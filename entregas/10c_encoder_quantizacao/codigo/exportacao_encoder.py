"""Exportação do encoder pré-treinado: o XLM-R com as duas cabeças sai do PyTorch e vira um arquivo ONNX (Etapa 10c).

É o par de `exportacao.py` (a rede do zero, Etapa 8) para o encoder da Etapa
10a. O que entra no arquivo e o que fica fora:

    A. texto -> subtokens                 `tokenizers` (o tokenizador do modelo, copiado para `tokenizador.json`)
    B a D. subtokens -> pontuações        ONNX: tabela de subtokens e de posições, as camadas do corpo e as duas cabeças
    montar a resposta                     Python (`Previsao`)

**O grafo é montado à mão**, operação por operação, como na Etapa 8, em vez
de sair do exportador do PyTorch. Três motivos:

- cada matriz de pesos tem no arquivo o mesmo nome que tem no PyTorch (por
  exemplo `encoder.encoder.layer.0.attention.self.query.weight`). A
  quantização aponta para elas pelo nome, e a conferência dos zeros da poda
  compara tensor com tensor;
- o grafo serve para qualquer tamanho de modelo: vocabulário cortado (a
  tabela tem menos linhas) e menos camadas (o laço dá menos voltas);
- não depende de nenhum pacote além do `onnx`.

O preço é o mesmo da Etapa 8: a conta da rede fica escrita duas vezes, aqui
e na biblioteca `transformers`. Por isso um teste confere que as duas dão as
mesmas pontuações, e a etapa confere que dão as mesmas respostas em todo o
`dev` e em todo o teste.

**A conta** é a do XLM-R (Conneau et al., arXiv 1911.02116), como a
biblioteca `transformers` a escreve:

- entrada: linha do subtoken + linha da posição + linha do tipo (uma só),
  e uma normalização de camada. A posição de um subtoken é a sua ordem no
  pedido, contada a partir do id de preenchimento mais um; o preenchimento
  fica sempre com a posição do próprio id de preenchimento;
- cada camada: atenção com várias cabeças, soma com a entrada, normalização;
  rede densa com GELU, soma, normalização (a normalização vem **depois** de
  cada soma);
- a intenção sai de uma camada linear sobre o estado do primeiro token; a
  tag de cada palavra, de uma camada linear sobre o estado do primeiro
  subtoken da palavra (Chen et al., JointBERT, arXiv 1902.10909).

O sistema exportado em si (`SistemaEncoderOnnx`) mora em
`sistema_encoder_onnx.py`, que roda sem PyTorch.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from especialista.compressao.exportacao import OPSET, _Grafo, ler_rede  # noqa: F401  (`ler_rede` serve aos dois sistemas exportados)
from especialista.compressao.sistema_encoder_onnx import (  # noqa: F401  (reexportados, como em `exportacao.py`)
    ARQUIVOS,
    FICHA,
    REDE,
    REDE_COMPACTADA,
    TOKENIZADOR,
    TOKENIZADOR_COMPACTADO,
    SistemaEncoderOnnx,
    arquivo_da_rede,
    arquivo_do_tokenizador,
)
from especialista.modelos.encoder import SistemaEncoder
from especialista.modelos.encoder.modelo import EncoderConjunto

TABELA_DE_SUBTOKENS = "encoder.embeddings.word_embeddings.weight"
TABELA_DE_POSICOES = "encoder.embeddings.position_embeddings.weight"


# --------------------------------------------------------------------------
# O grafo: a mesma conta de `EncoderConjunto.forward`, em operações do ONNX
# --------------------------------------------------------------------------


def construir_rede(modelo: EncoderConjunto) -> onnx.ModelProto:
    """O encoder ajustado, com as duas cabeças, como um grafo ONNX de 32 bits.

    Entradas (as três grades de `sistema_encoder_onnx.Alinhador`): `ids` e
    `atencao`, com forma (lote, subtokens), e `primeiros`, com forma (lote,
    palavras). Saídas: `intencoes` (lote, n_intents) e `emissoes` (lote,
    palavras, n_tags), as mesmas de `EncoderConjunto.forward`.

    Cada camada linear vira uma multiplicação de matrizes (`MatMul`) seguida
    da soma do viés (`Add`). A matriz é gravada já transposta, com o nome que
    tem no PyTorch. Os dois produtos da atenção (consulta × chave e pesos ×
    valor) são `MatMul` entre duas ativações, sem matriz gravada.
    """
    if modelo.tabela_dividida is not None:
        raise RuntimeError("A tabela está dividida (treino em andamento): conclua o treino antes de exportar.")
    c = modelo.encoder.config
    if getattr(c, "hidden_act", "gelu") != "gelu":
        raise ValueError(f"O grafo foi escrito para a ativação GELU (a do XLM-R); este encoder usa {c.hidden_act}.")
    if getattr(c, "position_embedding_type", "absolute") != "absolute":
        raise ValueError(f"O grafo foi escrito para posições absolutas (as do XLM-R); este encoder usa {c.position_embedding_type}.")
    estado = {nome: valor.detach().cpu().numpy() for nome, valor in modelo.state_dict().items()}
    if any(valor.dtype != np.float32 for nome, valor in estado.items() if nome.endswith((".weight", ".bias"))):
        raise ValueError("A exportação espera os pesos em 32 bits.")
    g = _Grafo()
    d, n_cabecas = c.hidden_size, c.num_attention_heads
    d_cabeca = d // n_cabecas
    preenchimento = int(modelo.encoder.embeddings.padding_idx)
    epsilon = float(c.layer_norm_eps)

    def linear(x: str, camada: str) -> str:
        g.peso(f"{camada}.weight", estado[f"{camada}.weight"].T)
        g.peso(f"{camada}.bias", estado[f"{camada}.bias"])
        return g.op("Add", [g.op("MatMul", [x, f"{camada}.weight"], f"{camada}.produto"), f"{camada}.bias"], f"{camada}.saida")

    def norma(x: str, camada: str) -> str:
        g.peso(f"{camada}.weight", estado[f"{camada}.weight"])
        g.peso(f"{camada}.bias", estado[f"{camada}.bias"])
        return g.op("LayerNormalization", [x, f"{camada}.weight", f"{camada}.bias"], f"{camada}.saida", axis=-1, epsilon=epsilon)

    def gelu(x: str, nome: str) -> str:
        """GELU exata: x · ½ · (1 + erf(x / √2)). O opset 17 não tem a operação pronta; o ONNX Runtime reconhece este desenho e o junta numa só."""
        erf = g.op("Erf", [g.op("Div", [x, "raiz_de_dois"], f"{nome}.dividido")], f"{nome}.erf")
        return g.op("Mul", [g.op("Mul", [x, g.op("Add", [erf, "um_real"], f"{nome}.um_mais_erf")], f"{nome}.vezes_x"), "meio"], f"{nome}.saida")

    def solto(nome: str, valor, tipo) -> str:
        """Um número só, sem eixos (`_Grafo.peso` daria a ele um eixo de tamanho 1, e buscar a posição 0 criaria um eixo a mais)."""
        g.pesos.append(numpy_helper.from_array(np.array(valor, dtype=tipo), nome))
        return nome

    # Números soltos de que as operações precisam
    solto("zero", 0, np.int64)
    solto("um", 1, np.int64)
    solto("id_do_preenchimento", preenchimento, np.int64)
    g.inteiros("eixos_1_2", [1, 2])
    g.inteiros("eixo_do_fim", [-1])
    g.inteiros("em_cabecas", [0, 0, n_cabecas, d_cabeca])     # 0 = manter a dimensão como está
    g.inteiros("de_volta", [0, 0, d])
    solto("um_real", 1.0, np.float32)
    solto("meio", 0.5, np.float32)
    solto("raiz_de_dois", math.sqrt(2.0), np.float32)
    solto("raiz_da_cabeca", math.sqrt(d_cabeca), np.float32)
    solto("menor_real", np.finfo(np.float32).min, np.float32)

    # A. Entrada. A posição de cada subtoken: a ordem dele no pedido, contada depois do id de preenchimento
    e = "encoder.embeddings"
    de_verdade = g.op("Cast", [g.op("Not", [g.op("Equal", ["ids", "id_do_preenchimento"], "e_preenchimento")], "nao_e_preenchimento")],
                      "subtoken_de_verdade", to=TensorProto.INT64)                                    # (B, S)
    ordem = g.op("Mul", [g.op("CumSum", [de_verdade, "um"], "contagem_dos_subtokens"), de_verdade], "ordem_no_pedido")
    posicoes = g.op("Add", [ordem, "id_do_preenchimento"], "posicoes")
    g.peso(TABELA_DE_SUBTOKENS, estado[TABELA_DE_SUBTOKENS])
    g.peso(TABELA_DE_POSICOES, estado[TABELA_DE_POSICOES])
    g.peso(f"{e}.token_type_embeddings.weight", estado[f"{e}.token_type_embeddings.weight"][0])   # o XLM-R só tem um tipo: a linha 0
    dos_subtokens = g.op("Gather", [TABELA_DE_SUBTOKENS, "ids"], f"{e}.word_embeddings.linhas", axis=0)           # (B, S, d)
    das_posicoes = g.op("Gather", [TABELA_DE_POSICOES, posicoes], f"{e}.position_embeddings.linhas", axis=0)
    com_tipo = g.op("Add", [dos_subtokens, f"{e}.token_type_embeddings.weight"], f"{e}.com_tipo")
    sequencia = norma(g.op("Add", [com_tipo, das_posicoes], f"{e}.soma"), f"{e}.LayerNorm")

    # O preenchimento não recebe atenção: soma-se o menor número real à pontuação dele, antes do softmax
    fora = g.op("Mul", [g.op("Sub", ["um_real", g.op("Cast", ["atencao"], "atencao_em_real", to=TensorProto.FLOAT)], "e_preenchimento_em_real"),
                        "menor_real"], "desconto_do_preenchimento")                                    # (B, S)
    desconto = g.op("Unsqueeze", [fora, "eixos_1_2"], "desconto_para_atencao")                         # (B, 1, 1, S)

    # B. As camadas do corpo
    for i in range(len(modelo.encoder.encoder.layer)):
        p = f"encoder.encoder.layer.{i}"
        a = f"{p}.attention.self"
        separar = lambda camada, ordem_dos_eixos, entrada=sequencia, a=a: g.op(
            "Transpose", [g.op("Reshape", [linear(entrada, f"{a}.{camada}"), "em_cabecas"], f"{a}.{camada}.em_cabecas")],
            f"{a}.{camada}.por_cabeca", perm=ordem_dos_eixos,
        )
        q = separar("query", [0, 2, 1, 3])    # (B, cabeças, S, d_cabeça)
        k = separar("key", [0, 2, 3, 1])      # (B, cabeças, d_cabeça, S): já virada para o produto
        v = separar("value", [0, 2, 1, 3])
        pontuacao = g.op("Div", [g.op("MatMul", [q, k], f"{a}.produto_consulta_chave"), "raiz_da_cabeca"], f"{a}.pontuacao")
        pesos = g.op("Softmax", [g.op("Add", [pontuacao, desconto], f"{a}.sem_preenchimento")], f"{a}.pesos", axis=-1)
        contexto = g.op("Reshape", [g.op("Transpose", [g.op("MatMul", [pesos, v], f"{a}.contexto")], f"{a}.contexto_por_subtoken", perm=[0, 2, 1, 3]),
                                    "de_volta"], f"{a}.contexto_junto")
        depois_da_atencao = norma(g.op("Add", [linear(contexto, f"{p}.attention.output.dense"), sequencia], f"{p}.attention.output.soma"),
                                  f"{p}.attention.output.LayerNorm")
        densa = linear(gelu(linear(depois_da_atencao, f"{p}.intermediate.dense"), f"{p}.intermediate.gelu"), f"{p}.output.dense")
        sequencia = norma(g.op("Add", [densa, depois_da_atencao], f"{p}.output.soma"), f"{p}.output.LayerNorm")

    # C. Intenção: uma camada linear sobre o estado do primeiro token
    g.op("Identity", [linear(g.op("Gather", [sequencia, "zero"], "estado_do_inicio", axis=1), "intencao")], "intencoes")

    # D. Entidades: uma camada linear sobre o estado do primeiro subtoken de cada palavra
    das_palavras = g.op("GatherND", [sequencia, g.op("Unsqueeze", ["primeiros", "eixo_do_fim"], "primeiros_em_coluna")],
                        "estado_das_palavras", batch_dims=1)                                           # (B, L, d)
    g.op("Identity", [linear(das_palavras, "entidades")], "emissoes")

    n_intents, n_tags = estado["intencao.bias"].shape[0], estado["entidades.bias"].shape[0]
    grafo = helper.make_graph(
        g.operacoes,
        "encoder_conjunto",
        inputs=[
            helper.make_tensor_value_info("ids", TensorProto.INT64, ["lote", "subtokens"]),
            helper.make_tensor_value_info("atencao", TensorProto.INT64, ["lote", "subtokens"]),
            helper.make_tensor_value_info("primeiros", TensorProto.INT64, ["lote", "palavras"]),
        ],
        outputs=[
            helper.make_tensor_value_info("intencoes", TensorProto.FLOAT, ["lote", n_intents]),
            helper.make_tensor_value_info("emissoes", TensorProto.FLOAT, ["lote", "palavras", n_tags]),
        ],
        initializer=g.pesos,
    )
    rede = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="especialista")
    rede.ir_version = 8   # a versão do formato que acompanha o opset 17
    return rede


# --------------------------------------------------------------------------
# Gravar um sistema exportado
# --------------------------------------------------------------------------


def gravar_tokenizador(tokenizador, caminho: Path) -> Path:
    """Grava o tokenizador do modelo num arquivo só, que o pacote `tokenizers` lê sem a biblioteca `transformers`.

    É o mesmo conteúdo do `tokenizer.json` do modelo (num modelo de
    vocabulário cortado, já é o tokenizador reduzido), sem as opções de
    preenchimento e de corte que a biblioteca liga durante o uso, e sem os
    espaços de formatação, que só aumentam o arquivo.
    """
    from tokenizers import Tokenizer

    limpo = Tokenizer.from_str(tokenizador.backend_tokenizer.to_str())
    limpo.no_padding()
    limpo.no_truncation()
    limpo.save(str(caminho), pretty=False)
    return caminho


def exportar(sistema: SistemaEncoder, pasta: Path, origem: str | None = None) -> Path:
    """Grava em `pasta` o sistema inteiro, pronto para rodar sem PyTorch.

        rede.onnx          a rede em 32 bits, da tabela de subtokens até as pontuações
        tokenizador.json   o tokenizador do modelo
        sistema.json       intenções, tags, o limite de subtokens, o que a rede é e de onde veio

    `origem` é um texto livre que fica anotado na ficha (por exemplo, a pasta do modelo).
    """
    modelo = sistema.modelo.eval()
    encoder = modelo.encoder
    if sistema.tokenizador.pad_token_id != encoder.embeddings.padding_idx:
        raise ValueError("O id de preenchimento do tokenizador não é o do encoder: as posições do XLM-R dependem dele.")
    pasta.mkdir(parents=True, exist_ok=True)
    rede = construir_rede(modelo)
    onnx.save(rede, pasta / REDE)
    onnx.checker.check_model(str(pasta / REDE))   # pelo arquivo: a rede do XLM-R tem mais de 1 GB
    del rede
    (pasta / REDE_COMPACTADA).unlink(missing_ok=True)
    (pasta / TOKENIZADOR_COMPACTADO).unlink(missing_ok=True)
    gravar_tokenizador(sistema.tokenizador, pasta / TOKENIZADOR)
    ficha = {
        "forma": "fp32",
        "nome": "ONNX, 32 bits",
        "origem": origem,
        "intents": sistema.intents,
        "tags": sistema.tags,
        "max_subtokens": modelo.config.max_subtokens,
        "id_do_preenchimento": int(encoder.embeddings.padding_idx),
        "config": modelo.config.como_dict(),
        "camadas": len(encoder.encoder.layer),
        "linhas_da_tabela": int(encoder.get_input_embeddings().num_embeddings),
        "d_modelo": int(encoder.config.hidden_size),
        "cabecas_de_atencao": int(encoder.config.num_attention_heads),
        "parametros": sistema.parametros,
        "opset": OPSET,
    }
    (pasta / FICHA).write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pasta


def copiar_acompanhantes(origem: Path, destino: Path, **troca) -> dict:
    """Copia para `destino` o que acompanha a rede (tokenizador e ficha), trocando na ficha os campos de `troca`."""
    destino.mkdir(parents=True, exist_ok=True)
    tokenizador = arquivo_do_tokenizador(origem)
    shutil.copyfile(tokenizador, destino / tokenizador.name)
    ficha = {**json.loads((origem / FICHA).read_text(encoding="utf-8")), **troca}
    (destino / FICHA).write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ficha


def _gzip(origem: Path, destino: Path, pedaco: int = 1 << 24) -> None:
    """O arquivo passado pelo gzip (nível 9, o de `exportacao.compactar`), lido aos pedaços: a rede pode ter mais de 1 GB."""
    with open(origem, "rb") as entrada, open(destino, "wb") as saida:
        with gzip.GzipFile(filename="", mode="wb", fileobj=saida, compresslevel=9, mtime=0) as compressor:
            shutil.copyfileobj(entrada, compressor, pedaco)


def compactar(pasta: Path, destino: Path) -> Path:
    """Uma cópia do sistema em que a rede e o tokenizador passam por um compressor comum (gzip, o mesmo algoritmo do zip).

    Como em `exportacao.compactar` (Etapa 8): serve para o modelo com pesos
    soltos em zero, em que um zero de 8 bits ocupa o mesmo byte que qualquer
    outro peso. É a última parte do Deep Compression (Han et al., arXiv
    1510.00149): podar, quantizar e, por fim, codificar. Aqui o tokenizador
    também é compactado: é um arquivo de texto com as peças do vocabulário, e
    no modelo em 8 bits passa a ser uma parte visível do sistema. A leitura
    devolve os mesmos bytes, então as respostas não mudam.
    """
    destino.mkdir(parents=True, exist_ok=True)
    for arquivo in (REDE, TOKENIZADOR):
        (destino / arquivo).unlink(missing_ok=True)
    _gzip(pasta / REDE, destino / REDE_COMPACTADA)
    _gzip(pasta / TOKENIZADOR, destino / TOKENIZADOR_COMPACTADO)
    shutil.copyfile(pasta / FICHA, destino / FICHA)
    return destino


def bytes_por_arquivo(pasta: Path) -> dict[str, int]:
    """O tamanho de cada arquivo de um sistema exportado: a rede, o tokenizador e a ficha."""
    return {arquivo.name: arquivo.stat().st_size for arquivo in (arquivo_da_rede(pasta), arquivo_do_tokenizador(pasta), pasta / FICHA)}


def bytes_do_sistema(pasta: Path) -> int:
    """Tamanho em disco de um sistema exportado: a rede e tudo o que ela precisa para responder."""
    return sum(bytes_por_arquivo(pasta).values())
