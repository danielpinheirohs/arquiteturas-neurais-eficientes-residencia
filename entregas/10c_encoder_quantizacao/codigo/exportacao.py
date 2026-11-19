"""Exportação: a rede treinada sai do PyTorch e vira um arquivo ONNX, que o ONNX Runtime executa.

ONNX é um formato de arquivo para redes neurais: uma lista de operações
(somar, multiplicar matrizes, normalizar...) ligadas umas às outras, com os
pesos dentro. O ONNX Runtime é um programa que lê esse arquivo e faz as
contas, sem PyTorch. É nele que a Etapa 8 quantiza a rede.

**O que entra no arquivo e o que fica fora.** Um pedido passa por quatro
trechos. Só o segundo vira ONNX:

    A. texto -> ids de features          Python (`Featurizador`, o mesmo de sempre)
    B a F. ids -> pontuações             ONNX: tabela de features, projeção, encoder
                                         e as duas cabeças, até a pontuação de cada
                                         intenção e de cada tag por palavra
    F. pontuações das tags -> tags       numpy: Viterbi do CRF, com as regras BIO
    montar a resposta                    Python (`Previsao`)

O Viterbi fica fora por dois motivos. É um laço cujo tamanho depende do
pedido, com um caminho de volta no fim, e laço dentro de um grafo ONNX é
lento e difícil de ler. E ele não tem multiplicação por pesos, que é onde a
quantização ganha: só soma as transições (12 mil números) e escolhe o maior.
As transições ficam em ponto flutuante, num arquivo ao lado.

**O grafo é montado à mão**, operação por operação, em vez de sair do
exportador do PyTorch. Assim cada operação e cada matriz de pesos tem no
arquivo o mesmo nome que tem em `modelos/diet/` (por exemplo
`blocos.0.atencao.consulta.weight`), e a quantização pode apontar para elas
pelo nome. O preço é que a conta da rede fica escrita duas vezes, aqui e no
PyTorch; por isso um teste confere que as duas dão as mesmas pontuações, e o
notebook da etapa confere que dão as mesmas respostas em todo o `dev` e em
todo o teste.

**A tabela de features muda de forma, não de conta.** No PyTorch ela é uma
`nn.EmbeddingBag`: recebe os ids de todas as palavras numa fila e soma por
fatia. Aqui os ids de cada palavra ocupam uma linha de tamanho fixo, e as
sobras são preenchidas com o id de uma linha extra da tabela, que só tem
zeros. Somar a linha de zeros não muda nada. Ganha-se uma operação simples
(`Gather`, "buscar linhas"), que é a que o ONNX Runtime sabe quantizar.

O sistema exportado em si (`SistemaOnnx`, o Viterbi em numpy e os nomes dos
arquivos) mora em `sistema_onnx.py`, que roda sem PyTorch; este módulo, que
constrói o grafo a partir do modelo treinado, precisa dele. Os nomes de lá
continuam disponíveis aqui.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
from onnx import TensorProto, helper, numpy_helper

from especialista.compressao.sistema_onnx import (  # noqa: F401  (reexportados: quem já importava daqui continua funcionando)
    ARQUIVOS,
    CRF,
    FEATURES,
    FICHA,
    REDE,
    REDE_COMPACTADA,
    SistemaOnnx,
    abrir_sessao,
    arquivo_da_rede,
    viterbi,
)
from especialista.modelos.diet import Diet, SistemaDiet

OPSET = 17               # a primeira versão do ONNX em que a normalização de camada é uma operação só


# --------------------------------------------------------------------------
# O grafo: a mesma conta de `Diet.codificar` e `Diet._pontuar`, em operações do ONNX
# --------------------------------------------------------------------------


class _Grafo:
    """Junta as operações e os pesos do grafo. O nome de cada operação é o nome do que ela produz."""

    def __init__(self):
        self.operacoes: list[onnx.NodeProto] = []
        self.pesos: list[onnx.TensorProto] = []

    def peso(self, nome: str, valores) -> str:
        self.pesos.append(numpy_helper.from_array(np.ascontiguousarray(valores), nome))
        return nome

    def inteiros(self, nome: str, valores) -> str:
        return self.peso(nome, np.asarray(valores, dtype=np.int64))

    def op(self, operador: str, entradas: list[str], nome: str, **atributos) -> str:
        self.operacoes.append(helper.make_node(operador, entradas, [nome], name=nome, **atributos))
        return nome


def construir_rede(modelo: Diet) -> onnx.ModelProto:
    """A rede treinada como um grafo ONNX de 32 bits.

    Entradas:

    - `ids` (lote, palavras, features): os ids de features de cada palavra. Onde
      sobra espaço (palavra com menos features, pedido mais curto), o id é o da
      linha de zeros, que é o número de features do vocabulário.
    - `comprimentos` (lote,): quantas palavras tem cada pedido.

    Saídas: `intencoes` (lote, n_intents) e `emissoes` (lote, palavras, n_tags),
    as mesmas de `Diet.forward`.

    Cada camada linear vira uma multiplicação de matrizes (`MatMul`) seguida da
    soma do viés (`Add`). A matriz é gravada já transposta, com o nome que tem
    no PyTorch.
    """
    c = modelo.config
    estado = {nome: valor.detach().cpu().numpy() for nome, valor in modelo.state_dict().items()}
    g = _Grafo()
    d_cabeca = c.d_modelo // c.n_cabecas
    epsilon = float(modelo.norma_final.eps)

    def linear(x: str, camada: str) -> str:
        g.peso(f"{camada}.weight", estado[f"{camada}.weight"].T)
        g.peso(f"{camada}.bias", estado[f"{camada}.bias"])
        return g.op("Add", [g.op("MatMul", [x, f"{camada}.weight"], f"{camada}.produto"), f"{camada}.bias"], f"{camada}.saida")

    def norma(x: str, camada: str) -> str:
        g.peso(f"{camada}.weight", estado[f"{camada}.weight"])
        g.peso(f"{camada}.bias", estado[f"{camada}.bias"])
        return g.op("LayerNormalization", [x, f"{camada}.weight", f"{camada}.bias"], f"{camada}.saida", axis=-1, epsilon=epsilon)

    def com_linha_de_zeros(tabela: np.ndarray) -> np.ndarray:
        return np.concatenate([tabela, np.zeros((1, tabela.shape[1]), dtype=tabela.dtype)])

    # Números soltos de que as operações precisam
    for nome, valor in (("zero", 0), ("um", 1)):
        g.inteiros(nome, valor)
    for eixos in ([1], [2], [1, 2]):
        g.inteiros("eixos_" + "_".join(map(str, eixos)), eixos)
    g.inteiros("fim_do_eixo", [np.iinfo(np.int64).max])
    g.inteiros("menos_um", [-1])
    g.inteiros("comeco", [0])

    # Quem é palavra de verdade, quem é o CLS (logo depois da última palavra) e quem é preenchimento
    palavras_mais_cls = g.op("Add", [g.op("Gather", [g.op("Shape", ["ids"], "forma_dos_ids"), "um"], "palavras", axis=0), "um"], "palavras_mais_cls")
    posicoes = g.op("Range", ["zero", palavras_mais_cls, "um"], "posicoes")                       # (L+1,)
    comprimento = g.op("Unsqueeze", ["comprimentos", "eixos_1"], "comprimento_em_coluna")         # (B, 1)
    valido = g.op("LessOrEqual", [posicoes, comprimento], "valido")                               # (B, L+1): palavra ou CLS
    no_cls = g.op("Equal", [posicoes, comprimento], "no_cls")                                     # (B, L+1): só o CLS
    quantas = g.op("Unsqueeze", [g.op("Cast", ["comprimentos"], "comprimentos_em_real", to=TensorProto.FLOAT), "eixos_1_2"], "quantas_palavras")

    # B. Tabela de features: busca as linhas dos features de cada palavra e soma
    g.peso("tabela.weight", com_linha_de_zeros(estado["tabela.weight"]))
    vetores = g.op("ReduceSum", [g.op("Gather", ["tabela.weight", "ids"], "tabela.linhas", axis=0), "eixos_2"], "tabela.soma", keepdims=0)  # (B, L, d_features)

    # C. O resumo do pedido (média dos vetores das palavras) entra como uma posição a mais, e tudo é projetado junto
    media = g.op("Div", [g.op("ReduceSum", [vetores, "eixos_1"], "soma_do_pedido", keepdims=1), quantas], "media_do_pedido")    # (B, 1, d_features)
    projetado = linear(g.op("Concat", [vetores, media], "palavras_e_media", axis=1), "projecao")                             # (B, L+1, d_modelo)
    g.peso("marca_cls", estado["marca_cls"])
    cls = g.op("Add", [g.op("Slice", [projetado, "menos_um", "fim_do_eixo", "eixos_1"], "media_projetada"), "marca_cls"], "cls_de_entrada")
    sequencia = g.op("Where", [g.op("Unsqueeze", [no_cls, "eixos_2"], "no_cls_em_coluna"), cls, projetado], "sequencia_de_entrada")

    # D. Encoder. A tabela de distâncias entre posições (j - i, limitada) é a mesma para todos os blocos
    g.inteiros("distancia_min", -c.distancia_max)
    g.inteiros("distancia_max", c.distancia_max)
    distancia = g.op("Sub", [g.op("Unsqueeze", [posicoes, "comeco"], "posicoes_em_linha"), g.op("Unsqueeze", [posicoes, "eixos_1"], "posicoes_em_coluna")], "distancia")
    indice_da_distancia = g.op("Add", [g.op("Clip", [distancia, "distancia_min", "distancia_max"], "distancia_limitada"), "distancia_max"], "indice_da_distancia")
    valido_para_atencao = g.op("Unsqueeze", [valido, "eixos_1_2"], "valido_para_atencao")          # (B, 1, 1, L+1)
    g.inteiros("em_cabecas", [0, 0, c.n_cabecas, d_cabeca])     # 0 = manter a dimensão como está
    g.inteiros("de_volta", [0, 0, c.d_modelo])
    g.peso("raiz_da_cabeca", np.float32(math.sqrt(d_cabeca)))
    g.peso("menos_infinito", np.float32(-np.inf))

    for i in range(len(modelo.blocos)):
        b, a = f"blocos.{i}", f"blocos.{i}.atencao"
        normalizado = norma(sequencia, f"{b}.norma_atencao")
        separar = lambda camada, ordem: g.op(
            "Transpose", [g.op("Reshape", [linear(normalizado, f"{a}.{camada}"), "em_cabecas"], f"{a}.{camada}.em_cabecas")], f"{a}.{camada}.por_cabeca", perm=ordem
        )
        q = separar("consulta", [0, 2, 1, 3])    # (B, cabeças, L+1, d_cabeça)
        k = separar("chave", [0, 2, 3, 1])       # (B, cabeças, d_cabeça, L+1): já virada para o produto
        v = separar("valor", [0, 2, 1, 3])
        conteudo = g.op("MatMul", [q, k], f"{a}.conteudo")                                          # q_i · k_j
        g.peso(f"{a}.posicao", estado[f"{a}.posicao"].T)
        por_distancia = g.op("MatMul", [q, f"{a}.posicao"], f"{a}.por_distancia")                   # q_i · a[d], para cada distância d
        indice = g.op("Expand", [indice_da_distancia, g.op("Shape", [conteudo], f"{a}.forma_das_pontuacoes")], f"{a}.indice")
        posicao = g.op("GatherElements", [por_distancia, indice], f"{a}.posicao_relativa", axis=3)  # q_i · a[j - i]
        pontuacao = g.op("Div", [g.op("Add", [conteudo, posicao], f"{a}.soma"), "raiz_da_cabeca"], f"{a}.pontuacao")
        pesos = g.op("Softmax", [g.op("Where", [valido_para_atencao, pontuacao, "menos_infinito"], f"{a}.sem_preenchimento")], f"{a}.pesos", axis=-1)
        contexto = g.op("Reshape", [g.op("Transpose", [g.op("MatMul", [pesos, v], f"{a}.contexto")], f"{a}.contexto_por_palavra", perm=[0, 2, 1, 3]), "de_volta"], f"{a}.contexto_junto")
        sequencia = g.op("Add", [sequencia, linear(contexto, f"{a}.saida")], f"{b}.depois_da_atencao")
        densa = linear(g.op("Relu", [linear(norma(sequencia, f"{b}.norma_ffn"), f"{b}.expande")], f"{b}.relu"), f"{b}.contrai")
        sequencia = g.op("Add", [sequencia, densa], f"{b}.depois_da_densa")
    sequencia = norma(sequencia, "norma_final")

    # E. Intenção: a saída do CLS de cada pedido, comparada com o vetor de cada intenção
    do_cls = g.op("GatherND", [sequencia, comprimento], "saida_do_cls", batch_dims=1)               # (B, d_modelo)
    g.peso("vetor_da_intencao", estado["vetor_da_intencao"].T)
    intencoes = g.op("MatMul", [linear(do_cls, "cls_para_rotulo"), "vetor_da_intencao"], "pontuacao_profunda")
    if c.atalho_intencao == "resumo":
        g.inteiros("eixo_do_meio", [1])
        intencoes = g.op("Add", [intencoes, linear(g.op("Squeeze", [media, "eixo_do_meio"], "media_solta"), "atalho")], "pontuacao_com_atalho")
    elif c.atalho_intencao == "features":
        # a média, entre todos os features do pedido, da linha de cada um na tabela do atalho
        g.peso("atalho.weight", com_linha_de_zeros(estado["atalho.weight"]))
        g.inteiros("linha_de_zeros", estado["tabela.weight"].shape[0])
        g.peso("um_real", np.float32(1.0))
        soma = g.op("ReduceSum", [g.op("Gather", ["atalho.weight", "ids"], "atalho.linhas", axis=0), "eixos_1_2"], "atalho.soma", keepdims=0)
        ligados = g.op("Cast", [g.op("Less", ["ids", "linha_de_zeros"], "atalho.ligado")], "atalho.ligado_em_real", to=TensorProto.FLOAT)
        g.inteiros("eixos_1_2_3", [1, 2])
        quantos = g.op("Max", [g.op("ReduceSum", [ligados, "eixos_1_2_3"], "atalho.quantos", keepdims=0), "um_real"], "atalho.quantos_ao_menos_um")
        g.inteiros("eixo_do_fim", [1])
        intencoes = g.op("Add", [intencoes, g.op("Div", [soma, g.op("Unsqueeze", [quantos, "eixo_do_fim"], "atalho.quantos_em_coluna")], "atalho.media")], "pontuacao_com_atalho")
    g.op("Identity", [intencoes], "intencoes")

    # F. Entidades: uma pontuação por tag para cada palavra (o CLS, na última posição, fica de fora)
    palavras = g.op("Slice", [sequencia, "comeco", "menos_um", "eixos_1"], "saida_das_palavras")
    g.op("Identity", [linear(palavras, "emissao")], "emissoes")

    n_intents, n_tags = estado["vetor_da_intencao"].shape[0], estado["emissao.bias"].shape[0]
    grafo = helper.make_graph(
        g.operacoes,
        "diet",
        inputs=[
            helper.make_tensor_value_info("ids", TensorProto.INT64, ["lote", "palavras", "features"]),
            helper.make_tensor_value_info("comprimentos", TensorProto.INT64, ["lote"]),
        ],
        outputs=[
            helper.make_tensor_value_info("intencoes", TensorProto.FLOAT, ["lote", n_intents]),
            helper.make_tensor_value_info("emissoes", TensorProto.FLOAT, ["lote", "palavras", n_tags]),
        ],
        initializer=g.pesos,
    )
    rede = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="especialista")
    rede.ir_version = 8   # a versão do formato que acompanha o opset 17
    onnx.checker.check_model(rede)
    return rede


# --------------------------------------------------------------------------
# Gravar um sistema exportado
# --------------------------------------------------------------------------


def exportar(sistema: SistemaDiet, pasta: Path, origem: str | None = None) -> Path:
    """Grava em `pasta` o sistema inteiro, pronto para rodar sem PyTorch.

        rede.onnx       a rede em 32 bits, da tabela de features até as pontuações
        features.json   o vocabulário de features (o mesmo arquivo do modelo)
        crf.npz         as transições do CRF, em ponto flutuante, já com as regras BIO
        sistema.json    intenções, tags, o que a rede é e de onde veio

    `origem` é um texto livre que fica anotado na ficha (por exemplo, a pasta do modelo).
    """
    modelo = sistema.modelo.eval()
    pasta.mkdir(parents=True, exist_ok=True)
    onnx.save(construir_rede(modelo), pasta / REDE)
    (pasta / REDE_COMPACTADA).unlink(missing_ok=True)
    sistema.featurizador.salvar(pasta / FEATURES)
    with torch.no_grad():
        transicao, inicio = modelo.crf._pesos()
    np.savez(pasta / CRF, transicao=transicao.cpu().numpy(), inicio=inicio.cpu().numpy(), fim=modelo.crf.fim.detach().cpu().numpy())
    ficha = {
        "forma": "fp32",
        "nome": "ONNX, 32 bits",
        "origem": origem,
        "intents": sistema.intents,
        "tags": sistema.tags,
        "usar_crf": modelo.config.usar_crf,
        "config": modelo.config.como_dict(),
        "features": len(sistema.featurizador),
        "parametros": sistema.parametros,
        "opset": OPSET,
    }
    (pasta / FICHA).write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pasta


def copiar_acompanhantes(origem: Path, destino: Path, **troca) -> dict:
    """Copia para `destino` o que acompanha a rede (vocabulário, CRF e ficha), trocando na ficha os campos de `troca`."""
    destino.mkdir(parents=True, exist_ok=True)
    for arquivo in (FEATURES, CRF):
        shutil.copyfile(origem / arquivo, destino / arquivo)
    ficha = {**json.loads((origem / FICHA).read_text(encoding="utf-8")), **troca}
    (destino / FICHA).write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ficha


def ler_rede(pasta: Path) -> onnx.ModelProto:
    """O grafo gravado na pasta, para inspeção. Lê o arquivo comum ou o compactado."""
    arquivo = arquivo_da_rede(pasta)
    conteudo = gzip.decompress(arquivo.read_bytes()) if arquivo.suffix == ".gz" else arquivo.read_bytes()
    return onnx.load_model_from_string(conteudo)


def compactar(pasta: Path, destino: Path) -> Path:
    """Uma cópia do sistema em que a rede passa por um compressor comum (gzip, o mesmo algoritmo do zip).

    Serve para o modelo podado por magnitude. Zerar pesos não diminui o
    arquivo: um zero de 8 bits ocupa o mesmo byte que qualquer outro peso. Um
    compressor comum resolve isso sem formato especial: uma fila de zeros
    quase some, e os pesos que ficaram são guardados como estão. A leitura
    devolve exatamente os mesmos bytes, então as respostas não mudam; o custo
    é descompactar uma vez, ao carregar. É a última parte do Deep Compression
    (Han et al., arXiv 1510.00149): podar, quantizar e, por fim, codificar.
    """
    copiar_acompanhantes(pasta, destino)
    (destino / REDE).unlink(missing_ok=True)
    (destino / REDE_COMPACTADA).write_bytes(gzip.compress(arquivo_da_rede(pasta).read_bytes(), compresslevel=9, mtime=0))
    return destino


def bytes_do_sistema(pasta: Path) -> int:
    """Tamanho em disco de um sistema exportado: a rede e tudo o que ela precisa para responder."""
    return sum((pasta / arquivo).stat().st_size for arquivo in (arquivo_da_rede(pasta).name, *ARQUIVOS))


def bytes_por_arquivo(pasta: Path) -> dict[str, int]:
    return {arquivo: (pasta / arquivo).stat().st_size for arquivo in (arquivo_da_rede(pasta).name, *ARQUIVOS)}


# --------------------------------------------------------------------------
# O que há dentro do arquivo
# --------------------------------------------------------------------------

_SUFIXO_QUANTIZADO = "_quantized"   # o nome que o ONNX Runtime dá a um peso depois de quantizar


def pesos_da_rede(rede: onnx.ModelProto) -> pd.DataFrame:
    """Uma linha por tensor de pesos gravado no arquivo: forma, tipo, quantos números tem, quantos bytes ocupa e quantos são zero.

    Os pesos quantizados aparecem com o nome original. `zeros` conta os
    números que valem zero de verdade: num tensor de inteiros, os que são
    iguais ao ponto zero da quantização.
    """
    tensores = {t.name: numpy_helper.to_array(t) for t in rede.graph.initializer}
    linhas = []
    for nome, valores in tensores.items():
        quantizado = nome.endswith(_SUFIXO_QUANTIZADO)
        if valores.ndim == 0 or nome.endswith(("_scale", "_zero_point")) or not (quantizado or valores.dtype == np.float32):
            continue   # números soltos, formas e eixos das operações, e os parâmetros da quantização: não são pesos da rede
        original = nome.removesuffix(_SUFIXO_QUANTIZADO)
        ponto_zero = tensores.get(f"{original}_zero_point") if quantizado else None
        zero = 0 if ponto_zero is None or ponto_zero.ndim > 0 else ponto_zero
        linhas.append({
            "tensor": original, "forma": "×".join(map(str, valores.shape)), "tipo": str(valores.dtype), "numeros": int(valores.size),
            "bytes": int(valores.nbytes), "zeros": int((valores == zero).sum()), "quantizado": quantizado,
        })
    return pd.DataFrame(linhas).set_index("tensor")


def operacoes_da_rede(rede: onnx.ModelProto) -> pd.Series:
    """Quantas operações de cada tipo o grafo tem."""
    return pd.Series([no.op_type for no in rede.graph.node]).value_counts().rename("operações")
