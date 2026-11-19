"""Poda do encoder pré-treinado: o que tirar do XLM-R e como reajustar o que ficou (Etapa 10b).

É o roteiro de `poda.py` (a poda da rede do zero, Etapa 7) com as peças do
encoder. O que é igual vem de lá (`aplicar`, `Agenda`, `esparsidade_cubica`,
o arquivo esparso); o que depende da rede está aqui.

1. **Pesos soltos, por magnitude** (Han et al., arXiv 1506.02626), de uma vez
   ou aos poucos (Zhu & Gupta, arXiv 1710.01878). Entram as 6 matrizes de
   cada camada do corpo: 4 da atenção e 2 da rede densa. São as mesmas 6 de
   Gordon et al. (arXiv 2002.08307, seção 3.1), que também cortam cada matriz
   em separado, com o seu próprio limite.

   **A tabela de subtokens fica fora da poda por magnitude.** Sanh et al.
   (arXiv 2005.07683, seção 5) também deixam as tabelas de entrada de fora e
   contam a esparsidade só no corpo. Gordon et al. podam a tabela "do mesmo
   jeito": aqui isso é um desvio declarado. O motivo: a tabela do XLM-R tem
   69% dos parâmetros, e 98% das suas linhas nunca são lidas em português.
   Zerar pesos nessas linhas daria esparsidade sem dizer nada sobre a rede. A
   ferramenta certa para a tabela é a de baixo: tirar as linhas inteiras.

2. **Peças inteiras.**
   - **Corte do vocabulário:** saem da tabela as linhas dos subtokens que não
     servem, e o tokenizador encolhe junto. É a "poda de vocabulário" do
     TextPruner (Yang et al., arXiv 2203.15996, seção 3.1), que no XLM-R
     fica só com os subtokens que aparecem no treino (seção 4.2), e a ideia
     de Abdaoui et al. (arXiv 2010.05609): num modelo multilíngue, a maior
     parte dos parâmetros está na tabela, e ficar só com os idiomas que
     interessam corta o modelo sem mexer no corpo.
   - **Camadas de cima:** Sajjad et al. (arXiv 2004.03844, seção 5.1)
     comparam seis formas de descartar camadas e a melhor é a mais simples,
     tirar as de cima; depois, ajuste fino.

3. **O reajuste** é o laço de `treino_encoder.treinar`, passo por passo, com
   a poda dentro.

"Esparsidade" é sempre a fração dos pesos podáveis (as matrizes do corpo)
que está em zero.
"""

from __future__ import annotations

import copy
import json
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro
from especialista.compressao import poda
from especialista.compressao.poda import Agenda, Mascaras, aplicar, esparsidade_cubica
from especialista.modelos.encoder import Config, SistemaEncoder
from especialista.modelos.encoder.modelo import EncoderConjunto
from especialista.treino_encoder import Receita, fila_de_lotes, fracao_da_taxa

__all__ = [
    "Agenda", "Mascaras", "aplicar", "esparsidade_cubica", "nomes_podaveis", "mascaras_por_magnitude", "mascaras_dos_zeros",
    "por_tensor", "contar", "ids_do_texto", "ids_por_alfabeto", "reduzir_tokenizador", "cortar_vocabulario", "cobertura",
    "manter_camadas", "reajustar", "salvar_esparso", "carregar_esparso", "bytes_com_zip", "tamanhos", "execucao_esparsa",
]


# --------------------------------------------------------------------------
# Quais pesos entram na poda por magnitude
# --------------------------------------------------------------------------


def nomes_podaveis(modelo: EncoderConjunto) -> list[str]:
    """As matrizes que a poda por magnitude pode cortar: as 6 de cada camada do corpo.

    Consulta, chave, valor e saída da atenção, e as duas da rede densa. No
    XLM-R base são 72 matrizes, com 84.934.656 pesos (99,9% do corpo). Ficam
    de fora os vieses, as normalizações, as posições, as cabeças e a tabela de
    subtokens (o motivo está no texto do módulo).
    """
    nomes = []
    for i in range(len(modelo.encoder.encoder.layer)):
        camada = f"encoder.encoder.layer.{i}"
        nomes += [f"{camada}.attention.self.{matriz}.weight" for matriz in ("query", "key", "value")]
        nomes += [f"{camada}.attention.output.dense.weight", f"{camada}.intermediate.dense.weight", f"{camada}.output.dense.weight"]
    return nomes


@torch.no_grad()
def mascaras_por_magnitude(modelo: EncoderConjunto, esparsidade: float, anteriores: Mascaras | None = None) -> Mascaras:
    """Em cada matriz podável, marca para sair a fração `esparsidade` dos pesos de menor valor absoluto.

    A mesma conta de `poda.mascaras_por_magnitude`: o corte é por matriz, e
    saem exatamente `round(esparsidade × número de pesos)` pesos de cada uma.
    `anteriores` são as máscaras de um corte anterior (poda gradual): quem já
    saiu continua fora.
    """
    if not 0.0 <= esparsidade < 1.0:
        raise ValueError(f"A esparsidade é uma fração entre 0 e 1 (sem chegar a 1); veio {esparsidade}.")
    mascaras = {}
    for nome in nomes_podaveis(modelo):
        peso = modelo.get_parameter(nome)
        tamanho = peso.abs().flatten()
        ja_saiu = None if anteriores is None else ~anteriores[nome].flatten()
        if ja_saiu is not None:
            tamanho = tamanho.masked_fill(ja_saiu, -1.0)   # entram primeiro na fila dos menores
        saem = round(esparsidade * tamanho.numel())
        fica = torch.ones_like(tamanho, dtype=torch.bool)
        fica[tamanho.argsort(stable=True)[:saem]] = False
        if ja_saiu is not None:
            fica &= ~ja_saiu
        mascaras[nome] = fica.view_as(peso)
    return mascaras


@torch.no_grad()
def mascaras_dos_zeros(modelo: EncoderConjunto) -> Mascaras:
    """As máscaras de um modelo já podado, lidas dos próprios pesos (o "pattern-lock" de Zafrir et al., arXiv 2111.05754)."""
    return {nome: modelo.get_parameter(nome) != 0 for nome in nomes_podaveis(modelo)}


def por_tensor(modelo: EncoderConjunto) -> pd.DataFrame:
    """Uma linha por tensor de parâmetros: quantos pesos tem, quantos são diferentes de zero e se entra na poda."""
    podaveis = set(nomes_podaveis(modelo))
    linhas = [
        {"tensor": nome, "pesos": p.numel(), "nao_nulos": int(torch.count_nonzero(p)), "podavel": nome in podaveis}
        for nome, p in modelo.named_parameters()
    ]
    return pd.DataFrame(linhas).set_index("tensor")


def contar(modelo: EncoderConjunto) -> dict[str, int | float]:
    """Os números de tamanho de um modelo, podado ou não (os mesmos campos de `poda.contar`)."""
    tabela = por_tensor(modelo)
    podaveis = tabela[tabela["podavel"]]
    return {
        "parametros": int(tabela["pesos"].sum()),
        "nao_nulos": int(tabela["nao_nulos"].sum()),
        "podaveis": int(podaveis["pesos"].sum()),
        "nao_nulos_podaveis": int(podaveis["nao_nulos"].sum()),
        "esparsidade": float(1 - podaveis["nao_nulos"].sum() / podaveis["pesos"].sum()),
    }


# --------------------------------------------------------------------------
# Peças inteiras (1): o corte do vocabulário
# --------------------------------------------------------------------------
#
# Duas formas de escolher as linhas que ficam. Nenhuma olha o texto de `dev` nem do teste.


def ids_do_texto(sistema: SistemaEncoder, enunciados) -> torch.Tensor:
    """Forma "treino": os subtokens que aparecem em `enunciados` (o texto de treino, sem rótulo), mais os tokens especiais.

    É a escolha do TextPruner para o XLM-R (Yang et al., arXiv 2203.15996,
    seção 4.2: "só os tokens que aparecem no treino").
    """
    return sistema.alinhador.ids_usados(enunciados)


# Fora das letras e do ASCII: os sinais do bloco Latin-1 (« » ª º ° £), o × e o ÷, a pontuação geral
# (travessão, aspas curvas, reticências) e os símbolos de moeda (€).
_SINAIS = ((0x00A0, 0x00BF), (0x00D7, 0x00D7), (0x00F7, 0x00F7), (0x2000, 0x206F), (0x20A0, 0x20CF))
MARCA_DE_PALAVRA = "▁"   # o sinal com que o tokenizador marca o começo de uma palavra


def do_alfabeto_latino(caractere: str) -> bool:
    """O caractere é letra latina (com ou sem acento), dígito, pontuação ou símbolo comum?"""
    codigo = ord(caractere)
    if caractere == MARCA_DE_PALAVRA or 0x20 <= codigo < 0x7F:   # o ASCII que se imprime: letras, dígitos, pontuação
        return True
    if any(inicio <= codigo <= fim for inicio, fim in _SINAIS):
        return True
    return caractere.isalpha() and unicodedata.name(caractere, "").startswith("LATIN")


def ids_por_alfabeto(tokenizador) -> torch.Tensor:
    """Forma "alfabeto": os subtokens escritos só com letras latinas, dígitos e pontuação, mais os tokens especiais.

    Não usa corpus nenhum: a escolha olha só para a grafia de cada subtoken.
    Saem as peças em cirílico, árabe, chinês e nos outros alfabetos que o
    português não usa. É uma escolha deste projeto, na linha de Abdaoui et
    al. (arXiv 2010.05609), que encolhem o vocabulário de um modelo
    multilíngue para os idiomas que interessam.
    """
    ficam = set(tokenizador.all_special_ids)
    ficam.update(i for peca, i in tokenizador.get_vocab().items() if all(do_alfabeto_latino(c) for c in peca))
    return torch.tensor(sorted(ficam), dtype=torch.long)


def _com_os_especiais(tokenizador, ficam) -> list[int]:
    return sorted({int(i) for i in torch.as_tensor(ficam).tolist()} | set(tokenizador.all_special_ids))


def _trocar_ids(processador: dict, novo_id: dict[int, int]) -> None:
    """Acerta, no pós-processador do tokenizador, os ids dos tokens especiais (o de início e o de fim)."""
    for outro in processador.get("processors", []):          # uma sequência de pós-processadores
        _trocar_ids(outro, novo_id)
    for especial in processador.get("special_tokens", {}).values():   # TemplateProcessing
        especial["ids"] = [novo_id[i] for i in especial["ids"]]
    for campo in ("sep", "cls"):                              # RobertaProcessing: ["</s>", 2]
        if isinstance(processador.get(campo), list):
            processador[campo][1] = novo_id[processador[campo][1]]


def reduzir_tokenizador(tokenizador, ficam):
    """O mesmo tokenizador, só com os subtokens de `ficam`. O subtoken de id antigo `ficam[i]` passa a ter o id `i`.

    O tokenizador do XLM-R é do tipo Unigram (SentencePiece): tem uma lista
    de peças, cada uma com uma pontuação, e parte cada palavra na sequência
    de peças de maior pontuação total. Tirar peças da lista não muda as
    pontuações das que ficam. Então:

    - uma palavra cujas peças ficaram todas é partida do mesmo jeito;
    - uma palavra que usava uma peça cortada é partida de novo, em peças
      menores que ficaram; um caractere que não ficou em peça nenhuma vira
      `<unk>`.

    É o que acontece no TextPruner e em Abdaoui et al.: o vocabulário do
    tokenizador encolhe junto com a tabela. O arquivo do tokenizador também
    encolhe, e é ele que conta no tamanho do sistema.
    """
    import transformers
    from tokenizers import Tokenizer

    descricao = json.loads(tokenizador.backend_tokenizer.to_str())
    modelo = descricao["model"]
    if modelo.get("type") != "Unigram":
        raise ValueError(f"O corte do vocabulário foi escrito para tokenizadores Unigram (o do XLM-R); este é {modelo.get('type')}.")
    ficam = _com_os_especiais(tokenizador, ficam)
    novo_id = {antigo: novo for novo, antigo in enumerate(ficam)}
    modelo["vocab"] = [modelo["vocab"][i] for i in ficam]
    modelo["unk_id"] = novo_id[modelo["unk_id"]]
    for token in descricao.get("added_tokens", []):
        token["id"] = novo_id[token["id"]]
    if descricao.get("post_processor"):
        _trocar_ids(descricao["post_processor"], novo_id)

    classe = getattr(transformers, "PreTrainedTokenizerFast", None) or transformers.TokenizersBackend
    nomes = ("bos_token", "eos_token", "unk_token", "pad_token", "cls_token", "sep_token", "mask_token")
    especiais = {nome: str(getattr(tokenizador, nome)) for nome in nomes if getattr(tokenizador, nome, None) is not None}
    return classe(tokenizer_object=Tokenizer.from_str(json.dumps(descricao)), model_max_length=tokenizador.model_max_length, **especiais)


def cortar_vocabulario(sistema: SistemaEncoder, ficam) -> SistemaEncoder:
    """Um sistema novo só com as linhas `ficam` da tabela de subtokens, e com o tokenizador reduzido a elas.

    O corpo e as cabeças são copiados sem mudança. Para um texto cujos
    subtokens ficaram todos, as respostas são exatamente as mesmas. O sistema
    recebido não muda.
    """
    modelo = sistema.modelo
    if modelo.tabela_dividida is not None:
        raise RuntimeError("A tabela está dividida (treino em andamento): conclua o treino antes de cortar o vocabulário.")
    ficam = _com_os_especiais(sistema.tokenizador, ficam)
    tokenizador = reduzir_tokenizador(sistema.tokenizador, ficam)
    encoder = modelo.encoder
    if tokenizador.pad_token_id != encoder.config.pad_token_id:
        raise ValueError("O token de preenchimento mudou de id: as posições do XLM-R dependem dele.")
    tabela = encoder.get_input_embeddings()
    indices = torch.tensor(ficam, dtype=torch.long, device=tabela.weight.device)
    menor = nn.Embedding.from_pretrained(tabela.weight.detach()[indices].clone(), freeze=False, padding_idx=tabela.padding_idx)
    encoder.set_input_embeddings(menor)      # por um momento, para a cópia não levar a tabela inteira
    try:
        copia = copy.deepcopy(modelo)
    finally:
        encoder.set_input_embeddings(tabela)
    copia.encoder.config.vocab_size = len(ficam)
    return SistemaEncoder(copia, tokenizador, sistema.intents, sistema.tags, sistema.dispositivo, sistema.tamanho_do_lote)


def cobertura(original, reduzido, ficam, enunciados, max_subtokens: int = 128) -> dict[str, int | float]:
    """Quanto de um texto o vocabulário cortado deixa de cobrir. É só diagnóstico: não escolhe nada.

    `original` e `reduzido` são os dois tokenizadores e `enunciados`, listas
    de palavras. Devolve:

    - `fracao_cortada`: dos subtokens que o tokenizador original produz, a
      fração cujas linhas saíram da tabela (e `enunciados_com_corte`, quantos
      pedidos têm pelo menos um);
    - `fracao_desconhecida`: dos subtokens que o tokenizador reduzido produz,
      a fração que é `<unk>`: o que sobrou sem representação depois de a
      palavra ser partida de novo;
    - `subtokens_antes` e `subtokens_depois`: partir de novo deixa os pedidos
      mais compridos.
    """
    enunciados = [list(palavras) for palavras in enunciados]
    ficam = set(_com_os_especiais(original, ficam))
    opcoes = {"is_split_into_words": True, "truncation": True, "max_length": max_subtokens, "add_special_tokens": False}
    antes = original(enunciados, **opcoes)["input_ids"]
    depois = reduzido(enunciados, **opcoes)["input_ids"]
    total_antes, total_depois = sum(map(len, antes)), sum(map(len, depois))
    cortados = sum(i not in ficam for linha in antes for i in linha)
    desconhecidos = sum(i == reduzido.unk_token_id for linha in depois for i in linha)
    return {
        "enunciados": len(enunciados),
        "subtokens_antes": total_antes,
        "subtokens_depois": total_depois,
        "subtokens_cortados": cortados,
        "fracao_cortada": cortados / max(1, total_antes),
        "enunciados_com_corte": sum(any(i not in ficam for i in linha) for linha in antes),
        "subtokens_desconhecidos": desconhecidos,
        "fracao_desconhecida": desconhecidos / max(1, total_depois),
        "enunciados_com_desconhecido": sum(any(i == reduzido.unk_token_id for i in linha) for linha in depois),
    }


# --------------------------------------------------------------------------
# Peças inteiras (2): as camadas de cima
# --------------------------------------------------------------------------


def manter_camadas(sistema: SistemaEncoder, quantas: int) -> SistemaEncoder:
    """Um sistema novo só com as `quantas` primeiras camadas do corpo: saem as de cima.

    Sajjad et al. (arXiv 2004.03844) tiram 2, 4 e 6 das 12 camadas de BERT,
    RoBERTa e XLNet, e tirar as de cima é a forma que melhor se sai. Cada
    camada do XLM-R termina numa normalização, então a saída da camada
    `quantas` tem a mesma forma que a da última: as cabeças passam a ler
    dela. O sistema recebido não muda.
    """
    corpo = sistema.modelo.encoder.encoder
    todas = corpo.layer
    if not 1 <= quantas <= len(todas):
        raise ValueError(f"O corpo tem {len(todas)} camadas; não dá para ficar com {quantas}.")
    corpo.layer = nn.ModuleList(list(todas)[:quantas])   # por um momento, para a cópia não levar as camadas que saem
    try:
        copia = copy.deepcopy(sistema.modelo)
    finally:
        corpo.layer = todas
    copia.encoder.config.num_hidden_layers = quantas
    return SistemaEncoder(copia, sistema.tokenizador, sistema.intents, sistema.tags, sistema.dispositivo, sistema.tamanho_do_lote)


# --------------------------------------------------------------------------
# O reajuste: o laço de ajuste fino com a poda dentro
# --------------------------------------------------------------------------


def reajustar(
    sistema: SistemaEncoder,
    treino: pd.DataFrame,
    dev: pd.DataFrame,
    receita: Receita = Receita(),
    semente: int = 0,
    mascaras: Mascaras | None = None,
    agenda: Agenda | None = None,
    partida_concorre: bool = True,
) -> pd.DataFrame:
    """Ajusta `sistema` no lugar, com a poda dentro do laço, e o deixa com os pesos do melhor ponto em `dev`.

    É o laço de `treino_encoder.treinar`, passo por passo (mesma receita,
    mesma agenda da taxa, mesmos sorteios), com os três usos de
    `poda.reajustar`:

    - `mascaras`: reajuste depois da poda de uma vez (Han et al.). A máscara
      fica travada: depois de cada passo do otimizador, os pesos podados
      voltam a zero.
    - `agenda`: poda gradual (Zhu & Gupta). A máscara é refeita a cada
      `passos_entre_cortes` passos, com a esparsidade da curva cúbica, até o
      fim da rampa; daí em diante fica travada. Só as épocas em que a rede já
      está na esparsidade final concorrem a melhor ponto.
    - nenhum dos dois: o controle sem poda e as redes de que saíram peças
      inteiras. Com `partida_concorre=False` é exatamente
      `treino_encoder.treinar` (um teste confere).

    `partida_concorre`: o modelo como chegou (já podado, antes de qualquer
    passo) concorre a melhor ponto, como "época 0". Se nenhuma época do
    reajuste for melhor que ele em `dev`, o reajuste devolve o modelo como
    entrou. Não vale para a poda gradual, em que o ponto de partida ainda não
    está na esparsidade final.

    Devolve o histórico: uma linha por época, com as perdas de treino, a
    esparsidade medida no fim da época e as métricas em `dev`.
    """
    if mascaras is not None and agenda is not None:
        raise ValueError("Ou a máscara já vem pronta (`mascaras`) ou é feita aos poucos (`agenda`): não as duas.")
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo = sistema.modelo
    na_placa = str(sistema.dispositivo).startswith("cuda")
    mista = receita.precisao_mista and na_placa

    if receita.tabela_dividida:
        sistema.preparar_treino(treino)
    tabela = modelo.tabela_dividida

    # decaimento só nas matrizes de pesos, não em vieses nem em normalizações
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    grupos = [
        {"params": [p for p in parametros if p.ndim >= 2], "weight_decay": receita.decaimento},
        {"params": [p for p in parametros if p.ndim < 2], "weight_decay": 0.0},
    ]
    otimizador = torch.optim.AdamW(grupos, lr=receita.taxa, fused=na_placa)
    passos_por_epoca = receita.passos_da_epoca(len(treino))
    total = receita.epocas * passos_por_epoca
    agenda_da_taxa = torch.optim.lr_scheduler.LambdaLR(otimizador, lambda passo: fracao_da_taxa(passo, total, receita.aquecimento))

    mascaras = dict(mascaras or {})
    aplicar(modelo, mascaras)
    fim_da_rampa = 0
    if agenda is not None:
        cortes = agenda.epocas_de_rampa * passos_por_epoca // agenda.passos_entre_cortes
        fim_da_rampa = cortes * agenda.passos_entre_cortes
        if cortes < 1 or agenda.epocas_de_rampa > receita.epocas:
            raise ValueError("A rampa da poda gradual precisa caber no reajuste e ter pelo menos um corte.")

    rotulos_dev, textos_dev = ouro(dev), list(dev["utt"])
    lotes = fila_de_lotes(len(treino), receita.lote, rng)
    historico, melhor_valor, melhor_estado = [], -1.0, None
    inicio = time.perf_counter()
    if partida_concorre and agenda is None:
        metricas = qualidade.avaliar(rotulos_dev, sistema.prever(textos_dev))
        melhor_valor, melhor_estado = metricas[receita.criterio], modelo.copiar_estado()
        historico.append({"epoca": 0, "lote": receita.lote, "passos": 0, "esparsidade": contar(modelo)["esparsidade"],
                          **{f"dev_{nome}": valor for nome, valor in metricas.items()}, "melhor": True, "segundos": 0.0})
    passo = 0
    for epoca in range(receita.epocas):
        modelo.train()
        somas: dict[str, torch.Tensor] = {}
        for _ in range(passos_por_epoca):
            lote = sistema.lote(treino.iloc[next(lotes)])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mista):
                perdas = modelo.perda(lote)
            otimizador.zero_grad(set_to_none=True)
            perdas["total"].backward()
            if receita.norma_maxima > 0:
                torch.nn.utils.clip_grad_norm_(parametros, receita.norma_maxima)
            taxa_do_passo = otimizador.param_groups[0]["lr"]
            otimizador.step()
            if tabela is not None:   # as linhas sem gradiente sofrem só o decaimento de pesos
                tabela.decair(1 - taxa_do_passo * receita.decaimento)
            agenda_da_taxa.step()
            passo += 1
            # --- a poda dentro do laço ---
            if agenda is not None and passo <= fim_da_rampa and passo % agenda.passos_entre_cortes == 0:
                alvo = esparsidade_cubica(passo, fim_da_rampa, agenda.esparsidade)
                mascaras = mascaras_por_magnitude(modelo, alvo, anteriores=mascaras or None)
            aplicar(modelo, mascaras)   # máscara travada: o que saiu volta a zero depois de cada passo
            # ------------------------------
            for nome, valor in perdas.items():
                somas[nome] = somas.get(nome, 0.0) + valor.detach()

        metricas = qualidade.avaliar(rotulos_dev, sistema.prever(textos_dev))
        linha = {"epoca": epoca + 1, "lote": receita.lote, "passos": passos_por_epoca, "taxa": taxa_do_passo,
                 "esparsidade": contar(modelo)["esparsidade"],
                 **{f"perda_{nome}": float(soma) / passos_por_epoca for nome, soma in somas.items()},
                 **{f"dev_{nome}": valor for nome, valor in metricas.items()}}
        na_esparsidade_final = passo >= fim_da_rampa   # na poda gradual, as épocas da rampa não concorrem
        if na_esparsidade_final and metricas[receita.criterio] > melhor_valor:
            melhor_valor, melhor_estado = metricas[receita.criterio], modelo.copiar_estado()
            linha["melhor"] = True
        linha["segundos"] = time.perf_counter() - inicio
        historico.append(linha)

    modelo.load_state_dict(melhor_estado, strict=False)
    sistema.concluir_treino()
    modelo.eval()
    historico = pd.DataFrame(historico)
    historico["melhor"] = historico["epoca"] == historico.loc[historico["melhor"].fillna(False).astype(bool), "epoca"].max()
    return historico


# --------------------------------------------------------------------------
# Armazenamento: o arquivo comum, o arquivo com zip e o arquivo que aproveita os zeros
# --------------------------------------------------------------------------

ARQUIVOS_DA_REDE = ("encoder/model.safetensors", "cabecas.pt")   # os pesos; o resto da pasta é tokenizador e configuração


def salvar_esparso(sistema: SistemaEncoder, pasta: Path) -> Path:
    """Grava o sistema com os pesos no arquivo esparso de `poda.salvar_esparso` (um bit por posição e só os não nulos).

    Ao lado ficam o tokenizador e as configurações, como em
    `SistemaEncoder.salvar`. `carregar_esparso` lê tudo de volta, com os
    mesmos números.
    """
    sistema.modelo.fundir_tabela()
    arquivo = poda.salvar_esparso(sistema, pasta)
    sistema.modelo.encoder.config.save_pretrained(pasta / "encoder")
    sistema.tokenizador.save_pretrained(pasta / "encoder")
    (pasta / "config.json").write_text(
        json.dumps({"config": sistema.modelo.config.como_dict(), "intents": sistema.intents, "tags": sistema.tags},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return arquivo


def carregar_esparso(pasta: Path, dispositivo: str = "cpu") -> SistemaEncoder:
    """Monta o sistema a partir de `pesos_esparsos.npz`, das configurações e do tokenizador (não lê `model.safetensors`)."""
    import transformers

    dados = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    config = transformers.AutoConfig.from_pretrained(pasta / "encoder")
    encoder = transformers.MODEL_MAPPING[type(config)](config, add_pooling_layer=False)
    tokenizador = transformers.AutoTokenizer.from_pretrained(pasta / "encoder")
    modelo = EncoderConjunto(Config(**dados["config"]), encoder, len(dados["intents"]), len(dados["tags"]))
    modelo.load_state_dict(poda.ler_esparso(pasta / poda.ARQUIVO_ESPARSO))
    return SistemaEncoder(modelo.eval(), tokenizador, dados["intents"], dados["tags"], dispositivo)


def bytes_com_zip(caminho: Path, pedaco: int = 1 << 24) -> int:
    """O mesmo número de `poda.bytes_com_zip` (deflate, nível 6), lendo o arquivo aos pedaços: o do XLM-R tem mais de 1 GB."""
    compressor, total = zlib.compressobj(6), 0
    with open(caminho, "rb") as arquivo:
        while bloco := arquivo.read(pedaco):
            total += len(compressor.compress(bloco))
    return total + len(compressor.flush())


def tamanhos(pasta: Path, arquivo_esparso: Path | None = None) -> dict[str, dict[str, int]]:
    """O tamanho em disco de um sistema gravado por `SistemaEncoder.salvar`, com a rede separada do resto.

    - `rede`: os pesos (`encoder/model.safetensors` e `cabecas.pt`), no
      arquivo comum, no arquivo comum com zip e, se `arquivo_esparso` for
      dado, no arquivo que só grava os não nulos;
    - `tokenizador_e_configuracao`: os outros arquivos da pasta, sem os quais
      o sistema não recebe texto;
    - `sistema`: a soma dos dois. É o que precisa estar no disco.
    """
    da_rede = [pasta / nome for nome in ARQUIVOS_DA_REDE]
    outros = sorted(a for a in pasta.rglob("*") if a.is_file() and a not in da_rede)
    rede = {"arquivo_comum": sum(a.stat().st_size for a in da_rede), "arquivo_comum_com_zip": sum(bytes_com_zip(a) for a in da_rede)}
    resto = {"arquivo_comum": sum(a.stat().st_size for a in outros), "arquivo_comum_com_zip": sum(bytes_com_zip(a) for a in outros)}
    if arquivo_esparso is not None:
        rede["arquivo_esparso"] = arquivo_esparso.stat().st_size
        resto["arquivo_esparso"] = resto["arquivo_comum"]   # o tokenizador e as configurações não mudam de formato
    return {"rede": rede, "tokenizador_e_configuracao": resto, "sistema": {formato: rede[formato] + resto[formato] for formato in rede}}


# --------------------------------------------------------------------------
# Execução que aproveita os zeros
# --------------------------------------------------------------------------


def execucao_esparsa(sistema: SistemaEncoder) -> SistemaEncoder:
    """Uma cópia do sistema, na CPU, em que as matrizes do corpo só fazem as contas dos pesos não nulos (`poda.LinearEsparsa`).

    Serve para medir a latência que a poda daria com uma execução que
    aproveita os zeros. Só para inferência.
    """
    copia = SistemaEncoder(copy.deepcopy(sistema.modelo).to("cpu"), sistema.tokenizador, sistema.intents, sistema.tags, "cpu",
                           sistema.tamanho_do_lote)
    modelo = copia.modelo.eval()
    for nome in nomes_podaveis(modelo):
        pai, _, filho = nome.removesuffix(".weight").rpartition(".")
        dono = modelo.get_submodule(pai)
        setattr(dono, filho, poda.LinearEsparsa(getattr(dono, filho)))
    return copia
