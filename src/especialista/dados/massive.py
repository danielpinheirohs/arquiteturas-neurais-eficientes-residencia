"""Amazon MASSIVE: do tarball oficial até partições prontas para treino.

Referência: FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural
Language Understanding Dataset with 51 Typologically-Diverse Languages"
(arXiv 2204.08582). Licença dos dados: CC BY 4.0.

Fluxo, na ordem em que o notebook da Etapa 1 chama:

    baixar() -> extrair_locale() -> ler_bruto() -> preparar()
             -> construir_rotulos() -> salvar_entrega()

As etapas seguintes usam só `carregar_entrega()`.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from especialista.bio import bio_para_spans
from especialista.caminhos import DADOS_BRUTOS, entrega

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
# Hash do arquivo baixado em 16/11/2026. Congela a versão usada no projeto:
# se a Amazon trocar o arquivo, o download falha em vez de mudar os dados calado.
SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
VERSAO = "1.1"
LOCALE = "pt-PT"
ETAPA = "01_dados"
PARTICOES = ("train", "dev", "test")

# Números de FitzGerald et al. para um locale. Servem de critério de pronto.
PUBLICADO = {"train": 11514, "dev": 2033, "test": 2974, "intents": 60, "slot_types": 55, "scenarios": 18}

# Resultados publicados no teste pt-PT, em % (FitzGerald et al., Tabelas 7, 8 e 9), com a
# margem de 95% informada no artigo. São a régua externa do projeto, com uma diferença a
# lembrar: esses modelos treinaram com os 51 idiomas juntos, não só com pt-PT.
REFERENCIA = {
    "XLM-R Base": {
        "parametros": 270_000_000,
        "intent_acc": (86.7, 1.2), "slot_f1": (73.3, 0.6), "exact_match": (64.6, 1.7),
    },
    "mT5 Base (encoder)": {
        "parametros": 258_000_000,
        "intent_acc": (86.9, 1.2), "slot_f1": (74.9, 0.6), "exact_match": (66.0, 1.7),
    },
    "mT5 Base (texto para texto)": {
        "parametros": 580_000_000,
        "intent_acc": (86.7, 1.2), "slot_f1": (76.6, 0.6), "exact_match": (67.6, 1.7),
    },
}

# Anotação do MASSIVE: "acorda-me às [time : cinco da manhã]"
_SLOT = re.compile(r"\[\s*(?P<slot>[^\]:]+?)\s*:\s*(?P<texto>[^\]]*?)\s*\]")


# --------------------------------------------------------------------------
# 1. Obter o dado bruto
# --------------------------------------------------------------------------

def _sha256(arquivo: Path) -> str:
    h = hashlib.sha256()
    with arquivo.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(destino: Path = DADOS_BRUTOS) -> Path:
    """Baixa o tarball oficial (uma vez) e confere o hash."""
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"amazon-massive-dataset-{VERSAO}.tar.gz"
    if not arquivo.exists():
        urllib.request.urlretrieve(URL, arquivo)
    obtido = _sha256(arquivo)
    if obtido != SHA256:
        raise ValueError(f"Hash inesperado em {arquivo.name}: {obtido}. Apague o arquivo e baixe de novo.")
    return arquivo


def extrair_locale(tarball: Path, locale: str = LOCALE) -> Path:
    """Tira do tarball só o JSONL do locale pedido."""
    saida = tarball.parent / f"massive-{VERSAO}" / f"{locale}.jsonl"
    if saida.exists():
        return saida
    saida.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        membro = tar.extractfile(f"{VERSAO}/data/{locale}.jsonl")
        if membro is None:
            raise FileNotFoundError(f"Locale {locale} não está no tarball.")
        saida.write_bytes(membro.read())
    return saida


def ler_bruto(jsonl: Path) -> pd.DataFrame:
    """Lê o JSONL como veio, sem transformar nada."""
    with jsonl.open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(linha) for linha in f])


# --------------------------------------------------------------------------
# 2. Anotação <-> BIO
# --------------------------------------------------------------------------

def anotacao_para_bio(annot_utt: str) -> tuple[list[str], list[str]]:
    """Converte a anotação do MASSIVE em tokens e tags BIO.

    >>> anotacao_para_bio("acorda-me às [time : cinco da manhã]")
    (['acorda-me', 'às', 'cinco', 'da', 'manhã'], ['O', 'O', 'B-time', 'I-time', 'I-time'])

    O token é a palavra separada por espaço, como no próprio MASSIVE. A quebra
    em pedaços menores (subpalavras, n-gramas) é decisão da arquitetura.
    """
    tokens: list[str] = []
    tags: list[str] = []
    cursor = 0
    for m in _SLOT.finditer(annot_utt):
        for palavra in annot_utt[cursor : m.start()].split():
            tokens.append(palavra)
            tags.append("O")
        for i, palavra in enumerate(m["texto"].split()):
            tokens.append(palavra)
            tags.append(("B-" if i == 0 else "I-") + m["slot"])
        cursor = m.end()
    for palavra in annot_utt[cursor:].split():
        tokens.append(palavra)
        tags.append("O")
    return tokens, tags


def bio_para_anotacao(tokens: list[str], tags: list[str]) -> str:
    """Caminho de volta: reconstrói a anotação do MASSIVE a partir de tokens e tags."""
    partes: list[str] = []
    cursor = 0
    for tipo, inicio, fim in bio_para_spans(tags):
        partes.extend(tokens[cursor:inicio])
        partes.append(f"[{tipo} : {' '.join(tokens[inicio:fim])}]")
        cursor = fim
    partes.extend(tokens[cursor:])
    return " ".join(partes)


# --------------------------------------------------------------------------
# 3. Preparar
# --------------------------------------------------------------------------

def preparar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Do registro bruto para o registro de treino.

    Mantém os nomes de campo do MASSIVE (`utt`, `annot_utt`, `intent`...) e
    acrescenta:

    - `tokens`, `tags`: o enunciado em palavras e as tags BIO alinhadas;
    - `n_tokens`, `n_slots`: tamanho e número de entidades;
    - `intent_agreement`, `slots_agreement`: quantos dos 3 avaliadores do
      MASSIVE disseram que o rótulo está certo;
    - `utt_in_train`: o texto exato aparece na partição de treino (em `train`
      é verdadeiro por definição; interessa em `dev` e `test`).

    Nada é removido nem corrigido: as partições ficam idênticas às publicadas,
    para que os números sejam comparáveis com os da literatura.
    """
    df = bruto[["id", "partition", "scenario", "intent", "utt", "annot_utt"]].copy()
    df["id"] = df["id"].astype(int)

    convertido = df["annot_utt"].map(anotacao_para_bio)
    df["tokens"] = convertido.map(lambda par: par[0])
    df["tags"] = convertido.map(lambda par: par[1])

    desalinhado = df[df["tokens"] != df["utt"].str.split()]
    if len(desalinhado):
        raise ValueError(f"{len(desalinhado)} enunciados em que os tokens da anotação não batem com `utt`.")

    df["n_tokens"] = df["tokens"].map(len)
    df["n_slots"] = df["tags"].map(lambda tags: len(bio_para_spans(tags)))

    julgamentos = bruto["judgments"]
    df["intent_agreement"] = julgamentos.map(lambda js: sum(j["intent_score"] == 1 for j in js))
    # slots_score: 1 = entidades corretas, 2 = enunciado sem entidade. Os dois contam como "de acordo".
    df["slots_agreement"] = julgamentos.map(lambda js: sum(j["slots_score"] in (1, 2) for j in js))

    textos_de_treino = set(df.loc[df["partition"] == "train", "utt"])
    df["utt_in_train"] = df["utt"].isin(textos_de_treino)

    return df.sort_values("id").reset_index(drop=True)


def construir_rotulos(df: pd.DataFrame) -> dict:
    """Inventário de rótulos: o "esquema" que modelo e LLM recebem.

    Usa as três partições do locale, porque o esquema é conhecido de antemão
    (como a lista de ferramentas de um agente), não aprendido dos dados.
    """
    intents = sorted(df["intent"].unique())
    slot_types = sorted({tag[2:] for tags in df["tags"] for tag in tags if tag != "O"})
    tags = ["O"] + [f"{prefixo}-{slot}" for slot in slot_types for prefixo in ("B", "I")]
    scenarios = {
        cenario: sorted(grupo["intent"].unique())
        for cenario, grupo in df.groupby("scenario")
    }
    return {"intents": intents, "slot_types": slot_types, "tags": tags, "scenarios": scenarios}


# --------------------------------------------------------------------------
# 4. Entregar e recarregar
# --------------------------------------------------------------------------

def salvar_entrega(df: pd.DataFrame, rotulos: dict, pasta: Path | None = None) -> Path:
    """Grava uma partição por arquivo Parquet e o inventário de rótulos."""
    pasta = pasta or entrega(ETAPA)
    for particao in PARTICOES:
        parte = df[df["partition"] == particao].reset_index(drop=True)
        parte.to_parquet(pasta / f"{particao}.parquet", index=False)
    (pasta / "rotulos.json").write_text(
        json.dumps(rotulos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return pasta


def carregar_entrega(pasta: Path | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Ponto de entrada das etapas seguintes: `particoes, rotulos = carregar_entrega()`."""
    pasta = pasta or entrega(ETAPA)
    particoes = {}
    for particao in PARTICOES:
        parte = pd.read_parquet(pasta / f"{particao}.parquet")
        for coluna in ("tokens", "tags"):  # o Parquet devolve arrays; o projeto usa listas
            parte[coluna] = parte[coluna].map(list)
        particoes[particao] = parte
    rotulos = json.loads((pasta / "rotulos.json").read_text(encoding="utf-8"))
    return particoes, rotulos
