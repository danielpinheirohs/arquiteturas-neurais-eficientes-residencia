"""Entidades como trechos (spans) e as duas formas de rotular palavra por palavra.

Um span é `(tipo, início, fim)`, em índices de palavra, com fim exclusivo.
É a forma neutra: serve para um modelo que rotula palavras e para uma LLM
que devolve trechos de texto.

Duas formas por palavra convertem para spans:

- tags BIO: `B-time I-time O`. É o formato dos dados da Etapa 1.
- rótulos simples: `time time O`. É a convenção do código do MASSIVE:
  palavras vizinhas com o mesmo rótulo formam uma entidade só.
"""

from __future__ import annotations

Span = tuple[str, int, int]


def bio_para_spans(tags: list[str]) -> list[Span]:
    """Agrupa tags BIO em spans.

    Um `I-` sem `B-` antes abre uma entidade (convenção do conlleval e do
    seqeval padrão), para que uma sequência malformada ainda seja avaliável.
    """
    spans: list[Span] = []
    tipo, inicio = None, 0
    for i, tag in enumerate([*tags, "O"]):  # o "O" extra fecha a última entidade
        continua = tag.startswith("I-") and tag[2:] == tipo
        if tipo is not None and not continua:
            spans.append((tipo, inicio, i))
            tipo = None
        if tag.startswith("B-") or (tag.startswith("I-") and tipo is None):
            tipo, inicio = tag[2:], i
    return spans


def spans_para_bio(spans: list[Span], n_palavras: int) -> list[str]:
    """Caminho de volta: spans (sem sobreposição) para tags BIO."""
    tags = ["O"] * n_palavras
    for tipo, inicio, fim in spans:
        tags[inicio] = f"B-{tipo}"
        for i in range(inicio + 1, fim):
            tags[i] = f"I-{tipo}"
    return tags


def rotulos_para_spans(rotulos: list[str], fora: str = "O") -> list[Span]:
    """Agrupa rótulos simples em spans: cada sequência de rótulos iguais é uma entidade."""
    spans: list[Span] = []
    inicio = 0
    for i in range(1, len(rotulos) + 1):
        if i == len(rotulos) or rotulos[i] != rotulos[inicio]:
            if rotulos[inicio] != fora:
                spans.append((rotulos[inicio], inicio, i))
            inicio = i
    return spans


def bio_para_rotulos(tags: list[str]) -> list[str]:
    """Tira o prefixo `B-`/`I-`: `B-time I-time O` vira `time time O`."""
    return [tag if tag == "O" else tag[2:] for tag in tags]
