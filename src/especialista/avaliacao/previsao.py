"""O contrato entre os sistemas e a régua.

Qualquer sistema recebe o texto do enunciado e devolve uma `Previsao`:
a intenção e as entidades como spans. A régua não sabe se quem respondeu
foi uma rede, uma tabela ou uma LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:   # pandas e pyarrow só são carregados por quem lê ou grava tabelas
    import pandas as pd

from especialista.bio import Span, bio_para_spans, rotulos_para_spans


@dataclass(frozen=True)
class Previsao:
    """A resposta de um sistema para um enunciado."""

    intent: str
    slots: tuple[Span, ...] = ()

    @classmethod
    def de_tags(cls, intent: str, tags: list[str]) -> "Previsao":
        """A partir de tags BIO, uma por palavra."""
        return cls(intent, tuple(bio_para_spans(tags)))

    @classmethod
    def de_rotulos(cls, intent: str, rotulos: list[str]) -> "Previsao":
        """A partir de rótulos simples, um por palavra (convenção do MASSIVE)."""
        return cls(intent, tuple(rotulos_para_spans(rotulos)))


class Sistema(Protocol):
    """Tudo o que a régua exige de um sistema."""

    nome: str

    def prever(self, textos: list[str]) -> list[Previsao]: ...


def ouro(df: pd.DataFrame) -> list[Previsao]:
    """Os rótulos humanos de uma partição, no mesmo formato das previsões."""
    return [Previsao.de_tags(intent, list(tags)) for intent, tags in zip(df["intent"], df["tags"])]


# --------------------------------------------------------------------------
# Guardar previsões: permite comparar sistemas depois sem rodá-los de novo
# --------------------------------------------------------------------------

def _esquema():
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.int64()),
            ("intent", pa.string()),
            ("slots", pa.list_(pa.struct([("tipo", pa.string()), ("inicio", pa.int32()), ("fim", pa.int32())]))),
        ]
    )


def salvar_previsoes(caminho: Path, ids, previsoes: list[Previsao]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    caminho.parent.mkdir(parents=True, exist_ok=True)
    tabela = pa.table(
        {
            "id": [int(i) for i in ids],
            "intent": [p.intent for p in previsoes],
            "slots": [[{"tipo": t, "inicio": i, "fim": f} for t, i, f in p.slots] for p in previsoes],
        },
        schema=_esquema(),
    )
    pq.write_table(tabela, caminho)
    return caminho


def carregar_previsoes(caminho: Path) -> tuple[list[int], list[Previsao]]:
    import pyarrow.parquet as pq

    linhas = pq.read_table(caminho).to_pylist()
    ids = [linha["id"] for linha in linhas]
    previsoes = [
        Previsao(linha["intent"], tuple((s["tipo"], s["inicio"], s["fim"]) for s in linha["slots"]))
        for linha in linhas
    ]
    return ids, previsoes
