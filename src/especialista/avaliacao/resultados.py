"""A tabela de resultados do projeto: uma linha por sistema e partição, sempre com as mesmas colunas.

Cada etapa grava as suas linhas em `entregas/NN_etapa/resultados.csv`. A
etapa de comparativo junta todas.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from especialista.avaliacao import eficiencia, qualidade
from especialista.avaliacao.previsao import Previsao


def registrar(
    sistema: str,
    particao: str,
    ouro: list[Previsao],
    prev: list[Previsao],
    tempos_ms: np.ndarray | None = None,
    parametros: int | None = None,
    bytes_em_disco: int | None = None,
    **extras,
) -> dict:
    """Monta a linha de um sistema: qualidade com intervalo, eficiência e ambiente.

    `extras` acrescenta colunas próprias da etapa (por exemplo, `threads=1`).
    """
    linha: dict = {"sistema": sistema, "particao": particao, "enunciados": len(ouro)}
    for metrica, valores in qualidade.intervalo(ouro, prev).iterrows():
        linha[metrica] = valores["valor"]
        linha[f"{metrica}_ic_min"] = valores["ic_min"]
        linha[f"{metrica}_ic_max"] = valores["ic_max"]
    if tempos_ms is not None:
        linha.update(eficiencia.resumir_latencia(tempos_ms))
    linha["parametros"] = parametros
    linha["bytes"] = bytes_em_disco
    linha.update(eficiencia.ambiente())
    linha.update(extras)
    linha["data"] = date.today().isoformat()
    return linha


def registrar_grupo(
    sistema: str,
    particao: str,
    ouro: list[Previsao],
    grupo: list[list[Previsao]],
    tempos_ms: np.ndarray | None = None,
    parametros: int | None = None,
    bytes_em_disco: int | None = None,
    **extras,
) -> dict:
    """A linha de um sistema treinado várias vezes: média entre sementes, intervalo da média e desvio entre sementes."""
    linha: dict = {"sistema": sistema, "particao": particao, "enunciados": len(ouro), "sementes": len(grupo)}
    for metrica, valores in qualidade.intervalo_do_grupo(ouro, grupo).iterrows():
        linha[metrica] = valores["valor"]
        linha[f"{metrica}_ic_min"] = valores["ic_min"]
        linha[f"{metrica}_ic_max"] = valores["ic_max"]
        linha[f"{metrica}_desvio_sementes"] = valores["desvio_sementes"]
    if tempos_ms is not None:
        linha.update(eficiencia.resumir_latencia(tempos_ms))
    linha["parametros"] = parametros
    linha["bytes"] = bytes_em_disco
    linha.update(eficiencia.ambiente())
    linha.update(extras)
    linha["data"] = date.today().isoformat()
    return linha


def salvar(linhas: list[dict], pasta: Path) -> Path:
    caminho = pasta / "resultados.csv"
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    return caminho


def carregar(pasta: Path) -> pd.DataFrame:
    return pd.read_csv(pasta / "resultados.csv")
