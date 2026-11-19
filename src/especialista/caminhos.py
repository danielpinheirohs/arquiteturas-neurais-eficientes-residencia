"""Onde cada coisa mora no projeto.

Regra do projeto: uma etapa só lê o que está em `entregas/` de etapas
anteriores. `data/` é cache (downloads e execuções de treino): pode ser
apagado, e é recriado rodando a etapa de novo.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DADOS_BRUTOS = RAIZ / "data" / "raw"
EXECUCOES = RAIZ / "data" / "execucoes"
ENTREGAS = RAIZ / "entregas"


def entrega(etapa: str) -> Path:
    """Pasta de entrega de uma etapa, por exemplo `entrega("01_dados")`."""
    pasta = ENTREGAS / etapa
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta
