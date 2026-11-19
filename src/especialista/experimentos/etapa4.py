"""Etapa 4: as variantes de treino com rótulos humanos.

Cada variante muda UMA coisa em relação ao padrão. Rodar pela linha de comando
(útil para várias em paralelo):

    uv run python -m especialista.experimentos.etapa4 --variante padrao --semente 0
"""

from __future__ import annotations

from pathlib import Path

from especialista.experimentos import execucao

ETAPA = "04_treino"
SEMENTES = (0, 1, 2)

VARIANTES: dict[str, dict] = {
    "padrao": {
        "nome": "Padrão (receita do artigo)",
        "config": {}, "receita": {},
    },
    "sem_mascara": {
        "nome": "Sem a perda de máscara",
        "config": {"peso_mascara": 0.0}, "receita": {},
    },
    "so_intencao": {
        "nome": "Só intenção (sem a tarefa de entidades)",
        "config": {"peso_entidades": 0.0}, "receita": {"criterio": "intent_acc"},
    },
    "so_entidades": {
        "nome": "Só entidades (sem a tarefa de intenção)",
        "config": {"peso_intencao": 0.0}, "receita": {"criterio": "slot_f1"},
    },
    "sem_crf": {
        "nome": "Sem CRF (cada palavra escolhe a sua tag)",
        "config": {"usar_crf": False}, "receita": {},
    },
    "sem_balanceamento": {
        "nome": "Sem balancear as intenções nos lotes",
        "config": {}, "receita": {"minimo_por_intencao": 0},
    },
    "mais_dropout": {
        "nome": "Dropout de 0,4 (o padrão é 0,2)",
        "config": {"dropout": 0.4, "dropout_features": 0.4}, "receita": {},
    },
}


def executar(variante: str, semente: int, definicao: dict | None = None) -> Path:
    """Treina a variante (ou devolve a pasta, se já rodou). `definicao` só é preciso para variantes fora da lista."""
    return execucao.executar(ETAPA, variante, semente, definicao or VARIANTES[variante])


def carregar(variante: str, semente: int) -> dict:
    return execucao.carregar(ETAPA, variante, semente)


if __name__ == "__main__":
    execucao.linha_de_comando(ETAPA, VARIANTES)
