"""Etapa 4b: candidatas a recuperar a acurácia de intenção.

O controle é a variante `padrao` da Etapa 4. Cada candidata muda UMA coisa.

    uv run python -m especialista.experimentos.etapa4b --variante atalho_features --semente 0
"""

from __future__ import annotations

from pathlib import Path

from especialista.experimentos import execucao

ETAPA = "04b_intencao"
SEMENTES = (0, 1, 2)

VARIANTES: dict[str, dict] = {
    # --- a hipótese original: a intenção precisa de acesso direto aos features ---
    "atalho_resumo": {
        "nome": "Atalho a partir do resumo (128 dimensões)",
        "hipotese": "O encoder e o espaço de 20 dimensões atrapalham a intenção",
        "config": {"atalho_intencao": "resumo"}, "receita": {},
    },
    "atalho_features": {
        "nome": "Atalho a partir dos features (classificador linear dentro da rede)",
        "hipotese": "Falta um caminho linear sobre os n-gramas, como o da linha de base",
        "config": {"atalho_intencao": "features"}, "receita": {},
    },
    "rotulo_64": {
        "nome": "Espaço de rótulos com 64 dimensões (o padrão é 20)",
        "hipotese": "O espaço de 20 dimensões é um gargalo para 60 intenções",
        "config": {"d_rotulo": 64}, "receita": {},
    },
    # --- a hipótese que saiu do diagnóstico: a rede varia demais de um treino para outro ---
    "media_dos_pesos": {
        "nome": "Média móvel dos pesos (fator 0,999)",
        "hipotese": "Os pesos do último passo oscilam; a média dos passos recentes generaliza melhor",
        "config": {}, "receita": {"media_dos_pesos": 0.999},
    },
    "suavizacao": {
        "nome": "Suavização de rótulos na intenção (0,1)",
        "hipotese": "A rede fica confiante demais no treino; suavizar o alvo reduz o sobreajuste",
        "config": {"suavizacao": 0.1}, "receita": {},
    },
}


def executar(variante: str, semente: int, definicao: dict | None = None) -> Path:
    return execucao.executar(ETAPA, variante, semente, definicao or VARIANTES[variante])


def carregar(variante: str, semente: int) -> dict:
    return execucao.carregar(ETAPA, variante, semente)


if __name__ == "__main__":
    execucao.linha_de_comando(ETAPA, VARIANTES)
