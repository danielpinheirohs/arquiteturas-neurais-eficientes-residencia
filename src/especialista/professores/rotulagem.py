"""Das respostas gravadas de uma LLM para um conjunto rotulado por ela.

Serve às duas LLMs: cada uma grava um registro por pedido (`id`, `texto`
da resposta, ...), e o caminho daí até os rótulos é o mesmo.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from especialista.avaliacao.previsao import Previsao
from especialista.bio import spans_para_bio
from especialista.professores import resposta


def ler_conjunto(
    registros: dict[str, dict], dados: pd.DataFrame, prefixo: str, intents: list[str], tipos: list[str]
) -> tuple[list[Previsao], Counter, list[dict]]:
    """As respostas de um conjunto, na ordem de `dados`.

    Devolve as previsões, o diagnóstico somado da leitura e os registros
    alinhados. Um pedido sem registro conta como `sem_registro` e vira uma
    resposta vazia, que a régua trata como erro.
    """
    previsoes, diagnostico, alinhados = [], Counter(), []
    for linha in dados.itertuples():
        registro = registros.get(f"{prefixo}-{linha.id}")
        if registro is None:
            diagnostico["sem_registro"] += 1
            registro = {"id": f"{prefixo}-{linha.id}", "texto": None, "parada": None, "uso": {}, "ms": None, "erro": "sem registro"}
        previsao, do_pedido = resposta.ler(registro["texto"], linha.utt, intents, tipos)
        previsoes.append(previsao)
        diagnostico.update(do_pedido)
        alinhados.append(registro)
    return previsoes, diagnostico, alinhados


def rotular(dados: pd.DataFrame, previsoes: list[Previsao], intents: list[str]) -> pd.DataFrame:
    """O conjunto com os rótulos da LLM no lugar dos humanos, nas colunas que o treino lê.

    `valido` é falso quando a resposta não pôde ser lida ou trouxe uma
    intenção fora da lista: esse exemplo não tem como ensinar o aluno.
    """
    return pd.DataFrame(
        {
            "id": dados["id"].to_numpy(),
            "utt": dados["utt"].to_numpy(),
            "tokens": [list(t) for t in dados["tokens"]],
            "intent": [p.intent for p in previsoes],
            "tags": [spans_para_bio(list(p.slots), len(t)) for p, t in zip(previsoes, dados["tokens"])],
            "valido": [p.intent in intents for p in previsoes],
        }
    )
