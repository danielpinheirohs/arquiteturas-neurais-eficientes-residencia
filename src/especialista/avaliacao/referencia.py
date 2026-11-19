"""A conta oficial, para conferir a nossa.

Reproduz o que o código de avaliação do MASSIVE faz (`eval_preds`, em
github.com/alexa/massive): rótulos simples por palavra viram BIO juntando
vizinhos iguais, o F1 de entidades vem do `seqeval` e o exact match exige
intenção e sequência inteira iguais. O macro-F1 vem do scikit-learn.

Só é usada em testes e no notebook da Etapa 2. A régua do projeto é
`qualidade.avaliar`, que precisa dar exatamente os mesmos números.
"""

from __future__ import annotations

import numpy as np
from seqeval.metrics import f1_score as f1_seqeval
from sklearn.metrics import accuracy_score, f1_score

Enunciado = tuple[str, list[str]]  # (intenção, rótulos simples por palavra)


def _para_bio(rotulos: list[str]) -> list[str]:
    bio, anterior = [], "O"
    for rotulo in rotulos:
        if rotulo == "O":
            bio.append("O")
        else:
            bio.append(("I-" if rotulo == anterior else "B-") + rotulo)
        anterior = rotulo
    return bio


def avaliar_como_massive(ouro: list[Enunciado], prev: list[Enunciado]) -> dict[str, float]:
    intents_ouro = [i for i, _ in ouro]
    intents_prev = [i for i, _ in prev]
    bio_ouro = [_para_bio(r) for _, r in ouro]
    bio_prev = [_para_bio(r) for _, r in prev]
    return {
        "intent_acc": float(accuracy_score(intents_ouro, intents_prev)),
        "intent_macro_f1": float(
            f1_score(intents_ouro, intents_prev, labels=sorted(set(intents_ouro)), average="macro", zero_division=0)
        ),
        "slot_f1": float(f1_seqeval(bio_ouro, bio_prev, zero_division=0)),
        "exact_match": float(
            np.mean([io == ip and bo == bp for io, ip, bo, bp in zip(intents_ouro, intents_prev, bio_ouro, bio_prev)])
        ),
    }


def corromper(
    enunciados: list[Enunciado], taxa: float, intents: list[str], tipos: list[str], semente: int = 0
) -> list[Enunciado]:
    """Estraga uma fração dos rótulos ao acaso: gera "previsões" com erros de todo tipo."""
    rng = np.random.default_rng(semente)
    opcoes = ["O", *tipos]
    saida = []
    for intent, rotulos in enunciados:
        if rng.random() < taxa:
            intent = str(rng.choice(intents))
        rotulos = [str(rng.choice(opcoes)) if rng.random() < taxa else r for r in rotulos]
        saida.append((intent, rotulos))
    return saida
