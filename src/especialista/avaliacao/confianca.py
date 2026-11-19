"""O que as probabilidades de um sistema dizem além da resposta escolhida.

Um sistema que devolve uma probabilidade para cada intenção entrega mais do
que a intenção escolhida: diz o quanto confia nela e quais são as outras
candidatas. É essa informação a mais que a destilação com logits aproveita
(Hinton et al., arXiv 1503.02531). Três medidas:

- **Calibração**: quando o sistema diz 90%, acerta 90% das vezes? Segue Guo
  et al. (arXiv 1706.04599): os pedidos são agrupados em faixas de
  confiança e, em cada faixa, compara-se a confiança média com a acurácia.
  O erro de calibração esperado (ECE) é a média dessas distâncias, pesada
  pelo tamanho da faixa.
- **Acerto entre as k mais prováveis**: quando a escolhida está errada, a
  certa está logo atrás?
- **Entropia**: o quanto a probabilidade está espalhada entre as intenções.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def probabilidades(logp: np.ndarray) -> np.ndarray:
    """Log-probabilidades (pedidos, intenções) para probabilidades que somam 1 em cada pedido."""
    logp = np.asarray(logp, dtype=np.float64)
    logp = logp - logp.max(axis=1, keepdims=True)
    prob = np.exp(logp)
    return prob / prob.sum(axis=1, keepdims=True)


def faixas_de_confianca(confianca: np.ndarray, acertou: np.ndarray, n_faixas: int = 10) -> pd.DataFrame:
    """Uma linha por faixa de confiança de mesma largura: quantos pedidos, confiança média e acurácia.

    Faixas sem nenhum pedido ficam fora.
    """
    confianca, acertou = np.asarray(confianca, dtype=float), np.asarray(acertou, dtype=bool)
    limites = np.linspace(0.0, 1.0, n_faixas + 1)
    faixa = np.clip(np.digitize(confianca, limites[1:-1]), 0, n_faixas - 1)
    linhas = []
    for f in range(n_faixas):
        dentro = faixa == f
        if dentro.any():
            linhas.append({"de": limites[f], "ate": limites[f + 1], "pedidos": int(dentro.sum()),
                           "confianca": float(confianca[dentro].mean()), "acuracia": float(acertou[dentro].mean())})
    return pd.DataFrame(linhas)


def erro_de_calibracao(faixas: pd.DataFrame) -> float:
    """ECE: a distância entre confiança e acurácia, na média das faixas, pesada pelo número de pedidos."""
    return float(((faixas["confianca"] - faixas["acuracia"]).abs() * faixas["pedidos"]).sum() / faixas["pedidos"].sum())


def acerto_entre_as_k(prob: np.ndarray, certa: np.ndarray, ks: tuple[int, ...] = (1, 2, 3, 5, 10)) -> pd.Series:
    """Em que parte dos pedidos a intenção certa está entre as `k` mais prováveis. `certa` traz o índice da coluna certa."""
    ordem = np.argsort(-np.asarray(prob), axis=1, kind="stable")
    colocacao = np.argmax(ordem == np.asarray(certa)[:, None], axis=1)      # 0 = a mais provável
    return pd.Series({k: float((colocacao < k).mean()) for k in ks}, name="acerto")


def entropia(prob: np.ndarray) -> np.ndarray:
    """Entropia de cada pedido, em bits: 0 quando toda a probabilidade está numa intenção só."""
    prob = np.asarray(prob, dtype=np.float64)
    return -(prob * np.log2(np.clip(prob, 1e-300, None))).sum(axis=1)
