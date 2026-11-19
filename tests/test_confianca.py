"""Calibração, acerto entre as k mais prováveis e entropia, em exemplos feitos à mão."""

import numpy as np
import pytest

from especialista.avaliacao import confianca


def test_probabilidades_somam_um_e_preservam_a_ordem():
    prob = confianca.probabilidades(np.array([[-1.0, -2.0, -900.0], [0.0, 0.0, 0.0]]))
    assert prob.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert prob[0, 0] > prob[0, 1] > prob[0, 2] and prob[1] == pytest.approx([1 / 3] * 3)


def test_calibracao_perfeita_e_exagerada():
    # 10 pedidos com 80% de confiança e 8 acertos: calibrado. Outros 10 com 95% e 5 acertos: confiante demais.
    conf = np.array([0.8] * 10 + [0.95] * 10)
    acertou = np.array([True] * 8 + [False] * 2 + [True] * 5 + [False] * 5)
    faixas = confianca.faixas_de_confianca(conf, acertou)
    assert faixas["pedidos"].tolist() == [10, 10] and faixas["acuracia"].tolist() == pytest.approx([0.8, 0.5])
    assert confianca.erro_de_calibracao(faixas) == pytest.approx((0 * 10 + 0.45 * 10) / 20)
    assert confianca.faixas_de_confianca(np.array([1.0]), np.array([True]))["ate"].iloc[0] == 1.0     # 100% cai na última faixa


def test_acerto_entre_as_k_e_entropia():
    prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    acerto = confianca.acerto_entre_as_k(prob, np.array([0, 2, 0]), ks=(1, 2, 3))
    assert acerto.to_dict() == pytest.approx({1: 1 / 3, 2: 2 / 3, 3: 1.0})
    assert confianca.entropia(np.array([[1.0, 0.0], [0.5, 0.5]])) == pytest.approx([0.0, 1.0])
