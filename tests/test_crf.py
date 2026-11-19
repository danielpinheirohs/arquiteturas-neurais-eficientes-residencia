"""O CRF contra a força bruta: em casos pequenos dá para enumerar todas as sequências e conferir."""

import itertools

import pytest
import torch

from especialista.modelos.diet.crf import CRF, restricoes_bio

TAGS = ["O", "B-a", "I-a", "B-b", "I-b"]


def crf_aleatorio(semente: int, com_restricoes: bool) -> CRF:
    torch.manual_seed(semente)
    crf = CRF(len(TAGS), *restricoes_bio(TAGS)) if com_restricoes else CRF(len(TAGS))
    for p in crf.parameters():
        torch.nn.init.normal_(p)
    return crf


def todas_as_sequencias(comprimento: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product(range(len(TAGS)), repeat=comprimento)))


@pytest.mark.parametrize("com_restricoes", [False, True])
@pytest.mark.parametrize("comprimento", [1, 2, 3, 4])
def test_particao_e_viterbi_iguais_a_forca_bruta(comprimento, com_restricoes):
    crf = crf_aleatorio(comprimento, com_restricoes)
    emissoes = torch.randn(1, comprimento, len(TAGS))
    real = torch.ones(1, comprimento, dtype=torch.bool)

    sequencias = todas_as_sequencias(comprimento)
    pontos = crf.pontuacao(emissoes.expand(len(sequencias), -1, -1), sequencias, real.expand(len(sequencias), -1))

    assert crf.particao(emissoes, real)[0].item() == pytest.approx(torch.logsumexp(pontos, 0).item(), abs=1e-4)
    assert crf.decodificar(emissoes, real)[0] == sequencias[pontos.argmax()].tolist()


def test_perda_e_positiva_e_as_probabilidades_somam_um():
    crf = crf_aleatorio(0, com_restricoes=True)
    emissoes = torch.randn(1, 3, len(TAGS))
    real = torch.ones(1, 3, dtype=torch.bool)
    sequencias = todas_as_sequencias(3)
    perdas = crf.perda(emissoes.expand(len(sequencias), -1, -1), sequencias, real.expand(len(sequencias), -1))
    assert (perdas > 0).all()
    assert torch.exp(-perdas).sum().item() == pytest.approx(1.0, abs=1e-4)


def test_preenchimento_nao_muda_o_resultado():
    crf = crf_aleatorio(1, com_restricoes=True)
    curta, longa = torch.randn(1, 2, len(TAGS)), torch.randn(1, 5, len(TAGS))
    lote = torch.zeros(2, 5, len(TAGS))
    lote[0, :2], lote[1] = curta[0], longa[0]
    real = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    tags = torch.tensor([[1, 2, 0, 0, 0], [0, 3, 4, 4, 0]])

    um = torch.ones(1, 2, dtype=torch.bool)
    cinco = torch.ones(1, 5, dtype=torch.bool)
    assert crf.perda(lote, tags, real)[0].item() == pytest.approx(crf.perda(curta, tags[:1, :2], um).item(), abs=1e-4)
    assert crf.perda(lote, tags, real)[1].item() == pytest.approx(crf.perda(longa, tags[1:], cinco).item(), abs=1e-4)
    assert crf.decodificar(lote, real) == [crf.decodificar(curta, um)[0], crf.decodificar(longa, cinco)[0]]


def test_restricoes_bio():
    transicao, inicio = restricoes_bio(TAGS)
    i = {tag: n for n, tag in enumerate(TAGS)}
    assert inicio.tolist() == [False, False, True, False, True]          # não se começa com I-
    assert not transicao[i["B-a"], i["I-a"]] and not transicao[i["I-a"], i["I-a"]]
    assert transicao[i["O"], i["I-a"]] and transicao[i["B-b"], i["I-a"]] and transicao[i["I-b"], i["I-a"]]
    assert not transicao[:, i["O"]].any() and not transicao[:, i["B-a"]].any()  # O e B- podem vir depois de tudo


@pytest.mark.parametrize("semente", range(5))
def test_saida_nunca_e_malformada(semente):
    crf = crf_aleatorio(semente, com_restricoes=True)
    transicao, inicio = restricoes_bio(TAGS)
    caminho = crf.decodificar(torch.randn(1, 12, len(TAGS)) * 5, torch.ones(1, 12, dtype=torch.bool))[0]
    assert not inicio[caminho[0]]
    assert not any(transicao[a, b] for a, b in zip(caminho, caminho[1:]))
