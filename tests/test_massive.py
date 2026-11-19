"""Conversão anotação <-> BIO: a peça da Etapa 1 que não pode errar calada."""

import pytest

from especialista.bio import bio_para_spans
from especialista.dados.massive import anotacao_para_bio, bio_para_anotacao


def test_exemplo_do_artigo():
    tokens, tags = anotacao_para_bio("acorda-me às [time : cinco da manhã] durante [date : esta semana]")
    assert tokens == ["acorda-me", "às", "cinco", "da", "manhã", "durante", "esta", "semana"]
    assert tags == ["O", "O", "B-time", "I-time", "I-time", "O", "B-date", "I-date"]


def test_sem_entidade():
    assert anotacao_para_bio("que horas são") == (["que", "horas", "são"], ["O", "O", "O"])


def test_entidades_vizinhas_do_mesmo_tipo_ficam_separadas():
    tokens, tags = anotacao_para_bio("[person : ana] [person : rui] chegaram")
    assert tags == ["B-person", "B-person", "O"]
    assert bio_para_spans(tags) == [("person", 0, 1), ("person", 1, 2)]


def test_spans():
    tags = ["O", "B-time", "I-time", "O", "B-date"]
    assert bio_para_spans(tags) == [("time", 1, 3), ("date", 4, 5)]


@pytest.mark.parametrize(
    "anotacao",
    [
        "acorda-me às [time : nove da manhã] na [date : sexta-feira]",
        "liga máquina do café",
        "[place_name : lisboa]",
        "toca [artist_name : tony carreira] na [device_type : coluna] da [house_place : sala]",
    ],
)
def test_ida_e_volta(anotacao):
    assert bio_para_anotacao(*anotacao_para_bio(anotacao)) == anotacao
