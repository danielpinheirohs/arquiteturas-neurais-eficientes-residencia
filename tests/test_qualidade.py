"""A régua de qualidade: exemplos feitos à mão e conferência contra a conta oficial."""

import pytest

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao
from especialista.avaliacao.referencia import avaliar_como_massive, corromper
from especialista.bio import bio_para_rotulos, rotulos_para_spans, spans_para_bio

OURO = [
    Previsao("alarm_set", (("time", 2, 5), ("date", 6, 8))),
    Previsao("iot_coffee", ()),
    Previsao("play_music", (("artist_name", 1, 3),)),
    Previsao("alarm_set", (("time", 0, 1),)),
]


def test_oraculo_tira_nota_maxima():
    notas = qualidade.avaliar(OURO, OURO)
    assert all(valor == 1.0 for valor in notas.values())


def test_exemplo_feito_a_mao():
    prev = [
        Previsao("alarm_set", (("time", 2, 5), ("date", 6, 7))),  # limite errado em `date`
        Previsao("iot_coffee", ()),                                 # tudo certo
        Previsao("play_radio", (("artist_name", 1, 3),)),          # intenção errada
        Previsao("alarm_set", (("timeofday", 0, 1),)),             # tipo errado
    ]
    notas = qualidade.avaliar(OURO, prev)
    assert notas["intent_acc"] == 3 / 4
    # entidades: 4 no ouro, 4 previstas, 2 certas (time e artist_name)
    assert notas["slot_precision"] == notas["slot_recall"] == notas["slot_f1"] == 0.5
    # só o segundo enunciado está inteiro certo
    assert notas["exact_match"] == 1 / 4
    # macro-F1: alarm_set 1.0, iot_coffee 1.0, play_music 0.0 -> média 2/3
    assert notas["intent_macro_f1"] == pytest.approx(2 / 3)


def test_intencao_fora_do_esquema_conta_como_erro():
    prev = [Previsao("inventada", o.slots) for o in OURO]
    notas = qualidade.avaliar(OURO, prev)
    assert notas["intent_acc"] == 0 and notas["intent_macro_f1"] == 0 and notas["exact_match"] == 0
    assert notas["slot_f1"] == 1.0


def test_sem_entidade_prevista_nao_quebra():
    notas = qualidade.avaliar(OURO, [Previsao(o.intent) for o in OURO])
    assert notas["slot_f1"] == 0.0 and notas["intent_acc"] == 1.0


def test_tamanhos_diferentes_dao_erro():
    with pytest.raises(ValueError):
        qualidade.avaliar(OURO, OURO[:-1])


def test_rotulos_simples_e_bio_dao_os_mesmos_spans():
    spans = [("time", 2, 5), ("date", 6, 8)]
    tags = spans_para_bio(spans, 9)
    assert rotulos_para_spans(bio_para_rotulos(tags)) == spans


@pytest.mark.parametrize("taxa", [0.0, 0.05, 0.3, 0.8])
@pytest.mark.parametrize("semente", [0, 1, 2])
def test_igual_a_conta_oficial_do_massive(taxa, semente):
    intents = ["alarm_set", "iot_coffee", "play_music", "weather_query"]
    tipos = ["time", "date", "artist_name"]
    base = [
        ("alarm_set", ["O", "O", "time", "time", "O", "date"]),
        ("iot_coffee", ["O", "O", "O"]),
        ("play_music", ["O", "artist_name", "artist_name", "O"]),
        ("weather_query", ["O", "O", "date", "O", "time"]),
    ] * 25
    ruidoso = corromper(base, taxa, intents, tipos, semente)

    oficial = avaliar_como_massive(base, ruidoso)
    nossa = qualidade.avaliar(
        [Previsao.de_rotulos(i, r) for i, r in base],
        [Previsao.de_rotulos(i, r) for i, r in ruidoso],
    )
    for metrica, valor in oficial.items():
        assert nossa[metrica] == pytest.approx(valor, abs=1e-12), metrica


def test_bootstrap_pareado_usa_os_mesmos_sorteios():
    a = qualidade.reamostrar(OURO, OURO, n=20, semente=7)
    b = qualidade.reamostrar(OURO, OURO, n=20, semente=7)
    assert a.equals(b)
    comparacao = qualidade.comparar(OURO, OURO, OURO, n=50)
    assert (comparacao["diferenca"] == 0).all() and not comparacao["distinguivel"].any()


def test_grupos_de_sementes():
    errado = [Previsao("inventada", o.slots) for o in OURO]
    grupo = qualidade.intervalo_do_grupo(OURO, [OURO, errado], n=50)
    assert grupo.loc["intent_acc", "valor"] == 0.5          # média de 100% e 0%
    assert grupo.loc["slot_f1", "desvio_sementes"] == 0     # as duas acertam todas as entidades

    comparacao = qualidade.comparar_grupos(OURO, [errado, errado], [OURO, OURO], n=50)
    assert comparacao.loc["intent_acc", "diferenca"] == 1.0
    assert comparacao.loc["intent_acc", "distinguivel"] and not comparacao.loc["slot_f1", "distinguivel"]


def test_por_tipo_de_entidade():
    prev = [Previsao(o.intent, tuple(s for s in o.slots if s[0] != "date")) for o in OURO]
    tabela = qualidade.por_tipo_de_entidade(OURO, prev)
    assert tabela.loc["time", "f1"] == 1.0 and tabela.loc["date", "recall"] == 0.0
    assert tabela.loc["time", "exemplos"] == 2


def test_erros_de_entidade_classifica_cada_entidade_dos_rotulos():
    prev = [
        Previsao("alarm_set", (("time", 2, 5), ("date", 6, 7))),            # `date` com limite diferente
        Previsao("iot_coffee", (("coffee_type", 0, 1),)),                    # uma entidade a mais
        Previsao("play_music", (("song_name", 1, 3),)),                      # mesmo trecho, outro tipo
        Previsao("alarm_set", ()),                                           # faltou
    ]
    destinos = qualidade.erros_de_entidade(OURO, prev)
    assert destinos.to_dict() == {
        "certa": 1, "limite diferente": 1, "tipo diferente": 1, "tipo e limite diferentes": 0, "faltou": 1, "a mais": 1,
    }
    assert destinos[list(qualidade.DESTINOS)].sum() == sum(len(o.slots) for o in OURO)

    so_toca = [Previsao("alarm_set", (("date", 4, 7),)), *OURO[1:]]          # toca `time` e `date`, sem acertar nenhuma
    destinos = qualidade.erros_de_entidade(OURO, so_toca)
    assert destinos["tipo e limite diferentes"] == 1 and destinos["limite diferente"] == 1 and destinos["a mais"] == 0
