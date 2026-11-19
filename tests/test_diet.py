"""A rede inteira num conjunto de brinquedo: formas, preenchimento, gradientes, salvar e aprender."""

import pandas as pd
import pytest
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro
from especialista.modelos.diet import Config, SistemaDiet, ngramas

ENUNCIADOS = [
    ("alarm_set", "acorda-me às nove da manhã", "O O B-time I-time I-time"),
    ("alarm_set", "põe um alarme para sexta", "O O O O B-date"),
    ("alarm_set", "acorda-me amanhã às sete", "O B-date O B-time"),
    ("play_music", "toca tony carreira", "O B-artist I-artist"),
    ("play_music", "quero ouvir rui veloso agora", "O O B-artist I-artist O"),
    ("play_music", "toca música", "O O"),
    ("weather_query", "como está o tempo em lisboa", "O O O O O B-place"),
    ("weather_query", "vai chover amanhã no porto", "O O B-date O B-place"),
]
ROTULOS = {
    "intents": ["alarm_set", "play_music", "weather_query"],
    "tags": ["O", "B-artist", "I-artist", "B-date", "I-date", "B-place", "I-place", "B-time", "I-time"],
}
PEQUENO = Config(d_modelo=32, n_cabecas=2, d_ffn=64, d_features=16, d_rotulo=8)


@pytest.fixture
def dados() -> pd.DataFrame:
    return pd.DataFrame(
        {"intent": i, "utt": texto, "tokens": texto.split(), "tags": tags.split()} for i, texto, tags in ENUNCIADOS
    )


@pytest.fixture
def sistema(dados) -> SistemaDiet:
    return SistemaDiet.novo(dados, ROTULOS, PEQUENO, contagem_minima=1)


def test_ngramas():
    assert ngramas("da", n_max=3) == ["d", "a", "^d", "da", "a$", "^da", "da$"]


def test_palavra_nova_usa_os_ngramas_conhecidos(sistema):
    assert "p:amanhãs" not in sistema.featurizador.vocabulario
    assert len(sistema.featurizador.ids("amanhãs")) > 0


def test_formas(sistema, dados):
    lote = sistema.lote(dados)
    intencoes, emissoes = sistema.modelo(lote)
    maior = max(len(t) for t in dados["tokens"])
    assert intencoes.shape == (len(dados), 3)
    assert emissoes.shape == (len(dados), maior, 9)
    assert sum(sistema.modelo.parametros_por_parte().values()) == sistema.parametros


def test_resposta_nao_depende_de_quem_esta_no_lote(sistema, dados):
    """O preenchimento de um lote não pode vazar para os enunciados curtos."""
    sistema.modelo.eval()
    textos = list(dados["utt"])
    juntos = sistema.prever(textos)
    separados = [sistema.prever([texto])[0] for texto in textos]
    assert juntos == separados

    with torch.no_grad():
        no_lote, _ = sistema.modelo(sistema.lote(dados))
        sozinho, _ = sistema.modelo(sistema.lote(dados.iloc[[5]]))  # o mais curto
    assert torch.allclose(no_lote[5], sozinho[0], atol=1e-5)


def test_gradiente_chega_a_todos_os_parametros(sistema, dados):
    torch.manual_seed(0)
    sistema.modelo.train()
    sistema.modelo.perda(sistema.lote(pd.concat([dados] * 4))).get("total").backward()
    sem_gradiente = [nome for nome, p in sistema.modelo.named_parameters() if p.grad is None or not p.grad.any()]
    assert sem_gradiente == []


def test_enunciado_mais_longo_que_qualquer_um_do_treino(sistema):
    previsao = sistema.prever([" ".join(["toca"] * 100)])[0]
    assert previsao.intent in ROTULOS["intents"]


def test_salvar_e_carregar(sistema, dados, tmp_path):
    relido = SistemaDiet.carregar(sistema.salvar(tmp_path / "modelo"))
    assert relido.prever(list(dados["utt"])) == sistema.prever(list(dados["utt"]))
    assert relido.modelo.config == sistema.modelo.config


def test_aprende_um_conjunto_pequeno(sistema, dados):
    """Se a rede não consegue decorar 8 exemplos, há erro na rede, na perda ou no alinhamento dos rótulos."""
    torch.manual_seed(0)
    otimizador = torch.optim.Adam(sistema.modelo.parameters(), lr=3e-3)
    lote = sistema.lote(dados)
    for _ in range(150):
        sistema.modelo.train()
        perda = sistema.modelo.perda(lote)["total"]
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    assert qualidade.avaliar(ouro(dados), sistema.prever(list(dados["utt"])))["exact_match"] == 1.0
