"""O laço de treino num conjunto de brinquedo: balanceamento, lotes e escolha do melhor ponto."""

import numpy as np
import pandas as pd

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro
from especialista.modelos.diet import Config, SistemaDiet
from especialista.treino import Receita, indices_da_epoca, lotes_por_comprimento, treinar

from test_diet import ENUNCIADOS, PEQUENO, ROTULOS


def dados() -> pd.DataFrame:
    return pd.DataFrame(
        {"intent": i, "utt": texto, "tokens": texto.split(), "tags": tags.split()} for i, texto, tags in ENUNCIADOS
    )


def test_lote_cresce_em_linha_reta():
    receita = Receita(epocas=5, lote_inicial=64, lote_final=128)
    assert [receita.tamanho_do_lote(e) for e in range(5)] == [64, 80, 96, 112, 128]


def test_balanceamento_garante_o_minimo_por_intencao():
    intents = pd.Series(["a"] * 10 + ["b"] * 3 + ["c"] * 1)
    ordem = indices_da_epoca(intents, minimo=5, rng=np.random.default_rng(0))
    vistos = intents.iloc[ordem].value_counts()
    assert vistos.to_dict() == {"a": 10, "b": 5, "c": 5}
    assert set(ordem) == set(range(len(intents)))            # ninguém fica de fora

    sem = indices_da_epoca(intents, minimo=0, rng=np.random.default_rng(0))
    assert sorted(sem) == list(range(len(intents)))          # sem balanceamento: cada exemplo uma vez


def test_lotes_por_comprimento_nao_perdem_nem_repetem_exemplos():
    rng = np.random.default_rng(0)
    comprimentos = rng.integers(1, 30, 500)
    ordem = rng.permutation(500)
    lotes = lotes_por_comprimento(ordem, comprimentos, tamanho=32, rng=rng, lotes_por_grupo=5)
    assert sorted(np.concatenate(lotes)) == list(range(500))
    assert all(len(lote) <= 32 for lote in lotes)
    # lotes mais homogêneos que o acaso: o maior comprimento do lote, em média, cai
    ao_acaso = np.mean([comprimentos[ordem[i : i + 32]].max() for i in range(0, 500, 32)])
    assert np.mean([comprimentos[lote].max() for lote in lotes]) < ao_acaso


def test_treinar_aprende_e_fica_com_o_melhor_ponto():
    d = dados()
    sistema = SistemaDiet.novo(d, ROTULOS, PEQUENO, contagem_minima=1)
    receita = Receita(epocas=60, taxa=3e-3, lote_inicial=4, lote_final=8, minimo_por_intencao=0, avaliar_a_cada=10)
    historico = treinar(sistema, d, d, receita, semente=0)

    assert len(historico) == 60 and historico["melhor"].sum() == 1
    avaliadas = historico.dropna(subset=["dev_exact_match"])
    assert list(avaliadas["epoca"]) == [10, 20, 30, 40, 50, 60]
    assert historico["perda_total"].iloc[-1] < historico["perda_total"].iloc[0]

    # o sistema termina com os pesos da época marcada como melhor
    melhor = avaliadas.loc[avaliadas["melhor"], "dev_exact_match"].iloc[0]
    assert melhor == avaliadas["dev_exact_match"].max()
    assert qualidade.avaliar(ouro(d), sistema.prever(list(d["utt"])))["exact_match"] == melhor


def test_media_dos_pesos_fica_com_os_pesos_medios():
    """Com a média móvel ligada, o sistema termina com pesos diferentes dos do último passo, e ainda aprende."""
    d = dados()
    base = dict(epocas=40, taxa=3e-3, lote_inicial=4, lote_final=8, minimo_por_intencao=0, avaliar_a_cada=10)
    comum = SistemaDiet.novo(d, ROTULOS, PEQUENO, contagem_minima=1)
    treinar(comum, d, d, Receita(**base), semente=0)
    com_media = SistemaDiet.novo(d, ROTULOS, PEQUENO, contagem_minima=1)
    historico = treinar(com_media, d, d, Receita(**base, media_dos_pesos=0.9), semente=0)

    diferentes = any(
        not (a == b).all() for a, b in zip(comum.modelo.state_dict().values(), com_media.modelo.state_dict().values())
    )
    assert diferentes
    assert historico.dropna(subset=["dev_exact_match"])["dev_exact_match"].max() > 0.5


def test_atalhos_de_intencao_treinam_e_salvam(tmp_path):
    d = dados()
    for modo in ("resumo", "features"):
        config = Config(**{**PEQUENO.como_dict(), "atalho_intencao": modo})
        sistema = SistemaDiet.novo(d, ROTULOS, config, contagem_minima=1)
        sistema.modelo.train()
        sistema.modelo.perda(sistema.lote(d))["total"].backward()
        assert sistema.modelo.atalho is not None
        assert all(p.grad is not None for p in sistema.modelo.atalho.parameters())
        # a resposta não depende do lote também com o atalho
        textos = list(d["utt"])
        assert sistema.prever(textos) == [sistema.prever([t])[0] for t in textos]
        relido = SistemaDiet.carregar(sistema.salvar(tmp_path / modo))
        assert relido.prever(textos) == sistema.prever(textos)
