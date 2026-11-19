"""Etapa 7: as máscaras, a poda gradual, o reajuste com máscara travada, as peças inteiras, os arquivos e a regra."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.compressao import poda
from especialista.dados import massive
from especialista.experimentos import etapa6, etapa7, execucao
from especialista.modelos.diet import Config, SistemaDiet
from especialista.professores.rotulagem import rotular
from especialista.treino import Receita, treinar

from test_destilacao import professor_que_troca
from test_diet import ENUNCIADOS, PEQUENO, ROTULOS

CURTA = dict(taxa=3e-3, lote_inicial=4, lote_final=8, minimo_por_intencao=0)   # a receita, em miniatura
TEXTOS = [texto for _, texto, _ in ENUNCIADOS]


def dados() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": i, "intent": intent, "utt": texto, "tokens": texto.split(), "tags": tags.split()}
        for i, (intent, texto, tags) in enumerate(ENUNCIADOS)
    )


def sistema_novo(semente: int = 0, **troca) -> SistemaDiet:
    return SistemaDiet.novo(dados(), ROTULOS, Config(**{**PEQUENO.como_dict(), **troca}), semente=semente, contagem_minima=1)


def sistema_treinado(semente: int = 0) -> SistemaDiet:
    sistema = sistema_novo(semente)
    treinar(sistema, dados(), dados(), Receita(epocas=30, avaliar_a_cada=10, **CURTA), semente=semente)
    return sistema


def pontuacoes(sistema: SistemaDiet, textos=TEXTOS) -> tuple[torch.Tensor, torch.Tensor]:
    """As saídas da rede antes de qualquer escolha: pontuação de cada intenção e de cada tag."""
    sistema.modelo.eval()
    with torch.no_grad():
        return sistema.modelo(sistema.featurizador.montar([t.split() for t in textos]))


def zeros(sistema: SistemaDiet) -> dict[str, int]:
    return {nome: int((p == 0).sum()) for nome, p in sistema.modelo.named_parameters()}


# --------------------------------------------------------------------------
# Máscaras por magnitude
# --------------------------------------------------------------------------


@pytest.mark.parametrize("esparsidade", [0.3, 0.5, 0.7, 0.9])
def test_a_esparsidade_pedida_e_a_obtida(esparsidade):
    sistema = sistema_novo()
    antes = {nome: p.detach().clone() for nome, p in sistema.modelo.named_parameters()}
    contagem_antes = poda.contar(sistema.modelo)
    podaveis = poda.nomes_podaveis(sistema.modelo)
    mascaras = poda.mascaras_por_magnitude(sistema.modelo, esparsidade)
    poda.aplicar(sistema.modelo, mascaras)

    assert set(mascaras) == set(podaveis) and len(podaveis) == 2 + 6 * PEQUENO.n_camadas
    for nome, p in sistema.modelo.named_parameters():
        if nome not in mascaras:
            assert torch.equal(p, antes[nome])                          # o que não entra na poda fica intacto
            continue
        saiu = ~mascaras[nome]
        assert int(saiu.sum()) == round(esparsidade * p.numel())        # a conta exata, matriz por matriz
        assert int((p == 0).sum()) == int(saiu.sum())
        assert torch.equal(p[mascaras[nome]], antes[nome][mascaras[nome]])   # quem fica não muda
        assert antes[nome][saiu].abs().max() <= antes[nome][mascaras[nome]].abs().min()   # saem os de menor valor absoluto

    contagem = poda.contar(sistema.modelo)
    pedidos = sum(round(esparsidade * antes[nome].numel()) for nome in podaveis)
    assert contagem["podaveis"] - contagem["nao_nulos_podaveis"] == pedidos
    assert contagem["esparsidade"] == pytest.approx(esparsidade, abs=0.01)
    assert contagem["parametros"] == contagem_antes["parametros"] == sistema.parametros     # zerar não tira parâmetro nenhum da conta comum
    assert contagem_antes["esparsidade"] == 0.0
    assert contagem["nao_nulos"] == contagem_antes["nao_nulos"] - pedidos                   # mas tira dos diferentes de zero


def test_as_matrizes_podaveis_tem_quase_todos_os_parametros_da_rede_de_verdade():
    """Na configuração do projeto (Etapa 4b), o que entra na poda soma 98,7% dos parâmetros."""
    treino = pd.DataFrame({"tokens": [[f"palavra{i}"] * 2 for i in range(40)]})
    sistema = SistemaDiet.novo(treino, ROTULOS, Config())
    tabela = poda.por_tensor(sistema.modelo)
    assert tabela.loc[tabela["podavel"], "pesos"].sum() / tabela["pesos"].sum() > 0.9
    assert not tabela.loc[[n for n in tabela.index if "bias" in n or "norma" in n or n.startswith(("crf", "emissao", "cls_para"))], "podavel"].any()


def test_a_mascara_so_aperta():
    sistema = sistema_novo()
    primeira = poda.mascaras_por_magnitude(sistema.modelo, 0.4)
    poda.aplicar(sistema.modelo, primeira)
    with torch.no_grad():   # o treino mexe nos pesos; aqui, um peso que tinha saído "volta" com valor alto
        sistema.modelo.tabela.weight[~primeira["tabela.weight"]] = 9.0
    segunda = poda.mascaras_por_magnitude(sistema.modelo, 0.6, anteriores=primeira)
    for nome in primeira:
        assert not (segunda[nome] & ~primeira[nome]).any()            # ninguém que saiu voltou
        assert int((~segunda[nome]).sum()) == round(0.6 * segunda[nome].numel())
    # pedir menos do que já saiu não devolve nada
    terceira = poda.mascaras_por_magnitude(sistema.modelo, 0.1, anteriores=segunda)
    assert all(torch.equal(terceira[nome], segunda[nome]) for nome in segunda)

    with pytest.raises(ValueError):
        poda.mascaras_por_magnitude(sistema.modelo, 1.0)


def test_mascaras_dos_zeros_le_a_mascara_do_modelo_podado():
    sistema = sistema_novo()
    mascaras = poda.mascaras_por_magnitude(sistema.modelo, 0.7)
    poda.aplicar(sistema.modelo, mascaras)
    lidas = poda.mascaras_dos_zeros(sistema.modelo)
    assert all(torch.equal(lidas[nome], mascaras[nome]) for nome in mascaras)


def test_esparsidade_cubica_confere_com_a_conta_a_mao():
    # Zhu & Gupta, equação 1, com s_inicial = 0: s(t) = s_final · (1 − (1 − t/T)³)
    assert poda.esparsidade_cubica(0, 1000, 0.9) == 0.0
    assert poda.esparsidade_cubica(1000, 1000, 0.9) == 0.9
    assert poda.esparsidade_cubica(500, 1000, 0.9) == pytest.approx(0.9 * (1 - 0.5**3))      # 0,7875
    assert poda.esparsidade_cubica(250, 1000, 0.8, inicial=0.2) == pytest.approx(0.8 - 0.6 * 0.75**3)
    curva = [poda.esparsidade_cubica(t, 1000, 0.9) for t in range(0, 1001, 100)]
    passos = np.diff(curva)
    assert (passos > 0).all() and (np.diff(passos) < 0).all()      # sempre sobe, e cada corte é menor que o anterior
    assert poda.esparsidade_cubica(5000, 1000, 0.9) == 0.9         # depois da rampa, fica


# --------------------------------------------------------------------------
# O reajuste
# --------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [{}, {"media_dos_pesos": 0.9}, {"decaimento": 0.01}])
def test_reajustar_sem_poda_e_o_treinar(extra):
    """Sem máscara nem agenda, o laço da poda dá os mesmos pesos, número por número, que o laço de treino do projeto."""
    receita = Receita(epocas=12, avaliar_a_cada=4, **CURTA, **extra)
    d = dados()
    pelo_treino, pela_poda = sistema_novo(), sistema_novo()
    historico_do_treino = treinar(pelo_treino, d, d, receita, semente=3)
    historico_da_poda = poda.reajustar(pela_poda, d, d, receita, semente=3, partida_concorre=False)

    assert all(torch.equal(a, b) for a, b in zip(pelo_treino.modelo.state_dict().values(), pela_poda.modelo.state_dict().values()))
    iguais = [c for c in historico_do_treino.columns if c != "segundos"]
    pd.testing.assert_frame_equal(historico_do_treino[iguais], historico_da_poda[iguais])
    assert poda.passos_por_epoca(d, receita) == historico_da_poda["passos"].tolist()


@pytest.mark.parametrize("extra", [{}, {"media_dos_pesos": 0.9}, {"decaimento": 0.01}])
def test_a_mascara_travada_continua_zerada_depois_do_treino(extra):
    d = dados()
    sistema = sistema_treinado()
    mascaras = poda.mascaras_por_magnitude(sistema.modelo, 0.5)
    poda.aplicar(sistema.modelo, mascaras)
    podado = {nome: p.detach().clone() for nome, p in sistema.modelo.named_parameters()}

    historico = poda.reajustar(sistema, d, d, Receita(epocas=12, avaliar_a_cada=4, **CURTA, **extra), semente=0, mascaras=mascaras,
                               partida_concorre=False)   # aqui interessa o que o treino faz com a máscara: fica uma época treinada

    for nome, mascara in mascaras.items():
        peso = sistema.modelo.get_parameter(nome)
        assert (peso[~mascara] == 0).all()                           # o que saiu continua em zero
        assert not torch.equal(peso[mascara], podado[nome][mascara])  # e o que ficou foi reajustado
        assert int((peso == 0).sum()) == round(0.5 * peso.numel())
    assert np.allclose(historico["esparsidade"], 0.5, atol=0.01)
    assert historico["melhor"].sum() == 1
    # o sistema termina com os pesos da época marcada como melhor
    melhor = historico.loc[historico["melhor"], "dev_exact_match"].iloc[0]
    assert qualidade.avaliar(ouro(d), sistema.prever(list(d["utt"])))["exact_match"] == melhor


def test_sem_travar_a_mascara_os_zeros_se_perdem():
    """O controle do teste acima: o mesmo treino, sem a máscara, tira os pesos podados do zero."""
    d = dados()
    sistema = sistema_treinado()
    mascaras = poda.mascaras_por_magnitude(sistema.modelo, 0.5)
    poda.aplicar(sistema.modelo, mascaras)
    poda.reajustar(sistema, d, d, Receita(epocas=4, avaliar_a_cada=4, **CURTA), semente=0, partida_concorre=False)
    assert poda.contar(sistema.modelo)["esparsidade"] < 0.05


def test_o_ponto_de_partida_concorre_a_melhor_ponto():
    d = dados()
    receita = Receita(epocas=12, avaliar_a_cada=4, **CURTA)

    # um modelo que já acerta tudo: nenhuma época do reajuste é melhor, e ele sai como entrou
    sistema = sistema_treinado()
    assert qualidade.avaliar(ouro(d), sistema.prever(TEXTOS))["exact_match"] == 1.0
    como_chegou = {nome: p.detach().clone() for nome, p in sistema.modelo.named_parameters()}
    historico = poda.reajustar(sistema, d, d, receita, semente=0)
    assert historico["epoca"].tolist() == list(range(13)) and historico.loc[historico["melhor"], "epoca"].tolist() == [0]
    assert historico.loc[0, "dev_exact_match"] == 1.0 and historico.loc[0, "passos"] == 0
    assert all(torch.equal(p, como_chegou[nome]) for nome, p in sistema.modelo.named_parameters())
    assert historico["passos"].iloc[1:].tolist() == poda.passos_por_epoca(d, receita)

    # um modelo que a poda derrubou: o reajuste melhora, e fica uma época treinada, com a máscara no lugar
    sistema = sistema_treinado()
    mascaras = poda.mascaras_por_magnitude(sistema.modelo, 0.9)
    poda.aplicar(sistema.modelo, mascaras)
    derrubado = qualidade.avaliar(ouro(d), sistema.prever(TEXTOS))["exact_match"]
    historico = poda.reajustar(sistema, d, d, receita, semente=0, mascaras=mascaras)
    escolhida = historico.loc[historico["melhor"]].iloc[0]
    assert historico.loc[0, "dev_exact_match"] == derrubado < 1.0
    assert escolhida["epoca"] > 0 and escolhida["dev_exact_match"] > derrubado
    assert qualidade.avaliar(ouro(d), sistema.prever(TEXTOS))["exact_match"] == escolhida["dev_exact_match"]
    assert poda.contar(sistema.modelo)["esparsidade"] == pytest.approx(0.9, abs=0.01)

    # na poda gradual a partida não concorre: ela ainda não está na esparsidade final
    gradual = poda.reajustar(sistema_treinado(), d, d, receita, semente=0, agenda=poda.Agenda(0.5, epocas_de_rampa=4, passos_entre_cortes=2))
    assert gradual["epoca"].min() == 1


@pytest.mark.parametrize("extra", [{}, {"media_dos_pesos": 0.9}])
def test_poda_gradual_chega_na_esparsidade_e_so_escolhe_epoca_depois_da_rampa(extra):
    d = dados()
    sistema = sistema_treinado()
    receita = Receita(epocas=12, avaliar_a_cada=2, **CURTA, **extra)
    agenda = poda.Agenda(esparsidade=0.9, epocas_de_rampa=6, passos_entre_cortes=2)
    historico = poda.reajustar(sistema, d, d, receita, semente=0, agenda=agenda)

    # a esparsidade sobe pela curva cúbica, corte a corte, e trava no valor pedido ao fim da rampa
    passos_ate = np.cumsum(poda.passos_por_epoca(d, receita))
    fim = passos_ate[5] // 2 * 2
    esperada = [poda.esparsidade_cubica(min(p // 2 * 2, fim), fim, 0.9) for p in passos_ate]
    assert historico["esparsidade"].tolist() == pytest.approx(esperada, abs=0.01)
    assert historico["esparsidade"].is_monotonic_increasing
    assert historico["esparsidade"].iloc[0] > 0 and historico["esparsidade"].iloc[4] < 0.9
    assert np.allclose(historico["esparsidade"].iloc[5:], 0.9, atol=0.002)

    # o modelo final tem exatamente os zeros pedidos, matriz por matriz
    for nome in poda.nomes_podaveis(sistema.modelo):
        peso = sistema.modelo.get_parameter(nome)
        assert int((peso == 0).sum()) == round(0.9 * peso.numel())

    # as épocas da rampa são avaliadas, mas a escolhida é a melhor entre as que já estão na esparsidade final
    avaliadas = historico.dropna(subset=["dev_exact_match"])
    assert avaliadas["epoca"].tolist() == [2, 4, 6, 8, 10, 12]
    escolhida = historico.loc[historico["melhor"]].iloc[0]
    assert escolhida["epoca"] >= 6
    assert escolhida["dev_exact_match"] == avaliadas.loc[avaliadas["epoca"] >= 6, "dev_exact_match"].max()
    assert qualidade.avaliar(ouro(d), sistema.prever(list(d["utt"])))["exact_match"] == escolhida["dev_exact_match"]


def test_poda_gradual_recusa_o_que_nao_cabe():
    d = dados()
    receita = Receita(epocas=4, avaliar_a_cada=2, **CURTA)
    with pytest.raises(ValueError, match="rampa"):   # nenhum corte cabe na rampa
        poda.reajustar(sistema_novo(), d, d, receita, agenda=poda.Agenda(0.5, epocas_de_rampa=1, passos_entre_cortes=1000))
    with pytest.raises(ValueError, match="rampa"):   # rampa maior que o treino
        poda.reajustar(sistema_novo(), d, d, receita, agenda=poda.Agenda(0.5, epocas_de_rampa=5, passos_entre_cortes=1))
    with pytest.raises(ValueError, match="não as duas"):
        mascaras = poda.mascaras_por_magnitude(sistema_novo().modelo, 0.5)
        poda.reajustar(sistema_novo(), d, d, receita, mascaras=mascaras, agenda=poda.Agenda(0.5, 2, 1))


# --------------------------------------------------------------------------
# Peças inteiras
# --------------------------------------------------------------------------


def test_remover_linha_que_nao_contribui_preserva_as_saidas():
    sistema = sistema_treinado()
    n = len(sistema.featurizador)
    saem = torch.arange(0, n, 3)
    with torch.no_grad():
        sistema.modelo.tabela.weight[saem] = 0.0          # estas linhas deixam de contribuir...
    ficam = torch.tensor([i for i in range(n) if i % 3])
    menor = poda.remover_linhas(sistema, ficam)             # ...e tirá-las não muda nenhuma saída

    assert len(menor.featurizador) == len(ficam) == menor.modelo.tabela.weight.shape[0]
    assert menor.parametros == sistema.parametros - len(saem) * PEQUENO.d_features
    for inteira, reduzida in zip(pontuacoes(sistema), pontuacoes(menor)):
        assert torch.allclose(inteira, reduzida, atol=1e-6)
    assert menor.prever(TEXTOS) == sistema.prever(TEXTOS)
    # cada feature que ficou continua com a sua linha
    nomes = list(sistema.featurizador.vocabulario)
    for novo, antigo in enumerate(ficam.tolist()):
        assert menor.featurizador.vocabulario[nomes[antigo]] == novo
        assert torch.equal(menor.modelo.tabela.weight[novo], sistema.modelo.tabela.weight[antigo])


def test_remover_features_de_um_pedido_nao_muda_a_resposta_dos_outros():
    sistema = sistema_treinado()
    do_primeiro = {i for palavra in TEXTOS[0].split() for i in sistema.featurizador.ids(palavra)}
    dos_outros = {i for texto in TEXTOS[1:] for palavra in texto.split() for i in sistema.featurizador.ids(palavra)}
    ficam = sorted(set(range(len(sistema.featurizador))) - (do_primeiro - dos_outros))
    menor = poda.remover_linhas(sistema, ficam)
    assert len(menor.featurizador) < len(sistema.featurizador)
    for inteira, reduzida in zip(pontuacoes(sistema, TEXTOS[1:]), pontuacoes(menor, TEXTOS[1:])):
        assert torch.allclose(inteira, reduzida, atol=1e-6)
    assert not torch.allclose(pontuacoes(sistema, TEXTOS[:1])[0], pontuacoes(menor, TEXTOS[:1])[0], atol=1e-6)


def test_remover_camada_que_nao_contribui_preserva_as_saidas():
    sistema = sistema_treinado()
    assert len(sistema.modelo.blocos) == 2
    bloco = sistema.modelo.blocos[1]
    with torch.no_grad():   # as duas saídas do bloco de cima passam a somar zero: o bloco vira uma passagem direta
        for camada in (bloco.atencao.saida, bloco.contrai):
            camada.weight.zero_()
            camada.bias.zero_()
    sem = poda.remover_camada(sistema)

    assert len(sem.modelo.blocos) == 1 and sem.modelo.config.n_camadas == 1
    assert sem.parametros == sistema.parametros - sum(p.numel() for p in bloco.parameters())
    for inteira, reduzida in zip(pontuacoes(sistema), pontuacoes(sem)):
        assert torch.allclose(inteira, reduzida, atol=1e-6)
    assert sem.prever(TEXTOS) == sistema.prever(TEXTOS)


def test_remover_camada_fica_com_os_pesos_certos():
    sistema = sistema_treinado()
    de_cima, de_baixo = poda.remover_camada(sistema), poda.remover_camada(sistema, qual=0)
    for nome, valor in sistema.modelo.blocos[0].state_dict().items():
        assert torch.equal(de_cima.modelo.blocos[0].state_dict()[nome], valor)      # padrão: sai o bloco de cima
    for nome, valor in sistema.modelo.blocos[1].state_dict().items():
        assert torch.equal(de_baixo.modelo.blocos[0].state_dict()[nome], valor)
    assert not torch.allclose(pontuacoes(sistema)[0], pontuacoes(de_cima)[0], atol=1e-4)   # um bloco treinado contribui
    with pytest.raises(ValueError):
        poda.remover_camada(de_cima)


def test_importancia_das_linhas():
    d = dados()
    sistema = sistema_treinado()
    vocabulario = sistema.featurizador.vocabulario
    antes = {nome: p.detach().clone() for nome, p in sistema.modelo.named_parameters()}
    config = sistema.modelo.config

    importancia = poda.importancia_das_linhas(sistema, d, tamanho_do_lote=4)
    assert importancia.shape == (len(vocabulario),) and (importancia >= 0).all() and importancia.max() > 0
    # medir não mexe no modelo: mesmos pesos, mesma configuração, nenhum gradiente pendurado, mesmo modo
    assert all(torch.equal(p, antes[nome]) for nome, p in sistema.modelo.named_parameters())
    assert sistema.modelo.config == config and not sistema.modelo.training
    assert all(p.grad is None for p in sistema.modelo.parameters())
    assert torch.equal(importancia, poda.importancia_das_linhas(sistema, d, tamanho_do_lote=4))   # não depende de sorteio

    # um feature que não aparece nos dados não pesa nada; os que aparecem pesam
    so_lisboa = [i for nome, i in vocabulario.items() if "lisbo" in nome]
    sem_lisboa = d[~d["utt"].str.contains("lisboa")]
    parcial = poda.importancia_das_linhas(sistema, sem_lisboa, tamanho_do_lote=4)
    assert len(so_lisboa) > 0 and (parcial[so_lisboa] == 0).all() and (importancia[so_lisboa] > 0).all()

    # a escolha: sai exatamente a fração pedida, das menos importantes
    ficam = poda.linhas_que_ficam(parcial, 0.25)
    saem = sorted(set(range(len(vocabulario))) - set(ficam.tolist()))
    assert len(saem) == round(0.25 * len(vocabulario)) and ficam.tolist() == sorted(ficam.tolist())
    assert parcial[saem].max() <= parcial[ficam].min()


def test_aluno_que_aprende_pelos_logits_e_podado_com_os_logits():
    """Num aluno destilado com temperatura, a importância e o reajuste usam a distribuição do professor, não o rótulo."""
    d = dados()
    indice = np.array([ROTULOS["intents"].index(i) for i in d["intent"]])
    logits = np.full((len(d), 3), -4.0, dtype=np.float32)
    logits[np.arange(len(d)), indice] = 0.0
    # os rótulos de intenção estão todos errados; só os logits dizem a intenção certa
    do_professor = d.assign(intent=[ROTULOS["intents"][(i + 1) % 3] for i in indice], logits_professor=list(logits))
    aluno = sistema_novo(temperatura=2.0)
    treinar(aluno, do_professor, d, Receita(epocas=40, avaliar_a_cada=10, **CURTA), semente=0)
    assert qualidade.avaliar(ouro(d), aluno.prever(TEXTOS))["intent_acc"] == 1.0

    with pytest.raises(ValueError, match="logits_professor"):     # sem os logits a medida nem roda: não há como usar o rótulo por engano
        poda.importancia_das_linhas(aluno, d, tamanho_do_lote=4)
    importancia = poda.importancia_das_linhas(aluno, do_professor, tamanho_do_lote=4)
    assert importancia.max() > 0 and aluno.modelo.config.peso_mascara == 1.0

    mascaras = poda.mascaras_por_magnitude(aluno.modelo, 0.7)
    poda.reajustar(aluno, do_professor, d, Receita(epocas=20, avaliar_a_cada=5, **CURTA), semente=0, mascaras=mascaras)
    assert poda.contar(aluno.modelo)["esparsidade"] == pytest.approx(0.7, abs=0.01)
    assert qualidade.avaliar(ouro(d), aluno.prever(TEXTOS))["intent_acc"] == 1.0   # segue a intenção dos logits depois de podado


# --------------------------------------------------------------------------
# Gravar e reler
# --------------------------------------------------------------------------


def test_modelo_podado_grava_rele_e_reproduz_as_previsoes(tmp_path):
    d = dados()
    sistema = sistema_treinado()
    mascaras = poda.mascaras_por_magnitude(sistema.modelo, 0.7)
    poda.reajustar(sistema, d, d, Receita(epocas=8, avaliar_a_cada=4, **CURTA), semente=0, mascaras=mascaras)
    previsto = sistema.prever(TEXTOS)

    relido = SistemaDiet.carregar(sistema.salvar(tmp_path / "podado"))
    assert relido.prever(TEXTOS) == previsto
    assert zeros(relido) == zeros(sistema) and poda.contar(relido.modelo) == poda.contar(sistema.modelo)   # os zeros sobrevivem ao disco


def test_arquivo_esparso_devolve_os_mesmos_pesos_e_e_menor(tmp_path):
    sistema = sistema_treinado()
    pasta = sistema.salvar(tmp_path / "modelo")
    sem_poda = poda.salvar_esparso(sistema, pasta).stat().st_size
    denso = (pasta / "pesos.pt").stat().st_size
    assert sem_poda < denso * 1.05                           # sem zeros para aproveitar, não fica maior que o comum

    tamanhos = {}
    for esparsidade in (0.5, 0.9):
        podado = sistema_treinado()
        poda.aplicar(podado.modelo, poda.mascaras_por_magnitude(podado.modelo, esparsidade))
        pasta = podado.salvar(tmp_path / f"podado_{esparsidade}")
        tamanhos[esparsidade] = poda.salvar_esparso(podado, pasta).stat().st_size
        assert (pasta / "pesos.pt").stat().st_size == denso  # o arquivo comum não muda com os zeros

        (pasta / "pesos.pt").unlink()                        # o sistema volta só do arquivo esparso
        relido = poda.carregar_esparso(pasta)
        original, lido = podado.modelo.state_dict(), relido.modelo.state_dict()
        assert list(original) == list(lido) and all(torch.equal(original[nome], lido[nome]) for nome in original)
        assert relido.prever(TEXTOS) == podado.prever(TEXTOS)
        assert poda.bytes_do_sistema(pasta, poda.ARQUIVO_ESPARSO) == sum(p.stat().st_size for p in pasta.iterdir())
    assert tamanhos[0.9] < tamanhos[0.5] < sem_poda
    # um compressor comum também aproveita os zeros do arquivo de sempre
    cheio, com_zeros = sistema_treinado(), sistema_treinado()
    poda.aplicar(com_zeros.modelo, poda.mascaras_por_magnitude(com_zeros.modelo, 0.9))
    assert poda.bytes_com_zip(com_zeros.salvar(tmp_path / "a") / "pesos.pt") < 0.5 * poda.bytes_com_zip(cheio.salvar(tmp_path / "b") / "pesos.pt")


def test_modelos_da_poda_estruturada_gravam_e_relem(tmp_path):
    sistema = sistema_treinado()
    importancia = poda.importancia_das_linhas(sistema, dados(), tamanho_do_lote=4)
    for nome, menor in (("linhas", poda.remover_linhas(sistema, poda.linhas_que_ficam(importancia, 0.5))), ("camada", poda.remover_camada(sistema))):
        relido = SistemaDiet.carregar(menor.salvar(tmp_path / nome))
        assert relido.prever(TEXTOS) == menor.prever(TEXTOS)
        assert relido.parametros == menor.parametros < sistema.parametros
        # aqui o arquivo comum encolhe de verdade, sem formato especial
        assert (tmp_path / nome / "pesos.pt").stat().st_size < (sistema.salvar(tmp_path / "inteiro") / "pesos.pt").stat().st_size
    assert (tmp_path / "linhas" / "features.json").stat().st_size < (tmp_path / "inteiro" / "features.json").stat().st_size


def test_execucao_esparsa_da_as_mesmas_respostas():
    sistema = sistema_treinado()
    poda.aplicar(sistema.modelo, poda.mascaras_por_magnitude(sistema.modelo, 0.8))
    esparso = poda.execucao_esparsa(sistema)
    assert isinstance(esparso.modelo.blocos[0].expande, poda.LinearEsparsa) and isinstance(sistema.modelo.blocos[0].expande, torch.nn.Linear)
    for densa, esparsa in zip(pontuacoes(sistema), pontuacoes(esparso)):
        assert torch.allclose(densa, esparsa, atol=1e-5)
    assert esparso.prever(TEXTOS) == sistema.prever(TEXTOS)
    assert esparso.prever(TEXTOS) == [esparso.prever([t])[0] for t in TEXTOS]   # uma consulta por vez, como na medida de latência


# --------------------------------------------------------------------------
# A execução: quem é podado, com que sinal, e o que fica gravado
# --------------------------------------------------------------------------


@pytest.fixture
def cenario(tmp_path, monkeypatch):
    """As entregas de que a Etapa 7 depende, em miniatura: o teto (Etapa 4b) e um aluno A (Etapas 5a e 6)."""
    treino, dev = dados(), dados().assign(id=lambda t: t["id"] + 100)
    teste = dados().assign(id=lambda t: t["id"] + 200).iloc[::-1].reset_index(drop=True)
    intents = ROTULOS["intents"]
    monkeypatch.setattr(massive, "carregar_entrega", lambda: ({"train": treino, "dev": dev, "test": teste}, ROTULOS))
    for modulo in (etapa6, etapa7):
        monkeypatch.setattr(modulo, "ENTREGAS", tmp_path / "entregas")
        monkeypatch.setattr(modulo, "RAIZ", tmp_path)
    monkeypatch.setattr(execucao, "EXECUCOES", tmp_path / "execucoes")
    monkeypatch.setattr(etapa7, "EPOCAS", 8)
    monkeypatch.setattr(etapa7, "EPOCAS_DE_RAMPA", 4)
    monkeypatch.setattr(etapa7, "PASSOS_ENTRE_CORTES", 2)

    # Etapa 4b: a rede, a receita e os modelos do teto
    receita = {"epocas": 30, "avaliar_a_cada": 2, "media_dos_pesos": 0.9, **CURTA}
    rede = tmp_path / "entregas" / "04b_intencao"
    for semente in etapa7.SEMENTES:
        sistema = SistemaDiet.novo(treino, ROTULOS, Config(**{**PEQUENO.como_dict(), "suavizacao": 0.1}), semente=semente, contagem_minima=1)
        treinar(sistema, treino, dev, Receita(**receita), semente=semente)
        sistema.salvar(rede / "modelos" / f"semente_{semente}")
    (rede / "decisao.json").write_text(json.dumps({
        "config": {**PEQUENO.como_dict(), "suavizacao": 0.1}, "receita": receita, "modelos": "entregas/04b_intencao/modelos",
    }), encoding="utf-8")

    # Etapa 5a: um professor que troca duas intenções rotula o treino e o dev
    professor = etapa6.pasta_do_professor("fechado")
    professor.mkdir(parents=True)
    rotular(treino, professor_que_troca(treino), intents).to_parquet(professor / "rotulos_treino.parquet", index=False)
    salvar_previsoes(professor / "previsoes" / "claude_dev.parquet", dev["id"], professor_que_troca(dev))
    return {"treino": treino, "dev": dev, "teste": teste, "pasta": tmp_path}


def entregar_o_aluno_a(cenario) -> None:
    """Etapa 6: treina o aluno A e grava a decisão que a Etapa 7 lê."""
    destilacao = cenario["pasta"] / "entregas" / "06_destilacao"
    for semente in etapa7.SEMENTES:
        origem = etapa6.executar("aluno_a", semente, "cpu")
        shutil.copytree(origem / "modelo", destilacao / "modelos" / "aluno_a" / f"semente_{semente}")
    meta = etapa6.carregar("aluno_a", 0)["meta"]
    (destilacao / "decisao.json").write_text(json.dumps({
        "seguem": ["A"], "principal": "A",
        "alunos": {"A": {"nome": "Aluno A (rótulos do professor fechado)", "variante": "aluno_a", "config": meta["config"],
                         "receita": meta["receita"], "modelos": "entregas/06_destilacao/modelos/aluno_a"}},
    }), encoding="utf-8")


def test_sem_a_entrega_da_etapa_6_so_o_teto_e_podado(cenario):
    assert list(etapa7.modelos_a_podar()) == ["teto"]
    assert len(etapa7.grade()) == len(etapa7.VARIANTES) * len(etapa7.SEMENTES) == 48
    # os níveis do plano em todas as formas de poda; os extras, só na poda de uma vez
    assert [v for v in etapa7.VARIANTES if v.startswith("gradual")] == [f"gradual_{s}" for s in (30, 50, 70, 90)]
    assert [v for v in etapa7.VARIANTES if v.startswith("magnitude")] == [f"magnitude_{s}" for s in (30, 50, 70, 90, 95, 98)]
    assert etapa7.receita_do_reajuste("teto").epocas == 8
    assert etapa7.receita_do_reajuste("teto").media_dos_pesos == 0.9      # o resto da receita é o de quem treinou o modelo
    entregar_o_aluno_a(cenario)
    assert list(etapa7.modelos_a_podar()) == ["aluno_a", "teto"]          # o aluno principal roda primeiro
    assert etapa7.faltam()[0] == ("aluno_a", "sem_poda", 0) and len(etapa7.faltam()) == 96


@pytest.mark.parametrize("variante", ["sem_poda", "magnitude_50", "magnitude_98", "gradual_90", "linhas_50", "sem_camada"])
def test_cada_variante_roda_e_grava_o_que_promete(cenario, variante):
    dev, teste = cenario["dev"], cenario["teste"]
    pasta = etapa7.executar("teto", variante, 1, "cpu")
    execucao_7 = etapa7.carregar("teto", variante, 1)
    meta, historico, previsoes = execucao_7["meta"], execucao_7["historico"], execucao_7["previsoes"]
    familia, fracao = etapa7.VARIANTES[variante]["familia"], etapa7.VARIANTES[variante]["nivel"] / 100
    de_partida = SistemaDiet.carregar(cenario["pasta"] / "entregas" / "04b_intencao" / "modelos" / "semente_1")
    podado = SistemaDiet.carregar(pasta / "modelo")

    # o reajuste tem o orçamento da etapa e a receita de quem treinou o modelo; fora da poda gradual, a partida concorre (época 0)
    assert historico["epoca"].tolist() == list(range(1 if familia == "gradual" else 0, 9))
    assert meta["receita"]["epocas"] == 8 and meta["receita"]["media_dos_pesos"] == 0.9
    assert meta["melhor_epoca"] == int(historico.loc[historico["melhor"], "epoca"].iloc[0])
    assert meta["de_partida"] == "entregas/04b_intencao/modelos/semente_1" and meta["origem"]["dev"] == "rótulos humanos de dev"
    # as respostas gravadas cobrem dev e teste, na ordem das partições, e o modelo gravado as reproduz
    assert carregar_previsoes(pasta / "previsoes_dev.parquet")[0] == dev["id"].tolist()
    assert carregar_previsoes(pasta / "previsoes_test.parquet")[0] == teste["id"].tolist()
    assert podado.prever(list(dev["utt"])) == previsoes["dev"] and podado.prever(list(teste["utt"])) == previsoes["test"]
    assert meta["dev"] == pytest.approx(qualidade.avaliar(ouro(dev), previsoes["dev"]))
    assert meta["contagem"] == poda.contar(podado.modelo) and meta["contagem_de_partida"] == poda.contar(de_partida.modelo)

    # antes do reajuste
    if familia == "gradual":
        assert set(previsoes) == {"dev", "test"} and meta["dev_antes"] is None
    else:
        assert set(previsoes) == {"dev", "test", "dev_antes", "test_antes"}
        assert meta["dev_antes"] == pytest.approx(qualidade.avaliar(ouro(dev), previsoes["dev_antes"]))
    if familia == "controle":
        assert previsoes["dev_antes"] == de_partida.prever(list(dev["utt"]))      # "antes" do controle é o modelo de partida
        assert meta["contagem"]["parametros"] == de_partida.parametros and meta["contagem"]["esparsidade"] < 0.01

    # o que cada família tira
    if familia in etapa7.PESOS_SOLTOS:
        assert meta["contagem"]["parametros"] == de_partida.parametros
        for nome in poda.nomes_podaveis(podado.modelo):
            assert int((podado.modelo.get_parameter(nome) == 0).sum()) == round(fracao * podado.modelo.get_parameter(nome).numel())
        assert (meta["poda_gradual"] is not None) == (familia == "gradual")
        if familia == "gradual":
            assert int(historico.loc[historico["melhor"], "epoca"].iloc[0]) >= 4
    if familia == "linhas":
        assert meta["features"] == len(podado.featurizador) == len(de_partida.featurizador) - round(fracao * len(de_partida.featurizador))
        assert meta["contagem"]["parametros"] == de_partida.parametros - (len(de_partida.featurizador) - meta["features"]) * PEQUENO.d_features
    if familia == "camada":
        assert meta["config"]["n_camadas"] == 1 and len(podado.modelo.blocos) == 1
        assert meta["contagem"]["parametros"] < de_partida.parametros

    # o que está gravado não roda de novo
    marca = (pasta / "meta.json").stat().st_mtime_ns
    assert etapa7.executar("teto", variante, 1, "cpu") == pasta and (pasta / "meta.json").stat().st_mtime_ns == marca
    assert ("teto", variante, 1) not in etapa7.faltam() and ("teto", variante, 0) in etapa7.faltam()


def test_o_aluno_e_reajustado_com_o_sinal_do_seu_professor(cenario):
    entregar_o_aluno_a(cenario)
    dev = cenario["dev"]
    pasta = etapa7.executar("aluno_a", "magnitude_50", 0, "cpu")
    execucao_7 = etapa7.carregar("aluno_a", "magnitude_50", 0)
    meta, historico, previsoes = execucao_7["meta"], execucao_7["historico"], execucao_7["previsoes"]

    # treino e parada com os rótulos do professor, como na Etapa 6
    assert "professor fechado" in meta["origem"]["treino"] and "professor fechado" in meta["origem"]["dev"]
    assert meta["de_partida"] == "entregas/06_destilacao/modelos/aluno_a/semente_0"
    do_professor = professor_que_troca(dev)
    assert meta["dev"] == pytest.approx(qualidade.avaliar(do_professor, previsoes["dev"]))
    assert meta["dev_antes"] == pytest.approx(qualidade.avaliar(do_professor, previsoes["dev_antes"]))
    avaliadas = historico.dropna(subset=["dev_exact_match"])
    assert meta["dev"]["exact_match"] == pytest.approx(avaliadas["dev_exact_match"].max())
    # o aluno podado continua imitando o professor, não os rótulos humanos
    assert qualidade.avaliar(do_professor, previsoes["dev"])["intent_acc"] > 0.8
    assert qualidade.avaliar(ouro(dev), previsoes["dev"])["intent_acc"] < 0.5
    assert SistemaDiet.carregar(pasta / "modelo").prever(list(dev["utt"])) == previsoes["dev"]


def test_modelo_de_partida_treinado_de_novo_e_refeito_pelo_mesmo_comando(cenario):
    """Se os pesos de partida mudam, a execução antiga deixa de valer: volta para a lista e é refeita, sem passo manual."""
    pasta = etapa7.executar("teto", "magnitude_50", 0, "cpu")
    antiga = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    assert antiga["digital_de_partida"] == etapa7.digital_de_partida("teto", 0)
    assert etapa7.pronta("teto", "magnitude_50", 0) and ("teto", "magnitude_50", 0) not in etapa7.faltam()

    # o teto, semente 0, é treinado de novo (aqui, com outra semente): os pesos mudam
    treino, dev = cenario["treino"], cenario["dev"]
    de_partida = cenario["pasta"] / "entregas" / "04b_intencao" / "modelos" / "semente_0"
    outro = SistemaDiet.novo(treino, ROTULOS, Config(**{**PEQUENO.como_dict(), "suavizacao": 0.1}), semente=7, contagem_minima=1)
    treinar(outro, treino, dev, Receita(epocas=10, avaliar_a_cada=5, **CURTA), semente=7)
    outro.salvar(de_partida)

    assert etapa7.digital_de_partida("teto", 0) != antiga["digital_de_partida"]
    assert not etapa7.pronta("teto", "magnitude_50", 0) and ("teto", "magnitude_50", 0) in etapa7.faltam()
    (pasta / "resto_da_execucao_antiga.txt").write_text("x", encoding="utf-8")
    assert etapa7.executar("teto", "magnitude_50", 0, "cpu") == pasta
    nova = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    assert nova["digital_de_partida"] == etapa7.digital_de_partida("teto", 0) and etapa7.pronta("teto", "magnitude_50", 0)
    assert not (pasta / "resto_da_execucao_antiga.txt").exists()          # a pasta antiga foi apagada inteira
    # a execução refeita partiu mesmo dos pesos novos: o seu "antes" é o modelo novo, podado
    poda.aplicar(outro.modelo, poda.mascaras_por_magnitude(outro.modelo, 0.5))
    assert etapa7.carregar("teto", "magnitude_50", 0)["previsoes"]["dev_antes"] == outro.prever(list(dev["utt"]))
    # as outras sementes não mudaram: continuam valendo
    etapa7.executar("teto", "magnitude_50", 1, "cpu")
    assert etapa7.pronta("teto", "magnitude_50", 1)


def test_modelo_de_partida_com_outra_configuracao_e_recusado(cenario):
    arquivo = cenario["pasta"] / "entregas" / "04b_intencao" / "decisao.json"
    decisao = json.loads(arquivo.read_text(encoding="utf-8"))
    decisao["config"]["dropout"] = 0.5
    arquivo.write_text(json.dumps(decisao), encoding="utf-8")
    with pytest.raises(ValueError, match="configuração"):
        etapa7.executar("teto", "sem_poda", 0, "cpu")


# --------------------------------------------------------------------------
# A regra do que segue para a quantização
# --------------------------------------------------------------------------


def respostas(gabarito: list[Previsao], erradas: int, desde: int = 0) -> list[Previsao]:
    """Uma cópia do gabarito com a intenção errada em `erradas` pedidos seguidos, a partir de `desde`."""
    return [Previsao("outra", p.slots) if desde <= i < desde + erradas else p for i, p in enumerate(gabarito)]


def test_quem_segue():
    gabarito = [Previsao("alarm_set")] * 400
    sem_poda = [respostas(gabarito, 80)] * 3                    # erra os pedidos 0 a 79
    # o modelo podado conserta metade desses erros e comete outros: é assim que dois modelos diferem de verdade.
    # zerar pesos soltos não muda o arquivo comum (1000 bytes, como o modelo sem poda); tirar peças inteiras muda
    solto = lambda erradas, nao_nulos: {"em_dev": [respostas(gabarito, erradas, desde=40)] * 3, "nao_nulos": nao_nulos, "bytes": 1000}
    peca = lambda erradas, tamanho: {"em_dev": [respostas(gabarito, erradas, desde=40)] * 3, "nao_nulos": tamanho, "bytes": tamanho}

    decisao = etapa7.quem_segue(gabarito, sem_poda, {
        "magnitude_30": solto(80, 700), "gradual_30": solto(78, 700),
        "magnitude_50": solto(84, 500), "gradual_50": solto(82, 500),     # 1 e 0,5 ponto abaixo: não dá para distinguir
        "magnitude_70": solto(120, 300), "gradual_70": solto(84, 300),    # só a gradual passa nos 70%
        "magnitude_90": solto(200, 100), "gradual_90": solto(160, 100),
        "linhas_30": peca(80, 800), "linhas_50": peca(82, 650), "linhas_70": peca(140, 520),
        "sem_camada": peca(83, 840),
    })
    tabela = decisao["tabela"]
    assert tabela.loc["magnitude_70", "diferenca"] == pytest.approx(-0.10) and not tabela.loc["magnitude_70", "passa"]
    assert tabela.loc["gradual_70", "passa"] and tabela.loc["sem_camada", "passa"] and not tabela.loc["linhas_70", "passa"]
    # a principal é a que passa com o menor arquivo comum: uma peça inteira, mesmo tendo mais parâmetros diferentes de zero
    assert decisao["principal"] == "linhas_50" and tabela.loc["linhas_50", "nao_nulos"] > tabela.loc["gradual_70", "nao_nulos"]
    # a secundária é a dos pesos soltos com a maior esparsidade que passa
    assert decisao["secundaria"] == "gradual_70"
    assert decisao["seguem"] == ["linhas_50", "gradual_70"]

    # pesos soltos: as duas formas passam na mesma esparsidade: fica a de maior exact match; empate, a de uma vez
    assert etapa7.quem_segue(gabarito, sem_poda, {"magnitude_50": solto(84, 500), "gradual_50": solto(82, 500)})["secundaria"] == "gradual_50"
    assert etapa7.quem_segue(gabarito, sem_poda, {"magnitude_50": solto(82, 500), "gradual_50": solto(82, 500)})["secundaria"] == "magnitude_50"
    # peças inteiras com o mesmo arquivo: a de maior exact match
    assert etapa7.quem_segue(gabarito, sem_poda, {"linhas_30": peca(84, 800), "sem_camada": peca(80, 800)})["principal"] == "sem_camada"
    # nenhuma peça inteira passa: a principal é o modelo sem poda, e a dos pesos soltos continua secundária
    so_soltos = etapa7.quem_segue(gabarito, sem_poda, {"magnitude_30": solto(40, 700), "linhas_90": peca(190, 400)})
    assert so_soltos["principal"] is None and so_soltos["secundaria"] == "magnitude_30" and so_soltos["seguem"] == ["magnitude_30"]
    # só uma peça inteira passa: não há secundária
    so_pecas = etapa7.quem_segue(gabarito, sem_poda, {"magnitude_90": solto(200, 100), "linhas_30": peca(80, 800)})
    assert so_pecas["principal"] == "linhas_30" and so_pecas["secundaria"] is None and so_pecas["seguem"] == ["linhas_30"]
    # nada passa: nada segue
    nada = etapa7.quem_segue(gabarito, sem_poda, {"magnitude_90": solto(200, 100), "linhas_90": peca(190, 400)})
    assert nada["seguem"] == [] and nada["principal"] is None and nada["secundaria"] is None and not nada["tabela"]["passa"].any()
