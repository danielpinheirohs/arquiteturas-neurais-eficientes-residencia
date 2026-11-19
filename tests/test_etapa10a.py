"""Etapa 10a: as regras escritas antes dos resultados, os quatro tempos da etapa e a retomada.

Os treinos usam o encoder minúsculo de `test_encoder.py` no lugar do XLM-R e
as entregas em miniatura da Etapa 6 (`cenario`, de `test_destilacao.py`).
"""

import json
from math import ceil

import pytest

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro
from especialista.caminhos import ENTREGAS
from especialista.experimentos import etapa6, etapa10a
from especialista.modelos.encoder import Config

from test_destilacao import cenario, professor_que_troca, respostas  # noqa: F401  (cenario é uma fixture)
from test_encoder import sistema_minusculo

RAPIDA = {"taxa": 1e-2, "epocas": 40, "lote": 4}   # o encoder minúsculo parte de pesos sorteados: precisa de taxa alta


@pytest.fixture
def etapa(cenario, monkeypatch):  # noqa: F811
    """A Etapa 10a em miniatura: as entregas da Etapa 6 e um encoder minúsculo no lugar do XLM-R."""
    minusculo = lambda config: Config(**{**config.como_dict(), "modelo_id": "minusculo", "revisao": ""})
    monkeypatch.setattr(etapa10a, "novo_sistema", lambda rotulos, config, semente, dispositivo: sistema_minusculo(semente, minusculo(config), rotulos))
    return cenario


def com_receita(receita: dict = RAPIDA, vencedora: str = "busca_taxa_3e-5_epocas_10") -> None:
    """Como se a busca já tivesse escolhido a receita."""
    etapa10a._gravar_json(etapa10a.pasta_da_etapa() / "receita.json", {"vencedora": vencedora, "receita": receita, "regra": ""})


def com_portao(passou: bool = True) -> None:
    etapa10a._gravar_json(etapa10a.pasta_da_etapa() / "portao.json", {"passou": passou, "no_teste": {}})


def erradas(gabarito: list[Previsao], intencao: int, entidade: int = 0) -> list[Previsao]:
    """Uma cópia do gabarito com a intenção errada nos primeiros pedidos e a entidade perdida nos últimos."""
    copia = list(gabarito)
    for i in range(intencao):
        copia[i] = Previsao("outra", copia[i].slots)
    for i in range(len(copia) - entidade, len(copia)):
        copia[i] = Previsao(copia[i].intent, ())
    return copia


COM_ENTIDADE = [Previsao("alarm_set", (("date", 0, 1),))] * 400


# --------------------------------------------------------------------------
# O desenho da etapa
# --------------------------------------------------------------------------


def test_a_busca_e_as_variantes_sao_as_declaradas():
    assert len(etapa10a.BUSCA) == 6                                       # no máximo 6 configurações
    assert all(2e-5 <= v["receita"]["taxa"] <= 5e-5 for v in etapa10a.BUSCA.values())
    assert etapa10a.variante_da_busca(2e-5, 10) == "busca_taxa_2e-5_epocas_10" in etapa10a.BUSCA
    assert list(etapa10a.VARIANTES) == ["teto", "piso", "aluno_a", "aluno_b1", "aluno_b2_t1", "aluno_b2_t2", "aluno_b2_t4", "aluno_b2_t8"]
    # os dados do piso e de cada aluno são os da variante de mesmo nome da Etapa 6
    assert all(v["dados"] in (None, nome) and (v["dados"] is None or nome in etapa6.VARIANTES) for nome, v in etapa10a.VARIANTES.items())
    for T in etapa10a.TEMPERATURAS:                                       # a perda do B2 é a da Etapa 6: só a distribuição do professor
        assert etapa10a.VARIANTES[etapa6.variante_do_b2(T)]["config"] == {"temperatura": float(T), "peso_rotulo": 0.0}
    assert len(etapa10a.grade()) == 6 + 8 * 3
    assert set(etapa10a.PUBLICADO) == {"intent_acc", "slot_f1", "exact_match"}


# --------------------------------------------------------------------------
# As regras
# --------------------------------------------------------------------------


def test_escolher_receita():
    gabarito = [Previsao("alarm_set")] * 100
    v = etapa10a.variante_da_busca
    iguais = {nome: respostas(gabarito, 20) for nome in etapa10a.BUSCA}

    # maior exact match em dev
    melhor = {**iguais, v(5e-5, 20): respostas(gabarito, 10)}
    escolhida, tabela = etapa10a.escolher_receita(gabarito, melhor)
    assert escolhida == v(5e-5, 20) and tabela.loc[escolhida, "exact_match"] == pytest.approx(0.90)
    assert list(tabela.index) == list(etapa10a.BUSCA) and tabela.loc[v(3e-5, 10), ["taxa", "epocas"]].tolist() == [3e-5, 10]

    # empate em exact match: decide a acurácia de intenção
    empate = {**iguais, v(3e-5, 20): respostas(gabarito, 10, 10)}        # 20 pedidos errados, só 10 de intenção
    assert etapa10a.escolher_receita(gabarito, empate)[0] == v(3e-5, 20)
    # empate nos dois: menos épocas e, depois, a menor taxa
    assert etapa10a.escolher_receita(gabarito, iguais)[0] == v(2e-5, 10)
    assert etapa10a.escolher_receita(gabarito, {v(5e-5, 10): iguais[v(5e-5, 10)], v(2e-5, 20): iguais[v(2e-5, 20)]})[0] == v(5e-5, 10)


def test_perto_do_publicado():
    # igual ao gabarito: acima do publicado nas três métricas
    passou, tabela = etapa10a.perto_do_publicado(COM_ENTIDADE, [COM_ENTIDADE] * 3)
    assert passou and tabela["perto"].all() and (tabela["diferenca"] > 0).all()
    assert tabela.loc["intent_acc", "publicado"] == 0.867 and tabela.loc["slot_f1", "margem_publicada"] == 0.006

    # um pouco abaixo do publicado em intenção (84,0 contra 86,7), mas os intervalos se tocam: perto
    pouco = [erradas(COM_ENTIDADE, 64)] * 3
    passou, tabela = etapa10a.perto_do_publicado(COM_ENTIDADE, pouco)
    linha = tabela.loc["intent_acc"]
    assert linha["teto"] == pytest.approx(0.84) and linha["diferenca"] < 0 and linha["ic_max"] >= 0.867 - 0.012 and linha["perto"]
    assert passou and tabela["perto"].all()                              # nas outras duas métricas ele está acima do publicado

    # longe em intenção: basta uma métrica longe para a etapa parar
    passou, tabela = etapa10a.perto_do_publicado(COM_ENTIDADE, [erradas(COM_ENTIDADE, 120)] * 3)
    assert not passou and not tabela.loc["intent_acc", "perto"] and tabela.loc["slot_f1", "perto"]


def test_fracao_do_caminho():
    piso, teto = [erradas(COM_ENTIDADE, 200)] * 3, [erradas(COM_ENTIDADE, 40)] * 3      # 50% e 90% de intenção
    aluno = [erradas(COM_ENTIDADE, 80), erradas(COM_ENTIDADE, 80), erradas(COM_ENTIDADE, 80)]   # 80%
    fracao = etapa10a.fracao_do_caminho(COM_ENTIDADE, piso, teto, aluno, n=300)
    linha = fracao.loc["intent_acc"]
    assert linha["valor"] == pytest.approx(0.75) and linha["ic_min"] < 0.75 < linha["ic_max"]
    assert list(fracao.columns) == ["valor", "ic_min", "ic_max"]
    # nos extremos: quem está no teto cobre tudo, quem está no piso não cobre nada
    assert etapa10a.fracao_do_caminho(COM_ENTIDADE, piso, teto, teto, n=50).loc["exact_match"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert etapa10a.fracao_do_caminho(COM_ENTIDADE, piso, teto, piso, n=50).loc["exact_match"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_quem_segue_leva_sempre_o_a_e_o_b2():
    gabarito = [Previsao("alarm_set")] * 300
    a, b1 = [respostas(gabarito, 60)] * 3, [respostas(gabarito, 90)] * 3

    com_ganho = etapa10a.quem_segue(gabarito, {"A": a, "B1": b1, "B2": [respostas(gabarito, 30)] * 3})
    assert com_ganho["seguem"] == ["A", "B2"] and com_ganho["b2_passa_na_regra_da_etapa_6"] and com_ganho["principal"] == "B2"

    # o B2 não passa na regra da Etapa 6: segue mesmo assim (os mesmos alunos do caminho do zero), e isso fica dito
    sem_ganho = etapa10a.quem_segue(gabarito, {"A": a, "B1": b1, "B2": [respostas(gabarito, 88)] * 3})
    assert sem_ganho["seguem"] == ["A", "B2"] and not sem_ganho["b2_passa_na_regra_da_etapa_6"] and sem_ganho["principal"] == "A"
    assert sem_ganho["exact_match_em_dev"] == pytest.approx({"A": 0.8, "B2": 212 / 300})
    assert "intent_acc" in sem_ganho["b2_menos_b1"].index


def test_comparar_tudo_segue_a_lista_fechada():
    nomes = {lado for _, a, b, _ in etapa10a.COMPARACOES for lado in (a, b)}
    assert nomes == {"zero:teto", "zero:piso", "zero:A", "zero:B1", "zero:B2", "teto", "piso", "A", "B1", "B2", "fechado", "aberto"}
    grupos = {nome: [erradas(COM_ENTIDADE, 100)] * 2 for nome in nomes}
    grupos["teto"], grupos["A"] = [erradas(COM_ENTIDADE, 40)] * 2, [erradas(COM_ENTIDADE, 80)] * 2
    tabela = etapa10a.comparar_tudo(COM_ENTIDADE, grupos)
    assert tabela["comparacao"].drop_duplicates().tolist() == [nome for nome, *_ in etapa10a.COMPARACOES]
    por = tabela.set_index(["comparacao", "metrica"])
    assert por.loc[("Aluno A − teto", "intent_acc"), "diferenca"] == pytest.approx(-0.10)          # B − A: aluno menos teto
    assert por.loc[("Teto: XLM-R − do zero", "intent_acc"), "diferenca"] == pytest.approx(+0.15)   # XLM-R menos do zero
    assert por.loc[("Aluno A − teto", "intent_acc"), "distinguivel"] and not por.loc[("Aluno B2 − aluno B1", "intent_acc"), "distinguivel"]


@pytest.mark.skipif(not (ENTREGAS / "06_destilacao" / "decisao.json").exists(), reason="as entregas das Etapas 4b e 6 não estão em disco")
def test_grupos_do_caminho_do_zero_le_as_entregas():
    grupos = etapa10a.grupos_do_caminho_do_zero("test")
    assert set(grupos) == {"zero:teto", "zero:piso", "zero:A", "zero:B1", "zero:B2"}
    tamanhos = {len(previsoes) for grupo in grupos.values() for previsoes in grupo}
    assert all(len(grupo) == 3 for grupo in grupos.values()) and len(tamanhos) == 1
    # o B2 do caminho do zero é o da temperatura que a Etapa 6 escolheu, e não outro
    da_6 = json.loads((ENTREGAS / "06_destilacao" / "decisao.json").read_text(encoding="utf-8"))
    do_arquivo = carregar_previsoes(ENTREGAS / "06_destilacao" / "previsoes" / f"aluno_b2_t{da_6['temperatura_do_b2']}_semente_0_test.parquet")[1]
    assert grupos["zero:B2"][0] == do_arquivo


# --------------------------------------------------------------------------
# Os quatro tempos e a retomada
# --------------------------------------------------------------------------


def test_os_quatro_tempos_e_a_retomada(etapa, monkeypatch):
    dev = etapa["dev"]
    monkeypatch.setattr(etapa10a, "PUBLICADO", {m: (0.0, 0.0) for m in etapa10a.PUBLICADO})   # qualquer teto fica "perto"

    # 1. no começo só a busca pode rodar; o resto depende da receita, que ainda não existe
    assert etapa10a.faltam() == [(v, 0) for v in etapa10a.BUSCA]
    assert len(etapa10a.faltam(tudo=True)) == 30
    with pytest.raises(FileNotFoundError, match="receita"):
        etapa10a.definicao("teto")
    with pytest.raises(FileNotFoundError, match="busca ainda não acabou"):
        etapa10a.decidir_receita()
    for variante, semente in etapa10a.faltam():
        etapa10a.executar(variante, semente, "cpu")
    assert etapa10a.faltam() == []                                         # a busca acabou: falta decidir

    # 2. a regra aponta a vencedora, e a decisão gravada não é refeita
    decidido = etapa10a.decidir_receita()
    em_dev = {v: etapa10a.carregar(v, 0)["previsoes_dev"] for v in etapa10a.BUSCA}
    assert decidido["vencedora"] == etapa10a.escolher_receita(ouro(dev), em_dev)[0]
    assert decidido["receita"] == etapa10a.BUSCA[decidido["vencedora"]]["receita"] == etapa10a.definicao("teto")["receita"]
    assert etapa10a.decidir_receita() == decidido

    # 3. agora só o teto; a semente 0 é a própria execução vencedora da busca
    assert etapa10a.faltam() == [("teto", 0), ("teto", 1), ("teto", 2)]
    pasta = etapa10a.executar("teto", 0, "cpu")
    teto_0, da_busca = etapa10a.carregar("teto", 0), etapa10a.carregar(decidido["vencedora"], 0)
    assert teto_0["meta"]["reaproveitada_de"] == decidido["vencedora"] and teto_0["meta"]["variante"] == "teto"
    assert teto_0["previsoes_dev"] == da_busca["previsoes_dev"] and (pasta / "modelo" / "cabecas.pt").exists()

    # uma pasta sem meta.json é sobra de execução interrompida: é apagada e refeita
    sobra = etapa10a.pasta_da_execucao("teto", 1)
    sobra.mkdir(parents=True)
    (sobra / "historico.csv").write_text("lixo", encoding="utf-8")
    assert ("teto", 1) in etapa10a.faltam()
    etapa10a.executar("teto", 1, "cpu")
    assert len(etapa10a.carregar("teto", 1)["historico"]) == decidido["receita"]["epocas"]
    etapa10a.executar("teto", 2, "cpu")

    # o que está gravado não é treinado de novo
    def nao_treina(*a, **k):
        raise AssertionError("treinou de novo")
    with monkeypatch.context() as m:
        m.setattr(etapa10a, "treinar", nao_treina)
        assert etapa10a.executar("teto", 1, "cpu") == sobra

    # 4. o teto é medido contra o publicado; só depois de passar os alunos podem rodar
    assert etapa10a.faltam() == []
    medido = etapa10a.medir_portao()
    assert medido["passou"] and set(medido["no_teste"]) == set(etapa10a.PUBLICADO)
    assert all((etapa10a.pasta_da_execucao("teto", s) / "previsoes_test.parquet").exists() for s in etapa10a.SEMENTES)
    restantes = etapa10a.faltam()
    assert len(restantes) == 7 * 3 and {v for v, _ in restantes} == set(etapa10a.VARIANTES) - {"teto"}
    assert "perto" in etapa10a.situacao() and "9 de 30" in etapa10a.situacao()

    # um teto longe do publicado tranca a etapa: nada mais pode rodar
    com_portao(passou=False)
    assert etapa10a.faltam() == [] and len(etapa10a.faltam(tudo=True)) == 21
    monkeypatch.setattr(etapa10a, "_rodar", nao_treina)
    assert etapa10a.onda(processos=1) is False


def test_a_onda_roda_a_etapa_inteira_e_abre_o_teste_so_para_os_escolhidos(etapa, monkeypatch):
    monkeypatch.setattr(etapa10a, "PUBLICADO", {m: (0.0, 0.0) for m in etapa10a.PUBLICADO})
    monkeypatch.setattr(etapa10a, "_rodar", lambda variante, semente: 0 if etapa10a.executar(variante, semente, "cpu") else 1)
    assert etapa10a.onda(processos=1) is True
    assert etapa10a.faltam(tudo=True) == []

    teste = json.loads((etapa10a.pasta_da_etapa() / "teste.json").read_text(encoding="utf-8"))
    T = teste["temperatura_do_b2"]
    em_dev = {t: etapa10a.previsoes(etapa6.variante_do_b2(t), "dev") for t in etapa10a.TEMPERATURAS}
    assert T == etapa6.escolher_temperatura(ouro(etapa["dev"]), em_dev)[0]                    # a regra da Etapa 6, sem mudança
    assert teste["abertas_no_teste"] == ["teto", "piso", "aluno_a", "aluno_b1", etapa6.variante_do_b2(T)]
    # as temperaturas que não foram escolhidas nunca são medidas no teste
    com_teste = {v for v in etapa10a.VARIANTES if (etapa10a.pasta_da_execucao(v, 0) / "previsoes_test.parquet").exists()}
    assert com_teste == set(teste["abertas_no_teste"])

    grupos = etapa10a.grupos_do_encoder("test", T)
    assert set(grupos) == {"teto", "piso", "A", "B1", "B2"} and all(len(g) == 3 for g in grupos.values())
    assert "teste aberto" in etapa10a.situacao() and "30 de 30" in etapa10a.situacao()

    # uma execução com erro para a onda
    monkeypatch.setattr(etapa10a, "faltam", lambda tudo=False: [("teto", 0)])
    monkeypatch.setattr(etapa10a, "_rodar", lambda variante, semente: 1)
    assert etapa10a.onda(processos=1) is False


# --------------------------------------------------------------------------
# Os alunos: os mesmos dados da Etapa 6, nenhum rótulo humano
# --------------------------------------------------------------------------


def test_aluno_treina_e_para_pelos_rotulos_do_professor(etapa):
    com_receita()
    dev = etapa["dev"]
    pasta = etapa10a.executar("aluno_a", 0, "cpu")
    do_aluno = etapa10a.carregar("aluno_a", 0)
    meta, historico, previsoes = do_aluno["meta"], do_aluno["historico"], do_aluno["previsoes_dev"]

    # a época escolhida é a melhor contra o dev do professor, e é com ele que `meta["dev"]` foi medido
    do_professor = professor_que_troca(dev)
    assert meta["melhor_epoca"] == int(historico.loc[historico["dev_exact_match"].idxmax(), "epoca"])
    assert meta["dev"] == pytest.approx(qualidade.avaliar(do_professor, previsoes))
    assert "professor fechado" in meta["origem"]["treino"] and "professor fechado" in meta["origem"]["dev"]
    assert "vocabulario" not in meta["origem"]                             # o vocabulário de features é coisa da rede do zero

    # o aluno aprendeu a troca do professor: concorda com ele, não com os rótulos humanos
    assert qualidade.avaliar(do_professor, previsoes)["intent_acc"] > 0.8
    assert qualidade.avaliar(ouro(dev), previsoes)["intent_acc"] < 0.5
    # as respostas gravadas cobrem dev inteiro, na ordem da Etapa 1
    assert carregar_previsoes(pasta / "previsoes_dev.parquet")[0] == dev["id"].tolist()
    assert meta["receita"]["passos_por_epoca"] == 0 and meta["passos_de_treino"] == RAPIDA["epocas"] * ceil(len(dev) / RAPIDA["lote"])
    assert sum(meta["parametros_por_parte"].values()) == meta["parametros"] and meta["bytes_em_disco"] > 0


def test_dados_dos_alunos_sao_os_da_etapa_6(etapa):
    for variante in ("piso", "aluno_a", "aluno_b1", "aluno_b2_t4"):
        daqui, de_la = etapa10a.dados_da_variante(variante), etapa6.dados_da_variante(variante)
        assert daqui["treino"].equals(de_la["treino"]) and daqui["dev"].equals(de_la["dev"]), variante
        assert "vocabulario" not in daqui and "vocabulario" not in daqui["origem"]
    assert len(etapa10a.dados_da_variante("teto")["treino"]) == len(etapa["treino"])
    assert "rótulos humanos" in etapa10a.dados_da_variante("teto")["origem"]["treino"]
    assert "logits_professor" in etapa10a.dados_da_variante("aluno_b2_t4")["treino"].columns
    assert "logits_professor" not in etapa10a.dados_da_variante("aluno_b1")["treino"].columns


def test_b2_aprende_a_intencao_pelos_logits(etapa):
    com_receita()
    pasta = etapa10a.executar("aluno_b2_t2", 0, "cpu")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    assert (meta["config"]["temperatura"], meta["config"]["peso_rotulo"]) == (2.0, 0.0)
    assert meta["exemplos_de_treino"] == len(etapa["treino"]) - 1          # o pedido sem resposta do professor fica fora
    assert meta["origem"]["logits"].endswith("logits_treino.npy")
    assert meta["dev"]["intent_acc"] > 0.8                                 # contra o dev do professor
    assert "perda_intencao" in etapa10a.carregar("aluno_b2_t2", 0)["historico"].columns


def test_piso_recebe_os_passos_do_treino_completo(etapa):
    com_receita({**RAPIDA, "epocas": 3})
    do_treino_completo = ceil(len(etapa["treino"]) / RAPIDA["lote"])       # 8 exemplos, lote de 4: 2 passos por época
    assert etapa10a.definicao("piso")["receita"]["passos_por_epoca"] == do_treino_completo
    assert "passos_por_epoca" not in etapa10a.definicao("aluno_a")["receita"]

    pasta = etapa10a.executar("piso", 0, "cpu")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    assert meta["exemplos_de_treino"] == 3 and "rótulos humanos" in meta["origem"]["dev"]
    # com 3 exemplos, uma época "por passada" teria 1 passo; o piso tem os mesmos 2 de quem treina com tudo
    assert etapa10a.carregar("piso", 0)["historico"]["passos"].tolist() == [do_treino_completo] * 3


def test_controle_de_32_bits_so_muda_a_precisao_e_fica_fora_da_grade(etapa):
    com_receita()
    teto, controle = etapa10a.definicao("teto"), etapa10a.definicao("teto_em_32_bits")
    assert controle["config"] == teto["config"]
    assert controle["receita"] == {**teto["receita"], "precisao_mista": False}
    assert etapa10a.dados_da_variante("teto_em_32_bits")["treino"].equals(etapa10a.dados_da_variante("teto")["treino"])
    # é um controle, não uma variante: não aparece na grade nem no que falta rodar
    assert all(v != "teto_em_32_bits" for v, _ in etapa10a.grade()) and "teto_em_32_bits" not in etapa10a.VARIANTES
