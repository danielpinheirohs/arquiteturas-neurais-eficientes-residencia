"""Etapa 11c: quem é comprimido, com que receita, a troca que não vaza para as etapas de origem e as regras do comparativo.

Nada aqui treina, usa a placa, a API ou a rede. Os testes que leem as entregas
de verdade só olham decisões e marcas; os que mexem em pastas usam uma pasta
temporária no lugar do cache de execuções.
"""

import json

import pytest

from especialista.caminhos import ENTREGAS
from especialista.experimentos import etapa7, etapa8, etapa10b, etapa10d, etapa11b, etapa11c, execucao

NOVO = etapa11c.PROFESSOR
TEM_ENTREGAS = all((ENTREGAS / etapa / nome).exists() for etapa, nome in (
    ("07_poda", "decisao.json"), ("08_quantizacao", "decisao.json"), ("10b_encoder_poda", "decisao.json"), ("10c_encoder_quantizacao", "decisao.json"),
    (etapa11b.ETAPA, "decisao.json"), (etapa11b.ETAPA, "sistemas.json"), ("11a_professor_aberto_maior", "sistemas.json"), (etapa10d.ETAPA, "regras_v2.json")))
com_entregas = pytest.mark.skipif(not TEM_ENTREGAS, reason="as entregas das Etapas 7 a 11b não estão em disco")


@pytest.fixture
def descartavel(tmp_path, monkeypatch):
    """O cache de execuções numa pasta temporária, com um `regras.json` qualquer: é o que o teste de fumaça faz."""
    monkeypatch.setattr(execucao, "EXECUCOES", tmp_path)
    (tmp_path / etapa11c.ETAPA).mkdir()
    (tmp_path / etapa11c.ETAPA / "regras.json").write_text(json.dumps({"escritas_em": "antes"}), encoding="utf-8")
    return tmp_path


def test_emprestar_devolve_o_que_estava_mesmo_com_erro():
    antes = (etapa7.ETAPA, etapa7.modelos_a_podar, etapa7.EPOCAS)
    with pytest.raises(RuntimeError):
        with etapa11c.emprestar(etapa7, ETAPA="outra", modelos_a_podar=lambda: {}, EPOCAS=1):
            assert (etapa7.ETAPA, etapa7.modelos_a_podar(), etapa7.EPOCAS) == ("outra", {}, 1)
            raise RuntimeError("no meio")
    assert (etapa7.ETAPA, etapa7.modelos_a_podar, etapa7.EPOCAS) == antes == ("07_poda", antes[1], 30)


@com_entregas
def test_alunos_saem_da_regra_de_quem_representa_e_do_b2_para_comparar():
    decisao = json.loads((ENTREGAS / etapa11c.DECISAO_DOS_ALUNOS).read_text(encoding="utf-8"))
    alunos = etapa11c.alunos()
    por_caminho = {c: [a["sigla"] for a in alunos.values() if a["caminho"] == c] for c in etapa11c.ORIGEM}
    for c, siglas in por_caminho.items():
        representa = decisao["quem_representa_o_cenario_b"][c]["representa"]
        assert siglas == [representa] + [s for s in etapa11c.TAMBEM if s != representa]            # o representante primeiro; o B2, se não for ele
        assert [a["representa"] for a in alunos.values() if a["caminho"] == c] == [True] + [False] * (len(siglas) - 1)
    assert por_caminho == {"zero": ["B2"], "xlmr": ["B1", "B2"]}                                     # o que a Etapa 11b relatou
    declaradas = json.loads((ENTREGAS / etapa11b.ETAPA / "sistemas.json").read_text(encoding="utf-8"))["sistemas"]
    for nome, a in alunos.items():
        assert NOVO in nome and nome.startswith(a["caminho"] + "__") and a["chave"] in declaradas and NOVO in a["sinal"]["treino"]
        assert all(pasta.is_relative_to(ENTREGAS / etapa11b.ETAPA) and (pasta / "config.json").exists() for pasta in a["pastas"].values())   # parte da entrega da 11b
    with pytest.raises(ValueError):
        etapa11c.alunos("qwen3_8b__fp16__colab_l4")                                                 # a 11b não treinou alunos para outro professor


@com_entregas
def test_receitas_sao_as_das_decisoes_de_origem_sem_escolha_nova():
    receitas = etapa11c.receitas()
    for caminho, origem in etapa11c.ORIGEM.items():
        poda, quantizacao = (json.loads((ENTREGAS / origem[e] / "decisao.json").read_text(encoding="utf-8")) for e in ("poda", "quantizacao"))
        assert receitas[caminho]["poda"] == poda["principal"]["variante"] and receitas[caminho]["forma"] == quantizacao["final"]["forma"]
        antigo = receitas[caminho]["no_aluno_antigo_do_cenario_b"]                                   # a cadeia antiga do cenário B recebeu a mesma receita
        assert (antigo["poda"], antigo["forma"]) == (receitas[caminho]["poda"], receitas[caminho]["forma"]) and "b2" in antigo["modelo"]
        assert etapa11c.variantes(caminho) == (receitas[caminho]["poda"], "sem_poda")
    assert receitas["zero"]["reajuste"]["epocas"] == etapa7.EPOCAS and receitas["xlmr"]["reajuste"]["epocas"] == etapa10b.EPOCAS
    assert receitas["zero"]["poda"] in etapa7.VARIANTES and receitas["xlmr"]["poda"] in etapa10b.VARIANTES and receitas["zero"]["forma"] in etapa8.CANDIDATAS


@com_entregas
def test_grade_tem_a_ordem_da_cadeia_e_as_pastas_trazem_o_professor(descartavel):
    grade = etapa11c.grade()
    passos = [e[0] for e in grade]
    assert passos == sorted(passos, key=etapa11c.PASSOS.index) and {p: passos.count(p) for p in etapa11c.PASSOS} == {"poda": 9, "quantizar": 18, "latencia": 6}
    assert all(s == etapa11c.SEMENTE_DO_CUSTO for p, _, _, s in grade if p == "latencia")
    assert etapa11c.faltam() == grade and "0 de 33" in etapa11c.situacao()
    regras = etapa11c.montar_regras()
    assert regras["execucoes"]["total"] == 33 and set(regras["alunos"]) == set(etapa11c.alunos()) and set(regras["conferencias"]) >= {"poda", "quantizacao", "se_nao_passar"}
    json.dumps(regras)
    for passo, nome, variante, semente in grade:
        pasta = etapa11c.pasta_da_poda(nome, semente) if passo == "poda" else etapa11c.pasta_da_quantizacao(nome, variante, semente)
        assert pasta.is_relative_to(descartavel / etapa11c.ETAPA) and NOVO in str(pasta)


@com_entregas
@pytest.mark.parametrize("caminho, origem", [("zero", etapa7), ("xlmr", etapa10b)])
def test_podar_chama_a_etapa_de_origem_com_o_aluno_novo_e_devolve_tudo_no_fim(descartavel, monkeypatch, caminho, origem):
    nome = next(n for n, a in etapa11c.alunos().items() if a["caminho"] == caminho)
    antes = (origem.ETAPA, origem.modelos_a_podar, origem.dados_do_modelo, origem.EPOCAS)
    visto = {}

    def espia(modelo, variante, semente, dispositivo=None):
        # as épocas do reajuste, como cada etapa de origem as lê dentro de `executar`
        epocas = origem.receita_do_reajuste(modelo).epocas if caminho == "zero" else origem.receita_do_reajuste(modelo, origem.definicao(variante)["epocas"]).epocas
        visto.update(etapa=origem.ETAPA, modelos=list(origem.modelos_a_podar()), pasta=origem.pasta_da_execucao(modelo, variante, semente), epocas=epocas, chamada=(modelo, variante, semente))
        if caminho == "zero":   # os dados que o reajuste leria: os do professor novo (lê os arquivos da Etapa 11a; no XLM-R é a mesma função)
            visto["origem"] = origem.dados_do_modelo(modelo)["origem"]
        return visto["pasta"]

    monkeypatch.setattr(origem, "executar", espia)
    pasta = etapa11c.podar(nome, 0)
    assert visto["chamada"] == (nome, etapa11c.receitas()[caminho]["poda"], 0) and visto["modelos"] == [nome]      # a etapa de origem só enxerga o aluno novo
    assert pasta == etapa11c.pasta_da_poda(nome, 0) and pasta.is_relative_to(descartavel) and visto["etapa"] == f"{etapa11c.ETAPA}/poda"
    assert visto["epocas"] == etapa11c.FUMACA["epocas"][caminho]                                                       # fora do cache de verdade, o reajuste é o curto
    if caminho == "zero":
        assert visto["origem"]["professor"] == NOVO and all(NOVO in visto["origem"][campo] for campo in ("treino", "dev", "logits"))
    assert (origem.ETAPA, origem.modelos_a_podar, origem.dados_do_modelo, origem.EPOCAS) == antes                     # nada ficou trocado
    assert nome not in origem.modelos_a_podar()


def test_sem_regras_nada_roda(tmp_path, monkeypatch):
    monkeypatch.setattr(execucao, "EXECUCOES", tmp_path)
    for chamada in (lambda: etapa11c.podar("x", 0), lambda: etapa11c.quantizar("x", "sem_poda", 0), lambda: etapa11c.medir_latencia("x", "sem_poda")):
        with pytest.raises(FileNotFoundError):
            chamada()


@com_entregas
def test_latencia_com_a_maquina_ocupada_e_guardada_e_refeita(descartavel, monkeypatch):
    nome = next(n for n, a in etapa11c.alunos().items() if a["caminho"] == "zero")
    pasta = etapa11c.pasta_da_quantizacao(nome, "sem_poda", 0)
    pasta.mkdir(parents=True)
    medidas = iter([0.9, 0.6, 0.03])

    def medir(modelo, variante, base, semente, rodadas):
        assert (pasta / "latencia.json").exists() is False and base == etapa11c.base_da_quantizacao()
        (pasta / "latencia.json").write_text(json.dumps({"nucleos_de_outros": next(medidas)}), encoding="utf-8")
        (pasta / "latencia.npz").write_bytes(b"tempos")

    monkeypatch.setattr(etapa8, "medir_latencia", medir)
    devolvida, quieta = etapa11c.medir_latencia(nome, "sem_poda", tentativas=3, espera=0)
    assert devolvida == pasta and quieta and json.loads((pasta / "latencia.json").read_text(encoding="utf-8"))["nucleos_de_outros"] == 0.03
    recusadas = sorted(a.name for a in pasta.glob("latencia_recusada_*"))
    assert recusadas == ["latencia_recusada_1.json", "latencia_recusada_1.npz", "latencia_recusada_2.json", "latencia_recusada_2.npz"]      # nenhuma medida é apagada
    assert json.loads((pasta / "latencia_recusada_1.json").read_text(encoding="utf-8"))["nucleos_de_outros"] == 0.9


@com_entregas
def test_o_comparativo_le_a_etapa_11_e_o_da_10d_continua_como_foi_entregue(tmp_path):
    antigas = json.loads((ENTREGAS / etapa10d.ETAPA / "regras_v2.json").read_text(encoding="utf-8"))
    assert list(etapa10d.registro(base=tmp_path)) == antigas["sistemas_no_registro"]                                   # a 10d não enxerga a Etapa 11
    fichas = etapa11c.registro()
    novas = [c for c in fichas if c not in antigas["sistemas_no_registro"]]
    alunos = etapa11c.alunos()
    assert list(fichas)[:44] == antigas["sistemas_no_registro"] and len(novas) >= 2 + 4 + 4 * len(alunos)             # professores da 11a, alunos da 11b e as formas desta etapa
    for a in alunos.values():
        assert [fichas[f"{a['chave']}/{p}"]["ponto"] for p in ("podado", "onnx", "int8", "int8_sem_poda")] == ["podado", "onnx", "int8", "int8_sem_poda"]
        assert fichas[f"{a['chave']}/int8"]["professor"] == NOVO and fichas[f"{a['chave']}/int8"]["modelo"] == fichas[a["chave"]]["modelo"] and not fichas[f"{a['chave']}/int8"].get("final")
        assert fichas[a["chave"]]["eficiencia"] == (etapa11c.ETAPA, {"modelo": next(n for n, x in alunos.items() if x is a), "variante": "sem_poda", "forma": "pytorch"})
    assert etapa10d.finais(fichas) == antigas["finais"]                                                                 # os modelos finais continuam os da 10d
    cadeias = etapa11c.cadeias_do_registro(fichas)
    assert list(cadeias)[:6] == list(antigas["cadeias"]) and [c for c in cadeias if NOVO in c] == [f"{a['caminho']}/{a['sigla']}@{NOVO}" for a in alunos.values()]
    assert all(chave in fichas for c in cadeias.values() for _, chave in c["degraus"])


@com_entregas
def test_lista_do_comparativo_guarda_as_43_da_10d_e_gera_as_familias_novas_por_regra():
    antigas = json.loads((ENTREGAS / etapa10d.ETAPA / "regras_v2.json").read_text(encoding="utf-8"))
    fichas = etapa11c.registro()
    lista = etapa11c.lista_de_comparacoes(fichas)
    pares = {(c["b"], c["a"]): c for c in lista}
    assert len(pares) == len(lista)
    for antiga in antigas["comparacoes"]:   # as da 10d estão todas, com o mesmo id e as mesmas famílias na frente
        daqui = pares[(antiga["b"], antiga["a"])]
        assert daqui["id"] == antiga["id"] and daqui["familias"][:len(antiga["familias"])] == antiga["familias"]
    b2, b1x, b2x = f"aluno_b2@{NOVO}", f"xlmr/aluno_b1@{NOVO}", f"xlmr/aluno_b2@{NOVO}"
    assert pares[(f"{b2}/int8", "aluno_b2/int8")]["familias"] == ["troca_de_professor"] and pares[(f"{b2x}/podado", "xlmr/aluno_b2/podado")]["familias"] == ["troca_de_professor"]
    assert (f"{b1x}/int8", "xlmr/aluno_b1/int8") not in pares                                          # o B1 antigo do XLM-R não foi comprimido: não há par
    assert pares[("aluno_a/int8", f"{b2}/int8")]["familias"] == ["cenario_a_menos_b"] and pares[("aluno_a/int8", "aluno_b2/int8")]["familias"] == ["ponta_a_ponta", "cenario_a_menos_b"]
    assert pares[("xlmr/aluno_a/int8", f"{b1x}/int8")]["familias"] == ["cenario_a_menos_b"] and pares[("xlmr/aluno_a", f"{b2x}")]["familias"] == ["cenario_a_menos_b"]
    assert pares[(f"{b1x}/int8", b1x)]["familias"] == ["compressao_inteira"] and pares[("teto/int8", "teto")]["familias"] == ["ponta_a_ponta", "compressao_inteira"]
    assert pares[(f"{b2x}/int8", f"{b2}/int8")]["familias"] == ["entre_caminhos"] and pares[(f"{b2}/int8", NOVO)]["familias"] == ["aluno_professor"]
    assert pares[(f"{b2}/int8", f"{b2}/int8_sem_poda")]["familias"] == ["poda_nos_8_bits"]
    assert pares[(NOVO, "claude")]["familias"] == ["entre_professores"] and pares[(NOVO, "qwen")]["familias"] == ["entre_professores"] and ("qwen", "qwen") not in pares
    principais = etapa11c.primarias(fichas)
    assert [(p["b"], p["a"]) for p in principais] == [("aluno_a/int8", f"{b2}/int8"), ("xlmr/aluno_a/int8", f"{b1x}/int8")] and all((p["b"], p["a"]) in pares for p in principais)
    # um professor que chegue depois entra sozinho, pela marca de professor aberto. O Qwen3-32B chegou assim em 19/11
    # (a extensão da Etapa 11a declarou a ficha dele); por isso o professor de mentira daqui tem outra chave
    com_mais_um = {**fichas, "qwen3_99b__de_mentira": {**fichas[NOVO], "curto": "Qwen3-99B"}}
    mais = {(c["b"], c["a"]) for c in etapa11c.lista_de_comparacoes(com_mais_um)} - set(pares)
    assert mais == {("qwen3_99b__de_mentira", "claude"), ("qwen3_99b__de_mentira", "qwen")}
    if "qwen3_32b__fp16__colab_g4" in fichas:
        assert pares[("qwen3_32b__fp16__colab_g4", "claude")]["familias"] == pares[("qwen3_32b__fp16__colab_g4", "qwen")]["familias"] == ["entre_professores"]


def test_regras_do_comparativo_continuam_a_numeracao_da_10d_e_nunca_sao_sobrescritas(tmp_path, monkeypatch):
    monkeypatch.setattr(execucao, "EXECUCOES", tmp_path)
    conteudo = {"comparacoes": [{"b": "x", "a": "y"}], "cadeias": {}, "sistemas_no_registro": ["x", "y"]}
    monkeypatch.setattr(etapa11c, "montar_regras_do_comparativo", lambda professor=NOVO, entregas=None: dict(conteudo))
    primeira = etapa11c.gravar_regras_do_comparativo()
    assert primeira.name == "regras_v3.json" and json.loads(primeira.read_text(encoding="utf-8"))["versao"] == 3 and etapa11c.gravar_regras_do_comparativo() == primeira
    texto = primeira.read_text(encoding="utf-8")
    conteudo["sistemas_no_registro"].append("qwen3_32b")   # o registro cresceu
    segunda = etapa11c.gravar_regras_do_comparativo()
    assert segunda.name == "regras_v4.json" and primeira.read_text(encoding="utf-8") == texto and json.loads(segunda.read_text(encoding="utf-8"))["versao"] == 4


# ---- depois da cadeia: as contas, a entrega e o comparativo (só leem o que está em disco)

ENTREGA = ENTREGAS / etapa11c.ETAPA
com_a_entrega = pytest.mark.skipif(not (ENTREGA / "comparativo" / "calculo.json").exists(), reason="a entrega da Etapa 11c ainda não foi gerada")


@com_a_entrega
def test_cada_conferencia_segue_a_regra_de_origem_e_a_decisao_so_relata():
    import pandas as pd

    conf = pd.read_csv(ENTREGA / "conferencias.csv")
    decisao = json.loads((ENTREGA / "decisao.json").read_text(encoding="utf-8"))
    principais = conf[conf["conta_para_a_resposta"]]
    assert len(principais) == 3 * len(etapa11c.alunos()) and set(principais["conferencia"]) == {"poda", "8 bits", "ONNX de 32 bits"}
    for _, l in principais.iterrows():   # o critério de cada conferência é o da regra de origem, em dev
        if l["conferencia"] == "poda":
            assert (l["metrica"], l["particao"]) == ("exact_match", "dev") and l["passa"] == (not l["ic_max"] < 0)
        elif l["conferencia"] == "8 bits":
            assert (l["metrica"], l["particao"]) == ("slot_f1", "dev") and l["passa"] == (l["diferenca"] >= -etapa8.LIMITE_DO_PLANO)
        else:
            assert l["passa"] == (l["com_a_receita"] == l["referencia"])
    assert decisao["decisoes_nesta_etapa"] == "nenhuma" and decisao["todas_as_receitas_passam"] == bool(principais["passa"].all())
    for nome, a in decisao["alunos"].items():
        assert a["passa_em_tudo"] == bool(principais[principais["aluno"] == nome]["passa"].all()) and a["poda"]["conferencia"]["passa"] == bool(
            principais[(principais["aluno"] == nome) & (principais["conferencia"] == "poda")]["passa"].iloc[0])
        for pastas in (a["pastas_por_semente"]["podado"], a["pastas_por_semente"]["int8"], a["pastas_por_semente"]["int8_sem_poda"]):
            for pasta in pastas.values():   # os modelos estão na entrega, por link físico para o cache
                arquivos = [x for x in (etapa11c.RAIZ / pasta).rglob("*") if x.is_file()]
                assert arquivos and all(x.stat().st_nlink > 1 for x in arquivos)


@com_a_entrega
def test_cada_ficha_declarada_aponta_para_uma_linha_e_para_respostas_que_existem():
    from especialista.experimentos import etapa9

    declarado = json.loads((ENTREGA / "sistemas.json").read_text(encoding="utf-8"))
    fichas, ler = etapa11c.registro(), etapa9._Tabelas(ENTREGAS)
    assert len(declarado["sistemas"]) == 4 * len(etapa11c.alunos()) and len(declarado["cadeias"]) == len(etapa11c.alunos())
    for chave in declarado["sistemas"]:
        f = fichas[chave]
        assert f["declarado_em"] == f"entregas/{etapa11c.ETAPA}/sistemas.json" and all(arquivo.exists() for lista in f["previsoes"].values() for arquivo in lista)
        for etapa, filtro in [f["resultado"], *f.get("outras_fontes", [])]:
            for particao in etapa11c.PARTICOES:
                etapa9.linha_gravada(ler(etapa), particao=particao, **filtro)   # dá erro se não houver exatamente uma
    for chave, mais in declarado["complementos"].items():   # a latência do aluno destilado, que a Etapa 11b não mediu, é lida desta etapa
        assert fichas[chave]["eficiencia"] == tuple(mais["eficiencia"]) and fichas[chave]["declarado_em"].startswith(f"entregas/{etapa11b.ETAPA}/")
        assert etapa9.linha_gravada(ler(etapa11c.ETAPA), particao="test", **mais["eficiencia"][1])["latencia_p50_ms"] > 0


@com_a_entrega
def test_o_comparativo_foi_calculado_pelas_regras_gravadas_e_reproduz_a_10d_dentro_dele():
    import pandas as pd

    calculo = json.loads((ENTREGA / "comparativo" / "calculo.json").read_text(encoding="utf-8"))
    regras = json.loads((ENTREGA / "comparativo" / calculo["regras"]["arquivo"]).read_text(encoding="utf-8"))
    assert calculo["regras"]["versao"] == regras["versao"] >= 4 and regras["gravado_em"] < calculo["calculado_em"]          # as regras vieram antes das contas
    assert etapa10d.sha256(ENTREGA / "comparativo" / calculo["regras"]["arquivo"]) == calculo["regras"]["sha256"]
    antigas = json.loads((ENTREGA / "comparativo" / "regras_v3.json").read_text(encoding="utf-8"))                        # a versão anterior continua lá, e só cresceu
    assert antigas["versao"] == 3 and set(antigas["sistemas_no_registro"]) < set(regras["sistemas_no_registro"])
    assert {(c["b"], c["a"]) for c in antigas["comparacoes"]} < {(c["b"], c["a"]) for c in regras["comparacoes"]}
    comparacoes = pd.read_csv(ENTREGA / "comparativo" / "comparacoes.csv")
    for particao in etapa11c.PARTICOES:   # a lista fechada inteira, e só ela, nas duas partições
        assert set(zip(comparacoes[comparacoes["particao"] == particao]["chave_b"], comparacoes[comparacoes["particao"] == particao]["chave_a"])) == {(c["b"], c["a"]) for c in regras["comparacoes"]}
    conferencia = pd.read_csv(ENTREGA / "comparativo" / "conferencia.csv")
    assert bool(conferencia[conferencia["gravado"] == True]["confere"].all())   # noqa: E712
    assert (conferencia["o_que"] == "linha da tabela final da Etapa 10d").sum() == 44 and (conferencia["o_que"] == "comparação da Etapa 10d").sum() == 2 * 43
    tabela = pd.read_csv(ENTREGA / "comparativo" / "tabela_final.csv")
    assert list(tabela["chave"]) == regras["sistemas_no_registro"] and not tabela[tabela["final"]]["chave"].str.contains("@").any()   # os modelos finais continuam os da 10d


@com_a_entrega
def test_as_regras_gravadas_do_comparativo_sao_as_do_registro_de_hoje():
    """Se uma entrega da lista mudar (um professor a mais, por exemplo), este teste avisa: é preciso gravar a versão seguinte das regras e calcular de novo."""
    regras, _ = etapa11c.carregar_regras_do_comparativo()
    hoje = json.loads(json.dumps(etapa11c.montar_regras_do_comparativo()))
    assert {k: v for k, v in regras.items() if k not in ("versao", "gravado_em")} == hoje
