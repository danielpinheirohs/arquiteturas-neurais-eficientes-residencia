"""Comparativo dos dois caminhos: o registro de sistemas, as comparações geradas por regra, a fronteira e o preço de cada ponto.

Os testes das regras usam registros de mentira, só com as marcas (caminho,
modelo, ponto, professor). Os que conferem a leitura das entregas usam as
entregas de verdade, se estão em disco, e não abrem nenhuma resposta: só
olham onde cada ficha manda ler. Nada aqui usa GPU, API ou rede.
"""

import json

import numpy as np
import pandas as pd
import pytest

from especialista.caminhos import ENTREGAS
from especialista.experimentos import etapa9, etapa10d

TEM_ENTREGAS = all((ENTREGAS / etapa / "decisao.json").exists() for etapa in ("08_quantizacao", "10a_encoder", "10b_encoder_poda", "10c_encoder_quantizacao"))
com_entregas = pytest.mark.skipif(not TEM_ENTREGAS, reason="as entregas das Etapas 8 e 10a a 10c não estão em disco")


def ficha(caminho, modelo, ponto, **campos) -> dict:
    return {"caminho": caminho, "modelo": modelo, "ponto": ponto, "curto": f"{modelo}, {ponto}", **campos}


def registro_de_mentira() -> dict:
    """Os dois caminhos em miniatura, com as chaves da Etapa 9 no caminho do zero e o prefixo no outro."""
    fichas = {"base_299": ficha(None, None, "base"), "base_11514": ficha(None, None, "base"), "claude": ficha(None, None, "professor"), "qwen": ficha(None, None, "professor")}
    for p in ("", "xlmr/"):
        c = "zero" if not p else "xlmr"
        fichas[p + "teto"] = ficha(c, "teto", "sem_compressao")
        fichas[p + "piso"] = ficha(c, "piso", "piso")
        for aluno, professor in (("aluno_a", "claude"), ("aluno_b1", "qwen"), ("aluno_b2", "qwen")):
            fichas[p + aluno] = ficha(c, aluno, "destilado", professor=professor)
        for modelo in ("aluno_a", "aluno_b2", "teto"):
            for ponto in ("podado", "onnx", "int8", "soltos", "int8_sem_poda"):
                fichas[f"{p}{modelo}/{ponto}"] = ficha(c, modelo, ponto, professor=fichas[p + modelo].get("professor"), de_partida=p + modelo, final=(modelo, ponto) == ("aluno_a", "int8"))
    return fichas


def test_filtro_em_leva_o_filtro_do_teste_para_dev():
    filtro = {"particao": "test", "comparacao": "test · Aluno A − professor fechado", "modelo": "teto", "forma": "fp32"}
    assert etapa10d.filtro_em(filtro, "dev") == {"particao": "dev", "comparacao": "dev · Aluno A − professor fechado", "modelo": "teto", "forma": "fp32"}
    assert etapa10d.filtro_em(filtro, "test") == filtro


def test_lista_de_comparacoes_sai_das_marcas_do_registro(tmp_path):
    fichas = registro_de_mentira()
    lista = etapa10d.lista_de_comparacoes(fichas, tmp_path)
    pares = {(c["b"], c["a"]): c for c in lista}
    assert len(pares) == len(lista)   # cada par aparece uma vez
    # entre caminhos: B é o outro caminho e A é a rede do zero, no mesmo modelo e no mesmo ponto; o ONNX e as formas fora da cadeia não entram
    entre = [c for c in lista if "entre_caminhos" in c["familias"]]
    assert len(entre) == 11 and all(c["b"] == "xlmr/" + c["a"] for c in entre)
    assert ("xlmr/aluno_a/onnx", "aluno_a/onnx") not in pares and ("xlmr/aluno_a/soltos", "aluno_a/soltos") not in pares
    # o final contra o final é o mesmo par do aluno A em INT8: uma comparação, duas famílias
    assert pares[("xlmr/aluno_a/int8", "aluno_a/int8")]["familias"] == ["entre_caminhos", "finais"]
    # cada aluno contra o seu professor, nos dois caminhos; a de ponta a ponta "a" é o mesmo par
    assert pares[("aluno_b1", "qwen")]["familias"] == ["aluno_professor"] and pares[("xlmr/aluno_a", "claude")]["familias"] == ["aluno_professor"]
    assert pares[("xlmr/aluno_a/int8", "claude")]["familias"] == ["aluno_professor", "ponta_a_ponta"]
    # as dez da Etapa 9 em cada caminho, com as chaves do caminho; o que os caminhos dividem não muda de chave
    ponta = [c for c in lista if "ponta_a_ponta" in c["familias"]]
    assert len(ponta) == 2 * len(etapa9.PONTA_A_PONTA)
    assert ("xlmr/aluno_a/int8", "base_299") in pares and ("xlmr/aluno_a/int8", "xlmr/piso") in pares and ("xlmr/teto/int8", "xlmr/teto") in pares
    # só as do caminho do zero têm o número da Etapa 9 para conferir
    assert pares[("aluno_a/int8", "teto")]["gravado"] == (etapa10d.E9, "comparacoes.csv", {"letra": "c"}) and pares[("xlmr/aluno_a/int8", "xlmr/teto")]["gravado"] is None
    assert [c["b"] for c in lista if c["familias"] == ["poda_nos_8_bits"]] == [f"{p}{m}/int8" for p in ("", "xlmr/") for m in ("aluno_a", "aluno_b2", "teto")]


def test_cadeias_sao_as_da_etapa_9_em_cada_caminho(tmp_path):
    fichas = registro_de_mentira()
    cadeias = etapa10d.cadeias_do_registro(fichas, tmp_path)
    assert list(cadeias) == ["zero/A", "zero/B", "zero/teto", "xlmr/A", "xlmr/B", "xlmr/teto"]
    assert cadeias["zero/A"]["degraus"] == etapa9.CADEIAS["A"]["degraus"]
    assert [chave for _, chave in cadeias["xlmr/B"]["degraus"]] == ["qwen", "xlmr/aluno_b2", "xlmr/aluno_b2/podado", "xlmr/aluno_b2/onnx", "xlmr/aluno_b2/int8"]
    del fichas["xlmr/teto/onnx"]   # uma cadeia sem um dos degraus não é montada pela metade
    assert "xlmr/teto" not in etapa10d.cadeias_do_registro(fichas, tmp_path)


def test_um_sistema_declarado_por_uma_etapa_nova_entra_sozinho(tmp_path):
    """É o que a Etapa 11 vai precisar: declarar um caminho e um sistema num `sistemas.json`, sem mexer neste módulo."""
    entregas = tmp_path / "entregas"
    (entregas / "11b_novo").mkdir(parents=True)
    declarado = {
        "caminhos": {"novo": {"nome": "Caminho novo", "curto": "Novo", "prefixo": "novo/"}},
        "sistemas": {"novo/aluno_b2": {"nome": "Aluno B2 novo", "curto": "Aluno B2", "caminho": "novo", "modelo": "aluno_b2", "ponto": "destilado", "professor": "qwen", "rotulos_humanos": 8,
                                       "resultado": ["11b_novo", {"variante": "aluno_b2"}], "previsoes": {"dev": ["entregas/11b_novo/previsoes/b2_dev.parquet"], "test": ["entregas/11b_novo/previsoes/b2_test.parquet"]}}},
    }
    (entregas / "11b_novo" / "sistemas.json").write_text(json.dumps(declarado), encoding="utf-8")
    caminhos, fichas_novas, _ = etapa10d.fichas_declaradas(entregas)   # sem lista, lê todas as entregas em disco
    assert etapa10d.fichas_declaradas(entregas, valem=("09_comparativo",)) == ({}, {}, {})   # com lista, só as que valem
    assert etapa10d.fichas_declaradas(entregas, valem=("11b_novo",))[1].keys() == fichas_novas.keys()
    assert list(caminhos) == ["novo"] and fichas_novas["novo/aluno_b2"]["resultado"] == ("11b_novo", {"variante": "aluno_b2"})
    assert fichas_novas["novo/aluno_b2"]["previsoes"]["test"] == [tmp_path / "entregas/11b_novo/previsoes/b2_test.parquet"] and fichas_novas["novo/aluno_b2"]["sementes"] == 1
    fichas = {**registro_de_mentira(), **fichas_novas}
    # o comparativo que quer a etapa nova diz que a entrega dela vale (`valem`); o da 10d, com a lista dele, não a enxerga
    pares = {(c["b"], c["a"]): c["familias"] for c in etapa10d.lista_de_comparacoes(fichas, entregas, valem=None)}
    assert pares[("novo/aluno_b2", "aluno_b2")] == ["entre_caminhos"] and pares[("novo/aluno_b2", "qwen")] == ["aluno_professor"]
    assert etapa10d.chave_no("novo", "aluno_b2", fichas, etapa10d.caminhos_do_registro(entregas, valem=None)) == "novo/aluno_b2"
    assert etapa10d.chave_no("novo", "qwen", fichas, etapa10d.caminhos_do_registro(entregas, valem=None)) == "qwen"
    assert "novo" not in etapa10d.caminhos_do_registro(entregas) and "11b_novo" not in etapa10d.ENTREGAS_QUE_VALEM


def test_regras_gravadas_nunca_sao_sobrescritas(tmp_path, monkeypatch):
    conteudo = {"comparacoes": [{"b": "x", "a": "y"}], "cadeias": {}}
    monkeypatch.setattr(etapa10d, "registro", lambda entregas=None, base=None: {})
    monkeypatch.setattr(etapa10d, "montar_regras", lambda fichas, entregas=None: dict(conteudo))
    with pytest.raises(FileNotFoundError):
        etapa10d.carregar_regras(tmp_path)
    primeira = etapa10d.gravar_regras(tmp_path)
    assert primeira.name == "regras.json" and etapa10d.gravar_regras(tmp_path) == primeira   # o mesmo conteúdo não vira versão nova
    texto = primeira.read_text(encoding="utf-8")
    conteudo["comparacoes"].append({"b": "z", "a": "y"})   # o registro cresceu
    segunda = etapa10d.gravar_regras(tmp_path)
    assert segunda.name == "regras_v2.json" and primeira.read_text(encoding="utf-8") == texto
    regras, arquivo = etapa10d.carregar_regras(tmp_path)
    assert arquivo == segunda and regras["versao"] == 2 and len(regras["comparacoes"]) == 2


def test_dominado_por_e_a_fronteira_da_etapa_9():
    rng = np.random.default_rng(0)
    pontos = pd.DataFrame({"custo": 10 ** rng.uniform(-1, 3, 40), "ganho": rng.uniform(0.2, 0.7, 40)}, index=[f"s{i}" for i in range(40)])
    pontos.loc["sem_custo"] = [np.nan, 0.9]   # sem uma das medidas: fica de fora
    for folga in (0.0, 0.14):
        vencedores = etapa10d.dominado_por(pontos, "custo", "ganho", folga)
        assert "sem_custo" not in vencedores
        assert sorted(c for c, quem in vencedores.items() if not quem) == sorted(etapa9.nao_dominados(pontos, "custo", "ganho", folga))
    tres = pd.DataFrame({"custo": [1.0, 1.05, 30.0], "ganho": [0.50, 0.53, 0.52]}, index=["barato", "vizinho", "caro"])
    assert etapa10d.dominado_por(tres, "custo", "ganho") == {"barato": [], "vizinho": [], "caro": ["vizinho"]}
    assert etapa10d.dominado_por(tres, "custo", "ganho", 0.14)["barato"] == ["vizinho"]   # com a folga, 1,0 e 1,05 são empate de custo, e o vizinho é melhor


def test_fronteiras_so_entre_os_sistemas_com_poucos_rotulos_e_pelo_arquivo_entregue():
    T = pd.DataFrame({
        "rotulos_humanos": [8, 8, 8, 40], "exact_match": [0.50, 0.52, 0.49, 0.60], "latencia_p50_ms": [0.3, 8.0, 0.3, 0.3],
        "bytes": [2e6, 90e6, 5e6, 2e6], "bytes_com_zip": [1.7e6, 60e6, 0.9e6, np.nan], "entregue_compactado": [False, False, True, False],
    }, index=["zero", "xlmr", "soltos", "teto"])
    F = etapa10d.fronteiras(T, folga=0.1).set_index(["custo", "chave"])
    assert "teto" not in F.index.get_level_values("chave")   # usa todos os rótulos: é referência, não candidato
    assert F.loc[("latencia", "soltos"), "dominado_por"] == "zero" and bool(F.loc[("latencia", "zero"), "na_fronteira"]) and bool(F.loc[("latencia", "xlmr"), "na_fronteira"])
    assert F.loc[("tamanho", "soltos"), "valor_do_custo"] == pytest.approx(0.9) and bool(F.loc[("tamanho", "soltos"), "na_fronteira"])   # vai compactado: vale o tamanho com zip
    trocada = etapa10d.fronteiras(T, folga=0.1, latencia=pd.Series({"zero": 0.5})).set_index(["custo", "chave"])
    assert trocada.loc[("latencia", "zero"), "valor_do_custo"] == 0.5 and bool(trocada.loc[("latencia", "soltos"), "na_fronteira"])


def test_preco_por_ponto_so_existe_quando_o_ganho_se_distingue_de_zero():
    preco = etapa10d.preco_por_ponto(88.0, 0.022, 0.011, 0.044)
    assert preco["definido"] and preco["valor"] == pytest.approx(40.0) and preco["ic_min"] == pytest.approx(20.0) and preco["ic_max"] == pytest.approx(80.0)
    assert not etapa10d.preco_por_ponto(88.0, 0.01, -0.001, 0.02)["definido"]
    assert not etapa10d.preco_por_ponto(88.0, -0.02, -0.03, -0.01)["definido"]


def test_quadro_de_artigos_compara_o_que_o_projeto_escreve_com_o_que_a_fonte_devolveu(tmp_path):
    artigos = (
        {"passo": "Destilação", "onde": "os dois", "tecnica": "t", "autor": "Hinton", "quem": "Hinton et al.", "titulo": "Distilling the Knowledge", "arxiv": "1503.02531"},
        {"passo": "Poda", "onde": "os dois", "tecnica": "t", "autor": "Han", "quem": "Han et al.", "titulo": "Deep Compression", "arxiv": "1510.00149"},
        {"passo": "Quantização", "onde": "os dois", "tecnica": "t", "autor": "Jacob", "quem": "Jacob et al.", "titulo": "", "arxiv": "1712.05877"},
        {"passo": "Régua", "onde": "os dois", "tecnica": "t", "autor": "Koehn", "quem": "Koehn", "titulo": "Statistical Significance Tests", "acl": "W04-3250", "ano": 2004},
        {"passo": "Exportação", "onde": "os dois", "tecnica": "t", "autor": None, "quem": "nenhum artigo", "titulo": ""},
    )
    assert list(etapa10d.quadro_de_artigos(tmp_path, artigos, tmp_path)["na_fonte"]) == ["não conferida"] * 4 + ["sem artigo"]   # sem consulta gravada, nada está conferido
    (tmp_path / "artigos").mkdir()
    fonte = {"consultado_em": "2026-11-18T20:00:00", "arxiv": {
        "1503.02531": {"titulo": "Distilling the Knowledge in a Neural Network", "autores": ["Geoffrey Hinton", "Oriol Vinyals", "Jeff Dean"], "publicado": "2015-03-09", "resumo": "um resumo"},
        "1510.00149": {"titulo": "Outro artigo qualquer", "autores": ["Fulano de Tal"], "publicado": "2015-10-01", "resumo": "um resumo"},
    }, "acl": {"W04-3250": {"titulo": "Statistical Significance Tests for Machine Translation Evaluation", "autores": ["Koehn, Philipp"], "publicado": "2004", "resumo": ""}}}
    (tmp_path / "artigos" / "arxiv_1.json").write_text(json.dumps(fonte), encoding="utf-8")
    quadro = etapa10d.quadro_de_artigos(tmp_path, artigos, tmp_path).set_index("identificador")
    assert quadro.loc["1503.02531", "na_fonte"] == "conferida" and quadro.loc["1503.02531", "autores_na_fonte"] == 3 and bool(quadro.loc["1503.02531", "resumo_lido"])
    assert quadro.loc["1510.00149", "na_fonte"] == "não confere"      # a fonte devolveu outro artigo com esse número
    assert quadro.loc["1712.05877", "na_fonte"] == "não conferida"    # a fonte não devolveu
    assert quadro.loc["W04-3250", "na_fonte"] == "conferida" and not quadro.loc["W04-3250", "resumo_lido"] and quadro.loc["W04-3250", "ano"] == 2004
    assert quadro.loc["", "na_fonte"] == "sem artigo"


def test_todo_artigo_do_quadro_da_etapa_9_continua_no_quadro():
    daqui = {a.get("arxiv") or a.get("acl") for a in etapa10d.ARTIGOS}
    assert {a["arxiv"] for a in etapa9.ARTIGOS if a.get("arxiv")} <= daqui and {"W04-3250", "D12-1091"} <= daqui
    assert {a["passo"] for a in etapa10d.ARTIGOS} == set(etapa10d.PASSOS)
    assert set(etapa10d.RESUMOS_LIDOS) <= {a["arxiv"] for a in etapa10d.ARTIGOS if a.get("arxiv")}


# ---- com as entregas de verdade: o registro lê a linha certa de cada etapa?


@com_entregas
def test_registro_tem_os_dois_caminhos_com_as_mesmas_chaves(tmp_path):
    fichas = etapa10d.registro(base=tmp_path)
    zero = [c for c, f in fichas.items() if f["caminho"] == "zero"]
    xlmr = [c for c, f in fichas.items() if f["caminho"] == "xlmr"]
    assert list(fichas)[:21] == list(etapa9.catalogo())   # as linhas da Etapa 9 vêm primeiro, na ordem dela
    assert sorted("xlmr/" + c for c in zero) == sorted(xlmr)
    assert etapa10d.finais(fichas) == {"zero": "aluno_a/int8", "xlmr": "xlmr/aluno_a/int8"}
    for chave, f in fichas.items():
        assert f["ponto"] in etapa10d.PONTOS and set(f["previsoes"]) == set(etapa10d.PARTICOES) and all(len(lista) == f["sementes"] for lista in f["previsoes"].values()), chave
    # o que os caminhos dividem não é de nenhum dos dois, e todo aluno aponta para uma LLM do registro
    assert all(fichas[c]["caminho"] is None for c in ("claude", "qwen", "base_299", "base_11514"))
    # o registro desta etapa só lê as entregas que existiam quando ela foi entregue: os dois professores das Etapas 5a e 5b, e nenhum sistema declarado depois
    assert {f["professor"] for f in fichas.values() if f.get("professor")} == {"claude", "qwen"} and not [c for c, f in fichas.items() if "declarado_em" in f]
    assert fichas["xlmr/aluno_a"]["professor"] == "claude" and fichas["xlmr/aluno_b2/int8"]["professor"] == "qwen" and fichas["aluno_b1"]["professor"] == "qwen"


@com_entregas
def test_cada_ficha_aponta_para_uma_linha_que_existe_em_dev_e_no_teste(tmp_path):
    fichas = etapa10d.registro(base=tmp_path)
    ler = etapa9._Tabelas(ENTREGAS)
    for chave, f in fichas.items():
        for etapa, filtro in [f["resultado"], *f.get("outras_fontes", [])]:
            for particao in etapa10d.PARTICOES:
                etapa9.linha_gravada(ler(etapa), particao=particao, **filtro)   # dá erro se não houver exatamente uma
        if "eficiencia" in f:
            etapa9.linha_gravada(ler(f["eficiencia"][0]), particao="test", **f["eficiencia"][1])
        if "zip" in f:
            etapa, arquivo, filtro, coluna = f["zip"]
            assert etapa9.linha_gravada(ler(etapa, arquivo), **filtro)[coluna] > 0, chave


@com_entregas
def test_as_respostas_de_origem_estao_nas_entregas_e_so_o_que_falta_vai_para_o_cache(tmp_path):
    fichas = etapa10d.registro(base=tmp_path)
    no_cache = {chave for chave, f in fichas.items() for lista in f["previsoes"].values() for c in lista if c.is_relative_to(tmp_path)}
    assert all("gerar" in fichas[c] for c in no_cache)   # tudo o que não está numa entrega tem como ser produzido
    assert no_cache == {"base_299"} | {f"{m}/{p}" for m in ("aluno_a", "aluno_b2", "teto") for p in ("int8", "soltos", "int8_sem_poda")}
    assert not [c for chave, f in fichas.items() if chave not in no_cache for lista in f["previsoes"].values() for c in lista if not c.exists()]
    assert all(c.is_relative_to(ENTREGAS) for c in fichas["aluno_a/int8"]["previsoes"]["test"])   # as do teste, a Etapa 9 entregou


@com_entregas
def test_regras_saem_do_registro_sem_ler_resultado(tmp_path):
    fichas = etapa10d.registro(base=tmp_path)
    regras = etapa10d.montar_regras(fichas)
    # as 43 comparações dos sistemas das Etapas 2 a 10c: o que as etapas seguintes declaram não entra aqui (entra no comparativo da Etapa 11c)
    assert len(regras["comparacoes"]) == 43 and len(fichas) == 44 and not [c for c in fichas.values() if "declarado_em" in c]
    assert all(d.split("/")[1] in etapa10d.ENTREGAS_QUE_VALEM for d in regras["decisoes_lidas"])
    assert list(regras["cadeias"]) == ["zero/A", "zero/B", "zero/teto", "xlmr/A", "xlmr/B", "xlmr/teto"]
    assert regras["metodo"]["intervalo"]["reamostragens"] == 2000 and regras["metodo"]["sementes"] == 3
    assert regras["finais"] == {"zero": "aluno_a/int8", "xlmr": "xlmr/aluno_a/int8"}
    json.dumps(regras)   # tem de caber num arquivo


ENTREGA_DA_10D = ENTREGAS / etapa10d.ETAPA
com_a_entrega_da_10d = pytest.mark.skipif(not (ENTREGA_DA_10D / "calculo.json").exists(), reason="a entrega da Etapa 10d não está em disco")


@com_entregas
@com_a_entrega_da_10d
def test_o_registro_reproduz_o_que_a_10d_entregou_mesmo_com_etapas_novas_em_disco(tmp_path):
    """O risco de versão: etapas posteriores declaram sistemas em `sistemas.json`, e o registro da 10d não pode passar a lê-los."""
    calculo = json.loads((ENTREGA_DA_10D / "calculo.json").read_text(encoding="utf-8"))
    entregue = json.loads((ENTREGA_DA_10D / calculo["regras"]["arquivo"]).read_text(encoding="utf-8"))
    fichas = etapa10d.registro(base=tmp_path)
    regras = json.loads(json.dumps(etapa10d.montar_regras(fichas)))
    assert list(fichas) == entregue["sistemas_no_registro"] and len(fichas) == calculo["sistemas"] == 44
    assert len(regras["comparacoes"]) == calculo["comparacoes"] == 43 and list(regras["cadeias"]) == calculo["cadeias"]
    assert {k: v for k, v in entregue.items() if k not in ("versao", "gravado_em")} == regras   # as regras inteiras, com a impressão digital de cada decisão lida
    assert len(pd.read_csv(ENTREGA_DA_10D / "registro.csv")) == len(fichas) and list(pd.read_csv(ENTREGA_DA_10D / "tabela_final.csv")["chave"]) == list(fichas)
    # com todas as entregas em disco valendo, o registro é outro: é por isso que a lista existe
    declaradas = sorted(ENTREGAS.glob("*/sistemas.json"))
    if declaradas:
        assert len(etapa10d.registro(base=tmp_path, valem=None)) > len(fichas)
