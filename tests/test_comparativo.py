"""Comparativo final: a junção do que as etapas gravaram, as cadeias e as comparações de ponta a ponta.

Os testes montam as entregas das Etapas 1 a 8 em miniatura, com respostas
inventadas, e conferem que a Etapa 9 lê a linha certa de cada arquivo, compara
os sistemas certos na ordem certa e acusa quando um número gravado não bate
com a conta dela. Nada aqui usa GPU, API ou os modelos de verdade.
"""

import json

import numpy as np
import pandas as pd
import pytest

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, ouro, salvar_previsoes
from especialista.compressao import exportacao
from especialista.experimentos import etapa9

N = 100   # sorteios do bootstrap nos testes: o bastante para haver intervalo, pouco para ser rápido
MOLDES = [
    ("alarm_set", "acorda-me às nove", "O O B-time"),
    ("play_music", "toca tony carreira", "O B-artist I-artist"),
    ("weather_query", "vai chover amanhã", "O O B-date"),
    ("play_music", "toca música", "O O"),
]
POUCOS, TODOS = 8, 40
# quanto cada sistema inventado erra: o bastante para as diferenças terem sinal
ERRO = {"base_poucos": 0.6, "base_todos": 0.35, "claude": 0.2, "qwen": 0.4, "teto": 0.1, "piso": 0.6, "aluno_a": 0.25, "aluno_b1": 0.45, "aluno_b2_t8": 0.42}
PODA = {"aluno_a": ("linhas_90", "magnitude_95"), "aluno_b2_t8": ("linhas_90", "gradual_90"), "teto": ("linhas_90", "magnitude_90")}


def particao(quantos: int, primeiro_id: int) -> pd.DataFrame:
    linhas = [MOLDES[i % len(MOLDES)] for i in range(quantos)]
    return pd.DataFrame({"id": range(primeiro_id, primeiro_id + quantos), "intent": [l[0] for l in linhas], "utt": [l[1] for l in linhas],
                         "tokens": [l[1].split() for l in linhas], "tags": [l[2].split() for l in linhas]})


def inventar(gabarito: list[Previsao], erro: float, semente: int) -> list[Previsao]:
    """Respostas de mentira: a certa, com a intenção trocada ou as entidades apagadas numa parte dos pedidos."""
    rng = np.random.default_rng(semente)
    return [Previsao(g.intent if rng.random() > erro else "outra", g.slots if rng.random() > erro else ()) for g in gabarito]


def linha(gabarito, grupo, **campos) -> dict:
    """Uma linha de `resultados.csv`, como `resultados.registrar_grupo` grava, com menos sorteios."""
    medida = qualidade.intervalo_do_grupo(gabarito, grupo, n=N)
    dados = {"particao": "test", "enunciados": len(gabarito), "cpu": "CPU de teste", **campos}
    for metrica in qualidade.METRICAS:
        dados.update({metrica: medida.loc[metrica, "valor"], f"{metrica}_ic_min": medida.loc[metrica, "ic_min"], f"{metrica}_ic_max": medida.loc[metrica, "ic_max"]})
    return dados


def latencia(p50: float) -> dict:
    return {"latencia_p50_ms": p50, "latencia_p95_ms": 1.4 * p50, "latencia_media_ms": 1.1 * p50}


def comparacao(gabarito, a, b, **campos) -> list[dict]:
    """As linhas de um `comparacoes.csv`: B − A, uma por métrica."""
    contra = qualidade.comparar_grupos(gabarito, a, b, n=N)
    return [{**campos, "metrica": m, **contra.loc[m, ["a", "b", "diferenca", "ic_min", "ic_max", "distinguivel"]].to_dict()} for m in qualidade.METRICAS]


def gravar_json(caminho, conteudo) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """As entregas das Etapas 1 a 8 em miniatura, e as respostas que `preparar` produziria."""
    raiz = tmp_path_factory.mktemp("projeto")
    entregas, base = raiz / "entregas", raiz / "data" / "execucoes" / etapa9.ETAPA
    teste = particao(TODOS + 20, 1000)
    gabarito = ouro(teste)
    ids = list(teste["id"])
    grupos: dict[str, list[list[Previsao]]] = {}

    def salvar(chave, arquivos, erro, sementes=(0,), de=None):
        """Inventa as respostas de um sistema (ou estraga um pouco as de `de`), grava e guarda."""
        grupos[chave] = [inventar(de[i] if de else gabarito, erro, 97 * len(grupos) + s) for i, s in enumerate(sementes)]
        for caminho, previsoes in zip(arquivos, grupos[chave], strict=True):
            salvar_previsoes(caminho, ids, previsoes)
        return grupos[chave]

    tres = etapa9.SEMENTES
    # Etapa 1: os dados
    for nome, tabela in (("train", particao(TODOS, 0)), ("dev", particao(10, 500)), ("test", teste)):
        (entregas / "01_dados").mkdir(parents=True, exist_ok=True)
        tabela.to_parquet(entregas / "01_dados" / f"{nome}.parquet", index=False)
    gravar_json(entregas / "01_dados" / "rotulos.json", {"intents": sorted({m[0] for m in MOLDES})})
    gravar_json(entregas / "01_dados" / "manifesto.json", {"enunciados": {"train": TODOS, "dev": 10, "test": len(teste)}})
    # Etapa 2: a linha de base com todos os rótulos
    gravar_json(entregas / "02_avaliacao" / "manifesto.json", {"linha_de_base": {"escolhida": "Linha de base (palavras + caracteres)"}})
    g = salvar("base_11514", [entregas / "02_avaliacao" / "previsoes" / "linha_de_base_test.parquet"], ERRO["base_todos"])
    pd.DataFrame([linha(gabarito, g, sistema="Linha de base (palavras + caracteres)", parametros=900, bytes=3300, **latencia(0.8)),
                  {"sistema": "Linha de base (palavras + caracteres)", "particao": "dev"}]).to_csv(entregas / "02_avaliacao" / "resultados.csv", index=False)
    # Etapa 4b: o teto
    g = salvar("teto", [entregas / "04b_intencao" / "previsoes" / f"semente_{s}_test.parquet" for s in tres], ERRO["teto"], tres)
    pd.DataFrame([linha(gabarito, g, sistema="DIET do zero (rótulos humanos)", parametros=4000, bytes=20000, **latencia(1.2))]).to_csv(entregas / "04b_intencao" / "resultados.csv", index=False)
    # Etapa 5a: o professor fechado e a linha de base com os poucos rótulos (só as métricas: as respostas ela não gravou)
    gravar_json(entregas / "05a_professor_fechado" / "configuracao.json", {"modelo": "fechado-1", "conjunto_de_demonstracao": {"exemplos": POUCOS}, "latencia": {"de_onde": "rede de teste"}})
    particao(POUCOS, 0).to_parquet(entregas / "05a_professor_fechado" / "demonstracoes.parquet", index=False)
    g = salvar("claude", [entregas / "05a_professor_fechado" / "previsoes" / "claude_test.parquet"], ERRO["claude"])
    b = salvar("base_299", [base / "previsoes" / "base_299_test.parquet"], ERRO["base_poucos"])
    pd.DataFrame([
        linha(gabarito, g, sistema="Fechado 1 (10 exemplos)", modelo="fechado-1", custo_por_mil_pedidos_usd=0.11, custo_por_mil_pedidos_em_lote_usd=0.06, **latencia(850.0)),
        linha(gabarito, b, sistema=f"Linha de base com {POUCOS} rótulos", parametros=50),
    ]).to_csv(entregas / "05a_professor_fechado" / "resultados.csv", index=False)
    # Etapa 5b: o professor aberto
    gravar_json(entregas / "05b_professor_aberto" / "escolha.json", {"modelo": "org/aberto-8b", "escolhido": "aberto_8b"})
    g = salvar("qwen", [entregas / "05b_professor_aberto" / "previsoes" / "aberto_8b_test.parquet"], ERRO["qwen"])
    pd.DataFrame([linha(gabarito, g, sistema="Aberto-8B em 4 bits (10 exemplos)", modelo="org/aberto-8b", parametros=8e9, vram_gb=6.1, gpu="Placa de teste", **latencia(1100.0))]).to_csv(
        entregas / "05b_professor_aberto" / "resultados.csv", index=False)
    # Etapa 6: o piso e os alunos
    alunos = {"A": "aluno_a", "B1": "aluno_b1", "B2": "aluno_b2_t8"}
    gravar_json(entregas / "06_destilacao" / "decisao.json", {
        "principal": "A", "seguem": ["A", "B2"], "temperatura_do_b2": 8,
        "alunos": {letra: {"nome": f"Aluno {letra} (de teste)", "variante": variante, "segue": letra != "B1"} for letra, variante in alunos.items()},
    })
    linhas, comparacoes = [], []
    for chave, variante in (("piso", "piso"), ("aluno_a", "aluno_a"), ("aluno_b1", "aluno_b1"), ("aluno_b2", "aluno_b2_t8")):
        g = salvar(chave, [entregas / "06_destilacao" / "previsoes" / f"{variante}_semente_{s}_test.parquet" for s in tres], ERRO[variante], tres)
        linhas.append(linha(gabarito, g, sistema=f"DIET do zero, {variante}", variante=variante, parametros=4000, bytes=20000, **latencia(1.15)))
    for letra, professor, papel in (("A", "claude", "professor fechado"), ("B1", "qwen", "professor aberto"), ("B2", "qwen", "professor aberto")):
        comparacoes += comparacao(gabarito, grupos[professor], grupos[f"aluno_{letra.lower()}"], comparacao=f"test · Aluno {letra} − {papel}")
    pd.DataFrame(linhas).to_csv(entregas / "06_destilacao" / "resultados.csv", index=False)
    pd.DataFrame(comparacoes).to_csv(entregas / "06_destilacao" / "comparacoes.csv", index=False)
    # Etapas 7 e 8: a poda principal e a de pesos soltos de cada modelo, em 32 bits e em INT8
    de_partida = {"aluno_a": "aluno_a", "aluno_b2_t8": "aluno_b2", "teto": "teto"}
    r7, c7, e7, r8, c8, fichas8 = [], [], [], [], [], {}
    for modelo, (principal, solta) in PODA.items():
        chave = de_partida[modelo]
        sem_poda = grupos[chave]
        podado = salvar(f"{chave}/podado", [entregas / "07_poda" / "previsoes" / f"{modelo}__{principal}_semente_{s}_test.parquet" for s in tres], 0.05, tres, de=sem_poda)
        grupos[f"{chave}/onnx"] = podado
        int8 = salvar(f"{chave}/int8", [base / "previsoes" / f"{modelo}__{principal}__dinamica_semente_{s}_test.parquet" for s in tres], 0.02, tres, de=podado)
        soltos = salvar(f"{chave}/soltos", [base / "previsoes" / f"{modelo}__{solta}__dinamica_semente_{s}_test.parquet" for s in tres], 0.08, tres, de=sem_poda)
        r7.append(linha(gabarito, podado, modelo=modelo, variante=principal, momento="depois do reajuste"))
        c7 += comparacao(gabarito, sem_poda, podado, modelo=modelo, variante=principal, comparacao="depois − sem poda", particao="test")
        e7 += [{"modelo": modelo, "variante": "(sem poda)", "bytes_com_zip": 18000}, {"modelo": modelo, "variante": principal, "bytes_com_zip": 7400}]
        r8 += [
            linha(gabarito, sem_poda, modelo=modelo, variante="sem_poda", forma="pytorch", parametros=4000, bytes=20000, **latencia(1.10)),
            linha(gabarito, podado, modelo=modelo, variante=principal, forma="pytorch", parametros=2000, bytes=8000, **latencia(1.11)),
            linha(gabarito, podado, modelo=modelo, variante=principal, forma="fp32", parametros=2000, bytes=7900, **latencia(0.43)),
            linha(gabarito, int8, modelo=modelo, variante=principal, forma="dinamica", parametros=2000, bytes=2100, bytes_compactado=1700, **latencia(0.26)),
            linha(gabarito, soltos, modelo=modelo, variante=solta, forma="dinamica", parametros=4000, bytes=5200, bytes_compactado=900, **latencia(0.27)),
        ]
        c8 += comparacao(gabarito, podado, podado, modelo=modelo, variante=principal, forma="fp32", particao="test")
        c8 += comparacao(gabarito, podado, int8, modelo=modelo, variante=principal, forma="dinamica", particao="test")
        for variante, compactado in ((principal, False), (solta, True)):
            fichas8[f"{modelo}/{variante}"] = {"forma": "dinamica", "nome_da_forma": "INT8, quantização dinâmica", "compactado": compactado,
                                               "pastas_por_semente": {str(s): f"entregas/08_quantizacao/modelos/{modelo}/{variante}/semente_{s}" for s in tres}}
    gravar_json(entregas / "07_poda" / "decisao.json", {"modelos": {
        modelo: {"principal": principal, "secundaria": solta, "seguem": [{"variante": principal, "nome": "Sem 90% das linhas"}, {"variante": solta, "nome": "Magnitude, pesos soltos"}]}
        for modelo, (principal, solta) in PODA.items()}})
    gravar_json(entregas / "08_quantizacao" / "decisao.json", {"final": {"modelo": "aluno_a", "variante": "linhas_90", "forma": "dinamica"}, "modelos": fichas8})
    for pasta, arquivo, linhas in (("07_poda", "resultados.csv", r7), ("07_poda", "comparacoes.csv", c7), ("07_poda", "eficiencia.csv", e7),
                                   ("08_quantizacao", "resultados.csv", r8), ("08_quantizacao", "comparacoes.csv", c8)):
        pd.DataFrame(linhas).to_csv(entregas / pasta / arquivo, index=False)

    with pytest.MonkeyPatch.context() as remendo:
        remendo.setattr(etapa9, "REAMOSTRAS", N)
        fichas = etapa9.catalogo(entregas, base)
        medidas = {"professor_aberto": {"bytes_dos_pesos_baixados": 16_000_000}}
        yield {"raiz": raiz, "entregas": entregas, "base": base, "teste": teste, "gabarito": gabarito, "ids": ids, "grupos": grupos, "fichas": fichas,
               "medidas": medidas, "tabela": etapa9.tabela_final(fichas, medidas, entregas)}


# --------------------------------------------------------------------------
# O catálogo e a tabela final
# --------------------------------------------------------------------------


def test_catalogo_tem_uma_linha_por_sistema_na_ordem_da_tabela(mini):
    chaves = list(mini["fichas"])
    assert chaves[:9] == ["base_299", "base_11514", "claude", "qwen", "teto", "piso", "aluno_a", "aluno_b1", "aluno_b2"]
    for modelo in ("aluno_a", "aluno_b2", "teto"):   # o aluno de cada cenário e o teto: podado, ONNX, INT8 e a poda de pesos soltos
        assert [c for c in chaves if c.startswith(modelo + "/")] == [f"{modelo}/{sufixo}" for sufixo in ("podado", "onnx", "int8", "soltos")]
    assert not any(c.startswith("aluno_b1/") for c in chaves)   # o B1 não foi para a poda
    assert len(chaves) == 21
    assert etapa9.chave_do_final(mini["fichas"]) == "aluno_a/int8"   # o que a decisão da Etapa 8 marcou
    assert mini["fichas"]["aluno_a/int8"]["bloco"] == "Cenário A" and mini["fichas"]["aluno_b2/int8"]["bloco"] == "Cenário B"


def test_o_que_nao_estava_gravado_esta_marcado_para_gerar(mini):
    a_gerar = {chave for chave, ficha in mini["fichas"].items() if "gerar" in ficha}
    assert a_gerar == {"base_299"} | {f"{m}/{s}" for m in ("aluno_a", "aluno_b2", "teto") for s in ("int8", "soltos")}
    assert mini["fichas"]["aluno_a/int8"]["gerar"]["de"][0] == mini["raiz"] / "entregas/08_quantizacao/modelos/aluno_a/linhas_90/semente_0"
    assert mini["fichas"]["teto/onnx"]["previsoes"] == mini["fichas"]["teto/podado"]["previsoes"]   # o ONNX de 32 bits responde igual ao PyTorch


def test_tabela_final_le_a_linha_certa_de_cada_etapa(mini):
    tabela, entregas = mini["tabela"], mini["entregas"]
    assert list(tabela.index) == list(mini["fichas"])
    da_etapa_6 = pd.read_csv(entregas / "06_destilacao" / "resultados.csv").set_index("variante")
    da_etapa_8 = pd.read_csv(entregas / "08_quantizacao" / "resultados.csv").set_index(["modelo", "variante", "forma"])
    assert tabela.loc["aluno_b2", "exact_match"] == pytest.approx(da_etapa_6.loc["aluno_b2_t8", "exact_match"])
    assert tabela.loc["aluno_b2", "exact_match_ic_min"] == pytest.approx(da_etapa_6.loc["aluno_b2_t8", "exact_match_ic_min"])
    # quem foi para a poda tem a latência da Etapa 8 (máquina quieta); quem não foi, a da etapa que o treinou
    assert tabela.loc["aluno_a", "latencia_p50_ms"] == 1.10 and tabela.loc["aluno_a", "latencia_medida_na"] == "Etapa 8"
    assert tabela.loc["aluno_b1", "latencia_p50_ms"] == 1.15 and tabela.loc["aluno_b1", "latencia_medida_na"] == "Etapa 6"
    assert tabela.loc["aluno_a/int8", "bytes"] == 2100 and tabela.loc["aluno_a/int8", "bytes_com_zip"] == 1700
    assert tabela.loc["aluno_a/int8", "intent_acc"] == pytest.approx(da_etapa_8.loc[("aluno_a", "linhas_90", "dinamica"), "intent_acc"])
    assert tabela.loc["aluno_a/podado", "bytes_com_zip"] == 7400 and tabela.loc["aluno_a", "bytes_com_zip"] == 18000   # o zip de 32 bits vem da Etapa 7
    assert tabela.loc["teto/soltos", "entregue_compactado"] and not tabela.loc["teto/int8", "entregue_compactado"]
    assert tabela.loc["aluno_a", "rotulos_humanos"] == POUCOS and tabela.loc["teto/int8", "rotulos_humanos"] == TODOS


def test_tabela_final_diz_a_condicao_de_cada_latencia_e_nao_inventa_o_que_falta(mini):
    tabela = mini["tabela"]
    assert tabela.loc["claude", "condicao_da_latencia"] == etapa9.API and "rede de teste" in tabela.loc["claude", "onde_foi_medida"]
    assert tabela.loc["qwen", "condicao_da_latencia"] == etapa9.GPU and "Placa de teste" in tabela.loc["qwen", "onde_foi_medida"]
    assert tabela.loc["aluno_a/int8", "condicao_da_latencia"] == etapa9.CPU and "ONNX Runtime" in tabela.loc["aluno_a/int8", "onde_foi_medida"]
    # o professor fechado: parâmetros e tamanho não divulgados; custo em dólar só ele tem
    assert not tabela.loc["claude", "divulgado"] and pd.isna(tabela.loc["claude", "parametros"]) and pd.isna(tabela.loc["claude", "bytes"])
    assert tabela.loc["claude", "custo_por_mil_usd"] == 0.11 and tabela["custo_por_mil_usd"].notna().sum() == 1
    # a linha de base com poucos rótulos: a etapa de origem não mediu latência nem tamanho
    assert pd.isna(tabela.loc["base_299", "latencia_p50_ms"]) and tabela.loc["base_299", "onde_foi_medida"] == "não medida"
    assert pd.isna(tabela.loc["base_299", "consultas_por_segundo"])
    assert tabela.loc["qwen", "bytes"] == 16_000_000   # o tamanho do professor aberto vem do que `preparar` mediu
    assert tabela.loc["aluno_a/int8", "consultas_por_segundo"] == pytest.approx(1000 / (1.1 * 0.26))
    assert tabela.loc["claude", "sistema"] == "Fechado 1 (professor fechado)" and tabela.loc["qwen", "curto"] == "Aberto-8B"


def test_linha_gravada_exige_exatamente_uma_linha():
    tabela = pd.DataFrame({"modelo": ["a", "a", "b"], "forma": ["x", "y", "x"], "valor": [1, 2, 3]})
    assert etapa9.linha_gravada(tabela, modelo="a", forma="y")["valor"] == 2
    with pytest.raises(LookupError, match="há 2"):
        etapa9.linha_gravada(tabela, modelo="a")
    with pytest.raises(LookupError, match="há 0"):
        etapa9.linha_gravada(tabela, modelo="c")


def test_respostas_confere_arquivo_e_ordem_dos_pedidos(mini, tmp_path):
    ficha = mini["fichas"]["aluno_a"]
    grupo = etapa9.respostas(ficha, mini["ids"])
    assert len(grupo) == 3 and grupo == mini["grupos"]["aluno_a"]
    with pytest.raises(ValueError, match="ordem"):
        etapa9.respostas(ficha, mini["ids"][::-1])
    with pytest.raises(FileNotFoundError, match="etapa9 preparar"):
        etapa9.respostas({"previsoes": [tmp_path / "nao_existe.parquet"]})


# --------------------------------------------------------------------------
# A conferência com as etapas anteriores
# --------------------------------------------------------------------------


def test_conferencia_passa_quando_bate_e_acusa_numero_trocado(mini):
    confere = etapa9.conferir_qualidade(mini["fichas"], mini["tabela"], mini["gabarito"], mini["grupos"], mini["entregas"])
    assert confere["confere"].all() and len(confere) == 21 + 3   # os três modelos podados estão gravados por duas etapas
    assert confere.loc[confere["chave"] == "aluno_a/int8", "respostas_geradas_agora"].all()
    trocada = mini["tabela"].copy()
    trocada.loc["aluno_a/int8", "exact_match"] += 0.01   # o RESUMO de uma etapa diria um número, a conta daqui outro
    acusou = etapa9.conferir_qualidade(mini["fichas"], trocada, mini["gabarito"], mini["grupos"], mini["entregas"]).set_index("chave")
    assert not acusou.loc["aluno_a/int8", "confere"] and acusou.loc["aluno_a/int8", "maior_diferenca"] == pytest.approx(0.01)
    assert acusou.drop(index="aluno_a/int8")["confere"].all()


def test_conferencia_dos_degraus_contra_as_comparacoes_gravadas(mini):
    degraus = etapa9.cadeias(mini["tabela"], mini["gabarito"], mini["grupos"])
    confere = etapa9.conferir_degraus(mini["fichas"], degraus, mini["entregas"])
    assert confere["confere"].all() and len(confere) == 4 + 4 + 3   # todo degrau com anterior tem uma comparação gravada
    assert set(confere["gravado_em"]) == {f"entregas/{e}/comparacoes.csv" for e in ("06_destilacao", "07_poda", "08_quantizacao")}
    degraus.loc[degraus["chave"] == "aluno_a/podado", "slot_f1_dif_ic_max"] += 0.02
    acusou = etapa9.conferir_degraus(mini["fichas"], degraus, mini["entregas"]).set_index("chave")
    assert not acusou.loc["aluno_a/podado", "confere"] and acusou.drop(index="aluno_a/podado")["confere"].all()


# --------------------------------------------------------------------------
# As cadeias e as comparações de ponta a ponta
# --------------------------------------------------------------------------


def test_cadeia_traz_cada_degrau_e_a_diferenca_para_o_anterior(mini):
    tabela, gabarito, grupos = mini["tabela"], mini["gabarito"], mini["grupos"]
    cadeia = etapa9.cadeia("A", etapa9.CADEIAS["A"]["degraus"], tabela, gabarito, grupos)
    assert list(cadeia["chave"]) == ["claude", "aluno_a", "aluno_a/podado", "aluno_a/onnx", "aluno_a/int8"]
    assert list(cadeia["degrau"]) == ["Professor (LLM)", "Aluno destilado", "Podado e reajustado", "ONNX, 32 bits", "INT8"]
    assert pd.isna(cadeia.loc[0, "anterior"]) and list(cadeia["anterior"].iloc[1:]) == ["claude", "aluno_a", "aluno_a/podado", "aluno_a/onnx"]
    assert pd.isna(cadeia.loc[0, "exact_match_diferenca"])   # o primeiro degrau não tem anterior
    esperado = qualidade.comparar_grupos(gabarito, grupos["claude"], grupos["aluno_a"], n=N)   # aluno − professor
    assert cadeia.loc[1, "exact_match_diferenca"] == pytest.approx(esperado.loc["exact_match", "diferenca"])
    assert cadeia.loc[1, "exact_match_dif_ic_min"] == pytest.approx(esperado.loc["exact_match", "ic_min"])
    assert cadeia.loc[1, "exact_match_diferenca"] == pytest.approx(cadeia.loc[1, "exact_match"] - cadeia.loc[0, "exact_match"])
    # a exportação não muda nenhuma resposta: diferença zero, intervalo zero
    assert cadeia.loc[3, [f"{m}_{c}" for m in qualidade.METRICAS for c in ("diferenca", "dif_ic_min", "dif_ic_max")]].astype(float).abs().max() == 0
    # qualidade, tamanho e latência de cada degrau vêm da tabela final
    assert list(cadeia["bytes"].iloc[1:]) == [20000, 8000, 7900, 2100] and list(cadeia["latencia_p50_ms"]) == [850.0, 1.10, 1.11, 0.43, 0.26]


def test_cadeias_tem_os_dois_cenarios_e_o_teto(mini):
    todas = etapa9.cadeias(mini["tabela"], mini["gabarito"], mini["grupos"])
    assert list(todas.groupby("cadeia", sort=False).size().items()) == [("A", 5), ("B", 5), ("teto", 4)]
    assert todas.loc[todas["cadeia"] == "teto", "chave"].iloc[0] == "teto" and todas.loc[todas["cadeia"] == "B", "chave"].iloc[0] == "qwen"
    assert todas.loc[todas["ordem"] == 0, "anterior"].isna().all()


def test_ponta_a_ponta_e_a_lista_fixada_e_a_diferenca_e_b_menos_a(mini):
    tabela, gabarito, grupos = mini["tabela"], mini["gabarito"], mini["grupos"]
    comparacoes = etapa9.ponta_a_ponta(tabela, gabarito, grupos)
    assert list(dict.fromkeys(comparacoes["letra"])) == ["a", "b", "c", "d", "e1", "e2", "f", "g1", "g2", "g3"]
    assert len(comparacoes) == 10 * len(qualidade.METRICAS) and not comparacoes["exploratoria"].any()
    assert np.allclose(comparacoes["diferenca"], comparacoes["valor_b"] - comparacoes["valor_a"])
    a = comparacoes[(comparacoes["letra"] == "a") & (comparacoes["metrica"] == "exact_match")].iloc[0]
    assert (a["chave_b"], a["chave_a"]) == ("aluno_a/int8", "claude")   # modelo final − professor
    esperado = qualidade.comparar_grupos(gabarito, grupos["claude"], grupos["aluno_a/int8"], n=N).loc["exact_match"]
    assert a["diferenca"] == pytest.approx(esperado["diferenca"]) and a["ic_max"] == pytest.approx(esperado["ic_max"])
    assert a["valor_b"] == pytest.approx(tabela.loc["aluno_a/int8", "exact_match"]) and a["valor_a"] == pytest.approx(tabela.loc["claude", "exact_match"])
    # o modelo final ganha da linha de base e do piso com os mesmos rótulos: os sistemas inventados foram feitos assim
    assert comparacoes[(comparacoes["letra"].isin(["g1", "g3"])) & (comparacoes["metrica"] == "exact_match")]["ic_min"].gt(0).all()


def test_toda_chave_das_listas_fixadas_existe_no_catalogo(mini):
    chaves = set(mini["fichas"])
    assert {b for _, b, _, _ in etapa9.PONTA_A_PONTA} | {a for _, _, a, _ in etapa9.PONTA_A_PONTA} <= chaves
    assert {chave for cadeia in etapa9.CADEIAS.values() for _, chave in cadeia["degraus"]} <= chaves
    assert {tipo for cadeia in etapa9.CADEIAS.values() for tipo, _ in cadeia["degraus"]} <= set(etapa9.DEGRAUS)


# --------------------------------------------------------------------------
# A fronteira, a poda em dev e no teste, e as contas pequenas
# --------------------------------------------------------------------------


def test_nao_dominados_e_a_fronteira_de_pareto():
    pontos = pd.DataFrame({"custo": [1.0, 2.0, 3.0, 4.0, 2.5, np.nan], "ganho": [50.0, 60.0, 55.0, 70.0, 60.0, 99.0]}, index=list("abcdef"))
    assert etapa9.nao_dominados(pontos, "custo", "ganho") == ["a", "b", "d"]   # c e e perdem para b; f não tem custo
    iguais = pd.DataFrame({"custo": [1.0, 1.0], "ganho": [5.0, 5.0]}, index=["x", "y"])
    assert etapa9.nao_dominados(iguais, "custo", "ganho") == ["x", "y"]        # empate nas duas medidas: nenhum vence o outro


def test_folga_do_custo_trata_medidas_proximas_como_empate():
    pontos = pd.DataFrame({"custo": [0.260, 0.261, 0.43], "ganho": [46.0, 51.0, 51.0]}, index=["rapido_e_pior", "final", "mais_lento"])
    assert etapa9.nao_dominados(pontos, "custo", "ganho") == ["rapido_e_pior", "final"]           # sem folga, um milésimo de ms põe um sistema pior na fronteira
    assert etapa9.nao_dominados(pontos, "custo", "ganho", folga_do_custo=0.1) == ["final"]         # com a folga da medida, não


def test_poda_em_dev_e_no_teste(tmp_path):
    entregas = tmp_path / "entregas"
    gravar_json(entregas / "07_poda" / "decisao.json", {"modelos": {"aluno_a": {"principal": "linhas_90", "secundaria": "magnitude_95"}, "teto": {"principal": "linhas_90", "secundaria": None}}})
    def gravada(modelo, variante, particao_, diferenca, minimo, maximo):
        return {"modelo": modelo, "variante": variante, "comparacao": "depois − sem poda", "particao": particao_, "metrica": "exact_match", "diferenca": diferenca, "ic_min": minimo, "ic_max": maximo}
    pd.DataFrame([
        gravada("aluno_a", "linhas_90", "dev", -0.004, -0.014, 0.006), gravada("aluno_a", "linhas_90", "test", 0.000, -0.008, 0.009),
        gravada("aluno_a", "magnitude_95", "dev", -0.008, -0.016, 0.000), gravada("aluno_a", "magnitude_95", "test", -0.009, -0.016, -0.002),
        gravada("teto", "linhas_90", "dev", -0.008, -0.020, 0.002), gravada("teto", "linhas_90", "test", -0.012, -0.021, -0.003),
        {**gravada("teto", "linhas_90", "test", 0.5, 0.4, 0.6), "comparacao": "antes − sem poda"},
    ]).to_csv(entregas / "07_poda" / "comparacoes.csv", index=False)
    tabela = etapa9.poda_em_dev_e_no_teste(entregas).set_index(["modelo", "variante"])
    assert len(tabela) == 3 and tabela["passa_em_dev"].all()   # a regra aceitou as três em dev
    assert list(tabela["abaixo_no_teste"]) == [False, True, True]   # no teste, duas ficam distinguivelmente abaixo
    assert tabela.loc[("teto", "linhas_90"), "test_diferenca"] == -0.012


def test_contas_pequenas():
    assert etapa9.consultas_por_segundo(0.25) == 4000 and np.isnan(etapa9.consultas_por_segundo(float("nan")))
    assert etapa9.milhar(11514) == "11.514" and etapa9.milhar(299) == "299"
    assert etapa9.etapa_curta("08_quantizacao") == "Etapa 8" and etapa9.etapa_curta("05a_professor_fechado") == "Etapa 5a"
    assert etapa9.ano_do_arxiv("2204.08582") == 2022 and etapa9.ano_do_arxiv("1503.02531") == 2015
    with pytest.raises(ValueError):
        etapa9.ano_do_arxiv("EMNLP 2004")


# --------------------------------------------------------------------------
# O quadro de artigos
# --------------------------------------------------------------------------


def test_artigo_fora_do_plano_e_dos_resumos_e_acusado(tmp_path):
    (tmp_path / "entregas" / "06_destilacao").mkdir(parents=True)
    (tmp_path / "entregas" / etapa9.ETAPA).mkdir()
    (tmp_path / "PLANO.md").write_text("Hinton et al. ([1503.02531](https://arxiv.org/abs/1503.02531))", encoding="utf-8")
    (tmp_path / "entregas" / "06_destilacao" / "RESUMO.md").write_text("He & Garner (arXiv 2305.13512)", encoding="utf-8")
    (tmp_path / "entregas" / etapa9.ETAPA / "RESUMO.md").write_text("arXiv 9999.00001", encoding="utf-8")   # o resumo desta etapa não vale como fonte de si mesmo
    artigos = ({"quem": "Hinton et al.", "arxiv": "1503.02531"}, {"quem": "He & Garner", "arxiv": "2305.13512"}, {"quem": "Novo", "arxiv": "9999.00001"}, {"quem": "Koehn", "veiculo": "EMNLP 2004"})
    assert etapa9.artigos_fora_das_fontes(artigos, tmp_path) == ["Novo (9999.00001)", "Koehn (EMNLP 2004)"]


def test_quadro_de_artigos_do_projeto():
    quadro = etapa9.quadro_de_artigos()
    assert len(quadro) == len(etapa9.ARTIGOS) and quadro["endereco"].str.startswith("https://").all()
    assert quadro.loc[quadro["quem"] == "Hinton et al.", "ano"].item() == 2015 and quadro.loc[quadro["quem"] == "Koehn", "ano"].item() == 2004
    if not (etapa9.ENTREGAS.parent / "PLANO.md").exists():
        pytest.skip("sem o PLANO.md do projeto")
    # a regra da etapa: o que não está no PLANO.md nem num RESUMO tem de ter sido conferido, e está marcado
    de_fora = set(etapa9.artigos_fora_das_fontes())
    marcados = {f"{a['quem']} ({a.get('arxiv') or a.get('veiculo')})" for a in etapa9.ARTIGOS if a.get("conferido")}
    assert de_fora == marcados


# --------------------------------------------------------------------------
# O que não estava gravado: `preparar`
# --------------------------------------------------------------------------


def test_preparar_gera_as_respostas_que_faltam_e_nao_repete(mini, tmp_path, monkeypatch):
    carregados = []

    class DeMentira:
        def prever(self, textos):
            return [Previsao("play_music") for _ in textos]

    monkeypatch.setattr(exportacao.SistemaOnnx, "carregar", classmethod(lambda cls, pasta: carregados.append(pasta) or DeMentira()))
    cache = tmp_path / "hub"
    (cache / "models--org--aberto-8b" / "blobs").mkdir(parents=True)
    (cache / "models--org--aberto-8b" / "blobs" / "pesos").write_bytes(b"x" * 1234)
    base = tmp_path / "execucoes"
    assert len(etapa9.faltam(etapa9.catalogo(mini["entregas"], base), base)) == 1 + 6 * 3 + 1   # linha de base, 6 sistemas × 3 sementes, medidas.json

    etapa9.preparar(mini["entregas"], base, cache=cache)
    fichas = etapa9.catalogo(mini["entregas"], base)
    assert etapa9.faltam(fichas, base) == [] and len(carregados) == 6 * 3
    assert carregados[0] == mini["raiz"] / "entregas/08_quantizacao/modelos/aluno_a/linhas_90/semente_0"
    assert etapa9.respostas(fichas["teto/soltos"], mini["ids"])[2] == [Previsao("play_music")] * len(mini["ids"])
    de_base = etapa9.respostas(fichas["base_299"], mini["ids"])[0]   # a linha de base é treinada de verdade, com os poucos exemplos
    assert {p.intent for p in de_base} <= {m[0] for m in MOLDES} and qualidade.avaliar(mini["gabarito"], de_base)["intent_acc"] > 0.5
    medidas = etapa9.carregar_medidas(base)
    assert medidas["professor_aberto"]["bytes_dos_pesos_baixados"] == 1234 and medidas["professor_aberto"]["arquivos"] == 1
    assert set(medidas["respostas_geradas"]) == {c for c, f in fichas.items() if "gerar" in f}

    etapa9.preparar(mini["entregas"], base, cache=tmp_path / "vazio")   # de novo, com o cache já limpo: nada é refeito, e a medida gravada continua valendo
    assert len(carregados) == 6 * 3
    assert etapa9.carregar_medidas(base)["professor_aberto"]["bytes_dos_pesos_baixados"] == 1234
    (base / "medidas.json").unlink()
    etapa9.preparar(mini["entregas"], base, cache=tmp_path / "vazio")   # sem medida gravada e sem cache: fica dito que não foi medido
    sem_cache = etapa9.carregar_medidas(base)["professor_aberto"]
    assert sem_cache["bytes_dos_pesos_baixados"] is None and "não medido" in sem_cache["de_onde"]


def test_bytes_no_cache(tmp_path):
    assert etapa9.bytes_no_cache("org/modelo", tmp_path) is None
    blobs = tmp_path / "models--org--modelo" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_bytes(b"1" * 10)
    (blobs / "b").write_bytes(b"2" * 5)
    assert etapa9.bytes_no_cache("org/modelo", tmp_path) == (15, 2)
