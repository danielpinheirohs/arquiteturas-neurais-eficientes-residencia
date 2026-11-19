"""Etapa 11b: a troca de professor que não vaza, a pasta própria, as regras gravadas antes e as contas do repasse.

Os treinos usam as entregas em miniatura da Etapa 6 (`cenario`, de
`test_destilacao.py`), o encoder minúsculo de `test_encoder.py` e um
professor novo de brinquedo, que acerta o que o antigo troca.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from especialista import caminhos
from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, EXECUCOES
from especialista.experimentos import etapa6, etapa10a, etapa11a, etapa11b, execucao
from especialista.modelos import encoder
from especialista.professores.rotulagem import rotular

from test_destilacao import cenario, professor_que_troca  # noqa: F401  (cenario é uma fixture)
from test_encoder import sistema_minusculo

NOVO, ANTIGO = etapa11b.PROFESSOR_NOVO, etapa11b.PROFESSOR_ANTIGO
RAPIDA = {"taxa": 1e-2, "epocas": 12, "lote": 4}   # a receita "da Etapa 10a" em miniatura
com_entregas = pytest.mark.skipif(not (ENTREGAS / etapa11a.ETAPA / "escolha.json").exists() or not (ENTREGAS / "10a_encoder" / "decisao.json").exists(),
                                  reason="as entregas das Etapas 6, 10a e 11a não estão no disco")


def professor_que_acerta(d: pd.DataFrame) -> list[Previsao]:
    return [Previsao.de_tags(intent, list(tags)) for intent, tags in zip(d["intent"], d["tags"])]


def com_erros(gabarito: list[Previsao], erros: int, inicio: int = 0) -> list[Previsao]:
    """Uma cópia do gabarito com a intenção errada em `erros` pedidos, a partir de `inicio`."""
    copia = list(gabarito)
    for i in range(inicio, inicio + erros):
        copia[i] = Previsao("outra", copia[i].slots)
    return copia


def arquivos_de(pasta) -> dict:
    """Uma fotografia de tudo o que está numa pasta: caminho, tamanho, data e conteúdo."""
    return {str(a.relative_to(pasta)): (a.stat().st_size, a.stat().st_mtime_ns, a.read_bytes()) for a in sorted(pasta.rglob("*")) if a.is_file()}


@pytest.fixture
def etapa(cenario, monkeypatch):  # noqa: F811
    """A Etapa 11b em miniatura: o cenário da Etapa 6, o encoder minúsculo e a pasta do professor novo, no formato da Etapa 11a."""
    pasta, treino, dev, intents = cenario["pasta"], cenario["treino"], cenario["dev"], cenario["intents"]
    entregas = pasta / "entregas"
    monkeypatch.setattr(etapa11b, "ENTREGAS", entregas)
    monkeypatch.setattr(etapa11b, "RAIZ", pasta)
    minusculo = lambda config: encoder.Config(**{**config.como_dict(), "modelo_id": "minusculo", "revisao": ""})
    monkeypatch.setattr(etapa10a, "novo_sistema", lambda rotulos, config, semente, dispositivo: sistema_minusculo(semente, minusculo(config), rotulos))

    # o professor novo: acerta a intenção que o antigo troca, e não deixa pedido sem resposta
    novo = entregas / etapa11a.como_professor(NOVO)["etapa"]
    (novo / "previsoes").mkdir(parents=True)
    rotular(treino, professor_que_acerta(treino), intents).to_parquet(novo / "rotulos_treino.parquet", index=False)
    logits = np.full((len(treino), len(intents)), -5.0, dtype=np.float32)
    logits[np.arange(len(treino)), [intents.index(i) for i in treino["intent"]]] = -0.01
    np.save(novo / "logits_treino.npy", logits)
    (novo / "intencoes.json").write_text(json.dumps(intents), encoding="utf-8")
    for particao in ("dev", "test"):
        salvar_previsoes(novo / "previsoes" / f"{NOVO}_{particao}.parquet", dev["id"], professor_que_acerta(dev))
    # os professores antigos também respondem ao teste (no cenário, o teste é o próprio dev)
    for professor in ("fechado", "aberto"):
        salvar_previsoes(etapa6.pasta_do_professor(professor) / "previsoes" / f"{etapa6.PROFESSORES[professor]['previsoes']}_test.parquet",
                         dev["id"], professor_que_troca(dev))

    # o que as etapas de origem entregaram: a receita do XLM-R, a temperatura de cada caminho e a escolha da Etapa 11a
    config = encoder.Config().como_dict()                                   # o encoder minúsculo entra por `etapa10a.novo_sistema`, como nos testes da 10a
    (entregas / "10a_encoder").mkdir()
    (entregas / "10a_encoder" / "decisao.json").write_text(json.dumps({"temperatura_do_b2": 2, "alunos": {"B1": {"config": config, "receita": RAPIDA}}}), encoding="utf-8")
    (entregas / "06_destilacao").mkdir()
    (entregas / "06_destilacao" / "decisao.json").write_text(json.dumps({"temperatura_do_b2": 8}), encoding="utf-8")
    metricas = {"exact_match": 0.5}
    (entregas / etapa11a.ETAPA / "escolha.json").write_text(json.dumps({
        "escolhido": NOVO, "candidatos": {NOVO: {"dev": metricas}, ANTIGO: {"dev": metricas}},
        "disputas": {etapa11b.FORA_DO_ESCOPO: {"contra": ANTIGO, "diferenca": 0.0, "ic_min": -0.01, "ic_max": 0.01},
                     NOVO: {"contra": ANTIGO, "diferenca": 0.06, "ic_min": 0.04, "ic_max": 0.08}},
    }), encoding="utf-8")
    return cenario


def entregar_os_alunos_antigos(etapa: dict) -> None:
    """As respostas que as Etapas 4b, 6 e 10a entregaram, de brinquedo: os alunos antigos erram mais que os professores."""
    entregas, dev = etapa["pasta"] / "entregas", etapa["dev"]
    gabarito = ouro(dev)
    grava = lambda arquivo, erros: salvar_previsoes(arquivo, dev["id"], com_erros(gabarito, erros))
    for particao in ("dev", "test"):
        for s in etapa11b.SEMENTES:
            grava(entregas / "04b_intencao" / "previsoes" / f"semente_{s}_{particao}.parquet", 0)
            for v, erros in (("piso", 6), ("aluno_a", 2), ("aluno_b1", 4), ("aluno_b1_sem_suavizacao", 4), *((f"aluno_b2_t{T}", 3) for T in etapa11b.TEMPERATURAS)):
                grava(entregas / "06_destilacao" / "previsoes" / f"{v}_semente_{s}_{particao}.parquet", erros)
            for v, erros in (("teto", 0), ("piso", 5), ("aluno_a", 1), ("aluno_b1", 4), *((f"aluno_b2_t{T}", 4) for T in etapa11b.TEMPERATURAS)):
                grava(entregas / "10a_encoder" / "previsoes" / f"{v}_semente_{s}_{particao}.parquet", erros)
    # as comparações que as etapas de origem gravaram: aqui, só o cabeçalho de cada formato (nenhuma linha para conferir)
    (entregas / "06_destilacao" / "comparacoes.csv").write_text("comparacao,metrica,a,b,diferenca,ic_min,ic_max,distinguivel\n", encoding="utf-8")
    (entregas / "10a_encoder" / "comparacoes.csv").write_text("particao,comparacao,metrica,a,b,diferenca,ic_min,ic_max,distinguivel\n", encoding="utf-8")
    (entregas / etapa11a.ETAPA / "comparacoes.csv").write_text("particao,b,a,comparacao,metrica,valor_a,valor_b,diferenca,ic_min,ic_max,distinguivel\n", encoding="utf-8")


# --------------------------------------------------------------------------
# O desenho: só o professor muda
# --------------------------------------------------------------------------


def test_a_grade_e_os_nomes_dizem_o_caminho_o_aluno_e_o_professor():
    assert etapa11b.TEMPERATURAS == etapa6.TEMPERATURAS == etapa10a.TEMPERATURAS
    assert etapa11b.ALUNOS["zero"] == tuple(v for v in etapa6.VARIANTES if etapa6.VARIANTES[v]["professor"] == "aberto")     # todos os alunos do professor aberto
    assert etapa11b.ALUNOS["xlmr"] == tuple(v for v in etapa10a.VARIANTES if v.startswith("aluno_b"))
    assert len(etapa11b.grade()) == (6 + 5) * 3 == 33
    assert etapa11b.variante("zero", "aluno_b1") == f"zero__aluno_b1__{NOVO}"
    assert all(NOVO in etapa11b.variante(c, a) and etapa11b.variante(c, a).startswith(f"{c}__") for c, a, _ in etapa11b.grade())
    with pytest.raises(KeyError):
        etapa11b.variante("xlmr", "aluno_b1_sem_suavizacao")              # no XLM-R o B1 já não tem suavização: não há controle
    # nenhuma execução desta etapa cai na pasta das Etapas 6 ou 10a
    pastas = {etapa11b.pasta_da_execucao(c, a, s) for c, a, s in etapa11b.grade()}
    assert len(pastas) == 33 and all(p.parts[-3] == etapa11b.ETAPA for p in pastas)
    assert etapa11b.CAMINHOS["zero"]["processos"] == 7 and etapa11b.CAMINHOS["xlmr"]["processos"] == 1


def test_com_professor_troca_so_dentro_do_bloco():
    antes = dict(etapa6.PROFESSORES["aberto"])
    with etapa11b.com_professor(NOVO) as professor:
        assert etapa6.PROFESSORES["aberto"] == professor == etapa11a.como_professor(NOVO)
        assert etapa6.pasta_do_professor("aberto").parts[-3:] == (etapa11a.ETAPA, "professores", NOVO)
    assert etapa6.PROFESSORES["aberto"] == antes
    with pytest.raises(RuntimeError):
        with etapa11b.com_professor(NOVO):
            raise RuntimeError("o treino quebrou no meio")
    assert etapa6.PROFESSORES["aberto"] == antes == {"nome": "Qwen3-8B", "etapa": "05b_professor_aberto", "previsoes": "qwen3_8b"}


@com_entregas
def test_as_receitas_sao_as_dos_alunos_antigos():
    da_6 = json.loads((ENTREGAS / "06_destilacao" / "decisao.json").read_text(encoding="utf-8"))
    da_10a = json.loads((ENTREGAS / "10a_encoder" / "decisao.json").read_text(encoding="utf-8"))
    completa = lambda caminho, d: (etapa11b._classes(caminho)[0](**d["config"]).como_dict(), etapa11b._classes(caminho)[1](**d["receita"]).como_dict())

    for sigla, aluno in (("B1", "aluno_b1"), ("B2", etapa6.variante_do_b2(da_6["temperatura_do_b2"]))):
        assert completa("zero", etapa11b.definicao("zero", aluno)) == (da_6["alunos"][sigla]["config"], da_6["alunos"][sigla]["receita"])
    for sigla, aluno in (("B1", "aluno_b1"), ("B2", etapa6.variante_do_b2(da_10a["temperatura_do_b2"]))):
        assert completa("xlmr", etapa11b.definicao("xlmr", aluno)) == (da_10a["alunos"][sigla]["config"], da_10a["alunos"][sigla]["receita"])
    for aluno in etapa11b.ALUNOS["zero"]:                                 # a definição é a da Etapa 6; só o nome diz o professor novo
        nova, antiga = etapa11b.definicao("zero", aluno), etapa6.definicao(aluno)
        assert (nova["config"], nova["receita"]) == (antiga["config"], antiga["receita"]) and "Qwen3-14B" in nova["nome"]
    for T in etapa11b.TEMPERATURAS:                                       # do B1 ao B2, no XLM-R, só a temperatura
        b1, b2 = etapa11b.definicao("xlmr", "aluno_b1"), etapa11b.definicao("xlmr", etapa6.variante_do_b2(T))
        assert {k for k in b1["config"] if b1["config"][k] != b2["config"][k]} == {"temperatura"} and b2["config"]["temperatura"] == float(T)
        assert b1["receita"] == b2["receita"]
    # contra o que as execuções antigas gravaram, quando o cache delas está no disco
    assert False not in etapa11b.receitas_contra_as_antigas().values()
    assert etapa11b.temperatura_antiga("zero") == da_6["temperatura_do_b2"] and etapa11b.temperatura_antiga("xlmr") == da_10a["temperatura_do_b2"]
    # o professor desta rodada tomou o lugar do titular na escada da Etapa 11a; o 8B sem os 4 bits, não, e fica sem alunos
    disputas = json.loads((ENTREGAS / etapa11a.ETAPA / "escolha.json").read_text(encoding="utf-8"))["disputas"]
    assert disputas[NOVO]["ic_min"] > 0 and not disputas[etapa11b.FORA_DO_ESCOPO]["ic_min"] > 0
    with pytest.raises(ValueError, match="escada"):
        etapa11b.montar_regras(etapa11b.FORA_DO_ESCOPO)


# --------------------------------------------------------------------------
# As listas fechadas e as duas contas novas
# --------------------------------------------------------------------------


def test_as_listas_cobrem_as_comparacoes_combinadas():
    lista = etapa11b.lista_de_comparacoes()
    pares = {(linha["bloco"], linha["b"], linha["a"]) for linha in lista}
    assert len({linha["id"] for linha in lista}) == len(lista) == len(pares) and {linha["bloco"] for linha in lista} == set(etapa11b.BLOCOS)
    for c in etapa11b.CAMINHOS:
        for s in ("B1", "B2"):
            assert ("novo_menos_antigo", f"{c}:{s}:novo", f"{c}:{s}:antigo") in pares          # aluno novo − aluno antigo
            for p in ("novo", "antigo"):
                assert ("aluno_menos_professor", f"{c}:{s}:{p}", f"prof:{p}") in pares         # aluno − o seu professor
                assert ("b_menos_a", f"{c}:{s}:{p}", f"{c}:A") in pares                        # aluno B − aluno A do mesmo caminho
        for p in ("novo", "antigo"):                                                           # os logits, com cada professor
            assert ("logits", f"{c}:B2:{p}", f"{c}:{etapa11b.SO_LOGITS[c]}:{p}") in pares
    assert ("logits", "zero:B2:novo", "zero:B1:novo") in pares and ("professores", "prof:novo", "prof:antigo") in pares
    # um aluno só é comparado com o professor que o ensinou
    assert all(linha["a"].split(":")[1] == linha["b"].split(":")[2] for linha in lista if linha["bloco"] == "aluno_menos_professor")

    diferencas, repasses = etapa11b.lista_de_diferencas(), etapa11b.lista_de_repasses()
    assert [d["id"] for d in diferencas if d["bloco"] == "logits"] == ["logits|zero|B2-B1s", "logits|zero|B2-B1", "logits|xlmr|B2-B1"]
    assert all(d["a_novo"].endswith("novo") and d["b_novo"].endswith("novo") and d["a_antigo"].endswith("antigo") and d["b_antigo"].endswith("antigo") for d in diferencas)
    assert [(r["caminho"], r["aluno"]) for r in repasses] == [("zero", "B1"), ("zero", "B1s"), ("zero", "B2"), ("xlmr", "B1"), ("xlmr", "B2")]
    assert all((r["professor_antigo"], r["professor_novo"]) == ("prof:antigo", "prof:novo") for r in repasses)
    assert all(texto in etapa11b.REGRA_DO_REPASSE for texto in ("÷", "reamostrados juntos", "exact match no teste"))


def test_repasse_e_o_ganho_do_aluno_dividido_pelo_do_professor():
    gabarito = [Previsao("alarm_set")] * 400
    professor_antigo, professor_novo = [com_erros(gabarito, 200)], [com_erros(gabarito, 100)]            # 50% e 75%: o professor ganha 25 pontos
    aluno_antigo = [com_erros(gabarito, 220), com_erros(gabarito, 230), com_erros(gabarito, 210)]       # 45% na média
    aluno_novo = [com_erros(gabarito, 170), com_erros(gabarito, 180), com_erros(gabarito, 160)]         # 57,5%: o aluno ganha 12,5
    tabela = etapa11b.repasse(gabarito, aluno_antigo, aluno_novo, professor_antigo, professor_novo, n=300)
    linha = tabela.loc["intent_acc"]
    assert (linha["ganho_do_aluno"], linha["ganho_do_professor"], linha["repasse"]) == pytest.approx((0.125, 0.25, 0.5))
    assert linha["ic_min"] < 0.5 < linha["ic_max"] and linha["professor_ic_min"] > 0 and linha["legivel"]
    assert linha["aluno_ic_min"] < 0.125 < linha["aluno_ic_max"]

    # o aluno que ganha exatamente o que o professor ganhou, nos mesmos pedidos: repasse 1, sem incerteza
    igual = etapa11b.repasse(gabarito, professor_antigo, professor_novo, professor_antigo, professor_novo, n=50).loc["exact_match"]
    assert igual[["repasse", "ic_min", "ic_max"]].tolist() == pytest.approx([1.0, 1.0, 1.0])
    # o aluno que não muda: repasse 0
    assert etapa11b.repasse(gabarito, aluno_antigo, aluno_antigo, professor_antigo, professor_novo, n=50).loc["intent_acc", "repasse"] == 0.0
    # o professor que quase não muda: a razão não se lê
    quase = etapa11b.repasse(gabarito, aluno_antigo, aluno_novo, professor_antigo, [com_erros(gabarito, 198)], n=300).loc["intent_acc"]
    assert not quase["legivel"] and quase["professor_ic_min"] <= 0 < quase["ganho_do_professor"]


def test_diferenca_de_diferencas_pareia_os_quatro_sistemas():
    gabarito = [Previsao("alarm_set")] * 400
    a_novo, b_novo = [com_erros(gabarito, 120)] * 3, [com_erros(gabarito, 80)] * 3          # B − A = +10 pontos com o professor novo
    a_antigo, b_antigo = [com_erros(gabarito, 160)] * 3, [com_erros(gabarito, 144)] * 3     # e +4 com o antigo
    tabela = etapa11b.diferenca_de_diferencas(gabarito, a_novo, b_novo, a_antigo, b_antigo, n=300)
    linha = tabela.loc["intent_acc"]
    assert (linha["com_o_novo"], linha["com_o_antigo"], linha["diferenca"]) == pytest.approx((0.10, 0.04, 0.06))
    assert 0 < linha["ic_min"] < 0.06 < linha["ic_max"] and linha["distinguivel"]
    # cada parcela é a diferença pareada de `qualidade.comparar_grupos`
    assert linha["com_o_novo"] == pytest.approx(qualidade.comparar_grupos(gabarito, a_novo, b_novo, n=50).loc["intent_acc", "diferenca"])
    # a mesma diferença dos dois lados: zero, sem incerteza
    nada = etapa11b.diferenca_de_diferencas(gabarito, a_novo, b_novo, a_novo, b_novo, n=50).loc["exact_match"]
    assert nada[["diferenca", "ic_min", "ic_max"]].tolist() == [0.0, 0.0, 0.0] and not nada["distinguivel"]


# --------------------------------------------------------------------------
# As regras vêm antes, e o treino não toca no que já existe
# --------------------------------------------------------------------------


def test_sem_regras_gravadas_nada_treina(etapa, monkeypatch):
    with pytest.raises(FileNotFoundError):
        etapa11b.executar("zero", "aluno_b1", 0, "cpu")
    monkeypatch.setattr(etapa11b, "_rodar", lambda *a, **k: pytest.fail("a rodada não pode treinar sem as regras"))
    with pytest.raises(FileNotFoundError):
        etapa11b.rodada()
    assert not etapa11b.pasta_da_etapa().exists()

    destino = etapa11b.gravar_regras()
    assert destino == etapa11b.pasta_da_etapa() / NOVO / "regras.json"       # as regras moram na pasta do professor
    regras = json.loads(destino.read_text(encoding="utf-8"))
    assert regras["professor_novo"]["versao"] == NOVO and regras["professor_antigo"]["versao"] == ANTIGO and regras["execucoes"] == 33
    assert regras["professor_novo"]["na_escada"]["contra"] == ANTIGO and regras["professor_novo"]["na_escada"]["ic_min"] > 0
    assert regras["temperaturas"] == [1, 2, 4, 8] and regras["sementes"] == [0, 1, 2] and regras["escopo"]["sem_alunos_para"] == etapa11b.FORA_DO_ESCOPO
    assert regras["comparacoes"] == etapa11b.lista_de_comparacoes() and regras["repasses"] == etapa11b.lista_de_repasses()
    assert set(regras["variantes"]) == {etapa11b.variante(c, a) for c, a, _ in etapa11b.grade()}
    # os arquivos do professor novo ficam registrados com a impressão digital
    assert any(caminho.endswith("rotulos_treino.parquet") and NOVO in caminho for caminho in regras["professor_novo"]["arquivos"])
    # gravar de novo não muda a hora: a data é a prova de que as regras vieram antes
    antes = destino.read_bytes()
    assert etapa11b.gravar_regras().read_bytes() == antes


def test_as_regras_recusam_quem_nao_venceu_a_escada_e_receita_diferente(etapa, monkeypatch):
    with pytest.raises(ValueError, match="escada"):                         # disputou e não tomou o lugar do titular
        etapa11b.montar_regras(etapa11b.FORA_DO_ESCOPO)
    with pytest.raises(ValueError, match="escada"):                         # nem disputou
        etapa11b.gravar_regras("qwen3_14b__bf16__colab_a100")
    assert not etapa11b.pasta_da_etapa().exists()
    monkeypatch.setattr(etapa11b, "receitas_contra_as_antigas", lambda professor: {"zero__aluno_b1": False})
    with pytest.raises(ValueError, match="receita"):
        etapa11b.montar_regras()


def test_outro_professor_entra_depois_sem_tocar_nos_alunos_do_primeiro(etapa, monkeypatch):
    """Uma segunda rodada, com outro professor da Etapa 11a: pastas, regras e variantes próprias."""
    outro = "qwen3_32b__fp16__colab_g4"
    monkeypatch.setitem(etapa11a.MODELOS, "qwen3_32b", {"repositorio": "Qwen/Qwen3-32B", "nome": "Qwen3-32B", "commit": None})
    entregas, dev = etapa["pasta"] / "entregas", etapa["dev"]
    do_novo, do_outro = entregas / etapa11a.como_professor(NOVO)["etapa"], entregas / etapa11a.como_professor(outro)["etapa"]
    (do_outro / "previsoes").mkdir(parents=True)
    for nome in ("rotulos_treino.parquet", "logits_treino.npy", "intencoes.json"):
        (do_outro / nome).write_bytes((do_novo / nome).read_bytes())
    for particao in ("dev", "test"):
        salvar_previsoes(do_outro / "previsoes" / f"{outro}_{particao}.parquet", dev["id"], professor_que_acerta(dev))
    escolha = json.loads((entregas / etapa11a.ETAPA / "escolha.json").read_text(encoding="utf-8"))
    escolha["disputas"][outro] = {"contra": NOVO, "diferenca": 0.03, "ic_min": 0.01, "ic_max": 0.05}
    escolha["candidatos"][outro] = {"dev": {"exact_match": 0.6}}
    (entregas / etapa11a.ETAPA / "escolha.json").write_text(json.dumps({**escolha, "escolhido": outro}), encoding="utf-8")

    # a primeira rodada: as regras e um aluno do primeiro professor
    etapa11b.gravar_regras()
    etapa11b.executar("zero", "aluno_b1", 0, "cpu")
    do_primeiro = {str(a): (a.stat().st_mtime_ns, a.read_bytes()) for a in sorted(etapa11b.pasta_da_etapa().rglob("*")) if a.is_file()}

    # o segundo professor não treina com as regras do primeiro: precisa das suas, gravadas antes
    with pytest.raises(FileNotFoundError):
        etapa11b.executar("zero", "aluno_b1", 0, "cpu", professor=outro)
    regras = json.loads(etapa11b.gravar_regras(outro).read_text(encoding="utf-8"))
    assert regras["professor_novo"]["versao"] == outro and regras["professor_novo"]["na_escada"]["contra"] == NOVO
    assert set(regras["variantes"]) == {etapa11b.variante(c, a, outro) for c, a, _ in etapa11b.grade()}
    assert regras["comparacoes"] == etapa11b.lista_de_comparacoes()         # as mesmas comparações, contra os alunos antigos

    for caminho, aluno in (("zero", "aluno_b1"), ("xlmr", "aluno_b2_t4")):
        pasta = etapa11b.executar(caminho, aluno, 0, "cpu", professor=outro)
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        assert pasta.parts[-2:] == (f"{caminho}__{aluno}__{outro}", "semente_0") and meta["variante"] == f"{caminho}__{aluno}__{outro}"
        assert meta["origem"]["professor"] == outro and f"professores/{outro}" in meta["origem"]["treino"] and "Qwen3-32B" in meta["nome"]
    assert len(etapa11b.faltam(None, outro)) == 31 and len(etapa11b.faltam()) == 32 and outro in etapa11b.situacao(outro)
    assert etapa11b.pasta_da_rodada(outro) != etapa11b.pasta_da_rodada() and etapa11b.exigir_regras()["professor_novo"]["versao"] == NOVO
    # nada do primeiro professor mudou
    assert {nome: (a, b) for nome, (a, b) in do_primeiro.items()} == {nome: (Path(nome).stat().st_mtime_ns, Path(nome).read_bytes()) for nome in do_primeiro}
    # regras de um professor na pasta de outro são recusadas
    (etapa11b.pasta_da_rodada(outro) / "regras.json").write_bytes((etapa11b.pasta_da_rodada() / "regras.json").read_bytes())
    with pytest.raises(ValueError, match="outro professor"):
        etapa11b.exigir_regras(outro)


def test_o_aluno_novo_aprende_do_professor_novo_e_o_antigo_fica_como_estava(etapa):
    dev = etapa["dev"]
    etapa6.executar("aluno_b1", 0, "cpu")                                   # o aluno antigo, na pasta da Etapa 6
    da_etapa_6 = execucao.EXECUCOES / etapa6.ETAPA
    antes = arquivos_de(da_etapa_6)

    etapa11b.gravar_regras()
    pasta = etapa11b.executar("zero", "aluno_b1", 0, "cpu")
    assert pasta == execucao.EXECUCOES / etapa11b.ETAPA / f"zero__aluno_b1__{NOVO}" / "semente_0"
    novo, antigo = etapa11b.carregar("zero", "aluno_b1", 0), etapa6.carregar("aluno_b1", 0)
    meta = novo["meta"]
    assert meta["etapa"] == etapa11b.ETAPA and meta["variante"] == f"zero__aluno_b1__{NOVO}" and "Qwen3-14B" in meta["nome"]
    assert meta["origem"]["professor"] == NOVO and all(f"professores/{NOVO}" in meta["origem"][campo] for campo in ("treino", "dev"))
    assert "05b_professor_aberto" in antigo["meta"]["origem"]["treino"]
    assert (meta["config"], meta["receita"]) == (antigo["meta"]["config"], antigo["meta"]["receita"])   # a mesma receita: só o professor mudou
    assert meta["exemplos_de_treino"] == len(etapa["treino"]) == antigo["meta"]["exemplos_de_treino"] + 1   # o novo não deixou pedido sem resposta

    # cada aluno aprendeu do seu professor: o novo acerta a intenção, o antigo repete a troca
    assert qualidade.avaliar(ouro(dev), novo["previsoes_dev"])["intent_acc"] > 0.8
    assert qualidade.avaliar(ouro(dev), antigo["previsoes_dev"])["intent_acc"] < 0.5
    assert meta["dev"] == pytest.approx(qualidade.avaliar(professor_que_acerta(dev), novo["previsoes_dev"]))   # a parada é pelo dev do professor novo

    # nada da Etapa 6 mudou, e o professor dela voltou ao lugar
    assert arquivos_de(da_etapa_6) == antes
    assert etapa6.PROFESSORES["aberto"]["etapa"] == "05b_professor_aberto"
    assert "05b_professor_aberto" in etapa6.dados_da_variante("aluno_b1")["origem"]["treino"]
    # o que já rodou não roda de novo
    gravado = arquivos_de(pasta)
    assert etapa11b.executar("zero", "aluno_b1", 0, "cpu") == pasta and arquivos_de(pasta) == gravado
    assert ("zero", "aluno_b1", 0) not in etapa11b.faltam() and len(etapa11b.faltam()) == 32 and len(etapa11b.faltam("xlmr")) == 15


def test_o_xlmr_grava_na_pasta_da_etapa_com_os_campos_da_10a(etapa):
    etapa10a._gravar_json(etapa10a.pasta_da_etapa() / "receita.json", {"vencedora": "busca", "receita": RAPIDA, "regra": ""})
    antigo = json.loads((etapa10a.executar("aluno_b2_t4", 0, "cpu") / "meta.json").read_text(encoding="utf-8"))
    da_etapa_10a = arquivos_de(etapa10a.pasta_da_etapa())

    etapa11b.gravar_regras()
    pasta = etapa11b.executar("xlmr", "aluno_b2_t4", 0, "cpu")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    assert pasta.parts[-3:] == (etapa11b.ETAPA, f"xlmr__aluno_b2_t4__{NOVO}", "semente_0")
    assert set(meta) == set(antigo) and (meta["config"], meta["receita"]) == (antigo["config"], antigo["receita"])
    assert (meta["config"]["temperatura"], meta["config"]["peso_rotulo"]) == (4.0, 0.0)
    assert f"professores/{NOVO}/logits_treino.npy" in meta["origem"]["logits"] and "05b_professor_aberto" in antigo["origem"]["logits"]
    assert "vocabulario" not in meta["origem"] and meta["passos_de_treino"] == antigo["passos_de_treino"]
    assert {a.name for a in pasta.iterdir()} == {"historico.csv", "meta.json", "modelo", "previsoes_dev.parquet"}
    assert arquivos_de(etapa10a.pasta_da_etapa()) == da_etapa_10a
    # o modelo gravado é relido pelo carregador do caminho e repete as respostas
    relido = etapa11b.carregar_sistema("xlmr", "aluno_b2_t4", 0).prever(list(etapa["dev"]["utt"]))
    assert relido == etapa11b.carregar("xlmr", "aluno_b2_t4", 0)["previsoes_dev"]


def test_a_receita_curta_so_vale_em_pasta_descartavel(etapa, monkeypatch, tmp_path):
    etapa11b.gravar_regras()
    with pytest.raises(ValueError, match="descartável"):                    # dentro do projeto, nunca
        etapa11b.usar_pasta_descartavel(etapa11b.RAIZ / "data" / "execucoes" / "qualquer")
    with monkeypatch.context() as m:                                        # como se o cache fosse o de verdade
        m.setattr(caminhos, "EXECUCOES", execucao.EXECUCOES)
        with pytest.raises(ValueError, match="descartável"):
            etapa11b.executar("zero", "aluno_b1", 0, "cpu", receita_curta={"epocas": 2})
    meta = json.loads((etapa11b.executar("zero", "aluno_b1", 0, "cpu", receita_curta={"epocas": 2, "avaliar_a_cada": 1}) / "meta.json").read_text(encoding="utf-8"))
    assert meta["receita"]["epocas"] == 2 and meta["nome"].endswith("[fumaça]")
    assert EXECUCOES == caminhos.EXECUCOES and not etapa11b.no_cache_de_verdade()


# --------------------------------------------------------------------------
# A rodada inteira, em miniatura
# --------------------------------------------------------------------------


def test_a_rodada_treina_tudo_um_caminho_de_cada_vez_e_as_contas_seguem_as_regras(etapa, monkeypatch, capsys):
    entregar_os_alunos_antigos(etapa)
    etapa11b.gravar_regras()
    ordem = []

    def roda(caminho, aluno, semente, professor, fumaca=None):
        ordem.append(caminho)
        etapa11b.executar(caminho, aluno, semente, "cpu", professor=professor)
        return 0

    monkeypatch.setattr(etapa11b, "_rodar", roda)
    monkeypatch.setitem(etapa11b.CAMINHOS["zero"], "processos", 1)          # no teste, um de cada vez também na rede do zero
    assert etapa11b.rodada() is True
    assert ordem == ["zero"] * 18 + ["xlmr"] * 15 and etapa11b.faltam() == []
    saida = capsys.readouterr().out.strip().splitlines()
    assert f"FIM: 33 de 33 execuções de {NOVO} prontas" in saida[-1]
    registro = json.loads(sorted((etapa11b.pasta_da_rodada() / "rodadas").glob("rodada_*.json"))[-1].read_text(encoding="utf-8"))
    assert registro["professor"] == NOVO and registro["prontas"] == 33 and registro["erros"] == [] and registro["codigo_igual_do_inicio_ao_fim"] and len(registro["segundos_de_treino"]) == 33
    assert all(NOVO in json.loads((etapa11b.pasta_da_execucao(c, a, s) / "meta.json").read_text(encoding="utf-8"))["origem"]["treino"] for c, a, s in etapa11b.grade())

    # a temperatura: a regra da Etapa 6, em dev, e o teste só para quem a regra do teste diz
    with pytest.raises(FileNotFoundError):
        etapa11b.grupos("dev")
    for caminho in etapa11b.CAMINHOS:
        decidido = etapa11b.decidir_temperatura(caminho, "cpu")
        em_dev = {T: etapa11b.previsoes(caminho, etapa6.variante_do_b2(T), "dev") for T in etapa11b.TEMPERATURAS}
        T = decidido["temperatura_do_b2"]
        assert T == etapa6.escolher_temperatura(ouro(etapa["dev"]), em_dev)[0] == etapa11b.temperatura_nova(caminho)
        esperadas = {a for a in etapa11b.ALUNOS[caminho] if "b2" not in a} | {etapa6.variante_do_b2(T), etapa6.variante_do_b2(etapa11b.temperatura_antiga(caminho))}
        com_teste = {a for a in etapa11b.ALUNOS[caminho] if (etapa11b.pasta_da_execucao(caminho, a, 0) / "previsoes_test.parquet").exists()}
        assert com_teste == esperadas == set(decidido["abertas_no_teste"])

    g = etapa11b.grupos("test")
    usadas = {linha[k] for linha in etapa11b.lista_de_comparacoes() if linha["bloco"] != "mesma_temperatura" for k in ("a", "b")}
    assert usadas <= set(g) and all(len(g[chave]) == (1 if chave.startswith("prof:") else 3) for chave in g)
    assert g["prof:novo"][0] == professor_que_acerta(etapa["dev"]) and g["prof:antigo"][0] == professor_que_troca(etapa["dev"])
    assert ("zero:B2:novo@T_antiga" in g) == (etapa11b.temperatura_nova("zero") != 8)

    destino = etapa11b.calcular(n=40)
    assert destino == etapa11b.pasta_da_rodada() / "calculo" and (etapa11b.pasta_da_rodada() / "temperatura_zero.json").exists()
    tabelas = {nome: pd.read_csv(destino / f"{nome}.csv") for nome in ("qualidade", "comparacoes", "diferencas_de_diferencas", "repasses")}
    assert all(set(t["particao"]) == {"dev", "test"} for t in tabelas.values())
    comparacoes = tabelas["comparacoes"].query("particao == 'test' and metrica == 'intent_acc'").set_index("id")
    assert set(comparacoes.index) >= {linha["id"] for linha in etapa11b.lista_de_comparacoes() if linha["bloco"] != "mesma_temperatura"}
    professores = comparacoes.loc["professores|prof:novo|prof:antigo"]
    assert professores["diferenca"] == pytest.approx(qualidade.comparar(ouro(etapa["dev"]), g["prof:antigo"][0], g["prof:novo"][0], n=50).loc["intent_acc", "diferenca"])
    repasses = tabelas["repasses"].query("particao == 'test' and metrica == 'intent_acc'").set_index("id")
    assert repasses.loc["repasse|zero|B1", "ganho_do_professor"] == pytest.approx(professores["diferenca"])
    assert len(tabelas["diferencas_de_diferencas"]["id"].unique()) == len(etapa11b.lista_de_diferencas())

    # o que é só relato: a grade em dev, a fração do caminho, a concordância, quem representa o cenário B e as linhas de resultados
    extras = {nome: pd.read_csv(destino / f"{nome}.csv") for nome in ("temperaturas", "fracao_do_caminho", "concordancia_com_o_professor", "resultados", "treinos", "conferencia")}
    grade_em_dev = extras["temperaturas"]
    assert len(grade_em_dev) == 2 * 2 * 4 and grade_em_dev.groupby(["caminho", "professor"])["escolhida"].sum().eq(1).all()
    assert set(grade_em_dev[(grade_em_dev["professor"] == "antigo") & grade_em_dev["escolhida"]]["temperatura"]) == {8, 2}
    assert len(extras["treinos"]) == 33 and extras["treinos"]["treino"].str.contains(NOVO).all()
    em_dev, no_teste = (extras["resultados"][extras["resultados"]["particao"] == p] for p in ("dev", "test"))
    assert len(em_dev) == 11 and set(no_teste["variante"]) == {etapa11b.variante(c, a) for c in etapa11b.CAMINHOS for a in json.loads((etapa11b.pasta_da_rodada() / f"temperatura_{c}.json").read_text(encoding="utf-8"))["abertas_no_teste"]}
    assert (extras["resultados"]["parametros"] > 0).all() and (extras["resultados"]["bytes"] > 0).all()
    representa = json.loads((destino / "quem_representa.json").read_text(encoding="utf-8"))
    assert set(representa) == {"zero", "xlmr"} and all(r["representa"] == ("B2" if r["b2_passa"] else "B1") for r in representa.values())
    concordancia = extras["concordancia_com_o_professor"].query("particao == 'test'").set_index("aluno")
    # o aluno antigo de brinquedo erra a intenção nos 4 primeiros pedidos; o professor antigo troca alarme e música: os dois só coincidem nos 2 pedidos de tempo
    assert concordancia.loc["zero:B1:antigo", "mesma_intencao"] == pytest.approx(2 / 8)
    casos = ["os dois acertam", "só o professor acerta", "só o aluno acerta", "erram com a mesma resposta", "erram com respostas diferentes"]
    assert concordancia[casos].sum(axis=1).tolist() == pytest.approx([1.0] * len(concordancia))

    # a conferência: sem linha na origem, nada fecha; com a linha gravada (e com o sinal trocado onde a origem é A − B), fecha
    assert len(extras["conferencia"]) == len(etapa11b.conferencias()) * 2 * 4 and not extras["conferencia"]["fecha"].any()
    feitas = tabelas["comparacoes"]
    como_a_10a = feitas[feitas["id"] == "logits|xlmr:B2:antigo|xlmr:B1:antigo"].assign(comparacao="Aluno B2 − aluno B1")
    como_a_10a.to_csv(etapa["pasta"] / "entregas" / "10a_encoder" / "comparacoes.csv", index=False)
    ao_contrario = feitas[feitas["id"] == "b_menos_a|zero:B1:antigo|zero:A"]
    como_a_6 = ao_contrario.assign(comparacao=ao_contrario["particao"] + " · Aluno A − aluno B1", diferenca=-ao_contrario["diferenca"], ic_min=-ao_contrario["ic_max"], ic_max=-ao_contrario["ic_min"])
    como_a_6.drop(columns="particao").to_csv(etapa["pasta"] / "entregas" / "06_destilacao" / "comparacoes.csv", index=False)
    conferida = etapa11b.conferir_com_as_origens(feitas).set_index("id")
    assert conferida.loc["logits|xlmr:B2:antigo|xlmr:B1:antigo", "fecha"].all() and conferida.loc["b_menos_a|zero:B1:antigo|zero:A", "fecha"].all()
    assert not conferida.loc["professores|prof:novo|prof:antigo", "fecha"].any()
    como_a_10a.assign(diferenca=como_a_10a["diferenca"] + 0.01).to_csv(etapa["pasta"] / "entregas" / "10a_encoder" / "comparacoes.csv", index=False)
    assert not etapa11b.conferir_com_as_origens(feitas).set_index("id").loc["logits|xlmr:B2:antigo|xlmr:B1:antigo", "fecha"].any()   # um número que não fecha aparece

    # a entrega: a decisão, os sistemas declarados para o comparativo e os modelos por link físico
    from especialista.experimentos import etapa9, etapa10d

    entrega = etapa11b.entregar()
    decisao, declarados = (json.loads((entrega / nome).read_text(encoding="utf-8")) for nome in ("decisao.json", "sistemas.json"))
    assert decisao["temperatura_do_b2"] == {c: etapa11b.temperatura_nova(c) for c in etapa11b.CAMINHOS} and decisao["temperatura_com_o_professor_antigo"] == {"zero": 8, "xlmr": 2}
    assert set(decisao["alunos"]["zero"]) == {"B1", "B2", "B1s"} and set(decisao["alunos"]["xlmr"]) == {"B1", "B2"} and decisao["alunos"]["zero"]["B1s"]["modelos"] is None
    for c in etapa11b.CAMINHOS:
        for sigla in etapa11b.ENTREGUES:
            aluno = decisao["alunos"][c][sigla]
            assert NOVO in aluno["variante"] and NOVO in aluno["sinal"]["treino"]
            for pasta in aluno["pastas_por_semente"].values():
                assert (etapa["pasta"] / pasta / "config.json").stat().st_nlink > 1          # link físico para o cache
            assert etapa11b.carregar_sistema(c, aluno["aluno"], 0).prever(["toca música"]) == type(etapa11b.carregar_sistema(c, aluno["aluno"], 0)).carregar(etapa["pasta"] / aluno["pastas_por_semente"]["0"]).prever(["toca música"])
    assert set(declarados["sistemas"]) == {f"{prefixo}aluno_{s}@{NOVO}" for prefixo in ("", "xlmr/") for s in ("b1", "b2")}
    _, fichas, _ = etapa10d.fichas_declaradas(etapa["pasta"] / "entregas")
    for chave, ficha in fichas.items():
        assert ficha["ponto"] == "destilado" and ficha["professor"] == NOVO and ficha["modelo"] not in etapa10d.NOME_DO_MODELO     # não toma o lugar do aluno antigo nas marcas
        assert ficha["caminho"] == ("xlmr" if chave.startswith("xlmr/") else "zero") and ficha["sementes"] == 3
        assert all(arquivo.exists() for lista in ficha["previsoes"].values() for arquivo in lista)
        for particao in ("dev", "test"):                                                    # a ficha aponta para uma linha, e uma só, de resultados.csv
            etapa9.linha_gravada(pd.read_csv(entrega / "resultados.csv"), particao=particao, **ficha["resultado"][1])
    assert {a.name for a in (entrega / "previsoes").iterdir() if "_dev" in a.name} == {f"{etapa11b.variante(c, a)}_semente_{s}_dev.parquet" for c, a, s in etapa11b.grade()}

    # se o código mudar a lista depois das regras gravadas, as contas não rodam
    monkeypatch.setattr(etapa11b, "lista_de_repasses", lambda: [])
    with pytest.raises(ValueError, match="regras.json"):
        etapa11b.calcular(n=40)


def test_uma_execucao_com_erro_nao_para_as_outras_e_a_ultima_linha_avisa(etapa, monkeypatch, capsys):
    etapa11b.gravar_regras()
    chamadas = []
    monkeypatch.setattr(etapa11b, "_rodar", lambda caminho, aluno, semente, professor, fumaca=None: chamadas.append((caminho, aluno, semente)) or 1)
    assert etapa11b.rodada() is False
    assert len(chamadas) == 33                                              # todas foram tentadas
    ultima = capsys.readouterr().out.strip().splitlines()[-1]
    assert "FIM COM ERROS: 0 de 33" in ultima and "zero/aluno_b1/semente_0" in ultima and "ATENÇÃO" not in ultima


def test_a_rodada_guarda_o_codigo_do_comeco_e_avisa_se_ele_mudar_no_meio(etapa, monkeypatch, capsys):
    pacote = etapa["pasta"] / "src" / "especialista"
    pacote.mkdir(parents=True)
    (pacote / "etapa_de_outro.py").write_text("VERSAO = 1\n", encoding="utf-8")
    etapa11b.gravar_regras()

    def roda(caminho, aluno, semente, professor, fumaca=None):             # outra etapa mexe no pacote enquanto a rodada anda
        (pacote / "etapa_de_outro.py").write_text("VERSAO = 2\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(etapa11b, "_rodar", roda)
    etapa11b.rodada()
    ultima = capsys.readouterr().out.strip().splitlines()[-1]
    assert "ATENÇÃO: código alterado durante a rodada em src/especialista/etapa_de_outro.py" in ultima
    registro = json.loads(sorted((etapa11b.pasta_da_rodada() / "rodadas").glob("rodada_*.json"))[-1].read_text(encoding="utf-8"))
    assert registro["codigo_alterado_durante_a_rodada"] == ["src/especialista/etapa_de_outro.py"] and not registro["codigo_igual_do_inicio_ao_fim"]
    assert (Path(registro["copia_do_codigo"]) / "src" / "especialista" / "etapa_de_outro.py").read_text(encoding="utf-8") == "VERSAO = 1\n"


# --------------------------------------------------------------------------
# O cache de verdade, quando existe
# --------------------------------------------------------------------------


@pytest.mark.skipif(not (EXECUCOES / etapa11b.ETAPA / NOVO / "regras.json").exists(), reason="as regras da Etapa 11b ainda não foram gravadas")
def test_as_regras_gravadas_vieram_antes_dos_treinos_e_sao_as_do_codigo():
    pasta = EXECUCOES / etapa11b.ETAPA
    arquivo = pasta / NOVO / "regras.json"
    regras = json.loads(arquivo.read_text(encoding="utf-8"))
    assert regras["comparacoes"] == etapa11b.lista_de_comparacoes() and regras["diferencas_de_diferencas"] == etapa11b.lista_de_diferencas()
    assert regras["repasses"] == etapa11b.lista_de_repasses() and regras["regra_do_repasse"] == etapa11b.REGRA_DO_REPASSE
    assert regras["regra_da_temperatura"] == etapa11b.REGRA_DA_TEMPERATURA and regras["temperaturas"] == list(etapa11b.TEMPERATURAS)
    assert regras["professor_novo"]["versao"] == NOVO and regras["execucoes"] == len(etapa11b.grade())
    assert False not in regras["receitas_iguais_as_antigas"]["resultado"].values()
    for meta in pasta.glob(f"*__{NOVO}/semente_*/meta.json"):               # nenhum treino terminou antes de as regras existirem
        assert arquivo.stat().st_mtime < meta.stat().st_mtime
        gravado = json.loads(meta.read_text(encoding="utf-8"))
        assert gravado["origem"]["professor"] == NOVO and gravado["variante"] in regras["variantes"]
        esperado = regras["variantes"][gravado["variante"]]
        Config, Receita = etapa11b._classes(esperado["caminho"])
        assert (gravado["config"], gravado["receita"]) == (Config(**esperado["config"]).como_dict(), Receita(**esperado["receita"]).como_dict())


@pytest.mark.skipif(not (EXECUCOES / "06_destilacao").exists() or not (EXECUCOES / "10a_encoder").exists(), reason="o cache das Etapas 6 e 10a não está no disco")
def test_os_alunos_antigos_continuam_sendo_do_professor_antigo():
    for etapa_antiga in ("06_destilacao", "10a_encoder"):
        metas = sorted((EXECUCOES / etapa_antiga).glob("aluno_b*/semente_*/meta.json"))
        assert metas
        for meta in metas:
            origem = json.loads(meta.read_text(encoding="utf-8"))["origem"]
            assert "05b_professor_aberto" in origem["treino"] and "05b_professor_aberto" in origem["dev"] and "11a_" not in json.dumps(origem)


com_a_entrega = pytest.mark.skipif(not (ENTREGAS / etapa11b.ETAPA / "sistemas.json").exists(), reason="a entrega da Etapa 11b ainda não foi gerada")


@com_a_entrega
def test_os_alunos_declarados_entram_no_comparativo_sem_tomar_o_lugar_dos_antigos(tmp_path):
    from especialista.experimentos import etapa9, etapa10d

    assert not [f for f in etapa10d.registro(base=tmp_path).values() if "declarado_em" in f]   # o comparativo da 10d não muda: só lê as entregas da lista dele
    registro = etapa10d.registro(base=tmp_path, valem=None)                                      # quem quer os alunos novos diz que a entrega desta etapa vale
    novas = {chave: ficha for chave, ficha in registro.items() if ficha.get("declarado_em", "").startswith(f"entregas/{etapa11b.ETAPA}/")}
    assert set(novas) == {f"{prefixo}aluno_{s}@{NOVO}" for prefixo in ("", "xlmr/") for s in ("b1", "b2")}
    # os alunos antigos continuam onde estavam, com o professor antigo
    assert registro["aluno_b1"]["professor"] == "qwen" and registro["xlmr/aluno_b2"]["professor"] == "qwen" and NOVO in registro
    pares = {(c["b"], c["a"]): c["familias"] for c in etapa10d.lista_de_comparacoes(registro, valem=None)}
    for s in ("b1", "b2"):
        assert pares[(f"xlmr/aluno_{s}@{NOVO}", f"aluno_{s}@{NOVO}")] == ["entre_caminhos"]      # XLM-R − rede do zero, com o mesmo professor
        assert pares[(f"aluno_{s}@{NOVO}", NOVO)] == ["aluno_professor"] and pares[(f"xlmr/aluno_{s}@{NOVO}", NOVO)] == ["aluno_professor"]
    assert pares[("xlmr/aluno_b1", "aluno_b1")] == ["entre_caminhos"] and pares[("aluno_b1", "qwen")] == ["aluno_professor"]   # as dos antigos não mudaram
    # o número que o comparativo lê é o desta etapa
    tabela, qualidade_da_etapa = etapa9.tabela_final(novas, entregas=ENTREGAS), pd.read_csv(ENTREGAS / etapa11b.ETAPA / "qualidade.csv")
    do_teste = qualidade_da_etapa[(qualidade_da_etapa["particao"] == "test") & (qualidade_da_etapa["metrica"] == "exact_match")].set_index("sistema")["valor"]
    for chave, ficha in novas.items():
        sigla = "B1" if "aluno_b1" in chave else "B2"
        assert tabela.loc[chave, "exact_match"] == pytest.approx(do_teste[f"{ficha['caminho']}:{sigla}:novo"])


@com_a_entrega
def test_a_decisao_entregue_e_a_do_cache_e_os_modelos_sao_links():
    entrega = ENTREGAS / etapa11b.ETAPA
    decisao = json.loads((entrega / "decisao.json").read_text(encoding="utf-8"))
    for c in etapa11b.CAMINHOS:
        do_cache = json.loads((EXECUCOES / etapa11b.ETAPA / NOVO / f"temperatura_{c}.json").read_text(encoding="utf-8"))
        assert decisao["temperatura_do_b2"][c] == do_cache["temperatura_do_b2"] and decisao["abertas_no_teste"][c] == do_cache["abertas_no_teste"]
        assert decisao["alunos"][c]["B2"]["aluno"] == etapa6.variante_do_b2(do_cache["temperatura_do_b2"])
        for sigla in etapa11b.ENTREGUES:
            for semente, pasta in decisao["alunos"][c][sigla]["pastas_por_semente"].items():
                no_cache = EXECUCOES / etapa11b.ETAPA / decisao["alunos"][c][sigla]["variante"] / f"semente_{semente}" / "modelo" / "config.json"
                assert (etapa11b.RAIZ / pasta / "config.json").samefile(no_cache)
    assert json.loads((entrega / "regras.json").read_text(encoding="utf-8"))["escritas_em"] == decisao["regras"]["escritas_em"]
    manifesto = entrega / "manifesto.json"
    if manifesto.exists():      # as tabelas pequenas ainda são as que o manifesto registrou
        saidas = json.loads(manifesto.read_text(encoding="utf-8"))["saidas"]
        for nome in ("decisao.json", "sistemas.json", "comparacoes.csv", "repasses.csv", "RESUMO.md"):
            assert etapa11a.sha256_do_arquivo(entrega / nome) == saidas[nome]["sha256"], nome
