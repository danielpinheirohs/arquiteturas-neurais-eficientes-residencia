"""Etapa 11a: as partes leves (nome de versão, retomada por lotes inteiros, pacote por lista explícita)."""

import json

import numpy as np
import pytest

from especialista.caminhos import ENTREGAS
from especialista.experimentos import etapa11a, etapa11a_colab

ENTREGA = ENTREGAS / etapa11a.ETAPA
com_entrega = pytest.mark.skipif(not (ENTREGA / "escolha.json").exists(), reason="a entrega da Etapa 11a ainda não foi gerada")


def test_o_nome_da_versao_diz_o_que_ela_e():
    v = etapa11a.descrever("qwen3_8b__fp16__colab_l4")
    assert (v["repositorio"], v["precisao"], v["tipo_de_calculo"], v["maquina"]) == ("Qwen/Qwen3-8B", "fp16", "float16", "colab_l4")
    assert etapa11a.descrever(etapa11a.VERSAO_DA_5B)["bits_por_peso"] == 4


def test_a_retomada_deixa_so_lotes_inteiros(tmp_path):
    arquivo = tmp_path / "respostas.jsonl"
    linhas = [json.dumps({"id": f"dev-{i}"}) for i in range(15)]
    arquivo.write_text("\n".join(linhas) + '\n{"id": "dev-15", "tex', encoding="utf-8")      # 2 lotes e meio, e uma linha cortada
    assert etapa11a.retomar(arquivo, total=20) == {f"dev-{i}" for i in range(12)}
    assert len(arquivo.read_text(encoding="utf-8").splitlines()) == 12
    # um arquivo completo fica como está, mesmo que o último lote seja menor
    arquivo.write_text("\n".join(linhas[:14]) + "\n", encoding="utf-8")
    assert len(etapa11a.retomar(arquivo, total=14)) == 14


def test_o_pacote_nao_leva_segredo_nem_o_que_nao_precisa():
    nomes = etapa11a_colab.ARQUIVOS_DO_PACOTE
    assert not any(".env" in nome or "claude" in nome for nome in nomes)
    assert all(nome.startswith(("src/especialista/", "entregas/01_dados/", "data/execucoes/11a_professor_aberto_maior/")) or nome in ("pyproject.toml", "uv.lock") for nome in nomes)


def test_o_limiar_de_ambiente_esta_ancorado_nos_atalhos_da_5b():
    atalhos = json.loads((etapa11a.RAIZ / "entregas" / "05b_professor_aberto" / "escolha.json").read_text(encoding="utf-8"))["conferencia_dos_atalhos"]["em_lote_contra_um_a_um"]
    limiar = etapa11a.LIMIAR_DE_AMBIENTE
    assert limiar["p95_da_diferenca_de_probabilidade_max"] >= atalhos["maior_diferenca_de_probabilidade"]
    assert limiar["mesma_resposta_lida_min"] == 1 - 2 * (1 - atalhos["mesma_resposta_depois_de_lida"] / 32)


def test_a_linha_que_troca_o_professor_aponta_para_a_pasta_da_versao():
    assert etapa11a.como_professor(etapa11a.VERSAO_DA_5B) == {"nome": "Qwen3-8B", "etapa": "05b_professor_aberto", "previsoes": "qwen3_8b"}
    novo = etapa11a.como_professor("qwen3_14b__fp16__colab_a100")
    assert novo["etapa"] == "11a_professor_aberto_maior/professores/qwen3_14b__fp16__colab_a100" and novo["previsoes"] == "qwen3_14b__fp16__colab_a100"


@com_entrega
def test_a_escolha_gravada_segue_a_regra_das_proprias_disputas():
    escolha = json.loads((ENTREGA / "escolha.json").read_text(encoding="utf-8"))
    titular = escolha["titular_de_partida"]
    for desafiante, disputa in escolha["disputas"].items():
        assert disputa["contra"] == titular
        titular = desafiante if disputa["ic_min"] > 0 else titular
    assert escolha["escolhido"] == titular and escolha["regra_escrita_em"] < escolha["gerado_em"]


@com_entrega
def test_o_codigo_da_etapa_6_le_cada_versao_trocando_so_o_caminho(monkeypatch):
    from especialista.dados import massive
    from especialista.experimentos import etapa6

    particoes, rotulos = massive.carregar_entrega()
    for versao in (*etapa11a.CANDIDATOS, etapa11a.EXTENSAO):
        monkeypatch.setitem(etapa6.PROFESSORES, "aberto", etapa11a.como_professor(versao))
        do_treino = etapa6.treino_do_professor("aberto", particoes["train"])
        logits = etapa6.logits_do_professor("treino", len(particoes["train"]), rotulos["intents"])
        assert len(etapa6.previsoes_do_professor("aberto", "dev", particoes["dev"])) == len(particoes["dev"])
        assert len(etapa6.previsoes_do_professor("aberto", "test", particoes["test"])) == len(particoes["test"])
        # nos pedidos com rótulo aproveitável, a intenção do rótulo é a de maior pontuação
        com_rotulo = do_treino["valido"].to_numpy()
        do_rotulo = np.array([rotulos["intents"].index(i) if i in rotulos["intents"] else 0 for i in do_treino["intent"]])
        assert logits.dtype == np.float32 and np.all(logits[com_rotulo, do_rotulo[com_rotulo]] >= logits[com_rotulo].max(axis=1) - 1e-4)


@com_entrega
def test_os_professores_declarados_entram_na_tabela_do_comparativo():
    from especialista.experimentos import etapa9, etapa10d

    _, fichas, _ = etapa10d.fichas_declaradas(ENTREGAS)
    novas = {chave: ficha for chave, ficha in fichas.items() if ficha["declarado_em"].startswith(f"entregas/{etapa11a.ETAPA}/")}
    assert set(novas) == {*etapa11a.CANDIDATOS, etapa11a.EXTENSAO}
    assert [chave for chave, ficha in novas.items() if ficha["escolhido_na_escada"]] == [json.loads((ENTREGA / "escolha_extensao_32b.json").read_text(encoding="utf-8"))["escolhido"]]
    assert all(arquivo.exists() for ficha in novas.values() for lista in ficha["previsoes"].values() for arquivo in lista)
    tabela = etapa9.tabela_final(novas, entregas=ENTREGAS)      # indexada pela chave do sistema
    escolha = json.loads((ENTREGA / "escolha.json").read_text(encoding="utf-8"))
    extensao = json.loads((ENTREGA / "escolha_extensao_32b.json").read_text(encoding="utf-8"))
    do_teste = {**{v: x["test"]["exact_match"] for v, x in escolha["candidatos"].items()}, etapa11a.EXTENSAO: extensao["candidatos"][etapa11a.EXTENSAO]["test"]["exact_match"]}
    for versao in novas:      # a linha do teste, e não a de dev
        assert tabela.loc[versao, "exact_match"] == pytest.approx(do_teste[versao])


@com_entrega
def test_nada_da_etapa_5b_mudou():
    retroativa = json.loads((ENTREGA / "professores" / etapa11a.VERSAO_DA_5B / "versao.json").read_text(encoding="utf-8"))
    for caminho, registro in retroativa["saidas"].items():
        assert etapa11a.sha256_do_arquivo(etapa11a.RAIZ / caminho) == registro["sha256"], caminho


def test_a_extensao_tem_modelo_com_commit_fixado_e_regra_propria():
    v = etapa11a.descrever(etapa11a.EXTENSAO)
    assert (v["repositorio"], v["tipo_de_calculo"], v["maquina"]) == ("Qwen/Qwen3-32B", "float16", "colab_g4") and len(v["commit"]) == 40
    regra = etapa11a.REGRA_DA_EXTENSAO
    assert regra["regra"] == etapa11a.REGRA_DA_ESCADA["regra"] and regra["titular"] == "qwen3_14b__fp16__colab_a100"
    assert "NÃO estava nas regras" in regra["o_que_ja_se_sabia"]


def test_a_projecao_de_custo_soma_o_que_passou_o_que_falta_e_a_folga():
    horas, unidades = etapa11a_colab.projetar_custo(decorrido_s=1800, segundos_por_pedido=0.9, pedidos_restantes=14000, taxa_por_hora=10.0, folga_s=0)
    assert horas == pytest.approx(4.0) and unidades == pytest.approx(40.0)
    assert etapa11a_colab.projetar_custo(1800, 1.5, 14488, 10.05)[1] > 60      # ritmo lento demais: passa do teto e a sessão é desligada


@com_entrega
def test_a_extensao_e_um_arquivo_novo_e_a_escolha_de_1811_fica_como_estava():
    guardada = ENTREGA / "versoes" / etapa11a.VERSAO_ENTREGUE_EM_1811
    assert etapa11a.sha256_do_arquivo(ENTREGA / "escolha.json") == etapa11a.sha256_do_arquivo(guardada / "escolha.json")
    assert json.loads((guardada / "LEIA.json").read_text(encoding="utf-8"))["todos_iguais"]
    assert all((guardada / nome).exists() for nome in ("RESUMO.md", "11a_professor_aberto_maior.html", "11a_professor_aberto_maior.ipynb", "manifesto.json", "sistemas.json"))
    escolha = json.loads((ENTREGA / "escolha.json").read_text(encoding="utf-8"))
    extensao = json.loads((ENTREGA / "escolha_extensao_32b.json").read_text(encoding="utf-8"))
    regra = json.loads((ENTREGA / "regra_extensao_32b.json").read_text(encoding="utf-8"))
    disputa = extensao["disputas"][etapa11a.EXTENSAO]
    # a extensão parte do escolhido da escada planejada, com a regra gravada antes, e diz que veio depois
    assert disputa["contra"] == escolha["escolhido"] == regra["titular"] and "NÃO estava" in extensao["o_que_ja_se_sabia"]
    assert extensao["toma_o_lugar"] == (disputa["ic_min"] > 0) and extensao["escolhido"] == (etapa11a.EXTENSAO if disputa["ic_min"] > 0 else escolha["escolhido"])
    assert regra["escrita_em"] < extensao["gerado_em"] and escolha["gerado_em"] < regra["escrita_em"]


def test_as_horas_de_uma_sessao_que_vira_o_dia():
    assert etapa11a._horas_entre("22:25:24", "00:54:02") == pytest.approx(2.4772, abs=1e-4)
    assert etapa11a._horas_entre("17:26:24", "21:03:40") == pytest.approx(3.6211, abs=1e-4)
