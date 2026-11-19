"""Etapa 12: as peças da investigação dos achados que não dependem de treino nem de modelo grande.

Nada aqui treina, usa a placa ou lê os modelos das etapas anteriores. Os grafos
ONNX são minúsculos e montados no próprio teste.
"""

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from especialista.avaliacao.previsao import Previsao
from especialista.compressao import poda_encoder
from especialista.compressao.sistema_onnx import abrir_sessao
from especialista.experimentos import etapa12


def test_teste_de_sinal():
    assert etapa12.binomial_bilateral(0, 0) == 1.0
    assert etapa12.binomial_bilateral(5, 10) == 1.0
    assert etapa12.binomial_bilateral(10, 10) == pytest.approx(2 / 2**10)
    # 78 contra 53: o número do achado 3
    assert 0.03 < etapa12.binomial_bilateral(78, 131) < 0.04
    assert etapa12.binomial_bilateral(53, 131) == etapa12.binomial_bilateral(78, 131)


def test_margem_entre_as_duas_primeiras():
    pontuacoes = np.array([[0.0, 0.0, -50.0], [10.0, 0.0, 0.0]])
    m = etapa12.margem(pontuacoes)
    assert m[0] == pytest.approx(0.0, abs=1e-9)
    assert m[1] == pytest.approx(1.0, abs=1e-3)


def test_tipo_e_camada_da_operacao():
    assert etapa12.tipo_da_operacao("encoder.encoder.layer.7.output.dense.produto") == "saida_da_rede_densa"
    assert etapa12.tipo_da_operacao("encoder.encoder.layer.0.attention.output.dense.produto") == "saida_da_atencao"
    assert etapa12.tipo_da_operacao("encoder.encoder.layer.11.attention.self.key.produto") == "consulta_chave_valor"
    assert etapa12.tipo_da_operacao("encoder.encoder.layer.3.intermediate.dense.produto") == "entrada_da_rede_densa"
    assert etapa12.tipo_da_operacao("intencao.produto") == "cabecas"
    assert etapa12.tipo_da_operacao("encoder.embeddings.word_embeddings.linhas") == "tabelas"
    assert etapa12.camada_da_operacao("encoder.encoder.layer.11.output.dense.produto") == 11
    assert etapa12.camada_da_operacao("entidades.produto") is None


def test_texto_do_intervalo_no_estilo_do_projeto():
    assert etapa12.texto_ic({"diferenca": -1.04, "ic_min": -1.8, "ic_max": -0.3}) == "−1,0 (−1,8 a −0,3)"
    assert etapa12.texto_ic({"diferenca": 0.05, "ic_min": -0.39, "ic_max": 0.5}, 2) == "+0,05 (−0,39 a +0,50)"


def test_copia_do_professor():
    certo, errado, outro = Previsao("a"), Previsao("b"), Previsao("c")
    gab = [certo, certo, certo, certo]
    prof = [certo, errado, errado, certo]          # o professor erra nos pedidos 1 e 2
    aluno = [certo, errado, outro, errado]          # copia o erro no 1, erra de outro jeito no 2
    r = etapa12.copia_do_professor(gab, prof, [aluno])
    assert r["pedidos_em_que_o_professor_erra"] == 2
    assert r["copia_o_erro"] == pytest.approx(50.0)
    assert r["acerta_onde_o_professor_erra"] == pytest.approx(0.0)
    assert r["concorda_com_o_professor"] == pytest.approx(50.0)


def test_gravador_guarda_as_respostas_e_devolve_as_metricas():
    gravador = etapa12.Gravador()
    gab = [Previsao("a"), Previsao("b")]
    metricas = gravador.avaliar(gab, [Previsao("a"), Previsao("a")])
    assert metricas["intent_acc"] == pytest.approx(0.5)
    assert gravador.respostas == [[Previsao("a"), Previsao("a")]]


def test_emprestar_devolve_o_que_estava_mesmo_com_erro():
    antes = poda_encoder.qualidade
    with pytest.raises(RuntimeError):
        with etapa12.emprestar(poda_encoder, qualidade="outra"):
            assert poda_encoder.qualidade == "outra"
            raise RuntimeError("no meio")
    assert poda_encoder.qualidade is antes


def _grafo_com_tabela(tabela: np.ndarray) -> onnx.ModelProto:
    no = helper.make_node("Gather", ["tabela", "ids"], ["linhas"], name="linhas", axis=0)
    grafo = helper.make_graph([no], "t", [helper.make_tensor_value_info("ids", TensorProto.INT64, ["n"])],
                              [helper.make_tensor_value_info("linhas", TensorProto.FLOAT, ["n", tabela.shape[1]])],
                              initializer=[numpy_helper.from_array(tabela, "tabela")])
    rede = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    rede.ir_version = 8
    return rede


def test_tabela_por_linha_erra_no_maximo_meio_degrau_de_cada_linha():
    rng = np.random.default_rng(0)
    tabela = (rng.normal(size=(6, 5)) * np.array([[0.01], [0.1], [1], [10], [3], [0.5]])).astype(np.float32)
    rede = etapa12.tabela_por_linha(_grafo_com_tabela(tabela), "tabela")
    onnx.checker.check_model(rede)
    ids = np.arange(6, dtype=np.int64)
    saida = abrir_sessao(rede.SerializeToString()).run(["linhas"], {"ids": ids})[0]
    faixa = np.maximum(tabela.max(axis=1), 0) - np.minimum(tabela.min(axis=1), 0)
    erro = np.abs(saida - tabela).max(axis=1)
    assert np.all(erro <= faixa / 255 / 2 + 1e-6)
    # com um degrau só para a tabela inteira, a linha de valores pequenos perderia quase tudo
    assert erro[0] < 1e-4


def test_plano_gravado_tem_hipoteses_testes_criterios_e_leitura_no_teste():
    arquivo = etapa12.pasta_da_etapa() / "plano.json"
    if not arquivo.exists():
        pytest.skip("o plano da Etapa 12 não está em disco")
    plano = json.loads(arquivo.read_text(encoding="utf-8"))
    for achado in ("achado_1", "achado_2", "achado_3"):
        assert {"hipoteses", "testes", "criterios", "leitura_no_teste"} <= set(plano[achado])
