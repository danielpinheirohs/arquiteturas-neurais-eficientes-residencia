"""Exportação para ONNX e quantização em 8 bits: a rede exportada faz a mesma conta, e a quantizada continua um sistema da régua."""

import numpy as np
import pandas as pd
import pytest
import torch

from especialista.avaliacao.previsao import Previsao
from especialista.compressao import exportacao, quantizacao
from especialista.compressao.exportacao import SistemaOnnx
from especialista.experimentos import etapa8
from especialista.modelos.diet import Config, SistemaDiet
from especialista.modelos.diet.crf import CRF, restricoes_bio

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
PEQUENO = dict(d_modelo=32, n_cabecas=2, d_ffn=64, d_features=16, d_rotulo=8)
MEDIO = dict(d_modelo=64, n_cabecas=4, d_ffn=256, d_features=64, d_rotulo=8)   # pesos bastantes para o tamanho do arquivo ser o dos pesos
TEXTOS = [texto for _, texto, _ in ENUNCIADOS]
# pedidos que o treino não tem: uma palavra só, palavra nunca vista, palavra sem nenhum feature conhecido, pedido longo
DIFICEIS = ["toca", "acorda-me xyzw amanhã", "ŧŧŧ às nove", " ".join(["toca"] * 40)]


@pytest.fixture
def dados() -> pd.DataFrame:
    return pd.DataFrame({"intent": i, "utt": texto, "tokens": texto.split(), "tags": tags.split()} for i, texto, tags in ENUNCIADOS)


def novo(dados, semente=0, **config) -> SistemaDiet:
    sistema = SistemaDiet.novo(dados, ROTULOS, Config(**{**PEQUENO, **config}), semente=semente, contagem_minima=1)
    sistema.modelo.eval()
    return sistema


def treinado(dados, **config) -> SistemaDiet:
    """Um sistema que decorou os 8 exemplos: as suas respostas têm folga, e não mudam por um arredondamento."""
    torch.manual_seed(0)
    sistema = novo(dados, **config)
    otimizador = torch.optim.Adam(sistema.modelo.parameters(), lr=3e-3)
    lote = sistema.lote(dados)
    for _ in range(150):
        sistema.modelo.train()
        perda = sistema.modelo.perda(lote)["total"]
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    sistema.modelo.eval()
    return sistema


def pontuacoes_do_pytorch(sistema: SistemaDiet, textos: list[str]):
    lote = sistema.featurizador.montar([texto.split() for texto in textos])
    with torch.no_grad():
        intencoes, emissoes = sistema.modelo(lote)
    real = np.arange(lote.forma[1])[None, :] < lote.comprimentos.numpy()[:, None]
    return intencoes.numpy(), emissoes.numpy(), real


# --------------------------------------------------------------------------
# A exportação sozinha não muda a conta
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [{}, {"n_camadas": 1}, {"n_camadas": 3}, {"distancia_max": 2}, {"atalho_intencao": "resumo"}, {"atalho_intencao": "features"}, {"usar_crf": False}],
    ids=["padrao", "uma_camada", "tres_camadas", "distancia_curta", "atalho_resumo", "atalho_features", "sem_crf"],
)
def test_onnx_de_32_bits_faz_a_mesma_conta_do_pytorch(dados, tmp_path, config):
    sistema = novo(dados, semente=3, **config)
    with torch.no_grad():   # pesos que nascem em zero (CRF, marca do CLS, atalho) ganham valores, para a conta depender deles
        for nome, p in sistema.modelo.named_parameters():
            if not p.any():
                p.normal_(std=0.3)
    exportado = SistemaOnnx.carregar(exportacao.exportar(sistema, tmp_path / "fp32"), tamanho_do_lote=64)
    textos = TEXTOS + DIFICEIS

    intencoes, emissoes, real = pontuacoes_do_pytorch(sistema, textos)
    do_onnx_i, do_onnx_e = exportado.pontuar([texto.split() for texto in textos])
    assert do_onnx_i.shape == intencoes.shape and do_onnx_e.shape == emissoes.shape
    assert np.allclose(do_onnx_i, intencoes, atol=1e-4)
    assert np.allclose(do_onnx_e[real], emissoes[real], atol=1e-4)
    # as respostas também, fora o pedido de 40 palavras iguais: numa rede sem treino ele empata as tags,
    # e um empate se decide na sétima casa decimal (o Viterbi tem o seu próprio teste, logo abaixo)
    assert exportado.prever(textos[:-1]) == sistema.prever(textos[:-1])


def test_viterbi_em_numpy_e_o_do_crf():
    torch.manual_seed(0)
    crf = CRF(len(ROTULOS["tags"]), *restricoes_bio(ROTULOS["tags"]))
    with torch.no_grad():
        for p in crf.parameters():
            p.normal_()
        transicao, inicio = crf._pesos()
    for comprimento in (1, 2, 5, 12):
        emissoes = torch.randn(4, comprimento, len(ROTULOS["tags"]))
        esperado = crf.decodificar(emissoes, torch.ones(4, comprimento, dtype=torch.bool))
        obtido = [exportacao.viterbi(e, transicao.numpy(), inicio.numpy(), crf.fim.detach().numpy()) for e in emissoes.numpy()]
        assert obtido == esperado


def test_lotes_com_pedidos_de_tamanhos_diferentes(dados, tmp_path):
    """Em 32 bits, um pedido tem a mesma resposta sozinho, num lote de iguais ou ao lado de um pedido bem mais longo."""
    sistema = treinado(dados)
    pasta = exportacao.exportar(sistema, tmp_path / "fp32")
    textos = TEXTOS + DIFICEIS
    um_por_vez = SistemaOnnx.carregar(pasta).prever(textos)
    assert um_por_vez == SistemaOnnx.carregar(pasta, tamanho_do_lote=3).prever(textos)
    assert um_por_vez == SistemaOnnx.carregar(pasta, tamanho_do_lote=64).prever(textos)
    assert um_por_vez == sistema.prever(textos)

    quantizado = SistemaOnnx.carregar(quantizacao.quantizar(pasta, tmp_path / "dinamica", "dinamica"), tamanho_do_lote=5)
    intencoes, emissoes = quantizado.pontuar([texto.split() for texto in textos])
    assert intencoes.shape == (len(textos), 3) and emissoes.shape == (len(textos), 40, 9)
    assert len(quantizado.prever(textos)) == len(textos)


# --------------------------------------------------------------------------
# O sistema quantizado
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forma", list(quantizacao.FORMAS))
def test_sistema_quantizado_cumpre_o_contrato_da_regua(dados, tmp_path, forma):
    sistema = treinado(dados)
    fp32 = exportacao.exportar(sistema, tmp_path / "fp32")
    quantizado = SistemaOnnx.carregar(quantizacao.quantizar(fp32, tmp_path / forma, forma, TEXTOS * 4))

    assert isinstance(quantizado.nome, str) and quantizacao.FORMAS[forma]["nome"] in quantizado.nome
    assert quantizado.parametros == sistema.parametros
    textos = TEXTOS + DIFICEIS
    previsoes = quantizado.prever(textos)
    assert len(previsoes) == len(textos)
    for texto, previsao in zip(textos, previsoes):
        assert isinstance(previsao, Previsao) and previsao.intent in ROTULOS["intents"]
        for tipo, inicio, fim in previsao.slots:   # entidades bem formadas, dentro do pedido
            assert f"B-{tipo}" in ROTULOS["tags"] and 0 <= inicio < fim <= len(texto.split())
    # num modelo que decorou o treino, 8 bits não mudam as respostas do treino
    assert quantizado.prever(TEXTOS) == sistema.prever(TEXTOS)


def test_estatica_exige_calibracao(dados, tmp_path):
    fp32 = exportacao.exportar(novo(dados), tmp_path / "fp32")
    with pytest.raises(ValueError, match="calibração"):
        quantizacao.quantizar(fp32, tmp_path / "estatica", "estatica")


@pytest.mark.parametrize("forma", ["dinamica", "estatica"])
def test_arquivo_int8_e_menor(dados, tmp_path, forma):
    sistema = novo(dados, **MEDIO)
    fp32 = exportacao.exportar(sistema, tmp_path / "fp32")
    int8 = quantizacao.quantizar(fp32, tmp_path / forma, forma, TEXTOS)

    rede = lambda pasta: (pasta / exportacao.REDE).stat().st_size
    assert rede(int8) < 0.4 * rede(fp32)   # 4 vezes menos nos pesos; o resto do arquivo (vieses, grafo) não encolhe
    assert exportacao.bytes_do_sistema(int8) < exportacao.bytes_do_sistema(fp32)

    pesos = quantizacao.pesos_antes_e_depois(fp32, int8)
    grandes = pesos.loc[["tabela.weight", "projecao.weight", "blocos.0.expande.weight", "emissao.weight"]]
    assert (grandes["tipo_antes"] == "float32").all()
    assert list(grandes["tipo_depois"]) == ["uint8", "int8", "int8", "int8"]   # a tabela sem sinal, as matrizes com sinal
    assert (grandes["bytes_depois"] * 4 == grandes["bytes_antes"]).all()
    assert pesos.loc["projecao.bias", "tipo_depois"] == "float32"   # os vieses ficam em 32 bits


@pytest.mark.parametrize("forma", ["dinamica", "estatica"])
def test_pesos_podados_continuam_em_zero_e_o_arquivo_compactado_encolhe(dados, tmp_path, forma):
    sistema = novo(dados, **MEDIO)
    podados = ["tabela.weight", "projecao.weight", "blocos.0.expande.weight", "blocos.1.atencao.consulta.weight"]
    with torch.no_grad():   # zera 80% dos pesos de menor valor absoluto de algumas matrizes
        for nome in podados:
            peso = sistema.modelo.get_parameter(nome)
            peso.mul_(peso.abs() > peso.abs().flatten().quantile(0.8))
    fp32 = exportacao.exportar(sistema, tmp_path / "fp32")
    int8 = quantizacao.quantizar(fp32, tmp_path / forma, forma, TEXTOS)

    zeros = quantizacao.zeros_conservados(fp32, int8)
    assert zeros["conservados"] and zeros["zeros_perdidos"] == 0
    assert zeros["zeros_depois"] >= zeros["zeros_antes"] > 0

    compactado = exportacao.compactar(int8, tmp_path / "compactado")
    assert exportacao.bytes_do_sistema(compactado) < 0.8 * exportacao.bytes_do_sistema(int8)
    assert SistemaOnnx.carregar(compactado).prever(TEXTOS + DIFICEIS) == SistemaOnnx.carregar(int8).prever(TEXTOS + DIFICEIS)


# --------------------------------------------------------------------------
# A execução da etapa e a regra de decisão
# --------------------------------------------------------------------------


def test_execucao_grava_tudo_o_que_o_notebook_le(dados, tmp_path):
    sistema = treinado(dados)
    modelo = sistema.salvar(tmp_path / "modelo")
    medir = {"dev": dados.assign(id=range(len(dados))), "test": dados.iloc[:5].assign(id=range(5))}
    medido = etapa8.quantizar_modelo(modelo, tmp_path / "saida", medir, TEXTOS)

    assert medido["parametros"] == sistema.parametros and medido["pedidos"] == {"dev": 8, "test": 5}
    assert medido["formas"]["fp32"]["iguais_ao_pytorch_dev"] == 8 and medido["formas"]["fp32"]["iguais_ao_pytorch_test"] == 5
    for forma in etapa8.MEDIDAS:
        for particao in medir:
            assert (tmp_path / "saida" / f"previsoes_{forma}_{particao}.parquet").exists()
    for forma in quantizacao.FORMAS:
        assert medido["formas"][forma]["conservados"] and medido["formas"][forma]["compactado_responde_igual"]
        assert medido["formas"][forma]["bytes_compactado"] > 0
    for forma in ("fp32", *etapa8.CANDIDATAS):   # os sistemas que ficam gravados carregam e respondem
        assert len(SistemaOnnx.carregar(tmp_path / "saida" / "sistemas" / forma).prever(TEXTOS)) == len(TEXTOS)


def _respostas(certas: int, total: int = 100) -> list[Previsao]:
    """`certas` respostas inteiramente certas e o resto com a entidade errada."""
    return [Previsao("a", (("x", 0, 1),)) if i < certas else Previsao("a", (("x", 1, 2),)) for i in range(total)]


def test_regra_da_forma_final():
    gabarito = _respostas(100)
    bom, queda_pequena, queda_grande = [_respostas(90)] * 3, [_respostas(90)[:-1] + _respostas(0, 1)] * 3, [_respostas(80)] * 3
    assert etapa8.LIMITE_DO_PLANO == 0.01 and etapa8.CRITERIO_DO_PLANO == "slot_f1"

    # as duas iguais ao modelo de 32 bits: fica a dinâmica, a não ser que a estática seja mais rápida em todas as rodadas
    iguais = {"pytorch": bom, "dinamica": bom, "estatica": bom}
    assert etapa8.escolher_forma(gabarito, iguais)["forma"] == "dinamica"
    assert etapa8.escolher_forma(gabarito, iguais, {"dinamica": [0.30, 0.31], "estatica": [0.29, 0.32]})["forma"] == "dinamica"
    assert etapa8.escolher_forma(gabarito, iguais, {"dinamica": [0.30, 0.31], "estatica": [0.29, 0.30]})["forma"] == "estatica"

    # uma queda de menos de 1 ponto de F1 passa
    decisao = etapa8.escolher_forma(gabarito, {"pytorch": bom, "dinamica": queda_pequena, "estatica": bom})
    assert decisao["passam"] == ["dinamica", "estatica"] and not decisao["precisa_de_qat"]

    # uma forma cai mais de 1 ponto: fica a outra
    decisao = etapa8.escolher_forma(gabarito, {"pytorch": bom, "dinamica": queda_grande, "estatica": bom})
    assert decisao["forma"] == "estatica" and decisao["passam"] == ["estatica"]
    assert not decisao["tabela"].loc["dinamica", "passa"] and decisao["tabela"].loc["dinamica", "diferenca"] < -0.01

    # as duas caem mais de 1 ponto: só então entra o treino ciente de quantização
    decisao = etapa8.escolher_forma(gabarito, {"pytorch": bom, "dinamica": queda_grande, "estatica": queda_grande})
    assert decisao["forma"] is None and decisao["precisa_de_qat"]


def test_maquina_quieta_desconta_o_proprio_processo():
    # em 10 s de relógio, a máquina gastou 15 s de CPU, 10 deles deste processo: os outros ocuparam meio núcleo
    assert etapa8.nucleos_de_outros((100.0, 50.0, 2.0), (110.0, 65.0, 12.0)) == pytest.approx(0.5)
    assert etapa8.nucleos_de_outros((0.0, 0.0, 0.0), (10.0, 9.9, 10.0)) == 0.0   # nunca negativo
    parede, de_todos, proprio = etapa8.relogios()
    assert parede > 0 and proprio > 0 and de_todos >= proprio


def test_le_a_decisao_da_poda_nos_dois_formatos():
    principal = {"variante": "linhas_50", "nome": "Sem 50% das linhas", "modelos": "entregas/07_poda/modelos/aluno_a/linhas_50", "familia": "linhas"}
    secundaria = {"variante": "magnitude_70", "nome": "Magnitude, 70%", "modelos": "entregas/07_poda/modelos/aluno_a/magnitude_70", "familia": "magnitude"}
    papeis = lambda fichas: {f["variante"]: f["papel"] for f in fichas}

    em_lista = etapa8._fichas_da_poda({"seguem": [secundaria, principal], "principal": "linhas_50"})
    assert papeis(em_lista) == {"magnitude_70": "secundaria", "linhas_50": "principal"}
    em_chaves = etapa8._fichas_da_poda({"principal": principal, "secundaria": secundaria})
    assert papeis(em_chaves) == {"linhas_50": "principal", "magnitude_70": "secundaria"}
    assert etapa8._fichas_da_poda({"seguem": [], "principal": None}) == []


def test_sistema_exportado_roda_sem_pytorch(dados, tmp_path):
    """Carregar e consultar o sistema exportado, num processo novo, não traz o PyTorch (nem pandas, nem o pacote onnx)."""
    import json
    import subprocess
    import sys

    sistema = treinado(dados)
    textos = TEXTOS[:5]
    pasta = exportacao.exportar(sistema, tmp_path / "fp32")
    esperado = [[p.intent, [list(s) for s in p.slots]] for p in SistemaOnnx.carregar(pasta).prever(textos)]
    programa = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from especialista.compressao.sistema_onnx import SistemaOnnx\n"
        "previsoes = SistemaOnnx.carregar(Path(sys.argv[1])).prever(json.loads(sys.argv[2]))\n"
        "pesados = [m for m in ('torch', 'pandas', 'pyarrow', 'onnx') if m in sys.modules]\n"
        "print(json.dumps({'pesados': pesados, 'previsoes': [[p.intent, [list(s) for s in p.slots]] for p in previsoes]}))\n"
    )
    saida = subprocess.run([sys.executable, "-c", programa, str(pasta), json.dumps(textos)], capture_output=True, text=True, check=True)
    resultado = json.loads(saida.stdout)
    assert resultado["pesados"] == []
    assert resultado["previsoes"] == esperado
