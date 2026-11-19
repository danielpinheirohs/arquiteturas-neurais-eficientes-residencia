"""Exportação do encoder pré-treinado para ONNX e quantização em 8 bits (Etapa 10c), num XLM-R minúsculo de pesos sorteados, na CPU.

O que está em jogo: o grafo montado à mão tem de fazer a conta da biblioteca
(com vocabulário cortado e com menos camadas também), o sistema sem PyTorch
tem de escolher o primeiro subtoken de cada palavra como o alinhador do
treino, e a quantização não pode desfazer os zeros da poda.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from especialista.avaliacao.previsao import Previsao
from especialista.compressao import exportacao, exportacao_encoder, poda_encoder, quantizacao
from especialista.compressao.sistema_encoder_onnx import Alinhador, SistemaEncoderOnnx
from especialista.experimentos import etapa8, etapa10c
from especialista.modelos.encoder import SistemaEncoder
from especialista.treino_encoder import Receita, treinar

from test_diet import ROTULOS
from test_encoder import dados
from test_poda_encoder import PEQUENO, TEXTOS, pontuacoes, sistema_minusculo, tokenizador_unigram

transformers = pytest.importorskip("transformers")

# pedidos que o treino não tem: uma palavra só, palavra com letras que o tokenizador não conhece, e um pedido que
# passa do limite de subtokens (as últimas palavras ficam sem primeiro subtoken)
DIFICEIS = ["toca", "acorda-me xyzw amanhã", " ".join(["acorda-me"] * 12)]
ENUNCIADOS = [texto.split() for texto in TEXTOS + DIFICEIS]


def do_treino(sistema: SistemaEncoder) -> torch.Tensor:
    return poda_encoder.ids_do_texto(sistema, [texto.split() for texto in TEXTOS])


VARIANTES = {
    "inteiro": lambda: sistema_minusculo(semente=3),
    "uma_camada": lambda: sistema_minusculo(semente=3, camadas=1),
    "vocabulario_cortado": lambda: (lambda s: poda_encoder.cortar_vocabulario(s, do_treino(s)))(sistema_minusculo(semente=3)),
    "menos_camadas": lambda: poda_encoder.manter_camadas(sistema_minusculo(semente=3), 2),
    "vocabulario_cortado_e_menos_camadas": lambda: (lambda s: poda_encoder.manter_camadas(poda_encoder.cortar_vocabulario(s, do_treino(s)), 2))(sistema_minusculo(semente=3)),
}


def sistema_medio(semente: int = 0) -> SistemaEncoder:
    """Um encoder com pesos bastantes para o tamanho do arquivo ser o dos pesos, e não o do grafo."""
    tokenizador = tokenizador_unigram()
    torch.manual_seed(semente)
    do_encoder = transformers.XLMRobertaConfig(
        vocab_size=len(tokenizador), hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=512,
        max_position_embeddings=66, pad_token_id=tokenizador.pad_token_id, type_vocab_size=1,
    )
    return SistemaEncoder.novo(ROTULOS, PEQUENO, semente=semente, encoder=transformers.XLMRobertaModel(do_encoder, add_pooling_layer=False), tokenizador=tokenizador)


@pytest.fixture(scope="module")
def pasta_do_treinado(tmp_path_factory):
    """Um sistema que decorou os 8 exemplos: as suas respostas têm folga, e não mudam por um arredondamento."""
    sistema = sistema_minusculo()
    treinar(sistema, dados(), dados(), Receita(epocas=40, taxa=1e-2, lote=4))
    return sistema.salvar(tmp_path_factory.mktemp("treinado") / "modelo")


# --------------------------------------------------------------------------
# A exportação sozinha não muda a conta
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variante", list(VARIANTES))
def test_onnx_de_32_bits_faz_a_mesma_conta_do_pytorch(tmp_path, variante):
    sistema = VARIANTES[variante]()
    with torch.no_grad():   # os vieses nascem em zero: ganham valores, para a conta depender deles
        for _, p in sistema.modelo.named_parameters():
            if not p.any():
                p.normal_(std=0.3)
    pasta = exportacao_encoder.exportar(sistema, tmp_path / "fp32")
    exportado = SistemaEncoderOnnx.carregar(pasta, tamanho_do_lote=64)
    textos = TEXTOS + DIFICEIS

    intencoes, emissoes = (t.numpy() for t in pontuacoes(sistema, textos))
    do_onnx_i, do_onnx_e = exportado.pontuar(ENUNCIADOS)      # todos num lote só: há preenchimento, e as posições dependem dele
    _, real = exportado.alinhador.alinhar(ENUNCIADOS)
    assert do_onnx_i.shape == intencoes.shape and do_onnx_e.shape == emissoes.shape
    assert np.allclose(do_onnx_i, intencoes, atol=1e-5)
    assert np.allclose(do_onnx_e[real], emissoes[real], atol=1e-5)
    assert exportado.prever(textos) == sistema.prever(textos)
    assert SistemaEncoderOnnx.carregar(pasta).prever(textos) == sistema.prever(textos)   # um pedido por vez, como na régua e na latência

    ficha = json.loads((pasta / exportacao_encoder.FICHA).read_text(encoding="utf-8"))
    assert ficha["camadas"] == len(sistema.modelo.encoder.encoder.layer)
    assert ficha["linhas_da_tabela"] == len(sistema.tokenizador) and ficha["parametros"] == sistema.parametros


def test_a_escolha_do_primeiro_subtoken_e_a_do_alinhador_do_treino(tmp_path):
    sistema = sistema_minusculo()
    alinhador = Alinhador.carregar(exportacao_encoder.exportar(sistema, tmp_path / "fp32"))
    enunciados = ENUNCIADOS + [[]]   # e um pedido vazio
    lote = sistema.alinhador.montar(enunciados)
    entradas, real = alinhador.alinhar(enunciados)
    assert np.array_equal(entradas["ids"], lote.ids.numpy()) and np.array_equal(entradas["atencao"], lote.atencao.numpy())
    assert np.array_equal(entradas["primeiros"], lote.primeiros.numpy()) and np.array_equal(real, lote.real.numpy())
    assert entradas["ids"].shape[1] == PEQUENO.max_subtokens                 # o pedido longo bate no limite...
    assert not real[len(TEXTOS) + 2, :12].all() and real[0, : len(ENUNCIADOS[0])].all()   # ...e as últimas palavras dele ficam sem subtoken


def test_palavra_sem_subtoken_recebe_O(tmp_path):
    sistema = sistema_minusculo(semente=1)
    exportado = SistemaEncoderOnnx.carregar(exportacao_encoder.exportar(sistema, tmp_path / "fp32"))
    longo = DIFICEIS[-1]
    _, real = exportado.alinhador.alinhar([longo.split()])
    previsao = exportado.prever([longo])[0]
    assert previsao == sistema.prever([longo])[0]
    assert all(fim <= int(real[0].sum()) for _, _, fim in previsao.slots)    # nenhuma entidade passa do último subtoken lido
    assert exportado.prever([""]) == sistema.prever([""])                    # pedido vazio: só a intenção


def test_lotes_com_pedidos_de_tamanhos_diferentes(pasta_do_treinado, tmp_path):
    """Em 32 bits, um pedido tem a mesma resposta sozinho, num lote de iguais ou ao lado de um pedido bem mais longo."""
    sistema = SistemaEncoder.carregar(pasta_do_treinado)
    pasta = exportacao_encoder.exportar(sistema, tmp_path / "fp32")
    textos = TEXTOS + DIFICEIS
    um_por_vez = SistemaEncoderOnnx.carregar(pasta).prever(textos)
    assert um_por_vez == SistemaEncoderOnnx.carregar(pasta, tamanho_do_lote=3).prever(textos)
    assert um_por_vez == SistemaEncoderOnnx.carregar(pasta, tamanho_do_lote=64).prever(textos)
    assert um_por_vez == sistema.prever(textos)

    quantizado = SistemaEncoderOnnx.carregar(quantizacao.quantizar_encoder(pasta, tmp_path / "dinamica", "dinamica"), tamanho_do_lote=5)
    intencoes, emissoes = quantizado.pontuar(ENUNCIADOS)
    assert intencoes.shape == (len(textos), 3) and emissoes.shape == (len(textos), 12, 9)
    assert len(quantizado.prever(textos)) == len(textos)


# --------------------------------------------------------------------------
# O sistema quantizado
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forma", list(quantizacao.FORMAS))
def test_sistema_quantizado_cumpre_o_contrato_da_regua(pasta_do_treinado, tmp_path, forma):
    sistema = SistemaEncoder.carregar(pasta_do_treinado)
    fp32 = exportacao_encoder.exportar(sistema, tmp_path / "fp32")
    quantizado = SistemaEncoderOnnx.carregar(quantizacao.quantizar_encoder(fp32, tmp_path / forma, forma, TEXTOS * 4))

    assert isinstance(quantizado.nome, str) and etapa10c.NOMES[forma] in quantizado.nome
    assert quantizado.parametros == sistema.parametros
    assert not (tmp_path / forma / "temporarios").exists()   # os temporários das ferramentas saem da pasta do sistema
    textos = TEXTOS + DIFICEIS
    previsoes = quantizado.prever(textos)
    assert len(previsoes) == len(textos)
    for texto, previsao in zip(textos, previsoes):
        assert isinstance(previsao, Previsao) and previsao.intent in ROTULOS["intents"]
        for tipo, inicio, fim in previsao.slots:   # entidades bem formadas, dentro do pedido
            assert f"B-{tipo}" in ROTULOS["tags"] and 0 <= inicio < fim <= len(texto.split())
    # num modelo que decorou o treino, 8 bits não mudam as respostas do treino
    assert quantizado.prever(TEXTOS) == sistema.prever(TEXTOS)


def test_estatica_exige_calibracao(tmp_path):
    fp32 = exportacao_encoder.exportar(sistema_minusculo(), tmp_path / "fp32")
    with pytest.raises(ValueError, match="calibração"):
        quantizacao.quantizar_encoder(fp32, tmp_path / "estatica", "estatica")


def test_o_que_fica_em_8_bits_e_o_que_fica_em_32(tmp_path):
    sistema = sistema_minusculo(camadas=2)
    fp32 = exportacao_encoder.exportar(sistema, tmp_path / "fp32")
    partes = quantizacao.operacoes_com_pesos(exportacao.ler_rede(fp32))
    assert len(partes["tabela"]) == 2            # a tabela de subtokens e a de posições
    assert len(partes["matrizes"]) == 6 * 2 + 2  # as 6 de cada camada e as das duas cabeças
    produtos = [no for no in exportacao.ler_rede(fp32).graph.node if no.op_type == "MatMul"]
    assert len(produtos) == len(partes["matrizes"]) + 2 * 2   # os dois produtos da atenção, por camada, não têm matriz gravada

    dinamica = exportacao.ler_rede(quantizacao.quantizar_encoder(fp32, tmp_path / "dinamica", "dinamica"))
    operacoes = exportacao.operacoes_da_rede(dinamica)
    assert operacoes["MatMulInteger"] == len(partes["matrizes"]) and operacoes["MatMul"] == 2 * 2   # só os produtos da atenção continuam em 32 bits
    assert operacoes["LayerNormalization"] == 1 + 2 * 2 and operacoes["Softmax"] == 2 and operacoes["Erf"] == 2


@pytest.mark.parametrize("forma", ["dinamica", "estatica"])
def test_arquivo_int8_e_menor(tmp_path, forma):
    sistema = sistema_medio()
    fp32 = exportacao_encoder.exportar(sistema, tmp_path / "fp32")
    int8 = quantizacao.quantizar_encoder(fp32, tmp_path / forma, forma, TEXTOS)

    rede = lambda pasta: (pasta / exportacao_encoder.REDE).stat().st_size
    assert rede(int8) < 0.4 * rede(fp32)   # 4 vezes menos nos pesos; o resto do arquivo (vieses, grafo) não encolhe
    assert exportacao_encoder.bytes_do_sistema(int8) < exportacao_encoder.bytes_do_sistema(fp32)
    arquivos = exportacao_encoder.bytes_por_arquivo(int8)
    assert set(arquivos) == {exportacao_encoder.REDE, exportacao_encoder.TOKENIZADOR, exportacao_encoder.FICHA}
    assert arquivos[exportacao_encoder.TOKENIZADOR] == exportacao_encoder.bytes_por_arquivo(fp32)[exportacao_encoder.TOKENIZADOR]   # o tokenizador não muda

    pesos = quantizacao.pesos_antes_e_depois(fp32, int8)
    camada = "encoder.encoder.layer.0"
    grandes = pesos.loc[[exportacao_encoder.TABELA_DE_SUBTOKENS, exportacao_encoder.TABELA_DE_POSICOES, f"{camada}.attention.self.query.weight",
                         f"{camada}.intermediate.dense.weight", "intencao.weight", "entidades.weight"]]
    assert (grandes["tipo_antes"] == "float32").all()
    assert list(grandes["tipo_depois"]) == ["uint8", "uint8", "int8", "int8", "int8", "int8"]   # as tabelas sem sinal, as matrizes com sinal
    assert (grandes["bytes_depois"] * 4 == grandes["bytes_antes"]).all()
    for em_32_bits in (f"{camada}.attention.self.query.bias", f"{camada}.output.LayerNorm.weight", "encoder.embeddings.LayerNorm.bias"):
        assert pesos.loc[em_32_bits, "tipo_depois"] == "float32"   # vieses e normalizações ficam em 32 bits


@pytest.mark.parametrize("forma", ["dinamica", "estatica"])
def test_pesos_podados_continuam_em_zero_e_o_arquivo_compactado_encolhe(tmp_path, forma):
    sistema = sistema_medio()
    poda_encoder.aplicar(sistema.modelo, poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.8))   # 80% das matrizes do corpo em zero
    fp32 = exportacao_encoder.exportar(sistema, tmp_path / "fp32")
    int8 = quantizacao.quantizar_encoder(fp32, tmp_path / forma, forma, TEXTOS)

    zeros = quantizacao.zeros_conservados(fp32, int8)
    assert zeros["conservados"] and zeros["zeros_perdidos"] == 0
    assert zeros["zeros_depois"] >= zeros["zeros_antes"] > 0
    depois = exportacao.pesos_da_rede(exportacao.ler_rede(int8))
    for nome in poda_encoder.nomes_podaveis(sistema.modelo):   # matriz por matriz, com o nome que ela tem no PyTorch
        assert depois.loc[nome, "tipo"] == "int8" and depois.loc[nome, "zeros"] >= round(0.8 * depois.loc[nome, "numeros"])

    compactado = exportacao_encoder.compactar(int8, tmp_path / "compactado")
    assert set(exportacao_encoder.bytes_por_arquivo(compactado)) == {exportacao_encoder.REDE_COMPACTADA, exportacao_encoder.TOKENIZADOR_COMPACTADO, exportacao_encoder.FICHA}
    assert exportacao_encoder.bytes_do_sistema(compactado) < 0.8 * exportacao_encoder.bytes_do_sistema(int8)
    assert SistemaEncoderOnnx.carregar(compactado).prever(TEXTOS + DIFICEIS) == SistemaEncoderOnnx.carregar(int8).prever(TEXTOS + DIFICEIS)


def test_quantizar_rede_e_a_mesma_chamada_de_quantizar(tmp_path):
    """`quantizar_rede` (o núcleo usado pelo encoder) repete as opções de `quantizar` (Etapa 8): na rede do zero, as duas dão os mesmos pesos."""
    from onnx import numpy_helper

    from test_quantizacao import ENUNCIADOS as DO_DIET
    from test_quantizacao import novo as diet_novo

    tabela = pd.DataFrame({"intent": i, "utt": texto, "tokens": texto.split(), "tags": tags.split()} for i, texto, tags in DO_DIET)
    fp32 = exportacao.exportar(diet_novo(tabela), tmp_path / "fp32")
    for forma in quantizacao.FORMAS:
        pela_etapa_8 = quantizacao.quantizar(fp32, tmp_path / forma, forma, TEXTOS)
        leitor = quantizacao._Leitor(exportacao.SistemaOnnx.carregar(fp32), TEXTOS) if forma in quantizacao.PRECISA_DE_CALIBRACAO else None
        quantizacao.quantizar_rede(exportacao.ler_rede(fp32), tmp_path / f"{forma}.onnx", forma, leitor)
        tensores = lambda rede: {t.name: numpy_helper.to_array(t) for t in rede.graph.initializer}
        a, b = tensores(exportacao.ler_rede(pela_etapa_8)), tensores(exportacao.onnx.load(str(tmp_path / f"{forma}.onnx")))
        assert a.keys() == b.keys() and all(np.array_equal(a[nome], b[nome]) for nome in a)


# --------------------------------------------------------------------------
# A execução da etapa e a regra de decisão
# --------------------------------------------------------------------------


@pytest.mark.parametrize("guardar_fp32", [True, False])
def test_execucao_grava_tudo_o_que_o_notebook_le(pasta_do_treinado, tmp_path, guardar_fp32):
    d = dados()
    medir = {"dev": d, "test": d.iloc[:5]}
    medido = etapa10c.quantizar_modelo(pasta_do_treinado, tmp_path / "saida", medir, TEXTOS, threads=1, guardar_fp32=guardar_fp32)

    sistema = SistemaEncoder.carregar(pasta_do_treinado)
    assert medido["parametros"] == sistema.parametros and medido["pedidos"] == {"dev": 8, "test": 5} and medido["camadas"] == 3
    assert medido["formas"]["fp32"]["iguais_ao_pytorch_dev"] == 8 and medido["formas"]["fp32"]["iguais_ao_pytorch_test"] == 5
    original = medido["formas"][etapa10c.REFERENCIA]
    assert original["bytes"] == original["rede"] + original["tokenizador_e_configuracao"] > 0
    for forma in etapa10c.MEDIDAS:
        for particao in medir:
            assert (tmp_path / "saida" / f"previsoes_{forma}_{particao}.parquet").exists() == (particao in etapa10c.particoes_da_forma(forma))
    for forma in quantizacao.FORMAS:
        assert medido["formas"][forma]["conservados"] and medido["formas"][forma]["bytes"] > 0
        assert (tmp_path / "saida" / "sistemas" / forma).exists() == (forma in etapa10c.CANDIDATAS)   # das formas de diagnóstico ficam só as respostas
    for forma in etapa10c.CANDIDATAS:
        assert medido["formas"][forma]["compactado_responde_igual"] and 0 < medido["formas"][forma]["bytes_compactado"]
        assert len(SistemaEncoderOnnx.carregar(tmp_path / "saida" / "sistemas" / forma).prever(TEXTOS)) == len(TEXTOS)
        assert not (tmp_path / "saida" / "sistemas" / f"{forma}_compactado").exists()
    assert (tmp_path / "saida" / "sistemas" / "fp32").exists() == guardar_fp32


def test_a_regra_e_a_da_etapa_8():
    assert etapa10c.escolher_forma is etapa8.escolher_forma
    assert etapa10c.LIMITE_DO_PLANO == 0.01 and etapa10c.CRITERIO_DO_PLANO == "slot_f1" and etapa10c.CRITERIO_DE_DESEMPATE == "exact_match"
    assert etapa10c.CANDIDATAS == ("dinamica", "estatica") and quantizacao.CALIBRACAO == 512 and etapa10c.RODADAS == 5
    assert etapa10c.particoes_da_forma("dinamica") == ("dev", "test") and etapa10c.particoes_da_forma("so_tabela") == ("dev",)


def test_le_os_modelos_com_e_sem_a_decisao_da_poda():
    pastas = lambda raiz: {str(s): f"{raiz}/semente_{s}" for s in (0, 1, 2)}
    aluno = lambda sigla, variante, segue=True: {"nome": f"Aluno {sigla} (rótulos)", "variante": variante, "segue": segue,
                                                 "pastas_por_semente": pastas(f"entregas/10a_encoder/modelos/{variante}") if segue else None}
    da_10a = {"principal": "A", "seguem": ["A", "B2"], "alunos": {"A": aluno("A", "aluno_a"), "B1": aluno("B1", "aluno_b1", False), "B2": aluno("B2", "aluno_b2_t2")},
              "referencias": {"teto": {"pastas_por_semente": pastas("entregas/10a_encoder/modelos/teto")}}}
    partida = etapa10c.modelos_de_partida(da_10a)
    assert list(partida) == ["aluno_a", "teto", "aluno_b2_t2"] and partida["aluno_a"]["principal"] and not partida["teto"]["principal"]

    # enquanto a Etapa 10b não entrega a decisão, só existem os modelos sem poda
    sem_decisao = etapa10c.fichas_dos_modelos(partida, None)
    assert list(sem_decisao) == [("aluno_a", "sem_poda"), ("teto", "sem_poda"), ("aluno_b2_t2", "sem_poda")]
    assert not any(ficha["final"] for ficha in sem_decisao.values())
    assert sem_decisao[("teto", "sem_poda")]["pastas"][2] == etapa10c.RAIZ / "entregas/10a_encoder/modelos/teto/semente_2"

    # a decisão da 10b, no formato da Etapa 7
    podada = lambda m, v, papel, nome: {"variante": v, "papel": papel, "nome": nome, "familia": "x", "pastas_por_semente": pastas(f"entregas/10b_encoder_poda/modelos/{m}/{v}")}
    do_modelo = lambda m: {"sem_poda": {"pastas_por_semente": pastas(f"entregas/10a_encoder/modelos/{m}")}, "principal": "vocabulario_treino_camadas_6", "secundaria": "magnitude_70",
                           "seguem": [podada(m, "vocabulario_treino_camadas_6", "principal", "Vocabulário do treino e 6 das 12 camadas"), podada(m, "magnitude_70", "secundaria", "Magnitude, 70% de uma vez")]}
    da_10b = {"principal": {"modelo": "aluno_a", "variante": "vocabulario_treino_camadas_6"},
              "modelos": {"aluno_a": do_modelo("aluno_a"), "aluno_b2_t2": do_modelo("aluno_b2_t2"), "teto": {**do_modelo("teto"), "secundaria": None, "seguem": do_modelo("teto")["seguem"][:1]}}}
    fichas = etapa10c.fichas_dos_modelos(partida, da_10b)
    assert list(fichas) == [(m, v) for m, variantes in (("aluno_a", 3), ("teto", 2), ("aluno_b2_t2", 3)) for v in ("sem_poda", "vocabulario_treino_camadas_6", "magnitude_70")[:variantes]]
    assert [chave for chave, ficha in fichas.items() if ficha["final"]] == [("aluno_a", "vocabulario_treino_camadas_6")]
    final = fichas[("aluno_a", "vocabulario_treino_camadas_6")]
    assert final["papel"] == "principal" and final["curto"] == "Aluno A, vocabulário do treino e 6 camadas"
    assert final["nome"] == "Aluno A (rótulos), vocabulário do treino e 6 das 12 camadas"
    assert final["pastas"][0] == etapa10c.RAIZ / "entregas/10b_encoder_poda/modelos/aluno_a/vocabulario_treino_camadas_6/semente_0"
    assert fichas[("aluno_b2_t2", "magnitude_70")]["papel"] == "secundaria" and fichas[("aluno_b2_t2", "magnitude_70")]["curto"] == "Aluno B2, magnitude 70%"
    # se a 10b não aprovar nenhuma poda para o aluno principal, o final é o modelo sem poda
    sem_principal = etapa10c.fichas_dos_modelos(partida, {**da_10b, "principal": {"modelo": "aluno_a", "variante": None}})
    assert [chave for chave, ficha in sem_principal.items() if ficha["final"]] == [("aluno_a", "sem_poda")]

    assert [etapa10c.rotulo_curto(v) for v in ("vocabulario_alfabeto", "camadas_8", "gradual_90")] == ["vocabulário latino", "8 camadas", "gradual 90%"]


def test_sistema_exportado_roda_sem_pytorch(pasta_do_treinado, tmp_path):
    """Carregar e consultar o encoder exportado, num processo novo, não traz o PyTorch (nem `transformers`, pandas ou o pacote onnx)."""
    textos = TEXTOS[:5]
    pasta = quantizacao.quantizar_encoder(exportacao_encoder.exportar(SistemaEncoder.carregar(pasta_do_treinado), tmp_path / "fp32"), tmp_path / "dinamica", "dinamica")
    esperado = [[p.intent, [list(s) for s in p.slots]] for p in SistemaEncoderOnnx.carregar(pasta).prever(textos)]
    programa = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from especialista.compressao.sistema_encoder_onnx import SistemaEncoderOnnx\n"
        "previsoes = SistemaEncoderOnnx.carregar(Path(sys.argv[1])).prever(json.loads(sys.argv[2]))\n"
        "pesados = [m for m in ('torch', 'transformers', 'pandas', 'pyarrow', 'onnx') if m in sys.modules]\n"
        "print(json.dumps({'pesados': pesados, 'previsoes': [[p.intent, [list(s) for s in p.slots]] for p in previsoes]}))\n"
    )
    saida = subprocess.run([sys.executable, "-c", programa, str(pasta), json.dumps(textos)], capture_output=True, text=True, check=True)
    resultado = json.loads(saida.stdout)
    assert resultado["pesados"] == []
    assert resultado["previsoes"] == esperado
