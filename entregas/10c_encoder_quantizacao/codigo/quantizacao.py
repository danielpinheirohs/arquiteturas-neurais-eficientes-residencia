"""Quantização: os pesos da rede passam de números reais de 32 bits para inteiros de 8 bits.

A ideia (Jacob et al., arXiv 1712.05877): um número real `r` é guardado como
um inteiro `q` de 8 bits, e dois números por tensor dizem como voltar:

    r = escala · (q − ponto_zero)

Com 8 bits há 256 valores possíveis. A escala é o tamanho do degrau entre um
valor e o seguinte; quanto maior a faixa que o tensor precisa cobrir, maior o
degrau e maior o erro de arredondamento. O zero real é sempre representado
sem erro, então um peso podado continua sendo zero.

**O que é quantizado** (o recorte de Zafrir et al., Q8BERT, arXiv 1910.06188:
as camadas lineares e a tabela de embeddings):

- a tabela de features (a operação `Gather`): é onde está a maior parte dos
  parâmetros;
- toda multiplicação por uma matriz de pesos (`MatMul` com a matriz gravada no
  arquivo): a projeção, as matrizes do encoder e as das cabeças.

Ficam em 32 bits os vieses, as normalizações de camada, o softmax e os
produtos da atenção entre duas ativações, que têm poucos pesos ou nenhum, e o
CRF, que nem está no grafo.

**Duas formas de tratar as ativações** (os valores que passam pela rede):

- *dinâmica*: a faixa de cada ativação é medida em cada pedido, na hora, logo
  antes de cada multiplicação. Não precisa de dados de calibração.
- *estática*: a faixa de cada ativação é medida antes, uma vez, num conjunto
  de calibração (pedidos do treino, sem rótulo), e fica gravada no arquivo.
  Além disso, a saída de cada multiplicação também é arredondada para 8 bits.

Os pesos são quantizados do mesmo jeito nas duas, com um degrau por tensor:

- as matrizes viram inteiros com sinal (`int8`), numa faixa simétrica em volta
  do zero, de −maior peso a +maior peso;
- a tabela de features vira inteiros sem sinal (`uint8`), numa faixa que vai
  do menor ao maior peso da tabela. É como o ONNX Runtime trata uma tabela de
  onde se buscam linhas: como um valor que passa pela rede, e não como o peso
  de uma multiplicação.

Tudo é feito pelas ferramentas do próprio ONNX Runtime
(`onnxruntime.quantization`), sem treino: é quantização pós-treino.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static

from especialista.compressao import exportacao
from especialista.compressao.exportacao import SistemaOnnx

PARTES = ("tabela", "matrizes")

# Cada forma de quantizar: como trata as ativações e que partes da rede entram.
# As duas primeiras concorrem a modelo final; as outras servem para ver onde está o custo.
FORMAS: dict[str, dict] = {
    "dinamica": {"nome": "INT8, quantização dinâmica", "ativacoes": "dinamica", "partes": PARTES, "concorre": True},
    "estatica": {"nome": "INT8, quantização estática", "ativacoes": "estatica", "partes": PARTES, "concorre": True},
    "so_tabela": {"nome": "INT8 só na tabela de features", "ativacoes": "dinamica", "partes": ("tabela",), "concorre": False},
    "so_matrizes": {"nome": "INT8 só nas matrizes (dinâmica)", "ativacoes": "dinamica", "partes": ("matrizes",), "concorre": False},
}
PRECISA_DE_CALIBRACAO = tuple(forma for forma, f in FORMAS.items() if f["ativacoes"] == "estatica")
CALIBRACAO = 512   # pedidos do treino usados para medir as faixas na quantização estática


def operacoes_com_pesos(rede: onnx.ModelProto) -> dict[str, list[str]]:
    """Os nomes das operações que a quantização pode converter, por parte da rede.

    - `tabela`: cada `Gather` que busca linhas numa tabela de pesos gravada no arquivo;
    - `matrizes`: cada `MatMul` cujo segundo fator é uma matriz gravada no arquivo.

    Um `MatMul` entre duas ativações (os produtos da atenção) não entra.
    """
    gravados = {t.name for t in rede.graph.initializer if t.data_type == onnx.TensorProto.FLOAT and len(t.dims) == 2}
    return {
        "tabela": [no.name for no in rede.graph.node if no.op_type == "Gather" and no.input[0] in gravados],
        "matrizes": [no.name for no in rede.graph.node if no.op_type == "MatMul" and no.input[1] in gravados],
    }


@contextmanager
def _sem_avisos():
    """As ferramentas do ONNX Runtime avisam, a cada chamada, que o grafo não passou pelo pré-processamento delas.

    O pré-processamento junta operações e infere formas num grafo vindo de um
    exportador. O nosso já é montado à mão, operação por operação.
    """
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


class _Leitor(CalibrationDataReader):
    """Entrega os pedidos de calibração à ferramenta de quantização, um por vez, como a rede vai recebê-los."""

    def __init__(self, sistema: SistemaOnnx, textos: list[str]):
        self._entradas = iter([sistema.entradas([texto.split()]) for texto in textos])

    def get_next(self):
        return next(self._entradas, None)


def quantizar(origem: Path, destino: Path, forma: str, textos_de_calibracao: list[str] | None = None) -> Path:
    """Lê o sistema de 32 bits exportado em `origem` e grava em `destino` o mesmo sistema com a rede em INT8.

    `forma` é uma chave de `FORMAS`. A estática precisa de
    `textos_de_calibracao`: pedidos sem rótulo, que não sejam de `dev` nem do
    teste. Só a rede muda; o vocabulário de features e o CRF são copiados.
    """
    definicao = FORMAS[forma]
    estatica = definicao["ativacoes"] == "estatica"
    if estatica and not textos_de_calibracao:
        raise ValueError("A quantização estática mede as faixas das ativações num conjunto de calibração: passe `textos_de_calibracao`.")
    rede = exportacao.ler_rede(origem)
    por_parte = operacoes_com_pesos(rede)
    operacoes = [nome for parte in definicao["partes"] for nome in por_parte[parte]]

    exportacao.copiar_acompanhantes(
        origem, destino, forma=forma, nome=definicao["nome"], partes_quantizadas=list(definicao["partes"]),
        operacoes_quantizadas=len(operacoes), calibracao=len(textos_de_calibracao) if estatica else None,
    )
    (destino / exportacao.REDE_COMPACTADA).unlink(missing_ok=True)
    comum = {
        "op_types_to_quantize": ["Gather", "MatMul"],
        "nodes_to_quantize": operacoes,
        "per_channel": False,            # um degrau por tensor
        "reduce_range": False,           # os 8 bits inteiros
        "weight_type": QuantType.QInt8,  # matrizes: inteiros com sinal, faixa simétrica em volta do zero
        "extra_options": {"MatMulConstBOnly": True, "WeightSymmetric": True},
    }
    with _sem_avisos():
        if estatica:
            quantize_static(
                rede, destino / exportacao.REDE, _Leitor(SistemaOnnx.carregar(origem), textos_de_calibracao),
                quant_format=QuantFormat.QDQ,                # as faixas ficam em operações "quantiza / desquantiza" em volta de cada conta
                activation_type=QuantType.QUInt8,            # ativações: inteiros sem sinal, faixa assimétrica (Jacob et al.)
                calibrate_method=CalibrationMethod.MinMax,   # a faixa vai do menor ao maior valor visto na calibração
                **comum,
            )
        else:
            quantize_dynamic(rede, destino / exportacao.REDE, **comum)
    return destino


# --------------------------------------------------------------------------
# Conferir o que a quantização fez com os pesos
# --------------------------------------------------------------------------


def pesos_antes_e_depois(pasta_fp32: Path, pasta_int8: Path) -> pd.DataFrame:
    """Para cada tensor de pesos, o tipo, os bytes e os zeros antes e depois de quantizar."""
    antes = exportacao.pesos_da_rede(exportacao.ler_rede(pasta_fp32))
    depois = exportacao.pesos_da_rede(exportacao.ler_rede(pasta_int8))
    antes = antes.drop(columns="quantizado").rename(columns={"tipo": "tipo_antes", "bytes": "bytes_antes", "zeros": "zeros_antes"})
    return antes.join(depois[["tipo", "bytes", "zeros"]].add_suffix("_depois"), how="left")


def zeros_conservados(pasta_fp32: Path, pasta_int8: Path) -> dict[str, int | bool]:
    """Confere que todo peso que era zero em 32 bits continua zero em 8 bits.

    Importa para o modelo podado por magnitude: os zeros são a poda. Compara
    posição por posição, nos tensores que foram quantizados.
    """
    antes = {t.name: onnx.numpy_helper.to_array(t) for t in exportacao.ler_rede(pasta_fp32).graph.initializer}
    depois = {t.name: onnx.numpy_helper.to_array(t) for t in exportacao.ler_rede(pasta_int8).graph.initializer}
    zeros_antes = zeros_depois = perdidos = 0
    for nome, original in antes.items():
        quantizado = depois.get(f"{nome}_quantized")
        if quantizado is None:
            continue
        era_zero, e_zero = original == 0, quantizado == depois[f"{nome}_zero_point"]
        zeros_antes += int(era_zero.sum())
        zeros_depois += int(e_zero.sum())
        perdidos += int((era_zero & ~e_zero).sum())
    return {"zeros_antes": zeros_antes, "zeros_depois": zeros_depois, "zeros_perdidos": perdidos, "conservados": perdidos == 0}


def erro_dos_pesos(pasta_fp32: Path, pasta_int8: Path) -> pd.DataFrame:
    """Quanto cada tensor quantizado se afasta do original: o degrau e o erro de arredondamento, em relação ao tamanho dos pesos.

    `erro_relativo` é a raiz do erro quadrático médio dividida pelo desvio-padrão
    dos pesos do tensor. Um tensor com poucos pesos muito grandes tem degrau
    grande, e o erro relativo dos pesos pequenos sobe.
    """
    antes = {t.name: onnx.numpy_helper.to_array(t) for t in exportacao.ler_rede(pasta_fp32).graph.initializer}
    depois = {t.name: onnx.numpy_helper.to_array(t) for t in exportacao.ler_rede(pasta_int8).graph.initializer}
    linhas = []
    for nome, original in antes.items():
        quantizado = depois.get(f"{nome}_quantized")
        if quantizado is None:
            continue
        escala, ponto_zero = float(depois[f"{nome}_scale"]), int(depois[f"{nome}_zero_point"])
        de_volta = escala * (quantizado.astype(np.float32) - ponto_zero)
        linhas.append({
            "tensor": nome, "numeros": int(original.size), "maior_peso": float(np.abs(original).max()), "desvio_dos_pesos": float(original.std()),
            "degrau": escala, "erro_relativo": float(np.sqrt(np.mean((de_volta - original) ** 2)) / original.std()),
            "valores_distintos": int(len(np.unique(quantizado))),
        })
    return pd.DataFrame(linhas).set_index("tensor")


# --------------------------------------------------------------------------
# Etapa 10c: o mesmo recorte e as mesmas opções, para o encoder pré-treinado
# --------------------------------------------------------------------------
#
# No XLM-R (ver `exportacao_encoder.py`), as partes de `PARTES` são:
#
# - `tabela`: a tabela de subtokens (no modelo sem poda, 250.002 linhas e 69% dos
#   parâmetros) e a tabela de posições, as duas lidas por `Gather`;
# - `matrizes`: as 6 matrizes de cada camada do corpo (consulta, chave, valor e
#   saída da atenção, e as duas da rede densa) e as das duas cabeças.
#
# Ficam em 32 bits, como na rede do zero: os vieses, as normalizações de camada,
# o softmax, a GELU, as somas e os dois produtos da atenção entre ativações.

NOMES_NO_ENCODER = {"so_tabela": "INT8 só nas tabelas de entrada"}   # no encoder não há "tabela de features"


@contextmanager
def _temporarios_em(pasta: Path | None):
    """Manda para `pasta` os arquivos temporários das ferramentas do ONNX Runtime.

    A quantização estática grava uma cópia da rede com as medidas de faixa, e
    a rede do XLM-R tem mais de 1 GB. A pasta temporária do sistema pode ficar
    na memória (é o caso do `/tmp` desta máquina); a pasta da execução fica no disco.
    """
    anterior = tempfile.tempdir
    if pasta is not None:
        pasta.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(pasta)
    try:
        yield
    finally:
        tempfile.tempdir = anterior


def quantizar_rede(
    rede: onnx.ModelProto | Path, arquivo: Path, forma: str, leitor: CalibrationDataReader | None = None, temporarios: Path | None = None
) -> list[str]:
    """Quantiza uma rede exportada e grava o resultado em `arquivo`. Devolve os nomes das operações convertidas.

    É a mesma chamada de `quantizar`, com as mesmas opções (um teste confere
    que as duas dão o mesmo arquivo), sem saber de que sistema a rede faz
    parte. `rede` é o grafo ou o caminho do arquivo `.onnx`; passar o caminho
    poupa memória numa rede grande. Na estática, `leitor` entrega os pedidos
    de calibração.
    """
    definicao = FORMAS[forma]
    estatica = definicao["ativacoes"] == "estatica"
    if estatica and leitor is None:
        raise ValueError("A quantização estática mede as faixas das ativações num conjunto de calibração: passe o `leitor`.")
    grafo = rede if isinstance(rede, onnx.ModelProto) else onnx.load(str(rede))
    por_parte = operacoes_com_pesos(grafo)
    del grafo
    operacoes = [nome for parte in definicao["partes"] for nome in por_parte[parte]]
    comum = {
        "op_types_to_quantize": ["Gather", "MatMul"],
        "nodes_to_quantize": operacoes,
        "per_channel": False,            # um degrau por tensor
        "reduce_range": False,           # os 8 bits inteiros
        "weight_type": QuantType.QInt8,  # matrizes: inteiros com sinal, faixa simétrica em volta do zero
        "extra_options": {"MatMulConstBOnly": True, "WeightSymmetric": True},
    }
    with _sem_avisos(), _temporarios_em(temporarios):
        if estatica:
            quantize_static(
                rede, arquivo, leitor,
                quant_format=QuantFormat.QDQ,                # as faixas ficam em operações "quantiza / desquantiza" em volta de cada conta
                activation_type=QuantType.QUInt8,            # ativações: inteiros sem sinal, faixa assimétrica (Jacob et al.)
                calibrate_method=CalibrationMethod.MinMax,   # a faixa vai do menor ao maior valor visto na calibração
                **comum,
            )
        else:
            quantize_dynamic(rede, arquivo, **comum)
    return operacoes


def quantizar_encoder(origem: Path, destino: Path, forma: str, textos_de_calibracao: list[str] | None = None) -> Path:
    """Lê o encoder de 32 bits exportado em `origem` e grava em `destino` o mesmo sistema com a rede em INT8.

    O par de `quantizar` para o sistema de `exportacao_encoder.py`: as mesmas
    formas, as mesmas opções. Só a rede muda; o tokenizador é copiado.
    """
    from especialista.compressao import exportacao_encoder
    from especialista.compressao.sistema_encoder_onnx import Alinhador

    definicao = FORMAS[forma]
    estatica = definicao["ativacoes"] == "estatica"
    if estatica and not textos_de_calibracao:
        raise ValueError("A quantização estática mede as faixas das ativações num conjunto de calibração: passe `textos_de_calibracao`.")
    leitor = _Leitor(Alinhador.carregar(origem), textos_de_calibracao) if estatica else None
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = exportacao.arquivo_da_rede(origem)
    rede = exportacao.ler_rede(origem) if arquivo.suffix == ".gz" else arquivo
    temporarios = destino / "temporarios"
    try:
        operacoes = quantizar_rede(rede, destino / exportacao.REDE, forma, leitor, temporarios)
    finally:
        shutil.rmtree(temporarios, ignore_errors=True)
    exportacao_encoder.copiar_acompanhantes(
        origem, destino, forma=forma, nome=NOMES_NO_ENCODER.get(forma, definicao["nome"]), partes_quantizadas=list(definicao["partes"]),
        operacoes_quantizadas=len(operacoes), calibracao=len(textos_de_calibracao) if estatica else None,
    )
    (destino / exportacao.REDE_COMPACTADA).unlink(missing_ok=True)
    return destino
