"""O sistema exportado, pronto para responder: roda sem PyTorch.

É tudo o que um pedido precisa depois que a rede saiu do PyTorch (Etapa 8):

    A. texto -> ids de features          Python (`Vocabulario`)
    B a F. ids -> pontuações             ONNX Runtime (`rede.onnx`)
    F. pontuações -> tags                Viterbi em numpy
    montar a resposta                    Python (`Previsao`)

Este módulo só importa numpy e ONNX Runtime, e nada do que ele importa do
projeto carrega PyTorch, pandas ou o pacote `onnx`. É por isso que está
separado de `exportacao.py`, que constrói o grafo a partir do modelo em
PyTorch. Um teste confere, num processo à parte, que carregar e consultar um
sistema por aqui não traz o PyTorch junto.

    from especialista.compressao.sistema_onnx import SistemaOnnx
    SistemaOnnx.carregar(pasta).prever(["acorda-me às nove da manhã"])
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from especialista.avaliacao.previsao import Previsao
from especialista.modelos.vocabulario import Vocabulario

REDE = "rede.onnx"       # a rede, da tabela de features até as pontuações
REDE_COMPACTADA = "rede.onnx.gz"   # a mesma rede, passada por um compressor comum (ver `compactar`)
FEATURES = "features.json"
CRF = "crf.npz"
FICHA = "sistema.json"
ARQUIVOS = (FEATURES, CRF, FICHA)   # o que acompanha a rede na pasta de um sistema exportado


# --------------------------------------------------------------------------
# O Viterbi do CRF, em numpy
# --------------------------------------------------------------------------


def viterbi(emissoes: np.ndarray, transicao: np.ndarray, inicio: np.ndarray, fim: np.ndarray) -> list[int]:
    """A sequência de tags de maior pontuação de um pedido. É a conta de `CRF.decodificar`, para um pedido só.

    `emissoes` tem forma (palavras, tags). `transicao` [origem, destino] e
    `inicio` já trazem somada a penalidade das regras BIO.
    """
    colunas = np.arange(emissoes.shape[1])
    melhor = inicio + emissoes[0]                       # a melhor pontuação de um caminho que termina em cada tag
    de_onde = []                                        # para cada passo, a melhor tag anterior
    for emissao in emissoes[1:]:
        candidatos = melhor[:, None] + transicao        # (origem, destino)
        origem = candidatos.argmax(axis=0)
        melhor = candidatos[origem, colunas] + emissao
        de_onde.append(origem)
    tag = int((melhor + fim).argmax())
    caminho = [tag]
    for origem in reversed(de_onde):                    # volta do fim para o começo
        tag = int(origem[tag])
        caminho.append(tag)
    return caminho[::-1]


# --------------------------------------------------------------------------
# Onde está a rede
# --------------------------------------------------------------------------


def arquivo_da_rede(pasta: Path) -> Path:
    """O arquivo da rede que está na pasta: o comum ou, se só houver ele, o compactado."""
    return pasta / REDE if (pasta / REDE).exists() else pasta / REDE_COMPACTADA


# --------------------------------------------------------------------------
# O sistema exportado, no contrato da régua
# --------------------------------------------------------------------------


def abrir_sessao(rede: Path | bytes, threads: int = 1) -> ort.InferenceSession:
    """Prepara o ONNX Runtime para rodar a rede na CPU, com o número de threads do protocolo de latência."""
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = threads
    opcoes.inter_op_num_threads = 1
    opcoes.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opcoes.log_severity_level = 3   # só erros
    return ort.InferenceSession(rede if isinstance(rede, bytes) else str(rede), opcoes, providers=["CPUExecutionProvider"])


class SistemaOnnx:
    """A rede exportada no contrato da régua: recebe textos, devolve `Previsao`.

    Faz os quatro trechos de um pedido: texto para ids (Python), ids para
    pontuações (ONNX Runtime), pontuações para tags (Viterbi em numpy) e a
    montagem da resposta.

    **Um pedido por vez.** O padrão é `tamanho_do_lote = 1`: cada pedido passa
    sozinho pela rede, como no protocolo de latência e como em produção. Na
    quantização dinâmica a faixa de valores de cada camada é medida em cada
    passada; com vários pedidos juntos, a resposta de um dependeria de quem
    mais está no lote. A rede aceita lotes (um teste confere), e em 32 bits a
    resposta é a mesma de um jeito ou de outro.
    """

    def __init__(
        self,
        sessao: ort.InferenceSession,
        featurizador: Vocabulario,
        ficha: dict,
        crf: dict[str, np.ndarray],
        tamanho_do_lote: int = 1,
    ):
        self.sessao = sessao
        self.featurizador = featurizador
        self.ficha = ficha
        self.intents: list[str] = ficha["intents"]
        self.tags: list[str] = ficha["tags"]
        self.usar_crf: bool = ficha["usar_crf"]
        self.transicao, self.inicio, self.fim = crf["transicao"], crf["inicio"], crf["fim"]
        self.tamanho_do_lote = tamanho_do_lote
        self.nome = f"DIET do zero, {ficha['nome']}"
        self._linha_de_zeros = len(featurizador)   # o id que preenche as sobras

    @classmethod
    def carregar(cls, pasta: Path, threads: int = 1, tamanho_do_lote: int = 1) -> "SistemaOnnx":
        arquivo = arquivo_da_rede(pasta)
        rede = gzip.decompress(arquivo.read_bytes()) if arquivo.suffix == ".gz" else arquivo
        ficha = json.loads((pasta / FICHA).read_text(encoding="utf-8"))
        with np.load(pasta / CRF) as arquivos:
            crf = {nome: arquivos[nome] for nome in arquivos.files}
        return cls(abrir_sessao(rede, threads), Vocabulario.carregar(pasta / FEATURES), ficha, crf, tamanho_do_lote)

    @property
    def parametros(self) -> int:
        return self.ficha["parametros"]

    def entradas(self, enunciados: list[list[str]]) -> dict[str, np.ndarray]:
        """Trecho A: palavras para a grade de ids que a rede recebe, com as sobras apontando para a linha de zeros."""
        listas = [[self.featurizador.ids(palavra) for palavra in tokens] for tokens in enunciados]
        palavras = max(len(tokens) for tokens in enunciados)
        features = max((len(ids) for linha in listas for ids in linha), default=0) or 1
        grade = np.full((len(enunciados), palavras, features), self._linha_de_zeros, dtype=np.int64)
        for b, linha in enumerate(listas):
            for i, ids in enumerate(linha):
                grade[b, i, : len(ids)] = ids
        return {"ids": grade, "comprimentos": np.array([len(tokens) for tokens in enunciados], dtype=np.int64)}

    def pontuar(self, enunciados: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """Trechos A e B a F: a pontuação de cada intenção (B, n_intents) e de cada tag por palavra (B, L, n_tags)."""
        intencoes, emissoes = self.sessao.run(["intencoes", "emissoes"], self.entradas(enunciados))
        return intencoes, emissoes

    def decodificar(self, emissoes: np.ndarray, comprimentos: list[int]) -> list[list[int]]:
        """Trecho F: a melhor sequência de tags de cada pedido."""
        if not self.usar_crf:
            return [linha[:quantas].argmax(axis=1).tolist() for linha, quantas in zip(emissoes, comprimentos)]
        return [viterbi(linha[:quantas], self.transicao, self.inicio, self.fim) for linha, quantas in zip(emissoes, comprimentos)]

    def prever(self, textos: list[str]) -> list[Previsao]:
        previsoes = []
        for inicio in range(0, len(textos), self.tamanho_do_lote):
            enunciados = [texto.split() for texto in textos[inicio : inicio + self.tamanho_do_lote]]
            intencoes, emissoes = self.pontuar(enunciados)
            caminhos = self.decodificar(emissoes, [len(tokens) for tokens in enunciados])
            previsoes.extend(
                Previsao.de_tags(self.intents[i], [self.tags[t] for t in caminho])
                for i, caminho in zip(intencoes.argmax(axis=1).tolist(), caminhos)
            )
        return previsoes
