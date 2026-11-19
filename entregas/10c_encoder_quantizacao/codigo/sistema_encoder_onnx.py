"""O encoder pré-treinado exportado, pronto para responder: roda sem PyTorch (Etapa 10c).

É o par de `sistema_onnx.py` (o sistema exportado da rede do zero) para o
XLM-R com as duas cabeças. Um pedido passa por três trechos:

    A. texto -> subtokens                 pacote `tokenizers` (o mesmo tokenizador do modelo, lido de `tokenizador.json`)
    B a D. subtokens -> pontuações        ONNX Runtime (`rede.onnx`): tabela de subtokens, corpo do Transformer e as duas cabeças
    montar a resposta                     Python (`Previsao`)

Não há CRF nem Viterbi: como em Chen et al. (JointBERT, arXiv 1902.10909), a
tag de cada palavra é a mais pontuada no **primeiro subtoken** da palavra. A
escolha desse subtoken é feita aqui, do mesmo jeito que em
`modelos/encoder/subtokens.py`: uma palavra que o tokenizador apaga, ou que
ficou além do limite de subtokens, não tem primeiro subtoken e recebe `O`.

Este módulo só importa numpy, ONNX Runtime e `tokenizers`. Nada do que ele
importa do projeto carrega PyTorch, `transformers`, pandas ou o pacote
`onnx`; um teste confere isso num processo à parte.

    from especialista.compressao.sistema_encoder_onnx import SistemaEncoderOnnx
    SistemaEncoderOnnx.carregar(pasta).prever(["acorda-me às nove da manhã"])
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from especialista.avaliacao.previsao import Previsao
from especialista.compressao.sistema_onnx import FICHA, REDE, REDE_COMPACTADA, abrir_sessao, arquivo_da_rede

TOKENIZADOR = "tokenizador.json"                 # o tokenizador inteiro: as peças, as pontuações e as regras
TOKENIZADOR_COMPACTADO = "tokenizador.json.gz"   # o mesmo arquivo, passado por um compressor comum
ARQUIVOS = (FICHA,)                              # o que acompanha a rede e o tokenizador na pasta de um sistema exportado

__all__ = ["REDE", "REDE_COMPACTADA", "TOKENIZADOR", "TOKENIZADOR_COMPACTADO", "FICHA", "ARQUIVOS", "Alinhador", "SistemaEncoderOnnx",
           "arquivo_da_rede", "arquivo_do_tokenizador"]


def arquivo_do_tokenizador(pasta: Path) -> Path:
    """O arquivo do tokenizador que está na pasta: o comum ou, se só houver ele, o compactado."""
    return pasta / TOKENIZADOR if (pasta / TOKENIZADOR).exists() else pasta / TOKENIZADOR_COMPACTADO


class Alinhador:
    """Trecho A: do texto para o que a rede recebe. É a conta de `modelos.encoder.subtokens.Alinhador`, em numpy.

    As palavras são as do projeto (`texto.split()`), entregues já separadas
    ao tokenizador. A rede recebe três grades de inteiros:

    - `ids` (lote, subtokens): os subtokens, com o token de início e o de
      fim; onde sobra espaço, o id de preenchimento;
    - `atencao` (lote, subtokens): 1 nos subtokens de verdade, 0 no preenchimento;
    - `primeiros` (lote, palavras): a posição, em `ids`, do primeiro subtoken
      de cada palavra (0 onde a palavra não tem subtoken).
    """

    def __init__(self, tokenizador: Tokenizer, max_subtokens: int, id_do_preenchimento: int):
        tokenizador.no_padding()                                   # o preenchimento é feito aqui, em numpy
        tokenizador.enable_truncation(max_length=max_subtokens)    # o mesmo limite do treino, contando o início e o fim
        self.tokenizador = tokenizador
        self.max_subtokens = max_subtokens
        self.id_do_preenchimento = id_do_preenchimento

    @classmethod
    def carregar(cls, pasta: Path) -> "Alinhador":
        ficha = json.loads((pasta / FICHA).read_text(encoding="utf-8"))
        arquivo = arquivo_do_tokenizador(pasta)
        conteudo = gzip.decompress(arquivo.read_bytes()) if arquivo.suffix == ".gz" else arquivo.read_bytes()
        return cls(Tokenizer.from_str(conteudo.decode("utf-8")), ficha["max_subtokens"], ficha["id_do_preenchimento"])

    def alinhar(self, enunciados: list[list[str]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """As três grades da rede e `real` (lote, palavras): a palavra existe e tem subtoken."""
        codificados = [self.tokenizador.encode(palavras, is_pretokenized=True) for palavras in enunciados]
        subtokens = max(len(c.ids) for c in codificados)
        palavras = max(1, max(len(p) for p in enunciados))
        ids = np.full((len(enunciados), subtokens), self.id_do_preenchimento, dtype=np.int64)
        atencao = np.zeros((len(enunciados), subtokens), dtype=np.int64)
        primeiros = np.zeros((len(enunciados), palavras), dtype=np.int64)
        real = np.zeros((len(enunciados), palavras), dtype=bool)
        for b, codificado in enumerate(codificados):
            ids[b, : len(codificado.ids)] = codificado.ids
            atencao[b, : len(codificado.ids)] = 1
            # `word_ids` diz, para cada subtoken, de que palavra ele veio (None nos tokens especiais)
            for posicao, palavra in enumerate(codificado.word_ids):
                if palavra is not None and not real[b, palavra]:
                    primeiros[b, palavra], real[b, palavra] = posicao, True
        return {"ids": ids, "atencao": atencao, "primeiros": primeiros}, real

    def entradas(self, enunciados: list[list[str]]) -> dict[str, np.ndarray]:
        """Só o que a rede recebe. É o que a calibração da quantização estática lê."""
        return self.alinhar(enunciados)[0]


class SistemaEncoderOnnx:
    """O encoder exportado no contrato da régua: recebe textos, devolve `Previsao`.

    **Um pedido por vez.** O padrão é `tamanho_do_lote = 1`, como no
    protocolo de latência. Na quantização dinâmica a faixa de valores de cada
    camada é medida em cada passada; com vários pedidos juntos, a resposta de
    um dependeria de quem mais está no lote. A rede aceita lotes (um teste
    confere), e em 32 bits a resposta é a mesma de um jeito ou de outro.
    """

    def __init__(self, sessao: ort.InferenceSession, alinhador: Alinhador, ficha: dict, tamanho_do_lote: int = 1):
        self.sessao = sessao
        self.alinhador = alinhador
        self.ficha = ficha
        self.intents: list[str] = ficha["intents"]
        self.tags: list[str] = ficha["tags"]
        self.tamanho_do_lote = tamanho_do_lote
        self.nome = f"XLM-R base, {ficha['nome']}"

    @classmethod
    def carregar(cls, pasta: Path, threads: int = 1, tamanho_do_lote: int = 1) -> "SistemaEncoderOnnx":
        arquivo = arquivo_da_rede(pasta)
        rede = gzip.decompress(arquivo.read_bytes()) if arquivo.suffix == ".gz" else arquivo
        ficha = json.loads((pasta / FICHA).read_text(encoding="utf-8"))
        return cls(abrir_sessao(rede, threads), Alinhador.carregar(pasta), ficha, tamanho_do_lote)

    @property
    def parametros(self) -> int:
        return self.ficha["parametros"]

    def entradas(self, enunciados: list[list[str]]) -> dict[str, np.ndarray]:
        return self.alinhador.entradas(enunciados)

    def pontuar(self, enunciados: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """A pontuação de cada intenção (B, n_intents) e de cada tag por palavra (B, L, n_tags)."""
        intencoes, emissoes = self.sessao.run(["intencoes", "emissoes"], self.entradas(enunciados))
        return intencoes, emissoes

    def montar(self, intencoes: np.ndarray, emissoes: np.ndarray, real: np.ndarray, comprimentos: list[int]) -> list[Previsao]:
        """Das pontuações às respostas: a intenção e a tag mais pontuadas; palavra sem subtoken fica fora de qualquer entidade."""
        melhores = emissoes.argmax(axis=-1)
        return [
            Previsao.de_tags(self.intents[i], [self.tags[t] if tem else "O" for t, tem in zip(linha[:quantas].tolist(), existe[:quantas].tolist())])
            for i, linha, existe, quantas in zip(intencoes.argmax(axis=1).tolist(), melhores, real, comprimentos)
        ]

    def prever(self, textos: list[str]) -> list[Previsao]:
        previsoes = []
        for inicio in range(0, len(textos), self.tamanho_do_lote):
            enunciados = [texto.split() for texto in textos[inicio : inicio + self.tamanho_do_lote]]
            entradas, real = self.alinhador.alinhar(enunciados)
            intencoes, emissoes = self.sessao.run(["intencoes", "emissoes"], entradas)
            previsoes.extend(self.montar(intencoes, emissoes, real, [len(palavras) for palavras in enunciados]))
        return previsoes
