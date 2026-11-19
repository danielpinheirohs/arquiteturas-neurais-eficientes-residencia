"""Os exemplos que a LLM vê no prompt.

O cenário do projeto: a equipe tem POUCOS exemplos rotulados por humanos e
muitos pedidos sem rótulo (os registros do agente). A LLM usa os poucos
como demonstração e rotula os muitos.

- O *conjunto de demonstração* é uma amostra pequena do treino, com um
  número fixo de exemplos por intenção. É todo o rótulo humano que a LLM vê.
- Para cada pedido, entram no prompt os `k` exemplos mais parecidos desse
  conjunto (Liu et al., "What Makes Good In-Context Examples for GPT-3?",
  arXiv 2101.06804).
- Um pedido nunca é demonstração de si mesmo: ao rotular o treino, o
  próprio exemplo é excluído da busca.

Usar o treino inteiro como fonte de demonstrações daria à LLM milhares de
rótulos humanos e desmontaria o cenário: o que ela "ensinasse" ao aluno
seria, em boa parte, rótulo humano repassado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from especialista.bio import bio_para_spans

POR_INTENCAO = 5
K = 10


def montar_conjunto(treino: pd.DataFrame, por_intencao: int = POR_INTENCAO, semente: int = 0) -> pd.DataFrame:
    """Sorteia até `por_intencao` exemplos de cada intenção do treino."""
    rng = np.random.default_rng(semente)
    escolhidos = [
        rng.choice(grupo.index.to_numpy(), size=min(por_intencao, len(grupo)), replace=False)
        for _, grupo in treino.groupby("intent", sort=True)
    ]
    return treino.loc[np.sort(np.concatenate(escolhidos))].reset_index(drop=True)


def entidades_em_texto(tokens: list[str], tags: list[str]) -> list[dict]:
    """As entidades de um exemplo no formato que a LLM lê e escreve: tipo e texto."""
    return [{"type": tipo, "text": " ".join(tokens[ini:fim])} for tipo, ini, fim in bio_para_spans(list(tags))]


class Demonstracoes:
    """O conjunto de demonstração e a busca dos exemplos mais parecidos com um pedido.

    Semelhança: cosseno entre vetores TF-IDF de n-gramas de caracteres. Não
    usa nenhum modelo externo, então é igual para as duas LLMs e reproduzível.
    """

    def __init__(self, conjunto: pd.DataFrame):
        self.conjunto = conjunto.reset_index(drop=True)
        self.ids = self.conjunto["id"].to_numpy()
        self._vetorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self._matriz = self._vetorizador.fit_transform(self.conjunto["utt"])
        self._exemplos = [
            {"utt": linha.utt, "intent": linha.intent, "slots": entidades_em_texto(list(linha.tokens), list(linha.tags))}
            for linha in self.conjunto.itertuples()
        ]

    def __len__(self) -> int:
        return len(self.conjunto)

    def mais_parecidos(self, texto: str, k: int = K, excluir_id: int | None = None) -> list[dict]:
        """Os `k` exemplos mais parecidos, do menos para o mais parecido (o mais parecido fica junto do pedido)."""
        if k == 0:
            return []
        semelhanca = (self._matriz @ self._vetorizador.transform([texto]).T).toarray().ravel()
        if excluir_id is not None:
            semelhanca[self.ids == excluir_id] = -1.0
        ordem = np.argsort(-semelhanca, kind="stable")[:k]
        return [self._exemplos[i] for i in ordem[::-1]]
