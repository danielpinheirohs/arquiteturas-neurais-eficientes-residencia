"""Piso e linha de base: o mínimo que qualquer modelo do projeto precisa superar.

Nenhum dos dois é uma rede neural. Servem para duas coisas: provar que a
régua funciona de ponta a ponta num sistema real, e dar o ponto de partida
contra o qual o ganho de cada etapa é medido.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from especialista.avaliacao.previsao import Previsao
from especialista.bio import bio_para_rotulos


class Piso:
    """Não olha o texto: responde sempre a intenção mais comum e nenhuma entidade."""

    nome = "Piso (classe majoritária)"

    def treinar(self, treino: pd.DataFrame) -> "Piso":
        self.intent = treino["intent"].mode()[0]
        return self

    def prever(self, textos: list[str]) -> list[Previsao]:
        return [Previsao(self.intent) for _ in textos]

    @property
    def parametros(self) -> int:
        return 1


class LinhaDeBase:
    """Saco de palavras, sem contexto.

    - Intenção: TF-IDF + regressão logística, o classificador de texto clássico.
    - Entidades: cada palavra recebe o rótulo que mais teve no treino; palavra
      nunca vista fica fora de entidade. Vizinhas com o mesmo rótulo formam
      uma entidade, como no código do MASSIVE.

    Hiperparâmetros fixos, sem busca: é um ponto de partida, não um concorrente.
    """

    def __init__(self, caracteres: bool = False):
        self.caracteres = caracteres
        self.nome = "Linha de base (palavras + caracteres)" if caracteres else "Linha de base (palavras)"

    def treinar(self, treino: pd.DataFrame) -> "LinhaDeBase":
        vetores = [TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), sublinear_tf=True)]
        if self.caracteres:
            # n-gramas de 2 a 5 caracteres, dentro de cada palavra
            vetores.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))
        self.intencao = make_pipeline(make_union(*vetores), LogisticRegression(C=10.0, max_iter=2000))
        self.intencao.fit(treino["utt"], treino["intent"])

        contagem: dict[str, Counter] = defaultdict(Counter)
        for tokens, tags in zip(treino["tokens"], treino["tags"]):
            for palavra, rotulo in zip(tokens, bio_para_rotulos(list(tags))):
                contagem[palavra][rotulo] += 1
        self.rotulo_da_palavra = {palavra: c.most_common(1)[0][0] for palavra, c in contagem.items()}
        return self

    def prever(self, textos: list[str]) -> list[Previsao]:
        intents = self.intencao.predict(textos)
        return [
            Previsao.de_rotulos(str(intent), [self.rotulo_da_palavra.get(p, "O") for p in texto.split()])
            for intent, texto in zip(intents, textos)
        ]

    @property
    def parametros(self) -> int:
        """Pesos da regressão logística mais uma entrada por palavra na tabela de entidades."""
        logistica = self.intencao[-1]
        return int(logistica.coef_.size + logistica.intercept_.size + len(self.rotulo_da_palavra))
