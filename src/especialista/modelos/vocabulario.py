"""O vocabulário de features da camada A, sem PyTorch.

É a parte da camada A que só mexe com texto: os n-gramas de uma palavra e a
tabela que leva cada feature ao seu id. Fica fora do pacote `diet` de
propósito: quem só quer rodar a rede exportada (`compressao/sistema_onnx.py`)
importa este módulo e não carrega o PyTorch. A montagem dos tensores de treino
continua em `diet/features.py` (`Featurizador`).

    "manhã" -> palavra:  p:manhã
               n-gramas: ^m, ma, an, nh, hã, ã$, ^ma, man, ... (até 5 caracteres)

`^` e `$` marcam começo e fim da palavra, para que "ma" no início seja
diferente de "ma" no meio (como no fastText, Bojanowski et al., arXiv 1607.04606).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

INICIO, FIM = "^", "$"


def ngramas(palavra: str, n_max: int = 5) -> list[str]:
    """Todos os n-gramas de 1 a `n_max` caracteres da palavra, com as marcas de borda."""
    marcada = INICIO + palavra + FIM
    return [
        marcada[i : i + n]
        for n in range(1, n_max + 1)
        for i in range(len(marcada) - n + 1)
        if marcada[i : i + n] not in (INICIO, FIM)  # a marca sozinha aparece em toda palavra: não informa nada
    ]


def features_do_texto(palavra: str, n_max: int = 5) -> list[str]:
    """Os nomes dos features de uma palavra: ela mesma e os seus n-gramas, sem repetição.

    Cada feature está ligado ou desligado. Um n-grama que aparece duas vezes
    na palavra (o "a" de "casa") conta uma vez só.
    """
    return list(dict.fromkeys([f"p:{palavra}", *(f"g:{g}" for g in ngramas(palavra, n_max))]))


class Vocabulario:
    """Conhece o vocabulário de features: de uma palavra para os ids dos features que ela liga."""

    def __init__(self, vocabulario: dict[str, int], n_max: int = 5):
        self.vocabulario = vocabulario
        self.n_max = n_max
        self._memoria: dict[str, list[int]] = {}

    @classmethod
    def ajustar(cls, enunciados: Iterable[list[str]], n_max: int = 5, contagem_minima: int = 2) -> "Vocabulario":
        """Monta o vocabulário com os features vistos pelo menos `contagem_minima` vezes no treino.

        Um feature visto uma vez só não generaliza: serve apenas para decorar
        aquele exemplo.
        """
        contagem = Counter(f for tokens in enunciados for palavra in tokens for f in features_do_texto(palavra, n_max))
        mantidos = sorted(f for f, c in contagem.items() if c >= contagem_minima)
        return cls({f: i for i, f in enumerate(mantidos)}, n_max)

    def __len__(self) -> int:
        return len(self.vocabulario)

    def ids(self, palavra: str) -> list[int]:
        """Ids dos features conhecidos da palavra. Palavra nova fica só com os n-gramas que já foram vistos."""
        achados = self._memoria.get(palavra)
        if achados is None:
            achados = [self.vocabulario[f] for f in features_do_texto(palavra, self.n_max) if f in self.vocabulario]
            self._memoria[palavra] = achados
        return achados

    def salvar(self, caminho: Path) -> None:
        caminho.write_text(
            json.dumps({"n_max": self.n_max, "features": list(self.vocabulario)}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def carregar(cls, caminho: Path) -> "Vocabulario":
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return cls({f: i for i, f in enumerate(dados["features"])}, dados["n_max"])
