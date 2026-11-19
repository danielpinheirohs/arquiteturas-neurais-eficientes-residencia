"""Camada A: do texto para ids de features.

No DIET, cada palavra entra na rede como um vetor esparso: um "1" na posição
da própria palavra e um "1" em cada n-grama de caracteres (n <= 5) que ela
contém. Aqui o vetor não é montado: guarda-se só a lista das posições
ligadas (os ids), que é a mesma informação.

    "manhã" -> palavra:  p:manhã
               n-gramas: ^m, ma, an, nh, hã, ã$, ^ma, man, ... (até 5 caracteres)

`^` e `$` marcam começo e fim da palavra, para que "ma" no início seja
diferente de "ma" no meio (como no fastText, Bojanowski et al., arXiv 1607.04606).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from especialista.modelos.vocabulario import FIM, INICIO, Vocabulario, features_do_texto, ngramas

__all__ = ["FIM", "INICIO", "Featurizador", "Lote", "Vocabulario", "features_do_texto", "ngramas"]


@dataclass
class Lote:
    """Um lote de enunciados pronto para a rede.

    Os ids de todas as palavras do lote ficam numa fila só (`features`);
    `inicios` diz onde começa a fatia de cada posição. Posições de
    preenchimento têm fatia vazia.
    """

    features: torch.Tensor      # (N,)    ids de features, em fila
    inicios: torch.Tensor       # (B*L,)  início da fatia de cada posição
    comprimentos: torch.Tensor  # (B,)    palavras em cada enunciado
    palavras: torch.Tensor      # (B, L)  um número por palavra distinta do lote (-1 no preenchimento)
    intents: torch.Tensor | None = None  # (B,)
    tags: torch.Tensor | None = None     # (B, L)
    logits_professor: torch.Tensor | None = None  # (B, n_intents)  só na destilação: a pontuação que o professor deu a cada intenção

    @property
    def forma(self) -> tuple[int, int]:
        return tuple(self.palavras.shape)

    def para(self, dispositivo) -> "Lote":
        mover = lambda t: None if t is None else t.to(dispositivo)
        return Lote(*(mover(getattr(self, campo)) for campo in self.__dataclass_fields__))


class Featurizador(Vocabulario):
    """O vocabulário de features, mais a montagem de enunciados em `Lote` (tensores do PyTorch).

    O vocabulário em si (`ajustar`, `ids`, `salvar`, `carregar`) está em
    `modelos/vocabulario.py`, que não depende do PyTorch.
    """

    def montar(
        self,
        enunciados: list[list[str]],
        intents: list[int] | None = None,
        tags: list[list[int]] | None = None,
    ) -> Lote:
        maior = max(len(tokens) for tokens in enunciados)
        features: list[int] = []
        inicios: list[int] = []
        numero_da_palavra: dict[str, int] = {}
        palavras = torch.full((len(enunciados), maior), -1, dtype=torch.long)
        for b, tokens in enumerate(enunciados):
            for i in range(maior):
                inicios.append(len(features))
                if i < len(tokens):
                    features.extend(self.ids(tokens[i]))
                    palavras[b, i] = numero_da_palavra.setdefault(tokens[i], len(numero_da_palavra))
        lote = Lote(
            features=torch.tensor(features, dtype=torch.long),
            inicios=torch.tensor(inicios, dtype=torch.long),
            comprimentos=torch.tensor([len(tokens) for tokens in enunciados], dtype=torch.long),
            palavras=palavras,
        )
        if intents is not None:
            lote.intents = torch.tensor(intents, dtype=torch.long)
        if tags is not None:
            lote.tags = torch.zeros((len(enunciados), maior), dtype=torch.long)
            for b, linha in enumerate(tags):
                lote.tags[b, : len(linha)] = torch.tensor(linha, dtype=torch.long)
        return lote
