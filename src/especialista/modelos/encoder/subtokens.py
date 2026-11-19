"""A. Do texto para os subtokens do encoder, sem perder onde começa cada palavra.

O encoder não lê palavras: lê subtokens (pedaços de palavra) do vocabulário
com que foi pré-treinado. "acorda-me" vira `▁acord`, `a`, `-`, `me`. Os
rótulos de entidade do projeto são por palavra. A ponte entre os dois é a de
Chen et al. (JointBERT, arXiv 1902.10909, seção 3.2): cada palavra é
representada pelo seu **primeiro subtoken**, e é sobre ele que a cabeça de
entidades decide a tag.

O que este arquivo garante:

- as palavras são as do projeto (`utt.split()`), entregues já separadas ao
  tokenizador, para que subtoken nenhum junte duas palavras;
- uma palavra que o tokenizador apaga (um caractere invisível, por exemplo)
  ou que ficou além do limite de subtokens não tem primeiro subtoken: fica
  marcada como não real, não entra na perda e recebe a tag `O` na previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Lote:
    """Um lote de enunciados pronto para o encoder."""

    ids: torch.Tensor           # (B, S)  subtokens, com o token de início e o de fim; preenchimento à direita
    atencao: torch.Tensor       # (B, S)  1 nos subtokens de verdade, 0 no preenchimento
    primeiros: torch.Tensor     # (B, L)  posição, em `ids`, do primeiro subtoken de cada palavra (0 onde não há)
    real: torch.Tensor          # (B, L)  a palavra existe e tem subtoken
    comprimentos: torch.Tensor  # (B,)    palavras em cada enunciado
    intents: torch.Tensor | None = None  # (B,)
    tags: torch.Tensor | None = None     # (B, L)
    logits_professor: torch.Tensor | None = None  # (B, n_intents)  só na destilação com logits

    def para(self, dispositivo) -> "Lote":
        mover = lambda t: None if t is None else t.to(dispositivo)
        return Lote(*(mover(getattr(self, campo)) for campo in self.__dataclass_fields__))


class Alinhador:
    """Transforma enunciados (listas de palavras) em `Lote`, com o tokenizador do encoder."""

    def __init__(self, tokenizador, max_subtokens: int = 128):
        self.tokenizador = tokenizador
        self.max_subtokens = max_subtokens

    def montar(
        self, enunciados: list[list[str]], intents: list[int] | None = None, tags: list[list[int]] | None = None
    ) -> Lote:
        codificado = self.tokenizador(
            [list(palavras) for palavras in enunciados], is_split_into_words=True, truncation=True,
            max_length=self.max_subtokens, padding=True, return_tensors="pt",
        )
        maior = max(1, max(len(palavras) for palavras in enunciados))
        primeiros = [[0] * maior for _ in enunciados]
        real = [[False] * maior for _ in enunciados]
        for i in range(len(enunciados)):
            # `word_ids` diz, para cada subtoken, de que palavra ele veio (None nos tokens especiais)
            for posicao, palavra in enumerate(codificado.word_ids(i)):
                if palavra is not None and not real[i][palavra]:
                    primeiros[i][palavra], real[i][palavra] = posicao, True

        lote = Lote(
            ids=codificado["input_ids"],
            atencao=codificado["attention_mask"],
            primeiros=torch.tensor(primeiros, dtype=torch.long),
            real=torch.tensor(real, dtype=torch.bool),
            comprimentos=torch.tensor([len(palavras) for palavras in enunciados], dtype=torch.long),
        )
        if intents is not None:
            lote.intents = torch.tensor(intents, dtype=torch.long)
        if tags is not None:
            lote.tags = torch.tensor([linha + [0] * (maior - len(linha)) for linha in tags], dtype=torch.long)
        return lote

    def ids_usados(self, enunciados) -> torch.Tensor:
        """Os subtokens distintos que aparecem em `enunciados`, mais os tokens especiais. Ids em ordem crescente."""
        usados = set(self.tokenizador.all_special_ids)
        enunciados = [list(palavras) for palavras in enunciados]
        for inicio in range(0, len(enunciados), 1024):
            codificado = self.tokenizador(
                enunciados[inicio : inicio + 1024], is_split_into_words=True, truncation=True, max_length=self.max_subtokens
            )
            usados.update(i for linha in codificado["input_ids"] for i in linha)
        return torch.tensor(sorted(usados), dtype=torch.long)
