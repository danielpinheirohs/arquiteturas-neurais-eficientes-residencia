"""Camada F: Conditional Random Field linear para as entidades.

Referências: Lafferty et al. (2001) para o CRF; Lample et al., "Neural
Architectures for Named Entity Recognition" (arXiv 1603.01360), para o uso
sobre uma rede neural, que é o que o DIET cita.

A rede dá a cada palavra uma pontuação por tag (as "emissões"). Escolher a
melhor tag palavra por palavra ignora que tags vizinhas dependem uma da
outra: `I-time` só faz sentido depois de `B-time` ou `I-time`. O CRF
acrescenta uma pontuação por par de tags vizinhas (as "transições") e passa
a pontuar a sequência inteira:

    pontuação(tags) = início[t_1] + Σ emissão[i, t_i] + Σ transição[t_i, t_i+1] + fim[t_n]

- Treino: maximizar a probabilidade da sequência certa entre todas as
  possíveis. A soma sobre todas (a "partição") sai do algoritmo forward.
- Inferência: a sequência de maior pontuação sai do algoritmo de Viterbi.

Transições impossíveis no esquema BIO recebem uma penalidade fixa, então o
modelo nunca devolve uma sequência malformada.
"""

from __future__ import annotations

import torch
from torch import nn

PROIBIDO = -1e4  # grande o bastante para nunca ser escolhida, finita para não gerar NaN


def restricoes_bio(tags: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """O que o esquema BIO proíbe: `I-x` no começo, e `I-x` depois de qualquer coisa que não seja `B-x` ou `I-x`.

    Devolve (transição proibida (T, T), início proibido (T,)).
    """
    n = len(tags)
    transicao = torch.zeros(n, n, dtype=torch.bool)
    inicio = torch.zeros(n, dtype=torch.bool)
    for j, destino in enumerate(tags):
        if destino.startswith("I-"):
            inicio[j] = True
            tipo = destino[2:]
            for i, origem in enumerate(tags):
                transicao[i, j] = origem not in (f"B-{tipo}", f"I-{tipo}")
    return transicao, inicio


class CRF(nn.Module):
    def __init__(
        self,
        n_tags: int,
        transicao_proibida: torch.Tensor | None = None,
        inicio_proibido: torch.Tensor | None = None,
    ):
        super().__init__()
        self.transicao = nn.Parameter(torch.zeros(n_tags, n_tags))  # [origem, destino]
        self.inicio = nn.Parameter(torch.zeros(n_tags))
        self.fim = nn.Parameter(torch.zeros(n_tags))
        zeros = lambda *forma: torch.zeros(*forma, dtype=torch.bool)
        transicao_proibida = zeros(n_tags, n_tags) if transicao_proibida is None else transicao_proibida
        inicio_proibido = zeros(n_tags) if inicio_proibido is None else inicio_proibido
        self.register_buffer("penalidade_transicao", transicao_proibida.float() * PROIBIDO)
        self.register_buffer("penalidade_inicio", inicio_proibido.float() * PROIBIDO)

    def _pesos(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transicao + self.penalidade_transicao, self.inicio + self.penalidade_inicio

    def pontuacao(self, emissoes: torch.Tensor, tags: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Pontuação de uma sequência de tags dada. emissoes (B, L, T), tags (B, L), real (B, L) -> (B,)."""
        transicao, inicio = self._pesos()
        lote = torch.arange(emissoes.shape[0], device=emissoes.device)
        das_tags = emissoes.gather(-1, tags[..., None]).squeeze(-1)             # emissão da tag de cada palavra
        entre_vizinhas = transicao[tags[:, :-1], tags[:, 1:]]                    # transição de cada par
        ultima = tags[lote, real.sum(1) - 1]
        return (
            inicio[tags[:, 0]]
            + (das_tags * real).sum(1)
            + (entre_vizinhas * real[:, 1:]).sum(1)
            + self.fim[ultima]
        )

    def particao(self, emissoes: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """log da soma de exp(pontuação) sobre TODAS as sequências possíveis (algoritmo forward) -> (B,)."""
        transicao, inicio = self._pesos()
        alfa = inicio + emissoes[:, 0]                                            # (B, T): caminhos que terminam em cada tag
        for i in range(1, emissoes.shape[1]):
            seguinte = torch.logsumexp(alfa[:, :, None] + transicao, dim=1) + emissoes[:, i]
            alfa = torch.where(real[:, i, None], seguinte, alfa)                  # preenchimento não avança
        return torch.logsumexp(alfa + self.fim, dim=1)

    def perda(self, emissoes: torch.Tensor, tags: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """-log P(sequência certa), por enunciado -> (B,). Sempre >= 0."""
        return self.particao(emissoes, real) - self.pontuacao(emissoes, tags, real)

    @torch.no_grad()
    def decodificar(self, emissoes: torch.Tensor, real: torch.Tensor) -> list[list[int]]:
        """A sequência de maior pontuação de cada enunciado (algoritmo de Viterbi)."""
        transicao, inicio = self._pesos()
        melhor = inicio + emissoes[:, 0]                                          # (B, T)
        de_onde = []                                                              # para cada passo, a melhor tag anterior
        for i in range(1, emissoes.shape[1]):
            candidatos = melhor[:, :, None] + transicao                           # (B, origem, destino)
            maximo, origem = candidatos.max(dim=1)
            melhor = torch.where(real[:, i, None], maximo + emissoes[:, i], melhor)
            de_onde.append(origem)
        ultima = (melhor + self.fim).argmax(dim=1)

        caminhos = []
        for b, comprimento in enumerate(real.sum(1).tolist()):
            tag = int(ultima[b])
            caminho = [tag]
            for origem in reversed(de_onde[: comprimento - 1]):                   # volta do fim para o começo
                tag = int(origem[b, tag])
                caminho.append(tag)
            caminhos.append(caminho[::-1])
        return caminhos
