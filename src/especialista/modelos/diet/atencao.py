"""Camada D: o encoder Transformer, com atenção de posição relativa.

Referências: Vaswani et al., "Attention Is All You Need" (arXiv 1706.03762),
para o bloco; Shaw et al., "Self-Attention with Relative Position
Representations" (arXiv 1803.02155), para a posição.

Tudo é feito de `nn.Linear` à vista, sem `nn.MultiheadAttention`: assim cada
matriz de pesos tem nome próprio, o que facilita podar e quantizar depois.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class AtencaoRelativa(nn.Module):
    """Autoatenção com várias cabeças em que a posição entra como distância entre palavras.

    Para a palavra i olhando a palavra j (Shaw et al.):

        pontuação(i, j) = q_i · (k_j + a[j - i]) / sqrt(d_cabeça)

    `a` é uma tabela aprendida com um vetor por distância, de -distancia_max
    a +distancia_max. Distâncias maiores usam o vetor do extremo. Como só a
    distância importa, a rede não depende do comprimento visto no treino.
    """

    def __init__(self, d_modelo: int, n_cabecas: int, distancia_max: int):
        super().__init__()
        assert d_modelo % n_cabecas == 0
        self.n_cabecas = n_cabecas
        self.d_cabeca = d_modelo // n_cabecas
        self.distancia_max = distancia_max
        self.consulta = nn.Linear(d_modelo, d_modelo)
        self.chave = nn.Linear(d_modelo, d_modelo)
        self.valor = nn.Linear(d_modelo, d_modelo)
        self.saida = nn.Linear(d_modelo, d_modelo)
        # um vetor por distância, compartilhado entre as cabeças (como em Shaw et al.)
        self.posicao = nn.Parameter(torch.empty(2 * distancia_max + 1, self.d_cabeca))
        nn.init.normal_(self.posicao, std=0.02)

    def pesos(self, x: torch.Tensor, valido: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devolve os pesos de atenção (B, cabeças, L, L) e os valores (B, cabeças, L, d_cabeça)."""
        lote, comprimento, _ = x.shape
        separar = lambda t: t.view(lote, comprimento, self.n_cabecas, self.d_cabeca).transpose(1, 2)
        q, k, v = separar(self.consulta(x)), separar(self.chave(x)), separar(self.valor(x))

        conteudo = q @ k.transpose(-1, -2)                                   # (B, h, L, L): q_i · k_j
        posicoes = torch.arange(comprimento, device=x.device)
        distancia = (posicoes[None, :] - posicoes[:, None]).clamp(-self.distancia_max, self.distancia_max)
        por_distancia = q @ self.posicao.T                                   # (B, h, L, 2k+1): q_i · a[d]
        indice = (distancia + self.distancia_max).expand(lote, self.n_cabecas, -1, -1)
        posicao = por_distancia.gather(-1, indice)                           # (B, h, L, L): q_i · a[j - i]

        pontuacao = (conteudo + posicao) / math.sqrt(self.d_cabeca)
        pontuacao = pontuacao.masked_fill(~valido[:, None, None, :], float("-inf"))  # ninguém olha o preenchimento
        return pontuacao.softmax(dim=-1), v

    def forward(self, x: torch.Tensor, valido: torch.Tensor) -> torch.Tensor:
        pesos, v = self.pesos(x, valido)
        lote, comprimento, d_modelo = x.shape
        return self.saida((pesos @ v).transpose(1, 2).reshape(lote, comprimento, d_modelo))


class Bloco(nn.Module):
    """Um bloco do encoder: atenção, depois uma rede densa, cada um com atalho residual.

    A normalização vem antes de cada subcamada (pré-norma), que treina de
    forma mais estável sem aquecimento da taxa de aprendizado (Xiong et al.,
    arXiv 2002.04745).
    """

    def __init__(self, d_modelo: int, n_cabecas: int, d_ffn: int, distancia_max: int, dropout: float):
        super().__init__()
        self.norma_atencao = nn.LayerNorm(d_modelo)
        self.atencao = AtencaoRelativa(d_modelo, n_cabecas, distancia_max)
        self.norma_ffn = nn.LayerNorm(d_modelo)
        self.expande = nn.Linear(d_modelo, d_ffn)
        self.contrai = nn.Linear(d_ffn, d_modelo)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valido: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.atencao(self.norma_atencao(x), valido))
        x = x + self.dropout(self.contrai(self.dropout(torch.relu(self.expande(self.norma_ffn(x))))))
        return x
