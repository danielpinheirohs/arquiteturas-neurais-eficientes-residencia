"""Desenho da arquitetura, com a forma dos dados em cada camada para um enunciado de exemplo."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from especialista import visual

_NEUTRO = "#f0efec"


def _caixa(ax, x, y, largura, altura, letra, titulo, detalhe, cor):
    ax.add_patch(FancyBboxPatch(
        (x, y), largura, altura, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=cor, edgecolor=visual.EIXO, linewidth=0.8,
    ))
    cabecalho = f"{letra} · {titulo}" if letra else titulo
    ax.text(x + 0.18, y + altura - 0.27, cabecalho, fontsize=10, fontweight="bold", color=visual.TINTA, va="center")
    ax.text(x + 0.18, y + 0.27, detalhe, fontsize=8.5, color=visual.TINTA_SECUNDARIA, va="center")


def _seta(ax, origem, destino):
    ax.add_patch(FancyArrowPatch(origem, destino, arrowstyle="-|>", mutation_scale=10,
                                 color=visual.TINTA_SUAVE, linewidth=1, shrinkA=0, shrinkB=0))


def _forma(ax, x, y, texto):
    ax.text(x, y, texto, fontsize=9, family="monospace", color=visual.TINTA_SECUNDARIA, va="center")


def desenhar(formas: dict[str, str], exemplo: str) -> Figure:
    """`formas` dá o texto da coluna da direita para cada camada: chaves texto, A, B, C, D, E, F, G."""
    base = 2.3  # altura do eixo abaixo das cabeças; a figura é cortada aqui
    fig, ax = plt.subplots(figsize=(9, 9 * (11.6 - base) / 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(base, 11.6)
    ax.axis("off")

    altura, largura, x = 0.82, 6.6, 0.3
    centro = x + largura / 2
    coluna = [
        ("texto", "", "Enunciado", f"“{exemplo}”", _NEUTRO),
        ("A", "A", "Features", "palavra inteira + n-gramas de 1 a 5 caracteres", visual.AZUL_CLARO),
        ("B", "B", "Tabela de features", "soma os vetores dos features de cada palavra", visual.AZUL_CLARO),
        ("C", "C", "Projeção e CLS", "camada linear; CLS = média das palavras, ao final", visual.AZUL_CLARO),
        ("D", "D", "Encoder Transformer × 2", "atenção com posição relativa + rede densa", visual.AZUL_CLARO),
    ]
    y = 10.5
    for i, (chave, letra, titulo, detalhe, cor) in enumerate(coluna):
        _caixa(ax, x, y, largura, altura, letra, titulo, detalhe, cor)
        _forma(ax, x + largura + 0.35, y + altura / 2, formas[chave])
        if i < len(coluna) - 1:
            _seta(ax, (centro, y), (centro, y - 0.48))
        y -= altura + 0.48
    base_do_encoder = y + altura + 0.48

    # as três cabeças, lado a lado
    y_cabecas = base_do_encoder - 2.0
    cabecas = [
        ("E", "Intenção", "CLS → 20 dimensões,\nproduto com cada intenção", visual.AZUL_CLARO),
        ("F", "Entidades", "pontuação por tag,\ndepois CRF (Viterbi)", visual.AZUL_CLARO),
        ("G", "Máscara", "só no treino: adivinhar\na palavra escondida", _NEUTRO),
    ]
    largura_cabeca, vao = 3.6, 0.3
    for i, (letra, titulo, detalhe, cor) in enumerate(cabecas):
        xc = x + i * (largura_cabeca + vao)
        ax.add_patch(FancyBboxPatch(
            (xc, y_cabecas), largura_cabeca, 1.3, boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=cor, edgecolor=visual.EIXO, linewidth=0.8,
        ))
        ax.text(xc + 0.18, y_cabecas + 1.03, f"{letra} · {titulo}", fontsize=10, fontweight="bold",
                color=visual.TINTA, va="center")
        ax.text(xc + 0.18, y_cabecas + 0.5, detalhe, fontsize=8.5, color=visual.TINTA_SECUNDARIA,
                va="center", linespacing=1.4)
        ax.text(xc + 0.18, y_cabecas - 0.3, formas[letra], fontsize=9, family="monospace",
                color=visual.TINTA_SECUNDARIA, va="center")
        _seta(ax, (centro, base_do_encoder), (xc + largura_cabeca / 2, y_cabecas + 1.3))

    ax.text(x, y_cabecas - 1.0, "Perda do treino  =  perda de intenção  +  perda de entidades  +  perda de máscara",
            fontsize=9.5, color=visual.TINTA, va="center")
    ax.text(x + largura + 0.35, 11.42, "forma dos dados", fontsize=8.5, color=visual.TINTA_SUAVE, va="center")
    return fig
