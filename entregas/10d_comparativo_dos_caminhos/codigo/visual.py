"""Padrão visual do projeto: um estilo de gráfico e um jeito de mostrar BIO.

Todos os notebooks usam estas funções, para que as figuras de todas as
etapas tenham a mesma cara no relatório final.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

# Eixos com vírgula decimal e sinal de menos tipográfico, como no texto do projeto.
VIRGULA = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ",").replace("-", "−"))

# Paleta: uma cor para o dado, tons de cinza para todo o resto.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
AZUL = "#2a78d6"
AZUL_CLARO = "#cde2fb"
LARANJA = "#eb6834"  # segunda série, quando há duas


def aplicar_estilo() -> None:
    """Chame uma vez no começo do notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": TINTA,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.edgecolor": EIXO,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": TINTA_SUAVE,
            "ytick.color": TINTA_SUAVE,
            "xtick.labelcolor": TINTA_SECUNDARIA,
            "ytick.labelcolor": TINTA_SECUNDARIA,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def barras_ranqueadas(
    valores,
    titulo: str,
    eixo_x: str,
    destacar: dict[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """Barras horizontais, da maior para a menor, numa cor só.

    `valores` é uma Series do pandas (índice = nome, valor = contagem).
    `destacar` rotula só as barras que importam: {nome: texto}.
    """
    valores = valores.sort_values(ascending=True)
    poucas = len(valores) <= 12
    fig, ax = plt.subplots(figsize=(7.5, max(2.2, (0.3 if poucas else 0.2) * len(valores) + 1)))
    ax.barh(valores.index, valores.values, height=0.42 if poucas else 0.62, color=AZUL)
    ax.set_title(titulo)
    ax.set_xlabel(eixo_x)
    ax.xaxis.grid(True)
    ax.spines["bottom"].set_visible(False)
    ax.margins(y=0.01)
    ax.tick_params(axis="y", labelsize=8.5)
    for nome, texto in (destacar or {}).items():
        ax.annotate(
            texto,
            (valores[nome], nome),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=TINTA,
        )
    return fig, ax


def histograma(
    valores,
    titulo: str,
    eixo_x: str,
    marcas: dict[str, float] | None = None,
    classes: int = 40,
    passo: float | None = None,
) -> tuple[Figure, Axes]:
    """Distribuição de uma medida, com linhas de referência rotuladas.

    `marcas` é {texto: posição no eixo x}. Se a medida só assume valores
    múltiplos de `passo` (uma acurácia sobre N enunciados anda de 1/N em 1/N),
    informe-o: as classes se alinham a ele e o histograma não sai serrilhado.
    """
    valores = np.asarray(valores, dtype=float)
    limites: int | np.ndarray = classes
    if passo:
        largura = passo * max(1, int(np.ceil((valores.max() - valores.min()) / (classes * passo))))
        limites = np.arange(valores.min() - passo / 2, valores.max() + largura, largura)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hist(valores, bins=limites, color=AZUL, rwidth=0.9)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.set_title(titulo)
    ax.set_xlabel(eixo_x)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    topo = ax.get_ylim()[1] * 1.3
    ax.set_ylim(0, topo)
    for i, (texto, x) in enumerate((marcas or {}).items()):
        ax.axvline(x, color=TINTA_SECUNDARIA, linewidth=0.8, zorder=0.5)
        ax.annotate(
            texto,
            (x, topo),
            xytext=(4, -11 - 12 * (i % 2)),  # alterna a altura para rótulos vizinhos não colidirem
            textcoords="offset points",
            fontsize=8.5,
            color=TINTA_SECUNDARIA,
            bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 1.5},  # o rótulo não cruza outra linha
        )
    return fig, ax


def pontos_com_intervalo(
    series: dict[str, dict[str, tuple[float, float, float]]],
    titulo: str,
    eixo_x: str,
) -> tuple[Figure, Axes]:
    """Compara até duas séries, métrica por métrica: um ponto e o seu intervalo.

    `series` é {nome da série: {métrica: (valor, mínimo, máximo)}}.
    """
    cores = [AZUL, LARANJA]
    marcadores = ["o", "D"]
    metricas = list(next(iter(series.values())))
    fig, ax = plt.subplots(figsize=(7.5, 0.75 * len(metricas) + 1.4))
    desvio = 0.16 if len(series) > 1 else 0
    for s, (nome, valores) in enumerate(series.items()):
        for m, metrica in enumerate(metricas):
            valor, minimo, maximo = valores[metrica]
            y = m + (-desvio if s == 0 else desvio)  # a primeira série fica em cima
            ax.plot([minimo, maximo], [y, y], color=cores[s], linewidth=2, solid_capstyle="round")
            ax.plot(valor, y, marcadores[s], color=cores[s], markersize=8, markeredgecolor=SUPERFICIE,
                    markeredgewidth=2, label=nome if m == 0 else None)
            ax.annotate(f"{valor:.1f}".replace(".", ","), (maximo, y), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(range(len(metricas)), metricas)
    ax.invert_yaxis()
    ax.set_ylim(len(metricas) - 0.5, -0.5)
    ax.set_title(titulo)
    ax.set_xlabel(eixo_x)
    ax.xaxis.grid(True)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.spines["bottom"].set_visible(False)
    if len(series) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, fontsize=9,
                  labelcolor=TINTA_SECUNDARIA, borderaxespad=0.2, handletextpad=0.3, columnspacing=1.6)
        ax.set_title(titulo, pad=30)
    return fig, ax


def diferencas_com_intervalo(
    dados: dict[str, tuple[float, float, float]],
    titulo: str,
    eixo_x: str,
) -> tuple[Figure, Axes]:
    """Uma diferença por linha: o ponto é a diferença medida, a barra é o intervalo de 95%.

    `dados` é {rótulo da linha: (diferença, mínimo, máximo)}. A linha vertical
    marca o zero: um intervalo que não a toca é uma diferença distinguível.
    """
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(dados) + 1.3))
    ax.axvline(0, color=TINTA_SUAVE, linewidth=1.2, zorder=0.6)
    for y, (rotulo, (valor, minimo, maximo)) in enumerate(dados.items()):
        ax.plot([minimo, maximo], [y, y], color=AZUL, linewidth=2, solid_capstyle="round")
        ax.plot(valor, y, "o", color=AZUL, markersize=8, markeredgecolor=SUPERFICIE, markeredgewidth=2)
        ax.annotate(f"{valor:+.1f}".replace(".", ",").replace("-", "−"), (maximo, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(range(len(dados)), list(dados))
    ax.set_ylim(len(dados) - 0.5, -0.5)
    ax.set_title(titulo)
    ax.set_xlabel(eixo_x)
    ax.xaxis.grid(True)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.margins(x=0.12)
    return fig, ax


def barras_em_ordem(valores: dict[str, float], titulo: str, eixo_x: str, formato=None) -> tuple[Figure, Axes]:
    """Barras horizontais na ordem dada, de cima para baixo, cada uma com o seu valor escrito.

    Para quando a ordem das barras já diz algo (as 1, 2, 3... mais prováveis) e não deve ser trocada pelo tamanho.
    """
    formato = formato or (lambda v: f"{v:.1f}".replace(".", ","))
    nomes, numeros = list(valores), list(valores.values())
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(nomes) + 1.2))
    ax.barh(nomes, numeros, height=0.5, color=AZUL)
    for nome, numero in zip(nomes, numeros):
        ax.annotate(formato(numero), (numero, nome), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5, color=TINTA)
    ax.invert_yaxis()
    ax.set_title(titulo)
    ax.set_xlabel(eixo_x)
    ax.xaxis.grid(True)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.spines["bottom"].set_visible(False)
    ax.margins(x=0.08)
    return fig, ax


def calibracao(faixas, titulo: str) -> tuple[Figure, Axes]:
    """Confiança contra acurácia, uma marca por faixa de confiança (Guo et al., arXiv 1706.04599).

    `faixas` é a tabela de `confianca.faixas_de_confianca`. Na diagonal, o sistema acerta tanto quanto diz
    que vai acertar; abaixo dela, confia demais. O número ao lado de cada marca é quantos pedidos a faixa tem.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 100], [0, 100], color=TINTA_SUAVE, linewidth=1.2, zorder=0.6)
    # o rótulo da diagonal fica acima dela: um sistema que confia demais tem as marcas abaixo
    ax.annotate("acerta tanto quanto diz", (60, 60), xytext=(-6, 6), textcoords="offset points", ha="right", va="bottom",
                fontsize=8.5, color=TINTA_SECUNDARIA, rotation=42, rotation_mode="anchor")
    x, y = 100 * faixas["confianca"], 100 * faixas["acuracia"]
    ax.plot(x, y, color=AZUL, linewidth=2, zorder=2)
    ax.plot(x, y, "o", color=AZUL, markersize=8, markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=3)
    for xi, yi, quantos in zip(x, y, faixas["pedidos"]):
        ax.annotate(f"{quantos:,}".replace(",", "."), (xi, yi), xytext=(-6, 7), textcoords="offset points", ha="right", fontsize=8, color=TINTA_SECUNDARIA)
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 104)
    ax.set_aspect("equal")
    ax.set_title(titulo)
    ax.set_xlabel("confiança na intenção escolhida (%)")
    ax.set_ylabel("acurácia (%)")
    ax.grid(True)
    return fig, ax


def curva_com_intervalo(
    series: dict[str, dict[float, tuple[float, float, float]]],
    titulo: str,
    eixo_x: str,
    eixo_y: str,
    marcas_x: list[float] | None = None,
    niveis: bool = False,
) -> tuple[Figure, Axes]:
    """Até duas séries ao longo de um eixo: o ponto é o valor e a barra vertical, o intervalo de 95%.

    `series` é {nome da série: {x: (valor, mínimo, máximo)}}. Serve para a curva
    de qualidade por esparsidade (Etapa 7). Com `niveis=True`, os valores de x
    são níveis postos a distâncias iguais (30, 50, 70, 90, 95 e 98% não cabem
    lado a lado numa régua): o eixo mostra o nome de cada nível, não uma escala.
    `marcas_x` escolhe onde o eixo x ganha rótulo, quando não são níveis.
    """
    cores, marcadores = [AZUL, LARANJA], ["o", "D"]
    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    todos = sorted({x for pontos in series.values() for x in pontos})
    lugar = {x: i for i, x in enumerate(todos)} if niveis else {x: x for x in todos}
    largura = (lugar[todos[-1]] - lugar[todos[0]]) or 1
    for s, (nome, pontos) in enumerate(series.items()):
        desvio = (s - (len(series) - 1) / 2) * 0.016 * largura  # as barras das duas séries não ficam uma em cima da outra
        xs = [lugar[x] + desvio for x in pontos]
        ax.plot(xs, [v[0] for v in pontos.values()], color=cores[s], linewidth=2, zorder=2)
        for x, (_, minimo, maximo) in zip(xs, pontos.values()):
            ax.plot([x, x], [minimo, maximo], color=cores[s], linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(xs, [v[0] for v in pontos.values()], marcadores[s], color=cores[s], markersize=7.5, markeredgecolor=SUPERFICIE,
                markeredgewidth=1.8, zorder=3, label=nome)
    if niveis:
        ax.set_xticks(range(len(todos)), [f"{x:g}".replace(".", ",") for x in todos])
    else:
        ax.set_xticks(marcas_x if marcas_x is not None else todos)
        ax.xaxis.set_major_formatter(VIRGULA)
    ax.yaxis.set_major_formatter(VIRGULA)
    ax.yaxis.grid(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(eixo_x)
    ax.set_ylabel(eixo_y)
    ax.set_title(titulo, pad=30 if len(series) > 1 else 14)
    if len(series) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, fontsize=9,
                  labelcolor=TINTA_SECUNDARIA, borderaxespad=0.2, handletextpad=0.3, columnspacing=1.6)
    return fig, ax


def salvar(fig: Figure, pasta: Path, nome: str) -> Path:
    """Grava a figura na pasta de entrega da etapa."""
    destino = pasta / "figuras"
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.png"
    fig.savefig(caminho)
    return caminho


def html_bio(tokens: list[str], tags: list[str]) -> str:
    """Um enunciado como o modelo vê: cada palavra com a sua tag embaixo.

    Palavras que pertencem a uma entidade ganham fundo; a tag identifica o tipo.
    """
    celulas = []
    for token, tag in zip(tokens, tags):
        em_entidade = tag != "O"
        fundo = f"background:{AZUL_CLARO};color:{TINTA};" if em_entidade else ""
        celulas.append(
            '<span style="display:inline-block;text-align:center;margin:0 1px 6px 0;">'
            f'<span style="display:block;padding:3px 7px;border-radius:4px;{fundo}">{escape(token)}</span>'
            f'<span style="display:block;font-size:11px;font-family:monospace;color:{TINTA_SUAVE};">{escape(tag)}</span>'
            "</span>"
        )
    return f'<div style="line-height:1.3;margin:4px 0 10px 0;">{"".join(celulas)}</div>'


# --------------------------------------------------------------------------
# Etapa 9: custo contra qualidade, e os degraus de uma cadeia
# --------------------------------------------------------------------------

# Eixos em escala logarítmica: 0,1 · 1 · 10 · 1.000, com vírgula decimal e ponto de milhar.
MILHAR = FuncFormatter(lambda valor, _: f"{valor:,.0f}".replace(",", ".") if valor >= 1000 else f"{valor:g}".replace(".", ","))

_LADOS = {"direita": ((8, 0), "left", "center"), "esquerda": ((-8, 0), "right", "center"), "cima": ((0, 8), "center", "bottom"), "baixo": ((0, -8), "center", "top")}


def fronteira(
    pontos: list[dict],
    titulo: str,
    eixo_x: str,
    eixo_y: str,
    series: list[str] | None = None,
    trilhas: list[dict] | None = None,
    escada: list[tuple[float, float]] | None = None,
    rotulo_da_escada: str | None = None,
    colunas: dict[str, float] | None = None,
    nota: str | None = None,
    altura: float = 5.0,
) -> tuple[Figure, Axes]:
    """Custo contra qualidade: um ponto por sistema, com o custo (eixo x) em escala logarítmica.

    `pontos` é uma lista de dicionários, um por sistema:

    - `x`, `y`: o custo e a qualidade;
    - `rotulo`: o texto ao lado do ponto (None: sem texto; quem identifica o ponto é a trilha e a coluna);
    - `serie`: 0 ou 1, a cor (no máximo duas séries, nomeadas em `series`); None para um ponto de referência que não é de nenhuma das duas (cinza, fora da legenda);
    - `marcador`: a forma do ponto, para uma segunda informação (por exemplo, a condição da medida);
    - `vazado`: ponto só com contorno;
    - `lado`: de que lado do ponto vai o texto (`direita`, `esquerda`, `cima`, `baixo`), e `desvio`, em pontos, para afastá-lo de um vizinho;
    - `fio`: um traço fino do texto até o ponto, para quando o texto precisa ficar longe;
    - `forte`: texto em negrito, para o ponto que é o assunto da figura.

    `trilhas` liga os pontos do mesmo sistema em formas diferentes: [{"pontos": [(x, y), ...], "serie": 0}].
    `escada` é a fronteira: os pontos que nenhum outro vence em custo e qualidade ao mesmo tempo, do mais barato
    para o mais caro. `colunas` dá nome a posições do eixo x que se repetem entre os sistemas: {texto: x}, ou
    {texto: (x, alinhamento)} quando duas colunas vizinhas precisam se afastar. `nota` é um texto de leitura,
    no canto de baixo à direita.
    """
    cores = [AZUL, LARANJA]
    fig, ax = plt.subplots(figsize=(7.5, altura))
    ax.set_xscale("log")
    if escada:
        xs, ys = [p[0] for p in escada], [p[1] for p in escada]
        ax.step(xs, ys, where="post", color=TINTA_SUAVE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1.5)
        if rotulo_da_escada and len(xs) > 1:
            # o nome da fronteira vai embaixo do patamar mais comprido, encostado no fim dele: é onde há espaço vazio
            i = int(np.argmax(np.log(xs[1:]) - np.log(xs[:-1])))
            ax.annotate(rotulo_da_escada, (xs[i + 1], ys[i]), xytext=(-8, -6), textcoords="offset points", ha="right", va="top",
                        fontsize=8.5, color=TINTA_SECUNDARIA)
    for trilha in trilhas or []:
        ax.plot([p[0] for p in trilha["pontos"]], [p[1] for p in trilha["pontos"]], color=cores[trilha.get("serie", 0)], linewidth=1.6, alpha=0.55, zorder=2)
    ja_na_legenda = set()
    for ponto in pontos:
        serie = ponto.get("serie", 0)
        vazado = ponto.get("vazado", False)
        cor = TINTA_SECUNDARIA if serie is None else cores[serie]
        na_legenda = bool(series) and serie is not None and not vazado and ponto.get("marcador", "o") == "o"
        ax.plot(
            ponto["x"], ponto["y"], ponto.get("marcador", "o"), color=cor, markersize=7 if vazado else 8.5,
            markerfacecolor=SUPERFICIE if vazado else cor, markeredgecolor=cor if vazado else SUPERFICIE, markeredgewidth=1.8,
            zorder=3, linestyle="none", label=series[serie] if na_legenda and serie not in ja_na_legenda else None,
        )
        if na_legenda:
            ja_na_legenda.add(serie)
        if ponto.get("rotulo"):
            (dx, dy), ha, va = _LADOS[ponto.get("lado", "direita")]
            ex, ey = ponto.get("desvio", (0, 0))
            fio = {"arrowstyle": "-", "color": TINTA_SUAVE, "linewidth": 0.8, "shrinkA": 1, "shrinkB": 5} if ponto.get("fio") else None
            ax.annotate(ponto["rotulo"], (ponto["x"], ponto["y"]), xytext=(dx + ex, dy + ey), textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                        color=TINTA, fontweight="bold" if ponto.get("forte") else "normal", linespacing=1.25, arrowprops=fio)
    for texto, onde in (colunas or {}).items():
        x, ha = onde if isinstance(onde, tuple) else (onde, "center")
        ax.annotate(texto, (x, 1.0), xycoords=("data", "axes fraction"), xytext=({"right": 5, "left": -5}.get(ha, 0), -6), textcoords="offset points",
                    ha=ha, va="top", fontsize=8.5, color=TINTA_SECUNDARIA, linespacing=1.25)
    if nota:
        ax.annotate(nota, (1.0, 0.0), xycoords="axes fraction", xytext=(-6, 8), textcoords="offset points", ha="right", va="bottom", fontsize=8.5,
                    color=TINTA_SECUNDARIA, linespacing=1.35)
    ax.xaxis.set_major_formatter(MILHAR)
    ax.yaxis.set_major_formatter(VIRGULA)
    ax.minorticks_off()
    ax.grid(True)
    ax.set_xlabel(eixo_x)
    ax.set_ylabel(eixo_y)
    ax.set_title(titulo, pad=30 if series and len(series) > 1 else 14)
    if series and len(series) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, fontsize=9, labelcolor=TINTA_SECUNDARIA,
                  borderaxespad=0.2, handletextpad=0.3, columnspacing=1.6)
    return fig, ax


def degraus_em_paineis(degraus: list[str], paineis: list[dict], titulo: str) -> tuple[Figure, list[Axes]]:
    """Os degraus de um caminho, de cima para baixo, em painéis lado a lado: cada painel mostra uma medida, no seu próprio eixo.

    `paineis` é uma lista de dicionários, um por medida:

    - `eixo_x`: o nome da medida e a unidade;
    - `valores`: um por degrau. Um número, ou (valor, mínimo, máximo) para desenhar o intervalo, ou None quando a medida não existe;
    - `rotulos`: o texto ao lado de cada ponto (para None, o texto que explica a falta: "não divulgado");
    - `log`: eixo em escala logarítmica, para medidas que mudam de ordem de grandeza entre os degraus.

    Não há dois eixos no mesmo painel: medidas de unidades diferentes ficam em painéis diferentes.
    """
    fig, eixos = plt.subplots(1, len(paineis), figsize=(3.6 * len(paineis) + 0.6, 0.52 * len(degraus) + 1.5), sharey=True,
                              gridspec_kw={"wspace": 0.18})
    eixos = list(np.atleast_1d(eixos))
    for ax, painel in zip(eixos, paineis):
        numeros = [v if isinstance(v, tuple) else (v, v, v) for v in painel["valores"] if v is not None]
        menor, maior = min(v[1] for v in numeros), max(v[2] for v in numeros)
        # o texto fica à direita de cada ponto: o eixo guarda, à direita do maior valor, a parte da largura que o texto mais comprido ocupa
        do_texto = min(0.62, 0.07 + 0.022 * max(len(r) for v, r in zip(painel["valores"], painel["rotulos"]) if v is not None))
        esquerda, direita = 0.07 / (1 - do_texto - 0.07), do_texto / (1 - do_texto - 0.07)
        if painel.get("log"):
            ax.set_xscale("log")
            decadas = max(float(np.log10(maior / menor)), 0.3)
            ax.set_xlim(menor / 10 ** (esquerda * decadas), maior * 10 ** (direita * decadas))
            marcas = [10.0**e for e in range(-6, 13) if menor / 3 <= 10.0**e <= maior * 3]   # só as potências de 10 onde há dado
            ax.set_xticks(marcas if len(marcas) <= 4 else marcas[::-1][::2][::-1])              # muitas décadas num painel estreito: uma sim, uma não
            ax.xaxis.set_major_formatter(MILHAR)
            ax.minorticks_off()
        else:
            faixa = (maior - menor) or 1.0
            ax.set_xlim(menor - esquerda * faixa, maior + direita * faixa)
            ax.xaxis.set_major_formatter(VIRGULA)
        for y, (valor, rotulo) in enumerate(zip(painel["valores"], painel["rotulos"])):
            if valor is None:
                ax.annotate(rotulo, (0.04, y), xycoords=("axes fraction", "data"), va="center", fontsize=8.5, color=TINTA_SUAVE)
                continue
            valor, minimo, maximo = valor if isinstance(valor, tuple) else (valor, valor, valor)
            if maximo > minimo:
                ax.plot([minimo, maximo], [y, y], color=AZUL, linewidth=2, solid_capstyle="round")
            ax.plot(valor, y, "o", color=AZUL, markersize=8, markeredgecolor=SUPERFICIE, markeredgewidth=2)
            ax.annotate(rotulo, (maximo, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5, color=TINTA)
        ax.set_xlabel(painel["eixo_x"])
        ax.xaxis.grid(True)
        ax.spines["bottom"].set_visible(False)
        if ax is not eixos[0]:
            ax.spines["left"].set_visible(False)
    eixos[0].set_yticks(range(len(degraus)), degraus)
    eixos[0].set_ylim(len(degraus) - 0.5, -0.5)
    eixos[0].set_title(titulo)
    return fig, eixos
