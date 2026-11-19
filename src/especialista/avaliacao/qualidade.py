"""Régua de qualidade: quatro métricas e a incerteza de cada uma.

As métricas seguem FitzGerald et al. (MASSIVE, arXiv 2204.08582): acurácia
de intenção, F1 micro de entidades e exact match. O macro-F1 de intenção
entra porque as classes são desbalanceadas (Etapa 1).

| Métrica           | Pergunta que responde                                        |
|-------------------|--------------------------------------------------------------|
| `intent_acc`      | Em quantos enunciados a intenção está certa?                 |
| `intent_macro_f1` | E se cada intenção pesar igual, rara ou frequente?           |
| `slot_f1`         | Das entidades, quantas saíram com tipo e limites certos?     |
| `exact_match`     | Em quantos enunciados a resposta inteira está certa?         |

A incerteza vem de bootstrap por enunciado (Koehn, 2004; Berg-Kirkpatrick
et al., 2012): sorteia-se o conjunto de teste de novo, com reposição, e
recalcula-se tudo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from especialista.avaliacao.previsao import Previsao

METRICAS = ("intent_acc", "intent_macro_f1", "slot_f1", "exact_match")
NOMES = {
    "intent_acc": "Acurácia de intenção",
    "intent_macro_f1": "Macro-F1 de intenção",
    "slot_precision": "Precisão de entidades",
    "slot_recall": "Recall de entidades",
    "slot_f1": "F1 de entidades",
    "exact_match": "Exact match",
}


def _por_enunciado(ouro: list[Previsao], prev: list[Previsao]) -> dict:
    """Reduz cada enunciado aos poucos números de que as métricas precisam."""
    if len(ouro) != len(prev):
        raise ValueError(f"{len(ouro)} rótulos para {len(prev)} previsões.")
    classes = sorted({o.intent for o in ouro})
    indice = {c: i for i, c in enumerate(classes)}
    entidades_ouro = [frozenset(o.slots) for o in ouro]
    entidades_prev = [frozenset(p.slots) for p in prev]
    return {
        "classes": classes,
        # intenção fora do esquema (uma LLM pode inventar) vira -1: nunca acerta, não cria classe
        "ouro": np.array([indice[o.intent] for o in ouro]),
        "prev": np.array([indice.get(p.intent, -1) for p in prev]),
        "certas": np.array([len(a & b) for a, b in zip(entidades_ouro, entidades_prev)]),
        "n_ouro": np.array([len(a) for a in entidades_ouro]),
        "n_prev": np.array([len(b) for b in entidades_prev]),
    }


def _razao(numerador: float, denominador: float) -> float:
    return float(numerador / denominador) if denominador else 0.0


def _metricas(e: dict, amostra: np.ndarray | None = None) -> dict[str, float]:
    ouro, prev, certas, n_ouro, n_prev = (
        e[k] if amostra is None else e[k][amostra] for k in ("ouro", "prev", "certas", "n_ouro", "n_prev")
    )
    k = len(e["classes"])
    acertou = ouro == prev

    # Macro-F1: F1 de cada intenção presente nos rótulos, depois a média simples.
    certas_k = np.bincount(ouro[acertou], minlength=k)
    ouro_k = np.bincount(ouro, minlength=k)
    prev_k = np.bincount(prev[prev >= 0], minlength=k)
    presentes = ouro_k > 0
    f1_k = 2 * certas_k[presentes] / (ouro_k[presentes] + prev_k[presentes])

    # Entidades: uma entidade só conta como certa com tipo, início e fim iguais.
    tp = certas.sum()
    return {
        "intent_acc": float(acertou.mean()),
        "intent_macro_f1": float(f1_k.mean()),
        "slot_precision": _razao(tp, n_prev.sum()),
        "slot_recall": _razao(tp, n_ouro.sum()),
        "slot_f1": _razao(2 * tp, n_prev.sum() + n_ouro.sum()),
        "exact_match": float((acertou & (certas == n_ouro) & (certas == n_prev)).mean()),
    }


def avaliar(ouro: list[Previsao], prev: list[Previsao]) -> dict[str, float]:
    """As métricas de um sistema, entre 0 e 1."""
    return _metricas(_por_enunciado(ouro, prev))


def reamostrar(ouro: list[Previsao], prev: list[Previsao], n: int = 2000, semente: int = 0) -> pd.DataFrame:
    """Bootstrap por enunciado: `n` conjuntos de teste sorteados, uma linha de métricas para cada.

    Com a mesma semente, dois sistemas recebem os mesmos sorteios. É isso que
    torna a comparação pareada.
    """
    e = _por_enunciado(ouro, prev)
    rng = np.random.default_rng(semente)
    tamanho = len(ouro)
    return pd.DataFrame([_metricas(e, rng.integers(0, tamanho, tamanho)) for _ in range(n)])


def intervalo(ouro: list[Previsao], prev: list[Previsao], n: int = 2000, semente: int = 0) -> pd.DataFrame:
    """Valor de cada métrica com intervalo de 95% de confiança."""
    amostras = reamostrar(ouro, prev, n, semente)
    return pd.DataFrame(
        {
            "valor": pd.Series(avaliar(ouro, prev)),
            "ic_min": amostras.quantile(0.025),
            "ic_max": amostras.quantile(0.975),
        }
    )


def comparar(
    ouro: list[Previsao], a: list[Previsao], b: list[Previsao], n: int = 2000, semente: int = 0
) -> pd.DataFrame:
    """Diferença pareada B − A, com intervalo de 95%.

    A diferença é "distinguível" quando o intervalo não contém zero.
    """
    diferencas = reamostrar(ouro, b, n, semente) - reamostrar(ouro, a, n, semente)
    tabela = pd.DataFrame(
        {
            "a": pd.Series(avaliar(ouro, a)),
            "b": pd.Series(avaliar(ouro, b)),
            "ic_min": diferencas.quantile(0.025),
            "ic_max": diferencas.quantile(0.975),
        }
    )
    tabela.insert(2, "diferenca", tabela["b"] - tabela["a"])
    tabela["distinguivel"] = (tabela["ic_min"] > 0) | (tabela["ic_max"] < 0)
    return tabela


# --------------------------------------------------------------------------
# Modelos treinados: várias sementes por sistema
# --------------------------------------------------------------------------
#
# Uma rede treinada duas vezes, com sementes diferentes, dá resultados um pouco
# diferentes (Reimers & Gurevych, arXiv 1707.09861). Por isso um "sistema"
# treinado é um grupo de execuções, e o seu resultado é a média do grupo.


def _media_das_reamostras(ouro: list[Previsao], grupo: list[list[Previsao]], n: int, semente: int) -> pd.DataFrame:
    """Para cada sorteio do conjunto, a média das métricas entre as execuções do grupo."""
    return sum(reamostrar(ouro, prev, n, semente) for prev in grupo) / len(grupo)


def intervalo_do_grupo(ouro: list[Previsao], grupo: list[list[Previsao]], n: int = 2000, semente: int = 0) -> pd.DataFrame:
    """Média entre sementes, intervalo de 95% dessa média e o desvio-padrão entre as sementes."""
    por_semente = pd.DataFrame([avaliar(ouro, prev) for prev in grupo])
    amostras = _media_das_reamostras(ouro, grupo, n, semente)
    return pd.DataFrame(
        {
            "valor": por_semente.mean(),
            "ic_min": amostras.quantile(0.025),
            "ic_max": amostras.quantile(0.975),
            "desvio_sementes": por_semente.std(ddof=1) if len(grupo) > 1 else 0.0,
        }
    )


def comparar_grupos(
    ouro: list[Previsao], a: list[list[Previsao]], b: list[list[Previsao]], n: int = 2000, semente: int = 0
) -> pd.DataFrame:
    """Diferença pareada B − A entre as médias de dois grupos de execuções, com intervalo de 95%."""
    diferencas = _media_das_reamostras(ouro, b, n, semente) - _media_das_reamostras(ouro, a, n, semente)
    media = lambda grupo: pd.DataFrame([avaliar(ouro, prev) for prev in grupo]).mean()
    tabela = pd.DataFrame(
        {"a": media(a), "b": media(b), "ic_min": diferencas.quantile(0.025), "ic_max": diferencas.quantile(0.975)}
    )
    tabela.insert(2, "diferenca", tabela["b"] - tabela["a"])
    tabela["distinguivel"] = (tabela["ic_min"] > 0) | (tabela["ic_max"] < 0)
    return tabela


def por_intencao(ouro: list[Previsao], prev: list[Previsao]) -> pd.DataFrame:
    """Precisão, recall e F1 de cada intenção: mostra onde o sistema erra."""
    e = _por_enunciado(ouro, prev)
    k = len(e["classes"])
    certas = np.bincount(e["ouro"][e["ouro"] == e["prev"]], minlength=k)
    n_ouro = np.bincount(e["ouro"], minlength=k)
    n_prev = np.bincount(e["prev"][e["prev"] >= 0], minlength=k)
    with np.errstate(divide="ignore", invalid="ignore"):
        tabela = pd.DataFrame(
            {
                "exemplos": n_ouro,
                "precisao": np.where(n_prev > 0, certas / n_prev, 0.0),
                "recall": certas / n_ouro,
                "f1": 2 * certas / (n_ouro + n_prev),
            },
            index=pd.Index(e["classes"], name="intent"),
        )
    return tabela


DESTINOS = ("certa", "limite diferente", "tipo diferente", "tipo e limite diferentes", "faltou")


def erros_de_entidade(ouro: list[Previsao], prev: list[Previsao]) -> pd.Series:
    """O que aconteceu com cada entidade dos rótulos, e quantas o sistema inventou.

    Cada entidade dos rótulos cai em um destino:

    - `certa`: o sistema devolveu o mesmo tipo, início e fim;
    - `limite diferente`: mesmo tipo, trecho que se sobrepõe mas começa ou acaba em outro lugar;
    - `tipo diferente`: mesmo trecho, outro tipo;
    - `tipo e limite diferentes`: só se sobrepõe;
    - `faltou`: nada do que o sistema devolveu toca nela.

    `a mais` conta as entidades do sistema que não tocam em nenhuma entidade
    errada dos rótulos. A soma dos cinco destinos é o total de entidades dos rótulos.
    """
    contagem = {destino: 0 for destino in (*DESTINOS, "a mais")}
    toca = lambda a, b: a[1] < b[2] and b[1] < a[2]
    for o, p in zip(ouro, prev):
        do_ouro, do_sistema = set(o.slots), set(p.slots)
        contagem["certa"] += len(do_ouro & do_sistema)
        erradas, sobras = do_ouro - do_sistema, do_sistema - do_ouro
        for entidade in erradas:
            vizinhas = [s for s in sobras if toca(entidade, s)]
            if not vizinhas:
                contagem["faltou"] += 1
            elif any(s[1:] == entidade[1:] for s in vizinhas):
                contagem["tipo diferente"] += 1
            elif any(s[0] == entidade[0] for s in vizinhas):
                contagem["limite diferente"] += 1
            else:
                contagem["tipo e limite diferentes"] += 1
        contagem["a mais"] += sum(not any(toca(s, entidade) for entidade in erradas) for s in sobras)
    return pd.Series(contagem, name="entidades")


def por_tipo_de_entidade(ouro: list[Previsao], prev: list[Previsao]) -> pd.DataFrame:
    """Precisão, recall e F1 de cada tipo de entidade."""
    no_ouro, previstas, certas = {}, {}, {}
    for o, p in zip(ouro, prev):
        for tipo, *_ in o.slots:
            no_ouro[tipo] = no_ouro.get(tipo, 0) + 1
        for tipo, *_ in p.slots:
            previstas[tipo] = previstas.get(tipo, 0) + 1
        for tipo, *_ in set(o.slots) & set(p.slots):
            certas[tipo] = certas.get(tipo, 0) + 1
    linhas = []
    for tipo in sorted(set(no_ouro) | set(previstas)):
        g, q, c = no_ouro.get(tipo, 0), previstas.get(tipo, 0), certas.get(tipo, 0)
        linhas.append({"tipo": tipo, "exemplos": g, "previstas": q, "precisao": _razao(c, q),
                       "recall": _razao(c, g), "f1": _razao(2 * c, g + q)})
    return pd.DataFrame(linhas).set_index("tipo")
