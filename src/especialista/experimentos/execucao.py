"""Uma execução de treino: treina, grava o que produziu e não repete se já estiver pronta.

É a peça comum aos experimentos de todas as etapas. Cada etapa só declara
as suas variantes (o que muda em `Config` e em `Receita`) e, quando não
treina com os rótulos humanos da Etapa 1, de onde vêm os seus dados.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import EXECUCOES
from especialista.dados import massive
from especialista.modelos.diet import Config, SistemaDiet
from especialista.treino import Receita, treinar


def pasta_da_execucao(etapa: str, variante: str, semente: int) -> Path:
    return EXECUCOES / etapa / variante / f"semente_{semente}"


def dados_humanos() -> dict:
    """O que o treino lê quando os rótulos são os humanos: o treino e o `dev` da Etapa 1."""
    particoes, rotulos = massive.carregar_entrega()
    return {"treino": particoes["train"], "dev": particoes["dev"], "rotulos": rotulos}


def executar(
    etapa: str, variante: str, semente: int, definicao: dict, dispositivo: str | None = None, dados=dados_humanos
) -> Path:
    """Treina a variante com a semente dada, ou devolve a pasta se já estiver pronta.

    `definicao` tem `nome`, `config` (campos de `Config` a trocar) e
    `receita` (campos de `Receita` a trocar).

    `dados` é a função que busca o que o treino lê; só é chamada se a execução
    ainda não existe. O padrão são os rótulos humanos da Etapa 1. Ela devolve
    um dicionário com:

    - `treino` e `dev`: os dois conjuntos, com as colunas `utt`, `tokens`,
      `intent` e `tags`. É `dev` que escolhe a época de parada;
    - `rotulos`: o inventário de intenções e de tags;
    - `vocabulario` (opcional): o conjunto de onde sai o vocabulário de
      features. Se faltar, é `treino`;
    - `avaliar` (opcional): o conjunto cujas respostas ficam gravadas em
      `previsoes_dev.parquet`. Se faltar, é `dev`;
    - `origem` (opcional): de quem são os rótulos, para ficar anotado em
      `meta.json`.
    """
    pasta = pasta_da_execucao(etapa, variante, semente)
    if (pasta / "meta.json").exists():
        return pasta

    config = Config(**definicao.get("config", {}))
    receita = Receita(**definicao.get("receita", {}))
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")

    lido = dados()
    treino, dev, rotulos = lido["treino"], lido["dev"], lido["rotulos"]
    avaliar = lido.get("avaliar", dev)
    sistema = SistemaDiet.novo(lido.get("vocabulario", treino), rotulos, config, semente=semente, dispositivo=dispositivo)

    inicio = time.perf_counter()
    historico = treinar(sistema, treino, dev, receita, semente=semente)
    duracao = time.perf_counter() - inicio

    pasta.mkdir(parents=True, exist_ok=True)
    previsoes = sistema.prever(list(avaliar["utt"]))
    no_dev_de_parada = previsoes if avaliar is dev else sistema.prever(list(dev["utt"]))
    historico.to_csv(pasta / "historico.csv", index=False)
    salvar_previsoes(pasta / "previsoes_dev.parquet", avaliar["id"], previsoes)
    sistema.salvar(pasta / "modelo")
    meta = {
        "etapa": etapa,
        "variante": variante,
        "nome": definicao.get("nome", variante),
        "semente": semente,
        "config": config.como_dict(),
        "receita": receita.como_dict(),
        "melhor_epoca": int(historico.loc[historico["melhor"], "epoca"].iloc[0]),
        "dev": qualidade.avaliar(ouro(dev), no_dev_de_parada),   # no conjunto que escolheu a época de parada
        "exemplos_de_treino": len(treino),
        "exemplos_de_dev": len(dev),
        **({"origem": lido["origem"]} if "origem" in lido else {}),
        "segundos_de_treino": duracao,
        "dispositivo": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # por último: marca "pronto"
    return pasta


def carregar(etapa: str, variante: str, semente: int) -> dict:
    """O que uma execução produziu: meta, histórico e previsões em dev."""
    pasta = pasta_da_execucao(etapa, variante, semente)
    _, previsoes = carregar_previsoes(pasta / "previsoes_dev.parquet")
    return {
        "pasta": pasta,
        "meta": json.loads((pasta / "meta.json").read_text(encoding="utf-8")),
        "historico": pd.read_csv(pasta / "historico.csv"),
        "previsoes_dev": previsoes,
    }


def linha_de_comando(etapa: str, variantes: dict[str, dict], executar_variante=None) -> None:
    """`python -m especialista.experimentos.<etapa> --variante NOME --semente N`

    `executar_variante(variante, semente)` é para a etapa que treina com dados próprios;
    sem ela, vale `executar` com os rótulos humanos.
    """
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--variante", required=True, choices=sorted(variantes))
    argumentos.add_argument("--semente", type=int, required=True)
    a = argumentos.parse_args()
    if executar_variante is None:
        pasta = executar(etapa, a.variante, a.semente, variantes[a.variante])
    else:
        pasta = executar_variante(a.variante, a.semente)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    print(a.variante, a.semente, f"época {meta['melhor_epoca']}", {k: round(v, 4) for k, v in meta["dev"].items()},
          f"{meta['segundos_de_treino']:.0f}s")
