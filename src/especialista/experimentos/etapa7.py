"""Etapa 7: poda e fine-tuning de recuperação.

Cada execução parte de um modelo já treinado, tira uma parte dele e reajusta
o que sobrou, com o mesmo sinal com que o modelo aprendeu:

    sem_poda        não tira nada e faz o mesmo reajuste (controle)
    magnitude_N     zera N% dos pesos de menor valor absoluto, de uma vez, e
                    reajusta com a máscara travada (Han et al., arXiv 1506.02626)
    gradual_N       chega aos mesmos N% aos poucos, durante o treino
                    (Zhu & Gupta, arXiv 1710.01878)
    linhas_N        remove N% das linhas da tabela de features (features
                    inteiros) e reajusta
    sem_camada      remove a camada de cima do encoder e reajusta

N é 30, 50, 70 ou 90, os níveis do plano. A poda de uma vez tem ainda 95 e 98,
para achar onde a rede quebra.

Quem é podado: o teto da Etapa 4b e os alunos que a Etapa 6 marcou em
`entregas/06_destilacao/decisao.json`.

    uv run python -m especialista.experimentos.etapa7 --modelo teto --variante magnitude_50 --semente 0
    uv run python -m especialista.experimentos.etapa7 --faltam      # lista o que ainda não rodou

**O reajuste usa o sinal de treino do próprio modelo.** Um aluno é reajustado
com os rótulos (ou logits) do seu professor, e a época de parada é escolhida
no `dev` rotulado pelo mesmo professor, como na Etapa 6. Nenhum rótulo humano
entra. O teto, que aprendeu com rótulos humanos, é reajustado com eles.

O orçamento do reajuste é o mesmo em todas as execuções (`EPOCAS`), e a
receita é a de quem treinou o modelo. A regra que escolhe o que segue para a
quantização, escrita antes de olhar os resultados, está em `quem_segue`.

**Se um modelo de partida for treinado de novo, o comando é o mesmo.** Cada
execução anota a impressão digital dos pesos de que partiu. Quando os pesos
mudam, as execuções antigas daquele modelo voltam a aparecer em `--faltam` e
são refeitas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.compressao import poda
from especialista.dados import massive
from especialista.experimentos import etapa6, execucao
from especialista.modelos.diet import Config, SistemaDiet
from especialista.treino import Receita

ETAPA = "07_poda"
SEMENTES = (0, 1, 2)
ESPARSIDADES = (30, 50, 70, 90)   # em %, as do plano
EXTRAS = (95, 98)                 # só na poda de uma vez: 20 e 50 vezes menos pesos, para achar onde a rede quebra

# O orçamento do reajuste, igual para todas as execuções
EPOCAS = 30                # de reajuste; o treino original tem 80 e chega ao melhor ponto entre a 25ª e a 45ª
EPOCAS_DE_RAMPA = 15       # poda gradual: a esparsidade sobe na primeira metade e a máscara fica travada na segunda
PASSOS_ENTRE_CORTES = 100  # poda gradual: a máscara é refeita a cada 100 passos de treino

DECISAO_DA_REDE = "04b_intencao/decisao.json"
DECISAO_DA_DESTILACAO = "06_destilacao/decisao.json"
CRITERIO = "exact_match"   # a métrica da regra do que segue

VARIANTES: dict[str, dict] = {
    "sem_poda": {"nome": "Sem poda, com o mesmo reajuste (controle)", "familia": "controle", "nivel": 0},
    **{f"magnitude_{s}": {"nome": f"Magnitude, {s}% de uma vez", "familia": "magnitude", "nivel": s} for s in (*ESPARSIDADES, *EXTRAS)},
    **{f"gradual_{s}": {"nome": f"Magnitude, {s}% gradual", "familia": "gradual", "nivel": s} for s in ESPARSIDADES},
    **{f"linhas_{s}": {"nome": f"Sem {s}% das linhas da tabela de features", "familia": "linhas", "nivel": s} for s in ESPARSIDADES},
    "sem_camada": {"nome": "Sem a camada de cima do encoder", "familia": "camada", "nivel": 1},
}
PESOS_SOLTOS = ("magnitude", "gradual")   # as duas formas da poda por magnitude
PECAS_INTEIRAS = ("linhas", "camada")     # a poda estruturada
SEM_ANTES = ("gradual",)                  # na poda gradual não existe "podado sem reajuste": o treino corre junto


# --------------------------------------------------------------------------
# Quem é podado, e com que sinal cada um é reajustado
# --------------------------------------------------------------------------


def modelos_a_podar() -> dict[str, dict]:
    """Os modelos de partida, na ordem em que as execuções rodam: o aluno principal, o teto, os outros alunos.

    Para cada um: `nome`, `pasta` (com `semente_0`, `semente_1`, ...), `config` e
    `receita` com que foi treinado, `previsoes` (onde a etapa de origem gravou
    as respostas de cada semente) e `variante_da_etapa_6` (None para o teto).
    Enquanto a Etapa 6 não entregar a sua decisão, só existe o teto.
    """
    rede = json.loads((ENTREGAS / DECISAO_DA_REDE).read_text(encoding="utf-8"))
    teto = {
        "nome": "Teto (rótulos humanos)", "curto": "Teto", "pasta": RAIZ / rede["modelos"], "config": rede["config"], "receita": rede["receita"],
        "previsoes": str(ENTREGAS / "04b_intencao" / "previsoes" / "semente_{semente}_{particao}.parquet"),
        "variante_da_etapa_6": None, "principal": False,
    }
    arquivo = ENTREGAS / DECISAO_DA_DESTILACAO
    if not arquivo.exists():
        return {"teto": teto}
    destilacao = json.loads(arquivo.read_text(encoding="utf-8"))
    alunos = {}
    for chave in destilacao["seguem"]:
        aluno = destilacao["alunos"][chave]
        alunos[aluno["variante"]] = {
            "nome": aluno["nome"], "curto": f"Aluno {chave}", "pasta": RAIZ / aluno["modelos"], "config": aluno["config"], "receita": aluno["receita"],
            "previsoes": str(arquivo.parent / "previsoes" / (aluno["variante"] + "_semente_{semente}_{particao}.parquet")),
            "variante_da_etapa_6": aluno["variante"], "principal": chave == destilacao["principal"],
        }
    principais = {v: a for v, a in alunos.items() if a["principal"]}
    return {**principais, "teto": teto, **{v: a for v, a in alunos.items() if not a["principal"]}}


def dados_do_modelo(modelo: str) -> dict:
    """O treino e o `dev` de parada do modelo: os mesmos com que ele foi treinado.

    Para um aluno são os da Etapa 6 (`etapa6.dados_da_variante`): rótulos do
    professor no treino e no `dev`, e os logits quando o aluno aprende com eles.
    """
    variante = modelos_a_podar()[modelo]["variante_da_etapa_6"]
    if variante is None:
        return {**execucao.dados_humanos(), "origem": {"treino": "rótulos humanos de treino", "dev": "rótulos humanos de dev"}}
    return etapa6.dados_da_variante(variante)


def receita_do_reajuste(modelo: str) -> Receita:
    """A receita com que o modelo foi treinado, com o número de épocas do reajuste."""
    return Receita(**{**modelos_a_podar()[modelo]["receita"], "epocas": EPOCAS})


# --------------------------------------------------------------------------
# Uma execução: podar, reajustar, gravar
# --------------------------------------------------------------------------


def pasta_da_execucao(modelo: str, variante: str, semente: int) -> Path:
    return execucao.pasta_da_execucao(f"{ETAPA}/{modelo}", variante, semente)


@lru_cache(maxsize=None)
def _digital_do_arquivo(caminho: str, modificado_em: int, tamanho: int) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:16]


def digital_de_partida(modelo: str, semente: int) -> str:
    """A impressão digital dos pesos do modelo de partida: muda se o modelo for treinado de novo."""
    pesos = modelos_a_podar()[modelo]["pasta"] / f"semente_{semente}" / "pesos.pt"
    estado = pesos.stat()
    return _digital_do_arquivo(str(pesos), estado.st_mtime_ns, estado.st_size)


def pronta(modelo: str, variante: str, semente: int) -> bool:
    """A execução está gravada e partiu dos pesos que o modelo de partida tem hoje?"""
    meta = pasta_da_execucao(modelo, variante, semente) / "meta.json"
    return meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("digital_de_partida") == digital_de_partida(modelo, semente)


def _respostas_na_cpu(sistema: SistemaDiet, particoes: dict[str, pd.DataFrame]) -> dict[str, list[Previsao]]:
    """As respostas saem sempre da CPU, que é onde o modelo vai rodar: não dependem da placa em que o reajuste rodou."""
    de_onde = sistema.dispositivo
    sistema.modelo.to("cpu")
    sistema.dispositivo = "cpu"
    try:
        return {nome: sistema.prever(list(particao["utt"])) for nome, particao in particoes.items()}
    finally:
        sistema.modelo.to(de_onde)
        sistema.dispositivo = de_onde


def executar(modelo: str, variante: str, semente: int, dispositivo: str | None = None) -> Path:
    """Poda e reajusta o modelo com a semente dada, ou devolve a pasta se já rodou.

    Grava o modelo podado e reajustado (uma pasta comum de `SistemaDiet`), o
    histórico do reajuste e as respostas em `dev` e no teste, antes e depois
    do reajuste. Uma execução gravada a partir de outros pesos (o modelo de
    partida foi treinado de novo) é apagada e refeita.
    """
    pasta = pasta_da_execucao(modelo, variante, semente)
    if pronta(modelo, variante, semente):
        return pasta
    if pasta.exists():
        shutil.rmtree(pasta)

    de_partida, propria = modelos_a_podar()[modelo], VARIANTES[variante]
    familia, fracao = propria["familia"], propria["nivel"] / 100
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    lido = dados_do_modelo(modelo)
    treino, dev = lido["treino"], lido["dev"]
    particoes, _ = massive.carregar_entrega()
    medir = {"dev": particoes["dev"], "test": particoes["test"]}
    receita = receita_do_reajuste(modelo)

    sistema = SistemaDiet.carregar(de_partida["pasta"] / f"semente_{semente}", dispositivo)
    if sistema.modelo.config != Config(**de_partida["config"]):
        raise ValueError(f"O modelo em {de_partida['pasta']} não tem a configuração anotada na decisão da sua etapa.")
    de_partida_contagem = poda.contar(sistema.modelo)

    inicio = time.perf_counter()
    mascaras = agenda = None
    if familia == "magnitude":
        mascaras = poda.mascaras_por_magnitude(sistema.modelo, fracao)
        poda.aplicar(sistema.modelo, mascaras)
    elif familia == "gradual":
        agenda = poda.Agenda(fracao, EPOCAS_DE_RAMPA, PASSOS_ENTRE_CORTES)
    elif familia == "linhas":
        importancia = poda.importancia_das_linhas(sistema, treino)
        sistema = poda.remover_linhas(sistema, poda.linhas_que_ficam(importancia, fracao))
    elif familia == "camada":
        sistema = poda.remover_camada(sistema)

    tem_antes = familia not in SEM_ANTES
    antes = _respostas_na_cpu(sistema, medir) if tem_antes else {}
    dev_antes = qualidade.avaliar(ouro(dev), sistema.prever(list(dev["utt"]))) if tem_antes else None
    historico = poda.reajustar(sistema, treino, dev, receita, semente=semente, mascaras=mascaras, agenda=agenda)
    duracao = time.perf_counter() - inicio
    depois = _respostas_na_cpu(sistema, medir)

    contagem = poda.contar(sistema.modelo)
    if familia in PESOS_SOLTOS:   # a esparsidade pedida tem de ser a obtida: se a máscara vazou, a execução não é gravada
        pedidos = sum(round(fracao * sistema.modelo.get_parameter(nome).numel()) for nome in poda.nomes_podaveis(sistema.modelo))
        em_zero = contagem["podaveis"] - contagem["nao_nulos_podaveis"]
        if not pedidos <= em_zero <= pedidos + 10:   # a folga cobre um peso que o treino leve a zero por acaso
            raise RuntimeError(f"Pedidos {pedidos} pesos em zero; o modelo tem {em_zero}.")

    pasta.mkdir(parents=True, exist_ok=True)
    historico.to_csv(pasta / "historico.csv", index=False)
    for particao, tabela in medir.items():
        salvar_previsoes(pasta / f"previsoes_{particao}.parquet", tabela["id"], depois[particao])
        if tem_antes:
            salvar_previsoes(pasta / f"previsoes_{particao}_antes.parquet", tabela["id"], antes[particao])
    sistema.salvar(pasta / "modelo")
    meta = {
        "etapa": ETAPA,
        "modelo": modelo,
        "variante": variante,
        "nome": propria["nome"],
        "familia": familia,
        "nivel": propria["nivel"],
        "semente": semente,
        "de_partida": str((de_partida["pasta"] / f"semente_{semente}").relative_to(RAIZ)),
        "digital_de_partida": digital_de_partida(modelo, semente),
        "config": sistema.modelo.config.como_dict(),
        "receita": receita.como_dict(),
        "poda_gradual": {"epocas_de_rampa": EPOCAS_DE_RAMPA, "passos_entre_cortes": PASSOS_ENTRE_CORTES} if agenda else None,
        "melhor_epoca": int(historico.loc[historico["melhor"], "epoca"].iloc[0]),
        "dev": qualidade.avaliar(ouro(dev), sistema.prever(list(dev["utt"]))),   # no dev que escolheu a época de parada
        "dev_antes": dev_antes,                                                  # no mesmo dev, antes do reajuste
        "contagem_de_partida": de_partida_contagem,
        "contagem": contagem,
        "features": len(sistema.featurizador),
        "exemplos_de_treino": len(treino),
        "exemplos_de_dev": len(dev),
        "origem": lido.get("origem"),
        "segundos": duracao,
        "dispositivo": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # por último: marca "pronto"
    return pasta


def carregar(modelo: str, variante: str, semente: int) -> dict:
    """O que uma execução produziu: meta, histórico e as respostas (`dev`, `test` e, quando existem, `dev_antes`, `test_antes`)."""
    pasta = pasta_da_execucao(modelo, variante, semente)
    previsoes = {
        arquivo.stem.removeprefix("previsoes_"): carregar_previsoes(arquivo)[1] for arquivo in sorted(pasta.glob("previsoes_*.parquet"))
    }
    return {
        "pasta": pasta,
        "meta": json.loads((pasta / "meta.json").read_text(encoding="utf-8")),
        "historico": pd.read_csv(pasta / "historico.csv"),
        "previsoes": previsoes,
    }


def grade(modelos=None, variantes=None) -> list[tuple[str, str, int]]:
    """Todas as execuções (modelo, variante, semente) da etapa, na ordem em que devem rodar."""
    return [(m, v, s) for m in (modelos or modelos_a_podar()) for v in (variantes or VARIANTES) for s in SEMENTES]


def faltam(modelos=None, variantes=None) -> list[tuple[str, str, int]]:
    """As execuções que ainda não estão gravadas, ou que partiram de pesos que o modelo de partida já não tem."""
    return [(m, v, s) for m, v, s in grade(modelos, variantes) if not pronta(m, v, s)]


# --------------------------------------------------------------------------
# A regra de decisão da etapa
# --------------------------------------------------------------------------


def quem_segue(gabarito: list[Previsao], sem_poda: list[list[Previsao]], candidatos: dict[str, dict]) -> dict:
    """A regra do que segue para a quantização (Etapa 8), para um modelo de partida.

    `gabarito` são os rótulos humanos de `dev`; `sem_poda`, as respostas do
    modelo de partida em `dev`, uma lista por semente. `candidatos` traz, para
    cada variante podada e reajustada, `em_dev` (as respostas de cada semente),
    `nao_nulos` (parâmetros diferentes de zero) e `bytes` (o tamanho medido do
    modelo no arquivo comum, que grava todos os pesos, zeros inclusive).

    1. Uma variante **passa** se o exact match dela em `dev`, depois do
       reajuste, não for distinguivelmente menor que o do modelo sem poda: o
       intervalo de 95% da diferença pareada (média das sementes) não fica todo
       abaixo de zero.
    2. A **principal** é a variante que passa com o menor arquivo comum. É
       esse arquivo que a quantização e uma execução comum consomem, e só
       tirar peças inteiras (linhas da tabela, camada) o diminui. Empate: a de
       maior exact match. Se nenhuma peça inteira passa, a principal é o
       modelo sem poda (`None`).
    3. A **secundária** é a dos **pesos soltos** (magnitude de uma vez e
       gradual) que passa com a maior esparsidade. Se as duas formas passam
       nessa esparsidade, a de maior exact match; se empatam, a de uma vez,
       que é a mais simples. O ganho dela depende de um formato que aproveite
       os zeros; segue para a Etapa 8 medir o que a quantização e a compressão
       fazem com ela. Se nenhuma passa, não há secundária (`None`).

    Devolve `principal`, `secundaria`, `seguem` (as variantes podadas que
    seguem, na ordem principal, secundária; pode ser vazia) e `tabela`, com a
    diferença e o intervalo de cada candidata.
    """
    linhas = []
    for variante, candidato in candidatos.items():
        comparacao = qualidade.comparar_grupos(gabarito, sem_poda, candidato["em_dev"]).loc[CRITERIO]
        linhas.append({
            "variante": variante, "familia": VARIANTES[variante]["familia"], "nivel": VARIANTES[variante]["nivel"],
            "nao_nulos": candidato["nao_nulos"], "bytes": candidato["bytes"], "sem_poda": comparacao["a"], "podado": comparacao["b"],
            "diferenca": comparacao["diferenca"], "ic_min": comparacao["ic_min"], "ic_max": comparacao["ic_max"],
            "passa": bool(not comparacao["ic_max"] < 0),
        })
    tabela = pd.DataFrame(linhas).set_index("variante")
    passam = tabela[tabela["passa"]]

    pecas = passam[passam["familia"].isin(PECAS_INTEIRAS)]
    principal = min(pecas.index, key=lambda v: (pecas.loc[v, "bytes"], -pecas.loc[v, "podado"])) if len(pecas) else None
    soltos = passam[passam["familia"].isin(PESOS_SOLTOS)]
    secundaria = (
        max(soltos.index, key=lambda v: (soltos.loc[v, "nivel"], soltos.loc[v, "podado"], soltos.loc[v, "familia"] == "magnitude"))
        if len(soltos) else None
    )
    return {"principal": principal, "secundaria": secundaria, "seguem": [v for v in (principal, secundaria) if v], "tabela": tabela}


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 7: uma execução de poda e reajuste.")
    argumentos.add_argument("--modelo", help="o modelo de partida: teto ou a variante de um aluno da Etapa 6")
    argumentos.add_argument("--variante", choices=sorted(VARIANTES))
    argumentos.add_argument("--semente", type=int)
    argumentos.add_argument("--dispositivo", default=None, help="cuda ou cpu; o padrão é a placa, se houver")
    argumentos.add_argument("--faltam", action="store_true", help="lista as execuções que ainda não rodaram, uma por linha")
    a = argumentos.parse_args()
    if a.faltam:
        for modelo, variante, semente in faltam():
            print(f"--modelo {modelo} --variante {variante} --semente {semente}")
        return
    if a.modelo is None or a.variante is None or a.semente is None:
        argumentos.error("informe --modelo, --variante e --semente (ou --faltam)")
    if a.modelo not in modelos_a_podar():
        argumentos.error(f"modelo desconhecido: {a.modelo}. Os que existem: {', '.join(modelos_a_podar())}")
    pasta = executar(a.modelo, a.variante, a.semente, a.dispositivo)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    antes = "" if meta["dev_antes"] is None else f"exact match antes {meta['dev_antes'][CRITERIO]:.4f}, "
    print(a.modelo, a.variante, a.semente, f"época {meta['melhor_epoca']}", f"{antes}depois {meta['dev'][CRITERIO]:.4f}",
          f"não nulos {meta['contagem']['nao_nulos']}", f"{meta['segundos']:.0f}s")


if __name__ == "__main__":
    _linha_de_comando()
