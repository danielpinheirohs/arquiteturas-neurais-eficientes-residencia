"""Etapa 6: destilação. Cada sistema treina com um sinal diferente.

Todos têm a rede e a receita decididas na Etapa 4b
(`entregas/04b_intencao/decisao.json`). O que muda de um para o outro é só
de onde vêm os rótulos do treino:

    piso          os 299 rótulos humanos que as LLMs veem, e mais nada
    aluno_a       os rótulos do professor fechado (Etapa 5a)
    aluno_b1      os rótulos do professor aberto (Etapa 5b)
    aluno_b2_tN   as entidades dos rótulos do professor aberto e, na intenção,
                  as probabilidades dele, com temperatura N
                  (Hinton et al., arXiv 1503.02531)

Há ainda um controle, `aluno_b1_sem_suavizacao`: o aluno B1 sem a suavização
de rótulos da Etapa 4b. O B2 difere do B1 em duas coisas (entram os logits,
sai a suavização); do controle ele difere só nos logits.

O teto, treinado com os 11.514 rótulos humanos, já existe em
`entregas/04b_intencao/` e não é treinado de novo.

    uv run python -m especialista.experimentos.etapa6 --variante aluno_a --semente 0

**Nenhum rótulo humano entra no treino de um aluno.** O aluno treina com os
rótulos do professor em todos os pedidos do treino, e a época de parada é
escolhida no `dev` rotulado pelo mesmo professor. Os rótulos humanos de
`dev` e de teste só aparecem depois, na análise (notebook da etapa).

As duas regras de decisão da etapa, escritas antes de olhar os resultados,
estão em `escolher_temperatura` e `quem_segue`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.dados import massive
from especialista.experimentos import execucao
from especialista.professores.rotulagem import rotular

ETAPA = "06_destilacao"
SEMENTES = (0, 1, 2)
TEMPERATURAS = (1, 2, 4, 8)   # a grade do aluno B2, em volta de T = 4, onde a Etapa 5b mediu o professor aberto mais calibrado

# Onde cada professor deixou os rótulos. `previsoes` é o prefixo dos arquivos de dev e teste.
PROFESSORES: dict[str, dict] = {
    "fechado": {"nome": "Claude Haiku 5.5", "etapa": "05a_professor_fechado", "previsoes": "claude"},
    "aberto": {"nome": "Qwen3-8B", "etapa": "05b_professor_aberto", "previsoes": "qwen3_8b"},
}
DEMONSTRACOES = "05a_professor_fechado/demonstracoes.parquet"   # os 299 exemplos com rótulo humano
DECISAO_DA_REDE = "04b_intencao/decisao.json"

# Cada variante diz de que professor vêm os rótulos (None: rótulos humanos), se a intenção usa os
# logits dele e o que muda na configuração da Etapa 4b.
VARIANTES: dict[str, dict] = {
    "piso": {
        "nome": "Piso do cenário (só os 299 rótulos humanos)",
        "professor": None, "logits": False, "config": {}, "receita": {},
    },
    "aluno_a": {
        "nome": "Aluno A (rótulos do professor fechado)",
        "professor": "fechado", "logits": False, "config": {}, "receita": {},
    },
    "aluno_b1": {
        "nome": "Aluno B1 (rótulos do professor aberto)",
        "professor": "aberto", "logits": False, "config": {}, "receita": {},
    },
    "aluno_b1_sem_suavizacao": {
        "nome": "Aluno B1 sem suavização de rótulos (controle do B2)",
        "professor": "aberto", "logits": False, "config": {"suavizacao": 0.0}, "receita": {},
    },
    **{
        f"aluno_b2_t{T}": {
            "nome": f"Aluno B2 (logits do professor aberto, temperatura {T})",
            "professor": "aberto", "logits": True,
            # o alvo da intenção já é uma distribuição: a suavização de rótulos da Etapa 4b sai
            "config": {"temperatura": float(T), "peso_rotulo": 0.0, "suavizacao": 0.0}, "receita": {},
        }
        for T in TEMPERATURAS
    },
}


def variante_do_b2(temperatura: int) -> str:
    return f"aluno_b2_t{temperatura}"


def definicao(variante: str) -> dict:
    """A rede e a receita da Etapa 4b, com o que a variante troca."""
    partida = json.loads((ENTREGAS / DECISAO_DA_REDE).read_text(encoding="utf-8"))
    propria = VARIANTES[variante]
    return {
        "nome": propria["nome"],
        "config": {**partida["config"], **propria["config"]},
        "receita": {**partida["receita"], **propria["receita"]},
    }


# --------------------------------------------------------------------------
# De onde vêm os rótulos
# --------------------------------------------------------------------------


def _relativo(caminho: Path) -> str:
    return str(caminho.relative_to(RAIZ))


def _ler_rotulado(caminho: Path) -> pd.DataFrame:
    dados = pd.read_parquet(caminho)
    for coluna in ("tokens", "tags"):  # o Parquet devolve arrays; o projeto usa listas
        dados[coluna] = dados[coluna].map(list)
    return dados


def _mesmos_pedidos(ids, particao: pd.DataFrame, arquivo: Path) -> None:
    """Rótulo de um pedido aplicado a outro é um erro que não aparece sozinho: conferir sempre."""
    if list(ids) != particao["id"].tolist():
        raise ValueError(f"{arquivo} não traz os mesmos pedidos, na mesma ordem, da partição da Etapa 1.")


def pasta_do_professor(professor: str) -> Path:
    return ENTREGAS / PROFESSORES[professor]["etapa"]


def previsoes_do_professor(professor: str, particao: str, dados: pd.DataFrame) -> list[Previsao]:
    """As respostas do professor em `dev` ou `test`, conferidas contra os pedidos de `dados`."""
    arquivo = pasta_do_professor(professor) / "previsoes" / f"{PROFESSORES[professor]['previsoes']}_{particao}.parquet"
    ids, previsoes = carregar_previsoes(arquivo)
    _mesmos_pedidos(ids, dados, arquivo)
    return previsoes


def treino_do_professor(professor: str, treino: pd.DataFrame) -> pd.DataFrame:
    """O treino com os rótulos do professor no lugar dos humanos. A coluna `valido` marca quem tem rótulo aproveitável."""
    arquivo = pasta_do_professor(professor) / "rotulos_treino.parquet"
    rotulado = _ler_rotulado(arquivo)
    _mesmos_pedidos(rotulado["id"], treino, arquivo)
    return rotulado


def dev_do_professor(professor: str, dev: pd.DataFrame, intents: list[str]) -> pd.DataFrame:
    """`dev` com os rótulos do professor no lugar dos humanos: é ele que escolhe a época de parada do aluno."""
    return rotular(dev, previsoes_do_professor(professor, "dev", dev), intents)


def logits_do_professor(particao: str, linhas: int, intents: list[str]) -> np.ndarray:
    """As log-probabilidades que o professor aberto deu a cada intenção: uma linha por pedido, uma coluna por intenção.

    `particao` é `treino`, `dev` ou `test`. As colunas têm de estar na ordem de `intents`.
    """
    pasta = pasta_do_professor("aberto")
    arquivo = pasta / f"logits_{particao}.npy"
    logits = np.load(arquivo)
    if logits.shape != (linhas, len(intents)):
        raise ValueError(f"{arquivo} tem forma {logits.shape}; o esperado é {(linhas, len(intents))}.")
    if np.isnan(logits).any():
        raise ValueError(f"{arquivo} tem valores que não são números.")
    ordem = pasta / "intencoes.json"
    if ordem.exists() and json.loads(ordem.read_text(encoding="utf-8")) != list(intents):
        raise ValueError(f"A ordem das colunas em {ordem} não é a de `rotulos['intents']`.")
    return logits


def dados_da_variante(variante: str) -> dict:
    """O que `execucao.executar` precisa para treinar a variante (o formato está descrito lá)."""
    particoes, rotulos = massive.carregar_entrega()
    treino, dev = particoes["train"], particoes["dev"]
    professor = VARIANTES[variante]["professor"]

    if professor is None:
        # Piso: tudo o que o sistema sabe vem dos 299 exemplos, inclusive o vocabulário de features.
        # A época de parada usa o dev humano, como no teto: uma vantagem que o cenário não daria.
        arquivo = ENTREGAS / DEMONSTRACOES
        return {
            "treino": _ler_rotulado(arquivo), "dev": dev, "rotulos": rotulos,
            "origem": {"treino": f"rótulos humanos: {_relativo(arquivo)}", "dev": "rótulos humanos de dev",
                       "vocabulario": "os exemplos do próprio treino"},
        }

    pasta = pasta_do_professor(professor)
    do_treino = treino_do_professor(professor, treino)
    do_dev = dev_do_professor(professor, dev, rotulos["intents"])
    origem = {
        "treino": f"rótulos do professor {professor}: {_relativo(pasta / 'rotulos_treino.parquet')}",
        "dev": f"rótulos do professor {professor}: {_relativo(pasta / 'previsoes')}/{PROFESSORES[professor]['previsoes']}_dev.parquet",
        "vocabulario": "todos os pedidos do treino, sem rótulo",
        "sem_rotulo_aproveitavel": {"treino": int((~do_treino["valido"]).sum()), "dev": int((~do_dev["valido"]).sum())},
    }
    if VARIANTES[variante]["logits"]:
        logits = logits_do_professor("treino", len(treino), rotulos["intents"])
        do_treino["logits_professor"] = list(logits)   # um vetor por pedido; segue para o lote em `SistemaDiet.lote`
        origem["logits"] = _relativo(pasta / "logits_treino.npy")
    return {
        # pedidos sem rótulo aproveitável não têm como ensinar nem como julgar: ficam fora
        "treino": do_treino[do_treino["valido"]].reset_index(drop=True),
        "dev": do_dev[do_dev["valido"]].reset_index(drop=True),
        "rotulos": rotulos,
        "vocabulario": treino,   # o texto dos pedidos não depende de rótulo: todos os alunos e o teto têm o mesmo vocabulário
        "avaliar": dev,          # as respostas gravadas cobrem dev inteiro, para a análise
        "origem": origem,
    }


# --------------------------------------------------------------------------
# Executar e ler
# --------------------------------------------------------------------------


def executar(variante: str, semente: int, dispositivo: str | None = None) -> Path:
    """Treina a variante com a semente dada, ou devolve a pasta se já rodou."""
    return execucao.executar(ETAPA, variante, semente, definicao(variante), dispositivo, dados=lambda: dados_da_variante(variante))


def carregar(variante: str, semente: int) -> dict:
    return execucao.carregar(ETAPA, variante, semente)


def faltam(variantes=None) -> list[tuple[str, int]]:
    """As execuções (variante, semente) que ainda não estão gravadas."""
    return [
        (variante, semente)
        for variante in (variantes or VARIANTES)
        for semente in SEMENTES
        if not (execucao.pasta_da_execucao(ETAPA, variante, semente) / "meta.json").exists()
    ]


# --------------------------------------------------------------------------
# O aluno ao lado do professor
# --------------------------------------------------------------------------

CASOS = ("os dois acertam", "só o professor acerta", "só o aluno acerta", "erram com a mesma resposta", "erram com respostas diferentes")


def aluno_e_professor(
    gabarito: list[Previsao], do_professor: list[Previsao], do_aluno: list[Previsao], parte: str = "intencao"
) -> pd.Series:
    """Em que parte dos pedidos o aluno acerta ou erra junto com o professor.

    `parte` é `intencao` (só a intenção) ou `tudo` (a intenção e todas as
    entidades). Cada pedido cai em um dos cinco casos de `CASOS`, e as cinco
    partes somam 1:

    - `só o professor acerta`: o que o aluno perdeu ao imitar;
    - `só o aluno acerta`: o aluno corrigiu o professor;
    - `erram com a mesma resposta`: o aluno copiou o erro.
    """
    if not len(gabarito) == len(do_professor) == len(do_aluno):
        raise ValueError("Gabarito, professor e aluno precisam ter os mesmos pedidos.")
    resposta = (lambda p: p.intent) if parte == "intencao" else (lambda p: (p.intent, frozenset(p.slots)))
    contagem = dict.fromkeys(CASOS, 0)
    for certo, professor, aluno in zip(map(resposta, gabarito), map(resposta, do_professor), map(resposta, do_aluno)):
        if professor == certo:
            contagem["os dois acertam" if aluno == certo else "só o professor acerta"] += 1
        elif aluno == certo:
            contagem["só o aluno acerta"] += 1
        else:
            contagem["erram com a mesma resposta" if aluno == professor else "erram com respostas diferentes"] += 1
    return pd.Series(contagem, name="parte dos pedidos") / len(gabarito)


# --------------------------------------------------------------------------
# As regras de decisão da etapa
# --------------------------------------------------------------------------


def escolher_temperatura(gabarito: list[Previsao], em_dev: dict[int, list[list[Previsao]]]) -> tuple[int, pd.DataFrame]:
    """A regra de escolha da temperatura do aluno B2.

    `em_dev` traz, para cada temperatura, as previsões em `dev` de cada
    semente; `gabarito` são os rótulos humanos de `dev`. Fica a temperatura de
    maior acurácia de intenção, na média das sementes: os logits só falam de
    intenção. Em caso de empate exato, a de maior exact match; se o empate
    continuar, a menor temperatura.

    Devolve a escolhida e a tabela com a média de cada métrica por temperatura.
    """
    medias = pd.DataFrame(
        {T: pd.DataFrame([qualidade.avaliar(gabarito, previsoes) for previsoes in grupo]).mean() for T, grupo in em_dev.items()}
    ).T.rename_axis("temperatura")
    escolhida = max(medias.index, key=lambda T: (medias.loc[T, "intent_acc"], medias.loc[T, "exact_match"], -T))
    return escolhida, medias


def quem_segue(gabarito: list[Previsao], em_dev: dict[str, list[list[Previsao]]]) -> dict:
    """A regra do que segue para a poda (Etapa 7).

    `em_dev` traz as previsões em `dev` de cada semente dos alunos `A`, `B1` e
    `B2` (este na temperatura escolhida); `gabarito` são os rótulos humanos.

    1. O aluno A segue: é o único do cenário em que o professor só entrega rótulos.
    2. Do professor aberto segue um aluno só. É o B2 se a acurácia de intenção
       dele for distinguivelmente maior que a do B1 (intervalo de 95% da
       diferença pareada todo acima de zero) e o exact match não for
       distinguivelmente menor. Senão é o B1, que é mais simples: não precisa
       dos logits.
    3. O principal, que a Etapa 7 poda primeiro, é o de maior exact match
       entre os que seguem.

    O B1 da regra é o da receita da Etapa 4b, com suavização de rótulos. O
    controle sem suavização não concorre: serve para dizer de onde vem a
    diferença, não para escolher o modelo.
    """
    b2_menos_b1 = qualidade.comparar_grupos(gabarito, em_dev["B1"], em_dev["B2"])
    b2_passa = bool(b2_menos_b1.loc["intent_acc", "ic_min"] > 0 and not b2_menos_b1.loc["exact_match", "ic_max"] < 0)
    seguem = ["A", "B2" if b2_passa else "B1"]
    exact_match = {a: float(np.mean([qualidade.avaliar(gabarito, p)["exact_match"] for p in em_dev[a]])) for a in seguem}
    return {
        "seguem": seguem,
        "principal": max(seguem, key=exact_match.get),
        "b2_passa": b2_passa,
        "b2_menos_b1": b2_menos_b1,
        "exact_match_em_dev": exact_match,
    }


if __name__ == "__main__":
    execucao.linha_de_comando(ETAPA, VARIANTES, executar)
