"""Etapa 11b: os alunos do cenário B com o professor aberto maior.

A pergunta: um professor aberto melhor faz alunos melhores? Quanto do ganho
do professor chega ao aluno, e com ele os logits passam a valer mais que os
rótulos?

A Etapa 11a trocou o professor aberto: saiu o Qwen3-8B em 4 bits (Etapa 5b)
e entrou o Qwen3-14B em float16. Aqui os alunos do cenário B são treinados
de novo com ele, nos dois caminhos, **com as receitas dos alunos antigos e
sem nova busca**. A única coisa que muda é o professor:

    caminho   alunos                                           a receita vem de
    zero      aluno_b1, aluno_b1_sem_suavizacao, aluno_b2_tN   Etapa 6  (`etapa6.definicao`)
    xlmr      aluno_b1, aluno_b2_tN                            Etapa 10a (`entregas/10a_encoder/decisao.json`)

**O professor é um parâmetro.** Toda função e todo comando recebem a versão
do professor (o id da Etapa 11a, como `qwen3_14b__fp16__colab_a100`); o
padrão é `PROFESSOR_NOVO`. O nome de cada variante diz o caminho, o aluno e
o professor (`zero__aluno_b1__qwen3_14b__fp16__colab_a100`), e o que é de
uma rodada e não é execução (regras, temperatura, contas, registros) mora
na pasta do professor:

    data/execucoes/11b_alunos_do_professor_maior/
        <caminho>__<aluno>__<professor>/semente_<n>/    uma execução de treino
        <professor>/regras.json                         as regras, gravadas antes dos treinos desse professor
        <professor>/temperatura_<caminho>.json          a decisão em dev e o que foi aberto no teste
        <professor>/calculo/, <professor>/rodadas/      as contas e o registro de cada rodada
        artigos/                                        as referências conferidas no arXiv

Assim uma rodada com outro professor entra depois sem tocar nos alunos de
nenhum outro. Só ganha alunos o professor que tomou o lugar do titular na
escada da Etapa 11a.

**Nada das Etapas 6 e 10a é sobrescrito.** O professor só é trocado dentro
de `com_professor`, que devolve o antigo na saída.

**As regras vêm antes dos treinos.** `regras` grava `regras.json` com a
hora; sem esse arquivo nada treina. A única decisão da etapa é a
temperatura do aluno B2, em `dev`, pela regra da Etapa 6. As comparações
(`lista_de_comparacoes`, `lista_de_diferencas`, `lista_de_repasses`) estão
fechadas no mesmo arquivo.

    uv run python -m especialista.experimentos.etapa11b regras       # antes de tudo
    uv run python -m especialista.experimentos.etapa11b artigos      # confere as referências no arXiv
    uv run python -m especialista.experimentos.etapa11b rodada       # todos os treinos: a rede do zero em ondas, depois o XLM-R em fila
    uv run python -m especialista.experimentos.etapa11b temperatura  # depois da rodada: a regra em dev e a abertura do teste
    uv run python -m especialista.experimentos.etapa11b calcular     # as comparações, gravadas no cache
    uv run python -m especialista.experimentos.etapa11b manifesto    # depois do notebook: o que a entrega contém

Com outro professor, os mesmos comandos, na mesma ordem, com `--professor <versão da Etapa 11a>`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from especialista import caminhos
from especialista import treino as treino_do_zero
from especialista import treino_encoder
from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.dados import massive
from especialista.experimentos import etapa6, etapa10a, etapa11a, execucao
from especialista.modelos import diet, encoder
from especialista.modelos.encoder.sistema import bytes_em_disco

ETAPA = "11b_alunos_do_professor_maior"
PERGUNTA = ("Um professor aberto melhor faz alunos melhores? Quanto do ganho do professor chega ao aluno, "
            "e com ele os logits passam a valer mais que os rótulos?")
SEMENTES = (0, 1, 2)
TEMPERATURAS = etapa6.TEMPERATURAS            # a grade do aluno B2 das Etapas 6 e 10a: 1, 2, 4 e 8
REAMOSTRAS = 2000                             # sorteios do bootstrap por enunciado, como em todas as etapas
TRES = ("intent_acc", "slot_f1", "exact_match")

# O professor da primeira rodada, e o padrão de toda função e de todo comando: o Qwen3-14B, que tomou o lugar do titular
# na escada da Etapa 11a. Outra versão da 11a entra por `professor=` (na linha de comando, `--professor`).
PROFESSOR_NOVO = "qwen3_14b__fp16__colab_a100"
PROFESSOR_ANTIGO = etapa11a.VERSAO_DA_5B            # o Qwen3-8B em 4 bits, que ensinou os alunos das Etapas 6 e 10a: a referência de toda rodada
FORA_DO_ESCOPO = "qwen3_8b__fp16__colab_l4"         # o degrau da Etapa 11a que não tomou o lugar do titular: sem alunos (ver `montar_regras`)

# Cada caminho: de que etapa vêm a receita e os alunos antigos, e quantos treinos cabem juntos na placa
# (medido nas Etapas 6 e 10a, na RTX 3080: 7 da rede do zero; do XLM-R, um por vez).
CAMINHOS: dict[str, dict] = {
    "zero": {"nome": "Rede do zero", "etapa": "06_destilacao", "processos": 7},
    "xlmr": {"nome": "XLM-R", "etapa": "10a_encoder", "processos": 1},
}
B2 = tuple(etapa6.variante_do_b2(T) for T in TEMPERATURAS)
ALUNOS: dict[str, tuple[str, ...]] = {
    "zero": ("aluno_b1", "aluno_b1_sem_suavizacao", *B2),   # os alunos do professor aberto da Etapa 6, com o controle do B2
    "xlmr": ("aluno_b1", *B2),                              # os da Etapa 10a (lá o B1 já não tem suavização: não há controle)
}


def pasta_da_etapa() -> Path:
    return execucao.EXECUCOES / ETAPA


def pasta_da_rodada(professor: str = PROFESSOR_NOVO) -> Path:
    """Onde mora o que é de um professor e não é execução: as regras, a temperatura, as contas e os registros das rodadas."""
    return pasta_da_etapa() / professor


def variante(caminho: str, aluno: str, professor: str = PROFESSOR_NOVO) -> str:
    """O nome da variante diz o caminho, o aluno e o professor: `zero__aluno_b1__qwen3_14b__fp16__colab_a100`."""
    if aluno not in ALUNOS[caminho]:
        raise KeyError(f"O caminho {caminho} não tem o aluno {aluno}.")
    return f"{caminho}__{aluno}__{professor}"


def pasta_da_execucao(caminho: str, aluno: str, semente: int, professor: str = PROFESSOR_NOVO) -> Path:
    return execucao.pasta_da_execucao(ETAPA, variante(caminho, aluno, professor), semente)


def _ler_json(caminho: Path) -> dict | None:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def _gravar_json(caminho: Path, conteudo: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _agora() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


# --------------------------------------------------------------------------
# A troca de professor
# --------------------------------------------------------------------------


@contextmanager
def com_professor(versao: str):
    """Dentro do bloco, o professor aberto das Etapas 6 e 10a é a versão dada; na saída volta o de antes.

    As Etapas 6 e 10a leem o professor por `etapa6.PROFESSORES["aberto"]`.
    Trocar essa entrada de vez faria qualquer código que rodasse depois (um
    notebook antigo, um teste) ler o professor novo sem avisar. Por isso a
    troca vale só dentro do bloco, mesmo que ele termine com erro.
    """
    antes = etapa6.PROFESSORES["aberto"]
    etapa6.PROFESSORES["aberto"] = etapa11a.como_professor(versao)
    try:
        yield etapa6.PROFESSORES["aberto"]
    finally:
        etapa6.PROFESSORES["aberto"] = antes


def nome_do_professor(versao: str) -> str:
    return etapa11a.descrever(versao)["nome"]


# --------------------------------------------------------------------------
# As receitas (as dos alunos antigos) e os dados (os do professor da rodada)
# --------------------------------------------------------------------------


def _nome(caminho: str, aluno: str, professor: str) -> str:
    quem = nome_do_professor(professor)
    if aluno == "aluno_b1":
        papel = f"Aluno B1 (rótulos do {quem})"
    elif aluno == "aluno_b1_sem_suavizacao":
        papel = f"Aluno B1 sem suavização de rótulos (controle do B2; rótulos do {quem})"
    else:
        papel = f"Aluno B2 (logits do {quem}, temperatura {aluno.rsplit('_t', 1)[1]})"
    return f"{CAMINHOS[caminho]['nome']} · {papel}"


def definicao(caminho: str, aluno: str, professor: str = PROFESSOR_NOVO) -> dict:
    """O nome, a configuração e a receita de um aluno: as do aluno de mesmo nome da etapa de origem, sem mudança.

    Rede do zero: `etapa6.definicao`, que parte de `entregas/04b_intencao/decisao.json`.
    XLM-R: a configuração e a receita que `entregas/10a_encoder/decisao.json`
    gravou para o aluno B1, com o que `etapa10a.VARIANTES` troca no B2 (a
    temperatura e o peso do rótulo). O professor só entra no nome.
    """
    variante(caminho, aluno, professor)   # confere que o aluno existe no caminho
    if caminho == "zero":
        antiga = etapa6.definicao(aluno)
        return {"nome": _nome(caminho, aluno, professor), "config": dict(antiga["config"]), "receita": dict(antiga["receita"])}
    b1 = json.loads((ENTREGAS / CAMINHOS["xlmr"]["etapa"] / "decisao.json").read_text(encoding="utf-8"))["alunos"]["B1"]
    return {"nome": _nome(caminho, aluno, professor), "config": {**b1["config"], **etapa10a.VARIANTES[aluno]["config"]}, "receita": dict(b1["receita"])}


def _classes(caminho: str) -> tuple[type, type]:
    """A configuração e a receita de cada caminho: completam a definição com os valores padrão, como o treino faz."""
    return (diet.Config, treino_do_zero.Receita) if caminho == "zero" else (encoder.Config, treino_encoder.Receita)


def receitas_contra_as_antigas(professor: str = PROFESSOR_NOVO) -> dict[str, bool | None]:
    """Para cada aluno, se a configuração e a receita daqui são as que o aluno antigo de mesmo nome treinou.

    A comparação é com o `meta.json` de cada execução antiga, no cache da
    etapa de origem (o que de fato rodou). None: o cache antigo não está no
    disco, e não há como conferir.
    """
    iguais: dict[str, bool | None] = {}
    for caminho in CAMINHOS:
        Config, Receita = _classes(caminho)
        for aluno in ALUNOS[caminho]:
            d = definicao(caminho, aluno, professor)
            daqui = {"config": Config(**d["config"]).como_dict(), "receita": Receita(**d["receita"]).como_dict()}
            metas = [_ler_json(execucao.EXECUCOES / CAMINHOS[caminho]["etapa"] / aluno / f"semente_{s}" / "meta.json") for s in SEMENTES]
            iguais[variante(caminho, aluno, professor)] = None if any(m is None for m in metas) else all(
                {"config": m["config"], "receita": m["receita"]} == daqui for m in metas)
    return iguais


def dados_do_aluno(caminho: str, aluno: str, professor: str = PROFESSOR_NOVO) -> dict:
    """O que o treino lê: o que a Etapa 6 (ou a 10a) montaria para o aluno, com o professor dado no lugar do antigo.

    Confere, antes de devolver, que os rótulos do treino, os de `dev` (que
    escolhem a época de parada) e os logits vieram mesmo da pasta desse
    professor: um aluno "novo" treinado com outro professor seria um erro
    que nenhuma métrica denuncia.
    """
    with com_professor(professor) as trocado:
        lido = etapa6.dados_da_variante(aluno) if caminho == "zero" else etapa10a.dados_da_variante(aluno)
    campos = ("treino", "dev", "logits") if etapa6.VARIANTES[aluno]["logits"] else ("treino", "dev")
    for campo in campos:
        if trocado["etapa"] not in lido["origem"].get(campo, ""):
            raise RuntimeError(f"{variante(caminho, aluno, professor)}: `{campo}` não veio do professor {professor} ({lido['origem'].get(campo)}).")
    lido["origem"] = {**lido["origem"], "professor": professor}
    return lido


# --------------------------------------------------------------------------
# Executar e ler
# --------------------------------------------------------------------------


def no_cache_de_verdade() -> bool:
    return execucao.EXECUCOES.resolve() == caminhos.EXECUCOES.resolve()


def exigir_regras(professor: str = PROFESSOR_NOVO) -> dict:
    """As regras gravadas para o professor. Sem elas nada treina e nada é calculado."""
    regras = _ler_json(pasta_da_rodada(professor) / "regras.json")
    if regras is None:
        raise FileNotFoundError(f"As regras de {professor} ainda não foram gravadas: rode o comando `regras` antes de qualquer treino.")
    if regras["professor_novo"]["versao"] != professor:
        raise ValueError(f"As regras em {pasta_da_rodada(professor)} são de outro professor ({regras['professor_novo']['versao']}).")
    return regras


def _ajustar_encoder(nome: str, semente: int, d: dict, dispositivo: str | None, dados) -> Path:
    """O ajuste fino do XLM-R como em `etapa10a.executar`, gravado na pasta desta etapa.

    `etapa10a.executar` só sabe gravar na pasta da Etapa 10a; por isso os
    passos dela estão repetidos aqui, com os mesmos campos em `meta.json`. O
    encoder nasce de `etapa10a.novo_sistema`, a mesma função de lá.
    """
    pasta = execucao.pasta_da_execucao(ETAPA, nome, semente)
    if (pasta / "meta.json").exists():
        return pasta
    if pasta.exists():   # sobra de uma execução interrompida
        shutil.rmtree(pasta)

    config, receita = encoder.Config(**d["config"]), treino_encoder.Receita(**d["receita"])
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    lido = dados()
    treino, dev, rotulos = lido["treino"], lido["dev"], lido["rotulos"]
    avaliar = lido.get("avaliar", dev)

    sistema = etapa10a.novo_sistema(rotulos, config, semente, dispositivo)
    partes = sistema.modelo.parametros_por_parte()
    linhas_usadas = len(sistema.alinhador.ids_usados(treino["tokens"]))
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inicio = time.perf_counter()
    historico = treino_encoder.treinar(sistema, treino, dev, receita, semente=semente)
    duracao = time.perf_counter() - inicio

    pasta.mkdir(parents=True, exist_ok=True)
    previsoes_gravadas = sistema.prever(list(avaliar["utt"]))
    no_dev_de_parada = previsoes_gravadas if avaliar is dev else sistema.prever(list(dev["utt"]))
    historico.to_csv(pasta / "historico.csv", index=False)
    salvar_previsoes(pasta / "previsoes_dev.parquet", avaliar["id"], previsoes_gravadas)
    sistema.salvar(pasta / "modelo")
    _gravar_json(pasta / "meta.json", {   # por último: marca "pronto"
        "etapa": ETAPA,
        "variante": nome,
        "nome": d["nome"],
        "semente": semente,
        "config": config.como_dict(),
        "receita": receita.como_dict(),
        "melhor_epoca": int(historico.loc[historico["melhor"], "epoca"].iloc[0]),
        "dev": qualidade.avaliar(ouro(dev), no_dev_de_parada),   # no conjunto que escolheu a época de parada
        "exemplos_de_treino": len(treino),
        "exemplos_de_dev": len(dev),
        "origem": lido["origem"],
        "passos_de_treino": int(historico["passos"].sum()),
        "segundos_de_treino": duracao,
        "segundos_por_epoca": float(historico["segundos"].diff().fillna(historico["segundos"]).median()),
        "pico_de_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if dispositivo == "cuda" else None,
        "parametros": sistema.parametros,
        "parametros_por_parte": partes,
        "linhas_da_tabela_no_treino": linhas_usadas,
        "bytes_em_disco": bytes_em_disco(pasta / "modelo"),
        "dispositivo": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu",
        "torch": torch.__version__,
    })
    return pasta


def executar(
    caminho: str, aluno: str, semente: int, dispositivo: str | None = None, receita_curta: dict | None = None, professor: str = PROFESSOR_NOVO,
) -> Path:
    """Treina um aluno com o professor dado e grava na pasta desta etapa, ou devolve a pasta se já rodou.

    Não treina sem as regras desse professor gravadas. `receita_curta` troca
    campos da receita (poucas épocas) e só é aceita numa pasta descartável:
    serve ao teste de fumaça, nunca ao cache de verdade.
    """
    exigir_regras(professor)
    d = definicao(caminho, aluno, professor)
    if receita_curta:
        if no_cache_de_verdade():
            raise ValueError("Receita curta só numa pasta descartável: o cache da etapa guarda as receitas das Etapas 6 e 10a, sem mudança.")
        d = {**d, "nome": f"{d['nome']} [fumaça]", "receita": {**d["receita"], **receita_curta}}
    nome, dados = variante(caminho, aluno, professor), lambda: dados_do_aluno(caminho, aluno, professor)
    if caminho == "zero":
        return execucao.executar(ETAPA, nome, semente, d, dispositivo, dados=dados)
    return _ajustar_encoder(nome, semente, d, dispositivo, dados)


def carregar(caminho: str, aluno: str, semente: int, professor: str = PROFESSOR_NOVO) -> dict:
    """O que uma execução produziu: meta, histórico e previsões em dev."""
    return execucao.carregar(ETAPA, variante(caminho, aluno, professor), semente)


def carregar_sistema(caminho: str, aluno: str, semente: int, dispositivo: str = "cpu", professor: str = PROFESSOR_NOVO):
    pasta = pasta_da_execucao(caminho, aluno, semente, professor) / "modelo"
    return diet.SistemaDiet.carregar(pasta, dispositivo) if caminho == "zero" else encoder.SistemaEncoder.carregar(pasta, dispositivo)


def previsoes(caminho: str, aluno: str, particao: str, professor: str = PROFESSOR_NOVO) -> list[list[Previsao]]:
    """As respostas das 3 sementes de um aluno em `dev` ou `test` (o teste só existe depois de aberto)."""
    return [carregar_previsoes(pasta_da_execucao(caminho, aluno, s, professor) / f"previsoes_{particao}.parquet")[1] for s in SEMENTES]


def grade() -> list[tuple[str, str, int]]:
    """Todas as execuções de uma rodada, qualquer que seja o professor: (caminho, aluno, semente)."""
    return [(caminho, aluno, semente) for caminho in CAMINHOS for aluno in ALUNOS[caminho] for semente in SEMENTES]


def faltam(caminho: str | None = None, professor: str = PROFESSOR_NOVO) -> list[tuple[str, str, int]]:
    """As execuções do professor que ainda não estão gravadas (de um caminho, ou de todos)."""
    return [(c, a, s) for c, a, s in grade() if caminho in (None, c) and not (pasta_da_execucao(c, a, s, professor) / "meta.json").exists()]


def situacao(professor: str = PROFESSOR_NOVO) -> str:
    prontas = len(grade()) - len(faltam(None, professor))
    por_caminho = ", ".join(f"{CAMINHOS[c]['nome']} {len(ALUNOS[c]) * len(SEMENTES) - len(faltam(c, professor))} de {len(ALUNOS[c]) * len(SEMENTES)}" for c in CAMINHOS)
    return f"{professor}: {prontas} de {len(grade())} execuções prontas ({por_caminho})"


# --------------------------------------------------------------------------
# Regra 1 (a única decisão): a temperatura do aluno B2, em dev, e o que é aberto no teste
# --------------------------------------------------------------------------

REGRA_DA_TEMPERATURA = (
    "a da Etapa 6 (`etapa6.escolher_temperatura`), aplicada em cada caminho, com o professor novo: maior acurácia de intenção "
    "em dev (rótulos humanos), média de 3 sementes; empate exato: maior exact match, depois a menor temperatura. "
    "A grade é a das Etapas 6 e 10a e não é ampliada, mesmo que a escolhida fique na ponta"
)
REGRA_DO_TESTE = (
    "o teste só é aberto depois da temperatura decidida, e só para: o aluno B1, o controle sem suavização (rede do zero), o aluno B2 "
    "na temperatura escolhida e, se ela for diferente da que o professor antigo teve, o B2 novo também na temperatura antiga "
    "(só para a comparação de sensibilidade 'mesma temperatura', que não decide nada). As outras temperaturas nunca são medidas no teste"
)


def temperatura_antiga(caminho: str) -> int:
    """A temperatura que a regra escolheu para o B2 com o professor antigo (Etapa 6: 8; Etapa 10a: 2)."""
    return int(json.loads((ENTREGAS / CAMINHOS[caminho]["etapa"] / "decisao.json").read_text(encoding="utf-8"))["temperatura_do_b2"])


def abrir_teste(caminho: str, aluno: str, semente: int, dispositivo: str | None = None, professor: str = PROFESSOR_NOVO) -> Path:
    """Grava as respostas da execução no teste (`previsoes_test.parquet`). Não repete se já existe."""
    arquivo = pasta_da_execucao(caminho, aluno, semente, professor) / "previsoes_test.parquet"
    if arquivo.exists():
        return arquivo
    particoes, _ = massive.carregar_entrega()
    teste = particoes["test"]
    # como nas etapas de origem: a rede do zero responde na CPU; o XLM-R, na placa se houver
    dispositivo = dispositivo or ("cpu" if caminho == "zero" or not torch.cuda.is_available() else "cuda")
    return salvar_previsoes(arquivo, teste["id"], carregar_sistema(caminho, aluno, semente, dispositivo, professor).prever(list(teste["utt"])))


def decidir_temperatura(caminho: str, dispositivo: str | None = None, professor: str = PROFESSOR_NOVO) -> dict:
    """Aplica `etapa6.escolher_temperatura` às respostas em `dev` do B2 do professor e abre o teste para quem `REGRA_DO_TESTE` diz."""
    arquivo = pasta_da_rodada(professor) / f"temperatura_{caminho}.json"
    if (decidido := _ler_json(arquivo)) is not None:
        return decidido
    exigir_regras(professor)
    if em_falta := faltam(caminho, professor):
        raise FileNotFoundError(f"Ainda faltam {len(em_falta)} execuções de {CAMINHOS[caminho]['nome']} com {professor}.")
    particoes, _ = massive.carregar_entrega()
    em_dev = {T: previsoes(caminho, etapa6.variante_do_b2(T), "dev", professor) for T in TEMPERATURAS}
    temperatura, medias = etapa6.escolher_temperatura(ouro(particoes["dev"]), em_dev)
    antiga = temperatura_antiga(caminho)
    abertas = [a for a in ALUNOS[caminho] if a not in B2] + [etapa6.variante_do_b2(T) for T in dict.fromkeys((int(temperatura), antiga))]
    for aluno in abertas:
        for semente in SEMENTES:
            abrir_teste(caminho, aluno, semente, dispositivo, professor)
    decidido = {
        "etapa": ETAPA, "caminho": caminho, "decidido_em": _agora(), "professor": professor,
        "temperatura_do_b2": int(temperatura), "temperatura_com_o_professor_antigo": antiga,
        "regra_da_temperatura": REGRA_DA_TEMPERATURA, "regra_do_teste": REGRA_DO_TESTE,
        "em_dev": json.loads(medias.to_json(orient="index")),
        "abertas_no_teste": abertas,
    }
    _gravar_json(arquivo, decidido)
    return decidido


def temperatura_nova(caminho: str, professor: str = PROFESSOR_NOVO) -> int:
    decidido = _ler_json(pasta_da_rodada(professor) / f"temperatura_{caminho}.json")
    if decidido is None:
        raise FileNotFoundError(f"A temperatura do B2 de {CAMINHOS[caminho]['nome']} com {professor} ainda não foi decidida: rode o comando `temperatura`.")
    return int(decidido["temperatura_do_b2"])


# --------------------------------------------------------------------------
# Regra 2: as comparações, fechadas antes dos treinos
# --------------------------------------------------------------------------
#
# Um sistema tem uma chave: `prof:novo`, `prof:antigo` e `prof:fechado` são os professores;
# `<caminho>:A` é o aluno A; `<caminho>:<B1|B1s|B2>:<novo|antigo>` são os alunos do cenário B com cada
# professor (B1s: o B1 sem suavização, só na rede do zero; cada B2 na temperatura que a regra escolheu
# em dev com o seu professor); `<caminho>:B2:novo@T_antiga` é o B2 novo na temperatura do antigo.
# "novo" é o professor da rodada; "antigo" é sempre o da Etapa 5b, o dos alunos das Etapas 6 e 10a.
# Toda comparação é a diferença pareada B − A, na média de 3 sementes, com intervalo de 95%
# (`qualidade.comparar_grupos`: bootstrap por enunciado, 2.000 reamostragens).

BLOCOS = {
    "novo_menos_antigo": "O professor melhor faz alunos melhores? Aluno novo − aluno antigo: mesma receita, muda o professor",
    "logits": "O que os logits valem (B2 − B1), com cada professor",
    "aluno_menos_professor": "O aluno ao lado do seu professor (aluno − professor)",
    "b_menos_a": "O cenário B ao lado do cenário A (aluno B − aluno A do mesmo caminho)",
    "professores": "Os professores entre si (o ganho do professor é o denominador do repasse; a Etapa 11a já o mediu)",
    "mesma_temperatura": "Sensibilidade: o B2 novo na temperatura do B2 antigo (só existe se a regra escolher outra temperatura)",
}
SIGLAS = {"B1": "aluno B1", "B1s": "aluno B1 sem suavização", "B2": "aluno B2"}
SO_LOGITS = {"zero": "B1s", "xlmr": "B1"}   # contra quem o B2 difere só nos logits (na rede do zero o B1 tem suavização de rótulos)


def _alunos_b(caminho: str) -> tuple[str, ...]:
    return ("B1", "B1s", "B2") if caminho == "zero" else ("B1", "B2")


def lista_de_comparacoes() -> list[dict]:
    """A lista fechada de comparações pareadas B − A: uma linha por comparação, com o bloco e a pergunta que ela responde."""
    linhas: list[dict] = []

    def junta(bloco: str, caminho: str | None, nome: str, a: str, b: str, pergunta: str) -> None:
        prefixo = f"{CAMINHOS[caminho]['nome']} · " if caminho else ""
        linhas.append({"id": f"{bloco}|{b}|{a}", "bloco": bloco, "caminho": caminho, "nome": prefixo + nome, "a": a, "b": b, "pergunta": pergunta})

    for c in CAMINHOS:
        for s in _alunos_b(c):
            junta("novo_menos_antigo", c, f"{SIGLAS[s]}: com o professor novo − com o antigo", f"{c}:{s}:antigo", f"{c}:{s}:novo",
                  "Com a mesma receita, o aluno do professor melhor é melhor?" + (" (cada B2 na temperatura escolhida em dev com o seu professor)" if s == "B2" else ""))
        for p in ("novo", "antigo"):
            junta("logits", c, f"professor {p}: B2 − {SIGLAS[SO_LOGITS[c]].replace('aluno ', '')} · só os logits mudam", f"{c}:{SO_LOGITS[c]}:{p}", f"{c}:B2:{p}",
                  f"Com o professor {p}, quanto os logits acrescentam aos rótulos, com o resto igual?")
            if c == "zero":
                junta("logits", c, f"professor {p}: B2 − B1 · entram os logits e sai a suavização", f"{c}:B1:{p}", f"{c}:B2:{p}",
                      f"Com o professor {p}, o B2 contra o B1 da receita da Etapa 4b (mudam duas coisas)")
            for s in ("B1", "B2"):
                junta("aluno_menos_professor", c, f"{SIGLAS[s]} − professor {p}", f"prof:{p}", f"{c}:{s}:{p}",
                      f"Quanto se perde (ou se ganha) ao trocar o professor {p} pelo {SIGLAS[s]} dele?")
                junta("b_menos_a", c, f"{SIGLAS[s]} do professor {p} − aluno A", f"{c}:A", f"{c}:{s}:{p}",
                      f"A que distância do aluno A (professor fechado) fica o {SIGLAS[s]} do professor {p}?")
        junta("mesma_temperatura", c, "aluno B2: novo na temperatura do antigo − antigo", f"{c}:B2:antigo", f"{c}:B2:novo@T_antiga",
              "Se a regra escolher outra temperatura: a diferença 'novo − antigo' do B2 se mantém com a temperatura igual?")
    junta("professores", None, "professor novo − professor antigo", "prof:antigo", "prof:novo", "Quanto o professor melhorou? (o denominador do repasse)")
    junta("professores", None, "professor novo − professor fechado", "prof:fechado", "prof:novo", "A que distância o professor aberto novo fica do fechado?")
    junta("professores", None, "professor antigo − professor fechado", "prof:fechado", "prof:antigo", "E o antigo?")
    return linhas


def lista_de_diferencas() -> list[dict]:
    """As diferenças de diferenças: (B − A com o professor novo) − (B − A com o antigo), com os quatro sistemas reamostrados juntos."""
    linhas = []
    for c in CAMINHOS:
        nome = CAMINHOS[c]["nome"]
        pares = [(SO_LOGITS[c], "B2", "o valor dos logits (só os logits mudam): com o professor novo − com o antigo",
                  "Os logits passam a valer mais com o professor novo?")]
        if c == "zero":
            pares.append(("B1", "B2", "B2 − B1 (entram os logits e sai a suavização): com o professor novo − com o antigo",
                          "O mesmo, contra o B1 da receita da Etapa 4b"))
        for a, b, o_que, pergunta in pares:
            linhas.append({"id": f"logits|{c}|{b}-{a}", "bloco": "logits", "caminho": c, "nome": f"{nome} · {o_que}", "pergunta": pergunta,
                           "a_novo": f"{c}:{a}:novo", "b_novo": f"{c}:{b}:novo", "a_antigo": f"{c}:{a}:antigo", "b_antigo": f"{c}:{b}:antigo"})
        for s in ("B1", "B2"):
            linhas.append({"id": f"ganho|{c}|{s}", "bloco": "ganho", "caminho": c,
                           "nome": f"{nome} · {SIGLAS[s]}: ganho do aluno − ganho do professor, em pontos", "pergunta":
                           "O aluno ganhou menos pontos do que o professor? (o repasse em diferença, que não depende de dividir)",
                           "a_novo": "prof:novo", "b_novo": f"{c}:{s}:novo", "a_antigo": "prof:antigo", "b_antigo": f"{c}:{s}:antigo"})
    return linhas


def lista_de_repasses() -> list[dict]:
    """Os repasses: ganho do aluno ÷ ganho do professor, um por aluno e caminho."""
    return [{"id": f"repasse|{c}|{s}", "caminho": c, "aluno": s, "nome": f"{CAMINHOS[c]['nome']} · {SIGLAS[s]}",
             "aluno_antigo": f"{c}:{s}:antigo", "aluno_novo": f"{c}:{s}:novo", "professor_antigo": "prof:antigo", "professor_novo": "prof:novo"}
            for c in CAMINHOS for s in _alunos_b(c)]


def _media(gabarito: list[Previsao], grupo: list[list[Previsao]]) -> pd.Series:
    return pd.DataFrame([qualidade.avaliar(gabarito, p) for p in grupo]).mean()


def _sorteios(gabarito: list[Previsao], grupo: list[list[Previsao]], n: int, semente: int) -> pd.DataFrame:
    """Para cada reamostra do conjunto, a média das métricas entre as sementes do grupo. A mesma `semente` dá os mesmos sorteios a todos os sistemas: é isso que pareia."""
    return sum(qualidade.reamostrar(gabarito, p, n, semente) for p in grupo) / len(grupo)


def diferenca_de_diferencas(
    gabarito: list[Previsao], a_novo: list[list[Previsao]], b_novo: list[list[Previsao]],
    a_antigo: list[list[Previsao]], b_antigo: list[list[Previsao]], n: int = REAMOSTRAS, semente: int = 0,
) -> pd.DataFrame:
    """(B − A com o professor novo) − (B − A com o antigo), em cada métrica, com intervalo de 95%.

    Os quatro sistemas são reamostrados juntos, nos mesmos sorteios de
    enunciados. Positivo: a diferença B − A cresceu com o professor novo.
    """
    novo = _media(gabarito, b_novo) - _media(gabarito, a_novo)
    antigo = _media(gabarito, b_antigo) - _media(gabarito, a_antigo)
    sorteios = (_sorteios(gabarito, b_novo, n, semente) - _sorteios(gabarito, a_novo, n, semente)) \
        - (_sorteios(gabarito, b_antigo, n, semente) - _sorteios(gabarito, a_antigo, n, semente))
    tabela = pd.DataFrame({"com_o_novo": novo, "com_o_antigo": antigo, "diferenca": novo - antigo,
                           "ic_min": sorteios.quantile(0.025), "ic_max": sorteios.quantile(0.975)})
    tabela["distinguivel"] = (tabela["ic_min"] > 0) | (tabela["ic_max"] < 0)
    return tabela


REGRA_DO_REPASSE = (
    "repasse = (aluno novo − aluno antigo) ÷ (professor novo − professor antigo), na mesma métrica e na mesma partição, contra os rótulos "
    "humanos. O aluno entra com a média das 3 sementes; cada professor é uma rotulagem só. O 'aluno antigo' e o 'aluno novo' têm a mesma "
    "receita e só diferem no professor (o B2, cada um na temperatura que a regra escolheu em dev com o seu professor). O intervalo de 95% "
    "vem do mesmo bootstrap por enunciado das outras medidas, com os quatro sistemas reamostrados juntos (2.000 reamostragens, semente 0): "
    "em cada reamostra calcula-se a razão, e ficam os percentis 2,5 e 97,5. Leitura: 1 = o aluno ganhou os mesmos pontos que o professor; "
    "entre 0 e 1 = só parte do ganho chegou; 0 = nada chegou; acima de 1 = o aluno ganhou mais pontos do que o professor. A métrica "
    "principal é o exact match no teste. A razão só é lida onde o ganho do professor é distinguível de zero e positivo (IC95 da diferença "
    "pareada acima de zero): com o denominador perto de zero a razão não quer dizer nada, e então relatam-se só os dois ganhos e a "
    "diferença entre eles, em pontos (lista de diferenças, bloco 'ganho')"
)


def repasse(
    gabarito: list[Previsao], aluno_antigo: list[list[Previsao]], aluno_novo: list[list[Previsao]],
    professor_antigo: list[list[Previsao]], professor_novo: list[list[Previsao]], n: int = REAMOSTRAS, semente: int = 0,
) -> pd.DataFrame:
    """Quanto do ganho do professor chegou ao aluno, em cada métrica (a definição está em `REGRA_DO_REPASSE`)."""
    do_aluno = _media(gabarito, aluno_novo) - _media(gabarito, aluno_antigo)
    do_professor = _media(gabarito, professor_novo) - _media(gabarito, professor_antigo)
    s_aluno = _sorteios(gabarito, aluno_novo, n, semente) - _sorteios(gabarito, aluno_antigo, n, semente)
    s_professor = _sorteios(gabarito, professor_novo, n, semente) - _sorteios(gabarito, professor_antigo, n, semente)
    with np.errstate(divide="ignore", invalid="ignore"):
        razoes = (s_aluno / s_professor).replace([np.inf, -np.inf], np.nan)
        valor = (do_aluno / do_professor).replace([np.inf, -np.inf], np.nan)
    tabela = pd.DataFrame({
        "ganho_do_aluno": do_aluno, "aluno_ic_min": s_aluno.quantile(0.025), "aluno_ic_max": s_aluno.quantile(0.975),
        "ganho_do_professor": do_professor, "professor_ic_min": s_professor.quantile(0.025), "professor_ic_max": s_professor.quantile(0.975),
        "repasse": valor, "ic_min": razoes.quantile(0.025), "ic_max": razoes.quantile(0.975),
    })
    tabela["legivel"] = tabela["professor_ic_min"] > 0   # só se lê a razão onde o professor ganhou de fato
    return tabela


LEITURAS = {
    "distinguivel": "uma diferença só vale quando o intervalo de 95% não contém zero",
    "onde_se_le": "a leitura principal é no teste; dev fica ao lado. A única decisão da etapa (a temperatura do B2) sai de dev",
    "alunos_melhores": ("para um aluno e um caminho, o professor melhor fez um aluno melhor se 'novo − antigo' em exact match no teste tem o "
                        "intervalo todo acima de zero; 'não se distingue' se o intervalo contém zero; 'pior' se fica todo abaixo"),
    "repasse": "lido em exact match no teste, com o intervalo; nas outras métricas é relatado ao lado",
    "logits_valem": ("com um professor, os logits valem se 'B2 − B1, só os logits mudam' em acurácia de intenção no teste tem o intervalo todo "
                     "acima de zero (os logits só falam de intenção); o exact match é relatado ao lado"),
    "logits_valem_mais": ("os logits passam a valer mais com o professor novo se a diferença das diferenças (novo − antigo) em acurácia de "
                          "intenção no teste tem o intervalo todo acima de zero; se o intervalo contém zero, a etapa diz que não dá para afirmar mudança"),
    "quem_representa_o_cenario_b": ("a regra da Etapa 6 (`etapa6.quem_segue`: o B2 se a intenção dele for distinguivelmente maior que a do B1 em dev e o exact match "
                                    "não for distinguivelmente menor; senão o B1) é aplicada em cada caminho, com o professor novo, e relatada. Ela só diz que aluno "
                                    "do cenário B o comparativo deve citar: nesta etapa nada é comprimido"),
    "descritivo": ("sem decidir nada, relatam-se também: cada sistema com intervalo; a grade de temperaturas em dev; a fração do caminho piso → teto "
                   "de cada aluno (`etapa10a.fracao_do_caminho`); e a concordância de cada aluno com o seu professor no teste"),
}


# --------------------------------------------------------------------------
# As regras gravadas
# --------------------------------------------------------------------------


def _arquivos_do_professor(versao: str) -> dict[str, str]:
    """A impressão digital dos arquivos que os alunos leem de um professor."""
    with com_professor(versao) as professor:
        pasta = etapa6.pasta_do_professor("aberto")
    nomes = ["rotulos_treino.parquet", "logits_treino.npy", f"previsoes/{professor['previsoes']}_dev.parquet", f"previsoes/{professor['previsoes']}_test.parquet"]
    return {str((pasta / n).relative_to(RAIZ)): etapa11a.sha256_do_arquivo(pasta / n) for n in nomes if (pasta / n).exists()}


def _com_rotulo_aproveitavel(versao: str) -> dict[str, int]:
    """Em quantos pedidos do treino e de `dev` o professor deixou um rótulo aproveitável: é o que o aluno dele recebe."""
    particoes, rotulos = massive.carregar_entrega()
    with com_professor(versao):
        treino = etapa6.treino_do_professor("aberto", particoes["train"])
        dev = etapa6.dev_do_professor("aberto", particoes["dev"], rotulos["intents"])
    return {"treino": int(treino["valido"].sum()), "dev": int(dev["valido"].sum())}


REGRA_DE_QUEM_GANHA_ALUNOS = (
    "só ganha alunos o professor que tomou o lugar do titular na escada da Etapa 11a (IC95 da diferença pareada de exact match em dev "
    "todo acima de zero, em `entregas/11a_professor_aberto_maior/escolha.json`)"
)


def montar_regras(professor: str = PROFESSOR_NOVO) -> dict:
    """Tudo o que a etapa fixa antes de treinar com um professor: quem ele é, as receitas, a grade, a regra da temperatura e as comparações."""
    escolha = json.loads((ENTREGAS / etapa11a.ETAPA / "escolha.json").read_text(encoding="utf-8"))
    disputa = escolha["disputas"].get(professor)
    if disputa is None or not disputa["ic_min"] > 0:
        raise ValueError(f"{professor} não tomou o lugar do titular na escada da Etapa 11a: {REGRA_DE_QUEM_GANHA_ALUNOS}.")
    iguais = receitas_contra_as_antigas(professor)
    if diferentes := [nome for nome, igual in iguais.items() if igual is False]:
        raise ValueError(f"A receita daqui não é a do aluno antigo em: {', '.join(diferentes)}.")
    fora = escolha["disputas"][FORA_DO_ESCOPO]
    definicoes = {variante(c, a, professor): {"caminho": c, "aluno": a, **definicao(c, a, professor)} for c in CAMINHOS for a in ALUNOS[c]}
    return {
        "etapa": ETAPA,
        "pergunta": PERGUNTA,
        "o_que_muda": "só o professor aberto: os rótulos do treino, os rótulos de dev que escolhem a época de parada e, no B2, os logits de intenção",
        "o_que_muda_junto": ("o número de exemplos de treino de cada aluno é o de pedidos em que o seu professor deixou rótulo aproveitável "
                             "(ver `pedidos_com_rotulo_aproveitavel` de cada professor): faz parte da troca de professor, não é uma escolha desta etapa"),
        "o_que_nao_muda": ("a rede, a receita, as sementes, a grade de temperaturas e a regra de escolha de cada caminho são as das Etapas 6 e 10a; "
                           "não há nova busca, nem mais épocas, nem mais sementes, nem temperatura fora da grade"),
        "professor_novo": {"versao": professor, "nome": nome_do_professor(professor), "de_onde": f"entregas/{etapa11a.ETAPA}/escolha.json",
                           "na_escada": {**disputa, "regra": REGRA_DE_QUEM_GANHA_ALUNOS, "escolhido_da_escada_nesta_data": escolha["escolhido"]},
                           "em_dev": escolha["candidatos"][professor]["dev"], "pedidos_com_rotulo_aproveitavel": _com_rotulo_aproveitavel(professor),
                           "arquivos": _arquivos_do_professor(professor)},
        "professor_antigo": {"versao": PROFESSOR_ANTIGO, "nome": nome_do_professor(PROFESSOR_ANTIGO), "de_onde": "entregas/05b_professor_aberto/ (o professor dos alunos das Etapas 6 e 10a)",
                             "em_dev": escolha["candidatos"][PROFESSOR_ANTIGO]["dev"], "pedidos_com_rotulo_aproveitavel": _com_rotulo_aproveitavel(PROFESSOR_ANTIGO),
                             "arquivos": _arquivos_do_professor(PROFESSOR_ANTIGO)},
        "escopo": {
            "sem_alunos_para": FORA_DO_ESCOPO,
            "por_que": (f"escolha de escopo: na escada da Etapa 11a, o Qwen3-8B sem os 4 bits não se distinguiu do de 4 bits (exact match em dev: "
                        f"{100 * fora['diferenca']:+.1f}, IC95 de {100 * fora['ic_min']:+.1f} a {100 * fora['ic_max']:+.1f} pontos). Treinar alunos para ele "
                        "repetiria os alunos antigos com um professor de mesma qualidade e não responderia à pergunta desta etapa"),
        },
        "caminhos": {c: {"nome": CAMINHOS[c]["nome"], "receita_de": f"entregas/{CAMINHOS[c]['etapa']}/", "alunos": list(ALUNOS[c]),
                         "temperatura_com_o_professor_antigo": temperatura_antiga(c), "treinos_juntos_na_placa": CAMINHOS[c]["processos"]} for c in CAMINHOS},
        "ordem_da_rodada": "primeiro a rede do zero, depois o XLM-R; os dois caminhos nunca dividem a placa",
        "sementes": list(SEMENTES),
        "temperaturas": list(TEMPERATURAS),
        "execucoes": len(grade()),
        "variantes": definicoes,
        "receitas_iguais_as_antigas": {"como": "a configuração e a receita de cada variante, comparadas com o meta.json das 3 execuções antigas de mesmo nome "
                                               "no cache da etapa de origem (null: cache antigo ausente)", "resultado": iguais},
        "pasta_das_execucoes": f"data/execucoes/{ETAPA}/<caminho>__<aluno>__<professor>/semente_<n>/",
        "regra_da_temperatura": REGRA_DA_TEMPERATURA,
        "regra_do_teste": REGRA_DO_TESTE,
        "estatistica": (f"toda comparação é pareada, B − A, na média de {len(SEMENTES)} sementes, com intervalo de 95% por bootstrap por enunciado "
                        f"({REAMOSTRAS} reamostragens, semente 0, percentis 2,5 e 97,5), em dev e no teste, nas métricas de `qualidade.METRICAS`"),
        "blocos": BLOCOS,
        "comparacoes": lista_de_comparacoes(),
        "diferencas_de_diferencas": lista_de_diferencas(),
        "repasses": lista_de_repasses(),
        "regra_do_repasse": REGRA_DO_REPASSE,
        "leituras": LEITURAS,
        "de_onde_vem_os_alunos_antigos": ("das respostas entregues: entregas/06_destilacao/previsoes/ e entregas/10a_encoder/previsoes/ (e entregas/04b_intencao/previsoes/ "
                                          "para o teto da rede do zero). Nenhum aluno antigo é treinado de novo"),
        "artigos": [{"quem": a["quem"], "arxiv": a["arxiv"], "para_que": a["tecnica"]} for a in ARTIGOS],
    }


def gravar_regras(professor: str = PROFESSOR_NOVO) -> Path:
    """Grava as regras de um professor, com a hora. Se já existem, não regrava: a data vale como prova de que vieram antes dos treinos."""
    destino = pasta_da_rodada(professor) / "regras.json"
    if not destino.exists():
        _gravar_json(destino, {"escritas_em": _agora(), **montar_regras(professor)})
    return destino


# --------------------------------------------------------------------------
# A rodada: todos os treinos de um professor, numa cadeia só
# --------------------------------------------------------------------------

# O teste de fumaça: os dados e o professor de verdade, poucas épocas, pasta descartável, uma execução de
# cada tipo (rótulos e logits) em cada caminho. Passa pelo mesmo caminho de código da rodada inteira.
FUMACA = {
    "grade": (("zero", "aluno_b1", 0), ("zero", "aluno_b2_t4", 0), ("xlmr", "aluno_b1", 0), ("xlmr", "aluno_b2_t4", 0)),
    "receita": {"zero": {"epocas": 2, "avaliar_a_cada": 1}, "xlmr": {"epocas": 1}},
}


def usar_pasta_descartavel(pasta: Path) -> Path:
    """Aponta o cache de execuções deste processo para `pasta`. Recusa qualquer pasta dentro de `data/` ou de `entregas/` do projeto."""
    pasta = Path(pasta).resolve()
    if any(pasta.is_relative_to(dono) for dono in (RAIZ / "data", RAIZ / "entregas")):
        raise ValueError(f"{pasta} fica dentro do projeto: o teste de fumaça só grava numa pasta descartável, fora de data/ e de entregas/.")
    execucao.EXECUCOES = pasta
    return pasta


def impressao_do_codigo() -> dict[str, str]:
    """A impressão digital de cada arquivo de código do pacote e das versões travadas: o projeto não é um repositório git."""
    arquivos = sorted((RAIZ / "src" / "especialista").rglob("*.py")) + [RAIZ / "uv.lock", RAIZ / "pyproject.toml"]
    return {str(a.relative_to(RAIZ)): etapa11a.sha256_do_arquivo(a) for a in arquivos if a.exists()}


def copiar_o_codigo(destino: Path) -> dict[str, str]:
    """Guarda em `destino` uma cópia do código com que a rodada começa e devolve a impressão digital de cada arquivo copiado.

    Outra etapa pode estar mexendo no pacote enquanto a rodada anda (cada
    treino é um processo novo, que importa o código do disco). Com a cópia
    dá para ver depois, arquivo por arquivo, o que mudou no meio.
    """
    copiados = {}
    for relativo in impressao_do_codigo():
        (destino / relativo).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAIZ / relativo, destino / relativo)
        copiados[relativo] = etapa11a.sha256_do_arquivo(destino / relativo)
    return copiados


def ambiente() -> dict:
    import transformers

    placa = torch.cuda.is_available()
    return {
        "python": platform.python_version(), "sistema": platform.platform(),
        "bibliotecas": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "placa": torch.cuda.get_device_name(0) if placa else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if placa else None,
    }


def _rodar(caminho: str, aluno: str, semente: int, professor: str = PROFESSOR_NOVO, fumaca: Path | None = None) -> int:
    """Uma execução num processo próprio: a memória da placa volta inteira quando ele acaba."""
    comando = [sys.executable, "-m", "especialista.experimentos.etapa11b", "treinar", "--professor", professor,
               "--caminho", caminho, "--aluno", aluno, "--semente", str(semente)]
    if fumaca is not None:
        comando += ["--fumaca", str(fumaca)]
    return subprocess.run(comando, env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2"}).returncode


def conferir_fumaca(professor: str = PROFESSOR_NOVO) -> list[dict]:
    """O que cada execução do teste de fumaça gravou, e se o modelo relido do disco repete as respostas gravadas."""
    particoes, _ = massive.carregar_entrega()
    textos = list(particoes["dev"]["utt"])[:200]
    linhas = []
    for caminho, aluno, semente in FUMACA["grade"]:
        feito = carregar(caminho, aluno, semente, professor)
        meta, historico = feito["meta"], feito["historico"]
        dispositivo = "cpu" if caminho == "zero" or not torch.cuda.is_available() else "cuda"
        relido = carregar_sistema(caminho, aluno, semente, dispositivo, professor).prever(textos)
        linhas.append({
            "variante": meta["variante"], "nome": meta["nome"], "epocas": meta["receita"]["epocas"], "exemplos_de_treino": meta["exemplos_de_treino"],
            "professor": meta["origem"]["professor"], "treino": meta["origem"]["treino"], "dev": meta["origem"]["dev"], "logits": meta["origem"].get("logits"),
            "temperatura": meta["config"]["temperatura"], "perda_na_primeira_epoca": float(historico["perda_total"].iloc[0]),
            "perda_na_ultima_epoca": float(historico["perda_total"].iloc[-1]), "segundos_de_treino": round(meta["segundos_de_treino"], 1),
            "respostas_iguais_ao_reler_o_modelo": f"{sum(a == b for a, b in zip(relido, feito['previsoes_dev'][:200]))} de {len(textos)}",
        })
    return linhas


def rodada(professor: str = PROFESSOR_NOVO, fumaca: Path | None = None) -> bool:
    """Roda todos os treinos que faltam de um professor: primeiro a rede do zero, em ondas; depois o XLM-R, em fila. Nunca os dois juntos.

    Pode ser interrompida e chamada de novo: o que já está gravado não é
    refeito. Uma execução com erro não para as outras. A última linha que
    imprime começa com `FIM:` ou com `FIM COM ERROS:`. Devolve True se
    tudo ficou pronto. Só mexe nas pastas cujo nome traz esse professor.

    No começo guarda uma cópia do código (`rodadas/codigo_<data>/`); no
    fim, diz na última linha que arquivos do pacote mudaram durante a
    rodada, se algum mudou.

    Com `fumaca`, roda só `FUMACA["grade"]`, com poucas épocas, na pasta
    descartável dada, pelo mesmo caminho de código.
    """
    avisar = lambda texto: print(f"[{datetime.now():%H:%M:%S}] {texto}", flush=True)
    if fumaca is not None:
        regras_de_verdade = pasta_da_rodada(professor) / "regras.json"
        usar_pasta_descartavel(fumaca)
        if not (pasta_da_rodada(professor) / "regras.json").exists():
            pasta_da_rodada(professor).mkdir(parents=True, exist_ok=True)
            shutil.copy2(regras_de_verdade, pasta_da_rodada(professor) / "regras.json")
    regras = exigir_regras(professor)
    inicio = datetime.now()
    registro = pasta_da_rodada(professor) / "rodadas" / f"rodada_{inicio:%Y%m%d-%H%M%S}.json"
    copia = registro.with_name(f"codigo_{inicio:%Y%m%d-%H%M%S}")
    codigo = copiar_o_codigo(copia)
    avisar(f"Etapa 11b, {'teste de fumaça' if fumaca else 'rodada'} (pid {os.getpid()}): regras de {regras['escritas_em']}; {situacao(professor)}")
    relato = {"etapa": ETAPA, "fumaca": fumaca is not None, "pid": os.getpid(), "iniciada_em": _agora(), "regras_escritas_em": regras["escritas_em"],
              "professor": professor, "ambiente": ambiente(), "codigo": codigo, "copia_do_codigo": str(copia), "pasta": str(pasta_da_etapa())}
    _gravar_json(registro, relato)

    feitas = FUMACA["grade"] if fumaca else grade()
    erros: list[str] = []
    for caminho in CAMINHOS:   # um caminho de cada vez: as duas famílias nunca dividem a placa
        pendentes = [t for t in feitas if t[0] == caminho and not (pasta_da_execucao(*t, professor) / "meta.json").exists()]
        if not pendentes:
            continue
        processos = CAMINHOS[caminho]["processos"]
        avisar(f"{CAMINHOS[caminho]['nome']}: {len(pendentes)} execuções, {processos} de cada vez")
        with ThreadPoolExecutor(max_workers=processos) as fila:
            codigos = list(fila.map(lambda t: _rodar(*t, professor, fumaca), pendentes))
        erros += [f"{c}/{a}/semente_{s}" for (c, a, s), saida in zip(pendentes, codigos) if saida != 0]

    prontas = [t for t in feitas if (pasta_da_execucao(*t, professor) / "meta.json").exists()]
    no_fim = impressao_do_codigo()
    alterados = sorted(relativo for relativo in set(codigo) | set(no_fim) if codigo.get(relativo) != no_fim.get(relativo))
    minutos = (datetime.now() - inicio).total_seconds() / 60
    relato.update({"terminada_em": _agora(), "minutos": round(minutos, 1), "prontas": len(prontas), "previstas": len(feitas), "erros": erros,
                   "codigo_igual_do_inicio_ao_fim": not alterados, "codigo_alterado_durante_a_rodada": alterados,
                   "segundos_de_treino": {f"{variante(c, a, professor)}/semente_{s}": round(carregar(c, a, s, professor)["meta"]["segundos_de_treino"], 1) for c, a, s in prontas}})
    if fumaca is not None and len(prontas) == len(feitas):
        relato["conferencia"] = conferir_fumaca(professor)
        print(json.dumps(relato["conferencia"], ensure_ascii=False, indent=2), flush=True)
    _gravar_json(registro, relato)

    resumo = f"{len(prontas)} de {len(feitas)} execuções de {professor} prontas em {minutos:.0f} min; registro em {registro}"
    if alterados:   # não é erro de execução, mas quem lê precisa saber: os treinos de antes e de depois podem ter importado código diferente
        resumo += f"; ATENÇÃO: código alterado durante a rodada em {', '.join(alterados)} (cópia do começo em {copia})"
    if erros or len(prontas) < len(feitas):
        avisar(f"FIM COM ERROS: {resumo}; problemas: {'; '.join(erros) or 'execuções sem meta.json'}")
        return False
    avisar(f"FIM: {resumo}")
    return True


# --------------------------------------------------------------------------
# Depois da rodada: os sistemas das comparações e as contas
# --------------------------------------------------------------------------


def _ler(arquivo: Path, ids: list[int]) -> list[Previsao]:
    """As respostas de um arquivo, conferidas contra os pedidos da partição: rótulo de um pedido aplicado a outro não aparece sozinho."""
    do_arquivo, respostas = carregar_previsoes(arquivo)
    if list(do_arquivo) != ids:
        raise ValueError(f"{arquivo} não traz os mesmos pedidos, na mesma ordem, da partição da Etapa 1.")
    return respostas


def grupos(particao: str, professor: str = PROFESSOR_NOVO) -> dict[str, list[list[Previsao]]]:
    """Todos os sistemas das comparações em `dev` ou `test`, pela chave (ver o comentário da Regra 2).

    Os professores e os alunos antigos vêm das entregas das etapas deles; os
    alunos novos, do cache desta etapa, nas pastas do professor dado. Só
    roda depois da temperatura decidida nos dois caminhos.
    """
    particoes, _ = massive.carregar_entrega()
    dados = particoes[particao]
    ids = dados["id"].tolist()
    g: dict[str, list[list[Previsao]]] = {"prof:fechado": [etapa6.previsoes_do_professor("fechado", particao, dados)]}
    for chave, versao in (("prof:antigo", PROFESSOR_ANTIGO), ("prof:novo", professor)):
        with com_professor(versao):
            g[chave] = [etapa6.previsoes_do_professor("aberto", particao, dados)]
    g["zero:teto"] = [_ler(ENTREGAS / "04b_intencao" / "previsoes" / f"semente_{s}_{particao}.parquet", ids) for s in SEMENTES]
    for c in CAMINHOS:
        entregue = ENTREGAS / CAMINHOS[c]["etapa"] / "previsoes"
        antiga, nova = temperatura_antiga(c), temperatura_nova(c, professor)
        antigos = {"piso": "piso", "A": "aluno_a", "B1:antigo": "aluno_b1", "B2:antigo": etapa6.variante_do_b2(antiga)}
        antigos.update({"B1s:antigo": "aluno_b1_sem_suavizacao"} if c == "zero" else {"teto": "teto"})
        for sigla, v in antigos.items():
            g[f"{c}:{sigla}"] = [_ler(entregue / f"{v}_semente_{s}_{particao}.parquet", ids) for s in SEMENTES]
        novos = {"B1:novo": "aluno_b1", "B2:novo": etapa6.variante_do_b2(nova)}
        novos.update({"B1s:novo": "aluno_b1_sem_suavizacao"} if c == "zero" else {})
        novos.update({"B2:novo@T_antiga": etapa6.variante_do_b2(antiga)} if nova != antiga else {})
        for sigla, v in novos.items():
            g[f"{c}:{sigla}"] = [_ler(pasta_da_execucao(c, v, s, professor) / f"previsoes_{particao}.parquet", ids) for s in SEMENTES]
    return g


def comparar_tudo(gabarito: list[Previsao], g: dict[str, list[list[Previsao]]], n: int = REAMOSTRAS) -> pd.DataFrame:
    """Todas as comparações da lista fechada cujos sistemas existem, uma linha por comparação e métrica."""
    partes = []
    for linha in lista_de_comparacoes():
        if linha["a"] not in g or linha["b"] not in g:   # só a de sensibilidade pode faltar: quando a temperatura não mudou
            if linha["bloco"] != "mesma_temperatura":
                raise KeyError(f"Falta um sistema da comparação {linha['id']}.")
            continue
        tabela = qualidade.comparar_grupos(gabarito, g[linha["a"]], g[linha["b"]], n=n).rename_axis("metrica").reset_index()
        for campo in ("nome", "caminho", "bloco", "id"):
            tabela.insert(0, campo, linha[campo])
        partes.append(tabela)
    return pd.concat(partes, ignore_index=True)


def diferencas_tudo(gabarito: list[Previsao], g: dict[str, list[list[Previsao]]], n: int = REAMOSTRAS) -> pd.DataFrame:
    partes = []
    for linha in lista_de_diferencas():
        tabela = diferenca_de_diferencas(gabarito, *(g[linha[k]] for k in ("a_novo", "b_novo", "a_antigo", "b_antigo")), n=n).rename_axis("metrica").reset_index()
        for campo in ("nome", "caminho", "bloco", "id"):
            tabela.insert(0, campo, linha[campo])
        partes.append(tabela)
    return pd.concat(partes, ignore_index=True)


def repasses_tudo(gabarito: list[Previsao], g: dict[str, list[list[Previsao]]], n: int = REAMOSTRAS) -> pd.DataFrame:
    partes = []
    for linha in lista_de_repasses():
        tabela = repasse(gabarito, *(g[linha[k]] for k in ("aluno_antigo", "aluno_novo", "professor_antigo", "professor_novo")), n=n).rename_axis("metrica").reset_index()
        for campo in ("nome", "aluno", "caminho", "id"):
            tabela.insert(0, campo, linha[campo])
        partes.append(tabela)
    return pd.concat(partes, ignore_index=True)


def calcular(professor: str = PROFESSOR_NOVO, n: int = REAMOSTRAS) -> Path:
    """Faz as contas de um professor e grava em `<professor>/calculo/`: qualidade de cada sistema, comparações, diferenças de diferenças e repasses, em dev e no teste.

    Não roda sem as regras gravadas nem antes da temperatura decidida nos dois caminhos.
    """
    regras = exigir_regras(professor)
    if regras["comparacoes"] != lista_de_comparacoes() or regras["diferencas_de_diferencas"] != lista_de_diferencas() or regras["repasses"] != lista_de_repasses():
        raise ValueError("As listas de comparações do código não são as gravadas em regras.json antes dos treinos.")
    destino = pasta_da_rodada(professor) / "calculo"
    destino.mkdir(parents=True, exist_ok=True)
    particoes, _ = massive.carregar_entrega()
    tabelas: dict[str, list[pd.DataFrame]] = {nome: [] for nome in ("qualidade", "comparacoes", "diferencas_de_diferencas", "repasses", "fracao_do_caminho",
                                                                    "concordancia_com_o_professor", "resultados")}
    for particao in ("dev", "test"):
        gabarito, g = ouro(particoes[particao]), grupos(particao, professor)
        if particao == "dev":
            _gravar_json(destino / "quem_representa.json", quem_representa(gabarito, g))
        medidas = pd.concat({chave: qualidade.intervalo_do_grupo(gabarito, grupo, n=n) for chave, grupo in g.items()}, names=["sistema", "metrica"]).reset_index()
        medidas.insert(1, "sementes", medidas["sistema"].map(lambda chave: len(g[chave])))
        for nome, tabela in (("qualidade", medidas), ("comparacoes", comparar_tudo(gabarito, g, n)), ("diferencas_de_diferencas", diferencas_tudo(gabarito, g, n)),
                             ("repasses", repasses_tudo(gabarito, g, n)), ("fracao_do_caminho", fracoes_do_caminho(gabarito, g, n)),
                             ("concordancia_com_o_professor", concordancia_com_o_professor(gabarito, g, n))):
            tabela.insert(0, "particao", particao)
            tabelas[nome].append(tabela)
        tabelas["resultados"].append(pd.DataFrame(linhas_de_resultados(particao, gabarito, professor)))   # já traz a coluna `particao`
    prontas = {nome: pd.concat(partes, ignore_index=True) for nome, partes in tabelas.items()}
    prontas.update({"temperaturas": tabela_de_temperaturas(professor, n), "treinos": tabela_de_treinos(professor),
                    "conferencia": conferir_com_as_origens(prontas["comparacoes"], professor)})
    for nome, tabela in prontas.items():
        tabela.to_csv(destino / f"{nome}.csv", index=False)
    _gravar_json(destino / "calculo.json", {
        "etapa": ETAPA, "professor": professor, "calculado_em": _agora(), "regras_escritas_em": regras["escritas_em"], "reamostras": n,
        "temperaturas": {c: {"nova": temperatura_nova(c, professor), "antiga": temperatura_antiga(c)} for c in CAMINHOS},
    })
    return destino


# --------------------------------------------------------------------------
# O que é só relato: a grade em dev, a fração do caminho, a concordância com o professor e a conferência
# --------------------------------------------------------------------------


def tabela_de_temperaturas(professor: str = PROFESSOR_NOVO, n: int = REAMOSTRAS) -> pd.DataFrame:
    """A grade do B2 em `dev`, em cada caminho, com o professor da rodada e com o antigo: uma linha por temperatura.

    `b2_menos_b1` é contra o aluno que só difere do B2 nos logits (`SO_LOGITS`);
    `escolhida_menos_esta` é a temperatura que a regra escolheu menos esta, em
    acurácia de intenção.
    """
    particoes, _ = massive.carregar_entrega()
    gabarito, ids = ouro(particoes["dev"]), particoes["dev"]["id"].tolist()
    so_logits = {"zero": "aluno_b1_sem_suavizacao", "xlmr": "aluno_b1"}
    linhas = []
    for c in CAMINHOS:
        entregue = ENTREGAS / CAMINHOS[c]["etapa"] / "previsoes"
        leitores = {
            "novo": lambda v, c=c: previsoes(c, v, "dev", professor),
            "antigo": lambda v, entregue=entregue: [_ler(entregue / f"{v}_semente_{s}_dev.parquet", ids) for s in SEMENTES],
        }
        escolhidas = {"novo": temperatura_nova(c, professor), "antigo": temperatura_antiga(c)}
        for quem, le in leitores.items():
            do_b1, da_escolhida = le(so_logits[c]), le(etapa6.variante_do_b2(escolhidas[quem]))
            for T in TEMPERATURAS:
                do_b2 = le(etapa6.variante_do_b2(T))
                media = _media(gabarito, do_b2)
                contra_o_b1 = qualidade.comparar_grupos(gabarito, do_b1, do_b2, n=n)
                contra_esta = qualidade.comparar_grupos(gabarito, do_b2, da_escolhida, n=n).loc["intent_acc"]
                linha = {"caminho": c, "professor": quem, "temperatura": T, "escolhida": T == escolhidas[quem], **{m: float(media[m]) for m in TRES}}
                for m in ("intent_acc", "exact_match"):
                    linha.update({f"b2_menos_b1_{m}{sufixo}": float(contra_o_b1.loc[m, campo]) for sufixo, campo in (("", "diferenca"), ("_ic_min", "ic_min"), ("_ic_max", "ic_max"))})
                linha.update({f"escolhida_menos_esta{sufixo}": float(contra_esta[campo]) for sufixo, campo in (("", "diferenca"), ("_ic_min", "ic_min"), ("_ic_max", "ic_max"))})
                linhas.append(linha)
    return pd.DataFrame(linhas)


def fracoes_do_caminho(gabarito: list[Previsao], g: dict[str, list[list[Previsao]]], n: int = REAMOSTRAS) -> pd.DataFrame:
    """A fração do caminho piso → teto de cada aluno (`etapa10a.fracao_do_caminho`), com o piso e o teto do próprio caminho."""
    partes = []
    for c in CAMINHOS:
        for sigla in ("A", "B1:antigo", "B2:antigo", "B1:novo", "B2:novo"):
            tabela = etapa10a.fracao_do_caminho(gabarito, g[f"{c}:piso"], g[f"{c}:teto"], g[f"{c}:{sigla}"], n=n).rename_axis("metrica").reset_index()
            tabela.insert(0, "sistema", f"{c}:{sigla}")
            tabela.insert(0, "caminho", c)
            partes.append(tabela)
    return pd.concat(partes, ignore_index=True)


def concordancia_com_o_professor(gabarito: list[Previsao], g: dict[str, list[list[Previsao]]], n: int = REAMOSTRAS) -> pd.DataFrame:
    """Em que parte dos pedidos cada aluno dá a mesma intenção, e a mesma resposta inteira, que o professor que o ensinou.

    Traz também os cinco casos de `etapa6.aluno_e_professor` na intenção (os
    dois acertam, só um acerta, erram igual, erram diferente), na média das sementes.
    """
    pares = [(f"{c}:A", "prof:fechado") for c in CAMINHOS] + [(f"{c}:{s}:{p}", f"prof:{p}") for c in CAMINHOS for s in ("B1", "B2") for p in ("antigo", "novo")]
    linhas = []
    for aluno, professor in pares:
        do_professor = g[professor][0]
        igual = qualidade.intervalo_do_grupo(do_professor, g[aluno], n=n)
        casos = sum(etapa6.aluno_e_professor(gabarito, do_professor, respostas, "intencao") for respostas in g[aluno]) / len(g[aluno])
        linha = {"aluno": aluno, "professor": professor}
        for nome, metrica in (("mesma_intencao", "intent_acc"), ("mesma_resposta", "exact_match")):
            linha.update({nome: float(igual.loc[metrica, "valor"]), f"{nome}_ic_min": float(igual.loc[metrica, "ic_min"]), f"{nome}_ic_max": float(igual.loc[metrica, "ic_max"])})
        linha.update({caso: float(parte) for caso, parte in casos.items()})
        linhas.append(linha)
    return pd.DataFrame(linhas)


def quem_representa(gabarito_dev: list[Previsao], g_dev: dict[str, list[list[Previsao]]]) -> dict:
    """A regra da Etapa 6 (`etapa6.quem_segue`), aplicada em dev aos alunos do professor da rodada. Só relata: nada é comprimido aqui."""
    saida = {}
    for c in CAMINHOS:
        r = etapa6.quem_segue(gabarito_dev, {"A": g_dev[f"{c}:A"], "B1": g_dev[f"{c}:B1:novo"], "B2": g_dev[f"{c}:B2:novo"]})
        em_dev = lambda chave: float(_media(gabarito_dev, g_dev[chave])["exact_match"])
        saida[c] = {
            "representa": "B2" if r["b2_passa"] else "B1", "b2_passa": bool(r["b2_passa"]),
            "b2_menos_b1_em_dev": {m: {k: float(r["b2_menos_b1"].loc[m, k]) for k in ("diferenca", "ic_min", "ic_max")} for m in ("intent_acc", "exact_match")},
            "exact_match_em_dev": {"A": em_dev(f"{c}:A"), "B1": em_dev(f"{c}:B1:novo"), "B2": em_dev(f"{c}:B2:novo")},
        }
    return saida


def conferencias(professor: str = PROFESSOR_NOVO) -> tuple:
    """As comparações desta etapa que uma etapa anterior já gravou: (id daqui, etapa de origem, como se chama lá, sinal)."""
    da_11a = "11a_professor_aberto_maior"
    return (
        ("logits|zero:B2:antigo|zero:B1s:antigo", "06_destilacao", "Aluno B2 − aluno B1 sem suavização", 1),
        ("logits|zero:B2:antigo|zero:B1:antigo", "06_destilacao", "Aluno B2 − aluno B1", 1),
        ("aluno_menos_professor|zero:B1:antigo|prof:antigo", "06_destilacao", "Aluno B1 − professor aberto", 1),
        ("aluno_menos_professor|zero:B2:antigo|prof:antigo", "06_destilacao", "Aluno B2 − professor aberto", 1),
        ("b_menos_a|zero:B1:antigo|zero:A", "06_destilacao", "Aluno A − aluno B1", -1),          # lá é A − B1: o sinal troca
        ("logits|xlmr:B2:antigo|xlmr:B1:antigo", "10a_encoder", "Aluno B2 − aluno B1", 1),
        ("aluno_menos_professor|xlmr:B1:antigo|prof:antigo", "10a_encoder", "Aluno B1 − professor aberto", 1),
        ("aluno_menos_professor|xlmr:B2:antigo|prof:antigo", "10a_encoder", "Aluno B2 − professor aberto", 1),
        ("professores|prof:novo|prof:antigo", da_11a, (professor, PROFESSOR_ANTIGO), 1),
        ("professores|prof:novo|prof:fechado", da_11a, (professor, etapa11a.FECHADO), 1),
        ("professores|prof:antigo|prof:fechado", da_11a, (PROFESSOR_ANTIGO, etapa11a.FECHADO), 1),
    )


def conferir_com_as_origens(comparacoes: pd.DataFrame, professor: str = PROFESSOR_NOVO, tolerancia: float = 1e-9) -> pd.DataFrame:
    """Cada comparação de `conferencias`, refeita aqui das respostas entregues, ao lado do número que a etapa de origem gravou.

    `fecha` diz se a diferença e os dois limites do intervalo coincidem. Um
    número que não fecha é relatado, não consertado.
    """
    daqui = comparacoes.set_index(["particao", "id", "metrica"])
    linhas = []
    for ident, etapa, la, sinal in conferencias(professor):
        gravadas = pd.read_csv(ENTREGAS / etapa / "comparacoes.csv")
        for particao in ("dev", "test"):
            if etapa == "06_destilacao":      # lá a partição vem dentro do nome da comparação
                da_origem = gravadas[gravadas["comparacao"] == f"{particao} · {la}"]
            elif etapa == "10a_encoder":
                da_origem = gravadas[(gravadas["particao"] == particao) & (gravadas["comparacao"] == la)]
            else:
                da_origem = gravadas[(gravadas["particao"] == particao) & (gravadas["b"] == la[0]) & (gravadas["a"] == la[1])]
            for metrica in qualidade.METRICAS:
                linha = {"particao": particao, "id": ident, "etapa_de_origem": etapa, "nome_na_origem": la if isinstance(la, str) else " − ".join(la), "metrica": metrica}
                gravada = da_origem[da_origem["metrica"] == metrica]
                refeita = daqui.loc[(particao, ident, metrica)]
                linha["daqui"] = float(refeita["diferenca"])
                if len(gravada) != 1:
                    linhas.append({**linha, "de_la": np.nan, "maior_distancia": np.nan, "fecha": False})
                    continue
                g = gravada.iloc[0]
                esperado = (g["diferenca"], g["ic_min"], g["ic_max"]) if sinal == 1 else (-g["diferenca"], -g["ic_max"], -g["ic_min"])
                distancia = max(abs(float(refeita[campo]) - float(valor)) for campo, valor in zip(("diferenca", "ic_min", "ic_max"), esperado))
                linhas.append({**linha, "de_la": float(esperado[0]), "maior_distancia": distancia, "fecha": bool(distancia <= tolerancia)})
    return pd.DataFrame(linhas)


def tabela_de_treinos(professor: str = PROFESSOR_NOVO) -> pd.DataFrame:
    """Uma linha por execução: a época escolhida, os exemplos, o tempo e de onde vieram os rótulos."""
    linhas = []
    for c, a, s in grade():
        meta = carregar(c, a, s, professor)["meta"]
        linhas.append({
            "caminho": c, "aluno": a, "semente": s, "variante": meta["variante"], "nome": meta["nome"], "melhor_epoca": meta["melhor_epoca"], "epocas": meta["receita"]["epocas"],
            "exemplos_de_treino": meta["exemplos_de_treino"], "exemplos_de_dev": meta["exemplos_de_dev"], "segundos_de_treino": meta["segundos_de_treino"],
            "segundos_por_epoca": meta.get("segundos_por_epoca", meta["segundos_de_treino"] / meta["receita"]["epocas"]), "pico_de_vram_gb": meta.get("pico_de_vram_gb"),
            "contra_o_professor_em_dev_intent_acc": meta["dev"]["intent_acc"], "contra_o_professor_em_dev_exact_match": meta["dev"]["exact_match"],
            "treino": meta["origem"]["treino"], "dev": meta["origem"]["dev"], "logits": meta["origem"].get("logits"),
        })
    return pd.DataFrame(linhas)


def _tamanho(caminho: str, aluno: str, professor: str) -> tuple[int, int]:
    """Parâmetros e bytes em disco de um aluno (semente 0)."""
    meta = carregar(caminho, aluno, 0, professor)["meta"]
    parametros = meta.get("parametros") or carregar_sistema(caminho, aluno, 0, "cpu", professor).parametros
    return int(parametros), int(bytes_em_disco(pasta_da_execucao(caminho, aluno, 0, professor) / "modelo"))


def linhas_de_resultados(particao: str, gabarito: list[Previsao], professor: str = PROFESSOR_NOVO) -> list[dict]:
    """As linhas de `resultados.csv` (`avaliacao/resultados.py`): cada aluno novo com respostas gravadas na partição."""
    from especialista.avaliacao import resultados

    linhas = []
    for c in CAMINHOS:
        for aluno in ALUNOS[c]:
            if not all((pasta_da_execucao(c, aluno, s, professor) / f"previsoes_{particao}.parquet").exists() for s in SEMENTES):
                continue   # as temperaturas que a regra não escolheu não têm teste
            parametros, em_disco = _tamanho(c, aluno, professor)
            linhas.append(resultados.registrar_grupo(
                definicao(c, aluno, professor)["nome"], particao, gabarito, previsoes(c, aluno, particao, professor), parametros=parametros, bytes_em_disco=em_disco,
                variante=variante(c, aluno, professor), caminho=c, aluno=aluno, professor=professor))
    return linhas


# --------------------------------------------------------------------------
# A entrega: a decisão, os sistemas declarados para o comparativo, os arquivos e o manifesto
# --------------------------------------------------------------------------

EXTENSAO_DA_ESCADA = (
    "quando esta etapa foi entregue, a extensão da escada da Etapa 11a (um quarto degrau, maior) ainda estava sendo fechada. "
    "Pela regra desta etapa, só se treinam alunos do professor que toma o lugar do titular na escada"
)


def siglas_dos_alunos(caminho: str, professor: str = PROFESSOR_NOVO) -> dict[str, str]:
    """{sigla: aluno} dos alunos medidos no teste: B1, o controle (rede do zero) e o B2 na temperatura escolhida."""
    alunos = {"B1": "aluno_b1", "B2": etapa6.variante_do_b2(temperatura_nova(caminho, professor))}
    return {**alunos, "B1s": "aluno_b1_sem_suavizacao"} if caminho == "zero" else alunos


ENTREGUES = ("B1", "B2")   # os alunos cujos modelos vão para a entrega, por link físico


def montar_decisao(professor: str = PROFESSOR_NOVO) -> dict:
    """O `decisao.json` da etapa: a temperatura de cada caminho, quem representa o cenário B e onde está cada aluno."""
    regras, calculo = exigir_regras(professor), pasta_da_rodada(professor) / "calculo"
    representa = json.loads((calculo / "quem_representa.json").read_text(encoding="utf-8"))
    alunos: dict[str, dict] = {}
    for c in CAMINHOS:
        alunos[c] = {}
        for sigla, aluno in siglas_dos_alunos(c, professor).items():
            metas = [carregar(c, aluno, s, professor)["meta"] for s in SEMENTES]
            v = variante(c, aluno, professor)
            alunos[c][sigla] = {
                "nome": metas[0]["nome"], "variante": v, "aluno": aluno,
                "modelos": f"entregas/{ETAPA}/modelos/{v}" if sigla in ENTREGUES else None,
                "pastas_por_semente": {str(s): f"entregas/{ETAPA}/modelos/{v}/semente_{s}" for s in SEMENTES} if sigla in ENTREGUES else None,
                "no_cache": f"data/execucoes/{ETAPA}/{v}", "sementes": list(SEMENTES), "config": metas[0]["config"], "receita": metas[0]["receita"],
                "sinal": metas[0]["origem"], "melhor_epoca": [m["melhor_epoca"] for m in metas],
                "exact_match_em_dev": representa[c]["exact_match_em_dev"].get(sigla),
            }
    temperaturas = {c: _ler_json(pasta_da_rodada(professor) / f"temperatura_{c}.json") for c in CAMINHOS}
    return {
        "como_ler": ("`temperatura_do_b2` é a única decisão da etapa, por caminho, tomada em dev. `alunos` traz, por caminho, o B1, o B2 na temperatura escolhida e, na rede do "
                     "zero, o controle sem suavização (B1s). As pastas de `pastas_por_semente` são lidas por `SistemaDiet.carregar` (zero) e `SistemaEncoder.carregar` (xlmr). "
                     "`sinal` diz de onde vêm os rótulos do treino e da parada de cada aluno. Nada segue para a compressão nesta etapa"),
        "etapa": ETAPA, "gerado_em": _agora(), "pergunta": PERGUNTA,
        "professor": {"versao": professor, "nome": nome_do_professor(professor), "de_onde": regras["professor_novo"]["de_onde"]},
        "professor_antigo": {"versao": PROFESSOR_ANTIGO, "nome": nome_do_professor(PROFESSOR_ANTIGO)},
        "regras": {"arquivo": f"entregas/{ETAPA}/regras.json", "escritas_em": regras["escritas_em"]},
        "temperatura_do_b2": {c: temperaturas[c]["temperatura_do_b2"] for c in CAMINHOS},
        "temperatura_com_o_professor_antigo": {c: temperaturas[c]["temperatura_com_o_professor_antigo"] for c in CAMINHOS},
        "regra_da_temperatura": REGRA_DA_TEMPERATURA,
        "em_dev_por_temperatura": {c: temperaturas[c]["em_dev"] for c in CAMINHOS},
        "abertas_no_teste": {c: temperaturas[c]["abertas_no_teste"] for c in CAMINHOS},
        "quem_representa_o_cenario_b": {"regra": LEITURAS["quem_representa_o_cenario_b"], **representa},
        "alunos": alunos,
        "alunos_antigos": {"zero": "entregas/06_destilacao/decisao.json", "xlmr": "entregas/10a_encoder/decisao.json"},
        "extensao_da_escada": EXTENSAO_DA_ESCADA,
    }


def montar_sistemas(professor: str = PROFESSOR_NOVO) -> dict:
    """Os alunos novos, declarados para o comparativo no formato de `etapa10d.fichas_declaradas`.

    Cada aluno fica no seu caminho (`zero` ou `xlmr`), com um `modelo` que traz
    o professor no nome (assim não toma o lugar do aluno antigo nas marcas do
    registro) e com `professor` apontando para a ficha que a Etapa 11a declarou.
    Com essas marcas ele entra, por regra, em "aluno − professor dele" e em
    "XLM-R − rede do zero".
    """
    from especialista.experimentos import etapa10d

    poucos = len(pd.read_parquet(ENTREGAS / etapa6.DEMONSTRACOES))
    quem = nome_do_professor(professor)
    fichas = {}
    for c in CAMINHOS:
        T = temperatura_nova(c, professor)
        for sigla, aluno in (("B1", "aluno_b1"), ("B2", etapa6.variante_do_b2(T))):
            modelo, v = f"aluno_{sigla.lower()}@{professor}", variante(c, aluno, professor)
            fichas[etapa10d.CAMINHOS[c]["prefixo"] + modelo] = {
                "nome": f"Aluno {sigla} ({'rótulos' if sigla == 'B1' else 'logits'} do {quem}" + (f", T = {T})" if sigla == "B2" else ")"),
                "curto": f"Aluno {sigla} · {quem}", "bloco": "Alunos destilados", "rotulos_humanos": poucos, "rotulos_como": "pelo professor",
                "dev_humano": f"temperatura T = {T} escolhida em dev (Etapa 11b)" if sigla == "B2" else "nenhuma no treino (parada no dev rotulado pelo professor)",
                "caminho": c, "modelo": modelo, "ponto": "destilado", "professor": professor, "papel": f"aluno do cenário B com o professor aberto maior, Etapa 11b",
                "variante": v, "temperatura": T if sigla == "B2" else None,
                "resultado": [ETAPA, {"variante": v}],
                "previsoes": {p: [f"entregas/{ETAPA}/previsoes/{v}_semente_{s}_{p}.parquet" for s in SEMENTES] for p in ("dev", "test")},
            }
    return {
        "sistemas": fichas,
        "fora_do_formato": {
            variante("zero", "aluno_b1_sem_suavizacao", professor): "é o controle do B2 na rede do zero: não tem par no XLM-R e não é um sistema do comparativo; está em resultados.csv",
            "as outras temperaturas do B2": "só têm respostas em dev (a regra não as abre no teste)",
        },
    }


def entregar(professor: str = PROFESSOR_NOVO) -> Path:
    """Monta `entregas/11b_alunos_do_professor_maior/` a partir do cache. Nada é tirado do cache; os modelos entram por link físico."""
    destino, rodada_do_professor = ENTREGAS / ETAPA, pasta_da_rodada(professor)
    destino.mkdir(parents=True, exist_ok=True)
    copiar = lambda origem, alvo: (alvo.parent.mkdir(parents=True, exist_ok=True), shutil.copy2(origem, alvo))
    for arquivo in sorted((rodada_do_professor / "calculo").glob("*.csv")) + [rodada_do_professor / "calculo" / "calculo.json", rodada_do_professor / "regras.json"]:
        copiar(arquivo, destino / arquivo.name)
    _gravar_json(destino / "decisao.json", montar_decisao(professor))
    _gravar_json(destino / "sistemas.json", montar_sistemas(professor))
    quadro_de_artigos().to_csv(destino / "artigos.csv", index=False)
    for consulta in sorted((pasta_da_etapa() / "artigos").glob("arxiv_*.json"))[-1:]:
        copiar(consulta, destino / "artigos_na_fonte.json")
    for registro in sorted((rodada_do_professor / "rodadas").glob("rodada_*.json")):
        copiar(registro, destino / "rodadas" / registro.name)
    for sobra in sorted(a for a in (rodada_do_professor / "fumaca").glob("*") if a.is_file()):   # o teste de fumaça e a impressão digital de antes dos treinos
        copiar(sobra, destino / "fumaca" / sobra.name)
    if (rodada_do_professor / "rodada.log").exists():
        copiar(rodada_do_professor / "rodada.log", destino / "rodadas" / "rodada.log")
    for c, aluno, s in grade():
        pasta, v = pasta_da_execucao(c, aluno, s, professor), variante(c, aluno, professor)
        copiar(pasta / "historico.csv", destino / "historicos" / f"{v}_semente_{s}.csv")
        for particao in ("dev", "test"):
            if (pasta / f"previsoes_{particao}.parquet").exists():
                copiar(pasta / f"previsoes_{particao}.parquet", destino / "previsoes" / f"{v}_semente_{s}_{particao}.parquet")
    for c in CAMINHOS:
        for sigla, aluno in siglas_dos_alunos(c, professor).items():
            if sigla not in ENTREGUES:
                continue
            for s in SEMENTES:
                modelo = pasta_da_execucao(c, aluno, s, professor) / "modelo"
                for arquivo in sorted(a for a in modelo.rglob("*") if a.is_file()):
                    etapa11a.ligar(arquivo, destino / "modelos" / variante(c, aluno, professor) / f"semente_{s}" / arquivo.relative_to(modelo))
    for nome in ("src/especialista/experimentos/etapa11b.py", "notebooks/construtores/build_nb11b.py", "tests/test_etapa11b.py"):
        if (RAIZ / nome).exists():
            copiar(RAIZ / nome, destino / "codigo" / Path(nome).name)
    return destino


def conferir_intactas(professor: str = PROFESSOR_NOVO, etapas: tuple[str, ...] = ("06_destilacao", "10a_encoder"), com_sha256_ate: int = 2 * 2**20) -> pd.DataFrame:
    """O cache e a entrega das etapas dadas, hoje, contra a impressão digital tirada antes do primeiro treino desta etapa.

    A impressão de antes está em `<professor>/fumaca/impressoes_antes.json`
    (tamanho, data e sha256 de cada arquivo). Um arquivo conta como igual se
    tem o mesmo tamanho e a mesma data de modificação e, até
    `com_sha256_ate` bytes, o mesmo sha256.
    """
    antes = json.loads((pasta_da_rodada(professor) / "fumaca" / "impressoes_antes.json").read_text(encoding="utf-8"))
    linhas = []
    for raiz in ("data/execucoes", "entregas"):
        for etapa in etapas:
            prefixo = f"{raiz}/{etapa}/"
            de_antes = {nome: registro for nome, registro in antes["arquivos"].items() if nome.startswith(prefixo)}
            de_hoje = {str(a.relative_to(RAIZ)): a for a in (RAIZ / raiz / etapa).rglob("*") if a.is_file()}
            iguais = 0
            for nome, registro in de_antes.items():
                if nome not in de_hoje:
                    continue
                estado = de_hoje[nome].stat()
                igual = estado.st_size == registro["bytes"] and estado.st_mtime_ns == registro["mtime_ns"]
                if igual and registro["sha256"] is not None and registro["bytes"] <= com_sha256_ate:
                    igual = etapa11a.sha256_do_arquivo(de_hoje[nome]) == registro["sha256"]
                iguais += igual
            linhas.append({"pasta": prefixo, "impressao_tirada_em": antes["tirada_em"], "arquivos_antes": len(de_antes), "arquivos_hoje": len(de_hoje), "iguais": iguais,
                           "novos": len(set(de_hoje) - set(de_antes)), "sumiram": len(set(de_antes) - set(de_hoje))})
    return pd.DataFrame(linhas)


def manifesto(professor: str = PROFESSOR_NOVO) -> Path:
    """O que a entrega contém (com a impressão digital de cada arquivo), do que ela foi feita e onde ficou o que é grande."""
    destino = ENTREGAS / ETAPA
    regras = exigir_regras(professor)
    registros = sorted((pasta_da_rodada(professor) / "rodadas").glob("rodada_*.json"))
    ultima = json.loads(registros[-1].read_text(encoding="utf-8"))
    entradas = [ENTREGAS / "01_dados" / n for n in ("train.parquet", "dev.parquet", "test.parquet", "rotulos.json")]
    entradas += [ENTREGAS / n for n in ("04b_intencao/decisao.json", "06_destilacao/decisao.json", "10a_encoder/decisao.json", f"{etapa11a.ETAPA}/escolha.json", etapa6.DEMONSTRACOES)]
    entradas += [RAIZ / n for versao in ("professor_novo", "professor_antigo") for n in regras[versao]["arquivos"]]
    entradas += [etapa6.pasta_do_professor("fechado") / "previsoes" / f"{etapa6.PROFESSORES['fechado']['previsoes']}_{p}.parquet" for p in ("dev", "test")]
    for pasta, molde in ((ENTREGAS / "04b_intencao" / "previsoes", "semente_*.parquet"), (ENTREGAS / "06_destilacao" / "previsoes", "*.parquet"), (ENTREGAS / "10a_encoder" / "previsoes", "*.parquet")):
        entradas += sorted(pasta.glob(molde))
    arquivos = sorted(a for a in destino.rglob("*") if a.is_file() and a.name != "manifesto.json")
    arquivo = destino / "manifesto.json"
    _gravar_json(arquivo, {
        "etapa": ETAPA, "gerado_em": _agora(),
        "o_que_e": "O que a Etapa 11b leu (com a impressão digital de cada arquivo), as regras, a rodada de treinos e o que ela entrega.",
        "professor": professor, "professor_antigo": PROFESSOR_ANTIGO,
        "regras": {"arquivo": f"entregas/{ETAPA}/regras.json", "escritas_em": regras["escritas_em"]},
        "rodada": {k: ultima.get(k) for k in ("iniciada_em", "terminada_em", "minutos", "prontas", "previstas", "erros", "codigo_igual_do_inicio_ao_fim",
                                             "codigo_alterado_durante_a_rodada", "ambiente", "copia_do_codigo")},
        "codigo_da_rodada": ultima["codigo"],
        "entradas": {str(a.relative_to(RAIZ)): etapa11a.sha256_do_arquivo(a) for a in entradas},
        "saidas": {str(a.relative_to(destino)): {"sha256": etapa11a.sha256_do_arquivo(a), "bytes": a.stat().st_size, "link_fisico": a.stat().st_nlink > 1} for a in arquivos},
        "cache": f"data/execucoes/{ETAPA}/ (as 33 execuções, as outras temperaturas do B2 e a cópia do código da rodada; os modelos da entrega são links físicos para lá)",
    })
    return arquivo


# --------------------------------------------------------------------------
# Os artigos de apoio, conferidos na fonte (`etapa10d.conferir_artigos`)
# --------------------------------------------------------------------------
#
# No formato de `etapa10d.ARTIGOS`. `RESUMOS_LIDOS` diz o que o **resumo** de cada um, lido no arXiv em
# 18/11/2026, sustenta e o que não sustenta nesta etapa. O corpo dos artigos não foi lido aqui.

ARTIGOS = (
    {"passo": "Destilação com temperatura", "onde": "os dois", "tecnica": "A perda do aluno B2: os logits do professor amolecidos por uma temperatura", "autor": "Hinton",
     "quem": "Hinton, Vinyals & Dean", "titulo": "Distilling the Knowledge in a Neural Network", "arxiv": "1503.02531"},
    {"passo": "Professor e aluno", "onde": "os dois", "tecnica": "Um professor mais exato não faz, por si só, um aluno melhor", "autor": "Cho",
     "quem": "Cho & Hariharan", "titulo": "On the Efficacy of Knowledge Distillation", "arxiv": "1910.01348"},
    {"passo": "Professor e aluno", "onde": "os dois", "tecnica": "A distância de tamanho entre professor e aluno atrapalha a destilação", "autor": "Mirzadeh",
     "quem": "Mirzadeh et al.", "titulo": "Improved Knowledge Distillation via Teacher Assistant", "arxiv": "1902.03393"},
    {"passo": "Professor e aluno", "onde": "os dois", "tecnica": "O que os logits valem depende de quão bem aproximam as probabilidades de cada classe", "autor": "Menon",
     "quem": "Menon et al.", "titulo": "Why distillation helps: a statistical perspective", "arxiv": "2005.10419"},
    {"passo": "Professor e aluno", "onde": "os dois", "tecnica": "Parte do ganho dos alvos suaves é regularização, como a suavização de rótulos (o controle do B2)", "autor": "Yuan",
     "quem": "Yuan et al.", "titulo": "Revisiting Knowledge Distillation via Label Smoothing Regularization", "arxiv": "1909.11723"},
    {"passo": "Professor e aluno", "onde": "os dois", "tecnica": "Imitar o professor e acertar a tarefa são medidas diferentes (aluno − professor e concordância)", "autor": "Stanton",
     "quem": "Stanton et al.", "titulo": "Does Knowledge Distillation Really Work?", "arxiv": "2106.05945"},
    {"passo": "Professor e aluno", "onde": "os dois", "tecnica": "Um aluno pode passar o seu professor", "autor": "Furlanello",
     "quem": "Furlanello et al.", "titulo": "Born Again Neural Networks", "arxiv": "1805.04770"},
)
RESUMOS_LIDOS = {
    "1503.02531": ("em parte", "o resumo fala em destilar um conjunto de modelos num modelo só; a temperatura, os alvos suaves e o fator T² estão no corpo do artigo, "
                               "que não foi relido aqui (a perda do projeto tem teste próprio contra a conta à mão)"),
    "1910.01348": ("em parte", "o resumo diz que professores mais exatos muitas vezes não são bons professores e que modelos maiores não costumam ensinar melhor, por "
                               "diferença de capacidade. Sustenta a pergunta (o ganho do professor não chega ao aluno por garantia), não a resposta: o resumo não diz "
                               "em que dados mediu, e lá o professor é uma rede treinada na mesma tarefa; aqui é uma LLM com poucos exemplos no prompt, e o aluno "
                               "não vê rótulo humano"),
    "1902.03393": ("em parte", "o resumo diz que o aluno piora quando a distância para o professor é grande e que um aluno fixo não aproveita um professor "
                               "arbitrariamente grande; experimentos em CIFAR-10, CIFAR-100 e ImageNet, com CNN e ResNet. Não sustenta nada sobre texto nem sobre LLM como "
                               "professora: aqui os dois professores são milhares de vezes maiores que o aluno, e a etapa mede um degrau só (8B para 14B)"),
    "2005.10419": ("em parte", "o resumo diz que o professor estima as probabilidades de classe de Bayes e que o aluno ganha com um conhecimento aproximado delas "
                               "(um compromisso entre viés e variância). Sustenta que o valor dos logits depende da qualidade das probabilidades, e não só do acerto; "
                               "o resumo não compara professores de acertos diferentes"),
    "1909.11723": ("em parte", "o resumo diz que a destilação é uma suavização de rótulos aprendida e que até um professor mal treinado melhora o aluno. Sustenta o "
                               "controle desta etapa (separar os logits da suavização); os experimentos são de visão, e não com rótulos de LLM"),
    "2106.05945": ("em parte", "o resumo diz que sobra uma distância grande entre as previsões do professor e do aluno, e que imitar melhor nem sempre generaliza melhor. "
                               "Sustenta medir a concordância com o professor à parte da qualidade; não trata de professor que só entrega rótulos"),
    "1805.04770": ("em parte", "o resumo diz que alunos com a mesma arquitetura do professor o superam, em visão e em modelagem de linguagem. Sustenta que um aluno passar "
                               "o professor é um fenômeno conhecido; aqui o aluno é muito menor que o professor e de outra família, caso que o resumo não cobre"),
}


def conferir_artigos() -> Path:
    """Pergunta ao arXiv o título, os autores e a data de cada referência e grava a resposta em `artigos/` do cache desta etapa."""
    from especialista.experimentos import etapa10d

    return etapa10d.conferir_artigos(base=pasta_da_etapa(), artigos=ARTIGOS)


def quadro_de_artigos() -> pd.DataFrame:
    """O quadro das referências com o que a fonte devolveu (`etapa10d.quadro_de_artigos`) e o que o resumo sustenta."""
    from especialista.experimentos import etapa10d

    lidos = {ident: {"sustenta": sustenta, "diz": diz} for ident, (sustenta, diz) in RESUMOS_LIDOS.items()}
    return etapa10d.quadro_de_artigos(base=pasta_da_etapa(), artigos=ARTIGOS, lidos=lidos)


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 11b: os alunos do cenário B com o professor aberto maior.")
    comandos = argumentos.add_subparsers(dest="comando", required=True)

    def comando(nome: str, ajuda: str, com_o_professor: bool = True) -> argparse.ArgumentParser:
        novo = comandos.add_parser(nome, help=ajuda)
        if com_o_professor:
            novo.add_argument("--professor", default=PROFESSOR_NOVO, help=f"a versão do professor, como a Etapa 11a a nomeia (padrão: {PROFESSOR_NOVO})")
        return novo

    comando("regras", "grava as regras do professor, com a hora (antes de qualquer treino com ele)")
    comando("artigos", "confere as referências no arXiv", com_o_professor=False)
    treino = comando("treinar", "treina um aluno com uma semente")
    treino.add_argument("--caminho", required=True, choices=sorted(CAMINHOS))
    treino.add_argument("--aluno", required=True, choices=sorted(set(ALUNOS["zero"]) | set(ALUNOS["xlmr"])))
    treino.add_argument("--semente", type=int, required=True)
    treino.add_argument("--fumaca", type=Path, help="pasta descartável: treina com a receita curta do teste de fumaça")
    comando("faltam", "lista o que ainda não rodou, uma execução por linha")
    comando("situacao", "quantas execuções estão prontas")
    da_rodada = comando("rodada", "roda tudo o que falta: a rede do zero em ondas, depois o XLM-R em fila")
    da_rodada.add_argument("--fumaca", type=Path, help="pasta descartável: roda só o teste de fumaça")
    comando("temperatura", "depois da rodada: escolhe a temperatura do B2 em dev e abre o teste")
    comando("calcular", "faz as comparações e grava em <professor>/calculo/")
    comando("entregar", "monta a entrega a partir do cache (o notebook também chama)")
    comando("manifesto", "depois do notebook e do HTML: o que a entrega contém, com impressão digital")
    a = argumentos.parse_args()

    if a.comando == "regras":
        destino = gravar_regras(a.professor)
        print(destino, "escritas em", json.loads(destino.read_text(encoding="utf-8"))["escritas_em"])
    elif a.comando == "artigos":
        print(conferir_artigos())
        print(quadro_de_artigos()[["quem", "identificador", "na_fonte", "titulo_na_fonte", "publicado_na_fonte"]].to_string(index=False))
    elif a.comando == "treinar":
        torch.set_num_threads(2)
        if a.fumaca is not None:
            usar_pasta_descartavel(a.fumaca)
        pasta = executar(a.caminho, a.aluno, a.semente, receita_curta=FUMACA["receita"][a.caminho] if a.fumaca is not None else None, professor=a.professor)
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        print(f"[{datetime.now():%H:%M:%S}]", meta["variante"], a.semente, f"época {meta['melhor_epoca']} de {meta['receita']['epocas']}",
              {k: round(v, 4) for k, v in meta["dev"].items()}, f"{meta['segundos_de_treino']:.0f}s", flush=True)
    elif a.comando == "faltam":
        for caminho, aluno, semente in faltam(None, a.professor):
            print(f"--professor {a.professor} --caminho {caminho} --aluno {aluno} --semente {semente}")
    elif a.comando == "situacao":
        print(situacao(a.professor))
    elif a.comando == "rodada":
        sys.exit(0 if rodada(a.professor, a.fumaca) else 1)
    elif a.comando == "temperatura":
        for caminho in CAMINHOS:
            decidido = decidir_temperatura(caminho, professor=a.professor)
            print(CAMINHOS[caminho]["nome"], "T =", decidido["temperatura_do_b2"], "(com o professor antigo:", f"{decidido['temperatura_com_o_professor_antigo']})",
                  "abertas no teste:", ", ".join(decidido["abertas_no_teste"]))
    elif a.comando == "calcular":
        print(calcular(a.professor))
    elif a.comando == "entregar":
        print(entregar(a.professor))
    elif a.comando == "manifesto":
        print(manifesto(a.professor))


if __name__ == "__main__":
    _linha_de_comando()
