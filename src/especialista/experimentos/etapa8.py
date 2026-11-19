"""Etapa 8: quantização. Cada modelo é exportado para ONNX e convertido para inteiros de 8 bits.

Cada execução parte de uma pasta de modelo (uma que `SistemaDiet.carregar`
lê), e produz, sem nenhum treino:

    pytorch       as respostas do modelo como ele chegou, em 32 bits (a referência)
    fp32          o mesmo modelo exportado para ONNX, ainda em 32 bits
    dinamica      INT8 com a faixa das ativações medida em cada pedido
    estatica      INT8 com a faixa das ativações medida antes, em pedidos do treino
    so_tabela     INT8 só na tabela de features   } só para ver onde
    so_matrizes   INT8 só nas matrizes            } está o custo

Quem é quantizado: o que a Etapa 7 marcou em `entregas/07_poda/decisao.json`
(as variantes podadas que seguem, de cada modelo de partida) e, como
referência, cada modelo de partida sem poda.

    uv run python -m especialista.experimentos.etapa8 faltam
    uv run python -m especialista.experimentos.etapa8 quantizar --modelo aluno_a --variante sem_poda --semente 0
    uv run python -m especialista.experimentos.etapa8 quantizar --tudo
    uv run python -m especialista.experimentos.etapa8 quantizar --tudo --modelo teto    # só um modelo de partida (para rodar vários em paralelo)
    uv run python -m especialista.experimentos.etapa8 latencia

A latência é medida num comando à parte, para rodar com a máquina quieta: o
notebook só lê o que ele gravou. Tudo roda na CPU.

As regras de decisão da etapa, escritas antes de olhar os resultados, estão
em `escolher_forma`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch

from especialista.avaliacao import eficiencia, qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, EXECUCOES, RAIZ
from especialista.compressao import exportacao, quantizacao
from especialista.compressao.exportacao import SistemaOnnx
from especialista.dados import massive
from especialista.modelos.diet import SistemaDiet

ETAPA = "08_quantizacao"
SEMENTES = (0, 1, 2)
DECISAO_DA_PODA = "07_poda/decisao.json"
DECISAO_DA_REDE = "04b_intencao/decisao.json"
DECISAO_DA_DESTILACAO = "06_destilacao/decisao.json"

FORMAS = quantizacao.FORMAS
CANDIDATAS = tuple(forma for forma, f in FORMAS.items() if f["concorre"])   # as que concorrem a modelo final
REFERENCIA = "pytorch"                          # o mesmo modelo em 32 bits, como chegou: é contra ele que tudo é comparado
MEDIDAS = (REFERENCIA, "fp32", *FORMAS)         # tudo o que tem respostas gravadas
COM_LATENCIA = (REFERENCIA, "fp32", *CANDIDATAS)
NOMES = {REFERENCIA: "PyTorch, 32 bits", "fp32": "ONNX, 32 bits", **{forma: f["nome"] for forma, f in FORMAS.items()}}

LIMITE_DO_PLANO = 0.01    # 1 ponto de F1 de entidades: a queda máxima que a quantização pós-treino pode ter (PLANO.md)
CRITERIO_DO_PLANO = "slot_f1"
CRITERIO_DE_DESEMPATE = "exact_match"   # a métrica das regras das Etapas 6 e 7
SEMENTE_DA_CALIBRACAO = 0
RODADAS = 5               # a latência é medida em rodadas alternadas entre os sistemas


def base_padrao() -> Path:
    return EXECUCOES / ETAPA


# --------------------------------------------------------------------------
# Quem é quantizado
# --------------------------------------------------------------------------


def _sem_poda_das_etapas_anteriores() -> dict[str, dict]:
    """Os modelos de partida, sem poda: o aluno principal, o teto e os outros alunos que a Etapa 6 mandou seguir."""
    rede = json.loads((ENTREGAS / DECISAO_DA_REDE).read_text(encoding="utf-8"))
    destilacao = json.loads((ENTREGAS / DECISAO_DA_DESTILACAO).read_text(encoding="utf-8"))
    alunos = {
        destilacao["alunos"][chave]["variante"]: {
            "nome": destilacao["alunos"][chave]["nome"], "curto": f"Aluno {chave}", "pasta": RAIZ / destilacao["alunos"][chave]["modelos"],
            "principal": chave == destilacao["principal"],
        }
        for chave in destilacao["seguem"]
    }
    teto = {"nome": "Teto (rótulos humanos)", "curto": "Teto", "pasta": RAIZ / rede["modelos"], "principal": False}
    return {**{m: a for m, a in alunos.items() if a["principal"]}, "teto": teto, **{m: a for m, a in alunos.items() if not a["principal"]}}


def _rotulo_curto(variante: str) -> str:
    """O nome de uma variante da poda para tabelas e gráficos: `linhas_50` vira "linhas 50%"."""
    palavras = variante.split("_")
    return " ".join(palavras[:-1] + [palavras[-1] + "%"]) if palavras[-1].isdigit() else " ".join(palavras)


def _fichas_da_poda(do_modelo: dict) -> list[dict]:
    """As variantes podadas que a Etapa 7 mandou seguir, de um modelo de partida, cada uma com o seu papel.

    Aceita os dois jeitos de escrever a decisão: uma lista `seguem` de fichas
    mais o nome da `principal`, ou as fichas direto nas chaves `principal` e
    `secundaria`. A principal é a de menor arquivo; a secundária, a melhor de
    pesos soltos (pesos densos com zeros).
    """
    fichas = [dict(f) for f in do_modelo.get("seguem", []) if isinstance(f, dict)]
    principal = do_modelo.get("principal")
    for papel in ("principal", "secundaria"):
        if isinstance(do_modelo.get(papel), dict) and do_modelo[papel].get("modelos"):
            if do_modelo[papel]["variante"] not in [f["variante"] for f in fichas]:
                fichas.append(dict(do_modelo[papel]))
    nome_da_principal = principal["variante"] if isinstance(principal, dict) else principal
    for ficha in fichas:
        ficha["papel"] = "principal" if ficha["variante"] == nome_da_principal else "secundaria"
    return fichas


def modelos_a_quantizar() -> dict[tuple[str, str], dict]:
    """Tudo o que a etapa quantiza, na ordem em que roda: (modelo de partida, variante) -> ficha.

    A ficha tem `nome`, `curto`, `pasta` (com `semente_0`, `semente_1`, ...),
    `papel` (`sem_poda`, `principal` ou `secundaria`) e `final`: verdadeiro
    para a variante que a Etapa 7 marcou como a principal do projeto.

    Enquanto a Etapa 7 não entregar a sua decisão, só existem os modelos sem poda.
    """
    partida = _sem_poda_das_etapas_anteriores()
    arquivo = ENTREGAS / DECISAO_DA_PODA
    poda = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {"modelos": {}, "principal": None}
    do_projeto = poda.get("principal") or {}
    modelos = list(poda["modelos"]) or list(partida)
    fichas: dict[tuple[str, str], dict] = {}
    for m in modelos:
        do_modelo = poda["modelos"].get(m, {})
        de_partida = partida.get(m, {"nome": do_modelo.get("nome", m), "curto": do_modelo.get("nome", m), "principal": False})
        pasta = RAIZ / do_modelo["sem_poda"]["modelos"] if "sem_poda" in do_modelo else de_partida["pasta"]
        podadas = _fichas_da_poda(do_modelo)
        fichas[(m, "sem_poda")] = {
            "nome": f"{de_partida['nome']}, sem poda", "curto": f"{de_partida['curto']}, sem poda", "modelo": m, "variante": "sem_poda",
            "pasta": pasta, "papel": "sem_poda", "familia": None,
            # se a Etapa 7 não aprovou nenhuma poda para o modelo principal, o final parte do modelo sem poda
            "final": do_projeto.get("modelo") == m and do_projeto.get("variante") is None,
        }
        for ficha in podadas:
            v = ficha["variante"]
            fichas[(m, v)] = {
                "nome": f"{de_partida['nome']}, {ficha.get('nome', v)[0].lower() + ficha.get('nome', v)[1:]}",
                "curto": f"{de_partida['curto']}, {_rotulo_curto(v)}", "modelo": m, "variante": v, "pasta": RAIZ / ficha["modelos"], "papel": ficha["papel"],
                "familia": ficha.get("familia"), "final": do_projeto.get("modelo") == m and do_projeto.get("variante") == v,
            }
    return fichas


def textos_de_calibracao(treino: pd.DataFrame, quantos: int = quantizacao.CALIBRACAO) -> list[str]:
    """Os pedidos do treino que medem as faixas da quantização estática: sorteados uma vez, sem rótulo.

    São os mesmos para todos os modelos e sementes. Nenhum pedido de `dev` ou
    do teste entra: a calibração faz parte de construir o modelo.
    """
    sorteio = np.random.default_rng(SEMENTE_DA_CALIBRACAO).choice(len(treino), min(quantos, len(treino)), replace=False)
    return [treino["utt"].iloc[i] for i in sorted(sorteio)]


# --------------------------------------------------------------------------
# Uma execução: exportar, quantizar de cada forma, responder e gravar
# --------------------------------------------------------------------------


def pasta_da_execucao(modelo: str, variante: str, semente: int, base: Path | None = None) -> Path:
    return (base or base_padrao()) / modelo / variante / f"semente_{semente}"


def bytes_do_modelo_original(pasta: Path) -> int:
    """Tamanho em disco do modelo como chegou: pesos em 32 bits, vocabulário e configuração."""
    return sum(arquivo.stat().st_size for arquivo in pasta.iterdir() if arquivo.is_file())


def quantizar_modelo(pasta_do_modelo: Path, destino: Path, medir: dict[str, pd.DataFrame], calibracao: list[str]) -> dict:
    """Leva um modelo (qualquer pasta que `SistemaDiet.carregar` leia) por todas as formas da etapa.

    Grava em `destino`:

    - `sistemas/<forma>/`: o sistema exportado, pronto para `SistemaOnnx.carregar`
      (`fp32` e as formas que concorrem; e `<forma>_compactado/`, com a rede
      passada por um compressor comum);
    - `previsoes_<forma>_<particao>.parquet`: as respostas de cada forma em cada
      partição de `medir`, sempre um pedido por vez.

    Devolve o que foi medido: tamanhos, zeros conservados e, para cada forma, em
    quantos pedidos a resposta é igual à do modelo em PyTorch.
    """
    original = SistemaDiet.carregar(pasta_do_modelo, "cpu")
    textos = {particao: list(tabela["utt"]) for particao, tabela in medir.items()}
    sistemas = destino / "sistemas"
    respostas: dict[str, dict[str, list[Previsao]]] = {REFERENCIA: {p: original.prever(t) for p, t in textos.items()}}

    fp32 = exportacao.exportar(original, sistemas / "fp32", origem=str(pasta_do_modelo))
    respostas["fp32"] = {p: SistemaOnnx.carregar(fp32).prever(t) for p, t in textos.items()}
    medido = {
        REFERENCIA: {"bytes": bytes_do_modelo_original(pasta_do_modelo)},
        "fp32": {"bytes": exportacao.bytes_do_sistema(fp32), "arquivos": exportacao.bytes_por_arquivo(fp32)},
    }
    pesos = exportacao.pesos_da_rede(exportacao.ler_rede(fp32))
    for forma in FORMAS:
        pasta = quantizacao.quantizar(fp32, sistemas / forma, forma, calibracao)
        respostas[forma] = {p: SistemaOnnx.carregar(pasta).prever(t) for p, t in textos.items()}
        compactado = exportacao.compactar(pasta, sistemas / f"{forma}_compactado")
        de_volta = SistemaOnnx.carregar(compactado).prever(textos[next(iter(textos))][:200])
        medido[forma] = {
            "bytes": exportacao.bytes_do_sistema(pasta), "arquivos": exportacao.bytes_por_arquivo(pasta),
            "bytes_compactado": exportacao.bytes_do_sistema(compactado),
            "compactado_responde_igual": de_volta == respostas[forma][next(iter(textos))][:200],
            **quantizacao.zeros_conservados(fp32, pasta),
        }
        if not FORMAS[forma]["concorre"]:   # das formas de diagnóstico ficam só as respostas e os tamanhos
            shutil.rmtree(pasta)
            shutil.rmtree(compactado)
    for forma, por_particao in respostas.items():
        for particao, previsoes in por_particao.items():
            salvar_previsoes(destino / f"previsoes_{forma}_{particao}.parquet", medir[particao]["id"], previsoes)
            if forma != REFERENCIA:
                iguais = sum(a == b for a, b in zip(previsoes, respostas[REFERENCIA][particao]))
                medido[forma][f"iguais_ao_pytorch_{particao}"] = iguais
    return {
        "parametros": original.parametros,
        "features": len(original.featurizador),
        "camadas": len(original.modelo.blocos),
        "pesos_gravados": int(pesos["numeros"].sum()),
        "pesos_em_zero": int(pesos["zeros"].sum()),
        "pedidos": {particao: len(t) for particao, t in textos.items()},
        "formas": medido,
    }


def executar(modelo: str, variante: str, semente: int, base: Path | None = None) -> Path:
    """Quantiza o modelo com a semente dada, ou devolve a pasta se já rodou."""
    pasta = pasta_da_execucao(modelo, variante, semente, base)
    if (pasta / "meta.json").exists():
        return pasta
    ficha = modelos_a_quantizar()[(modelo, variante)]
    particoes, _ = massive.carregar_entrega()
    calibracao = textos_de_calibracao(particoes["train"])
    de_onde = ficha["pasta"] / f"semente_{semente}"

    inicio = time.perf_counter()
    medido = quantizar_modelo(de_onde, pasta, {"dev": particoes["dev"], "test": particoes["test"]}, calibracao)
    meta = {
        "etapa": ETAPA, "modelo": modelo, "variante": variante, "nome": ficha["nome"], "papel": ficha["papel"], "semente": semente,
        "de_partida": str(de_onde.relative_to(RAIZ)) if de_onde.is_relative_to(RAIZ) else str(de_onde),
        **medido,
        "calibracao": {"pedidos": len(calibracao), "de_onde": "pedidos do treino, sem rótulo", "semente_do_sorteio": SEMENTE_DA_CALIBRACAO},
        "segundos": time.perf_counter() - inicio,
        "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, **ambiente(),
    }
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # por último: marca "pronto"
    return pasta


def carregar(modelo: str, variante: str, semente: int, base: Path | None = None) -> dict:
    """O que uma execução produziu: meta e as respostas, `previsoes[forma][particao]`."""
    pasta = pasta_da_execucao(modelo, variante, semente, base)
    previsoes: dict[str, dict[str, list[Previsao]]] = {}
    for forma in MEDIDAS:
        previsoes[forma] = {p: carregar_previsoes(pasta / f"previsoes_{forma}_{p}.parquet")[1] for p in ("dev", "test")}
    return {"pasta": pasta, "meta": json.loads((pasta / "meta.json").read_text(encoding="utf-8")), "previsoes": previsoes}


def grade() -> list[tuple[str, str, int]]:
    return [(m, v, s) for m, v in modelos_a_quantizar() for s in SEMENTES]


def faltam(base: Path | None = None) -> list[tuple[str, str, int]]:
    """As execuções que ainda não estão gravadas."""
    return [(m, v, s) for m, v, s in grade() if not (pasta_da_execucao(m, v, s, base) / "meta.json").exists()]


# --------------------------------------------------------------------------
# Latência: medida à parte, com a máquina quieta
# --------------------------------------------------------------------------


def ambiente() -> dict:
    """A máquina, como em `eficiencia.ambiente`, e as instruções de CPU que mudam a conta com inteiros.

    Sem as instruções VNNI, o ONNX Runtime faz o produto de inteiros de 8 bits
    com uma instrução que pode saturar, e o resultado da rede quantizada pode
    mudar de uma máquina para outra.
    """
    instrucoes = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            marcas = next(linha for linha in f if linha.startswith("flags")).split()
        instrucoes = " ".join(sorted(m for m in marcas if "vnni" in m or m in ("avx2", "avx512f")))
    except (OSError, StopIteration):
        pass
    return {**eficiencia.ambiente(), "instrucoes_da_cpu": instrucoes, "tem_vnni": "vnni" in instrucoes}


def carga_da_maquina() -> float:
    """A carga média do último minuto (quantos processos disputavam a CPU). Perto de zero numa máquina quieta."""
    return float(os.getloadavg()[0])


def relogios() -> tuple[float, float, float]:
    """Três relógios, em segundos: o de parede, o tempo de CPU gasto por todos os processos da máquina e o gasto por este processo."""
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            campos = [int(x) for x in f.readline().split()[1:9]]   # user, nice, system, idle, iowait, irq, softirq, steal
        de_todos = (sum(campos) - campos[3] - campos[4]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        de_todos = float("nan")
    return time.perf_counter(), de_todos, sum(os.times()[:4])


def nucleos_de_outros(antes: tuple[float, float, float], depois: tuple[float, float, float]) -> float:
    """Quantos núcleos de CPU os outros processos ocuparam, em média, entre duas leituras de `relogios`.

    É a medida de "máquina quieta" da etapa: perto de zero, só este processo
    estava trabalhando. A carga média do sistema não serve para isso durante a
    medida, porque conta também o próprio processo que mede.
    """
    parede, de_todos, proprio = (d - a for a, d in zip(antes, depois))
    return max(0.0, (de_todos - proprio) / parede)


def tempo_por_trecho(sistema: SistemaOnnx, textos: list[str], consultas: int = eficiencia.CONSULTAS, semente: int = 0) -> dict[str, float]:
    """Onde vai o tempo de um pedido no sistema exportado, em milissegundos (média).

    Os quatro trechos somam o que `prever` faz: texto para ids, a rede no ONNX
    Runtime, o Viterbi e a montagem da resposta.
    """
    rng = np.random.default_rng(semente)
    sorteados = [textos[i] for i in rng.integers(0, len(textos), consultas)]
    soma = dict.fromkeys(("texto para ids", "rede (ONNX Runtime)", "Viterbi do CRF", "montar a resposta"), 0)
    for texto in sorteados:
        t0 = time.perf_counter_ns()
        tokens = texto.split()
        entradas = sistema.entradas([tokens])
        t1 = time.perf_counter_ns()
        intencoes, emissoes = sistema.sessao.run(["intencoes", "emissoes"], entradas)
        t2 = time.perf_counter_ns()
        caminhos = sistema.decodificar(emissoes, [len(tokens)])
        t3 = time.perf_counter_ns()
        Previsao.de_tags(sistema.intents[int(intencoes[0].argmax())], [sistema.tags[t] for t in caminhos[0]])
        t4 = time.perf_counter_ns()
        for trecho, gasto in zip(soma, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            soma[trecho] += gasto
    return {trecho: total / consultas / 1e6 for trecho, total in soma.items()}


def medir_latencia(modelo: str, variante: str, base: Path | None = None, semente: int = 0, rodadas: int = RODADAS, refazer: bool = False) -> Path:
    """Mede a latência do modelo em PyTorch e de cada forma exportada, no protocolo de `eficiencia.py`.

    Uma consulta por vez, 1 thread, do texto à resposta, com os pedidos de
    `dev`. Os sistemas se alternam em `rodadas` rodadas de 1.024 consultas:
    se a máquina oscilar durante a medida, a oscilação cai sobre todos. Grava
    `latencia.npz` (os tempos, uma linha por rodada) e `latencia.json` (o
    resumo, a máquina e quantos núcleos de CPU os outros processos ocuparam
    durante a medida) na pasta da execução.
    """
    pasta = pasta_da_execucao(modelo, variante, semente, base)
    if (pasta / "latencia.json").exists() and not refazer:
        return pasta
    if not (pasta / "meta.json").exists():
        raise FileNotFoundError(f"Quantize antes: falta {pasta / 'meta.json'}.")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    de_onde = Path(meta["de_partida"])
    torch.set_num_threads(1)
    particoes, _ = massive.carregar_entrega()
    textos = list(particoes["dev"]["utt"])
    sistemas = {REFERENCIA: SistemaDiet.carregar(de_onde if de_onde.is_absolute() else RAIZ / de_onde, "cpu")}
    sistemas.update({forma: SistemaOnnx.carregar(pasta / "sistemas" / forma, threads=1) for forma in COM_LATENCIA if forma != REFERENCIA})

    carga_antes, antes = carga_da_maquina(), relogios()
    tempos = {forma: [] for forma in sistemas}
    for rodada in range(rodadas):
        for forma, sistema in sistemas.items():
            tempos[forma].append(eficiencia.medir_latencia(sistema, textos, semente=rodada))
    trechos = {forma: tempo_por_trecho(sistemas[forma], textos) for forma in sistemas if forma != REFERENCIA}
    carga_depois, depois = carga_da_maquina(), relogios()

    np.savez(pasta / "latencia.npz", **{forma: np.stack(linhas) for forma, linhas in tempos.items()})
    resumo = {
        "modelo": modelo, "variante": variante, "semente": semente, "threads": 1, "rodadas": rodadas, "consultas_por_rodada": eficiencia.CONSULTAS,
        "p50_por_rodada_ms": {forma: [float(np.percentile(t, 50)) for t in linhas] for forma, linhas in tempos.items()},
        "resumo": {forma: eficiencia.resumir_latencia(np.concatenate(linhas)) for forma, linhas in tempos.items()},
        "tempo_por_trecho_ms": trechos,
        "nucleos_de_outros": nucleos_de_outros(antes, depois), "carga_antes": carga_antes, "carga_depois": carga_depois,
        "medido_em": datetime.now().isoformat(timespec="seconds"), "onnxruntime": onnxruntime.__version__, "torch": torch.__version__,
        "python": platform.python_version(), **ambiente(),
    }
    (pasta / "latencia.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pasta


def carregar_latencia(modelo: str, variante: str, base: Path | None = None, semente: int = 0) -> dict | None:
    """Os tempos gravados por `medir_latencia` (`tempos[forma]` com uma linha por rodada) e o resumo. None se ainda não mediu."""
    pasta = pasta_da_execucao(modelo, variante, semente, base)
    if not (pasta / "latencia.json").exists():
        return None
    with np.load(pasta / "latencia.npz") as arquivos:
        tempos = {forma: arquivos[forma] for forma in arquivos.files}
    return {"tempos": tempos, **json.loads((pasta / "latencia.json").read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------
# A regra de decisão da etapa
# --------------------------------------------------------------------------


def escolher_forma(
    gabarito: list[Previsao],
    em_dev: dict[str, list[list[Previsao]]],
    p50_por_rodada: dict[str, list[float]] | None = None,
) -> dict:
    """A regra que diz qual forma de quantizar fica, para um modelo. Tudo em `dev`, com os rótulos humanos.

    `em_dev` traz as respostas de cada semente do modelo em PyTorch
    (`pytorch`) e de cada forma candidata. `p50_por_rodada` traz a latência
    mediana de cada candidata em cada rodada da medida.

    1. **Passa** a forma cujo F1 de entidades cai no máximo 1 ponto em relação
       ao mesmo modelo em 32 bits (diferença pareada, média das sementes). É o
       limite do plano do projeto para a quantização pós-treino.
    2. Se **nenhuma passa**, a quantização pós-treino não serve para esse
       modelo, e entra o treino ciente de quantização (QAT). É o único caso em
       que ele entra.
    3. Se **só uma passa**, é ela.
    4. Se **as duas passam**, sai a que for distinguivelmente pior que a outra
       em exact match (intervalo de 95% da diferença pareada todo de um lado do
       zero). Se nenhuma sair, fica a estática só se a latência mediana dela
       for menor que a da dinâmica em todas as rodadas da medida; senão fica a
       dinâmica, que é a mais simples: não precisa de dados de calibração nem
       guarda faixas no arquivo. O tamanho não desempata: as duas guardam os
       mesmos pesos em 8 bits.

    Devolve `forma` (ou None, se precisar de QAT), `precisa_de_qat`, `motivo` e
    `tabela`, com a diferença de cada candidata para o modelo em 32 bits.
    """
    comparacoes = {forma: qualidade.comparar_grupos(gabarito, em_dev[REFERENCIA], em_dev[forma]) for forma in CANDIDATAS}
    tabela = pd.DataFrame({
        forma: {
            "em_32_bits": c.loc[CRITERIO_DO_PLANO, "a"], "em_8_bits": c.loc[CRITERIO_DO_PLANO, "b"], "diferenca": c.loc[CRITERIO_DO_PLANO, "diferenca"],
            "ic_min": c.loc[CRITERIO_DO_PLANO, "ic_min"], "ic_max": c.loc[CRITERIO_DO_PLANO, "ic_max"],
            "passa": bool(c.loc[CRITERIO_DO_PLANO, "diferenca"] >= -LIMITE_DO_PLANO),
        }
        for forma, c in comparacoes.items()
    }).T
    passam = [forma for forma in CANDIDATAS if tabela.loc[forma, "passa"]]
    entre_as_duas = None
    if not passam:
        forma, motivo = None, "nenhuma forma pós-treino fica dentro de 1 ponto de F1 de entidades"
    elif len(passam) == 1:
        forma, motivo = passam[0], "é a única que fica dentro de 1 ponto de F1 de entidades"
    else:
        entre_as_duas = qualidade.comparar_grupos(gabarito, em_dev["dinamica"], em_dev["estatica"]).loc[CRITERIO_DE_DESEMPATE]   # estática − dinâmica
        if entre_as_duas["ic_max"] < 0:
            forma, motivo = "dinamica", "as duas passam, e a estática é distinguivelmente pior em exact match"
        elif entre_as_duas["ic_min"] > 0:
            forma, motivo = "estatica", "as duas passam, e a dinâmica é distinguivelmente pior em exact match"
        elif p50_por_rodada and all(e < d for e, d in zip(p50_por_rodada["estatica"], p50_por_rodada["dinamica"], strict=True)):
            forma, motivo = "estatica", "as duas passam, sem diferença distinguível de qualidade, e a estática é mais rápida em todas as rodadas"
        else:
            forma, motivo = "dinamica", "as duas passam, sem diferença distinguível de qualidade, e a estática não é mais rápida em todas as rodadas: fica a mais simples"
    return {"forma": forma, "precisa_de_qat": forma is None, "motivo": motivo, "passam": passam, "tabela": tabela, "estatica_menos_dinamica": entre_as_duas}


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 8: exportação para ONNX e quantização em 8 bits. Tudo na CPU.")
    argumentos.add_argument("comando", choices=("faltam", "quantizar", "latencia"))
    argumentos.add_argument("--modelo", help="o modelo de partida: teto ou a variante de um aluno da Etapa 6")
    argumentos.add_argument("--variante", help="sem_poda ou uma variante que a Etapa 7 mandou seguir")
    argumentos.add_argument("--semente", type=int)
    argumentos.add_argument("--tudo", action="store_true", help="quantizar: roda, em sequência, tudo o que falta (com --modelo, só o desse modelo de partida)")
    argumentos.add_argument("--refazer", action="store_true", help="latencia: mede de novo o que já foi medido")
    argumentos.add_argument("--rodadas", type=int, default=RODADAS)
    argumentos.add_argument("--pasta", type=Path, default=None, help="onde gravar; o padrão é data/execucoes/08_quantizacao")
    a = argumentos.parse_args()
    fichas = modelos_a_quantizar()

    if a.comando == "faltam":
        for modelo, variante, semente in faltam(a.pasta):
            print(f"--modelo {modelo} --variante {variante} --semente {semente}")
        return
    if a.comando == "quantizar":
        if a.tudo:
            escolhidas = [execucao for execucao in faltam(a.pasta) if a.modelo in (None, execucao[0])]
        elif a.modelo is None or a.variante is None or a.semente is None:
            argumentos.error("informe --modelo, --variante e --semente (ou --tudo)")
        elif (a.modelo, a.variante) not in fichas:
            argumentos.error(f"não existe {a.modelo} {a.variante}. O que existe: " + ", ".join(f"{m} {v}" for m, v in fichas))
        else:
            escolhidas = [(a.modelo, a.variante, a.semente)]
        for modelo, variante, semente in escolhidas:
            meta = json.loads((executar(modelo, variante, semente, a.pasta) / "meta.json").read_text(encoding="utf-8"))
            f = meta["formas"]
            print(modelo, variante, semente, f"ONNX igual ao PyTorch em {f['fp32']['iguais_ao_pytorch_dev']} de {meta['pedidos']['dev']} pedidos de dev;",
                  "; ".join(f"{forma}: {f[forma]['bytes'] / 1e6:.2f} MB, {f[forma]['iguais_ao_pytorch_dev']} iguais" for forma in CANDIDATAS),
                  f"({f[REFERENCIA]['bytes'] / 1e6:.2f} MB em 32 bits); {meta['segundos']:.0f}s", flush=True)
        return
    # latencia: por padrão, todos os modelos que já foram quantizados com a semente 0
    escolhidas = [(a.modelo, a.variante)] if a.modelo and a.variante else list(fichas)
    print(f"Carga da máquina antes de medir: {carga_da_maquina():.2f} (perto de zero = quieta)", flush=True)
    for modelo, variante in escolhidas:
        medir_latencia(modelo, variante, a.pasta, rodadas=a.rodadas, refazer=a.refazer)
        lido = carregar_latencia(modelo, variante, a.pasta)
        print(modelo, variante, "p50 (ms):", ", ".join(f"{forma} {lido['resumo'][forma]['latencia_p50_ms']:.3f}" for forma in lido["resumo"]),
              f"| outros processos: {lido['nucleos_de_outros']:.2f} núcleo(s) de CPU", flush=True)


if __name__ == "__main__":
    _linha_de_comando()
