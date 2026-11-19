"""Etapa 10c: quantização do XLM-R. Cada modelo é exportado para ONNX e convertido para inteiros de 8 bits.

O mesmo roteiro da Etapa 8 (a quantização da rede do zero), agora no encoder
pré-treinado das Etapas 10a e 10b. Cada execução parte de uma pasta de modelo
(uma que `SistemaEncoder.carregar` lê) e produz, sem nenhum treino:

    pytorch       as respostas do modelo como ele chegou, em 32 bits, na CPU (a referência)
    fp32          o mesmo modelo exportado para ONNX, ainda em 32 bits
    dinamica      INT8 com a faixa das ativações medida em cada pedido
    estatica      INT8 com a faixa das ativações medida antes, em 512 pedidos do treino
    so_tabela     INT8 só nas tabelas de entrada   } só em `dev`, para ver
    so_matrizes   INT8 só nas matrizes             } onde está o custo

Quem é quantizado: o que a Etapa 10b marcou em
`entregas/10b_encoder_poda/decisao.json` (a variante principal e a secundária
de cada modelo de partida) e, como referência, cada modelo de partida sem
poda. Enquanto a 10b não entrega a decisão, só existem os modelos sem poda
da Etapa 10a.

    uv run python -m especialista.experimentos.etapa10c faltam
    uv run python -m especialista.experimentos.etapa10c quantizar --modelo aluno_a --variante sem_poda --semente 0
    uv run python -m especialista.experimentos.etapa10c quantizar --tudo
    uv run python -m especialista.experimentos.etapa10c quantizar --tudo --modelo teto    # só um modelo de partida (para rodar vários em paralelo)
    uv run python -m especialista.experimentos.etapa10c latencia

Tudo roda na CPU. A latência é medida num comando à parte, com a máquina
quieta: o notebook só lê o que ele gravou.

**O que ficou decidido antes de qualquer resultado.** São as regras da Etapa 8,
sem mudança; o código da regra é o mesmo (`etapa8.escolher_forma`).

1. **O que é quantizado** (`quantizacao.py`): o recorte de Zafrir et al.
   (Q8BERT, arXiv 1910.06188), as tabelas de entrada e as camadas lineares,
   com um degrau por tensor (Jacob et al., arXiv 1712.05877). Em 8 bits: a
   tabela de subtokens, a de posições, as 6 matrizes de cada camada do corpo
   e as das duas cabeças. Em 32 bits: vieses, normalizações de camada,
   softmax, GELU, somas e os dois produtos da atenção entre ativações.
2. **Duas formas concorrem**: dinâmica e estática. A estática é calibrada com
   512 pedidos do treino, sem rótulo, os mesmos para todos os modelos
   (`etapa8.textos_de_calibracao`). Nenhum pedido de `dev` ou do teste entra.
3. **Onde se decide**: em `dev`, com os rótulos humanos, na média de 3
   sementes, contra o mesmo modelo em 32 bits (diferença pareada).
4. **Quem passa**: a forma cujo F1 de entidades cai no máximo 1 ponto.
5. **Treino ciente de quantização (QAT)**: só entra se nenhuma forma passar,
   isto é, se o F1 de entidades cair mais de 1 ponto com a quantização
   pós-treino. É o único caso.
6. **Se as duas passam**: sai a que for distinguivelmente pior em exact
   match (intervalo de 95% da diferença pareada todo de um lado do zero). Se
   nenhuma sair, a estática só fica se for mais rápida que a dinâmica em
   todas as rodadas de latência; senão fica a dinâmica, que não precisa de
   calibração.
7. **O teste** é aberto para reportar a forma escolhida em `dev`.
8. **O modelo final deste caminho**: o aluno A, na variante que a Etapa 10b
   marcou como principal, na forma escolhida por esta regra.
9. **Tamanho** é o dos arquivos gravados, com a rede separada do tokenizador.
   **Latência** é a do protocolo de `eficiencia.py` (uma consulta por vez, 1
   thread, do texto à resposta), em rodadas alternadas, com a máquina quieta
   (`etapa8.nucleos_de_outros`).

**Disco.** A rede do XLM-R em 32 bits tem até 1,1 GB. O ONNX de 32 bits só
fica guardado na semente em que o custo é medido (a 0), para a linha "ONNX,
32 bits" da latência. Das formas de diagnóstico e dos arquivos compactados
ficam só as respostas e os tamanhos.
"""

from __future__ import annotations

import argparse
import gc
import json
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

from especialista.avaliacao import eficiencia
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, salvar_previsoes
from especialista.caminhos import ENTREGAS, EXECUCOES, RAIZ
from especialista.compressao import exportacao, exportacao_encoder, poda_encoder, quantizacao
from especialista.compressao.sistema_encoder_onnx import SistemaEncoderOnnx
from especialista.dados import massive
from especialista.experimentos import etapa8
from especialista.modelos.encoder import SistemaEncoder

ETAPA = "10c_encoder_quantizacao"
SEMENTES = (0, 1, 2)
SEMENTE_DO_CUSTO = 0              # tamanho e latência são medidos nos modelos desta semente, como na Etapa 10b
DECISAO_DA_PODA = "10b_encoder_poda/decisao.json"
DECISAO_DO_ENCODER = "10a_encoder/decisao.json"

# As formas, a referência e a regra são as da Etapa 8
FORMAS = quantizacao.FORMAS
CANDIDATAS = etapa8.CANDIDATAS                  # as que concorrem a modelo final: dinâmica e estática
DIAGNOSTICOS = tuple(forma for forma in FORMAS if forma not in CANDIDATAS)
REFERENCIA = etapa8.REFERENCIA                  # o mesmo modelo em 32 bits, no PyTorch: é contra ele que tudo é comparado
MEDIDAS = (REFERENCIA, "fp32", *FORMAS)         # tudo o que tem respostas gravadas
COM_LATENCIA = (REFERENCIA, "fp32", *CANDIDATAS)
NOMES = {**etapa8.NOMES, **quantizacao.NOMES_NO_ENCODER}

LIMITE_DO_PLANO = etapa8.LIMITE_DO_PLANO                # 1 ponto de F1 de entidades
CRITERIO_DO_PLANO = etapa8.CRITERIO_DO_PLANO            # slot_f1
CRITERIO_DE_DESEMPATE = etapa8.CRITERIO_DE_DESEMPATE    # exact_match
RODADAS = etapa8.RODADAS                                # 5 rodadas de latência, alternadas entre os sistemas
THREADS = 4               # para responder `dev` e teste ao quantizar (a latência é sempre com 1)
CONFERIR_COMPACTADO = 200   # pedidos de `dev` em que o sistema compactado é comparado com o comum

escolher_forma = etapa8.escolher_forma                  # a regra de decisão: a mesma função da Etapa 8
textos_de_calibracao = etapa8.textos_de_calibracao      # os 512 pedidos do treino, sorteados uma vez
ambiente, carga_da_maquina, relogios, nucleos_de_outros = etapa8.ambiente, etapa8.carga_da_maquina, etapa8.relogios, etapa8.nucleos_de_outros


def base_padrao() -> Path:
    return EXECUCOES / ETAPA


def particoes_da_forma(forma: str) -> tuple[str, ...]:
    """Em que partições cada forma responde: as de diagnóstico, só em `dev`."""
    return ("dev",) if forma in DIAGNOSTICOS else ("dev", "test")


# --------------------------------------------------------------------------
# Quem é quantizado
# --------------------------------------------------------------------------


def _pastas(por_semente: dict) -> dict[int, Path]:
    return {int(semente): RAIZ / pasta for semente, pasta in por_semente.items()}


def modelos_de_partida(decisao: dict) -> dict[str, dict]:
    """Os modelos sem poda da Etapa 10a, na ordem em que rodam: o aluno principal, o teto e os outros alunos que seguem."""
    alunos, principal = decisao["alunos"], decisao["principal"]
    ficha = lambda sigla: {
        "nome": alunos[sigla]["nome"], "curto": f"Aluno {sigla}", "pastas": _pastas(alunos[sigla]["pastas_por_semente"]), "principal": sigla == principal,
    }
    teto = {"nome": "Teto (rótulos humanos)", "curto": "Teto", "pastas": _pastas(decisao["referencias"]["teto"]["pastas_por_semente"]), "principal": False}
    outros = [sigla for sigla in decisao["seguem"] if sigla != principal]
    return {alunos[principal]["variante"]: ficha(principal), "teto": teto, **{alunos[sigla]["variante"]: ficha(sigla) for sigla in outros}}


def rotulo_curto(variante: str) -> str:
    """O nome de uma variante da Etapa 10b para tabelas e gráficos: `vocabulario_treino_camadas_6` vira "vocabulário do treino e 6 camadas"."""
    palavras = variante.split("_")
    if palavras[0] in ("magnitude", "gradual") and palavras[-1].isdigit():
        return f"{palavras[0]} {palavras[-1]}%"
    partes = []
    if "vocabulario" in palavras[:-1]:
        forma = palavras[palavras.index("vocabulario") + 1]
        partes.append({"treino": "vocabulário do treino", "alfabeto": "vocabulário latino"}.get(forma, f"vocabulário {forma}"))
    if "camadas" in palavras[:-1]:
        partes.append(f"{palavras[palavras.index('camadas') + 1]} camadas")
    return " e ".join(partes) or variante.replace("_", " ")


def fichas_dos_modelos(partida: dict[str, dict], poda: dict | None) -> dict[tuple[str, str], dict]:
    """Junta os modelos de partida com a decisão da Etapa 10b (`poda`; None enquanto ela não existe).

    A decisão tem o formato da Etapa 7: em cada modelo de partida, `sem_poda`,
    `principal`, `secundaria` e `seguem` (a ficha de cada variante podada que
    segue, com `pastas_por_semente`).
    """
    poda = poda or {"modelos": {}, "principal": None}
    do_projeto = poda.get("principal") or {}
    modelos = [m for m in partida if not poda["modelos"] or m in poda["modelos"]] + [m for m in poda["modelos"] if m not in partida]
    fichas: dict[tuple[str, str], dict] = {}
    for m in modelos:
        do_modelo = poda["modelos"].get(m, {})
        de_partida = partida.get(m) or {"nome": do_modelo.get("nome", m), "curto": do_modelo.get("nome", m), "pastas": {}, "principal": False}
        sem_poda = do_modelo.get("sem_poda") or {}
        fichas[(m, "sem_poda")] = {
            "nome": f"{de_partida['nome']}, sem poda", "curto": f"{de_partida['curto']}, sem poda", "modelo": m, "variante": "sem_poda",
            "pastas": _pastas(sem_poda["pastas_por_semente"]) if sem_poda.get("pastas_por_semente") else de_partida["pastas"],
            "papel": "sem_poda", "familia": None, "do_principal": de_partida["principal"],
            # se a Etapa 10b não aprovou nenhuma poda para o modelo principal, o final parte do modelo sem poda
            "final": do_projeto.get("modelo") == m and do_projeto.get("variante") is None,
        }
        for ficha in etapa8._fichas_da_poda(do_modelo):
            v, nome = ficha["variante"], ficha.get("nome", ficha["variante"])
            fichas[(m, v)] = {
                "nome": f"{de_partida['nome']}, {nome[0].lower() + nome[1:]}", "curto": f"{de_partida['curto']}, {rotulo_curto(v)}", "modelo": m, "variante": v,
                "pastas": _pastas(ficha["pastas_por_semente"]), "papel": ficha["papel"], "familia": ficha.get("familia"), "do_principal": de_partida["principal"],
                "final": do_projeto.get("modelo") == m and do_projeto.get("variante") == v,
            }
    return fichas


def modelos_a_quantizar() -> dict[tuple[str, str], dict]:
    """Tudo o que a etapa quantiza, na ordem em que roda: (modelo de partida, variante) -> ficha.

    A ficha tem `nome`, `curto`, `pastas` (a de cada semente, lida por
    `SistemaEncoder.carregar`), `papel` (`sem_poda`, `principal` ou
    `secundaria`), `do_principal` (o modelo de partida é o aluno principal) e
    `final`: verdadeiro para a variante que a Etapa 10b marcou como a
    principal deste caminho.
    """
    partida = modelos_de_partida(json.loads((ENTREGAS / DECISAO_DO_ENCODER).read_text(encoding="utf-8")))
    arquivo = ENTREGAS / DECISAO_DA_PODA
    return fichas_dos_modelos(partida, json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else None)


# --------------------------------------------------------------------------
# Uma execução: exportar, quantizar de cada forma, responder e gravar
# --------------------------------------------------------------------------


def pasta_da_execucao(modelo: str, variante: str, semente: int, base: Path | None = None) -> Path:
    return (base or base_padrao()) / modelo / variante / f"semente_{semente}"


def bytes_do_modelo_original(pasta: Path) -> dict[str, int]:
    """Tamanho em disco do modelo como chegou (a pasta de `SistemaEncoder.salvar`), com a rede separada do tokenizador e das configurações."""
    da_rede = [pasta / nome for nome in poda_encoder.ARQUIVOS_DA_REDE]
    rede = sum(arquivo.stat().st_size for arquivo in da_rede)
    resto = sum(arquivo.stat().st_size for arquivo in pasta.rglob("*") if arquivo.is_file() and arquivo not in da_rede)
    return {"bytes": rede + resto, "rede": rede, "tokenizador_e_configuracao": resto}


def quantizar_modelo(
    pasta_do_modelo: Path, destino: Path, medir: dict[str, pd.DataFrame], calibracao: list[str], threads: int = THREADS, guardar_fp32: bool = True
) -> dict:
    """Leva um modelo (qualquer pasta que `SistemaEncoder.carregar` leia) por todas as formas da etapa.

    Grava em `destino`:

    - `sistemas/<forma>/`: o sistema exportado, pronto para
      `SistemaEncoderOnnx.carregar`. Ficam as formas que concorrem e, com
      `guardar_fp32`, o de 32 bits;
    - `previsoes_<forma>_<particao>.parquet`: as respostas de cada forma,
      sempre um pedido por vez (as formas de diagnóstico, só em `dev`);
    - `erro_dos_pesos.csv` (com `guardar_fp32`): o degrau e o erro de
      arredondamento de cada tensor quantizado.

    Devolve o que foi medido: tamanhos (com e sem compactar), zeros
    conservados e, para cada forma, em quantos pedidos a resposta é igual à
    do modelo em PyTorch.
    """
    torch.set_num_threads(threads)
    textos = {particao: list(tabela["utt"]) for particao, tabela in medir.items()}
    primeira = next(iter(textos))
    sistemas = destino / "sistemas"
    segundos, relogio = {}, time.perf_counter()

    def marcar(trecho: str) -> None:
        nonlocal relogio
        segundos[trecho], relogio = time.perf_counter() - relogio, time.perf_counter()

    original = SistemaEncoder.carregar(pasta_do_modelo, "cpu")
    respostas: dict[str, dict[str, list[Previsao]]] = {REFERENCIA: {p: original.prever(t) for p, t in textos.items()}}
    marcar("pytorch")
    fp32 = exportacao_encoder.exportar(original, sistemas / "fp32", origem=str(pasta_do_modelo))
    do_modelo = {
        "parametros": original.parametros, "camadas": len(original.modelo.encoder.encoder.layer),
        "linhas_da_tabela": int(original.modelo.encoder.get_input_embeddings().num_embeddings),
        "pesos_podaveis": poda_encoder.contar(original.modelo),
    }
    del original
    gc.collect()   # o modelo em PyTorch ocupa mais de 1 GB; daqui em diante só o ONNX Runtime trabalha
    marcar("exportar")

    respostas["fp32"] = {p: SistemaEncoderOnnx.carregar(fp32, threads=threads).prever(t) for p, t in textos.items()}
    marcar("fp32")
    medido = {
        REFERENCIA: bytes_do_modelo_original(pasta_do_modelo),
        "fp32": {"bytes": exportacao_encoder.bytes_do_sistema(fp32), "arquivos": exportacao_encoder.bytes_por_arquivo(fp32)},
    }
    rede = exportacao.ler_rede(fp32)
    pesos = exportacao.pesos_da_rede(rede)
    medido["fp32"]["operacoes_do_grafo"] = {tipo: int(quantas) for tipo, quantas in exportacao.operacoes_da_rede(rede).items()}
    del rede
    for forma in FORMAS:
        pasta = quantizacao.quantizar_encoder(fp32, sistemas / forma, forma, calibracao)
        marcar(f"quantizar_{forma}")
        quantizado = SistemaEncoderOnnx.carregar(pasta, threads=threads)
        respostas[forma] = {p: quantizado.prever(textos[p]) for p in particoes_da_forma(forma) if p in textos}
        del quantizado
        marcar(forma)
        medido[forma] = {
            "bytes": exportacao_encoder.bytes_do_sistema(pasta), "arquivos": exportacao_encoder.bytes_por_arquivo(pasta),
            "operacoes_quantizadas": json.loads((pasta / exportacao_encoder.FICHA).read_text(encoding="utf-8"))["operacoes_quantizadas"],
            "operacoes_do_grafo": {tipo: int(quantas) for tipo, quantas in exportacao.operacoes_da_rede(exportacao.ler_rede(pasta)).items()},
            **quantizacao.zeros_conservados(fp32, pasta),
        }
        if forma == CANDIDATAS[0] and guardar_fp32:   # o erro de arredondamento de cada tensor: os pesos em 8 bits são os mesmos nas duas formas
            quantizacao.erro_dos_pesos(fp32, pasta).to_csv(destino / "erro_dos_pesos.csv")
        if forma in CANDIDATAS:
            compactado = exportacao_encoder.compactar(pasta, sistemas / f"{forma}_compactado")
            de_volta = SistemaEncoderOnnx.carregar(compactado, threads=threads).prever(textos[primeira][:CONFERIR_COMPACTADO])
            medido[forma].update({
                "bytes_compactado": exportacao_encoder.bytes_do_sistema(compactado), "arquivos_compactados": exportacao_encoder.bytes_por_arquivo(compactado),
                "compactado_responde_igual": de_volta == respostas[forma][primeira][:CONFERIR_COMPACTADO],
            })
            shutil.rmtree(compactado)   # fica só o tamanho; `exportacao_encoder.compactar` refaz o arquivo quando for para a entrega
        else:                           # das formas de diagnóstico ficam só as respostas e os tamanhos
            shutil.rmtree(pasta)
        marcar(f"conferir_{forma}")
    if not guardar_fp32:
        shutil.rmtree(fp32)
    for forma, por_particao in respostas.items():
        for particao, previsoes in por_particao.items():
            salvar_previsoes(destino / f"previsoes_{forma}_{particao}.parquet", medir[particao]["id"], previsoes)
            if forma != REFERENCIA:
                medido[forma][f"iguais_ao_pytorch_{particao}"] = sum(a == b for a, b in zip(previsoes, respostas[REFERENCIA][particao]))
    return {
        **do_modelo,
        "pesos_gravados": int(pesos["numeros"].sum()),
        "pesos_em_zero": int(pesos["zeros"].sum()),
        "pedidos": {particao: len(t) for particao, t in textos.items()},
        "fp32_guardado": guardar_fp32,
        "formas": medido,
        "segundos_por_trecho": segundos,
    }


def executar(modelo: str, variante: str, semente: int, base: Path | None = None, threads: int = THREADS) -> Path:
    """Quantiza o modelo com a semente dada, ou devolve a pasta se já rodou."""
    pasta = pasta_da_execucao(modelo, variante, semente, base)
    if (pasta / "meta.json").exists():
        return pasta
    ficha = modelos_a_quantizar()[(modelo, variante)]
    particoes, _ = massive.carregar_entrega()
    calibracao = textos_de_calibracao(particoes["train"])
    de_onde = ficha["pastas"][semente]

    inicio = time.perf_counter()
    medido = quantizar_modelo(de_onde, pasta, {"dev": particoes["dev"], "test": particoes["test"]}, calibracao, threads, guardar_fp32=semente == SEMENTE_DO_CUSTO)
    meta = {
        "etapa": ETAPA, "modelo": modelo, "variante": variante, "nome": ficha["nome"], "papel": ficha["papel"], "semente": semente,
        "de_partida": str(de_onde.relative_to(RAIZ)) if de_onde.is_relative_to(RAIZ) else str(de_onde),
        **medido,
        "calibracao": {"pedidos": len(calibracao), "de_onde": "pedidos do treino, sem rótulo", "semente_do_sorteio": etapa8.SEMENTE_DA_CALIBRACAO},
        "threads": threads, "segundos": time.perf_counter() - inicio,
        "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, **ambiente(),
    }
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # por último: marca "pronto"
    return pasta


def carregar(modelo: str, variante: str, semente: int, base: Path | None = None) -> dict:
    """O que uma execução produziu: meta e as respostas, `previsoes[forma][particao]`."""
    pasta = pasta_da_execucao(modelo, variante, semente, base)
    previsoes = {forma: {p: carregar_previsoes(pasta / f"previsoes_{forma}_{p}.parquet")[1] for p in particoes_da_forma(forma)} for forma in MEDIDAS}
    return {"pasta": pasta, "meta": json.loads((pasta / "meta.json").read_text(encoding="utf-8")), "previsoes": previsoes}


def grade() -> list[tuple[str, str, int]]:
    return [(m, v, s) for m, v in modelos_a_quantizar() for s in SEMENTES]


def faltam(base: Path | None = None) -> list[tuple[str, str, int]]:
    """As execuções que ainda não estão gravadas."""
    return [(m, v, s) for m, v, s in grade() if not (pasta_da_execucao(m, v, s, base) / "meta.json").exists()]


# --------------------------------------------------------------------------
# Latência: medida à parte, com a máquina quieta
# --------------------------------------------------------------------------

TRECHOS = ("texto para subtokens", "rede (ONNX Runtime)", "montar a resposta")


def tempo_por_trecho(sistema: SistemaEncoderOnnx, textos: list[str], consultas: int = eficiencia.CONSULTAS, semente: int = 0) -> dict[str, float]:
    """Onde vai o tempo de um pedido no sistema exportado, em milissegundos (média). Os três trechos somam o que `prever` faz."""
    rng = np.random.default_rng(semente)
    sorteados = [textos[i] for i in rng.integers(0, len(textos), consultas)]
    soma = dict.fromkeys(TRECHOS, 0)
    for texto in sorteados:
        t0 = time.perf_counter_ns()
        palavras = texto.split()
        entradas, real = sistema.alinhador.alinhar([palavras])
        t1 = time.perf_counter_ns()
        intencoes, emissoes = sistema.sessao.run(["intencoes", "emissoes"], entradas)
        t2 = time.perf_counter_ns()
        sistema.montar(intencoes, emissoes, real, [len(palavras)])
        t3 = time.perf_counter_ns()
        for trecho, gasto in zip(soma, (t1 - t0, t2 - t1, t3 - t2)):
            soma[trecho] += gasto
    return {trecho: total / consultas / 1e6 for trecho, total in soma.items()}


def medir_latencia(
    modelo: str, variante: str, base: Path | None = None, semente: int = SEMENTE_DO_CUSTO, rodadas: int = RODADAS, refazer: bool = False,
    consultas: int = eficiencia.CONSULTAS,
) -> Path:
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
    if not (pasta / "sistemas" / "fp32" / exportacao_encoder.REDE).exists():
        raise FileNotFoundError(f"O ONNX de 32 bits só fica guardado na semente {SEMENTE_DO_CUSTO}: falta {pasta / 'sistemas' / 'fp32'}.")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    de_onde = Path(meta["de_partida"])
    torch.set_num_threads(1)
    particoes, _ = massive.carregar_entrega()
    textos = list(particoes["dev"]["utt"])
    sistemas = {REFERENCIA: SistemaEncoder.carregar(de_onde if de_onde.is_absolute() else RAIZ / de_onde, "cpu")}
    sistemas.update({forma: SistemaEncoderOnnx.carregar(pasta / "sistemas" / forma, threads=1) for forma in COM_LATENCIA if forma != REFERENCIA})

    carga_antes, antes = carga_da_maquina(), relogios()
    tempos = {forma: [] for forma in sistemas}
    for rodada in range(rodadas):
        for forma, sistema in sistemas.items():
            tempos[forma].append(eficiencia.medir_latencia(sistema, textos, consultas=consultas, semente=rodada))
    trechos = {forma: tempo_por_trecho(sistemas[forma], textos, consultas) for forma in sistemas if forma != REFERENCIA}
    carga_depois, depois = carga_da_maquina(), relogios()

    np.savez(pasta / "latencia.npz", **{forma: np.stack(linhas) for forma, linhas in tempos.items()})
    resumo = {
        "modelo": modelo, "variante": variante, "semente": semente, "threads": 1, "rodadas": rodadas, "consultas_por_rodada": consultas,
        "p50_por_rodada_ms": {forma: [float(np.percentile(t, 50)) for t in linhas] for forma, linhas in tempos.items()},
        "resumo": {forma: eficiencia.resumir_latencia(np.concatenate(linhas)) for forma, linhas in tempos.items()},
        "tempo_por_trecho_ms": trechos,
        "nucleos_de_outros": nucleos_de_outros(antes, depois), "carga_antes": carga_antes, "carga_depois": carga_depois,
        "medido_em": datetime.now().isoformat(timespec="seconds"), "onnxruntime": onnxruntime.__version__, "torch": torch.__version__,
        "python": platform.python_version(), **ambiente(),
    }
    (pasta / "latencia.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pasta


def carregar_latencia(modelo: str, variante: str, base: Path | None = None, semente: int = SEMENTE_DO_CUSTO) -> dict | None:
    """Os tempos gravados por `medir_latencia` (`tempos[forma]` com uma linha por rodada) e o resumo. None se ainda não mediu."""
    pasta = pasta_da_execucao(modelo, variante, semente, base)
    if not (pasta / "latencia.json").exists():
        return None
    with np.load(pasta / "latencia.npz") as arquivos:
        tempos = {forma: arquivos[forma] for forma in arquivos.files}
    return {"tempos": tempos, **json.loads((pasta / "latencia.json").read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 10c: exportação do XLM-R para ONNX e quantização em 8 bits. Tudo na CPU.")
    argumentos.add_argument("comando", choices=("faltam", "quantizar", "latencia"))
    argumentos.add_argument("--modelo", help="o modelo de partida: teto ou a variante de um aluno da Etapa 10a")
    argumentos.add_argument("--variante", help="sem_poda ou uma variante que a Etapa 10b mandou seguir")
    argumentos.add_argument("--semente", type=int)
    argumentos.add_argument("--tudo", action="store_true", help="quantizar: roda, em sequência, tudo o que falta (com --modelo, só o desse modelo de partida)")
    argumentos.add_argument("--threads", type=int, default=THREADS, help="quantizar: threads de CPU para responder dev e teste")
    argumentos.add_argument("--refazer", action="store_true", help="latencia: mede de novo o que já foi medido")
    argumentos.add_argument("--rodadas", type=int, default=RODADAS)
    argumentos.add_argument("--consultas", type=int, default=eficiencia.CONSULTAS, help="latencia: consultas por rodada (o protocolo usa 1.024)")
    argumentos.add_argument("--pasta", type=Path, default=None, help="onde gravar; o padrão é data/execucoes/10c_encoder_quantizacao")
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
            meta = json.loads((executar(modelo, variante, semente, a.pasta, a.threads) / "meta.json").read_text(encoding="utf-8"))
            f = meta["formas"]
            print(modelo, variante, semente, f"ONNX igual ao PyTorch em {f['fp32']['iguais_ao_pytorch_dev']} de {meta['pedidos']['dev']} pedidos de dev;",
                  "; ".join(f"{forma}: {f[forma]['bytes'] / 1e6:.1f} MB, {f[forma]['iguais_ao_pytorch_dev']} iguais" for forma in CANDIDATAS),
                  f"({f[REFERENCIA]['bytes'] / 1e6:.1f} MB em 32 bits); {meta['segundos']:.0f}s", flush=True)
        return
    # latencia: por padrão, todos os modelos que já foram quantizados com a semente do custo
    escolhidas = [(a.modelo, a.variante)] if a.modelo and a.variante else [chave for chave in fichas if (pasta_da_execucao(*chave, SEMENTE_DO_CUSTO, a.pasta) / "meta.json").exists()]
    print(f"Carga da máquina antes de medir: {carga_da_maquina():.2f} (perto de zero = quieta)", flush=True)
    for modelo, variante in escolhidas:
        medir_latencia(modelo, variante, a.pasta, rodadas=a.rodadas, refazer=a.refazer, consultas=a.consultas)
        lido = carregar_latencia(modelo, variante, a.pasta)
        print(modelo, variante, "p50 (ms):", ", ".join(f"{forma} {lido['resumo'][forma]['latencia_p50_ms']:.2f}" for forma in lido["resumo"]),
              f"| outros processos: {lido['nucleos_de_outros']:.2f} núcleo(s) de CPU", flush=True)


if __name__ == "__main__":
    _linha_de_comando()
