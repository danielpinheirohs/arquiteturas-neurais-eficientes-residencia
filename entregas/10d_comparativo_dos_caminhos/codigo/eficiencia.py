"""Régua de eficiência: quanto custa obter uma resposta.

Protocolo de latência, inspirado no cenário single-stream do MLPerf
Inference (Reddi et al., arXiv 1911.02549):

- uma consulta por vez (batch 1), em sequência;
- o relógio cobre tudo o que o sistema faz com o texto: separar palavras,
  calcular e montar a resposta;
- chamadas de aquecimento antes de medir;
- pelo menos 1.024 consultas medidas; reportam-se percentis, não só a média.

O ambiente é gravado junto com cada medida: latência sem o nome da máquina
não é comparável (Dehghani et al., "The Efficiency Misnomer", arXiv 2110.12894).
"""

from __future__ import annotations

import os
import pickle
import platform
import time

import numpy as np

from especialista.avaliacao.previsao import Sistema

CONSULTAS = 1024
AQUECIMENTO = 50


def medir_latencia(
    sistema: Sistema,
    textos: list[str],
    consultas: int = CONSULTAS,
    aquecimento: int = AQUECIMENTO,
    semente: int = 0,
) -> np.ndarray:
    """Tempo de cada consulta, em milissegundos."""
    rng = np.random.default_rng(semente)
    sorteados = [textos[i] for i in rng.integers(0, len(textos), aquecimento + consultas)]
    for texto in sorteados[:aquecimento]:
        sistema.prever([texto])
    tempos = np.empty(consultas)
    for i, texto in enumerate(sorteados[aquecimento:]):
        inicio = time.perf_counter_ns()
        sistema.prever([texto])
        tempos[i] = (time.perf_counter_ns() - inicio) / 1e6
    return tempos


def resumir_latencia(tempos_ms: np.ndarray) -> dict[str, float]:
    p50, p90, p95, p99 = np.percentile(tempos_ms, [50, 90, 95, 99])
    return {
        "latencia_p50_ms": float(p50),
        "latencia_p90_ms": float(p90),
        "latencia_p95_ms": float(p95),
        "latencia_p99_ms": float(p99),
        "latencia_media_ms": float(tempos_ms.mean()),
    }


def bytes_em_disco(objeto) -> int:
    """Tamanho do sistema serializado. Para redes, as etapas seguintes medem o arquivo exportado."""
    return len(pickle.dumps(objeto, protocol=pickle.HIGHEST_PROTOCOL))


def ambiente() -> dict[str, str | int]:
    """A máquina em que a medida foi feita."""
    cpu = platform.processor() or "desconhecida"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(linha.split(":", 1)[1].strip() for linha in f if linha.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "cpu": cpu,
        "cpus_logicas": os.cpu_count() or 0,
        "plataforma": "WSL2" if "microsoft" in platform.release().lower() else platform.system(),
        "python": platform.python_version(),
    }
