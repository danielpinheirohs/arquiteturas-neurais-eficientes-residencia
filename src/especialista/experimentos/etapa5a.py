"""Etapa 5a: as chamadas à LLM fechada (Claude Haiku 5.5).

Estas são as únicas partes do projeto que custam dinheiro por execução. Por
isso ficam em comandos explícitos, e não dentro do notebook:

    uv run python -m especialista.experimentos.etapa5a piloto
    uv run python -m especialista.experimentos.etapa5a lote dev      # ou test, ou train
    uv run python -m especialista.experimentos.etapa5a latencia

Cada comando grava as respostas em `data/execucoes/05a_professor_fechado/`
e não repete o que já está gravado. O notebook só lê.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from especialista.avaliacao import eficiencia
from especialista.caminhos import ENTREGAS, EXECUCOES
from especialista.dados import massive
from especialista.professores import claude
from especialista.professores.claude import Ajuste, ProfessorClaude
from especialista.professores.demonstracoes import Demonstracoes, montar_conjunto

ETAPA = "05a_professor_fechado"
PASTA = EXECUCOES / ETAPA
TAMANHO_DO_PILOTO = 200
TAMANHO_DO_ENVIO = 1500   # pedidos por lote: um envio de 80 MB com o treino inteiro foi recusado pelo servidor (erro 502)

# O piloto compara três jeitos de chamar a LLM, em 200 enunciados de dev.
AJUSTES: dict[str, Ajuste] = {
    "com_exemplos": Ajuste(),                                     # o padrão: 10 demonstrações, sem raciocínio
    "sem_exemplos": Ajuste(k=0),                                  # referência: só as instruções
    "com_raciocinio": Ajuste(raciocinio=True, esforco="low"),     # o modelo pode pensar antes de responder
}


def professor(ajuste: Ajuste, diario: Path | None = None) -> tuple[ProfessorClaude, dict]:
    particoes, rotulos = massive.carregar_entrega()
    demonstracoes = Demonstracoes(montar_conjunto(particoes["train"]))
    return ProfessorClaude(demonstracoes, rotulos, ajuste, diario=diario), particoes


CONJUNTOS_EM_LOTE = ("dev", "test", "train")
CHAMADAS_DIRETAS = ("teste", "latencia")   # o teste de formato e a medição de latência


def pasta_das_respostas(conjunto: str) -> Path:
    """Onde estão as respostas de um conjunto: no cache ou, se ele foi apagado, na entrega da etapa.

    As respostas custaram dinheiro. O notebook as copia para a entrega, e
    assim a etapa pode ser refeita sem chamar a API de novo.
    """
    no_cache = PASTA / conjunto
    return no_cache if (no_cache / "respostas.jsonl").exists() else ENTREGAS / ETAPA / "respostas" / conjunto


def respostas(conjunto: str) -> dict[str, dict]:
    """Os registros gravados de um conjunto, por identificador. O notebook lê por aqui."""
    return claude.ler_registros(pasta_das_respostas(conjunto) / "respostas.jsonl")


def chamadas(conjunto: str) -> list[dict]:
    """Os registros de chamadas diretas (teste de formato, latência), na ordem em que foram feitas."""
    return claude.ler_em_ordem(pasta_das_respostas(conjunto) / "respostas.jsonl")


def amostra_do_piloto(dev):
    return dev.sample(TAMANHO_DO_PILOTO, random_state=0).sort_values("id")


def piloto() -> None:
    for nome, ajuste in AJUSTES.items():
        destino = PASTA / f"piloto_{nome}" / "respostas.jsonl"
        if destino.exists():
            print(f"piloto {nome}: já gravado")
            continue
        prof, particoes = professor(ajuste)
        amostra = amostra_do_piloto(particoes["dev"])
        registros = [prof.chamar(linha.utt, identificador=f"dev-{linha.id}") for linha in amostra.itertuples()]
        claude.gravar(registros, destino)
        (destino.parent / "ajuste.json").write_text(json.dumps(ajuste.como_dict()), encoding="utf-8")
        gasto = sum(claude.custo(r["uso"]) for r in registros)
        erros = sum(r["erro"] is not None for r in registros)
        print(f"piloto {nome}: {len(registros)} chamadas, {erros} com erro, US$ {gasto:.4f}, "
              f"mediana de {np.median([r['ms'] for r in registros]):.0f} ms")


def _ids_da_marca(marca: Path) -> list[str | None]:
    """Os lotes anotados num `lote.json`, um por parte; `None` é uma parte ainda não enviada."""
    if not marca.exists():
        return []
    estado = json.loads(marca.read_text(encoding="utf-8"))
    return estado.get("ids") or [estado["id"]]


def lotes_anotados() -> set[str]:
    """Todos os lotes que já têm dono, de qualquer conjunto."""
    return {i for marca in PASTA.glob("*/lote.json") for i in _ids_da_marca(marca) if i is not None}


def lote(conjunto: str, nome_do_ajuste: str, espera: int = 30) -> None:
    pasta = PASTA / conjunto
    destino = pasta / "respostas.jsonl"
    if destino.exists():
        print(f"lote {conjunto}: já gravado")
        return
    ajuste = AJUSTES[nome_do_ajuste]
    prof, particoes = professor(ajuste)
    dados = particoes[conjunto]
    pasta.mkdir(parents=True, exist_ok=True)

    # no treino, o próprio exemplo nunca entra nas suas demonstrações
    excluir = (lambda linha: int(linha.id)) if conjunto == "train" else (lambda linha: None)
    itens = [(f"{conjunto}-{linha.id}", linha.utt, excluir(linha)) for linha in dados.itertuples()]
    partes = [itens[i : i + TAMANHO_DO_ENVIO] for i in range(0, len(itens), TAMANHO_DO_ENVIO)]

    # `lote.json` guarda o lote de cada parte: uma execução interrompida continua de onde parou, sem reenviar
    marca = pasta / "lote.json"
    ids = (_ids_da_marca(marca) + [None] * len(partes))[: len(partes)]
    for i, parte in enumerate(partes):
        if ids[i] is None:
            ids[i] = claude.enviar_com_cuidado(prof, parte, lotes_anotados)
            marca.write_text(json.dumps({"ids": ids, "ajuste": nome_do_ajuste, "pedidos": len(itens), "por_envio": TAMANHO_DO_ENVIO}), encoding="utf-8")
            print(f"lote {conjunto}: parte {i + 1} de {len(partes)} enviada ({len(parte)} pedidos)", flush=True)

    while True:
        estados = [claude.situacao(prof, lote_id) for lote_id in ids]
        if all(e["terminou"] for e in estados):
            break
        print(f"lote {conjunto}: processando {sum(e['processando'] for e in estados)}, prontas {sum(e['certas'] for e in estados)}", flush=True)
        time.sleep(espera)
    registros = [registro for lote_id in ids for registro in claude.coletar(prof, lote_id)]
    # só grava se o que voltou é exatamente o que foi pedido: um lote trocado ou repetido não passa daqui
    esperados, recebidos = {identificador for identificador, *_ in itens}, [r["id"] for r in registros]
    if set(recebidos) != esperados or len(recebidos) != len(esperados):
        raise RuntimeError(f"lote {conjunto}: voltaram {len(recebidos)} respostas ({len(set(recebidos) & esperados)} das {len(esperados)} esperadas). Nada foi gravado.")
    claude.gravar(registros, destino)
    (pasta / "ajuste.json").write_text(json.dumps(ajuste.como_dict()), encoding="utf-8")
    gasto = sum(claude.custo(r["uso"], em_lote=True) for r in registros)
    print(f"lote {conjunto}: {len(registros)} respostas, {sum(r['erro'] is not None for r in registros)} com erro, US$ {gasto:.4f}", flush=True)


def latencia(nome_do_ajuste: str) -> None:
    pasta = PASTA / "latencia"
    if (pasta / "tempos_ms.npy").exists():
        print("latência: já gravada")
        return
    # o diário guarda cada chamada assim que ela volta: se a medição for interrompida, o que foi pago não se perde
    prof, particoes = professor(AJUSTES[nome_do_ajuste], diario=pasta / f"diario_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    # o mesmo protocolo dos modelos locais: aquecimento, 1.024 consultas de dev, uma por vez, do texto à resposta
    tempos = eficiencia.medir_latencia(prof, list(particoes["dev"]["utt"]))
    pasta.mkdir(parents=True, exist_ok=True)
    np.save(pasta / "tempos_ms.npy", tempos)
    medidas = prof.registros[-len(tempos):]
    claude.gravar(medidas, pasta / "respostas.jsonl")
    (pasta / "meta.json").write_text(json.dumps({
        "ajuste": nome_do_ajuste, "consultas": len(tempos), "quando": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "de_onde": "rede do usuário, Brasil", **eficiencia.ambiente(),
        "prazo_por_chamada_s": claude.PRAZO_EM_SEGUNDOS, "novas_tentativas": claude.TENTATIVAS,
        # o aquecimento também é cobrado, mas só as consultas medidas ficam em `respostas.jsonl`
        "chamadas_com_aquecimento": len(prof.registros),
        "dolares_com_aquecimento": sum(claude.custo(r["uso"]) for r in prof.registros),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"latência: p50 {np.percentile(tempos, 50):.0f} ms, p95 {np.percentile(tempos, 95):.0f} ms, "
          f"US$ {sum(claude.custo(r['uso']) for r in prof.registros):.4f}")


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    comandos = argumentos.add_subparsers(dest="comando", required=True)
    comandos.add_parser("piloto")
    do_lote = comandos.add_parser("lote")
    do_lote.add_argument("conjunto", choices=["dev", "test", "train"])
    do_lote.add_argument("--ajuste", default="com_exemplos", choices=sorted(AJUSTES))
    da_latencia = comandos.add_parser("latencia")
    da_latencia.add_argument("--ajuste", default="com_exemplos", choices=sorted(AJUSTES))
    a = argumentos.parse_args()
    if a.comando == "piloto":
        piloto()
    elif a.comando == "lote":
        lote(a.conjunto, a.ajuste)
    else:
        latencia(a.ajuste)
