"""Etapa 5b: a LLM aberta, rodando na GPU local.

    uv run python -m especialista.experimentos.etapa5b rotular --modelo qwen3_8b --conjunto dev
    uv run python -m especialista.experimentos.etapa5b conferir --modelo qwen3_8b
    uv run python -m especialista.experimentos.etapa5b latencia --modelo qwen3_8b
    uv run python -m especialista.experimentos.etapa5b escolhido     # aplica a regra e diz qual é o professor

Não custa dinheiro, custa horas de GPU. Por isso segue a mesma disciplina da
Etapa 5a: os comandos gravam em `data/execucoes/05b_professor_aberto/`, cada
resposta vai para o disco assim que sai, uma execução interrompida continua
de onde parou, e o notebook só lê.

**A escolha do professor.** Os candidatos rotulam `dev` inteiro. A regra,
escrita antes de olhar os resultados, está em `escolher`: parte-se do
candidato mais barato, e um mais caro só toma o lugar se o exact match dele
em `dev` for distinguivelmente maior (intervalo de 95% da diferença pareada
todo acima de zero).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from especialista.avaliacao import eficiencia, qualidade
from especialista.avaliacao.previsao import ouro
from especialista.caminhos import ENTREGAS, EXECUCOES
from especialista.dados import massive
from especialista.professores import claude
from especialista.professores.demonstracoes import Demonstracoes, montar_conjunto
from especialista.professores.local import DEPOIS, INICIO, ProfessorLocal
from especialista.professores.rotulagem import ler_conjunto

ETAPA = "05b_professor_aberto"
PASTA = EXECUCOES / ETAPA
CONJUNTOS = ("dev", "test", "train")
RESERVA_DE_VRAM_GB = 0.5     # abaixo disso de memória livre na placa, o lote diminui

# Candidatos sem restrição de acesso no Hugging Face, do mais barato para o mais caro.
# `lote` é quantos pedidos vão juntos pela placa de 10 GB.
CANDIDATOS: dict[str, dict] = {
    "qwen3_4b": {"id": "Qwen/Qwen3-4B", "nome": "Qwen3-4B", "licenca": "Apache 2.0", "lote": 8},
    "gaia_4b": {"id": "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it", "nome": "GAIA 4B (CEIA-UFG)", "licenca": "Gemma", "lote": 8},
    "qwen3_8b": {"id": "Qwen/Qwen3-8B", "nome": "Qwen3-8B", "licenca": "Apache 2.0", "lote": 6},
}


def professor(apelido: str) -> tuple[ProfessorLocal, dict]:
    particoes, rotulos = massive.carregar_entrega()
    candidato = CANDIDATOS[apelido]
    demonstracoes = Demonstracoes(montar_conjunto(particoes["train"]))
    return ProfessorLocal(candidato["id"], demonstracoes, rotulos, nome=candidato["nome"]), particoes


def pasta_das_respostas(apelido: str, parte: str) -> Path:
    """Onde estão as respostas: no cache ou, se ele foi apagado, na entrega da etapa."""
    no_cache = PASTA / apelido / parte
    return no_cache if no_cache.exists() else ENTREGAS / ETAPA / "respostas" / apelido / parte


def respostas(apelido: str, conjunto: str) -> dict[str, dict]:
    """Os registros gravados de um conjunto, por identificador. O notebook lê por aqui."""
    return claude.ler_registros(pasta_das_respostas(apelido, conjunto) / "respostas.jsonl")


def parametros_do_modelo(apelido: str) -> int:
    """Quantos parâmetros o modelo tem, lidos dos arquivos de pesos em disco.

    Não dá para contar no modelo carregado: em 4 bits os pesos ficam
    compactados, e a contagem sai pela metade. No GAIA entra também a parte
    de visão do Gemma 3, que este projeto não usa.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    pasta = Path(snapshot_download(CANDIDATOS[apelido]["id"], local_files_only=True, allow_patterns=["*.safetensors"]))
    total = 0
    for arquivo in sorted(pasta.glob("*.safetensors")):
        with safe_open(arquivo, framework="np") as pesos:
            total += sum(int(np.prod(pesos.get_slice(nome).get_shape())) for nome in pesos.keys())
    return total


def _maquina(prof: ProfessorLocal, apelido: str) -> dict:
    return {
        "quantizacao": "4 bits (NF4, bitsandbytes)",
        "vram_do_modelo_gb": round(prof.vram_do_modelo_gb, 2),
        "vram_de_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0),
        "parametros": parametros_do_modelo(apelido),
        "tokens_das_instrucoes_em_memoria": len(prof._prefixo),
        "quando": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def rotular(apelido: str, conjunto: str, limite: int | None = None) -> None:
    pasta = PASTA / apelido / conjunto
    destino = pasta / "respostas.jsonl"
    particoes, _ = massive.carregar_entrega()
    dados = particoes[conjunto] if limite is None else particoes[conjunto].iloc[:limite]
    feitos = set(claude.ler_registros(destino)) if destino.exists() else set()
    # no treino, o próprio exemplo nunca entra nas suas demonstrações
    excluir = (lambda linha: int(linha.id)) if conjunto == "train" else (lambda linha: None)
    itens = [(f"{conjunto}-{linha.id}", linha.utt, excluir(linha)) for linha in dados.itertuples() if f"{conjunto}-{linha.id}" not in feitos]
    if not itens:
        print(f"{apelido} {conjunto}: já gravado ({len(feitos)} respostas)")
        return

    torch.cuda.reset_peak_memory_stats()
    prof, _ = professor(apelido)
    # pedidos de tamanho parecido vão juntos: menos enchimento, menos conta jogada fora.
    # Os maiores primeiro: se o lote não couber na placa, isso aparece no começo, e não depois de horas.
    itens.sort(key=lambda item: -len(prof.tokens_do_prompt(item[1], item[2])))
    pasta.mkdir(parents=True, exist_ok=True)
    lote, inicio, prontos = CANDIDATOS[apelido]["lote"], time.perf_counter(), 0
    with destino.open("a", encoding="utf-8") as arquivo:
        while prontos < len(itens):
            # se outro programa ocupar a placa, a memória passa a vir da RAM e tudo fica dezenas de vezes mais lento
            if lote > 1 and torch.cuda.mem_get_info()[0] / 1e9 < RESERVA_DE_VRAM_GB:
                torch.cuda.empty_cache()
                if torch.cuda.mem_get_info()[0] / 1e9 < RESERVA_DE_VRAM_GB:
                    lote = max(1, lote // 2)
                    print(f"{apelido} {conjunto}: pouca memória livre na placa, lote reduzido para {lote}", flush=True)
            for registro in prof.responder(itens[prontos : prontos + lote]):
                arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
            arquivo.flush()
            antes, prontos = prontos, min(prontos + lote, len(itens))
            if prontos // 200 > antes // 200 or prontos == len(itens):
                ritmo = (time.perf_counter() - inicio) / prontos
                print(f"{apelido} {conjunto}: {prontos} de {len(itens)}, {ritmo:.2f} s por pedido, faltam {ritmo * (len(itens) - prontos) / 60:.0f} min", flush=True)

    segundos = time.perf_counter() - inicio
    anterior = json.loads((pasta / "meta.json").read_text(encoding="utf-8")) if (pasta / "meta.json").exists() else {}
    (pasta / "meta.json").write_text(json.dumps({
        **{k: v for k, v in CANDIDATOS[apelido].items() if k != "lote"},
        **_maquina(prof, apelido), "lote": CANDIDATOS[apelido]["lote"],
        "pedidos": len(feitos) + len(itens),
        # soma os trechos de uma execução retomada; não inclui o tempo de carregar o modelo
        "segundos_de_gpu": round(anterior.get("segundos_de_gpu", 0.0) + segundos, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{apelido} {conjunto}: {len(itens)} respostas em {segundos / 60:.1f} min", flush=True)


def conferir(apelido: str, quantos: int = 32) -> None:
    """Compara o caminho rápido com as contas diretas, sem reaproveitamento, nos mesmos pedidos de `dev`."""
    destino = PASTA / apelido / "conferencia.json"
    if destino.exists():
        print(f"{apelido} conferência: já gravada")
        return
    prof, particoes = professor(apelido)
    amostra = particoes["dev"].sample(quantos, random_state=0)
    sequencias = [prof.tokens_do_prompt(texto) for texto in amostra["utt"]]
    lote = CANDIDATOS[apelido]["lote"]

    um_a_um = [prof._pontuar_e_gerar([seq]) for seq in sequencias]
    em_lote = [prof._pontuar_e_gerar(sequencias[i : i + lote]) for i in range(0, quantos, lote)]
    pontos_1 = torch.cat([r[0] for r in um_a_um])
    pontos_l = torch.cat([r[0] for r in em_lote])
    texto_1 = [r[2][0][0] for r in um_a_um]
    texto_l = [texto for r in em_lote for texto, _ in r[2]]
    escolha_1 = [r[1][0] for r in um_a_um]
    escolha_l = [i for r in em_lote for i in r[1]]

    # a conta direta dos 60 nomes custa 60 passadas por pedido: só nos primeiros
    diretos = min(8, quantos)
    referencia = torch.stack([prof.pontuar_de_referencia(seq) for seq in sequencias[:diretos]])
    da_biblioteca = [prof.gerar_de_referencia(seq, i) for seq, i in zip(sequencias, escolha_1)]

    def lida(pedido: str, escolhida: int, texto: str | None):
        """A previsão que sai de uma intenção escolhida e do texto gerado para as entidades."""
        completa = None if texto is None else INICIO + prof.intents[escolhida] + DEPOIS + texto
        return prof.interpretar({"texto": completa}, pedido)[0]

    def mesma_resposta(escolhas, textos) -> int:
        """Em quantos pedidos a resposta, depois de lida, é igual à do caminho rápido um a um."""
        return sum(lida(p, i, x) == lida(p, j, y) for p, i, x, j, y in zip(amostra["utt"], escolha_1, texto_1, escolhas, textos))

    resultado = {
        "pedidos": quantos, "lote": lote,
        "pontuacao_rapida_contra_direta": {
            "pedidos": diretos,
            "maior_diferenca_de_probabilidade": float((pontos_1[:diretos].exp() - referencia.exp()).abs().max()),
            "mesma_intencao": int((pontos_1[:diretos].argmax(1) == referencia.argmax(1)).sum()),
        },
        "geracao_propria_contra_a_da_biblioteca": {
            "mesmo_texto": sum(a == b for a, b in zip(texto_1, da_biblioteca)),
            "mesma_resposta_depois_de_lida": mesma_resposta(escolha_1, da_biblioteca),
        },
        "em_lote_contra_um_a_um": {
            "maior_diferenca_de_probabilidade": float((pontos_l.exp() - pontos_1.exp()).abs().max()),
            "mesma_intencao": sum(a == b for a, b in zip(escolha_1, escolha_l)),
            "mesmo_texto": sum(a == b for a, b in zip(texto_1, texto_l)),
            "mesma_resposta_depois_de_lida": mesma_resposta(escolha_l, texto_l),
        },
        "quando": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


def latencia(apelido: str) -> None:
    pasta = PASTA / apelido / "latencia"
    if (pasta / "tempos_ms.npy").exists():
        print(f"{apelido} latência: já gravada")
        return
    torch.cuda.reset_peak_memory_stats()
    prof, particoes = professor(apelido)
    # o mesmo protocolo de todos os sistemas: aquecimento, 1.024 consultas de dev, uma por vez, do texto à resposta
    tempos = eficiencia.medir_latencia(prof, list(particoes["dev"]["utt"]))
    pasta.mkdir(parents=True, exist_ok=True)
    np.save(pasta / "tempos_ms.npy", tempos)
    claude.gravar(prof.registros[-len(tempos):], pasta / "respostas.jsonl")
    (pasta / "meta.json").write_text(json.dumps({
        **{k: v for k, v in CANDIDATOS[apelido].items() if k != "lote"}, **_maquina(prof, apelido), "consultas": len(tempos), **eficiencia.ambiente(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{apelido} latência: p50 {np.percentile(tempos, 50):.0f} ms, p95 {np.percentile(tempos, 95):.0f} ms", flush=True)


def escolher(candidatos_em_dev: dict[str, list]) -> tuple[str, dict]:
    """A regra de escolha do professor. `candidatos_em_dev` traz as previsões em `dev` de cada apelido.

    Na ordem de `CANDIDATOS` (do mais barato para o mais caro): o primeiro é o
    titular; cada candidato seguinte só toma o lugar se o exact match dele for
    distinguivelmente maior que o do titular (intervalo de 95% da diferença
    pareada todo acima de zero).
    """
    particoes, _ = massive.carregar_entrega()
    gabarito = ouro(particoes["dev"])
    ordem = [apelido for apelido in CANDIDATOS if apelido in candidatos_em_dev]
    titular, disputas = ordem[0], {}
    for desafiante in ordem[1:]:
        comparacao = qualidade.comparar(gabarito, candidatos_em_dev[titular], candidatos_em_dev[desafiante]).loc["exact_match"]
        disputas[desafiante] = {"contra": titular, **{k: float(comparacao[k]) for k in ("a", "b", "diferenca", "ic_min", "ic_max")}}
        if comparacao["ic_min"] > 0:
            titular = desafiante
    return titular, disputas


def escolhido() -> str:
    """Aplica a regra aos candidatos que já rotularam `dev` inteiro e devolve o apelido do professor."""
    particoes, _ = massive.carregar_entrega()
    completos = {}
    for apelido in CANDIDATOS:
        arquivo = pasta_das_respostas(apelido, "dev") / "respostas.jsonl"
        if arquivo.exists() and len(claude.ler_registros(arquivo)) == len(particoes["dev"]):
            completos[apelido] = previsoes(apelido, "dev")[0]
    return escolher(completos)[0]


def previsoes(apelido: str, conjunto: str) -> tuple[list, dict, list[dict]]:
    """As respostas gravadas de um conjunto, lidas na ordem dos dados: previsões, diagnóstico e registros."""
    particoes, rotulos = massive.carregar_entrega()
    return ler_conjunto(respostas(apelido, conjunto), particoes[conjunto], conjunto, rotulos["intents"], rotulos["slot_types"])


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    comandos = argumentos.add_subparsers(dest="comando", required=True)
    for nome in ("rotular", "conferir", "latencia"):
        comando = comandos.add_parser(nome)
        comando.add_argument("--modelo", required=True, choices=list(CANDIDATOS))
        if nome == "rotular":
            comando.add_argument("--conjunto", required=True, choices=CONJUNTOS)
            comando.add_argument("--limite", type=int, default=None, help="só os primeiros N pedidos (para testar)")
    comandos.add_parser("escolhido")
    a = argumentos.parse_args()
    if a.comando == "escolhido":
        print(escolhido())
    elif a.comando == "rotular":
        rotular(a.modelo, a.conjunto, a.limite)
    elif a.comando == "conferir":
        conferir(a.modelo)
    else:
        latencia(a.modelo)
