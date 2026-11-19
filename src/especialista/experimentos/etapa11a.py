"""Etapa 11a: o professor aberto sem o limite dos 10 GB.

A pergunta: o cenário B perde por ser aberto, ou porque o professor teve de
caber em 10 GB? A Etapa 5b escolheu o Qwen3-8B em 4 bits porque era o que
cabia na placa local. Aqui a mesma escada continua numa placa alugada
(Colab), com a mesma regra, o mesmo prompt, o mesmo código e o mesmo lote:

1. Qwen3-8B sem os 4 bits (só sai a compressão);
2. Qwen3-14B sem os 4 bits.

**Cada professor é uma versão.** O nome de uma versão diz o que ela é,
`<modelo>__<precisão>__<máquina>`, e cada uma tem a sua pasta em
`data/execucoes/11a_professor_aberto_maior/<versão>/`, com um `versao.json`
que registra tudo o que é preciso para refazer ou comparar. Nada da Etapa 5b
é sobrescrito: a versão dela (`qwen3_8b__nf4__local_3080`) ganha só um
`versao.json` retroativo, que aponta para os arquivos onde eles já estão.

**Tudo o que decide está escrito aqui, antes de qualquer resultado de dev**:
`REGRA_DA_ESCADA`, `REGRA_DE_PRECISAO` e `LIMIAR_DE_AMBIENTE`.

Este arquivo roda em dois lugares. Na máquina local só se usam as funções
leves (regra, comparação, registro de versão). As pesadas (`trabalho`)
rodam no Colab, levadas pelo condutor `etapa11a_colab.py`; por isso `torch`
e `transformers` são importados dentro das funções que precisam deles.

    python -m especialista.experimentos.etapa11a regra          # grava a regra, com data
    python -m especialista.experimentos.etapa11a retroativa     # versao.json da versão da Etapa 5b
    python -m especialista.experimentos.etapa11a lotes          # os lotes da conferência de ambiente
    python -m especialista.experimentos.etapa11a comparar --versao qwen3_8b__nf4__colab_l4
    python -m especialista.experimentos.etapa11a trabalho --versao ... --tarefas dev,test,train   # no Colab
    python -m especialista.experimentos.etapa11a fechar         # formato da 5b, escada, tabelas, custo (depois das rodadas)
    python -m especialista.experimentos.etapa11a manifesto      # depois do notebook: o que a entrega contém
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from especialista.caminhos import ENTREGAS, EXECUCOES, RAIZ, entrega
from especialista.dados import massive
from especialista.professores import prompt, resposta
from especialista.professores.demonstracoes import K, POR_INTENCAO, Demonstracoes, montar_conjunto

ETAPA = "11a_professor_aberto_maior"
PASTA = EXECUCOES / ETAPA
ETAPA_5B = "05b_professor_aberto"
CONJUNTOS = ("dev", "test", "train")
LOTE = 6                         # o mesmo da Etapa 5b para o Qwen3-8B; não muda com a placa

# ---------------------------------------------------------------------------
# As versões
# ---------------------------------------------------------------------------

MODELOS = {
    # o commit do 8B é o que está no cache local e que a Etapa 5b usou
    "qwen3_8b": {"repositorio": "Qwen/Qwen3-8B", "nome": "Qwen3-8B", "commit": "b968826d9c46dd6066d109eabc6255188de91218"},
    # o commit do 14B é lido do Hugging Face na primeira carga e fica gravado em `commit.txt` da versão
    "qwen3_14b": {"repositorio": "Qwen/Qwen3-14B", "nome": "Qwen3-14B", "commit": None},
    # extensão da escada (ver `REGRA_DA_EXTENSAO`): commit lido do Hugging Face em 18/11, antes da rodada
    "qwen3_32b": {"repositorio": "Qwen/Qwen3-32B", "nome": "Qwen3-32B", "commit": "9216db5781bf21249d130ec9da846c4624c16137"},
}
PRECISOES = {
    "nf4": {"descricao": "4 bits (NF4, bitsandbytes, quantização dupla)", "bits_por_peso": 4, "tipo_de_calculo": "float16"},
    "fp16": {"descricao": "sem quantização, float16", "bits_por_peso": 16, "tipo_de_calculo": "float16"},
    "bf16": {"descricao": "sem quantização, bfloat16", "bits_por_peso": 16, "tipo_de_calculo": "bfloat16"},
}
VERSAO_DA_5B = "qwen3_8b__nf4__local_3080"


def descrever(versao: str) -> dict:
    """Do nome de uma versão (`modelo__precisao__maquina`) para o que ela é."""
    modelo, precisao, maquina = versao.split("__")
    return {"id": versao, "modelo": modelo, **MODELOS[modelo], "precisao": precisao, **PRECISOES[precisao], "maquina": maquina}


# ---------------------------------------------------------------------------
# As regras, escritas antes de qualquer resultado
# ---------------------------------------------------------------------------

REGRA_DA_ESCADA = {
    "regra": (
        "do mais barato para o mais caro; um candidato mais caro só toma o lugar se o exact match em dev "
        "for distinguivelmente maior (IC95 da diferença pareada acima de zero)"
    ),
    "origem": "a mesma frase de entregas/05b_professor_aberto/escolha.json; a escada continua de onde a 5b parou",
    "titular_de_partida": VERSAO_DA_5B,
    "ordem": ["qwen3_8b__nf4", "qwen3_8b sem quantização (fp16, ou bf16 pela regra de precisão)", "qwen3_14b sem quantização"],
    "medida": "exact match em dev (2.033 pedidos), diferença pareada, bootstrap por pedido, 2.000 reamostras, semente 0",
    "relato": "a decisão sai de dev; o teste só é relatado",
    "o_que_roda": "todo candidato que rodar rotula dev, teste e treino inteiros, ganhe ou perca: todas as versões ficam guardadas",
}

REGRA_DE_PRECISAO = {
    "regra": (
        "sem quantização, o cálculo é em float16, o mesmo tipo de cálculo da versão em 4 bits. Só se o float16 "
        "produzir NaN ou infinito nas 60 pontuações de intenção a versão é refeita inteira em bfloat16 (outra versão, outro nome)"
    ),
    "como_se_confere": (
        "antes de rotular, 10 lotes de dev espalhados pela ordem de execução (60 pedidos) passam pelo modelo e só se olha "
        "se as pontuações são números finitos; durante a rotulagem, todo lote é conferido do mesmo jeito e um não finito para o trabalho"
    ),
    "o_que_nao_se_olha": "nenhuma métrica de dev entra nessa decisão",
    "lotes": 10,
}

# A conferência dos atalhos da Etapa 5b (escolha.json) mediu quanto o resultado muda quando só
# muda a companhia do pedido no lote (lote contra um a um, 32 pedidos): mesma intenção em 32 de 32,
# mesma resposta lida em 31 de 32, maior diferença de probabilidade 0,0985. É o ruído numérico que
# o projeto já aceitou. A troca de ambiente (outra placa, outro driver) é ruído da mesma natureza:
# a mesma conta em ponto flutuante feita em outra ordem. O limiar pede que ela não seja pior do
# que o dobro do que os atalhos mostraram.
LIMIAR_DE_AMBIENTE = {
    "pedidos": 300,
    "lotes": 50,
    "ancora": "conferencia_dos_atalhos.em_lote_contra_um_a_um de entregas/05b_professor_aberto/escolha.json",
    "mesmos_tokens_de_entrada_min": 1.0,          # o prompt é texto: tem de ser idêntico em todos os pedidos
    "mesma_intencao_min": 0.97,                   # nos atalhos, 32 de 32; em 300 pedidos há empates apertados
    "mesma_resposta_lida_min": 1 - 2 * (1 / 32),  # nos atalhos, 1 em 32 mudou; aceita-se até o dobro (0,9375)
    "p95_da_diferenca_de_probabilidade_max": 0.0985,   # 95% dos pedidos abaixo da MAIOR diferença vista nos atalhos
    "maior_diferenca_de_probabilidade": "relatada, sem limiar: é o máximo de 300 pedidos, e nos atalhos era o de 32",
    "se_reprovar": "o trabalho longo não é lançado; relata-se a diferença",
    "desenho": (
        "a configuração da 5b (4 bits) roda no Colab em 50 lotes inteiros da rodada local de dev, espalhados pela ordem "
        "de execução e com os mesmos pedidos na mesma ordem; assim a companhia no lote é a mesma e a única coisa que muda é o ambiente"
    ),
}


# A extensão: um quarto degrau, decidido DEPOIS de ver o resultado dos três primeiros. Fica num arquivo
# próprio (`regra_extensao_32b.json`); `regra.json`, das 17h16, não muda.
EXTENSAO = "qwen3_32b__fp16__colab_g4"
TITULAR_DA_EXTENSAO = "qwen3_14b__fp16__colab_a100"
REGRA_DA_EXTENSAO = {
    "o_que_e": "extensão da escada da Etapa 11a com mais um degrau: o Qwen3-32B, sem os 4 bits, desafia o titular atual",
    "o_que_ja_se_sabia": (
        "esta extensão NÃO estava nas regras das 17h16. Foi decidida depois de ver o resultado da escada de três degraus "
        "(em dev, o Qwen3-14B tomou o lugar do Qwen3-8B; no teste, ele ficou abaixo do professor fechado). Um degrau acrescentado "
        "depois de ver o resultado é uma segunda olhada nos mesmos dados: a regra não muda por causa disso, e o resultado será "
        "relatado como extensão, não como parte da escada planejada"
    ),
    "autorizacao": (
        "do usuário, em 18/11/2026 perto das 22h, com custo previsto de cerca de 40 unidades. Ela chegou a esta rodada pela "
        "mensagem do orquestrador; o agente que preparou e lançou a rodada não falou com o usuário"
    ),
    "regra": REGRA_DA_ESCADA["regra"],
    "titular": TITULAR_DA_EXTENSAO, "desafiante": EXTENSAO,
    "medida": "exact match em dev (2.033 pedidos), diferença pareada desafiante − titular, bootstrap por pedido, 2.000 reamostras, semente 0 (`qualidade.comparar`)",
    "relato": "a decisão sai de dev; o teste só é relatado",
    "o_que_roda": "o desafiante rotula dev, teste e treino inteiros, ganhe ou perca, e a versão fica guardada",
    "precisao": REGRA_DE_PRECISAO["regra"] + ". " + REGRA_DE_PRECISAO["como_se_confere"],
    "igual_as_outras_versoes": "mesmo prompt, mesmas demonstrações, mesma pontuação e geração, lote 6",
    "modelo": {k: MODELOS["qwen3_32b"][k] for k in ("repositorio", "commit")},
    "placa": "G4 do Colab (RTX PRO 6000 Blackwell, 96 GB), 10,05 unidades por hora. Se não alocar, a rodada não acontece: não se troca de placa nem de precisão",
    "travas_de_custo": {
        "teto_de_unidades_do_condutor": 60, "piso_de_saldo": 500, "teto_de_horas": 7,
        "projecao_depois_de_dev": "quando dev terminar (12% dos pedidos), se o custo total projetado passar de 60 unidades, a sessão é desligada e o motivo fica registrado",
        "desligamento": "pelo condutor, no fim ou na falha; pela própria VM, se o condutor sumir",
    },
    "o_que_fica_como_esta": "`regra.json`, `fecho/escolha.json` e a entrega `entregas/11a_professor_aberto_maior/` não são alterados por esta rodada; a extensão entra depois, como versão nova da escolha",
}


def gravar_regra_da_extensao() -> Path:
    """Grava a regra da extensão com a hora e com o que já se sabia quando ela foi escrita. Se já existe, não regrava."""
    destino = PASTA / "regra_extensao_32b.json"
    if not destino.exists():
        escolha = PASTA / "fecho" / "escolha.json"
        visto = json.loads(escolha.read_text(encoding="utf-8")) if escolha.exists() else {}
        destino.write_text(json.dumps({
            "etapa": ETAPA, "escrita_em": time.strftime("%Y-%m-%d %H:%M:%S %z"), **REGRA_DA_EXTENSAO,
            "resultado_ja_visto_ao_escrever": {
                "escolhido_na_escada_de_tres_degraus": visto.get("escolhido"), "disputas": visto.get("disputas"),
                "exact_match": {v: {c: m["exact_match"] for c, m in x.items() if c in ("dev", "test")} for v, x in visto.get("candidatos", {}).items()},
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def gravar_regra() -> Path:
    """Grava as regras com a hora. Se já existem, não regrava: a data vale como prova de que vieram antes."""
    destino = PASTA / "regra.json"
    if not destino.exists():
        PASTA.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps({
            "etapa": ETAPA, "escrita_em": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "escada": REGRA_DA_ESCADA, "precisao": REGRA_DE_PRECISAO, "ambiente": LIMIAR_DE_AMBIENTE,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# Registro: o que identifica uma versão
# ---------------------------------------------------------------------------

def sha256_do_arquivo(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def sha256_do_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def impressao_do_prompt() -> dict:
    """O que o prompt tem de fixo: as instruções e o conjunto de demonstração. Não precisa do modelo."""
    particoes, rotulos = massive.carregar_entrega()
    demonstracoes = Demonstracoes(montar_conjunto(particoes["train"]))
    return {
        "sha256_das_instrucoes": sha256_do_texto(prompt.instrucoes(rotulos["intents"], rotulos["slot_types"])),
        "sha256_das_demonstracoes": sha256_do_texto(json.dumps(demonstracoes._exemplos, ensure_ascii=False, sort_keys=True)),
        "demonstracoes": len(demonstracoes), "por_intencao": POR_INTENCAO, "semente_das_demonstracoes": 0, "k": K,
        "escolha_das_demonstracoes": "as k mais parecidas por TF-IDF de n-gramas de caracteres",
        "sha256_dos_dados": {
            nome: sha256_do_arquivo(massive.entrega(massive.ETAPA) / nome)
            for nome in ("train.parquet", "dev.parquet", "test.parquet", "rotulos.json")
        },
    }


def constantes_de(arquivo: Path, nomes: tuple[str, ...]) -> dict:
    """Lê constantes de um arquivo de código sem importá-lo (`local.py` puxa o torch)."""
    achadas = {}
    for no in ast.parse(arquivo.read_text(encoding="utf-8")).body:
        alvo = no.targets[0] if isinstance(no, ast.Assign) else no.target if isinstance(no, ast.AnnAssign) else None
        if isinstance(alvo, ast.Name) and alvo.id in nomes and no.value is not None:
            achadas[alvo.id] = ast.literal_eval(no.value)
    return achadas


def geracao() -> dict:
    """Os parâmetros de geração, lidos do código que gera."""
    daqui = Path(__file__).resolve().parents[1] / "professores" / "local.py"
    c = constantes_de(daqui, ("MAX_TOKENS", "INICIO", "DEPOIS"))
    return {
        "lote": LOTE, "ordem": "prompts mais longos primeiro (ordenação estável), lotes consecutivos",
        "intencao": "a mais provável das 60, pela soma dos logaritmos dos tokens do nome",
        "entidades": "decodificação gulosa (o token mais provável), sem amostragem",
        "max_tokens": c["MAX_TOKENS"], "inicio_forcado": c["INICIO"], "depois_da_intencao": c["DEPOIS"],
        "enable_thinking": False,
    }


def ler_registros(caminho: Path) -> dict[str, dict]:
    """Os registros de um arquivo, por identificador. Uma última linha cortada (download no meio da escrita) é ignorada."""
    registros = {}
    if caminho.exists():
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            try:
                registro = json.loads(linha)
            except ValueError:
                continue
            registros[registro["id"]] = registro
    return registros


def retomar(destino: Path, total: int, lote: int = LOTE) -> set[str]:
    """Deixa no arquivo só lotes inteiros e devolve os identificadores já gravados.

    Uma execução interrompida pode deixar uma linha cortada ou um lote pela
    metade. Tirando essas sobras, a continuação monta exatamente os lotes
    que uma execução sem interrupção montaria. Um arquivo completo (`total`
    registros) fica como está: o último lote pode ser menor.
    """
    if not destino.exists():
        return set()
    inteiras = []
    for linha in destino.read_text(encoding="utf-8").splitlines():
        try:
            json.loads(linha)
        except ValueError:
            break
        inteiras.append(linha)
    if len(inteiras) != total:
        inteiras = inteiras[: len(inteiras) - len(inteiras) % lote]
    conteudo = "".join(linha + "\n" for linha in inteiras)
    if conteudo != destino.read_text(encoding="utf-8"):
        provisorio = destino.with_suffix(".tmp")
        provisorio.write_text(conteudo, encoding="utf-8")
        provisorio.replace(destino)
    return {json.loads(linha)["id"] for linha in inteiras}


# ---------------------------------------------------------------------------
# A conferência de ambiente (parte leve: escolher os lotes e comparar)
# ---------------------------------------------------------------------------

def respostas_da_5b(conjunto: str) -> Path:
    return EXECUCOES / ETAPA_5B / "qwen3_8b" / conjunto / "respostas.jsonl"


def lotes_da_conferencia() -> list[list[str]]:
    """Cinquenta lotes inteiros da rodada local de dev, espalhados pela ordem em que rodaram.

    Os lotes da rodada local se reconhecem no arquivo: `ms` é o tempo do lote
    dividido pelos pedidos, igual para todos os pedidos de um mesmo lote.
    """
    lotes: list[tuple[float, list[str]]] = []
    with respostas_da_5b("dev").open(encoding="utf-8") as arquivo:
        for registro in map(json.loads, arquivo):
            if lotes and lotes[-1][0] == registro["ms"]:
                lotes[-1][1].append(registro["id"])
            else:
                lotes.append((registro["ms"], [registro["id"]]))
    inteiros = [ids for _, ids in lotes if len(ids) == LOTE]
    escolhidos = np.linspace(0, len(inteiros) - 1, LIMIAR_DE_AMBIENTE["lotes"]).round().astype(int)
    return [inteiros[i] for i in escolhidos]


def comparar_ambiente(versao: str) -> dict:
    """Compara as respostas da conferência (Colab, 4 bits) com as da rodada local, pedido a pedido, e aplica o limiar."""
    particoes, rotulos = massive.carregar_entrega()
    pedido = {f"dev-{linha.id}": linha.utt for linha in particoes["dev"].itertuples()}
    locais = ler_registros(respostas_da_5b("dev"))
    remotos = ler_registros(PASTA / versao / "conferencia" / "respostas.jsonl")
    ids = list(remotos)
    lida = lambda registro, i: resposta.ler(registro["texto"], pedido[i], rotulos["intents"], rotulos["slot_types"])[0]
    logp_l = np.array([locais[i]["logp"] for i in ids])
    logp_r = np.array([remotos[i]["logp"] for i in ids])
    diferenca = np.abs(np.exp(logp_l) - np.exp(logp_r)).max(axis=1)      # a mesma medida da conferência dos atalhos
    n = len(ids)
    medido = {
        "pedidos": n,
        "mesmos_tokens_de_entrada": sum(locais[i]["uso"]["entrada"] == remotos[i]["uso"]["entrada"] for i in ids) / n,
        "mesma_intencao": float((logp_l.argmax(1) == logp_r.argmax(1)).mean()),
        "mesmo_texto": sum(locais[i]["texto"] == remotos[i]["texto"] for i in ids) / n,
        "mesma_resposta_lida": sum(lida(locais[i], i) == lida(remotos[i], i) for i in ids) / n,
        "p50_da_diferenca_de_probabilidade": float(np.percentile(diferenca, 50)),
        "p95_da_diferenca_de_probabilidade": float(np.percentile(diferenca, 95)),
        "maior_diferenca_de_probabilidade": float(diferenca.max()),
    }
    L = LIMIAR_DE_AMBIENTE
    criterios = {
        "pedidos": n == L["pedidos"],
        "mesmos_tokens_de_entrada": medido["mesmos_tokens_de_entrada"] >= L["mesmos_tokens_de_entrada_min"],
        "mesma_intencao": medido["mesma_intencao"] >= L["mesma_intencao_min"],
        "mesma_resposta_lida": medido["mesma_resposta_lida"] >= L["mesma_resposta_lida_min"],
        "p95_da_diferenca_de_probabilidade": medido["p95_da_diferenca_de_probabilidade"] <= L["p95_da_diferenca_de_probabilidade_max"],
    }
    resultado = {
        "versao": versao, "contra": VERSAO_DA_5B, "medido": medido, "limiar": L, "criterios": criterios,
        "aprovado": all(criterios.values()), "quando": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (PASTA / versao / "conferencia_de_ambiente.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


# ---------------------------------------------------------------------------
# A versão que já existia (Etapa 5b): só o registro, sem tocar nas saídas
# ---------------------------------------------------------------------------

ARQUIVOS_DA_5B = (
    "src/especialista/professores/local.py", "src/especialista/professores/prompt.py",
    "src/especialista/professores/demonstracoes.py", "src/especialista/professores/resposta.py",
    "src/especialista/professores/rotulagem.py", "src/especialista/experimentos/etapa5b.py",
    "src/especialista/dados/massive.py", "src/especialista/bio.py", "src/especialista/avaliacao/previsao.py",
)


def versao_retroativa() -> Path:
    """O `versao.json` da versão da Etapa 5b, montado depois do fato a partir do que ficou em disco."""
    from importlib import metadata

    v = descrever(VERSAO_DA_5B)
    cache, entrega = EXECUCOES / ETAPA_5B / "qwen3_8b", RAIZ / "entregas" / ETAPA_5B
    metas = {c: json.loads((cache / c / "meta.json").read_text(encoding="utf-8")) for c in CONJUNTOS}
    fim_da_rodada = max(os.path.getmtime(cache / c / "respostas.jsonl") for c in CONJUNTOS)
    arquivos = {}
    for nome in ARQUIVOS_DA_5B:
        alterado = os.path.getmtime(RAIZ / nome)
        arquivos[nome] = {
            "sha256": sha256_do_arquivo(RAIZ / nome),
            "alterado_em": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alterado)),
            # sem git, a prova de que o arquivo é o da rodada é a data: ele não mudou depois que ela terminou
            "igual_ao_da_rodada": bool(alterado <= fim_da_rodada),
        }
    torch_py = constantes_de(Path(metadata.distribution("torch").locate_file("torch/version.py")), ("__version__", "git_version", "cuda"))
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        driver = None
    saidas = {}
    for c in CONJUNTOS:
        arquivo = cache / c / "respostas.jsonl"
        saidas[str(arquivo.relative_to(RAIZ))] = {"sha256": sha256_do_arquivo(arquivo), "pedidos": len(ler_registros(arquivo))}
    for arquivo in sorted(entrega.glob("logits_*.npy")) + [entrega / "rotulos_treino.parquet", entrega / "escolha.json"]:
        saidas[str(arquivo.relative_to(RAIZ))] = {"sha256": sha256_do_arquivo(arquivo)}
    registro = {
        "id": VERSAO_DA_5B, "etapa": ETAPA, "papel": "titular de partida: o professor aberto escolhido na Etapa 5b",
        "retroativo": "montado em %s a partir do que a Etapa 5b deixou em disco; as saídas não foram tocadas" % time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "modelo": {"repositorio": v["repositorio"], "nome": v["nome"], "commit": v["commit"],
                   "de_onde_vem_o_commit": "único snapshot no cache local do Hugging Face"},
        "precisao": {"id": v["precisao"], "descricao": v["descricao"], "tipo_de_calculo": v["tipo_de_calculo"]},
        "prompt": impressao_do_prompt(),
        "geracao": geracao(),
        "codigo": {"pacote": None, "observacao": "rodou direto da pasta do projeto, que não é um repositório git", "arquivos": arquivos},
        "ambiente": {
            "maquina": "local (WSL2)", "gpu": metas["dev"]["gpu"], "vram_total_gb": 10, "driver": driver,
            "python": platform.python_version(), "torch": torch_py["__version__"], "torch_git": torch_py["git_version"], "cuda": torch_py.get("cuda"),
            **{nome: metadata.version(nome) for nome in ("transformers", "bitsandbytes", "accelerate", "tokenizers", "safetensors", "huggingface-hub", "numpy", "pandas", "scikit-learn")},
            "observacao": "versões lidas hoje do ambiente do projeto (uv.lock); a rodada foi na madrugada do mesmo dia",
        },
        "execucao": {
            "fim_de_cada_conjunto": {c: metas[c]["quando"] for c in CONJUNTOS},
            "pedidos": {c: metas[c]["pedidos"] for c in CONJUNTOS},
            "segundos_de_gpu": {c: metas[c]["segundos_de_gpu"] for c in CONJUNTOS},
            "vram_do_modelo_gb": metas["dev"]["vram_do_modelo_gb"], "vram_de_pico_gb": max(m["vram_de_pico_gb"] for m in metas.values()),
            "tokens_das_instrucoes_em_memoria": metas["dev"]["tokens_das_instrucoes_em_memoria"],
            "unidades_de_computacao": 0,
        },
        "onde_estao_as_saidas": {"cache": str(cache.relative_to(RAIZ)), "entrega": str(entrega.relative_to(RAIZ))},
        "saidas": saidas,
    }
    destino = PASTA / VERSAO_DA_5B / "versao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# A parte pesada: roda no Colab
# ---------------------------------------------------------------------------

class PrecisaoInsuficiente(RuntimeError):
    """O cálculo produziu NaN ou infinito: a versão não serve nessa precisão."""


def _commit(v: dict, pasta: Path) -> str:
    """O commit do modelo: o fixado na tabela ou, se não houver, o atual do Hugging Face, gravado para as retomadas."""
    if v["commit"]:
        return v["commit"]
    arquivo = pasta / "commit.txt"
    if not arquivo.exists():
        from huggingface_hub import HfApi

        arquivo.write_text(HfApi().model_info(v["repositorio"]).sha, encoding="utf-8")
    return arquivo.read_text(encoding="utf-8").strip()


def carregar(versao: str, pasta: Path):
    """O professor de uma versão: mesmos pesos, mesmo prompt, mesmo código; só muda como os pesos ficam na placa."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from especialista.professores.local import ProfessorLocal

    v = descrever(versao)
    commit = _commit(v, pasta)
    tokenizador = AutoTokenizer.from_pretrained(v["repositorio"], revision=commit)
    if v["precisao"] == "nf4":
        # exatamente a carga de `ProfessorLocal`, com o commit fixado
        como = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )}
    else:
        como = {"dtype": {"fp16": torch.float16, "bf16": torch.bfloat16}[v["precisao"]]}
    modelo = AutoModelForCausalLM.from_pretrained(v["repositorio"], revision=commit, device_map={"": 0}, **como)
    particoes, rotulos = massive.carregar_entrega()
    demonstracoes = Demonstracoes(montar_conjunto(particoes["train"]))
    prof = ProfessorLocal(v["repositorio"], demonstracoes, rotulos, nome=v["nome"], modelo=modelo, tokenizador=tokenizador)
    return prof, particoes, commit


def ambiente(prof, commit: str) -> dict:
    """A máquina e as bibliotecas em que o trabalho está rodando de verdade."""
    from collections import Counter
    from importlib import metadata

    import torch

    consulta = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.strip()
    memoria = dict(linha.split(":", 1) for linha in Path("/proc/meminfo").read_text().splitlines()[:1])
    prof._estado_inicial(1)     # lê as instruções: `_prefixo` passa a existir
    return {
        "python": platform.python_version(), "sistema": platform.platform(),
        "torch": torch.__version__, "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        **{nome: metadata.version(nome) for nome in ("transformers", "bitsandbytes", "accelerate", "tokenizers", "safetensors", "huggingface-hub", "numpy", "pandas", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(0), "nvidia_smi": consulta,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "cpus": os.cpu_count(), "ram": memoria.get("MemTotal", "").strip(),
        "commit_do_modelo": commit,
        "tipos_dos_parametros": dict(Counter(str(p.dtype) for p in prof.modelo.parameters())),
        "tipo_das_contas_do_professor": str(prof._tipo),
        "vram_do_modelo_gb": round(prof.vram_do_modelo_gb, 2),
        "tokens_das_instrucoes_em_memoria": len(prof._prefixo),
        "sha256_das_instrucoes_no_formato_do_modelo": sha256_do_texto(prof.tokenizador.decode(prof._prefixo)),
        "prompt": impressao_do_prompt(), "geracao": geracao(),
        "quando_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }


def _conferir_finitos(registros: list[dict]) -> None:
    ruins = [r["id"] for r in registros if not all(math.isfinite(valor) for valor in r["logp"])]
    if ruins:
        raise PrecisaoInsuficiente(f"pontuação não finita em {ruins[:6]}")


def _itens(dados, conjunto: str) -> list[tuple[str, str, int | None]]:
    """Os pedidos de um conjunto. No treino, o próprio exemplo nunca entra nas suas demonstrações."""
    excluir = (lambda linha: int(linha.id)) if conjunto == "train" else (lambda linha: None)
    return [(f"{conjunto}-{linha.id}", linha.utt, excluir(linha)) for linha in dados.itertuples()]


def _em_ordem(prof, itens: list) -> list:
    """Os prompts mais longos primeiro, como na Etapa 5b (ordenação estável)."""
    return sorted(itens, key=lambda item: -len(prof.tokens_do_prompt(item[1], item[2])))


def _gravar_lotes(prof, lotes: list[list], destino: Path, rotulo: str) -> float:
    """Passa os lotes pelo professor e grava cada um inteiro, de uma vez. Devolve os segundos gastos."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    inicio, prontos, total = time.perf_counter(), 0, sum(map(len, lotes))
    with destino.open("a", encoding="utf-8") as arquivo:
        for lote in lotes:
            registros = prof.responder(lote)
            _conferir_finitos(registros)
            arquivo.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros))
            arquivo.flush()
            antes, prontos = prontos, prontos + len(lote)
            if prontos // 120 > antes // 120 or prontos == total:
                ritmo = (time.perf_counter() - inicio) / prontos
                print(f"{rotulo}: {prontos} de {total}, {ritmo:.2f} s por pedido, faltam {ritmo * (total - prontos) / 60:.0f} min", flush=True)
    return time.perf_counter() - inicio


def _meta(pasta: Path, versao: str, pedidos: int, segundos: float, extra: dict | None = None) -> None:
    import torch

    arquivo = pasta / "meta.json"
    anterior = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}
    arquivo.write_text(json.dumps({
        "versao": versao, "lote": LOTE, "pedidos": pedidos,
        # soma os trechos de uma execução retomada; não inclui o tempo de carregar o modelo
        "segundos_de_gpu": round(anterior.get("segundos_de_gpu", 0.0) + segundos, 1),
        "trechos": anterior.get("trechos", 0) + 1,
        "vram_de_pico_gb": round(max(anterior.get("vram_de_pico_gb", 0.0), torch.cuda.max_memory_allocated() / 1e9), 2),
        "gpu": torch.cuda.get_device_name(0), "quando_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        **(extra or {}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def rotular(prof, particoes: dict, versao: str, conjunto: str) -> None:
    """Rotula um conjunto inteiro. Retomável: continua do último lote inteiro gravado."""
    pasta = PASTA / versao / conjunto
    destino = pasta / "respostas.jsonl"
    todos = _itens(particoes[conjunto], conjunto)
    feitos = retomar(destino, len(todos))
    if len(feitos) == len(todos):
        print(f"{versao} {conjunto}: já gravado ({len(feitos)} respostas)", flush=True)
        return
    # a ordem é a do conjunto inteiro; como só há lotes inteiros gravados, o que falta recomeça numa fronteira de lote
    faltam = [item for item in _em_ordem(prof, todos) if item[0] not in feitos]
    lotes = [faltam[i : i + LOTE] for i in range(0, len(faltam), LOTE)]
    segundos = _gravar_lotes(prof, lotes, destino, f"{versao} {conjunto}")
    _meta(pasta, versao, len(feitos) + len(faltam), segundos)
    print(f"{versao} {conjunto}: {len(faltam)} respostas em {segundos / 60:.1f} min", flush=True)


def conferir_ambiente(prof, particoes: dict, versao: str) -> None:
    """Roda os lotes da conferência de ambiente (lidos de `lotes_da_conferencia.json`, que veio da máquina local)."""
    pasta = PASTA / versao / "conferencia"
    destino = pasta / "respostas.jsonl"
    lotes_de_ids = json.loads((PASTA / "lotes_da_conferencia.json").read_text(encoding="utf-8"))
    feitos = retomar(destino, sum(map(len, lotes_de_ids)))
    por_id = {item[0]: item for item in _itens(particoes["dev"], "dev")}
    lotes = [[por_id[i] for i in ids] for ids in lotes_de_ids if ids[0] not in feitos]
    if not lotes:
        print(f"{versao} conferência: já gravada ({len(feitos)} respostas)", flush=True)
        return
    segundos = _gravar_lotes(prof, lotes, destino, f"{versao} conferência")
    _meta(pasta, versao, len(feitos) + sum(map(len, lotes)), segundos)


def conferir_precisao(prof, particoes: dict, versao: str) -> None:
    """A regra de precisão: as pontuações de 10 lotes de dev são números finitos? Não grava resposta nem olha acerto."""
    import torch

    destino = PASTA / versao / "precisao.json"
    if destino.exists() and json.loads(destino.read_text(encoding="utf-8"))["finito"]:
        print(f"{versao} precisão: já conferida", flush=True)
        return
    ordem = _em_ordem(prof, _itens(particoes["dev"], "dev"))
    inteiros = [ordem[i : i + LOTE] for i in range(0, len(ordem) - LOTE + 1, LOTE)]
    escolhidos = np.linspace(0, len(inteiros) - 1, REGRA_DE_PRECISAO["lotes"]).round().astype(int)
    nao_finitos, pedidos = 0, 0
    for i in escolhidos:
        pontos, _, _ = prof._pontuar_e_gerar([prof.tokens_do_prompt(texto, excluir) for _, texto, excluir in inteiros[i]])
        nao_finitos += int((~torch.isfinite(pontos)).sum())
        pedidos += len(inteiros[i])
    resultado = {"versao": versao, "tipo_das_contas": str(prof._tipo), "lotes": len(escolhidos), "pedidos": pedidos,
                 "pontuacoes_nao_finitas": nao_finitos, "finito": nao_finitos == 0,
                 "quando_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{versao} precisão: {json.dumps(resultado, ensure_ascii=False)}", flush=True)
    if nao_finitos:
        raise PrecisaoInsuficiente(f"{nao_finitos} pontuações não finitas em {pedidos} pedidos")


def trabalho(versao: str, tarefas: list[str]) -> int:
    """O que roda no Colab: carrega o professor uma vez e faz as tarefas em ordem. Código de saída 3 = precisão insuficiente."""
    import torch

    pasta = PASTA / versao
    pasta.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    print(f"{versao}: carregando {descrever(versao)['repositorio']} ({descrever(versao)['descricao']})", flush=True)
    prof, particoes, commit = carregar(versao, pasta)
    onde = ambiente(prof, commit)
    (pasta / "ambiente.json").write_text(json.dumps(onde, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{versao}: pronto em {time.perf_counter() - inicio:.0f} s; {onde['gpu']}, torch {onde['torch']}, transformers {onde['transformers']}, "
          f"modelo com {onde['vram_do_modelo_gb']} GB, tipos {onde['tipos_dos_parametros']}", flush=True)
    try:
        for tarefa in tarefas:
            torch.cuda.reset_peak_memory_stats()
            if tarefa == "conferencia":
                conferir_ambiente(prof, particoes, versao)
            elif tarefa == "precisao":
                conferir_precisao(prof, particoes, versao)
            else:
                rotular(prof, particoes, versao, tarefa)
    except PrecisaoInsuficiente as erro:
        print(f"{versao}: PRECISAO INSUFICIENTE: {erro}", flush=True)
        return 3
    print(f"{versao}: tarefas concluídas: {','.join(tarefas)}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# A escada (fase 2: só depois de as versões terem dev inteiro)
# ---------------------------------------------------------------------------

def previsoes(versao: str, conjunto: str) -> tuple[list, dict, list[dict]]:
    """As respostas gravadas de uma versão, lidas na ordem dos dados: previsões, diagnóstico e registros."""
    from especialista.professores.rotulagem import ler_conjunto

    particoes, rotulos = massive.carregar_entrega()
    arquivo = respostas_da_5b(conjunto) if versao == VERSAO_DA_5B else PASTA / versao / conjunto / "respostas.jsonl"
    return ler_conjunto(ler_registros(arquivo), particoes[conjunto], conjunto, rotulos["intents"], rotulos["slot_types"])


def escolher(desafiantes: list[str], titular: str = VERSAO_DA_5B) -> tuple[str, dict]:
    """Aplica `REGRA_DA_ESCADA`: `desafiantes` vem do mais barato para o mais caro, e só entra quem tem dev inteiro.

    `titular` é de onde a escada parte: a versão da Etapa 5b na escada planejada; o escolhido dela, na extensão.
    """
    from especialista.avaliacao import qualidade
    from especialista.avaliacao.previsao import ouro

    particoes, _ = massive.carregar_entrega()
    gabarito = ouro(particoes["dev"])
    em_dev = {versao: previsoes(versao, "dev")[0] for versao in [titular, *desafiantes]}
    disputas = {}
    for desafiante in desafiantes:
        comparacao = qualidade.comparar(gabarito, em_dev[titular], em_dev[desafiante]).loc["exact_match"]
        disputas[desafiante] = {"contra": titular, **{k: float(comparacao[k]) for k in ("a", "b", "diferenca", "ic_min", "ic_max")}}
        if comparacao["ic_min"] > 0:
            titular = desafiante
    return titular, disputas


# ---------------------------------------------------------------------------
# O fecho (fase 2): saídas no formato da Etapa 5b, a escada, as tabelas, o custo e a entrega
# ---------------------------------------------------------------------------

CANDIDATOS = ("qwen3_8b__fp16__colab_l4", "qwen3_14b__fp16__colab_a100")     # os degraus novos, do mais barato para o mais caro
CONFERENCIA = "qwen3_8b__nf4__colab_l4"
ESCADA = (VERSAO_DA_5B, *CANDIDATOS)
FECHADO = "claude"                                                           # a referência: o professor fechado da Etapa 5a
NOMES = {
    VERSAO_DA_5B: "Qwen3-8B em 4 bits (RTX 3080 local, Etapa 5b)",
    "qwen3_8b__fp16__colab_l4": "Qwen3-8B sem os 4 bits, em float16 (Colab, L4)",
    "qwen3_14b__fp16__colab_a100": "Qwen3-14B sem os 4 bits, em float16 (Colab, A100)",
    CONFERENCIA: "Qwen3-8B em 4 bits (Colab, L4): conferência de ambiente",
    FECHADO: "Claude Haiku 5.5 (professor fechado, Etapa 5a)",
}
CURTOS = {VERSAO_DA_5B: "8B, 4 bits", "qwen3_8b__fp16__colab_l4": "8B, float16", "qwen3_14b__fp16__colab_a100": "14B, float16", FECHADO: "Claude Haiku 5.5"}
NOMES[EXTENSAO], CURTOS[EXTENSAO] = "Qwen3-32B sem os 4 bits, em float16 (Colab, G4)", "32B, float16"
ABERTOS = (*ESCADA, EXTENSAO)                                                # todos os professores abertos, com o da extensão
PARES = (   # as comparações pareadas B − A da tabela (as duas primeiras linhas de cada desafiante são as da escada)
    ("qwen3_8b__fp16__colab_l4", VERSAO_DA_5B), ("qwen3_14b__fp16__colab_a100", VERSAO_DA_5B),
    ("qwen3_14b__fp16__colab_a100", "qwen3_8b__fp16__colab_l4"),
    (VERSAO_DA_5B, FECHADO), ("qwen3_8b__fp16__colab_l4", FECHADO), ("qwen3_14b__fp16__colab_a100", FECHADO),
)
PARES_DA_EXTENSAO = ((EXTENSAO, TITULAR_DA_EXTENSAO), (EXTENSAO, VERSAO_DA_5B), (EXTENSAO, FECHADO))
CONDUTOR_QUE_RODOU = "pacotes/etapa11a_colab_506bad4134c4.py"     # o das três rodadas de 18/11; o da extensão está no versao.json dela
CONTA_DA_EXTENSAO = {"saldo_antes": 577.17, "saldo_depois": 553.02,
                     "de_onde": "`colab usage`: 577,17 antes da rodada do 32B; 553,02 lido pelo orquestrador depois do fim, com taxa 0 e nenhuma sessão ativa"}
VERSAO_ENTREGUE_EM_1811 = "2026-11-18_escada_de_tres_degraus"
TAXAS_POR_HORA = {"L4": 1.94, "A100": 6.73, "G4": 10.05}     # unidades por hora, medidas com `colab usage` em 18/11
CONTA = {"saldo_antes": 600.00, "saldo_depois": 577.17,
         "de_onde": "`colab usage`: 600,00 antes da etapa; 577,17 lido pelo orquestrador depois do fim das duas rodadas, com taxa 0 e nenhuma sessão ativa"}
# Sessões abertas à mão na fase 1, fora dos condutores (horários tirados da saída dos comandos)
SONDAS = (
    {"o_que": "sonda da L4 (máquina, kernel que segue sem a ligação local, desligamento pela própria VM)", "gpu": "L4", "de": "17:08:43", "ate": "17:10:35"},
    {"o_que": "sonda da G4 (placa e taxa)", "gpu": "G4", "de": "17:23:40", "ate": "17:24:28"},
)
A100_CRIADA = "17:19:01"      # a sessão da sonda da A100 foi reaproveitada pelo candidato 2


def pasta_de_consumo(versao: str) -> tuple[Path, str]:
    """Onde os alunos leem um professor (rótulos, logits, previsões) e o prefixo dos arquivos de `previsoes/`."""
    if versao == VERSAO_DA_5B:
        return ENTREGAS / ETAPA_5B, "qwen3_8b"
    return PASTA / versao, versao


def como_professor(versao: str) -> dict:
    """A linha que troca o professor nas Etapas 6 e 10a: `etapa6.PROFESSORES["aberto"] = etapa11a.como_professor(versao)`."""
    if versao == VERSAO_DA_5B:
        return {"nome": MODELOS["qwen3_8b"]["nome"], "etapa": ETAPA_5B, "previsoes": "qwen3_8b"}
    return {"nome": descrever(versao)["nome"], "etapa": f"{ETAPA}/professores/{versao}", "previsoes": versao}


def derivar(versao: str) -> dict[str, Path]:
    """Das respostas gravadas para os arquivos que os alunos leem, no formato da Etapa 5b, dentro da pasta da versão.

    `logits_{treino,dev,test}.npy` (float32, uma linha por pedido na ordem da
    partição, uma coluna por intenção na ordem de `intencoes.json`),
    `rotulos_treino.parquet` e `previsoes/<versão>_{dev,test}.parquet`. Um
    arquivo que já existe não é regravado.
    """
    from especialista.avaliacao.previsao import salvar_previsoes
    from especialista.professores.rotulagem import rotular as rotular_conjunto

    particoes, rotulos = massive.carregar_entrega()
    pasta = PASTA / versao
    lidos = {c: previsoes(versao, c) for c in CONJUNTOS}
    if any(lidos[c][1]["sem_registro"] for c in CONJUNTOS):
        raise ValueError(f"{versao}: há pedidos sem resposta gravada.")
    feitos: dict[str, Path] = {}

    def gravar(nome: str, escreve) -> None:
        destino = pasta / nome
        if not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            escreve(destino)
        feitos[nome] = destino

    for conjunto, nome in (("train", "treino"), ("dev", "dev"), ("test", "test")):
        logp = np.array([r["logp"] for r in lidos[conjunto][2]], dtype=np.float32)
        gravar(f"logits_{nome}.npy", lambda destino, logp=logp: np.save(destino, logp))
    gravar("intencoes.json", lambda destino: destino.write_text(json.dumps(rotulos["intents"], indent=1) + "\n", encoding="utf-8"))
    gravar("rotulos_treino.parquet", lambda destino: rotular_conjunto(particoes["train"], lidos["train"][0], rotulos["intents"]).to_parquet(destino, index=False))
    for conjunto in ("dev", "test"):
        gravar(f"previsoes/{versao}_{conjunto}.parquet", lambda destino, c=conjunto: salvar_previsoes(destino, particoes[c]["id"], lidos[c][0]))
    return feitos


def _hora(texto: str) -> int:
    h, m, s = map(int, texto.split(":"))
    return 3600 * h + 60 * m + s


def _marca(versao: str, trecho: str = "", ultima: bool = False) -> str:
    """A hora (HH:MM:SS) da primeira linha (ou da última) do registro do condutor que contém `trecho`."""
    linhas = [linha for linha in (PASTA / versao / "condutor.log").read_text(encoding="utf-8").splitlines()
              if trecho in linha and len(linha) > 8 and linha[2] == ":" and linha[5] == ":"]
    if not linhas:
        raise LookupError(f"{versao}: não há linha com {trecho!r} no registro do condutor.")
    return linhas[-1 if ultima else 0][:8]


def _horas_entre(de: str, ate: str) -> float:
    """Horas de `de` até `ate` (HH:MM:SS); se `ate` é menor, a rodada virou o dia."""
    return ((_hora(ate) - _hora(de)) % 86400) / 3600


def custo() -> dict:
    """As unidades de computação, repartidas por taxa × tempo de cada sessão, e o total da conta.

    Os condutores mediram o gasto pela diferença de saldo, mas o saldo é da
    conta inteira e as duas sessões rodaram ao mesmo tempo: os dois números
    se sobrepõem. Aqui cada trecho vale o tempo de sessão ligada (dos
    registros dos condutores) vezes a taxa da placa.
    """
    c1, c2 = CANDIDATOS
    partes = [
        {"o_que": "conferência de ambiente (4 bits na L4), do começo da sessão até o candidato 1 assumi-la", "versao": CONFERENCIA, "gpu": "L4",
         "de": _marca(CONFERENCIA, "sessão prof11a_l4"), "ate": _marca(c1)},
        {"o_que": "candidato 1: precisão, dev, teste e treino", "versao": c1, "gpu": "L4", "de": _marca(c1), "ate": _marca(c1, "parando a sessão")},
        {"o_que": "candidato 2: precisão, dev, teste e treino", "versao": c2, "gpu": "A100", "de": _marca(c2), "ate": _marca(c2, "parando a sessão")},
        {"o_que": "sonda da A100, até o condutor do candidato 2 assumir a sessão", "versao": None, "gpu": "A100", "de": A100_CRIADA, "ate": _marca(c2)},
        *({**sonda, "versao": None} for sonda in SONDAS),
    ]
    for parte in partes:
        parte["horas"] = round(_horas_entre(parte["de"], parte["ate"]), 4)
        parte["taxa_por_hora"] = TAXAS_POR_HORA[parte["gpu"]]
        parte["unidades"] = round(parte["horas"] * parte["taxa_por_hora"], 2)
    soma = round(sum(parte["unidades"] for parte in partes), 2)
    gasto = round(CONTA["saldo_antes"] - CONTA["saldo_depois"], 2)
    for parte in partes:      # a mesma repartição, na escala do que a conta de fato cobrou
        parte["unidades_fechadas_na_conta"] = round(parte["unidades"] * gasto / soma, 2)
    medido = {}
    for versao in (CONFERENCIA, *CANDIDATOS):
        conducoes = json.loads((PASTA / versao / "versao.json").read_text(encoding="utf-8"))["conducoes"]
        medido[versao] = conducoes[-1]["unidades"]["gastas_pelo_saldo"]
    extensao = None
    if (PASTA / EXTENSAO / "FIM_REMOTO.json").exists():
        da_extensao = json.loads((PASTA / EXTENSAO / "versao.json").read_text(encoding="utf-8"))["conducoes"]
        parte = {"o_que": "extensão: Qwen3-32B, da sessão pronta até ser parada (precisão, dev, teste e treino)", "versao": EXTENSAO, "gpu": "G4",
                 "de": _marca(EXTENSAO, "Session READY"), "ate": _marca(EXTENSAO, "parando a sessão", ultima=True)}
        parte["horas"] = round(_horas_entre(parte["de"], parte["ate"]), 4)
        parte["taxa_por_hora"] = TAXAS_POR_HORA[parte["gpu"]]
        parte["unidades"] = round(parte["horas"] * parte["taxa_por_hora"], 2)
        gasto_da_extensao = round(CONTA_DA_EXTENSAO["saldo_antes"] - CONTA_DA_EXTENSAO["saldo_depois"], 2)
        parte["unidades_fechadas_na_conta"] = gasto_da_extensao          # só houve esta sessão entre os dois saldos
        extensao = {
            "partes": [parte], "soma_por_taxa_e_tempo": parte["unidades"], "conta": {**CONTA_DA_EXTENSAO, "gasto": gasto_da_extensao},
            "diferenca_entre_a_soma_e_a_conta": round(parte["unidades"] - gasto_da_extensao, 2),
            "medido_pelo_condutor_pelo_saldo": da_extensao[-1]["unidades"]["gastas_pelo_saldo"],
            "projecao_depois_de_dev": da_extensao[-1].get("projecao_depois_de_dev"),
            "alocacao": "a G4 respondeu 'Service Unavailable' três vezes antes de alocar; tentativa sem sessão não custa",
            "previsto_na_autorizacao": "cerca de 40 unidades", "teto_do_condutor": 60,
        }
    return {
        "extensao_32b": extensao,
        "total_no_colab": {"saldo_antes": CONTA["saldo_antes"], "saldo_depois": (CONTA_DA_EXTENSAO if extensao else CONTA)["saldo_depois"],
                           "gasto": round(CONTA["saldo_antes"] - (CONTA_DA_EXTENSAO if extensao else CONTA)["saldo_depois"], 2)},
        "como": ("`unidades` = tempo de sessão ligada (horários dos registros dos condutores, de 18/11) vezes a taxa da placa; "
                 "`unidades_fechadas_na_conta` = a mesma repartição multiplicada por gasto da conta ÷ soma, para as partes somarem o que a conta cobrou"),
        "taxas_por_hora": TAXAS_POR_HORA, "partes": partes, "soma_por_taxa_e_tempo": soma,
        "conta": {**CONTA, "gasto": gasto}, "diferenca_entre_a_soma_e_a_conta": round(soma - gasto, 2),
        "por_que_diferem": "a taxa tem duas casas, o saldo é cobrado em passos e os horários das sondas são de relógio de comando; a conta é o número que vale",
        "o_que_os_condutores_mediram_pelo_saldo": medido,
        "por_que_nao_servem": "o saldo é da conta inteira; com duas sessões ao mesmo tempo, cada condutor contou também o gasto da outra",
        "teto_da_etapa": 150,
    }


def parametros_do_modelo(versao: str, commit: str, vram_do_modelo_gb: float) -> dict:
    """Quantos parâmetros o modelo tem: do cabeçalho dos pesos no Hugging Face; se não der, estimado pela memória (2 bytes por peso)."""
    try:
        from huggingface_hub import HfApi

        contagem = HfApi().get_safetensors_metadata(descrever(versao)["repositorio"], revision=commit).parameter_count
        return {"parametros": int(sum(contagem.values())), "de_onde": "cabeçalho dos arquivos de pesos no Hugging Face, no commit da versão"}
    except Exception as erro:      # sem rede, a etapa não para por causa de um número descritivo
        return {"parametros": int(round(vram_do_modelo_gb * 1e9 / 2, -7)), "de_onde": f"estimado pela memória do modelo em float16 ({type(erro).__name__} ao consultar o Hugging Face)"}


def completar_versoes(custos: dict) -> None:
    """Acrescenta ao `versao.json` de cada versão do Colab o que só se sabe no fim: condutor, custo, tempos, placa e derivados.

    Uma versão já completada não é regravada: o registro de 18/11 fica como foi fechado.
    """
    for versao in (CONFERENCIA, *CANDIDATOS, EXTENSAO):
        pasta = PASTA / versao
        arquivo = pasta / "versao.json"
        if not (pasta / "FIM_REMOTO.json").exists():
            continue
        registro = json.loads(arquivo.read_text(encoding="utf-8"))
        if "completado_em" in registro:
            continue
        bloco = custos["extensao_32b"] if versao == EXTENSAO else custos
        onde = json.loads((pasta / "ambiente.json").read_text(encoding="utf-8"))
        fim = json.loads((pasta / "FIM_REMOTO.json").read_text(encoding="utf-8"))
        partes = [parte for parte in bloco["partes"] if parte["versao"] == versao]
        metas = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(pasta.glob("*/meta.json"))}
        registro["papel"] = NOMES[versao]
        if registro["modelo"].get("parametros") is None:
            registro["modelo"].update(parametros_do_modelo(versao, registro["modelo"]["commit"], onde["vram_do_modelo_gb"]))
        for conducao in registro["conducoes"]:      # um condutor parado de propósito deixava a situação como "rodando"
            if conducao["situacao"] == "rodando" and conducao.get("fim"):
                conducao["situacao"] = ("interrompida de propósito antes de criar sessão: as tentativas de alocação iam passar dos 20 minutos combinados; "
                                        "o condutor foi relançado em seguida com menos tentativas. Sem custo")
                conducao["situacao_corrigida_no_fecho"] = "o condutor tinha deixado 'rodando'"
        condutor = PASTA / CONDUTOR_QUE_RODOU
        registro.setdefault("condutor", {     # o condutor da extensão já grava o seu; este é o das rodadas de 18/11
            "arquivo": f"data/execucoes/{ETAPA}/{CONDUTOR_QUE_RODOU}", "sha256": sha256_do_arquivo(condutor),
            "observacao": "cópia do condutor como estava quando os três condutores foram lançados (17h19 a 17h26 de 18/11); o arquivo em src/ ganhou depois o modo de só acompanhar uma sessão ocupada",
        })
        registro["gpu"] = {"placa": onde["gpu"], "nvidia_smi": onde["nvidia_smi"], "vram_total_gb": onde["vram_total_gb"],
                           "vram_do_modelo_gb": onde["vram_do_modelo_gb"], "vram_de_pico_gb": max(m["vram_de_pico_gb"] for m in metas.values())}
        registro["tempos"] = {
            "condutor": {"inicio": registro["conducoes"][-1]["inicio"], "fim": registro["conducoes"][-1]["fim"]},
            "trabalho_remoto_s": fim["segundos"], "codigo_de_saida_do_trabalho": fim["codigo"],
            "segundos_de_gpu": {c: m["segundos_de_gpu"] for c, m in metas.items()},
            "segundos_por_pedido": {c: round(m["segundos_de_gpu"] / m["pedidos"], 3) for c, m in metas.items()},
        }
        registro["custo"] = {"como": custos["como"], "partes": partes, "unidades": round(sum(parte["unidades"] for parte in partes), 2),
                             "unidades_fechadas_na_conta": round(sum(parte["unidades_fechadas_na_conta"] for parte in partes), 2),
                             "total_da_conta_na_etapa": bloco["conta"]["gasto"],
                             "aviso": "o campo conducoes[].unidades.gastas_pelo_saldo é do saldo da conta inteira; se outra sessão rodava ao mesmo tempo, o gasto dela está lá dentro. Vale o número daqui"}
        if versao in (*CANDIDATOS, EXTENSAO):
            registro["derivados"] = {nome: {"sha256": sha256_do_arquivo(caminho)} for nome, caminho in derivar(versao).items()}
            registro["para_os_alunos"] = como_professor(versao)
        registro["completado_em"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        arquivo.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")


def fechar() -> Path:
    """Calcula tudo o que o notebook mostra e grava em `fecho/`: a escada, as tabelas, as mudanças, os diagnósticos e o custo."""
    import pandas as pd

    from especialista.avaliacao import qualidade, resultados
    from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro

    particoes, rotulos = massive.carregar_entrega()
    fecho = PASTA / "fecho"
    fecho.mkdir(parents=True, exist_ok=True)
    for versao in (*CANDIDATOS, EXTENSAO):
        derivar(versao)
    custos = custo()
    completar_versoes(custos)
    (fecho / "custo.json").write_text(json.dumps(custos, ensure_ascii=False, indent=2), encoding="utf-8")

    gabarito = {c: ouro(particoes[c]) for c in CONJUNTOS}
    lidos = {versao: {c: previsoes(versao, c) for c in CONJUNTOS} for versao in ABERTOS}
    prev = {versao: {c: lidos[versao][c][0] for c in CONJUNTOS} for versao in ABERTOS}
    da_5a = ENTREGAS / "05a_professor_fechado"
    prev[FECHADO] = {}
    for c in ("dev", "test"):
        ids, prev[FECHADO][c] = carregar_previsoes(da_5a / "previsoes" / f"claude_{c}.parquet")
        if list(ids) != particoes[c]["id"].tolist():
            raise ValueError(f"As respostas do professor fechado em {c} não estão na ordem da partição.")
    do_fechado = pd.read_parquet(da_5a / "rotulos_treino.parquet")
    if do_fechado["id"].tolist() != particoes["train"]["id"].tolist():
        raise ValueError("Os rótulos de treino do professor fechado não estão na ordem da partição.")
    prev[FECHADO]["train"] = [Previsao.de_tags(i, list(t)) for i, t in zip(do_fechado["intent"], do_fechado["tags"])]

    # 1. a escada, pela regra gravada antes
    escolhido, disputas = escolher(list(CANDIDATOS))

    # 2. a qualidade de cada professor, com intervalo (o treino é descritivo: é a qualidade dos rótulos que o aluno recebe)
    linhas = []
    for sistema in (*ABERTOS, FECHADO):
        for c in CONJUNTOS:
            for metrica, valores in qualidade.intervalo(gabarito[c], prev[sistema][c]).loc[list(qualidade.METRICAS)].iterrows():
                linhas.append({"sistema": sistema, "nome": NOMES[sistema], "particao": c, "metrica": metrica, **{k: float(valores[k]) for k in ("valor", "ic_min", "ic_max")}})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(fecho / "qualidade.csv", index=False)

    # 3. as comparações pareadas B − A
    linhas = []
    for c in ("dev", "test"):
        for b, a in (*PARES, *PARES_DA_EXTENSAO):
            comparacao = qualidade.comparar(gabarito[c], prev[a][c], prev[b][c]).loc[list(qualidade.METRICAS)]
            for metrica, v in comparacao.iterrows():
                linhas.append({"particao": c, "b": b, "a": a, "comparacao": f"{CURTOS[b]} − {CURTOS[a]}", "metrica": metrica,
                               "valor_a": float(v["a"]), "valor_b": float(v["b"]),
                               **{k: float(v[k]) for k in ("diferenca", "ic_min", "ic_max")}, "distinguivel": bool(v["distinguivel"])})
    pd.DataFrame(linhas).to_csv(fecho / "comparacoes.csv", index=False)

    # 4. quanto as respostas mudam de um professor aberto para outro (descritivo, fora da regra)
    linhas = []
    for c in CONJUNTOS:
        for b, a in (*PARES[:3], *PARES_DA_EXTENSAO[:2]):
            textos = [(ra["texto"], rb["texto"]) for ra, rb in zip(lidos[a][c][2], lidos[b][c][2])]
            pares = list(zip(prev[a][c], prev[b][c]))
            linhas.append({
                "particao": c, "b": b, "a": a, "comparacao": f"{CURTOS[b]} e {CURTOS[a]}", "pedidos": len(pares),
                "mesma_intencao": sum(x.intent == y.intent for x, y in pares) / len(pares),
                "mesmas_entidades": sum(x.slots == y.slots for x, y in pares) / len(pares),
                "mesma_resposta_lida": sum(x == y for x, y in pares) / len(pares),
                "mesmo_texto_bruto": sum(x == y for x, y in textos) / len(textos),
            })
    pd.DataFrame(linhas).to_csv(fecho / "mudancas_entre_professores.csv", index=False)

    # 5. os diagnósticos de leitura das respostas
    linhas = []
    for versao in ABERTOS:
        for c in CONJUNTOS:
            _, diagnostico, registros = lidos[versao][c]
            linhas.append({
                "versao": versao, "particao": c, "pedidos": len(registros),
                **{campo: int(diagnostico.get(campo, 0)) for campo in ("sem_registro", "sem_resposta", "json_invalido", "intencao_fora_da_lista", "entidades", "tipo_fora_da_lista", "texto_nao_encontrado")},
                "cortadas_no_limite_de_tokens": sum(r.get("parada") == "max_tokens" for r in registros),
                "sem_rotulo_aproveitavel": sum(p.intent not in rotulos["intents"] for p in lidos[versao][c][0]),
            })
    pd.DataFrame(linhas).to_csv(fecho / "diagnosticos.csv", index=False)

    # 6. a tabela de resultados no formato do projeto (uma linha por sistema e partição)
    e5b = resultados.carregar(ENTREGAS / ETAPA_5B)
    parametros_8b = int(e5b[e5b["modelo"] == MODELOS["qwen3_8b"]["repositorio"]]["parametros"].iloc[0])
    linhas = []
    for versao in ABERTOS:
        if versao == VERSAO_DA_5B:
            metas = {c: json.loads((EXECUCOES / ETAPA_5B / "qwen3_8b" / c / "meta.json").read_text(encoding="utf-8")) for c in CONJUNTOS}
            extra = {"parametros": parametros_8b, "vram_gb": metas["dev"]["vram_do_modelo_gb"], "gpu": metas["dev"]["gpu"]}
        else:
            registro = json.loads((PASTA / versao / "versao.json").read_text(encoding="utf-8"))
            metas = {c: json.loads((PASTA / versao / c / "meta.json").read_text(encoding="utf-8")) for c in CONJUNTOS}
            extra = {"parametros": registro["modelo"]["parametros"], "vram_gb": registro["gpu"]["vram_do_modelo_gb"], "gpu": registro["gpu"]["placa"]}
        ritmo = sum(m["segundos_de_gpu"] for m in metas.values()) / sum(m["pedidos"] for m in metas.values())
        for c in ("dev", "test"):
            linhas.append(resultados.registrar(
                f"{NOMES[versao]} (10 exemplos de 299 rótulos)", c, gabarito[c], prev[versao][c], parametros=extra["parametros"],
                modelo=descrever(versao)["repositorio"], versao=versao, precisao=descrever(versao)["descricao"], vram_gb=extra["vram_gb"], gpu=extra["gpu"],
                segundos_por_pedido_em_lote=ritmo, respostas_de="Etapa 5b" if versao == VERSAO_DA_5B else "Etapa 11a",
            ))
    resultados.salvar(linhas, fecho)

    # 7. a escolha, com todas as disputas
    regra = json.loads((PASTA / "regra.json").read_text(encoding="utf-8"))
    valor = lambda sistema, c, metrica: float(tabela[(tabela["sistema"] == sistema) & (tabela["particao"] == c) & (tabela["metrica"] == metrica)]["valor"].iloc[0])
    candidatos = lambda versoes: {versao: {"nome": NOMES[versao], **{c: {m: valor(versao, c, m) for m in qualidade.METRICAS} for c in ("dev", "test")}} for versao in versoes}
    if not (fecho / "escolha.json").exists():      # a escolha da escada planejada, gravada em 18/11, não é regravada
        (fecho / "escolha.json").write_text(json.dumps({
        "etapa": ETAPA, "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "regra": regra["escada"]["regra"], "regra_escrita_em": regra["escrita_em"], "medida": regra["escada"]["medida"],
        "titular_de_partida": VERSAO_DA_5B, "ordem": list(ESCADA), "escolhido": escolhido, "modelo": descrever(escolhido)["repositorio"],
        "disputas": disputas,
        "candidatos": {versao: {"nome": NOMES[versao], **{c: {m: valor(versao, c, m) for m in qualidade.METRICAS} for c in ("dev", "test")}} for versao in ESCADA},
        "o_teste_nao_decide": "os números do teste estão aqui só para relato",
        "ate_onde_a_escada_foi": "até o Qwen3-14B; candidatos maiores não foram rodados",
        "para_os_alunos": como_professor(escolhido),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 7b. a extensão: o 32B desafia o escolhido da escada planejada. Arquivo novo; `escolha.json` fica como está
    regra_da_extensao = json.loads((PASTA / "regra_extensao_32b.json").read_text(encoding="utf-8"))
    if escolhido != regra_da_extensao["titular"]:
        raise ValueError("O titular da regra da extensão não é o escolhido da escada de três degraus.")
    final, disputa_da_extensao = escolher([EXTENSAO], titular=escolhido)
    if not (fecho / "escolha_extensao_32b.json").exists():
        (fecho / "escolha_extensao_32b.json").write_text(json.dumps({
            "etapa": ETAPA, "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "o_que_e": regra_da_extensao["o_que_e"], "o_que_ja_se_sabia": regra_da_extensao["o_que_ja_se_sabia"], "autorizacao": regra_da_extensao["autorizacao"],
            "regra": regra_da_extensao["regra"], "regra_escrita_em": regra_da_extensao["escrita_em"], "medida": regra_da_extensao["medida"],
            "titular": escolhido, "desafiante": EXTENSAO, "disputas": disputa_da_extensao,
            "toma_o_lugar": final == EXTENSAO, "escolhido": final, "modelo": descrever(final)["repositorio"],
            "escolha_anterior": {"arquivo": "escolha.json", "escolhido": escolhido, "continua_valendo": final == escolhido},
            "candidatos": candidatos((escolhido, EXTENSAO)),
            "o_teste_nao_decide": "os números do teste estão aqui só para relato",
            "ate_onde_a_escada_foi": "até o Qwen3-32B, na extensão; nada maior foi rodado",
            "para_os_alunos": como_professor(final),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 8. os professores novos, declarados para o comparativo (formato em `etapa10d.fichas_declaradas`)
    poucos = impressao_do_prompt()["demonstracoes"]
    papel = lambda versao: "professor aberto da extensão" if versao == EXTENSAO else "professor aberto maior" if versao == escolhido else "professor aberto sem compressão"
    ficha = lambda versao: {
        "nome": f"{descrever(versao)['nome']} sem os 4 bits ({papel(versao)}, Etapa 11a)",
        "curto": f"{descrever(versao)['nome']}, float16", "bloco": "Professores", "rotulos_humanos": poucos, "rotulos_como": "no prompt",
        "dev_humano": "escada do professor aberto (Etapas 5b e 11a): escolha entre versões pelo exact match em dev",
        "caminho": None, "modelo": None, "ponto": "professor", "papel": "professor aberto, Etapa 11a", "modelo_id": descrever(versao)["repositorio"],
        "versao": versao, "escolhido_na_escada": versao == final,
        "resultado": [ETAPA, {"versao": versao}],
        "previsoes": {c: [f"entregas/{ETAPA}/professores/{versao}/previsoes/{versao}_{c}.parquet"] for c in ("dev", "test")},
    }
    (fecho / "sistemas.json").write_text(json.dumps({
        "sistemas": {versao: ficha(versao) for versao in (*CANDIDATOS, EXTENSAO)},
        "fora_do_formato": {
            CONFERENCIA: "só tem 300 pedidos de dev (a conferência de ambiente): sem dev e teste inteiros não há linha de resultado nem previsões para declarar",
            VERSAO_DA_5B: "já está no comparativo com a chave `qwen` (entrega da Etapa 5b)",
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return fecho


def ligar(origem: Path, destino: Path) -> None:
    """Link físico: o arquivo grande aparece na entrega sem ocupar o disco duas vezes. Se o disco não deixar, copia."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        if destino.samefile(origem):
            return
        destino.unlink()
    try:
        os.link(origem, destino)
    except OSError:
        shutil.copy2(origem, destino)


def entregar() -> Path:
    """Monta `entregas/11a_professor_aberto_maior/` a partir das pastas das versões e do fecho. Nada é tirado do cache."""
    destino, fecho = entrega(ETAPA), PASTA / "fecho"
    copiar = lambda origem, alvo: (alvo.parent.mkdir(parents=True, exist_ok=True), shutil.copy2(origem, alvo))
    for nome in ("escolha.json", "escolha_extensao_32b.json", "resultados.csv", "qualidade.csv", "comparacoes.csv", "mudancas_entre_professores.csv", "diagnosticos.csv", "custo.json", "sistemas.json"):
        if (fecho / nome).exists():
            copiar(fecho / nome, destino / nome)
    copiar(PASTA / "regra.json", destino / "regra.json")
    if (PASTA / "regra_extensao_32b.json").exists():
        copiar(PASTA / "regra_extensao_32b.json", destino / "regra_extensao_32b.json")
    copiar(PASTA / CONFERENCIA / "conferencia_de_ambiente.json", destino / "conferencia_de_ambiente.json")
    for versao in (v for v in (*CANDIDATOS, EXTENSAO) if (PASTA / v / "logits_treino.npy").exists()):
        pasta, alvo = PASTA / versao, destino / "professores" / versao
        for nome in ("versao.json", "ambiente.json", "pacotes_instalados.txt", "precisao.json", "FIM_REMOTO.json", "condutor.log", "intencoes.json"):
            copiar(pasta / nome, alvo / nome)
        for nome in ("logits_treino.npy", "logits_dev.npy", "logits_test.npy", "rotulos_treino.parquet", f"previsoes/{versao}_dev.parquet", f"previsoes/{versao}_test.parquet"):
            ligar(pasta / nome, alvo / nome)
        for c in CONJUNTOS:
            ligar(pasta / c / "respostas.jsonl", alvo / "respostas" / c / "respostas.jsonl")
            copiar(pasta / c / "meta.json", alvo / "respostas" / c / "meta.json")
    pasta, alvo = PASTA / CONFERENCIA, destino / "professores" / CONFERENCIA
    for nome in ("versao.json", "ambiente.json", "pacotes_instalados.txt", "conferencia_de_ambiente.json", "condutor.log", "conferencia/respostas.jsonl", "conferencia/meta.json"):
        copiar(pasta / nome, alvo / nome)
    copiar(PASTA / "lotes_da_conferencia.json", alvo / "conferencia" / "lotes_da_conferencia.json")
    copiar(PASTA / VERSAO_DA_5B / "versao.json", destino / "professores" / VERSAO_DA_5B / "versao.json")
    for pacote in sorted((PASTA / "pacotes").glob("pacote_*.tar.gz")):
        ligar(pacote, destino / "codigo" / pacote.name)
    for condutor in sorted((PASTA / "pacotes").glob("etapa11a_colab_*.py")):      # cada condutor que rodou, com o sha256 no nome
        copiar(condutor, destino / "codigo" / condutor.name)
    for nome in ("src/especialista/experimentos/etapa11a.py", "src/especialista/experimentos/etapa11a_colab.py", "notebooks/construtores/build_nb11a.py"):
        if (RAIZ / nome).exists():
            copiar(RAIZ / nome, destino / "codigo" / Path(nome).name)
    return destino


def guardar_versao_entregue(nome: str = VERSAO_ENTREGUE_EM_1811) -> Path:
    """Antes de regenerar a entrega: guarda, dentro dela, o que foi entregue (resumo, notebook, HTML, escolha, tabelas, figuras e código).

    Não regrava uma versão já guardada. As pastas `professores/` não entram: cada versão de professor já é imutável na sua pasta.
    """
    origem = entrega(ETAPA)
    destino = origem / "versoes" / nome
    if destino.exists():
        return destino
    do_manifesto = json.loads((origem / "manifesto.json").read_text(encoding="utf-8"))
    destino.mkdir(parents=True)
    iguais = {}
    for arquivo in sorted(p for p in origem.iterdir() if p.is_file()):
        shutil.copy2(arquivo, destino / arquivo.name)
        if arquivo.name in do_manifesto["saidas"]:
            iguais[arquivo.name] = sha256_do_arquivo(arquivo) == do_manifesto["saidas"][arquivo.name]["sha256"]
    for pasta in ("figuras", "codigo"):
        shutil.copytree(origem / pasta, destino / pasta)
    notebook = RAIZ / "notebooks" / f"{ETAPA}.ipynb"
    shutil.copy2(notebook, destino / notebook.name)
    do_fecho = PASTA / "fecho_versoes" / nome       # o mesmo no cache: as tabelas de onde a entrega saiu
    if not do_fecho.exists():
        shutil.copytree(PASTA / "fecho", do_fecho)
    (destino / "LEIA.json").write_text(json.dumps({
        "o_que_e": "a entrega da Etapa 11a como foi fechada, antes de ser regenerada: resumo, notebook executado, HTML, escolha, regra, tabelas, figuras, manifesto e código",
        "guardada_em": time.strftime("%Y-%m-%d %H:%M:%S %z"), "manifesto_gerado_em": do_manifesto["gerado_em"],
        "arquivos_iguais_aos_do_manifesto": iguais, "todos_iguais": all(iguais.values()),
        "notebook": {"arquivo": notebook.name, "alterado_em": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(notebook))),
                     "observacao": "o notebook não estava no manifesto; a data é a prova de que é o executado no fecho"},
        "o_que_nao_esta_aqui": f"as pastas das versões de professor, que não mudam: entregas/{ETAPA}/professores/",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destino


def manifesto() -> Path:
    """O que a entrega contém (com a impressão digital de cada arquivo), do que ela foi feita e onde ficou o que é grande."""
    destino = entrega(ETAPA)
    arquivos = sorted(p for p in destino.rglob("*") if p.is_file() and p.name != "manifesto.json")
    entradas = [ENTREGAS / "01_dados" / n for n in ("train.parquet", "dev.parquet", "test.parquet", "rotulos.json")]
    entradas += [ENTREGAS / ETAPA_5B / n for n in ("escolha.json", "rotulos_treino.parquet", "logits_treino.npy", "previsoes/qwen3_8b_dev.parquet", "previsoes/qwen3_8b_test.parquet")]
    entradas += [respostas_da_5b(c) for c in CONJUNTOS]
    entradas += [ENTREGAS / "05a_professor_fechado" / n for n in ("previsoes/claude_dev.parquet", "previsoes/claude_test.parquet", "rotulos_treino.parquet")]
    escolha = json.loads((destino / "escolha.json").read_text(encoding="utf-8"))
    extensao = json.loads((destino / "escolha_extensao_32b.json").read_text(encoding="utf-8")) if (destino / "escolha_extensao_32b.json").exists() else None
    arquivo = destino / "manifesto.json"
    arquivo.write_text(json.dumps({
        "etapa": ETAPA, "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "o_que_e": "O que a Etapa 11a leu (com a impressão digital de cada arquivo), as regras, as versões de professor, o custo e o que ela entrega.",
        "nada_foi_treinado": True,
        "regras": {"arquivo": f"entregas/{ETAPA}/regra.json", "escritas_em": json.loads((destino / "regra.json").read_text(encoding="utf-8"))["escrita_em"]},
        "escolhido": escolha["escolhido"], "para_os_alunos": escolha["para_os_alunos"],
        "extensao": extensao and {
            "regra": f"entregas/{ETAPA}/regra_extensao_32b.json", "regra_escrita_em": extensao["regra_escrita_em"], "escolha": f"entregas/{ETAPA}/escolha_extensao_32b.json",
            "decidida_depois_de_ver_a_escada_de_tres_degraus": True, "toma_o_lugar": extensao["toma_o_lugar"], "escolhido_depois_da_extensao": extensao["escolhido"],
        },
        "versoes_guardadas_desta_entrega": {p.name: f"entregas/{ETAPA}/versoes/{p.name}/" for p in sorted((destino / "versoes").glob("*")) if p.is_dir()},
        "versoes": {versao: f"entregas/{ETAPA}/professores/{versao}/versao.json" for versao in (VERSAO_DA_5B, CONFERENCIA, *CANDIDATOS, EXTENSAO) if (destino / "professores" / versao).exists()},
        "custo": json.loads((destino / "custo.json").read_text(encoding="utf-8")),
        "entradas": {str(p.relative_to(RAIZ)): sha256_do_arquivo(p) for p in entradas},
        "saidas": {str(p.relative_to(destino)): {"sha256": sha256_do_arquivo(p), "bytes": p.stat().st_size, "link_fisico": p.stat().st_nlink > 1} for p in arquivos},
        "cache": f"data/execucoes/{ETAPA}/ (as pastas das versões; os arquivos grandes da entrega são links físicos para lá)",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return arquivo


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    comandos = argumentos.add_subparsers(dest="comando", required=True)
    for nome in ("regra", "regra-extensao", "retroativa", "lotes", "guardar-versao", "fechar", "manifesto"):
        comandos.add_parser(nome)
    comandos.add_parser("comparar").add_argument("--versao", required=True)
    do_trabalho = comandos.add_parser("trabalho")
    do_trabalho.add_argument("--versao", required=True)
    do_trabalho.add_argument("--tarefas", required=True, help="separadas por vírgula: conferencia, precisao, dev, test, train")
    a = argumentos.parse_args()
    if a.comando == "regra":
        print(gravar_regra())
    elif a.comando == "regra-extensao":
        print(gravar_regra_da_extensao())
    elif a.comando == "retroativa":
        print(versao_retroativa())
    elif a.comando == "lotes":
        destino = PASTA / "lotes_da_conferencia.json"
        if not destino.exists():
            PASTA.mkdir(parents=True, exist_ok=True)
            destino.write_text(json.dumps(lotes_da_conferencia()), encoding="utf-8")
        print(destino)
    elif a.comando == "guardar-versao":
        print(guardar_versao_entregue())
    elif a.comando == "fechar":
        print(fechar())
    elif a.comando == "manifesto":
        print(manifesto())
    elif a.comando == "comparar":
        print(json.dumps(comparar_ambiente(a.versao), ensure_ascii=False, indent=2))
    else:
        sys.exit(trabalho(a.versao, a.tarefas.split(",")))
