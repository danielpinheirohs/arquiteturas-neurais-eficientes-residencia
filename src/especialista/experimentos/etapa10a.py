"""Etapa 10a: o teto e os alunos com um encoder pré-treinado (XLM-R base).

O segundo caminho do PLANO.md. A rede do zero dá lugar ao XLM-R base com
cabeça conjunta (`modelos/encoder/`); os dados, os professores e as sementes
são os mesmos das Etapas 4 e 6:

    teto          os 11.514 rótulos humanos
    piso          os 299 rótulos humanos que as LLMs veem, e mais nada
    aluno_a       os rótulos do professor fechado (Etapa 5a)
    aluno_b1      os rótulos do professor aberto (Etapa 5b)
    aluno_b2_tN   as entidades dos rótulos do professor aberto e, na intenção,
                  as probabilidades dele, com temperatura N

Os dados de cada aluno vêm de `etapa6.dados_da_variante`: são os mesmos
pedidos, com os mesmos rótulos, que a rede do zero recebeu. Nenhum rótulo
humano entra no treino nem na parada de um aluno.

A etapa anda em quatro tempos, e cada um só começa com o anterior gravado:

    1. busca      6 configurações da receita do teto, 1 semente cada
    2. receita    a regra `escolher_receita` aponta a vencedora  -> receita.json
    3. teto       3 sementes da vencedora; a regra `perto_do_publicado`
                  confere o teto contra a linha publicada         -> portao.json
    4. o resto    piso e alunos, com a receita do teto; no fim, a temperatura
                  do B2 sai de `dev` e o teste é aberto            -> teste.json

    uv run python -m especialista.experimentos.etapa10a faltam
    uv run python -m especialista.experimentos.etapa10a treinar --variante aluno_a --semente 0
    uv run python -m especialista.experimentos.etapa10a onda

**As regras da etapa, escritas antes de qualquer treino**, estão em
`escolher_receita`, `perto_do_publicado`, `etapa6.escolher_temperatura`
(a mesma da Etapa 6), `COMPARACOES`, `fracao_do_caminho` e `quem_segue`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao import eficiencia, qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.dados import massive
from especialista.experimentos import etapa6, execucao
from especialista.modelos.encoder import Config, SistemaEncoder
from especialista.modelos.encoder.sistema import bytes_em_disco
from especialista.treino_encoder import Receita, treinar

ETAPA = "10a_encoder"
SEMENTES = (0, 1, 2)
TEMPERATURAS = etapa6.TEMPERATURAS   # a mesma grade do aluno B2 da Etapa 6: 1, 2, 4 e 8

# A linha publicada do XLM-R base no teste pt-PT do MASSIVE (FitzGerald et al., arXiv 2204.08582,
# tabelas 7 a 9): valor e margem de 95% do artigo. O modelo de lá treinou com os 51 idiomas juntos.
PUBLICADO: dict[str, tuple[float, float]] = {
    "intent_acc": (0.867, 0.012),
    "slot_f1": (0.733, 0.006),
    "exact_match": (0.646, 0.017),
}

# --------------------------------------------------------------------------
# A busca da receita do teto: 6 configurações, 1 semente cada
# --------------------------------------------------------------------------
#
# Taxa de aprendizado: os três valores que o artigo do BERT recomenda no ajuste fino
# (Devlin et al., arXiv 1810.04805, apêndice A.3). Épocas: dois valores da lista do JointBERT
# (Chen et al., arXiv 1902.10909, seção 4.2: 1, 5, 10, 20, 30, 40). O resto da receita é o
# padrão de `treino_encoder.Receita` e não entra na busca.

TAXAS = (2e-5, 3e-5, 5e-5)
EPOCAS = (10, 20)
SEMENTE_DA_BUSCA = 0


def variante_da_busca(taxa: float, epocas: int) -> str:
    return f"busca_taxa_{taxa:.0e}_epocas_{epocas}".replace("e-0", "e-")


BUSCA: dict[str, dict] = {
    variante_da_busca(taxa, epocas): {
        "nome": f"Busca do teto: taxa {taxa:.0e}, {epocas} épocas".replace("e-0", "e-"),
        "dados": None, "config": {}, "receita": {"taxa": taxa, "epocas": epocas},
    }
    for taxa in TAXAS
    for epocas in EPOCAS
}

# Cada variante diz de que variante da Etapa 6 vêm os dados (None: rótulos humanos de todos os
# pedidos) e o que ela troca na configuração. A receita de todas é a que a busca escolher.
VARIANTES: dict[str, dict] = {
    "teto": {"nome": "Teto (rótulos humanos de todos os pedidos)", "dados": None, "config": {}},
    "piso": {"nome": "Piso do cenário (só os 299 rótulos humanos)", "dados": "piso", "config": {}},
    "aluno_a": {"nome": "Aluno A (rótulos do professor fechado)", "dados": "aluno_a", "config": {}},
    "aluno_b1": {"nome": "Aluno B1 (rótulos do professor aberto)", "dados": "aluno_b1", "config": {}},
    **{
        etapa6.variante_do_b2(T): {
            "nome": f"Aluno B2 (logits do professor aberto, temperatura {T})",
            "dados": etapa6.variante_do_b2(T),
            # a mesma perda do B2 da Etapa 6: T² · KL, sem termo de rótulo rígido
            "config": {"temperatura": float(T), "peso_rotulo": 0.0},
        }
        for T in TEMPERATURAS
    },
}
# Um controle, fora da grade: o teto da semente 0 treinado todo em 32 bits, sem a precisão mista.
# Serve só para medir o que a precisão mista muda. Não entra em nenhuma decisão nem em `faltam`.
CONTROLES: dict[str, dict] = {
    "teto_em_32_bits": {
        "nome": "Controle: teto treinado todo em 32 bits (sem precisão mista)",
        "dados": None, "config": {}, "receita": {"precisao_mista": False},
    },
}
SEMENTE_DO_CONTROLE = 0
ALUNOS = {"A": "aluno_a", "B1": "aluno_b1"}   # o B2 depende da temperatura: `etapa6.variante_do_b2(T)`
PROFESSOR_DE = {"A": "fechado", "B1": "aberto", "B2": "aberto"}


def pasta_da_etapa() -> Path:
    return execucao.EXECUCOES / ETAPA


def pasta_da_execucao(variante: str, semente: int) -> Path:
    return execucao.pasta_da_execucao(ETAPA, variante, semente)


def _ler_json(caminho: Path) -> dict | None:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def _gravar_json(caminho: Path, conteudo: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Regra 1: a receita do teto
# --------------------------------------------------------------------------

REGRA_DA_RECEITA = (
    "maior exact match em dev (rótulos humanos), na melhor época de cada configuração, 1 semente; "
    "empate exato: maior acurácia de intenção, depois menos épocas, depois a menor taxa"
)


def escolher_receita(gabarito: list[Previsao], em_dev: dict[str, list[Previsao]]) -> tuple[str, pd.DataFrame]:
    """A regra de escolha da receita do teto.

    `em_dev` traz, para cada configuração da busca, as previsões em `dev` do
    seu melhor ponto; `gabarito` são os rótulos humanos de `dev`. Fica a de
    maior exact match, a métrica que já escolhe a época de parada. Empate
    exato: a de maior acurácia de intenção; se continuar, a mais barata
    (menos épocas) e, por fim, a de menor taxa.

    Devolve a escolhida e a tabela com as métricas de cada configuração.
    """
    tabela = pd.DataFrame({variante: qualidade.avaliar(gabarito, previsoes) for variante, previsoes in em_dev.items()}).T
    tabela.insert(0, "epocas", [BUSCA[v]["receita"]["epocas"] for v in tabela.index])
    tabela.insert(0, "taxa", [BUSCA[v]["receita"]["taxa"] for v in tabela.index])
    ordem = lambda v: (tabela.loc[v, "exact_match"], tabela.loc[v, "intent_acc"], -tabela.loc[v, "epocas"], -tabela.loc[v, "taxa"])
    return max(tabela.index, key=ordem), tabela.rename_axis("variante")


def decidir_receita() -> dict:
    """Aplica `escolher_receita` às 6 execuções da busca e grava `receita.json`. Não refaz se já decidiu."""
    arquivo = pasta_da_etapa() / "receita.json"
    if (decidido := _ler_json(arquivo)) is not None:
        return decidido
    if em_falta := [v for v in BUSCA if not _pronta(v, SEMENTE_DA_BUSCA)]:
        raise FileNotFoundError(f"A busca ainda não acabou; faltam: {', '.join(em_falta)}.")
    particoes, _ = massive.carregar_entrega()
    em_dev = {v: carregar(v, SEMENTE_DA_BUSCA)["previsoes_dev"] for v in BUSCA}
    vencedora, tabela = escolher_receita(ouro(particoes["dev"]), em_dev)
    decidido = {
        "etapa": ETAPA, "decidido_em": datetime.now().isoformat(timespec="seconds"), "regra": REGRA_DA_RECEITA,
        "vencedora": vencedora, "receita": BUSCA[vencedora]["receita"], "semente": SEMENTE_DA_BUSCA,
        "em_dev": json.loads(tabela.to_json(orient="index")),
        "melhor_epoca": {v: carregar(v, SEMENTE_DA_BUSCA)["meta"]["melhor_epoca"] for v in BUSCA},
    }
    _gravar_json(arquivo, decidido)
    return decidido


def receita_escolhida() -> dict:
    decidido = _ler_json(pasta_da_etapa() / "receita.json")
    if decidido is None:
        raise FileNotFoundError("A receita do teto ainda não foi escolhida: rode a busca e depois o comando `receita`.")
    return decidido


def definicao(variante: str) -> dict:
    """O nome, a configuração e a receita de uma variante. Fora da busca, a receita é a que a busca escolheu."""
    if variante in BUSCA:
        return {"nome": BUSCA[variante]["nome"], "config": {}, "receita": dict(BUSCA[variante]["receita"])}
    propria = {**VARIANTES, **CONTROLES}[variante]
    receita = {**receita_escolhida()["receita"], **propria.get("receita", {})}
    if variante == "piso":
        # O piso tem 299 exemplos. Com a receita contada em épocas ele daria 36 vezes menos passos
        # que os outros e pararia antes de aprender. Fica com o mesmo número de passos de quem
        # treina com o conjunto inteiro (Mosbach et al., arXiv 2006.04884).
        particoes, _ = massive.carregar_entrega()
        receita["passos_por_epoca"] = Receita(**receita).passos_da_epoca(len(particoes["train"]))
    return {"nome": propria["nome"], "config": dict(propria["config"]), "receita": receita}


# --------------------------------------------------------------------------
# De onde vêm os rótulos
# --------------------------------------------------------------------------


def dados_da_variante(variante: str) -> dict:
    """O que o treino lê. Para o piso e os alunos, exatamente o que a Etapa 6 entregou à rede do zero."""
    da_etapa_6 = {**BUSCA, **VARIANTES, **CONTROLES}[variante]["dados"]
    if da_etapa_6 is None:
        return {**execucao.dados_humanos(), "origem": {"treino": "rótulos humanos de todos os pedidos", "dev": "rótulos humanos de dev"}}
    lido = etapa6.dados_da_variante(da_etapa_6)
    lido.pop("vocabulario", None)                 # o vocabulário de features é da rede do zero; o encoder traz o seu
    lido["origem"] = {chave: valor for chave, valor in lido["origem"].items() if chave != "vocabulario"}
    return lido


# --------------------------------------------------------------------------
# Executar e ler
# --------------------------------------------------------------------------


def novo_sistema(rotulos: dict, config: Config, semente: int, dispositivo: str) -> SistemaEncoder:
    """O encoder pré-treinado com as cabeças recém-sorteadas. (Os testes trocam esta função por um encoder minúsculo.)"""
    return SistemaEncoder.novo(rotulos, config, semente=semente, dispositivo=dispositivo)


def _pronta(variante: str, semente: int) -> bool:
    return (pasta_da_execucao(variante, semente) / "meta.json").exists()


def _reaproveitar_da_busca(pasta: Path) -> bool:
    """A semente 0 do teto é a própria execução vencedora da busca: mesma receita, mesmos dados, mesma semente."""
    vencedora = receita_escolhida()["vencedora"]
    origem = pasta_da_execucao(vencedora, SEMENTE_DA_BUSCA)
    if not (origem / "meta.json").exists():
        return False
    shutil.copytree(origem, pasta, ignore=shutil.ignore_patterns("meta.json"))
    meta = json.loads((origem / "meta.json").read_text(encoding="utf-8"))
    meta.update({"variante": "teto", "nome": VARIANTES["teto"]["nome"], "reaproveitada_de": vencedora})
    _gravar_json(pasta / "meta.json", meta)
    return True


def executar(variante: str, semente: int, dispositivo: str | None = None) -> Path:
    """Ajusta a variante com a semente dada e grava tudo, ou devolve a pasta se já rodou.

    O que fica na pasta: `historico.csv`, `previsoes_dev.parquet` (as
    respostas em `dev` inteiro), `modelo/` e, por último, `meta.json`, que
    marca a execução como pronta. Uma pasta sem `meta.json` é sobra de uma
    execução interrompida: é apagada e refeita.
    """
    pasta = pasta_da_execucao(variante, semente)
    if (pasta / "meta.json").exists():
        return pasta
    if pasta.exists():
        shutil.rmtree(pasta)
    if variante == "teto" and semente == SEMENTE_DA_BUSCA and _reaproveitar_da_busca(pasta):
        return pasta

    d = definicao(variante)
    config, receita = Config(**d["config"]), Receita(**d["receita"])
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    lido = dados_da_variante(variante)
    treino, dev, rotulos = lido["treino"], lido["dev"], lido["rotulos"]
    avaliar = lido.get("avaliar", dev)

    sistema = novo_sistema(rotulos, config, semente, dispositivo)
    partes = sistema.modelo.parametros_por_parte()
    linhas_usadas = len(sistema.alinhador.ids_usados(treino["tokens"]))
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()

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
        "etapa": ETAPA,
        "variante": variante,
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
        "linhas_da_tabela_no_treino": linhas_usadas,   # das linhas da tabela de subtokens, quantas aparecem neste treino
        "bytes_em_disco": bytes_em_disco(pasta / "modelo"),
        "dispositivo": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    _gravar_json(pasta / "meta.json", meta)  # por último: marca "pronto"
    return pasta


def carregar(variante: str, semente: int) -> dict:
    """O que uma execução produziu: meta, histórico e previsões em dev."""
    return execucao.carregar(ETAPA, variante, semente)


def carregar_sistema(variante: str, semente: int, dispositivo: str = "cpu") -> SistemaEncoder:
    return SistemaEncoder.carregar(pasta_da_execucao(variante, semente) / "modelo", dispositivo)


def previsoes(variante: str, particao: str) -> list[list[Previsao]]:
    """As respostas das 3 sementes de uma variante em `dev` ou `test` (o teste só existe depois de aberto)."""
    return [carregar_previsoes(pasta_da_execucao(variante, s) / f"previsoes_{particao}.parquet")[1] for s in SEMENTES]


# --------------------------------------------------------------------------
# Regra 2: o teto chega perto da linha publicada?
# --------------------------------------------------------------------------

REGRA_DO_PORTAO = (
    "em cada uma das três métricas (intenção, entidades, exact match), no teste, o intervalo de 95% do teto "
    "(média de 3 sementes, bootstrap por enunciado) alcança o intervalo publicado (valor ± margem do artigo), ou o teto fica acima"
)


def perto_do_publicado(gabarito: list[Previsao], grupo: list[list[Previsao]]) -> tuple[bool, pd.DataFrame]:
    """A regra que decide se o teto reproduz a linha publicada do XLM-R base no MASSIVE pt-PT.

    Não há como parear com o modelo publicado: só existem os três números
    dele e as margens de 95% do artigo. A regra é a que sobra nesse caso: os
    dois intervalos têm de se tocar. O teto passa, numa métrica, se o topo do
    seu intervalo de 95% chega ao piso do intervalo publicado (valor − margem).
    Ficar acima do publicado também passa. Vale para a etapa só se passar
    nas três métricas.

    `gabarito` são os rótulos humanos do teste e `grupo`, as previsões das 3
    sementes do teto. Devolve o veredito e a tabela, uma linha por métrica.
    """
    nosso = qualidade.intervalo_do_grupo(gabarito, grupo)
    tabela = pd.DataFrame(
        [
            {
                "metrica": metrica, "teto": nosso.loc[metrica, "valor"], "ic_min": nosso.loc[metrica, "ic_min"],
                "ic_max": nosso.loc[metrica, "ic_max"], "publicado": valor, "margem_publicada": margem,
                "diferenca": nosso.loc[metrica, "valor"] - valor,
                "perto": bool(nosso.loc[metrica, "ic_max"] >= valor - margem),
            }
            for metrica, (valor, margem) in PUBLICADO.items()
        ]
    ).set_index("metrica")
    return bool(tabela["perto"].all()), tabela


def abrir_teste(variante: str, semente: int, dispositivo: str | None = None) -> Path:
    """Grava as respostas da execução no teste (`previsoes_test.parquet`). Não repete se já existe."""
    arquivo = pasta_da_execucao(variante, semente) / "previsoes_test.parquet"
    if arquivo.exists():
        return arquivo
    particoes, _ = massive.carregar_entrega()
    teste = particoes["test"]
    sistema = carregar_sistema(variante, semente, dispositivo or ("cuda" if torch.cuda.is_available() else "cpu"))
    return salvar_previsoes(arquivo, teste["id"], sistema.prever(list(teste["utt"])))


def medir_portao() -> dict:
    """Abre o teste para as 3 sementes do teto, aplica `perto_do_publicado` e grava `portao.json`."""
    arquivo = pasta_da_etapa() / "portao.json"
    if (medido := _ler_json(arquivo)) is not None:
        return medido
    if em_falta := [s for s in SEMENTES if not _pronta("teto", s)]:
        raise FileNotFoundError(f"Faltam sementes do teto: {em_falta}.")
    for semente in SEMENTES:
        abrir_teste("teto", semente)
    particoes, _ = massive.carregar_entrega()
    passou, tabela = perto_do_publicado(ouro(particoes["test"]), previsoes("teto", "test"))
    medido = {
        "etapa": ETAPA, "medido_em": datetime.now().isoformat(timespec="seconds"), "regra": REGRA_DO_PORTAO,
        "passou": passou, "no_teste": json.loads(tabela.to_json(orient="index")),
    }
    _gravar_json(arquivo, medido)
    return medido


# --------------------------------------------------------------------------
# Regra 3: a temperatura do aluno B2 (a mesma regra da Etapa 6) e a abertura do teste
# --------------------------------------------------------------------------


def decidir_temperatura() -> dict:
    """Aplica `etapa6.escolher_temperatura` às previsões em `dev` do B2 e abre o teste para quem vai ser medido nele.

    No teste entram o piso, o aluno A, o B1 e o B2 só na temperatura
    escolhida (o teto já foi aberto em `medir_portao`). Grava `teste.json`.
    """
    arquivo = pasta_da_etapa() / "teste.json"
    if (decidido := _ler_json(arquivo)) is not None:
        return decidido
    particoes, _ = massive.carregar_entrega()
    em_dev = {T: previsoes(etapa6.variante_do_b2(T), "dev") for T in TEMPERATURAS}
    temperatura, medias = etapa6.escolher_temperatura(ouro(particoes["dev"]), em_dev)
    abertas = ["piso", "aluno_a", "aluno_b1", etapa6.variante_do_b2(temperatura)]
    for variante in abertas:
        for semente in SEMENTES:
            abrir_teste(variante, semente)
    decidido = {
        "etapa": ETAPA, "decidido_em": datetime.now().isoformat(timespec="seconds"),
        "temperatura_do_b2": int(temperatura),
        "regra_da_temperatura": "a da Etapa 6: maior acurácia de intenção em dev (rótulos humanos), média de 3 sementes; "
                                "empate: maior exact match, depois menor temperatura",
        "em_dev": json.loads(medias.to_json(orient="index")),
        "abertas_no_teste": ["teto", *abertas],
    }
    _gravar_json(arquivo, decidido)
    return decidido


# --------------------------------------------------------------------------
# Regra 4: a lista fechada de comparações
# --------------------------------------------------------------------------
#
# Cada linha é uma diferença pareada B − A, na média das sementes, com intervalo de 95%
# (`qualidade.comparar_grupos`). "zero:" é o sistema do caminho do zero (Etapas 4b e 6);
# "fechado" e "aberto" são os professores. Nenhuma comparação entra depois de ver os resultados.

COMPARACOES: tuple[tuple[str, str, str, str], ...] = (
    # (nome, A, B, o que ela responde)
    ("Teto: XLM-R − do zero", "zero:teto", "teto", "O que o pré-treino acrescenta quando há rótulos humanos para tudo?"),
    ("Piso: XLM-R − do zero", "zero:piso", "piso", "E quando só há 299 rótulos?"),
    ("Aluno A: XLM-R − do zero", "zero:A", "A", "O pré-treino muda o que se aprende do professor fechado?"),
    ("Aluno B1: XLM-R − do zero", "zero:B1", "B1", "E dos rótulos do professor aberto?"),
    ("Aluno B2: XLM-R − do zero", "zero:B2", "B2", "E dos logits do professor aberto? (cada B2 na sua temperatura)"),
    ("Aluno A − professor fechado", "fechado", "A", "Quanto se perde (ou se ganha) ao trocar a LLM fechada pelo aluno?"),
    ("Aluno B1 − professor aberto", "aberto", "B1", "E ao trocar a LLM aberta pelo aluno de rótulos?"),
    ("Aluno B2 − professor aberto", "aberto", "B2", "E pelo aluno de logits?"),
    ("Aluno A − teto", "teto", "A", "Quanto custa não ter os 11.514 rótulos humanos?"),
    ("Aluno B2 − aluno B1", "B1", "B2", "Quanto valem os logits? (aqui o B1 não tem suavização: só os logits mudam)"),
)


def grupos_do_caminho_do_zero(particao: str) -> dict[str, list[list[Previsao]]]:
    """As respostas da rede do zero em `dev` ou `test`: o teto da Etapa 4b e o piso e os alunos da Etapa 6."""
    da_6 = json.loads((ENTREGAS / "06_destilacao" / "decisao.json").read_text(encoding="utf-8"))
    variantes = {"piso": "piso", "A": "aluno_a", "B1": "aluno_b1", "B2": etapa6.variante_do_b2(da_6["temperatura_do_b2"])}
    ler = lambda arquivo: carregar_previsoes(arquivo)[1]
    grupos = {"zero:teto": [ler(ENTREGAS / "04b_intencao" / "previsoes" / f"semente_{s}_{particao}.parquet") for s in SEMENTES]}
    for sigla, variante in variantes.items():
        grupos[f"zero:{sigla}"] = [
            ler(ENTREGAS / "06_destilacao" / "previsoes" / f"{variante}_semente_{s}_{particao}.parquet") for s in SEMENTES
        ]
    return grupos


def grupos_do_encoder(particao: str, temperatura: int) -> dict[str, list[list[Previsao]]]:
    """As respostas do XLM-R em `dev` ou `test`: teto, piso e alunos, o B2 na temperatura dada."""
    variantes = {"teto": "teto", "piso": "piso", **ALUNOS, "B2": etapa6.variante_do_b2(temperatura)}
    return {sigla: previsoes(variante, particao) for sigla, variante in variantes.items()}


def grupos_dos_professores(particao: str) -> dict[str, list[list[Previsao]]]:
    """As respostas dos dois professores, cada um como um grupo de uma execução só."""
    particoes, _ = massive.carregar_entrega()
    return {p: [etapa6.previsoes_do_professor(p, particao, particoes[particao])] for p in ("fechado", "aberto")}


def comparar_tudo(gabarito: list[Previsao], grupos: dict[str, list[list[Previsao]]]) -> pd.DataFrame:
    """Todas as comparações de `COMPARACOES`, uma linha por comparação e métrica."""
    partes = []
    for nome, a, b, _ in COMPARACOES:
        tabela = qualidade.comparar_grupos(gabarito, grupos[a], grupos[b]).rename_axis("metrica").reset_index()
        tabela.insert(0, "comparacao", nome)
        partes.append(tabela)
    return pd.concat(partes, ignore_index=True)


def fracao_do_caminho(
    gabarito: list[Previsao], piso: list[list[Previsao]], teto: list[list[Previsao]], aluno: list[list[Previsao]],
    n: int = 2000, semente: int = 0,
) -> pd.DataFrame:
    """Que fração da distância entre o piso e o teto o aluno cobre, em cada métrica, com intervalo de 95%.

        fração = (aluno − piso) / (teto − piso)

    0 é ficar no piso (a LLM não acrescentou nada aos 299 rótulos) e 1 é
    chegar ao teto (os rótulos da LLM valeram o mesmo que os humanos). Cada
    um dos três entra com a média das suas sementes. O intervalo vem do
    mesmo bootstrap por enunciado das outras medidas, com os três sistemas
    reamostrados juntos.
    """
    media = lambda grupo: pd.DataFrame([qualidade.avaliar(gabarito, p) for p in grupo]).mean()
    amostras = lambda grupo: sum(qualidade.reamostrar(gabarito, p, n, semente) for p in grupo) / len(grupo)
    valor = (media(aluno) - media(piso)) / (media(teto) - media(piso))
    sorteios = (amostras(aluno) - amostras(piso)) / (amostras(teto) - amostras(piso))
    return pd.DataFrame({"valor": valor, "ic_min": sorteios.quantile(0.025), "ic_max": sorteios.quantile(0.975)})


# --------------------------------------------------------------------------
# Regra 5: quem segue para a compressão (Etapa 10b)
# --------------------------------------------------------------------------

REGRA_DO_QUE_SEGUE = (
    "seguem o aluno A e o aluno B2 (na temperatura escolhida em dev), os mesmos do caminho do zero, para que a "
    "Etapa 10c compare cadeia com cadeia; o principal é o de maior exact match em dev; a regra da Etapa 6 "
    "(B2 − B1 em dev) é aplicada e relatada como conferência, sem mudar quem segue"
)


def quem_segue(gabarito: list[Previsao], em_dev: dict[str, list[list[Previsao]]]) -> dict:
    """A regra do que segue para a poda e a quantização do XLM-R (Etapa 10b).

    `em_dev` traz as previsões em `dev` de cada semente dos alunos `A`, `B1`
    e `B2` (este na temperatura escolhida); `gabarito` são os rótulos humanos.

    1. Seguem o aluno A e o aluno B2. São os que seguiram no caminho do zero
       (Etapa 6), e a Etapa 10c só compara as duas cadeias de compressão se
       os dois caminhos comprimirem os mesmos alunos.
    2. O principal, que a Etapa 10b comprime primeiro, é o de maior exact
       match em `dev` entre os dois.
    3. A regra da Etapa 6 para o B2 (intenção distinguivelmente maior que a
       do B1 e exact match não distinguivelmente menor) é aplicada e
       relatada em `b2_passa_na_regra_da_etapa_6`. Se o B2 não passar, ele
       segue mesmo assim, e o resumo da etapa tem de dizer isso.
    """
    da_etapa_6 = etapa6.quem_segue(gabarito, em_dev)
    seguem = ["A", "B2"]
    exact_match = {a: float(np.mean([qualidade.avaliar(gabarito, p)["exact_match"] for p in em_dev[a]])) for a in seguem}
    return {
        "seguem": seguem,
        "principal": max(seguem, key=exact_match.get),
        "exact_match_em_dev": exact_match,
        "b2_passa_na_regra_da_etapa_6": da_etapa_6["b2_passa"],
        "b2_menos_b1": da_etapa_6["b2_menos_b1"],
    }


# --------------------------------------------------------------------------
# O que falta e a onda de treinos
# --------------------------------------------------------------------------


def grade() -> list[tuple[str, int]]:
    """Todas as execuções da etapa: a busca (1 semente) e as variantes (3 sementes)."""
    return [(v, SEMENTE_DA_BUSCA) for v in BUSCA] + [(v, s) for v in VARIANTES for s in SEMENTES]


def faltam(tudo: bool = False) -> list[tuple[str, int]]:
    """As execuções que ainda não estão gravadas e que já podem rodar.

    A ordem dos quatro tempos é respeitada: sem `receita.json` só a busca
    pode rodar; sem `portao.json` aprovado, só o teto. Com `tudo=True`,
    devolve tudo o que falta, possa rodar agora ou não.
    """
    em_falta = [(v, s) for v, s in grade() if not _pronta(v, s)]
    if tudo:
        return em_falta
    da_busca = [(v, s) for v, s in em_falta if v in BUSCA]
    if da_busca or _ler_json(pasta_da_etapa() / "receita.json") is None:
        return da_busca
    do_teto = [(v, s) for v, s in em_falta if v == "teto"]
    portao = _ler_json(pasta_da_etapa() / "portao.json")
    if do_teto or portao is None or not portao["passou"]:
        return do_teto
    return em_falta


def situacao() -> str:
    """Em que tempo a etapa está, em uma linha."""
    prontas, todas = len(grade()) - len(faltam(tudo=True)), len(grade())
    receita, portao, teste = (_ler_json(pasta_da_etapa() / f"{nome}.json") for nome in ("receita", "portao", "teste"))
    partes = [f"{prontas} de {todas} execuções prontas"]
    partes.append(f"receita: {receita['vencedora']}" if receita else "receita: ainda não escolhida")
    partes.append("teto contra o publicado: " + ("ainda não medido" if portao is None else "perto" if portao["passou"] else "LONGE (a etapa para aqui)"))
    partes.append(f"teste aberto, T do B2 = {teste['temperatura_do_b2']}" if teste else "teste dos alunos: ainda fechado")
    return "; ".join(partes)


def _rodar(variante: str, semente: int) -> int:
    """Uma execução num processo próprio: a memória da placa volta inteira quando ele acaba."""
    comando = [sys.executable, "-m", "especialista.experimentos.etapa10a", "treinar", "--variante", variante, "--semente", str(semente)]
    return subprocess.run(comando, env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2"}).returncode


def onda(processos: int = 1) -> bool:
    """Roda a etapa inteira, um tempo depois do outro, com `processos` treinos de cada vez na placa.

    O padrão é um treino por vez. Medido nesta etapa, na RTX 3080: um treino
    sozinho leva 11,6 s por época; dois juntos levam 27 s por época cada um,
    ou seja, rendem menos que um só, e ocupam 7,5 dos 10 GB da placa.

    Para sozinha, sem treinar os alunos, se o teto não chegar perto da linha
    publicada. Pode ser interrompida e chamada de novo: o que já está gravado
    não é refeito. Devolve True se chegou ao fim.
    """
    avisar = lambda texto: print(f"[{datetime.now():%H:%M:%S}] {texto}", flush=True)
    while True:
        pendentes = faltam()
        if pendentes:
            avisar(f"{len(pendentes)} execuções a rodar, {processos} de cada vez: {', '.join(f'{v}/{s}' for v, s in pendentes)}")
            with ThreadPoolExecutor(max_workers=processos) as fila:
                codigos = list(fila.map(lambda par: _rodar(*par), pendentes))
            if any(codigos):
                avisar(f"PAROU: {sum(c != 0 for c in codigos)} execuções terminaram com erro. Veja o registro acima.")
                return False
        elif _ler_json(pasta_da_etapa() / "receita.json") is None:
            decidido = decidir_receita()
            avisar(f"receita do teto: {decidido['vencedora']} ({decidido['regra']})")
        elif _ler_json(pasta_da_etapa() / "portao.json") is None:
            medido = medir_portao()
            for metrica, linha in medido["no_teste"].items():
                avisar(f"teto no teste, {metrica}: {100 * linha['teto']:.1f} ({100 * linha['ic_min']:.1f} a {100 * linha['ic_max']:.1f}); "
                       f"publicado {100 * linha['publicado']:.1f} ± {100 * linha['margem_publicada']:.1f}; perto: {linha['perto']}")
        elif not _ler_json(pasta_da_etapa() / "portao.json")["passou"]:
            avisar("PAROU: o teto não chegou perto da linha publicada. Os alunos não foram treinados.")
            return False
        else:
            decidido = decidir_temperatura()
            avisar(f"temperatura do B2: {decidido['temperatura_do_b2']}; teste aberto para {', '.join(decidido['abertas_no_teste'])}")
            avisar("FIM: " + situacao())
            return True


# --------------------------------------------------------------------------
# Custo: latência em CPU
# --------------------------------------------------------------------------

RODADAS = 5


def medir_latencia(variante: str = "aluno_a", semente: int = 0, rodadas: int = RODADAS, refazer: bool = False) -> Path:
    """Mede a latência do XLM-R ajustado, no protocolo de `eficiencia.py`, ao lado da rede do zero.

    Uma consulta por vez, 1 thread de CPU, do texto à resposta, com os
    pedidos de `dev`. Os dois sistemas (o XLM-R desta etapa e o aluno A da
    rede do zero, em PyTorch) se alternam em `rodadas` rodadas de 1.024
    consultas: se a máquina oscilar, a oscilação cai sobre os dois. Precisa
    da máquina quieta. Grava `latencia.json` e `latencia.npz`, com quantos
    núcleos de CPU os outros processos ocuparam durante a medida.
    """
    from especialista.experimentos import etapa8   # os relógios de "máquina quieta" da Etapa 8
    from especialista.modelos.diet import SistemaDiet

    pasta = pasta_da_etapa() / "latencia"
    if (pasta / "latencia.json").exists() and not refazer:
        return pasta
    torch.set_num_threads(1)
    particoes, _ = massive.carregar_entrega()
    textos = list(particoes["dev"]["utt"])
    do_zero = ENTREGAS / "06_destilacao" / "modelos" / "aluno_a" / f"semente_{semente}"
    sistemas = {"xlmr": carregar_sistema(variante, semente, "cpu"), "do_zero": SistemaDiet.carregar(do_zero, "cpu")}

    carga_antes, antes = etapa8.carga_da_maquina(), etapa8.relogios()
    tempos = {nome: [] for nome in sistemas}
    for rodada in range(rodadas):
        for nome, sistema in sistemas.items():
            tempos[nome].append(eficiencia.medir_latencia(sistema, textos, semente=rodada))
    carga_depois, depois = etapa8.carga_da_maquina(), etapa8.relogios()

    pasta.mkdir(parents=True, exist_ok=True)
    np.savez(pasta / "latencia.npz", **{nome: np.stack(linhas) for nome, linhas in tempos.items()})
    _gravar_json(pasta / "latencia.json", {
        "sistemas": {"xlmr": str(pasta_da_execucao(variante, semente).relative_to(RAIZ)), "do_zero": str(do_zero.relative_to(RAIZ))},
        "threads": 1, "rodadas": rodadas, "consultas_por_rodada": eficiencia.CONSULTAS,
        "p50_por_rodada_ms": {nome: [float(np.percentile(t, 50)) for t in linhas] for nome, linhas in tempos.items()},
        "resumo": {nome: eficiencia.resumir_latencia(np.concatenate(linhas)) for nome, linhas in tempos.items()},
        "nucleos_de_outros": etapa8.nucleos_de_outros(antes, depois), "carga_antes": carga_antes, "carga_depois": carga_depois,
        "medido_em": datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__, **eficiencia.ambiente(),
    })
    return pasta


def carregar_latencia() -> dict | None:
    """Os tempos gravados por `medir_latencia` e o resumo. None se ainda não mediu."""
    pasta = pasta_da_etapa() / "latencia"
    if not (pasta / "latencia.json").exists():
        return None
    with np.load(pasta / "latencia.npz") as arquivos:
        tempos = {nome: arquivos[nome] for nome in arquivos.files}
    return {"tempos": tempos, **json.loads((pasta / "latencia.json").read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 10a: teto e alunos com o XLM-R base.")
    comandos = argumentos.add_subparsers(dest="comando", required=True)
    treino = comandos.add_parser("treinar", help="ajusta uma variante com uma semente")
    treino.add_argument("--variante", required=True, choices=sorted({**BUSCA, **VARIANTES, **CONTROLES}))
    treino.add_argument("--semente", type=int, required=True)
    falta = comandos.add_parser("faltam", help="lista o que ainda não rodou e já pode rodar, uma execução por linha")
    falta.add_argument("--tudo", action="store_true", help="lista tudo o que falta, possa rodar agora ou não")
    comandos.add_parser("situacao", help="em que tempo a etapa está")
    comandos.add_parser("receita", help="aplica a regra da receita do teto à busca")
    comandos.add_parser("portao", help="mede o teto no teste contra a linha publicada")
    comandos.add_parser("teste", help="escolhe a temperatura do B2 em dev e abre o teste")
    da_onda = comandos.add_parser("onda", help="roda a etapa inteira, um tempo depois do outro")
    da_onda.add_argument("--processos", type=int, default=1)
    latencia = comandos.add_parser("latencia", help="mede a latência em CPU (precisa da máquina quieta)")
    latencia.add_argument("--refazer", action="store_true")
    a = argumentos.parse_args()

    if a.comando == "treinar":
        torch.set_num_threads(2)
        meta = json.loads((executar(a.variante, a.semente) / "meta.json").read_text(encoding="utf-8"))
        vram = f", pico de {meta['pico_de_vram_gb']:.1f} GB na placa" if meta.get("pico_de_vram_gb") else ""
        print(f"[{datetime.now():%H:%M:%S}]", a.variante, a.semente, f"época {meta['melhor_epoca']} de {meta['receita']['epocas']}",
              {k: round(v, 4) for k, v in meta["dev"].items()},
              f"{meta['segundos_de_treino']:.0f}s ({meta['segundos_por_epoca']:.1f}s por época{vram})", flush=True)
    elif a.comando == "faltam":
        for variante, semente in faltam(a.tudo):
            print(f"--variante {variante} --semente {semente}")
    elif a.comando == "situacao":
        print(situacao())
    elif a.comando == "receita":
        decidido = decidir_receita()
        print(decidido["vencedora"], decidido["receita"])
    elif a.comando == "portao":
        medido = medir_portao()
        print(json.dumps(medido, ensure_ascii=False, indent=2))
        sys.exit(0 if medido["passou"] else 1)
    elif a.comando == "teste":
        print(json.dumps(decidir_temperatura(), ensure_ascii=False, indent=2))
    elif a.comando == "onda":
        sys.exit(0 if onda(a.processos) else 1)
    elif a.comando == "latencia":
        pasta = medir_latencia(refazer=a.refazer)
        print(json.dumps(json.loads((pasta / "latencia.json").read_text(encoding="utf-8"))["resumo"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _linha_de_comando()
