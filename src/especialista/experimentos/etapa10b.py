"""Etapa 10b: poda e fine-tuning de recuperação do XLM-R.

O mesmo roteiro da Etapa 7 (a poda da rede do zero), agora no encoder
pré-treinado da Etapa 10a. Cada execução parte de um modelo já ajustado, tira
uma parte dele e reajusta o que sobrou, com o mesmo sinal com que o modelo
aprendeu:

    sem_poda                       não tira nada e faz o mesmo reajuste (controle)
    magnitude_N                    zera N% dos pesos de menor valor absoluto das matrizes do
                                   corpo, de uma vez, e reajusta com a máscara travada
                                   (Han et al., arXiv 1506.02626)
    gradual_N                      chega aos mesmos N% aos poucos, durante o reajuste
                                   (Zhu & Gupta, arXiv 1710.01878)
    vocabulario_treino             a tabela de subtokens fica só com as linhas dos subtokens
                                   que aparecem no texto de treino (TextPruner, Yang et al.,
                                   arXiv 2203.15996)
    vocabulario_alfabeto           a tabela fica só com os subtokens escritos em letras
                                   latinas, dígitos e pontuação (sem corpus nenhum)
    camadas_8, camadas_6           o corpo fica com as 8 ou as 6 camadas de baixo, das 12
                                   (Sajjad et al., arXiv 2004.03844: tirar 4 e 6 de cima)
    vocabulario_treino_camadas_K   as duas coisas juntas, com K = 8 e K = 6

N é 30, 50, 70 ou 90, os níveis do plano.

    uv run python -m especialista.experimentos.etapa10b --faltam
    uv run python -m especialista.experimentos.etapa10b executar --modelo aluno_a --variante magnitude_50 --semente 0
    uv run python -m especialista.experimentos.etapa10b onda

**O que ficou decidido antes de qualquer resultado** (as regras estão no código, nos lugares indicados):

1. **Modelos de partida** (`modelos_a_podar`). A grade completa roda no aluno
   principal da Etapa 10a (o aluno A, que é o modelo final deste caminho) e
   no teto (a referência: a compressão sem o efeito do professor). O outro
   aluno que segue (o B2) recebe só as variantes que a regra apontar como
   principal e secundária para o aluno A, para existir a cadeia do cenário B
   no comparativo. Ele não tem grade nem regra própria.
2. **Onde a poda por magnitude age** (`poda_encoder.nomes_podaveis`): nas 72
   matrizes do corpo, com o corte por matriz. A tabela de subtokens fica
   fora; ela é tratada pelo corte do vocabulário.
3. **Corte do vocabulário** (`FORMAS`): duas formas de escolher as linhas que
   ficam, e nenhuma olha o texto de `dev` nem do teste. O tokenizador é
   reduzido junto (`poda_encoder.reduzir_tokenizador`), e os arquivos dele
   contam no tamanho. A fração dos subtokens de `dev` que cada forma deixa
   sem linha é anotada em `meta.json` como diagnóstico: não escolhe nada.
4. **Reajuste** (`EPOCAS`, `receita_do_reajuste`): 10 épocas em todas as
   execuções, com a receita de quem treinou o modelo (taxa, lote, agenda da
   taxa) e o sinal de quem treinou o modelo. É o orçamento do ajuste fino da
   Etapa 10a inteiro, como nos artigos de poda de encoders pré-treinados:
   Sajjad et al. tiram as camadas e fazem o ajuste fino de sempre; Sanh et
   al. podam durante o ajuste fino inteiro. É também o critério da Etapa 7,
   que deu ao reajuste as épocas que o treino original leva para chegar ao
   melhor ponto (30 de 80); aqui o treino chega ao melhor ponto entre a 7ª e
   a 10ª época, de 10. A máscara fica travada, o ponto de partida concorre
   como "época 0", e a parada dos alunos usa o `dev` rotulado pelo professor.
   Na poda gradual a esparsidade sobe nas 5 primeiras épocas, com um corte a
   cada 100 passos, e a máscara fica travada nas 5 últimas.

   **Este número veio de um piloto no teto** (`piloto.json`, na pasta das
   execuções): a primeira versão do desenho dava ao reajuste 4 épocas, 40%
   do treino, a proporção da Etapa 7. Com 4 épocas, o teto sem as 6 camadas
   de cima não reaprende a intenção; com 10, reaprende. O orçamento curto
   decidiria o resultado, então o orçamento passou a ser o do treino. Só o
   teto (semente 0) foi visto; os alunos, não.
5. **Regra do que segue para a quantização** (`quem_segue`): a da Etapa 7.
6. **Comparações** (`COMPARACOES`): a lista fechada do que é comparado.
7. **Controle do orçamento** (`CURTAS`): duas variantes do aluno A são
   reajustadas também com 4 épocas (40% do treino, a proporção da Etapa 7).
   Serve só para medir, no aluno, quanto o orçamento do reajuste pesa na
   recuperação. Não entra na regra.

**As respostas saem da placa de vídeo, em 32 bits**, como as dos modelos sem
poda na Etapa 10a: é contra elas que cada variante é comparada.

**Disco.** Cada modelo ocupa até 1,13 GB e a grade tem perto de cem
execuções. As respostas, as métricas e o histórico de todas ficam gravados.
Os pesos ficam enquanto a variante pode seguir para a quantização
(`limpar_pesos`); os da semente 0 ficam sempre, para medir tamanho e
latência. Um modelo apagado é refeito pelo mesmo comando, apagando a pasta
da execução.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao import eficiencia, qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.compressao import poda_encoder
from especialista.dados import massive
from especialista.experimentos import etapa10a, execucao
from especialista.modelos.encoder import Config, SistemaEncoder
from especialista.modelos.encoder.sistema import bytes_em_disco
from especialista.treino_encoder import Receita

ETAPA = "10b_encoder_poda"
SEMENTES = (0, 1, 2)
SEMENTE_DO_CUSTO = 0              # tamanho e latência são medidos nos modelos desta semente
ESPARSIDADES = (30, 50, 70, 90)   # em %, as do plano
CAMADAS = (8, 6)                  # quantas ficam, das 12: Sajjad et al. tiram 4 e 6 de cima
FORMAS: dict[str, str] = {        # as duas formas de escolher as linhas da tabela de subtokens que ficam
    "treino": "os subtokens que aparecem no texto de treino pt-PT (os pedidos, sem rótulo), mais os tokens especiais",
    "alfabeto": "os subtokens escritos só com letras latinas, dígitos e pontuação, mais os tokens especiais (sem corpus)",
}
COMBINACOES = (("treino", 8), ("treino", 6))   # no máximo duas: o menor vocabulário com cada corte de camadas

# O orçamento do reajuste, igual para todas as execuções da grade
EPOCAS = 10                # as do ajuste fino da Etapa 10a, que chega ao melhor ponto entre a 7ª e a 10ª (ver o texto do módulo)
EPOCAS_DE_RAMPA = 5        # poda gradual: a esparsidade sobe na primeira metade e a máscara fica travada na segunda
PASSOS_ENTRE_CORTES = 100  # poda gradual: a máscara é refeita a cada 100 passos, como na Etapa 7 (18 cortes nas 5 épocas de rampa)
EPOCAS_CURTAS = 4          # o controle do orçamento: 40% do treino, a proporção que o reajuste teve na Etapa 7 (30 de 80)

DECISAO_DO_ENCODER = "10a_encoder/decisao.json"
PREVISOES_SEM_PODA = "10a_encoder/previsoes"
CRITERIO = "exact_match"   # a métrica da regra do que segue

VARIANTES: dict[str, dict] = {   # na ordem em que rodam
    "sem_poda": {"nome": "Sem poda, com o mesmo reajuste (controle)", "familia": "controle", "nivel": 0},
    "vocabulario_treino": {"nome": "Vocabulário do treino", "familia": "vocabulario", "nivel": None, "forma": "treino"},
    "vocabulario_alfabeto": {"nome": "Vocabulário do alfabeto latino", "familia": "vocabulario", "nivel": None, "forma": "alfabeto"},
    **{f"camadas_{k}": {"nome": f"{k} das 12 camadas", "familia": "camadas", "nivel": 12 - k, "camadas": k} for k in CAMADAS},
    **{
        f"vocabulario_{forma}_camadas_{k}": {
            "nome": f"Vocabulário do treino e {k} das 12 camadas", "familia": "combinacao", "nivel": 12 - k, "forma": forma, "camadas": k,
        }
        for forma, k in COMBINACOES
    },
    **{
        f"{familia}_{s}": {"nome": f"Magnitude, {s}% {'de uma vez' if familia == 'magnitude' else 'gradual'}", "familia": familia, "nivel": s}
        for s in sorted(ESPARSIDADES, reverse=True)   # da maior esparsidade para a menor: ver `limpar_pesos`
        for familia in ("magnitude", "gradual")
    },
}
# O controle do orçamento, só no aluno principal e fora da regra: a mesma variante com o reajuste de 4 épocas.
CURTAS: dict[str, dict] = {
    f"{base}_curto": {**VARIANTES[base], "nome": f"{VARIANTES[base]['nome']}, reajuste de {EPOCAS_CURTAS} épocas", "familia": "curto",
                      "familia_de_base": VARIANTES[base]["familia"], "base": base, "epocas": EPOCAS_CURTAS}
    for base in ("magnitude_90", "camadas_6")
}
PESOS_SOLTOS = ("magnitude", "gradual")                   # as duas formas da poda por magnitude
PECAS_INTEIRAS = ("vocabulario", "camadas", "combinacao")  # a poda estruturada
SEM_ANTES = ("gradual",)                                  # na poda gradual não existe "podado sem reajuste"
FORA_DA_REGRA = ("controle", "curto")                     # não concorrem a seguir para a quantização

# A lista fechada de comparações da etapa. Todas são pareadas, na média das 3 sementes, no teste, nas três
# métricas (intenção, entidades, exact match), com intervalo de 95% por bootstrap de enunciados.
COMPARACOES: tuple[tuple[str, str], ...] = (
    ("Cada variante − o modelo sem poda", "depois do reajuste; para os pesos soltos de uma vez e as peças inteiras, também antes"),
    ("Depois − antes do reajuste", "na mesma variante: é o que o reajuste recupera"),
    ("Gradual − de uma vez", "no mesmo nível de esparsidade"),
    ("Reajuste de 10 épocas − reajuste de 4", "nas duas variantes de `CURTAS`, só no aluno principal: quanto o orçamento pesa na recuperação?"),
    ("A curva do XLM-R ao lado da curva da rede do zero", "aluno A, exact match por esparsidade; lado a lado no texto, sem teste entre as duas"),
)

REGRA = (
    "em dev, com rótulos humanos: passa a variante cujo exact match depois do reajuste não é distinguivelmente menor que o do "
    "modelo sem poda (IC95 da diferença pareada, média de 3 sementes). A principal é a que passa com o menor arquivo comum "
    "(denso); se nenhuma peça inteira passa, é o modelo sem poda. A secundária é a de pesos soltos que passa com a maior "
    "esparsidade."
)


def definicao(variante: str) -> dict:
    """O que a variante tira e por quantas épocas é reajustada."""
    propria = {**VARIANTES, **CURTAS}[variante]
    return {"epocas": EPOCAS, **propria}


def orcamento(variante: str) -> dict:
    """O orçamento do reajuste da variante. Fica anotado em cada execução: se mudar, a execução antiga deixa de valer."""
    propria = definicao(variante)
    gradual = propria.get("familia_de_base", propria["familia"]) == "gradual"
    return {"epocas": propria["epocas"], "epocas_de_rampa": EPOCAS_DE_RAMPA if gradual else None,
            "passos_entre_cortes": PASSOS_ENTRE_CORTES if gradual else None}


def pasta_da_etapa() -> Path:
    return execucao.EXECUCOES / ETAPA


def pasta_da_execucao(modelo: str, variante: str, semente: int) -> Path:
    return execucao.pasta_da_execucao(f"{ETAPA}/{modelo}", variante, semente)


def _ler_json(caminho: Path) -> dict | None:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def _gravar_json(caminho: Path, conteudo: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Quem é podado, e com que sinal cada um é reajustado
# --------------------------------------------------------------------------


def modelos_a_podar() -> dict[str, dict]:
    """Os modelos de partida, lidos de `entregas/10a_encoder/decisao.json`: o aluno principal, o outro aluno, o teto.

    Para cada um: `nome`, `curto`, `variante_da_10a` (de onde vêm os dados do
    reajuste), `pastas` (a de cada semente, lida por `SistemaEncoder.carregar`),
    `config`, `receita`, `sinal`, `previsoes` (onde a Etapa 10a entregou as
    respostas do modelo sem poda), `principal` e `grade`: "completa" (todas
    as variantes) ou "do_principal" (só as que seguem do aluno principal).
    """
    decisao = json.loads((ENTREGAS / DECISAO_DO_ENCODER).read_text(encoding="utf-8"))

    def ficha(chave: str, origem: dict, nome: str, curto: str, grade: str, principal: bool = False) -> dict:
        return {
            "nome": nome, "curto": curto, "variante_da_10a": chave,
            "pastas": {int(semente): RAIZ / pasta for semente, pasta in origem["pastas_por_semente"].items()},
            "config": origem["config"], "receita": origem["receita"], "sinal": origem["sinal"],
            "previsoes": str(ENTREGAS / PREVISOES_SEM_PODA / (chave + "_semente_{semente}_{particao}.parquet")),
            "grade": grade, "principal": principal,
        }

    alunos, principal = decisao["alunos"], decisao["principal"]
    ordem = [principal, *(sigla for sigla in decisao["seguem"] if sigla != principal)]
    modelos = {
        alunos[sigla]["variante"]: ficha(alunos[sigla]["variante"], alunos[sigla], alunos[sigla]["nome"], f"Aluno {sigla}",
                                         "completa" if sigla == principal else "do_principal", sigla == principal)
        for sigla in ordem
    }
    modelos["teto"] = ficha("teto", decisao["referencias"]["teto"], "Teto (rótulos humanos)", "Teto", "completa")
    return modelos


def modelo_principal() -> str:
    return next(chave for chave, ficha in modelos_a_podar().items() if ficha["principal"])


def dados_do_modelo(modelo: str) -> dict:
    """O treino e o `dev` de parada do modelo: os mesmos com que a Etapa 10a o treinou.

    Para um aluno, rótulos do professor no treino e no `dev` (e os logits,
    quando o aluno aprende com eles). Para o teto, rótulos humanos.
    """
    return etapa10a.dados_da_variante(modelos_a_podar()[modelo]["variante_da_10a"])


def receita_do_reajuste(modelo: str, epocas: int = EPOCAS) -> Receita:
    """A receita com que o modelo foi treinado, com o número de épocas do reajuste."""
    return Receita(**{**modelos_a_podar()[modelo]["receita"], "epocas": epocas})


def previsoes_sem_poda(modelo: str, particao: str) -> list[list[Previsao]]:
    """As respostas do modelo de partida (sem poda) em `dev` ou `test`, uma lista por semente, como a Etapa 10a entregou."""
    particoes, _ = massive.carregar_entrega()
    esperados = list(particoes[particao]["id"])
    grupo = []
    for semente in SEMENTES:
        ids, previsoes = carregar_previsoes(Path(modelos_a_podar()[modelo]["previsoes"].format(semente=semente, particao=particao)))
        if list(ids) != esperados:
            raise ValueError(f"As respostas sem poda de {modelo} (semente {semente}, {particao}) não estão na ordem da partição.")
        grupo.append(previsoes)
    return grupo


# --------------------------------------------------------------------------
# Uma execução: podar, reajustar, gravar
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _digital_do_arquivo(caminho: str, modificado_em: int, tamanho: int) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:16]


def digital_de_partida(modelo: str, semente: int) -> str:
    """A impressão digital do modelo de partida: muda se ele for treinado de novo.

    É a das cabeças (`cabecas.pt`, que todo treino muda), junto com o tamanho
    do arquivo do encoder. Ler o encoder inteiro, de mais de 1 GB, a cada
    consulta de `--faltam` custaria segundos por execução.
    """
    pasta = modelos_a_podar()[modelo]["pastas"][semente]
    cabecas = (pasta / "cabecas.pt").stat()
    encoder = (pasta / "encoder" / "model.safetensors").stat().st_size
    return f"{_digital_do_arquivo(str(pasta / 'cabecas.pt'), cabecas.st_mtime_ns, cabecas.st_size)}-{encoder}"


def pronta(modelo: str, variante: str, semente: int) -> bool:
    """A execução está gravada, partiu dos pesos que o modelo de partida tem hoje e teve o orçamento de reajuste de hoje?"""
    meta = _ler_json(pasta_da_execucao(modelo, variante, semente) / "meta.json")
    return (meta is not None and meta.get("digital_de_partida") == digital_de_partida(modelo, semente)
            and meta.get("orcamento") == orcamento(variante))


def ids_da_forma(forma: str, sistema: SistemaEncoder) -> torch.Tensor:
    """As linhas da tabela de subtokens que ficam, em cada forma de `FORMAS`. Nenhuma lê `dev` nem o teste."""
    if forma == "treino":
        particoes, _ = massive.carregar_entrega()   # o texto de todos os pedidos de treino, sem rótulo: o mesmo para todo modelo
        return poda_encoder.ids_do_texto(sistema, particoes["train"]["tokens"])
    if forma == "alfabeto":
        return poda_encoder.ids_por_alfabeto(sistema.tokenizador)
    raise ValueError(f"Forma de corte do vocabulário desconhecida: {forma}. As que existem: {', '.join(FORMAS)}.")


def executar(modelo: str, variante: str, semente: int, dispositivo: str | None = None) -> Path:
    """Poda e reajusta o modelo com a semente dada, ou devolve a pasta se já rodou.

    Grava o histórico do reajuste, as respostas em `dev` e no teste (antes e
    depois do reajuste), o modelo podado e reajustado (uma pasta comum de
    `SistemaEncoder`) e, por último, `meta.json`, que marca a execução como
    pronta. Uma pasta sem `meta.json`, ou gravada a partir de outros pesos ou
    com outro orçamento de reajuste, é apagada e refeita.
    """
    pasta = pasta_da_execucao(modelo, variante, semente)
    if pronta(modelo, variante, semente):
        return pasta
    if pasta.exists():
        shutil.rmtree(pasta)

    de_partida, propria = modelos_a_podar()[modelo], definicao(variante)
    familia = propria.get("familia_de_base", propria["familia"])
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    lido = dados_do_modelo(modelo)
    treino, dev = lido["treino"], lido["dev"]
    particoes, _ = massive.carregar_entrega()
    medir = {"dev": particoes["dev"], "test": particoes["test"]}
    receita = receita_do_reajuste(modelo, propria["epocas"])

    sistema = SistemaEncoder.carregar(de_partida["pastas"][semente], dispositivo)
    if sistema.modelo.config != Config(**de_partida["config"]):
        raise ValueError(f"O modelo em {de_partida['pastas'][semente]} não tem a configuração anotada na decisão da Etapa 10a.")
    contagem_de_partida = poda_encoder.contar(sistema.modelo)
    linhas_de_partida = sistema.modelo.encoder.get_input_embeddings().num_embeddings
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inicio = time.perf_counter()
    mascaras = agenda = vocabulario = None
    if familia == "magnitude":
        mascaras = poda_encoder.mascaras_por_magnitude(sistema.modelo, propria["nivel"] / 100)
        poda_encoder.aplicar(sistema.modelo, mascaras)
    elif familia == "gradual":
        agenda = poda_encoder.Agenda(propria["nivel"] / 100, EPOCAS_DE_RAMPA, PASSOS_ENTRE_CORTES)
    if "forma" in propria:
        ficam = ids_da_forma(propria["forma"], sistema)
        menor = poda_encoder.cortar_vocabulario(sistema, ficam)
        medida = lambda enunciados: poda_encoder.cobertura(sistema.tokenizador, menor.tokenizador, ficam, enunciados,
                                                           sistema.modelo.config.max_subtokens)
        vocabulario = {   # diagnóstico: não escolhe nada
            "forma": propria["forma"], "linhas_antes": linhas_de_partida, "linhas": len(ficam),
            "cobertura": {"treino": medida(treino["tokens"]), "dev": medida(particoes["dev"]["tokens"]), "test": medida(particoes["test"]["tokens"])},
        }
        sistema = menor
    if "camadas" in propria:
        sistema = poda_encoder.manter_camadas(sistema, propria["camadas"])

    tem_antes = familia not in SEM_ANTES
    antes = {nome: sistema.prever(list(particao["utt"])) for nome, particao in medir.items()} if tem_antes else {}
    dev_antes = qualidade.avaliar(ouro(dev), sistema.prever(list(dev["utt"]))) if tem_antes else None
    historico = poda_encoder.reajustar(sistema, treino, dev, receita, semente=semente, mascaras=mascaras, agenda=agenda)
    duracao = time.perf_counter() - inicio
    depois = {nome: sistema.prever(list(particao["utt"])) for nome, particao in medir.items()}

    contagem = poda_encoder.contar(sistema.modelo)
    if familia in PESOS_SOLTOS:   # a esparsidade pedida tem de ser a obtida: se a máscara vazou, a execução não é gravada
        fracao = propria["nivel"] / 100
        pedidos = sum(round(fracao * sistema.modelo.get_parameter(nome).numel()) for nome in poda_encoder.nomes_podaveis(sistema.modelo))
        em_zero = contagem["podaveis"] - contagem["nao_nulos_podaveis"]
        if not pedidos <= em_zero <= pedidos + 100:   # a folga cobre algum peso que o reajuste leve a zero por acaso
            raise RuntimeError(f"Pedidos {pedidos} pesos em zero; o modelo tem {em_zero}.")

    pasta.mkdir(parents=True, exist_ok=True)
    historico.to_csv(pasta / "historico.csv", index=False)
    for particao, tabela in medir.items():
        salvar_previsoes(pasta / f"previsoes_{particao}.parquet", tabela["id"], depois[particao])
        if tem_antes:
            salvar_previsoes(pasta / f"previsoes_{particao}_antes.parquet", tabela["id"], antes[particao])
    sistema.salvar(pasta / "modelo")
    epocas_do_laco = historico[historico["epoca"] > 0]
    meta = {
        "etapa": ETAPA,
        "modelo": modelo,
        "variante": variante,
        "nome": propria["nome"],
        "familia": propria["familia"],
        "nivel": propria["nivel"],
        "semente": semente,
        "de_partida": str(de_partida["pastas"][semente].relative_to(RAIZ)),
        "digital_de_partida": digital_de_partida(modelo, semente),
        "config": sistema.modelo.config.como_dict(),
        "receita": receita.como_dict(),
        "orcamento": orcamento(variante),
        "vocabulario": vocabulario,
        "camadas": len(sistema.modelo.encoder.encoder.layer),
        "melhor_epoca": int(historico.loc[historico["melhor"], "epoca"].iloc[0]),
        "dev": qualidade.avaliar(ouro(dev), sistema.prever(list(dev["utt"]))),   # no dev que escolheu a época de parada
        "dev_antes": dev_antes,                                                  # no mesmo dev, antes do reajuste
        "contagem_de_partida": contagem_de_partida,
        "contagem": contagem,
        "parametros_por_parte": sistema.modelo.parametros_por_parte(),
        "bytes_em_disco": bytes_em_disco(pasta / "modelo"),   # o sistema inteiro no arquivo comum: rede, tokenizador e configurações
        "exemplos_de_treino": len(treino),
        "exemplos_de_dev": len(dev),
        "origem": lido.get("origem"),
        "segundos": duracao,
        "segundos_por_epoca": float(epocas_do_laco["segundos"].diff().fillna(epocas_do_laco["segundos"]).median()),
        "pico_de_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if dispositivo == "cuda" else None,
        "respostas_em": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    _gravar_json(pasta / "meta.json", meta)  # por último: marca "pronto"
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


# --------------------------------------------------------------------------
# A regra de decisão da etapa
# --------------------------------------------------------------------------


def linha_da_regra(gabarito: list[Previsao], sem_poda: list[list[Previsao]], em_dev: list[list[Previsao]]) -> dict:
    """O passo 1 da regra, para uma variante: a diferença pareada de exact match em `dev` e se ela passa."""
    comparacao = qualidade.comparar_grupos(gabarito, sem_poda, em_dev).loc[CRITERIO]
    return {
        "sem_poda": float(comparacao["a"]), "podado": float(comparacao["b"]), "diferenca": float(comparacao["diferenca"]),
        "ic_min": float(comparacao["ic_min"]), "ic_max": float(comparacao["ic_max"]), "passa": bool(not comparacao["ic_max"] < 0),
    }


def escolher(tabela: pd.DataFrame) -> dict:
    """Os passos 2 e 3 da regra, a partir da tabela das candidatas (`familia`, `nivel`, `bytes`, `podado`, `passa`)."""
    passam = tabela[tabela["passa"]] if len(tabela) else tabela
    pecas = passam[passam["familia"].isin(PECAS_INTEIRAS)] if len(passam) else passam
    principal = min(pecas.index, key=lambda v: (pecas.loc[v, "bytes"], -pecas.loc[v, "podado"])) if len(pecas) else None
    soltos = passam[passam["familia"].isin(PESOS_SOLTOS)] if len(passam) else passam
    secundaria = (
        max(soltos.index, key=lambda v: (soltos.loc[v, "nivel"], soltos.loc[v, "podado"], soltos.loc[v, "familia"] == "magnitude"))
        if len(soltos) else None
    )
    return {"principal": principal, "secundaria": secundaria, "seguem": [v for v in (principal, secundaria) if v]}


def quem_segue(gabarito: list[Previsao], sem_poda: list[list[Previsao]], candidatos: dict[str, dict]) -> dict:
    """A regra do que segue para a quantização (Etapa 10c), para um modelo de partida. É a regra da Etapa 7.

    `gabarito` são os rótulos humanos de `dev`; `sem_poda`, as respostas do
    modelo de partida em `dev`, uma lista por semente. `candidatos` traz, para
    cada variante podada e reajustada, `em_dev` (as respostas de cada semente),
    `nao_nulos` (parâmetros diferentes de zero) e `bytes` (o tamanho medido do
    sistema no arquivo comum, que grava todos os pesos, zeros inclusive).

    1. Uma variante **passa** se o exact match dela em `dev`, depois do
       reajuste, não for distinguivelmente menor que o do modelo sem poda: o
       intervalo de 95% da diferença pareada (média das sementes) não fica todo
       abaixo de zero.
    2. A **principal** é a variante que passa com o menor arquivo comum. É
       esse arquivo que a quantização e uma execução comum consomem, e só
       tirar peças inteiras (linhas da tabela, camadas) o diminui. Empate: a de
       maior exact match. Se nenhuma peça inteira passa, a principal é o
       modelo sem poda (`None`).
    3. A **secundária** é a dos **pesos soltos** (magnitude de uma vez e
       gradual) que passa com a maior esparsidade. Se as duas formas passam
       nessa esparsidade, a de maior exact match; se empatam, a de uma vez,
       que é a mais simples. Se nenhuma passa, não há secundária (`None`).

    O controle e as variantes de `CURTAS` não são candidatos. Devolve
    `principal`, `secundaria`, `seguem` (as variantes podadas que seguem, na
    ordem principal, secundária; pode ser vazia) e `tabela`, com a diferença e
    o intervalo de cada candidata.
    """
    linhas = [
        {"variante": variante, "familia": VARIANTES[variante]["familia"], "nivel": VARIANTES[variante]["nivel"],
         "nao_nulos": candidato["nao_nulos"], "bytes": candidato["bytes"], **linha_da_regra(gabarito, sem_poda, candidato["em_dev"])}
        for variante, candidato in candidatos.items()
    ]
    colunas = ["variante", "familia", "nivel", "nao_nulos", "bytes", "sem_poda", "podado", "diferenca", "ic_min", "ic_max", "passa"]
    tabela = pd.DataFrame(linhas, columns=colunas).set_index("variante")
    return {**escolher(tabela), "tabela": tabela}


def completa(modelo: str, variante: str) -> bool:
    return all(pronta(modelo, variante, semente) for semente in SEMENTES)


def tabela_da_regra(modelo: str) -> pd.DataFrame:
    """A tabela da regra para as variantes do modelo que já têm as 3 sementes. A linha de cada uma fica guardada ao lado das execuções."""
    candidatas = [v for v in VARIANTES if VARIANTES[v]["familia"] not in FORA_DA_REGRA and completa(modelo, v)]
    linhas, gabarito, sem_poda = [], None, None
    for variante in candidatas:
        arquivo = pasta_da_execucao(modelo, variante, 0).parent / "regra_em_dev.json"
        digitais = [digital_de_partida(modelo, s) for s in SEMENTES]
        guardada = _ler_json(arquivo)
        if guardada is None or guardada["digitais"] != digitais:
            if gabarito is None:
                particoes, _ = massive.carregar_entrega()
                gabarito, sem_poda = ouro(particoes["dev"]), previsoes_sem_poda(modelo, "dev")
            metas = [carregar(modelo, variante, s) for s in SEMENTES]
            guardada = {
                "variante": variante, "familia": VARIANTES[variante]["familia"], "nivel": VARIANTES[variante]["nivel"],
                "nao_nulos": metas[SEMENTE_DO_CUSTO]["meta"]["contagem"]["nao_nulos"], "bytes": metas[SEMENTE_DO_CUSTO]["meta"]["bytes_em_disco"],
                **linha_da_regra(gabarito, sem_poda, [m["previsoes"]["dev"] for m in metas]), "digitais": digitais,
            }
            _gravar_json(arquivo, guardada)
        linhas.append({chave: valor for chave, valor in guardada.items() if chave != "digitais"})
    colunas = ["variante", "familia", "nivel", "nao_nulos", "bytes", "sem_poda", "podado", "diferenca", "ic_min", "ic_max", "passa"]
    return pd.DataFrame(linhas, columns=colunas).set_index("variante")


def seguem_do_principal() -> dict | None:
    """O que a regra aponta para o aluno principal, quando a grade dele está completa. Fica gravado e não é refeito.

    É o que define as execuções do outro aluno (`grade: "do_principal"`).
    Devolve None enquanto faltar alguma execução da grade do aluno principal.
    """
    arquivo = pasta_da_etapa() / "segue_do_principal.json"
    if (decidido := _ler_json(arquivo)) is not None:
        return decidido
    principal = modelo_principal()
    if any(not completa(principal, v) for v in VARIANTES):
        return None
    tabela = tabela_da_regra(principal)
    decidido = {
        "etapa": ETAPA, "decidido_em": datetime.now().isoformat(timespec="seconds"), "modelo": principal, "regra": REGRA,
        **escolher(tabela), "em_dev": json.loads(tabela.to_json(orient="index")),
    }
    _gravar_json(arquivo, decidido)
    return decidido


# --------------------------------------------------------------------------
# A grade, o que falta e o disco
# --------------------------------------------------------------------------


def variantes_do_modelo(modelo: str) -> list[str] | None:
    """As variantes que o modelo recebe na grade principal. None: ainda depende da decisão do aluno principal."""
    if modelos_a_podar()[modelo]["grade"] == "completa":
        return list(VARIANTES)
    decidido = seguem_do_principal()
    return None if decidido is None else list(decidido["seguem"])


def grade() -> list[tuple[str, str, int]]:
    """As execuções (modelo, variante, semente) que já se sabe que a etapa tem, na ordem em que rodam.

    Primeiro a grade do aluno principal; depois o outro aluno, com as
    variantes que seguem do principal (só entram quando a grade do principal
    termina); depois o teto; por último o controle do orçamento.
    """
    execucoes = [(m, v, s) for m in modelos_a_podar() for v in (variantes_do_modelo(m) or ()) for s in SEMENTES]
    return execucoes + [(modelo_principal(), v, s) for v in CURTAS for s in SEMENTES]


def faltam() -> list[tuple[str, str, int]]:
    """As execuções que ainda não estão gravadas, ou que partiram de pesos que o modelo de partida já não tem."""
    return [(m, v, s) for m, v, s in grade() if not pronta(m, v, s)]


def situacao() -> str:
    """Em que ponto a etapa está, em uma linha."""
    todas, em_falta = grade(), faltam()
    partes = [f"{len(todas) - len(em_falta)} de {len(todas)} execuções prontas"]
    decidido = seguem_do_principal()
    if decidido is None:
        dependentes = [m for m, ficha in modelos_a_podar().items() if ficha["grade"] == "do_principal"]
        partes.append(f"{', '.join(dependentes)}: espera a grade de {modelo_principal()} (mais {len(SEMENTES)} execuções por variante que seguir)")
    else:
        partes.append(f"do {decidido['modelo']} seguem: principal {decidido['principal']}, secundária {decidido['secundaria']}")
    return "; ".join(partes)


def limpar_pesos() -> list[Path]:
    """Apaga do cache os pesos das execuções que já não podem seguir para a quantização. Devolve as pastas apagadas.

    O que nunca é apagado: as respostas, as métricas e o histórico de toda
    execução; os pesos da semente 0 das variantes podadas (para medir tamanho
    e latência); os pesos das variantes que ainda podem seguir.

    O que é apagado, assim que as 3 sementes da variante estão prontas:

    - o controle e as variantes de `CURTAS`, que estão fora da regra;
    - a variante que não passa no passo 1 da regra (sementes 1 e 2);
    - a de pesos soltos que passa, quando outra de pesos soltos já passou com
      esparsidade maior (sementes 1 e 2): pela regra, ela não pode mais ser a
      secundária. Por isso a grade roda da maior esparsidade para a menor.

    As peças inteiras que passam ficam todas até o fim. O outro aluno
    (`grade: "do_principal"`) fica inteiro: as variantes dele seguem por
    desenho. Um modelo apagado é reproduzível: basta apagar a pasta da
    execução e rodar o mesmo comando.
    """
    apagadas = []
    for modelo, ficha in modelos_a_podar().items():
        if ficha["grade"] != "completa":
            continue
        tabela = tabela_da_regra(modelo)
        passam = tabela[tabela["passa"]] if len(tabela) else tabela
        soltos = passam[passam["familia"].isin(PESOS_SOLTOS)] if len(passam) else passam
        maior_nivel = soltos["nivel"].max() if len(soltos) else None
        for variante in [*VARIANTES, *(CURTAS if ficha["principal"] else ())]:
            if not completa(modelo, variante):
                continue
            familia = definicao(variante)["familia"]
            if familia in FORA_DA_REGRA:
                sementes = SEMENTES
            elif not tabela.loc[variante, "passa"] or (familia in PESOS_SOLTOS and tabela.loc[variante, "nivel"] < maior_nivel):
                sementes = tuple(s for s in SEMENTES if s != SEMENTE_DO_CUSTO)
            else:
                continue
            for semente in sementes:
                pesos = pasta_da_execucao(modelo, variante, semente) / "modelo"
                if pesos.exists():
                    shutil.rmtree(pesos)
                    apagadas.append(pesos)
    return apagadas


def limpar_depois_da_decisao() -> list[Path]:
    """Com a entrega gravada: apaga do cache os pesos de tudo o que não segue para a quantização. Devolve as pastas apagadas.

    Lê `entregas/10b_encoder_poda/decisao.json`. Antes de apagar qualquer
    coisa, confere que todo modelo que segue está na entrega, com as 3
    sementes: é de lá que a Etapa 10c lê. Ficam no cache os pesos das
    variantes que seguem (são os mesmos arquivos da entrega, sem ocupar o
    disco duas vezes) e, de todas as execuções, as respostas, as métricas e os
    históricos. Um modelo apagado é refeito pelo comando `executar`, depois
    de apagar a pasta da execução.
    """
    decisao = _ler_json(ENTREGAS / ETAPA / "decisao.json")
    if decisao is None:
        raise FileNotFoundError("A entrega da etapa ainda não tem decisao.json: nada é apagado antes dela.")
    seguem = set()
    for modelo, do_modelo in decisao["modelos"].items():
        for ficha in do_modelo["seguem"]:
            for semente in SEMENTES:
                entregue = RAIZ / ficha["pastas_por_semente"][str(semente)]
                if not all((entregue / arquivo).exists() for arquivo in (*poda_encoder.ARQUIVOS_DA_REDE, "config.json", "encoder/tokenizer.json")):
                    raise FileNotFoundError(f"Falta na entrega o modelo que segue: {entregue}. Nada foi apagado.")
            seguem.add((modelo, ficha["variante"]))
    apagadas = []
    for pesos in sorted(pasta_da_etapa().glob("*/*/semente_*/modelo")):
        if (pesos.parts[-4], pesos.parts[-3]) not in seguem:
            shutil.rmtree(pesos)
            apagadas.append(pesos)
    return apagadas


# --------------------------------------------------------------------------
# A onda: tudo o que falta, uma execução por vez na placa
# --------------------------------------------------------------------------


def _rodar(modelo: str, variante: str, semente: int) -> int:
    """Uma execução num processo próprio: a memória da placa volta inteira quando ele acaba."""
    comando = [sys.executable, "-m", "especialista.experimentos.etapa10b", "executar",
               "--modelo", modelo, "--variante", variante, "--semente", str(semente)]
    return subprocess.run(comando, env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "TQDM_DISABLE": "1"}).returncode


def onda() -> bool:
    """Roda tudo o que falta, uma execução por vez, e limpa o disco pelo caminho. Devolve True se nada falhou.

    Uma execução por vez: medido na Etapa 10a, dois treinos juntos na RTX
    3080 rendem menos que um. Pode ser interrompida e chamada de novo: o que
    já está gravado não é refeito. Uma execução que termina com erro é
    anotada e pulada; as outras continuam. No fim, mede os tamanhos em disco
    (`medir_tamanhos`), que não usam a placa.
    """
    avisar = lambda texto: print(f"[{datetime.now():%H:%M:%S}] {texto}", flush=True)
    falhas: list[tuple[str, str, int]] = []
    _gravar_json(pasta_da_etapa() / "onda.json", {"pid": os.getpid(), "inicio": datetime.now().isoformat(timespec="seconds")})
    avisar(f"INÍCIO (processo {os.getpid()}): " + situacao())
    while pendentes := [e for e in faltam() if e not in falhas]:
        modelo, variante, semente = pendentes[0]
        avisar(f"faltam {len(pendentes)}; agora: {modelo} {variante} semente {semente}")
        if _rodar(modelo, variante, semente) != 0:
            falhas.append((modelo, variante, semente))
            avisar(f"ERRO em {modelo} {variante} semente {semente}: pulada; veja o registro acima")
        try:
            apagadas = limpar_pesos()
        except Exception as erro:   # a limpeza não pode derrubar a onda
            avisar(f"limpeza do disco falhou ({type(erro).__name__}: {erro}); a onda continua")
        else:
            if apagadas:
                avisar(f"disco: apagados os pesos de {len(apagadas)} execuções que não seguem")
    avisar("execuções terminadas: " + situacao())
    # os tamanhos em disco não usam a placa: 4 processos de uma vez, cada um com a sua fatia dos modelos
    medidores = [subprocess.Popen([sys.executable, "-m", "especialista.experimentos.etapa10b", "tamanhos", "--fatia", str(i), "--de", "4"],
                                  stdout=subprocess.DEVNULL, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "TQDM_DISABLE": "1"})
                 for i in range(4)]
    if any([medidor.wait() != 0 for medidor in medidores]):
        avisar("a medida dos tamanhos falhou em algum modelo; rode o comando `tamanhos` para completar")
    avisar(f"tamanhos em disco medidos em {len(list((pasta_da_etapa() / 'tamanhos').glob('*.json')))} modelos")
    if falhas:
        avisar("FIM COM ERROS: " + ", ".join(f"{m}/{v}/{s}" for m, v, s in falhas))
        return False
    avisar("FIM: " + situacao())
    return True


# --------------------------------------------------------------------------
# Custo: tamanhos em disco e latência em CPU
# --------------------------------------------------------------------------

DE_PARTIDA = "de_partida"   # o nome, nas medidas de custo, do modelo sem poda
RODADAS = 5


def pasta_do_modelo(modelo: str, variante: str, semente: int = SEMENTE_DO_CUSTO) -> Path:
    """Onde estão os pesos: a pasta da Etapa 10a para o modelo de partida, a da execução para uma variante."""
    if variante == DE_PARTIDA:
        return modelos_a_podar()[modelo]["pastas"][semente]
    return pasta_da_execucao(modelo, variante, semente) / "modelo"


def _arquivo_dos_tamanhos(modelo: str, variante: str) -> Path:
    return pasta_da_etapa() / "tamanhos" / f"{modelo}__{variante}.json"


def medir_tamanhos(refazer: bool = False, fatia: int = 0, de: int = 1) -> dict[tuple[str, str], dict]:
    """O tamanho em disco, em três formatos, do modelo de partida e de cada variante com pesos gravados (semente 0).

    Para cada um, `poda_encoder.tamanhos`: a rede, o tokenizador com as
    configurações e o sistema inteiro, no arquivo comum, no arquivo comum com
    zip e no arquivo esparso (gravado de verdade numa pasta temporária, medido
    e apagado). Junto vão os parâmetros e os diferentes de zero. Não usa a
    placa de vídeo. O que já foi medido não é refeito.

    O controle fica de fora: ele tem a rede do modelo de partida, e os pesos
    dele não são guardados. `fatia` e `de` repartem o trabalho entre
    processos: a fatia `i` de `n` mede um em cada `n` modelos.
    """
    medidos = {}
    alvos = [(m, v) for m in modelos_a_podar()
             for v in [DE_PARTIDA, *(x for x in (variantes_do_modelo(m) or ()) if VARIANTES[x]["familia"] not in FORA_DA_REGRA)]]
    for posicao, (modelo, variante) in enumerate(alvos):
        pasta, arquivo = pasta_do_modelo(modelo, variante), _arquivo_dos_tamanhos(modelo, variante)
        if arquivo.exists() and not refazer:
            medidos[modelo, variante] = _ler_json(arquivo)
            continue
        if posicao % de != fatia or not (pasta / "config.json").exists():
            continue   # é de outra fatia, ou a execução ainda não rodou
        sistema = SistemaEncoder.carregar(pasta, "cpu")
        temporaria = pasta_da_etapa() / "tamanhos" / f"_esparso_{modelo}__{variante}"
        try:
            esparso = poda_encoder.salvar_esparso(sistema, temporaria)
            medida = {"modelo": modelo, "variante": variante, "semente": SEMENTE_DO_CUSTO, "pasta": str(pasta.relative_to(RAIZ)),
                      **poda_encoder.contar(sistema.modelo), "parametros_por_parte": sistema.modelo.parametros_por_parte(),
                      "bytes": poda_encoder.tamanhos(pasta, esparso)}
        finally:
            shutil.rmtree(temporaria, ignore_errors=True)
        _gravar_json(arquivo, medida)
        medidos[modelo, variante] = medida
    return medidos


def medir_latencia(modelo: str, variantes: list[str], grupo: str, esparsa: bool = False, rodadas: int = RODADAS, refazer: bool = False) -> Path:
    """Mede a latência em CPU de um grupo de variantes do modelo (semente 0), no protocolo de `eficiencia.py`.

    Uma consulta por vez, 1 thread, do texto à resposta, com os pedidos de
    `dev`. O modelo de partida entra sempre no grupo, como referência, e os
    sistemas se alternam em `rodadas` rodadas de 1.024 consultas: se a máquina
    oscilar, a oscilação cai sobre todos. Com `esparsa=True`, cada variante é
    medida com a execução que só faz as contas dos pesos não nulos
    (`poda_encoder.execucao_esparsa`). Precisa da máquina quieta. Cada
    sistema do XLM-R ocupa cerca de 1 GB de memória: por isso a medida é
    feita em grupos. Grava `latencia/{modelo}__{grupo}.json` e `.npz`.
    """
    from especialista.experimentos import etapa8   # os relógios de "máquina quieta" da Etapa 8

    base = pasta_da_etapa() / "latencia" / f"{modelo}__{grupo}"
    if base.with_suffix(".json").exists() and not refazer:
        return base.with_suffix(".json")
    torch.set_num_threads(1)
    particoes, _ = massive.carregar_entrega()
    textos = list(particoes["dev"]["utt"])
    sistemas = {DE_PARTIDA: SistemaEncoder.carregar(pasta_do_modelo(modelo, DE_PARTIDA), "cpu")}
    for variante in variantes:
        sistema = SistemaEncoder.carregar(pasta_do_modelo(modelo, variante), "cpu")
        sistemas[variante] = poda_encoder.execucao_esparsa(sistema) if esparsa else sistema

    carga_antes, antes = etapa8.carga_da_maquina(), etapa8.relogios()
    tempos = {nome: [] for nome in sistemas}
    for rodada in range(rodadas):
        for nome, sistema in sistemas.items():
            tempos[nome].append(eficiencia.medir_latencia(sistema, textos, semente=rodada))
    carga_depois, depois = etapa8.carga_da_maquina(), etapa8.relogios()

    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez(base.with_suffix(".npz"), **{nome: np.stack(linhas) for nome, linhas in tempos.items()})
    _gravar_json(base.with_suffix(".json"), {
        "modelo": modelo, "grupo": grupo, "execucao": "esparsa (CSR)" if esparsa else "comum", "semente": SEMENTE_DO_CUSTO,
        "threads": 1, "rodadas": rodadas, "consultas_por_rodada": eficiencia.CONSULTAS,
        "p50_por_rodada_ms": {nome: [float(np.percentile(t, 50)) for t in linhas] for nome, linhas in tempos.items()},
        "resumo": {nome: eficiencia.resumir_latencia(np.concatenate(linhas)) for nome, linhas in tempos.items()},
        "nucleos_de_outros": etapa8.nucleos_de_outros(antes, depois), "carga_antes": carga_antes, "carga_depois": carga_depois,
        "medido_em": datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__, **eficiencia.ambiente(),
    })
    return base.with_suffix(".json")


def carregar_latencias(modelo: str) -> dict[str, dict]:
    """Tudo o que `medir_latencia` gravou para o modelo: uma entrada por grupo, com os tempos de cada consulta."""
    lidas = {}
    for arquivo in sorted((pasta_da_etapa() / "latencia").glob(f"{modelo}__*.json")):
        with np.load(arquivo.with_suffix(".npz")) as tempos:
            lidas[arquivo.stem.split("__", 1)[1]] = {**_ler_json(arquivo), "tempos": {nome: tempos[nome] for nome in tempos.files}}
    return lidas


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 10b: poda e reajuste do XLM-R.")
    argumentos.add_argument("--faltam", action="store_true", help="lista as execuções que ainda não rodaram, uma por linha")
    comandos = argumentos.add_subparsers(dest="comando")
    uma = comandos.add_parser("executar", help="poda e reajusta um modelo, numa variante, com uma semente")
    uma.add_argument("--modelo", required=True, help="aluno_a, aluno_b2_t2 ou teto")
    uma.add_argument("--variante", required=True, choices=sorted({**VARIANTES, **CURTAS}))
    uma.add_argument("--semente", type=int, required=True)
    uma.add_argument("--dispositivo", default=None, help="cuda ou cpu; o padrão é a placa, se houver")
    comandos.add_parser("faltam", help="o mesmo que --faltam")
    comandos.add_parser("situacao", help="em que ponto a etapa está")
    comandos.add_parser("onda", help="roda tudo o que falta, uma execução por vez")
    da_limpeza = comandos.add_parser("limpar", help="apaga do cache os pesos das execuções que não podem seguir")
    da_limpeza.add_argument("--depois-da-decisao", action="store_true",
                            help="com a entrega gravada: apaga os pesos de tudo o que não segue, inclusive os da semente 0")
    dos_tamanhos = comandos.add_parser("tamanhos", help="mede os tamanhos em disco (não usa a placa)")
    dos_tamanhos.add_argument("--refazer", action="store_true")
    dos_tamanhos.add_argument("--fatia", type=int, default=0, help="com --de: mede só um em cada `de` modelos, para rodar em paralelo")
    dos_tamanhos.add_argument("--de", type=int, default=1)
    da_latencia = comandos.add_parser("latencia", help="mede a latência em CPU de um grupo de variantes (precisa da máquina quieta)")
    da_latencia.add_argument("--modelo", required=True)
    da_latencia.add_argument("--variantes", nargs="+", required=True)
    da_latencia.add_argument("--grupo", required=True, help="o nome do grupo, para o arquivo")
    da_latencia.add_argument("--esparsa", action="store_true", help="mede com a execução que aproveita os zeros")
    da_latencia.add_argument("--rodadas", type=int, default=RODADAS)
    da_latencia.add_argument("--refazer", action="store_true")
    a = argumentos.parse_args()

    if a.faltam or a.comando == "faltam":
        for modelo, variante, semente in faltam():
            print(f"executar --modelo {modelo} --variante {variante} --semente {semente}")
    elif a.comando == "executar":
        if a.modelo not in modelos_a_podar():
            argumentos.error(f"modelo desconhecido: {a.modelo}. Os que existem: {', '.join(modelos_a_podar())}")
        torch.set_num_threads(2)
        meta = _ler_json(executar(a.modelo, a.variante, a.semente, a.dispositivo) / "meta.json")
        antes = "" if meta["dev_antes"] is None else f"exact match antes {meta['dev_antes'][CRITERIO]:.4f}, "
        vram = f", pico de {meta['pico_de_vram_gb']:.1f} GB na placa" if meta.get("pico_de_vram_gb") else ""
        print(f"[{datetime.now():%H:%M:%S}]", a.modelo, a.variante, a.semente, f"época {meta['melhor_epoca']} de {meta['receita']['epocas']},",
              f"{antes}depois {meta['dev'][CRITERIO]:.4f} (no dev de parada);", f"esparsidade {meta['contagem']['esparsidade']:.4f},",
              f"não nulos {meta['contagem']['nao_nulos']}, {meta['bytes_em_disco'] / 1e6:.0f} MB;",
              f"{meta['segundos']:.0f}s ({meta['segundos_por_epoca']:.1f}s por época{vram})", flush=True)
    elif a.comando == "situacao":
        print(situacao())
    elif a.comando == "onda":
        sys.exit(0 if onda() else 1)
    elif a.comando == "limpar":
        for pasta in (limpar_depois_da_decisao() if a.depois_da_decisao else limpar_pesos()):
            print("apagado:", pasta.relative_to(RAIZ))
    elif a.comando == "tamanhos":
        for (modelo, variante), medida in medir_tamanhos(a.refazer, a.fatia, a.de).items():
            print(modelo, variante, {formato: round(valor / 1e6, 1) for formato, valor in medida["bytes"]["sistema"].items()})
    elif a.comando == "latencia":
        medido = _ler_json(medir_latencia(a.modelo, a.variantes, a.grupo, a.esparsa, a.rodadas, a.refazer))
        print(json.dumps({nome: resumo["latencia_p50_ms"] for nome, resumo in medido["resumo"].items()}, ensure_ascii=False, indent=2))
    else:
        argumentos.error("informe um comando (executar, onda, situacao, limpar, tamanhos, latencia) ou --faltam")


if __name__ == "__main__":
    _linha_de_comando()
