"""Etapa 12: investigação dos três achados que as etapas mediram e não explicaram.

**Pergunta.** Por que acontecem os três efeitos que as etapas mediram e não
explicaram, e o que cada um ensina para o artigo?

É uma investigação **depois dos resultados**: tudo aqui é diagnóstico e
exploratório. Nenhuma decisão das etapas anteriores muda, e os modelos finais
continuam os mesmos. O plano (hipóteses, testes e o que contaria como
confirmação ou refutação) foi gravado antes de qualquer execução em
`data/execucoes/12_investigacao_dos_achados/plano.json`. Explora-se em `dev`;
o teste entra no fim, uma leitura por achado, a fixada no plano.

Os três achados:

1. **A poda do XLM-R perde cerca de 1 ponto nos alunos do cenário B** e não
   no Aluno A nem no teto (Etapas 10b e 11c). Testes: o mesmo reajuste sem
   corte nenhum (controle), a curva por época contra os rótulos humanos e os
   do professor, a cópia dos erros do professor, um reajuste com taxa 10
   vezes menor e os pedidos que o corte afeta.
2. **A quantização estática do XLM-R desaba numa das três sementes** (Etapa
   10c). Testes: a faixa das ativações por tensor e por semente, a
   sensibilidade por grupo de tensores, outras calibrações e deixar os
   piores tensores em 32 bits.
3. **A quantização dinâmica tira cerca de 0,4 ponto de acurácia de intenção
   do modelo final do XLM-R** (Etapa 10c). Testes: que parte da rede, que
   pedidos trocam de resposta e para que lado, e o que acontece com a cabeça
   de intenção em 32 bits, com um degrau por canal ou por linha da tabela.

As execuções novas reaproveitam as funções das etapas de origem sem mudar uma
linha delas (`etapa10b.executar`, `quantizacao.quantizar_rede`), como a
Etapa 11c: só o modelo de partida, a receita e a pasta de gravação são
outros. Tudo o que é novo fica em `data/execucoes/12_investigacao_dos_achados/`
e `entregas/12_investigacao_dos_achados/`.

    uv run --no-sync python -m especialista.experimentos.etapa12 reajustes      # achado 1: os reajustes, na placa, um por vez
    uv run --no-sync python -m especialista.experimentos.etapa12 treino         # achado 1: respostas no treino, antes e depois
    uv run --no-sync python -m especialista.experimentos.etapa12 dinamica       # achado 3: as alternativas da dinâmica, em CPU
    uv run --no-sync python -m especialista.experimentos.etapa12 estatica       # achado 2: faixas, sensibilidade e calibrações, em CPU
    uv run --no-sync python -m especialista.experimentos.etapa12 contas         # tudo o que vai para a entrega
    uv run --no-sync python -m especialista.experimentos.etapa12 teste          # a leitura no teste fixada no plano
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, EXECUCOES, RAIZ

ETAPA = "12_investigacao_dos_achados"
SEMENTES = (0, 1, 2)
REAMOSTRAS = 2000
TRES = ("intent_acc", "slot_f1", "exact_match")
PROFESSOR_NOVO = "qwen3_14b__fp16__colab_a100"
B1 = f"xlmr__aluno_b1__{PROFESSOR_NOVO}"
B2 = f"xlmr__aluno_b2_t8__{PROFESSOR_NOVO}"
B2_ANTIGO = "aluno_b2_t2"
ALUNO_A = "aluno_a"
TETO = "teto"
NOMES = {B1: "Aluno B1 · Qwen3-14B", B2: "Aluno B2 · Qwen3-14B", B2_ANTIGO: "Aluno B2 · Qwen3-8B", ALUNO_A: "Aluno A · Claude Haiku 5.5", TETO: "Teto (rótulos humanos)"}
TAXA_MENOR = 5e-6   # a taxa da receita (5e-5) dividida por 10


def pasta_da_etapa() -> Path:
    return EXECUCOES / ETAPA


def pasta_da_entrega() -> Path:
    return ENTREGAS / ETAPA


def _json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def _gravar_json(caminho: Path, conteudo) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")


def plano() -> dict:
    """O plano gravado antes de qualquer execução. Sem ele nada roda."""
    arquivo = pasta_da_etapa() / "plano.json"
    if not arquivo.exists():
        raise FileNotFoundError("O plano da Etapa 12 ainda não foi gravado: ele vem antes de qualquer execução.")
    return _json(arquivo)


@lru_cache(maxsize=None)
def particoes() -> dict[str, pd.DataFrame]:
    """As partições com os rótulos humanos (Etapa 1), lidas uma vez por processo."""
    from especialista.dados import massive
    return massive.carregar_entrega()[0]


@lru_cache(maxsize=None)
def gabarito(particao: str) -> tuple[Previsao, ...]:
    return tuple(ouro(particoes()[particao]))


def ler_previsoes(arquivo: Path, particao: str) -> list[Previsao]:
    """Respostas gravadas, conferindo que estão na ordem da partição."""
    ids, previsoes = carregar_previsoes(arquivo)
    if list(ids) != list(particoes()[particao]["id"]):
        raise ValueError(f"{arquivo} não está na ordem de {particao}.")
    return previsoes


def comparar(particao: str, a: list[list[Previsao]], b: list[list[Previsao]], gab: list[Previsao] | None = None) -> dict[str, dict]:
    """B − A, média de 3 sementes, bootstrap pareado por enunciado (2.000 sorteios, semente 0), em pontos percentuais."""
    gab = list(gab if gab is not None else gabarito(particao))
    tabela = qualidade.comparar_grupos(gab, a, b, n=REAMOSTRAS, semente=0)
    return {m: {"a": 100 * tabela.loc[m, "a"], "b": 100 * tabela.loc[m, "b"], "diferenca": 100 * tabela.loc[m, "diferenca"],
                "ic_min": 100 * tabela.loc[m, "ic_min"], "ic_max": 100 * tabela.loc[m, "ic_max"], "distinguivel": bool(tabela.loc[m, "distinguivel"])}
            for m in TRES}


def media(particao: str, grupo: list[list[Previsao]], gab: list[Previsao] | None = None) -> dict[str, float]:
    gab = list(gab if gab is not None else gabarito(particao))
    return {m: 100 * float(np.mean([qualidade.avaliar(gab, p)[m] for p in grupo])) for m in TRES}


def diferenca_das_diferencas(particao: str, a0, a1, b0, b1, metrica: str = "exact_match") -> dict:
    """(B1 − B0) − (A1 − A0): o efeito de um passo num modelo menos o efeito do mesmo passo em outro, nos mesmos pedidos.

    Cada grupo é a lista de respostas das 3 sementes. Os sorteios do bootstrap são os mesmos para os quatro grupos (semente 0), então a
    diferença fica pareada por enunciado.
    """
    gab = list(gabarito(particao))
    amostra = lambda grupo: sum(qualidade.reamostrar(gab, p, REAMOSTRAS, 0)[metrica] for p in grupo) / len(grupo)
    dd = (amostra(b1) - amostra(b0)) - (amostra(a1) - amostra(a0))
    ponto = (media(particao, b1)[metrica] - media(particao, b0)[metrica]) - (media(particao, a1)[metrica] - media(particao, a0)[metrica])
    return {"diferenca": ponto, "ic_min": 100 * dd.quantile(0.025), "ic_max": 100 * dd.quantile(0.975),
            "distinguivel": bool(dd.quantile(0.025) > 0 or dd.quantile(0.975) < 0)}


def texto_ic(c: dict, casas: int = 1) -> str:
    """'−1,0 (−1,8 a −0,3)', no estilo do projeto."""
    f = lambda x: f"{x:+.{casas}f}".replace("-", "−").replace(".", ",")
    return f"{f(c['diferenca'])} ({f(c['ic_min'])} a {f(c['ic_max'])})"


# ==========================================================================
# Achado 1: a poda do XLM-R nos alunos do cenário B
# ==========================================================================

# Os reajustes novos. Todos usam `etapa10b.executar` sem mudança: a variante é uma das da Etapa 10b
# ("sem_poda" é o controle dela: o mesmo reajuste, sem tirar nada), e o que muda é a receita e a pasta.
REAJUSTES: dict[str, dict] = {
    "sem_corte": {"variante": "sem_poda", "taxa": None, "alunos": (B1, B2),
                  "o_que_e": "T1: o reajuste de 10 épocas da receita, sem cortar nada (o controle da Etapa 10b)"},
    "refeito": {"variante": "vocabulario_treino", "taxa": None, "alunos": (B1, B2, ALUNO_A),
                "o_que_e": "T2: o corte e o reajuste das Etapas 10b e 11c, refeitos com as respostas em dev gravadas a cada época"},
    "taxa_menor": {"variante": "vocabulario_treino", "taxa": TAXA_MENOR, "alunos": (B1, B2),
                   "o_que_e": "T4: o corte e o reajuste de 10 épocas com a taxa 10 vezes menor (5e-6), mesma agenda"},
}
ORDEM_DOS_REAJUSTES = [(tipo, aluno) for tipo in ("sem_corte", "refeito", "taxa_menor") for aluno in REAJUSTES[tipo]["alunos"] if aluno != ALUNO_A] + [("refeito", ALUNO_A)]


def ficha_do_aluno(aluno: str) -> dict:
    """A ficha que `etapa10b.executar` espera (como `etapa10b.modelos_a_podar`), e os dados do aluno (`dados`)."""
    from especialista.experimentos import etapa10b, etapa11b, etapa11c
    if aluno in (ALUNO_A, TETO, B2_ANTIGO):
        ficha = dict(etapa10b.modelos_a_podar()[aluno])
        return {"ficha": ficha, "dados": lambda: etapa10b.dados_do_modelo(aluno)}
    a = etapa11c.alunos(PROFESSOR_NOVO)[aluno]
    ficha = {"nome": a["nome"], "curto": a["curto"], "variante_da_10a": a["aluno"], "pastas": a["pastas"], "config": a["config"], "receita": a["receita"],
             "sinal": a["sinal"], "previsoes": None, "grade": "do_principal", "principal": False}
    return {"ficha": ficha, "dados": lambda: etapa11b.dados_do_aluno("xlmr", a["aluno"], PROFESSOR_NOVO)}


class Gravador:
    """Faz as vezes do módulo `qualidade` dentro de `poda_encoder.reajustar`: guarda as respostas em dev de cada época e devolve as mesmas métricas."""

    def __init__(self):
        self.respostas: list[list[Previsao]] = []

    def avaliar(self, rotulos, previsoes):
        self.respostas.append(list(previsoes))
        return qualidade.avaliar(rotulos, previsoes)


@contextmanager
def emprestar(modulo, **trocas):
    antes = {nome: getattr(modulo, nome) for nome in trocas}
    for nome, valor in trocas.items():
        setattr(modulo, nome, valor)
    try:
        yield modulo
    finally:
        for nome, valor in antes.items():
            setattr(modulo, nome, valor)


def pasta_do_reajuste(tipo: str, aluno: str, semente: int) -> Path:
    return pasta_da_etapa() / "reajustes" / tipo / aluno / REAJUSTES[tipo]["variante"] / f"semente_{semente}"


def reajustar(tipo: str, aluno: str, semente: int, dispositivo: str | None = None) -> Path:
    """Um reajuste do achado 1, com `etapa10b.executar` sem mudança. Grava as respostas em dev de cada época em `epocas/`."""
    from especialista.compressao import poda_encoder
    from especialista.experimentos import etapa10b

    plano()
    definicao = REAJUSTES[tipo]
    pasta = pasta_do_reajuste(tipo, aluno, semente)
    if (pasta / "epocas" / "pronto.json").exists():
        return pasta
    lido = ficha_do_aluno(aluno)
    ficha = lido["ficha"]
    if definicao["taxa"] is not None:
        ficha["receita"] = {**ficha["receita"], "taxa": definicao["taxa"]}
    dados = lido["dados"]()
    gravador = Gravador()
    trocas = {"ETAPA": f"{ETAPA}/reajustes/{tipo}", "modelos_a_podar": lambda: {aluno: ficha}, "dados_do_modelo": lambda modelo: dados}
    with emprestar(etapa10b, **trocas), emprestar(poda_encoder, qualidade=gravador):
        destino = etapa10b.pasta_da_execucao(aluno, definicao["variante"], semente)
        if destino != pasta:
            raise RuntimeError(f"O reajuste iria para {destino}, e não para {pasta}.")
        etapa10b.executar(aluno, definicao["variante"], semente, dispositivo)
    epocas = pasta / "epocas"
    epocas.mkdir(exist_ok=True)
    ids = particoes()["dev"]["id"]
    for epoca, respostas in enumerate(gravador.respostas):
        salvar_previsoes(epocas / f"dev_epoca_{epoca:02d}.parquet", ids, respostas)
    _gravar_json(epocas / "pronto.json", {"tipo": tipo, "aluno": aluno, "semente": semente, "epocas_gravadas": len(gravador.respostas),
                                          "receita": ficha["receita"], "o_que_e": definicao["o_que_e"]})
    return pasta


def rodar_reajustes() -> None:
    """Todos os reajustes do achado 1, um por vez, na ordem do plano."""
    import torch
    for tipo, aluno in ORDEM_DOS_REAJUSTES:
        for semente in SEMENTES:
            inicio = time.perf_counter()
            reajustar(tipo, aluno, semente)
            print(f"{time.strftime('%H:%M:%S')} {tipo} {aluno} semente {semente}: {time.perf_counter() - inicio:.0f} s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()


def pasta_10b(modelo: str, variante: str, semente: int) -> Path:
    return EXECUCOES / "10b_encoder_poda" / modelo / variante / f"semente_{semente}"


def pasta_11c(aluno: str, semente: int) -> Path:
    return EXECUCOES / "11c_cenario_b_ate_o_fim" / "poda" / aluno / "vocabulario_treino" / f"semente_{semente}"


def modelos_do_treino(aluno: str, semente: int) -> dict[str, Path]:
    """Os modelos em PyTorch de cada momento, para responder no treino (T3)."""
    partida = ficha_do_aluno(aluno)["ficha"]["pastas"][semente]
    if aluno == ALUNO_A:
        return {"partida": partida, "corte_e_reajuste": pasta_10b(ALUNO_A, "vocabulario_treino", semente) / "modelo"}
    return {"partida": partida, "corte_e_reajuste": pasta_11c(aluno, semente) / "modelo",
            "sem_corte_e_reajuste": pasta_do_reajuste("sem_corte", aluno, semente) / "modelo"}


def rodar_treino() -> None:
    """T3 do achado 1: as respostas no treino de cada momento (partida, corte e reajuste, reajuste sem corte), na placa."""
    import torch

    from especialista.modelos.encoder import SistemaEncoder
    plano()
    textos = list(particoes()["train"]["utt"])
    for aluno in (B1, B2, ALUNO_A):
        for semente in SEMENTES:
            for momento, pasta in modelos_do_treino(aluno, semente).items():
                arquivo = pasta_da_etapa() / "treino" / aluno / momento / f"semente_{semente}.parquet"
                if arquivo.exists():
                    continue
                sistema = SistemaEncoder.carregar(pasta, "cuda")
                arquivo.parent.mkdir(parents=True, exist_ok=True)
                salvar_previsoes(arquivo, particoes()["train"]["id"], sistema.prever(textos))
                del sistema
                gc.collect()
                torch.cuda.empty_cache()
                print(f"{time.strftime('%H:%M:%S')} treino {aluno} {momento} {semente}", flush=True)


def _grupo(arquivos: list[Path], particao: str = "dev") -> list[list[Previsao]]:
    return [ler_previsoes(a, particao) for a in arquivos]


def momentos_em_dev(aluno: str) -> dict[str, list[list[Previsao]]]:
    """As respostas em dev de cada momento do achado 1, 3 sementes, das etapas de origem e desta."""
    s3 = SEMENTES
    if aluno in (B1, B2):
        p = ENTREGAS / "11c_cenario_b_ate_o_fim" / "previsoes"
        saida = {
            "partida": _grupo([p / f"{aluno}__sem_poda__pytorch_semente_{s}_dev.parquet" for s in s3]),
            "so_corte": _grupo([p / f"{aluno}__vocabulario_treino__antes_do_reajuste_semente_{s}_dev.parquet" for s in s3]),
            "corte_e_reajuste": _grupo([p / f"{aluno}__vocabulario_treino_semente_{s}_dev.parquet" for s in s3]),
        }
    else:
        saida = {
            "partida": _grupo([ENTREGAS / "10a_encoder" / "previsoes" / f"{aluno}_semente_{s}_dev.parquet" for s in s3]),
            "so_corte": _grupo([pasta_10b(aluno, "vocabulario_treino", s) / "previsoes_dev_antes.parquet" for s in s3]),
            "corte_e_reajuste": _grupo([pasta_10b(aluno, "vocabulario_treino", s) / "previsoes_dev.parquet" for s in s3]),
        }
        if aluno != B2_ANTIGO:
            saida["sem_corte_e_reajuste"] = _grupo([pasta_10b(aluno, "sem_poda", s) / "previsoes_dev.parquet" for s in s3])
    for tipo, d in REAJUSTES.items():
        if aluno in d["alunos"] and all((pasta_do_reajuste(tipo, aluno, s) / "epocas" / "pronto.json").exists() for s in s3):
            nome = {"sem_corte": "sem_corte_e_reajuste", "refeito": "corte_e_reajuste_refeito", "taxa_menor": "corte_e_reajuste_taxa_menor"}[tipo]
            saida[nome] = _grupo([pasta_do_reajuste(tipo, aluno, s) / "previsoes_dev.parquet" for s in s3])
    return saida


@lru_cache(maxsize=None)
def rotulos_do_professor(aluno: str, particao: str) -> tuple[Previsao, ...]:
    """Os rótulos do professor do aluno em dev ou no treino, na ordem da partição."""
    dados = ficha_do_aluno(aluno)["dados"]()
    tabela = dados["dev" if particao == "dev" else "treino"].set_index("id").loc[particoes()[particao]["id"]]
    return tuple(ouro(tabela.reset_index()))


def _iguais(a: Previsao, b: Previsao) -> bool:
    return a.intent == b.intent and set(a.slots) == set(b.slots)


def copia_do_professor(gab: list[Previsao], prof: list[Previsao], grupo: list[list[Previsao]]) -> dict[str, float]:
    """Nos pedidos em que o professor erra (exact match contra o humano): em que fração o aluno repete a resposta dele, e em que fração acerta."""
    erra = [k for k, (g, p) in enumerate(zip(gab, prof)) if not _iguais(g, p)]
    copia = np.mean([np.mean([_iguais(prev[k], prof[k]) for k in erra]) for prev in grupo])
    acerta = np.mean([np.mean([_iguais(prev[k], gab[k]) for k in erra]) for prev in grupo])
    concorda = np.mean([np.mean([_iguais(a, b) for a, b in zip(prev, prof)]) for prev in grupo])
    return {"pedidos_em_que_o_professor_erra": len(erra), "copia_o_erro": 100 * copia, "acerta_onde_o_professor_erra": 100 * acerta,
            "concorda_com_o_professor": 100 * concorda}


def curva_por_epoca(tipo: str, aluno: str) -> pd.DataFrame:
    """T2: exact match em dev, contra os humanos e contra o professor, em cada época (média de 3 sementes)."""
    gab, prof = list(gabarito("dev")), list(rotulos_do_professor(aluno, "dev"))
    linhas = []
    for s in SEMENTES:
        for arquivo in sorted((pasta_do_reajuste(tipo, aluno, s) / "epocas").glob("dev_epoca_*.parquet")):
            prev = ler_previsoes(arquivo, "dev")
            linhas.append({"tipo": tipo, "aluno": aluno, "semente": s, "epoca": int(arquivo.stem.split("_")[-1]),
                           **{f"humano_{m}": 100 * v for m, v in qualidade.avaliar(gab, prev).items() if m in TRES},
                           **{f"professor_{m}": 100 * v for m, v in qualidade.avaliar(prof, prev).items() if m in TRES},
                           **{f"professor_erra_{k}": v for k, v in copia_do_professor(gab, prof, [prev]).items() if k != "pedidos_em_que_o_professor_erra"}})
    return pd.DataFrame(linhas)


def afetados_pelo_corte(aluno: str = ALUNO_A, particao: str = "dev") -> np.ndarray:
    """T5: os pedidos que têm pelo menos um subtoken fora das linhas que o corte 'vocabulário do treino' deixa (as do texto do treino)."""
    from especialista.modelos.encoder import SistemaEncoder
    sistema = SistemaEncoder.carregar(ficha_do_aluno(aluno)["ficha"]["pastas"][0], "cpu")
    ficam = set(sistema.alinhador.ids_usados(particoes()["train"]["tokens"]).tolist())
    opcoes = {"is_split_into_words": True, "truncation": True, "max_length": sistema.modelo.config.max_subtokens, "add_special_tokens": False}
    ids = sistema.tokenizador([list(t) for t in particoes()[particao]["tokens"]], **opcoes)["input_ids"]
    return np.array([any(i not in ficam for i in linha) for linha in ids])


def contas_achado_1() -> dict:
    """Todas as contas do achado 1, em dev."""
    gab = list(gabarito("dev"))
    saida: dict = {"comparacoes": [], "professor": [], "curvas": [], "corte": {}}
    pares = (("so_corte", "partida"), ("corte_e_reajuste", "partida"), ("sem_corte_e_reajuste", "partida"), ("corte_e_reajuste", "sem_corte_e_reajuste"),
             ("corte_e_reajuste", "so_corte"), ("corte_e_reajuste_refeito", "partida"), ("corte_e_reajuste_taxa_menor", "partida"),
             ("corte_e_reajuste_taxa_menor", "corte_e_reajuste"), ("corte_e_reajuste_taxa_menor", "so_corte"))
    afetados = afetados_pelo_corte()
    saida["corte"] = {"pedidos_de_dev": len(afetados), "afetados": int(afetados.sum())}
    for aluno in (B1, B2, B2_ANTIGO, ALUNO_A, TETO):
        momentos = momentos_em_dev(aluno)
        for b, a in pares:
            if a in momentos and b in momentos:
                for m, c in comparar("dev", momentos[a], momentos[b]).items():
                    saida["comparacoes"].append({**c, "aluno": aluno, "depois": b, "antes": a, "metrica": m})
        if aluno != TETO:
            prof = list(rotulos_do_professor(aluno, "dev"))
            saida["professor"].append({"aluno": aluno, "momento": "professor", **{m: 100 * v for m, v in qualidade.avaliar(gab, prof).items() if m in TRES}})
            for nome, grupo in momentos.items():
                saida["professor"].append({"aluno": aluno, "momento": nome, **copia_do_professor(gab, prof, grupo),
                                           **{f"contra_o_professor_{m}": 100 * np.mean([qualidade.avaliar(prof, p)[m] for p in grupo]) for m in TRES}})
        # T5: o corte sozinho, nos pedidos afetados e nos outros
        for nome, mascara in (("afetados", afetados), ("nao_afetados", ~afetados)):
            idx = np.flatnonzero(mascara)
            sub = lambda grupo: [[p[k] for k in idx] for p in grupo]
            for b, a in (("so_corte", "partida"), ("corte_e_reajuste", "so_corte")):
                c = comparar("dev", sub(momentos[a]), sub(momentos[b]), [gab[k] for k in idx])["exact_match"]
                saida["comparacoes"].append({**c, "aluno": aluno, "depois": b, "antes": a, "metrica": "exact_match", "subconjunto": nome, "pedidos": len(idx)})
    for tipo, d in REAJUSTES.items():
        for aluno in d["alunos"]:
            if all((pasta_do_reajuste(tipo, aluno, s) / "epocas" / "pronto.json").exists() for s in SEMENTES):
                saida["curvas"].append(curva_por_epoca(tipo, aluno))
    saida["curvas"] = pd.concat(saida["curvas"]) if saida["curvas"] else pd.DataFrame()
    saida["comparacoes"] = pd.DataFrame(saida["comparacoes"])
    saida["professor"] = pd.DataFrame(saida["professor"])
    saida["treino"] = contas_do_treino()
    return saida


def contas_do_treino() -> pd.DataFrame:
    """T3 no treino: o acerto contra o rótulo do professor, separado em pedidos 'limpos' (o professor acerta o humano) e 'ruidosos' (erra)."""
    gab = list(gabarito("train"))
    linhas = []
    for aluno in (B1, B2, ALUNO_A):
        prof = list(rotulos_do_professor(aluno, "train"))
        ruido = np.array([not _iguais(g, p) for g, p in zip(gab, prof)])
        for s in SEMENTES:
            for momento in modelos_do_treino(aluno, s):
                arquivo = pasta_da_etapa() / "treino" / aluno / momento / f"semente_{s}.parquet"
                if not arquivo.exists():
                    continue
                prev = ler_previsoes(arquivo, "train")
                igual_prof = np.array([_iguais(a, b) for a, b in zip(prev, prof)])
                igual_hum = np.array([_iguais(a, b) for a, b in zip(prev, gab)])
                linhas.append({"aluno": aluno, "semente": s, "momento": momento, "ruido_do_professor": 100 * ruido.mean(),
                               "igual_ao_professor_nos_limpos": 100 * igual_prof[~ruido].mean(), "igual_ao_professor_nos_ruidosos": 100 * igual_prof[ruido].mean(),
                               "igual_ao_humano_nos_ruidosos": 100 * igual_hum[ruido].mean()})
    return pd.DataFrame(linhas)


# ==========================================================================
# Ferramentas comuns aos achados 2 e 3: exportar, quantizar e responder no ONNX Runtime
# ==========================================================================

THREADS = 4
MODELOS_DA_10C = {   # os modelos em PyTorch que a Etapa 10c quantizou, por semente
    (ALUNO_A, "vocabulario_treino"): "entregas/10b_encoder_poda/modelos/aluno_a/vocabulario_treino/semente_{s}",
    (ALUNO_A, "sem_poda"): "entregas/10a_encoder/modelos/aluno_a/semente_{s}",
    (ALUNO_A, "gradual_70"): "entregas/10b_encoder_poda/modelos/aluno_a/gradual_70/semente_{s}",
    (TETO, "sem_poda"): "entregas/10a_encoder/modelos/teto/semente_{s}",
}
PREVISOES_DA_10C = ENTREGAS / "10c_encoder_quantizacao" / "previsoes"


def previsoes_da_10c(modelo: str, variante: str, forma: str, particao: str = "dev") -> list[list[Previsao]]:
    return [ler_previsoes(PREVISOES_DA_10C / f"{modelo}__{variante}__{forma}_semente_{s}_{particao}.parquet", particao) for s in SEMENTES]


def fp32_exportado(modelo: str, variante: str, semente: int) -> Path:
    """O sistema ONNX de 32 bits do modelo, exportado de novo do PyTorch (a 10c só guardou o da semente 0). Confere as respostas em dev com as da 10c."""
    from especialista.compressao import exportacao_encoder
    from especialista.compressao.sistema_encoder_onnx import SistemaEncoderOnnx
    from especialista.modelos.encoder import SistemaEncoder

    destino = pasta_da_etapa() / "onnx" / modelo / variante / f"semente_{semente}" / "fp32"
    if (destino / "conferido.json").exists():
        return destino
    original = SistemaEncoder.carregar(RAIZ / MODELOS_DA_10C[(modelo, variante)].format(s=semente), "cpu")
    exportacao_encoder.exportar(original, destino, origem=MODELOS_DA_10C[(modelo, variante)].format(s=semente))
    del original
    gc.collect()
    respostas = SistemaEncoderOnnx.carregar(destino, threads=THREADS).prever(list(particoes()["dev"]["utt"]))
    da_10c = previsoes_da_10c(modelo, variante, "fp32")[semente]
    iguais = sum(a == b for a, b in zip(respostas, da_10c))
    _gravar_json(destino / "conferido.json", {"iguais_a_10c_em_dev": iguais, "pedidos": len(da_10c)})
    if iguais != len(da_10c):
        print(f"AVISO: o ONNX de 32 bits refeito de {modelo}/{variante}/{semente} difere da 10c em {len(da_10c) - iguais} pedidos de dev")
    return destino


def responder(pasta: Path, particao: str = "dev", threads: int = THREADS) -> tuple[list[Previsao], np.ndarray]:
    """As respostas de um sistema ONNX, um pedido por vez (como na 10c), e as pontuações de intenção de cada pedido."""
    from especialista.compressao.sistema_encoder_onnx import SistemaEncoderOnnx

    sistema = SistemaEncoderOnnx.carregar(pasta, threads=threads)
    previsoes, pontuacoes = [], []
    for texto in particoes()[particao]["utt"]:
        palavras = texto.split()
        entradas, real = sistema.alinhador.alinhar([palavras])
        intencoes, emissoes = sistema.sessao.run(["intencoes", "emissoes"], entradas)
        previsoes.extend(sistema.montar(intencoes, emissoes, real, [len(palavras)]))
        pontuacoes.append(intencoes[0])
    return previsoes, np.stack(pontuacoes)


def gravar_respostas(pasta: Path, particao: str, previsoes: list[Previsao], pontuacoes: np.ndarray | None = None) -> None:
    salvar_previsoes(pasta / f"previsoes_{particao}.parquet", particoes()[particao]["id"], previsoes)
    if pontuacoes is not None:
        np.save(pasta / f"intencoes_{particao}.npy", pontuacoes.astype(np.float32))


def ler_respostas(pasta: Path, particao: str = "dev") -> list[Previsao]:
    return ler_previsoes(pasta / f"previsoes_{particao}.parquet", particao)


def bytes_da_rede(pasta: Path) -> int:
    return (pasta / "rede.onnx").stat().st_size


# ==========================================================================
# Achado 3: a quantização dinâmica e a acurácia de intenção
# ==========================================================================

# As alternativas de T4. Todas são a quantização dinâmica da 10c (`quantizacao.quantizar_rede`, forma "dinamica"),
# com uma mudança cada uma. A forma "so_matrizes" (tabelas em 32 bits) já tem respostas gravadas na 10c.
ALTERNATIVAS_DINAMICAS: dict[str, str] = {
    "dinamica": "a quantização dinâmica da 10c, refeita (confere que se reproduz)",
    "cabeca_de_intencao_em_32_bits": "a dinâmica, com a multiplicação da cabeça de intenção fora da quantização",
    "degrau_por_canal": "a dinâmica, com um degrau por coluna em cada matriz (per_channel=True)",
    "tabela_por_linha": "a dinâmica, com a tabela de subtokens em 8 bits com um degrau e um ponto zero por linha (grafo editado)",
}


def _opcoes_dinamicas(operacoes: list[str], por_canal: bool = False) -> dict:
    """As opções de `quantizacao.quantizar_rede` para a forma dinâmica, com o que T4 muda."""
    from onnxruntime.quantization import QuantType
    return {"op_types_to_quantize": ["Gather", "MatMul"], "nodes_to_quantize": operacoes, "per_channel": por_canal, "reduce_range": False,
            "weight_type": QuantType.QInt8, "extra_options": {"MatMulConstBOnly": True, "WeightSymmetric": True}}


def tabela_por_linha(rede, nome_da_tabela: str):
    """Troca a busca na tabela de subtokens (Gather sobre a tabela de 32 bits) por: linhas em 8 bits sem sinal, com um degrau e um ponto zero por linha.

    valor = degrau[linha] · (q − zero[linha]). Cada linha usa a faixa do seu menor ao seu maior valor (como a 10c faz com a tabela inteira).
    """
    from onnx import helper, numpy_helper

    tabela = next(t for t in rede.graph.initializer if t.name == nome_da_tabela)
    w = numpy_helper.to_array(tabela).astype(np.float32)
    menor, maior = np.minimum(w.min(axis=1), 0.0), np.maximum(w.max(axis=1), 0.0)
    degrau = np.where(maior > menor, (maior - menor) / 255.0, 1.0).astype(np.float32)
    zero = np.clip(np.round(-menor / degrau), 0, 255).astype(np.float32)
    q = np.clip(np.round(w / degrau[:, None] + zero[:, None]), 0, 255).astype(np.uint8)
    rede.graph.initializer.remove(tabela)
    rede.graph.initializer.extend([numpy_helper.from_array(q, f"{nome_da_tabela}.q"), numpy_helper.from_array(degrau[:, None], f"{nome_da_tabela}.degrau"),
                                   numpy_helper.from_array(zero[:, None], f"{nome_da_tabela}.zero")])
    no = next(n for n in rede.graph.node if n.op_type == "Gather" and n.input[0] == nome_da_tabela)
    ids, saida, posicao = no.input[1], no.output[0], list(rede.graph.node).index(no)
    novos = [
        helper.make_node("Gather", [f"{nome_da_tabela}.q", ids], [f"{saida}.q"], name=f"{saida}.q", axis=0),
        helper.make_node("Cast", [f"{saida}.q"], [f"{saida}.q_real"], name=f"{saida}.q_real", to=1),
        helper.make_node("Gather", [f"{nome_da_tabela}.zero", ids], [f"{saida}.zero"], name=f"{saida}.zero", axis=0),
        helper.make_node("Gather", [f"{nome_da_tabela}.degrau", ids], [f"{saida}.degrau"], name=f"{saida}.degrau", axis=0),
        helper.make_node("Sub", [f"{saida}.q_real", f"{saida}.zero"], [f"{saida}.centrado"], name=f"{saida}.centrado"),
        helper.make_node("Mul", [f"{saida}.centrado", f"{saida}.degrau"], [saida], name=saida),
    ]
    rede.graph.node.remove(no)
    for i, n in enumerate(novos):
        rede.graph.node.insert(posicao + i, n)
    return rede


def quantizar_alternativa_dinamica(fp32: Path, destino: Path, alternativa: str) -> Path:
    """Grava em `destino` o sistema com a alternativa dada de T4 do achado 3."""
    from onnxruntime.quantization import quantize_dynamic

    from especialista.compressao import exportacao, exportacao_encoder, quantizacao

    if (destino / exportacao.REDE).exists():
        return destino
    rede = exportacao.ler_rede(fp32)
    por_parte = quantizacao.operacoes_com_pesos(rede)
    operacoes = por_parte["tabela"] + por_parte["matrizes"]
    if alternativa == "cabeca_de_intencao_em_32_bits":
        operacoes = [o for o in operacoes if o != "intencao.produto"]
    if alternativa == "tabela_por_linha":
        rede = tabela_por_linha(rede, exportacao_encoder.TABELA_DE_SUBTOKENS)
        operacoes = [o for o in operacoes if o != "encoder.embeddings.word_embeddings.linhas"]
    destino.mkdir(parents=True, exist_ok=True)
    with quantizacao._sem_avisos():
        quantize_dynamic(rede, destino / exportacao.REDE, **_opcoes_dinamicas(operacoes, por_canal=alternativa == "degrau_por_canal"))
    exportacao_encoder.copiar_acompanhantes(fp32, destino, forma=f"etapa12_{alternativa}", nome=ALTERNATIVAS_DINAMICAS[alternativa], operacoes_quantizadas=len(operacoes))
    return destino


def pasta_da_dinamica(semente: int) -> Path:
    return pasta_da_etapa() / "dinamica" / ALUNO_A / "vocabulario_treino" / f"semente_{semente}"


def rodar_dinamica() -> None:
    """Achado 3, T3 e T4: o ONNX de 32 bits (com as pontuações de intenção) e cada alternativa, em dev, 3 sementes do Aluno A, vocabulário do treino."""
    plano()
    for semente in SEMENTES:
        fp32 = fp32_exportado(ALUNO_A, "vocabulario_treino", semente)
        base = pasta_da_dinamica(semente)
        if not (base / "fp32" / "previsoes_dev.parquet").exists():
            (base / "fp32").mkdir(parents=True, exist_ok=True)
            gravar_respostas(base / "fp32", "dev", *responder(fp32))
        for alternativa in ALTERNATIVAS_DINAMICAS:
            pasta = base / alternativa
            if (pasta / "previsoes_dev.parquet").exists():
                continue
            inicio = time.perf_counter()
            sistema = quantizar_alternativa_dinamica(fp32, pasta / "sistema", alternativa)
            gravar_respostas(pasta, "dev", *responder(sistema))
            _gravar_json(pasta / "meta.json", {"alternativa": alternativa, "o_que_e": ALTERNATIVAS_DINAMICAS[alternativa], "semente": semente,
                                               "bytes_da_rede": bytes_da_rede(sistema), "segundos": time.perf_counter() - inicio})
            print(f"{time.strftime('%H:%M:%S')} dinâmica {alternativa} semente {semente}: {time.perf_counter() - inicio:.0f} s", flush=True)


def binomial_bilateral(k: int, n: int) -> float:
    """Teste de sinal: a chance de um lado ter k ou mais (ou k ou menos) de n trocas, se os dois lados fossem igualmente prováveis."""
    from math import comb
    if n == 0:
        return 1.0
    cauda = sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2**n
    return min(1.0, 2 * cauda)


def e_intents_do_sistema(pasta: Path) -> list[str]:
    """A ordem das intenções nas pontuações de um sistema exportado (a da ficha)."""
    from especialista.compressao.sistema_encoder_onnx import FICHA
    return _json(pasta / FICHA)["intents"]


def margem(pontuacoes: np.ndarray) -> np.ndarray:
    """p1 − p2: a diferença entre as duas maiores probabilidades de intenção (softmax das pontuações)."""
    p = np.exp(pontuacoes - pontuacoes.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    duas = np.sort(p, axis=1)[:, -2:]
    return duas[:, 1] - duas[:, 0]


NOVE_MODELOS = [(m, v) for m, vs in ((ALUNO_A, ("sem_poda", "vocabulario_treino", "gradual_70")), (TETO, ("sem_poda", "vocabulario_treino", "magnitude_50")),
                                      (B2_ANTIGO, ("sem_poda", "vocabulario_treino", "gradual_70"))) for v in vs]


def contas_achado_3() -> dict:
    """T1, T2, T3, T4 e T5 do achado 3, em dev."""
    gab = list(gabarito("dev"))
    certas = np.array([g.intent for g in gab])
    fp32 = previsoes_da_10c(ALUNO_A, "vocabulario_treino", "fp32")
    din = previsoes_da_10c(ALUNO_A, "vocabulario_treino", "dinamica")
    saida: dict = {"partes": {}, "trocas": [], "alternativas": {}, "nove_modelos": []}
    # T1: que parte da rede
    for forma in ("dinamica", "so_tabela", "so_matrizes"):
        saida["partes"][forma] = comparar("dev", fp32, previsoes_da_10c(ALUNO_A, "vocabulario_treino", forma))
    # T2 e T3: as trocas de intenção, o lado e a margem
    linhas, margens_todas, margens_trocas = [], [], []
    primeira_certa = segunda_certa = 0   # nos pedidos de margem baixa: a 1ª ou a 2ª intenção do fp32 é a certa (o lado que uma troca ao acaso tomaria)
    for s in SEMENTES:
        i32 = np.array([p.intent for p in fp32[s]])
        i8 = np.array([p.intent for p in din[s]])
        pont = np.load(pasta_da_dinamica(s) / "fp32" / "intencoes_dev.npy")
        refeito = ler_respostas(pasta_da_dinamica(s) / "fp32")
        if [p.intent for p in refeito] != list(i32):
            raise RuntimeError(f"O fp32 refeito da semente {s} não dá as mesmas intenções da 10c.")
        m = margem(pont)
        trocou = i32 != i8
        intents = e_intents_do_sistema(fp32_exportado(ALUNO_A, "vocabulario_treino", s))
        baixa = m < np.percentile(m, 10)
        ordem = np.argsort(pont, axis=1)
        primeira_certa += int(sum(intents[ordem[k, -1]] == certas[k] for k in np.flatnonzero(baixa)))
        segunda_certa += int(sum(intents[ordem[k, -2]] == certas[k] for k in np.flatnonzero(baixa)))
        margens_todas.append(m)
        margens_trocas.append(m[trocou])
        for k in np.flatnonzero(trocou):
            linhas.append({"semente": s, "pedido": int(particoes()["dev"]["id"].iloc[k]), "certa": certas[k], "fp32": i32[k], "int8": i8[k],
                           "lado": "certo_para_errado" if i32[k] == certas[k] else ("errado_para_certo" if i8[k] == certas[k] else "errado_para_errado"),
                           "margem_fp32": float(m[k]), "percentil_da_margem": float((m < m[k]).mean())})
    trocas = pd.DataFrame(linhas)
    saida["trocas"] = trocas
    lados = trocas["lado"].value_counts().to_dict()
    ce, ec = lados.get("certo_para_errado", 0), lados.get("errado_para_certo", 0)
    todas, so_trocas = np.concatenate(margens_todas), np.concatenate(margens_trocas)
    saida["resumo_das_trocas"] = {
        "trocas": len(trocas), "por_semente": trocas.groupby("semente").size().to_dict(), "lados": lados,
        "teste_de_sinal_p": binomial_bilateral(ce, ce + ec), "pedidos_por_semente": len(gab),
        "fracao_certo_para_errado": ce / (ce + ec),
        "margem_baixa_primeira_certa": primeira_certa, "margem_baixa_segunda_certa": segunda_certa,
        "fracao_esperada_se_a_troca_fosse_ao_acaso_entre_as_duas_primeiras": primeira_certa / (primeira_certa + segunda_certa),
        "margem_mediana_das_trocas": float(np.median(so_trocas)), "margem_mediana_de_todos": float(np.median(todas)),
        "percentil_10_da_margem_de_todos": float(np.percentile(todas, 10)),
        "fracao_das_trocas_com_margem_abaixo_do_p10": float((so_trocas < np.percentile(todas, 10)).mean()),
        "fracao_das_trocas_com_margem_abaixo_de_0_1": float((so_trocas < 0.1).mean()), "fracao_de_todos_abaixo_de_0_1": float((todas < 0.1).mean()),
        "intencoes_que_mais_perdem": trocas[trocas["lado"] == "certo_para_errado"]["certa"].value_counts().head(8).to_dict(),
        "para_onde_vao": trocas[trocas["lado"] == "certo_para_errado"]["int8"].value_counts().head(8).to_dict(),
    }
    # T4: as alternativas
    for alternativa in ALTERNATIVAS_DINAMICAS:
        grupo = [ler_respostas(pasta_da_dinamica(s) / alternativa) for s in SEMENTES]
        metas = [_json(pasta_da_dinamica(s) / alternativa / "meta.json") for s in SEMENTES]
        saida["alternativas"][alternativa] = {"contra_fp32": comparar("dev", fp32, grupo), "bytes_da_rede": metas[0]["bytes_da_rede"],
                                              "o_que_e": ALTERNATIVAS_DINAMICAS[alternativa]}
        if alternativa == "dinamica":
            saida["dinamica_reproduz_a_10c"] = sum(a == b for x, y in zip(grupo, din) for a, b in zip(x, y))
    saida["alternativas"]["tabelas_em_32_bits"] = {"contra_fp32": saida["partes"]["so_matrizes"], "bytes_da_rede": None,
                                                   "o_que_e": "a dinâmica só nas matrizes (forma 'so_matrizes' da 10c, respostas gravadas)"}
    saida["bytes_fp32"] = bytes_da_rede(fp32_exportado(ALUNO_A, "vocabulario_treino", 0))
    # T5: os nove modelos da 10c
    for modelo, variante in NOVE_MODELOS:
        c = comparar("dev", previsoes_da_10c(modelo, variante, "fp32"), previsoes_da_10c(modelo, variante, "dinamica"))["intent_acc"]
        saida["nove_modelos"].append({"modelo": modelo, "variante": variante, **c})
    return saida


# ==========================================================================
# Achado 2: a quantização estática e a semente que desaba
# ==========================================================================

# Os modelos do achado: os dois que o enunciado cita (o 47,4 do Aluno A é o da variante "vocabulário do treino",
# não o do "sem poda", como o plano previa conferir) e, para contraste das faixas, o Aluno A com poda gradual de 70%.
MODELOS_DO_ACHADO_2 = ((TETO, "sem_poda"), (ALUNO_A, "vocabulario_treino"))
CONTRASTE_DO_ACHADO_2 = ((ALUNO_A, "gradual_70"),)
TIPOS = {   # os grupos de multiplicações por matriz gravada, pelo nome da camada (o que vem antes de ".produto")
    "consulta_chave_valor": ("attention.self.query", "attention.self.key", "attention.self.value"),
    "saida_da_atencao": ("attention.output.dense",),
    "entrada_da_rede_densa": ("intermediate.dense",),
    "saida_da_rede_densa": ("output.dense",),
    "cabecas": ("intencao", "entidades"),
}
BINS = np.concatenate([[0.0], np.logspace(-6, 4, 2001)])   # faixas de |x| para os percentis (passo de cerca de 1,2%)


def tipo_da_operacao(nome: str) -> str:
    if nome.endswith(".linhas"):
        return "tabelas"
    camada = nome.removesuffix(".produto")
    if camada in ("intencao", "entidades"):
        return "cabecas"
    sufixo = ".".join(camada.split(".")[4:])   # encoder.encoder.layer.N.<sufixo>
    return next(tipo for tipo, nomes in TIPOS.items() if sufixo in nomes)


def camada_da_operacao(nome: str) -> int | None:
    partes = nome.split(".")
    return int(partes[3]) if nome.startswith("encoder.encoder.layer.") else None


@lru_cache(maxsize=None)
def textos_de_calibracao() -> tuple[str, ...]:
    from especialista.experimentos import etapa8
    return tuple(etapa8.textos_de_calibracao(particoes()["train"]))


def faixas_das_ativacoes(fp32: Path) -> pd.DataFrame:
    """T1: a entrada de cada multiplicação por matriz gravada (o que a estática quantiza), nos 512 pedidos de calibração.

    Por tensor: menor e maior valor (a faixa que a calibração MinMax usa), percentis de |x| e as dimensões de onde vêm os maiores valores.
    """
    import onnx
    from onnx import helper

    from especialista.compressao import exportacao, quantizacao
    from especialista.compressao.sistema_encoder_onnx import Alinhador
    from especialista.compressao.sistema_onnx import abrir_sessao

    rede = exportacao.ler_rede(fp32)
    matrizes = quantizacao.operacoes_com_pesos(rede)["matrizes"]
    entrada = {no.name: no.input[0] for no in rede.graph.node if no.name in matrizes}
    tensores = sorted(set(entrada.values()), key=list(entrada.values()).index)
    for nome in tensores:
        rede.graph.output.append(helper.make_tensor_value_info(nome, onnx.TensorProto.FLOAT, None))
    sessao = abrir_sessao(rede.SerializeToString(), THREADS)
    del rede
    alinhador = Alinhador.carregar(fp32)
    menor = {t: np.inf for t in tensores}
    maior = {t: -np.inf for t in tensores}
    histograma = {t: np.zeros(len(BINS) - 1) for t in tensores}
    por_dimensao: dict[str, np.ndarray] = {}
    for texto in textos_de_calibracao():
        valores = sessao.run(tensores, alinhador.entradas([texto.split()]))
        for t, v in zip(tensores, valores):
            menor[t], maior[t] = min(menor[t], float(v.min())), max(maior[t], float(v.max()))
            a = np.abs(v.reshape(-1, v.shape[-1]))
            histograma[t] += np.histogram(np.clip(a, 0, BINS[-1]), BINS)[0]
            pd_ = a.max(axis=0)
            por_dimensao[t] = np.maximum(por_dimensao.get(t, pd_), pd_)
    linhas = []
    for operacao, t in entrada.items():
        acumulado = np.cumsum(histograma[t]) / histograma[t].sum()
        percentil = lambda q: float(BINS[1:][np.searchsorted(acumulado, q)])
        dims = por_dimensao[t]
        ordem = np.argsort(dims)[::-1]
        linhas.append({
            "operacao": operacao, "tensor": t, "tipo": tipo_da_operacao(operacao), "camada": camada_da_operacao(operacao), "dimensoes": len(dims),
            "menor": menor[t], "maior": maior[t], "faixa": maior[t] - min(menor[t], 0.0), "maximo_absoluto": max(abs(menor[t]), abs(maior[t])),
            "p99": percentil(0.99), "p99_9": percentil(0.999), "p99_99": percentil(0.9999),
            "dimensao_do_maximo": int(ordem[0]), "maximo_da_dimensao_2": float(dims[ordem[1]]), "mediana_dos_maximos_por_dimensao": float(np.median(dims)),
            "dimensoes_acima_de_6x_a_mediana": int((dims > 6 * np.median(dims)).sum()),
            "cinco_maiores_dimensoes": ";".join(f"{int(d)}:{dims[d]:.1f}" for d in ordem[:5]),
        })
    tabela = pd.DataFrame(linhas)
    tabela["razao_maximo_p99_99"] = tabela["maximo_absoluto"] / tabela["p99_99"]
    tabela["degrau_minmax"] = tabela["faixa"] / 255
    tabela["niveis_ate_p99"] = tabela["p99"] / tabela["degrau_minmax"]   # quantos degraus da faixa MinMax cobrem 99% dos valores
    return tabela


class _UmSo:
    """Um leitor de calibração que entrega um pedido só."""

    def __init__(self, entrada):
        self._entrada = [entrada]

    def get_next(self):
        return self._entrada.pop() if self._entrada else None


@contextmanager
def _histograma_pedido_a_pedido():
    """A calibração por histograma (percentil, entropia) do ONNX Runtime junta as ativações de todos os pedidos num só arranjo, e falha quando os
    pedidos têm comprimentos diferentes (é o caso: um pedido por vez, sem preenchimento). Dentro do bloco, ela recebe um pedido por vez e soma os
    histogramas (o coletor dela já sabe juntar histogramas de chamadas sucessivas). A calibração MinMax não passa por aqui."""
    from onnxruntime.quantization import calibrate

    original = calibrate.HistogramCalibrater.collect_data

    def pedido_a_pedido(self, data_reader):
        while (entrada := data_reader.get_next()) is not None:
            original(self, _UmSo(entrada))

    calibrate.HistogramCalibrater.collect_data = pedido_a_pedido
    try:
        yield
    finally:
        calibrate.HistogramCalibrater.collect_data = original


def quantizar_estatica(fp32: Path, destino: Path, operacoes: list[str], metodo: str = "MinMax") -> Path:
    """A estática da 10c (`quantizacao.quantizar_rede`, forma "estatica": QDQ, ativações uint8 assimétricas, pesos int8 simétricos, um degrau por tensor),
    com a lista de operações e o método de calibração dados. Com todas as operações e "MinMax", é a mesma chamada da 10c."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    from especialista.compressao import exportacao, exportacao_encoder, quantizacao
    from especialista.compressao.sistema_encoder_onnx import Alinhador

    if (destino / exportacao.REDE).exists():
        return destino
    destino.mkdir(parents=True, exist_ok=True)
    leitor = quantizacao._Leitor(Alinhador.carregar(fp32), list(textos_de_calibracao()))
    temporarios = destino / "temporarios"
    # A ferramenta grava uma cópia da rede com as formas inferidas AO LADO do arquivo de entrada (`rede-inferred.onnx`) e a apaga no fim.
    # Dois processos com a mesma rede de entrada apagariam a cópia um do outro; por isso a entrada é um link físico na pasta temporária desta quantização.
    temporarios.mkdir(parents=True, exist_ok=True)
    entrada = temporarios / exportacao.REDE
    entrada.unlink(missing_ok=True)
    entrada.hardlink_to(fp32 / exportacao.REDE)
    try:
        with quantizacao._sem_avisos(), quantizacao._temporarios_em(temporarios), _histograma_pedido_a_pedido():
            quantize_static(
                entrada, destino / exportacao.REDE, leitor, quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                calibrate_method=getattr(CalibrationMethod, metodo), op_types_to_quantize=["Gather", "MatMul"], nodes_to_quantize=operacoes,
                per_channel=False, reduce_range=False, weight_type=QuantType.QInt8, extra_options={"MatMulConstBOnly": True, "WeightSymmetric": True},
            )
    finally:
        shutil.rmtree(temporarios, ignore_errors=True)
    exportacao_encoder.copiar_acompanhantes(fp32, destino, forma=f"etapa12_estatica_{metodo}", nome=f"INT8 estática ({metodo})", operacoes_quantizadas=len(operacoes))
    return destino


def todas_as_operacoes(fp32: Path) -> list[str]:
    from especialista.compressao import exportacao, quantizacao
    por_parte = quantizacao.operacoes_com_pesos(exportacao.ler_rede(fp32))
    return por_parte["tabela"] + por_parte["matrizes"]


def pasta_da_estatica(modelo: str, variante: str, semente: int) -> Path:
    return pasta_da_etapa() / "estatica" / modelo / variante / f"semente_{semente}"


def medir_estatica(modelo: str, variante: str, semente: int, nome: str, operacoes: list[str], metodo: str = "MinMax", descricao: str = "") -> dict:
    """Quantiza, responde em dev, grava respostas e tamanho, e apaga a rede quantizada (só fica o que foi medido)."""
    pasta = pasta_da_estatica(modelo, variante, semente) / nome
    if (pasta / "meta.json").exists():
        return _json(pasta / "meta.json")
    fp32 = fp32_exportado(modelo, variante, semente)
    inicio = time.perf_counter()
    sistema = quantizar_estatica(fp32, pasta / "sistema", operacoes, metodo)
    previsoes, _ = responder(sistema)
    gravar_respostas(pasta, "dev", previsoes)
    meta = {"modelo": modelo, "variante": variante, "semente": semente, "nome": nome, "descricao": descricao, "metodo": metodo,
            "operacoes": len(operacoes), "bytes_da_rede": bytes_da_rede(sistema), "slot_f1_dev": 100 * qualidade.avaliar(list(gabarito("dev")), previsoes)["slot_f1"],
            "segundos": time.perf_counter() - inicio}
    shutil.rmtree(pasta / "sistema")
    _gravar_json(pasta / "meta.json", meta)
    print(f"{time.strftime('%H:%M:%S')} estática {modelo}/{variante}/{semente} {nome}: F1 {meta['slot_f1_dev']:.1f} ({meta['segundos']:.0f} s)", flush=True)
    return meta


def rodar_faixas() -> None:
    """T1 do achado 2: as faixas das ativações por tensor, nos modelos do achado e no contraste, 3 sementes."""
    plano()
    for modelo, variante in (*MODELOS_DO_ACHADO_2, *CONTRASTE_DO_ACHADO_2):
        for semente in SEMENTES:
            arquivo = pasta_da_estatica(modelo, variante, semente) / "faixas.csv"
            if arquivo.exists():
                continue
            inicio = time.perf_counter()
            tabela = faixas_das_ativacoes(fp32_exportado(modelo, variante, semente))
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            tabela.to_csv(arquivo, index=False)
            print(f"{time.strftime('%H:%M:%S')} faixas {modelo}/{variante}/{semente}: {time.perf_counter() - inicio:.0f} s", flush=True)


def rodar_sensibilidade() -> None:
    """T2 do achado 2: no teto sem poda, sementes 1 e 0, a estática num grupo de multiplicações por vez (o resto fica em 32 bits)."""
    plano()
    for semente in (1, 0):
        fp32 = fp32_exportado(TETO, "sem_poda", semente)
        todas = todas_as_operacoes(fp32)
        medir_estatica(TETO, "sem_poda", semente, "todas", todas, descricao="a estática da 10c, refeita")
        for tipo in (*TIPOS, "tabelas"):
            medir_estatica(TETO, "sem_poda", semente, f"so_{tipo}", [o for o in todas if tipo_da_operacao(o) == tipo], descricao=f"só {tipo} em 8 bits")


def piores_pela_faixa(faixas: pd.DataFrame, olhar: int = 20) -> list[str]:
    """T4: as multiplicações com a maior razão máximo/percentil 99,99 da entrada, até o maior salto (em log) entre as `olhar` primeiras.

    O corte sai só das faixas (T1), sem olhar a qualidade em dev, como o plano pede.
    """
    ordem = faixas.sort_values("razao_maximo_p99_99", ascending=False).head(olhar)
    logs = np.log(ordem["razao_maximo_p99_99"].to_numpy())
    k = int(np.argmax(logs[:-1] - logs[1:])) + 1
    return list(ordem["operacao"].iloc[:k])


def rodar_sensibilidade_por_camada_cli() -> None:
    """T2, segunda parte: o tipo que mais perdeu na primeira parte ("saida_da_rede_densa"), uma camada por vez.

    O plano previa só o teto na semente 1; a semente 0 do teto e a semente 1 do Aluno A entram para contraste (desvio relatado).
    """
    plano()
    for modelo, variante, semente in ((TETO, "sem_poda", 1), (TETO, "sem_poda", 0), (ALUNO_A, "vocabulario_treino", 1)):
        todas = todas_as_operacoes(fp32_exportado(modelo, variante, semente))
        for camada in range(12):
            medir_estatica(modelo, variante, semente, f"so_saida_da_rede_densa_camada_{camada}",
                           [o for o in todas if tipo_da_operacao(o) == "saida_da_rede_densa" and camada_da_operacao(o) == camada],
                           descricao=f"só a entrada da saída da rede densa da camada {camada} em 8 bits")


def rodar_acumulado() -> None:
    """T2, terceira parte (acrescentada depois de ver que nenhuma camada sozinha explica a queda: desvio relatado).

    A entrada da saída da rede densa em 8 bits nas camadas 0 a k (as de cima em 32 bits), para ver como a perda se acumula. Teto sem poda, sementes 1 e 0.
    """
    plano()
    for semente in (1, 0):
        todas = todas_as_operacoes(fp32_exportado(TETO, "sem_poda", semente))
        for k in (1, 3, 5, 7, 9):
            medir_estatica(TETO, "sem_poda", semente, f"so_saida_da_rede_densa_camadas_0_a_{k}",
                           [o for o in todas if tipo_da_operacao(o) == "saida_da_rede_densa" and camada_da_operacao(o) <= k],
                           descricao=f"só a entrada da saída da rede densa das camadas 0 a {k} em 8 bits")


def rodar_um_de_fora() -> None:
    """T4, segunda parte (acrescentada depois do acumulado: desvio relatado): a estática da 10c inteira, com UMA multiplicação da saída da rede
    densa em 32 bits por vez. Diz qual tensor, sozinho, devolve a qualidade. Teto sem poda e Aluno A, vocabulário do treino, semente 1."""
    plano()
    for modelo, variante in MODELOS_DO_ACHADO_2:
        todas = todas_as_operacoes(fp32_exportado(modelo, variante, 1))
        for camada in range(12):
            fora = f"encoder.encoder.layer.{camada}.output.dense.produto"
            medir_estatica(modelo, variante, 1, f"todas_menos_saida_da_rede_densa_camada_{camada}", [o for o in todas if o != fora],
                           descricao=f"a estática inteira, com a saída da rede densa da camada {camada} em 32 bits")


def rodar_exclusoes() -> None:
    """T4 do achado 2: a estática MinMax com os piores tensores em 32 bits, 3 sementes dos dois modelos.

    - `sem_os_piores_pela_faixa`: o que o plano fixou (k pelo maior salto da razão máximo/percentil, `piores_pela_faixa`);
    - `sem_a_saida_da_rede_densa`: as 12 multiplicações do tipo que mais perdeu em T2 ficam em 32 bits (acrescentada depois de ver T2: desvio relatado).
    """
    plano()
    for modelo, variante in MODELOS_DO_ACHADO_2:
        for semente in SEMENTES:
            fp32 = fp32_exportado(modelo, variante, semente)
            todas = todas_as_operacoes(fp32)
            piores = piores_pela_faixa(pd.read_csv(pasta_da_estatica(modelo, variante, semente) / "faixas.csv"))
            medir_estatica(modelo, variante, semente, "sem_os_piores_pela_faixa", [o for o in todas if o not in piores],
                           descricao=f"{len(piores)} multiplicações em 32 bits: " + ", ".join(piores))
            medir_estatica(modelo, variante, semente, "sem_a_saida_da_rede_densa", [o for o in todas if tipo_da_operacao(o) != "saida_da_rede_densa"],
                           descricao="as 12 multiplicações da saída da rede densa em 32 bits")
            medir_estatica(modelo, variante, semente, "todas", todas, descricao="a estática da 10c, refeita")


ALTERNATIVAS_ESTATICAS = {
    "todas": "a estática da 10c, refeita (MinMax, todas as multiplicações e tabelas)",
    "calibracao_percentile": "calibração por percentil (99,999)",
    "calibracao_entropy": "calibração por entropia",
    "sem_os_piores_pela_faixa": "MinMax, com as multiplicações de maior razão máximo/percentil em 32 bits (k pelo maior salto, sem olhar dev)",
    "sem_a_saida_da_rede_densa": "MinMax, com as 12 multiplicações da saída da rede densa em 32 bits",
}


def contas_achado_2() -> dict:
    """T1 a T4 do achado 2, em dev."""
    faixas = []
    for modelo, variante in (*MODELOS_DO_ACHADO_2, *CONTRASTE_DO_ACHADO_2):
        for s in SEMENTES:
            t = pd.read_csv(pasta_da_estatica(modelo, variante, s) / "faixas.csv")
            faixas.append(t.assign(modelo=modelo, variante=variante, semente=s))
    faixas = pd.concat(faixas, ignore_index=True)
    metas = pd.DataFrame([_json(m) for m in sorted((pasta_da_etapa() / "estatica").glob("*/*/semente_*/*/meta.json"))])
    fp32_f1 = {(m, v, s): 100 * qualidade.avaliar(list(gabarito("dev")), p)["slot_f1"]
               for m, v in (*MODELOS_DO_ACHADO_2, *CONTRASTE_DO_ACHADO_2) for s, p in enumerate(previsoes_da_10c(m, v, "fp32"))}
    metas["slot_f1_fp32"] = [fp32_f1[(r.modelo, r.variante, r.semente)] for r in metas.itertuples()]
    metas["queda"] = metas["slot_f1_dev"] - metas["slot_f1_fp32"]
    alternativas = []
    for modelo, variante in MODELOS_DO_ACHADO_2:
        fp32 = previsoes_da_10c(modelo, variante, "fp32")
        for nome, o_que_e in ALTERNATIVAS_ESTATICAS.items():
            pastas = [pasta_da_estatica(modelo, variante, s) / nome for s in SEMENTES]
            if not all((p / "meta.json").exists() for p in pastas):
                continue
            c = comparar("dev", fp32, [ler_respostas(p) for p in pastas])
            alternativas.append({"modelo": modelo, "variante": variante, "alternativa": nome, "o_que_e": o_que_e,
                                 "bytes_da_rede": _json(pastas[0] / "meta.json")["bytes_da_rede"],
                                 **{f"{m}_{k}": c[m][k] for m in TRES for k in ("diferenca", "ic_min", "ic_max")},
                                 **{f"slot_f1_semente_{s}": _json(p / "meta.json")["slot_f1_dev"] for s, p in enumerate(pastas)}})
    return {"faixas": faixas, "medidas": metas, "alternativas": pd.DataFrame(alternativas)}


def rodar_calibracoes() -> None:
    """T3 do achado 2: a estática com todas as operações, calibrada por percentil (99,999) e por entropia, 3 sementes dos dois modelos."""
    plano()
    for metodo in ("Percentile", "Entropy"):
        for modelo, variante in MODELOS_DO_ACHADO_2:
            for semente in SEMENTES:
                fp32 = fp32_exportado(modelo, variante, semente)
                medir_estatica(modelo, variante, semente, f"calibracao_{metodo.lower()}", todas_as_operacoes(fp32), metodo, descricao=f"calibração {metodo}")


# ==========================================================================
# Os artigos, conferidos na fonte (API do arXiv, gratuita)
# ==========================================================================

ARTIGOS = (
    {"passo": "Achado 1", "onde": "XLM-R", "tecnica": "Redes aprendem primeiro os padrões simples e só depois memorizam o ruído dos rótulos", "autor": "Arpit",
     "quem": "Arpit et al.", "titulo": "A Closer Look at Memorization in Deep Networks", "arxiv": "1706.05394"},
    {"passo": "Achado 1", "onde": "XLM-R", "tecnica": "Redes grandes encaixam até rótulos sorteados", "autor": "Zhang",
     "quem": "Zhang et al.", "titulo": "Understanding deep learning requires rethinking generalization", "arxiv": "1611.03530"},
    {"passo": "Achado 1", "onde": "XLM-R", "tecnica": "Concordar mais com o professor nem sempre faz o aluno generalizar melhor", "autor": "Stanton",
     "quem": "Stanton et al.", "titulo": "Does Knowledge Distillation Really Work?", "arxiv": "2106.05945"},
    {"passo": "Achado 1", "onde": "XLM-R", "tecnica": "Ajuste fino instável entre sementes; diferenças de generalização com a mesma perda de treino", "autor": "Mosbach",
     "quem": "Mosbach, Andriushchenko & Klakow", "titulo": "On the Stability of Fine-tuning BERT", "arxiv": "2006.04884"},
    {"passo": "Achado 1", "onde": "XLM-R", "tecnica": "Reaquecer a taxa de aprendizado primeiro aumenta a perda", "autor": "Gupta",
     "quem": "Gupta et al.", "titulo": "Continual Pre-Training of Large Language Models: How to (re)warm your model?", "arxiv": "2308.04014"},
    {"passo": "Achado 1", "onde": "XLM-R", "tecnica": "Cortar o vocabulário de um modelo multilíngue para uma língua conserva a qualidade", "autor": "Abdaoui",
     "quem": "Abdaoui, Pradel & Sigel", "titulo": "Load What You Need: Smaller Versions of Multilingual BERT", "arxiv": "2010.05609"},
    {"passo": "Achado 1", "onde": "XLM-R", "tecnica": "Corte do vocabulário para a língua-alvo (cerca de 50% do vocabulário basta)", "autor": "Ushio",
     "quem": "Ushio, Zhou & Camacho-Collados", "titulo": "An Efficient Multilingual Language Model Compression through Vocabulary Trimming", "arxiv": "2305.15020"},
    {"passo": "Achado 2", "onde": "XLM-R", "tecnica": "Ativações de faixa dinâmica alta, com extremos estruturados, dificultam a quantização de Transformers", "autor": "Bondarenko",
     "quem": "Bondarenko, Nagel & Blankevoort", "titulo": "Understanding and Overcoming the Challenges of Efficient Transformer Quantization", "arxiv": "2109.12948"},
    {"passo": "Achado 2", "onde": "XLM-R", "tecnica": "Dimensões extremas surgem cedo no pré-treino e ficam na mesma posição", "autor": "Kovaleva",
     "quem": "Kovaleva et al.", "titulo": "BERT Busters: Outlier Dimensions that Disrupt Transformers", "arxiv": "2105.06990"},
    {"passo": "Achado 2", "onde": "XLM-R", "tecnica": "Poucas dimensões extremas quebram a quantização por tensor; separá-las em mais bits resolve", "autor": "Dettmers",
     "quem": "Dettmers et al.", "titulo": "LLM.int8()", "arxiv": "2208.07339"},
    {"passo": "Achado 2", "onde": "XLM-R", "tecnica": "Calibração por percentil e por entropia na quantização de inteiros, BERT incluído", "autor": "Wu",
     "quem": "Wu et al.", "titulo": "Integer Quantization for Deep Learning Inference", "arxiv": "2004.09602"},
    {"passo": "Achados 2 e 3", "onde": "XLM-R", "tecnica": "O esquema de 8 bits (degrau e ponto zero)", "autor": "Jacob",
     "quem": "Jacob et al.", "titulo": "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", "arxiv": "1712.05877"},
    {"passo": "Achado 3", "onde": "XLM-R", "tecnica": "Modelos comprimidos trocam respostas certas por erradas e o contrário, mesmo com acurácia parecida", "autor": "Dutta",
     "quem": "Dutta et al.", "titulo": "Accuracy is Not All You Need", "arxiv": "2407.09141"},
)

# O que o resumo de cada artigo (o texto que a API devolve) sustenta. Só o resumo foi lido: nenhum PDF foi aberto nesta etapa.
RESUMOS_LIDOS = {
    "1706.05394": ("em parte", "o resumo diz que redes conseguem memorizar ruído, mas aprendem primeiro os padrões simples, e que os dados determinam quanto se memoriza. Sustenta "
                               "a leitura de que treino a mais sobre rótulos com erro passa a encaixar os erros; não fala de destilação, de encoders pré-treinados nem de reajuste depois de poda"),
    "1611.03530": ("em parte", "o resumo diz que redes treinadas com descida de gradiente encaixam facilmente rótulos sorteados. Sustenta que o aluno tem capacidade de copiar os erros do "
                               "professor; é sobre redes convolucionais de imagens"),
    "2106.05945": ("sim", "o resumo diz que concordar mais com o professor, paradoxalmente, nem sempre leva a generalizar melhor. É o padrão medido aqui: o reajuste sobe a concordância "
                          "com o professor em dev e desce o acerto contra os humanos"),
    "2006.04884": ("em parte", "o resumo diz que o ajuste fino de BERT varia muito com a semente e que modelos com a mesma perda de treino generalizam diferente. Sustenta tratar a semente "
                               "como fonte de variação; não trata de quantização nem de reajuste"),
    "2308.04014": ("em parte", "o resumo diz que reaquecer a taxa de aprendizado primeiro aumenta a perda e só depois melhora. Sustenta a queda da 1ª época do reajuste; é pré-treino "
                               "contínuo de um modelo de linguagem, não ajuste fino"),
    "2010.05609": ("em parte", "o resumo diz que versões menores do BERT multilíngue, com o vocabulário das línguas-alvo, mantêm resultados comparáveis. O corte de lá é por língua; "
                               "o daqui, pelo texto do treino, é mais agressivo e deixa pedidos de dev e do teste com subtokens sem linha"),
    "2305.15020": ("em parte", "o resumo diz que cortar o vocabulário para a língua-alvo mantém o desempenho, e que cerca de 50% do vocabulário costuma bastar. Mesma ressalva: "
                               "o corte daqui fica com 5.196 linhas (cerca de 2% de 250.002)"),
    "2109.12948": ("em parte", "o resumo diz que Transformers têm ativações de faixa dinâmica alta, difíceis de representar em ponto fixo de poucos bits, com extremos estruturados nas "
                               "conexões residuais. Sustenta o mecanismo geral; aqui o tensor que desaba é a entrada da 2ª camada da rede densa (saída da GELU), que o resumo não cita"),
    "2105.06990": ("em parte", "o resumo diz que as dimensões extremas surgem cedo no pré-treino e aparecem sempre na mesma posição. Sustenta que as dimensões extremas daqui se "
                               "repetem entre sementes e modelos (vêm do XLM-R pré-treinado); lá elas estão nos parâmetros da normalização, aqui na ativação da rede densa"),
    "2208.07339": ("em parte", "o resumo diz que Transformers têm atributos extremos sistemáticos, em poucas dimensões, e que isolar essas dimensões numa multiplicação de "
                               "16 bits mantém a qualidade com mais de 99,9% dos valores em 8 bits. Sustenta a ideia de deixar em mais bits o pouco que tem extremos; o artigo é "
                               "sobre modelos de 6,7 bilhões de parâmetros para cima, e a decomposição é por dimensão, não por tensor como aqui"),
    "2004.09602": ("em parte", "o resumo diz que um fluxo de 8 bits mantém a acurácia a 1% do ponto flutuante em todas as redes estudadas, BERT-large incluído. Sustenta que a "
                               "escolha da calibração importa; o resumo não diz qual calibração serve ao BERT"),
    "1712.05877": ("não conferido", "o esquema já é o das Etapas 8 e 10c; resumo não relido"),
    "2407.09141": ("sim", "o resumo diz que, com acurácia parecida, modelos comprimidos trocam respostas de certas para erradas e o contrário, em proporção. É o que T2 e T3 mediram"),
}


def conferir_artigos() -> Path:
    from especialista.experimentos import etapa10d
    return etapa10d.conferir_artigos(base=pasta_da_etapa(), artigos=ARTIGOS)


def quadro_de_artigos() -> pd.DataFrame:
    from especialista.experimentos import etapa10d
    lidos = {i: {"sustenta": s, "diz": d} for i, (s, d) in RESUMOS_LIDOS.items() if s != "não conferido"}
    return etapa10d.quadro_de_artigos(base=pasta_da_etapa(), artigos=ARTIGOS, lidos=lidos)


# ==========================================================================
# As contas, a leitura no teste e a entrega
# ==========================================================================


def pasta_das_contas() -> Path:
    return pasta_da_etapa() / "contas"


def _linhas(c: dict) -> dict:
    return {f"{m}_{k}": c[m][k] for m in TRES for k in ("a", "b", "diferenca", "ic_min", "ic_max", "distinguivel")}


def calcular() -> Path:
    """Todas as contas em dev, gravadas em `contas/`: é daqui que o notebook lê."""
    plano()
    pasta = pasta_das_contas()
    pasta.mkdir(parents=True, exist_ok=True)
    # Achado 1
    a1 = contas_achado_1()
    a1["comparacoes"].to_csv(pasta / "achado1_comparacoes.csv", index=False)
    a1["professor"].to_csv(pasta / "achado1_professor.csv", index=False)
    a1["curvas"].to_csv(pasta / "achado1_curvas.csv", index=False)
    a1["treino"].to_csv(pasta / "achado1_treino.csv", index=False)
    dd = []
    a = momentos_em_dev(ALUNO_A)
    for aluno in (B1, B2):
        b = momentos_em_dev(aluno)
        for momento in ("sem_corte_e_reajuste", "corte_e_reajuste"):
            for metrica in ("exact_match", "intent_acc", "slot_f1"):
                dd.append({"aluno": aluno, "contra": ALUNO_A, "momento": momento, "metrica": metrica,
                           **diferenca_das_diferencas("dev", a["partida"], a[momento], b["partida"], b[momento], metrica)})
    pd.DataFrame(dd).to_csv(pasta / "achado1_diferenca_das_diferencas.csv", index=False)
    afetados_teste = afetados_pelo_corte(particao="test")   # só a contagem de pedidos com subtoken cortado: não lê resposta nenhuma
    afetados = afetados_pelo_corte()
    iguais = {}   # o corte só muda o que o modelo vê nos pedidos afetados: fora deles, com e sem corte, o reajuste dá as mesmas respostas?
    for aluno in (B1, B2):
        m = momentos_em_dev(aluno)
        iguais[aluno] = [{"nao_afetados_iguais": int(sum(x == y for k, (x, y) in enumerate(zip(c, s)) if not afetados[k])), "nao_afetados": int((~afetados).sum()),
                          "afetados_iguais": int(sum(x == y for k, (x, y) in enumerate(zip(c, s)) if afetados[k])), "afetados": int(afetados.sum())}
                         for c, s in zip(m["corte_e_reajuste"], m["sem_corte_e_reajuste"])]
    treino_igual = {aluno: all(ler_previsoes(pasta_da_etapa() / "treino" / aluno / "corte_e_reajuste" / f"semente_{s}.parquet", "train")
                               == ler_previsoes(pasta_da_etapa() / "treino" / aluno / "sem_corte_e_reajuste" / f"semente_{s}.parquet", "train") for s in SEMENTES)
                    for aluno in (B1, B2)}
    _gravar_json(pasta / "achado1_corte.json", {**a1["corte"], "pedidos_do_teste": len(afetados_teste), "afetados_no_teste": int(afetados_teste.sum()),
                                                "com_e_sem_corte_por_semente": iguais, "com_e_sem_corte_iguais_no_treino_inteiro": treino_igual})
    # Achado 3
    a3 = contas_achado_3()
    a3["trocas"].to_csv(pasta / "achado3_trocas.csv", index=False)
    pd.DataFrame([{"forma": f, **_linhas(c)} for f, c in a3["partes"].items()]).to_csv(pasta / "achado3_partes.csv", index=False)
    pd.DataFrame([{"alternativa": n, "o_que_e": d["o_que_e"], "bytes_da_rede": d["bytes_da_rede"], **_linhas(d["contra_fp32"])}
                  for n, d in a3["alternativas"].items()]).to_csv(pasta / "achado3_alternativas.csv", index=False)
    pd.DataFrame(a3["nove_modelos"]).to_csv(pasta / "achado3_nove_modelos.csv", index=False)
    _gravar_json(pasta / "achado3_resumo.json", {**a3["resumo_das_trocas"], "bytes_fp32": a3["bytes_fp32"],
                                                 "dinamica_refeita_igual_a_10c": a3.get("dinamica_reproduz_a_10c"), "pedidos_3_sementes": 3 * len(gabarito("dev"))})
    # Achado 2
    a2 = contas_achado_2()
    a2["faixas"].to_csv(pasta / "achado2_faixas.csv", index=False)
    a2["medidas"].to_csv(pasta / "achado2_medidas.csv", index=False)
    a2["alternativas"].to_csv(pasta / "achado2_alternativas.csv", index=False)
    return pasta


def leitura_no_teste() -> Path:
    """A leitura no teste fixada no plano, uma por achado, e nenhuma outra. Grava `contas/teste.json`."""
    regras = plano()
    pasta = pasta_das_contas()
    saida: dict = {"regras": {k: regras[k]["leitura_no_teste"] for k in ("achado_1", "achado_2", "achado_3")}}
    # Achado 1: a regra da 10b para a taxa menor, em dev, nos dois alunos novos
    passa = {}
    for aluno in (B1, B2):
        m = momentos_em_dev(aluno)
        c = comparar("dev", m["partida"], m["corte_e_reajuste_taxa_menor"])["exact_match"]
        passa[aluno] = {"dev": c, "passa": c["ic_max"] >= 0}   # a regra da 10b: não distinguivelmente menor que o modelo sem poda
    momento = "taxa_menor" if all(p["passa"] for p in passa.values()) else "sem_corte"
    lidas = {}
    p11c = ENTREGAS / "11c_cenario_b_ate_o_fim" / "previsoes"
    for aluno in (B1, B2):
        partida = _grupo([p11c / f"{aluno}__sem_poda__pytorch_semente_{s}_test.parquet" for s in SEMENTES], "test")
        depois = _grupo([pasta_do_reajuste(momento, aluno, s) / "previsoes_test.parquet" for s in SEMENTES], "test")
        lidas[aluno] = comparar("test", partida, depois)["exact_match"]
    saida["achado_1"] = {"regra_em_dev": passa, "lido": f"{momento} − partida (sem poda)", "teste": lidas}
    # Achado 3: só se alguma alternativa resolve em dev
    alt = pd.read_csv(pasta / "achado3_alternativas.csv")
    resolve = alt[(alt["intent_acc_ic_min"] <= 0) & (alt["intent_acc_ic_max"] >= 0) & (alt["intent_acc_diferenca"] >= -0.15)
                  & (alt["slot_f1_diferenca"] >= -1) & (alt["alternativa"] != "dinamica")]
    saida["achado_3"] = {"resolvem_em_dev": list(resolve["alternativa"]), "teste": None,
                         "nota": "nenhuma alternativa resolve em dev pelo critério do plano: nada vai ao teste" if resolve.empty else None}
    if not resolve.empty:
        raise NotImplementedError("Alguma alternativa resolveu: a leitura no teste do achado 3 ainda não foi escrita.")
    # Achado 2: a alternativa estática de menor perda média de F1 em dev nos dois modelos
    alt2 = pd.read_csv(pasta / "achado2_alternativas.csv")
    alt2 = alt2[alt2["alternativa"] != "todas"]
    media_por = alt2.groupby("alternativa").agg(queda=("slot_f1_diferenca", "mean"), bytes=("bytes_da_rede", "mean")).sort_values(["queda", "bytes"], ascending=[False, True])
    melhor = media_por.index[0]
    lidas2 = {}
    for modelo, variante in MODELOS_DO_ACHADO_2:
        fp32 = previsoes_da_10c(modelo, variante, "fp32", "test")
        grupo = []
        for s in SEMENTES:
            destino = pasta_da_estatica(modelo, variante, s) / melhor / "teste"
            if not (destino / "previsoes_test.parquet").exists():
                fp = fp32_exportado(modelo, variante, s)
                todas = todas_as_operacoes(fp)
                operacoes = {"calibracao_percentile": todas, "calibracao_entropy": todas,
                             "sem_a_saida_da_rede_densa": [o for o in todas if tipo_da_operacao(o) != "saida_da_rede_densa"],
                             "sem_os_piores_pela_faixa": [o for o in todas if o not in piores_pela_faixa(pd.read_csv(pasta_da_estatica(modelo, variante, s) / "faixas.csv"))]}[melhor]
                metodo = {"calibracao_percentile": "Percentile", "calibracao_entropy": "Entropy"}.get(melhor, "MinMax")
                sistema = quantizar_estatica(fp, destino / "sistema", operacoes, metodo)
                previsoes, _ = responder(sistema)
                if previsoes != ler_respostas(pasta_da_estatica(modelo, variante, s) / melhor):
                    raise RuntimeError("A alternativa quantizada de novo não repete as respostas em dev.")
                gravar_respostas(destino, "test", responder(sistema, "test")[0])
                shutil.rmtree(destino / "sistema")
            grupo.append(ler_previsoes(destino / "previsoes_test.parquet", "test"))
        lidas2[f"{modelo}/{variante}"] = comparar("test", fp32, grupo)["slot_f1"]
    saida["achado_2"] = {"escolhida_em_dev": melhor, "media_em_dev": media_por.reset_index().to_dict("records"), "teste": lidas2}
    _gravar_json(pasta / "teste.json", saida)
    return pasta / "teste.json"


def entregar() -> Path:
    """Copia para `entregas/12_investigacao_dos_achados/` o plano, as contas, a leitura no teste, os artigos, os históricos e as respostas em dev
    das execuções novas, as impressões digitais e o código. As figuras, o RESUMO e o HTML vêm do notebook."""
    destino = pasta_da_entrega()
    destino.mkdir(parents=True, exist_ok=True)
    origem = pasta_da_etapa()
    for nome in ("plano.json", "plano.sha256"):
        shutil.copy2(origem / nome, destino / nome)
    for arquivo in sorted(pasta_das_contas().glob("*")):
        shutil.copy2(arquivo, destino / arquivo.name)
    quadro_de_artigos().to_csv(destino / "artigos.csv", index=False)
    consulta = sorted((origem / "artigos").glob("arxiv_*.json"))[-1]
    (destino / "artigos").mkdir(exist_ok=True)
    shutil.copy2(consulta, destino / "artigos" / consulta.name)
    for tipo in REAJUSTES:
        for aluno in REAJUSTES[tipo]["alunos"]:
            for s in SEMENTES:
                pasta = pasta_do_reajuste(tipo, aluno, s)
                alvo = destino / "reajustes" / tipo / aluno / f"semente_{s}"
                alvo.mkdir(parents=True, exist_ok=True)
                for nome in ("historico.csv", "meta.json", "previsoes_dev.parquet", "previsoes_dev_antes.parquet"):
                    if (pasta / nome).exists():
                        shutil.copy2(pasta / nome, alvo / nome)
                shutil.copytree(pasta / "epocas", alvo / "epocas", dirs_exist_ok=True)
    for arquivo in sorted((origem / "dinamica").glob("*/*/semente_*/*/previsoes_dev.parquet")) + sorted((origem / "dinamica").glob("*/*/semente_*/*/meta.json")) \
            + sorted((origem / "estatica").glob("*/*/semente_*/*/previsoes_dev.parquet")) + sorted((origem / "estatica").glob("*/*/semente_*/*/meta.json")) \
            + sorted((origem / "estatica").glob("*/*/semente_*/*/teste/previsoes_test.parquet")):
        alvo = destino / arquivo.relative_to(origem)
        alvo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arquivo, alvo)
    (destino / "impressoes").mkdir(exist_ok=True)
    for arquivo in sorted((origem / "impressoes").glob("*")):
        shutil.copy2(arquivo, destino / "impressoes" / arquivo.name)
    (destino / "codigo").mkdir(exist_ok=True)
    for arquivo in (Path(__file__), RAIZ / "notebooks" / "construtores" / "build_nb12.py", RAIZ / "tests" / "test_etapa12.py"):
        if arquivo.exists():
            shutil.copy2(arquivo, destino / "codigo" / arquivo.name)
    return destino


def manifesto() -> Path:
    """O que a entrega contém, com a impressão digital de cada arquivo, e de onde as contas leram."""
    import hashlib

    def sha(caminho: Path) -> str:
        return hashlib.sha256(caminho.read_bytes()).hexdigest()

    destino = pasta_da_entrega()
    arquivos = sorted(a for a in destino.rglob("*") if a.is_file() and a.name != "manifesto.json")
    leu = ["entregas/10a_encoder/previsoes", "entregas/10b_encoder_poda/modelos e data/execucoes/10b_encoder_poda (respostas do controle sem poda e antes do reajuste)",
           "entregas/10c_encoder_quantizacao/previsoes", "entregas/11b_alunos_do_professor_maior/modelos", "entregas/11c_cenario_b_ate_o_fim/previsoes",
           "data/execucoes/11c_cenario_b_ate_o_fim/poda (modelos podados e reajustados)", "entregas/05a_professor_fechado e entregas/11a_professor_aberto_maior (rótulos dos professores)",
           "entregas/10a_encoder/modelos (modelos de partida)"]
    _gravar_json(destino / "manifesto.json", {
        "etapa": ETAPA, "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "o_que_e": "Investigação depois dos resultados (diagnóstico e exploração). Nenhuma decisão das etapas anteriores muda.",
        "plano": {"arquivo": "plano.json", "sha256": sha(destino / "plano.json")},
        "leu": leu,
        "saidas": {str(a.relative_to(destino)): {"sha256": sha(a) if a.stat().st_size <= 64 * 2**20 else None, "bytes": a.stat().st_size} for a in arquivos},
        "cache": f"data/execucoes/{ETAPA}/ (os reajustes com os modelos, os ONNX de 32 bits refeitos, as quantizações de diagnóstico, as contas e os logs)",
    })
    return destino / "manifesto.json"


# ==========================================================================
# Linha de comando
# ==========================================================================


def _linha_de_comando() -> None:
    comandos = {
        "reajustes": rodar_reajustes,
        "treino": rodar_treino,
        "contas": calcular,
        "artigos": conferir_artigos,
        "entregar": entregar,
        "manifesto": manifesto,
        "teste": leitura_no_teste,
        "dinamica": rodar_dinamica,
        "faixas": rodar_faixas,
        "sensibilidade": rodar_sensibilidade,
        "calibracoes": rodar_calibracoes,
        "por_camada": rodar_sensibilidade_por_camada_cli,
        "exclusoes": rodar_exclusoes,
        "acumulado": lambda: (rodar_acumulado(), rodar_um_de_fora()),
        "estatica": lambda: (rodar_sensibilidade_por_camada_cli(), rodar_exclusoes(), rodar_calibracoes()),
    }
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("comando", choices=sorted(comandos))
    comandos[parser.parse_args().comando]()


if __name__ == "__main__":
    _linha_de_comando()
