"""Etapa 13 (piloto, exploratório): pré-treinar a rede do zero só com a perda de máscara.

A rede do projeto já treina três perdas juntas (intenção, entidades e máscara), mas só vê as
11.514 frases do MASSIVE. Aqui a mesma rede, com o mesmo vocabulário de features, é antes
treinada só com a máscara em frases da Wikipédia em português; depois o teto e o aluno A são
treinados com a receita de sempre, partindo desses pesos. Só o ponto de partida muda.

    uv run python -m especialista.experimentos.etapa13 regras
    uv run python -m especialista.experimentos.etapa13 preparar
    uv run python -m especialista.experimentos.etapa13 pretreinar --minutos 55
    uv run python -m especialista.experimentos.etapa13 ajustar --variante teto --semente 0
    uv run python -m especialista.experimentos.etapa13 calcular
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import carregar_previsoes, ouro, salvar_previsoes
from especialista.caminhos import DADOS_BRUTOS, ENTREGAS, EXECUCOES
from especialista.dados import massive
from especialista.experimentos import etapa6, execucao
from especialista.modelos.diet import Config, SistemaDiet
from especialista.modelos.diet.features import Featurizador
from especialista.modelos.diet.modelo import Diet

ETAPA = "13_pretreino_piloto"
PASTA = EXECUCOES / ETAPA
WIKIPEDIA = DADOS_BRUTOS / "wikipedia_pt" / "20231101.pt"
FRASES = PASTA / "frases.txt"
PRETREINO = PASTA / "pretreino"
SEMENTES = (0, 1, 2)
DECISAO_DA_REDE = ENTREGAS / "04b_intencao" / "decisao.json"
VOCABULARIO = ENTREGAS / "04b_intencao" / "modelos" / "semente_0" / "features.json"   # o mesmo de todos os alunos e do teto
CABECAS_DA_TAREFA = ("cls_para_rotulo.", "vetor_da_intencao", "emissao.", "crf.")     # não vêm do pré-treino
CONTROLES = {
    "teto": lambda s: ENTREGAS / "04b_intencao" / "modelos" / f"semente_{s}",
    "aluno_a": lambda s: ENTREGAS / "06_destilacao" / "modelos" / "aluno_a" / f"semente_{s}",
}

REGRAS = {
    "etapa": ETAPA,
    "natureza": "piloto exploratório, com prazo de uma manhã: uma configuração de pré-treino, sem busca; nenhuma decisão das etapas anteriores muda",
    "pergunta": "Pré-treinar a rede do zero só com a perda de máscara, em texto de português sem rótulo, melhora o teto e o aluno A?",
    "o_que_muda": "só o ponto de partida dos pesos da tabela de features, da projeção, do encoder e da cabeça de máscara; as cabeças de intenção e de entidades começam do zero",
    "o_que_nao_muda": "a arquitetura, o vocabulário de features (o do treino do MASSIVE, 25.202 features), a receita da Etapa 4b, os dados e os rótulos de cada sistema, as 3 sementes",
    "corpus": "Wikipédia em português (wikimedia/wikipedia, 20231101.pt): frases de 4 a 25 palavras, em minúsculas e sem pontuação, como os pedidos do MASSIVE; no máximo 6 frases por artigo",
    "pretreino": "só a perda de máscara do DIET (15% das palavras; erradas = as outras escondidas do lote), Adam 1e-3 com aquecimento de 1.000 passos e decaimento linear até 10%, lotes de 256 frases de comprimento parecido, uma passada, limitada pelo tempo",
    "controle": "os modelos já entregues: teto (entregas/04b_intencao/modelos) e aluno A (entregas/06_destilacao/modelos/aluno_a), 3 sementes cada",
    "medida": "exact match; diferença pareada pré-treinado − controle, média de 3 sementes, bootstrap por enunciado com 2.000 reamostragens e semente 0 (qualidade.comparar_grupos), IC95; 'distinguível' = IC sem o zero",
    "decisao_em": "dev (rótulos humanos)",
    "leitura_no_teste": "uma por sistema (teto e aluno A), a mesma diferença, relatada qualquer que seja o resultado em dev",
    "o_que_fica_de_fora": "poda, quantização, alunos do cenário B, outro corpus, outra taxa de ajuste, vocabulário de features maior",
}


# --------------------------------------------------------------------------
# O corpus
# --------------------------------------------------------------------------

_FIM_DE_FRASE = re.compile(r"(?<=[.!?])\s+")
_NAO_PALAVRA = re.compile(r"[^\w\s\-']|_", re.UNICODE)


def frases_do_texto(texto: str, maximo: int = 20, minimo_de_palavras: int = 4, maximo_de_palavras: int = 25) -> list[str]:
    """Frases de um artigo, no formato dos pedidos do MASSIVE: minúsculas, sem pontuação, palavras separadas por espaço."""
    achadas = []
    for paragrafo in texto.split("\n"):
        if len(paragrafo) < 40:                      # títulos de seção e itens de lista
            continue
        for frase in _FIM_DE_FRASE.split(paragrafo):
            palavras = [p.strip("-'") for p in _NAO_PALAVRA.sub(" ", frase.lower()).split()]
            palavras = [p for p in palavras if p]
            if not minimo_de_palavras <= len(palavras) <= maximo_de_palavras:
                continue
            if sum(p.isalpha() or "-" in p for p in palavras) < 0.8 * len(palavras):
                continue
            achadas.append(" ".join(palavras))
            if len(achadas) == maximo:
                return achadas
    return achadas


def _frases_do_arquivo(arquivo: Path) -> list[str]:
    textos = pd.read_parquet(arquivo, columns=["text"])["text"]
    return [frase for texto in textos for frase in frases_do_texto(texto)]


def preparar(limite: int = 12_000_000) -> Path:
    if FRASES.exists():
        return FRASES
    arquivos = sorted(WIKIPEDIA.glob("*.parquet"))
    with Pool(len(arquivos)) as fila:
        frases = [f for parte in fila.map(_frases_do_arquivo, arquivos) for f in parte]
    rng = np.random.default_rng(0)
    ordem = rng.permutation(len(frases))[:limite]
    PASTA.mkdir(parents=True, exist_ok=True)
    FRASES.write_text("\n".join(frases[i] for i in ordem) + "\n", encoding="utf-8")
    palavras = sum(len(frases[i].split()) for i in ordem[:100_000]) / min(100_000, len(ordem))
    (PASTA / "frases.json").write_text(json.dumps(
        {"frases_achadas": len(frases), "frases_gravadas": len(ordem), "palavras_por_frase_media": palavras,
         "arquivos": [a.name for a in arquivos]}, ensure_ascii=False, indent=1), encoding="utf-8")
    return FRASES


# --------------------------------------------------------------------------
# O pré-treino: só a máscara
# --------------------------------------------------------------------------

class _Montador:
    """Roda nos processos auxiliares: de uma lista de frases para um `Lote`."""

    def __init__(self, featurizador: Featurizador):
        self.featurizador = featurizador

    def __call__(self, frases: list[str]):
        return self.featurizador.montar([f.split() for f in frases])


def _lotes(n: int, comprimentos: np.ndarray, tamanho: int, rng: np.random.Generator, por_grupo: int = 50) -> list[np.ndarray]:
    ordem, lotes = rng.permutation(n), []
    for inicio in range(0, n, tamanho * por_grupo):
        pedaco = ordem[inicio : inicio + tamanho * por_grupo]
        pedaco = pedaco[np.argsort(comprimentos[pedaco], kind="stable")]
        lotes.extend(pedaco[i : i + tamanho] for i in range(0, len(pedaco), tamanho))
    return [lotes[i] for i in rng.permutation(len(lotes))]


def config_do_pretreino() -> Config:
    partida = json.loads(DECISAO_DA_REDE.read_text(encoding="utf-8"))["config"]
    return Config(**{**partida, "peso_intencao": 0.0, "peso_entidades": 0.0, "peso_mascara": 1.0})


def pretreinar(minutos: float, pasta: Path = PRETREINO, frases_max: int | None = None, tamanho: int = 256,
               taxa: float = 1e-3, aquecimento: int = 1000, auxiliares: int = 10, semente: int = 0) -> Path:
    """Treina a rede só com a perda de máscara, por no máximo `minutos`, e grava os pesos."""
    if (pasta / "meta.json").exists():
        return pasta
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    frases = FRASES.read_text(encoding="utf-8").splitlines()[:frases_max]
    comprimentos = np.array([f.count(" ") + 1 for f in frases])
    featurizador = Featurizador.carregar(VOCABULARIO)
    _, rotulos = massive.carregar_entrega()
    config = config_do_pretreino()
    modelo = Diet(config, len(featurizador), len(rotulos["intents"]), rotulos["tags"]).to("cuda")
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)

    lotes = _lotes(len(frases), comprimentos, tamanho, rng)
    carga = torch.utils.data.DataLoader(
        frases, batch_sampler=lotes, collate_fn=_Montador(featurizador), num_workers=auxiliares, prefetch_factor=8)
    pasta.mkdir(parents=True, exist_ok=True)
    registro, soma, acertos, vistos = [], 0.0, 0, 0
    inicio = time.perf_counter()
    limite = minutos * 60
    passo = 0
    modelo.train()
    for passo, lote in enumerate(carga, 1):
        decorrido = time.perf_counter() - inicio
        if decorrido > limite:
            passo -= 1
            break
        # aquecimento por passos; decaimento linear pelo tempo (ou pela passada, o que acabar antes), até 10%
        fracao = max(decorrido / limite, passo / len(lotes))
        fator = min(1.0, passo / aquecimento) * (1.0 - 0.9 * fracao)
        for grupo in otimizador.param_groups:
            grupo["lr"] = taxa * fator
        perdas = modelo.perda(lote.para("cuda"))
        otimizador.zero_grad()
        perdas["total"].backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
        otimizador.step()
        soma += perdas["mascara"].item()
        vistos += 1
        if passo % 500 == 0:
            linha = {"passo": passo, "frases": passo * tamanho, "perda_mascara": soma / vistos, "taxa": taxa * fator,
                     "minutos": decorrido / 60}
            registro.append(linha)
            print({k: round(v, 4) for k, v in linha.items()}, flush=True)
            soma, vistos = 0.0, 0
        if passo % 5000 == 0:
            torch.save(modelo.state_dict(), pasta / "pesos_parcial.pt")
    duracao = time.perf_counter() - inicio
    torch.save(modelo.state_dict(), pasta / "pesos.pt")
    featurizador.salvar(pasta / "features.json")
    pd.DataFrame(registro).to_csv(pasta / "historico.csv", index=False)
    (pasta / "meta.json").write_text(json.dumps({
        "passos": passo, "frases_vistas": passo * tamanho, "frases_no_corpus": len(frases), "passada": passo / len(lotes),
        "minutos": duracao / 60, "taxa": taxa, "aquecimento": aquecimento, "tamanho_do_lote": tamanho, "semente": semente,
        "config": config.como_dict(), "perda_mascara_inicial": registro[0]["perda_mascara"] if registro else None,
        "perda_mascara_final": registro[-1]["perda_mascara"] if registro else None,
        "acaso": math.log(0.15 * tamanho * float(comprimentos.mean())),
        "dispositivo": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return pasta


# --------------------------------------------------------------------------
# O ajuste: a receita de sempre, partindo do pré-treino
# --------------------------------------------------------------------------

def pesos_do_pretreino(pasta: Path = PRETREINO) -> dict:
    estado = torch.load(pasta / "pesos.pt", map_location="cpu", weights_only=True)
    return {nome: valor for nome, valor in estado.items() if not nome.startswith(CABECAS_DA_TAREFA)}


@contextlib.contextmanager
def partindo_do_pretreino(pasta: Path = PRETREINO):
    """Dentro do bloco, todo `SistemaDiet.novo` de `execucao` começa com os pesos do pré-treino."""
    pesos = pesos_do_pretreino(pasta)
    vocabulario = Featurizador.carregar(pasta / "features.json").vocabulario

    class ComPretreino(SistemaDiet):
        @classmethod
        def novo(cls, *args, **kwargs):
            sistema = super().novo(*args, **kwargs)
            if sistema.featurizador.vocabulario != vocabulario:
                raise ValueError("O vocabulário de features do sistema não é o do pré-treino.")
            faltam, sobram = sistema.modelo.load_state_dict(pesos, strict=False)
            if sobram or not all(nome.startswith(CABECAS_DA_TAREFA) for nome in faltam):
                raise ValueError(f"Pesos do pré-treino não encaixam: faltam {faltam}, sobram {sobram}.")
            return sistema

    original = execucao.SistemaDiet
    execucao.SistemaDiet = ComPretreino
    try:
        yield
    finally:
        execucao.SistemaDiet = original


def definicao(variante: str) -> dict:
    if variante == "teto":
        partida = json.loads(DECISAO_DA_REDE.read_text(encoding="utf-8"))
        return {"nome": "Teto, partindo do pré-treino", "config": partida["config"], "receita": partida["receita"]}
    return {**etapa6.definicao(variante), "nome": etapa6.VARIANTES[variante]["nome"] + ", partindo do pré-treino"}


def ajustar(variante: str, semente: int) -> Path:
    """Treina `teto` ou `aluno_a` com a receita e os dados de sempre, partindo do pré-treino; grava dev e teste."""
    dados = execucao.dados_humanos if variante == "teto" else (lambda: etapa6.dados_da_variante(variante))
    with partindo_do_pretreino():
        pasta = execucao.executar(ETAPA, f"{variante}__pretreino", semente, definicao(variante), dados=dados)
    if not (pasta / "previsoes_test.parquet").exists():
        particoes, _ = massive.carregar_entrega()
        sistema = SistemaDiet.carregar(pasta / "modelo", "cuda" if torch.cuda.is_available() else "cpu")
        for particao in ("dev", "test"):
            parte = particoes[particao]
            salvar_previsoes(pasta / f"previsoes_{particao}_humano.parquet" if particao == "dev" else pasta / "previsoes_test.parquet",
                             parte["id"], sistema.prever(list(parte["utt"])))
    return pasta


# --------------------------------------------------------------------------
# As contas
# --------------------------------------------------------------------------

def calcular() -> pd.DataFrame:
    particoes, _ = massive.carregar_entrega()
    linhas = []
    for variante in CONTROLES:
        for particao in ("dev", "test"):
            parte = particoes[particao]
            gabarito, textos = ouro(parte), list(parte["utt"])
            controle = [SistemaDiet.carregar(CONTROLES[variante](s), "cuda").prever(textos) for s in SEMENTES]
            arquivo = "previsoes_dev_humano.parquet" if particao == "dev" else "previsoes_test.parquet"
            novo = [carregar_previsoes(execucao.pasta_da_execucao(ETAPA, f"{variante}__pretreino", s) / arquivo)[1] for s in SEMENTES]
            tabela = qualidade.comparar_grupos(gabarito, controle, novo)
            for metrica, r in tabela.iterrows():
                linhas.append({"sistema": variante, "particao": particao, "metrica": metrica, "controle": r["a"],
                               "pretreino": r["b"], "diferenca": r["diferenca"], "ic_min": r["ic_min"], "ic_max": r["ic_max"],
                               "distinguivel": bool(r["distinguivel"])})
    resultado = pd.DataFrame(linhas)
    resultado.to_csv(PASTA / "comparacoes.csv", index=False)
    return resultado


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser()
    comandos = argumentos.add_subparsers(dest="comando", required=True)
    comandos.add_parser("regras")
    comandos.add_parser("preparar")
    p = comandos.add_parser("pretreinar")
    p.add_argument("--minutos", type=float, required=True)
    p.add_argument("--pasta", type=Path, default=PRETREINO)
    p.add_argument("--frases", type=int, default=None)
    p = comandos.add_parser("ajustar")
    p.add_argument("--variante", required=True, choices=sorted(CONTROLES))
    p.add_argument("--semente", type=int, required=True)
    comandos.add_parser("calcular")
    a = argumentos.parse_args()
    if a.comando == "regras":
        PASTA.mkdir(parents=True, exist_ok=True)
        destino = PASTA / "regras.json"
        if destino.exists():
            raise SystemExit("As regras já estão gravadas; não são regravadas.")
        destino.write_text(json.dumps({**REGRAS, "gravado_em": time.strftime("%Y-%m-%d %H:%M:%S %z")}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("regras gravadas", destino)
    elif a.comando == "preparar":
        print(preparar(), (PASTA / "frases.json").read_text(encoding="utf-8"))
    elif a.comando == "pretreinar":
        print("FIM:", pretreinar(a.minutos, a.pasta, a.frases))
    elif a.comando == "ajustar":
        pasta = ajustar(a.variante, a.semente)
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        print(a.variante, a.semente, f"época {meta['melhor_epoca']}", {k: round(v, 4) for k, v in meta["dev"].items()})
    elif a.comando == "calcular":
        r = calcular()
        print(r[r.metrica.isin(["exact_match", "intent_acc", "slot_f1"])].to_string())
