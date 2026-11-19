"""Etapa 13, rodada 2 (piloto, exploratório): por que o pré-treino da rodada 1 não rendeu?

Reaproveita `etapa13` (corpus, pré-treino só com a máscara, comparações) e testa:

- (1) o ajuste apagar o pré-treino: taxa menor no ajuste, taxa menor só nas partes pré-treinadas,
  aquecimento da taxa no começo do ajuste. Usa o pré-treino da rodada 1.
- (2)+(3) vocabulário de features maior (MASSIVE + Wikipédia) com pré-treino mais longo (2 passadas),
  contra o controle justo: a mesma rede, com o mesmo vocabulário, sem pré-treino.

Nada da rodada 1 é regravado. Tudo fica em `data/execucoes/13_pretreino_piloto/rodada2/`.

    uv run python -m especialista.experimentos.etapa13b regras --bloco A
    uv run python -m especialista.experimentos.etapa13b vocabulario
    uv run python -m especialista.experimentos.etapa13b pretreinar --nome vocabg_2passadas --vocab vocab_grande --passadas 2 --minutos 35
    uv run python -m especialista.experimentos.etapa13b ajustar --variante lr3e-4 --sistema teto --semente 0
    uv run python -m especialista.experimentos.etapa13b calcular --particao dev
    uv run python -m especialista.experimentos.etapa13b calcular --particao test --finalistas teto:X aluno_a:Y
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import carregar_previsoes, ouro, salvar_previsoes
from especialista.dados import massive
from especialista.experimentos import etapa6, etapa13, execucao
from especialista.modelos.diet import SistemaDiet
from especialista.modelos.diet.features import Featurizador
from especialista.modelos.diet.modelo import Diet
from especialista.modelos.vocabulario import features_do_texto, ngramas

ETAPA = "13_pretreino_piloto/rodada2"          # execucao.pasta_da_execucao junta EXECUCOES / ETAPA / variante / semente_N
PASTA = etapa13.PASTA / "rodada2"
SEMENTES = etapa13.SEMENTES

# --------------------------------------------------------------------------
# As variantes. Cada uma diz de onde vêm os pesos, qual vocabulário e como é o ajuste.
# --------------------------------------------------------------------------

PRETREINO_R1 = str(etapa13.PRETREINO)
VARIANTES: dict[str, dict] = {
    # bloco A: o pré-treino da rodada 1, outra receita de ajuste; controle = modelos entregues
    "r1_lr3e-4": {"bloco": "A", "pretreino": PRETREINO_R1, "vocab": None, "taxa": 3e-4, "fator_pretreinadas": 1.0, "aquecimento": 0,
                  "descricao": "pré-treino da rodada 1; ajuste com taxa 3e-4 em tudo (em vez de 1e-3)"},
    "r1_pre_x0.1": {"bloco": "A", "pretreino": PRETREINO_R1, "vocab": None, "taxa": 1e-3, "fator_pretreinadas": 0.1, "aquecimento": 0,
                    "descricao": "pré-treino da rodada 1; partes pré-treinadas com taxa 1e-4, cabeças novas com 1e-3"},
    "r1_aquec500": {"bloco": "A", "pretreino": PRETREINO_R1, "vocab": None, "taxa": 1e-3, "fator_pretreinadas": 1.0, "aquecimento": 500,
                    "descricao": "pré-treino da rodada 1; taxa 1e-3 com aquecimento linear nos primeiros 500 passos (~4 épocas)"},
    # bloco B: vocabulário grande; controle = mesma rede e vocabulário, sem pré-treino
    "vocabg_controle": {"bloco": "B", "pretreino": None, "vocab": "vocab_grande", "taxa": 1e-3, "fator_pretreinadas": 1.0, "aquecimento": 0,
                        "descricao": "vocabulário grande (MASSIVE + Wikipédia), SEM pré-treino: o controle justo do bloco B"},
    "vocabg_2p": {"bloco": "B", "pretreino": "pretreino_vocabg_2p", "vocab": "vocab_grande", "taxa": 1e-3, "fator_pretreinadas": 1.0, "aquecimento": 0,
                  "descricao": "vocabulário grande, pré-treino de 2 passadas; receita de sempre"},
}
# Variantes do bloco C (combinação), acrescentadas depois do bloco A: o pré-treino do bloco B com a receita de ajuste
# que se saiu melhor no bloco A. Gravadas nas regras do bloco C antes de rodar.
for _nome, _base in (("vocabg_2p_lr3e-4", "r1_lr3e-4"), ("vocabg_2p_pre_x0.1", "r1_pre_x0.1"), ("vocabg_2p_aquec500", "r1_aquec500")):
    VARIANTES[_nome] = {**VARIANTES[_base], "bloco": "C", "pretreino": "pretreino_vocabg_2p", "vocab": "vocab_grande",
                        "descricao": "vocabulário grande, pré-treino de 2 passadas; ajuste como em " + _base}
for _nome, _base in (("vocabg_controle_lr3e-4", "r1_lr3e-4"), ("vocabg_controle_aquec500", "r1_aquec500")):
    VARIANTES[_nome] = {**VARIANTES[_base], "bloco": "C", "pretreino": None, "vocab": "vocab_grande",
                        "descricao": "vocabulário grande, SEM pré-treino; ajuste como em " + _base + " (controle da mesma receita)"}

# Bloco D (acrescentado pelo orquestrador às 11:00, depois de fechados os blocos A a C): mais pré-treino, com o
# vocabulário do MASSIVE (a rede não cresce) e a placa só para ele. Gravado em regras_bloco_D.json antes de rodar.
VARIANTES["massive_3p"] = {"bloco": "D", "pretreino": "pretreino_massive_3p", "vocab": None, "taxa": 1e-3, "fator_pretreinadas": 1.0,
                           "aquecimento": 0, "descricao": "vocabulário do MASSIVE, pré-treino de até 3 passadas; receita de sempre"}


def _caminho(nome: str | None) -> Path | None:
    if nome is None:
        return None
    p = Path(nome)
    return p if p.is_absolute() else PASTA / nome


def controle_de(variante: str) -> str:
    """'entregue' (os modelos das Etapas 4b e 6) ou o nome da variante-controle."""
    v = VARIANTES[variante]
    if v["vocab"] is None:
        return "entregue"
    if v["pretreino"] is None:
        return "entregue"            # o controle sem pré-treino também é comparado com o modelo entregue
    return "vocabg_controle"


# --------------------------------------------------------------------------
# Vocabulário grande
# --------------------------------------------------------------------------

def _contar_palavras(linhas: list[str]) -> Counter:
    c = Counter()
    for linha in linhas:
        c.update(linha.split())
    return c


def vocabulario(palavras: int = 60_000, n_gramas: int = 20_000, destino: Path = PASTA / "vocab_grande") -> Path:
    """MASSIVE (25.202 features, na mesma ordem) + as `palavras` mais frequentes da Wikipédia
    + os `n_gramas` mais frequentes (por ocorrência de palavra) que ainda não estão lá."""
    if (destino / "features.json").exists():
        return destino
    inicio = time.perf_counter()
    linhas = etapa13.FRASES.read_text(encoding="utf-8").splitlines()
    pedacos = [linhas[i : i + 200_000] for i in range(0, len(linhas), 200_000)]
    with Pool(20) as fila:
        contagem = Counter()
        for c in fila.imap_unordered(_contar_palavras, pedacos):
            contagem.update(c)
    del linhas, pedacos
    massive_vocab = Featurizador.carregar(etapa13.VOCABULARIO)
    base = list(massive_vocab.vocabulario)
    conhecidos = set(base)
    novas_palavras = []
    for palavra, _ in contagem.most_common():
        if len(novas_palavras) >= palavras:
            break
        if len(palavra) > 30:
            continue
        f = f"p:{palavra}"
        if f not in conhecidos:
            novas_palavras.append(f)
            conhecidos.add(f)
    contagem_ng = Counter()
    for palavra, n in contagem.items():
        if n < 5 or len(palavra) > 30:
            continue
        for g in set(ngramas(palavra, massive_vocab.n_max)):
            contagem_ng[g] += n
    novos_ng = []
    for g, _ in contagem_ng.most_common():
        if len(novos_ng) >= n_gramas:
            break
        f = f"g:{g}"
        if f not in conhecidos:
            novos_ng.append(f)
            conhecidos.add(f)
    final = base + novas_palavras + novos_ng
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "features.json").write_text(json.dumps({"n_max": massive_vocab.n_max, "features": final}, ensure_ascii=False), encoding="utf-8")
    (destino / "meta.json").write_text(json.dumps({
        "features_massive": len(base), "palavras_novas_da_wikipedia": len(novas_palavras), "ngramas_novos_da_wikipedia": len(novos_ng),
        "total": len(final), "palavras_distintas_no_corpus": len(contagem), "corpus": str(etapa13.FRASES),
        "regra": "MASSIVE primeiro (mesma ordem); depois as palavras mais frequentes do corpus (até 30 caracteres); "
                 "depois os n-gramas (1 a 5 caracteres, com ^ e $) mais frequentes, contando ocorrências de palavras vistas >= 5 vezes",
        "segundos": time.perf_counter() - inicio,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino


# --------------------------------------------------------------------------
# Pré-treino com vocabulário e número de passadas à escolha
# --------------------------------------------------------------------------

def pretreinar(nome: str, vocab: str | None, passadas: int, minutos: float, tamanho: int = 256, taxa: float = 1e-3,
               aquecimento: int = 1000, auxiliares: int = 10, semente: int = 0) -> Path:
    pasta = PASTA / nome
    if (pasta / "meta.json").exists():
        return pasta
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    frases = etapa13.FRASES.read_text(encoding="utf-8").splitlines()
    comprimentos = np.array([f.count(" ") + 1 for f in frases])
    featurizador = Featurizador.carregar(_caminho(vocab) / "features.json" if vocab else etapa13.VOCABULARIO)
    _, rotulos = massive.carregar_entrega()
    config = etapa13.config_do_pretreino()
    modelo = Diet(config, len(featurizador), len(rotulos["intents"]), rotulos["tags"]).to("cuda")
    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa)
    por_passada = math.ceil(len(frases) / tamanho)
    total = por_passada * passadas
    pasta.mkdir(parents=True, exist_ok=True)
    registro, soma, vistos, passo = [], 0.0, 0, 0
    inicio = time.perf_counter()
    limite = minutos * 60
    parou_pelo_tempo = False
    modelo.train()
    for passada in range(passadas):
        lotes = etapa13._lotes(len(frases), comprimentos, tamanho, rng)
        carga = torch.utils.data.DataLoader(frases, batch_sampler=lotes, collate_fn=etapa13._Montador(featurizador),
                                            num_workers=auxiliares, prefetch_factor=8)
        for lote in carga:
            decorrido = time.perf_counter() - inicio
            if decorrido > limite:
                parou_pelo_tempo = True
                break
            passo += 1
            fracao = max(decorrido / limite, passo / total)
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
                linha = {"passo": passo, "passada": passo / por_passada, "perda_mascara": soma / vistos, "taxa": taxa * fator,
                         "minutos": decorrido / 60}
                registro.append(linha)
                print({k: round(v, 4) for k, v in linha.items()}, flush=True)
                soma, vistos = 0.0, 0
            if passo % 10000 == 0:
                torch.save(modelo.state_dict(), pasta / "pesos_parcial.pt")
        del carga
        if parou_pelo_tempo:
            break
    duracao = time.perf_counter() - inicio
    torch.save(modelo.state_dict(), pasta / "pesos.pt")
    featurizador.salvar(pasta / "features.json")
    pd.DataFrame(registro).to_csv(pasta / "historico.csv", index=False)
    (pasta / "meta.json").write_text(json.dumps({
        "nome": nome, "vocab": vocab or str(etapa13.VOCABULARIO), "n_features": len(featurizador),
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "parametros_por_parte": modelo.parametros_por_parte(),
        "passos": passo, "frases_vistas": passo * tamanho, "frases_no_corpus": len(frases), "passadas_pedidas": passadas,
        "passadas_feitas": passo / por_passada, "parou_pelo_tempo": parou_pelo_tempo,
        "minutos": duracao / 60, "taxa": taxa, "aquecimento": aquecimento, "tamanho_do_lote": tamanho, "semente": semente,
        "config": config.como_dict(), "perda_mascara_inicial": registro[0]["perda_mascara"] if registro else None,
        "perda_mascara_final": registro[-1]["perda_mascara"] if registro else None,
        "acaso": math.log(0.15 * tamanho * float(comprimentos.mean())),
        "dispositivo": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return pasta


# --------------------------------------------------------------------------
# O ajuste
# --------------------------------------------------------------------------

_PRETREINADAS: set[int] = set()     # ids dos parâmetros que vieram do pré-treino
_AJUSTE = {"fator": 1.0, "aquecimento": 0}


class _AdamDoAjuste(torch.optim.Adam):
    """Adam com taxa própria para as partes pré-treinadas e aquecimento linear por passos.
    `treino.treinar` cria `torch.optim.Adam(modelo.parameters(), lr=receita.taxa)`; esta classe toma o lugar dela."""

    def __init__(self, params, lr):
        params = list(params)
        novas = [p for p in params if id(p) not in _PRETREINADAS]
        velhas = [p for p in params if id(p) in _PRETREINADAS]
        grupos = [{"params": novas, "lr": lr, "base": lr}]
        if velhas:
            grupos.append({"params": velhas, "lr": lr * _AJUSTE["fator"], "base": lr * _AJUSTE["fator"]})
        super().__init__(grupos, lr=lr)
        self._passos = 0
        self._aplicar_taxa()

    def _aplicar_taxa(self):
        aq = _AJUSTE["aquecimento"]
        mult = min(1.0, (self._passos + 1) / aq) if aq > 0 else 1.0
        for g in self.param_groups:
            g["lr"] = g["base"] * mult

    def step(self, closure=None):
        self._aplicar_taxa()
        resultado = super().step(closure)
        self._passos += 1
        return resultado


def _sistema_da_variante(v: dict):
    vocab = _caminho(v["vocab"])
    pretreino = _caminho(v["pretreino"])

    class Sistema(SistemaDiet):
        @classmethod
        def novo(cls, treino, rotulos, config, semente=0, contagem_minima=2, dispositivo="cpu"):
            torch.manual_seed(semente)
            if vocab is not None:
                featurizador = Featurizador.carregar(vocab / "features.json")
            else:
                featurizador = Featurizador.ajustar(treino["tokens"], contagem_minima=contagem_minima)
            modelo = Diet(config, len(featurizador), len(rotulos["intents"]), rotulos["tags"])
            if pretreino is not None:
                if Featurizador.carregar(pretreino / "features.json").vocabulario != featurizador.vocabulario:
                    raise ValueError("O vocabulário de features do sistema não é o do pré-treino.")
                pesos = etapa13.pesos_do_pretreino(pretreino)
                faltam, sobram = modelo.load_state_dict(pesos, strict=False)
                if sobram or not all(n.startswith(etapa13.CABECAS_DA_TAREFA) for n in faltam):
                    raise ValueError(f"Pesos do pré-treino não encaixam: faltam {faltam}, sobram {sobram}.")
                _PRETREINADAS.update(id(p) for n, p in modelo.named_parameters() if n in pesos)
            return cls(modelo, featurizador, rotulos["intents"], rotulos["tags"], dispositivo)

    return Sistema


def definicao(sistema: str, variante: str) -> dict:
    v = VARIANTES[variante]
    base = etapa13.definicao(sistema)
    return {**base, "nome": f"{sistema}, rodada 2, {variante}: {v['descricao']}",
            "receita": {**base.get("receita", {}), "taxa": v["taxa"]}}


def ajustar(variante: str, sistema: str, semente: int) -> Path:
    v = VARIANTES[variante]
    _AJUSTE.update(fator=v["fator_pretreinadas"], aquecimento=v["aquecimento"])
    dados = execucao.dados_humanos if sistema == "teto" else (lambda: etapa6.dados_da_variante(sistema))
    original_sistema, original_adam = execucao.SistemaDiet, torch.optim.Adam
    execucao.SistemaDiet = _sistema_da_variante(v)
    torch.optim.Adam = _AdamDoAjuste
    try:
        pasta = execucao.executar(ETAPA, f"{sistema}__{variante}", semente, definicao(sistema, variante), dados=dados)
    finally:
        execucao.SistemaDiet, torch.optim.Adam = original_sistema, original_adam
    manifesto = pasta.parent / "manifesto.json"
    if not manifesto.exists():
        manifesto.write_text(json.dumps({"sistema": sistema, "variante": variante, **v,
                                         "controle": controle_de(variante)}, ensure_ascii=False, indent=1), encoding="utf-8")
    if not (pasta / "previsoes_dev_humano.parquet").exists():
        _prever(pasta, "dev")
    return pasta


def _prever(pasta: Path, particao: str) -> Path:
    destino = pasta / ("previsoes_dev_humano.parquet" if particao == "dev" else "previsoes_test.parquet")
    if not destino.exists():
        particoes, _ = massive.carregar_entrega()
        s = SistemaDiet.carregar(pasta / "modelo", "cuda")
        parte = particoes[particao]
        salvar_previsoes(destino, parte["id"], s.prever(list(parte["utt"])))
    return destino


# --------------------------------------------------------------------------
# As contas
# --------------------------------------------------------------------------

def _previsoes(sistema: str, variante: str, particao: str) -> list:
    if variante == "entregue":
        particoes, _ = massive.carregar_entrega()
        textos = list(particoes[particao]["utt"])
        return [SistemaDiet.carregar(etapa13.CONTROLES[sistema](s), "cuda").prever(textos) for s in SEMENTES]
    saida = []
    for s in SEMENTES:
        pasta = execucao.pasta_da_execucao(ETAPA, f"{sistema}__{variante}", s)
        saida.append(carregar_previsoes(_prever(pasta, particao))[1])
    return saida


def calcular(particao: str, pares: list[tuple[str, str]]) -> pd.DataFrame:
    particoes, _ = massive.carregar_entrega()
    gabarito = ouro(particoes[particao])
    memoria: dict = {}
    linhas = []
    for sistema, variante in pares:
        controle = controle_de(variante)
        comparacoes = [controle] + (["entregue"] if controle != "entregue" else [])
        for ctrl in comparacoes:
            chave = (sistema, ctrl)
            if chave not in memoria:
                memoria[chave] = _previsoes(sistema, ctrl, particao)
            novo = _previsoes(sistema, variante, particao)
            tabela = qualidade.comparar_grupos(gabarito, memoria[chave], novo)
            for metrica, r in tabela.iterrows():
                linhas.append({"sistema": sistema, "variante": variante, "controle": ctrl, "particao": particao, "metrica": metrica,
                               "valor_controle": r["a"], "valor_variante": r["b"], "diferenca": r["diferenca"],
                               "ic_min": r["ic_min"], "ic_max": r["ic_max"], "distinguivel": bool(r["distinguivel"])})
    return pd.DataFrame(linhas)


def tamanho(variante_ou_pasta: Path) -> dict:
    s = SistemaDiet.carregar(variante_ou_pasta, "cpu")
    p = s.parametros
    return {"parametros": p, "mb_float32": p * 4 / 1e6, "n_features": len(s.featurizador),
            "por_parte": s.modelo.parametros_por_parte()}


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser()
    comandos = argumentos.add_subparsers(dest="comando", required=True)
    comandos.add_parser("vocabulario")
    p = comandos.add_parser("pretreinar")
    p.add_argument("--nome", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--passadas", type=int, default=1)
    p.add_argument("--minutos", type=float, required=True)
    p = comandos.add_parser("ajustar")
    p.add_argument("--variante", required=True, choices=sorted(VARIANTES))
    p.add_argument("--sistema", required=True, choices=sorted(etapa13.CONTROLES))
    p.add_argument("--semente", type=int, required=True)
    p = comandos.add_parser("calcular")
    p.add_argument("--particao", required=True, choices=["dev", "test"])
    p.add_argument("--pares", nargs="+", required=True, help="sistema:variante")
    p.add_argument("--saida", required=True)
    a = argumentos.parse_args()
    if a.comando == "vocabulario":
        print(vocabulario(), (PASTA / "vocab_grande" / "meta.json").read_text(encoding="utf-8"))
    elif a.comando == "pretreinar":
        print("FIM:", pretreinar(a.nome, a.vocab, a.passadas, a.minutos))
    elif a.comando == "ajustar":
        pasta = ajustar(a.variante, a.sistema, a.semente)
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        print(a.sistema, a.variante, a.semente, f"época {meta['melhor_epoca']}", f"{meta['segundos_de_treino']:.0f}s",
              {k: round(v, 4) for k, v in meta["dev"].items()}, flush=True)
    elif a.comando == "calcular":
        if a.particao == "test" and not a.saida.startswith("teste_finalistas"):
            raise SystemExit("O teste só é lido para as finalistas: --saida teste_finalistas*.csv")
        pares = [tuple(x.split(":")) for x in a.pares]
        destino = PASTA / a.saida
        if destino.exists():
            raise SystemExit(f"{destino} já existe; não é regravado.")
        r = calcular(a.particao, pares)
        r.to_csv(destino, index=False)
        m = r[r.metrica.isin(["intent_acc", "slot_f1", "exact_match"])].copy()
        for c in ("valor_controle", "valor_variante", "diferenca", "ic_min", "ic_max"):
            m[c] = (m[c] * 100).round(1)
        print(m.drop(columns=["particao"]).to_string(index=False))
