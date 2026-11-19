"""Etapa 9: comparativo final. Junta o que as Etapas 2 a 8 entregaram, numa tabela, nas cadeias e nas comparações de ponta a ponta.

Nada é treinado aqui, nada usa a GPU e nada chama a API. Tudo sai do que
está em `entregas/`:

    qualidade, parâmetros, tamanho, latência, custo   `resultados.csv`, `eficiencia.csv` e `decisao.json` de cada etapa
    respostas no teste, para as comparações pareadas   `previsoes/*.parquet` de cada etapa

Três coisas não estavam gravadas, e o comando `preparar` as produz na CPU,
em segundos, em `data/execucoes/09_comparativo/`:

    respostas dos sistemas INT8    a Etapa 8 entregou os modelos, não as respostas: cada modelo é carregado da entrega e responde ao teste
    respostas da linha de base     a Etapa 5a gravou as métricas da linha de base com 299 rótulos, não as respostas: ela é treinada de novo
    tamanho do professor aberto    os pesos baixados, medidos no cache do Hugging Face desta máquina (se ainda estiverem lá)

    uv run python -m especialista.experimentos.etapa9 preparar

O que foi produzido de novo é conferido contra o que as etapas gravaram
(`conferir_qualidade`, `conferir_degraus`): se a conta desta etapa e a de uma
etapa anterior discordarem, o notebook para.

As comparações de ponta a ponta estão em `PONTA_A_PONTA`. A lista foi fixada
antes de calcular qualquer uma delas.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, salvar_previsoes
from especialista.caminhos import ENTREGAS, EXECUCOES

ETAPA = "09_comparativo"
SEMENTES = (0, 1, 2)
PARTICAO = "test"
TRES = ("intent_acc", "slot_f1", "exact_match")   # as três métricas das leituras
REAMOSTRAS = 2000                                 # sorteios do bootstrap por enunciado, como em todas as etapas
TOLERANCIA = 1e-9                                 # duas contas iguais só diferem no último dígito do número real

# As três condições em que a latência foi medida. Não são comparáveis entre si sem dizer qual é qual.
API, GPU, CPU = "API pela rede", "GPU local", "CPU local"
COMO_PREPARAR = "uv run python -m especialista.experimentos.etapa9 preparar"

# Os tipos de degrau de uma cadeia e o nome com que cada um aparece nas tabelas e nos gráficos.
DEGRAUS = {
    "professor": "Professor (LLM)",
    "destilado": "Aluno destilado",
    "sem_compressao": "Sem compressão",
    "podado": "Podado e reajustado",
    "onnx": "ONNX, 32 bits",
    "int8": "INT8",
}

# As comparações de ponta a ponta, fixadas antes de olhar os números: (letra, B, A, o que a diferença B − A responde).
# As chaves são as de `catalogo`. Qualquer comparação fora desta lista tem de ser marcada como exploratória.
PONTA_A_PONTA = (
    ("a", "aluno_a/int8", "claude", "o modelo final contra o seu professor"),
    ("b", "aluno_a/int8", "aluno_a", "o custo de toda a compressão: poda, exportação e 8 bits"),
    ("c", "aluno_a/int8", "teto", "o que custa não ter os rótulos humanos do treino"),
    ("d", "teto/int8", "teto", "a compressão sozinha, sem o efeito do professor"),
    ("e1", "aluno_b2/int8", "qwen", "o final do cenário B contra o seu professor"),
    ("e2", "aluno_b2/int8", "aluno_b2", "o custo de toda a compressão no cenário B"),
    ("f", "aluno_a/int8", "aluno_b2/int8", "o final do cenário A contra o final do cenário B"),
    ("g1", "aluno_a/int8", "base_299", "o modelo final contra a linha de base com os mesmos rótulos humanos"),
    ("g2", "aluno_a/int8", "base_11514", "o modelo final contra a linha de base com todos os rótulos humanos"),
    ("g3", "aluno_a/int8", "piso", "o modelo final contra a mesma rede treinada só com os rótulos humanos que a LLM vê"),
)

# As cadeias: professor -> aluno destilado -> podado e reajustado -> ONNX de 32 bits -> INT8.
CADEIAS = {
    "A": {"nome": "Cenário A: professor fechado", "degraus": (("professor", "claude"), ("destilado", "aluno_a"), ("podado", "aluno_a/podado"), ("onnx", "aluno_a/onnx"), ("int8", "aluno_a/int8"))},
    "B": {"nome": "Cenário B: professor aberto", "degraus": (("professor", "qwen"), ("destilado", "aluno_b2"), ("podado", "aluno_b2/podado"), ("onnx", "aluno_b2/onnx"), ("int8", "aluno_b2/int8"))},
    "teto": {"nome": "Referência: teto (rótulos humanos)", "degraus": (("sem_compressao", "teto"), ("podado", "teto/podado"), ("onnx", "teto/onnx"), ("int8", "teto/int8"))},
}


def base_padrao() -> Path:
    return EXECUCOES / ETAPA


def milhar(x: float) -> str:
    """Inteiro com ponto de milhar, como no texto do projeto: 11514 vira "11.514"."""
    return f"{int(x):,}".replace(",", ".")


def etapa_curta(pasta: str) -> str:
    """`08_quantizacao` vira "Etapa 8"; `05a_professor_fechado` vira "Etapa 5a"."""
    return "Etapa " + pasta.split("_")[0].lstrip("0")


def _json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# O catálogo: cada sistema da tabela final e de onde vem cada coisa dele
# --------------------------------------------------------------------------


def catalogo(entregas: Path | None = None, base: Path | None = None) -> dict[str, dict]:
    """Cada sistema da tabela final, na ordem das linhas.

    Nada aqui é número: a ficha de um sistema só diz **onde ler**. Os nomes das
    variantes (qual poda, qual forma de 8 bits, qual temperatura) saem dos
    `decisao.json` das Etapas 6, 7 e 8, não deste arquivo.

    Campos da ficha:

    - `nome`, `curto`, `bloco`: como o sistema aparece na tabela e nos gráficos;
    - `rotulos_humanos`: quantos rótulos humanos de treino o sistema usa (no treino ou no prompt);
    - `dev_humano`: onde os rótulos humanos de `dev` entraram em decisões que levam a este sistema;
    - `resultado`: (etapa, filtro) da linha do teste no `resultados.csv` daquela etapa: qualidade, parâmetros e tamanho;
    - `eficiencia`: (etapa, filtro) da linha que traz a latência, quando não é a mesma;
    - `zip`: (etapa, arquivo, filtro, coluna) do tamanho compactado, quando ele está em outro arquivo;
    - `previsoes`: os arquivos com as respostas no teste, um por semente;
    - `gerar`: se as respostas não estavam gravadas, como `preparar` as produz;
    - `condicao`, `onde`: em que condição a latência foi medida;
    - `degrau_gravado`: (etapa, filtro) da comparação "este sistema − o degrau anterior da cadeia" que uma etapa já gravou.
    """
    entregas = entregas or ENTREGAS
    base = base or base_padrao()
    dados = _json(entregas / "01_dados" / "manifesto.json")["enunciados"]
    fechado = _json(entregas / "05a_professor_fechado" / "configuracao.json")
    aberto = _json(entregas / "05b_professor_aberto" / "escolha.json")
    destilacao = _json(entregas / "06_destilacao" / "decisao.json")
    poda = _json(entregas / "07_poda" / "decisao.json")
    quantizacao = _json(entregas / "08_quantizacao" / "decisao.json")
    linha_de_base = _json(entregas / "02_avaliacao" / "manifesto.json")["linha_de_base"]["escolhida"]

    poucos, todos = fechado["conjunto_de_demonstracao"]["exemplos"], dados["train"]
    n_poucos, n_todos = milhar(poucos), milhar(todos)
    gerado = base / "previsoes"
    por_semente = lambda pasta, molde: [entregas / pasta / "previsoes" / molde.format(s=s) for s in SEMENTES]
    cpu_1 = "1 thread, PyTorch"

    fichas: dict[str, dict] = {}
    fichas["base_299"] = {
        "nome": f"Linha de base com {n_poucos} rótulos", "curto": f"Linha de base, {n_poucos} rótulos", "bloco": "Linhas de base", "rotulos_humanos": poucos,
        "dev_humano": "nenhuma", "resultado": ("05a_professor_fechado", {"sistema": f"Linha de base com {n_poucos} rótulos"}),
        "previsoes": [gerado / "base_299_test.parquet"],
        "gerar": {"como": "linha_de_base", "de": [entregas / "05a_professor_fechado" / "demonstracoes.parquet"]},
        "condicao": None, "onde": "não medida",
    }
    fichas["base_11514"] = {
        "nome": f"Linha de base com {n_todos} rótulos", "curto": f"Linha de base, {n_todos} rótulos", "bloco": "Linhas de base", "rotulos_humanos": todos,
        "dev_humano": "escolha entre duas linhas de base (Etapa 2)", "resultado": ("02_avaliacao", {"sistema": linha_de_base}),
        "previsoes": [entregas / "02_avaliacao" / "previsoes" / "linha_de_base_test.parquet"],
        "condicao": CPU, "onde": "scikit-learn, threads não registradas",
    }
    fichas["claude"] = {
        "nome": None, "curto": None, "bloco": "Professores", "rotulos_humanos": poucos, "rotulos_como": "no prompt",
        "dev_humano": "nenhuma (configuração sem piloto)", "resultado": ("05a_professor_fechado", {"modelo": fechado["modelo"]}),
        "previsoes": [entregas / "05a_professor_fechado" / "previsoes" / "claude_test.parquet"],
        "condicao": API, "onde": f"{fechado['latencia']['de_onde']}, uma chamada por vez", "nao_divulgado": True, "papel": "professor fechado",
    }
    fichas["qwen"] = {
        "nome": None, "curto": None, "bloco": "Professores", "rotulos_humanos": poucos, "rotulos_como": "no prompt",
        "dev_humano": "escolha entre 3 candidatos e acerto do formato da resposta (Etapa 5b)", "resultado": ("05b_professor_aberto", {"modelo": aberto["modelo"]}),
        "previsoes": [entregas / "05b_professor_aberto" / "previsoes" / f"{aberto['escolhido']}_test.parquet"],
        "condicao": GPU, "onde": "um pedido por vez", "papel": "professor aberto", "modelo_id": aberto["modelo"],
    }
    fichas["teto"] = {
        "nome": f"Teto (DIET do zero com os {n_todos} rótulos humanos)", "curto": "Teto", "bloco": "Referências", "rotulos_humanos": todos,
        "dev_humano": "época de parada e escolha da receita (Etapas 4 e 4b)", "resultado": ("04b_intencao", {}),
        "eficiencia": ("08_quantizacao", {"modelo": "teto", "variante": "sem_poda", "forma": "pytorch"}),
        "zip": ("07_poda", "eficiencia.csv", {"modelo": "teto", "variante": "(sem poda)"}, "bytes_com_zip"),
        "previsoes": por_semente("04b_intencao", "semente_{s}_test.parquet"), "condicao": CPU, "onde": cpu_1,
    }
    fichas["piso"] = {
        "nome": f"Piso (DIET do zero só com os {n_poucos} rótulos humanos)", "curto": "Piso", "bloco": "Referências", "rotulos_humanos": poucos,
        "dev_humano": "época de parada (Etapa 6)", "resultado": ("06_destilacao", {"variante": "piso"}),
        "previsoes": por_semente("06_destilacao", "piso_semente_{s}_test.parquet"), "condicao": CPU, "onde": cpu_1,
    }
    com_o_dev = {"A": "nenhuma no treino (parada no dev rotulado pelo professor)", "B1": "nenhuma no treino (parada no dev rotulado pelo professor)",
                 "B2": f"temperatura T = {destilacao['temperatura_do_b2']} escolhida em dev (Etapa 6)"}
    professor_de = {"A": "professor fechado", "B1": "professor aberto", "B2": "professor aberto"}
    for letra, aluno in destilacao["alunos"].items():
        chave, variante = f"aluno_{letra.lower()}", aluno["variante"]
        fichas[chave] = {
            "nome": aluno["nome"], "curto": f"Aluno {letra}", "bloco": "Alunos destilados", "rotulos_humanos": poucos, "rotulos_como": "pelo professor",
            "dev_humano": com_o_dev.get(letra, "nenhuma no treino"), "resultado": ("06_destilacao", {"variante": variante}),
            "previsoes": por_semente("06_destilacao", variante + "_semente_{s}_test.parquet"), "condicao": CPU, "onde": cpu_1, "modelo_da_poda": variante,
            "degrau_gravado": ("06_destilacao", {"comparacao": f"{PARTICAO} · Aluno {letra} − {professor_de.get(letra, '')}"}),
        }
        if aluno["segue"]:   # os que foram para a poda têm latência medida de novo na Etapa 8, com a máquina quieta
            fichas[chave]["eficiencia"] = ("08_quantizacao", {"modelo": variante, "variante": "sem_poda", "forma": "pytorch"})
            fichas[chave]["zip"] = ("07_poda", "eficiencia.csv", {"modelo": variante, "variante": "(sem poda)"}, "bytes_com_zip")
    fichas["teto"]["modelo_da_poda"] = "teto"

    # Os modelos comprimidos: o aluno de cada cenário e o teto, pela poda principal, pela exportação e pelos 8 bits; e a poda de pesos soltos em INT8
    final = quantizacao["final"]
    cenarios = [(f"aluno_{letra.lower()}", "Cenário A" if letra == destilacao["principal"] else "Cenário B") for letra in destilacao["seguem"]]
    for chave, bloco in [*cenarios, ("teto", "Teto comprimido")]:
        de_partida = fichas[chave]
        modelo = de_partida["modelo_da_poda"]
        do_modelo = poda["modelos"][modelo]
        nome_da_poda = {f["variante"]: f["nome"] for f in do_modelo["seguem"]}
        principal, solta = do_modelo["principal"], do_modelo["secundaria"]
        decidido = f"{de_partida['dev_humano'].replace('nenhuma no treino (parada no dev rotulado pelo professor)', 'nenhuma no treino')}; regra da poda (Etapa 7)"
        em_32 = por_semente("07_poda", f"{modelo}__{principal}" + "_semente_{s}_test.parquet")
        comum = {"bloco": bloco, "rotulos_humanos": de_partida["rotulos_humanos"], "rotulos_como": de_partida.get("rotulos_como"), "condicao": CPU, "de_partida": chave}
        fichas[f"{chave}/podado"] = {
            **comum, "nome": f"{de_partida['curto']}, {_minuscula(nome_da_poda[principal])}, PyTorch, 32 bits", "curto": f"{de_partida['curto']}, podado",
            "dev_humano": decidido, "resultado": ("08_quantizacao", {"modelo": modelo, "variante": principal, "forma": "pytorch"}),
            "zip": ("07_poda", "eficiencia.csv", {"modelo": modelo, "variante": principal}, "bytes_com_zip"), "previsoes": em_32, "onde": cpu_1,
            "degrau_gravado": ("07_poda", {"modelo": modelo, "variante": principal, "comparacao": "depois − sem poda", "particao": PARTICAO}),
            "outra_fonte": ("07_poda", {"modelo": modelo, "variante": principal, "momento": "depois do reajuste"}),
        }
        fichas[f"{chave}/onnx"] = {
            **comum, "nome": f"{de_partida['curto']}, {_minuscula(nome_da_poda[principal])}, ONNX, 32 bits", "curto": f"{de_partida['curto']}, ONNX 32 bits",
            "dev_humano": decidido, "resultado": ("08_quantizacao", {"modelo": modelo, "variante": principal, "forma": "fp32"}),
            # a Etapa 8 conferiu que o ONNX de 32 bits responde igual ao PyTorch em todos os pedidos: as respostas são as mesmas
            "previsoes": em_32, "mesmas_respostas_de": f"{chave}/podado", "onde": "1 thread, ONNX Runtime",
            "degrau_gravado": ("08_quantizacao", {"modelo": modelo, "variante": principal, "forma": "fp32", "particao": PARTICAO}),
        }
        for sufixo, variante, rotulo in (("int8", principal, "INT8"), ("soltos", solta, "pesos soltos, INT8 com zip")):
            ficha = quantizacao["modelos"][f"{modelo}/{variante}"]
            fichas[f"{chave}/{sufixo}"] = {
                **comum, "nome": f"{de_partida['curto']}, {_minuscula(nome_da_poda[variante])}, {ficha['nome_da_forma']}" + (", arquivo compactado" if ficha["compactado"] else ""),
                "curto": f"{de_partida['curto']}, {rotulo}", "dev_humano": f"{decidido}; regra da forma de 8 bits (Etapa 8)",
                "resultado": ("08_quantizacao", {"modelo": modelo, "variante": variante, "forma": ficha["forma"]}),
                "previsoes": [gerado / f"{modelo}__{variante}__{ficha['forma']}_semente_{s}_test.parquet" for s in SEMENTES],
                "gerar": {"como": "onnx", "de": [entregas.parent / ficha["pastas_por_semente"][str(s)] for s in SEMENTES]},
                "onde": "1 thread, ONNX Runtime", "compactado": ficha["compactado"],
            }
        fichas[f"{chave}/int8"]["degrau_gravado"] = ("08_quantizacao", {"modelo": modelo, "variante": principal, "forma": quantizacao["modelos"][f"{modelo}/{principal}"]["forma"], "particao": PARTICAO})
        fichas[f"{chave}/int8"]["final"] = (final["modelo"], final["variante"]) == (modelo, principal)
    for ficha in fichas.values():
        ficha.setdefault("sementes", len(ficha["previsoes"]))
    return fichas


def _minuscula(texto: str) -> str:
    return texto[0].lower() + texto[1:]


def chave_do_final(fichas: dict[str, dict]) -> str:
    """A chave do modelo final do projeto: a que a Etapa 8 marcou em `decisao.json`."""
    finais = [chave for chave, ficha in fichas.items() if ficha.get("final")]
    if len(finais) != 1:
        raise LookupError(f"Era para haver um modelo final, e há {len(finais)}: {finais}.")
    return finais[0]


# --------------------------------------------------------------------------
# O que não estava gravado
# --------------------------------------------------------------------------


def bytes_no_cache(modelo_id: str, cache: Path | None = None) -> tuple[int, int] | None:
    """Tamanho dos arquivos de um modelo baixado do Hugging Face: (bytes, arquivos). None se ele não está no cache."""
    cache = cache or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    blobs = cache / f"models--{modelo_id.replace('/', '--')}" / "blobs"
    if not blobs.is_dir():
        return None
    tamanhos = [arquivo.stat().st_size for arquivo in blobs.iterdir() if arquivo.is_file()]
    return (sum(tamanhos), len(tamanhos)) if tamanhos else None


def preparar(entregas: Path | None = None, base: Path | None = None, refazer: bool = False, cache: Path | None = None) -> Path:
    """Produz o que as etapas anteriores não gravaram, ou devolve a pasta se já está pronto.

    - As respostas no teste de cada sistema com `gerar` na ficha: os modelos
      INT8 da Etapa 8 (carregados da entrega, um pedido por vez, como na Etapa 8)
      e a linha de base com os 299 rótulos (treinada de novo, como na Etapa 5a).
    - `medidas.json`: o que foi produzido, de onde, e o tamanho em disco dos
      pesos do professor aberto, se ainda estão no cache desta máquina.

    Roda na CPU, em segundos. Não treina rede, não usa a GPU e não chama a API.
    """
    from especialista.compressao.exportacao import SistemaOnnx
    from especialista.dados import massive
    from especialista.modelos.linha_de_base import LinhaDeBase

    entregas = entregas or ENTREGAS
    base = base or base_padrao()
    fichas = catalogo(entregas, base)
    particoes, _ = massive.carregar_entrega(entregas / "01_dados")
    teste = particoes[PARTICAO]
    textos = list(teste["utt"])
    geradas = {}
    for chave, ficha in fichas.items():
        gerar = ficha.get("gerar")
        if not gerar:
            continue
        for origem, destino in zip(gerar["de"], ficha["previsoes"], strict=True):
            if destino.exists() and not refazer:
                continue
            if gerar["como"] == "onnx":
                previsoes = SistemaOnnx.carregar(origem).prever(textos)
            else:   # a linha de base da Etapa 5a: palavras + caracteres, treinada só com os exemplos que a LLM vê
                previsoes = LinhaDeBase(caracteres=True).treinar(pd.read_parquet(origem)).prever(textos)
            salvar_previsoes(destino, teste["id"], previsoes)
        geradas[chave] = {"como": gerar["como"], "de": [_relativo(c, entregas.parent) for c in gerar["de"]], "arquivos": [c.name for c in ficha["previsoes"]]}
    modelo_id = fichas["qwen"]["modelo_id"]
    no_cache = bytes_no_cache(modelo_id, cache)
    professor_aberto = {
        "modelo": modelo_id, "bytes_dos_pesos_baixados": no_cache[0] if no_cache else None, "arquivos": no_cache[1] if no_cache else None,
        "o_que_e": "os arquivos do modelo como foram baixados (pesos de 16 bits, tokenizador e configuração); a Etapa 5b os carrega em 4 bits na placa",
        "de_onde": "cache do Hugging Face desta máquina" if no_cache else "o modelo não está mais no cache desta máquina: não medido",
        "medido_em": date.today().isoformat() if no_cache else None,
    }
    if not no_cache and (base / "medidas.json").exists():   # o cache foi limpo depois de medir: a medida que já estava gravada continua valendo
        antes = _json(base / "medidas.json").get("professor_aberto", {})
        if antes.get("modelo") == modelo_id and antes.get("bytes_dos_pesos_baixados"):
            professor_aberto = antes
    medidas = {"etapa": ETAPA, "gerado_em": date.today().isoformat(), "particao": PARTICAO, "pedidos": len(textos), "respostas_geradas": geradas,
               "professor_aberto": professor_aberto}
    base.mkdir(parents=True, exist_ok=True)
    (base / "medidas.json").write_text(json.dumps(medidas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")   # por último: marca "pronto"
    return base


def _relativo(caminho: Path, raiz: Path) -> str:
    return str(caminho.relative_to(raiz)) if caminho.is_relative_to(raiz) else str(caminho)


def faltam(fichas: dict[str, dict], base: Path | None = None) -> list[Path]:
    """Os arquivos de respostas que ainda não existem, e o `medidas.json`."""
    base = base or base_padrao()
    ausentes = [caminho for ficha in fichas.values() for caminho in ficha["previsoes"] if not caminho.exists()]
    return ausentes + ([] if (base / "medidas.json").exists() else [base / "medidas.json"])


def carregar_medidas(base: Path | None = None) -> dict:
    return _json((base or base_padrao()) / "medidas.json")


def respostas(ficha: dict, ids: list[int] | None = None) -> list[list[Previsao]]:
    """As respostas no teste de um sistema: uma lista por semente (uma só, se o sistema não é treinado com sementes).

    Com `ids`, confere que cada arquivo traz os pedidos do teste na mesma ordem: sem isso a comparação não é pareada.
    """
    grupo = []
    for caminho in ficha["previsoes"]:
        if not caminho.exists():
            raise FileNotFoundError(f"Falta {caminho}. Rode: {COMO_PREPARAR}")
        lidos, previsoes = carregar_previsoes(caminho)
        if ids is not None and list(lidos) != list(ids):
            raise ValueError(f"{caminho} não traz os pedidos do teste na ordem da entrega de dados.")
        grupo.append(previsoes)
    return grupo


# --------------------------------------------------------------------------
# A tabela final: uma linha por sistema, lida do que cada etapa gravou
# --------------------------------------------------------------------------


def linha_gravada(tabela: pd.DataFrame, **filtro) -> pd.Series:
    """A única linha de `tabela` que casa com o filtro.

    Dá erro se não houver exatamente uma: um nome trocado numa etapa anterior não passa calado.
    """
    escolhidas = tabela
    for coluna, valor in filtro.items():
        escolhidas = escolhidas[escolhidas[coluna] == valor]
    if len(escolhidas) != 1:
        raise LookupError(f"Era para haver uma linha com {filtro}, e há {len(escolhidas)}.")
    return escolhidas.iloc[0]


def consultas_por_segundo(latencia_media_ms: float) -> float:
    """Quantas consultas por segundo um sistema responde, uma depois da outra: o inverso do tempo médio de uma consulta."""
    return 1000.0 / latencia_media_ms if latencia_media_ms and not np.isnan(latencia_media_ms) else float("nan")


class _Tabelas:
    """Os CSV das entregas, lidos uma vez cada."""

    def __init__(self, entregas: Path):
        self.entregas, self.lidas = entregas, {}

    def __call__(self, etapa: str, arquivo: str = "resultados.csv") -> pd.DataFrame:
        if (etapa, arquivo) not in self.lidas:
            self.lidas[(etapa, arquivo)] = pd.read_csv(self.entregas / etapa / arquivo)
        return self.lidas[(etapa, arquivo)]


def _do_teste(tabela: pd.DataFrame, filtro: dict) -> pd.Series:
    return linha_gravada(tabela, **({"particao": PARTICAO} if "particao" in tabela.columns else {}), **filtro)


def tabela_final(fichas: dict[str, dict], medidas: dict | None = None, entregas: Path | None = None) -> pd.DataFrame:
    """A tabela final do projeto, no teste: uma linha por sistema, com tudo lido dos arquivos das etapas.

    Colunas de qualidade: cada métrica com `_ic_min` e `_ic_max` (intervalo de
    95%), como a etapa de origem gravou. `bytes` é o arquivo como é usado;
    `bytes_com_zip`, o mesmo arquivo passado por um compressor comum, quando
    foi medido. `consultas_por_segundo` é o inverso da latência média, uma
    consulta por vez. `custo_por_mil_usd` só existe para a API.
    """
    ler = _Tabelas(entregas or ENTREGAS)
    aberto = (medidas or {}).get("professor_aberto", {})
    linhas = []
    for chave, ficha in fichas.items():
        etapa, filtro = ficha["resultado"]
        r = _do_teste(ler(etapa), filtro)
        etapa_da_latencia, filtro_da_latencia = ficha.get("eficiencia", ficha["resultado"])
        e = _do_teste(ler(etapa_da_latencia), filtro_da_latencia)
        nome = ficha["nome"] or f"{str(r['sistema']).split(' (')[0]} ({ficha['papel']})"
        linha = {
            "chave": chave, "sistema": nome, "curto": ficha["curto"] or str(r["sistema"]).split(" (")[0].replace(" em 4 bits", ""), "bloco": ficha["bloco"],
            "rotulos_humanos": ficha["rotulos_humanos"], "rotulos_como": ficha.get("rotulos_como") or "no treino", "dev_humano_nas_decisoes": ficha["dev_humano"],
            "sementes": ficha["sementes"],
        }
        for metrica in qualidade.METRICAS:
            linha.update({metrica: r[metrica], f"{metrica}_ic_min": r[f"{metrica}_ic_min"], f"{metrica}_ic_max": r[f"{metrica}_ic_max"]})
        mede_latencia = ficha["condicao"] is not None and pd.notna(e.get("latencia_p50_ms"))
        media = e["latencia_media_ms"] if mede_latencia else np.nan
        linha.update({
            "parametros": r.get("parametros", np.nan), "bytes": r.get("bytes", np.nan), "bytes_com_zip": r.get("bytes_compactado", np.nan),
            "divulgado": not ficha.get("nao_divulgado", False), "entregue_compactado": bool(ficha.get("compactado", False)),
            "latencia_p50_ms": e["latencia_p50_ms"] if mede_latencia else np.nan, "latencia_p95_ms": e["latencia_p95_ms"] if mede_latencia else np.nan,
            "latencia_media_ms": media, "condicao_da_latencia": ficha["condicao"] if mede_latencia else None,
            "onde_foi_medida": _onde(ficha, e) if mede_latencia else "não medida", "latencia_medida_na": etapa_curta(etapa_da_latencia) if mede_latencia else None,
            "consultas_por_segundo": consultas_por_segundo(media),
            "custo_por_mil_usd": r.get("custo_por_mil_pedidos_usd", np.nan), "custo_por_mil_em_lote_usd": r.get("custo_por_mil_pedidos_em_lote_usd", np.nan),
            "memoria_na_placa_gb": r.get("vram_gb", np.nan), "qualidade_lida_de": f"entregas/{etapa}/resultados.csv",
        })
        if "zip" in ficha:
            etapa_do_zip, arquivo, filtro_do_zip, coluna = ficha["zip"]
            linha["bytes_com_zip"] = linha_gravada(ler(etapa_do_zip, arquivo), **filtro_do_zip)[coluna]
        if chave == "qwen" and aberto.get("bytes_dos_pesos_baixados"):
            linha["bytes"] = aberto["bytes_dos_pesos_baixados"]
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("chave")


def _onde(ficha: dict, linha: pd.Series) -> str:
    """Onde e como a latência de um sistema foi medida, por extenso."""
    if ficha["condicao"] == API:
        return f"{API} ({ficha['onde']})"
    if ficha["condicao"] == GPU:
        return f"{GPU} ({linha.get('gpu', 'placa desta máquina')}), {ficha['onde']}"
    return f"CPU ({linha.get('cpu', 'desta máquina')}), {ficha['onde']}"


# --------------------------------------------------------------------------
# A conta desta etapa confere com a das etapas anteriores?
# --------------------------------------------------------------------------


def conferir_qualidade(fichas: dict[str, dict], tabela: pd.DataFrame, gabarito: list[Previsao], grupos: dict[str, list[list[Previsao]]],
                       entregas: Path | None = None) -> pd.DataFrame:
    """Recalcula, das respostas, cada métrica e cada intervalo da tabela final, e compara com o que a etapa de origem gravou.

    Uma linha por sistema: a maior diferença absoluta entre as duas contas, em
    qualquer métrica, no valor ou nos limites do intervalo. Onde a mesma
    medida está gravada por duas etapas (`outra_fonte` na ficha), a segunda
    também é comparada.
    """
    ler = _Tabelas(entregas or ENTREGAS)
    linhas = []
    for chave, ficha in fichas.items():
        refeito = qualidade.intervalo_do_grupo(gabarito, grupos[chave], n=REAMOSTRAS)
        fontes = [(ficha["resultado"][0], tabela.loc[chave])]
        if "outra_fonte" in ficha:
            fontes.append((ficha["outra_fonte"][0], _do_teste(ler(ficha["outra_fonte"][0]), ficha["outra_fonte"][1])))
        for etapa, gravado in fontes:
            diferencas = [abs(refeito.loc[m, "valor"] - gravado[m]) for m in qualidade.METRICAS]
            diferencas += [abs(refeito.loc[m, limite] - gravado[f"{m}_{limite}"]) for m in qualidade.METRICAS for limite in ("ic_min", "ic_max")]
            linhas.append({"chave": chave, "gravado_em": f"entregas/{etapa}/resultados.csv", "respostas_geradas_agora": "gerar" in ficha,
                           "maior_diferenca": float(max(diferencas)), "confere": bool(max(diferencas) <= TOLERANCIA)})
    return pd.DataFrame(linhas)


def conferir_degraus(fichas: dict[str, dict], degraus: pd.DataFrame, entregas: Path | None = None) -> pd.DataFrame:
    """Compara cada degrau das cadeias, calculado aqui, com a mesma comparação gravada pela etapa que deu aquele passo.

    Uma linha por degrau que tem `degrau_gravado` na ficha: a maior diferença
    absoluta, em qualquer métrica, na diferença ou nos limites do intervalo.
    """
    ler = _Tabelas(entregas or ENTREGAS)
    linhas = []
    for _, degrau in degraus.dropna(subset=["anterior"]).iterrows():
        gravado_em = fichas[degrau["chave"]].get("degrau_gravado")
        if not gravado_em:
            continue
        etapa, filtro = gravado_em
        gravado = ler(etapa, "comparacoes.csv")
        for coluna, valor in filtro.items():
            gravado = gravado[gravado[coluna] == valor]
        gravado = gravado[gravado["metrica"].isin(qualidade.METRICAS)].set_index("metrica")
        if sorted(gravado.index) != sorted(qualidade.METRICAS):
            raise LookupError(f"Era para haver uma comparação gravada com {filtro} em {etapa}, uma linha por métrica, e há {len(gravado)} linhas.")
        diferencas = [abs(degrau[f"{m}_{c}"] - gravado.loc[m, g]) for m in qualidade.METRICAS for c, g in (("diferenca", "diferenca"), ("dif_ic_min", "ic_min"), ("dif_ic_max", "ic_max"))]
        linhas.append({"cadeia": degrau["cadeia"], "chave": degrau["chave"], "anterior": degrau["anterior"], "gravado_em": f"entregas/{etapa}/comparacoes.csv",
                       "maior_diferenca": float(max(diferencas)), "confere": bool(max(diferencas) <= TOLERANCIA)})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# As cadeias e as comparações de ponta a ponta
# --------------------------------------------------------------------------


def comparar(gabarito: list[Previsao], a: list[list[Previsao]], b: list[list[Previsao]]) -> pd.DataFrame:
    """Diferença pareada B − A entre dois sistemas (cada um, a média das suas sementes), com intervalo de 95%."""
    return qualidade.comparar_grupos(gabarito, a, b, n=REAMOSTRAS)


def cadeia(nome: str, degraus: tuple[tuple[str, str], ...], tabela: pd.DataFrame, gabarito: list[Previsao], grupos: dict[str, list[list[Previsao]]]) -> pd.DataFrame:
    """Uma cadeia, degrau por degrau: qualidade, tamanho e latência de cada um e a diferença pareada para o degrau anterior.

    `degraus` é a sequência de (tipo do degrau, chave do sistema). A primeira
    linha não tem degrau anterior. Colunas da diferença: `<métrica>_diferenca`,
    `<métrica>_dif_ic_min`, `<métrica>_dif_ic_max` e `<métrica>_distinguivel`.
    """
    linhas, anterior = [], None
    for ordem, (tipo, chave) in enumerate(degraus):
        s = tabela.loc[chave]
        linha = {"cadeia": nome, "ordem": ordem, "degrau": DEGRAUS[tipo], "chave": chave, "sistema": s["sistema"], "anterior": anterior}
        for metrica in qualidade.METRICAS:
            linha.update({metrica: s[metrica], f"{metrica}_ic_min": s[f"{metrica}_ic_min"], f"{metrica}_ic_max": s[f"{metrica}_ic_max"]})
        linha.update({c: s[c] for c in ("parametros", "bytes", "bytes_com_zip", "latencia_p50_ms", "condicao_da_latencia", "onde_foi_medida", "consultas_por_segundo")})
        if anterior is not None:
            contra = comparar(gabarito, grupos[anterior], grupos[chave])
            for metrica in qualidade.METRICAS:
                linha.update({f"{metrica}_diferenca": contra.loc[metrica, "diferenca"], f"{metrica}_dif_ic_min": contra.loc[metrica, "ic_min"],
                              f"{metrica}_dif_ic_max": contra.loc[metrica, "ic_max"], f"{metrica}_distinguivel": bool(contra.loc[metrica, "distinguivel"])})
        linhas.append(linha)
        anterior = chave
    return pd.DataFrame(linhas)


def cadeias(tabela: pd.DataFrame, gabarito: list[Previsao], grupos: dict[str, list[list[Previsao]]], quais: dict | None = None) -> pd.DataFrame:
    """Todas as cadeias de `CADEIAS`, uma embaixo da outra."""
    quais = quais or CADEIAS
    return pd.concat([cadeia(nome, c["degraus"], tabela, gabarito, grupos) for nome, c in quais.items()], ignore_index=True)


def ponta_a_ponta(tabela: pd.DataFrame, gabarito: list[Previsao], grupos: dict[str, list[list[Previsao]]], lista: tuple = PONTA_A_PONTA) -> pd.DataFrame:
    """As comparações de ponta a ponta, B − A, em todas as métricas: uma linha por comparação e métrica."""
    linhas = []
    for letra, b, a, pergunta in lista:
        contra = comparar(gabarito, grupos[a], grupos[b])
        for metrica in qualidade.METRICAS:
            linhas.append({
                "letra": letra, "comparacao": f"{tabela.loc[b, 'curto']} − {tabela.loc[a, 'curto']}", "chave_b": b, "chave_a": a, "pergunta": pergunta, "metrica": metrica,
                "valor_a": contra.loc[metrica, "a"], "valor_b": contra.loc[metrica, "b"], "diferenca": contra.loc[metrica, "diferenca"],
                "ic_min": contra.loc[metrica, "ic_min"], "ic_max": contra.loc[metrica, "ic_max"], "distinguivel": bool(contra.loc[metrica, "distinguivel"]),
                "exploratoria": False,
            })
    return pd.DataFrame(linhas)


def nao_dominados(pontos: pd.DataFrame, custo: str, ganho: str, folga_do_custo: float = 0.0) -> list:
    """Os pontos da fronteira de Pareto: os que nenhum outro vence nas duas medidas ao mesmo tempo.

    Um ponto é dominado quando existe outro com `custo` menor ou igual e
    `ganho` maior ou igual, e melhor em pelo menos um dos dois. Devolve os
    índices dos que sobram, do menor custo para o maior. Linhas sem uma das
    duas medidas ficam de fora.

    `folga_do_custo` é a variação relativa da própria medida de custo (0,1 =
    10%): dois custos mais próximos que isso contam como empate. Sem ela, um
    sistema entraria na fronteira por ser um milésimo de milissegundo mais
    rápido numa medida que varia mais do que isso. O ganho é comparado pelo
    valor medido, sem folga.
    """
    validos = pontos.dropna(subset=[custo, ganho])
    c, g, f = validos[custo], validos[ganho], 1.0 + folga_do_custo
    fronteira = []
    for indice, p in validos.iterrows():
        vencido = (((c <= p[custo] * f) & (g > p[ganho])) | ((c * f < p[custo]) & (g >= p[ganho]))).any()
        if not vencido:
            fronteira.append(indice)
    return sorted(fronteira, key=lambda indice: validos.loc[indice, custo])


def variacao_da_latencia(entregas: Path | None = None) -> float:
    """A maior variação relativa da latência mediana entre as rodadas da medida da Etapa 8: (maior p50 − menor p50) ÷ p50.

    É a folga com que duas latências desta máquina podem ser chamadas de diferentes.
    """
    e = pd.read_csv((entregas or ENTREGAS) / "08_quantizacao" / "eficiencia.csv")
    return float(((e["p50_max_das_rodadas"] - e["p50_min_das_rodadas"]) / e["latencia_p50_ms"]).max())


def poda_em_dev_e_no_teste(entregas: Path | None = None, metrica: str = "exact_match") -> pd.DataFrame:
    """Para cada variante podada que a Etapa 7 mandou seguir: a diferença para o modelo sem poda em `dev` (onde a regra decidiu) e no teste.

    A regra da Etapa 7 aceita a variante que **não fica distinguivelmente
    abaixo** do modelo sem poda em `dev`. `abaixo_no_teste` diz se, no teste,
    ela ficou.
    """
    entregas = entregas or ENTREGAS
    decisao = _json(entregas / "07_poda" / "decisao.json")
    comparacoes = pd.read_csv(entregas / "07_poda" / "comparacoes.csv")
    linhas = []
    for modelo, do_modelo in decisao["modelos"].items():
        for papel in ("principal", "secundaria"):
            variante = do_modelo.get(papel)
            if not variante:
                continue
            linha = {"modelo": modelo, "variante": variante, "papel": papel}
            for particao in ("dev", PARTICAO):
                c = linha_gravada(comparacoes, modelo=modelo, variante=variante, comparacao="depois − sem poda", particao=particao, metrica=metrica)
                linha.update({f"{particao}_diferenca": c["diferenca"], f"{particao}_ic_min": c["ic_min"], f"{particao}_ic_max": c["ic_max"]})
            linha["passa_em_dev"] = bool(linha["dev_ic_max"] >= 0)
            linha["abaixo_no_teste"] = bool(linha[f"{PARTICAO}_ic_max"] < 0)
            linhas.append(linha)
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Cada passo e o seu artigo
# --------------------------------------------------------------------------

# Só entram artigos que já estão no PLANO.md ou num RESUMO.md de entrega. As exceções têm `conferido`: a página em que o
# título e os autores foram conferidos nesta etapa (são os três que os notebooks das Etapas 2 e 4 citam e o PLANO.md não).
# `quem` e `apelido` são como o PLANO.md e os RESUMOs escrevem; nada vem de memória.
ARTIGOS = (
    {"passo": "O caminho inteiro", "tecnica": "Trocar a LLM por um modelo pequeno, feito para a tarefa", "quem": "Belcak et al.", "arxiv": "2506.02153"},
    {"passo": "O caminho inteiro", "tecnica": "LLM professora, destilação, poda e quantização em sequência", "quem": "Ni et al.", "apelido": "From Large to Super-Tiny", "arxiv": "2504.13471"},
    {"passo": "O caminho inteiro", "tecnica": "A ordem: destilar, podar, quantizar", "quem": "Han et al.", "apelido": "Deep Compression", "arxiv": "1510.00149"},
    {"passo": "O caminho inteiro", "tecnica": "A ordem: destilar, podar, quantizar", "quem": "Harma et al.", "arxiv": "2405.20935"},
    {"passo": "O caminho inteiro", "tecnica": "Poda e quantização juntas podem interagir (discussão da ordem)", "quem": "Nema et al. (preprint; só o resumo foi lido)",
     "apelido": "Compression Technique Interactions in Small Language Models", "arxiv": "2611.00464"},
    {"passo": "Etapas 1 e 2 · Dados e régua", "tecnica": "MASSIVE pt-PT; acurácia de intenção, F1 de entidades e exact match", "quem": "FitzGerald et al.", "apelido": "MASSIVE", "arxiv": "2204.08582"},
    {"passo": "Etapa 2 · Régua", "tecnica": "Intervalo de 95% e comparação pareada por bootstrap, por enunciado", "quem": "Koehn", "veiculo": "EMNLP 2004",
     "apelido": "Statistical Significance Tests for Machine Translation Evaluation", "conferido": "https://aclanthology.org/W04-3250/"},
    {"passo": "Etapa 2 · Régua", "tecnica": "Intervalo de 95% e comparação pareada por bootstrap, por enunciado", "quem": "Berg-Kirkpatrick et al.", "veiculo": "EMNLP-CoNLL 2012",
     "apelido": "An Empirical Investigation of Statistical Significance in NLP", "conferido": "https://aclanthology.org/D12-1091/"},
    {"passo": "Etapa 2 · Régua", "tecnica": "Latência: uma consulta por vez, percentis, máquina registrada", "quem": "Reddi et al.", "apelido": "MLPerf Inference", "arxiv": "1911.02549"},
    {"passo": "Etapas 2 e 9 · Régua e comparativo", "tecnica": "Custo reportado em várias medidas, com a condição de cada uma", "quem": "Dehghani et al.", "arxiv": "2110.12894"},
    {"passo": "Etapas 3 e 4 · Arquitetura", "tecnica": "Rede conjunta de intenção e entidades, do zero: features esparsas, Transformer de 2 camadas, CRF", "quem": "Bunk et al.", "apelido": "DIET", "arxiv": "2004.09936"},
    {"passo": "Etapa 4 · Treino", "tecnica": "Três sementes por modelo treinado; o resultado é a média", "quem": "Reimers & Gurevych", "arxiv": "1707.09861",
     "apelido": "Reporting Score Distributions Makes a Difference", "conferido": "https://arxiv.org/abs/1707.09861"},
    {"passo": "Etapa 5a · Professor fechado", "tecnica": "LLM com poucos exemplos no prompt: intenção perto do supervisionado, entidades atrás", "quem": "He & Garner", "arxiv": "2305.13512"},
    {"passo": "Etapa 6 · Destilação", "tecnica": "Destilação caixa-preta: o aluno aprende só com os rótulos da LLM", "quem": "Xu et al.", "apelido": "survey", "arxiv": "2402.13116"},
    {"passo": "Etapa 6 · Destilação", "tecnica": "Destilação caixa-preta: o aluno aprende só com os rótulos da LLM", "quem": "Di Palo et al.", "arxiv": "2411.05045"},
    {"passo": "Etapa 6 · Destilação", "tecnica": "Destilação com logits e temperatura", "quem": "Hinton et al.", "arxiv": "1503.02531"},
    {"passo": "Etapa 6 · Destilação", "tecnica": "Destilação com logits e temperatura", "quem": "Jiao et al.", "apelido": "TinyBERT", "arxiv": "1909.10351"},
    {"passo": "Etapa 7 · Poda", "tecnica": "Poda por magnitude: treinar, podar, retreinar", "quem": "Han et al.", "arxiv": "1506.02626"},
    {"passo": "Etapa 7 · Poda", "tecnica": "Poda gradual, com a esparsidade subindo durante o treino", "quem": "Zhu & Gupta", "arxiv": "1710.01878"},
    {"passo": "Etapa 7 · Poda", "tecnica": "Poda de peças inteiras (estruturada)", "quem": "Xia et al.", "apelido": "CoFi", "arxiv": "2204.00408"},
    {"passo": "Etapa 7 · Reajuste", "tecnica": "Fine-tuning de recuperação com a máscara travada, com o sinal do professor", "quem": "Zafrir et al.", "apelido": "Prune Once for All", "arxiv": "2111.05754"},
    {"passo": "Etapa 7 · Reajuste", "tecnica": "Fine-tuning de recuperação com a máscara travada, com o sinal do professor", "quem": "Sanh et al.", "arxiv": "2005.07683"},
    {"passo": "Etapa 8 · Quantização", "tecnica": "Inteiros de 8 bits: escala e ponto zero", "quem": "Jacob et al.", "arxiv": "1712.05877"},
    {"passo": "Etapa 8 · Quantização", "tecnica": "Um Transformer em 8 bits: o que quantizar e o que deixar em 32 bits", "quem": "Zafrir et al.", "apelido": "Q8BERT", "arxiv": "1910.06188"},
    {"passo": "Etapa 10 · Em aberto", "tecnica": "Encoder pré-treinado com cabeça conjunta de intenção e entidades", "quem": "Chen et al.", "apelido": "JointBERT", "arxiv": "1902.10909"},
)


def ano_do_arxiv(identificador: str) -> int:
    """O ano em que um artigo entrou no arXiv, lido do próprio identificador: `2204.08582` é de abril de 2022."""
    if not re.fullmatch(r"\d{4}\.\d{4,5}", identificador):
        raise ValueError(f"Não é um identificador do arXiv: {identificador!r}")
    return 2000 + int(identificador[:2])


def quadro_de_artigos(artigos: tuple = ARTIGOS) -> pd.DataFrame:
    """O quadro "cada passo e o seu artigo": uma linha por artigo, com o ano e o endereço."""
    linhas = []
    for artigo in artigos:
        arxiv = artigo.get("arxiv")
        linhas.append({
            "passo": artigo["passo"], "tecnica": artigo["tecnica"], "quem": artigo["quem"], "apelido": artigo.get("apelido", ""),
            "ano": ano_do_arxiv(arxiv) if arxiv else int(artigo["veiculo"][-4:]), "onde": f"arXiv {arxiv}" if arxiv else artigo["veiculo"],
            "endereco": f"https://arxiv.org/abs/{arxiv}" if arxiv else artigo.get("conferido", ""), "conferido_nesta_etapa": bool(artigo.get("conferido")),
        })
    return pd.DataFrame(linhas)


def artigos_fora_das_fontes(artigos: tuple = ARTIGOS, raiz: Path | None = None) -> list[str]:
    """Os artigos do quadro cujo identificador do arXiv não aparece nem no PLANO.md nem em nenhum RESUMO.md de entrega.

    A regra da etapa: só entra no quadro o que já estava citado nesses
    arquivos. O que esta função devolve tem de estar marcado com `conferido`.
    Só valem os resumos das etapas anteriores a esta (uma etapa só lê as
    entregas das anteriores): o resumo de uma etapa posterior que cite o
    mesmo artigo não muda o que estava citado quando este quadro foi feito.
    """
    raiz = raiz or ENTREGAS.parent
    fontes = [raiz / "PLANO.md", *sorted(f for f in (raiz / "entregas").glob("*/RESUMO.md") if f.parent.name < ETAPA)]
    texto = "\n".join(f.read_text(encoding="utf-8") for f in fontes if f.exists())
    return [f"{a['quem']} ({a.get('arxiv') or a.get('veiculo')})" for a in artigos if not (a.get("arxiv") and a["arxiv"] in texto)]


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 9: comparativo final. Só lê as entregas; `preparar` produz, na CPU, as respostas que não estavam gravadas.")
    argumentos.add_argument("comando", choices=("faltam", "preparar"))
    argumentos.add_argument("--refazer", action="store_true", help="preparar: produz de novo o que já existe")
    argumentos.add_argument("--pasta", type=Path, default=None, help="onde gravar; o padrão é data/execucoes/09_comparativo")
    a = argumentos.parse_args()
    if a.comando == "faltam":
        for caminho in faltam(catalogo(base=a.pasta), a.pasta):
            print(caminho)
        return
    pasta = preparar(base=a.pasta, refazer=a.refazer)
    medidas = carregar_medidas(pasta)
    pesos = medidas["professor_aberto"]["bytes_dos_pesos_baixados"]
    print(f"Respostas no teste geradas para {len(medidas['respostas_geradas'])} sistemas, em {pasta}.")
    print("Pesos do professor aberto no cache:", f"{pesos / 1e9:.1f} GB" if pesos else "não estão mais no cache (não medido)")


if __name__ == "__main__":
    _linha_de_comando()
