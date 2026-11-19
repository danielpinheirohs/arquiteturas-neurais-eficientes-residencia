"""Etapa 10d: comparativo dos dois caminhos. A rede do zero (Etapas 2 a 9) ao lado do encoder pré-treinado (Etapas 10a a 10c).

Nada é treinado aqui. Tudo sai do que as etapas anteriores entregaram:

    caminho do zero     o catálogo da Etapa 9 (`etapa9.catalogo`), sem mudança
    caminho do XLM-R    os `decisao.json`, `resultados.csv` e `previsoes/` das Etapas 10a, 10b e 10c

**O registro de sistemas** (`registro`) é a peça central. Cada sistema é uma
ficha que diz onde ler a qualidade, as respostas (em `dev` e no teste), o
tamanho e a latência, e que traz quatro marcas: `caminho` (zero, xlmr),
`modelo` (teto, piso, aluno_a, aluno_b1, aluno_b2), `ponto` da cadeia
(sem_compressao, destilado, podado, onnx, int8...) e `professor`. As
comparações não são uma lista escrita à mão: são **geradas por regra** a
partir dessas marcas (`lista_de_comparacoes`, `cadeias_do_registro`). Um
sistema novo no registro entra sozinho na tabela, na fronteira e nas
comparações.

**Para acrescentar linhas depois** (a Etapa 11, por exemplo), há dois jeitos:
uma função nova em `FONTES`, que leia o `decisao.json` da etapa nova, ou um
arquivo `sistemas.json` na entrega da etapa nova (formato em
`fichas_declaradas`), que o registro lê sem mudança de código.

**Cada comparativo diz que entregas valem para ele** (`valem`). O desta etapa
lê as de `ENTREGAS_QUE_VALEM`, as que existiam quando ela foi entregue: assim
ela continua reproduzindo o que entregou (44 sistemas, 43 comparações) depois
que etapas novas declaram sistemas. O comparativo seguinte (Etapa 11c) chama
as mesmas funções com a lista dele.

A ordem dos comandos é a ordem em que as coisas podem ser feitas:

    uv run python -m especialista.experimentos.etapa10d regras      # 1. grava as regras, antes de qualquer conta nova
    uv run python -m especialista.experimentos.etapa10d preparar    # 2. produz, na CPU, as respostas que nenhuma etapa gravou
    uv run python -m especialista.experimentos.etapa10d latencia    # 3. mede de novo os dois modelos finais, na mesma sessão (máquina quieta)
    uv run python -m especialista.experimentos.etapa10d artigos     # 4. confere cada referência no arXiv (gratuito)
    uv run python -m especialista.experimentos.etapa10d calcular    # 5. tabela, conferência, cadeias, comparações, fronteira e preço

`calcular` não roda sem `regras.json`, e grava a impressão digital dele junto
do que calculou. Tudo fica em `data/execucoes/10d_comparativo_dos_caminhos/`;
o notebook só lê.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, EXECUCOES
from especialista.experimentos import etapa9

ETAPA = "10d_comparativo_dos_caminhos"
PERGUNTA = ("Com os mesmos professores e as mesmas regras, o que o encoder pré-treinado compra em qualidade, "
            "e quanto custa em tamanho e latência, em relação à rede feita do zero?")
SEMENTES = etapa9.SEMENTES
PARTICOES = ("dev", "test")
TRES = etapa9.TRES
REAMOSTRAS = etapa9.REAMOSTRAS
TOLERANCIA = etapa9.TOLERANCIA
API, GPU, CPU = etapa9.API, etapa9.GPU, etapa9.CPU
LIMITE_DE_OUTROS = 0.25    # núcleos de CPU que os outros processos podem ocupar numa medida de latência: o limite das Etapas 8 e 10c
E9, E10A, E10B, E10C = "09_comparativo", "10a_encoder", "10b_encoder_poda", "10c_encoder_quantizacao"

# Os caminhos. `prefixo` é o que vem antes da chave de cada sistema: as do caminho do zero são as da Etapa 9, sem prefixo.
ZERO, XLMR = "zero", "xlmr"
CAMINHOS = {
    ZERO: {"nome": "Rede do zero (DIET)", "curto": "Do zero", "prefixo": "", "etapas": "2 a 9"},
    XLMR: {"nome": "Encoder pré-treinado (XLM-R base)", "curto": "XLM-R", "prefixo": "xlmr/", "etapas": "10a a 10c"},
}
REFERENCIA = ZERO          # nas comparações entre caminhos, B é o outro caminho e A é este

# Os pontos de uma cadeia e o nome de cada um. Os de `ENTRE_CAMINHOS` são comparados entre os caminhos, no mesmo modelo.
PONTOS = {
    "base": "Linha de base", "professor": "Professor (LLM)", "sem_compressao": "Sem compressão", "piso": "Piso", "destilado": "Aluno destilado",
    "podado": "Podado e reajustado", "onnx": "ONNX, 32 bits", "int8": "INT8", "soltos": "Pesos soltos, INT8", "int8_sem_poda": "INT8, sem poda",
}
ENTRE_CAMINHOS = ("sem_compressao", "piso", "destilado", "podado", "int8")
CONTRA_O_PROFESSOR = ("destilado", "int8")
NOME_DO_MODELO = {"teto": "Teto", "piso": "Piso", "aluno_a": "Aluno A", "aluno_b1": "Aluno B1", "aluno_b2": "Aluno B2"}

# As entregas que valem para esta etapa: as que existiam quando ela foi entregue, em 18/11/2026. É delas que o registro lê
# os `sistemas.json` (nenhuma tem) e as regras leem os `decisao.json`. Sem esta lista, o registro leria o `sistemas.json`
# de qualquer etapa posterior, e reexecutar a 10d deixaria de reproduzir o que ela entregou. Quem quer ler outras entregas
# (o comparativo da Etapa 11c) passa a sua lista em `valem`; `valem=None` quer dizer "todas as que estão em disco".
ENTREGAS_QUE_VALEM = (
    "01_dados", "02_avaliacao", "03_arquitetura", "04_treino", "04b_intencao", "05a_professor_fechado", "05b_professor_aberto", "06_destilacao",
    "07_poda", "08_quantizacao", "09_comparativo", "10a_encoder", "10b_encoder_poda", "10c_encoder_quantizacao", ETAPA,
)


def base_padrao() -> Path:
    return EXECUCOES / ETAPA


def _json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _minuscula(texto: str) -> str:
    return texto[0].lower() + texto[1:]


# --------------------------------------------------------------------------
# O registro: cada sistema dos dois caminhos e de onde vem cada coisa dele
# --------------------------------------------------------------------------


def _em_dev(caminho: Path, entregas: Path, base: Path) -> Path:
    """O arquivo de `dev` que corresponde a um arquivo de respostas do teste: ao lado dele, se é de uma entrega; no cache desta etapa, se foi gerado."""
    nome = caminho.name.replace("_test.parquet", "_dev.parquet")
    return caminho.with_name(nome) if caminho.is_relative_to(entregas) and caminho.with_name(nome).exists() else base / "previsoes" / nome


def fichas_do_zero(entregas: Path, base: Path) -> dict[str, dict]:
    """O caminho do zero: o catálogo da Etapa 9, com as marcas do registro, as respostas de `dev` e o INT8 sem poda de cada modelo.

    As respostas do teste que a Etapa 9 produziu (INT8 e linha de base com
    poucos rótulos) são lidas da **entrega** dela. As de `dev` desses mesmos
    sistemas não existem em nenhuma entrega: `preparar` as produz.
    """
    fichas = etapa9.catalogo(entregas, entregas / E9)
    quantizacao = _json(entregas / "08_quantizacao" / "decisao.json")
    for chave, ficha in fichas.items():
        partes = chave.split("/")
        if chave.startswith("base_"):
            marcas = {"caminho": None, "modelo": None, "ponto": "base"}
        elif "papel" in ficha:
            marcas = {"caminho": None, "modelo": None, "ponto": "professor"}
        elif len(partes) == 1:
            marcas = {"caminho": ZERO, "modelo": chave, "ponto": {"teto": "sem_compressao", "piso": "piso"}.get(chave, "destilado")}
            if "degrau_gravado" in ficha:   # um aluno: o professor está na comparação que a Etapa 6 gravou
                marcas["professor"] = "claude" if "fechado" in ficha["degrau_gravado"][1]["comparacao"] else "qwen"
        else:
            marcas = {"caminho": ZERO, "modelo": partes[0], "ponto": partes[1], "professor": fichas[partes[0]].get("professor")}
        ficha.update(marcas)
        teste = ficha["previsoes"]
        ficha["previsoes"] = {"test": teste, "dev": [_em_dev(c, entregas, base) for c in teste]}
    for chave in [c for c, f in fichas.items() if f["ponto"] == "int8"]:   # o mesmo modelo, em INT8, sem passar pela poda
        partida = fichas[chave]["de_partida"]
        modelo = fichas[partida]["modelo_da_poda"]
        q = quantizacao["modelos"][f"{modelo}/sem_poda"]
        fichas[f"{partida}/int8_sem_poda"] = {
            "nome": f"{fichas[partida]['curto']}, sem poda, {q['nome_da_forma']}", "curto": f"{fichas[partida]['curto']}, INT8 sem poda", "bloco": "Quantizado sem poda",
            "rotulos_humanos": fichas[partida]["rotulos_humanos"], "rotulos_como": fichas[partida].get("rotulos_como"),
            "dev_humano": f"{fichas[partida]['dev_humano']}; regra da forma de 8 bits (Etapa 8)", "resultado": ("08_quantizacao", {"modelo": modelo, "variante": "sem_poda", "forma": q["forma"]}),
            "previsoes": {p: [base / "previsoes" / f"{modelo}__sem_poda__{q['forma']}_semente_{s}_{p}.parquet" for s in SEMENTES] for p in PARTICOES},
            "gerar": {"como": "onnx", "de": [entregas.parent / q["pastas_por_semente"][str(s)] for s in SEMENTES]},
            "condicao": CPU, "onde": "1 thread, ONNX Runtime", "compactado": bool(q["compactado"]), "de_partida": partida, "sementes": len(SEMENTES),
            "caminho": ZERO, "modelo": partida, "ponto": "int8_sem_poda", "professor": fichas[partida].get("professor"),
        }
    final = quantizacao["final"]
    for chave, ficha in fichas.items():   # como carregar cada forma do modelo final, para a medida de latência na mesma sessão
        if ficha.get("final"):
            ficha["carregar"] = {
                "int8": ("onnx_diet", entregas.parent / final["pastas_por_semente"]["0"]),
                "onnx": ("exportar_diet", entregas.parent / final["em_32_bits"] / "semente_0"),
                "pytorch": ("diet", entregas.parent / final["em_32_bits"] / "semente_0"),
            }
    return fichas


def fichas_do_encoder(entregas: Path, base: Path) -> dict[str, dict]:
    """O caminho do XLM-R, lido dos `decisao.json` das Etapas 10a, 10b e 10c, com as mesmas chaves do caminho do zero atrás do prefixo.

    Como na Etapa 9: a qualidade do modelo sem poda vem da etapa que o treinou
    (10a) e a latência, da etapa de quantização (10c), que mediu todos na
    mesma sessão. O modelo podado em 32 bits tem as respostas da Etapa 10b e
    o número gravado pela 10c: as duas etapas são conferidas.
    """
    d10a, d10b, d10c = (_json(entregas / e / "decisao.json") for e in (E10A, E10B, E10C))
    poucos = _json(entregas / "05a_professor_fechado" / "configuracao.json")["conjunto_de_demonstracao"]["exemplos"]
    todos = _json(entregas / "01_dados" / "manifesto.json")["enunciados"]["train"]
    P = CAMINHOS[XLMR]["prefixo"]
    respostas = lambda etapa, molde: {p: [entregas / etapa / "previsoes" / f"{molde}_semente_{s}_{p}.parquet" for s in SEMENTES] for p in PARTICOES}
    cpu_1, onnx_1 = "1 thread, PyTorch", "1 thread, ONNX Runtime"
    fichas: dict[str, dict] = {}

    def sem_poda(chave: str, variante: str, comparacao: str, **campos) -> None:
        ficha = {"resultado": (E10A, {"variante": variante}), "previsoes": respostas(E10A, variante), "condicao": CPU, "onde": cpu_1, "sementes": len(SEMENTES),
                 "caminho": XLMR, "modelo": chave, "entre_caminhos_gravado": (E10A, "comparacoes.csv", {"particao": "test", "comparacao": f"{comparacao}: XLM-R − do zero"}), **campos}
        if f"{variante}/sem_poda" in d10c["modelos"]:   # foi até a quantização: a latência é a de lá, e o número de lá é conferido
            na_10c = (E10C, {"modelo": variante, "variante": "sem_poda", "forma": "pytorch"})
            ficha.update({"eficiencia": na_10c, "outras_fontes": [na_10c], "variante_da_10a": variante,
                          "zip": (E10B, "eficiencia.csv", {"modelo": variante, "variante": "de_partida"}, "sistema_com_zip")})
        fichas[P + chave] = ficha

    sem_poda("teto", "teto", "Teto", nome=f"Teto (XLM-R base com os {etapa9.milhar(todos)} rótulos humanos)", curto="Teto", bloco="Referências", rotulos_humanos=todos,
             dev_humano="época de parada e busca da receita (Etapa 10a)", ponto="sem_compressao")
    sem_poda("piso", "piso", "Piso", nome=f"Piso (XLM-R base só com os {etapa9.milhar(poucos)} rótulos humanos)", curto="Piso", bloco="Referências", rotulos_humanos=poucos,
             dev_humano="época de parada (Etapa 10a)", ponto="piso")
    for letra, aluno in d10a["alunos"].items():
        com_t = letra == "B2"
        sem_poda(f"aluno_{letra.lower()}", aluno["variante"], f"Aluno {letra}", nome=aluno["nome"][:-1] + f", T = {d10a['temperatura_do_b2']})" if com_t else aluno["nome"],
                 curto=f"Aluno {letra}", bloco="Alunos destilados", rotulos_humanos=poucos, rotulos_como="pelo professor", ponto="destilado",
                 dev_humano=f"temperatura T = {d10a['temperatura_do_b2']} escolhida em dev (Etapa 10a)" if com_t else "nenhuma no treino (parada no dev rotulado pelo professor)",
                 professor="claude" if letra == "A" else "qwen",
                 degrau_gravado=(E10A, {"particao": "test", "comparacao": f"Aluno {letra} − professor {'fechado' if letra == 'A' else 'aberto'}"}))

    de_quem = {f["variante_da_10a"]: c for c, f in fichas.items() if "variante_da_10a" in f}   # a variante da 10a de cada modelo que foi para a compressão
    final = d10c["final"]
    for modelo, do_modelo in d10b["modelos"].items():
        partida = de_quem[modelo]
        de_partida = fichas[partida]
        principal, solta = do_modelo["principal"], do_modelo["secundaria"]
        nome_da_poda = {f["variante"]: f["nome"] for f in do_modelo["seguem"]}
        bloco = {"aluno_a": "Cenário A", "teto": "Teto comprimido"}.get(de_partida["modelo"], "Cenário B")
        decidido = f"{de_partida['dev_humano'].replace('nenhuma no treino (parada no dev rotulado pelo professor)', 'nenhuma no treino')}; regra da poda (Etapa 10b)"
        comum = {"bloco": bloco, "rotulos_humanos": de_partida["rotulos_humanos"], "rotulos_como": de_partida.get("rotulos_como"), "condicao": CPU, "de_partida": partida,
                 "sementes": len(SEMENTES), "caminho": XLMR, "modelo": de_partida["modelo"], "professor": de_partida.get("professor")}
        em_8 = lambda variante: d10c["modelos"][f"{modelo}/{variante}"]
        linha_da_10c = lambda variante, forma: (E10C, {"modelo": modelo, "variante": variante, "forma": forma})
        if principal:
            fichas[f"{partida}/podado"] = {
                **comum, "nome": f"{de_partida['curto']}, {_minuscula(nome_da_poda[principal])}, PyTorch, 32 bits", "curto": f"{de_partida['curto']}, podado", "dev_humano": decidido, "ponto": "podado",
                "resultado": linha_da_10c(principal, "pytorch"), "outras_fontes": [(E10B, {"modelo": modelo, "variante": principal, "momento": "depois do reajuste"})],
                "zip": (E10B, "eficiencia.csv", {"modelo": modelo, "variante": principal}, "sistema_com_zip"), "previsoes": respostas(E10B, f"{modelo}__{principal}"), "onde": cpu_1,
                "degrau_gravado": (E10B, {"modelo": modelo, "variante": principal, "comparacao": "depois − sem poda", "particao": "test"}),
            }
            fichas[f"{partida}/onnx"] = {
                **comum, "nome": f"{de_partida['curto']}, {_minuscula(nome_da_poda[principal])}, ONNX, 32 bits", "curto": f"{de_partida['curto']}, ONNX 32 bits", "dev_humano": decidido, "ponto": "onnx",
                "resultado": linha_da_10c(principal, "fp32"), "previsoes": respostas(E10C, f"{modelo}__{principal}__fp32"), "onde": onnx_1,
                "degrau_gravado": (E10C, {"modelo": modelo, "variante": principal, "forma": "fp32", "particao": "test"}),
            }
        em_8_bits = [("int8", principal, "INT8"), ("soltos", solta, "pesos soltos, INT8 com zip"), ("int8_sem_poda", "sem_poda", "INT8 sem poda")]
        for ponto, variante, rotulo in em_8_bits:
            if not variante:
                continue
            q = em_8(variante)
            compactado = bool(q["pastas_compactadas_por_semente"])
            nome_da_variante = "sem poda" if variante == "sem_poda" else _minuscula(nome_da_poda[variante])
            fichas[f"{partida}/{ponto}"] = {
                **comum, "nome": f"{de_partida['curto']}, {nome_da_variante}, {q['nome_da_forma']}" + (", arquivo compactado" if compactado else ""),
                "curto": f"{de_partida['curto']}, {rotulo}", "ponto": ponto, "dev_humano": f"{decidido if variante != 'sem_poda' else de_partida['dev_humano']}; regra da forma de 8 bits (Etapa 10c)",
                "resultado": linha_da_10c(variante, q["forma"]), "previsoes": respostas(E10C, f"{modelo}__{variante}__{q['forma']}"), "onde": onnx_1, "compactado": compactado,
                **({"bloco": "Quantizado sem poda"} if variante == "sem_poda" else {}),
            }
        if principal:
            forma = em_8(principal)["forma"]
            fichas[f"{partida}/int8"]["degrau_gravado"] = (E10C, {"modelo": modelo, "variante": principal, "forma": forma, "particao": "test"})
            fichas[f"{partida}/int8"]["final"] = (final["modelo"], final["variante"]) == (modelo, principal)
            if fichas[f"{partida}/int8"]["final"]:
                fichas[f"{partida}/int8"]["carregar"] = {
                    "int8": ("onnx_encoder", entregas.parent / final["pastas_por_semente"]["0"]), "onnx": ("onnx_encoder", entregas.parent / final["onnx_de_32_bits"]),
                    "pytorch": ("encoder", entregas.parent / final["em_32_bits"]["0"]),
                }
    return fichas


def vale(pasta_da_entrega: Path, valem: tuple[str, ...] | None) -> bool:
    """A entrega está na lista das que valem? `valem=None`: todas valem."""
    return valem is None or pasta_da_entrega.name in valem


def fichas_declaradas(entregas: Path, valem: tuple[str, ...] | None = None) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Os sistemas que uma etapa posterior declara num arquivo `sistemas.json` da sua entrega: (caminhos novos, fichas, cadeias).

    `valem` é a lista das entregas (o nome da pasta) cujo arquivo é lido; sem
    ela, todas as que estão em disco. As funções do registro passam a lista
    do seu comparativo (aqui, `ENTREGAS_QUE_VALEM`).

    É a porta de entrada das etapas que ainda não existem. O arquivo tem:

        {"caminhos": {"id": {"nome": ..., "curto": ..., "prefixo": "id/"}},          opcional: um caminho novo
         "sistemas": {"chave": {ficha}},                                              as fichas, nos mesmos campos de `registro`
         "cadeias":  {"id": {"nome": ..., "caminho": ..., "degraus": [[tipo, chave], ...]}}}   opcional

    Na ficha, `resultado` é [etapa, filtro], `previsoes` é {"dev": [...],
    "test": [...]} com caminhos a partir da raiz do projeto, e as marcas
    `caminho`, `modelo`, `ponto` e `professor` decidem em que comparações o
    sistema entra.
    """
    caminhos, fichas, cadeias = {}, {}, {}
    for arquivo in sorted(entregas.glob("*/sistemas.json")):
        if not vale(arquivo.parent, valem):
            continue
        do_arquivo = fichas_de(_json(arquivo), entregas.parent, str(arquivo.relative_to(entregas.parent)))
        for junto, novo in zip((caminhos, fichas, cadeias), do_arquivo):
            junto.update(novo)
    return caminhos, fichas, cadeias


def fichas_de(declarado: dict, raiz: Path, declarado_em: str) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """O conteúdo de um `sistemas.json` (já lido), levado para o formato do registro: (caminhos, fichas, cadeias).

    `raiz` é a pasta do projeto, de onde partem os caminhos das respostas;
    `declarado_em` fica anotado em cada ficha.
    """
    par = lambda valor: (valor[0], valor[1]) if valor else None
    caminhos = dict(declarado.get("caminhos", {}))
    cadeias = {nome: {**c, "degraus": tuple(tuple(d) for d in c["degraus"])} for nome, c in declarado.get("cadeias", {}).items()}
    fichas = {}
    for chave, ficha in declarado.get("sistemas", {}).items():
        ficha = {"condicao": None, "onde": "não medida", "bloco": "Declarados", "dev_humano": "não informado", **ficha, "declarado_em": declarado_em}
        ficha["resultado"] = par(ficha["resultado"])
        for campo in ("eficiencia", "degrau_gravado"):
            if campo in ficha:
                ficha[campo] = par(ficha[campo])
        ficha["previsoes"] = {p: [raiz / c for c in lista] for p, lista in ficha["previsoes"].items()}
        ficha.setdefault("sementes", len(ficha["previsoes"]["test"]))
        fichas[chave] = ficha
    return caminhos, fichas, cadeias


FONTES = (fichas_do_zero, fichas_do_encoder)   # uma função por caminho: recebe (entregas, base) e devolve as fichas


def caminhos_do_registro(entregas: Path | None = None, valem: tuple[str, ...] | None = ENTREGAS_QUE_VALEM) -> dict[str, dict]:
    """Os caminhos conhecidos: os deste módulo e os que as entregas que valem declaram."""
    return {**CAMINHOS, **fichas_declaradas(entregas or ENTREGAS, valem)[0]}


def registro(entregas: Path | None = None, base: Path | None = None, valem: tuple[str, ...] | None = ENTREGAS_QUE_VALEM) -> dict[str, dict]:
    """Todos os sistemas dos dois caminhos (e os declarados pelas entregas que valem), na ordem das linhas da tabela.

    Com o padrão (`ENTREGAS_QUE_VALEM`), é o registro com que esta etapa foi
    entregue. Outro comparativo passa a sua lista em `valem`.

    Campos de uma ficha, além dos da Etapa 9 (`etapa9.catalogo`):

    - `caminho`: `zero`, `xlmr`, ou None para o que os caminhos dividem (professores e linhas de base);
    - `modelo`, `ponto`: quem é o sistema e em que ponto da cadeia está (as chaves de `PONTOS`);
    - `professor`: a chave da LLM que rotulou o treino dele, quando há;
    - `previsoes`: {"dev": [...], "test": [...]}, um arquivo por semente;
    - `outras_fontes`: outras etapas que gravaram a mesma medida, para a conferência;
    - `entre_caminhos_gravado`: a comparação "este sistema − o mesmo no caminho do zero" que uma etapa já gravou.
    """
    entregas, base = entregas or ENTREGAS, base or base_padrao()
    fichas: dict[str, dict] = {}
    for fonte in FONTES:
        fichas.update(fonte(entregas, base))
    fichas.update(fichas_declaradas(entregas, valem)[1])
    return fichas


def chave_no(caminho: str, chave: str, fichas: dict[str, dict], caminhos: dict[str, dict] | None = None) -> str:
    """A chave, em outro caminho, do sistema que no caminho do zero se chama `chave`. O que os caminhos dividem não muda de chave."""
    if fichas[chave].get("caminho") is None:
        return chave
    return (caminhos or CAMINHOS)[caminho]["prefixo"] + chave


def finais(fichas: dict[str, dict]) -> dict[str, str]:
    """O modelo final de cada caminho: {caminho: chave}."""
    return {f["caminho"]: c for c, f in fichas.items() if f.get("final")}


# --------------------------------------------------------------------------
# As regras: a lista fechada de comparações, gerada das marcas do registro
# --------------------------------------------------------------------------


def cadeias_do_registro(fichas: dict[str, dict], entregas: Path | None = None, valem: tuple[str, ...] | None = ENTREGAS_QUE_VALEM) -> dict[str, dict]:
    """As cadeias de cada caminho: as três da Etapa 9 (cenário A, cenário B, teto), com as chaves de cada caminho, mais as declaradas."""
    caminhos = caminhos_do_registro(entregas, valem)
    cadeias = {}
    for caminho, do_caminho in caminhos.items():
        for nome, c in etapa9.CADEIAS.items():
            degraus = tuple((tipo, chave_no(caminho, chave, fichas, caminhos)) for tipo, chave in c["degraus"])
            if all(chave in fichas for _, chave in degraus):
                cadeias[f"{caminho}/{nome}"] = {"nome": f"{do_caminho['curto']} · {c['nome']}", "caminho": caminho, "cenario": nome, "degraus": degraus}
    cadeias.update(fichas_declaradas(entregas or ENTREGAS, valem)[2])
    return cadeias


def lista_de_comparacoes(fichas: dict[str, dict], entregas: Path | None = None, valem: tuple[str, ...] | None = ENTREGAS_QUE_VALEM) -> list[dict]:
    """A lista fechada das comparações pareadas B − A, gerada por regra. Um par que cai em duas famílias aparece uma vez, com as duas.

    Famílias:

    - `entre_caminhos`: o mesmo modelo, no mesmo ponto da cadeia (`ENTRE_CAMINHOS`), no outro caminho menos no do zero;
    - `finais`: o modelo final de cada caminho menos o final do caminho do zero;
    - `aluno_professor`: cada aluno, como saiu da destilação e na forma final de 8 bits, menos o professor dele;
    - `ponta_a_ponta`: as dez comparações de ponta a ponta da Etapa 9 (`etapa9.PONTA_A_PONTA`), repetidas em cada caminho;
    - `poda_nos_8_bits`: em cada caminho, o modelo podado em INT8 menos o mesmo modelo em INT8 sem poda.
    """
    caminhos = caminhos_do_registro(entregas, valem)
    pares: dict[tuple[str, str], dict] = {}

    def por(familia: str, ident: str, b: str, a: str, pergunta: str, gravado: tuple | None = None) -> None:
        item = pares.setdefault((b, a), {"id": ident, "familias": [], "b": b, "a": a, "pergunta": pergunta, "gravado": None})
        item["familias"].append(familia)
        item["gravado"] = item["gravado"] or gravado

    do_zero = {(f["modelo"], f["ponto"]): c for c, f in fichas.items() if f.get("caminho") == REFERENCIA}
    fim = finais(fichas)
    for chave, f in fichas.items():   # 1. entre caminhos, no mesmo ponto
        if f.get("caminho") not in (None, REFERENCIA) and f["ponto"] in ENTRE_CAMINHOS and (f["modelo"], f["ponto"]) in do_zero:
            por("entre_caminhos", f"caminhos:{chave}", chave, do_zero[(f["modelo"], f["ponto"])], f"{f['curto']}: {caminhos[f['caminho']]['curto']} menos a rede do zero", f.get("entre_caminhos_gravado"))
    for caminho, chave in fim.items():   # 2. final contra final
        if caminho != REFERENCIA and REFERENCIA in fim:
            por("finais", f"finais:{caminho}", chave, fim[REFERENCIA], f"o modelo final do caminho {caminhos[caminho]['curto']} menos o modelo final da rede do zero")
    for chave, f in fichas.items():   # 3. cada aluno contra o seu professor
        if f.get("professor") and f["ponto"] in CONTRA_O_PROFESSOR:
            por("aluno_professor", f"professor:{chave}", chave, f["professor"], f"{caminhos[f['caminho']]['curto']}: {f['curto']} menos o professor dele")
    for caminho, do_caminho in caminhos.items():   # 4. as de ponta a ponta da Etapa 9, em cada caminho
        for letra, b, a, pergunta in etapa9.PONTA_A_PONTA:
            if b not in fichas or a not in fichas:
                continue
            b_, a_ = chave_no(caminho, b, fichas, caminhos), chave_no(caminho, a, fichas, caminhos)
            if b_ in fichas and a_ in fichas:
                por("ponta_a_ponta", f"ponta:{caminho}:{letra}", b_, a_, f"{do_caminho['curto']}: {pergunta}", (E9, "comparacoes.csv", {"letra": letra}) if caminho == ZERO else None)
    for chave, f in fichas.items():   # 5. a poda, vista dentro dos 8 bits
        sem_poda = f"{f.get('de_partida')}/int8_sem_poda"
        if f["ponto"] == "int8" and sem_poda in fichas:
            por("poda_nos_8_bits", f"poda8:{chave}", chave, sem_poda, f"{caminhos[f['caminho']]['curto']}: {NOME_DO_MODELO.get(f['modelo'], f['modelo'])} podado em INT8 menos o mesmo modelo em INT8 sem poda")
    return list(pares.values())


def montar_regras(fichas: dict[str, dict], entregas: Path | None = None, valem: tuple[str, ...] | None = ENTREGAS_QUE_VALEM) -> dict:
    """O texto das regras desta etapa: o método, a lista fechada de comparações e os critérios de leitura. Não lê nenhum resultado."""
    entregas = entregas or ENTREGAS
    lista = lista_de_comparacoes(fichas, entregas, valem)
    cadeias = cadeias_do_registro(fichas, entregas, valem)
    fim = finais(fichas)
    decisoes = sorted({str(c.relative_to(entregas.parent)) for c in entregas.glob("*/decisao.json") if vale(c.parent, valem)})
    return {
        "etapa": ETAPA, "pergunta": PERGUNTA,
        "metodo": {
            "decisoes_nesta_etapa": "nenhuma: os modelos finais já foram escolhidos em dev pelas regras das Etapas 6, 7, 8, 10a, 10b e 10c",
            "particoes": "dev e teste, contra os rótulos humanos; a leitura principal é a do teste, e dev vai ao lado; quando as duas discordam, a discordância é dita",
            "sementes": len(SEMENTES), "media": "um sistema treinado é a média das suas 3 sementes",
            "intervalo": {"metodo": "bootstrap por enunciado", "reamostragens": REAMOSTRAS, "nivel": 0.95, "pareado": "os mesmos sorteios para os dois sistemas", "semente_do_sorteio": 0},
            "metricas_das_leituras": list(TRES),
            "leitura": "uma diferença só vale quando o intervalo de 95% não contém zero; senão, 'sem diferença distinguível'",
            "comparacoes_multiplas": "sem correção; o número esperado de intervalos fora do zero só por acaso (5%) é dito ao lado da contagem",
            "primarias": "o modelo final de cada caminho contra o final da rede do zero, nas três métricas, no teste; as outras descrevem de onde a diferença vem",
        },
        "familias": {
            "entre_caminhos": f"o mesmo modelo no mesmo ponto da cadeia ({', '.join(ENTRE_CAMINHOS)}), outro caminho menos rede do zero. O ONNX de 32 bits fica de fora: nos dois caminhos ele dá as mesmas respostas do modelo podado",
            "finais": "modelo final de cada caminho menos o modelo final da rede do zero",
            "aluno_professor": f"cada aluno ({', '.join(CONTRA_O_PROFESSOR)}) menos o professor que rotulou o treino dele, nos dois caminhos",
            "ponta_a_ponta": "as dez comparações de ponta a ponta fixadas na Etapa 9, repetidas com as chaves de cada caminho",
            "poda_nos_8_bits": "em cada caminho, o modelo podado em INT8 menos o mesmo modelo em INT8 sem poda (os dois estão na fronteira; esta comparação diz se eles se distinguem)",
        },
        "comparacoes": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in item.items() if k != "gravado"} for item in lista],
        "cadeias": {nome: {"nome": c["nome"], "degraus": [list(d) for d in c["degraus"]]} for nome, c in cadeias.items()},
        "degraus_das_cadeias": "em cada cadeia, cada degrau menos o anterior; a diferença calculada aqui tem de ser igual à que a etapa de origem gravou",
        "finais": fim,
        "fronteira": {
            "definicao": "a da Etapa 9 (etapa9.nao_dominados): um sistema é dominado quando existe outro com custo menor ou igual e exact match maior ou igual, e melhor em pelo menos um dos dois",
            "candidatos": "todos os sistemas do registro que usam só os poucos rótulos humanos (o cenário do projeto), dos dois caminhos; teto e linha de base com todos os rótulos aparecem como referência",
            "custos": "latência p50 (uma consulta, 1 thread) e tamanho do arquivo como é entregue (o compactado, quando o sistema é entregue compactado)",
            "folga_da_latencia": "duas latências mais próximas que a maior variação relativa da mediana entre as rodadas das medidas de origem (Etapas 8 e 10c) contam como empate; o tamanho não tem folga",
            "qualidade": "exact match no teste, pelo valor medido, sem folga: quem diz se dois vizinhos se distinguem são as comparações pareadas",
        },
        "preco_de_cada_ponto": {
            "o_que_e": "entre os dois modelos finais: (tamanho do final do outro caminho − tamanho do final do zero) ÷ (diferença de exact match, em pontos percentuais); o mesmo com a latência p50",
            "intervalo": "os limites do intervalo de 95% da diferença pareada de exact match, levados para a razão; só é definido quando o intervalo inteiro está acima de zero",
            "onde": "no teste (leitura principal) e em dev; a latência é a das etapas de origem, e a medida na mesma sessão vai ao lado",
        },
        "latencia_na_mesma_sessao": {
            "sistemas": "os dois modelos finais, semente 0, em INT8, em ONNX de 32 bits e em PyTorch de 32 bits",
            "protocolo": "o das Etapas 8 e 10c (eficiencia.medir_latencia): uma consulta por vez, 1 thread, do texto à resposta, pedidos de dev, 50 de aquecimento, 5 rodadas de 1.024 consultas, sistemas alternados dentro de cada rodada",
            "maquina_quieta": f"os outros processos ocupam no máximo {LIMITE_DE_OUTROS} núcleo de CPU durante a medida (etapa8.nucleos_de_outros), o limite das Etapas 8 e 10c",
            "leitura": "os números de origem continuam sendo os da tabela; a medida nova vai ao lado. Uma latência nova 'confirma' a de origem quando a diferença relativa entre as duas é menor que a folga da fronteira",
        },
        "conferencia": {"tolerancia": TOLERANCIA, "regra": "cada número de qualidade e cada diferença que uma etapa de origem gravou é refeito das respostas; se discordar, é relatado, não consertado"},
        "ja_era_conhecido_ao_escrever": [
            "as comparações entre caminhos nos modelos sem poda (teto, piso, alunos A, B1 e B2) já estavam no RESUMO da Etapa 10a",
            "os degraus de cada cadeia e as comparações de ponta a ponta do caminho do zero já estavam nas Etapas 6 a 10c",
            "os valores de qualidade, tamanho e latência de cada sistema, um a um, já estavam nos RESUMOs",
            "novas nesta etapa: entre caminhos nos pontos podado e INT8 (inclui final contra final), aluno contra professor em INT8 no XLM-R, ponta a ponta do XLM-R, poda nos 8 bits, a fronteira conjunta e o preço de cada ponto",
        ],
        "decisoes_lidas": {d: sha256(entregas.parent / d) for d in decisoes},
        "sistemas_no_registro": list(fichas),
    }


def gravar_regras(base: Path | None = None, entregas: Path | None = None) -> Path:
    """Grava `regras.json`. Se já existe um com outro conteúdo (o registro cresceu), grava `regras_vN.json` e não toca no anterior."""
    base = base or base_padrao()
    base.mkdir(parents=True, exist_ok=True)
    regras = montar_regras(registro(entregas, base), entregas)
    existentes = sorted(base.glob("regras*.json"), key=lambda c: _json(c)["versao"])
    if existentes:
        ultima = _json(existentes[-1])
        if {k: v for k, v in ultima.items() if k not in ("versao", "gravado_em")} == json.loads(json.dumps(regras)):
            return existentes[-1]
    versao = len(existentes) + 1
    destino = base / ("regras.json" if versao == 1 else f"regras_v{versao}.json")
    destino.write_text(json.dumps({"versao": versao, "gravado_em": datetime.now().isoformat(timespec="seconds"), **regras}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destino


def carregar_regras(base: Path | None = None) -> tuple[dict, Path]:
    """A versão mais nova das regras gravadas, e o arquivo. Dá erro se nenhuma foi gravada: as contas não rodam sem regra escrita."""
    existentes = sorted((base or base_padrao()).glob("regras*.json"), key=lambda c: _json(c)["versao"])
    if not existentes:
        raise FileNotFoundError("Grave as regras antes de calcular: uv run python -m especialista.experimentos.etapa10d regras")
    return _json(existentes[-1]), existentes[-1]


# --------------------------------------------------------------------------
# O que não estava gravado
# --------------------------------------------------------------------------


def faltam(fichas: dict[str, dict]) -> list[Path]:
    """Os arquivos de respostas que ainda não existem."""
    return [c for f in fichas.values() for lista in f["previsoes"].values() for c in lista if not c.exists()]


def preparar(entregas: Path | None = None, base: Path | None = None, refazer: bool = False) -> Path:
    """Produz, na CPU, as respostas que nenhuma etapa gravou, do mesmo jeito que a Etapa 9 fez para o teste.

    São as do caminho do zero: os modelos INT8 em `dev` (a Etapa 9 só precisou
    do teste), o INT8 sem poda em `dev` e no teste, e a linha de base com os
    poucos rótulos em `dev`. Cada modelo é carregado da entrega da Etapa 8.
    """
    from especialista.compressao.sistema_onnx import SistemaOnnx
    from especialista.dados import massive
    from especialista.modelos.linha_de_base import LinhaDeBase

    entregas, base = entregas or ENTREGAS, base or base_padrao()
    fichas = registro(entregas, base)
    particoes, _ = massive.carregar_entrega(entregas / "01_dados")
    geradas = {}
    for chave, ficha in fichas.items():
        gerar = ficha.get("gerar")
        if not gerar:
            continue
        for particao, destinos in ficha["previsoes"].items():
            aqui = [d for d in destinos if d.is_relative_to(base)]
            if not aqui:
                continue
            origens = gerar["de"] if len(gerar["de"]) == len(destinos) else gerar["de"] * len(destinos)
            for origem, destino in zip(origens, destinos, strict=True):
                if destino.exists() and not refazer:
                    continue
                textos = list(particoes[particao]["utt"])
                previsoes = SistemaOnnx.carregar(origem).prever(textos) if gerar["como"] == "onnx" else LinhaDeBase(caracteres=True).treinar(pd.read_parquet(origem)).prever(textos)
                salvar_previsoes(destino, particoes[particao]["id"], previsoes)
            geradas.setdefault(chave, {"como": gerar["como"], "de": [etapa9._relativo(c, entregas.parent) for c in gerar["de"]], "arquivos": {}})["arquivos"][particao] = [d.name for d in destinos]
    base.mkdir(parents=True, exist_ok=True)
    (base / "medidas.json").write_text(json.dumps({"etapa": ETAPA, "gerado_em": datetime.now().isoformat(timespec="seconds"), "respostas_geradas": geradas}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return base


# --------------------------------------------------------------------------
# A tabela, a conferência, as cadeias e as comparações
# --------------------------------------------------------------------------


def filtro_em(filtro: dict, particao: str) -> dict:
    """O filtro de uma linha gravada, escrito para o teste, levado para outra partição: troca `test` onde ele aparece como valor ou no começo de um nome."""
    troca = lambda v: particao if v == "test" else particao + v[4:] if isinstance(v, str) and v.startswith("test · ") else v
    return {coluna: troca(valor) for coluna, valor in filtro.items()}


def respostas(fichas: dict[str, dict], particao: str, ids: list[int]) -> dict[str, list[list]]:
    """As respostas de cada sistema numa partição, uma lista por semente, com a ordem dos pedidos conferida."""
    return {chave: etapa9.respostas({"previsoes": ficha["previsoes"][particao]}, ids) for chave, ficha in fichas.items()}


def tabela_final(fichas: dict[str, dict], entregas: Path | None = None) -> pd.DataFrame:
    """A tabela final estendida: a da Etapa 9 (`etapa9.tabela_final`, no teste) com as linhas dos dois caminhos, as marcas do registro e a qualidade em `dev`.

    As colunas de `dev` têm o prefixo `dev_` e vêm do mesmo `resultados.csv` de cada etapa de origem.
    """
    entregas = entregas or ENTREGAS
    T = etapa9.tabela_final(fichas, _json(entregas / E9 / "medidas_novas.json"), entregas)
    for marca in ("caminho", "modelo", "ponto", "professor"):
        T[marca] = [fichas[c].get(marca) for c in T.index]
    T["final"] = [bool(fichas[c].get("final")) for c in T.index]
    ler = etapa9._Tabelas(entregas)
    for chave, ficha in fichas.items():
        etapa, filtro = ficha["resultado"]
        try:
            r = etapa9.linha_gravada(ler(etapa), particao="dev", **filtro)
        except LookupError:
            continue
        for metrica in qualidade.METRICAS:
            for sufixo in ("", "_ic_min", "_ic_max"):
                T.loc[chave, f"dev_{metrica}{sufixo}"] = r[f"{metrica}{sufixo}"]
    return T


def em_dev(T: pd.DataFrame) -> pd.DataFrame:
    """A mesma tabela com a qualidade de `dev` nas colunas de qualidade: é o que `etapa9.cadeias` lê para montar as cadeias em `dev`."""
    D = T.copy()
    for metrica in qualidade.METRICAS:
        for sufixo in ("", "_ic_min", "_ic_max"):
            D[f"{metrica}{sufixo}"] = T[f"dev_{metrica}{sufixo}"]
    return D


def conferir_qualidade(fichas: dict[str, dict], gabaritos: dict[str, list], grupos: dict[str, dict], entregas: Path | None = None) -> pd.DataFrame:
    """Recalcula, das respostas, cada métrica e cada intervalo, em `dev` e no teste, e compara com cada etapa que gravou o mesmo número.

    Uma linha por (sistema, partição, etapa que gravou): a maior diferença
    absoluta, no valor ou nos limites do intervalo. `gravado` é falso quando a
    etapa de origem não tem aquela linha (não há contra o que conferir).
    """
    ler = etapa9._Tabelas(entregas or ENTREGAS)
    linhas = []
    for chave, ficha in fichas.items():
        fontes = [ficha["resultado"], *ficha.get("outras_fontes", []), *([ficha["outra_fonte"]] if "outra_fonte" in ficha else [])]
        for particao in PARTICOES:
            refeito = qualidade.intervalo_do_grupo(gabaritos[particao], grupos[particao][chave], n=REAMOSTRAS)
            for etapa, filtro in fontes:
                linha = {"o_que": "qualidade de um sistema", "chave": chave, "particao": particao, "gravado_em": f"entregas/{etapa}/resultados.csv",
                         "respostas_geradas_agora": "gerar" in ficha and any(base_padrao().name in str(c) for c in ficha["previsoes"][particao])}
                try:
                    gravado = etapa9.linha_gravada(ler(etapa), particao=particao, **filtro)
                except LookupError:
                    linhas.append({**linha, "gravado": False, "maior_diferenca": np.nan, "confere": None})
                    continue
                diferencas = [abs(refeito.loc[m, "valor"] - gravado[m]) for m in qualidade.METRICAS]
                diferencas += [abs(refeito.loc[m, limite] - gravado[f"{m}_{limite}"]) for m in qualidade.METRICAS for limite in ("ic_min", "ic_max")]
                linhas.append({**linha, "gravado": True, "maior_diferenca": float(max(diferencas)), "confere": bool(max(diferencas) <= TOLERANCIA)})
    return pd.DataFrame(linhas)


def _gravada(ler: etapa9._Tabelas, etapa: str, arquivo: str, filtro: dict) -> pd.DataFrame | None:
    """As linhas de uma comparação gravada, uma por métrica, ou None se a etapa não gravou essa comparação."""
    gravado = ler(etapa, arquivo)
    for coluna, valor in filtro.items():
        gravado = gravado[gravado[coluna] == valor]
    gravado = gravado[gravado["metrica"].isin(qualidade.METRICAS)].set_index("metrica")
    return gravado if sorted(gravado.index) == sorted(qualidade.METRICAS) else None


def conferir_degraus(fichas: dict[str, dict], degraus: pd.DataFrame, particao: str, entregas: Path | None = None) -> pd.DataFrame:
    """Cada degrau de cada cadeia, calculado aqui, contra a mesma diferença gravada pela etapa que deu aquele passo."""
    ler = etapa9._Tabelas(entregas or ENTREGAS)
    linhas = []
    for _, degrau in degraus.dropna(subset=["anterior"]).iterrows():
        gravado_em = fichas[degrau["chave"]].get("degrau_gravado")
        if not gravado_em:
            continue
        etapa, filtro = gravado_em
        gravado = _gravada(ler, etapa, "comparacoes.csv", filtro_em(filtro, particao))
        linha = {"o_que": "degrau de uma cadeia", "chave": degrau["chave"], "anterior": degrau["anterior"], "cadeia": degrau["cadeia"], "particao": particao, "gravado_em": f"entregas/{etapa}/comparacoes.csv"}
        if gravado is None:
            linhas.append({**linha, "gravado": False, "maior_diferenca": np.nan, "confere": None})
            continue
        diferencas = [abs(degrau[f"{m}_{c}"] - gravado.loc[m, g]) for m in qualidade.METRICAS for c, g in (("diferenca", "diferenca"), ("dif_ic_min", "ic_min"), ("dif_ic_max", "ic_max"))]
        linhas.append({**linha, "gravado": True, "maior_diferenca": float(max(diferencas)), "confere": bool(max(diferencas) <= TOLERANCIA)})
    return pd.DataFrame(linhas)


def comparar_lista(lista: list[dict], T: pd.DataFrame, gabarito: list, grupos: dict[str, list[list]], particao: str) -> pd.DataFrame:
    """As comparações da lista fechada numa partição, pela função da Etapa 9 (`etapa9.ponta_a_ponta`): uma linha por comparação e métrica."""
    tabela = etapa9.ponta_a_ponta(T, gabarito, grupos, tuple((item["id"], item["b"], item["a"], item["pergunta"]) for item in lista))
    familias = {item["id"]: "; ".join(item["familias"]) for item in lista}
    tabela = tabela.rename(columns={"letra": "id"})
    tabela.insert(1, "familias", tabela["id"].map(familias))
    tabela.insert(2, "particao", particao)
    return tabela


def conferir_comparacoes(lista: list[dict], comparacoes: pd.DataFrame, entregas: Path | None = None) -> pd.DataFrame:
    """As comparações da lista que alguma etapa já tinha gravado (`gravado` no item), contra o que essa etapa gravou."""
    ler = etapa9._Tabelas(entregas or ENTREGAS)
    linhas = []
    for item in lista:
        if not item["gravado"]:
            continue
        etapa, arquivo, filtro = item["gravado"]
        for particao, daqui in comparacoes[comparacoes["id"] == item["id"]].groupby("particao"):
            origem = ler(etapa, arquivo)
            if "particao" not in origem.columns and particao != "test":   # a Etapa 9 só gravou o teste
                continue
            gravado = _gravada(ler, etapa, arquivo, filtro_em(filtro, particao))
            linha = {"o_que": "comparação entre sistemas", "chave": item["b"], "anterior": item["a"], "particao": particao, "gravado_em": f"entregas/{etapa}/{arquivo}"}
            if gravado is None:
                linhas.append({**linha, "gravado": False, "maior_diferenca": np.nan, "confere": None})
                continue
            daqui = daqui.set_index("metrica")
            diferencas = [abs(daqui.loc[m, c] - gravado.loc[m, c]) for m in qualidade.METRICAS for c in ("diferenca", "ic_min", "ic_max")]
            linhas.append({**linha, "gravado": True, "maior_diferenca": float(max(diferencas)), "confere": bool(max(diferencas) <= TOLERANCIA)})
    return pd.DataFrame(linhas)


def conferir_com_a_etapa_9(T: pd.DataFrame, entregas: Path | None = None) -> pd.DataFrame:
    """As linhas da tabela final da Etapa 9, uma a uma, contra as mesmas linhas da tabela estendida: têm de ser idênticas."""
    da_9 = pd.read_csv((entregas or ENTREGAS) / E9 / "tabela_final.csv").set_index("chave")
    linhas = []
    for chave, antiga in da_9.iterrows():
        nova = T.loc[chave] if chave in T.index else None
        diferentes = list(da_9.columns) if nova is None else [c for c in da_9.columns if not (pd.isna(antiga[c]) and pd.isna(nova[c])) and not _igual(antiga[c], nova[c])]
        linhas.append({"o_que": "linha da tabela final da Etapa 9", "chave": chave, "particao": "test", "gravado_em": f"entregas/{E9}/tabela_final.csv", "gravado": True,
                       "maior_diferenca": 0.0 if not diferentes else np.nan, "confere": not diferentes, "colunas_diferentes": ", ".join(diferentes)})
    return pd.DataFrame(linhas)


def _igual(a, b) -> bool:
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)) and not isinstance(a, bool):
        return bool(abs(float(a) - float(b)) <= TOLERANCIA * max(1.0, abs(float(a))))
    return a == b


# --------------------------------------------------------------------------
# A fronteira e o preço de cada ponto
# --------------------------------------------------------------------------


def variacao_da_latencia(entregas: Path | None = None) -> float:
    """A folga da fronteira: a maior variação relativa da latência mediana entre as rodadas, nas medidas de origem dos dois caminhos (Etapas 8 e 10c)."""
    entregas = entregas or ENTREGAS
    variacoes = [pd.read_csv(entregas / etapa / "eficiencia.csv") for etapa in ("08_quantizacao", E10C)]
    return float(max(((e["p50_max_das_rodadas"] - e["p50_min_das_rodadas"]) / e["latencia_p50_ms"]).max() for e in variacoes))


def dominado_por(pontos: pd.DataFrame, custo: str, ganho: str, folga_do_custo: float = 0.0) -> dict:
    """Para cada ponto, quem o domina, pela definição da Etapa 9 (`etapa9.nao_dominados`): {índice: [índices dos que o vencem]}.

    Os pontos com lista vazia são a fronteira. Linhas sem uma das duas medidas ficam de fora.
    """
    validos = pontos.dropna(subset=[custo, ganho])
    f = 1.0 + folga_do_custo
    vence = lambda q, p: (q[custo] <= p[custo] * f and q[ganho] > p[ganho]) or (q[custo] * f < p[custo] and q[ganho] >= p[ganho])
    return {i: [j for j, q in validos.iterrows() if j != i and vence(q, p)] for i, p in validos.iterrows()}


def megas_entregues(T: pd.DataFrame) -> pd.Series:
    """O tamanho do arquivo como é entregue, em MB: o compactado, quando o sistema vai compactado."""
    return pd.Series(np.where(T["entregue_compactado"], T["bytes_com_zip"], T["bytes"]), index=T.index, dtype=float) / 1e6


def fronteiras(T: pd.DataFrame, folga: float, latencia: pd.Series | None = None) -> pd.DataFrame:
    """As duas fronteiras (latência e tamanho) entre os sistemas com os poucos rótulos humanos: quem está nelas e quem domina cada um dos outros.

    `latencia` troca a latência de alguns sistemas (a medida na mesma sessão), para ver se a fronteira muda.
    """
    candidatos = T[T["rotulos_humanos"] == T["rotulos_humanos"].min()].copy()
    candidatos["megas"] = megas_entregues(candidatos)
    if latencia is not None:
        candidatos.loc[latencia.index.intersection(candidatos.index), "latencia_p50_ms"] = latencia
    linhas = []
    for nome, custo, f in (("latencia", "latencia_p50_ms", folga), ("tamanho", "megas", 0.0)):
        vencedores = dominado_por(candidatos, custo, "exact_match", f)
        assert sorted(c for c, v in vencedores.items() if not v) == sorted(etapa9.nao_dominados(candidatos, custo, "exact_match", f))
        for chave, quem in vencedores.items():
            linhas.append({"custo": nome, "chave": chave, "valor_do_custo": candidatos.loc[chave, custo], "exact_match": candidatos.loc[chave, "exact_match"],
                           "na_fronteira": not quem, "dominado_por": "; ".join(quem), "quantos_o_dominam": len(quem)})
    return pd.DataFrame(linhas)


def preco_por_ponto(custo_a_mais: float, diferenca: float, ic_min: float, ic_max: float) -> dict:
    """Quanto custo a mais por ponto percentual de qualidade a mais, com o intervalo que vem do intervalo da diferença de qualidade.

    Só é definido quando a diferença de qualidade se distingue de zero para
    cima: com zero dentro do intervalo, o preço de um ponto não tem limite.
    """
    if not ic_min > 0:
        return {"definido": False, "valor": np.nan, "ic_min": np.nan, "ic_max": np.nan}
    return {"definido": True, "valor": custo_a_mais / (100 * diferenca), "ic_min": custo_a_mais / (100 * ic_max), "ic_max": custo_a_mais / (100 * ic_min)}


def precos(T: pd.DataFrame, comparacoes: pd.DataFrame, fichas: dict[str, dict], latencias: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """O preço de cada ponto de exact match entre os modelos finais: MB a mais e ms a mais por ponto, em `dev` e no teste.

    `latencias` são as fontes de latência: {"origem": a da tabela, "mesma sessão": a medida de novo}.
    """
    fim = finais(fichas)
    latencias = {"etapas de origem": T["latencia_p50_ms"], **(latencias or {})}
    linhas = []
    for caminho, chave in fim.items():
        if caminho == REFERENCIA:
            continue
        zero = fim[REFERENCIA]
        for particao in PARTICOES:
            c = comparacoes[(comparacoes["chave_b"] == chave) & (comparacoes["chave_a"] == zero) & (comparacoes["particao"] == particao) & (comparacoes["metrica"] == "exact_match")].iloc[0]
            custos = [("tamanho", "MB", megas_entregues(T).loc[chave], megas_entregues(T).loc[zero], "arquivos gravados")]
            custos += [("latencia", "ms", l.loc[chave], l.loc[zero], fonte) for fonte, l in latencias.items() if chave in l.index and zero in l.index]
            for custo, unidade, do_outro, do_zero, fonte in custos:
                linhas.append({"caminho": caminho, "particao": particao, "custo": custo, "unidade": unidade, "fonte_do_custo": fonte, "do_final": do_outro, "do_final_do_zero": do_zero,
                               "custo_a_mais": do_outro - do_zero, "vezes": do_outro / do_zero, "exact_match_a_mais": c["diferenca"], "exact_match_ic_min": c["ic_min"], "exact_match_ic_max": c["ic_max"],
                               **{f"por_ponto_{k}" if k != "definido" else k: v for k, v in preco_por_ponto(do_outro - do_zero, c["diferenca"], c["ic_min"], c["ic_max"]).items()}})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# A latência dos modelos finais, na mesma sessão
# --------------------------------------------------------------------------

FORMAS_MEDIDAS = ("int8", "onnx", "pytorch")   # as três formas em que cada etapa de origem mediu o modelo final
PONTO_DA_FORMA = {"int8": "int8", "onnx": "onnx", "pytorch": "podado"}


def _carregar_sistema(como: str, pasta: Path, onde_exportar: Path):
    if como == "onnx_diet":
        from especialista.compressao.sistema_onnx import SistemaOnnx
        return SistemaOnnx.carregar(pasta, threads=1)
    if como == "onnx_encoder":
        from especialista.compressao.sistema_encoder_onnx import SistemaEncoderOnnx
        return SistemaEncoderOnnx.carregar(pasta, threads=1)
    if como == "encoder":
        from especialista.modelos.encoder import SistemaEncoder
        return SistemaEncoder.carregar(pasta, "cpu")
    from especialista.modelos.diet import SistemaDiet
    sistema = SistemaDiet.carregar(pasta, "cpu")
    if como == "diet":
        return sistema
    # exportar_diet: a Etapa 8 não entregou o ONNX de 32 bits da rede do zero; ele é refeito do modelo entregue pela Etapa 7, com o código da Etapa 8
    from especialista.compressao import exportacao
    from especialista.compressao.sistema_onnx import SistemaOnnx
    return SistemaOnnx.carregar(exportacao.exportar(sistema, onde_exportar, origem=str(pasta)), threads=1)


def medir_latencia(base: Path | None = None, entregas: Path | None = None, rodadas: int = 5, consultas: int | None = None, conferir: int = 200) -> Path:
    """Mede, na mesma sessão, a latência dos dois modelos finais em INT8, ONNX de 32 bits e PyTorch de 32 bits.

    O protocolo é o das Etapas 8 e 10c, com a mesma função
    (`eficiencia.medir_latencia`) e o mesmo medidor de "outros processos"
    (`etapa8.nucleos_de_outros`). Os sistemas se alternam dentro de cada
    rodada. Cada medida vai para uma pasta própria, `latencia/<data-hora>/`:
    nenhuma medida anterior é sobrescrita.
    """
    import onnxruntime
    import torch

    from especialista.avaliacao import eficiencia
    from especialista.avaliacao.previsao import carregar_previsoes
    from especialista.dados import massive
    from especialista.experimentos import etapa8

    base, entregas = base or base_padrao(), entregas or ENTREGAS
    consultas = consultas or eficiencia.CONSULTAS
    fichas = registro(entregas, base)
    pasta = base / "latencia" / datetime.now().strftime("%Y%m%d-%H%M%S")
    pasta.mkdir(parents=True)
    torch.set_num_threads(1)
    particoes, _ = massive.carregar_entrega(entregas / "01_dados")
    textos = list(particoes["dev"]["utt"])
    sistemas, de_onde, reproduz = {}, {}, {}
    for forma in FORMAS_MEDIDAS:   # a ordem dentro da rodada: os dois caminhos lado a lado em cada forma
        for caminho, chave in finais(fichas).items():
            como, origem = fichas[chave]["carregar"][forma]
            nome = f"{caminho}/{forma}"
            sistemas[nome] = _carregar_sistema(como, origem, pasta / "sistemas" / f"{caminho}_fp32")
            de_onde[nome] = {"como": como, "pasta": etapa9._relativo(origem, entregas.parent)}
            # é mesmo o modelo da tabela? as primeiras respostas em dev têm de ser as gravadas (semente 0)
            ponto = chave.replace("/int8", "/" + PONTO_DA_FORMA[forma])
            gravadas = carregar_previsoes(fichas[ponto]["previsoes"]["dev"][0])[1][:conferir]
            reproduz[nome] = sistemas[nome].prever(textos[:conferir]) == gravadas

    tempos, outros_por_rodada = {nome: [] for nome in sistemas}, []
    carga_antes, antes = etapa8.carga_da_maquina(), etapa8.relogios()
    for rodada in range(rodadas):
        inicio = etapa8.relogios()
        for nome, sistema in sistemas.items():
            tempos[nome].append(eficiencia.medir_latencia(sistema, textos, consultas=consultas, semente=rodada))
        outros_por_rodada.append(etapa8.nucleos_de_outros(inicio, etapa8.relogios()))
    carga_depois, depois = etapa8.carga_da_maquina(), etapa8.relogios()

    np.savez(pasta / "latencia.npz", **{nome.replace("/", "__"): np.stack(linhas) for nome, linhas in tempos.items()})
    outros = etapa8.nucleos_de_outros(antes, depois)
    resumo = {
        "etapa": ETAPA, "medido_em": datetime.now().isoformat(timespec="seconds"), "semente": 0, "threads": 1, "rodadas": rodadas, "consultas_por_rodada": consultas,
        "aquecimento": eficiencia.AQUECIMENTO, "pedidos": "dev", "ordem_na_rodada": list(sistemas), "sistemas": de_onde, "reproduz_as_respostas_gravadas": reproduz,
        "p50_por_rodada_ms": {nome: [float(np.percentile(t, 50)) for t in linhas] for nome, linhas in tempos.items()},
        "resumo": {nome: eficiencia.resumir_latencia(np.concatenate(linhas)) for nome, linhas in tempos.items()},
        "nucleos_de_outros": outros, "nucleos_de_outros_por_rodada": outros_por_rodada, "limite_de_outros": LIMITE_DE_OUTROS, "maquina_quieta": bool(max([outros, *outros_por_rodada]) <= LIMITE_DE_OUTROS),
        "carga_antes": carga_antes, "carga_depois": carga_depois, "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, **etapa8.ambiente(),
    }
    (pasta / "latencia.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")   # por último: marca "pronto"
    return pasta


def latencias_medidas(base: Path | None = None) -> list[dict]:
    """Todas as medidas de latência desta etapa já gravadas, da mais antiga para a mais nova."""
    pastas = sorted(p for p in ((base or base_padrao()) / "latencia").glob("*") if (p / "latencia.json").exists())
    return [{"pasta": p, **_json(p / "latencia.json")} for p in pastas]


def latencia_ao_lado(medida: dict, T: pd.DataFrame, fichas: dict[str, dict]) -> pd.DataFrame:
    """A medida na mesma sessão ao lado da medida de origem de cada sistema: uma linha por (caminho, forma)."""
    fim = finais(fichas)
    linhas = []
    for nome, r in medida["resumo"].items():
        caminho, forma = nome.split("/")
        chave = fim[caminho].replace("/int8", "/" + PONTO_DA_FORMA[forma])
        rodadas = medida["p50_por_rodada_ms"][nome]
        linhas.append({"caminho": caminho, "forma": forma, "chave": chave, "p50_mesma_sessao_ms": r["latencia_p50_ms"], "p95_mesma_sessao_ms": r["latencia_p95_ms"], "media_mesma_sessao_ms": r["latencia_media_ms"],
                       "p50_min_das_rodadas": min(rodadas), "p50_max_das_rodadas": max(rodadas), "p50_de_origem_ms": T.loc[chave, "latencia_p50_ms"], "medida_de_origem": T.loc[chave, "latencia_medida_na"],
                       "diferenca_relativa": r["latencia_p50_ms"] / T.loc[chave, "latencia_p50_ms"] - 1, "reproduz_as_respostas": medida["reproduz_as_respostas_gravadas"][nome]})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Cada passo e o seu artigo, nos dois caminhos
# --------------------------------------------------------------------------

# O quadro parte do da Etapa 9 (`etapa9.ARTIGOS`) e junta o que as Etapas 10a, 10b e 10c citam. `passo` é um dos passos do
# caminho; `onde` diz em que caminho a técnica foi usada. `autor` é o sobrenome do primeiro autor e `titulo`, um trecho do
# título, como as etapas escrevem: é contra eles que a resposta do arXiv é conferida (`conferir_artigos`).
PASSOS = ("O caminho inteiro", "Dados e régua", "Professores", "Destilação", "Arquitetura", "Poda de vocabulário", "Corte de camadas", "Poda por magnitude", "Reajuste", "Quantização", "Exportação")
ARTIGOS = (
    {"passo": "O caminho inteiro", "onde": "os dois", "tecnica": "Trocar a LLM por um modelo pequeno, feito para a tarefa", "autor": "Belcak", "quem": "Belcak et al.", "titulo": "Small Language Models", "arxiv": "2506.02153"},
    {"passo": "O caminho inteiro", "onde": "os dois", "tecnica": "LLM professora, destilação, poda e quantização em sequência", "autor": "Ni", "quem": "Ni et al.", "titulo": "From Large to Super-Tiny", "arxiv": "2504.13471"},
    {"passo": "O caminho inteiro", "onde": "os dois", "tecnica": "A ordem: podar, quantizar e, por fim, codificar o arquivo", "autor": "Han", "quem": "Han et al.", "titulo": "Deep Compression", "arxiv": "1510.00149"},
    {"passo": "O caminho inteiro", "onde": "os dois", "tecnica": "A ordem das técnicas de compressão importa", "autor": "Harma", "quem": "Harma et al.", "titulo": "", "arxiv": "2405.20935"},
    {"passo": "O caminho inteiro", "onde": "os dois", "tecnica": "Poda e quantização juntas podem interagir (discussão da ordem)", "autor": "Nema", "quem": "Nema et al.", "titulo": "Compression Technique Interactions in Small Language Models", "arxiv": "2611.00464"},
    {"passo": "Dados e régua", "onde": "os dois", "tecnica": "MASSIVE pt-PT; acurácia de intenção, F1 de entidades e exact match; a linha publicada do XLM-R base", "autor": "FitzGerald", "quem": "FitzGerald et al.", "titulo": "MASSIVE", "arxiv": "2204.08582"},
    {"passo": "Dados e régua", "onde": "os dois", "tecnica": "Intervalo de 95% e comparação pareada por bootstrap, por enunciado", "autor": "Koehn", "quem": "Koehn", "titulo": "Statistical Significance Tests for Machine Translation Evaluation", "acl": "W04-3250", "ano": 2004},
    {"passo": "Dados e régua", "onde": "os dois", "tecnica": "Intervalo de 95% e comparação pareada por bootstrap, por enunciado", "autor": "Berg-Kirkpatrick", "quem": "Berg-Kirkpatrick et al.", "titulo": "An Empirical Investigation of Statistical Significance in NLP", "acl": "D12-1091", "ano": 2012},
    {"passo": "Dados e régua", "onde": "os dois", "tecnica": "Três sementes por modelo treinado; o resultado é a média", "autor": "Reimers", "quem": "Reimers & Gurevych", "titulo": "Reporting Score Distributions Makes a Difference", "arxiv": "1707.09861"},
    {"passo": "Dados e régua", "onde": "os dois", "tecnica": "Latência: uma consulta por vez, percentis, máquina registrada", "autor": "Reddi", "quem": "Reddi et al.", "titulo": "MLPerf Inference", "arxiv": "1911.02549"},
    {"passo": "Dados e régua", "onde": "os dois", "tecnica": "Custo reportado em várias medidas (parâmetros, bytes, tempo), com a condição de cada uma", "autor": "Dehghani", "quem": "Dehghani et al.", "titulo": "The Efficiency Misnomer", "arxiv": "2110.12894"},
    {"passo": "Professores", "onde": "os dois", "tecnica": "LLM com poucos exemplos no prompt: intenção perto do supervisionado, entidades atrás", "autor": "He", "quem": "He & Garner", "titulo": "", "arxiv": "2305.13512"},
    {"passo": "Destilação", "onde": "os dois", "tecnica": "Destilação caixa-preta: o aluno aprende só com os rótulos da LLM", "autor": "Xu", "quem": "Xu et al.", "titulo": "Survey on Knowledge Distillation of Large Language Models", "arxiv": "2402.13116"},
    {"passo": "Destilação", "onde": "os dois", "tecnica": "Destilação caixa-preta: o aluno aprende só com os rótulos da LLM", "autor": "Di Palo", "quem": "Di Palo et al.", "titulo": "", "arxiv": "2411.05045"},
    {"passo": "Destilação", "onde": "os dois", "tecnica": "Destilação com logits e temperatura", "autor": "Hinton", "quem": "Hinton et al.", "titulo": "Distilling the Knowledge in a Neural Network", "arxiv": "1503.02531"},
    {"passo": "Destilação", "onde": "os dois", "tecnica": "Destilação de um Transformer para um aluno menor", "autor": "Jiao", "quem": "Jiao et al.", "titulo": "TinyBERT", "arxiv": "1909.10351"},
    {"passo": "Arquitetura", "onde": "do zero", "tecnica": "Rede conjunta de intenção e entidades, do zero: features esparsas, Transformer de 2 camadas, CRF", "autor": "Bunk", "quem": "Bunk et al.", "titulo": "DIET", "arxiv": "2004.09936"},
    {"passo": "Arquitetura", "onde": "XLM-R", "tecnica": "Encoder pré-treinado com cabeça conjunta de intenção e entidades", "autor": "Chen", "quem": "Chen et al.", "titulo": "BERT for Joint Intent Classification and Slot Filling", "arxiv": "1902.10909"},
    {"passo": "Arquitetura", "onde": "XLM-R", "tecnica": "O encoder multilíngue pré-treinado (XLM-R base)", "autor": "Conneau", "quem": "Conneau et al.", "titulo": "Unsupervised Cross-lingual Representation Learning at Scale", "arxiv": "1911.02116"},
    {"passo": "Arquitetura", "onde": "XLM-R", "tecnica": "A receita de ajuste fino de um encoder pré-treinado (taxas, lote, épocas)", "autor": "Devlin", "quem": "Devlin et al.", "titulo": "BERT: Pre-training of Deep Bidirectional Transformers", "arxiv": "1810.04805"},
    {"passo": "Arquitetura", "onde": "XLM-R", "tecnica": "AdamW: o decaimento de pesos separado do gradiente", "autor": "Loshchilov", "quem": "Loshchilov & Hutter", "titulo": "Decoupled Weight Decay Regularization", "arxiv": "1711.05101"},
    {"passo": "Arquitetura", "onde": "XLM-R", "tecnica": "Ajuste fino com poucos exemplos: o que decide é o número de passos (o piso com 299 rótulos)", "autor": "Mosbach", "quem": "Mosbach et al.", "titulo": "On the Stability of Fine-tuning BERT", "arxiv": "2006.04884"},
    {"passo": "Poda de vocabulário", "onde": "XLM-R", "tecnica": "Cortar da tabela de subtokens o que o corpus da tarefa não usa, com o tokenizador junto", "autor": "Yang", "quem": "Yang, Cui & Chen", "titulo": "TextPruner", "arxiv": "2203.15996"},
    {"passo": "Poda de vocabulário", "onde": "XLM-R", "tecnica": "Num modelo multilíngue a maior parte dos parâmetros está na tabela; ficar só com os idiomas que interessam", "autor": "Abdaoui", "quem": "Abdaoui, Pradel & Sigel", "titulo": "Load What You Need", "arxiv": "2010.05609"},
    {"passo": "Poda de vocabulário", "onde": "do zero", "tecnica": "Poda de peças inteiras (estruturada): as linhas da tabela de features", "autor": "Xia", "quem": "Xia et al.", "titulo": "Structured Pruning Learns Compact and Accurate Models", "arxiv": "2204.00408"},
    {"passo": "Corte de camadas", "onde": "os dois", "tecnica": "Tirar as camadas de cima de um encoder já treinado e ajustar (testado nos dois caminhos; não ficou em nenhum modelo final)", "autor": "Sajjad", "quem": "Sajjad et al.", "titulo": "On the Effect of Dropping Layers of Pre-trained Transformer Models", "arxiv": "2004.03844"},
    {"passo": "Poda por magnitude", "onde": "os dois", "tecnica": "Treinar, podar pelos pesos de menor valor absoluto e treinar de novo", "autor": "Han", "quem": "Han et al.", "titulo": "Learning both Weights and Connections", "arxiv": "1506.02626"},
    {"passo": "Poda por magnitude", "onde": "os dois", "tecnica": "Poda gradual, com a esparsidade subindo durante o treino", "autor": "Zhu", "quem": "Zhu & Gupta", "titulo": "To prune, or not to prune", "arxiv": "1710.01878"},
    {"passo": "Poda por magnitude", "onde": "XLM-R", "tecnica": "Poda por magnitude num BERT, matriz por matriz: de 30% a 40% sai sem perda", "autor": "Gordon", "quem": "Gordon, Duh & Andrews", "titulo": "Compressing BERT", "arxiv": "2002.08307"},
    {"passo": "Poda por magnitude", "onde": "os dois", "tecnica": "Como reportar poda: tamanho e velocidade, controle, sementes", "autor": "Blalock", "quem": "Blalock et al.", "titulo": "What is the State of Neural Network Pruning", "arxiv": "2003.03033"},
    {"passo": "Reajuste", "onde": "os dois", "tecnica": "Fine-tuning de recuperação com a máscara travada, com o sinal do professor", "autor": "Zafrir", "quem": "Zafrir et al.", "titulo": "Prune Once for All", "arxiv": "2111.05754"},
    {"passo": "Reajuste", "onde": "os dois", "tecnica": "Magnitude contra outros critérios num encoder pré-treinado; as tabelas de entrada fora da poda", "autor": "Sanh", "quem": "Sanh, Wolf & Rush", "titulo": "Movement Pruning", "arxiv": "2005.07683"},
    {"passo": "Quantização", "onde": "os dois", "tecnica": "Inteiros de 8 bits: escala e ponto zero", "autor": "Jacob", "quem": "Jacob et al.", "titulo": "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", "arxiv": "1712.05877"},
    {"passo": "Quantização", "onde": "os dois", "tecnica": "Um Transformer em 8 bits: o que quantizar e o que deixar em 32 bits", "autor": "Zafrir", "quem": "Zafrir et al.", "titulo": "Q8BERT", "arxiv": "1910.06188"},
    {"passo": "Quantização", "onde": "os dois", "tecnica": "As formas de quantizar, lado a lado: dinâmica, estática, com treino", "autor": "Gholami", "quem": "Gholami et al.", "titulo": "A Survey of Quantization Methods for Efficient Neural Network Inference", "arxiv": "2103.13630"},
    {"passo": "Quantização", "onde": "os dois", "tecnica": "Por que as ativações de um Transformer são difíceis de quantizar com faixa fixa (a estática que não passou)", "autor": "Bondarenko", "quem": "Bondarenko et al.", "titulo": "Understanding and Overcoming the Challenges of Efficient Transformer Quantization", "arxiv": "2109.12948"},
    {"passo": "Quantização", "onde": "do zero", "tecnica": "Quantização pós-treino, só dos pesos ou também das ativações, e quando o treino é necessário", "autor": "Krishnamoorthi", "quem": "Krishnamoorthi", "titulo": "Quantizing deep convolutional networks for efficient inference", "arxiv": "1806.08342"},
    {"passo": "Quantização", "onde": "do zero", "tecnica": "Um Transformer quantizado em 8 bits depois de treinado", "autor": "Bhandare", "quem": "Bhandare et al.", "titulo": "Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model", "arxiv": "1906.00532"},
    {"passo": "Exportação", "onde": "os dois", "tecnica": "Levar a rede do PyTorch para o ONNX Runtime: passo de engenharia, sem artigo citado em nenhuma etapa", "autor": None, "quem": "nenhum artigo", "titulo": ""},
)

# O que foi lido na fonte nesta etapa, em 18/11/2026: o resumo de cada artigo, como o arXiv o devolve (guardado em `artigos/arxiv_*.json`).
# `sustenta` diz se o **resumo** sustenta a técnica como o projeto a usa: "sim", ou "em parte" quando o resumo confirma o assunto mas o
# ponto exato que o projeto cita está no corpo do artigo, que não foi lido aqui. Os dois artigos da ACL Anthology só tiveram título,
# autores e ano conferidos (a página não traz resumo no arquivo consultado).
RESUMOS_LIDOS = {
    "2506.02153": ("sim", "artigo de posição: modelos pequenos bastam e saem mais baratos em tarefas repetidas e especializadas; não traz medida própria"),
    "2504.13471": ("sim", "pipeline de três estágios: protótipo com a LLM como professora, transferência de conhecimento e compressão do modelo"),
    "1510.00149": ("sim", "três estágios em sequência: poda, quantização treinada e codificação de Huffman; não trata de destilação"),
    "2405.20935": ("sim", "esparsidade e quantização não são independentes, e a ordem importa: quantizar antes de podar pode tirar elementos importantes"),
    "2611.00464": ("sim", "estudo fatorial em modelos gerativos de 2,7 a 7B: INT8 quase sem perda; 4 bits (NF4) com poda por magnitude derruba o modelo; entre as configurações viáveis a ordem não teve efeito significativo"),
    "2204.08582": ("em parte", "o conjunto (51 idiomas, 60 intenções, 55 tipos de entidade) e as três métricas, com resultados do XLM-R e do mT5; os números da linha de pt-PT estão nas tabelas do artigo, não no resumo"),
    "1707.09861": ("sim", "a semente muda o resultado de forma significativa; reportar um número só não basta"),
    "1911.02549": ("em parte", "apresenta o benchmark de inferência; o cenário de uma consulta por vez (single-stream) está no corpo do artigo"),
    "2110.12894": ("sim", "os indicadores de custo (parâmetros, operações, tempo) não andam juntos; reportar só um dá um retrato incompleto"),
    "2305.13512": ("sim", "os maiores modelos chegam perto do supervisionado em intenção e ficam atrás em entidades"),
    "2402.13116": ("em parte", "levantamento da destilação de LLMs, incluindo das fechadas para modelos abertos; a divisão caixa-preta e caixa-branca está no corpo"),
    "2411.05045": ("sim", "destila uma LLM num modelo pequeno de classificação de texto; o método deles tem um laço de aprendizado ativo que este projeto não usa"),
    "1503.02531": ("em parte", "destilar um conjunto de modelos num modelo só; a temperatura está no corpo do artigo"),
    "1909.10351": ("em parte", "destilação de um Transformer para um aluno menor, camada a camada; este projeto usa só os logits de saída"),
    "2004.09936": ("sim", "arquitetura conjunta de intenção e entidades, que funciona sem pesos pré-treinados; 2 camadas, CRF e features esparsas estão no corpo"),
    "1902.10909": ("sim", "modelo conjunto de intenção e preenchimento de slots sobre o BERT"),
    "1911.02116": ("sim", "o XLM-R: modelo de linguagem mascarado, multilíngue, treinado em cem idiomas"),
    "1810.04805": ("em parte", "o BERT é ajustado com uma camada de saída a mais; taxas, lote e épocas do ajuste fino estão no corpo"),
    "1711.05101": ("sim", "separa o decaimento de pesos do passo do gradiente no Adam (AdamW)"),
    "2006.04884": ("em parte", "a instabilidade do ajuste fino vem da otimização, não do tamanho pequeno do conjunto; a recomendação sobre o número de passos está no corpo"),
    "2203.15996": ("sim", "ferramenta com poda de vocabulário e poda do Transformer, que reduz o modelo sem treinar de novo"),
    "2010.05609": ("sim", "num modelo multilíngue a maior parte dos parâmetros está na tabela de entrada; reduzir o vocabulário aos idiomas de interesse encolhe o modelo"),
    "2204.00408": ("em parte", "poda estruturada de camadas, cabeças e unidades com destilação; não trata de linhas de uma tabela de features, que é como este projeto a usa"),
    "2004.03844": ("sim", "tirar camadas de modelos pré-treinados: até 40% com 98% do desempenho; as camadas de baixo são as mais importantes"),
    "1506.02626": ("sim", "poda das conexões redundantes, com treino depois do corte, sem perda de acurácia"),
    "1710.01878": ("sim", "poda gradual, incorporada ao treino"),
    "2002.08307": ("sim", "poda de pesos no BERT: de 30% a 40% não afeta; acima disso a transferência piora"),
    "2003.03033": ("sim", "meta-análise de 81 artigos: faltam referências e métricas padronizadas para comparar métodos de poda"),
    "2111.05754": ("sim", "modelos pré-treinados esparsos, com poda e destilação, que são ajustados mantendo o padrão de esparsidade"),
    "2005.07683": ("em parte", "poda de movimento contra magnitude no ajuste fino, com vantagem em esparsidade alta; deixar as tabelas de entrada fora da poda está no corpo"),
    "1712.05877": ("sim", "esquema de quantização para inferência só com inteiros; a escala e o ponto zero estão no corpo"),
    "1910.06188": ("em parte", "BERT em 8 bits, 4 vezes menor, mas com treino ciente de quantização; este projeto quantiza depois do treino e usa do artigo só o recorte do que é quantizado, que está no corpo"),
    "2103.13630": ("em parte", "levantamento dos métodos de quantização; a comparação entre dinâmica, estática e com treino está no corpo"),
    "2109.12948": ("sim", "as ativações do Transformer têm faixas dinâmicas largas, difíceis de representar em ponto fixo de poucos bits"),
    "1806.08342": ("sim", "quantização pós-treino em 8 bits: pesos por canal e ativações por camada ficam a 2% do ponto flutuante; só os pesos já dão 4 vezes menos"),
    "1906.00532": ("sim", "um Transformer de tradução quantizado em INT8 depois de treinado, com as instruções VNNI, com menos de 0,5% de perda"),
}


def onde_esta_citado(artigo: dict, raiz: Path | None = None) -> list[str]:
    """Em que arquivos do projeto o identificador do artigo já aparecia: PLANO.md, RESUMO.md das entregas e construtores dos notebooks (fora os desta etapa)."""
    raiz = raiz or ENTREGAS.parent
    ident = artigo.get("arxiv") or artigo.get("acl")
    if not ident:
        return []
    fontes = [raiz / "PLANO.md", *sorted((raiz / "entregas").glob("*/RESUMO.md")), *sorted((raiz / "notebooks" / "construtores").glob("build_nb*.py"))]
    achados = []
    for f in fontes:
        if f.exists() and ETAPA not in str(f) and "nb10d" not in f.name and ident in f.read_text(encoding="utf-8"):
            achados.append("PLANO.md" if f.name == "PLANO.md" else f"RESUMO da Etapa {f.parent.name.split('_')[0].lstrip('0')}" if f.name == "RESUMO.md" else f"notebook da Etapa {f.stem.replace('build_nb', '').lstrip('0')}")
    return achados


def conferir_artigos(base: Path | None = None, artigos: tuple = ARTIGOS) -> Path:
    """Pergunta ao arXiv (API pública, gratuita) o título, os autores e a data de cada identificador do quadro, e grava a resposta inteira.

    Grava `artigos/arxiv_<data-hora>.json`: o que a fonte devolveu, sem
    julgamento. Quem compara com o que o projeto escreve é `quadro_de_artigos`.
    Os dois artigos da ACL Anthology são buscados na página de cada um.
    """
    import urllib.request
    import xml.etree.ElementTree as ET

    base = base or base_padrao()
    (base / "artigos").mkdir(parents=True, exist_ok=True)
    ids = [a["arxiv"] for a in artigos if a.get("arxiv")]
    pedido = urllib.request.Request(f"http://export.arxiv.org/api/query?id_list={','.join(ids)}&max_results={len(ids)}", headers={"User-Agent": "conferencia-de-referencias/1.0"})
    with urllib.request.urlopen(pedido, timeout=60) as resposta:
        raiz = ET.fromstring(resposta.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    limpo = lambda texto: re.sub(r"\s+", " ", texto or "").strip()
    devolvidos = {}
    for entrada in raiz.findall("a:entry", ns):
        ident = re.sub(r"v\d+$", "", limpo(entrada.findtext("a:id", "", ns)).rsplit("/abs/", 1)[-1])
        devolvidos[ident] = {"titulo": limpo(entrada.findtext("a:title", "", ns)), "autores": [limpo(x.findtext("a:name", "", ns)) for x in entrada.findall("a:author", ns)],
                             "publicado": limpo(entrada.findtext("a:published", "", ns))[:10], "resumo": limpo(entrada.findtext("a:summary", "", ns))}
    da_acl = {}
    for artigo in artigos:
        if artigo.get("acl"):
            try:
                with urllib.request.urlopen(urllib.request.Request(f"https://aclanthology.org/{artigo['acl']}.bib", headers={"User-Agent": "conferencia-de-referencias/1.0"}), timeout=60) as resposta:
                    bib = resposta.read().decode("utf-8")
                campo = lambda nome: limpo((re.search(nome + r'\s*=\s*"(.*?)",?\n', bib, re.S) or [None, ""])[1])
                da_acl[artigo["acl"]] = {"titulo": campo("title").replace("{", "").replace("}", ""), "autores": [limpo(x) for x in campo("author").split(" and ")], "publicado": campo("year"), "resumo": ""}
            except Exception as erro:   # a página não respondeu: fica registrado, e a referência sai como não conferida
                da_acl[artigo["acl"]] = {"erro": repr(erro)}
    destino = base / "artigos" / f"arxiv_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    destino.write_text(json.dumps({"consultado_em": datetime.now().isoformat(timespec="seconds"), "fonte": "http://export.arxiv.org/api/query e https://aclanthology.org/<id>.bib",
                                   "pedidos": ids, "arxiv": devolvidos, "acl": da_acl}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destino


def _sem_acento(texto: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)).lower()


def quadro_de_artigos(base: Path | None = None, artigos: tuple = ARTIGOS, raiz: Path | None = None, lidos: dict | None = None) -> pd.DataFrame:
    """O quadro "cada passo e o seu artigo", com o que a fonte devolveu para cada referência.

    `na_fonte` tem três valores: "conferida" (o identificador existe na fonte e
    o primeiro autor e o título são os que o projeto escreve), "não confere"
    (a fonte devolveu outro artigo) e "não conferida" (a fonte não devolveu o
    identificador, ou a consulta não foi feita). `resumo_lido` vem de
    `RESUMOS_LIDOS`: as referências cujo resumo foi lido nesta etapa, com o
    que o resumo diz sobre a técnica.
    """
    base = base or base_padrao()
    consultas = sorted((base / "artigos").glob("arxiv_*.json"))
    fonte = _json(consultas[-1]) if consultas else {"arxiv": {}, "acl": {}, "consultado_em": None}
    if lidos is None:   # só vale como "lido" o resumo que a fonte devolveu na consulta gravada
        lidos = {ident: {"sustenta": s, "diz": d} for ident, (s, d) in RESUMOS_LIDOS.items() if fonte["arxiv"].get(ident, {}).get("resumo")}
    linhas = []
    for a in artigos:
        ident = a.get("arxiv") or a.get("acl")
        devolvido = (fonte["arxiv"].get(a["arxiv"]) if a.get("arxiv") else fonte["acl"].get(a.get("acl"))) if ident else None
        if not ident:
            na_fonte = "sem artigo"
        elif not devolvido or "erro" in devolvido or not devolvido.get("titulo"):
            na_fonte = "não conferida"
        else:
            primeiro = _sem_acento(devolvido["autores"][0]) if devolvido["autores"] else ""
            autor_ok = _sem_acento(a["autor"]) in primeiro
            titulo_ok = _sem_acento(a["titulo"]) in _sem_acento(devolvido["titulo"])
            na_fonte = "conferida" if autor_ok and titulo_ok else "não confere"
        lido = lidos.get(ident, {}) if ident else {}
        linhas.append({
            "passo": a["passo"], "caminho": a["onde"], "tecnica": a["tecnica"], "quem": a["quem"], "titulo_no_projeto": a["titulo"], "identificador": ident or "",
            "onde": f"arXiv {a['arxiv']}" if a.get("arxiv") else f"ACL Anthology {a['acl']}" if a.get("acl") else "",
            "endereco": f"https://arxiv.org/abs/{a['arxiv']}" if a.get("arxiv") else f"https://aclanthology.org/{a['acl']}/" if a.get("acl") else "",
            "ano": etapa9.ano_do_arxiv(a["arxiv"]) if a.get("arxiv") else a.get("ano"), "citado_em": "; ".join(onde_esta_citado(a, raiz)),
            "na_fonte": na_fonte, "titulo_na_fonte": (devolvido or {}).get("titulo", ""), "primeiro_autor_na_fonte": ((devolvido or {}).get("autores") or [""])[0],
            "autores_na_fonte": len((devolvido or {}).get("autores") or []), "publicado_na_fonte": (devolvido or {}).get("publicado", ""),
            "resumo_lido": bool(lido), "o_resumo_sustenta": lido.get("sustenta", ""), "o_que_o_resumo_diz": lido.get("diz", ""), "consultado_em": fonte.get("consultado_em"),
        })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# O cálculo inteiro, gravado no cache
# --------------------------------------------------------------------------

ARQUIVOS = ("tabela_final.csv", "conferencia.csv", "cadeias.csv", "comparacoes.csv", "fronteira.csv", "precos.csv", "latencia_ao_lado.csv", "calculo.json")


def calcular(base: Path | None = None, entregas: Path | None = None) -> Path:
    """Faz todas as contas da etapa e grava em `resultados/`. Não roda sem as regras gravadas e sem as respostas preparadas."""
    from especialista.dados import massive

    base, entregas = base or base_padrao(), entregas or ENTREGAS
    regras, arquivo_das_regras = carregar_regras(base)
    fichas = registro(entregas, base)
    ausentes = faltam(fichas)
    if ausentes:
        raise FileNotFoundError(f"Faltam {len(ausentes)} arquivo(s) de respostas (o primeiro: {ausentes[0]}). Rode: uv run python -m especialista.experimentos.etapa10d preparar")
    lista = lista_de_comparacoes(fichas, entregas)
    cadeias = cadeias_do_registro(fichas, entregas)
    em_regra = {(c["b"], c["a"]) for c in regras["comparacoes"]}
    fora_da_regra = [(c["b"], c["a"]) for c in lista if (c["b"], c["a"]) not in em_regra]
    if fora_da_regra or list(regras["cadeias"]) != list(cadeias):
        raise RuntimeError(f"O registro mudou depois das regras gravadas ({len(fora_da_regra)} comparações novas). Grave uma versão nova das regras antes de calcular: ... etapa10d regras")

    particoes, _ = massive.carregar_entrega(entregas / "01_dados")
    gabaritos = {p: ouro(particoes[p]) for p in PARTICOES}
    grupos = {p: respostas(fichas, p, list(particoes[p]["id"])) for p in PARTICOES}
    T = tabela_final(fichas, entregas)
    tabelas = {"test": T, "dev": em_dev(T)}

    confere = [conferir_qualidade(fichas, gabaritos, grupos, entregas), conferir_com_a_etapa_9(T, entregas)]
    CAD, PP = [], []
    for p in PARTICOES:
        cad = etapa9.cadeias(tabelas[p], gabaritos[p], grupos[p], cadeias)
        cad.insert(1, "particao", p)
        CAD.append(cad)
        confere.append(conferir_degraus(fichas, cad, p, entregas))
        PP.append(comparar_lista(lista, tabelas[p], gabaritos[p], grupos[p], p))
    CAD, PP = pd.concat(CAD, ignore_index=True), pd.concat(PP, ignore_index=True)
    confere.append(conferir_comparacoes(lista, PP, entregas))
    confere = pd.concat(confere, ignore_index=True)

    folga = variacao_da_latencia(entregas)
    medidas = latencias_medidas(base)
    ao_lado = latencia_ao_lado(medidas[-1], T, fichas) if medidas else pd.DataFrame()
    nova = ao_lado.set_index("chave")["p50_mesma_sessao_ms"] if medidas else None
    F = fronteiras(T, folga).assign(latencia_usada="etapas de origem")
    if medidas:
        F = pd.concat([F, fronteiras(T, folga, nova).assign(latencia_usada="mesma sessão, nos sistemas medidos de novo")], ignore_index=True)
    PR = precos(T, PP, fichas, {"mesma sessão": nova} if medidas else None)

    saida = base / "resultados"
    saida.mkdir(parents=True, exist_ok=True)
    T.reset_index().to_csv(saida / "tabela_final.csv", index=False)
    confere.to_csv(saida / "conferencia.csv", index=False)
    CAD.to_csv(saida / "cadeias.csv", index=False)
    PP.to_csv(saida / "comparacoes.csv", index=False)
    F.to_csv(saida / "fronteira.csv", index=False)
    PR.to_csv(saida / "precos.csv", index=False)
    ao_lado.to_csv(saida / "latencia_ao_lado.csv", index=False)
    entradas = sorted({c for f in fichas.values() for lista_ in f["previsoes"].values() for c in lista_})
    calculo = {
        "etapa": ETAPA, "calculado_em": datetime.now().isoformat(timespec="seconds"),
        "regras": {"arquivo": arquivo_das_regras.name, "versao": regras["versao"], "gravado_em": regras["gravado_em"], "sha256": sha256(arquivo_das_regras)},
        "sistemas": len(fichas), "comparacoes": len(lista), "cadeias": list(cadeias), "folga_da_latencia": folga,
        "latencia_na_mesma_sessao": str(medidas[-1]["pasta"].relative_to(base)) if medidas else None,
        "respostas_lidas": {"arquivos": len(entradas), "sha256": {etapa9._relativo(c, entregas.parent): sha256(c) for c in entradas}},
    }
    (saida / "calculo.json").write_text(json.dumps(calculo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")   # por último: marca "pronto"
    return saida


def carregar_calculo(base: Path | None = None) -> dict:
    """O que `calcular` gravou: {nome sem extensão: tabela ou dicionário}."""
    saida = (base or base_padrao()) / "resultados"
    if not (saida / "calculo.json").exists():
        raise FileNotFoundError("Falta o cálculo. Rode: uv run python -m especialista.experimentos.etapa10d calcular")
    lido = {}
    for nome in ARQUIVOS:
        if nome.endswith(".json"):
            lido[nome[:-5]] = _json(saida / nome)
        else:
            try:
                lido[nome[:-4]] = pd.read_csv(saida / nome)
            except pd.errors.EmptyDataError:
                lido[nome[:-4]] = pd.DataFrame()
    lido["tabela_final"] = lido["tabela_final"].set_index("chave")
    lido["fronteira"]["dominado_por"] = lido["fronteira"]["dominado_por"].fillna("")   # quem está na fronteira não tem quem o domine: o CSV guarda vazio
    return lido


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 10d: comparativo dos dois caminhos. Só lê as entregas; nada é treinado.")
    argumentos.add_argument("comando", choices=("regras", "faltam", "preparar", "latencia", "artigos", "calcular"))
    argumentos.add_argument("--refazer", action="store_true", help="preparar: produz de novo as respostas que já existem")
    argumentos.add_argument("--rodadas", type=int, default=5, help="latencia: rodadas alternadas (o protocolo usa 5)")
    argumentos.add_argument("--consultas", type=int, default=None, help="latencia: consultas por rodada (o protocolo usa 1.024)")
    argumentos.add_argument("--pasta", type=Path, default=None, help="onde gravar; o padrão é data/execucoes/10d_comparativo_dos_caminhos")
    a = argumentos.parse_args()
    if a.comando == "regras":
        destino = gravar_regras(a.pasta)
        regras = _json(destino)
        print(f"{destino}: versão {regras['versao']}, gravada em {regras['gravado_em']}; {len(regras['comparacoes'])} comparações, {len(regras['cadeias'])} cadeias, {len(regras['sistemas_no_registro'])} sistemas.")
    elif a.comando == "faltam":
        for caminho in faltam(registro(base=a.pasta)):
            print(caminho)
    elif a.comando == "preparar":
        pasta = preparar(base=a.pasta, refazer=a.refazer)
        print(f"Respostas geradas para {len(_json(pasta / 'medidas.json')['respostas_geradas'])} sistemas, em {pasta}.")
    elif a.comando == "latencia":
        pasta = medir_latencia(a.pasta, rodadas=a.rodadas, consultas=a.consultas)
        medida = _json(pasta / "latencia.json")
        print(pasta, "| p50 (ms):", ", ".join(f"{nome} {r['latencia_p50_ms']:.3f}" for nome, r in medida["resumo"].items()),
              f"| outros processos: {medida['nucleos_de_outros']:.3f} núcleo(s) | quieta: {medida['maquina_quieta']} | reproduz: {all(medida['reproduz_as_respostas_gravadas'].values())}")
    elif a.comando == "artigos":
        destino = conferir_artigos(a.pasta)
        quadro = quadro_de_artigos(a.pasta)
        print(destino, "|", quadro["na_fonte"].value_counts().to_dict())
    else:
        saida = calcular(a.pasta)
        confere = pd.read_csv(saida / "conferencia.csv")
        print(saida, "| conferência:", confere.groupby("o_que")["confere"].agg(lambda c: f"{int((c == True).sum())} de {int(c.notna().sum())}").to_dict())   # noqa: E712


if __name__ == "__main__":
    _linha_de_comando()
