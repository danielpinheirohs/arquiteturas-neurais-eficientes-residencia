"""Monta notebooks/10d_comparativo_dos_caminhos.ipynb a partir das células abaixo."""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Etapa 10d · Comparativo dos dois caminhos: do zero ou pré-treinado?

| | |
|---|---|
| **Pergunta** | Com os mesmos professores e as mesmas regras, o que o encoder pré-treinado compra em qualidade, e quanto custa em tamanho e latência, em relação à rede feita do zero? |
| **Entrada** | As entregas das Etapas 2 a 9 (o caminho do zero, pelo catálogo da Etapa 9) e das Etapas 10a, 10b e 10c (o caminho do XLM-R): `decisao.json`, `resultados.csv`, `comparacoes.csv`, `eficiencia.csv` e as respostas de cada sistema em `dev` e no teste. E o que os comandos de `especialista/experimentos/etapa10d.py` gravaram em `data/execucoes/10d_comparativo_dos_caminhos/`: as regras, as respostas que nenhuma etapa tinha guardado, a latência medida na mesma sessão, a consulta das referências e as contas. |
| **Entrega** | `entregas/10d_comparativo_dos_caminhos/`: a tabela final estendida, as cadeias dos dois caminhos, as comparações pareadas, a fronteira, o preço de cada ponto, a latência lado a lado, o quadro de artigos, os limites, as regras e o manifesto. |
| **Pronto quando** | (1) as regras foram gravadas antes das contas, e as contas são as das regras; (2) a tabela tem as linhas da Etapa 9, idênticas, e as do XLM-R, com qualidade e intervalo em `dev` e no teste; (3) cada número de qualidade e cada diferença que uma etapa de origem gravou foi refeito das respostas, e o que não bate está relatado; (4) cada cadeia tem os degraus dos dois caminhos lado a lado; (5) a fronteira e o preço de cada ponto saem pelas regras; (6) a latência dos dois modelos finais foi medida na mesma sessão, com a máquina quieta, ou a falta disso está nos limites; (7) cada referência do quadro diz se foi conferida na fonte; (8) os limites dos dois caminhos estão juntos; (9) nada foi treinado, e a entrega tem manifesto e cópia do código. |

**O que esta etapa faz.** Nada é treinado. O projeto percorreu duas vezes o mesmo caminho (LLM professora, destilação, poda, reajuste, quantização): uma com uma rede pequena feita do zero, outra com um encoder pré-treinado. Aqui os dois caminhos são postos na mesma tabela e comparados ponto por ponto, com as mesmas respostas gravadas e a mesma régua.

**Artigos de apoio**

- Custo não é um número só, e cada medida precisa da sua condição: Dehghani et al., *The Efficiency Misnomer* ([arXiv 2110.12894](https://arxiv.org/abs/2110.12894)).
- Latência com uma consulta por vez, percentis e a máquina registrada: Reddi et al., *MLPerf Inference Benchmark* ([arXiv 1911.02549](https://arxiv.org/abs/1911.02549)).
- Intervalo e comparação pareada por bootstrap: Koehn ([EMNLP 2004](https://aclanthology.org/W04-3250/)); Berg-Kirkpatrick et al. ([EMNLP-CoNLL 2012](https://aclanthology.org/D12-1091/)).
- A rede do zero: Bunk et al., *DIET* ([arXiv 2004.09936](https://arxiv.org/abs/2004.09936)). O encoder pré-treinado com cabeça conjunta: Chen et al., *BERT for Joint Intent Classification and Slot Filling* ([arXiv 1902.10909](https://arxiv.org/abs/1902.10909)); Conneau et al., XLM-R ([arXiv 1911.02116](https://arxiv.org/abs/1911.02116)).

Os endereços acima foram consultados nesta etapa (seção 10, com o quadro de todos os passos).

**Roteiro**

1. O registro: quem entra, e de onde vem cada número
2. As regras, escritas antes das contas
3. A conta desta etapa é igual à das etapas de origem?
4. A tabela final, com os dois caminhos
5. As cadeias, lado a lado
6. Os dois caminhos no mesmo ponto
7. Cada aluno contra o seu professor, e de ponta a ponta
8. A fronteira e o preço de cada ponto
9. A latência dos dois modelos finais, na mesma sessão
10. Cada passo e o seu artigo
11. Rótulos humanos em cada ponto e limites
12. Entrega
13. Resposta
''')

code(r'''
import importlib.metadata
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from especialista import visual
from especialista.avaliacao import qualidade
from especialista.caminhos import ENTREGAS, RAIZ, entrega
from especialista.dados import massive
from especialista.experimentos import etapa9, etapa10d

visual.aplicar_estilo()
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

PASTA = entrega("10d_comparativo_dos_caminhos")
BASE = etapa10d.base_padrao()
TRES = etapa10d.TRES
API, GPU, CPU = etapa10d.API, etapa10d.GPU, etapa10d.CPU
ZERO, XLMR = etapa10d.ZERO, etapa10d.XLMR
CAMINHOS = etapa10d.caminhos_do_registro()

particoes, _ = massive.carregar_entrega()
N_TREINO, N_DEV, N_TESTE = (len(particoes[p]) for p in ("train", "dev", "test"))
NOME_DA_PARTICAO = {"dev": "dev", "test": "teste"}


def n(x):
    return f"{int(round(x)):,}".replace(",", ".")


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def pts(x, casas=1):
    return num(100 * x, casas)


def dif(x, casas=1):
    return f"{100 * x:+.{casas}f}".replace(".", ",").replace("-", "−")


def com_intervalo(linha, prefixo="", casas=1):
    """Uma diferença com o seu intervalo de 95%, em pontos percentuais: "−2,8 (−4,3 a −1,5)"."""
    d, a, b = (linha[prefixo + c] for c in (("diferenca", "dif_ic_min", "dif_ic_max") if prefixo else ("diferenca", "ic_min", "ic_max")))
    return f"{dif(d, casas)} ({dif(a, casas)} a {dif(b, casas)})"


def valor_com_intervalo(linha, metrica, prefixo=""):
    return f"{pts(linha[prefixo + metrica])} ({pts(linha[prefixo + metrica + '_ic_min'])} a {pts(linha[prefixo + metrica + '_ic_max'])})"


def posicao(minimo, maximo):
    """Onde B fica em relação a A numa comparação pareada B − A."""
    return "acima" if minimo > 0 else "abaixo" if maximo < 0 else "sem diferença distinguível"


def metrica_na_frase(m):
    nome = qualidade.NOMES[m]
    return nome if nome.startswith("F1") else nome[0].lower() + nome[1:]


def ms(x):
    return "não medida" if pd.isna(x) else f"{n(x)} ms" if x >= 100 else f"{num(x, 2)} ms"


def tamanho(b):
    return "—" if pd.isna(b) else f"{num(b / 1e9, 1)} GB" if b >= 1e9 else f"{num(b / 1e6, 2 if b < 1e7 else 1)} MB"


def vezes(maior, menor):
    razao = maior / menor
    return n(razao) if razao >= 100 else num(razao, 1)


def em_markdown(tabela):
    linhas = ["| " + " | ".join(tabela.columns) + " |", "|" + "---|" * len(tabela.columns)]
    return "\n".join(linhas + ["| " + " | ".join(str(v) for v in linha) + " |" for linha in tabela.itertuples(index=False)])


def trio(linha, colunas=("diferenca", "ic_min", "ic_max")):
    return tuple(100 * linha[c] for c in colunas)


def lista(itens):
    """Uma enumeração numa frase. Se algum item já tem vírgula, os itens são separados por ponto e vírgula."""
    itens = list(itens)
    if any("," in item for item in itens):
        return "; ".join(itens)
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def minusculo(texto):
    """A inicial em minúscula, para o meio de uma frase; siglas (INT8, ONNX) ficam como estão."""
    return texto if texto[:2].isupper() else texto[0].lower() + texto[1:]


def diferencas(dados, titulo, eixo_x):
    """`visual.diferencas_com_intervalo`, com o número das diferenças negativas à esquerda da barra."""
    fig, ax = visual.diferencas_com_intervalo(dados, titulo, eixo_x)
    for anotacao, (valor, minimo, maximo) in zip(ax.texts, dados.values()):
        if valor < 0:
            anotacao.xy = (minimo, anotacao.xy[1])
            anotacao.set_position((-7, 0))
            anotacao.set_ha("right")
    esquerda, direita = ax.get_xlim()
    ax.set_xlim(esquerda - 0.07 * (direita - esquerda), direita)
    return fig, ax
''')

md(r'''
## 1. O registro: quem entra, e de onde vem cada número

A peça central da etapa é um **registro de sistemas** (`etapa10d.registro`). Cada sistema é uma ficha que diz em que arquivo está a qualidade, em que arquivos estão as respostas (em `dev` e no teste) e de que etapa vêm o tamanho e a latência. Nada no registro é número: ele só diz onde ler.

- **O caminho do zero** é o catálogo da Etapa 9 (`etapa9.catalogo`), sem mudança.
- **O caminho do XLM-R** é lido dos `decisao.json` das Etapas 10a, 10b e 10c: qual poda ficou, qual forma de 8 bits, qual temperatura, onde está cada modelo.
- Cada ficha leva quatro marcas: o **caminho**, o **modelo** (teto, piso, aluno A, B1 ou B2), o **ponto** da cadeia (sem compressão, destilado, podado, ONNX, INT8) e o **professor**. As comparações da seção 2 são geradas a partir dessas marcas.

**Feito para crescer.** Um sistema novo entra de dois jeitos, sem reescrever tabela: uma função nova em `etapa10d.FONTES`, que leia o `decisao.json` da etapa nova; ou um arquivo `sistemas.json` na entrega da etapa nova, que o registro lê sozinho. Com as marcas preenchidas, o sistema aparece na tabela, na fronteira e nas comparações que as regras geram.

Três grupos de respostas não estavam em nenhuma entrega, todos do caminho do zero. O comando `preparar` os produz na CPU, em segundos, carregando os modelos entregues pela Etapa 8, do mesmo jeito que a Etapa 9 fez para o teste:

```bash
uv run python -m especialista.experimentos.etapa10d preparar
```
''')

code(r'''
fichas = etapa10d.registro()
assert not etapa10d.faltam(fichas), "Faltam respostas. Rode: uv run python -m especialista.experimentos.etapa10d preparar"
C = etapa10d.carregar_calculo()
T, CONF, CAD, PP, FRONT, PRECOS, LAT, CALC = (C[k] for k in ("tabela_final", "conferencia", "cadeias", "comparacoes", "fronteira", "precos", "latencia_ao_lado", "calculo"))
assert list(T.index) == list(fichas), "O cálculo gravado não é do registro de agora. Rode: uv run python -m especialista.experimentos.etapa10d calcular"
medidas = json.loads((BASE / "medidas.json").read_text(encoding="utf-8"))

FINAIS = etapa10d.finais(fichas)
FZ, FX = FINAIS[ZERO], FINAIS[XLMR]            # o modelo final de cada caminho
POUCOS, TODOS = int(T["rotulos_humanos"].min()), int(T["rotulos_humanos"].max())
PROFESSOR, ABERTO = T.loc["claude", "curto"], T.loc["qwen", "curto"]
FOLGA = CALC["folga_da_latencia"]
SEMENTES = len(etapa10d.SEMENTES)
PARES = list(dict.fromkeys(zip(PP["chave_b"], PP["chave_a"])))


def caminho_de(chave):
    c = T.loc[chave, "caminho"]
    return None if pd.isna(c) else c


def nome(chave):
    """O nome curto de um sistema, com o caminho na frente: "XLM-R · Aluno A, INT8"."""
    c = caminho_de(chave)
    return T.loc[chave, "curto"] if c is None else f"{CAMINHOS[c]['curto']} · {T.loc[chave, 'curto']}"


def comp(b, a, metrica="exact_match", particao="test"):
    """Uma comparação pareada B − A da lista fechada."""
    linhas = PP[(PP["chave_b"] == b) & (PP["chave_a"] == a) & (PP["metrica"] == metrica) & (PP["particao"] == particao)]
    assert len(linhas) == 1, (b, a, metrica, particao)
    return linhas.iloc[0]


def frase(b, a, particao="test", metricas=TRES):
    """Uma comparação em palavras: onde B fica em relação a A, métrica por métrica."""
    partes = [f"{posicao(l['ic_min'], l['ic_max'])} em {metrica_na_frase(m)} ({com_intervalo(l)})" for m in metricas for l in [comp(b, a, m, particao)]]
    return lista(partes)


def de_onde(chave, particao):
    caminhos = fichas[chave]["previsoes"][particao]
    if caminhos[0].is_relative_to(BASE):
        return "geradas nesta etapa (modelo da Etapa 8)" if fichas[chave]["gerar"]["como"] == "onnx" else "geradas nesta etapa (linha de base treinada de novo)"
    return str(caminhos[0].parent.relative_to(RAIZ)) + "/"


origem = pd.DataFrame([
    {"caminho": "os dois" if caminho_de(c) is None else CAMINHOS[caminho_de(c)]["curto"], "sistema": T.loc[c, "sistema"], "ponto": etapa10d.PONTOS[f["ponto"]], "sementes": f["sementes"],
     "qualidade lida de": T.loc[c, "qualidade_lida_de"], "respostas em dev": de_onde(c, "dev"), "respostas no teste": de_onde(c, "test"),
     "latência medida na": T.loc[c, "latencia_medida_na"] if pd.notna(T.loc[c, "latencia_medida_na"]) else "não medida"}
    for c, f in fichas.items()
])
display(origem.set_index(["caminho", "sistema"]))

por_caminho = T["caminho"].fillna("os dois").value_counts()
geradas = medidas["respostas_geradas"]
arquivos_gerados = sum(len(v) for g in geradas.values() for v in g["arquivos"].values())
display(Markdown(f"""
**Leitura.** São {len(fichas)} sistemas: {por_caminho[ZERO]} do caminho do zero, {por_caminho[XLMR]} do XLM-R e {por_caminho['os dois']} que os dois caminhos dividem (as duas LLMs professoras e as duas linhas de base).
As {len(etapa9.catalogo())} primeiras linhas são as da Etapa 9. As que esta etapa acrescenta ao caminho do zero são as do INT8 sem poda ({lista(T.loc[c, 'curto'] for c in fichas if fichas[c]['ponto'] == 'int8_sem_poda' and caminho_de(c) == ZERO)}), que a Etapa 8 entregou e a Etapa 9 não pôs na tabela: no XLM-R essa forma disputa a fronteira, e os dois caminhos precisam das mesmas linhas.
O modelo final de cada caminho, pelas decisões das Etapas 8 e 10c: **{T.loc[FZ, 'sistema']}** (do zero) e **{T.loc[FX, 'sistema']}** (XLM-R).
Respostas produzidas agora: {arquivos_gerados} arquivos de {len(geradas)} sistemas do caminho do zero (INT8 em `dev`, INT8 sem poda em `dev` e no teste, e a linha de base com {n(POUCOS)} rótulos em `dev`). Do XLM-R nada precisou ser refeito: a Etapa 10c guardou as respostas de todas as linhas.
"""))
''')

md(r'''
## 2. As regras, escritas antes das contas

As regras desta etapa estão em `regras.json`, gravado pelo comando abaixo **antes** de qualquer conta nova. O comando `calcular` não roda sem esse arquivo e guarda a impressão digital dele junto do que calculou.

```bash
uv run python -m especialista.experimentos.etapa10d regras
```

**O método é o do projeto.** Nenhuma decisão é tomada aqui: os modelos finais já foram escolhidos em `dev`, pelas regras das Etapas 6, 7, 8, 10a, 10b e 10c. Tudo é reportado em `dev` e no teste, contra os rótulos humanos; a leitura principal é a do teste. Um sistema treinado é a média das 3 sementes. Toda diferença é pareada, com bootstrap por enunciado, 2.000 reamostragens e intervalo de 95%; só vale quando o intervalo não contém zero. Não há correção para comparações múltiplas: as **primárias** são as do modelo final de um caminho contra o do outro, nas três métricas, no teste; as demais descrevem de onde a diferença vem.

**A lista de comparações é fechada e gerada por regra**, a partir das marcas do registro:

| Família | Regra |
|---|---|
| Entre caminhos | o mesmo modelo, no mesmo ponto da cadeia (sem compressão, piso, destilado, podado, INT8): XLM-R menos rede do zero. O ONNX de 32 bits fica de fora: nos dois caminhos ele dá as mesmas respostas do modelo podado |
| Finais | o modelo final do XLM-R menos o modelo final da rede do zero |
| Aluno e professor | cada aluno, como saiu da destilação e na forma final de 8 bits, menos a LLM que rotulou o treino dele, nos dois caminhos |
| De ponta a ponta | as dez comparações fixadas na Etapa 9, repetidas com as chaves de cada caminho |
| Poda nos 8 bits | em cada caminho, o modelo podado em INT8 menos o mesmo modelo em INT8 sem poda |
| Degraus das cadeias | em cada cadeia (cenário A, cenário B, teto; dois caminhos), cada degrau menos o anterior |

**A fronteira** usa a definição da Etapa 9 (`etapa9.nao_dominados`): um sistema é **dominado** quando existe outro com custo menor ou igual e exact match maior ou igual, e melhor em pelo menos um dos dois. Concorrem todos os sistemas do registro que usam só os poucos rótulos humanos, dos dois caminhos. Duas latências mais próximas que a maior variação da mediana entre as rodadas das medidas de origem contam como empate; o tamanho não tem folga; a qualidade é comparada pelo valor medido.

**O preço de cada ponto**, entre os dois modelos finais: (custo do final do XLM-R − custo do final do zero) ÷ (diferença de exact match, em pontos). O intervalo vem do intervalo de 95% da diferença pareada de exact match, e o preço só é definido quando esse intervalo está todo acima de zero.

**A latência na mesma sessão**: os dois modelos finais, semente 0, em INT8, ONNX de 32 bits e PyTorch de 32 bits, no protocolo das Etapas 8 e 10c. Os números de origem continuam sendo os da tabela; a medida nova vai ao lado e "confirma" a de origem quando a diferença relativa é menor que a folga da fronteira.
''')

code(r'''
regras, arquivo_das_regras = etapa10d.carregar_regras()
versoes = sorted((json.loads(c.read_text(encoding="utf-8")) for c in BASE.glob("regras*.json")), key=lambda r: r["versao"])
mesmos_pares = all([(c["b"], c["a"], c["familias"]) for c in v["comparacoes"]] == [(c["b"], c["a"], c["familias"]) for c in versoes[0]["comparacoes"]] for v in versoes)
so_o_texto = all({k for k in v if v[k] != versoes[0][k]} <= {"versao", "gravado_em", "comparacoes"} for v in versoes)
regras_antes = bool(regras["gravado_em"] < CALC["calculado_em"] and CALC["regras"]["sha256"] == etapa10d.sha256(arquivo_das_regras) and CALC["regras"]["versao"] == regras["versao"])
lista_das_regras = [(c["b"], c["a"]) for c in regras["comparacoes"]]
contas_sao_as_das_regras = bool(PARES == lista_das_regras and list(dict.fromkeys(CAD["cadeia"])) == list(regras["cadeias"]) and not PP["exploratoria"].any())

familias = pd.Series([f for c in regras["comparacoes"] for f in c["familias"]]).value_counts()
NOME_DA_FAMILIA = {"entre_caminhos": "Entre caminhos", "finais": "Finais", "aluno_professor": "Aluno e professor", "ponta_a_ponta": "De ponta a ponta", "poda_nos_8_bits": "Poda nos 8 bits"}
degraus_por_particao = int(CAD[(CAD["particao"] == "test")]["anterior"].notna().sum())
display(pd.DataFrame({"comparações": familias.rename(index=NOME_DA_FAMILIA)}))
display(pd.DataFrame([{"comparação (B − A)": f"{nome(c['b'])} − {nome(c['a'])}", "famílias": "; ".join(NOME_DA_FAMILIA[f] for f in c["familias"]), "o que responde": c["pergunta"]} for c in regras["comparacoes"]]))

em_duas = sum(len(c["familias"]) > 1 for c in regras["comparacoes"])
display(Markdown(f"""
**Leitura.** A lista tem {len(regras['comparacoes'])} comparações ({em_duas} caem em duas famílias e são contadas uma vez) e {len(regras['cadeias'])} cadeias, com {degraus_por_particao} degraus. Cada uma é calculada em `dev` e no teste.
As regras foram gravadas em {versoes[0]['gravado_em'].replace('T', ' às ')}{f" (versão 1) e em {regras['gravado_em'].replace('T', ' às ')} (versão {regras['versao']})" if len(versoes) > 1 else ""}; as contas, em {CALC['calculado_em'].replace('T', ' às ')}.
{f"A versão {regras['versao']} só corrige o texto que descreve cada comparação: os pares e as famílias são os mesmos da versão 1 ({'conferido' if mesmos_pares and so_o_texto else '**NÃO CONFERE**'}), e as duas foram gravadas antes das contas." if len(versoes) > 1 else ""}
As contas gravadas são exatamente as da lista: {"sim" if contas_sao_as_das_regras else "**NÃO**"}; nenhuma comparação foi acrescentada depois.

**O que já era conhecido quando as regras foram escritas** (está no próprio `regras.json`): {"; ".join(regras['ja_era_conhecido_ao_escrever'][:3])}. As regras não foram escritas no escuro: quem as escreveu já tinha lido os resumos das etapas anteriores. O que elas garantem é que a lista não foi escolhida depois de ver as contas novas.
"""))
assert regras_antes and contas_sao_as_das_regras and mesmos_pares and so_o_texto
''')

md(r'''
## 3. A conta desta etapa é igual à das etapas de origem?

Antes de comparar, a conferência. Cada número que uma etapa de origem gravou é refeito aqui, a partir das respostas, e comparado com o gravado:

- **a qualidade de cada sistema** (as quatro métricas e os limites dos intervalos), em `dev` e no teste, contra o `resultados.csv` da etapa que o mediu; quando duas etapas gravaram o mesmo número (o modelo sem poda do XLM-R, na 10a e na 10c; o podado, na 10b e na 10c; o podado da rede do zero, na 7 e na 8), as duas são conferidas;
- **cada degrau de cada cadeia**, contra o `comparacoes.csv` da etapa que deu aquele passo;
- **as comparações entre caminhos nos modelos sem poda**, contra as que a Etapa 10a gravou, e **as de ponta a ponta do caminho do zero**, contra as da Etapa 9;
- **as linhas da tabela final da Etapa 9**, coluna por coluna, contra as mesmas linhas da tabela estendida.

A regra da etapa: se um número não bate, ele é **relatado**, não consertado.
''')

code(r'''
CONF["situacao"] = np.where(~CONF["gravado"].astype(bool), "sem número gravado na origem", np.where(CONF["confere"] == True, "igual", "DIFERENTE"))   # noqa: E712
resumo_da_conferencia = CONF.groupby(["o_que", "particao", "situacao"]).size().unstack(fill_value=0).reset_index()
for coluna in ("igual", "DIFERENTE", "sem número gravado na origem"):
    if coluna not in resumo_da_conferencia.columns:
        resumo_da_conferencia[coluna] = 0
resumo_da_conferencia = resumo_da_conferencia.rename(columns={"o_que": "o que foi conferido", "particao": "partição"})[["o que foi conferido", "partição", "igual", "DIFERENTE", "sem número gravado na origem"]]
display(resumo_da_conferencia.set_index(["o que foi conferido", "partição"]))

conferidas = CONF[CONF["gravado"].astype(bool)]
diferentes = conferidas[conferidas["confere"] != True]   # noqa: E712
sem_origem = CONF[~CONF["gravado"].astype(bool)]
maior = float(conferidas["maior_diferenca"].max())
por_tipo = conferidas.groupby("o_que").size()
QUALIDADE, DEGRAU, COMPARACAO, LINHA_DA_9 = "qualidade de um sistema", "degrau de uma cadeia", "comparação entre sistemas", "linha da tabela final da Etapa 9"
if len(diferentes):
    display(diferentes)
if len(sem_origem):
    display(sem_origem[["o_que", "chave", "anterior", "particao", "gravado_em"]])
geradas_agora = conferidas[(conferidas["o_que"] == QUALIDADE) & (conferidas["respostas_geradas_agora"] == True)]   # noqa: E712
display(Markdown(f"""
**Leitura.** Foram conferidas **{n(len(conferidas))} linhas**: {n(por_tipo[QUALIDADE])} de qualidade ({len(fichas)} sistemas, em `dev` e no teste, contra cada etapa que gravou o número), {n(por_tipo[DEGRAU])} degraus de cadeia, {n(por_tipo[COMPARACAO])} comparações entre sistemas e {n(por_tipo[LINHA_DA_9])} linhas da tabela final da Etapa 9.
{"**Todas iguais**" if not len(diferentes) else f"**{len(diferentes)} DIFERENTES** (na tabela acima)"}: a maior diferença entre a conta desta etapa e um número gravado é de {maior:.1e} (em fração, não em pontos).
{f"Entre elas estão as {len(geradas_agora)} linhas dos sistemas cujas respostas foram produzidas agora: os modelos da Etapa 8, carregados de novo, dão em `dev` e no teste exatamente as métricas que a Etapa 8 gravou." if len(geradas_agora) else ""}
{f"Sem número na origem para conferir: {len(sem_origem)} linha(s), na tabela acima." if len(sem_origem) else "Nenhuma linha ficou sem número de origem para conferir."}
"""))
''')

md(r'''
## 4. A tabela final, com os dois caminhos

Uma linha por sistema, no teste. Qualidade em %, com intervalo de 95%; nos sistemas treinados, média das 3 sementes. Tamanho e latência dos modelos comprimidos são os da semente 0. As colunas são as da Etapa 9, mais a do caminho. A mesma tabela, com a qualidade em `dev` nas colunas `dev_...`, está em `tabela_final.csv`.
''')

code(r'''
def onde_curto(chave):
    s = T.loc[chave]
    return "não medida" if pd.isna(s["latencia_p50_ms"]) else f"{s['condicao_da_latencia']}: {fichas[chave]['onde']} ({s['latencia_medida_na']})"


def linha_da_tabela(chave):
    s = T.loc[chave]
    oculto = not s["divulgado"]
    c = caminho_de(chave)
    return {
        "Caminho": "os dois" if c is None else CAMINHOS[c]["curto"],
        "Sistema": s["sistema"] + (" **(modelo final)**" if s["final"] else ""),
        "Rótulos humanos": f"{n(s['rotulos_humanos'])} ({s['rotulos_como']})",
        "Acurácia de intenção": valor_com_intervalo(s, "intent_acc"), "F1 de entidades": valor_com_intervalo(s, "slot_f1"), "Exact match": valor_com_intervalo(s, "exact_match"),
        "Parâmetros": "não divulgado" if oculto else n(s["parametros"]),
        "Tamanho em disco": "não divulgado" if oculto else "não medido" if pd.isna(s["bytes"]) else tamanho(s["bytes"]),
        "Com zip": tamanho(s["bytes_com_zip"]),
        "Latência p50": ms(s["latencia_p50_ms"]), "Onde a latência foi medida": onde_curto(chave),
        "Consultas por segundo": "—" if pd.isna(s["consultas_por_segundo"]) else num(s["consultas_por_segundo"], 1) if s["consultas_por_segundo"] < 10 else n(s["consultas_por_segundo"]),
        "US$ por 1.000 consultas": "—" if pd.isna(s["custo_por_mil_usd"]) else num(s["custo_por_mil_usd"], 3),
    }


mostra = pd.DataFrame([linha_da_tabela(chave) for chave in fichas])
display(Markdown(em_markdown(mostra)))

fz, fx, p = T.loc[FZ], T.loc[FX], T.loc["claude"]
eficiencia10c = pd.read_csv(ENTREGAS / "10c_encoder_quantizacao" / "eficiencia.csv")
maquina = {"cpu": eficiencia10c["cpu"].iloc[0], "gpu": pd.read_csv(ENTREGAS / "05b_professor_aberto" / "resultados.csv")["gpu"].iloc[0], "vnni": bool(eficiencia10c["tem_vnni"].iloc[0])}
tres = lambda s: f"{pts(s['intent_acc'])} / {pts(s['slot_f1'])} / {pts(s['exact_match'])}"
notas_da_tabela = [
    f"**Máquina.** CPU local: {maquina['cpu']}. GPU local: {maquina['gpu']}. A API foi chamada desta máquina.",
    f"**Rótulos humanos.** \"no prompt\": a LLM vê {n(POUCOS)} exemplos rotulados por humanos. \"pelo professor\": a rede nunca vê um rótulo humano de treino; aprende com os {n(TODOS)} pedidos rotulados pela LLM. Onde os rótulos humanos de `dev` entraram em decisões está na seção 11.",
    "**Latência.** As do caminho do zero são da Etapa 8 e as do XLM-R, da Etapa 10c: mesmo protocolo, sessões diferentes. A seção 9 mede os dois modelos finais na mesma sessão. Do piso e do aluno B1 do XLM-R a latência não foi medida (a rede é a mesma do aluno A sem poda).",
    "**Tamanho.** É o do sistema inteiro como é usado; no XLM-R inclui o tokenizador. Os sistemas de pesos soltos em INT8 são entregues compactados: o tamanho deles em disco é o da coluna \"com zip\".",
    "**INT8 sem poda.** O mesmo modelo quantizado sem passar pela poda. Não é o modelo final de nenhum caminho (a regra da poda escolheu o menor arquivo que não se distingue do modelo sem poda em `dev`); entra porque foi entregue e disputa a fronteira.",
]
display(Markdown("\n".join(f"- {nota}" for nota in notas_da_tabela) + f"""

**Leitura.** No teste, o modelo final do XLM-R faz {tres(fx)} (intenção / entidades / exact match) com {tamanho(fx['bytes'])} e {ms(fx['latencia_p50_ms'])} por consulta; o da rede do zero faz {tres(fz)} com {tamanho(fz['bytes'])} e {ms(fz['latencia_p50_ms'])}.
São {vezes(fx['bytes'], fz['bytes'])} vezes o tamanho e {vezes(fx['latencia_p50_ms'], fz['latencia_p50_ms'])} vezes a latência. As seções seguintes dizem quais diferenças de qualidade se distinguem de zero e de que ponto do caminho elas vêm.
"""))
''')

md(r'''
## 5. As cadeias, lado a lado

Cada cadeia é lida de cima para baixo, e em cada degrau muda uma coisa só. Os degraus são os mesmos nos dois caminhos; o que cada um faz, não:

| Degrau | Rede do zero | XLM-R |
|---|---|---|
| Professor (LLM) | o ponto de partida: a LLM resolvendo a tarefa | o mesmo professor, as mesmas respostas |
| Aluno destilado | a rede do zero aprende com os rótulos (e, no cenário B, os logits) da LLM (Etapa 6) | o XLM-R é ajustado com os mesmos rótulos e logits (Etapa 10a) |
| Podado e reajustado | saem 90% das linhas da tabela de features (Etapa 7) | a tabela de subtokens fica só com o vocabulário do treino (Etapa 10b) |
| ONNX, 32 bits | a rede sai do PyTorch e roda no ONNX Runtime (Etapa 8) | o mesmo (Etapa 10c) |
| INT8 | os pesos passam de 32 para 8 bits, quantização dinâmica (Etapa 8) | o mesmo (Etapa 10c) |

A coluna "diferença para o degrau anterior" é pareada: diz o que aquele passo, sozinho, fez com a qualidade.
''')

code(r'''
CENARIOS = {"A": "Cenário A: professor fechado", "B": "Cenário B: professor aberto", "teto": "Referência: teto (rótulos humanos)"}
ARQUIVO_DO_CENARIO = {"A": "cenario_a", "B": "cenario_b", "teto": "teto"}
D = etapa9.DEGRAUS
poda_gravada = {ZERO: pd.read_csv(ENTREGAS / "07_poda" / "comparacoes.csv"), XLMR: pd.read_csv(ENTREGAS / "10b_encoder_poda" / "comparacoes.csv")}
ETAPA_DA_PODA = {ZERO: "Etapa 7", XLMR: "Etapa 10b"}


def cadeia(caminho, cenario, particao="test"):
    return CAD[(CAD["cadeia"] == f"{caminho}/{cenario}") & (CAD["particao"] == particao)]


def casas_da(d, m):
    return 2 if abs(d[f"{m}_diferenca"]) < 0.0015 else 1


def tabela_da_cadeia(caminho, cenario, particao="test"):
    return pd.DataFrame([
        {
            "Degrau": d["degrau"], "Sistema": nome(d["chave"]), "Exact match": valor_com_intervalo(d, "exact_match"),
            **{f"{qualidade.NOMES[m]}: diferença para o degrau anterior": "—" if pd.isna(d["anterior"]) else com_intervalo(d, f"{m}_", casas_da(d, m)) for m in ("exact_match", "intent_acc", "slot_f1")},
            "Tamanho em disco": "não divulgado" if not T.loc[d["chave"], "divulgado"] else tamanho(d["bytes"]), "Latência p50": ms(d["latencia_p50_ms"]), "Onde": d["condicao_da_latencia"],
        }
        for _, d in cadeia(caminho, cenario, particao).iterrows()
    ])


def lado_a_lado(cenario, particao="test"):
    """Os dois caminhos na mesma tabela: em cada degrau, o exact match e a diferença para o degrau anterior de cada um."""
    z, x = cadeia(ZERO, cenario, particao).set_index("degrau"), cadeia(XLMR, cenario, particao).set_index("degrau")
    linhas = []
    for degrau in z.index:
        linha = {"Degrau": degrau}
        for rotulo, c in ((CAMINHOS[ZERO]["curto"], z), (CAMINHOS[XLMR]["curto"], x)):
            d = c.loc[degrau]
            linha.update({f"{rotulo}: exact match": valor_com_intervalo(d, "exact_match"), f"{rotulo}: diferença para o anterior": "—" if pd.isna(d["anterior"]) else com_intervalo(d, "exact_match_", casas_da(d, "exact_match")),
                          f"{rotulo}: tamanho": "não divulgado" if not T.loc[d["chave"], "divulgado"] else tamanho(d["bytes"]), f"{rotulo}: latência p50": ms(d["latencia_p50_ms"])})
        linhas.append(linha)
    return pd.DataFrame(linhas)


def figura_lado_a_lado(cenario):
    z, x = cadeia(ZERO, cenario), cadeia(XLMR, cenario)
    serie = lambda c: {d["degrau"]: trio(d, ("exact_match", "exact_match_ic_min", "exact_match_ic_max")) for _, d in c.iterrows()}
    return visual.pontos_com_intervalo({CAMINHOS[ZERO]["nome"]: serie(z), CAMINHOS[XLMR]["nome"]: serie(x)}, f"{CENARIOS[cenario]}, no teste", "exact match (%), com intervalo de 95%")


def figura_da_cadeia(caminho, cenario):
    """A cadeia de um caminho no formato da Etapa 9: exact match, tamanho e latência em cada degrau."""
    c = cadeia(caminho, cenario)
    com_nome = 2 if c["degrau"].iloc[0] == D["professor"] else 1
    degraus = [f"{d}\n{T.loc[k, 'curto']}" if i < com_nome else f"{d}\nmodelo final" if T.loc[k, "final"] else d for i, (d, k) in enumerate(zip(c["degrau"], c["chave"]))]
    megas = [None if pd.isna(b) else b / 1e6 for b in c["bytes"]]
    latencias = list(c["latencia_p50_ms"])
    return visual.degraus_em_paineis(
        degraus,
        [
            {"eixo_x": "exact match (%)", "valores": [trio(d, ("exact_match", "exact_match_ic_min", "exact_match_ic_max")) for _, d in c.iterrows()], "rotulos": [pts(v) for v in c["exact_match"]]},
            {"eixo_x": "tamanho em disco (MB, escala logarítmica)", "valores": megas, "rotulos": ["não divulgado" if v is None else tamanho(v * 1e6) for v in megas], "log": True},
            {"eixo_x": "latência p50 (ms, escala logarítmica)", "valores": latencias, "rotulos": [f"{ms(v)} · {onde}" for v, onde in zip(latencias, c["condicao_da_latencia"])], "log": True},
        ],
        f"{CAMINHOS[caminho]['curto']} · {CENARIOS[cenario]}",
    )


def leitura_da_cadeia(caminho, cenario):
    """A cadeia de um caminho em frases: o que cada passo fez com a qualidade, o tamanho e a latência."""
    c = cadeia(caminho, cenario).set_index("degrau")
    em_dev = cadeia(caminho, cenario, "dev").set_index("degrau")
    frases = []
    if D["professor"] in c.index:
        d = c.loc[D["destilado"]]
        frases.append(f"**Destilação.** O aluno fica {posicao(d['exact_match_dif_ic_min'], d['exact_match_dif_ic_max'])}{' do professor' if posicao(d['exact_match_dif_ic_min'], d['exact_match_dif_ic_max']) != 'sem diferença distinguível' else ' para o professor'} em exact match: "
                      f"{com_intervalo(d, 'exact_match_')} no teste e {com_intervalo(em_dev.loc[D['destilado']], 'exact_match_')} em `dev`. A consulta sai de {ms(c.loc[D['professor'], 'latencia_p50_ms'])} ({c.loc[D['professor'], 'condicao_da_latencia']}) "
                      f"para {ms(d['latencia_p50_ms'])} ({d['condicao_da_latencia']}, 1 thread), e o sistema passa a ser um arquivo de {tamanho(d['bytes'])}.")
    d, antes = c.loc[D["podado"]], c.iloc[c.index.get_loc(D["podado"]) - 1]
    filtro = {k: v for k, v in fichas[d["chave"]]["degrau_gravado"][1].items() if k in ("modelo", "variante")}
    da_poda = lambda comparacao: etapa9.linha_gravada(poda_gravada[caminho], **filtro, comparacao=comparacao, particao="test", metrica="exact_match")
    sem_reajuste, recupera = da_poda("antes − sem poda"), da_poda("depois − antes")
    frases.append(f"**Poda e reajuste.** Logo depois do corte, sem reajuste, o exact match fica {com_intervalo(sem_reajuste)} em relação ao modelo sem poda; o reajuste muda {com_intervalo(recupera)} ({ETAPA_DA_PODA[caminho]}). "
                  f"Depois do reajuste, o modelo podado fica {posicao(d['exact_match_dif_ic_min'], d['exact_match_dif_ic_max'])} em relação ao modelo sem poda: {com_intervalo(d, 'exact_match_', casas_da(d, 'exact_match'))} no teste e {com_intervalo(em_dev.loc[D['podado']], 'exact_match_', casas_da(em_dev.loc[D['podado']], 'exact_match'))} em `dev`. "
                  f"O arquivo vai de {tamanho(antes['bytes'])} para {tamanho(d['bytes'])} ({vezes(antes['bytes'], d['bytes'])} vezes menos); "
                  f"a latência {'não muda' if abs(d['latencia_p50_ms'] / antes['latencia_p50_ms'] - 1) <= FOLGA else 'muda'} ({ms(antes['latencia_p50_ms'])} e {ms(d['latencia_p50_ms'])}).")
    o, i = c.loc[D["onnx"]], c.loc[D["int8"]]
    frases.append(f"**Exportação.** Sair do PyTorch não muda nenhuma resposta ({com_intervalo(o, 'exact_match_', 2)}) e leva a consulta de {ms(d['latencia_p50_ms'])} para {ms(o['latencia_p50_ms'])} ({vezes(d['latencia_p50_ms'], o['latencia_p50_ms'])} vezes menos).")
    outras = [f"{metrica_na_frase(m)} {posicao(i[f'{m}_dif_ic_min'], i[f'{m}_dif_ic_max'])} ({com_intervalo(i, f'{m}_', 2)} no teste, {com_intervalo(em_dev.loc[D['int8']], f'{m}_', 2)} em `dev`)" for m in ("intent_acc", "slot_f1")]
    frases.append(f"**8 bits.** Passar para INT8 muda o exact match em {com_intervalo(i, 'exact_match_', 2)} ({posicao(i['exact_match_dif_ic_min'], i['exact_match_dif_ic_max'])}); {'; '.join(outras)}. "
                  f"O arquivo vai de {tamanho(o['bytes'])} para {tamanho(i['bytes'])} ({vezes(o['bytes'], i['bytes'])} vezes menos) e a consulta, de {ms(o['latencia_p50_ms'])} para {ms(i['latencia_p50_ms'])} ({vezes(o['latencia_p50_ms'], i['latencia_p50_ms'])} vezes menos).")
    return frases


leituras_das_cadeias, FIGURAS_DAS_CADEIAS = {}, []
for cenario in CENARIOS:
    display(Markdown(f"### {CENARIOS[cenario]}"))
    display(lado_a_lado(cenario).set_index("Degrau"))
    fig, _ = figura_lado_a_lado(cenario)
    visual.salvar(fig, PASTA, f"cadeias_lado_a_lado_{ARQUIVO_DO_CENARIO[cenario]}")
    plt.show()
    fig, _ = figura_da_cadeia(XLMR, cenario)
    visual.salvar(fig, PASTA, f"cadeia_xlmr_{ARQUIVO_DO_CENARIO[cenario]}")
    plt.show()
    FIGURAS_DAS_CADEIAS += [f"cadeias_lado_a_lado_{ARQUIVO_DO_CENARIO[cenario]}", f"cadeia_xlmr_{ARQUIVO_DO_CENARIO[cenario]}"]
    for caminho in (ZERO, XLMR):
        leituras_das_cadeias[(caminho, cenario)] = leitura_da_cadeia(caminho, cenario)
        display(Markdown(f"**{CAMINHOS[caminho]['nome']}**\n\n" + "\n".join(f"- {f}" for f in leituras_das_cadeias[(caminho, cenario)])))
''')

code(r'''
passos = CAD[CAD["anterior"].notna()]
NOME_DA_CADEIA = {"A": "cenário A", "B": "cenário B", "teto": "teto"}


def distinguiveis_da(caminho, particao):
    """Os (cenário, degrau, métrica) em que um degrau do caminho se distingue de zero."""
    c = passos[(passos["cadeia"].str.startswith(caminho + "/")) & (passos["particao"] == particao)]
    return [(d["cadeia"].split("/")[1], d["degrau"], m) for _, d in c.iterrows() for m in TRES if d[f"{m}_distinguivel"] == True]   # noqa: E712


def frase_dos_degraus(caminho, particao="test"):
    achados = distinguiveis_da(caminho, particao)
    linha = lambda cen, g: passos[(passos["cadeia"] == f"{caminho}/{cen}") & (passos["degrau"] == g) & (passos["particao"] == particao)].iloc[0]
    return "; ".join(f"{NOME_DA_CADEIA[cen]}, {minusculo(g)}, {metrica_na_frase(m)} ({com_intervalo(linha(cen, g), m + '_', casas_da(linha(cen, g), m))})" for cen, g, m in achados) or "nenhuma"


total_de_degraus = int((passos["particao"] == "test").sum() / len(CAMINHOS))
compressao_x = [(cen, g, m) for cen, g, m in distinguiveis_da(XLMR, "test") if g != D["destilado"]]
compressao_z = [(cen, g, m) for cen, g, m in distinguiveis_da(ZERO, "test") if g != D["destilado"] and cen != "teto"]
nas_duas = lambda caminho: sorted(set(distinguiveis_da(caminho, "test")) & set(distinguiveis_da(caminho, "dev")))
leitura_das_cadeias = [
    f"**{CAMINHOS[ZERO]['nome']}.** De {total_de_degraus} degraus × {len(TRES)} métricas, {len(distinguiveis_da(ZERO, 'test'))} diferenças se distinguem de zero no teste: {frase_dos_degraus(ZERO)}. "
    + ("Nos dois cenários, a única mudança de qualidade que se distingue de zero é a da destilação; nenhum degrau de compressão se distingue (é a leitura da Etapa 9)." if not compressao_z else "Há degraus de compressão que se distinguem nos cenários."),
    f"**{CAMINHOS[XLMR]['nome']}.** De {total_de_degraus} degraus × {len(TRES)} métricas, {len(distinguiveis_da(XLMR, 'test'))} se distinguem de zero no teste: {frase_dos_degraus(XLMR)}. "
    + (f"Aqui a compressão não é neutra em todos os degraus: {len(compressao_x)} das diferenças distinguíveis estão em degraus de compressão." if compressao_x else "Nenhum degrau de compressão se distingue."),
    f"**`dev` e teste.** As diferenças que se distinguem de zero nas duas partições ao mesmo tempo são {len(nas_duas(ZERO))} na rede do zero e {len(nas_duas(XLMR))} no XLM-R"
    + (f" (no XLM-R: {'; '.join(f'{NOME_DA_CADEIA[cen]}, {minusculo(g)}, {metrica_na_frase(m)}' for cen, g, m in nas_duas(XLMR))})" if nas_duas(XLMR) else "") + ". "
    f"São {2 * total_de_degraus * len(TRES)} intervalos por partição, sem correção: cerca de {num(0.05 * 2 * total_de_degraus * len(TRES), 1)} sairia do zero só por acaso.",
    "**O custo.** Nos dois caminhos a poda diminui o arquivo e não muda a latência, a exportação diminui a latência, e os 8 bits diminuem os dois. O que muda é a proporção: na rede do zero a exportação rende mais que os 8 bits; no XLM-R, o contrário (Etapa 10c).",
]
display(Markdown("**Leitura das seis cadeias.**\n\n" + "\n".join(f"- {l}" for l in leitura_das_cadeias)))
''')

md(r'''
## 6. Os dois caminhos no mesmo ponto

A pergunta da etapa, ponto por ponto: o mesmo modelo, no mesmo ponto da cadeia, XLM-R menos rede do zero. A última linha de cada aluno é a forma final de 8 bits; a do aluno A é a comparação entre os dois modelos finais.

Uma ressalva vale para todas as linhas: os dois caminhos não diferem só no pré-treino. Mudam juntos a arquitetura, o tamanho, a receita de treino, a cabeça de entidades (com CRF na rede do zero, sem no XLM-R) e, nos pontos podados, a própria poda. A comparação mede o pacote.
''')

code(r'''
entre = [c for c in regras["comparacoes"] if "entre_caminhos" in c["familias"]]
ORDEM_DOS_MODELOS = ["teto", "piso", "aluno_a", "aluno_b1", "aluno_b2"]
entre = sorted(entre, key=lambda c: (ORDEM_DOS_MODELOS.index(T.loc[c["b"], "modelo"]), list(etapa10d.PONTOS).index(T.loc[c["b"], "ponto"])))
rotulo_do_ponto = lambda c: T.loc[c["a"], "curto"] + (" (os dois modelos finais)" if c["b"] == FX and c["a"] == FZ else "")


def tabela_entre_caminhos(particao):
    return pd.DataFrame([
        {"Ponto (XLM-R − rede do zero)": rotulo_do_ponto(c), **{qualidade.NOMES[m]: com_intervalo(comp(c["b"], c["a"], m, particao)) for m in TRES},
         "Exact match: do zero → XLM-R": f"{pts(comp(c['b'], c['a'], 'exact_match', particao)['valor_a'])} → {pts(comp(c['b'], c['a'], 'exact_match', particao)['valor_b'])}"}
        for c in entre
    ])


tabelas_entre = {p: tabela_entre_caminhos(p) for p in ("test", "dev")}
for p in ("test", "dev"):
    display(Markdown(f"**Em {'`dev`' if p == 'dev' else 'teste'}** (diferença pareada, em pontos percentuais, intervalo de 95%)"))
    display(tabelas_entre[p].set_index("Ponto (XLM-R − rede do zero)"))

FIGURAS_ENTRE = {"exact_match": "entre_caminhos_exact_match", "intent_acc": "entre_caminhos_intencao", "slot_f1": "entre_caminhos_entidades"}
for metrica, arquivo in FIGURAS_ENTRE.items():
    fig, ax = diferencas({rotulo_do_ponto(c).replace(" (os dois modelos finais)", " (finais)"): trio(comp(c["b"], c["a"], metrica)) for c in entre},
                         f"XLM-R menos rede do zero, no mesmo ponto, no teste: {metrica_na_frase(metrica)}", "diferença, em pontos percentuais (intervalo de 95%)")
    visual.salvar(fig, PASTA, arquivo)
    plt.show()
''')

code(r'''
def conta(particao, metrica, sinal):
    linhas = [comp(c["b"], c["a"], metrica, particao) for c in entre]
    return sum((l["ic_min"] > 0) if sinal > 0 else (l["ic_max"] < 0) for l in linhas)


def discordam(lista_de_pares, metricas=TRES):
    """As comparações em que dev e teste não dizem o mesmo (acima, abaixo ou sem diferença distinguível)."""
    return [(b, a, m) for b, a in lista_de_pares for m in metricas if posicao(comp(b, a, m, "dev")["ic_min"], comp(b, a, m, "dev")["ic_max"]) != posicao(comp(b, a, m, "test")["ic_min"], comp(b, a, m, "test")["ic_max"])]


pares_entre = [(c["b"], c["a"]) for c in entre]
discordam_entre = discordam(pares_entre)
sem_distincao_em = [rotulo_do_ponto(c) for c in entre if not comp(c["b"], c["a"])["distinguivel"]]
do_aluno = lambda modelo: [c for c in entre if T.loc[c["b"], "modelo"] == modelo]
trajeto = lambda modelo: " → ".join(f"{dif(comp(c['b'], c['a'])['diferenca'])} ({minusculo(etapa10d.PONTOS[T.loc[c['b'], 'ponto']])})" for c in do_aluno(modelo))
leitura_entre = [
    f"**Os dois modelos finais.** No teste, o final do XLM-R fica {frase(FX, FZ)} em relação ao final da rede do zero. Em `dev`: {frase(FX, FZ, 'dev')}.",
    f"**Em todos os pontos.** Em exact match, no teste, o XLM-R fica acima da rede do zero em {conta('test', 'exact_match', +1)} dos {len(entre)} pontos e abaixo em {conta('test', 'exact_match', -1)}"
    + (f"; não se distingue em: {lista(sem_distincao_em)}" if sem_distincao_em else "") + f". Em acurácia de intenção, acima em {conta('test', 'intent_acc', +1)} e abaixo em {conta('test', 'intent_acc', -1)}; em F1 de entidades, acima em {conta('test', 'slot_f1', +1)} e abaixo em {conta('test', 'slot_f1', -1)}.",
    f"**Ao longo da cadeia** (exact match, teste). Aluno A: {trajeto('aluno_a')}. Aluno B2: {trajeto('aluno_b2')}. Teto: {trajeto('teto')}. "
    + " ".join(f"No {minusculo(etapa10d.NOME_DO_MODELO[m])}, a diferença se distingue de zero em {sum(bool(comp(c['b'], c['a'])['distinguivel']) for c in do_aluno(m))} dos {len(do_aluno(m))} pontos." for m in ("aluno_a", "aluno_b2", "teto"))
    + " Onde há vantagem do XLM-R, ela já está no modelo sem poda e chega até o INT8; onde não há no modelo sem poda (aluno B2), a compressão não cria.",
    f"**`dev` e teste.** Das {len(entre) * len(TRES)} comparações (ponto × métrica), `dev` e teste dizem o mesmo em {len(entre) * len(TRES) - len(discordam_entre)}"
    + (f"; discordam em: {'; '.join(f'{T.loc[a, chr(99) + chr(117) + chr(114) + chr(116) + chr(111)]}, {metrica_na_frase(m)} ({com_intervalo(comp(b, a, m, chr(100) + chr(101) + chr(118)))} em dev e {com_intervalo(comp(b, a, m))} no teste)' for b, a, m in discordam_entre)}" if discordam_entre else "") + ".",
    f"São {len(entre) * len(TRES)} intervalos por partição, sem correção: cerca de {num(0.05 * len(entre) * len(TRES), 1)} sairia do zero só por acaso. As primárias são as três do modelo final.",
]
display(Markdown("**Leitura.**\n\n" + "\n".join(f"- {l}" for l in leitura_entre)))
''')

md(r'''
## 7. Cada aluno contra o seu professor, e de ponta a ponta

Primeiro, cada aluno menos a LLM que rotulou o treino dele, nos dois caminhos: como saiu da destilação e na forma final de 8 bits. Depois, as dez comparações de ponta a ponta da Etapa 9, repetidas com os sistemas do XLM-R. Por fim, a poda vista dentro dos 8 bits: o modelo podado em INT8 menos o mesmo modelo em INT8 sem poda.
''')

code(r'''
contra_o_professor = [c for c in regras["comparacoes"] if "aluno_professor" in c["familias"]]


def tabela_de_pares(itens, particao, rotulo="Comparação (B − A)"):
    return pd.DataFrame([{rotulo: f"{nome(c['b'])} − {nome(c['a'])}", **{qualidade.NOMES[m]: com_intervalo(comp(c["b"], c["a"], m, particao)) for m in TRES}} for c in itens])


tabelas_professor = {p: tabela_de_pares(contra_o_professor, p, "Aluno − professor") for p in ("test", "dev")}
for p in ("test", "dev"):
    display(Markdown(f"**Aluno menos professor, em {'`dev`' if p == 'dev' else 'teste'}**"))
    display(tabelas_professor[p].set_index("Aluno − professor"))
fig, ax = diferencas({f"{nome(c['b'])} − {T.loc[c['a'], 'curto']}": trio(comp(c["b"], c["a"])) for c in contra_o_professor},
                     "Cada aluno menos o seu professor, no teste: exact match", "diferença, em pontos percentuais (intervalo de 95%)")
visual.salvar(fig, PASTA, "aluno_menos_professor_exact_match")
plt.show()

# De ponta a ponta: as dez letras da Etapa 9, nos dois caminhos
LETRAS = [letra for letra, *_ in etapa9.PONTA_A_PONTA]
par_da_letra = {caminho: {letra: (etapa10d.chave_no(caminho, b, fichas, CAMINHOS), etapa10d.chave_no(caminho, a, fichas, CAMINHOS)) for letra, b, a, _ in etapa9.PONTA_A_PONTA} for caminho in (ZERO, XLMR)}
ROTULO_DA_LETRA = {"a": "a. Final − professor fechado", "b": "b. Final − aluno A sem compressão", "c": "c. Final − teto do caminho", "d": "d. Teto comprimido − teto", "e1": "e1. Final do cenário B − professor aberto",
                   "e2": "e2. Final do cenário B − aluno B2 sem compressão", "f": "f. Final do cenário A − final do cenário B", "g1": f"g1. Final − linha de base, {n(POUCOS)} rótulos",
                   "g2": f"g2. Final − linha de base, {n(TODOS)} rótulos", "g3": "g3. Final − piso do caminho"}


def tabela_de_ponta_a_ponta(particao):
    return pd.DataFrame([
        {"Comparação (B − A), em cada caminho": ROTULO_DA_LETRA[letra],
         **{f"{CAMINHOS[c]['curto']}: {qualidade.NOMES[m].lower() if not qualidade.NOMES[m].startswith('F1') else qualidade.NOMES[m]}": com_intervalo(comp(*par_da_letra[c][letra], m, particao)) for m in ("exact_match", "intent_acc", "slot_f1") for c in (ZERO, XLMR)}}
        for letra in LETRAS
    ])


tabelas_pp = {p: tabela_de_ponta_a_ponta(p) for p in ("test", "dev")}
display(Markdown("**De ponta a ponta, no teste**"))
display(tabelas_pp["test"].set_index("Comparação (B − A), em cada caminho"))
fig, ax = visual.pontos_com_intervalo({CAMINHOS[c]["nome"]: {ROTULO_DA_LETRA[letra]: trio(comp(*par_da_letra[c][letra])) for letra in LETRAS} for c in (ZERO, XLMR)},
                                      "De ponta a ponta, no teste: exact match", "diferença (o primeiro menos o segundo), em pontos percentuais (intervalo de 95%)")
ax.axvline(0, color=visual.TINTA_SUAVE, linewidth=1.2, zorder=0.6)
for anotacao in ax.texts:   # diferenças: o sinal fica explícito
    valor = float(anotacao.get_text().replace(",", "."))
    anotacao.set_text(f"{valor:+.1f}".replace(".", ",").replace("-", "−"))
visual.salvar(fig, PASTA, "ponta_a_ponta_exact_match")
plt.show()

poda8 = [c for c in regras["comparacoes"] if "poda_nos_8_bits" in c["familias"]]
tabelas_poda8 = {p: tabela_de_pares(poda8, p, "Podado em INT8 − sem poda em INT8") for p in ("test", "dev")}
display(Markdown("**A poda dentro dos 8 bits, no teste**"))
display(tabelas_poda8["test"].set_index("Podado em INT8 − sem poda em INT8"))
''')

code(r'''
pz, px = par_da_letra[ZERO], par_da_letra[XLMR]
em = lambda par, particao="test": comp(*par, "exact_match", particao)
passam = [(c["b"], m) for c in contra_o_professor for m in TRES if comp(c["b"], c["a"], m)["ic_min"] > 0]
abaixo = [(c["b"], m) for c in contra_o_professor for m in TRES if comp(c["b"], c["a"], m)["ic_max"] < 0]
todos_os_pares = [(c["b"], c["a"]) for c in regras["comparacoes"]]
total_de_intervalos = len(todos_os_pares) * len(TRES)
distinguiveis_no_teste = sum(bool(comp(b, a, m)["distinguivel"]) for b, a in todos_os_pares for m in TRES)
distinguiveis_em_dev = sum(bool(comp(b, a, m, "dev")["distinguivel"]) for b, a in todos_os_pares for m in TRES)
sem_poda_x, sem_poda_z = f"{fichas[FX]['de_partida']}/int8_sem_poda", f"{fichas[FZ]['de_partida']}/int8_sem_poda"
leituras_pp = [
    f"**a. Contra o professor fechado.** O final da rede do zero fica {frase(*pz['a'])} em relação ao {PROFESSOR}; o do XLM-R, {frase(*px['a'])}. Em `dev`, em exact match: {com_intervalo(em(pz['a'], 'dev'))} e {com_intervalo(em(px['a'], 'dev'))}.",
    f"**Quem passa o professor** (intervalo todo acima de zero, no teste): {lista(f'{nome(b)} em {metrica_na_frase(m)}' for b, m in passam) if passam else 'ninguém'}. "
    f"**Quem fica abaixo**: {lista(f'{nome(b)} em {metrica_na_frase(m)}' for b, m in abaixo) if abaixo else 'ninguém'}.",
    f"**b. O custo de toda a compressão.** Na rede do zero: {frase(*pz['b'])}. No XLM-R: {frase(*px['b'])}. "
    f"Em tamanho, a compressão leva a rede do zero de {tamanho(T.loc[pz['b'][1], 'bytes'])} para {tamanho(T.loc[FZ, 'bytes'])} ({vezes(T.loc[pz['b'][1], 'bytes'], T.loc[FZ, 'bytes'])} vezes) e o XLM-R de {tamanho(T.loc[px['b'][1], 'bytes'])} para {tamanho(T.loc[FX, 'bytes'])} ({vezes(T.loc[px['b'][1], 'bytes'], T.loc[FX, 'bytes'])} vezes); "
    f"em latência, de {ms(T.loc[pz['b'][1], 'latencia_p50_ms'])} para {ms(T.loc[FZ, 'latencia_p50_ms'])} ({vezes(T.loc[pz['b'][1], 'latencia_p50_ms'], T.loc[FZ, 'latencia_p50_ms'])} vezes) e de {ms(T.loc[px['b'][1], 'latencia_p50_ms'])} para {ms(T.loc[FX, 'latencia_p50_ms'])} ({vezes(T.loc[px['b'][1], 'latencia_p50_ms'], T.loc[FX, 'latencia_p50_ms'])} vezes).",
    f"**c. Contra o teto do próprio caminho.** Sem os {n(TODOS)} rótulos humanos de treino, o final da rede do zero fica {com_intervalo(em(pz['c']))} em exact match e o do XLM-R, {com_intervalo(em(px['c']))}. "
    + ("Pelos valores medidos, a distância para o teto não diminui com o encoder pré-treinado: o teto sobe tanto quanto o aluno, ou mais. " if em(px['c'])['diferenca'] <= em(pz['c'])['diferenca'] else "Pelos valores medidos, a distância para o teto é menor com o encoder pré-treinado. ")
    + "Se as duas distâncias diferem entre si não foi testado.",
    f"**d. A compressão sozinha, no teto.** Rede do zero: {com_intervalo(em(pz['d']))}; XLM-R: {com_intervalo(em(px['d']))} em exact match.",
    f"**e. O cenário B.** O final do cenário B fica {com_intervalo(em(pz['e1']))} em relação ao {ABERTO} na rede do zero e {com_intervalo(em(px['e1']))} no XLM-R; contra o aluno B2 sem compressão, {com_intervalo(em(pz['e2']))} e {com_intervalo(em(px['e2']))}.",
    f"**f. Um cenário contra o outro.** O final do cenário A menos o do cenário B: {com_intervalo(em(pz['f']))} na rede do zero e {com_intervalo(em(px['f']))} no XLM-R. "
    + (f"Pelos valores medidos, trocar de professor pesa mais que trocar de caminho: as duas distâncias entre cenários são maiores que a distância entre os dois modelos finais ({com_intervalo(comp(FX, FZ))})." if min(em(pz['f'])['diferenca'], em(px['f'])['diferenca']) > comp(FX, FZ)['diferenca']
       else f"A distância entre os dois modelos finais é {com_intervalo(comp(FX, FZ))}."),
    f"**g. Contra as referências.** Contra a linha de base com os mesmos {n(POUCOS)} rótulos: {com_intervalo(em(pz['g1']))} e {com_intervalo(em(px['g1']))}; contra a linha de base com os {n(TODOS)}: {com_intervalo(em(pz['g2']))} e {com_intervalo(em(px['g2']))}; contra o piso do próprio caminho: {com_intervalo(em(pz['g3']))} e {com_intervalo(em(px['g3']))} (o piso do XLM-R é mais alto: {pts(T.loc[px['g3'][1], 'exact_match'])}% contra {pts(T.loc[pz['g3'][1], 'exact_match'])}%).",
    f"**A poda dentro dos 8 bits.** O final do XLM-R menos o mesmo aluno em INT8 sem poda: {frase(FX, sem_poda_x)} no teste; em `dev`, {com_intervalo(comp(FX, sem_poda_x, 'exact_match', 'dev'))} em exact match. Na rede do zero: {com_intervalo(comp(FZ, sem_poda_z))} no teste e {com_intervalo(comp(FZ, sem_poda_z, 'exact_match', 'dev'))} em `dev`. "
    f"O INT8 sem poda do XLM-R tem {tamanho(T.loc[sem_poda_x, 'bytes'])}, contra {tamanho(T.loc[FX, 'bytes'])} do final.",
    f"**Quantas se distinguem.** A lista fechada tem {len(todos_os_pares)} comparações × {len(TRES)} métricas = {total_de_intervalos} intervalos por partição; {distinguiveis_no_teste} se distinguem de zero no teste e {distinguiveis_em_dev} em `dev`. "
    f"Sem correção para comparações múltiplas: cerca de {num(0.05 * total_de_intervalos, 0)} sairiam do zero só por acaso. `dev` e teste discordam em {len(discordam(todos_os_pares))} dos {total_de_intervalos}.",
]
display(Markdown("**Leitura.**\n\n" + "\n".join(f"- {l}" for l in leituras_pp)))
''')

md(r'''
## 8. A fronteira e o preço de cada ponto

Os sistemas num plano: custo no eixo horizontal, em **escala logarítmica**, e exact match no vertical. A **cor** diz o caminho. Uma **linha** liga as formas do mesmo aluno (sem compressão, exportado ou podado, INT8). A **linha tracejada** é a fronteira dos sistemas com os poucos rótulos humanos, dos dois caminhos juntos. Em cinza, as LLMs professoras, que os dois caminhos dividem. Círculo vazado: formas entregues que não estão na cadeia (INT8 sem poda; pesos soltos com zip).
''')

code(r'''
SERIES = [CAMINHOS[ZERO]["nome"], CAMINHOS[XLMR]["nome"]]
MARCADOR = {CPU: "o", GPU: "s", API: "D"}
serie = lambda chave: None if caminho_de(chave) is None else int(caminho_de(chave) == XLMR)
em_pontos = lambda chave: 100 * T.loc[chave, "exact_match"]
megas_de = etapa10d.megas_entregues(T)
ROTULO_DA_FRONTEIRA = f"fronteira dos sistemas com {n(POUCOS)} rótulos humanos"
ALUNOS = ("aluno_a", "aluno_b2")
ROTULO_DO_FINAL = {ZERO: "final\ndo zero", XLMR: "final do XLM-R"}
P = {ZERO: CAMINHOS[ZERO]["prefixo"], XLMR: CAMINHOS[XLMR]["prefixo"]}
front = lambda custo, usada="etapas de origem": FRONT[(FRONT["custo"] == custo) & (FRONT["latencia_usada"] == usada)].set_index("chave")
fronteira_lat = list(front("latencia")[front("latencia")["na_fronteira"]].sort_values("valor_do_custo").index)
fronteira_tam = list(front("tamanho")[front("tamanho")["na_fronteira"]].sort_values("valor_do_custo").index)


def ponto(chave, custo, rotulo=None, **extras):
    return {"x": custo(chave), "y": em_pontos(chave), "rotulo": rotulo, "serie": serie(chave), **extras}


# ---- qualidade contra latência
lat = lambda chave: T.loc[chave, "latencia_p50_ms"]
pontos = []
for c in (ZERO, XLMR):
    for a in ALUNOS:
        k = P[c] + a
        final = T.loc[f"{k}/int8", "final"]
        pontos += [ponto(k, lat, T.loc[k, "curto"], lado="baixo" if (c, a) == (ZERO, "aluno_a") else "direita", desvio=(0, 0) if a == "aluno_a" else (0, -3)), ponto(f"{k}/onnx", lat),
                   ponto(f"{k}/int8", lat, ROTULO_DO_FINAL[c] if final else None, lado="cima" if c == ZERO else "esquerda", forte=True),
                   ponto(f"{k}/int8_sem_poda", lat, "INT8 sem poda" if (c, a) == (XLMR, "aluno_a") else None, lado="cima", vazado=True)]
pontos += [
    ponto("aluno_b1", lat, T.loc["aluno_b1", "curto"], lado="direita", desvio=(0, -9)),
    ponto("piso", lat, f"{T.loc['piso', 'curto']} (do zero)", lado="direita"),
    ponto("claude", lat, f"{PROFESSOR}\n{API}", lado="baixo", marcador=MARCADOR[API]),
    ponto("qwen", lat, f"{ABERTO}\n{GPU}", lado="esquerda", marcador=MARCADOR[GPU]),
]
fig, ax = visual.fronteira(
    pontos, "Qualidade contra latência, no teste, com os dois caminhos", "latência p50 de uma consulta (ms, escala logarítmica)", "exact match (%)", series=SERIES,
    trilhas=[{"pontos": [(lat(k), em_pontos(k)) for k in (f"{P[c]}{a}/int8", f"{P[c]}{a}/onnx", P[c] + a)], "serie": int(c == XLMR)} for c in (ZERO, XLMR) for a in ALUNOS],
    escada=[(lat(k), em_pontos(k)) for k in fronteira_lat] + [(3000, em_pontos(fronteira_lat[-1]))],
    nota=f"Tracejado: {ROTULO_DA_FRONTEIRA}.\nCírculos: CPU local, 1 thread. Vazados: INT8 sem poda.\nO podado em PyTorch cai sobre o ponto sem poda.\nPiso e aluno B1 do XLM-R: latência não medida.",
)
ax.set_xlim(0.12, 3000)
ax.set_ylim(17, 60)
visual.salvar(fig, PASTA, "fronteira_latencia")
plt.show()

# ---- qualidade contra tamanho: o arquivo como é entregue
megas = lambda chave: megas_de.loc[chave]
pontos = []
for c in (ZERO, XLMR):
    for a in ALUNOS:
        k = P[c] + a
        final = T.loc[f"{k}/int8", "final"]
        pontos += [ponto(k, megas, T.loc[k, "curto"], lado="baixo" if (c, a) == (ZERO, "aluno_a") else "direita", desvio=(0, 0) if a == "aluno_a" else (0, 3)), ponto(f"{k}/podado", megas),
                   ponto(f"{k}/int8", megas, ROTULO_DO_FINAL[c] if final else None, lado="cima" if c == ZERO else "esquerda", forte=True),
                   ponto(f"{k}/int8_sem_poda", megas, "INT8 sem poda" if (c, a) == (XLMR, "aluno_a") else None, lado="cima", vazado=True), ponto(f"{k}/soltos", megas, vazado=True)]
    pontos += [ponto(P[c] + "aluno_b1", megas, T.loc[P[c] + "aluno_b1", "curto"], lado="direita", desvio=(0, -9)), ponto(P[c] + "piso", megas, T.loc[P[c] + "piso", "curto"], lado="direita")]
if pd.notna(T.loc["qwen", "bytes"]):
    pontos.append(ponto("qwen", megas, f"{ABERTO}\n{tamanho(T.loc['qwen', 'bytes'])} baixados", lado="baixo", marcador=MARCADOR[GPU]))
fig, ax = visual.fronteira(
    pontos, "Qualidade contra tamanho em disco, no teste, com os dois caminhos", "tamanho do arquivo entregue (MB, escala logarítmica)", "exact match (%)", series=SERIES,
    trilhas=[{"pontos": [(megas(k), em_pontos(k)) for k in (f"{P[c]}{a}/int8", f"{P[c]}{a}/podado", P[c] + a)], "serie": int(c == XLMR)} for c in (ZERO, XLMR) for a in ALUNOS],
    escada=[(megas(k), em_pontos(k)) for k in fronteira_tam] + [(60000, em_pontos(fronteira_tam[-1]))],
    nota=f"Tracejado: {ROTULO_DA_FRONTEIRA}.\nVazados: INT8 sem poda e pesos soltos com zip.\n{PROFESSOR}: tamanho não divulgado, fora do gráfico.\nO ONNX de 32 bits cai sobre o ponto podado.",
)
ax.set_xlim(0.4, 60000)
ax.set_ylim(17, 60)
visual.salvar(fig, PASTA, "fronteira_tamanho")
plt.show()
''')

code(r'''
def quem_domina(custo, usada="etapas de origem"):
    """Para cada sistema fora da fronteira: quem o domina, pelos valores medidos."""
    f = front(custo, usada)
    return pd.DataFrame([
        {"Sistema": nome(c), ("Latência p50" if custo == "latencia" else "Tamanho entregue"): ms(l["valor_do_custo"]) if custo == "latencia" else tamanho(1e6 * l["valor_do_custo"]),
         "Exact match": pts(l["exact_match"]), "Na fronteira": "sim" if l["na_fronteira"] else "não",
         "Dominado por": "—" if l["na_fronteira"] else lista(nome(k) for k in l["dominado_por"].split("; "))}
        for c, l in f.sort_values("valor_do_custo").iterrows()
    ])


tabelas_da_fronteira = {custo: quem_domina(custo) for custo in ("latencia", "tamanho")}
for custo, titulo in (("latencia", "Qualidade contra latência"), ("tamanho", "Qualidade contra tamanho")):
    display(Markdown(f"**{titulo}: quem está na fronteira e quem domina quem**"))
    display(tabelas_da_fronteira[custo].set_index("Sistema"))

# O preço de cada ponto, entre os dois modelos finais
def linha_do_preco(l):
    preco = (f"{num(l['por_ponto_valor'], 1 if l['unidade'] == 'MB' else 2)} {l['unidade']} ({num(l['por_ponto_ic_min'], 1 if l['unidade'] == 'MB' else 2)} a {num(l['por_ponto_ic_max'], 1 if l['unidade'] == 'MB' else 2)})"
             if l["definido"] else "não definido: o ganho não se distingue de zero")
    fmt = (lambda v: tamanho(1e6 * v)) if l["unidade"] == "MB" else ms
    return {"Partição": NOME_DA_PARTICAO[l["particao"]], "Custo": "tamanho em disco" if l["custo"] == "tamanho" else f"latência p50 ({l['fonte_do_custo']})",
            "Final do XLM-R": fmt(l["do_final"]), "Final do zero": fmt(l["do_final_do_zero"]), "A mais": fmt(l["custo_a_mais"]), "Vezes": vezes(l["do_final"], l["do_final_do_zero"]),
            "Exact match a mais (pontos)": f"{dif(l['exact_match_a_mais'])} ({dif(l['exact_match_ic_min'])} a {dif(l['exact_match_ic_max'])})", "Preço de um ponto de exact match": preco}


tabela_dos_precos = pd.DataFrame([linha_do_preco(l) for _, l in PRECOS.sort_values("particao", ascending=False, kind="stable").iterrows()])
display(Markdown("**O preço de cada ponto de exact match, entre os dois modelos finais**"))
display(tabela_dos_precos)
''')

code(r'''
preco = lambda custo, particao="test", fonte=None: PRECOS[(PRECOS["custo"] == custo) & (PRECOS["particao"] == particao) & ((PRECOS["fonte_do_custo"] == fonte) if fonte else True)].iloc[0]
texto_do_preco = lambda l, casas: f"{num(l['por_ponto_valor'], casas)} {l['unidade']} ({num(l['por_ponto_ic_min'], casas)} a {num(l['por_ponto_ic_max'], casas)})" if l["definido"] else "não definido (o ganho não se distingue de zero)"
p_mb, p_ms = preco("tamanho"), preco("latencia", fonte="etapas de origem")
p_mb_dev, p_ms_dev = preco("tamanho", "dev"), preco("latencia", "dev", "etapas de origem")
nomes_com_valor = lambda chaves: lista(f"{nome(c)} ({pts(T.loc[c, 'exact_match'])}%)" for c in chaves)
f_lat, f_tam = front("latencia"), front("tamanho")
dominadores = lambda f, chave: lista(nome(k) for k in f.loc[chave, "dominado_por"].split("; "))
situacao = lambda f, chave: "está na fronteira" if f.loc[chave, "na_fronteira"] else f"é dominado por {dominadores(f, chave)}"
medida_nova = len(LAT) > 0
f_lat_nova = front("latencia", "mesma sessão, nos sistemas medidos de novo") if medida_nova else None
fronteira_lat_nova = list(f_lat_nova[f_lat_nova["na_fronteira"]].sort_values("valor_do_custo").index) if medida_nova else None
de_outro_caminho = lambda f, chave: [k for k in f.loc[chave, "dominado_por"].split("; ") if k and caminho_de(k) not in (None, caminho_de(chave))]
ninguem_cruza = not any(de_outro_caminho(f, chave) for f in (f_lat, f_tam) for chave in (FZ, FX))
poda8_z, poda8_x = comp(FZ, sem_poda_z), comp(FX, sem_poda_x)

# Exploratória (acrescentada depois de ver a fronteira; não está nas regras): a mesma fronteira só com as linhas da Etapa 9 e as equivalentes do XLM-R, sem o INT8 sem poda
SEM_INT8_SEM_PODA = etapa10d.fronteiras(T[T["ponto"] != "int8_sem_poda"], FOLGA).assign(exploratoria=True)
da_cadeia = lambda custo: list(SEM_INT8_SEM_PODA[(SEM_INT8_SEM_PODA["custo"] == custo) & SEM_INT8_SEM_PODA["na_fronteira"]].sort_values("valor_do_custo")["chave"])

leitura_da_fronteira = [
    "**Quem domina quem, entre os caminhos.** " + ("Nenhum sistema de um caminho domina o modelo final do outro, nem em latência nem em tamanho: a rede do zero ocupa o lado barato das duas fronteiras e o XLM-R, o lado de mais qualidade. "
                                                  if ninguem_cruza else "Há sistema de um caminho que domina o modelo final do outro (tabelas acima). ")
    + ("Nenhum sistema do cenário B (alunos B1 e B2, em qualquer forma, nos dois caminhos) está em nenhuma das duas fronteiras. " if not any(f.loc[c, "na_fronteira"] for f in (f_lat, f_tam) for c in f.index if T.loc[c, "modelo"] in ("aluno_b1", "aluno_b2")) else "Há sistema do cenário B na fronteira. ")
    + f"Em latência, o {ABERTO} {'é dominado por ' + str(int(f_lat.loc['qwen', 'quantos_o_dominam'])) + ' sistemas, dos dois caminhos' if not f_lat.loc['qwen', 'na_fronteira'] else 'está na fronteira'}, e o {PROFESSOR} {situacao(f_lat, 'claude')}.",
    f"**Na latência, a poda não muda o custo.** Nos dois caminhos o modelo podado e o mesmo modelo sem poda respondem no mesmo tempo (seção 5), e a regra da fronteira trata latências a menos de {num(100 * FOLGA, 0)}% como empate. Entre dois empatados, fica na fronteira o de maior exact match medido. "
    f"Por isso o modelo final de cada caminho aparece dominado pelo seu próprio INT8 sem poda: na rede do zero por {num(abs(100 * poda8_z['diferenca']), 2)} ponto, que não se distingue de zero ({com_intervalo(poda8_z, casas=2)}); "
    f"no XLM-R por {num(abs(100 * poda8_x['diferenca']), 1)} ponto, que {'se distingue' if poda8_x['distinguivel'] else 'não se distingue'} no teste ({com_intervalo(poda8_x)}) e {'se distingue' if comp(FX, sem_poda_x, 'exact_match', 'dev')['distinguivel'] else 'não se distingue'} em `dev` ({com_intervalo(comp(FX, sem_poda_x, 'exact_match', 'dev'))}).",
    f"**Latência.** Entre os sistemas com {n(POUCOS)} rótulos humanos, dos dois caminhos, a fronteira tem: {nomes_com_valor(fronteira_lat)}. O final da rede do zero {situacao(f_lat, FZ)}; o final do XLM-R {situacao(f_lat, FX)}; "
    f"o {PROFESSOR} ({ms(lat('claude'))} pela API) {situacao(f_lat, 'claude')}. Latências a menos de {num(100 * FOLGA, 0)}% uma da outra contam como empate."
    + (f" Com as latências medidas na mesma sessão no lugar das de origem, a fronteira {'é a mesma' if fronteira_lat_nova == fronteira_lat else 'muda para: ' + nomes_com_valor(fronteira_lat_nova)}." if medida_nova else ""),
    f"**Tamanho.** A fronteira tem: {nomes_com_valor(fronteira_tam)}. O final da rede do zero {situacao(f_tam, FZ)}; o final do XLM-R {situacao(f_tam, FX)}.",
    f"**O INT8 sem poda do XLM-R.** Ele aparece na fronteira com {pts(T.loc[sem_poda_x, 'exact_match'])}% de exact match, {tamanho(T.loc[sem_poda_x, 'bytes'])} e {ms(lat(sem_poda_x))}. Não é o modelo final: a regra da Etapa 10b, escrita antes, escolhe o menor arquivo que não se distingue do modelo sem poda em `dev`. "
    f"Contra ele, o final fica {frase(FX, sem_poda_x)} no teste, e em `dev` {posicao(comp(FX, sem_poda_x, 'exact_match', 'dev')['ic_min'], comp(FX, sem_poda_x, 'exact_match', 'dev')['ic_max'])} em exact match ({com_intervalo(comp(FX, sem_poda_x, 'exact_match', 'dev'))}). "
    f"A fronteira ordena valores medidos; a diferença entre esses dois vizinhos é a que está nesta linha. O INT8 sem poda aparece acima do {PROFESSOR} pelo valor medido ({pts(T.loc[sem_poda_x, 'exact_match'])}% contra {pts(T.loc['claude', 'exact_match'])}%), mas essa comparação não está na lista fechada e não foi feita: não se pode dizer que ele passa o professor.",
    f"**Sem o INT8 sem poda (exploratória).** Esta conta foi acrescentada depois de ver a fronteira e não está nas regras. Só com as linhas da Etapa 9 e as equivalentes do XLM-R, a fronteira de latência seria: {nomes_com_valor(da_cadeia('latencia'))}; a de tamanho: {nomes_com_valor(da_cadeia('tamanho'))}. "
    f"O final da rede do zero {'estaria' if FZ in da_cadeia('latencia') else 'não estaria'} na de latência e o do XLM-R {'estaria' if FX in da_cadeia('latencia') else 'não estaria'}.",
    f"**O preço de cada ponto.** Entre os dois modelos finais, no teste, o XLM-R tem {dif(p_mb['exact_match_a_mais'])} ({dif(p_mb['exact_match_ic_min'])} a {dif(p_mb['exact_match_ic_max'])}) pontos de exact match a mais, {tamanho(1e6 * p_mb['custo_a_mais'])} a mais ({vezes(p_mb['do_final'], p_mb['do_final_do_zero'])} vezes) e {ms(p_ms['custo_a_mais'])} a mais por consulta ({vezes(p_ms['do_final'], p_ms['do_final_do_zero'])} vezes). "
    f"Cada ponto de exact match custa {texto_do_preco(p_mb, 1)} e {texto_do_preco(p_ms, 2)}. Em `dev`: {texto_do_preco(p_mb_dev, 1)} e {texto_do_preco(p_ms_dev, 2)}."
    + (f" Com a latência medida na mesma sessão: {texto_do_preco(preco('latencia', fonte='mesma sessão'), 2)} no teste." if medida_nova else ""),
    "**Como ler.** A fronteira usa os valores medidos, sem os intervalos: serve para ver a ordem de grandeza do custo. Dois vizinhos nela podem não se distinguir em qualidade; quem diz isso são as comparações pareadas. O intervalo do preço vem só da incerteza da qualidade: tamanho e latência são os de um modelo (semente 0).",
]
display(Markdown("**Leitura.**\n\n" + "\n".join(f"- {l}" for l in leitura_da_fronteira)))
''')

md(r'''
## 9. A latência dos dois modelos finais, na mesma sessão

As latências dos dois caminhos vieram de sessões diferentes: as da rede do zero, da Etapa 8; as do XLM-R, da Etapa 10c. O protocolo é o mesmo nas duas, e é a mesma função (`eficiencia.medir_latencia`, chamada por `etapa8.medir_latencia` e por `etapa10c.medir_latencia`):

| | Etapa 8 (rede do zero) | Etapa 10c (XLM-R) | Esta etapa |
|---|---|---|---|
| Threads | 1 | 1 | 1 |
| Consultas | uma por vez, do texto à resposta | igual | igual |
| Pedidos | os de `dev`, sorteados com a semente da rodada | igual | igual |
| Aquecimento | 50 consultas por rodada | igual | igual |
| Rodadas | 5 de 1.024 consultas, sistemas alternados | igual | igual |
| Sistemas na mesma rodada | as formas de um modelo | as formas de um modelo | **os dois modelos finais**, em cada forma |
| Máquina quieta | outros processos até 0,25 núcleo | igual | igual |

O que as medidas de origem não têm é os dois caminhos **na mesma rodada**. O comando abaixo faz isso: os dois modelos finais (semente 0) em INT8, em ONNX de 32 bits e em PyTorch de 32 bits, alternados, com o medidor de "outros processos" das Etapas 8 e 10c. Cada medida vai para uma pasta com data e hora; nenhuma é sobrescrita.

```bash
uv run python -m especialista.experimentos.etapa10d latencia
```

O ONNX de 32 bits da rede do zero não está na entrega da Etapa 8 (ela entregou só as formas de 8 bits). Ele é exportado de novo, do modelo que a Etapa 7 entregou, com o código da Etapa 8. Antes de medir, cada sistema responde aos 200 primeiros pedidos de `dev` e as respostas são comparadas com as gravadas: é a garantia de que o que foi medido é o modelo da tabela.
''')

code(r'''
medidas_de_latencia = etapa10d.latencias_medidas()
if not medida_nova:
    tabela_da_latencia, FIGURAS_DA_LATENCIA, latencia_confirma, medida = pd.DataFrame(), [], None, None
    leitura_da_latencia = ["**A latência não foi medida na mesma sessão.** As latências dos dois caminhos vêm de sessões diferentes (Etapas 8 e 10c), com o mesmo protocolo. Fica como limite."]
else:
    medida = medidas_de_latencia[-1]
    assert str(medida["pasta"].relative_to(BASE)) == CALC["latencia_na_mesma_sessao"]
    NOME_DA_FORMA = {"int8": "INT8", "onnx": "ONNX, 32 bits", "pytorch": "PyTorch, 32 bits"}
    origem_das_rodadas = {ZERO: pd.read_csv(ENTREGAS / "08_quantizacao" / "eficiencia.csv"), XLMR: eficiencia10c}
    FORMA_NA_ORIGEM = {"int8": None, "onnx": "fp32", "pytorch": "pytorch"}

    def rodadas_de_origem(l):
        """A menor e a maior mediana entre as rodadas da medida de origem do mesmo sistema."""
        filtro = dict(fichas[l["chave"]]["resultado"][1])
        e = etapa9.linha_gravada(origem_das_rodadas[l["caminho"]], **filtro)
        return e["p50_min_das_rodadas"], e["p50_max_das_rodadas"]

    LAT["confirma"] = LAT["diferenca_relativa"].abs() <= FOLGA
    tabela_da_latencia = pd.DataFrame([
        {"Sistema": nome(l["chave"]) + (" (modelo final)" if T.loc[l["chave"], "final"] else ""), "Forma": NOME_DA_FORMA[l["forma"]],
         "p50 de origem": f"{ms(l['p50_de_origem_ms'])} ({l['medida_de_origem']})", "p50 na mesma sessão": ms(l["p50_mesma_sessao_ms"]),
         "Entre as rodadas (mesma sessão)": f"{num(l['p50_min_das_rodadas'], 2)} a {num(l['p50_max_das_rodadas'], 2)} ms", "Mesma sessão ÷ origem": num(1 + l["diferenca_relativa"], 2),
         "Confirma a de origem": "sim" if l["confirma"] else "não", "Reproduz as respostas gravadas": "sim" if l["reproduz_as_respostas"] else "NÃO"}
        for _, l in LAT.iterrows()
    ])
    display(tabela_da_latencia.set_index(["Sistema", "Forma"]))
    FIGURAS_DA_LATENCIA = []
    for caminho in (ZERO, XLMR):
        do_caminho = LAT[LAT["caminho"] == caminho]
        fig, ax = visual.pontos_com_intervalo(
            {"etapa de origem": {NOME_DA_FORMA[l["forma"]]: (l["p50_de_origem_ms"], *rodadas_de_origem(l)) for _, l in do_caminho.iterrows()},
             "mesma sessão (esta etapa)": {NOME_DA_FORMA[l["forma"]]: (l["p50_mesma_sessao_ms"], l["p50_min_das_rodadas"], l["p50_max_das_rodadas"]) for _, l in do_caminho.iterrows()}},
            f"Latência do modelo final, {CAMINHOS[caminho]['nome']}", "latência p50 de uma consulta (ms); a barra vai da menor à maior mediana entre as 5 rodadas")
        # o padrão escreve com 1 casa; a latência precisa de 2. Os textos estão na ordem em que foram desenhados: a primeira série inteira, depois a segunda
        for anotacao, valor in zip(ax.texts, [*do_caminho["p50_de_origem_ms"], *do_caminho["p50_mesma_sessao_ms"]], strict=True):
            anotacao.set_text(num(valor, 2))
        visual.salvar(fig, PASTA, f"latencia_na_mesma_sessao_{caminho}")
        plt.show()
        FIGURAS_DA_LATENCIA.append(f"latencia_na_mesma_sessao_{caminho}")

    na = lambda caminho, forma: LAT[(LAT["caminho"] == caminho) & (LAT["forma"] == forma)].iloc[0]
    latencia_confirma = bool(LAT["confirma"].all())
    razao = lambda forma, coluna: na(XLMR, forma)[coluna] / na(ZERO, forma)[coluna]
    leitura_da_latencia = [
        f"**A medida.** Feita em {medida['medido_em'].replace('T', ' às ')}, com {medida['rodadas']} rodadas de {n(medida['consultas_por_rodada'])} consultas, 1 thread, os {len(LAT)} sistemas alternados em cada rodada. "
        f"Os outros processos ocuparam {num(medida['nucleos_de_outros'], 3)} núcleo de CPU no conjunto (o máximo numa rodada foi {num(max(medida['nucleos_de_outros_por_rodada']), 3)}; o limite é {num(medida['limite_de_outros'], 2)}): máquina quieta, {'sim' if medida['maquina_quieta'] else '**NÃO**'}. "
        f"Os {len(LAT)} sistemas reproduzem as respostas gravadas nos 200 primeiros pedidos de `dev`: {'sim' if LAT['reproduz_as_respostas'].all() else '**NÃO**'}.",
        f"**Os dois modelos finais, em INT8.** Na mesma sessão: {ms(na(ZERO, 'int8')['p50_mesma_sessao_ms'])} (rede do zero) e {ms(na(XLMR, 'int8')['p50_mesma_sessao_ms'])} (XLM-R), {num(razao('int8', 'p50_mesma_sessao_ms'), 1)} vezes. "
        f"Nas etapas de origem: {ms(na(ZERO, 'int8')['p50_de_origem_ms'])} e {ms(na(XLMR, 'int8')['p50_de_origem_ms'])}, {num(razao('int8', 'p50_de_origem_ms'), 1)} vezes.",
        f"**Em 32 bits.** ONNX: {ms(na(ZERO, 'onnx')['p50_mesma_sessao_ms'])} e {ms(na(XLMR, 'onnx')['p50_mesma_sessao_ms'])} ({num(razao('onnx', 'p50_mesma_sessao_ms'), 1)} vezes; na origem, {num(razao('onnx', 'p50_de_origem_ms'), 1)}). "
        f"PyTorch: {ms(na(ZERO, 'pytorch')['p50_mesma_sessao_ms'])} e {ms(na(XLMR, 'pytorch')['p50_mesma_sessao_ms'])} ({num(razao('pytorch', 'p50_mesma_sessao_ms'), 1)} vezes; na origem, {num(razao('pytorch', 'p50_de_origem_ms'), 1)}).",
        f"**Confirma a origem?** A regra: a diferença relativa entre a medida nova e a de origem é menor que a folga da fronteira ({num(100 * FOLGA, 0)}%). "
        + ("Vale para os " + str(len(LAT)) + " sistemas: a maior diferença é de " + num(100 * LAT["diferenca_relativa"].abs().max(), 1) + "%." if latencia_confirma
           else "Não vale para: " + lista(f"{nome(l['chave'])}, {NOME_DA_FORMA[l['forma']]} ({dif(l['diferenca_relativa'])}%)" for _, l in LAT[~LAT["confirma"]].iterrows()) + ".")
        + " Os números da tabela final continuam sendo os de origem; estes ficam ao lado.",
    ]
display(Markdown("**Leitura.**\n\n" + "\n".join(f"- {l}" for l in leitura_da_latencia)))
''')

md(r'''
## 10. Cada passo e o seu artigo

O quadro parte do da Etapa 9 e junta as referências que as Etapas 10a, 10b e 10c citam. Para cada passo do caminho: a técnica, o artigo, em que caminho foi usada e **se a referência foi conferida na fonte**.

A conferência foi feita nesta etapa, e é gratuita. O comando abaixo pergunta ao arXiv (API pública) o título, os autores e a data de cada identificador, busca os dois artigos da ACL Anthology na página de cada um, e grava a resposta inteira:

```bash
uv run python -m especialista.experimentos.etapa10d artigos
```

Há dois níveis, e o quadro diz qual vale para cada linha:

- **conferida**: o identificador existe na fonte, e o primeiro autor e o título que a fonte devolve são os que o projeto escreve;
- **resumo lido**: além disso, o resumo devolvido pelo arXiv foi lido nesta etapa. "sim" quer dizer que o resumo sustenta a técnica como o projeto a usa; "em parte", que o resumo confirma o assunto, mas o ponto exato que o projeto cita está no corpo do artigo.

**O corpo de nenhum artigo foi lido nesta etapa.** Os números publicados que o projeto cita (a linha do XLM-R base no MASSIVE, por exemplo) não foram conferidos de novo aqui.
''')

code(r'''
quadro = etapa10d.quadro_de_artigos()
com_artigo = quadro[quadro["na_fonte"] != "sem artigo"]
conferidas_na_fonte = int((com_artigo["na_fonte"] == "conferida").sum())
lidas = com_artigo[com_artigo["resumo_lido"]]
da_etapa_9 = {a.get("arxiv") for a in etapa9.ARTIGOS if a.get("arxiv")} | {"W04-3250", "D12-1091"}
novas = com_artigo[~com_artigo["identificador"].isin(da_etapa_9)]
fonte_dos_artigos = sorted((BASE / "artigos").glob("arxiv_*.json"))[-1]
tabela_de_artigos = pd.DataFrame({
    "Passo": quadro["passo"], "Caminho": quadro["caminho"], "Técnica": quadro["tecnica"],
    "Artigo": [q + (f", *{t}*" if t else "") for q, t in zip(quadro["quem"], quadro["titulo_na_fonte"].where(quadro["titulo_na_fonte"] != "", quadro["titulo_no_projeto"]))],
    "Ano": ["—" if pd.isna(a) else int(a) for a in quadro["ano"]],
    "Onde": [f"[{o}]({e})" if e else "—" for o, e in zip(quadro["onde"], quadro["endereco"])],
    "Estava citado em": [c or "—" for c in quadro["citado_em"]],
    "Conferida na fonte": quadro["na_fonte"],
    "Resumo lido": ["—" if s == "sem artigo" else (r if lido else "não") for s, lido, r in zip(quadro["na_fonte"], quadro["resumo_lido"], quadro["o_resumo_sustenta"])],
})
display(Markdown(em_markdown(tabela_de_artigos)))
em_parte = lidas[lidas["o_resumo_sustenta"] == "em parte"]
display(Markdown("**O que o resumo diz, nas linhas marcadas \"em parte\"**\n\n" + "\n".join(f"- {l['quem']} ({l['onde']}): {l['o_que_o_resumo_diz']}." for _, l in em_parte.iterrows())))
sem_artigo = quadro[quadro["na_fonte"] == "sem artigo"]
leitura_dos_artigos = (
    f"O quadro tem {len(quadro)} linhas: {len(com_artigo)} referências ({com_artigo['identificador'].nunique()} artigos diferentes) e {len(sem_artigo)} passo sem artigo ({lista(sem_artigo['passo'])}: é um passo de engenharia, e nenhuma etapa cita artigo para ele). "
    f"Das {len(com_artigo)} referências, {len(com_artigo) - len(novas)} já estavam no quadro da Etapa 9 e {len(novas)} vêm das Etapas 8, 10a, 10b e 10c. "
    f"**Conferidas na fonte: {conferidas_na_fonte} de {len(com_artigo)}** (identificador, primeiro autor e título, na consulta de {quadro['consultado_em'].iloc[0].replace('T', ' às ')}). "
    f"**Com o resumo lido: {len(lidas)}**; em {int((lidas['o_resumo_sustenta'] == 'sim').sum())} o resumo sustenta a técnica como o projeto a usa, e em {len(em_parte)} sustenta em parte. "
    f"As {len(com_artigo) - len(lidas)} sem resumo lido são as da ACL Anthology, de que só o título, os autores e o ano foram conferidos. "
    "Antes desta etapa, o `PLANO.md` registrava como lidos só dois resumos e parte de um texto, e a Etapa 9 tinha conferido título e autores de três referências."
)
display(Markdown(f"**Leitura.** {leitura_dos_artigos}"))
''')

md(r'''
## 11. Rótulos humanos em cada ponto e limites

O cenário do projeto promete **poucos rótulos humanos**: os que a LLM vê no prompt. A tabela diz, ponto por ponto dos dois caminhos, onde entrou rótulo humano, de treino ou de `dev`. Depois vêm os limites dos dois caminhos, juntos.
''')

code(r'''
ler_json = lambda *partes: json.loads(ENTREGAS.joinpath(*partes).read_text(encoding="utf-8"))
decisao6, decisao7, decisao8 = ler_json("06_destilacao", "decisao.json"), ler_json("07_poda", "decisao.json"), ler_json("08_quantizacao", "decisao.json")
d10a, d10b, d10c = ler_json("10a_encoder", "decisao.json"), ler_json("10b_encoder_poda", "decisao.json"), ler_json("10c_encoder_quantizacao", "decisao.json")
T_ZERO, T_XLMR = decisao6["temperatura_do_b2"], d10a["temperatura_do_b2"]
busca10a = pd.read_csv(ENTREGAS / "10a_encoder" / "busca.csv")
calibracao = d10c["calibracao_da_estatica"]["pedidos"]
piloto10b = (ENTREGAS / "10b_encoder_poda" / "piloto.json").exists()

rotulos_no_caminho = pd.DataFrame([
    ("Etapas 5a e 5b · as duas LLMs", "os dois", f"{n(POUCOS)} no prompt", "na LLM fechada, nenhuma escolha; na aberta, a escolha entre candidatos e o formato da resposta"),
    ("Etapas 4 e 4b · teto e receita", "do zero", f"{n(TODOS)}: o teto treina com todos", "época de parada e escolha da receita, que os alunos herdam"),
    ("Etapa 6 · alunos", "do zero", "nenhum: rótulos e logits das LLMs", f"a temperatura do B2 (T = {T_ZERO}) e a regra de quem segue para a poda"),
    ("Etapa 7 · poda e reajuste", "do zero", "nenhum: o reajuste usa o sinal do professor", "a regra do que segue para a quantização"),
    ("Etapa 8 · quantização", "do zero", "nenhum: a calibração usa pedidos do treino, sem rótulo", "a regra da forma de 8 bits"),
    ("Etapa 10a · teto, receita e alunos", "XLM-R", f"{n(TODOS)} no teto; nenhum nos alunos", f"a busca da receita do teto ({len(busca10a)} configurações, 1 semente), que os alunos herdam; a época de parada do teto e do piso; a temperatura do B2 (T = {T_XLMR})"),
    ("Etapa 10b · poda e reajuste", "XLM-R", "nenhum nos alunos: o reajuste usa o sinal do professor", "a regra do que segue para a quantização" + ("; o orçamento do reajuste saiu de um piloto no teto, visto em dev" if piloto10b else "")),
    ("Etapa 10c · quantização", "XLM-R", f"nenhum: a calibração usa {n(calibracao)} pedidos do treino, sem rótulo", "a regra da forma de 8 bits (a mesma da Etapa 8)"),
    ("Etapas 9 e 10d · comparativos", "os dois", "nenhum", "nenhuma decisão"),
], columns=["Ponto do caminho", "Caminho", f"Rótulos humanos de treino (de {n(TODOS)})", f"Decisões tomadas com os rótulos humanos de dev ({n(N_DEV)} pedidos)"])
display(rotulos_no_caminho.set_index("Ponto do caminho"))

# Os números que os limites citam, lidos das contas desta etapa e das etapas de origem
int8_x = lambda particao: cadeia(XLMR, "A", particao).set_index("degrau").loc[D["int8"]]
int8_z = lambda particao: cadeia(ZERO, "A", particao).set_index("degrau").loc[D["int8"]]
podado_b2_x = lambda particao: cadeia(XLMR, "B", particao).set_index("degrau").loc[D["podado"]]
estatica = pd.read_csv(ENTREGAS / "10c_encoder_quantizacao" / "comparacoes.csv")
estatica = estatica[(estatica["forma"] == "estatica") & (estatica["particao"] == "dev") & (estatica["metrica"] == "slot_f1")]
ef10b = pd.read_csv(ENTREGAS / "10b_encoder_poda" / "eficiencia.csv")
com_esparsa = ef10b[ef10b["latencia_esparsa_p50_ms"].notna()]
corte = pd.read_csv(ENTREGAS / "10b_encoder_poda" / "corte_do_vocabulario.csv")
aberturas_do_teste = sorted(p.parent.name for p in ENTREGAS.glob("*/resultados.csv") if "test" in set(pd.read_csv(p).get("particao", [])))
variante_final_x, variante_final_z = d10c["final"]["variante"], decisao8["final"]["variante"]
nome_da_poda = lambda decisao, modelo, variante: next(f["nome"] for f in decisao["modelos"][modelo]["seguem"] if f["variante"] == variante)
custo10a = ler_json("10a_encoder", "custo.json")

limites = [
    f"**Rótulos humanos de `dev` nas decisões (a ressalva dos {n(POUCOS)} rótulos).** O cenário promete {n(POUCOS)} rótulos humanos. No treino isso vale nos dois caminhos: nenhum aluno viu um rótulo humano de treino. Mas os {n(N_DEV)} rótulos humanos de `dev` decidiram, nos dois caminhos, a temperatura do B2, qual poda fica e a forma de 8 bits; no XLM-R, também a receita de ajuste fino (buscada no teto) e o orçamento do reajuste. "
    f"Num uso real com só {n(POUCOS)} rótulos, essas escolhas teriam de sair de outro critério (a concordância com o professor, por exemplo); isso só foi conferido para a temperatura da rede do zero (Etapa 9).",
    "**Os dois caminhos não diferem só no pré-treino.** Mudam juntos a arquitetura (12 camadas contra 2), o tamanho, a receita de treino, a cabeça de entidades (sem CRF contra com CRF) e o que a poda alcança. Cada diferença \"XLM-R menos rede do zero\" mede o pacote, não uma peça.",
    f"**A poda principal não é a mesma técnica nos dois caminhos.** Na rede do zero, \"{nome_da_poda(decisao7, decisao8['final']['modelo'], variante_final_z)}\" (as linhas da tabela de features); no XLM-R, \"{nome_da_poda(d10b, d10c['final']['modelo'], variante_final_x)}\" (as linhas da tabela de subtokens). Cada uma saiu da regra da sua etapa, em `dev`. "
    f"O orçamento do reajuste também difere: {d10b['reajuste']['epocas']} épocas no XLM-R (o treino inteiro), e uma fração do treino na rede do zero (Etapa 7).",
    f"**A temperatura do B2 difere entre os caminhos.** T = {T_ZERO} na rede do zero e T = {T_XLMR} no XLM-R: cada etapa escolheu a sua em `dev`. No XLM-R a escolhida não se distingue das outras três (Etapa 10a). O B1 da rede do zero tem suavização de rótulos; o do XLM-R, não. E o piso não foi treinado do mesmo jeito nos dois caminhos (Etapa 10a).",
    f"**A quantização do XLM-R mexe na intenção do modelo final, e a regra só olhava entidades.** No degrau INT8 do modelo final do XLM-R, a acurácia de intenção muda {com_intervalo(int8_x('dev'), 'intent_acc_', 2)} em `dev` e {com_intervalo(int8_x('test'), 'intent_acc_', 2)} no teste: distinguível nas duas. "
    f"Na rede do zero, o mesmo degrau: {com_intervalo(int8_z('dev'), 'intent_acc_', 2)} e {com_intervalo(int8_z('test'), 'intent_acc_', 2)}. A regra das Etapas 8 e 10c decide pelo F1 de entidades e foi escrita antes dos resultados; não foi mudada.",
    f"**A quantização estática foi testada com uma calibração só.** {n(calibracao)} pedidos do treino e a faixa do menor ao maior valor visto. No XLM-R ela não passou em nenhum dos {len(estatica)} modelos (o F1 de entidades cai de {num(-100 * estatica['diferenca'].max(), 1)} a {num(-100 * estatica['diferenca'].min(), 1)} pontos em `dev`). Outras calibrações (percentis, entropia) e o treino ciente de quantização não foram testados: a regra não os pediu.",
    f"**A poda do B2 no XLM-R não teve grade própria.** O aluno B2 recebeu as variantes escolhidas para o aluno A (Etapa 10b). A principal passa em `dev` ({com_intervalo(podado_b2_x('dev'), 'exact_match_')}) e fica {posicao(podado_b2_x('test')['exact_match_dif_ic_min'], podado_b2_x('test')['exact_match_dif_ic_max'])} no teste ({com_intervalo(podado_b2_x('test'), 'exact_match_')}).",
    f"**`dev` e teste discordam em parte das escolhas.** A regra das podas aceita o que não fica distinguivelmente abaixo em `dev`; com o tamanho de `dev`, uma queda de menos de cerca de 1 ponto pode passar. Aconteceu nos dois caminhos (Etapas 7, 9 e 10b). Nesta etapa, `dev` e teste discordam em {len(discordam(todos_os_pares))} dos {total_de_intervalos} intervalos da lista fechada.",
    f"**A execução esparsa só foi medida no aluno A.** No XLM-R, {len(com_esparsa)} variantes de pesos soltos do aluno A têm latência com a execução que só faz as contas dos pesos não nulos (Etapa 10b); nos outros modelos, não. Em nenhum dos dois caminhos os pesos soltos renderam latência no executor comum.",
    f"**O corte do vocabulário depende do texto do treino.** O modelo final do XLM-R só conhece os {n(d10c['final']['linhas_da_tabela'])} subtokens que o treino usa. Palavra nova é partida em peças menores ou vira `<unk>`: em `dev`, {num(100 * corte.set_index('forma').loc['treino', 'dev_pedidos_com_corte'], 1)}% dos pedidos têm algum subtoken trocado (Etapa 10b). Fora do domínio do MASSIVE isso não foi medido." if "dev_pedidos_com_corte" in corte.columns
    else f"**O corte do vocabulário depende do texto do treino.** O modelo final do XLM-R só conhece os {n(d10c['final']['linhas_da_tabela'])} subtokens que o treino usa. Palavra nova é partida em peças menores ou vira `<unk>` (Etapa 10b). Fora do domínio do MASSIVE isso não foi medido.",
    f"**A latência é de uma CPU, de uma semente e de pedidos curtos.** {maquina['cpu']}, {'com' if maquina['vnni'] else 'sem'} as instruções VNNI, 1 thread, semente 0, pedidos de poucas palavras. A mediana variou até {num(100 * FOLGA, 0)}% entre as rodadas das medidas de origem. Em outra CPU, sem VNNI, as respostas dos modelos em 8 bits podem mudar um pouco. Com mais threads, lotes ou textos longos, a razão entre os caminhos seria outra.",
    ("**As latências dos dois caminhos vêm de sessões diferentes.** " + (f"A medida na mesma sessão (seção 9) {'confirma' if latencia_confirma else 'não confirma todas'} as de origem pela regra da etapa, mas é uma medida só, de um dia." if medida_nova else "Não foi possível medir na mesma sessão nesta etapa.")),
    f"**O teste foi aberto mais de uma vez.** Ele aparece nos resultados de {len(aberturas_do_teste)} entregas, desde a Etapa 2. Nenhuma decisão saiu dele, mas são os mesmos {n(N_TESTE)} pedidos medindo tudo, e esta etapa soma {total_de_intervalos} intervalos de comparação por partição, mais os dos degraus, sem correção para comparações múltiplas.",
    "**Parte das comparações já era conhecida quando as regras foram escritas.** As dos modelos sem poda (Etapa 10a) e todas as do caminho do zero (Etapa 9). As regras garantem que a lista não foi escolhida depois das contas novas, não que quem as escreveu não conhecia nenhum número.",
    "**A fronteira usa os valores medidos**, sem os intervalos, e inclui o INT8 sem poda, que não é o modelo escolhido pela regra de nenhum caminho. Serve para ver a ordem de grandeza do custo, não para separar vizinhos.",
    "**O preço de cada ponto é uma razão entre dois modelos**, não uma curva: só há dois pontos, um por caminho. O intervalo vem só da incerteza da qualidade.",
    f"**O custo de treino não está na tabela.** A tabela mede o custo de usar. Treinar e reajustar o XLM-R pede placa de vídeo ({num(custo10a.get('horas_de_treino_na_etapa', float('nan')), 1)} horas de treino na Etapa 10a e mais as da 10b); a rede do zero treina em minutos. A memória em uso durante a consulta também não foi medida no XLM-R." if "horas_de_treino_na_etapa" in custo10a
    else "**O custo de treino não está na tabela.** A tabela mede o custo de usar. Treinar e reajustar o XLM-R pede horas de placa de vídeo (Etapas 10a e 10b); a rede do zero treina em minutos. A memória em uso durante a consulta também não foi medida no XLM-R.",
    "**Um conjunto, uma língua, uma rotulagem.** MASSIVE, recorte pt-PT; uma rotulagem do treino por professor; 3 sementes por modelo treinado. As LLMs podem conhecer o MASSIVE, que é público. O XLM-R foi pré-treinado em cem idiomas e aqui só vê pt-PT.",
    f"**As referências foram conferidas pelo identificador, pelo primeiro autor, pelo título e pelo resumo.** O corpo dos artigos não foi lido nesta etapa; em {len(em_parte)} referências o resumo sustenta só em parte o uso que o projeto faz (seção 10).",
    "**O que as etapas de origem já declararam continua valendo**: os limites de cada uma estão no seu `RESUMO.md` (o professor aberto é o que coube na placa; a temperatura do B2 da rede do zero está na borda da grade; a busca da receita do XLM-R tem 1 semente; o corte por matriz e a poda de movimento não testados; um degrau por tensor na quantização).",
]
display(Markdown("### Limites\n\n" + "\n".join(f"- {l}" for l in limites)))
''')

md(r'''
## 12. Entrega
''')

code(r'''
# 1. As tabelas da etapa, como o cálculo as gravou, e as tabelas de leitura
for arquivo in ("tabela_final.csv", "cadeias.csv", "comparacoes.csv", "conferencia.csv", "fronteira.csv", "precos.csv", "latencia_ao_lado.csv", "calculo.json"):
    shutil.copy2(BASE / "resultados" / arquivo, PASTA / arquivo)
quadro.to_csv(PASTA / "artigos.csv", index=False)
origem.to_csv(PASTA / "registro.csv", index=False)
SEM_INT8_SEM_PODA.to_csv(PASTA / "fronteira_sem_int8_sem_poda_exploratoria.csv", index=False)

# 2. As regras (todas as versões), o que foi produzido agora, a consulta das referências e as medidas de latência
for caminho in sorted(BASE.glob("regras*.json")):
    shutil.copy2(caminho, PASTA / caminho.name)
shutil.copy2(BASE / "medidas.json", PASTA / "medidas_novas.json")
(PASTA / "artigos_na_fonte").mkdir(exist_ok=True)
shutil.copy2(fonte_dos_artigos, PASTA / "artigos_na_fonte" / fonte_dos_artigos.name)
(PASTA / "previsoes").mkdir(exist_ok=True)
for chave in geradas:
    for lista_de_arquivos in fichas[chave]["previsoes"].values():
        for caminho in lista_de_arquivos:
            if caminho.is_relative_to(BASE):
                shutil.copy2(caminho, PASTA / "previsoes" / caminho.name)
for m in medidas_de_latencia:
    destino = PASTA / "latencia" / m["pasta"].name
    destino.mkdir(parents=True, exist_ok=True)
    for arquivo in ("latencia.json", "latencia.npz"):
        shutil.copy2(m["pasta"] / arquivo, destino / arquivo)

# 3. O código usado, copiado (o projeto não é um repositório git)
CODIGO = ["src/especialista/experimentos/etapa10d.py", "src/especialista/experimentos/etapa9.py", "src/especialista/avaliacao/qualidade.py", "src/especialista/avaliacao/eficiencia.py",
          "src/especialista/visual.py", "notebooks/construtores/build_nb10d.py", "tests/test_etapa10d.py"]
(PASTA / "codigo").mkdir(exist_ok=True)
for arquivo in CODIGO:
    shutil.copy2(RAIZ / arquivo, PASTA / "codigo" / Path(arquivo).name)

FIGURAS = ["fronteira_latencia", "fronteira_tamanho", *FIGURAS_DAS_CADEIAS, *FIGURAS_ENTRE.values(), "aluno_menos_professor_exact_match", "ponta_a_ponta_exact_match", *FIGURAS_DA_LATENCIA]
print("Figuras:", ", ".join(f for f in FIGURAS if (PASTA / "figuras" / f"{f}.png").exists()))
''')

code(r'''
# 4. A conclusão em dados: o que a etapa seguinte (ou o artigo) lê sem abrir o notebook
tres_de = lambda b, a, particao: {m: {k: float(comp(b, a, m, particao)[k]) for k in ("diferenca", "ic_min", "ic_max")} | {"distinguivel": bool(comp(b, a, m, particao)["distinguivel"])} for m in TRES}
custo_de = lambda chave: {"bytes": float(T.loc[chave, "bytes"]), "latencia_p50_ms": float(T.loc[chave, "latencia_p50_ms"]), "latencia_medida_na": T.loc[chave, "latencia_medida_na"], "parametros": int(T.loc[chave, "parametros"])}
conclusao = {
    "etapa": etapa10d.ETAPA, "gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"), "pergunta": etapa10d.PERGUNTA,
    "finais": {caminho: {"chave": chave, "sistema": T.loc[chave, "sistema"], "no_teste": {m: float(T.loc[chave, m]) for m in TRES}, "em_dev": {m: float(T.loc[chave, f"dev_{m}"]) for m in TRES}, **custo_de(chave)} for caminho, chave in FINAIS.items()},
    "final_do_xlmr_menos_final_do_zero": {"test": tres_de(FX, FZ, "test"), "dev": tres_de(FX, FZ, "dev")},
    "custo_do_final_do_xlmr_em_vezes_o_do_zero": {"tamanho": float(T.loc[FX, "bytes"] / T.loc[FZ, "bytes"]), "latencia_de_origem": float(T.loc[FX, "latencia_p50_ms"] / T.loc[FZ, "latencia_p50_ms"]),
                                                   "latencia_na_mesma_sessao": float(razao("int8", "p50_mesma_sessao_ms")) if medida_nova else None},
    "preco_de_um_ponto_de_exact_match": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in l.items()} for l in PRECOS.to_dict("records")],
    "fronteira": {"latencia": fronteira_lat, "tamanho": fronteira_tam, "latencia_com_a_medida_na_mesma_sessao": fronteira_lat_nova,
                  "dominados": {custo: {c: l["dominado_por"].split("; ") for c, l in front(custo).iterrows() if not l["na_fronteira"]} for custo in ("latencia", "tamanho")}, "folga_da_latencia": FOLGA},
    "latencia_na_mesma_sessao": None if not medida_nova else {"pasta": CALC["latencia_na_mesma_sessao"], "medido_em": medida["medido_em"], "maquina_quieta": medida["maquina_quieta"], "nucleos_de_outros": medida["nucleos_de_outros"],
                                                              "confirma_as_de_origem": latencia_confirma, "p50_ms": {f"{l['caminho']}/{l['forma']}": {"mesma_sessao": float(l["p50_mesma_sessao_ms"]), "origem": float(l["p50_de_origem_ms"])} for _, l in LAT.iterrows()}},
    "conferencia": {"linhas_conferidas": int(len(conferidas)), "diferentes": int(len(diferentes)), "sem_numero_na_origem": int(len(sem_origem)), "maior_diferenca": maior, "por_tipo": {k: int(v) for k, v in por_tipo.items()}},
    "comparacoes": {"na_lista_fechada": len(todos_os_pares), "intervalos_por_particao": total_de_intervalos, "distinguiveis_no_teste": int(distinguiveis_no_teste), "distinguiveis_em_dev": int(distinguiveis_em_dev),
                    "dev_e_teste_discordam": len(discordam(todos_os_pares)), "degraus_de_cadeia_por_particao": degraus_por_particao},
    "artigos": {"referencias": int(len(com_artigo)), "conferidas_na_fonte": conferidas_na_fonte, "com_resumo_lido": int(len(lidas)), "resumo_sustenta": int((lidas["o_resumo_sustenta"] == "sim").sum()),
                "resumo_sustenta_em_parte": int(len(em_parte)), "passos_sem_artigo": list(sem_artigo["passo"])},
    "regras": CALC["regras"], "sistemas_no_registro": len(fichas), "limites": len(limites),
    "como_acrescentar_um_sistema": "uma função nova em etapa10d.FONTES, ou um sistemas.json na entrega da etapa nova (formato em etapa10d.fichas_declaradas); depois: regras, preparar, calcular",
}
(PASTA / "conclusao.json").write_text(json.dumps(conclusao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# 5. O manifesto: de que arquivos a etapa leu, com que código, em que máquina
from especialista.experimentos import etapa8
ETAPAS_DE_ORIGEM = sorted({f["resultado"][0] for f in fichas.values()} | {fonte[0] for f in fichas.values() for fonte in f.get("outras_fontes", [])} | {"07_poda", "09_comparativo", "10b_encoder_poda", "01_dados"})
lidos_das_entregas = {}
for etapa in ETAPAS_DE_ORIGEM:
    for arquivo in ("decisao.json", "resultados.csv", "comparacoes.csv", "eficiencia.csv", "tabela_final.csv", "medidas_novas.json", "manifesto.json", "configuracao.json", "escolha.json"):
        if (ENTREGAS / etapa / arquivo).exists():
            lidos_das_entregas[f"entregas/{etapa}/{arquivo}"] = etapa10d.sha256(ENTREGAS / etapa / arquivo)
versao = lambda pacote: importlib.metadata.version(pacote)
manifesto = {
    "etapa": etapa10d.ETAPA, "gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
    "o_que_e": "De que arquivos a Etapa 10d leu (com a impressão digital de cada um), as regras, o que ela produziu, o código usado (copiado em codigo/), as versões das bibliotecas e a máquina.",
    "nada_foi_treinado": True,
    "entradas": {"tabelas_e_decisoes": lidos_das_entregas, "respostas": CALC["respostas_lidas"]},
    "regras": [{"arquivo": c.name, "versao": json.loads(c.read_text(encoding="utf-8"))["versao"], "gravado_em": json.loads(c.read_text(encoding="utf-8"))["gravado_em"], "sha256": etapa10d.sha256(c)} for c in sorted(BASE.glob("regras*.json"))],
    "calculo": {"calculado_em": CALC["calculado_em"], "com_as_regras": CALC["regras"]},
    "produzido_nesta_etapa": {"respostas": medidas["respostas_geradas"], "latencia": [{"pasta": f"latencia/{m['pasta'].name}", "medido_em": m["medido_em"], "maquina_quieta": m["maquina_quieta"], "nucleos_de_outros": m["nucleos_de_outros"]} for m in medidas_de_latencia],
                              "consulta_das_referencias": f"artigos_na_fonte/{fonte_dos_artigos.name}"},
    "saidas": {str(c.relative_to(PASTA)): etapa10d.sha256(c) for c in sorted(PASTA.glob("*.csv")) + sorted(PASTA.glob("regras*.json")) + [PASTA / "conclusao.json", PASTA / "calculo.json"]},
    "codigo": {"pasta": "codigo/", "sha256": {arquivo: etapa10d.sha256(RAIZ / arquivo) for arquivo in CODIGO}},
    "versoes": {pacote: versao(pacote) for pacote in ("numpy", "pandas", "onnxruntime", "onnx", "torch", "tokenizers", "transformers", "scikit-learn", "matplotlib")},
    "maquina": etapa8.ambiente(),
    "cache": "data/execucoes/10d_comparativo_dos_caminhos",
}
(PASTA / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Arquivos gravados:", ", ".join(sorted(c.name for c in PASTA.iterdir() if c.is_file())))
''')

md(r'''
### Critério de pronto
''')

code(r'''
da_9 = pd.read_csv(ENTREGAS / "09_comparativo" / "tabela_final.csv")
linhas_da_9_iguais = bool(len(CONF[CONF["o_que"] == LINHA_DA_9]) == len(da_9) and (CONF.loc[CONF["o_que"] == LINHA_DA_9, "confere"] == True).all())   # noqa: E712
colunas_de_qualidade = [f"{prefixo}{m}{s}" for prefixo in ("", "dev_") for m in qualidade.METRICAS for s in ("", "_ic_min", "_ic_max")]
qualidade_completa = bool(T[colunas_de_qualidade].notna().all().all())
esperados_do_xlmr = [f"{P[XLMR]}{m}" for m in ("teto", "aluno_a", "aluno_b2")] + [f"{P[XLMR]}{m}/{p}" for m in ("teto", "aluno_a", "aluno_b2") for p in ("podado", "onnx", "int8")]
cadeias_completas = bool(CAD[["exact_match", "latencia_p50_ms"]].notna().all().all() and passos[[f"{m}_{c}" for m in qualidade.METRICAS for c in ("diferenca", "dif_ic_min", "dif_ic_max")]].notna().all().all()
                         and sorted(dict.fromkeys(CAD["cadeia"])) == sorted(f"{c}/{cen}" for c in (ZERO, XLMR) for cen in CENARIOS))
degraus_conferidos = CONF[(CONF["o_que"] == DEGRAU) & CONF["gravado"].astype(bool)]
preco_pela_regra = bool((PRECOS["definido"] == (PRECOS["exact_match_ic_min"] > 0)).all() and set(PRECOS["particao"]) == {"dev", "test"})
pronto = pd.Series({
    f"As regras foram gravadas antes das contas, e as contas são exatamente as das regras ({len(regras['comparacoes'])} comparações, {len(regras['cadeias'])} cadeias; nenhuma acrescentada depois)": bool(regras_antes and contas_sao_as_das_regras and mesmos_pares),
    f"A tabela tem as {len(da_9)} linhas da Etapa 9, idênticas, e as do XLM-R (teto, alunos, podados, ONNX de 32 bits e INT8), com qualidade e intervalo em dev e no teste ({len(T)} linhas)":
        bool(linhas_da_9_iguais and all(c in T.index for c in esperados_do_xlmr) and qualidade_completa),
    f"Cada número que uma etapa de origem gravou foi refeito das respostas ({n(len(conferidas))} linhas conferidas; {len(diferentes)} diferentes; {len(sem_origem)} sem número de origem), e o que não bate está relatado":
        bool(len(conferidas) > 0 and (len(diferentes) == 0 or "DIFERENTE" in set(CONF["situacao"]))),
    f"Cada cadeia tem os degraus dos dois caminhos, com qualidade, tamanho, latência e a diferença pareada para o anterior, em dev e no teste ({len(regras['cadeias'])} cadeias; {len(degraus_conferidos)} degraus conferidos com a origem)": cadeias_completas,
    "A fronteira (com quem domina quem) e o preço de cada ponto saem pelas regras, com a definição de dominado da Etapa 9": bool(len(fronteira_lat) > 0 and len(fronteira_tam) > 0 and preco_pela_regra),
    ("A latência dos dois modelos finais foi medida na mesma sessão, no protocolo das Etapas 8 e 10c, com a máquina quieta e reproduzindo as respostas gravadas" if medida_nova else "A latência não foi medida na mesma sessão, e isso está nos limites"):
        bool(medida_nova and medida["maquina_quieta"] and LAT["reproduz_as_respostas"].all() and medida["rodadas"] == 5 and medida["consultas_por_rodada"] == 1024 and medida["threads"] == 1) if medida_nova else True,
    f"Cada referência do quadro diz se foi conferida na fonte ({conferidas_na_fonte} de {len(com_artigo)} conferidas; {len(lidas)} com o resumo lido)": bool(set(quadro["na_fonte"]) <= {"conferida", "não confere", "não conferida", "sem artigo"} and len(com_artigo) >= len(da_etapa_9)),
    f"Os limites dos dois caminhos estão juntos ({len(limites)} limites)": bool(len(limites) >= 12),
    f"As figuras foram gravadas ({len(FIGURAS)})": all((PASTA / "figuras" / f"{f}.png").exists() for f in FIGURAS),
    "Nenhuma rede foi treinada; a entrega tem manifesto, conclusão em dados e cópia do código": bool((PASTA / "manifesto.json").exists() and (PASTA / "conclusao.json").exists() and all((PASTA / "codigo" / Path(a).name).exists() for a in CODIGO)),
}).map({True: "sim", False: "NÃO"}).rename("atendido").to_frame()
display(pronto)
assert (pronto["atendido"] == "sim").all()
''')

md(r'''
## 13. Resposta

O resumo abaixo é gerado a partir dos números lidos e calculados neste notebook e gravado em `entregas/10d_comparativo_dos_caminhos/RESUMO.md`.
''')

code(r'''
onde_nasce = comp(P[XLMR] + "aluno_a", "aluno_a")
resposta = (
    f"No teste, o modelo final do XLM-R ({T.loc[FX, 'sistema']}) faz {tres(fx)} (acurácia de intenção / F1 de entidades / exact match) e o da rede do zero ({T.loc[FZ, 'sistema']}), {tres(fz)}. "
    f"O XLM-R fica {frase(FX, FZ)}; em `dev`, {frase(FX, FZ, 'dev')}. "
    f"**O que ele custa:** {tamanho(fx['bytes'])} contra {tamanho(fz['bytes'])} ({vezes(fx['bytes'], fz['bytes'])} vezes o tamanho) e {ms(fx['latencia_p50_ms'])} contra {ms(fz['latencia_p50_ms'])} por consulta ({vezes(fx['latencia_p50_ms'], fz['latencia_p50_ms'])} vezes a latência, 1 thread de CPU"
    + (f"; medidos na mesma sessão, {ms(na(XLMR, 'int8')['p50_mesma_sessao_ms'])} contra {ms(na(ZERO, 'int8')['p50_mesma_sessao_ms'])}, {num(razao('int8', 'p50_mesma_sessao_ms'), 1)} vezes" if medida_nova else "") + "). "
    f"**O preço de cada ponto de exact match**, entre os dois modelos finais: {texto_do_preco(p_mb, 1)} e {texto_do_preco(p_ms, 2)} no teste; em `dev`, {texto_do_preco(p_mb_dev, 1)} e {texto_do_preco(p_ms_dev, 2)}. "
    f"**De onde a diferença vem:** ela já existe quando o aluno sai da destilação ({com_intervalo(onde_nasce)} de exact match no aluno A sem poda) e atravessa a compressão: {trajeto('aluno_a')}. "
    f"Contra o professor ({PROFESSOR}), o final do XLM-R fica {frase(*px['a'])}; o da rede do zero, {frase(*pz['a'])}. "
    f"**Na fronteira** de qualidade contra latência, entre os sistemas com {n(POUCOS)} rótulos humanos, ficam: {nomes_com_valor(fronteira_lat)}; na de tamanho: {nomes_com_valor(fronteira_tam)}. "
    f"O final da rede do zero {situacao(f_lat, FZ)} (latência) e {situacao(f_tam, FZ)} (tamanho); o final do XLM-R {situacao(f_lat, FX)} (latência) e {situacao(f_tam, FX)} (tamanho). "
    f"Na latência a poda não muda o custo, e o final de cada caminho empata com o seu próprio INT8 sem poda: a diferença de exact match entre os dois é {com_intervalo(poda8_z, casas=2)} na rede do zero e {com_intervalo(poda8_x)} no XLM-R (em `dev`, {com_intervalo(comp(FX, sem_poda_x, 'exact_match', 'dev'))}). "
    + (f"**Nenhum caminho domina o outro**: nenhum sistema de um caminho domina o modelo final do outro. A rede do zero é o ponto barato, o XLM-R é o ponto de mais qualidade, e a escolha depende de quanto valem {dif(p_mb['exact_match_a_mais'])} pontos de exact match diante de {vezes(fx['bytes'], fz['bytes'])} vezes o arquivo e {vezes(fx['latencia_p50_ms'], fz['latencia_p50_ms'])} vezes a latência."
       if ninguem_cruza else "Há sistema de um caminho que domina o modelo final do outro: ver a seção da fronteira.")
)

colunas_do_resumo = ["Caminho", "Sistema", "Rótulos humanos", "Acurácia de intenção", "F1 de entidades", "Exact match", "Parâmetros", "Tamanho em disco", "Com zip", "Latência p50", "Onde a latência foi medida", "Consultas por segundo", "US$ por 1.000 consultas"]
tabela_pronto = "\n".join(f"| {criterio} | {atendido} |" for criterio, atendido in pronto["atendido"].items())
secoes_das_cadeias = "\n\n".join(
    f"### {CENARIOS[cen]}\n\n{em_markdown(lado_a_lado(cen))}\n\n![{CENARIOS[cen]}, lado a lado](figuras/cadeias_lado_a_lado_{ARQUIVO_DO_CENARIO[cen]}.png)\n\n"
    f"**{CAMINHOS[XLMR]['nome']}, degrau a degrau**\n\n{em_markdown(tabela_da_cadeia(XLMR, cen))}\n\n![{CAMINHOS[XLMR]['curto']} · {CENARIOS[cen]}](figuras/cadeia_xlmr_{ARQUIVO_DO_CENARIO[cen]}.png)\n\n"
    + "\n".join(f"- {f}" for f in leituras_das_cadeias[(XLMR, cen)])
    + f"\n\n**{CAMINHOS[ZERO]['nome']}, degrau a degrau** (os números são os da Etapa 9)\n\n{em_markdown(tabela_da_cadeia(ZERO, cen))}\n\n" + "\n".join(f"- {f}" for f in leituras_das_cadeias[(ZERO, cen)])
    for cen in CENARIOS
)
figuras_da_latencia = "\n\n".join(f"![Latência na mesma sessão, {CAMINHOS[c]['curto']}](figuras/latencia_na_mesma_sessao_{c}.png)" for c in (ZERO, XLMR)) if medida_nova else ""
pasta_da_latencia = f"`latencia/{medida['pasta'].name}/`" if medida_nova else "(não houve)"

resumo = f"""# Etapa 10d · Comparativo dos dois caminhos: resumo da entrega

**Pergunta.** {etapa10d.PERGUNTA}

**Resposta.** {resposta}

Nenhuma rede foi treinada nesta etapa. A linha de base com 299 rótulos, que não é rede neural, foi treinada de novo para gerar as respostas que a Etapa 5a não gravou, e os modelos em INT8 da rede do zero responderam de novo os pedidos; em todos os casos os números batem com os das etapas de origem. As regras foram gravadas antes das contas ({versoes[0]['gravado_em'].replace('T', ' às ')}; contas em {CALC['calculado_em'].replace('T', ' às ')}). Todos os números de qualidade vêm das respostas gravadas pelas Etapas 2 a 10c e foram refeitos: {n(len(conferidas))} linhas conferidas contra o que as etapas de origem gravaram, {"todas iguais" if not len(diferentes) else f"{len(diferentes)} diferentes (relatadas abaixo)"} (maior diferença: {maior:.1e}).

## As regras, escritas antes

- **Método:** nenhuma decisão nesta etapa; `dev` e teste, contra os rótulos humanos, com a leitura principal no teste; média de {SEMENTES} sementes; diferenças pareadas, bootstrap por enunciado, {n(etapa10d.REAMOSTRAS)} reamostragens, intervalo de 95%. Uma diferença só vale quando o intervalo não contém zero.
- **Lista fechada:** {len(regras['comparacoes'])} comparações, geradas por regra a partir das marcas do registro ({lista(f"{NOME_DA_FAMILIA[f].lower()}: {q}" for f, q in familias.items())}; {em_duas} caem em duas famílias), e {len(regras['cadeias'])} cadeias, com {degraus_por_particao} degraus. Primárias: o final de um caminho contra o do outro, nas três métricas, no teste.
- **Dominado** (a definição da Etapa 9): existe outro sistema com custo menor ou igual e exact match maior ou igual, e melhor em pelo menos um dos dois. Concorrem os sistemas com {n(POUCOS)} rótulos humanos, dos dois caminhos. Latências a menos de {num(100 * FOLGA, 0)}% uma da outra são empate.
- **Preço de cada ponto:** custo a mais ÷ pontos de exact match a mais, entre os dois modelos finais; o intervalo vem do intervalo da diferença de exact match e só existe quando ele está todo acima de zero.
- `regras.json` traz o texto inteiro{f", e `regras_v{regras['versao']}.json`, a versão em uso (só corrige o texto que descreve cada comparação; os pares são os mesmos)" if len(versoes) > 1 else ""}. O que já era conhecido ao escrever está declarado no arquivo e nos limites.

## A conferência

{em_markdown(resumo_da_conferencia)}

- {n(por_tipo[QUALIDADE])} linhas de qualidade ({len(fichas)} sistemas, `dev` e teste, contra cada etapa que gravou o número), {n(por_tipo[DEGRAU])} degraus de cadeia, {n(por_tipo[COMPARACAO])} comparações entre sistemas (as da Etapa 10a entre caminhos e as de ponta a ponta da Etapa 9) e as {n(por_tipo[LINHA_DA_9])} linhas da tabela final da Etapa 9, coluna por coluna.
- {"Nenhum número de etapa anterior deixou de fechar com as respostas gravadas." if not len(diferentes) else "**Há números que não fecham:** " + "; ".join(f"{l['chave']} ({l['particao']}, {l['gravado_em']})" for _, l in diferentes.iterrows())}

## A tabela final (teste, {n(N_TESTE)} pedidos)

Qualidade em % (intervalo de 95%); nos sistemas treinados, média de {SEMENTES} sementes. Tamanho e latência dos modelos comprimidos são os da semente 0. A mesma tabela com a qualidade em `dev` está em `tabela_final.csv` (colunas `dev_...`).

{em_markdown(mostra[colunas_do_resumo])}

{chr(10).join(f"- {nota}" for nota in notas_da_tabela)}

## As cadeias, lado a lado

Em cada cadeia, de cima para baixo, muda uma coisa por degrau. As diferenças são pareadas, em pontos percentuais (intervalo de 95%), de cada degrau para o anterior, no teste.

{secoes_das_cadeias}

**Nas seis cadeias.**

{chr(10).join(f"- {l}" for l in leitura_das_cadeias)}

## Os dois caminhos no mesmo ponto

XLM-R menos rede do zero, o mesmo modelo no mesmo ponto da cadeia. Diferença pareada, em pontos percentuais (intervalo de 95%).

**No teste**

{em_markdown(tabelas_entre['test'])}

**Em dev**

{em_markdown(tabelas_entre['dev'])}

![Entre caminhos, exact match](figuras/entre_caminhos_exact_match.png)

![Entre caminhos, intenção](figuras/entre_caminhos_intencao.png)

![Entre caminhos, entidades](figuras/entre_caminhos_entidades.png)

{chr(10).join(f"- {l}" for l in leitura_entre)}

## Cada aluno contra o seu professor

**No teste**

{em_markdown(tabelas_professor['test'])}

**Em dev**

{em_markdown(tabelas_professor['dev'])}

![Aluno menos professor](figuras/aluno_menos_professor_exact_match.png)

## De ponta a ponta, nos dois caminhos

As dez comparações da Etapa 9, B − A, com os sistemas de cada caminho.

**No teste**

{em_markdown(tabelas_pp['test'])}

**Em dev**

{em_markdown(tabelas_pp['dev'])}

![De ponta a ponta](figuras/ponta_a_ponta_exact_match.png)

**A poda dentro dos 8 bits** (podado em INT8 menos sem poda em INT8), no teste e em dev:

{em_markdown(tabelas_poda8['test'])}

{em_markdown(tabelas_poda8['dev'])}

{chr(10).join(f"- {l}" for l in leituras_pp)}

## A fronteira e o preço de cada ponto

![Qualidade contra latência](figuras/fronteira_latencia.png)

{em_markdown(tabelas_da_fronteira['latencia'])}

![Qualidade contra tamanho](figuras/fronteira_tamanho.png)

{em_markdown(tabelas_da_fronteira['tamanho'])}

**O preço de cada ponto de exact match, entre os dois modelos finais**

{em_markdown(tabela_dos_precos)}

{chr(10).join(f"- {l}" for l in leitura_da_fronteira)}

## A latência na mesma sessão

{em_markdown(tabela_da_latencia) if medida_nova else ""}

{figuras_da_latencia}

{chr(10).join(f"- {l}" for l in leitura_da_latencia)}
- O protocolo das Etapas 8 e 10c é o mesmo (1 thread, uma consulta por vez, do texto à resposta, pedidos de `dev`, 50 consultas de aquecimento, 5 rodadas de 1.024 consultas, sistemas alternados, a mesma função). O que as medidas de origem não tinham era os dois caminhos na mesma rodada.

## Cada passo e o seu artigo

{em_markdown(tabela_de_artigos)}

- {leitura_dos_artigos}
- O que o resumo diz, nas linhas marcadas "em parte": {" ".join(f"{l['quem']} ({l['onde']}): {l['o_que_o_resumo_diz']}." for _, l in em_parte.iterrows())}
- **O corpo de nenhum artigo foi lido nesta etapa**, e os números publicados que o projeto cita não foram conferidos de novo aqui. A resposta inteira da fonte está em `artigos_na_fonte/`.

## Rótulos humanos em cada ponto dos dois caminhos

{em_markdown(rotulos_no_caminho)}

## Limites

{chr(10).join(f"- {l}" for l in limites)}

## Como acrescentar um sistema depois

O comparativo é montado a partir de um registro (`etapa10d.registro`), não de tabelas escritas à mão. Para uma etapa nova (a Etapa 11, por exemplo) entrar:

1. ou uma função nova em `etapa10d.FONTES`, que leia o `decisao.json` da etapa e devolva as fichas; ou um arquivo `sistemas.json` na entrega da etapa (formato em `etapa10d.fichas_declaradas`);
2. cada ficha diz onde ler a qualidade e as respostas (`dev` e teste) e leva as marcas `caminho`, `modelo`, `ponto` e `professor`;
3. `regras` grava uma versão nova das regras (a anterior não é tocada), `preparar` e `calcular` refazem as contas, e este notebook é executado de novo.

Com as marcas preenchidas, o sistema entra sozinho na tabela, na fronteira e nas comparações (entre caminhos, contra o professor, de ponta a ponta).

## O que mudou fora desta etapa

Nenhuma entrega e nenhum cache de outra etapa foi alterado. Dois arquivos de código que outras etapas também usam mudaram, sem mudar o resultado delas:

- `visual.py`: `visual.fronteira` passou a aceitar um ponto de referência em cinza (`serie=None`), para as LLMs que os dois caminhos dividem. Com `serie` 0 ou 1, como a Etapa 9 usa, o desenho é o mesmo.
- `etapa9.py`: `artigos_fora_das_fontes` passou a ler só os resumos das etapas anteriores à Etapa 9. Antes lia os de todas as entregas, e o resumo desta etapa, que cita Reimers & Gurevych (arXiv 1707.09861), fazia a conferência do quadro de artigos da Etapa 9 falhar (um teste e o critério de pronto do notebook 9). Com a mudança, a função devolve o mesmo que devolvia quando a Etapa 9 foi entregue.

## Critério de pronto

| Critério | Atendido |
|---|---|
{tabela_pronto}

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `tabela_final.csv` | Uma linha por sistema: as colunas da Etapa 9 (qualidade no teste com intervalo, rótulos humanos, parâmetros, tamanho, latência e a condição da medida, consultas por segundo, custo), as marcas do registro e a qualidade em `dev` |
| `cadeias.csv` | Cada degrau de cada cadeia, dos dois caminhos, em `dev` e no teste, com a diferença pareada para o degrau anterior |
| `comparacoes.csv` | As {len(regras['comparacoes'])} comparações da lista fechada, B − A, em `dev` e no teste, em todas as métricas |
| `conferencia.csv` | Cada número refeito aqui contra o que a etapa de origem gravou |
| `fronteira.csv` | Para cada sistema com {n(POUCOS)} rótulos: se está na fronteira (latência e tamanho) e quem o domina; com a latência de origem e com a medida na mesma sessão |
| `fronteira_sem_int8_sem_poda_exploratoria.csv` | A mesma conta sem as linhas do INT8 sem poda. **Exploratória**: acrescentada depois de ver a fronteira, fora das regras |
| `precos.csv` | O preço de cada ponto de exact match entre os dois modelos finais |
| `latencia_ao_lado.csv`, `latencia/` | A latência medida na mesma sessão ao lado da de origem; em {pasta_da_latencia}, os tempos de cada rodada e o resumo |
| `artigos.csv`, `artigos_na_fonte/` | O quadro de artigos e a resposta inteira do arXiv e da ACL Anthology |
| `regras.json`{", `regras_v" + str(regras['versao']) + ".json`" if len(versoes) > 1 else ""} | As regras, como foram gravadas antes das contas |
| `conclusao.json` | A resposta da etapa em dados |
| `manifesto.json`, `calculo.json` | De que arquivos a etapa leu (com impressão digital), as versões, a máquina e o código usado |
| `registro.csv` | De onde vem cada coisa de cada sistema |
| `medidas_novas.json`, `previsoes/` | As respostas produzidas nesta etapa, por não estarem em nenhuma entrega |
| `codigo/` | Cópia do código da etapa |
| `figuras/` | Os gráficos deste notebook |
| `10d_comparativo_dos_caminhos.html` | Este notebook executado, para leitura |
"""
(PASTA / "RESUMO.md").write_text(resumo, encoding="utf-8")
display(Markdown(resumo))
''')

code(r'''
arquivos_da_entrega = sorted(p for p in PASTA.rglob("*") if p.is_file())
pd.DataFrame(
    {"tamanho (KB)": [round(p.stat().st_size / 1024, 1) for p in arquivos_da_entrega]},
    index=[str(p.relative_to(RAIZ)) for p in arquivos_da_entrega],
)
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("ok", len(cells), "células")
