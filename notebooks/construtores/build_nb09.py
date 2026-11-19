"""Monta notebooks/09_comparativo.ipynb a partir das células abaixo."""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Etapa 9 · Comparativo: o caminho inteiro numa tabela

| | |
|---|---|
| **Pergunta** | Somando tudo: o que cada etapa do caminho custou e rendeu, e onde fica o modelo final em relação aos professores, ao teto e às linhas de base? |
| **Entrada** | As entregas das Etapas 2 a 8 (`resultados.csv`, `comparacoes.csv`, `eficiencia.csv`, `decisao.json` e as respostas no teste de cada uma), `entregas/01_dados/` (os rótulos humanos do teste) e o que o comando `preparar` de `especialista/experimentos/etapa9.py` gravou: as respostas no teste que nenhuma etapa tinha guardado. |
| **Entrega** | `entregas/09_comparativo/`: a tabela final (`tabela_final.csv`), as cadeias dos dois cenários e do teto (`cadeias.csv`), as comparações de ponta a ponta (`comparacoes.csv`), as figuras, o quadro de artigos e os limites do trabalho, juntos. |
| **Pronto quando** | (1) a tabela final tem uma linha para cada sistema previsto, com qualidade e intervalo de 95% no teste, e diz em cada linha a condição em que a latência foi medida; (2) nada foi treinado nem medido de novo na GPU ou na API, e o que foi produzido agora está listado; (3) cada número de qualidade, recalculado das respostas, é igual ao que a etapa de origem gravou; (4) cada cadeia tem qualidade, tamanho e latência em cada degrau e a diferença pareada para o degrau anterior, igual à que a etapa de origem gravou; (5) as comparações de ponta a ponta são as da lista fixada antes, todas pareadas, com 2.000 sorteios; (6) as figuras foram gravadas; (7) todo artigo do quadro está no `PLANO.md` ou num `RESUMO.md`, ou foi conferido e está marcado; (8) os limites de cada etapa estão juntos no resumo. |

**O que esta etapa faz.** Nada novo é treinado. As Etapas 2 a 8 mediram, cada uma, um pedaço do caminho que vai de uma LLM a um modelo pequeno. Aqui os pedaços são postos lado a lado, na mesma régua, e o modelo final é comparado de uma ponta à outra com quem ele substitui.

**Artigos de apoio**

- Um modelo pequeno, feito para a tarefa, no lugar da LLM: Belcak et al. ([arXiv 2506.02153](https://arxiv.org/abs/2506.02153)).
- O esqueleto do caminho, LLM professora, destilação, poda e quantização: Ni et al., *From Large to Super-Tiny* ([arXiv 2504.13471](https://arxiv.org/abs/2504.13471)).
- Custo não é um número só, e cada medida de custo precisa da condição em que foi feita: Dehghani et al., *The Efficiency Misnomer* ([arXiv 2110.12894](https://arxiv.org/abs/2110.12894)).
- Latência com uma consulta por vez, percentis e a máquina registrada: Reddi et al., *MLPerf Inference* ([arXiv 1911.02549](https://arxiv.org/abs/1911.02549)).
- A ordem das etapas de compressão: Han et al., *Deep Compression* ([arXiv 1510.00149](https://arxiv.org/abs/1510.00149)).

O artigo de cada técnica do caminho está na seção 8.

**Roteiro**

1. O que entra, e de onde vem cada número
2. O que ficou decidido antes de calcular
3. A conta desta etapa é igual à das etapas anteriores?
4. A tabela final
5. O que cada etapa custou e rendeu: as cadeias
6. De ponta a ponta
7. As fronteiras: qualidade contra latência e contra tamanho
8. Cada passo e o seu artigo
9. Rótulos humanos em cada ponto, limites e o que fica em aberto
10. Entrega
11. Resposta
''')

code(r'''
import json
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from especialista import visual
from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro
from especialista.caminhos import ENTREGAS, RAIZ, entrega
from especialista.dados import massive
from especialista.experimentos import etapa9

visual.aplicar_estilo()
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 40)

PASTA = entrega("09_comparativo")
TRES = etapa9.TRES   # as três métricas das leituras
API, GPU, CPU = etapa9.API, etapa9.GPU, etapa9.CPU

particoes, rotulos = massive.carregar_entrega()
treino, dev, teste = (particoes[p] for p in ("train", "dev", "test"))
gabarito = ouro(teste)
POUCOS, TODOS, N_DEV, N_TESTE = None, len(treino), len(dev), len(teste)   # POUCOS: os rótulos que a LLM vê; lido na seção 1


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


def valor_com_intervalo(linha, metrica):
    return f"{pts(linha[metrica])} ({pts(linha[metrica + '_ic_min'])} a {pts(linha[metrica + '_ic_max'])})"


def posicao(minimo, maximo):
    """Onde B fica em relação a A numa comparação pareada B − A."""
    return "acima" if minimo > 0 else "abaixo" if maximo < 0 else "sem diferença distinguível"


def em_relacao_a(minimo, maximo):
    """O mesmo, pronto para ser seguido do nome de A: "abaixo do", "acima do", "sem diferença distinguível para o"."""
    return {"acima": "acima do", "abaixo": "abaixo do"}.get(posicao(minimo, maximo), "sem diferença distinguível para o")


def metrica_na_frase(m):
    """O nome de uma métrica no meio de uma frase: "acurácia de intenção", "F1 de entidades"."""
    nome = qualidade.NOMES[m]
    return nome if nome.startswith("F1") else nome[0].lower() + nome[1:]


def ms(x):
    """Latência: "854 ms" para a LLM, "0,26 ms" para a rede."""
    return "não medida" if pd.isna(x) else f"{n(x)} ms" if x >= 100 else f"{num(x, 2)} ms"


def tamanho(b):
    """Tamanho em disco: MB para as redes, GB para a LLM."""
    return "—" if pd.isna(b) else f"{num(b / 1e9, 1)} GB" if b >= 1e9 else f"{num(b / 1e6, 2 if b < 1e7 else 1)} MB"


def vezes(maior, menor):
    """Quantas vezes um número é o outro: "9,4" ou "3.271"."""
    razao = maior / menor
    return n(razao) if razao >= 100 else num(razao, 1)


def em_markdown(tabela):
    """Uma tabela de textos no formato do RESUMO.md."""
    linhas = ["| " + " | ".join(tabela.columns) + " |", "|" + "---|" * len(tabela.columns)]
    return "\n".join(linhas + ["| " + " | ".join(str(v) for v in linha) + " |" for linha in tabela.itertuples(index=False)])


def trio(linha, colunas=("diferenca", "ic_min", "ic_max")):
    """Valor e intervalo, em pontos percentuais, para os gráficos."""
    return tuple(100 * linha[c] for c in colunas)


def diferencas(dados, titulo, eixo_x):
    """`visual.diferencas_com_intervalo`, com o número das diferenças negativas à esquerda da barra: à direita ele ficaria em cima da linha do zero."""
    fig, ax = visual.diferencas_com_intervalo(dados, titulo, eixo_x)
    for anotacao, (valor, minimo, maximo) in zip(ax.texts, dados.values()):
        if valor < 0:
            anotacao.xy = (minimo, anotacao.xy[1])
            anotacao.set_position((-7, 0))
            anotacao.set_ha("right")
    esquerda, direita = ax.get_xlim()
    ax.set_xlim(esquerda - 0.07 * (direita - esquerda), direita)   # espaço para o número da barra mais à esquerda não encostar nos nomes das linhas
    return fig, ax
''')

md(r'''
## 1. O que entra, e de onde vem cada número

Cada linha da tabela final é um sistema que alguma etapa já mediu. O catálogo abaixo (em `etapa9.catalogo`) diz, para cada um, em que arquivo está a qualidade, em que arquivo estão as respostas no teste e em que etapa a latência foi medida. Os nomes das variantes (qual poda, qual forma de 8 bits, qual temperatura) saem dos `decisao.json` das Etapas 6, 7 e 8.

Três coisas não estavam gravadas em nenhuma entrega. O comando abaixo as produz na CPU, em segundos, sem treino de rede, sem GPU e sem API:

```bash
uv run python -m especialista.experimentos.etapa9 preparar
```

| O que faltava | Como foi produzido |
|---|---|
| As respostas no teste dos sistemas em INT8 | A Etapa 8 entregou os modelos, não as respostas. Cada modelo é carregado da entrega (`SistemaOnnx.carregar`) e responde ao teste, um pedido por vez |
| As respostas no teste da linha de base com os poucos rótulos | A Etapa 5a gravou as métricas dela, não as respostas. Ela é treinada de novo, do mesmo jeito (leva um segundo; não é rede neural) |
| O tamanho em disco do professor aberto | Os arquivos do modelo como foram baixados, medidos no cache do Hugging Face desta máquina |

A seção 3 confere que essas respostas refeitas dão exatamente as métricas que as etapas de origem gravaram.
''')

code(r'''
fichas = etapa9.catalogo()
faltam = etapa9.faltam(fichas)
assert not faltam, f"Falta produzir {len(faltam)} arquivo(s). Rode: {etapa9.COMO_PREPARAR}"
medidas = etapa9.carregar_medidas()
T = etapa9.tabela_final(fichas, medidas)
grupos = {chave: etapa9.respostas(ficha, list(teste["id"])) for chave, ficha in fichas.items()}   # as respostas no teste, uma lista por semente

FINAL = etapa9.chave_do_final(fichas)                    # o modelo final do projeto, como a Etapa 8 decidiu
PRINCIPAL = fichas[FINAL]["de_partida"]                   # o aluno de que ele saiu
POUCOS = int(T["rotulos_humanos"].min())                  # os rótulos humanos que a LLM vê
assert int(T["rotulos_humanos"].max()) == TODOS and all(len(g) == N_TESTE for grupo in grupos.values() for g in grupo)
PROFESSOR, ABERTO = T.loc["claude", "curto"], T.loc["qwen", "curto"]
MODELOS = sorted((c for c in fichas if f"{c}/int8" in fichas), key=lambda c: T.loc[c, "rotulos_humanos"])   # os que passaram pela compressão: o aluno de cada cenário e, por último, o teto
SEMENTES = len(etapa9.SEMENTES)
FOLGA = etapa9.variacao_da_latencia()                     # quanto a latência mediana variou entre as rodadas da Etapa 8: abaixo disso, duas latências são empate

# As decisões e configurações das etapas anteriores que os textos citam
ler_json = lambda *partes: json.loads(ENTREGAS.joinpath(*partes).read_text(encoding="utf-8"))
config5a, custo5a = ler_json("05a_professor_fechado", "configuracao.json"), ler_json("05a_professor_fechado", "custo.json")
escolha5b, decisao6 = ler_json("05b_professor_aberto", "escolha.json"), ler_json("06_destilacao", "decisao.json")
decisao7, decisao8 = ler_json("07_poda", "decisao.json"), ler_json("08_quantizacao", "decisao.json")
K = config5a["ajuste"]["k"]                               # quantos exemplos rotulados entram em cada pedido à LLM


def de_onde(caminhos):
    pasta = caminhos[0].parent
    return str(pasta.relative_to(RAIZ)) + "/"


origem = pd.DataFrame([
    {
        "sistema": T.loc[chave, "sistema"], "sementes": ficha["sementes"], "qualidade lida de": T.loc[chave, "qualidade_lida_de"],
        "respostas no teste": ("geradas agora: " + ("modelo de " + str(ficha["gerar"]["de"][0].parent.relative_to(RAIZ)) + "/" if ficha["gerar"]["como"] == "onnx" else "linha de base treinada de novo")
                               if "gerar" in ficha else "as do PyTorch (a Etapa 8 conferiu que são iguais)" if "mesmas_respostas_de" in ficha else de_onde(ficha["previsoes"])),
        "latência medida na": T.loc[chave, "latencia_medida_na"] or "não medida",
    }
    for chave, ficha in fichas.items()
]).set_index("sistema")
display(origem)

aberto = medidas["professor_aberto"]
geradas = [c for c, f in fichas.items() if "gerar" in f]
display(Markdown(f"""
**Leitura.** São {len(fichas)} sistemas. Para {len(fichas) - len(geradas)} deles as respostas no teste já estavam nas entregas; para {len(geradas)} foram produzidas agora
({len([c for c in geradas if fichas[c]['gerar']['como'] == 'onnx'])} sistemas em INT8 da Etapa 8 e a linha de base com {n(POUCOS)} rótulos).
O modelo final do projeto, pela decisão da Etapa 8, é **{T.loc[FINAL, 'sistema']}**.
O tamanho em disco do {ABERTO}: {tamanho(aberto['bytes_dos_pesos_baixados']) + f" em {aberto['arquivos']} arquivos ({aberto['o_que_e']})" if aberto['bytes_dos_pesos_baixados'] else "não medido (o modelo não está mais no cache desta máquina)"}.
"""))
''')

md(r'''
## 2. O que ficou decidido antes de calcular

Tudo abaixo foi escrito, e posto no código (`etapa9.py`), antes de calcular qualquer comparação desta etapa.

**1. Só o teste, e nenhuma decisão.** O modelo final já foi escolhido, em `dev`, pelas regras das Etapas 6, 7 e 8. Esta etapa só reporta. Tudo o que aparece aqui é do teste, contra os rótulos humanos.

**2. Os números são lidos, não refeitos.** A qualidade de cada sistema vem do `resultados.csv` da etapa que o mediu; parâmetros, tamanho, latência e custo, dos arquivos dessa mesma etapa. Como conferência, cada número de qualidade é recalculado a partir das respostas gravadas. **Se a conta daqui e a de uma etapa anterior discordarem, o notebook para**: não se escolhe um dos dois.

**3. O ONNX de 32 bits não tem respostas próprias gravadas.** A Etapa 8 conferiu que ele responde igual ao PyTorch em todos os pedidos de `dev` e do teste. As respostas dele são, portanto, as do modelo podado em PyTorch.

**4. A latência de cada sistema é a da condição em que ele roda de verdade**, e a tabela diz qual é, linha por linha. São três condições, que não se comparam como se fossem a mesma coisa:

| Condição | Quem | O que o tempo inclui |
|---|---|---|
| API pela rede | a LLM fechada | a ida e a volta até o provedor, a fila e o cálculo, uma chamada por vez |
| GPU local | a LLM aberta | o cálculo na placa desta máquina, um pedido por vez |
| CPU local | as redes do zero e a linha de base | do texto à resposta, uma consulta por vez, 1 thread |

Quando o mesmo modelo foi medido em mais de uma etapa, vale a medida da **Etapa 8**: é a única feita com a máquina conferidamente quieta e com os sistemas alternados nas mesmas rodadas. Só o piso e o aluno B1, que não foram até lá, ficam com a medida da Etapa 6; a linha de base, com a da Etapa 2.

**5. O custo.** Em dólar, só onde existe um preço: a API. Para os sistemas locais não se inventa preço: a tabela traz **consultas por segundo**, que é o inverso da latência média, com uma consulta por vez.

**6. O tamanho** é o do arquivo como o sistema o usa. Onde uma etapa mediu também o arquivo compactado (zip), ele aparece ao lado. Do professor fechado, parâmetros e tamanho não são divulgados.

**7. As cadeias.** Para cada cenário, o caminho professor → aluno destilado → podado e reajustado → ONNX de 32 bits → INT8, com qualidade, tamanho e latência em cada degrau e a **diferença pareada de cada degrau para o anterior**. O teto percorre os mesmos degraus de compressão, como referência: nele não há professor.

**8. As comparações de ponta a ponta** são as da lista abaixo, e só elas. Qualquer outra, acrescentada depois de ver os resultados, tem de ser marcada como exploratória.

**9. Como ler uma diferença.** Toda comparação é pareada (`qualidade.comparar_grupos`): bootstrap por enunciado, 2.000 sorteios, os mesmos sorteios para os dois sistemas, média de 3 sementes quando o sistema é treinado. Uma diferença só vale quando o intervalo de 95% não contém zero.

**10. A fronteira.** Um sistema está na fronteira quando nenhum outro é, ao mesmo tempo, mais barato e melhor. Ela é traçada entre os sistemas que usam só os poucos rótulos humanos, que é o cenário do projeto; o teto e a linha de base com todos os rótulos aparecem como referência. Duas latências mais próximas do que a variação da própria medida (a maior variação da mediana entre as rodadas da Etapa 8) contam como empate.
''')

code(r'''
nome_na_comparacao = {FINAL: "modelo final", "aluno_b2/int8": "final do cenário B", "teto/int8": "teto comprimido", "aluno_a": "aluno A sem compressão", "aluno_b2": "aluno B2 sem compressão",
                      "teto": "teto", "piso": "piso", "claude": PROFESSOR, "qwen": ABERTO, "base_299": T.loc["base_299", "curto"].lower(), "base_11514": T.loc["base_11514", "curto"].lower()}


def rotulo_da_comparacao(letra, b, a):
    return f"{letra}. {nome_na_comparacao[b][0].upper() + nome_na_comparacao[b][1:]} − {nome_na_comparacao[a]}"


ROTULO = {letra: rotulo_da_comparacao(letra, b, a) for letra, b, a, _ in etapa9.PONTA_A_PONTA}
pd.DataFrame([
    {"comparação (B − A)": ROTULO[letra], "B": T.loc[b, "sistema"], "A": T.loc[a, "sistema"], "o que responde": pergunta}
    for letra, b, a, pergunta in etapa9.PONTA_A_PONTA
]).set_index("comparação (B − A)")
''')

md(r'''
## 3. A conta desta etapa é igual à das etapas anteriores?

Antes de usar as respostas para comparar sistemas, a conferência da decisão 2. Para cada sistema, as quatro métricas e os limites dos intervalos são recalculados das respostas e comparados com a linha que a etapa de origem gravou. Para os modelos podados em 32 bits, que duas etapas mediram (a 7 e a 8), as duas são conferidas.

Depois, o mesmo para os degraus das cadeias: cada diferença pareada calculada aqui é comparada com a que a etapa responsável por aquele passo gravou no seu `comparacoes.csv`.
''')

code(r'''
confere_qualidade = etapa9.conferir_qualidade(fichas, T, gabarito, grupos)
CAD = etapa9.cadeias(T, gabarito, grupos)
confere_degraus = etapa9.conferir_degraus(fichas, CAD)

display(confere_qualidade.assign(sistema=[T.loc[c, "sistema"] for c in confere_qualidade["chave"]])[["sistema", "gravado_em", "respostas_geradas_agora", "maior_diferenca", "confere"]])
display(confere_degraus.assign(degrau=[f"{T.loc[c, 'curto']} − {T.loc[a, 'curto']}" for c, a in zip(confere_degraus["chave"], confere_degraus["anterior"])])[["cadeia", "degrau", "gravado_em", "maior_diferenca", "confere"]])

tudo_confere = bool(confere_qualidade["confere"].all() and confere_degraus["confere"].all())
maior_diferenca = max(confere_qualidade["maior_diferenca"].max(), confere_degraus["maior_diferenca"].max())
display(Markdown(f"""
**Leitura.** {len(confere_qualidade)} linhas de qualidade conferidas ({len(fichas)} sistemas; os {len(confere_qualidade) - len(fichas)} modelos podados em 32 bits, contra duas etapas) e {len(confere_degraus)} degraus.
{"Todas iguais" if tudo_confere else "**HÁ DIFERENÇA**"}: a maior diferença entre a conta desta etapa e o que estava gravado é de {maior_diferenca:.1e} (em fração, não em pontos).
{"O que as etapas anteriores reportaram e o que esta etapa calcula são os mesmos números, inclusive para as respostas refeitas agora (INT8 e linha de base)." if tudo_confere else "**A etapa não pode seguir: uma conta anterior e a desta etapa discordam. Veja as linhas com `confere = False`.**"}
"""))
assert tudo_confere, "A conta desta etapa e a de uma etapa anterior discordam: pare e avise, em vez de escolher uma das duas."
''')

md(r'''
## 4. A tabela final

Uma linha por sistema, no teste. Qualidade em %, com intervalo de 95%; nos sistemas treinados, média das sementes. Tamanho e latência dos modelos comprimidos são os da semente 0, como na Etapa 8.
''')

code(r'''
def onde_curto(chave):
    s = T.loc[chave]
    return "não medida" if pd.isna(s["latencia_p50_ms"]) else f"{s['condicao_da_latencia']}: {fichas[chave]['onde']} ({s['latencia_medida_na']})"


def linha_da_tabela(chave):
    s = T.loc[chave]
    oculto = not s["divulgado"]
    return {
        "Sistema": s["sistema"] + (" **(modelo final)**" if chave == FINAL else ""),
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

f, p, t_ = T.loc[FINAL], T.loc["claude"], T.loc["teto"]
maquina = {"cpu": pd.read_csv(ENTREGAS / "08_quantizacao" / "eficiencia.csv")["cpu"].iloc[0], "gpu": pd.read_csv(ENTREGAS / "05b_professor_aberto" / "resultados.csv")["gpu"].iloc[0]}
notas_da_tabela = [
    f"**Máquina.** CPU local: {maquina['cpu']}. GPU local: {maquina['gpu']}. A API foi chamada desta máquina ({fichas['claude']['onde'].split(',')[0]}, {fichas['claude']['onde'].split(', ')[1]}).",
    f"**Rótulos humanos.** \"no prompt\": a LLM vê {n(POUCOS)} exemplos rotulados por humanos, {K} em cada pedido. \"pelo professor\": a rede nunca vê um rótulo humano de treino; aprende com os {n(TODOS)} pedidos rotulados pela LLM, que viu os {n(POUCOS)}. Onde os rótulos humanos de `dev` entraram em decisões está na tabela \"Rótulos humanos em cada ponto do caminho\".",
    f"**Consultas por segundo** é o inverso da latência média, uma consulta por vez. **US$ por 1.000 consultas** é o preço de lista da chamada direta (em lote, US$ {num(p['custo_por_mil_em_lote_usd'], 3)}); dos sistemas locais não há preço medido.",
    f"**Tamanho.** Do {PROFESSOR}, parâmetros e tamanho não são divulgados. Do {ABERTO}, o tamanho é o dos arquivos baixados, com pesos de 16 bits; carregado em 4 bits ele ocupa {num(T.loc['qwen', 'memoria_na_placa_gb'], 1)} GB na placa. A poda de pesos soltos em INT8 é entregue já compactada: o tamanho dela em disco é o da coluna \"com zip\".",
    f"**Linha de base com {n(POUCOS)} rótulos.** A Etapa 5a mediu só a qualidade; tamanho e latência não foram medidos.",
]
display(Markdown("\n".join(f"- {nota}" for nota in notas_da_tabela) + f"""

**Leitura.** O modelo final faz {pts(f['exact_match'])}% de exact match com {tamanho(f['bytes'])} e {ms(f['latencia_p50_ms'])} por consulta numa thread de CPU ({n(f['consultas_por_segundo'])} consultas por segundo).
O professor dele, {PROFESSOR}, faz {pts(p['exact_match'])}% em {ms(p['latencia_p50_ms'])} pela API, a US$ {num(p['custo_por_mil_usd'], 3)} por mil consultas.
O teto, a mesma rede treinada com os {n(TODOS)} rótulos humanos, faz {pts(t_['exact_match'])}%. As seções seguintes dizem quais dessas distâncias se distinguem de zero e de que etapa cada uma vem.
"""))
''')

md(r'''
## 5. O que cada etapa custou e rendeu: as cadeias

Cada cadeia é lida de cima para baixo. Em cada degrau muda uma coisa só:

| Degrau | O que muda em relação ao anterior | Etapa |
|---|---|---|
| Professor (LLM) | é o ponto de partida: a LLM resolvendo a tarefa | 5a, 5b |
| Aluno destilado | a rede do zero aprende com os rótulos (e, no cenário B, os logits) da LLM | 6 |
| Podado e reajustado | saem 90% das linhas da tabela de features; a rede é reajustada com o mesmo sinal | 7 |
| ONNX, 32 bits | a mesma rede sai do PyTorch e roda no ONNX Runtime | 8 |
| INT8 | os pesos passam de 32 para 8 bits | 8 |

A coluna "diferença para o degrau anterior" é pareada: diz o que aquele passo, sozinho, fez com a qualidade.
''')

code(r'''
def tabela_da_cadeia(nome):
    c = CAD[CAD["cadeia"] == nome]
    return pd.DataFrame([
        {
            "Degrau": d["degrau"], "Sistema": T.loc[d["chave"], "curto"], "Exact match": valor_com_intervalo(d, "exact_match"),
            **{f"{qualidade.NOMES[m]}: diferença para o degrau anterior": "—" if pd.isna(d["anterior"]) else com_intervalo(d, f"{m}_", 2 if abs(d[f"{m}_diferenca"]) < 0.0015 else 1) for m in ("exact_match", "intent_acc", "slot_f1")},
            "Tamanho em disco": "não divulgado" if not T.loc[d["chave"], "divulgado"] else tamanho(d["bytes"]), "Latência p50": ms(d["latencia_p50_ms"]), "Onde": d["condicao_da_latencia"],
        }
        for _, d in c.iterrows()
    ])


def figura_da_cadeia(nome):
    c = CAD[CAD["cadeia"] == nome]
    com_nome = 2 if c["degrau"].iloc[0] == etapa9.DEGRAUS["professor"] else 1   # os primeiros degraus levam o nome do sistema embaixo
    degraus = [f"{d}\n{T.loc[k, 'curto']}" if i < com_nome else f"{d}\nmodelo final" if k == FINAL else d for i, (d, k) in enumerate(zip(c["degrau"], c["chave"]))]
    megas = [None if pd.isna(b) else b / 1e6 for b in c["bytes"]]
    latencias = list(c["latencia_p50_ms"])
    return visual.degraus_em_paineis(
        degraus,
        [
            {"eixo_x": "exact match (%)", "valores": [trio(d, ("exact_match", "exact_match_ic_min", "exact_match_ic_max")) for _, d in c.iterrows()], "rotulos": [pts(v) for v in c["exact_match"]]},
            {"eixo_x": "tamanho em disco (MB, escala logarítmica)", "valores": megas, "rotulos": ["não divulgado" if v is None else tamanho(v * 1e6) for v in megas], "log": True},
            {"eixo_x": "latência p50 (ms, escala logarítmica)", "valores": latencias, "rotulos": [f"{ms(v)} · {onde}" for v, onde in zip(latencias, c["condicao_da_latencia"])], "log": True},
        ],
        etapa9.CADEIAS[nome]["nome"],
    )


poda7 = pd.read_csv(ENTREGAS / "07_poda" / "comparacoes.csv")


def leitura_da_cadeia(nome):
    """A cadeia em frases: o que cada passo fez com a qualidade, o tamanho e a latência."""
    c = CAD[CAD["cadeia"] == nome].set_index("degrau")
    D = etapa9.DEGRAUS
    frases = []
    if D["professor"] in c.index:
        d = c.loc[D["destilado"]]
        frases.append(f"**Destilação.** O aluno fica {em_relacao_a(d['exact_match_dif_ic_min'], d['exact_match_dif_ic_max'])} professor em exact match: "
                      f"{com_intervalo(d, 'exact_match_')}. Em troca, a consulta sai de {ms(c.loc[D['professor'], 'latencia_p50_ms'])} ({c.loc[D['professor'], 'condicao_da_latencia']}) "
                      f"para {ms(d['latencia_p50_ms'])} ({d['condicao_da_latencia']}, 1 thread), e o sistema passa a ser um arquivo de {tamanho(d['bytes'])}.")
    d, antes = c.loc[D["podado"]], c.iloc[c.index.get_loc(D["podado"]) - 1]
    modelo = fichas[d["chave"]]["resultado"][1]["modelo"]
    variante = decisao7["modelos"][modelo]["principal"]
    sem_reajuste = etapa9.linha_gravada(poda7, modelo=modelo, variante=variante, comparacao="antes − sem poda", particao="test", metrica="exact_match")
    recupera = etapa9.linha_gravada(poda7, modelo=modelo, variante=variante, comparacao="depois − antes", particao="test", metrica="exact_match")
    frases.append(f"**Poda e reajuste.** Logo depois do corte, sem reajuste, o exact match fica {com_intervalo(sem_reajuste)} em relação ao modelo sem poda; o reajuste devolve {com_intervalo(recupera)} (Etapa 7). "
                  f"Depois do reajuste, o modelo podado fica {em_relacao_a(d['exact_match_dif_ic_min'], d['exact_match_dif_ic_max'])} modelo sem poda: {com_intervalo(d, 'exact_match_')}. "
                  f"O arquivo vai de {tamanho(antes['bytes'])} para {tamanho(d['bytes'])} ({vezes(antes['bytes'], d['bytes'])} vezes menos); "
                  f"a latência {'não muda' if abs(d['latencia_p50_ms'] / antes['latencia_p50_ms'] - 1) <= FOLGA else 'muda'} ({ms(antes['latencia_p50_ms'])} e {ms(d['latencia_p50_ms'])}).")
    o, i = c.loc[D["onnx"]], c.loc[D["int8"]]
    frases.append(f"**Exportação.** Sair do PyTorch não muda nenhuma resposta ({com_intervalo(o, 'exact_match_')}) e leva a consulta de {ms(d['latencia_p50_ms'])} para {ms(o['latencia_p50_ms'])} ({vezes(d['latencia_p50_ms'], o['latencia_p50_ms'])} vezes menos).")
    frases.append(f"**8 bits.** Passar para INT8 muda o exact match em {com_intervalo(i, 'exact_match_', 2)} ({posicao(i['exact_match_dif_ic_min'], i['exact_match_dif_ic_max'])}), "
                  f"e leva o arquivo de {tamanho(o['bytes'])} para {tamanho(i['bytes'])} ({vezes(o['bytes'], i['bytes'])} vezes menos) e a consulta de {ms(o['latencia_p50_ms'])} para {ms(i['latencia_p50_ms'])}.")
    return frases


leituras_das_cadeias = {}
for nome in etapa9.CADEIAS:
    display(Markdown(f"### {etapa9.CADEIAS[nome]['nome']}"))
    display(tabela_da_cadeia(nome).set_index("Degrau"))
    fig, _ = figura_da_cadeia(nome)
    visual.salvar(fig, PASTA, f"cadeia_{'cenario_' + nome.lower() if nome != 'teto' else 'teto'}")
    plt.show()
    leituras_das_cadeias[nome] = leitura_da_cadeia(nome)
    display(Markdown("\n".join(f"- {frase}" for frase in leituras_das_cadeias[nome])))
''')

code(r'''
passos = CAD.dropna(subset=["anterior"])
distinguiveis = [(d["cadeia"], d["degrau"], m) for _, d in passos.iterrows() for m in TRES if d[f"{m}_distinguivel"]]
NOME_DA_CADEIA = {"A": "cenário A", "B": "cenário B", "teto": "teto"}
frase_dos_degraus = "; ".join(f"{NOME_DA_CADEIA[c]}, {g.lower()}, {metrica_na_frase(m)} ({com_intervalo(passos[(passos['cadeia'] == c) & (passos['degrau'] == g)].iloc[0], m + '_')})" for c, g, m in distinguiveis)
D = etapa9.DEGRAUS
so_na_destilacao = all(g == D["destilado"] for c, g, _ in distinguiveis if c != "teto")
poda_do_teto = [m for c, g, m in distinguiveis if c == "teto" and g == D["podado"]]
leitura_das_cadeias = (
    f"De {len(passos)} degraus × {len(TRES)} métricas, {len(distinguiveis)} diferenças se distinguem de zero: {frase_dos_degraus}. "
    + ("Nos dois cenários, **toda a mudança de qualidade que se distingue de zero acontece na destilação**; nenhum degrau de compressão se distingue. " if so_na_destilacao
       else "Nos dois cenários há degraus de compressão que se distinguem de zero: a mudança de qualidade não fica só na destilação. ")
    + (f"No teto, a poda principal fica abaixo do modelo sem poda mesmo depois do reajuste ({' e '.join(metrica_na_frase(m) for m in poda_do_teto)}). " if poda_do_teto
       else "No teto, nenhum degrau de compressão se distingue de zero. ")
    + "O que a compressão muda é o custo: a poda diminui o arquivo, a exportação diminui a latência, os 8 bits diminuem os dois."
)
display(Markdown(f"**Leitura das três cadeias.** {leitura_das_cadeias}"))
''')

md(r'''
## 6. De ponta a ponta

As comparações da lista da seção 2, B − A, no teste. Cada uma junta vários degraus de uma vez: responde a uma pergunta que nenhuma etapa, sozinha, respondeu.
''')

code(r'''
PP = etapa9.ponta_a_ponta(T, gabarito, grupos)
assert list(dict.fromkeys(PP["letra"])) == [letra for letra, *_ in etapa9.PONTA_A_PONTA] and not PP["exploratoria"].any()   # só a lista fixada antes
PP["rotulo"] = PP["letra"].map(ROTULO)
pp = PP.set_index(["letra", "metrica"])

tabela_pp = pd.DataFrame([
    {"Comparação (B − A)": ROTULO[letra], **{qualidade.NOMES[m]: com_intervalo(pp.loc[(letra, m)]) for m in TRES}, "O que responde": pergunta}
    for letra, _, _, pergunta in etapa9.PONTA_A_PONTA
])
display(tabela_pp.set_index("Comparação (B − A)"))

FIGURAS_PP = {"exact_match": "ponta_a_ponta_exact_match", "intent_acc": "ponta_a_ponta_intencao", "slot_f1": "ponta_a_ponta_entidades"}
for metrica, arquivo in FIGURAS_PP.items():
    fig, ax = diferencas(
        {ROTULO[letra]: trio(pp.loc[(letra, metrica)]) for letra, *_ in etapa9.PONTA_A_PONTA},
        f"De ponta a ponta, no teste: {metrica_na_frase(metrica)}", "diferença (o primeiro menos o segundo), em pontos percentuais (intervalo de 95%)",
    )
    visual.salvar(fig, PASTA, arquivo)
    plt.show()
''')

code(r'''
def frase(letra, metricas=TRES):
    """Uma comparação em palavras: onde B fica em relação a A, métrica por métrica."""
    partes = []
    for m in metricas:
        l = pp.loc[(letra, m)]
        onde = posicao(l["ic_min"], l["ic_max"])
        partes.append(f"{onde} em {metrica_na_frase(m)} ({com_intervalo(l)})")
    return ", ".join(partes[:-1]) + " e " + partes[-1] if len(partes) > 1 else partes[0]


em = lambda letra: pp.loc[(letra, "exact_match")]
destilacao_a = CAD[(CAD["cadeia"] == "A") & (CAD["degrau"] == etapa9.DEGRAUS["destilado"])].iloc[0]
leituras_pp = [
    f"**a. Contra o professor.** O modelo final fica {frase('a')}. " + (f"A destilação sozinha tinha deixado {com_intervalo(destilacao_a, 'exact_match_')} de exact match (cadeia do cenário A), "
       f"e a compressão {'não se distingue de zero (comparação b)' if not em('b')['distinguivel'] else 'também pesa (comparação b)'}."),
    f"**b. O custo de toda a compressão.** Poda, exportação e 8 bits, juntas, deixam o modelo final {frase('b')} em relação ao aluno A como saiu da destilação. "
    f"Nesse caminho o arquivo foi de {tamanho(T.loc['aluno_a', 'bytes'])} para {tamanho(T.loc[FINAL, 'bytes'])} ({vezes(T.loc['aluno_a', 'bytes'], T.loc[FINAL, 'bytes'])} vezes menos) "
    f"e a consulta de {ms(T.loc['aluno_a', 'latencia_p50_ms'])} para {ms(T.loc[FINAL, 'latencia_p50_ms'])} ({vezes(T.loc['aluno_a', 'latencia_p50_ms'], T.loc[FINAL, 'latencia_p50_ms'])} vezes menos).",
    f"**c. Contra o teto.** Sem os {n(TODOS)} rótulos humanos de treino, o modelo final fica {frase('c')}." + (" É a maior queda de exact match da lista." if em('c')['diferenca'] == PP.loc[PP['metrica'] == 'exact_match', 'diferenca'].min() else ""),
    f"**d. A compressão sozinha.** No teto, onde não há professor, a mesma compressão deixa o modelo {frase('d')}. "
    f"Em exact match, esse custo {'se distingue' if em('d')['distinguivel'] else 'não se distingue'} de zero no teto e {'se distingue' if em('b')['distinguivel'] else 'não se distingue'} no aluno A (comparação b). "
    f"Se os dois custos diferem entre si não foi testado.",
    f"**e. O cenário B.** O final do cenário B fica {frase('e1')} em relação ao {ABERTO}, o professor dele, e {frase('e2', ('exact_match',))} em relação ao aluno B2 sem compressão.",
    f"**f. Um cenário contra o outro.** O final do cenário A fica {frase('f')} em relação ao final do cenário B. Os dois têm a mesma rede e passaram pela mesma compressão; o que muda entre eles é o professor e o sinal de treino.",
    f"**g. Contra as referências.** Contra a linha de base treinada com os mesmos {n(POUCOS)} rótulos, o modelo final fica {frase('g1', ('exact_match',))}; contra a mesma rede treinada só com esses {n(POUCOS)} rótulos (o piso), {frase('g3', ('exact_match',))}. "
    f"Contra a linha de base que usa os {n(TODOS)} rótulos, fica {frase('g2')}.",
]
distinguiveis_pp = int(PP[PP["metrica"].isin(TRES)]["distinguivel"].sum())
total_pp = len(etapa9.PONTA_A_PONTA) * len(TRES)
nota_do_acaso = (f"São {total_pp} comparações ({len(etapa9.PONTA_A_PONTA)} × {len(TRES)} métricas), e {distinguiveis_pp} se distinguem de zero. Não há correção para comparações múltiplas: "
                 f"com intervalos de 95%, cerca de {num(0.05 * total_pp, 1)} sairia do zero só por acaso. Nenhuma comparação foi acrescentada depois de ver os resultados.")
display(Markdown("**Leitura.**\n\n" + "\n".join(f"- {l}" for l in leituras_pp) + f"\n- {nota_do_acaso}"))
''')

md(r'''
## 7. As fronteiras: qualidade contra latência e contra tamanho

Os mesmos sistemas, agora num plano: custo no eixo horizontal, em **escala logarítmica**, e exact match no vertical. Em cada figura:

- a **cor** diz quantos rótulos humanos de treino o sistema usa;
- uma **linha** liga as formas do mesmo modelo (sem compressão, podado ou exportado, INT8);
- a **linha tracejada** é a fronteira dos sistemas com os poucos rótulos humanos: à direita e abaixo dela não há por que escolher.

Na figura de latência, a **forma do ponto** diz a condição da medida: círculo para a CPU local, quadrado para a GPU local, losango para a API pela rede.
''')

code(r'''
SERIES = [f"usa {n(POUCOS)} rótulos humanos", f"usa {n(TODOS)} rótulos humanos"]
MARCADOR = {CPU: "o", GPU: "s", API: "D"}
serie = lambda chave: int(T.loc[chave, "rotulos_humanos"] == TODOS)
em_pontos = lambda chave: 100 * T.loc[chave, "exact_match"]
duas_linhas = lambda texto: texto.replace(", ", "\n", 1)
ROTULO_DA_FRONTEIRA = f"fronteira dos sistemas com {n(POUCOS)} rótulos humanos"
do_cenario = T[T["rotulos_humanos"] == POUCOS]


def ponto(chave, custo, rotulo=None, **extras):
    return {"x": custo(chave), "y": em_pontos(chave), "rotulo": rotulo, "serie": serie(chave), **extras}


# ---- qualidade contra latência
lat = lambda chave: T.loc[chave, "latencia_p50_ms"]
fronteira_lat = etapa9.nao_dominados(do_cenario, "latencia_p50_ms", "exact_match", FOLGA)
pontos = []
for m in MODELOS:
    pontos += [ponto(m, lat, T.loc[m, "curto"], lado="cima" if f"{m}/int8" == FINAL else "direita"), ponto(f"{m}/onnx", lat),
               ponto(f"{m}/int8", lat, "modelo final" if f"{m}/int8" == FINAL else None, lado="cima", forte=True)]
pontos += [
    ponto("aluno_b1", lat, T.loc["aluno_b1", "curto"], lado="direita", desvio=(0, -8)),
    ponto("piso", lat, T.loc["piso", "curto"], lado="direita"),
    ponto("base_11514", lat, duas_linhas(T.loc["base_11514", "curto"]), lado="baixo", desvio=(-34, -12), fio=True),
    ponto("claude", lat, f"{PROFESSOR}\n{API}", lado="esquerda", marcador=MARCADOR[API]),
    ponto("qwen", lat, f"{ABERTO}\n{GPU}", lado="esquerda", marcador=MARCADOR[GPU]),
]
fig, ax = visual.fronteira(
    pontos, "Qualidade contra latência, no teste", "latência p50 de uma consulta (ms, escala logarítmica)", "exact match (%)", series=SERIES,
    trilhas=[{"pontos": [(lat(c), em_pontos(c)) for c in (f"{m}/int8", f"{m}/onnx", m)], "serie": serie(m)} for m in MODELOS],
    escada=[(lat(c), em_pontos(c)) for c in fronteira_lat] + [(3000, em_pontos(fronteira_lat[-1]))], rotulo_da_escada=ROTULO_DA_FRONTEIRA,
    colunas={"INT8": (lat(FINAL) * 0.9, "right"), "ONNX\n32 bits": lat(f"{PRINCIPAL}/onnx"), "PyTorch\n32 bits": lat(PRINCIPAL)},
    nota="Círculos: medidos na CPU local, 1 thread.\nO modelo podado em PyTorch tem a latência do modelo\nsem poda: cai sobre o mesmo ponto e não é desenhado.",
)
ax.set_xlim(0.12, 3000)
ax.set_ylim(15, 72)
visual.salvar(fig, PASTA, "fronteira_latencia")
plt.show()

# ---- qualidade contra tamanho: o arquivo como é entregue (a poda de pesos soltos em INT8 vai compactada)
megas = lambda chave: (T.loc[chave, "bytes_com_zip"] if T.loc[chave, "entregue_compactado"] else T.loc[chave, "bytes"]) / 1e6
entregue = T.assign(megas=[megas(c) for c in T.index])
fronteira_tam = etapa9.nao_dominados(entregue[entregue["rotulos_humanos"] == POUCOS], "megas", "exact_match")
pontos = []
for m in MODELOS:
    pontos += [ponto(m, megas, T.loc[m, "curto"], lado="direita" if serie(m) else "cima", desvio=(0, 0) if serie(m) else (0, -3)), ponto(f"{m}/podado", megas),
               ponto(f"{m}/int8", megas, "modelo final" if f"{m}/int8" == FINAL else None, lado="cima", forte=True), ponto(f"{m}/soltos", megas, vazado=True)]
pontos += [
    ponto("aluno_b1", megas, T.loc["aluno_b1", "curto"], lado="baixo"),
    ponto("piso", megas, T.loc["piso", "curto"], lado="direita"),
    ponto("base_11514", megas, T.loc["base_11514", "curto"], lado="direita"),
]
if pd.notna(T.loc["qwen", "bytes"]):
    pontos.append(ponto("qwen", megas, f"{ABERTO}\n{tamanho(T.loc['qwen', 'bytes'])} baixados", lado="esquerda"))
fig, ax = visual.fronteira(
    pontos, "Qualidade contra tamanho em disco, no teste", "tamanho do arquivo entregue (MB, escala logarítmica)", "exact match (%)", series=SERIES,
    trilhas=[{"pontos": [(megas(c), em_pontos(c)) for c in (f"{m}/int8", f"{m}/podado", m)], "serie": serie(m)} for m in MODELOS],
    escada=[(megas(c), em_pontos(c)) for c in fronteira_tam] + [(40000, em_pontos(fronteira_tam[-1]))], rotulo_da_escada=ROTULO_DA_FRONTEIRA,
    colunas={"INT8": megas(FINAL), "podado\n32 bits": megas(f"{PRINCIPAL}/podado"), "sem poda\n32 bits": (megas(PRINCIPAL), "left")},
    nota=f"Círculo vazado: poda de pesos soltos, INT8 com zip.\n{PROFESSOR}: tamanho não divulgado, fora do gráfico.\nO ONNX de 32 bits cai sobre o ponto podado de 32 bits.",
)
ax.set_xlim(0.4, 40000)
ax.set_ylim(15, 72)
visual.salvar(fig, PASTA, "fronteira_tamanho")
plt.show()
''')

code(r'''
def nomes(chaves):
    return ", ".join(f"{T.loc[c, 'curto']} ({pts(T.loc[c, 'exact_match'])}%)" for c in chaves)


# Dois pontos vizinhos da fronteira podem ser o mesmo modelo em duas formas, separados por menos do que o intervalo da diferença: a cadeia diz se é o caso
onnx_do_final = FINAL.replace("/int8", "/onnx")
int8_contra_onnx = CAD[CAD["chave"] == FINAL].iloc[0]
ressalva = ""
if onnx_do_final in fronteira_lat or onnx_do_final in fronteira_tam:
    ressalva = (f" O ONNX de 32 bits do mesmo modelo também aparece na fronteira, por {num(abs(100 * int8_contra_onnx['exact_match_diferenca']), 2)} ponto de exact match; "
                f"essa diferença não se distingue de zero ({com_intervalo(int8_contra_onnx, 'exact_match_', 2)}, último degrau da cadeia): na prática, é o mesmo ponto com mais custo.")
soltos = FINAL.replace("/int8", "/soltos")
leitura_da_fronteira = [
    f"**Latência.** Entre os sistemas com {n(POUCOS)} rótulos humanos, a fronteira tem: {nomes(fronteira_lat)}. O modelo final responde em {ms(lat(FINAL))} e o professor em {ms(lat('claude'))}: "
    f"{vezes(lat('claude'), lat(FINAL))} vezes menos, com a ressalva de que são condições diferentes (CPU local contra API pela rede). Contra o {ABERTO} na GPU local ({ms(lat('qwen'))}), {vezes(lat('qwen'), lat(FINAL))} vezes menos."
    f" Latências a menos de {num(100 * FOLGA, 0)}% uma da outra contam como empate: é o quanto a mediana variou entre as rodadas da Etapa 8.{ressalva}",
    f"**Tamanho.** A fronteira tem: {nomes(fronteira_tam)}. O modelo final ocupa {tamanho(T.loc[FINAL, 'bytes'])}"
    + (f"; os arquivos do {ABERTO}, {tamanho(T.loc['qwen', 'bytes'])}: {vezes(T.loc['qwen', 'bytes'], T.loc[FINAL, 'bytes'])} vezes mais." if pd.notna(T.loc["qwen", "bytes"]) else ".")
    + f" A poda de pesos soltos em INT8, compactada, é menor ({tamanho(T.loc[soltos, 'bytes_com_zip'])}) e faz {pts(T.loc[soltos, 'exact_match'])}% de exact match; fora do disco ela volta a ocupar {tamanho(T.loc[soltos, 'bytes'])} (Etapa 8).",
    f"**O teto.** Com os {n(TODOS)} rótulos humanos, a mesma rede comprimida do mesmo jeito fica em {pts(T.loc['teto/int8', 'exact_match'])}%, com o tamanho e a latência do modelo final. "
    f"A distância vertical entre as duas trilhas é o que os rótulos humanos de treino compram (comparação c).",
    "**Como ler a fronteira.** Ela usa os valores medidos, sem os intervalos. Dois sistemas vizinhos nela podem não se distinguir em qualidade; quem diz isso são as comparações pareadas das cadeias e de ponta a ponta.",
]
display(Markdown("**Leitura.**\n\n" + "\n".join(f"- {l}" for l in leitura_da_fronteira)))
''')

md(r'''
## 8. Cada passo e o seu artigo

Cada técnica usada no caminho, com o artigo que a sustenta. O quadro está em `etapa9.ARTIGOS`.

A regra: só entra artigo que já estava citado no `PLANO.md` ou no `RESUMO.md` de alguma entrega. Três referências do método de avaliação estavam só nos notebooks das Etapas 2 e 4; elas entram marcadas, depois de o título e os autores terem sido conferidos na página de cada uma. O ano é o do identificador do arXiv (os quatro primeiros dígitos são ano e mês), ou o do evento.
''')

code(r'''
quadro = etapa9.quadro_de_artigos()
de_fora = etapa9.artigos_fora_das_fontes()
conferidos = [f"{a['quem']} ({a.get('arxiv') or a.get('veiculo')})" for a in etapa9.ARTIGOS if a.get("conferido")]
artigos_conferem = set(de_fora) == set(conferidos)
tabela_de_artigos = pd.DataFrame({
    "Passo": quadro["passo"], "Técnica": quadro["tecnica"],
    "Artigo": [q + (f", *{a}*" if a else "") for q, a in zip(quadro["quem"], quadro["apelido"])], "Ano": quadro["ano"],
    "Onde": [f"[{o}]({e})" for o, e in zip(quadro["onde"], quadro["endereco"])],
    "Estava citado em": ["notebook da etapa (conferido agora)" if c else "PLANO.md ou RESUMO.md" for c in quadro["conferido_nesta_etapa"]],
})
display(Markdown(em_markdown(tabela_de_artigos)))
display(Markdown(f"""
**Leitura.** {len(quadro)} linhas, {quadro['endereco'].nunique()} artigos. Fora do `PLANO.md` e dos `RESUMO.md`: {len(de_fora)} ({'; '.join(de_fora)}).
{"São exatamente os marcados como conferidos nesta etapa." if artigos_conferem else "**Há artigo no quadro que não está nas fontes nem foi conferido.**"}
Do preprint arXiv 2611.00464 o projeto leu só o resumo (`PLANO.md`): ele entra na discussão da ordem das etapas, não como apoio de um resultado.
"""))
''')

md(r'''
## 9. Rótulos humanos em cada ponto, limites e o que fica em aberto

O cenário do projeto é: há **poucos rótulos humanos**, os que a LLM vê no prompt, e o resto vem da LLM. A tabela abaixo diz, ponto por ponto do caminho, onde entrou rótulo humano, de treino ou de `dev`. É a conta que um leitor precisa para saber o que o resultado promete e o que não promete.
''')

code(r'''
temperaturas = pd.read_csv(ENTREGAS / "06_destilacao" / "temperaturas.csv")
T_B2 = decisao6["temperatura_do_b2"]
# o professor fechado contra o teto, como a Etapa 5a gravou (professor − rede do zero com todos os rótulos)
professor_contra_teto = etapa9.linha_gravada(pd.read_csv(ENTREGAS / "05a_professor_fechado" / "comparacoes.csv"), comparacao=f"test · Rede do zero com {n(TODOS)} rótulos", metrica="exact_match")
calibracao = decisao8["calibracao_da_estatica"]["pedidos"]
# O B2 na temperatura escolhida: em quantas sementes o melhor ponto foi a última época do treino
historicos = [pd.read_csv(ENTREGAS / "06_destilacao" / "historicos" / f"{decisao6['alunos']['B2']['variante']}_semente_{s}.csv") for s in etapa9.SEMENTES]
pararam_no_fim = sum(int(h.loc[h["melhor"], "epoca"].iloc[-1] == h["epoca"].max()) for h in historicos)
pelo_professor = int(temperaturas.loc[temperaturas["intencao_igual_a_do_professor"].idxmax(), "temperatura"])   # a temperatura que a concordância com o professor escolheria
memoria_da_placa = re.search(r"GPU tem (\d+) GB", (RAIZ / "README.md").read_text(encoding="utf-8"))
ano_do_massive = etapa9.ano_do_arxiv(re.search(r"\d{4}\.\d{5}", ler_json("01_dados", "manifesto.json")["referencia"]).group())

rotulos_no_caminho = pd.DataFrame([
    ("Etapas 4 e 4b · o teto e a receita de treino", f"{n(TODOS)}: o teto treina com todos", f"época de parada do teto e escolha da receita (suavização de rótulos e média móvel dos pesos), que os alunos herdam"),
    (f"Etapa 5a · {PROFESSOR}", f"{n(POUCOS)} no prompt ({K} por pedido, os mais parecidos)", "nenhuma escolha: a configuração não teve piloto"),
    (f"Etapa 5b · {ABERTO}", f"{n(POUCOS)} no prompt ({K} por pedido)", f"escolha entre {len(escolha5b['candidatos'])} modelos candidatos e acerto do formato da resposta"),
    ("Etapa 6 · aluno A", f"nenhum: aprende com os {n(TODOS)} pedidos rotulados pela LLM", "nenhuma no treino (a parada usa o dev rotulado pelo professor); a regra de quem segue para a poda"),
    ("Etapa 6 · aluno B2", "nenhum: rótulos e logits da LLM aberta", f"a temperatura (T = {T_B2}, entre {len(temperaturas)} tentativas) e a regra de quem segue para a poda"),
    ("Etapa 6 · piso", f"{n(POUCOS)} no treino", "época de parada"),
    ("Etapa 7 · poda e reajuste", "nenhum: o reajuste usa o sinal do professor", "a regra do que segue para a quantização (qual poda fica)"),
    ("Etapa 8 · quantização", f"nenhum: a calibração usa {n(calibracao)} pedidos do treino, sem rótulo", "a regra da forma de 8 bits e a do treino ciente de quantização"),
    ("Etapa 9 · este comparativo", "nenhum", "nenhuma decisão"),
], columns=["Ponto do caminho", f"Rótulos humanos de treino (de {n(TODOS)})", f"Decisões tomadas com os rótulos humanos de dev ({n(N_DEV)} pedidos)"])
display(rotulos_no_caminho.set_index("Ponto do caminho"))

poda_nas_duas = etapa9.poda_em_dev_e_no_teste()
discordam = poda_nas_duas[poda_nas_duas["passa_em_dev"] & poda_nas_duas["abaixo_no_teste"]]
curto_do_modelo = {fichas[m]["modelo_da_poda"]: T.loc[m, "curto"] for m in MODELOS}
display(poda_nas_duas.assign(modelo=poda_nas_duas["modelo"].map(curto_do_modelo), **{
    "em dev": [f"{dif(d)} ({dif(a)} a {dif(b)})" for d, a, b in zip(poda_nas_duas["dev_diferenca"], poda_nas_duas["dev_ic_min"], poda_nas_duas["dev_ic_max"])],
    "no teste": [f"{dif(d)} ({dif(a)} a {dif(b)})" for d, a, b in zip(poda_nas_duas["test_diferenca"], poda_nas_duas["test_ic_min"], poda_nas_duas["test_ic_max"])],
})[["modelo", "variante", "papel", "em dev", "no teste", "passa_em_dev", "abaixo_no_teste"]])

# A mesma latência, medida em três sessões
eficiencia8 = pd.read_csv(ENTREGAS / "08_quantizacao" / "eficiencia.csv")
variante_do_aluno = fichas[PRINCIPAL]["modelo_da_poda"]
medidas_do_aluno = {
    "Etapa 6": etapa9.linha_gravada(pd.read_csv(ENTREGAS / "06_destilacao" / "resultados.csv"), variante=variante_do_aluno, particao="test")["latencia_p50_ms"],
    "Etapa 7": decisao7["modelos"][variante_do_aluno]["sem_poda"]["latencia_p50_ms"],
    "Etapa 8": etapa9.linha_gravada(eficiencia8, modelo=variante_do_aluno, variante="sem_poda", forma="pytorch")["latencia_p50_ms"],
}
frase_da_poda = "; ".join(f"{curto_do_modelo[l.modelo]}, {l.variante.replace('_', ' ')}% ({dif(l.dev_diferenca)} ({dif(l.dev_ic_min)} a {dif(l.dev_ic_max)}) em dev e {dif(l.test_diferenca)} ({dif(l.test_ic_min)} a {dif(l.test_ic_max)}) no teste)" for l in discordam.itertuples())
final_na_poda = poda_nas_duas[(poda_nas_duas["modelo"] == variante_do_aluno) & (poda_nas_duas["papel"] == "principal")].iloc[0]

limites = [
    f"**Rótulos humanos de `dev` nas decisões.** O cenário promete {n(POUCOS)} rótulos humanos. No treino isso vale: nenhum aluno viu um rótulo humano de treino. Mas os {n(N_DEV)} rótulos humanos de `dev` decidiram a temperatura do B2 (Etapa 6), quem segue para a poda (Etapa 6), qual poda fica (Etapa 7) e a forma de 8 bits (Etapa 8). "
    f"Num uso real com só {n(POUCOS)} rótulos, essas escolhas teriam de ser feitas de outro jeito, por exemplo pela concordância com o professor. Para a temperatura, esse critério escolheria T = {pelo_professor} "
    f"({'a mesma' if pelo_professor == T_B2 else 'outra'} escolha, pela grade gravada na Etapa 6); para a poda e a quantização, isso não foi testado.",
    f"**A rede e a receita vieram do teto.** As variantes de arquitetura (Etapa 4) e a receita de treino (Etapa 4b) foram comparadas treinando com os {n(TODOS)} rótulos humanos e medindo no `dev` humano. Os alunos herdam essas escolhas.",
    f"**O aluno herda o teto do professor.** O modelo final fica {com_intervalo(em('a'))} pontos de exact match em relação ao {PROFESSOR}, e {com_intervalo(em('c'))} em relação ao teto; o próprio professor fica {com_intervalo(professor_contra_teto)} em relação ao teto (Etapa 5a). "
    f"Um professor fechado maior, com raciocínio ligado ou com mais exemplos poderia subir esse limite; a configuração do professor não teve piloto (Etapa 5a).",
    f"**O professor aberto é o que coube na placa.** Foram {len(escolha5b['candidatos'])} candidatos em 4 bits, numa placa de {memoria_da_placa.group(1)} GB ({maquina['gpu']}); o escolhido ocupa {num(T.loc['qwen', 'memoria_na_placa_gb'], 1)} GB. "
    f"Um modelo aberto maior não foi testado. A diferença entre os finais dos dois cenários é de {com_intervalo(em('f'))} em exact match (A − B).",
    f"**A temperatura do B2 está na borda da grade.** T = {T_B2} é {'a maior' if T_B2 == temperaturas['temperatura'].max() else 'uma'} das {len(temperaturas)} testadas ({', '.join(str(int(t)) for t in temperaturas['temperatura'])}), "
    f"e em {pararam_no_fim} das {SEMENTES} sementes o melhor ponto do treino foi a última época. Uma temperatura maior ou mais épocas poderiam dar outro B2 (Etapa 6).",
    f"**`dev` e teste discordam em parte das escolhas da poda.** A regra da Etapa 7 aceita a variante que não fica distinguivelmente abaixo do modelo sem poda em `dev`. Das {len(poda_nas_duas)} variantes que seguiram, {len(discordam)} passam em `dev` e ficam abaixo no teste: {frase_da_poda}. "
    f"A do modelo final {'não está' if not final_na_poda['abaixo_no_teste'] else 'está'} entre elas ({dif(final_na_poda['test_diferenca'])} ({dif(final_na_poda['test_ic_min'])} a {dif(final_na_poda['test_ic_max'])}) no teste). "
    f"A regra aceita o que não se distingue, e com o tamanho de `dev` uma queda pequena pode passar (limite declarado na Etapa 7).",
    f"**A latência é de uma CPU, de uma semente e de pedidos curtos.** {maquina['cpu']}, 1 thread, semente 0, pedidos de poucas palavras. A mediana variou até {num(100 * FOLGA, 0)}% entre as rodadas da Etapa 8, e o mesmo modelo, medido em sessões diferentes, deu "
    + ", ".join(f"{ms(v)} ({e})" for e, v in medidas_do_aluno.items()) + f" ({T.loc[PRINCIPAL, 'curto']} sem poda, PyTorch). Em outra CPU, sem as instruções VNNI, as respostas do modelo em 8 bits podem mudar um pouco (Etapa 8).",
    f"**As três condições de latência não são a mesma medida.** A da API inclui a rede e a fila do provedor, num horário, de uma rede; a da LLM aberta é de uma GPU; a das redes, de uma thread de CPU. A razão entre elas diz o que muda para quem usa o sistema, não quanto cálculo cada um faz.",
    f"**O custo em dólar existe só para a API**, pelo preço de lista de {config5a['gerado_em'][8:10]}/{config5a['gerado_em'][5:7]}/{config5a['gerado_em'][:4]}. Dos sistemas locais não se mediu energia nem custo de máquina; rotular o treino com o professor fechado custou US$ {num(custo5a['partes']['Lote de treino']['dolares'], 2)}, uma vez.",
    f"**O teste foi aberto em todas as etapas desde a 4.** Nenhuma decisão saiu dele, mas são os mesmos {n(N_TESTE)} pedidos medindo tudo, e aqui há {total_pp} comparações de ponta a ponta sem correção para comparações múltiplas.",
    f"**Um conjunto, uma língua, uma rotulagem.** MASSIVE, recorte pt-PT; uma rotulagem do treino por professor; {SEMENTES} sementes por modelo treinado, que dão uma ideia da variação do treino, não uma medida precisa. As LLMs podem conhecer o MASSIVE, que é público desde {ano_do_massive} (Etapas 5a e 5b).",
    f"**O que foi refeito aqui.** As respostas dos sistemas em INT8 e da linha de base com {n(POUCOS)} rótulos não estavam gravadas e foram produzidas agora; as métricas delas são iguais às gravadas pelas etapas de origem (`conferencia.csv`). "
    f"Da linha de base com {n(POUCOS)} rótulos não há tamanho nem latência. O tamanho do {ABERTO} é o dos arquivos baixados, não o do modelo em 4 bits.",
    "**A fronteira usa os valores medidos**, sem os intervalos: serve para ver a ordem de grandeza do custo, não para separar sistemas vizinhos.",
]
em_aberto = [
    "**Etapa 10 · encoder pré-treinado.** Repetir a destilação e a compressão partindo de um encoder pronto (o `PLANO.md` aponta o XLM-R base, com linha publicada no teste pt-PT). Depende de escolher o encoder.",
    f"**Um professor fechado melhor.** O modelo final fica {com_intervalo(em('a'))} em relação ao professor, e o professor, {com_intervalo(professor_contra_teto)} em relação ao teto. Falta medir quanto um professor maior, ou a mesma LLM com outra configuração, move o aluno.",
    f"**Um professor aberto maior**, numa máquina com mais de {memoria_da_placa.group(1)} GB de placa, e uma grade de temperatura além de T = {T_B2}.",
    f"**Decidir sem rótulos humanos de `dev`.** Refazer as regras das Etapas 6, 7 e 8 usando só a concordância com o professor, e ver se as escolhas mudam.",
    "**Latência em outra máquina** (outra CPU, um processador ARM, um celular), com mais de uma semente, e a memória em uso, que não foi medida.",
]
display(Markdown("### Limites\n\n" + "\n".join(f"- {l}" for l in limites) + "\n\n### O que fica em aberto\n\n" + "\n".join(f"- {l}" for l in em_aberto)))
''')

md(r'''
## 10. Entrega
''')

code(r'''
# 1. As três tabelas da etapa
T.reset_index().to_csv(PASTA / "tabela_final.csv", index=False)
CAD.to_csv(PASTA / "cadeias.csv", index=False)
PP.to_csv(PASTA / "comparacoes.csv", index=False)

# 2. A conferência, o quadro de artigos e o que foi produzido agora
pd.concat([confere_qualidade.assign(o_que="qualidade de um sistema"), confere_degraus.assign(o_que="degrau de uma cadeia")], ignore_index=True).to_csv(PASTA / "conferencia.csv", index=False)
quadro.to_csv(PASTA / "artigos.csv", index=False)
(PASTA / "medidas_novas.json").write_text(json.dumps(medidas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for chave in geradas:   # as respostas que não estavam em nenhuma entrega
    for caminho in fichas[chave]["previsoes"]:
        (PASTA / "previsoes").mkdir(exist_ok=True)
        shutil.copy2(caminho, PASTA / "previsoes" / caminho.name)

FIGURAS = ["fronteira_latencia", "fronteira_tamanho", "cadeia_cenario_a", "cadeia_cenario_b", "cadeia_teto", *FIGURAS_PP.values()]
print("Arquivos gravados:", ", ".join(sorted(p.name for p in PASTA.iterdir() if p.is_file())))
print("Figuras:", ", ".join(f for f in FIGURAS if (PASTA / "figuras" / f"{f}.png").exists()))
''')

md(r'''
### Critério de pronto
''')

code(r'''
ESPERADOS = ["base_299", "base_11514", "claude", "qwen", "teto", "piso", "aluno_a", "aluno_b1", "aluno_b2"] + [f"{m}/{s}" for m in ("aluno_a", "aluno_b2", "teto") for s in ("podado", "onnx", "int8", "soltos")]
qualidade_completa = T[[f"{m}{s}" for m in qualidade.METRICAS for s in ("", "_ic_min", "_ic_max")]].notna().all().all()
condicao_dita = bool((T["latencia_p50_ms"].isna() == (T["onde_foi_medida"] == "não medida")).all() and T.loc[T["latencia_p50_ms"].notna(), "condicao_da_latencia"].isin([API, GPU, CPU]).all())
so_cpu = all(v["como"] in ("onnx", "linha_de_base") for v in medidas["respostas_geradas"].values()) and set(medidas["respostas_geradas"]) == set(geradas)
cadeias_completas = bool(CAD[["exact_match", "latencia_p50_ms"]].notna().all().all() and CAD.loc[~CAD["chave"].isin(["claude", "qwen"]), "bytes"].notna().all()
                         and passos[[f"{m}_{c}" for m in qualidade.METRICAS for c in ("diferenca", "dif_ic_min", "dif_ic_max")]].notna().all().all())
pronto = pd.Series({
    f"A tabela final tem uma linha para cada sistema previsto ({len(ESPERADOS)}), com qualidade e intervalo de 95% no teste, e diz a condição da latência em cada linha":
        bool(list(T.index) == ESPERADOS and qualidade_completa and condicao_dita),
    f"Nada foi treinado nem medido na GPU ou na API; o que foi produzido agora está listado ({len(geradas)} sistemas respondidos na CPU e o tamanho de um modelo em disco)": bool(so_cpu),
    f"Cada número de qualidade, recalculado das respostas, é igual ao que a etapa de origem gravou ({len(confere_qualidade)} linhas)": bool(confere_qualidade["confere"].all()),
    f"Cada cadeia tem qualidade, tamanho e latência em cada degrau, e a diferença pareada para o anterior é igual à gravada pela etapa de origem ({len(confere_degraus)} degraus)":
        bool(cadeias_completas and confere_degraus["confere"].all() and len(confere_degraus) == len(passos)),
    f"As comparações de ponta a ponta são as {len(etapa9.PONTA_A_PONTA)} da lista fixada antes, pareadas, com {n(etapa9.REAMOSTRAS)} sorteios; nenhuma exploratória":
        bool(list(dict.fromkeys(PP["letra"])) == [l for l, *_ in etapa9.PONTA_A_PONTA] and not PP["exploratoria"].any() and etapa9.REAMOSTRAS == 2000),
    f"As figuras foram gravadas ({len(FIGURAS)})": all((PASTA / "figuras" / f"{f}.png").exists() for f in FIGURAS),
    f"Todo artigo do quadro está no PLANO.md ou num RESUMO.md, ou foi conferido e está marcado ({len(quadro)} linhas, {len(conferidos)} conferidos)": bool(artigos_conferem),
    f"Os limites de cada etapa estão juntos ({len(limites)} limites, {len(em_aberto)} pontos em aberto)": bool(len(limites) >= 10 and len(em_aberto) >= 4),
}).map({True: "sim", False: "NÃO"}).rename("atendido").to_frame()
display(pronto)
assert (pronto["atendido"] == "sim").all()
''')

md(r'''
## 11. Resposta

O resumo abaixo é gerado a partir dos números lidos e calculados neste notebook e gravado em `entregas/09_comparativo/RESUMO.md`.
''')

code(r'''
resposta = (
    f"O modelo final ({T.loc[FINAL, 'sistema']}) faz no teste {pts(f['intent_acc'])}% de acurácia de intenção, {pts(f['slot_f1'])}% de F1 de entidades e {pts(f['exact_match'])}% de exact match, "
    f"usando {n(POUCOS)} rótulos humanos, num arquivo de {tamanho(f['bytes'])} que responde em {ms(f['latencia_p50_ms'])} numa thread de CPU. "
    f"Contra o professor ({PROFESSOR}: {pts(p['exact_match'])}% de exact match, {ms(p['latencia_p50_ms'])} pela API, US$ {num(p['custo_por_mil_usd'], 3)} por mil consultas), fica {frase('a')}. "
    + ("**Essa distância vem da destilação** " if so_na_destilacao else "A destilação responde por ") + f"({com_intervalo(destilacao_a, 'exact_match_')} de exact match no degrau professor → aluno); "
    f"poda, exportação e 8 bits, juntas, ficam em {com_intervalo(em('b'))}, e levam o arquivo de {tamanho(T.loc['aluno_a', 'bytes'])} para {tamanho(f['bytes'])} e a consulta de {ms(T.loc['aluno_a', 'latencia_p50_ms'])} para {ms(f['latencia_p50_ms'])}. "
    f"Contra o teto, a mesma rede com os {n(TODOS)} rótulos humanos, fica {com_intervalo(em('c'))}: é o que custa não ter os rótulos. "
    f"Contra a linha de base com os mesmos {n(POUCOS)} rótulos, {com_intervalo(em('g1'))}; contra a linha de base com os {n(TODOS)}, {com_intervalo(em('g2'))} em exact match e {com_intervalo(pp.loc[('g2', 'intent_acc')])} em intenção. "
    f"No cenário B, o aluno comprimido fica {em_relacao_a(em('e1')['ic_min'], em('e1')['ic_max'])} seu professor ({ABERTO}: {com_intervalo(em('e1'))}) e {em_relacao_a(-em('f')['ic_max'], -em('f')['ic_min'])} final do cenário A ({com_intervalo(em('f'))}, A − B). "
    f"No teto, onde não há professor, a mesma compressão deixa o modelo {em_relacao_a(em('d')['ic_min'], em('d')['ic_max'])} teto sem compressão: {com_intervalo(em('d'))}."
)

colunas_do_resumo = ["Sistema", "Rótulos humanos", "Acurácia de intenção", "F1 de entidades", "Exact match", "Parâmetros", "Tamanho em disco", "Com zip", "Latência p50", "Onde a latência foi medida", "Consultas por segundo", "US$ por 1.000 consultas"]
tabela_pronto = "\n".join(f"| {criterio} | {atendido} |" for criterio, atendido in pronto["atendido"].items())
secoes_das_cadeias = "\n\n".join(
    f"### {etapa9.CADEIAS[nome]['nome']}\n\n{em_markdown(tabela_da_cadeia(nome))}\n\n![{etapa9.CADEIAS[nome]['nome']}](figuras/cadeia_{'cenario_' + nome.lower() if nome != 'teto' else 'teto'}.png)\n\n"
    + "\n".join(f"- {frase_}" for frase_ in leituras_das_cadeias[nome])
    for nome in etapa9.CADEIAS
)
rotulos_md = em_markdown(rotulos_no_caminho)

resumo = f"""# Etapa 9 · Comparativo: resumo da entrega

**Pergunta.** Somando tudo: o que cada etapa do caminho custou e rendeu, e onde fica o modelo final em relação aos professores, ao teto e às linhas de base?

**Resposta.** {resposta}

Nada foi treinado nesta etapa. Todos os números de qualidade vêm dos arquivos das Etapas 2 a 8 e foram recalculados das respostas gravadas: {len(confere_qualidade)} linhas de qualidade e {len(confere_degraus)} degraus, todos iguais ao que as etapas de origem gravaram.

## A tabela final (teste, {n(N_TESTE)} pedidos)

Qualidade em % (intervalo de 95%); nos sistemas treinados, média de {SEMENTES} sementes. Tamanho e latência dos modelos comprimidos são os da semente 0.

{em_markdown(mostra[colunas_do_resumo])}

{chr(10).join(f"- {nota}" for nota in notas_da_tabela)}

## O que cada etapa custou e rendeu

Em cada cadeia, de cima para baixo, muda uma coisa por degrau. As diferenças são pareadas, em pontos percentuais (intervalo de 95%), de cada degrau para o anterior.

{secoes_das_cadeias}

**Nas três cadeias.** {leitura_das_cadeias}

## De ponta a ponta

Comparações fixadas antes de calcular. Diferença pareada B − A, no teste, em pontos percentuais (intervalo de 95%); uma diferença só vale quando o intervalo não contém zero.

{em_markdown(tabela_pp)}

![De ponta a ponta, exact match](figuras/ponta_a_ponta_exact_match.png)

![De ponta a ponta, intenção](figuras/ponta_a_ponta_intencao.png)

![De ponta a ponta, entidades](figuras/ponta_a_ponta_entidades.png)

{chr(10).join(f"- {l}" for l in leituras_pp)}
- {nota_do_acaso}

## As fronteiras

![Qualidade contra latência](figuras/fronteira_latencia.png)

![Qualidade contra tamanho](figuras/fronteira_tamanho.png)

{chr(10).join(f"- {l}" for l in leitura_da_fronteira)}

## Cada passo e o seu artigo

{em_markdown(tabela_de_artigos)}

- Só entram artigos já citados no `PLANO.md` ou num `RESUMO.md`. As exceções ({'; '.join(de_fora)}) estavam nos notebooks das Etapas 2 e 4 e tiveram título e autores conferidos na página de cada uma.
- O ano é o do identificador do arXiv, ou o do evento.
- Do preprint arXiv 2611.00464 o projeto leu só o resumo.

## Rótulos humanos em cada ponto do caminho

{rotulos_md}

## Limites

{chr(10).join(f"- {l}" for l in limites)}

## O que fica em aberto

{chr(10).join(f"- {l}" for l in em_aberto)}

## Critério de pronto

| Critério | Atendido |
|---|---|
{tabela_pronto}

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `tabela_final.csv` | Uma linha por sistema, no teste: qualidade com intervalo, rótulos humanos, parâmetros, tamanho, latência e a condição da medida, consultas por segundo e custo |
| `cadeias.csv` | Cada degrau de cada cadeia: qualidade, tamanho, latência e a diferença pareada para o degrau anterior, em todas as métricas |
| `comparacoes.csv` | As comparações de ponta a ponta, B − A, em todas as métricas |
| `conferencia.csv` | Cada número recalculado aqui contra o que a etapa de origem gravou |
| `artigos.csv` | O quadro de artigos |
| `medidas_novas.json` | O que foi produzido nesta etapa por não estar gravado: de que modelos saíram as respostas e o tamanho em disco do professor aberto |
| `previsoes/` | As respostas no teste produzidas nesta etapa: os sistemas em INT8 e a linha de base com {n(POUCOS)} rótulos |
| `figuras/` | Os gráficos deste notebook |
| `09_comparativo.html` | Este notebook executado, para leitura |
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
