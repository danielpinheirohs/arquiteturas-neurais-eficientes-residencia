"""Monta notebooks/05a_professor_fechado.ipynb a partir das células abaixo."""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Etapa 5a · O professor fechado: uma LLM que só se alcança pela API

| | |
|---|---|
| **Pergunta** | Quão bem uma LLM fechada resolve a tarefa quando tudo o que se tem dela é o texto da resposta, quanto custa e quanto demora, e que rótulos ela deixa para o aluno? |
| **Entrada** | `entregas/01_dados/` (os dados), `entregas/02_avaliacao/` (linha de base), `entregas/04b_intencao/` (a rede do zero treinada com rótulos humanos) e as respostas da API, gravadas pelos comandos de `especialista/experimentos/etapa5a.py`. |
| **Entrega** | `entregas/05a_professor_fechado/`: o treino rotulado pela LLM, as respostas dela em `dev` e teste, as respostas brutas da API, latência e custo medidos. |
| **Pronto quando** | (1) todo pedido enviado tem uma resposta gravada, e as falhas estão contadas; (2) o professor está medido na régua, com intervalo, e comparado de forma pareada com as referências; (3) latência e custo vêm de medida, não de estimativa; (4) o treino rotulado está gravado no formato que a destilação lê. |

**O que é "fechada".** A LLM roda no provedor. Não há pesos, nem a probabilidade que ela dá a cada rótulo: só o texto da resposta. É o cenário de **caixa-preta** da destilação, em que o aluno aprende com rótulos rígidos (Etapa 6, aluno A). O cenário de caixa-branca, com uma LLM aberta rodando nesta máquina, é a Etapa 5b.

**A LLM.** Claude Haiku 5.5, o modelo mais barato da família, escolhido pelo custo. Não houve piloto para comparar jeitos de chamá-la: a configuração é a que estava definida de antemão, e isso entra nos limites da etapa.

**Artigos de apoio**

- Rótulo do professor como alvo do aluno: Hinton et al., *Distilling the Knowledge in a Neural Network* ([arXiv 1503.02531](https://arxiv.org/abs/1503.02531)).
- Destilação de LLMs, caixa-preta e caixa-branca: Xu et al., *A Survey on Knowledge Distillation of Large Language Models* ([arXiv 2402.13116](https://arxiv.org/abs/2402.13116)).
- O workflow que serve de esqueleto ao projeto: Ni et al., *From Large to Super-Tiny* ([arXiv 2504.13471](https://arxiv.org/abs/2504.13471)).
- Escolha dos exemplos do prompt pelos vizinhos mais próximos: Liu et al., *What Makes Good In-Context Examples for GPT-3?* ([arXiv 2101.06804](https://arxiv.org/abs/2101.06804)).
- LLMs em intenção e entidades: He & Garner, *Can ChatGPT Detect Intent? Evaluating Large Language Models for Spoken Language Understanding* ([arXiv 2305.13512](https://arxiv.org/abs/2305.13512)).
- Latência medida uma consulta por vez: Reddi et al., *MLPerf Inference Benchmark* ([arXiv 1911.02549](https://arxiv.org/abs/1911.02549)).

**Roteiro**

1. O cenário e o pedido
2. O que voltou da API
3. O professor na régua
4. Onde o professor erra
5. Os rótulos que o aluno vai receber
6. Latência
7. Custo
8. Entrega
9. Resposta
''')

code(r'''
import json
import shutil
from collections import Counter
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, display

from especialista import visual
from especialista.avaliacao import eficiencia, qualidade, resultados
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.bio import spans_para_bio
from especialista.caminhos import ENTREGAS, RAIZ, entrega
from especialista.dados import massive
from especialista.experimentos import etapa5a
from especialista.modelos.linha_de_base import LinhaDeBase
from especialista.professores import claude, resposta
from especialista.professores.claude import Ajuste, ProfessorClaude
from especialista.professores.demonstracoes import Demonstracoes, montar_conjunto
from especialista.professores.rotulagem import ler_conjunto, rotular

visual.aplicar_estilo()
pd.set_option("display.max_colwidth", 200)

PASTA = entrega("05a_professor_fechado")
SEMENTES = (0, 1, 2)
PROFESSOR = "Claude Haiku 5.5"

particoes, rotulos = massive.carregar_entrega()
treino, dev, teste = (particoes[p] for p in ("train", "dev", "test"))
intents, tipos = rotulos["intents"], rotulos["slot_types"]
ouro_treino, ouro_dev, ouro_teste = ouro(treino), ouro(dev), ouro(teste)


def n(x):
    return f"{int(x):,}".replace(",", ".")


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def pts(x, casas=1):
    return num(100 * x, casas)


def dif(x):
    return f"{100 * x:+.1f}".replace(".", ",").replace("-", "−")


def usd(x, casas=2):
    return "US$ " + num(x, casas)


def com_intervalo(linha):
    return f"{dif(linha['diferenca'])} ({dif(linha['ic_min'])} a {dif(linha['ic_max'])})"


def valor_com_intervalo(linha):
    return f"{pts(linha['valor'])} ({pts(linha['ic_min'])} a {pts(linha['ic_max'])})"


def posicao(linha):
    """Onde B fica em relação a A numa comparação pareada B − A."""
    return "acima" if linha["ic_min"] > 0 else "abaixo" if linha["ic_max"] < 0 else "sem diferença distinguível"
''')

md(r'''
## 1. O cenário e o pedido

**O cenário.** Uma equipe tem um assistente virtual, muitos pedidos de usuários guardados e **poucos** deles rotulados por pessoas. Ela quer um modelo pequeno e não tem como rotular tudo à mão. A LLM entra como rotuladora: vê os poucos exemplos rotulados e rotula os muitos.

No projeto, isso fica assim:
''')

code(r'''
conjunto_demo = montar_conjunto(treino)
demos = Demonstracoes(conjunto_demo)
ajuste = Ajuste(**json.loads((etapa5a.pasta_das_respostas("train") / "ajuste.json").read_text(encoding="utf-8")))
professor = ProfessorClaude(demos, rotulos, ajuste)   # aqui só monta pedidos: nenhuma chamada sai deste notebook

por_intencao = conjunto_demo["intent"].value_counts()
pd.Series({
    "Pedidos de treino": n(len(treino)),
    "Rótulos humanos que a LLM vê (conjunto de demonstração)": f"{n(len(conjunto_demo))} ({pts(len(conjunto_demo) / len(treino))}% do treino)",
    "Exemplos por intenção no conjunto de demonstração": f"{por_intencao.max()} (mínimo {por_intencao.min()}, em intenções com menos de {por_intencao.max()} exemplos no treino)",
    "Exemplos que entram em cada pedido": f"os {ajuste.k} mais parecidos com o pedido",
    "Pedidos que a LLM rotula": f"{n(len(treino))} de treino, {n(len(dev))} de dev e {n(len(teste))} de teste",
}).rename("").to_frame()
''')

md(r'''
Os rótulos humanos de `dev` e de teste nunca entram em prompt nenhum: servem só para medir. Ao rotular o treino, um pedido que pertence ao conjunto de demonstração nunca recebe a si mesmo como exemplo.

**O pedido.** Cada chamada tem duas partes. As **instruções** são fixas: a tarefa, as regras e as listas de 60 intenções e 55 tipos de entidade. A **mensagem** muda a cada pedido: os exemplos resolvidos mais parecidos e, no fim, o pedido a resolver. Abaixo, um pedido de `dev` como ele foi enviado.
''')

code(r'''
exemplo = dev.iloc[50]
pedido = professor.pedido(exemplo.utt)
instrucoes = pedido["system"][0]["text"]
regras, listas = instrucoes.split("\n\nIntenções:\n")
print("INSTRUÇÕES (fixas, guardadas em cache pelo provedor)\n")
print(regras)
print(f"\nIntenções:\n{chr(10).join(listas.splitlines()[:3])}\n... ({len(intents)} intenções e {len(tipos)} tipos de entidade, um por linha)")
''')

code(r'''
print("MENSAGEM (muda a cada pedido)\n")
print(pedido["messages"][0]["content"])
print("\nRESPOSTA DA LLM\n")
print(etapa5a.respostas("dev")[f"dev-{exemplo.id}"]["texto"])
''')

md(r'''
A LLM devolve o **texto** de cada entidade, não a posição dela. Achar a posição no pedido é trabalho do projeto (`professores/resposta.py`), com uma regra só: procurar a sequência de palavras no pedido, em ordem; o que não for encontrado é descartado e conta como erro.

**A configuração da chamada**
''')

code(r'''
pd.Series({
    "Modelo": pedido["model"],
    "Formato da resposta": "JSON imposto na geração por um esquema: a intenção e os tipos só podem ser os das listas",
    "Raciocínio antes de responder": "ligado" if ajuste.raciocinio else "desligado",
    "Esforço": ajuste.esforco,
    "Temperatura": "não existe neste modelo: a API recusa o parâmetro",
    "Limite da resposta": f"{pedido['max_tokens']} tokens",
    "Cache": "as instruções fixas são marcadas para cache",
    "Como os pedidos foram enviados": "dev, teste e treino em lote (metade do preço, sem pressa); latência com chamadas diretas, uma por vez",
}).rename("").to_frame()
''')

md(r'''
**Duas leituras combinadas antes de olhar os números.**

1. **O professor acrescenta algo aos 299 rótulos?** A referência é a linha de base da Etapa 2 treinada **só com os 299 exemplos** que a LLM vê. Se, em `dev`, o exact match do professor não ficar acima dela com intervalo de 95% todo acima de zero, a LLM não traz nada além do que os rótulos já davam, e destilar dela não se justifica.
2. **Onde o professor fica em relação aos 11.514 rótulos humanos?** A referência é a rede do zero da Etapa 4b. Não é critério de aprovação: diz se o aluno que aprender com o professor parte de um teto mais alto ou mais baixo do que o de quem tem todos os rótulos humanos.
''')

md(r'''
## 2. O que voltou da API

Cada resposta foi gravada em disco no momento em que chegou. Este notebook só lê esses arquivos.
''')

code(r'''
dados_de = {"dev": dev, "test": teste, "train": treino}
nome_de = {"dev": "dev", "test": "teste", "train": "treino"}
lidos = {}
for conjunto, dados in dados_de.items():
    previsoes, diagnostico, registros = ler_conjunto(etapa5a.respostas(conjunto), dados, conjunto, intents, tipos)
    lidos[conjunto] = {"previsoes": previsoes, "diagnostico": diagnostico, "registros": registros}
claude_dev, claude_teste, claude_treino = (lidos[c]["previsoes"] for c in ("dev", "test", "train"))

linhas = []
for conjunto, lido in lidos.items():
    d, registros = lido["diagnostico"], lido["registros"]
    linhas.append({
        "conjunto": nome_de[conjunto],
        "pedidos": len(registros),
        "sem registro": d["sem_registro"],
        "sem resposta (erro, recusa ou corte)": d["sem_resposta"] - d["sem_registro"],
        "JSON ilegível": d["json_invalido"],
        "intenção fora da lista": d["intencao_fora_da_lista"],
        "entidades devolvidas": d["entidades"],
        "tipo fora da lista": d["tipo_fora_da_lista"],
        "texto que não está no pedido": d["texto_nao_encontrado"],
    })
leitura = pd.DataFrame(linhas).set_index("conjunto")
leitura.loc["total"] = leitura.sum()
paradas = Counter(r["parada"] or f"erro: {r['erro']}" for lido in lidos.values() for r in lido["registros"])
display(leitura.map(n))
print("Como cada resposta terminou:", ", ".join(f"{motivo}: {n(q)}" for motivo, q in paradas.most_common()))
''')

code(r'''
total = leitura.loc["total"]
sem_uso = total["sem registro"] + total["sem resposta (erro, recusa ou corte)"] + total["JSON ilegível"]
perdidas = total["tipo fora da lista"] + total["texto que não está no pedido"]

fora_do_pedido = []
for linha, registro in zip(dev.itertuples(), lidos["dev"]["registros"]):
    if resposta.ler(registro["texto"], linha.utt, intents, tipos)[1]["texto_nao_encontrado"]:
        fora_do_pedido.append({"pedido": linha.utt, "resposta da LLM": registro["texto"]})

if sem_uso == 0:
    frase_das_falhas = f"Os {n(total['pedidos'])} pedidos voltaram com uma resposta aproveitável: nenhuma falha, recusa ou resposta cortada."
else:
    frase_das_falhas = (f"De {n(total['pedidos'])} pedidos, **{n(sem_uso)}** ficaram sem resposta aproveitável ({pts(sem_uso / total['pedidos'], 2)}%). "
                        "Na régua, cada um deles conta como erro.")
display(Markdown(f"""
**Leitura.**

- {frase_das_falhas}
- O esquema imposto na geração cumpriu o papel: {n(total['intenção fora da lista'])} intenções e {n(total['tipo fora da lista'])} tipos de entidade fora das listas.
- De {n(total['entidades devolvidas'])} entidades devolvidas, **{n(total['texto que não está no pedido'])}** ({pts(total['texto que não está no pedido'] / total['entidades devolvidas'], 2)}%) traziam um texto que não está no pedido palavra por palavra, e foram descartadas. É a perda própria de pedir texto em vez de posição.
"""))
if fora_do_pedido:
    print(f"Exemplos de dev em que o texto de alguma entidade não está no pedido ({len(fora_do_pedido)} no total):")
    display(pd.DataFrame(fora_do_pedido).head(5))
''')

md(r'''
## 3. O professor na régua

A mesma régua da Etapa 2, com os rótulos humanos como gabarito. Primeiro o professor sozinho, nas duas partições.
''')

code(r'''
medida_dev, medida_teste = qualidade.intervalo(ouro_dev, claude_dev), qualidade.intervalo(ouro_teste, claude_teste)
pd.DataFrame({
    "dev (intervalo de 95%)": [valor_com_intervalo(l) for _, l in medida_dev.iterrows()],
    "teste (intervalo de 95%)": [valor_com_intervalo(l) for _, l in medida_teste.iterrows()],
}, index=medida_dev.index).rename(index=qualidade.NOMES)
''')

md(r'''
Agora contra as referências. São três, e cada uma responde a uma pergunta diferente:

| Referência | Rótulos humanos que usa | O que a comparação diz |
|---|---|---|
| Linha de base com 299 rótulos | os mesmos 299 que a LLM vê | o que a LLM acrescenta aos rótulos que recebeu |
| Linha de base (Etapa 2) | 11.514 | a LLM contra o método mais simples com todos os rótulos |
| Rede do zero (Etapa 4b), 3 sementes | 11.514 | a LLM contra o melhor modelo do projeto até aqui |
''')

code(r'''
base_299 = LinhaDeBase(caracteres=True).treinar(conjunto_demo)
base_299.nome = "Linha de base com 299 rótulos"
referencias = {}
for particao, dados in (("dev", dev), ("test", teste)):
    referencias[particao] = {
        "Linha de base com 299 rótulos": [base_299.prever(list(dados["utt"]))],
        "Linha de base com 11.514 rótulos": [carregar_previsoes(ENTREGAS / "02_avaliacao" / "previsoes" / f"linha_de_base_{particao}.parquet")[1]],
        "Rede do zero com 11.514 rótulos": [carregar_previsoes(ENTREGAS / "04b_intencao" / "previsoes" / f"semente_{s}_{particao}.parquet")[1] for s in SEMENTES],
    }
gabarito = {"dev": ouro_dev, "test": ouro_teste}
do_professor = {"dev": claude_dev, "test": claude_teste}
# sempre professor − referência, pareado: os mesmos sorteios do conjunto para os dois lados
contra = {p: {ref: qualidade.comparar_grupos(gabarito[p], grupo, [do_professor[p]]) for ref, grupo in referencias[p].items()} for p in ("dev", "test")}


def quadro(particao):
    colunas = {ref: tabela["a"].map(pts) for ref, tabela in contra[particao].items()}
    colunas[PROFESSOR] = next(iter(contra[particao].values()))["b"].map(pts)
    return pd.DataFrame(colunas).loc[list(qualidade.METRICAS)].rename(index=qualidade.NOMES)


print("Teste: valor de cada sistema (%)")
display(quadro("test"))
print("Teste: professor menos referência, em pontos percentuais (intervalo de 95%)")
pd.DataFrame({ref: [com_intervalo(l) for _, l in tabela.loc[list(qualidade.METRICAS)].iterrows()] for ref, tabela in contra["test"].items()},
             index=[qualidade.NOMES[m] for m in qualidade.METRICAS])
''')

code(r'''
rede_teste = qualidade.intervalo_do_grupo(ouro_teste, referencias["test"]["Rede do zero com 11.514 rótulos"])
fig, ax = visual.pontos_com_intervalo(
    {
        f"{PROFESSOR} (vê 299 rótulos)": {qualidade.NOMES[m]: tuple(100 * medida_teste.loc[m, c] for c in ("valor", "ic_min", "ic_max")) for m in qualidade.METRICAS},
        "Rede do zero (11.514 rótulos, 3 sementes)": {qualidade.NOMES[m]: tuple(100 * rede_teste.loc[m, c] for c in ("valor", "ic_min", "ic_max")) for m in qualidade.METRICAS},
    },
    "O professor e a rede do zero no teste", "% (ponto = valor, barra = intervalo de 95%)",
)
visual.salvar(fig, PASTA, "professor_na_regua")
plt.show()

fig, ax = visual.diferencas_com_intervalo(
    {ref: tuple(100 * tabela.loc["exact_match", c] for c in ("diferenca", "ic_min", "ic_max")) for ref, tabela in contra["test"].items()},
    "Exact match: o professor menos cada referência (teste)", "diferença em pontos percentuais (professor − referência)",
)
visual.salvar(fig, PASTA, "professor_menos_referencias")
plt.show()
''')

code(r'''
leitura_1 = contra["dev"]["Linha de base com 299 rótulos"].loc["exact_match"]
acrescenta = bool(leitura_1["ic_min"] > 0)
contra_rede = contra["test"]["Rede do zero com 11.514 rótulos"]
display(Markdown(f"""
**As duas leituras combinadas.**

1. **O professor acrescenta algo aos 299 rótulos?** {"Sim" if acrescenta else "Não"}. Em `dev`, o exact match do professor é {pts(leitura_1['b'])}% e o da linha de base treinada com os mesmos 299 rótulos é {pts(leitura_1['a'])}%: diferença de {com_intervalo(leitura_1)} pontos. {"O intervalo fica todo acima de zero." if acrescenta else "O intervalo não fica todo acima de zero."}
2. **Onde ele fica em relação aos 11.514 rótulos humanos?** No teste, contra a rede do zero: intenção {posicao(contra_rede.loc['intent_acc'])} ({com_intervalo(contra_rede.loc['intent_acc'])}), entidades {posicao(contra_rede.loc['slot_f1'])} ({com_intervalo(contra_rede.loc['slot_f1'])}), exact match {posicao(contra_rede.loc['exact_match'])} ({com_intervalo(contra_rede.loc['exact_match'])}).

**O que a literatura esperava.** He & Garner (arXiv 2305.13512) mediram LLMs nesta mesma família de tarefas: com poucos exemplos, as maiores chegam perto de modelos supervisionados em **intenção** e ficam atrás em **entidades**. É o mesmo desenho que aparece acima; a seção 4 mostra de onde vem a distância.
"""))
''')

md(r'''
## 4. Onde o professor erra

Esta análise é feita em `dev`, para o teste continuar servindo só de medida final. Ela importa para a Etapa 6: o aluno vai aprender com estes erros.

### 4.1 Intenção ou entidades?
''')

code(r'''
def acertos(gabarito, previsoes):
    intencao = np.array([p.intent == o.intent for o, p in zip(gabarito, previsoes)])
    entidades = np.array([set(p.slots) == set(o.slots) for o, p in zip(gabarito, previsoes)])
    return intencao, entidades


def reparticao(gabarito, previsoes):
    intencao, entidades = acertos(gabarito, previsoes)
    return pd.Series({
        "Tudo certo (exact match)": (intencao & entidades).mean(),
        "Só a intenção errada": (~intencao & entidades).mean(),
        "Só as entidades erradas": (intencao & ~entidades).mean(),
        "Intenção e entidades erradas": (~intencao & ~entidades).mean(),
    })


rede_dev = referencias["dev"]["Rede do zero com 11.514 rótulos"]
partes = pd.DataFrame({
    PROFESSOR: reparticao(ouro_dev, claude_dev),
    "Rede do zero (média de 3 sementes)": sum(reparticao(ouro_dev, p) for p in rede_dev) / len(rede_dev),
})
partes.map(lambda v: pts(v) + "%")
''')

md(r'''
### 4.2 O que acontece com cada entidade

Cada entidade dos rótulos humanos cai em um destino: o sistema acertou, errou o limite (o trecho começa ou acaba em outro lugar), errou o tipo, errou os dois, ou não devolveu nada ali. "A mais" são entidades que o sistema devolveu onde os rótulos humanos não têm nenhuma.
''')

code(r'''
destinos = pd.DataFrame({
    PROFESSOR: qualidade.erros_de_entidade(ouro_dev, claude_dev),
    "Rede do zero (média de 3 sementes)": sum(qualidade.erros_de_entidade(ouro_dev, p) for p in rede_dev) / len(rede_dev),
})
entidades_em_dev = destinos[PROFESSOR][list(qualidade.DESTINOS)].sum()
nomes_dos_destinos = {
    "certa": "Certa", "limite diferente": "Limite diferente", "tipo diferente": "Tipo diferente",
    "tipo e limite diferentes": "Tipo e limite diferentes", "faltou": "Faltou", "a mais": "A mais (não está nos rótulos)",
}
erros_do_professor = destinos[PROFESSOR].drop("certa").rename(index=nomes_dos_destinos)
fig, ax = visual.barras_ranqueadas(
    erros_do_professor, "Erros de entidade do professor em dev", f"entidades (os rótulos humanos de dev têm {n(entidades_em_dev)})",
    destacar={nome: n(valor) for nome, valor in erros_do_professor.items()},
)
visual.salvar(fig, PASTA, "erros_de_entidade")
plt.show()

destinos.rename(index=nomes_dos_destinos).assign(**{
    "% das entidades dos rótulos, professor": lambda t: (t[PROFESSOR] / entidades_em_dev).map(lambda v: pts(v) + "%"),
    "% das entidades dos rótulos, rede": lambda t: (t["Rede do zero (média de 3 sementes)"] / entidades_em_dev).map(lambda v: pts(v) + "%"),
}).assign(**{PROFESSOR: lambda t: t[PROFESSOR].map(n), "Rede do zero (média de 3 sementes)": lambda t: t["Rede do zero (média de 3 sementes)"].map(lambda v: n(round(v)))})
''')

code(r'''
por_tipo = qualidade.por_tipo_de_entidade(ouro_dev, claude_dev)
por_tipo_rede = qualidade.por_tipo_de_entidade(ouro_dev, rede_dev[0])
por_tipo["erros"] = (por_tipo["exemplos"] * (1 - por_tipo["recall"]) + por_tipo["previstas"] * (1 - por_tipo["precisao"])).round().astype(int)
piores_tipos = por_tipo.sort_values("erros", ascending=False).head(10)
print("Os 10 tipos de entidade com mais erros do professor em dev (entidades que faltaram mais entidades devolvidas erradas)")
pd.DataFrame({
    "nos rótulos humanos": piores_tipos["exemplos"].map(n),
    "devolvidas pelo professor": piores_tipos["previstas"].map(n),
    "precisão": piores_tipos["precisao"].map(pts),
    "recall": piores_tipos["recall"].map(pts),
    "F1 do professor": piores_tipos["f1"].map(pts),
    "F1 da rede (semente 0)": por_tipo_rede["f1"].reindex(piores_tipos.index).fillna(0).map(pts),
})
''')

md(r'''
Alguns pedidos de `dev` em que o professor acertou a intenção e errou as entidades. Em cada par, a primeira linha são os rótulos humanos e a segunda é a resposta do professor.
''')

code(r'''
intencao_ok, entidades_ok = acertos(ouro_dev, claude_dev)
so_entidades = np.flatnonzero(intencao_ok & ~entidades_ok)
escolhidos = np.random.default_rng(0).choice(so_entidades, size=min(6, len(so_entidades)), replace=False)
blocos = []
for i in sorted(escolhidos):
    linha = dev.iloc[i]
    tokens = list(linha.tokens)
    blocos.append(
        f'<div style="margin:0 0 14px 0;"><div style="font-size:12px;color:{visual.TINTA_SUAVE};">intenção: {linha.intent} · humano, depois professor</div>'
        + visual.html_bio(tokens, list(linha.tags))
        + visual.html_bio(tokens, spans_para_bio(list(claude_dev[i].slots), len(tokens)))
        + "</div>"
    )
HTML("".join(blocos))
''')

md(r'''
### 4.3 Intenções trocadas
''')

code(r'''
certa = np.array([o.intent for o in ouro_dev])
dada = np.array([p.intent for p in claude_dev])
trocas = pd.Series(Counter(zip(certa[certa != dada], dada[certa != dada]))).sort_values(ascending=False)
exemplos_da_troca = {par: dev["utt"].to_numpy()[(certa == par[0]) & (dada == par[1])][0] for par in trocas.index[:10]}
print(f"As 10 trocas de intenção mais frequentes do professor em dev ({n((certa != dada).sum())} erros de intenção no total)")
display(pd.DataFrame({
    "rótulo humano": [par[0] for par in trocas.index[:10]],
    "resposta do professor": [par[1] for par in trocas.index[:10]],
    "vezes": trocas.to_numpy()[:10],
    "um exemplo": [exemplos_da_troca[par] for par in trocas.index[:10]],
}))

f1_professor = qualidade.por_intencao(ouro_dev, claude_dev)
f1_rede = sum(qualidade.por_intencao(ouro_dev, p)["f1"] for p in rede_dev) / len(rede_dev)
piores = f1_professor.sort_values("f1").head(10)
print("As 10 intenções em que o professor vai pior em dev")
pd.DataFrame({
    "exemplos em dev": piores["exemplos"],
    "F1 do professor": piores["f1"].map(pts),
    "F1 da rede (média de 3 sementes)": f1_rede.reindex(piores.index).map(pts),
})
''')

md(r'''
### 4.4 Quanto o professor depende dos exemplos do prompt

Sem piloto, não há uma versão sem exemplos para comparar. O que dá para medir de graça: em quantos pedidos a intenção certa **aparece entre os 10 exemplos** do prompt, e como o professor se sai nos dois casos. A rede do zero entra como controle, porque pedidos sem exemplo parecido também tendem a ser os mais difíceis.
''')

code(r'''
intencoes_no_prompt = [{e["intent"] for e in demos.mais_parecidos(texto, ajuste.k)} for texto in dev["utt"]]
aparece = np.array([o.intent in vistas for o, vistas in zip(ouro_dev, intencoes_no_prompt)])
acerto_rede = np.mean([[p.intent == o.intent for o, p in zip(ouro_dev, previsoes)] for previsoes in rede_dev], axis=0)
dependencia = pd.DataFrame([
    {
        "pedidos de dev": nome,
        "quantos": n(m.sum()),
        "parte de dev": pts(m.mean()) + "%",
        "acurácia de intenção do professor": pts(intencao_ok[m].mean()),
        "acurácia de intenção da rede do zero": pts(acerto_rede[m].mean()),
    }
    for nome, m in (("A intenção certa aparece entre os exemplos", aparece), ("A intenção certa não aparece", ~aparece))
]).set_index("pedidos de dev")
dependencia
''')

md(r'''
## 5. Os rótulos que o aluno vai receber

O professor rotulou os 11.514 pedidos do treino. Como o treino tem rótulos humanos, dá para medir a qualidade desses rótulos diretamente. Em um caso real isso não existiria: só haveria a medida em uma amostra pequena, como `dev`.
''')

code(r'''
medida_treino = qualidade.intervalo(ouro_treino, claude_treino)
concordancia = pd.DataFrame({
    "treino (os rótulos do aluno)": [valor_com_intervalo(l) for _, l in medida_treino.iterrows()],
    "dev": [valor_com_intervalo(l) for _, l in medida_dev.iterrows()],
    "teste": [valor_com_intervalo(l) for _, l in medida_teste.iterrows()],
}, index=medida_treino.index).rename(index=qualidade.NOMES)
print("Concordância do professor com os rótulos humanos, em % (intervalo de 95%)")
display(concordancia)

rotulado = rotular(treino, claude_treino, intents)
partes_treino = reparticao(ouro_treino, claude_treino)
print(f"Pedidos de treino sem rótulo aproveitável: {n((~rotulado['valido']).sum())} de {n(len(rotulado))}")
partes_treino.map(lambda v: pts(v) + "%").rename("parte do treino").to_frame()
''')

code(r'''
humanas = pd.Series([o.intent for o in ouro_treino]).value_counts()
do_professor_treino = pd.Series([p.intent for p in claude_treino]).value_counts().reindex(humanas.index).fillna(0).astype(int)
desvio = (do_professor_treino - humanas).sort_values()
extremos = pd.concat([desvio.head(5), desvio.tail(5)])
print("Intenções que o professor usa menos (em cima) e mais (embaixo) do que os rótulos humanos, no treino")
display(pd.DataFrame({
    "nos rótulos humanos": humanas.reindex(extremos.index).map(n),
    "nos rótulos do professor": do_professor_treino.reindex(extremos.index).map(n),
    "diferença": extremos.map(lambda v: f"{v:+d}".replace("-", "−")),
}))

entidades_humanas = sum(len(o.slots) for o in ouro_treino)
entidades_do_professor = sum(len(p.slots) for p in claude_treino)
display(Markdown(f"""
**Leitura.**

- O aluno da Etapa 6 vai treinar com rótulos em que **{pts(partes_treino['Tudo certo (exact match)'])}%** dos pedidos estão inteiramente iguais aos humanos. Em {pts(partes_treino['Só as entidades erradas'] + partes_treino['Intenção e entidades erradas'])}% há pelo menos uma entidade diferente, e em {pts(partes_treino['Só a intenção errada'] + partes_treino['Intenção e entidades erradas'])}% a intenção é outra.
- A qualidade medida em `dev` ({pts(medida_dev.loc['exact_match', 'valor'])}% de exact match) e no teste ({pts(medida_teste.loc['exact_match', 'valor'])}%) é uma boa estimativa da qualidade dos rótulos do treino ({pts(medida_treino.loc['exact_match', 'valor'])}%): numa situação real, medir numa amostra pequena bastaria.
- O professor devolve {n(entidades_do_professor)} entidades no treino; os rótulos humanos têm {n(entidades_humanas)} ({dif(entidades_do_professor / entidades_humanas - 1)}%).
"""))
''')

md(r'''
## 6. Latência

O mesmo protocolo dos modelos locais (Etapa 2): uma consulta por vez, 50 de aquecimento e 1.024 medidas, sorteadas de `dev`. O relógio cobre tudo: escolher os exemplos, a ida e a volta pela rede, a geração no provedor e a leitura da resposta. A medida vale para a rede de onde foi feita.

Uma primeira medição foi descartada. Ela rodou ao mesmo tempo que o envio dos lotes, que ocupou a mesma rede com dezenas de megabytes, e ficou presa quando a conexão oscilou. A medida abaixo foi refeita depois, com a rede livre.
''')

code(r'''
pasta_latencia = etapa5a.pasta_das_respostas("latencia")
tempos = np.load(pasta_latencia / "tempos_ms.npy")
meta_latencia = json.loads((pasta_latencia / "meta.json").read_text(encoding="utf-8"))
chamadas_medidas = etapa5a.chamadas("latencia")
assert len(chamadas_medidas) == len(tempos)
falhas_na_latencia = sum(r["texto"] is None for r in chamadas_medidas)
latencia = eficiencia.resumir_latencia(tempos)

rede_resultados = resultados.carregar(ENTREGAS / "04b_intencao").query("particao == 'test'").iloc[0]
p99 = latencia["latencia_p99_ms"]
fig, ax = visual.histograma(
    tempos[tempos <= p99] / 1000, f"Latência do professor por pedido ({n(len(tempos))} chamadas diretas)", "segundos (o 1% mais lento fica fora do gráfico)",
    marcas={f"p50: {num(latencia['latencia_p50_ms'] / 1000, 2)} s": latencia["latencia_p50_ms"] / 1000,
            f"p95: {num(latencia['latencia_p95_ms'] / 1000, 2)} s": latencia["latencia_p95_ms"] / 1000},
)
visual.salvar(fig, PASTA, "latencia")
plt.show()

vezes = latencia["latencia_p50_ms"] / rede_resultados["latencia_p50_ms"]
print(f"Chamadas medidas sem resposta aproveitável: {falhas_na_latencia} de {len(chamadas_medidas)}. Medido em {meta_latencia['quando']}, de: {meta_latencia['de_onde']}.")
print(f"Cada chamada tem prazo de {meta_latencia['prazo_por_chamada_s']:.0f} s e até {meta_latencia['novas_tentativas']} novas tentativas automáticas em erro passageiro; o tempo delas entra na medida.")
pd.DataFrame({
    PROFESSOR + " (API)": [num(latencia[f"latencia_{p}_ms"], 0) for p in ("p50", "p90", "p95", "p99")] + [num(tempos.max(), 0)],
    "Rede do zero (1 núcleo de CPU)": [num(rede_resultados[f"latencia_{p}_ms"], 2) for p in ("p50", "p90", "p95", "p99")] + ["—"],
}, index=["p50 (ms)", "p90 (ms)", "p95 (ms)", "p99 (ms)", "máximo (ms)"])
''')

md(r'''
## 7. Custo

A API informa, em cada resposta, quantos tokens cobrou e de que tipo. O custo abaixo é a soma desses tokens pelo preço de lista. Não é estimativa de tamanho de prompt.

| Tipo de token | O que é | Preço por milhão |
|---|---|---|
| Entrada | o que foi enviado e não estava em cache | US$ 0,10 |
| Entrada lida do cache | as instruções fixas, quando o provedor já as tinha guardado | US$ 0,01 |
| Entrada escrita no cache | as instruções fixas, na vez em que são guardadas | US$ 0,125 |
| Saída | a resposta | US$ 0,50 |

Pedidos em lote custam metade em todos os tipos.
''')

code(r'''
def por_pedido(registros, em_lote):
    "Médias por pedido respondido: tokens de entrada (todos os tipos), de saída, parte lida do cache e custo."
    respondidos = [r for r in registros if r["uso"]]
    entrada = np.array([r["uso"]["entrada"] + r["uso"]["cache_lido"] + r["uso"]["cache_escrito"] for r in respondidos])
    return {
        "tokens de entrada": entrada.mean(),
        "tokens de saída": np.mean([r["uso"]["saida"] for r in respondidos]),
        "pedidos com leitura de cache": np.mean([r["uso"]["cache_lido"] > 0 for r in respondidos]),
        "custo": np.mean([claude.custo(r["uso"], em_lote) for r in respondidos]),
        "custo sem cache": np.mean([claude.custo({"entrada": e, "saida": r["uso"]["saida"]}, em_lote) for e, r in zip(entrada, respondidos)]),
    }


em_lote = por_pedido([r for c in ("dev", "test", "train") for r in lidos[c]["registros"]], em_lote=True)
direto = por_pedido(chamadas_medidas, em_lote=False)
print("Médias por pedido")
pd.DataFrame({
    "Em lote (como o treino foi rotulado)": em_lote,
    "Direto (como um agente chamaria)": direto,
}).T.assign(**{
    "tokens de entrada": lambda t: t["tokens de entrada"].map(lambda v: n(round(v))),
    "tokens de saída": lambda t: t["tokens de saída"].map(lambda v: num(v, 1)),
    "pedidos com leitura de cache": lambda t: t["pedidos com leitura de cache"].map(lambda v: pts(v) + "%"),
    "custo por 1.000 pedidos": lambda t: t["custo"].map(lambda v: usd(1000 * v, 3)),
    "custo por 1.000 pedidos sem cache": lambda t: t["custo sem cache"].map(lambda v: usd(1000 * v, 3)),
}).drop(columns=["custo", "custo sem cache"])
''')

md(r'''
A conta da etapa soma os tokens de tudo o que foi chamado e tem registro. As chamadas diretas da latência incluem o aquecimento. Algumas chamadas ficaram **sem registro** (uma tentativa interrompida do teste de formato e a medição de latência descartada); para elas só dá para dar um teto, pelo custo médio de uma chamada direta.
''')

code(r'''
diarios = [json.loads(linha) for arquivo in sorted(pasta_latencia.glob("diario_*.jsonl")) for linha in arquivo.open(encoding="utf-8")]
arquivo_sem_registro = pasta_latencia / "sem_registro.json"
sem_registro = json.loads(arquivo_sem_registro.read_text(encoding="utf-8")) if arquivo_sem_registro.exists() else []

uso = {
    "Teste de formato (chamadas diretas)": claude.resumir_uso(etapa5a.chamadas("teste")),
    "Lote de dev": claude.resumir_uso(lidos["dev"]["registros"], em_lote=True),
    "Lote de teste": claude.resumir_uso(lidos["test"]["registros"], em_lote=True),
    "Lote de treino": claude.resumir_uso(lidos["train"]["registros"], em_lote=True),
    "Latência (todas as chamadas diretas com registro)": claude.resumir_uso(diarios),
}
conta = pd.DataFrame(uso).T
gasto_registrado = float(conta["dolares"].sum())
custo_do_treino = uso["Lote de treino"]["dolares"]
chamadas_sem_registro = sum(item["chamadas_no_maximo"] for item in sem_registro)
teto_sem_registro = chamadas_sem_registro * direto["custo"]

conta_formatada = pd.DataFrame({
    "pedidos": conta["pedidos"].map(n),
    "entrada": conta["entrada"].map(n),
    "lida do cache": conta["cache_lido"].map(n),
    "escrita no cache": conta["cache_escrito"].map(n),
    "saída": conta["saida"].map(n),
    "custo": conta["dolares"].map(lambda v: usd(v, 4)),
})
vazio = {"entrada": "", "lida do cache": "", "escrita no cache": "", "saída": ""}
conta_formatada.loc["Total com registro"] = {"pedidos": n(conta["pedidos"].sum()), **vazio, "custo": usd(gasto_registrado, 4)}
conta_formatada.loc["Chamadas sem registro (teto)"] = {"pedidos": f"até {n(chamadas_sem_registro)}", **vazio, "custo": f"até {usd(teto_sem_registro, 4)}"}
print("Tokens cobrados e custo de cada parte da etapa")
display(conta_formatada)
fora_da_medida = len(diarios) - len(chamadas_medidas) - eficiencia.AQUECIMENTO
print(f"Chamadas diretas da latência: {n(len(chamadas_medidas))} medidas, {eficiencia.AQUECIMENTO} de aquecimento e {fora_da_medida} de uma execução interrompida logo no início.")
for item in sem_registro:
    print(f"Sem registro: {item['o_que']} (no máximo {n(item['chamadas_no_maximo'])} chamadas)")
''')

code(r'''
display(Markdown(f"""
**Leitura.**

- A etapa custou **{usd(gasto_registrado)}** em {n(conta['pedidos'].sum())} pedidos com registro, mais um teto de {usd(teto_sem_registro)} em chamadas sem registro. Rotular os {n(len(treino))} pedidos do treino custou {usd(custo_do_treino)}.
- Cada pedido leva em média {n(round(direto['tokens de entrada']))} tokens de entrada e {num(direto['tokens de saída'], 0)} de saída (o JSON da resposta). Quase todo o custo é entrada: as instruções com as listas de rótulos e os {ajuste.k} exemplos.
- Como um agente usaria (chamada direta, com as instruções em cache em {pts(direto['pedidos com leitura de cache'])}% das chamadas): **{usd(1000 * direto['custo'], 3)} por mil pedidos**, ou {usd(1e6 * direto['custo'], 0)} por milhão. Sem o cache seriam {usd(1000 * direto['custo sem cache'], 3)} por mil.
- Latência mediana de **{num(latencia['latencia_p50_ms'] / 1000, 2)} s** por pedido (p95 de {num(latencia['latencia_p95_ms'] / 1000, 2)} s). A rede do zero responde em {num(rede_resultados['latencia_p50_ms'], 1)} ms num núcleo de CPU: cerca de {n(round(vezes, -2))} vezes mais rápido.
"""))
''')

md(r'''
## 8. Entrega

As respostas brutas da API vão para a entrega: elas custaram dinheiro, e com elas a etapa pode ser refeita sem nenhuma chamada nova.
''')

code(r'''
# 1. O treino rotulado pelo professor, nas colunas que o treino da rede lê
rotulado.to_parquet(PASTA / "rotulos_treino.parquet", index=False)
conjunto_demo[["id", "intent", "utt", "tokens", "tags"]].to_parquet(PASTA / "demonstracoes.parquet", index=False)

# 2. As respostas do professor em dev e teste, no formato da régua
salvar_previsoes(PASTA / "previsoes" / "claude_dev.parquet", dev["id"], claude_dev)
salvar_previsoes(PASTA / "previsoes" / "claude_test.parquet", teste["id"], claude_teste)

# 3. As respostas brutas
for conjunto in (*etapa5a.CONJUNTOS_EM_LOTE, *etapa5a.CHAMADAS_DIRETAS):
    origem = etapa5a.PASTA / conjunto
    if origem.exists():
        shutil.copytree(origem, PASTA / "respostas" / conjunto, dirs_exist_ok=True)

# 4. A tabela de resultados
custo_por_mil = {"custo_por_mil_pedidos_usd": 1000 * direto["custo"], "custo_por_mil_pedidos_em_lote_usd": 1000 * em_lote["custo"]}
linhas_resultado = [
    resultados.registrar(f"{PROFESSOR} ({ajuste.k} exemplos de {len(conjunto_demo)} rótulos)", particao, gabarito[particao], do_professor[particao],
                         tempos_ms=tempos, modelo=claude.MODELO, **custo_por_mil)
    for particao in ("dev", "test")
] + [
    resultados.registrar(base_299.nome, particao, gabarito[particao], referencias[particao]["Linha de base com 299 rótulos"][0], parametros=base_299.parametros)
    for particao in ("dev", "test")
]
resultados.salvar(linhas_resultado, PASTA)
pd.concat({f"{p} · {ref}": t.loc[list(qualidade.METRICAS)] for p in ("dev", "test") for ref, t in contra[p].items()},
          names=["comparacao", "metrica"]).to_csv(PASTA / "comparacoes.csv")

# 5. Custo e configuração
(PASTA / "custo.json").write_text(json.dumps({
    "moeda": "USD", "precos_por_milhao_de_tokens": claude.PRECO, "desconto_do_lote": claude.DESCONTO_DO_LOTE,
    "partes": uso, "total_com_registro": gasto_registrado,
    "sem_registro": sem_registro, "teto_das_chamadas_sem_registro": teto_sem_registro,
    "por_pedido_em_lote": em_lote, "por_pedido_direto": direto,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(PASTA / "configuracao.json").write_text(json.dumps({
    "etapa": "05a_professor_fechado", "gerado_em": date.today().isoformat(), "modelo": claude.MODELO, "ajuste": ajuste.como_dict(),
    "max_tokens": claude.MAX_TOKENS, "conjunto_de_demonstracao": {"exemplos": len(conjunto_demo), "por_intencao": int(por_intencao.max()), "semente": 0},
    "piloto": "não houve: a configuração não foi comparada com alternativas",
    "latencia": meta_latencia,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(PASTA / "pedido_de_exemplo.md").write_text(
    f"# Um pedido como foi enviado ao {PROFESSOR}\n\n## Instruções (fixas)\n\n```\n{instrucoes}\n```\n\n## Mensagem\n\n```\n{pedido['messages'][0]['content']}\n```\n\n"
    f"## Esquema da resposta\n\n```json\n{json.dumps(pedido['output_config']['format']['schema'], ensure_ascii=False)}\n```\n", encoding="utf-8")

relido = pd.read_parquet(PASTA / "rotulos_treino.parquet")
relido_ok = [Previsao.de_tags(i, list(t)) for i, t in zip(relido["intent"], relido["tags"])] == claude_treino
print("Treino rotulado relido do disco reproduz as respostas do professor:", "sim" if relido_ok else "NÃO")
relido.head(3)
''')

md(r'''
### Critério de pronto
''')

code(r'''
sem_registro = int(leitura.loc["total", "sem registro"])
pronto = pd.Series({
    f"Todo pedido enviado tem resposta gravada ({n(total['pedidos'])} pedidos; {n(sem_uso)} sem resposta aproveitável, contados como erro)": sem_registro == 0,
    "Professor medido na régua, com intervalo, e comparado de forma pareada com as três referências": all(len(contra[p]) == 3 for p in ("dev", "test")),
    f"Latência medida em {n(len(tempos))} chamadas e custo somado dos tokens cobrados": len(tempos) >= eficiencia.CONSULTAS and gasto_registrado > 0,
    "Treino rotulado gravado no formato que a destilação lê": bool(relido_ok and len(relido) == len(treino)),
}).map({True: "sim", False: "NÃO"}).rename("atendido").to_frame()
assert (pronto["atendido"] == "sim").all()
pronto
''')

md(r'''
## 9. Resposta

O resumo abaixo é gerado a partir dos números calculados neste notebook e gravado em `entregas/05a_professor_fechado/RESUMO.md`.
''')

code(r'''
tabela_teste = "\n".join(
    f"| {qualidade.NOMES[m]} | {valor_com_intervalo(medida_teste.loc[m])} | "
    + " | ".join(f"{pts(t.loc[m, 'a'])} · {com_intervalo(t.loc[m])}" for t in contra["test"].values()) + " |"
    for m in qualidade.METRICAS
)
tabela_destinos = "\n".join(
    f"| {nomes_dos_destinos[d]} | {n(destinos.loc[d, PROFESSOR])} ({pts(destinos.loc[d, PROFESSOR] / entidades_em_dev)}%) | {n(round(destinos.loc[d, 'Rede do zero (média de 3 sementes)']))} ({pts(destinos.loc[d, 'Rede do zero (média de 3 sementes)'] / entidades_em_dev)}%) |"
    for d in destinos.index
)
tabela_rotulos = "\n".join(
    f"| {qualidade.NOMES[m]} | {valor_com_intervalo(medida_treino.loc[m])} | {valor_com_intervalo(medida_dev.loc[m])} | {valor_com_intervalo(medida_teste.loc[m])} |"
    for m in qualidade.METRICAS
)
tabela_custo = "\n".join(f"| {parte} | {linha['pedidos']} | {linha['custo']} |" for parte, linha in conta_formatada.iterrows())
troca_1 = trocas.index[0]

if acrescenta:
    primeira = (f"**Vale como professor**: em `dev`, o exact match dele ({pts(leitura_1['b'])}%) fica {com_intervalo(leitura_1)} pontos acima do da linha de base "
                f"treinada com os mesmos 299 rótulos ({pts(leitura_1['a'])}%).")
else:
    primeira = (f"**Não passou na primeira leitura**: em `dev`, o exact match dele ({pts(leitura_1['b'])}%) não fica acima do da linha de base "
                f"treinada com os mesmos 299 rótulos ({pts(leitura_1['a'])}%): {com_intervalo(leitura_1)} pontos.")

resumo = f"""# Etapa 5a · O professor fechado: resumo da entrega

**Pergunta.** Quão bem uma LLM fechada resolve a tarefa quando tudo o que se tem dela é o texto da resposta, quanto custa e quanto demora, e que rótulos ela deixa para o aluno?

**Resposta.** O {PROFESSOR}, vendo {len(conjunto_demo)} rótulos humanos como exemplos, faz no teste {pts(medida_teste.loc['intent_acc', 'valor'])}% de acurácia de intenção, {pts(medida_teste.loc['slot_f1', 'valor'])}% de F1 de entidades e {pts(medida_teste.loc['exact_match', 'valor'])}% de exact match. {primeira} Contra a rede do zero treinada com os {n(len(treino))} rótulos humanos, no teste, ele fica {posicao(contra_rede.loc['intent_acc'])} em intenção ({com_intervalo(contra_rede.loc['intent_acc'])}), {posicao(contra_rede.loc['slot_f1'])} em entidades ({com_intervalo(contra_rede.loc['slot_f1'])}) e {posicao(contra_rede.loc['exact_match'])} em exact match ({com_intervalo(contra_rede.loc['exact_match'])}). Cada resposta leva {num(latencia['latencia_p50_ms'] / 1000, 2)} s (mediana) e custa {usd(1000 * direto['custo'], 3)} por mil pedidos; a rede do zero responde em {num(rede_resultados['latencia_p50_ms'], 1)} ms. O treino rotulado por ele concorda inteiramente com os rótulos humanos em {pts(medida_treino.loc['exact_match', 'valor'])}% dos pedidos, e custou {usd(custo_do_treino)}.

O desenho (intenção perto de um modelo supervisionado, entidades atrás) é o que He & Garner ([arXiv 2305.13512](https://arxiv.org/abs/2305.13512)) relatam para LLMs com poucos exemplos.

## O cenário

- A LLM vê {len(conjunto_demo)} exemplos rotulados por humanos ({pts(len(conjunto_demo) / len(treino))}% do treino, até {int(por_intencao.max())} por intenção). Em cada pedido entram os {ajuste.k} mais parecidos.
- Modelo `{claude.MODELO}`, raciocínio {"ligado" if ajuste.raciocinio else "desligado"}, esforço `{ajuste.esforco}`, resposta em JSON imposta por esquema.
- {n(total['pedidos'])} pedidos rotulados em lote: {n(len(treino))} de treino, {n(len(dev))} de dev, {n(len(teste))} de teste.

## Qualidade no teste

Cada coluna de referência traz o valor da referência e, depois do ponto, a diferença professor − referência com intervalo de 95%.

| Métrica | {PROFESSOR} (intervalo de 95%) | {" | ".join(contra["test"])} |
|---|---|---|---|---|
{tabela_teste}

![O professor e a rede do zero](figuras/professor_na_regua.png)

![Professor menos referências](figuras/professor_menos_referencias.png)

## O que voltou da API

- Sem resposta aproveitável: {n(sem_uso)} de {n(total['pedidos'])} pedidos ({pts(sem_uso / total['pedidos'], 2)}%).
- Intenções fora da lista: {n(total['intenção fora da lista'])}. Tipos de entidade fora da lista: {n(total['tipo fora da lista'])}.
- Entidades com texto que não está no pedido, descartadas: {n(total['texto que não está no pedido'])} de {n(total['entidades devolvidas'])} ({pts(total['texto que não está no pedido'] / total['entidades devolvidas'], 2)}%).

## Onde o professor erra (dev)

Em dev, o professor acerta tudo em {pts(partes.loc['Tudo certo (exact match)', PROFESSOR])}% dos pedidos, erra só as entidades em {pts(partes.loc['Só as entidades erradas', PROFESSOR])}%, só a intenção em {pts(partes.loc['Só a intenção errada', PROFESSOR])}% e as duas em {pts(partes.loc['Intenção e entidades erradas', PROFESSOR])}%.

| Destino de cada entidade dos rótulos humanos | {PROFESSOR} | Rede do zero (média de 3 sementes) |
|---|---|---|
{tabela_destinos}

![Erros de entidade](figuras/erros_de_entidade.png)

- A troca de intenção mais frequente: `{troca_1[0]}` respondida como `{troca_1[1]}` ({trocas.iloc[0]} vezes em dev).
- Quando a intenção certa aparece entre os {ajuste.k} exemplos do prompt ({dependencia.iloc[0]['parte de dev']} de dev), o professor acerta a intenção em {dependencia.iloc[0]['acurácia de intenção do professor']}% dos pedidos; quando não aparece, em {dependencia.iloc[1]['acurácia de intenção do professor']}%. A rede do zero faz {dependencia.iloc[0]['acurácia de intenção da rede do zero']}% e {dependencia.iloc[1]['acurácia de intenção da rede do zero']}% nos mesmos grupos.

## Os rótulos do aluno

Concordância do professor com os rótulos humanos, em % (intervalo de 95%):

| Métrica | Treino | Dev | Teste |
|---|---|---|---|
{tabela_rotulos}

- Pedidos de treino sem rótulo aproveitável: {n((~rotulado['valido']).sum())}.
- A medida em `dev` antecipa bem a qualidade dos rótulos do treino.

## Latência e custo

- Latência por pedido, uma chamada por vez, {n(len(tempos))} chamadas: p50 {num(latencia['latencia_p50_ms'] / 1000, 2)} s, p95 {num(latencia['latencia_p95_ms'] / 1000, 2)} s, p99 {num(latencia['latencia_p99_ms'] / 1000, 2)} s. Medida em {meta_latencia['quando'][:10]}, de: {meta_latencia['de_onde']}.
- Por pedido: {n(round(direto['tokens de entrada']))} tokens de entrada e {num(direto['tokens de saída'], 0)} de saída, em média.
- Chamada direta: {usd(1000 * direto['custo'], 3)} por mil pedidos ({usd(1000 * direto['custo sem cache'], 3)} sem cache). Em lote: {usd(1000 * em_lote['custo'], 3)} por mil, com leitura de cache em {pts(em_lote['pedidos com leitura de cache'])}% dos pedidos.

![Latência](figuras/latencia.png)

| Parte da etapa | Pedidos | Custo |
|---|---|---|
{tabela_custo}

Envio dos lotes: o treino inteiro num único envio (84 MB) foi recusado pelo servidor; os lotes foram enviados em partes de {n(etapa5a.TAMANHO_DO_ENVIO)} pedidos. Nenhum pedido foi cobrado em dobro.

## Limites

- **Sem piloto.** A configuração ({ajuste.k} exemplos, raciocínio {"ligado" if ajuste.raciocinio else "desligado"}) não foi comparada com alternativas. Mais exemplos, raciocínio ligado ou um modelo maior poderiam dar um professor melhor, a outro custo.
- **Um modelo, uma execução.** A API não tem temperatura neste modelo e a saída não é garantidamente determinística; cada pedido foi feito uma vez.
- **A latência é de uma rede e de um horário.** Inclui a ida e a volta até o provedor.
- **O custo é o preço de lista** em {date.today().strftime('%d/%m/%Y')}, somado dos tokens que a API informou. As chamadas sem registro entram só como teto ({usd(teto_sem_registro)}); o valor exato está no painel de uso da conta.
- **A latência foi medida duas vezes.** A primeira medição concorreu com o envio dos lotes e foi descartada; vale a segunda, feita com a rede livre.
- **O modelo pode conhecer o MASSIVE.** O conjunto é público desde 2022 e pode estar nos dados de treino da LLM. Isso não se mede daqui.

## Para a Etapa 6

- O aluno A treina com `rotulos_treino.parquet`: as colunas que o treino da rede lê (`id`, `utt`, `tokens`, `intent`, `tags`), com `intent` e `tags` do professor. A coluna `valido` marca os pedidos sem rótulo aproveitável.
- A referência "sem LLM" do cenário é um modelo treinado só com os {len(conjunto_demo)} rótulos de `demonstracoes.parquet`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `rotulos_treino.parquet` | O treino rotulado pelo professor |
| `demonstracoes.parquet` | Os {len(conjunto_demo)} exemplos com rótulo humano que a LLM vê |
| `previsoes/` | As respostas do professor em dev e teste, no formato da régua |
| `respostas/` | As respostas brutas da API, com os tokens cobrados; a latência medida |
| `resultados.csv`, `comparacoes.csv` | O professor e a linha de base com {len(conjunto_demo)} rótulos na régua; cada comparação pareada |
| `custo.json`, `configuracao.json` | Tokens e dólares por parte; modelo e ajustes da chamada |
| `pedido_de_exemplo.md` | Um pedido completo, como foi enviado |
| `figuras/` | Os gráficos deste notebook |
| `05a_professor_fechado.html` | Este notebook executado, para leitura |
"""
(PASTA / "RESUMO.md").write_text(resumo, encoding="utf-8")
display(Markdown(resumo))
''')

code(r'''
arquivos = sorted(p for p in PASTA.rglob("*") if p.is_file())
pd.DataFrame(
    {"tamanho (KB)": [round(p.stat().st_size / 1024, 1) for p in arquivos]},
    index=[str(p.relative_to(RAIZ)) for p in arquivos],
)
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("ok", len(cells), "células")
