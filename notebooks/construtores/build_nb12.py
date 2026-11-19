"""Monta notebooks/12_investigacao_dos_achados.ipynb a partir das células abaixo."""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Etapa 12 · Investigação dos achados

| | |
|---|---|
| **Pergunta** | Por que acontecem os três efeitos que as etapas mediram e não explicaram, e o que cada um ensina para o artigo? |
| **Entrada** | As respostas gravadas e os modelos das Etapas 10a, 10b, 10c, 11b e 11c (`entregas/` e o cache delas em `data/execucoes/`), os rótulos dos professores (`entregas/05a_professor_fechado/`, `entregas/11a_professor_aberto_maior/`) e as execuções novas desta etapa, gravadas por `especialista/experimentos/etapa12.py` em `data/execucoes/12_investigacao_dos_achados/`. |
| **Entrega** | `entregas/12_investigacao_dos_achados/`: para cada achado, as hipóteses, os testes, os números com intervalo e a conclusão (explicado, explicado em parte, não explicado); o plano gravado antes; a leitura única no teste; os artigos conferidos. |
| **Pronto quando** | o plano foi gravado antes de qualquer execução; cada hipótese do plano tem o seu teste feito e a sua leitura; o teste foi lido só como o plano fixou; cada explicação tem um artigo conferido na fonte, com o que ele sustenta e o que não; nada das outras etapas mudou. |

**É uma investigação depois dos resultados.** Tudo aqui é diagnóstico e exploratório: os efeitos já tinham sido vistos, e os testes foram escolhidos depois deles. Nenhuma decisão das etapas anteriores muda, e os modelos finais continuam os mesmos. O que parecer recomendação é recomendação para um próximo trabalho, não troca de receita.

**Método.** O plano (hipóteses, testes, o que contaria como confirmação ou refutação, e a leitura no teste de cada achado) foi gravado em `plano.json` antes de qualquer execução. A exploração é toda em `dev`, com os rótulos humanos. Cada comparação é a diferença pareada da média de 3 sementes, com intervalo de 95% por bootstrap de enunciados (2.000 sorteios), como em todas as etapas. O teste entra uma vez por achado, como o plano fixou.

**Artigos de apoio.** Cada um foi conferido na API do arXiv (título, primeiro autor, data) e o resumo foi lido; nenhum PDF foi aberto. O quadro, com o que cada resumo sustenta e o que não sustenta, está na seção 6.

- Arpit et al., *A Closer Look at Memorization in Deep Networks*, [arXiv 1706.05394](https://arxiv.org/abs/1706.05394)
- Zhang et al., *Understanding deep learning requires rethinking generalization*, [arXiv 1611.03530](https://arxiv.org/abs/1611.03530)
- Stanton et al., *Does Knowledge Distillation Really Work?*, [arXiv 2106.05945](https://arxiv.org/abs/2106.05945)
- Mosbach, Andriushchenko & Klakow, *On the Stability of Fine-tuning BERT*, [arXiv 2006.04884](https://arxiv.org/abs/2006.04884)
- Gupta et al., *Continual Pre-Training of Large Language Models: How to (re)warm your model?*, [arXiv 2308.04014](https://arxiv.org/abs/2308.04014)
- Abdaoui, Pradel & Sigel, *Load What You Need*, [arXiv 2010.05609](https://arxiv.org/abs/2010.05609); Ushio, Zhou & Camacho-Collados, *Vocabulary Trimming*, [arXiv 2305.15020](https://arxiv.org/abs/2305.15020)
- Bondarenko, Nagel & Blankevoort, *Understanding and Overcoming the Challenges of Efficient Transformer Quantization*, [arXiv 2109.12948](https://arxiv.org/abs/2109.12948)
- Kovaleva et al., *BERT Busters*, [arXiv 2105.06990](https://arxiv.org/abs/2105.06990); Dettmers et al., *LLM.int8()*, [arXiv 2208.07339](https://arxiv.org/abs/2208.07339)
- Wu et al., *Integer Quantization for Deep Learning Inference*, [arXiv 2004.09602](https://arxiv.org/abs/2004.09602); Jacob et al., [arXiv 1712.05877](https://arxiv.org/abs/1712.05877)
- Dutta et al., *Accuracy is Not All You Need*, [arXiv 2407.09141](https://arxiv.org/abs/2407.09141)

**Seções**

1. O plano, gravado antes
2. Achado 1: a poda do XLM-R nos alunos do cenário B
3. Achado 3: a INT8 dinâmica e a intenção
4. Achado 2: a INT8 estática e a semente que desaba
5. A leitura no teste
6. Os artigos
7. Nada das outras etapas mudou
8. Critério de pronto e resumo
''')

code(r'''
import json
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from especialista import visual
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.experimentos import etapa12 as e

visual.aplicar_estilo()
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

BASE = e.pasta_da_etapa()           # data/execucoes/12_investigacao_dos_achados
CONTAS = e.pasta_das_contas()       # gravado por `python -m especialista.experimentos.etapa12 contas`
PASTA = e.pasta_da_entrega()
plano = e.plano()
ler = lambda nome: pd.read_csv(CONTAS / nome)
ler_json = lambda nome: json.loads((CONTAS / nome).read_text(encoding="utf-8"))
C1, PROF, CURVAS, TREINO, DD = (ler(f"achado1_{n}.csv") for n in ("comparacoes", "professor", "curvas", "treino", "diferenca_das_diferencas"))
CORTE = ler_json("achado1_corte.json")
P3, T3, ALT3, NOVE = (ler(f"achado3_{n}.csv") for n in ("partes", "trocas", "alternativas", "nove_modelos"))
R3 = ler_json("achado3_resumo.json")
FAIXAS, MED, ALT2 = (ler(f"achado2_{n}.csv") for n in ("faixas", "medidas", "alternativas"))
TESTE = ler_json("teste.json")

NOMES = {**e.NOMES}
CURTO = {e.B1: "B1 · Qwen3-14B", e.B2: "B2 · Qwen3-14B", e.B2_ANTIGO: "B2 · Qwen3-8B", e.ALUNO_A: "Aluno A", e.TETO: "Teto"}
MOMENTO = {"partida": "modelo de partida (sem poda)", "so_corte": "só o corte", "corte_e_reajuste": "corte e reajuste (receita da 10b)",
           "sem_corte_e_reajuste": "reajuste sem corte (controle)", "corte_e_reajuste_refeito": "corte e reajuste, refeito nesta etapa",
           "corte_e_reajuste_taxa_menor": "corte e reajuste com taxa 10× menor"}


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


def n(x):
    return f"{int(x):,}".replace(",", ".")


def ci(linha, casas=1, prefixo=""):
    """'−1,0 (−1,8 a −0,3)' a partir de uma linha com diferença e intervalo (em pontos)."""
    g = lambda k: linha[prefixo + k]
    f = lambda x: f"{x:+.{casas}f}".replace(".", ",").replace("-", "−")
    return f"{f(g('diferenca'))} ({f(g('ic_min'))} a {f(g('ic_max'))})"


def c1(aluno, depois, antes, metrica="exact_match", sub=None):
    """Uma linha das comparações do achado 1 (em dev)."""
    t = C1[(C1["aluno"] == aluno) & (C1["depois"] == depois) & (C1["antes"] == antes) & (C1["metrica"] == metrica)]
    t = t[t["subconjunto"].isna()] if sub is None else t[t["subconjunto"] == sub]
    assert len(t) == 1, (aluno, depois, antes, metrica, sub, len(t))
    return t.iloc[0]


def distinguivel(linha, prefixo=""):
    return linha[prefixo + "ic_min"] > 0 or linha[prefixo + "ic_max"] < 0
''')

md(r'''
## 1. O plano, gravado antes

O plano foi escrito depois de ler os resumos e as decisões das etapas de origem e antes de rodar qualquer coisa nova ou de fazer qualquer conta nova sobre as respostas gravadas. A hora em que ele foi gravado fica comparada, abaixo, com a hora da primeira execução desta etapa.
''')

code(r'''
arquivo_do_plano = BASE / "plano.json"
primeira_execucao = min(p.stat().st_mtime for p in BASE.glob("reajustes/*/*/*/semente_*/meta.json"))
for p in list(BASE.glob("onnx/*/*/semente_*/fp32/conferido.json")) + list(BASE.glob("estatica/*/*/semente_*/faixas.csv")):
    primeira_execucao = min(primeira_execucao, p.stat().st_mtime)
impressoes_antes = json.loads((BASE / "impressoes" / "impressoes_antes.json").read_text(encoding="utf-8"))["tirada_em"]
import hashlib
sha_do_plano = hashlib.sha256(arquivo_do_plano.read_bytes()).hexdigest()
plano_antes = arquivo_do_plano.stat().st_mtime < primeira_execucao
plano_intacto = sha_do_plano == (BASE / "plano.sha256").read_text().split()[0]
from datetime import datetime
quando = lambda t: datetime.fromtimestamp(t).strftime("%d/%m %H:%M:%S")
print(f"plano gravado em {quando(arquivo_do_plano.stat().st_mtime)}; primeira execução em {quando(primeira_execucao)}; antes: {plano_antes}; "
      f"sha256 igual ao registrado logo depois de gravar: {plano_intacto}")
pd.DataFrame([{"achado": k, "hipóteses": ", ".join(plano[k]["hipoteses"]), "testes": ", ".join(plano[k]["testes"]), "leitura no teste": plano[k]["leitura_no_teste"][:160] + "…"}
              for k in ("achado_1", "achado_3", "achado_2")])
''')

md(r'''
## 2. Achado 1: a poda do XLM-R perde nos alunos do cenário B

**O que se sabia.** A receita "vocabulário do treino" da Etapa 10b corta a tabela de subtokens do XLM-R para as 5.196 linhas que o texto do treino usa e depois reajusta 10 épocas, com a receita e o sinal de quem treinou o modelo (a taxa sobe de novo até 5e-5 e desce até zero). Com o Aluno A e o teto ela passa; com os alunos do cenário B, perde cerca de 1 ponto de exact match (10b e 11c).

**Hipóteses.** (a) O reajuste é treino a mais sobre os rótulos do professor aberto, que erram mais, e o aluno passa a copiar os erros dele. (b) O recomeço da taxa de aprendizado tira o modelo do ponto em que estava. (c) O corte em si: pedidos com subtokens fora das 5.196 linhas.

**Testes.** T1, o mesmo reajuste sem cortar nada (o controle da 10b, que o Aluno A e o teto já tinham e os alunos B não); T2, o corte e o reajuste refeitos com as respostas em `dev` gravadas a cada época; T3, a cópia dos erros do professor em `dev` e no treino; T4, o reajuste com a taxa 10 vezes menor; T5, os pedidos que o corte afeta.
''')

code(r'''
ALUNOS1 = (e.B1, e.B2, e.B2_ANTIGO, e.ALUNO_A, e.TETO)
linhas = []
for aluno in ALUNOS1:
    for depois, antes in (("so_corte", "partida"), ("corte_e_reajuste", "partida"), ("sem_corte_e_reajuste", "partida"), ("corte_e_reajuste", "sem_corte_e_reajuste"),
                          ("corte_e_reajuste_taxa_menor", "partida"), ("corte_e_reajuste_taxa_menor", "so_corte")):
        try:
            r = c1(aluno, depois, antes)
        except AssertionError:
            continue
        linhas.append({"modelo": CURTO[aluno], "comparação": f"{MOMENTO[depois]} − {MOMENTO[antes]}", "exact match em dev (pontos)": ci(r),
                       "distinguível": distinguivel(r)})
TAB1 = pd.DataFrame(linhas)
TAB1
''')

code(r'''
dados = {}
for aluno in (e.B1, e.B2, e.ALUNO_A, e.TETO):
    for depois, rotulo in (("sem_corte_e_reajuste", "reajuste sem corte"), ("corte_e_reajuste", "corte e reajuste")):
        r = c1(aluno, depois, "partida")
        dados[f"{CURTO[aluno]}: {rotulo}"] = (r["diferenca"], r["ic_min"], r["ic_max"])
fig, ax = visual.diferencas_com_intervalo(dados, "Achado 1 · T1: o reajuste, com e sem corte, menos o modelo de partida (exact match em dev)", "diferença pareada (pontos)")
visual.salvar(fig, PASTA, "achado1_controle")
''')

md(r'''
### T2: a curva do reajuste, época a época

O reajuste guarda a época que mais concorda com o `dev` **rotulado pelo professor** (o modelo de partida concorre como época 0). A curva abaixo mostra, para cada época, a mudança do exact match desde a época 0 contra os dois rótulos: o do professor (o que escolhe a época) e o humano (o que o projeto mede). O ponto é a média das 3 sementes e a barra vai da menor à maior semente.
''')

code(r'''
def curva(tipo, aluno, coluna):
    t = CURVAS[(CURVAS["tipo"] == tipo) & (CURVAS["aluno"] == aluno)]
    base = t[t["epoca"] == 0].set_index("semente")[coluna]
    t = t.assign(mudanca=t[coluna] - t["semente"].map(base))
    g = t.groupby("epoca")["mudanca"]
    return {int(k): (g.mean()[k], g.min()[k], g.max()[k]) for k in g.mean().index}


for tipo, aluno, nome in (("refeito", e.B1, "achado1_curva_b1"), ("refeito", e.ALUNO_A, "achado1_curva_aluno_a"), ("taxa_menor", e.B1, "achado1_curva_b1_taxa_menor")):
    series = {"contra o professor (escolhe a época)": curva(tipo, aluno, "professor_exact_match"), "contra os humanos (o que se mede)": curva(tipo, aluno, "humano_exact_match")}
    titulo = f"{CURTO[aluno]}: {'corte e reajuste' if tipo == 'refeito' else 'corte e reajuste com taxa 10× menor'}, mudança desde a época 0"
    fig, ax = visual.curva_com_intervalo(series, titulo, "época do reajuste", "exact match em dev (pontos)")
    ax.axhline(0, color=visual.TINTA_SUAVE, linewidth=1, zorder=0.5)
    visual.salvar(fig, PASTA, nome)

resumo_curvas = []
for tipo in ("refeito", "sem_corte", "taxa_menor"):
    for aluno in e.REAJUSTES[tipo]["alunos"]:
        t = CURVAS[(CURVAS["tipo"] == tipo) & (CURVAS["aluno"] == aluno)]
        m = t.groupby("epoca")[["humano_exact_match", "professor_exact_match"]].mean()
        escolhida = {s: json.loads((e.pasta_do_reajuste(tipo, aluno, s) / "meta.json").read_text())["melhor_epoca"] for s in e.SEMENTES}
        resumo_curvas.append({"reajuste": tipo, "modelo": CURTO[aluno], "época 1 − época 0, humanos": m.loc[1, "humano_exact_match"] - m.loc[0, "humano_exact_match"],
                              "época 1 − época 0, professor": m.loc[1, "professor_exact_match"] - m.loc[0, "professor_exact_match"],
                              "época 10 − época 0, humanos": m.loc[10, "humano_exact_match"] - m.loc[0, "humano_exact_match"],
                              "época 10 − época 0, professor": m.loc[10, "professor_exact_match"] - m.loc[0, "professor_exact_match"],
                              "época escolhida por semente": list(escolhida.values())})
RC = pd.DataFrame(resumo_curvas)
RC.round(2)
''')

code(r'''
# O reajuste refeito repete o das etapas de origem? (mesma função, mesma receita, mesma semente)
reproduz = {}
for aluno in (e.B1, e.B2, e.ALUNO_A):
    origem = e.momentos_em_dev(aluno)
    reproduz[CURTO[aluno]] = [sum(a == b for a, b in zip(x, y)) for x, y in zip(origem["corte_e_reajuste"], origem["corte_e_reajuste_refeito"])]
print("respostas em dev iguais às da etapa de origem, por semente (de 2.033):", reproduz)
''')

md(r'''
### T3: o aluno copia os erros do professor?

Nos pedidos de `dev` em que o professor erra (exact match contra o humano), duas medidas: em que fração o aluno dá **a mesma resposta errada** do professor, e em que fração acerta. No treino, o acerto do aluno contra o rótulo do professor, separado em pedidos em que o professor acerta o humano ("limpos") e em que erra ("ruidosos"): se o reajuste encaixa os erros do professor, o acerto contra o professor sobe mais nos ruidosos.
''')

code(r'''
TP = PROF.copy()
TP["modelo"] = TP["aluno"].map(CURTO)
TP["momento"] = TP["momento"].map(lambda m: MOMENTO.get(m, m))
cols = ["modelo", "momento", "intent_acc", "slot_f1", "exact_match", "pedidos_em_que_o_professor_erra", "copia_o_erro", "acerta_onde_o_professor_erra", "concorda_com_o_professor"]
display(TP[cols].round(1))

TT = TREINO.groupby(["aluno", "momento"]).mean(numeric_only=True).drop(columns="semente").reset_index()
TT["modelo"] = TT["aluno"].map(CURTO)
TT[["modelo", "momento", "ruido_do_professor", "igual_ao_professor_nos_limpos", "igual_ao_professor_nos_ruidosos", "igual_ao_humano_nos_ruidosos"]].round(1)
''')

code(r'''
def treino(aluno, momento, coluna):
    return TT[(TT["aluno"] == aluno) & (TT["momento"] == momento)][coluna].iloc[0]


def prof(aluno, momento, coluna):
    return PROF[(PROF["aluno"] == aluno) & (PROF["momento"] == momento)][coluna].iloc[0]


subida = {aluno: {"ruidosos": treino(aluno, "corte_e_reajuste", "igual_ao_professor_nos_ruidosos") - treino(aluno, "partida", "igual_ao_professor_nos_ruidosos"),
                  "limpos": treino(aluno, "corte_e_reajuste", "igual_ao_professor_nos_limpos") - treino(aluno, "partida", "igual_ao_professor_nos_limpos")}
          for aluno in (e.B1, e.B2, e.ALUNO_A)}
copia = {aluno: {m: prof(aluno, m, "copia_o_erro") - prof(aluno, "partida", "copia_o_erro") for m in ("corte_e_reajuste", "sem_corte_e_reajuste") if m in set(PROF[PROF["aluno"] == aluno]["momento"])}
         for aluno in (e.B1, e.B2, e.B2_ANTIGO, e.ALUNO_A)}
pd.DataFrame(subida).T.round(2), pd.DataFrame(copia).T.round(2)
''')

md(r'''
### O Aluno A perde menos: é uma diferença distinguível?

A diferença das diferenças: (efeito do passo no aluno B) − (efeito do mesmo passo no Aluno A), com os mesmos sorteios de pedidos para os quatro grupos.
''')

code(r'''
DD["modelo"] = DD["aluno"].map(CURTO)
DD["texto"] = DD.apply(ci, axis=1)
DD[DD["metrica"] == "exact_match"][["modelo", "momento", "texto", "distinguivel"]]
''')

md(r'''
### T5: os pedidos que o corte afeta
''')

code(r'''
linhas = []
for aluno in ALUNOS1:
    for sub in ("afetados", "nao_afetados"):
        r = c1(aluno, "so_corte", "partida", sub=sub)
        r2 = c1(aluno, "corte_e_reajuste", "so_corte", sub=sub)
        linhas.append({"modelo": CURTO[aluno], "pedidos": sub.replace("_", " "), "quantos": int(r["pedidos"]), "só o corte − partida": ci(r), "reajuste depois do corte": ci(r2)})
T5 = pd.DataFrame(linhas)
print(f"Pedidos com pelo menos um subtoken fora das 5.196 linhas: {CORTE['afetados']} de {n(CORTE['pedidos_de_dev'])} em dev "
      f"({num(100 * CORTE['afetados'] / CORTE['pedidos_de_dev'])}%), {CORTE['afetados_no_teste']} de {n(CORTE['pedidos_do_teste'])} no teste "
      f"({num(100 * CORTE['afetados_no_teste'] / CORTE['pedidos_do_teste'])}%; contagem só pelo texto, sem ler resposta)")
T5
''')

code(r'''
# Os números da leitura do achado 1
b1, b2, a_ = e.B1, e.B2, e.ALUNO_A
X1 = {
    "controle_b1": c1(b1, "sem_corte_e_reajuste", "partida"), "controle_b2": c1(b2, "sem_corte_e_reajuste", "partida"),
    "controle_a": c1(a_, "sem_corte_e_reajuste", "partida"), "controle_teto": c1(e.TETO, "sem_corte_e_reajuste", "partida"),
    "receita_b1": c1(b1, "corte_e_reajuste", "partida"), "receita_b2": c1(b2, "corte_e_reajuste", "partida"), "receita_a": c1(a_, "corte_e_reajuste", "partida"),
    "corte_vs_controle_b1": c1(b1, "corte_e_reajuste", "sem_corte_e_reajuste"), "corte_vs_controle_b2": c1(b2, "corte_e_reajuste", "sem_corte_e_reajuste"),
    "so_corte_b1": c1(b1, "so_corte", "partida"), "so_corte_b2": c1(b2, "so_corte", "partida"),
    "taxa_b1": c1(b1, "corte_e_reajuste_taxa_menor", "partida"), "taxa_b2": c1(b2, "corte_e_reajuste_taxa_menor", "partida"),
    "taxa_vs_so_corte_b1": c1(b1, "corte_e_reajuste_taxa_menor", "so_corte"), "taxa_vs_so_corte_b2": c1(b2, "corte_e_reajuste_taxa_menor", "so_corte"),
    "taxa_vs_receita_b1": c1(b1, "corte_e_reajuste_taxa_menor", "corte_e_reajuste"), "taxa_vs_receita_b2": c1(b2, "corte_e_reajuste_taxa_menor", "corte_e_reajuste"),
    "afetados_b1": c1(b1, "so_corte", "partida", sub="afetados"), "afetados_b2": c1(b2, "so_corte", "partida", sub="afetados"),
    "nao_afetados_b1": c1(b1, "so_corte", "partida", sub="nao_afetados"), "afetados_a": c1(a_, "so_corte", "partida", sub="afetados"),
}
dd = lambda aluno, momento: DD[(DD["aluno"] == aluno) & (DD["momento"] == momento) & (DD["metrica"] == "exact_match")].iloc[0]
X1["dd_controle_b1"], X1["dd_controle_b2"] = dd(b1, "sem_corte_e_reajuste"), dd(b2, "sem_corte_e_reajuste")
X1["dd_receita_b1"], X1["dd_receita_b2"] = dd(b1, "corte_e_reajuste"), dd(b2, "corte_e_reajuste")
rc = lambda tipo, aluno, col: RC[(RC["reajuste"] == tipo) & (RC["modelo"] == CURTO[aluno])][col].iloc[0]
ruido = {aluno: treino(aluno, "partida", "ruido_do_professor") for aluno in (b1, a_)}
prof_dev = {aluno: PROF[(PROF["aluno"] == aluno) & (PROF["momento"] == "professor")]["exact_match"].iloc[0] for aluno in (b1, a_)}
{k: (ci(v) if "diferenca" in v else v) for k, v in X1.items()}
''')

md(r'''
## 3. Achado 3: a INT8 dinâmica tira cerca de 0,4 ponto de intenção do modelo final do XLM-R

**O que se sabia.** No modelo final do XLM-R (Aluno A, vocabulário do treino), a INT8 dinâmica muda a acurácia de intenção em −0,41 (−0,79 a −0,03) em `dev` e −0,44 (−0,77 a −0,10) no teste, contra o mesmo modelo em 32 bits. A regra da 10c olhava só as entidades.

**Hipóteses e testes.** h1, de que parte da rede vem a queda (as formas de diagnóstico `so_tabela` e `so_matrizes` da 10c); h2, as trocas se concentram em pedidos de margem pequena entre a 1ª e a 2ª intenção; h3, as trocas vão mais para o lado certo → errado; h4, o efeito aparece nos 9 modelos da 10c. E T4, as alternativas: a cabeça de intenção em 32 bits, um degrau por canal nas matrizes, a tabela com um degrau por linha, a tabela em 32 bits.
''')

code(r'''
P3["texto"] = P3.apply(lambda r: ci(r, 2, "intent_acc_"), axis=1)
display(P3[["forma", "texto", "intent_acc_distinguivel"]].rename(columns={"texto": "intenção − fp32, dev (pontos)"}))
print(json.dumps({k: R3[k] for k in ("trocas", "por_semente", "lados", "teste_de_sinal_p", "fracao_certo_para_errado", "fracao_esperada_se_a_troca_fosse_ao_acaso_entre_as_duas_primeiras",
                                      "margem_mediana_das_trocas", "margem_mediana_de_todos", "percentil_10_da_margem_de_todos", "fracao_das_trocas_com_margem_abaixo_do_p10")},
                 ensure_ascii=False, indent=1))
''')

code(r'''
margens_todas = np.concatenate([e.margem(np.load(e.pasta_da_dinamica(s) / "fp32" / "intencoes_dev.npy")) for s in e.SEMENTES])
fig, ax = visual.histograma(T3["margem_fp32"], "Achado 3 · margem, no modelo de 32 bits, dos pedidos que trocam de intenção na INT8",
                            "margem entre a 1ª e a 2ª intenção (probabilidade)", marcas={"10º percentil de todos os pedidos": float(np.percentile(margens_todas, 10)),
                                                                                         "mediana de todos": float(np.median(margens_todas))}, classes=25)
visual.salvar(fig, PASTA, "achado3_margem")
''')

code(r'''
ALT3["texto_intencao"] = ALT3.apply(lambda r: ci(r, 2, "intent_acc_"), axis=1)
ALT3["texto_entidades"] = ALT3.apply(lambda r: ci(r, 2, "slot_f1_"), axis=1)
ALT3["MB_da_rede"] = ALT3["bytes_da_rede"] / 1e6
display(ALT3[["alternativa", "o_que_e", "texto_intencao", "texto_entidades", "MB_da_rede"]].round(2))
NOVE["modelo"] = NOVE["modelo"].map(CURTO) + ", " + NOVE["variante"].str.replace("_", " ")
dados = {r.modelo: (r.diferenca, r.ic_min, r.ic_max) for r in NOVE.itertuples()}
fig, ax = visual.diferencas_com_intervalo(dados, "Achado 3 · INT8 dinâmica − 32 bits, acurácia de intenção em dev, nos 9 modelos da 10c", "diferença pareada (pontos)")
visual.salvar(fig, PASTA, "achado3_nove_modelos")
negativos = int((NOVE["diferenca"] < 0).sum())
print(f"{negativos} de 9 modelos com diferença negativa em dev; média {num(NOVE['diferenca'].mean(), 2)}")
''')

md(r'''
## 4. Achado 2: a INT8 estática desaba numa das três sementes

**O que se sabia.** Na 10c, a INT8 estática (calibração com 512 pedidos do treino, faixa do menor ao maior valor, um degrau por tensor) não passa em nenhum dos 9 modelos, e uma semente desaba. Conferido nas respostas gravadas: o 41,9 do teto é o do teto sem poda, e o 47,4 do Aluno A é o da variante **vocabulário do treino** (o "sem poda" do Aluno A dá 47,8), e por isso é ela que entra aqui, como o plano previa.

**Hipóteses e testes.** h1, poucas ativações com valores extremos estouram a faixa (T1: a faixa de cada entrada de multiplicação, por semente); h2, a perda vem de poucos tensores (T2: um grupo por vez em 8 bits; T4: tensores em 32 bits); h3, outra calibração resolve (T3: percentil e entropia); h4, a poda gradual diminui os extremos (só descritivo).
''')

code(r'''
NOME_DO_MODELO = {("teto", "sem_poda"): "Teto, sem poda", ("aluno_a", "vocabulario_treino"): "Aluno A, vocabulário do treino", ("aluno_a", "gradual_70"): "Aluno A, gradual 70%"}
F = FAIXAS[FAIXAS["tipo"] == "saida_da_rede_densa"].copy()
F["modelo_"] = [NOME_DO_MODELO[(m, v)] for m, v in zip(F["modelo"], F["variante"])]
tabela_faixas = F.pivot_table(index=["modelo_", "semente"], columns="camada", values="maximo_absoluto").round(0)
display(Markdown("**Maior valor absoluto da entrada da 2ª camada da rede densa (a saída da GELU), por camada**, nos 512 pedidos de calibração"))
display(tabela_faixas)
por_tipo = FAIXAS.groupby(["modelo", "variante", "semente", "tipo"])["maximo_absoluto"].max().unstack("tipo").round(1)
display(Markdown("**Maior valor absoluto por tipo de entrada**"))
display(por_tipo)
L8 = F[F["camada"] == 8].set_index(["modelo_", "semente"])[["maximo_absoluto", "p99", "p99_99", "cinco_maiores_dimensoes"]]
L8
''')

code(r'''
M = MED.copy()
M["modelo_"] = [NOME_DO_MODELO[(m, v)] for m, v in zip(M["modelo"], M["variante"])]
tipos = M[M["nome"].str.match(r"^so_[a-z_]+$") & ~M["nome"].str.contains("camada")]
display(Markdown("**T2: só um grupo em 8 bits (o resto em 32 bits), teto sem poda.** F1 de entidades em dev e a queda para o fp32 da mesma semente."))
display(tipos.pivot_table(index="nome", columns="semente", values="queda").round(1))
por_camada = M[M["nome"].str.startswith("so_saida_da_rede_densa_camada_")].assign(camada=lambda t: t["nome"].str.extract(r"(\d+)$")[0].astype(int))
display(Markdown("**T2: só a entrada da saída da rede densa de UMA camada em 8 bits.** Queda de F1 em dev."))
display(por_camada.pivot_table(index="camada", columns=["modelo_", "semente"], values="queda").round(1))
acumulado = M[M["nome"].str.startswith("so_saida_da_rede_densa_camadas_0_a_")].assign(k=lambda t: t["nome"].str.extract(r"(\d+)$")[0].astype(int))
display(Markdown("**T2: as camadas 0 a k juntas em 8 bits** (desvio do plano: acrescentado depois de ver que nenhuma camada sozinha explica)."))
display(acumulado.pivot_table(index="k", columns="semente", values="queda").round(1))
um_fora = M[M["nome"].str.startswith("todas_menos_saida_da_rede_densa_camada_")].assign(camada=lambda t: t["nome"].str.extract(r"(\d+)$")[0].astype(int))
display(Markdown("**T4: a estática inteira, com UMA multiplicação da saída da rede densa em 32 bits** (semente 1; desvio do plano)."))
um_fora.pivot_table(index="camada", columns="modelo_", values="queda").round(1)
''')

code(r'''
serie = {}
for s in (1, 0):
    t = acumulado[acumulado["semente"] == s].set_index("k")["slot_f1_dev"]
    todas_ = M[(M["modelo"] == "teto") & (M["semente"] == s) & (M["nome"] == "todas")]
    so_tipo = M[(M["modelo"] == "teto") & (M["semente"] == s) & (M["nome"] == "so_saida_da_rede_densa")]["slot_f1_dev"].iloc[0]
    pontos = {-1: M[(M["modelo"] == "teto") & (M["semente"] == s)]["slot_f1_fp32"].iloc[0], **t.to_dict(), 11: so_tipo}
    serie[f"semente {s}"] = {k + 1: (v, v, v) for k, v in sorted(pontos.items())}
fig, ax = visual.curva_com_intervalo(serie, "Achado 2 · teto sem poda: a perda se acumula pelas camadas",
                                     "camadas (de baixo para cima) com a entrada da saída da rede densa em 8 bits", "F1 de entidades em dev (%)", marcas_x=[0, 2, 4, 6, 8, 10, 12])
visual.salvar(fig, PASTA, "achado2_acumulado")
dados = {}
for r in um_fora.sort_values(["modelo_", "camada"]).itertuples():
    dados[f"{r.modelo_}: camada {r.camada} em 32 bits"] = (r.queda, r.queda, r.queda)
fig, ax = visual.diferencas_com_intervalo({k: v for k, v in dados.items()}, "Achado 2 · semente 1: a estática inteira com uma multiplicação em 32 bits (queda de F1 em dev)", "F1 de entidades − fp32 (pontos)")
visual.salvar(fig, PASTA, "achado2_um_de_fora")
''')

code(r'''
A2 = ALT2.copy()
A2["modelo_"] = [NOME_DO_MODELO[(m, v)] for m, v in zip(A2["modelo"], A2["variante"])]
A2["F1 − fp32 (dev)"] = A2.apply(lambda r: ci(r, 1, "slot_f1_"), axis=1)
A2["intenção − fp32 (dev)"] = A2.apply(lambda r: ci(r, 1, "intent_acc_"), axis=1)
A2["MB da rede"] = A2["bytes_da_rede"] / 1e6
display(A2[["modelo_", "alternativa", "F1 − fp32 (dev)", "intenção − fp32 (dev)", "slot_f1_semente_0", "slot_f1_semente_1", "slot_f1_semente_2", "MB da rede"]].round(1))
dados = {f"{r.modelo_}: {r.alternativa.replace('_', ' ')}": (r.slot_f1_diferenca, r.slot_f1_ic_min, r.slot_f1_ic_max) for r in A2.itertuples()}
fig, ax = visual.diferencas_com_intervalo(dados, "Achado 2 · cada forma estática − 32 bits, F1 de entidades em dev (média de 3 sementes)", "diferença pareada (pontos)")
ax.axvline(-1, color=visual.LARANJA, linewidth=1.2, linestyle=(0, (4, 3)), zorder=0.6)
visual.salvar(fig, PASTA, "achado2_alternativas")
piores = {(r.modelo, r.semente): r.descricao for r in M[M["nome"] == "sem_os_piores_pela_faixa"].itertuples()}
pd.Series(piores, name="o que ficou em 32 bits (regra do maior salto)")
''')

md(r'''
## 5. A leitura no teste

Uma leitura por achado, a que o plano fixou antes de qualquer execução.
''')

code(r'''
print(json.dumps(TESTE["regras"], ensure_ascii=False, indent=1))
t1 = TESTE["achado_1"]
for aluno, c in t1["teste"].items():
    print(f"Achado 1, {CURTO[aluno]}: {t1['lido']}, exact match no teste: {ci(c)}")
print("Achado 1, a regra em dev para a taxa menor:", {CURTO[a]: (ci(p["dev"]), p["passa"]) for a, p in t1["regra_em_dev"].items()})
print("Achado 3:", TESTE["achado_3"])
t2 = TESTE["achado_2"]
for modelo, c in t2["teste"].items():
    print(f"Achado 2, {modelo}: {t2['escolhida_em_dev']}, F1 de entidades − fp32 no teste: {ci(c)}")
''')

md(r'''
## 6. Os artigos
''')

code(r'''
ART = (pd.read_csv(PASTA / "artigos.csv", dtype={"identificador": str}) if (PASTA / "artigos.csv").exists() else e.quadro_de_artigos()).fillna("")
ART[["quem", "identificador", "na_fonte", "titulo_na_fonte", "o_resumo_sustenta", "o_que_o_resumo_diz"]]
''')

md(r'''
## 7. Nada das outras etapas mudou

As impressões digitais (tamanho e data de todo arquivo de `entregas/` e `data/execucoes/`, fora as pastas desta etapa; sha256 dos arquivos de até 64 MB) foram tiradas antes da primeira execução e de novo no fim. Um arquivo "novo" fora desta etapa não foi escrito por ela: aparece aqui para ficar registrado.
''')

code(r'''
ler_imp = lambda nome: json.loads((BASE / "impressoes" / nome).read_text(encoding="utf-8"))
antes_ = ler_imp("impressoes_antes.json")
depois_ = ler_imp(sorted(p.name for p in (BASE / "impressoes").glob("impressoes_*.json") if p.name != "impressoes_antes.json")[-1])
a_arq, d_arq = antes_["arquivos"], depois_["arquivos"]
mudaram = sorted(k for k in set(a_arq) & set(d_arq) if a_arq[k] != d_arq[k])
sumiram = sorted(set(a_arq) - set(d_arq))
novos = sorted(k for k in set(d_arq) - set(a_arq) if e.ETAPA not in k)   # os desta etapa não contam (o script só exclui a pasta dela em data/execucoes)
print(f"antes: {antes_['tirada_em']}; depois: {depois_['tirada_em']}; {len(a_arq)} arquivos antes, {len(d_arq)} depois")
print(f"mudaram: {len(mudaram)}; sumiram: {len(sumiram)}; novos: {len(novos)}")
for k in novos:
    print("   novo:", k)
nada_mudou = not mudaram and not sumiram
pastas_novas = sorted({"/".join(k.split("/")[:3]) for k in novos})
''')

md(r'''
## 8. Critério de pronto e resumo
''')

code(r'''
# ---------- os números que o resumo usa (todos calculados acima) ----------
fp = lambda aluno, m: TT[(TT["aluno"] == aluno) & (TT["momento"] == m)]
mem = {a: (fp(a, "partida")["igual_ao_professor_nos_ruidosos"].iloc[0], fp(a, "corte_e_reajuste")["igual_ao_professor_nos_ruidosos"].iloc[0],
           fp(a, "partida")["igual_ao_professor_nos_limpos"].iloc[0], fp(a, "corte_e_reajuste")["igual_ao_professor_nos_limpos"].iloc[0]) for a in (b1, b2, a_)}
acerta = {a: (prof(a, "partida", "acerta_onde_o_professor_erra"), prof(a, "corte_e_reajuste", "acerta_onde_o_professor_erra")) for a in (b1, b2, a_)}
copia_ = {a: (prof(a, "partida", "copia_o_erro"), prof(a, "corte_e_reajuste", "copia_o_erro")) for a in (b1, b2, a_)}
queda1 = {a: rc("refeito", a, "época 1 − época 0, humanos") for a in (b1, b2, a_)}
fim_h = {a: rc("refeito", a, "época 10 − época 0, humanos") for a in (b1, b2, a_)}
fim_p = {a: rc("refeito", a, "época 10 − época 0, professor") for a in (b1, b2, a_)}
taxa_q1 = rc("taxa_menor", b1, "época 1 − época 0, humanos")
iguais_fora = all(x["nao_afetados_iguais"] == x["nao_afetados"] for v in CORTE["com_e_sem_corte_por_semente"].values() for x in v)
treino_igual = all(CORTE["com_e_sem_corte_iguais_no_treino_inteiro"].values())

alt3 = ALT3.set_index("alternativa")
cab = alt3.loc["cabeca_de_intencao_em_32_bits"]; canal = alt3.loc["degrau_por_canal"]; linha_ = alt3.loc["tabela_por_linha"]; din = alt3.loc["dinamica"]
part = P3.set_index("forma")
mb = lambda b: num(b / 1e6)

A2i = A2.set_index(["modelo", "alternativa"])
f1 = lambda m, alt: A2i.loc[(m, alt)]
def f1_sementes(m, alt):
    r = f1(m, alt)
    return " · ".join(num(r[f"slot_f1_semente_{s}"]) for s in e.SEMENTES)
fp32_f1 = {m: " / ".join(num(x) for x in M[(M["modelo"] == m) & (M["nome"] == "todas")].sort_values("semente")["slot_f1_fp32"]) for m in ("teto", "aluno_a")}
tipo_q = tipos.pivot_table(index="nome", columns="semente", values="queda")
so_densa = {s: M[(M["modelo"] == "teto") & (M["semente"] == s) & (M["nome"] == "so_saida_da_rede_densa")]["slot_f1_dev"].iloc[0] for s in (0, 1)}
todas_t = {s: M[(M["modelo"] == "teto") & (M["semente"] == s) & (M["nome"] == "todas")]["slot_f1_dev"].iloc[0] for s in (0, 1)}
outros_tipos_max = tipo_q.drop(index="so_saida_da_rede_densa").abs().max().max()
pc1 = por_camada[(por_camada["modelo"] == "teto") & (por_camada["semente"] == 1)]
pior_camada = pc1.sort_values("queda").iloc[0]
acu1 = acumulado[acumulado["semente"] == 1].set_index("k")["slot_f1_dev"]
acu0 = acumulado[acumulado["semente"] == 0].set_index("k")["slot_f1_dev"]
uf = um_fora.set_index(["modelo", "camada"])
melhor_fora = {m: um_fora[um_fora["modelo"] == m].sort_values("queda", ascending=False).iloc[0] for m in ("teto", "aluno_a")}
perda_toda = {m: M[(M["modelo"] == m) & (M["semente"] == 1) & (M["nome"] == "todas")]["queda"].iloc[0] for m in ("teto", "aluno_a")}
volta = {m: 1 - melhor_fora[m]["queda"] / perda_toda[m] for m in ("teto", "aluno_a")}
L8i = F[F["camada"] == 8].set_index(["modelo", "variante", "semente"])
fx = lambda m, s, col: L8i.loc[(m, {"teto": "sem_poda", "aluno_a": "vocabulario_treino"}[m], s), col]
fp32_bytes = R3["bytes_fp32"]
bytes_sem_densa = f1("aluno_a", "sem_a_saida_da_rede_densa")["bytes_da_rede"]
bytes_todas = f1("aluno_a", "todas")["bytes_da_rede"]
t1, t2 = TESTE["achado_1"], TESTE["achado_2"]
ESC2 = t2["escolhida_em_dev"]
grad = F[F["modelo_"] == "Aluno A, gradual 70%"].groupby("semente")["maximo_absoluto"].max()
semp = F[F["modelo_"] == "Aluno A, vocabulário do treino"].groupby("semente")["maximo_absoluto"].max()
ART_ok = (ART["na_fonte"] == "conferida").all()
{k: v for k, v in {"volta": volta, "ESC2": ESC2}.items()}
''')

code(r'''
estatica_igual = all(e.ler_respostas(e.pasta_da_estatica(m, v, s) / "todas") == e.previsoes_da_10c(m, v, "estatica")[s]
                     for m, v in e.MODELOS_DO_ACHADO_2 for s in e.SEMENTES)
criterios = [
    ("O plano foi gravado antes de qualquer execução nova, e não mudou depois", plano_antes and plano_intacto),
    ("As 21 execuções do achado 1 estão gravadas, cada uma com as respostas em dev de cada época", len(CURVAS.groupby(["tipo", "aluno", "semente"])) == 21 and CURVAS.groupby(["tipo", "aluno", "semente"]).size().eq(11).all()),
    ("O reajuste refeito repete, resposta por resposta, o das Etapas 10b e 11c", all(x == 2033 for v in reproduz.values() for x in v)),
    ("O ONNX de 32 bits exportado de novo repete as respostas da 10c em dev", all(json.loads(p.read_text())["iguais_a_10c_em_dev"] == 2033 for p in BASE.glob("onnx/*/*/semente_*/fp32/conferido.json"))),
    ("A INT8 dinâmica e a estática refeitas repetem as da 10c, resposta por resposta", R3["dinamica_refeita_igual_a_10c"] == 3 * 2033 and estatica_igual),
    ("Cada hipótese do plano tem o seu teste e a sua leitura", True),
    ("O teste foi lido uma vez por achado, como o plano fixou (achado 3: nada ao teste, porque nenhuma alternativa resolveu em dev)",
     set(TESTE) == {"regras", "achado_1", "achado_2", "achado_3"} and TESTE["achado_3"]["teste"] is None),
    ("Cada artigo citado foi conferido na API do arXiv", ART_ok),
    ("Nada das outras etapas mudou nem sumiu (impressões digitais antes e depois); os arquivos novos fora desta etapa estão listados", nada_mudou),
]
for criterio, ok in criterios:
    assert ok, criterio
linhas_de_pronto = "\n".join(f"| {c} | {'sim' if ok else 'NÃO'} |" for c, ok in criterios)
print(linhas_de_pronto)
''')

code(r'''
r = lambda chave: ci(X1[chave])
resumo = f"""# Etapa 12 · Investigação dos achados: resumo da entrega

**Pergunta.** Por que acontecem os três efeitos que as etapas mediram e não explicaram, e o que cada um ensina para o artigo?

**Natureza.** É uma investigação **depois dos resultados**: os efeitos já tinham sido vistos, e os testes foram escolhidos depois deles. Tudo aqui é diagnóstico e exploratório. Nenhuma decisão das etapas anteriores muda, e os modelos finais continuam os mesmos (o Aluno A, em cada caminho). O plano, com as hipóteses, os testes, o que contaria como confirmação e a leitura no teste de cada achado, foi gravado em `plano.json` antes de qualquer execução. Diferenças pareadas da média de 3 sementes, com intervalo de 95% por bootstrap de enunciados (2.000 sorteios), em pontos percentuais; tudo em `dev`, salvo a leitura única no teste.

**Resposta curta.**

- **Achado 1 (a poda do XLM-R perde nos alunos B): explicado em parte.** A perda vem na maior parte do **reajuste**, não do corte: o mesmo reajuste de 10 épocas, sem cortar nada, perde {r('controle_b1')} (B1) e {r('controle_b2')} (B2) de exact match em dev. O recomeço da taxa derruba o aluno na 1ª época ({num(queda1[b1])} e {num(queda1[b2])} pontos contra os humanos), e as épocas seguintes o levam de volta **para perto do professor**, não dos humanos: no treino, o aluno B1 passa a repetir o rótulo do professor em {num(mem[b1][1])}% dos pedidos em que o professor erra (eram {num(mem[b1][0])}%). Como a época é escolhida pelo dev rotulado pelo professor, ganha a que mais concorda com ele. Com a taxa 10 vezes menor o reajuste deixa de perder (corte e reajuste − só o corte: {r('taxa_vs_so_corte_b1')} e {r('taxa_vs_so_corte_b2')}). O corte em si pesa só nos {CORTE['afetados']} pedidos de dev ({num(100 * CORTE['afetados'] / CORTE['pedidos_de_dev'])}%) com subtoken fora das 5.196 linhas. **Em aberto:** a diferença entre o Aluno A e os alunos B não é distinguível (efeito do reajuste no B1 menos no A: {ci(X1['dd_controle_b1'])}); o Aluno A também encaixa os erros do professor no treino ({num(mem[a_][0])}% → {num(mem[a_][1])}%). "Passa com o A e não passa com o B" separa dois intervalos que se sobrepõem. **No teste** (a leitura única do plano), o reajuste sem corte também perde: {'; '.join(f"{CURTO[a]} {ci(c)}" for a, c in t1['teste'].items())}.
- **Achado 3 (a INT8 dinâmica tira cerca de 0,4 ponto de intenção): explicado.** A queda vem das matrizes ({ci(part.loc['so_matrizes'], 2, 'intent_acc_')}), não da tabela ({ci(part.loc['so_tabela'], 2, 'intent_acc_')}). É ruído de arredondamento que vira decisões quase empatadas: {num(100 * R3['fracao_das_trocas_com_margem_abaixo_do_p10'], 0)}% das {R3['trocas']} trocas de intenção (3 sementes) estão entre os 10% de pedidos de menor margem. As trocas vão {R3['lados'].get('certo_para_errado', 0)} vezes de certo para errado e {R3['lados'].get('errado_para_certo', 0)} de errado para certo (teste de sinal, p = {num(R3['teste_de_sinal_p'], 3)}), mas essa proporção ({num(R3['fracao_certo_para_errado'], 2)}) é a que uma troca ao acaso entre as duas primeiras intenções daria ({num(R3['fracao_esperada_se_a_troca_fosse_ao_acaso_entre_as_duas_primeiras'], 2)}): não há viés além do ruído. O efeito é geral: a diferença de intenção em dev é negativa em {negativos} dos 9 modelos da 10c. Nenhuma alternativa simples resolve: a cabeça de intenção em 32 bits dá exatamente o mesmo ({ci(cab, 2, 'intent_acc_')}), um degrau por canal piora ({ci(canal, 2, 'intent_acc_')}), a tabela com um degrau por linha fica em {ci(linha_, 2, 'intent_acc_')}. Nada foi ao teste.
- **Achado 2 (a INT8 estática desaba numa semente): explicado em parte.** A perda vem toda de um grupo de tensores: a entrada da 2ª camada da rede densa (a saída da GELU). Só esse grupo em 8 bits já derruba o teto na semente 1 para {num(so_densa[1])} de F1 (a estática inteira dá {num(todas_t[1])}); os outros grupos, sozinhos, mudam no máximo {num(outros_tipos_max)} ponto. Esse tensor tem valores extremos em poucas dimensões, as mesmas em todas as sementes e modelos (herança do XLM-R pré-treinado), e com a faixa do menor ao maior valor 99% dos valores caem em poucos degraus. Nenhuma camada sozinha derruba o modelo; a perda **se acumula** (camadas 0 a 7 em 8 bits: {num(acu1[7])}; 0 a 9: {num(acu1[9])}, na semente 1). Deixar **um** tensor em 32 bits devolve {num(100 * volta['teto'], 0)}% da perda no teto (camada {int(melhor_fora['teto']['camada'])}) e {num(100 * volta['aluno_a'], 0)}% no Aluno A (camada {int(melhor_fora['aluno_a']['camada'])}); deixar os 12 devolve tudo. A semente 1 do teto tem, na camada 8, ativações 3 vezes mais largas que as outras (maior valor {num(fx('teto', 1, 'maximo_absoluto'))} e percentil 99 de {num(fx('teto', 1, 'p99'), 2)}, contra {num(fx('teto', 0, 'maximo_absoluto'))} e {num(fx('teto', 0, 'p99'), 2)} na semente 0). **Em aberto:** a semente 1 do Aluno A desaba sem ter faixas maiores, e a calibração por percentil a conserta ({f1_sementes('aluno_a', 'calibracao_percentile')} de F1 nas sementes 0 · 1 · 2) sem consertar a do teto ({f1_sementes('teto', 'calibracao_percentile')}). **Recomendação:** deixar em 32 bits as 12 multiplicações da saída da rede densa; no teste (a leitura única do plano), essa forma fica a {'; '.join(f"{ci(c)}" + (' no teto' if k.startswith('teto') else ' no Aluno A') for k, c in t2['teste'].items())} de F1 do modelo de 32 bits.

## O plano

`plano.json` (sha256 `{sha_do_plano[:16]}…`) foi gravado às {quando(arquivo_do_plano.stat().st_mtime)}, antes da primeira execução ({quando(primeira_execucao)}) e logo depois das impressões digitais das outras etapas ({impressoes_antes}). Para cada achado: as hipóteses, os testes, o critério de confirmação e de refutação, e a única leitura no teste. Os desvios estão na seção própria.

## Achado 1 · A poda do XLM-R perde cerca de 1 ponto nos alunos do cenário B

**O que se sabia.** A receita "vocabulário do treino" da 10b corta a tabela de subtokens para as 5.196 linhas do texto de treino e reajusta 10 épocas com a receita e o sinal de quem treinou o modelo. Passa com o Aluno A e o teto; perde com os alunos B (B2 antigo −1,0 no teste; B1 e B2 novos −1,0 e −0,9 em dev na 11c).

**Hipóteses.** (a) treino a mais sobre os rótulos do professor aberto: o aluno copia os erros dele; (b) o recomeço da taxa de aprendizado; (c) o corte em si, pelos subtokens de dev e teste fora das 5.196 linhas.

**O que foi testado** (21 reajustes novos, 3 sementes cada; exact match em dev, contra o modelo de partida sem poda):

| Teste | B1 · Qwen3-14B | B2 · Qwen3-14B | Aluno A |
|---|---|---|---|
| Corte e reajuste (a receita, como na 11c e na 10b) | {r('receita_b1')} | {r('receita_b2')} | {r('receita_a')} |
| T1 · reajuste sem corte (controle) | {r('controle_b1')} | {r('controle_b2')} | {r('controle_a')} (já na 10b) |
| Corte e reajuste − reajuste sem corte | {r('corte_vs_controle_b1')} | {r('corte_vs_controle_b2')} | — |
| Só o corte, sem reajuste | {r('so_corte_b1')} | {r('so_corte_b2')} | — |
| T4 · corte e reajuste com taxa 10× menor | {r('taxa_b1')} | {r('taxa_b2')} | — |
| T4 − só o corte | {r('taxa_vs_so_corte_b1')} | {r('taxa_vs_so_corte_b2')} | — |
| Efeito do reajuste sem corte, B − A (diferença das diferenças) | {ci(X1['dd_controle_b1'])} | {ci(X1['dd_controle_b2'])} | — |

O teto, com o mesmo reajuste sem corte: {r('controle_teto')}.

- **O corte e o reajuste se separam exatamente.** O texto do treino só usa as 5.196 linhas que ficam; as outras não recebem gradiente. Por isso, com a mesma semente, o reajuste com corte e sem corte dá respostas idênticas em todo o treino ({'sim' if treino_igual else 'não'}) e em todos os pedidos de dev sem subtoken cortado ({'sim' if iguais_fora else 'não'}, nas 6 execuções). Toda a diferença entre os dois mora nos {CORTE['afetados']} pedidos afetados de dev; no teste são {CORTE['afetados_no_teste']} de {n(CORTE['pedidos_do_teste'])} ({num(100 * CORTE['afetados_no_teste'] / CORTE['pedidos_do_teste'])}%, contados só pelo texto).
- **T2, a curva.** O reajuste refeito repete o das etapas de origem resposta por resposta (2.033 de 2.033 em cada semente, nos três alunos). Na 1ª época, com a taxa de volta ao pico, o exact match contra os humanos cai {num(queda1[b1])} (B1), {num(queda1[b2])} (B2) e {num(queda1[a_])} (Aluno A). Na 10ª época, contra o professor o aluno termina acima de onde começou ({num(fim_p[b1], 2)}, {num(fim_p[b2], 2)} e {num(fim_p[a_], 2)}) e contra os humanos, abaixo ({num(fim_h[b1], 2)}, {num(fim_h[b2], 2)} e {num(fim_h[a_], 2)}). A época é escolhida pelo dev do professor. Com a taxa 10 vezes menor não há queda na 1ª época ({num(taxa_q1, 2)} no B1).
- **T3, a cópia do professor.** Nos pedidos de dev em que o professor erra, o aluno passa a repetir a resposta dele com mais frequência (B1 {num(copia_[b1][0])}% → {num(copia_[b1][1])}%; B2 {num(copia_[b2][0])}% → {num(copia_[b2][1])}%; Aluno A {num(copia_[a_][0])}% → {num(copia_[a_][1])}%) e a acertar menos onde o professor erra (B1 {num(acerta[b1][0])}% → {num(acerta[b1][1])}%; B2 {num(acerta[b2][0])}% → {num(acerta[b2][1])}%; Aluno A {num(acerta[a_][0])}% → {num(acerta[a_][1])}%). No treino, o acerto contra o rótulo do professor sobe muito mais onde o professor erra do que onde acerta (B1: {num(mem[b1][0])}% → {num(mem[b1][1])}% nos pedidos "ruidosos", {num(mem[b1][2])}% → {num(mem[b1][3])}% nos "limpos"; Aluno A: {num(mem[a_][0])}% → {num(mem[a_][1])}% e {num(mem[a_][2])}% → {num(mem[a_][3])}%). O professor aberto discorda do humano (exact match) em {num(ruido[b1])}% dos pedidos de treino; o fechado, em {num(ruido[a_])}%.
- **T5, o corte.** Nos pedidos afetados, só o corte muda o exact match em {r('afetados_b1')} (B1) e {r('afetados_b2')} (B2); nos outros, em exatamente zero.

**Conclusão: explicado em parte.** (a) **confirmada como mecanismo**: o reajuste encaixa os erros do professor (memorização do ruído, como em Arpit et al. e Zhang et al.), e concordar mais com o professor não melhora o acerto contra os humanos (Stanton et al.). (b) **refutada como causa única** (o teto passa pelo mesmo recomeço sem perder: {r('controle_teto')}) e **confirmada como gatilho**: o recomeço tira o aluno do ponto de partida, e sem ele (taxa 10× menor) o reajuste não perde nada além do que o corte já perdeu. O reaquecimento da taxa aumentar a perda antes de melhorar é o que Gupta et al. relatam no pré-treino contínuo. (c) **confirmada só para a parte do corte**, que mora inteira nos pedidos com subtoken cortado; em dev ela é menor ou parecida com a do reajuste ({r('corte_vs_controle_b1')} e {r('corte_vs_controle_b2')}), e a conta exploratória feita antes desta etapa sobre as respostas gravadas da 11c (não refeita aqui, para não abrir o teste de novo) dava, no teste, +0,1 e −0,1 para só o corte. Os cortes de vocabulário de Abdaoui et al. e Ushio et al. são por língua, não pelo texto do treino, e por isso não deixam pedidos sem linha.

**O que fica em aberto.** (1) Por que o Aluno A "passa": a diferença entre o efeito no B e no A não é distinguível ({ci(X1['dd_controle_b1'])} para o B1; {ci(X1['dd_controle_b2'])} para o B2), e o A mostra o mesmo mecanismo, um pouco menor. A explicação "rótulos menos ruidosos" é coerente com os números (o fechado discorda menos do humano e o A perde menos) mas não está provada. (2) Por que só o corte custa em dev e não no teste, com a mesma fração de pedidos afetados: não foi investigado (o teste não foi aberto para isso).

**Recomendação (para trabalho futuro; nada foi trocado).** Depois do corte do vocabulário, reajustar com taxa bem menor (5e-6) ou não reajustar (em dev, a taxa menor não perde nada além do corte; isso não foi ao teste): o corte não mexe nos pedidos cobertos pelo treino, e um reajuste com a taxa cheia, num aluno de professor ruidoso, reaprende os erros do professor. Para o artigo: "passa/não passa" com intervalos que se sobrepõem não sustenta dizer que a receita funciona com um aluno e não com outro.

## Achado 3 · A INT8 dinâmica tira cerca de 0,4 ponto de acurácia de intenção do modelo final do XLM-R

**O que se sabia.** Aluno A, vocabulário do treino, INT8 dinâmica − 32 bits: {ci(din, 2, 'intent_acc_')} de intenção em dev (−0,44 no teste, na 10c). A regra da 10c olhava só as entidades.

**Hipóteses e testes.** h1, a parte da rede (as formas `so_tabela` e `so_matrizes` gravadas na 10c); h2, as trocas se concentram em margem pequena (a pontuação de intenção do ONNX de 32 bits, refeito, em dev); h3, as trocas vão mais para o lado certo → errado; h4, o efeito aparece nos 9 modelos; T4, alternativas.

| Forma (Aluno A, vocabulário do treino) | Intenção − 32 bits, dev | Entidades − 32 bits, dev | MB da rede |
|---|---|---|---|
| INT8 dinâmica (a da 10c, refeita: {R3['dinamica_refeita_igual_a_10c']} de {3 * 2033} respostas iguais) | {ci(din, 2, 'intent_acc_')} | {ci(din, 2, 'slot_f1_')} | {mb(din['bytes_da_rede'])} |
| Só a tabela em 8 bits | {ci(part.loc['so_tabela'], 2, 'intent_acc_')} | {ci(part.loc['so_tabela'], 2, 'slot_f1_')} | — |
| Só as matrizes em 8 bits (a tabela em 32 bits) | {ci(part.loc['so_matrizes'], 2, 'intent_acc_')} | {ci(part.loc['so_matrizes'], 2, 'slot_f1_')} | — |
| Dinâmica com a cabeça de intenção em 32 bits | {ci(cab, 2, 'intent_acc_')} | {ci(cab, 2, 'slot_f1_')} | {mb(cab['bytes_da_rede'])} |
| Dinâmica com um degrau por canal nas matrizes | {ci(canal, 2, 'intent_acc_')} | {ci(canal, 2, 'slot_f1_')} | {mb(canal['bytes_da_rede'])} |
| Dinâmica com a tabela de subtokens com um degrau por linha | {ci(linha_, 2, 'intent_acc_')} | {ci(linha_, 2, 'slot_f1_')} | {mb(linha_['bytes_da_rede'])} |

O ONNX de 32 bits do mesmo modelo tem {mb(fp32_bytes)} MB de rede.

- **h1 confirmada:** a queda vem das matrizes; a tabela sozinha não move a intenção de forma distinguível.
- **h2 confirmada:** a margem mediana dos pedidos que trocam de intenção é {num(R3['margem_mediana_das_trocas'], 2)}, contra {num(R3['margem_mediana_de_todos'], 3)} em todos os pedidos; {num(100 * R3['fracao_das_trocas_com_margem_abaixo_do_p10'], 0)}% das trocas estão abaixo do 10º percentil da margem ({num(R3['percentil_10_da_margem_de_todos'], 2)}).
- **h3, pelo critério do plano, confirmada (p = {num(R3['teste_de_sinal_p'], 3)}), mas sem viés:** a fração certo → errado ({num(R3['fracao_certo_para_errado'], 2)}) é a esperada se cada troca escolhesse ao acaso entre as duas primeiras intenções do modelo de 32 bits ({num(R3['fracao_esperada_se_a_troca_fosse_ao_acaso_entre_as_duas_primeiras'], 2)}: nos pedidos de margem baixa, a 1ª intenção está certa mais vezes que a 2ª). As trocas são ruído; elas tiram mais acerto do que dão porque o modelo de 32 bits acerta mais do que erra nos empates.
- **h4 confirmada:** negativa em {negativos} dos 9 modelos em dev (média {num(NOVE['diferenca'].mean(), 2)}); no teste, a 10c já registrava 7 de 9 negativas (média −0,13).
- **Nenhuma alternativa resolve** pelo critério do plano (diferença com intervalo contendo o zero e ponto ≥ −0,15). A cabeça de intenção em 32 bits muda zero resposta: as trocas nascem antes da cabeça. O degrau por canal piora. A tabela por linha fica com o intervalo contendo o zero, mas o ponto ({num(linha_['intent_acc_diferenca'], 2)}) não atinge o critério.

**Conclusão: explicado.** A INT8 dinâmica arredonda as matrizes; o erro, pequeno, só vira a decisão nos pedidos quase empatados, e aí tira um pouco mais de acerto do que dá. É o fenômeno das "trocas" de Dutta et al.: acurácias parecidas escondem respostas que mudam nos dois sentidos. Os −0,4 do modelo final estão dentro da faixa dos 9 modelos ({num(NOVE['diferenca'].min(), 2)} a {num(NOVE['diferenca'].max(), 2)}): o modelo final não tem nada de especial; foi o intervalo dele que não tocou o zero.

**O que fica em aberto.** Que camadas concentram o erro que vira a intenção (não testado); por que o degrau por canal piora (não investigado).

**Recomendação.** A regra de quantização deveria olhar também a intenção (ou o exact match) e contar as trocas, além da média. Nenhuma das alternativas baratas testadas compra a intenção de volta.

## Achado 2 · A INT8 estática desaba numa das três sementes

**O que se sabia.** Na 10c a INT8 estática (MinMax, 512 pedidos do treino, um degrau por tensor) não passa em nenhum dos 9 modelos, e a semente 1 desaba. Conferido nas respostas gravadas: o 41,9 é do teto sem poda (F1 em 32 bits {fp32_f1['teto']}, por semente) e o 47,4 é do Aluno A **vocabulário do treino** (32 bits: {fp32_f1['aluno_a']}); a variante sem poda do A dá 47,8. A queda também aparece no B2 antigo (30,7 na semente 1).

**Hipóteses e testes.** h1, valores extremos estouram a faixa (T1: faixa de cada entrada de multiplicação, 512 pedidos de calibração); h2, poucos tensores (T2: um grupo por vez em 8 bits; T4: tensores em 32 bits); h3, outra calibração (percentil 99,999 e entropia); h4, a poda gradual diminui os extremos (descritivo).

- **T2, por tipo** (teto sem poda, só um grupo em 8 bits, queda de F1 em dev): a entrada da saída da rede densa {num(tipo_q.loc['so_saida_da_rede_densa', 1])} na semente 1 e {num(tipo_q.loc['so_saida_da_rede_densa', 0])} na semente 0; todos os outros grupos (consulta, chave e valor; saída da atenção; entrada da rede densa; cabeças; tabelas) entre {num(-outros_tipos_max)} e +{num(outros_tipos_max)}.
- **T2, por camada:** uma camada sozinha nunca custa mais que {num(-pior_camada['queda'])} ponto (camada {int(pior_camada['camada'])}, semente 1). **Acumulado** (desvio do plano): camadas 0 a 5 em 8 bits, {num(acu1[5])}; 0 a 7, {num(acu1[7])}; 0 a 9, {num(acu1[9])} (semente 1). Na semente 0: {num(acu0[5])}, {num(acu0[7])}, {num(acu0[9])}.
- **T4, um tensor em 32 bits** (desvio do plano; semente 1): a estática inteira com só a saída da rede densa da camada {int(melhor_fora['teto']['camada'])} em 32 bits vai de {num(todas_t[1])} para {num(melhor_fora['teto']['slot_f1_dev'])} no teto ({num(100 * volta['teto'], 0)}% da perda volta, com 1 de 74 multiplicações); no Aluno A, a camada {int(melhor_fora['aluno_a']['camada'])} leva de 47,4 para {num(melhor_fora['aluno_a']['slot_f1_dev'])} ({num(100 * volta['aluno_a'], 0)}%).
- **T1, as faixas.** As dimensões com valores extremos se repetem em todos os modelos e sementes (por exemplo a 754 na camada 7, a 1194 e a 3025 na 6), como Kovaleva et al. descrevem para dimensões que vêm do pré-treino. No maior caso, o maior valor é cerca de 60 e o percentil 99 fica perto de 1: com 256 degraus do menor ao maior valor, 99% dos valores ficam em uns 4 degraus. Na semente 1 do teto, a camada 8 tem a distribuição inteira 3 vezes mais larga (percentil 99 de {num(fx('teto', 1, 'p99'), 2)} contra {num(fx('teto', 0, 'p99'), 2)} e {num(fx('teto', 2, 'p99'), 2)}); na semente 1 do Aluno A, não ({num(fx('aluno_a', 1, 'p99'), 2)}); quem tem a camada 8 larga no A é a semente 2 ({num(fx('aluno_a', 2, 'p99'), 2)}), que também perde mais que a 0.

| Forma estática (F1 de entidades − 32 bits, dev) | Teto, sem poda | Aluno A, vocabulário do treino | F1 por semente (0 · 1 · 2) | MB da rede (A) |
|---|---|---|---|---|
""" + "\n".join(
    f"| {e.ALTERNATIVAS_ESTATICAS[alt]} | {ci(f1('teto', alt), 1, 'slot_f1_')} | {ci(f1('aluno_a', alt), 1, 'slot_f1_')} | teto {f1_sementes('teto', alt)}; A {f1_sementes('aluno_a', alt)} | {mb(f1('aluno_a', alt)['bytes_da_rede'])} |"
    for alt in e.ALTERNATIVAS_ESTATICAS) + f"""

O ONNX de 32 bits do Aluno A, vocabulário do treino, tem {mb(fp32_bytes)} MB de rede; a INT8 dinâmica, {mb(din['bytes_da_rede'])}.

- **h1 confirmada em parte:** os extremos existem e estão no grupo que perde, mas a semente que desaba nem sempre é a de faixa maior (Aluno A, semente 1).
- **h2 confirmada:** um tensor em 32 bits (1,4% das multiplicações) devolve mais de 2/3 da perda na semente 1 dos dois modelos; os 12 da saída da rede densa (16%) devolvem quase tudo. A regra do plano para escolher os piores só pela faixa (o maior salto da razão máximo/percentil) devolve menos ({ci(f1('teto', 'sem_os_piores_pela_faixa'), 1, 'slot_f1_')} no teto): a razão máximo/percentil aponta camadas que, sozinhas, pesam pouco.
- **h3 confirmada no Aluno A e refutada no teto**, pelo critério do plano (perda abaixo de 1 ponto, ou reduzida à metade ou menos): no Aluno A o percentil leva a perda de {ci(f1('aluno_a', 'todas'), 1, 'slot_f1_')} para {ci(f1('aluno_a', 'calibracao_percentile'), 1, 'slot_f1_')} (a entropia, para {ci(f1('aluno_a', 'calibracao_entropy'), 1, 'slot_f1_')}); no teto, de {ci(f1('teto', 'todas'), 1, 'slot_f1_')} para {ci(f1('teto', 'calibracao_percentile'), 1, 'slot_f1_')}. Nenhuma calibração deixa a perda abaixo de 1 ponto.
- **h4 não sustentada:** a faixa do Aluno A com poda gradual de 70% é parecida com a sem poda (maior valor por semente: {', '.join(num(x) for x in grad)} contra {', '.join(num(x) for x in semp)}); por que a poda gradual sofre menos fica em aberto.

**Conclusão: explicado em parte.** A estática falha porque um tipo de ativação, a saída da GELU que entra na 2ª camada da rede densa, tem valores extremos em poucas dimensões (Bondarenko et al. apontam extremos estruturados como o desafio próprio da quantização de Transformers; Dettmers et al., que separar as poucas dimensões extremas em mais bits resolve), e o erro desse tensor, pequeno em cada camada, se acumula pela pilha até derrubar o modelo. Por que a semente 1 desaba mais fica explicado só no teto, onde ela coincide com uma camada 8 três vezes mais larga; que cada semente fique a uma distância diferente desse ponto de queda é uma hipótese, não testada, e no Aluno A não há diferença de faixa que a sustente.

**O que fica em aberto.** O que torna a semente 1 do Aluno A frágil (as faixas não mostram; a calibração por percentil conserta); por que a poda gradual sofre menos; e se um degrau por grupo de dimensões (o que Bondarenko et al. propõem) resolveria sem deixar nada em 32 bits (não testado: o ONNX Runtime não tem essa forma pronta).

**Recomendação.** Se a estática for necessária, deixar as 12 multiplicações da saída da rede densa em 32 bits; custa {mb(bytes_sem_densa - bytes_todas)} MB a mais no Aluno A ({mb(bytes_sem_densa)} contra {mb(bytes_todas)} MB). Para o artigo: medir a INT8 estática em 3 sementes, porque uma só pode dar um número qualquer entre "quase sem perda" e "desaba".

## A leitura no teste

Uma por achado, como o plano fixou.

- **Achado 1.** A taxa menor passaria a seguir a regra da 10b em dev nos dois alunos? {'; '.join(f"{CURTO[a]}: {ci(p['dev'])}, {'passa' if p['passa'] else 'não passa'}" for a, p in t1['regra_em_dev'].items())}. Como não passa nos dois, leu-se no teste o reajuste sem corte − modelo de partida: {'; '.join(f"{CURTO[a]} {ci(c)}" for a, c in t1['teste'].items())} de exact match.
- **Achado 3.** Nenhuma alternativa resolveu em dev; nada foi ao teste.
- **Achado 2.** A alternativa de menor perda média em dev foi `{ESC2}`; no teste, F1 de entidades − 32 bits: {'; '.join(f"{k.replace('teto/sem_poda', 'teto sem poda').replace('aluno_a/vocabulario_treino', 'Aluno A, vocabulário do treino')} {ci(c)}" for k, c in t2['teste'].items())}.

## Os artigos

Conferidos na API do arXiv (título, primeiro autor e data); só o resumo foi lido.

| Artigo | Sustenta? | O que o resumo diz, e o que não sustenta |
|---|---|---|
""" + "\n".join(f"| {a.quem}, [arXiv {a.identificador}](https://arxiv.org/abs/{a.identificador}) | {a.o_resumo_sustenta or 'não lido'} | {a.o_que_o_resumo_diz or '—'} |" for a in ART.itertuples()) + f"""

## Desvios do plano

- **Achado 2, modelo.** O 47,4 citado é do Aluno A "vocabulário do treino", não do "sem poda"; usou-se essa variante, como o plano previa ao conferir.
- **Achado 2, testes acrescentados depois de ver resultados:** a sensibilidade por camada também na semente 0 do teto e na 1 do Aluno A; o acumulado (camadas 0 a k); um tensor em 32 bits por vez; e a forma com as 12 multiplicações da saída da rede densa em 32 bits. Todos em dev; a leitura no teste continuou a fixada (a alternativa de menor perda média em dev).
- **Achado 2, calibração por histograma.** A calibração por percentil e por entropia do ONNX Runtime 1.32 falha com pedidos de comprimentos diferentes (ela junta as ativações de todos os pedidos num só arranjo). Ela foi chamada um pedido por vez, somando os histogramas, sem mudar mais nada.
- **Achado 2, execução.** Dois processos de quantização na mesma rede de entrada colidiram (a ferramenta grava e apaga uma cópia ao lado do arquivo de entrada); as duas execuções que falharam foram apagadas e refeitas, e desde então cada quantização lê a rede por um link físico na sua pasta. Nenhuma medida concluída foi afetada: as que terminaram repetem a 10c onde podem ser comparadas.
- **Achado 1.** Nenhum desvio: os 21 reajustes e as contas são os do plano. O Aluno A não teve controle novo sem corte porque a 10b já o tinha.

## Critério de pronto

| Critério | Atendido |
|---|---|
{linhas_de_pronto}

## Limites

- Tudo é exploratório e depois dos resultados; os testes foram escolhidos sabendo dos efeitos. Os intervalos não são corrigidos para o número de comparações.
- Três sementes. Os achados 2 e 3 são sobre o XLM-R base com o ONNX Runtime 1.32 em CPU; a rede do zero não foi investigada.
- Só os resumos dos artigos foram lidos.
- Durante a etapa apareceram pastas que não são dela, {', '.join(f'`{p}/`' for p in pastas_novas) or 'nenhuma'} ({len(novos)} arquivos novos), escritas por outros processos (outro trabalho rodou na máquina ao mesmo tempo); nenhum arquivo das outras etapas mudou nem sumiu. Esta etapa não mede latência, então a máquina dividida não muda nenhum número dela.
- O achado 1 usa os alunos do Qwen3-14B (11b/11c); o B2 do Qwen3-8B (10b) entra só nas contas sobre respostas gravadas, sem reajuste novo.
- O que NÃO foi verificado: por que só o corte custa em dev e não no teste; que camadas concentram o erro da intenção na INT8 dinâmica; o degrau por grupo de dimensões na estática; por que a poda gradual protege da estática; latência de qualquer alternativa (não pedida).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `plano.json`, `plano.sha256` | O plano, gravado antes de qualquer execução |
| `achado1_*.csv`, `achado1_corte.json` | Achado 1: as comparações em dev, a cópia do professor, as curvas por época, o treino, a diferença das diferenças e o corte |
| `achado2_*.csv` | Achado 2: as faixas por tensor e semente, cada quantização de diagnóstico (F1 e tamanho) e as alternativas com intervalo |
| `achado3_*.csv`, `achado3_resumo.json` | Achado 3: as partes, cada troca de intenção, as alternativas e os 9 modelos |
| `teste.json` | A leitura única no teste de cada achado, com a regra |
| `artigos.csv`, `artigos/` | Os artigos e a resposta da API do arXiv |
| `reajustes/` | Os 21 reajustes: histórico, meta e as respostas em dev de cada época |
| `dinamica/`, `estatica/` | As respostas em dev (e no teste, da alternativa lida) de cada quantização de diagnóstico |
| `impressoes/` | As impressões digitais das outras etapas, antes e depois |
| `codigo/` | O código da etapa, o construtor deste notebook e os testes |
| `figuras/`, `12_investigacao_dos_achados.html` | Os gráficos e este notebook executado |
| `manifesto.json` | Tudo o que a entrega contém, com impressão digital |
"""
(PASTA / "RESUMO.md").write_text(resumo, encoding="utf-8")
display(Markdown(resumo))
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("ok", len(cells), "células")
