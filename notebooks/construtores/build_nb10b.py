"""Monta notebooks/10b_encoder_poda.ipynb a partir das células abaixo.

O orçamento do reajuste e as peças que saem (épocas, rampa, camadas, formas
do vocabulário), que aparecem no texto "antes de medir", são lidos de
`especialista.experimentos.etapa10b` na hora em que o notebook é gerado. Se
mudarem, gere o notebook de novo: a primeira célula de código confere.

    uv run python notebooks/construtores/build_nb10b.py notebooks/10b_encoder_poda.ipynb
"""
import sys
import nbformat as nbf

from especialista.experimentos import etapa10b

cells = []
ORCAMENTO = {
    "EPOCAS": etapa10b.EPOCAS,
    "RAMPA": etapa10b.EPOCAS_DE_RAMPA,
    "TRAVADA": etapa10b.EPOCAS - etapa10b.EPOCAS_DE_RAMPA,
    "CORTES": etapa10b.PASSOS_ENTRE_CORTES,
    "CURTAS": etapa10b.EPOCAS_CURTAS,
    "CAMADAS": " e ".join(str(k) for k in etapa10b.CAMADAS),
    "NIVEIS": ", ".join(f"{s}%" for s in etapa10b.ESPARSIDADES),
}


def com_orcamento(texto: str) -> str:
    for chave, valor in ORCAMENTO.items():
        texto = texto.replace(f"«{chave}»", str(valor))
    return texto


md = lambda s: cells.append(nbf.v4.new_markdown_cell(com_orcamento(s.strip("\n"))))
code = lambda s: cells.append(nbf.v4.new_code_cell(com_orcamento(s.strip("\n"))))

md(r'''
# Etapa 10b · Poda e fine-tuning de recuperação do XLM-R: quanto dá para tirar, e de onde

| | |
|---|---|
| **Pergunta** | Quanto dá para tirar do XLM-R, de que parte, e quanto o reajuste recupera? |
| **Entrada** | `entregas/01_dados/` (os pedidos), `entregas/10a_encoder/` (o teto e os alunos com o XLM-R, a decisão e o sinal de treino de cada um), `entregas/07_poda/` (a curva da rede do zero, para pôr ao lado) e as execuções gravadas pelos comandos de `especialista/experimentos/etapa10b.py`. |
| **Entrega** | `entregas/10b_encoder_poda/`: a curva de qualidade por esparsidade, antes e depois do reajuste; o corte do vocabulário e das camadas; o que cada corte muda de verdade em parâmetros, bytes e milissegundos; os modelos que seguem para a quantização (Etapa 10c) e o `decisao.json` que diz onde estão. |
| **Pronto quando** | (1) cada variante tem 3 sementes, com o mesmo orçamento de reajuste; (2) a esparsidade pedida é a obtida; (3) nenhum rótulo humano entrou no reajuste dos alunos; (4) a escolha das linhas do vocabulário não usou `dev` nem o teste; (5) cada variante tem a medida antes e depois do reajuste, com comparação pareada contra o mesmo modelo sem poda e intervalo de 95%, no teste; (6) tamanho e latência foram medidos em arquivos gravados e modelos carregados, com a rede separada do tokenizador; (7) o que segue para a quantização saiu de `dev`, pela regra escrita antes; (8) os modelos entregues reproduzem as respostas do teste. |

**De onde esta etapa parte.** A Etapa 7 podou a rede do zero (4,9 milhões de parâmetros). Aqui o mesmo roteiro é aplicado ao XLM-R base da Etapa 10a, que é 57 vezes maior e tem uma forma muito diferente: 69% dos parâmetros estão numa tabela de subtokens feita para 100 idiomas, e o resto, num corpo de 12 camadas. **Podar** é tirar o que menos faz falta e treinar mais um pouco para o que ficou compensar; esse treino a mais é o **reajuste**. Tira-se de dois jeitos: **pesos soltos**, espalhados pelas matrizes, ou **peças inteiras** (linhas da tabela, camadas).

**Artigos de apoio**

- Treinar, podar pelos pesos de menor valor absoluto e treinar de novo: Han et al., *Learning both Weights and Connections for Efficient Neural Networks* ([arXiv 1506.02626](https://arxiv.org/abs/1506.02626)).
- Poda gradual, com a esparsidade subindo durante o treino: Zhu & Gupta, *To prune, or not to prune* ([arXiv 1710.01878](https://arxiv.org/abs/1710.01878)).
- Poda por magnitude no BERT, matriz por matriz (6 por camada): Gordon, Duh & Andrews, *Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning* ([arXiv 2002.08307](https://arxiv.org/abs/2002.08307)). De 30% a 40% dos pesos saem sem perda; daí para cima a qualidade cai.
- Magnitude contra outros critérios num encoder pré-treinado, com a mesma curva cúbica: Sanh, Wolf & Rush, *Movement Pruning: Adaptive Sparsity by Fine-Tuning* ([arXiv 2005.07683](https://arxiv.org/abs/2005.07683)). As tabelas de entrada ficam fora da poda e a esparsidade é contada no corpo.
- Tirar as camadas de cima de um encoder já treinado e ajustar: Sajjad, Dalvi, Durrani & Nakov, *On the Effect of Dropping Layers of Pre-trained Transformer Models* ([arXiv 2004.03844](https://arxiv.org/abs/2004.03844)).
- Cortar da tabela os subtokens que o corpus da tarefa não usa, e do tokenizador junto: Yang, Cui & Chen, *TextPruner: A Model Pruning Toolkit for Pre-Trained Language Models* ([arXiv 2203.15996](https://arxiv.org/abs/2203.15996)).
- Num modelo multilíngue, a maior parte dos parâmetros está na tabela, e ficar só com os idiomas que interessam encolhe o modelo: Abdaoui, Pradel & Sigel, *Load What You Need: Smaller Versions of Multilingual BERT* ([arXiv 2010.05609](https://arxiv.org/abs/2010.05609)).
- Máscara travada no reajuste, só com o sinal do professor: Zafrir et al., *Prune Once for All* ([arXiv 2111.05754](https://arxiv.org/abs/2111.05754)).
- Guardar só os pesos que ficaram, e a ordem podar e depois quantizar: Han et al., *Deep Compression* ([arXiv 1510.00149](https://arxiv.org/abs/1510.00149)).
- Como reportar poda (tamanho e velocidade, controle, sementes): Blalock et al., *What is the State of Neural Network Pruning?* ([arXiv 2003.03033](https://arxiv.org/abs/2003.03033)). Parâmetros, bytes e tempo não são a mesma medida: Dehghani et al., *The Efficiency Misnomer* ([arXiv 2110.12894](https://arxiv.org/abs/2110.12894)).
- O encoder e a cabeça conjunta: Conneau et al., XLM-R ([arXiv 1911.02116](https://arxiv.org/abs/1911.02116)); Chen et al., JointBERT ([arXiv 1902.10909](https://arxiv.org/abs/1902.10909)).
''')

code(r'''
import json
import os
import shutil
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

from especialista import visual
from especialista.avaliacao import qualidade, resultados
from especialista.avaliacao.previsao import ouro, salvar_previsoes
from especialista.caminhos import ENTREGAS, RAIZ, entrega
from especialista.compressao import poda_encoder
from especialista.dados import massive
from especialista.experimentos import etapa10b
from especialista.modelos.encoder import SistemaEncoder

visual.aplicar_estilo()
pd.set_option("display.max_colwidth", 200)

PASTA = entrega("10b_encoder_poda")
SEMENTES, NIVEIS, VARIANTES, CURTAS = etapa10b.SEMENTES, etapa10b.ESPARSIDADES, etapa10b.VARIANTES, etapa10b.CURTAS
EPOCAS, RAMPA, CORTES = etapa10b.EPOCAS, etapa10b.EPOCAS_DE_RAMPA, etapa10b.PASSOS_ENTRE_CORTES
assert (EPOCAS, RAMPA, CORTES, etapa10b.EPOCAS_CURTAS) == («EPOCAS», «RAMPA», «CORTES», «CURTAS»), "O texto deste notebook foi gerado com outro orçamento de reajuste: gere o notebook de novo."

MODELOS = etapa10b.modelos_a_podar()                                        # o aluno principal, o outro aluno, o teto
PRINCIPAL = etapa10b.modelo_principal()
COMPLETOS = [m for m in MODELOS if MODELOS[m]["grade"] == "completa"]       # os que têm a grade inteira
DEPENDENTES = [m for m in MODELOS if MODELOS[m]["grade"] != "completa"]     # o que só recebe o que segue do principal
ALUNOS = [m for m in MODELOS if m != "teto"]
CURTO = {m: MODELOS[m]["curto"] for m in MODELOS}
NOME = {**{v: d["nome"] for v, d in VARIANTES.items()}, **{v: d["nome"] for v, d in CURTAS.items()}}

DE_UMA_VEZ = {s: f"magnitude_{s}" for s in NIVEIS}
GRADUAL = {s: f"gradual_{s}" for s in NIVEIS}
PODADAS = [v for v in VARIANTES if VARIANTES[v]["familia"] != "controle"]
PECAS = [v for v in PODADAS if VARIANTES[v]["familia"] in etapa10b.PECAS_INTEIRAS]
SOLTAS = [v for v in PODADAS if VARIANTES[v]["familia"] in etapa10b.PESOS_SOLTOS]
COM_ANTES = [v for v in PODADAS if VARIANTES[v]["familia"] not in etapa10b.SEM_ANTES]
ORDEM = ["sem_poda", *DE_UMA_VEZ.values(), *GRADUAL.values(), *PECAS]      # a ordem das tabelas

particoes, rotulos = massive.carregar_entrega()
treino, dev, teste = (particoes[p] for p in ("train", "dev", "test"))
gabarito = {"dev": ouro(dev), "test": ouro(teste)}
TRES = ("intent_acc", "slot_f1", "exact_match")   # as três métricas das leituras


def n(x):
    return f"{int(x):,}".replace(",", ".")


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def pts(x, casas=1):
    return num(100 * x, casas)


def dif(x):
    return f"{100 * x:+.1f}".replace(".", ",").replace("-", "−")


def mb(x):
    return f"{x / 1e6:,.1f}".replace(",", "#").replace(".", ",").replace("#", ".")


def com_intervalo(linha):
    return f"{dif(linha['diferenca'])} ({dif(linha['ic_min'])} a {dif(linha['ic_max'])})"


def valor_com_intervalo(linha):
    return f"{pts(linha['valor'])} ({pts(linha['ic_min'])} a {pts(linha['ic_max'])})"


def posicao(linha):
    """Onde B fica em relação a A numa comparação pareada B − A."""
    return "acima" if linha["ic_min"] > 0 else "abaixo" if linha["ic_max"] < 0 else "sem diferença distinguível"


def trio(linha, coluna_do_valor="valor"):
    """Valor e intervalo de uma linha de tabela, em pontos percentuais, para os gráficos."""
    return tuple(100 * linha[c] for c in (coluna_do_valor, "ic_min", "ic_max"))


def lista(itens, vazio="nenhum"):
    """"a", "a e b", "a, b e c"."""
    itens = [str(i) for i in itens]
    return vazio if not itens else itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def minusculo(texto):
    return texto[0].lower() + texto[1:]


curva = visual.curva_com_intervalo
''')

md(r'''
## 1. Onde estão os pesos do XLM-R

Antes de cortar, a conta de onde os parâmetros estão. É ela que diz o que cada tipo de corte pode alcançar.
''')

code(r'''
de_partida = json.loads((ENTREGAS / etapa10b.DECISAO_DO_ENCODER).read_text(encoding="utf-8"))["custo_de_partida"]
partes, total = de_partida["parametros_por_parte"], de_partida["parametros"]
PODAVEIS = 12 * (4 * 768 * 768 + 2 * 768 * 3072)      # as 72 matrizes do corpo (conferido num teste e na seção 4)
LINHAS, USADAS = de_partida["linhas_da_tabela_de_subtokens"], de_partida["linhas_da_tabela_usadas_no_treino"]
quem_alcanca = {
    "A. Tabela de subtokens": "o corte do vocabulário (linhas inteiras)",
    "A. Posições e normalização da entrada": "nada",
    "B. Corpo do Transformer": "a poda por magnitude (pesos soltos) e o corte de camadas",
    "C. Cabeça de intenção": "nada", "D. Cabeça de entidades": "nada",
}
display(pd.DataFrame([
    {"parte": parte, "parâmetros": n(quantos), "parte do modelo": f"{pts(quantos / total)}%", "o que esta etapa pode tirar daí": quem_alcanca[parte]}
    for parte, quantos in partes.items()
] + [{"parte": "Total", "parâmetros": n(total), "parte do modelo": "100,0%", "o que esta etapa pode tirar daí": ""}]).set_index("parte"))
display(Markdown(f"""
**Leitura.**

- **A poda por magnitude age em {n(PODAVEIS)} pesos**: as 6 matrizes de cada uma das 12 camadas. São {pts(PODAVEIS / partes['B. Corpo do Transformer'])}% do corpo, mas só {pts(PODAVEIS / total)}% do modelo. Mesmo com 90% desses pesos em zero, {pts(1 - 0.9 * PODAVEIS / total)}% dos parâmetros do modelo continuam diferentes de zero.
- **A tabela tem {n(LINHAS)} linhas e o treino em português usa {n(USADAS)}** ({pts(USADAS / LINHAS)}%). Cada linha tem 768 números. É aqui que está a maior parte do tamanho, e é o corte do vocabulário que chega nela.
- **Cada camada do corpo tem {n(partes['B. Corpo do Transformer'] / 12)} parâmetros** ({pts(partes['B. Corpo do Transformer'] / 12 / total)}% do modelo). Tirar camadas diminui pouco o arquivo, mas é o único corte que diminui a conta feita a cada consulta.
"""))
''')

md(r'''
## 2. O que ficou decidido antes de medir

Tudo abaixo estava escrito em `experimentos/etapa10b.py` e em `compressao/poda_encoder.py` antes de a grade rodar.

**1. Quem é podado.** A grade completa roda no **aluno A** (o modelo final deste caminho) e no **teto** (a referência: a compressão sem o efeito do professor). O **aluno B2** recebe só as duas variantes que a regra apontar como principal e secundária para o aluno A, para que exista a cadeia do cenário B no comparativo. Ele não tem grade nem regra própria.

**2. Com que sinal é reajustado.** Com o sinal de quem treinou o modelo: o aluno, com os rótulos (ou os logits) do seu professor, e a época de parada escolhida no `dev` rotulado pelo mesmo professor; o teto, com os rótulos humanos. Nenhum rótulo humano entra no reajuste de um aluno. É o que Zafrir et al. fazem ao reajustar um modelo esparso.

**3. Pesos soltos, por magnitude.** Em cada uma das 72 matrizes do corpo, saem os pesos de menor valor absoluto, na mesma fração em todas (corte por matriz, como em Gordon et al. e na Etapa 7). Níveis: «NIVEIS». Duas formas: **de uma vez** (Han et al.), com medida antes e depois do reajuste, e **gradual** (Zhu & Gupta), com a esparsidade subindo pela curva cúbica nas «RAMPA» primeiras épocas do reajuste, um corte a cada «CORTES» passos, e a máscara travada nas «TRAVADA» últimas.

**4. A tabela de subtokens fica fora da poda por magnitude.** Sanh et al. também deixam as tabelas de entrada de fora e contam a esparsidade no corpo. Gordon et al. podam a tabela "do mesmo jeito": aqui isso é um **desvio declarado**. O motivo: 98% das linhas da tabela nunca são lidas em português; zerar pesos nelas daria esparsidade de graça, sem dizer nada sobre a rede. A tabela é tratada pelo corte do vocabulário.

**5. Peças inteiras (i): o corte do vocabulário.** Saem linhas inteiras da tabela, e o tokenizador é reduzido às que ficam (é o que fazem o TextPruner e Abdaoui et al.). É o análogo das "linhas da tabela de features" da Etapa 7. Duas formas de escolher as linhas que ficam, e **nenhuma olha o texto de `dev` nem do teste**:

- **treino**: os subtokens que aparecem no texto de treino pt-PT (os pedidos, sem rótulo). É a escolha do TextPruner para o XLM-R.
- **alfabeto**: os subtokens escritos só com letras latinas, dígitos e pontuação. Não usa corpus nenhum; é uma escolha deste projeto, na linha de Abdaoui et al.

Uma palavra que usava uma peça cortada é partida de novo pelo tokenizador, em peças menores que ficaram; o que não tem peça nenhuma vira `<unk>`. A fração dos subtokens de `dev` que cada forma corta é relatada na seção 7, como diagnóstico: não escolhe nada.

**6. Peças inteiras (ii): as camadas de cima.** O corpo fica com as «CAMADAS» camadas de baixo, das 12. Sajjad et al. comparam seis formas de descartar camadas, tirar as de cima é a melhor, e eles testam tirar 2, 4 e 6 das 12. **(iii) Duas combinações**, definidas antes: o vocabulário do treino com cada um dos dois cortes de camadas.

**7. O reajuste.** «EPOCAS» épocas em todas as execuções, com a receita de quem treinou o modelo (taxa 5e-5, lote 32, a mesma agenda da taxa) e a máscara travada. É o orçamento do ajuste fino da Etapa 10a inteiro, como nos artigos de poda de encoders pré-treinados: Sajjad et al. tiram as camadas e fazem o ajuste fino de sempre; Sanh et al. podam durante o ajuste fino inteiro. É também o critério da Etapa 7, que deu ao reajuste as épocas que o treino original leva para chegar ao melhor ponto (30 de 80); aqui o treino chega ao melhor ponto entre a 7ª e a 10ª época, de 10. Fica o melhor ponto em `dev`, e o ponto de partida concorre como "época 0": se o reajuste não melhora o modelo, o modelo sai como entrou. Um **controle** faz o mesmo reajuste sem tirar nada. **Este orçamento veio de um piloto no teto**, mostrado logo abaixo: o primeiro desenho dava «CURTAS» épocas ao reajuste.

**8. O controle do orçamento.** Duas variantes do aluno A (magnitude de 90% e 6 camadas) são reajustadas também com «CURTAS» épocas: 40% do treino, a proporção que o reajuste teve na Etapa 7. Serve só para medir, no aluno, quanto o orçamento pesa na recuperação. Não entra na regra.

**9. A regra do que segue para a quantização**, a mesma da Etapa 7. Em `dev`, com os rótulos humanos: **passa** a variante cujo exact match, depois do reajuste, não é distinguivelmente menor que o do modelo sem poda (intervalo de 95% da diferença pareada, média de 3 sementes). A **principal** é a que passa com o menor arquivo comum; se nenhuma peça inteira passa, é o modelo sem poda. A **secundária** é a de pesos soltos que passa com a maior esparsidade.

**10. As comparações**, fixadas antes: cada variante contra o modelo sem poda (pareada, no teste, nas três métricas); depois × antes do reajuste; gradual × de uma vez no mesmo nível; reajuste de «EPOCAS» épocas × de «CURTAS»; e, no texto, a curva do XLM-R ao lado da curva da rede do zero, para o aluno A.

**11. O custo de cada variante.** Parâmetros e diferentes de zero; tamanho em três formatos (arquivo comum, com zip, arquivo esparso), com a rede separada do tokenizador; latência p50 em CPU com 1 thread, do texto à resposta.

**12. Como ler os resultados.** Gordon et al. veem o BERT aguentar de 30% a 40% dos pesos a menos sem perda, e cair daí para cima. Sanh et al. medem, com magnitude e 10% dos pesos do corpo, 77,8% no MNLI contra 84,5% do modelo inteiro. Sajjad et al. medem menos de 1 ponto de queda média no GLUE ao tirar 6 das 12 camadas do RoBERTa, e 2,9 pontos no BERT. O TextPruner corta o XLM-R de 1.060 MB para 406 MB só no vocabulário, sem perda no XNLI em inglês. A expectativa, então: os pesos soltos devem custar mais aqui do que na rede do zero, que aguentou 90%; o vocabulário deve sair quase de graça; as camadas ficam no meio.

**O que já tinha sido visto.** Um piloto com 6 execuções do teto (semente 0) rodou antes da grade, para medir o tempo e conferir a esparsidade. Ele mudou uma coisa no desenho: o orçamento do reajuste. As outras regras não mudaram depois dele, e as medidas dos alunos não tinham sido vistas. O que o piloto mostrou está gravado em `piloto.json`, ao lado das execuções:
''')

code(r'''
piloto = json.loads((etapa10b.pasta_da_etapa() / "piloto.json").read_text(encoding="utf-8"))
do_piloto = lambda variante, epocas: next(e for e in piloto["execucoes"] if e["variante"] == variante and e["epocas"] == epocas)
display(pd.DataFrame([
    {
        "variante (teto, semente 0)": NOME[e["variante"]], "épocas de reajuste": e["epocas"],
        "exact match em dev antes do reajuste": pts(e["dev_antes"]["exact_match"]) if e.get("dev_antes") else "—",
        "exact match em dev depois": pts(e["dev"]["exact_match"]), "época escolhida": e["melhor_epoca"],
        "acurácia de intenção em dev, época por época": " ".join(pts(h["dev_intent_acc"], 0) for h in e["historico"]),
    }
    for e in piloto["execucoes"]
]).set_index("variante (teto, semente 0)"))
curto_do_piloto, longo_do_piloto = do_piloto("camadas_6", etapa10b.EPOCAS_CURTAS), do_piloto("camadas_6", EPOCAS)
decola = next(h["epoca"] for h in longo_do_piloto["historico"] if h["dev_intent_acc"] > 0.5)
display(Markdown(f"""
**Leitura do piloto (teto, semente 0, `dev`).**

- **Sem as 6 camadas de cima, {etapa10b.EPOCAS_CURTAS} épocas não bastam:** o exact match fica em {pts(curto_do_piloto["dev"]["exact_match"])}%. As entidades voltam (F1 de {pts(curto_do_piloto["historico"][-1]["dev_slot_f1"])}%), mas a intenção não: {pts(curto_do_piloto["historico"][-1]["dev_intent_acc"])}% de acurácia. A cabeça de intenção lê o primeiro token, e na 6ª camada ele ainda não resume o pedido; a rede precisa reaprender isso.
- **Com {EPOCAS} épocas, a mesma rede chega a {pts(longo_do_piloto["dev"]["exact_match"])}%** de exact match. A intenção só passa de 50% na {decola}ª época: num reajuste de {etapa10b.EPOCAS_CURTAS} épocas a taxa de aprendizado já está quase em zero nesse ponto.
- **Por isso o orçamento passou de {etapa10b.EPOCAS_CURTAS} para {EPOCAS} épocas, para todas as variantes.** Com o orçamento curto, o resultado das camadas seria decidido pelo orçamento, não pelo corte. O controle da seção 7 mede a mesma coisa no aluno A, com 3 sementes.
"""))
''')

md(r'''
## 3. As execuções

As execuções rodam fora do notebook (`uv run python -m especialista.experimentos.etapa10b onda`) e ficam gravadas em `data/execucoes/10b_encoder_poda/`. Aqui elas só são lidas.
''')

code(r'''
faltam = etapa10b.faltam()
assert not faltam, ("Faltam execuções. Para listar: uv run python -m especialista.experimentos.etapa10b --faltam. Faltam: "
                    + ", ".join(f"{m} {v} {s}" for m, v, s in faltam[:12]) + (" ..." if len(faltam) > 12 else ""))
VARIANTES_DE = {m: etapa10b.variantes_do_modelo(m) for m in MODELOS}
exe = {m: {v: [etapa10b.carregar(m, v, s) for s in SEMENTES] for v in VARIANTES_DE[m]} for m in MODELOS}
exe[PRINCIPAL].update({v: [etapa10b.carregar(PRINCIPAL, v, s) for s in SEMENTES] for v in CURTAS})
# O modelo sem poda: as respostas que a Etapa 10a entregou
sem_poda = {m: {p: etapa10b.previsoes_sem_poda(m, p) for p in ("dev", "test")} for m in MODELOS}


def grupo(m, v, particao, momento="depois"):
    """As respostas das 3 sementes de uma variante, antes ou depois do reajuste."""
    return [e["previsoes"][particao if momento == "depois" else f"{particao}_antes"] for e in exe[m][v]]


todas = [(m, v, e) for m in MODELOS for v in exe[m] for e in exe[m][v]]
da_grade = [(m, v, e) for m, v, e in todas if v in VARIANTES]

# 1. Os modelos de partida são os que a Etapa 10a entregou: o "antes" do controle bate com as respostas gravadas lá
iguais_as_entregues = min(
    np.mean([a == b for a, b in zip(daqui, de_la)])
    for m in COMPLETOS for p in ("dev", "test") for daqui, de_la in zip(grupo(m, "sem_poda", p, "antes"), sem_poda[m][p])
)
# 2. O orçamento: toda execução tem as épocas declaradas e a receita de quem treinou o modelo
epocas_de = lambda v: etapa10b.definicao(v)["epocas"]
orcamento_confere = all(e["meta"]["receita"] == etapa10b.receita_do_reajuste(m, epocas_de(v)).como_dict()
                        and e["historico"]["epoca"].max() == epocas_de(v) for m, v, e in todas)
com_partida = [(m, v, e) for m, v, e in da_grade if VARIANTES[v]["familia"] not in etapa10b.SEM_ANTES]
ficou_na_partida = [(m, v, e) for m, v, e in com_partida if e["meta"]["melhor_epoca"] == 0]
# 3. A esparsidade pedida é a obtida, peso por peso
zeros_de = lambda e: e["meta"]["contagem"]["podaveis"] - e["meta"]["contagem"]["nao_nulos_podaveis"]
pesos_por_matriz = [768 * 768] * 48 + [768 * 3072] * 24
zeros_pedidos = lambda v: sum(round(VARIANTES[v]["nivel"] / 100 * pesos) for pesos in pesos_por_matriz)
soltas = [(m, v, e) for m, v, e in da_grade if v in SOLTAS]
sobra_de_zeros = max(zeros_de(e) - zeros_pedidos(v) for m, v, e in soltas)
falta_de_zeros = min(zeros_de(e) - zeros_pedidos(v) for m, v, e in soltas)
esparsidade_confere = 0 <= falta_de_zeros and sobra_de_zeros <= 100 and all(e["meta"]["contagem_de_partida"]["podaveis"] == PODAVEIS for m, v, e in todas)
sem_zeros_onde_nao_se_pediu = max(e["meta"]["contagem"]["esparsidade"] for m, v, e in da_grade if v not in SOLTAS)
# 4. A poda gradual seguiu a curva e só escolheu época depois da rampa
passos = int(exe[PRINCIPAL]["sem_poda"][0]["historico"]["passos"].max())
fim_da_rampa = RAMPA * passos // CORTES * CORTES
planejada = lambda s: np.array([poda_encoder.esparsidade_cubica(min(epoca * passos // CORTES * CORTES, fim_da_rampa), fim_da_rampa, s / 100)
                                for epoca in range(1, EPOCAS + 1)])
desvio_da_curva = max(np.abs(e["historico"]["esparsidade"].to_numpy() - planejada(s)).max() for s in NIVEIS for e in exe[PRINCIPAL][GRADUAL[s]])
gradual_confere = desvio_da_curva < 0.005 and all(e["meta"]["melhor_epoca"] >= RAMPA for m in COMPLETOS for s in NIVEIS for e in exe[m][GRADUAL[s]])
# 5. Nenhum rótulo humano no reajuste dos alunos: o sinal anotado em cada execução é o do professor
sem_humano = all(e["meta"]["origem"]["treino"].startswith("rótulos do professor") and e["meta"]["origem"]["dev"].startswith("rótulos do professor")
                 for m, v, e in todas if m in ALUNOS)
do_teto_e_humano = all(e["meta"]["origem"]["treino"].startswith("rótulos humanos") for m, v, e in todas if m == "teto")
# 6. O corte do vocabulário: as linhas que ficam não dependem de dev nem do teste (a forma "treino" cobre o treino inteiro, por construção)
cortes = [(m, v, e) for m, v, e in todas if e["meta"]["vocabulario"]]
treino_coberto = all(e["meta"]["vocabulario"]["cobertura"]["treino"]["subtokens_cortados"] == 0 for m, v, e in cortes if e["meta"]["vocabulario"]["forma"] == "treino")
mesmas_linhas = {forma: {e["meta"]["vocabulario"]["linhas"] for m, v, e in cortes if e["meta"]["vocabulario"]["forma"] == forma} for forma in etapa10b.FORMAS}

horas = sum(e["meta"]["segundos"] for m, v, e in todas) / 3600
onde_rodou = sorted({e["meta"]["respostas_em"] for m, v, e in todas})

epocas_escolhidas = lambda m, v: ", ".join(str(e["meta"]["melhor_epoca"]) for e in exe[m][v])
no_dev = lambda m, v, quando: np.mean([e["meta"][quando]["exact_match"] for e in exe[m][v]])
print(f"{MODELOS[PRINCIPAL]['nome']}: cada variante no dev que escolhe a época de parada (média de 3 sementes)")
pd.DataFrame([
    {
        "variante": NOME[v],
        "época escolhida (sementes 0, 1 e 2)": epocas_escolhidas(PRINCIPAL, v),
        "exact match antes do reajuste": pts(no_dev(PRINCIPAL, v, "dev_antes")) if exe[PRINCIPAL][v][0]["meta"]["dev_antes"] else "—",
        "exact match depois": pts(no_dev(PRINCIPAL, v, "dev")),
        "minutos por execução": num(np.mean([e["meta"]["segundos"] for e in exe[PRINCIPAL][v]]) / 60, 1),
    }
    for v in [*ORDEM, *CURTAS]
]).set_index("variante")
''')

code(r'''
display(Markdown(f"""
**Leitura.**

- **São {len(todas)} execuções**: {len(COMPLETOS)} modelos com a grade de {len(VARIANTES)} variantes × {len(SEMENTES)} sementes, {lista([f"{len(VARIANTES_DE[m])} variantes do {minusculo(CURTO[m])}" for m in DEPENDENTES])} e {len(CURTAS)} variantes do controle do orçamento. Somam {num(horas, 1)} horas de processamento, em {lista(onde_rodou)}.
- **Modelos de partida:** as respostas dos modelos sem poda, refeitas aqui antes do reajuste do controle, são iguais às que a Etapa 10a entregou em pelo menos {pts(iguais_as_entregues, 2)}% dos pedidos de cada conjunto.
- **Orçamento igual para todas:** {"confere" if orcamento_confere else "NÃO confere"} ({EPOCAS} épocas na grade, {etapa10b.EPOCAS_CURTAS} no controle do orçamento, com a receita de quem treinou o modelo).
- **Quando o reajuste não ajuda, ele não fica.** Em {len(ficou_na_partida)} das {len(com_partida)} execuções em que o ponto de partida concorre, o melhor ponto em `dev` foi o próprio ponto de partida (época 0).
- **Esparsidade pedida e obtida:** {"confere" if esparsidade_confere else "NÃO confere"}. Nas {len(soltas)} execuções de pesos soltos, o número de zeros nas matrizes podáveis é o pedido (faltam no máximo {max(0, -falta_de_zeros)} e sobram no máximo {sobra_de_zeros} pesos, em {n(PODAVEIS)}). Nas demais, a maior fração de zeros é {pts(sem_zeros_onde_nao_se_pediu, 3)}%.
- **Poda gradual:** {"confere" if gradual_confere else "NÃO confere"}. A esparsidade medida ao fim de cada época segue a curva cúbica (maior desvio: {pts(desvio_da_curva, 2)} ponto), e toda época escolhida é do fim da rampa para a frente (a partir da {RAMPA}ª).
- **Sem rótulo humano no reajuste dos alunos:** {"confere" if sem_humano and do_teto_e_humano else "NÃO confere"}.
- **O corte do vocabulário não olhou `dev` nem o teste:** a forma "treino" fica com {lista(n(x) for x in sorted(mesmas_linhas["treino"]))} linhas e cobre o treino inteiro ({"confere" if treino_coberto else "NÃO confere"}); a forma "alfabeto" fica com {lista(n(x) for x in sorted(mesmas_linhas["alfabeto"]))}, escolhidas só pela grafia.
"""))
''')

md(r'''
## 4. Pesos soltos: antes e depois do reajuste

A poda por magnitude feita de uma vez, nos quatro níveis, no teste. Cada nível tem duas medidas: o modelo logo depois do corte (**antes** do reajuste) e o mesmo modelo reajustado com a máscara travada (**depois**). A referência é o modelo sem poda. O ponto de 0% da série "depois" é o controle: o mesmo reajuste, sem tirar nada.
''')

code(r'''
_medidas, _contra, _recupera = {}, {}, {}


def medida(m, v, momento="depois", p="test"):
    """Valor e intervalo de 95% de cada métrica, média de 3 sementes. `v=None` é o modelo sem poda."""
    chave = (m, v, momento, p)
    if chave not in _medidas:
        _medidas[chave] = qualidade.intervalo_do_grupo(gabarito[p], sem_poda[m][p] if v is None else grupo(m, v, p, momento))
    return _medidas[chave]


def contra(m, v, momento="depois", p="test"):
    """Comparação pareada: a variante menos o mesmo modelo sem poda."""
    chave = (m, v, momento, p)
    if chave not in _contra:
        _contra[chave] = qualidade.comparar_grupos(gabarito[p], sem_poda[m][p], grupo(m, v, p, momento))
    return _contra[chave]


def recupera(m, v, p="test"):
    """Comparação pareada: depois do reajuste menos antes."""
    chave = (m, v, p)
    if chave not in _recupera:
        _recupera[chave] = qualidade.comparar_grupos(gabarito[p], grupo(m, v, p, "antes"), grupo(m, v, p, "depois"))
    return _recupera[chave]


def tem_antes(v):
    return etapa10b.definicao(v).get("familia_de_base", etapa10b.definicao(v)["familia"]) not in etapa10b.SEM_ANTES


def quadro(m, variantes, metrica="exact_match", p="test"):
    """Uma linha por variante: antes, depois, cada um contra o modelo sem poda, e o que o reajuste recupera."""
    linhas = {"Sem poda": {"antes do reajuste": valor_com_intervalo(medida(m, None, p=p).loc[metrica]), "depois do reajuste": "—",
                           "antes − sem poda": "—", "depois − sem poda": "—", "o que o reajuste recupera (depois − antes)": "—"}}
    for v in variantes:
        antes = tem_antes(v) and v != "sem_poda"
        linhas[NOME[v]] = {
            "antes do reajuste": valor_com_intervalo(medida(m, v, "antes", p).loc[metrica]) if antes else "—",
            "depois do reajuste": valor_com_intervalo(medida(m, v, "depois", p).loc[metrica]),
            "antes − sem poda": com_intervalo(contra(m, v, "antes", p).loc[metrica]) if antes else "—",
            "depois − sem poda": com_intervalo(contra(m, v, "depois", p).loc[metrica]),
            "o que o reajuste recupera (depois − antes)": com_intervalo(recupera(m, v, p).loc[metrica]) if antes else "—",
        }
    return pd.DataFrame(linhas).T


def pontos_da_curva(m, por_nivel, momento, metrica="exact_match", p="test"):
    """Os pontos de uma série da curva: 0% é o modelo sem poda (antes) ou o controle (depois)."""
    zero = medida(m, None, p=p) if momento == "antes" else medida(m, "sem_poda", "depois", p)
    return {0: trio(zero.loc[metrica]), **{s: trio(medida(m, v, momento, p).loc[metrica]) for s, v in por_nivel.items()}}


for m in COMPLETOS:
    fig, ax = curva(
        {"Depois do reajuste": pontos_da_curva(m, DE_UMA_VEZ, "depois"), "Antes do reajuste": pontos_da_curva(m, DE_UMA_VEZ, "antes")},
        f"{CURTO[m]} (XLM-R): exact match no teste, por esparsidade (poda de uma vez)", "pesos do corpo em zero (%); os níveis estão a distâncias iguais",
        "exact match (%), média de 3 sementes", niveis=True,
    )
    visual.salvar(fig, PASTA, f"curva_{m}")
    plt.show()

print(f"{MODELOS[PRINCIPAL]['nome']}: exact match no teste, em % (intervalo de 95%); diferenças em pontos percentuais")
display(quadro(PRINCIPAL, ["sem_poda", *DE_UMA_VEZ.values()]))
print(f"{MODELOS[PRINCIPAL]['nome']}: as três métricas no teste, depois do reajuste, contra o modelo sem poda (pontos percentuais, intervalo de 95%)")
display(pd.DataFrame({
    ("0% (controle)" if v == "sem_poda" else f"{VARIANTES[v]['nivel']}%"): {qualidade.NOMES[metrica]: com_intervalo(contra(PRINCIPAL, v).loc[metrica]) for metrica in TRES}
    for v in ["sem_poda", *DE_UMA_VEZ.values()]
}).T)
for m in COMPLETOS:
    if m != PRINCIPAL:
        print(f"\n{MODELOS[m]['nome']}: exact match no teste")
        display(quadro(m, ["sem_poda", *DE_UMA_VEZ.values()]))
''')

code(r'''
def niveis_onde(m, por_nivel, momento, condicao, p="test"):
    return [f"{s}%" for s, v in por_nivel.items() if condicao(contra(m, v, momento, p).loc["exact_match"])]


def leitura_dos_pesos_soltos(m):
    cai_antes = niveis_onde(m, DE_UMA_VEZ, "antes", lambda l: l["ic_max"] < 0)
    cai_depois = niveis_onde(m, DE_UMA_VEZ, "depois", lambda l: l["ic_max"] < 0)
    recuperou = [f"{s}%" for s, v in DE_UMA_VEZ.items() if recupera(m, v).loc["exact_match", "ic_min"] > 0]
    ultimo = DE_UMA_VEZ[max(NIVEIS)]
    return (
        f"sem reajuste, o exact match fica distinguivelmente abaixo do modelo sem poda em {lista(cai_antes, 'nenhum nível')}; "
        f"com reajuste, em {lista(cai_depois, 'nenhum nível')}. O reajuste melhora o modelo podado, com intervalo todo acima de zero, em {lista(recuperou, 'nenhum nível')}. "
        f"Em {max(NIVEIS)}%: {com_intervalo(contra(m, ultimo, 'antes').loc['exact_match'])} antes e {com_intervalo(contra(m, ultimo, 'depois').loc['exact_match'])} depois"
    )


controle = contra(PRINCIPAL, "sem_poda")
display(Markdown(f"""
**Leitura (teste, exact match).**

- **{CURTO[PRINCIPAL]}:** {leitura_dos_pesos_soltos(PRINCIPAL)}.
- **O que o reajuste recupera**, nível a nível ({CURTO[PRINCIPAL]}): {"; ".join(f"{s}%: {com_intervalo(recupera(PRINCIPAL, v).loc['exact_match'])}" for s, v in DE_UMA_VEZ.items())}.
- **O controle.** O mesmo reajuste, sem tirar nada, deixa o {minusculo(CURTO[PRINCIPAL])} {posicao(controle.loc['exact_match'])} em relação a si mesmo ({com_intervalo(controle.loc['exact_match'])} em exact match; {com_intervalo(controle.loc['intent_acc'])} em intenção; {com_intervalo(controle.loc['slot_f1'])} em entidades).
""" + "".join(f"- **{CURTO[m]}:** {leitura_dos_pesos_soltos(m)}.\n" for m in COMPLETOS if m != PRINCIPAL)))
''')

md(r'''
## 5. De uma vez ou gradual?

No mesmo nível de esparsidade e com o mesmo orçamento de «EPOCAS» épocas: cortar tudo no começo ou subir a esparsidade aos poucos nas «RAMPA» primeiras épocas?
''')

code(r'''
_formas = {}


def gradual_contra_de_uma_vez(m, s, p="test"):
    if (m, s, p) not in _formas:
        _formas[m, s, p] = qualidade.comparar_grupos(gabarito[p], grupo(m, DE_UMA_VEZ[s], p), grupo(m, GRADUAL[s], p))
    return _formas[m, s, p]


def quadro_das_formas(m, metrica="exact_match"):
    return pd.DataFrame({
        f"{s}%": {
            "de uma vez, depois do reajuste": valor_com_intervalo(medida(m, DE_UMA_VEZ[s]).loc[metrica]),
            "gradual": valor_com_intervalo(medida(m, GRADUAL[s]).loc[metrica]),
            "gradual − de uma vez": com_intervalo(gradual_contra_de_uma_vez(m, s).loc[metrica]),
            "gradual − sem poda": com_intervalo(contra(m, GRADUAL[s]).loc[metrica]),
        }
        for s in NIVEIS
    }).T


def leitura_das_formas(m):
    acima = [f"{s}%" for s in NIVEIS if gradual_contra_de_uma_vez(m, s).loc["exact_match", "ic_min"] > 0]
    abaixo = [f"{s}%" for s in NIVEIS if gradual_contra_de_uma_vez(m, s).loc["exact_match", "ic_max"] < 0]
    return f"a gradual fica acima da poda de uma vez em {lista(acima, 'nenhum nível')} e abaixo em {lista(abaixo, 'nenhum nível')}"


fig, ax = curva(
    {"Gradual": pontos_da_curva(PRINCIPAL, GRADUAL, "depois"), "De uma vez": pontos_da_curva(PRINCIPAL, DE_UMA_VEZ, "depois")},
    f"{CURTO[PRINCIPAL]} (XLM-R): exact match no teste, depois do reajuste, nas duas formas", "pesos do corpo em zero (%); os níveis estão a distâncias iguais",
    "exact match (%), média de 3 sementes", niveis=True,
)
visual.salvar(fig, PASTA, "de_uma_vez_ou_gradual")
plt.show()
for m in COMPLETOS:
    print(f"{MODELOS[m]['nome']}: exact match no teste")
    display(quadro_das_formas(m))
display(Markdown("**Leitura (teste, exact match).**\n\n" + "".join(f"- **{CURTO[m]}:** {leitura_das_formas(m)}.\n" for m in COMPLETOS)))
''')

md(r'''
## 6. Peças inteiras: o vocabulário e as camadas

Aqui a rede que sobra é uma rede comum, só que menor. Primeiro o diagnóstico do corte do vocabulário: quanto do texto cada forma deixa sem linha na tabela. Ele foi calculado junto com cada execução e **não escolheu nada**.
''')

code(r'''
um_de = lambda forma: next(e for m, v, e in cortes if m == PRINCIPAL and e["meta"]["vocabulario"]["forma"] == forma and e["meta"]["camadas"] == 12)
diagnostico = pd.DataFrame([
    {
        "forma": forma, "o que fica": etapa10b.FORMAS[forma],
        "linhas que ficam": n(um_de(forma)["meta"]["vocabulario"]["linhas"]),
        "parte da tabela": f"{pts(um_de(forma)['meta']['vocabulario']['linhas'] / um_de(forma)['meta']['vocabulario']['linhas_antes'])}%",
        **{
            f"{nome}: subtokens cortados": f"{pts(um_de(forma)['meta']['vocabulario']['cobertura'][p]['fracao_cortada'], 2)}%"
            for p, nome in (("treino", "treino"), ("dev", "dev"), ("test", "teste"))
        },
        "dev: pedidos com algum corte": f"{pts(um_de(forma)['meta']['vocabulario']['cobertura']['dev']['enunciados_com_corte'] / len(dev))}%",
        "dev: subtokens que viram <unk>": f"{pts(um_de(forma)['meta']['vocabulario']['cobertura']['dev']['fracao_desconhecida'], 2)}%",
        "dev: subtokens por pedido, antes → depois": (
            f"{num(um_de(forma)['meta']['vocabulario']['cobertura']['dev']['subtokens_antes'] / len(dev), 2)} → "
            f"{num(um_de(forma)['meta']['vocabulario']['cobertura']['dev']['subtokens_depois'] / len(dev), 2)}"),
    }
    for forma in etapa10b.FORMAS
]).set_index("forma")
display(diagnostico)

# Um exemplo de palavra partida de novo, com os tokenizadores gravados (semente 0)
original = SistemaEncoder.carregar(etapa10b.pasta_do_modelo(PRINCIPAL, etapa10b.DE_PARTIDA), "cpu").tokenizador
reduzido = SistemaEncoder.carregar(etapa10b.pasta_do_modelo(PRINCIPAL, "vocabulario_treino"), "cpu").tokenizador
exemplos = []
for texto in dev["utt"]:
    for palavra in texto.split():
        antes, depois = original.tokenize(palavra), reduzido.tokenize(palavra)
        if antes != depois and len(exemplos) < 6 and palavra not in [e["palavra"] for e in exemplos]:
            ids = reduzido([[palavra]], is_split_into_words=True, add_special_tokens=False)["input_ids"][0]
            exemplos.append({"palavra": palavra, "tokenizador original": " ".join(antes), "tokenizador reduzido (forma treino)": " ".join(depois),
                             "vira <unk>": "sim" if reduzido.unk_token_id in ids else "não"})
display(pd.DataFrame(exemplos).set_index("palavra"))
''')

code(r'''
def quadro_das_pecas(m, variantes=PECAS, p="test"):
    tabela = quadro(m, variantes, p=p).drop(index="Sem poda")
    tabela.insert(0, "parâmetros", [n(exe[m][v][0]["meta"]["contagem"]["parametros"]) for v in variantes])
    tabela.insert(1, "sistema no arquivo comum (MB)", [mb(exe[m][v][0]["meta"]["bytes_em_disco"]) for v in variantes])
    return tabela


fig, ax = visual.diferencas_com_intervalo(
    {NOME[v]: trio(contra(PRINCIPAL, v).loc["exact_match"], "diferenca") for v in PECAS},
    f"{CURTO[PRINCIPAL]} (XLM-R): exact match no teste, depois do reajuste, contra o modelo sem poda", "diferença em pontos percentuais (cortado e reajustado − sem poda)",
)
visual.salvar(fig, PASTA, "pecas_inteiras")
plt.show()
for m in COMPLETOS:
    print(f"{MODELOS[m]['nome']}: exact match no teste, em % (intervalo de 95%); diferenças em pontos percentuais")
    display(quadro_das_pecas(m))
print(f"{MODELOS[PRINCIPAL]['nome']}: as três métricas no teste, depois do reajuste, contra o modelo sem poda")
display(pd.DataFrame({NOME[v]: {qualidade.NOMES[metrica]: com_intervalo(contra(PRINCIPAL, v).loc[metrica]) for metrica in TRES} for v in PECAS}).T)


def leitura_das_pecas(m):
    passam = [minusculo(NOME[v]) for v in PECAS if not contra(m, v).loc["exact_match", "ic_max"] < 0]
    caem = [minusculo(NOME[v]) for v in PECAS if contra(m, v).loc["exact_match", "ic_max"] < 0]
    sem_reajuste = [minusculo(NOME[v]) for v in PECAS if not contra(m, v, "antes").loc["exact_match", "ic_max"] < 0]
    return (f"depois do reajuste, não se distinguem do modelo sem poda em exact match: {lista(passam, 'nenhuma')}; ficam abaixo: {lista(caem, 'nenhuma')}. "
            f"Já sem reajuste, não se distinguem: {lista(sem_reajuste, 'nenhuma')}")


display(Markdown("**Leitura (teste, exact match).**\n\n" + "".join(f"- **{CURTO[m]}:** {leitura_das_pecas(m)}.\n" for m in COMPLETOS)
                 + "".join(f"- **{NOME[v]}** ({CURTO[PRINCIPAL]}): {com_intervalo(contra(PRINCIPAL, v, 'antes').loc['exact_match'])} antes do reajuste, "
                           f"{com_intervalo(contra(PRINCIPAL, v).loc['exact_match'])} depois; o reajuste recupera {com_intervalo(recupera(PRINCIPAL, v).loc['exact_match'])}.\n"
                           for v in PECAS)))
''')

md(r'''
## 7. O orçamento do reajuste limita a recuperação?

Duas variantes do aluno A foram reajustadas também com «CURTAS» épocas (40% do treino, a proporção da Etapa 7), além das «EPOCAS» da grade. A comparação é pareada: a mesma variante, as mesmas sementes, só o número de épocas muda. Este controle não entra na regra do que segue.
''')

code(r'''
_orcamento = {}


def longo_contra_curto(v_curta, p="test"):
    """Comparação pareada: a variante com o reajuste da grade menos a mesma variante com o reajuste curto."""
    if (v_curta, p) not in _orcamento:
        _orcamento[v_curta, p] = qualidade.comparar_grupos(gabarito[p], grupo(PRINCIPAL, v_curta, p), grupo(PRINCIPAL, CURTAS[v_curta]["base"], p))
    return _orcamento[v_curta, p]


C, L = etapa10b.EPOCAS_CURTAS, EPOCAS
display(pd.DataFrame({
    NOME[CURTAS[v]["base"]]: {
        f"{C} épocas": valor_com_intervalo(medida(PRINCIPAL, v).loc["exact_match"]),
        f"{L} épocas": valor_com_intervalo(medida(PRINCIPAL, CURTAS[v]["base"]).loc["exact_match"]),
        **{f"{L} − {C} épocas, {qualidade.NOMES[metrica].lower()}": com_intervalo(longo_contra_curto(v).loc[metrica]) for metrica in TRES},
        f"{C} épocas − sem poda (exact match)": com_intervalo(contra(PRINCIPAL, v).loc["exact_match"]),
        f"{L} épocas − sem poda (exact match)": com_intervalo(contra(PRINCIPAL, CURTAS[v]["base"]).loc["exact_match"]),
        f"época escolhida ({C} épocas)": epocas_escolhidas(PRINCIPAL, v),
        f"época escolhida ({L} épocas)": epocas_escolhidas(PRINCIPAL, CURTAS[v]["base"]),
    }
    for v in CURTAS
}).T)


def leitura_do_orcamento(v):
    base = CURTAS[v]["base"]
    return (f"com {L} épocas em vez de {C}, o exact match fica {posicao(longo_contra_curto(v).loc['exact_match'])} ({com_intervalo(longo_contra_curto(v).loc['exact_match'])}); "
            f"contra o modelo sem poda, {com_intervalo(contra(PRINCIPAL, v).loc['exact_match'])} com {C} épocas e {com_intervalo(contra(PRINCIPAL, base).loc['exact_match'])} com {L}")


display(Markdown("**Leitura (teste).**\n\n" + "".join(f"- **{NOME[CURTAS[v]['base']]}:** {leitura_do_orcamento(v)}.\n" for v in CURTAS)))
''')

md(r'''
## 8. O XLM-R ao lado da rede do zero

A mesma poda por magnitude, de uma vez, no aluno A dos dois caminhos: o da rede do zero (Etapa 7) e o do XLM-R (aqui). Cada um é comparado com o **seu próprio** modelo sem poda, no teste. As duas curvas ficam lado a lado; não há teste entre elas, porque mudam juntos a rede, o tamanho e o orçamento do reajuste.
''')

code(r'''
do_zero = pd.read_csv(ENTREGAS / "07_poda" / "comparacoes.csv")
do_zero = do_zero[(do_zero["modelo"] == "aluno_a") & (do_zero["particao"] == "test") & (do_zero["metrica"] == "exact_match")]
linha_do_zero = lambda s, comparacao: do_zero[(do_zero["variante"] == f"magnitude_{s}") & (do_zero["comparacao"] == comparacao)].iloc[0]
print("Comparações da Etapa 7 lidas:", sorted(do_zero["comparacao"].unique()))

lado_a_lado = {}
for momento, comparacao in (("depois", "depois − sem poda"), ("antes", "antes − sem poda")):
    lado_a_lado[momento] = {
        "XLM-R": {s: trio(contra(PRINCIPAL, DE_UMA_VEZ[s], momento).loc["exact_match"], "diferenca") for s in NIVEIS},
        "Rede do zero": {s: trio(linha_do_zero(s, comparacao), "diferenca") for s in NIVEIS},
    }
    fig, ax = curva(lado_a_lado[momento], f"Aluno A: exact match no teste contra o próprio modelo sem poda, {momento} do reajuste",
                    "pesos podáveis em zero (%); os níveis estão a distâncias iguais", "diferença para o modelo sem poda (pontos percentuais)", niveis=True)
    visual.salvar(fig, PASTA, f"xlmr_e_rede_do_zero_{momento}")
    plt.show()

em_texto = lambda t: f"{dif(t[0] / 100)} ({dif(t[1] / 100)} a {dif(t[2] / 100)})"
display(pd.DataFrame({
    f"{s}%": {f"{rede}, {momento} do reajuste": em_texto(lado_a_lado[momento][rede][s]) for momento in ("antes", "depois") for rede in ("XLM-R", "Rede do zero")}
    for s in NIVEIS
}).T)
cai = lambda rede, momento: [f"{s}%" for s in NIVEIS if lado_a_lado[momento][rede][s][2] < 0]
display(Markdown(f"""
**Leitura (aluno A, teste, exact match, diferença para o próprio modelo sem poda).**

- **Sem reajuste**, ficam distinguivelmente abaixo: o XLM-R em {lista(cai("XLM-R", "antes"), "nenhum nível")}; a rede do zero em {lista(cai("Rede do zero", "antes"), "nenhum nível")}.
- **Com reajuste**: o XLM-R em {lista(cai("XLM-R", "depois"), "nenhum nível")}; a rede do zero em {lista(cai("Rede do zero", "depois"), "nenhum nível")}.
- **Em 90%**: o XLM-R fica a {em_texto(lado_a_lado["depois"]["XLM-R"][90])} do seu modelo sem poda, e a rede do zero, a {em_texto(lado_a_lado["depois"]["Rede do zero"][90])}.
- **O que não é igual nos dois lados:** na rede do zero a poda alcança 98,7% dos parâmetros (a tabela de features entra); no XLM-R, {pts(PODAVEIS / total)}% (só o corpo). O reajuste teve 30 épocas lá (37,5% do treino) e {EPOCAS} aqui (o treino inteiro).
"""))
''')

md(r'''
## 9. O que muda de verdade: parâmetros, bytes e milissegundos

Tamanho e latência são medidos nos arquivos gravados e nos modelos carregados (semente 0), não estimados. O tamanho vem em três formatos: o **arquivo comum** (grava todos os pesos, zeros inclusive), o arquivo comum **com zip** e o **arquivo esparso** (um bit por posição e só os pesos diferentes de zero). A **rede** é contada separada do **tokenizador e das configurações**, sem os quais o sistema não recebe texto. A latência é o p50 em CPU, com 1 thread, uma consulta por vez, do texto à resposta.
''')

code(r'''
tamanhos = etapa10b.medir_tamanhos()      # o que já foi medido pelo comando `tamanhos` só é lido
latencias = {m: etapa10b.carregar_latencias(m) for m in MODELOS}


def latencia(m, v, execucao="comum"):
    """O p50 da variante, em ms, na execução comum ou na esparsa. None se não foi medida."""
    for grupo_medido in latencias[m].values():
        if grupo_medido["execucao"].startswith(execucao) and v in grupo_medido["resumo"]:
            return grupo_medido["resumo"][v]["latencia_p50_ms"]
    return None


def custo(m, v):
    """A linha de custo de uma variante (`v=None`: o modelo de partida)."""
    medido = tamanhos[m, v or etapa10b.DE_PARTIDA]
    rede, resto, sistema = (medido["bytes"][parte] for parte in ("rede", "tokenizador_e_configuracao", "sistema"))
    return {
        "modelo": m, "variante": v or "de_partida", "nome": "Sem poda" if v is None else NOME[v],
        "parametros": medido["parametros"], "nao_nulos": medido["nao_nulos"], "esparsidade_do_corpo": medido["esparsidade"],
        "linhas_da_tabela": medido["parametros_por_parte"]["A. Tabela de subtokens"] // 768,
        "rede_arquivo_comum": rede["arquivo_comum"], "rede_com_zip": rede["arquivo_comum_com_zip"], "rede_arquivo_esparso": rede["arquivo_esparso"],
        "tokenizador_e_configuracao": resto["arquivo_comum"],
        "sistema_arquivo_comum": sistema["arquivo_comum"], "sistema_com_zip": sistema["arquivo_comum_com_zip"], "sistema_arquivo_esparso": sistema["arquivo_esparso"],
        "latencia_p50_ms": latencia(m, v or etapa10b.DE_PARTIDA), "latencia_esparsa_p50_ms": latencia(m, v, "esparsa") if v else None,
    }


com_pesos = lambda m: [v for v in ORDEM if (m, v) in tamanhos]      # o controle não guarda pesos: o custo dele é o do modelo sem poda
eficiencia = pd.DataFrame([custo(m, v) for m in MODELOS for v in [None, *(v for v in ORDEM if (m, v) in tamanhos)]])
ms = lambda x: "—" if x is None or pd.isna(x) else num(x, 1)
vista = eficiencia[eficiencia["modelo"] == PRINCIPAL]
print(f"{MODELOS[PRINCIPAL]['nome']}, semente 0. Tamanhos em MB; latência p50 em ms, 1 thread de CPU")
display(pd.DataFrame([
    {
        "variante": l["nome"], "parâmetros": n(l["parametros"]), "diferentes de zero": n(l["nao_nulos"]),
        "rede: arquivo comum": mb(l["rede_arquivo_comum"]), "rede: com zip": mb(l["rede_com_zip"]), "rede: arquivo esparso": mb(l["rede_arquivo_esparso"]),
        "tokenizador e configurações": mb(l["tokenizador_e_configuracao"]), "sistema: arquivo comum": mb(l["sistema_arquivo_comum"]),
        "latência": ms(l["latencia_p50_ms"]), "latência, execução esparsa": ms(l["latencia_esparsa_p50_ms"]),
    }
    for _, l in vista.iterrows()
]).set_index("variante"))

fig, ax = visual.barras_em_ordem(
    {l["nome"]: l["sistema_arquivo_comum"] / 1e6 for _, l in vista.iterrows() if l["variante"] == "de_partida" or l["variante"] in PECAS},
    f"{CURTO[PRINCIPAL]} (XLM-R): tamanho do sistema no arquivo comum (rede, tokenizador e configurações)", "MB em disco", formato=lambda x: n(round(x)),
)
visual.salvar(fig, PASTA, "tamanho_em_disco")
plt.show()
''')

code(r'''
c = lambda v: vista[vista["variante"] == (v or "de_partida")].iloc[0]
base, topo = c(None), c(DE_UMA_VEZ[max(NIVEIS)])
menor = min((c(v) for v in PECAS), key=lambda l: l["sistema_arquivo_comum"])
medidas_de_latencia = vista.dropna(subset=["latencia_p50_ms"])
display(Markdown(f"""
**Leitura ({CURTO[PRINCIPAL]}, semente 0).**

- **Com os pesos soltos**, o número de parâmetros não muda e o arquivo comum também não: com {max(NIVEIS)}% do corpo em zero, a rede ocupa {mb(topo['rede_arquivo_comum'])} MB, contra {mb(base['rede_arquivo_comum'])} MB sem poda. Mudam os diferentes de zero ({n(topo['nao_nulos'])}, contra {n(base['nao_nulos'])}) e o tamanho num formato que aproveite os zeros: {mb(topo['rede_com_zip'])} MB com zip e {mb(topo['rede_arquivo_esparso'])} MB no arquivo esparso. O piso é a tabela de subtokens, que a poda por magnitude não toca: {mb(partes['A. Tabela de subtokens'] * 4)} MB.
- **Com as peças inteiras**, o arquivo comum diminui sem formato especial. O menor sistema é {minusculo(menor['nome'])}: {mb(menor['sistema_arquivo_comum'])} MB ({n(menor['parametros'])} parâmetros), contra {mb(base['sistema_arquivo_comum'])} MB ({num(base['sistema_arquivo_comum'] / menor['sistema_arquivo_comum'], 1)} vezes menor).
- **O tokenizador conta.** Sem poda, o tokenizador e as configurações ocupam {mb(base['tokenizador_e_configuracao'])} MB; com o vocabulário do treino, {mb(c('vocabulario_treino')['tokenizador_e_configuracao'])} MB.
- **Latência** (medida em {len(medidas_de_latencia)} das {len(vista)} linhas): {"; ".join(f"{minusculo(l['nome'])}: {ms(l['latencia_p50_ms'])} ms" for _, l in medidas_de_latencia.iterrows()) or "ainda não medida"}.
"""))
''')

md(r'''
## 10. O que segue para a quantização

A regra da seção 2, aplicada em `dev` com os rótulos humanos. É a única decisão da etapa. O aluno B2 não tem regra própria: recebe a principal e a secundária do aluno A, e a tabela dele só mostra como elas se saem nele.
''')

code(r'''
candidatos = lambda m, variantes: {
    v: {"em_dev": grupo(m, v, "dev"), "nao_nulos": exe[m][v][0]["meta"]["contagem"]["nao_nulos"], "bytes": exe[m][v][0]["meta"]["bytes_em_disco"]}
    for v in variantes
}
decisoes = {m: etapa10b.quem_segue(gabarito["dev"], sem_poda[m]["dev"], candidatos(m, PODADAS)) for m in COMPLETOS}
gravada = etapa10b.seguem_do_principal()      # o que a onda decidiu para o aluno principal, e que definiu as execuções do outro aluno
assert (gravada["principal"], gravada["secundaria"]) == (decisoes[PRINCIPAL]["principal"], decisoes[PRINCIPAL]["secundaria"]), "A regra refeita aqui não dá o que a onda gravou."
for m in DEPENDENTES:
    do_principal = decisoes[PRINCIPAL]
    tabela = etapa10b.quem_segue(gabarito["dev"], sem_poda[m]["dev"], candidatos(m, VARIANTES_DE[m]))["tabela"]
    decisoes[m] = {"principal": do_principal["principal"], "secundaria": do_principal["secundaria"], "seguem": list(do_principal["seguem"]), "tabela": tabela}

for m in MODELOS:
    tabela = decisoes[m]["tabela"]
    papel = {decisoes[m]["principal"]: "principal", decisoes[m]["secundaria"]: "secundária"}
    print(f"{MODELOS[m]['nome']}: exact match em dev (rótulos humanos), depois do reajuste, contra o modelo sem poda")
    display(pd.DataFrame([
        {
            "variante": NOME[v], "sistema no arquivo comum (MB)": mb(l["bytes"]), "diferentes de zero": n(l["nao_nulos"]),
            "podado": pts(l["podado"]), "podado − sem poda": com_intervalo(l), "passa": "sim" if l["passa"] else "não", "segue como": papel.get(v, ""),
            "no teste: podado − sem poda": com_intervalo(contra(m, v).loc["exact_match"]),
            "abaixo no teste": "sim" if contra(m, v).loc["exact_match", "ic_max"] < 0 else "não",
        }
        for v, l in tabela.iterrows()
    ]).set_index("variante"))


def discordam(m):
    """Onde dev e teste não dizem o mesmo: (passam em dev e ficam abaixo no teste, não passam em dev e não se distinguem no teste)."""
    tabela = decisoes[m]["tabela"]
    abaixo_no_teste = lambda v: contra(m, v).loc["exact_match", "ic_max"] < 0
    return ([v for v in tabela.index if tabela.loc[v, "passa"] and abaixo_no_teste(v)],
            [v for v in tabela.index if not tabela.loc[v, "passa"] and not abaixo_no_teste(v)])


def leitura_de_dev_e_teste(m):
    so_dev, so_teste = discordam(m)
    com_numeros = lambda v: f"{minusculo(NOME[v])} ({com_intervalo(decisoes[m]['tabela'].loc[v])} em dev, {com_intervalo(contra(m, v).loc['exact_match'])} no teste)"
    return (f"passam em dev e ficam distinguivelmente abaixo no teste: {lista([com_numeros(v) for v in so_dev], 'nenhuma')}; "
            f"não passam em dev e não se distinguem no teste: {lista([com_numeros(v) for v in so_teste], 'nenhuma')}")


def descricao(m, v):
    if v is None:
        return "nenhuma (segue o modelo sem poda)"
    l = decisoes[m]["tabela"].loc[v]
    return f"{minusculo(NOME[v])}: {com_intervalo(l)} em dev, {mb(l['bytes'])} MB no arquivo comum, {n(l['nao_nulos'])} parâmetros diferentes de zero"


display(Markdown("**Leitura.**\n\n" + "".join(
    f"- **{CURTO[m]}:** principal: {descricao(m, decisoes[m]['principal'])}; secundária: {descricao(m, decisoes[m]['secundaria'])}."
    + ("" if m in COMPLETOS else " (As variantes são as do aluno principal; "
       + lista([f"{minusculo(NOME[v])} {'passa' if decisoes[m]['tabela'].loc[v, 'passa'] else 'NÃO passa'} na regra aplicada a ele" for v in decisoes[m]["seguem"]], "nenhuma") + ".)")
    + "\n" for m in MODELOS)
    + "\n**Dev e teste dizem o mesmo?** A regra só olha `dev`. O teste não decide nada, mas mostra onde a decisão é frágil.\n\n"
    + "".join(f"- **{CURTO[m]}:** {leitura_de_dev_e_teste(m)}.\n" for m in MODELOS)))
''')

md(r'''
## 11. Entrega
''')

code(r'''
relativo_a_raiz = lambda caminho: str(Path(caminho).relative_to(RAIZ))


def copiar(origem, destino):
    """Copia a pasta sem ocupar o disco duas vezes quando dá (ligação entre arquivos); senão, copia de verdade."""
    try:
        shutil.copytree(origem, destino, copy_function=os.link)
    except OSError:
        shutil.rmtree(destino, ignore_errors=True)
        shutil.copytree(origem, destino)


# 1. Os modelos que seguem, com as respostas e os históricos
for antiga in ("modelos", "previsoes", "historicos"):
    if (PASTA / antiga).exists():
        shutil.rmtree(PASTA / antiga)
(PASTA / "historicos").mkdir()
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
reproduz, mantem_os_zeros = [], []
for m in MODELOS:
    for v in decisoes[m]["seguem"]:
        for i, semente in enumerate(SEMENTES):
            destino = PASTA / "modelos" / m / v / f"semente_{semente}"
            copiar(exe[m][v][i]["pasta"] / "modelo", destino)
            entregue = SistemaEncoder.carregar(destino, dispositivo)
            reproduz.append(np.mean([a == b for a, b in zip(entregue.prever(list(teste["utt"])), exe[m][v][i]["previsoes"]["test"])]))
            if v in SOLTAS:
                contagem = poda_encoder.contar(entregue.modelo)
                mantem_os_zeros.append(contagem["podaveis"] - contagem["nao_nulos_podaveis"] == zeros_de(exe[m][v][i]))
            del entregue
            for particao, ids in (("dev", dev["id"]), ("test", teste["id"])):
                salvar_previsoes(PASTA / "previsoes" / f"{m}__{v}_semente_{semente}_{particao}.parquet", ids, exe[m][v][i]["previsoes"][particao])
    for v in exe[m]:
        for semente, e in zip(SEMENTES, exe[m][v]):
            e["historico"].to_csv(PASTA / "historicos" / f"{m}__{v}_semente_{semente}.csv", index=False)

# 2. A tabela de resultados: o modelo sem poda e cada variante, antes e depois do reajuste, em dev e no teste
linhas_resultado = []
for m in MODELOS:
    for p in ("dev", "test"):
        linhas_resultado.append(resultados.registrar_grupo(f"{m}: sem poda", p, gabarito[p], sem_poda[m][p], modelo=m, variante="de_partida", momento="sem poda"))
        for v in exe[m]:
            medido = tamanhos.get((m, v))
            extras = {"parametros": medido["parametros"], "bytes_em_disco": medido["bytes"]["sistema"]["arquivo_comum"]} if medido else {}
            linhas_resultado.append(resultados.registrar_grupo(f"{m}: {NOME[v]}", p, gabarito[p], grupo(m, v, p), modelo=m, variante=v, momento="depois do reajuste", **extras))
            if f"{p}_antes" in exe[m][v][0]["previsoes"] and v != "sem_poda":
                linhas_resultado.append(resultados.registrar_grupo(f"{m}: {NOME[v]}, sem reajuste", p, gabarito[p], grupo(m, v, p, "antes"),
                                                                    modelo=m, variante=v, momento="antes do reajuste", **extras))
resultados.salvar(linhas_resultado, PASTA)

# 3. Todas as comparações pareadas da lista fechada
comparacoes = {}
for m in MODELOS:
    for p in ("dev", "test"):
        for v in exe[m]:
            comparacoes[m, v, "depois − sem poda", p] = contra(m, v, "depois", p)
            if f"{p}_antes" in exe[m][v][0]["previsoes"] and v != "sem_poda":
                comparacoes[m, v, "antes − sem poda", p] = contra(m, v, "antes", p)
                comparacoes[m, v, "depois − antes", p] = recupera(m, v, p)
        if m in COMPLETOS:
            for s in NIVEIS:
                comparacoes[m, GRADUAL[s], "gradual − de uma vez", p] = gradual_contra_de_uma_vez(m, s, p)
    for v in CURTAS:
        for p in ("dev", "test"):
            comparacoes[PRINCIPAL, CURTAS[v]["base"], f"{EPOCAS} épocas − {etapa10b.EPOCAS_CURTAS} épocas", p] = longo_contra_curto(v, p)
pd.concat(comparacoes, names=["modelo", "variante", "comparacao", "particao", "metrica"]).to_csv(PASTA / "comparacoes.csv")
eficiencia.to_csv(PASTA / "eficiencia.csv", index=False)
diagnostico.to_csv(PASTA / "corte_do_vocabulario.csv")
for guardado in ("piloto.json", "latencia.sh", "segue_do_principal.json"):
    shutil.copy(etapa10b.pasta_da_etapa() / guardado, PASTA / guardado)
(PASTA / "latencia").mkdir(exist_ok=True)
for medido in sorted((etapa10b.pasta_da_etapa() / "latencia").glob("*.json")):
    shutil.copy(medido, PASTA / "latencia" / medido.name)


# 4. A tabela principal: cada variante de cada modelo de partida, no teste
def linha_principal(m, v):
    valores = medida(m, v)
    tem_custo = (m, v or etapa10b.DE_PARTIDA) in tamanhos
    k = custo(m, v) if tem_custo else custo(m, None) if v == "sem_poda" else {}
    papel = {decisoes[m]["principal"]: "principal", decisoes[m]["secundaria"]: "secundaria"}.get(v, "") if v else ""
    return {
        "modelo": m, "variante": v or "de_partida", "nome": "Sem poda" if v is None else NOME[v], "papel": papel,
        **{metrica: valores.loc[metrica, "valor"] for metrica in TRES},
        "exact_match_antes": medida(m, v, "antes").loc["exact_match", "valor"] if v and v != "sem_poda" and tem_antes(v) else None,
        **({f"{c}_exact_match": contra(m, v).loc["exact_match", c] for c in ("diferenca", "ic_min", "ic_max")} if v else {}),
        **{chave: k.get(chave) for chave in ("parametros", "nao_nulos", "rede_arquivo_comum", "rede_com_zip", "rede_arquivo_esparso", "tokenizador_e_configuracao",
                                             "sistema_arquivo_comum", "sistema_com_zip", "sistema_arquivo_esparso", "latencia_p50_ms", "latencia_esparsa_p50_ms")},
    }


tabela_principal = pd.DataFrame([linha_principal(m, v) for m in MODELOS for v in [None, *(x for x in [*ORDEM, *CURTAS] if x in exe[m])]])
tabela_principal.to_csv(PASTA / "tabela_principal.csv", index=False)


# 5. A decisão, no formato da Etapa 7 (é o que a Etapa 10c lê)
def bytes_de(k):
    return {
        "arquivo_comum": int(k["sistema_arquivo_comum"]), "arquivo_comum_com_zip": int(k["sistema_com_zip"]), "arquivo_esparso": int(k["sistema_arquivo_esparso"]),
        "so_a_rede": {"arquivo_comum": int(k["rede_arquivo_comum"]), "arquivo_comum_com_zip": int(k["rede_com_zip"]), "arquivo_esparso": int(k["rede_arquivo_esparso"])},
        "tokenizador_e_configuracao": int(k["tokenizador_e_configuracao"]),
    }


def ficha(m, v, papel):
    k, l, meta = custo(m, v), decisoes[m]["tabela"].loc[v], exe[m][v][0]["meta"]
    pastas = {str(s): relativo_a_raiz(PASTA / "modelos" / m / v / f"semente_{s}") for s in SEMENTES}
    return {
        "variante": v, "papel": papel, "nome": NOME[v], "familia": VARIANTES[v]["familia"], "nivel": VARIANTES[v]["nivel"],
        "modelos": relativo_a_raiz(PASTA / "modelos" / m / v), "pastas_por_semente": pastas, "sementes": list(SEMENTES),
        "parametros": int(k["parametros"]), "nao_nulos": int(k["nao_nulos"]), "camadas": meta["camadas"], "linhas_da_tabela": int(k["linhas_da_tabela"]),
        "bytes": bytes_de(k), "latencia_p50_ms": k["latencia_p50_ms"],
        "exact_match_em_dev": float(l["podado"]), "passa_na_regra": bool(l["passa"]),
        "diferenca_para_sem_poda_em_dev": {c: float(l[c]) for c in ("diferenca", "ic_min", "ic_max")},
        "como_ler": ("SistemaEncoder.carregar(pasta da semente). " + (
            "Os pesos podados estão gravados como zeros em encoder/model.safetensors; a máscara é lida deles (poda_encoder.mascaras_dos_zeros). "
            "O arquivo esparso é gravado por poda_encoder.salvar_esparso." if v in SOLTAS else
            "É um XLM-R comum da biblioteca, só que menor" + (", com o tokenizador reduzido na mesma pasta." if meta["vocabulario"] else "."))),
    }


def do_modelo(m):
    k = custo(m, None)
    return {
        "nome": MODELOS[m]["nome"], "variante_da_etapa_10a": MODELOS[m]["variante_da_10a"], "grade": MODELOS[m]["grade"],
        "sinal_do_reajuste": MODELOS[m]["sinal"], "config": MODELOS[m]["config"], "receita_do_reajuste": etapa10b.receita_do_reajuste(m).como_dict(),
        "sem_poda": {
            "pastas_por_semente": {str(s): relativo_a_raiz(MODELOS[m]["pastas"][s]) for s in SEMENTES}, "sementes": list(SEMENTES),
            "parametros": int(k["parametros"]), "nao_nulos": int(k["nao_nulos"]), "bytes": bytes_de(k), "latencia_p50_ms": k["latencia_p50_ms"],
            "exact_match_em_dev": float(medida(m, None, p="dev").loc["exact_match", "valor"]),
        },
        "principal": decisoes[m]["principal"], "secundaria": decisoes[m]["secundaria"],
        "seguem": [ficha(m, v, "principal" if v == decisoes[m]["principal"] else "secundaria") for v in decisoes[m]["seguem"]],
    }


v_principal = decisoes[PRINCIPAL]["principal"]
decisao = {
    "etapa": etapa10b.ETAPA, "gerado_em": date.today().isoformat(), "regra": etapa10b.REGRA,
    "como_ler": ("Em cada modelo de partida: `principal` e `secundaria` trazem o nome da variante (null na principal quer dizer: o modelo sem poda; null na "
                 "secundária: nenhuma de pesos soltos passou). `seguem` traz a ficha de cada variante podada que segue, e `sem_poda`, a do modelo de partida. "
                 "Toda pasta em `pastas_por_semente` é lida por especialista.modelos.encoder.SistemaEncoder.carregar. Em `bytes`, os três formatos são do sistema "
                 "inteiro (rede, tokenizador e configurações); `so_a_rede` e `tokenizador_e_configuracao` separam as partes. Os modelos com `grade: do_principal` "
                 "recebem as variantes do aluno principal por desenho; `passa_na_regra` diz como elas se saem neles."),
    "reajuste": {"epocas": EPOCAS, "epocas_de_rampa_da_poda_gradual": RAMPA, "passos_entre_cortes": CORTES,
                 "receita": "a de quem treinou o modelo, com o sinal de quem treinou o modelo"},
    "principal": {"modelo": PRINCIPAL, "variante": v_principal,
                  "pastas_por_semente": ({str(s): relativo_a_raiz(PASTA / "modelos" / PRINCIPAL / v_principal / f"semente_{s}") for s in SEMENTES} if v_principal
                                         else {str(s): relativo_a_raiz(MODELOS[PRINCIPAL]["pastas"][s]) for s in SEMENTES})},
    "modelos": {m: do_modelo(m) for m in MODELOS},
}
(PASTA / "decisao.json").write_text(json.dumps(decisao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Seguem:", {m: decisoes[m]["seguem"] for m in MODELOS})
''')

md(r'''
### Critério de pronto
''')

code(r'''
completas = all(len(exe[m][v]) == len(SEMENTES) for m in MODELOS for v in exe[m])
tem_antes_e_depois = all(f"{p}_antes" in e["previsoes"] and p in e["previsoes"] for m, v, e in da_grade if v in COM_ANTES for p in ("dev", "test"))
custo_medido = all((m, v) in tamanhos for m in MODELOS for v in decisoes[m]["seguem"]) and all((m, etapa10b.DE_PARTIDA) in tamanhos for m in MODELOS) \
    and all(pd.notna(custo(PRINCIPAL, v)["latencia_p50_ms"]) for v in [None, *decisoes[PRINCIPAL]["seguem"]])
criterios = {
    f"Cada variante tem 3 sementes, com o mesmo orçamento de reajuste ({len(todas)} execuções)": completas and orcamento_confere,
    "A esparsidade pedida é a obtida": esparsidade_confere and gradual_confere and (all(mantem_os_zeros) if mantem_os_zeros else True),
    "Nenhum rótulo humano entrou no reajuste dos alunos": sem_humano,
    "A escolha das linhas do vocabulário não usou dev nem o teste": treino_coberto and all(len(x) == 1 for x in mesmas_linhas.values()),
    f"Cada variante tem antes e depois do reajuste, com comparação pareada contra o modelo sem poda e intervalo de 95%, no teste ({len(comparacoes)} comparações)": tem_antes_e_depois,
    "Tamanho e latência foram medidos em arquivos gravados e modelos carregados, com a rede separada do tokenizador": custo_medido,
    "O que segue para a quantização saiu de dev, pela regra escrita antes": (gravada["principal"], gravada["secundaria"]) == (decisoes[PRINCIPAL]["principal"], decisoes[PRINCIPAL]["secundaria"]),
    "Os modelos entregues são pastas comuns de SistemaEncoder e reproduzem as respostas do teste": bool(reproduz) and min(reproduz) > 0.999,
}
for criterio, atendido in criterios.items():
    assert atendido, criterio
pd.DataFrame({"atendido": ["sim" if ok else "NÃO" for ok in criterios.values()]}, index=list(criterios))
''')

md(r'''
## 12. Resposta
''')

code(r'''
P = PRINCIPAL
leitura = leitura_dos_pesos_soltos(P)
linha_md = lambda celulas: "| " + " | ".join(str(c) for c in celulas) + " |"


def tabela_md(tabela, primeira):
    """Uma tabela do pandas em Markdown, com o índice na primeira coluna."""
    return "\n".join([linha_md([primeira, *tabela.columns]), linha_md(["---"] * (len(tabela.columns) + 1)),
                      *(linha_md([indice, *linha]) for indice, linha in zip(tabela.index, tabela.to_numpy()))])


def linha_do_custo(l):
    return [l["nome"], n(l["parametros"]), n(l["nao_nulos"]), mb(l["rede_arquivo_comum"]), mb(l["rede_com_zip"]), mb(l["rede_arquivo_esparso"]),
            mb(l["tokenizador_e_configuracao"]), mb(l["sistema_arquivo_comum"]), ms(l["latencia_p50_ms"]), ms(l["latencia_esparsa_p50_ms"])]


def linha_de_tudo(l):
    celula = lambda x, formato: "—" if x is None or pd.isna(x) else formato(x)
    diferenca = "—" if pd.isna(l.get("diferenca_exact_match")) else com_intervalo({"diferenca": l["diferenca_exact_match"], "ic_min": l["ic_min_exact_match"], "ic_max": l["ic_max_exact_match"]})
    nome = l["nome"] + (f" **({'principal' if l['papel'] == 'principal' else 'secundária'})**" if l["papel"] else "")
    return [CURTO[l["modelo"]], nome, pts(l["intent_acc"]), pts(l["slot_f1"]), celula(l["exact_match_antes"], pts), pts(l["exact_match"]), diferenca,
            celula(l["nao_nulos"], n), celula(l["sistema_arquivo_comum"], mb), celula(l["sistema_arquivo_esparso"], mb), celula(l["latencia_p50_ms"], lambda x: num(x, 1))]


def linha_da_decisao(m):
    return [CURTO[m], descricao(m, decisoes[m]["principal"]), descricao(m, decisoes[m]["secundaria"])]


segue = decisoes[P]
resumo = f"""# Etapa 10b · Poda e fine-tuning de recuperação do XLM-R: resumo da entrega

**Pergunta.** Quanto dá para tirar do XLM-R, de que parte, e quanto o reajuste recupera?

**Resposta.** No {minusculo(CURTO[P])}, com os pesos soltos do corpo: {leitura}. Com peças inteiras: {leitura_das_pecas(P)}. **No custo**, com {max(NIVEIS)}% dos pesos do corpo em zero a rede continua com {mb(topo['rede_arquivo_comum'])} MB no arquivo comum e cai para {mb(topo['rede_arquivo_esparso'])} MB no arquivo esparso; o menor sistema de peças inteiras ({minusculo(menor['nome'])}) tem {mb(menor['sistema_arquivo_comum'])} MB, contra {mb(base['sistema_arquivo_comum'])} MB sem poda. A latência só muda com as camadas: {ms(c('camadas_6')['latencia_p50_ms'])} ms com 6 das 12, contra {ms(base['latencia_p50_ms'])} ms sem poda; o vocabulário do treino fica em {ms(c('vocabulario_treino')['latencia_p50_ms'])} ms.

Segue para a quantização (Etapa 10c), do {minusculo(CURTO[P])}: a principal é **{descricao(P, segue['principal'])}**; a secundária é **{descricao(P, segue['secundaria'])}**.

## O que foi feito

- **Modelos de partida:** {lista([MODELOS[m]['nome'] for m in MODELOS])}. A grade completa roda em {lista([minusculo(CURTO[m]) for m in COMPLETOS])}; {lista([minusculo(CURTO[m]) for m in DEPENDENTES])} recebe só a principal e a secundária do {minusculo(CURTO[P])}. Cada um é reajustado com o sinal com que aprendeu; a época de parada dos alunos usa o `dev` rotulado pelo professor.
- **Pesos soltos, por magnitude** (Han et al., arXiv 1506.02626): nas 72 matrizes do corpo ({n(PODAVEIS)} pesos, {pts(PODAVEIS / total)}% do modelo), com o corte por matriz, em {lista([f"{s}%" for s in NIVEIS])}; de uma vez, com medida antes e depois do reajuste, e gradual (Zhu & Gupta, arXiv 1710.01878), com a esparsidade subindo nas {RAMPA} primeiras épocas. A tabela de subtokens fica fora (Sanh et al., arXiv 2005.07683); Gordon et al. (arXiv 2002.08307) a podam: desvio declarado.
- **Peças inteiras:** o corte do vocabulário (TextPruner, arXiv 2203.15996; Abdaoui et al., arXiv 2010.05609), em duas formas que não olham `dev` nem o teste, com o tokenizador reduzido junto; as camadas de cima (Sajjad et al., arXiv 2004.03844), ficando com {lista(etapa10b.CAMADAS)} das 12; e duas combinações.
- **Reajuste:** {EPOCAS} épocas em todas as execuções (o orçamento do ajuste fino da Etapa 10a), com a receita de quem treinou o modelo e a máscara travada. Fica o melhor ponto em `dev`, e o ponto de partida concorre: em {len(ficou_na_partida)} das {len(com_partida)} execuções o modelo saiu como entrou. Um controle faz o mesmo reajuste sem tirar nada, e duas variantes do {minusculo(CURTO[P])} foram reajustadas também com {etapa10b.EPOCAS_CURTAS} épocas (a proporção da Etapa 7).
- **O orçamento veio de um piloto no teto** (semente 0, `dev`): sem as 6 camadas de cima, {pts(curto_do_piloto["dev"]["exact_match"])}% de exact match com {etapa10b.EPOCAS_CURTAS} épocas de reajuste e {pts(longo_do_piloto["dev"]["exact_match"])}% com {EPOCAS}.
- {len(todas)} execuções, {num(horas, 1)} horas de placa de vídeo.

## Pesos soltos: antes e depois do reajuste ({CURTO[P]}, teste)

Exact match em % (intervalo de 95%), média de 3 sementes; diferenças pareadas em pontos percentuais.

{tabela_md(quadro(P, ["sem_poda", *DE_UMA_VEZ.values()]), "")}

![Curva](figuras/curva_{P}.png)

## De uma vez ou gradual ({CURTO[P]}, teste)

{tabela_md(quadro_das_formas(P), "Esparsidade")}

- {leitura_das_formas(P).capitalize()}.

## Peças inteiras ({CURTO[P]}, teste)

{tabela_md(quadro_das_pecas(P), "Variante")}

![Peças inteiras](figuras/pecas_inteiras.png)

Diagnóstico do corte do vocabulário (não escolheu nada):

{tabela_md(diagnostico.drop(columns=["o que fica"]), "Forma")}

## O orçamento do reajuste ({CURTO[P]}, teste)

{chr(10).join(f"- **{NOME[CURTAS[v]['base']]}:** {leitura_do_orcamento(v)}." for v in CURTAS)}

## O XLM-R ao lado da rede do zero (aluno A, teste, magnitude de uma vez)

Diferença de exact match para o próprio modelo sem poda, em pontos percentuais (intervalo de 95%).

{tabela_md(pd.DataFrame({f"{s}%": {f"{rede}, {momento} do reajuste": em_texto(lado_a_lado[momento][rede][s]) for momento in ("antes", "depois") for rede in ("XLM-R", "Rede do zero")} for s in NIVEIS}).T, "Esparsidade")}

![Lado a lado](figuras/xlmr_e_rede_do_zero_depois.png)

## O que muda de verdade ({CURTO[P]}, semente 0)

Tamanhos em MB; latência p50 em ms, 1 thread de CPU, uma consulta por vez.

{linha_md(["Variante", "Parâmetros", "Diferentes de zero", "Rede: arquivo comum", "Rede: com zip", "Rede: arquivo esparso", "Tokenizador e configurações", "Sistema: arquivo comum", "Latência", "Latência, execução esparsa"])}
{linha_md(["---"] * 10)}
{chr(10).join(linha_md(linha_do_custo(l)) for _, l in vista.iterrows())}

![Tamanho em disco](figuras/tamanho_em_disco.png)

## Todas as variantes, no teste

Métricas em %, média de 3 sementes, depois do reajuste (a coluna "antes" é o exact match logo depois do corte). A diferença é pareada, contra o mesmo modelo sem poda. Tamanhos do sistema em MB e latência p50 em ms, medidos na semente 0.

{linha_md(["Modelo de partida", "Variante", "Intenção", "F1 de entidades", "Exact match antes", "Exact match", "Exact match − sem poda", "Diferentes de zero", "Arquivo comum", "Esparso", "Latência"])}
{linha_md(["---"] * 11)}
{chr(10).join(linha_md(linha_de_tudo(l)) for _, l in tabela_principal.iterrows())}

## O que segue para a quantização

Regra, decidida em `dev` com os rótulos humanos: passa a variante cujo exact match, depois do reajuste, não é distinguivelmente menor que o do modelo sem poda. A **principal** é a que passa com o menor arquivo comum; se nenhuma peça inteira passa, é o modelo sem poda. A **secundária** é a de pesos soltos que passa com a maior esparsidade. {lista([CURTO[m] for m in DEPENDENTES])} recebe as variantes do {minusculo(CURTO[P])}, por desenho.

{linha_md(["Modelo de partida", "Principal", "Secundária"])}
{linha_md(["---"] * 3)}
{chr(10).join(linha_md(linha_da_decisao(m)) for m in MODELOS)}

Dev e teste dizem o mesmo? A regra só olha `dev`; o teste não decide nada.

{chr(10).join(f"- **{CURTO[m]}:** {leitura_de_dev_e_teste(m)}." for m in MODELOS)}

## Critério de pronto

| Critério | Atendido |
|---|---|
{chr(10).join(f"| {criterio} | {'sim' if ok else 'NÃO'} |" for criterio, ok in criterios.items())}

## Limites

- **A tabela de subtokens ficou fora da poda por magnitude.** A esparsidade é do corpo ({pts(PODAVEIS / total)}% do modelo). Gordon et al. podam a tabela também.
- **O orçamento do reajuste ({EPOCAS} épocas) foi escolhido depois de um piloto no teto**: o primeiro desenho tinha {etapa10b.EPOCAS_CURTAS}. É um orçamento maior, em proporção, que o da Etapa 7 (o treino inteiro, contra 37,5%). O controle com {etapa10b.EPOCAS_CURTAS} épocas só existe em duas variantes do {minusculo(CURTO[P])}.
- **O corte é por matriz e a poda é de magnitude.** O corte global e a poda de movimento (Sanh et al.), que os artigos apontam como melhor em esparsidade alta num encoder pré-treinado, não foram testados.
- **Só duas formas de corte do vocabulário e dois cortes de camadas.** A forma "treino" depende do texto de treino: palavra nova é partida em peças menores ou vira `<unk>`. A forma "alfabeto" é uma escolha deste projeto, sem artigo que a teste.
- **{lista([CURTO[m] for m in DEPENDENTES])} não tem grade própria**: recebe as variantes escolhidas para o {minusculo(CURTO[P])}.
- **O teto serviu ao piloto.** Seis execuções dele (semente 0) foram vistas antes de a grade rodar. Fora o orçamento do reajuste, as regras não mudaram depois.
- **A regra aceita o que não se distingue.** Com o tamanho de `dev`, uma queda de menos de cerca de 1 ponto de exact match pode passar.
- **As decisões de análise usam os rótulos humanos de `dev`**, que o cenário dos alunos não tem. O reajuste e a parada não usam.
- **As respostas saem da placa de vídeo, em 32 bits**, como as da Etapa 10a; o reajuste usa precisão mista.
- **O teste foi aberto para todas as variantes**, porque a curva inteira é o resultado da etapa. Nenhuma decisão saiu dele.
- **A curva da rede do zero ao lado da do XLM-R não é um teste**: mudam juntos a rede, o que a poda alcança e o orçamento do reajuste.
- **Latência só em PyTorch, na CPU, com 1 thread.** No {minusculo(CURTO[P])}, todas as variantes; nos outros modelos, só as que seguem. A execução esparsa (CSR do PyTorch) só foi medida no {minusculo(CURTO[P])}. O modelo sem poda, medido em cada grupo, variou entre {num(min(g["resumo"][etapa10b.DE_PARTIDA]["latencia_p50_ms"] for g in latencias[P].values()), 1)} e {num(max(g["resumo"][etapa10b.DE_PARTIDA]["latencia_p50_ms"] for g in latencias[P].values()), 1)} ms de um grupo para outro.
- **Os pesos das variantes que não seguem foram apagados do cache** (ficam as respostas, as métricas e os históricos de todas). São refeitos pelo comando `executar`.
- **O arquivo esparso é o formato mais simples** (um bit por posição) e não é entregue: `poda_encoder.salvar_esparso` o grava a partir da pasta comum.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `decisao.json` | Para cada modelo de partida: a principal, a secundária, a pasta de cada semente, o sinal do reajuste e os tamanhos medidos |
| `modelos/` | Os modelos que seguem, 3 sementes cada, em pastas comuns de `SistemaEncoder` (pesos densos, com zeros no lugar dos pesos podados, ou a rede menor com o tokenizador reduzido) |
| `tabela_principal.csv` | Cada variante de cada modelo de partida no teste: qualidade, diferença para o modelo sem poda, tamanhos e latência |
| `previsoes/` | As respostas dos modelos que seguem, em `dev` e no teste |
| `resultados.csv` | Cada variante de cada modelo de partida, antes e depois do reajuste, em `dev` e no teste |
| `comparacoes.csv` | Cada comparação pareada, em todas as métricas |
| `eficiencia.csv` | Parâmetros, tamanhos em disco (rede, tokenizador, sistema) e latência de cada variante (semente 0) |
| `corte_do_vocabulario.csv` | O diagnóstico das duas formas de corte: linhas que ficam e fração dos subtokens cortada em treino, `dev` e teste |
| `historicos/` | Perdas, esparsidade e métricas no `dev` de parada, época por época, de cada execução |
| `piloto.json` | O que foi visto no teto antes de a grade rodar, e que mudou o orçamento do reajuste |
| `segue_do_principal.json` | A regra aplicada ao aluno principal durante a onda: é o que definiu as execuções do outro aluno |
| `latencia/`, `latencia.sh` | O resumo de cada grupo da medida de latência e o roteiro que os mediu |
| `figuras/` | Os gráficos deste notebook |
| `10b_encoder_poda.html` | Este notebook executado, para leitura |
"""
(PASTA / "RESUMO.md").write_text(resumo, encoding="utf-8")
display(Markdown(resumo))
''')

code(r'''
arquivos = sorted(p for p in PASTA.rglob("*") if p.is_file() and "historicos" not in p.parts)
pd.DataFrame(
    {"tamanho (KB)": [round(p.stat().st_size / 1024, 1) for p in arquivos]},
    index=[str(p.relative_to(RAIZ)) for p in arquivos],
)
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
for celula in cells:   # um erro de digitação numa célula aparece aqui, e não só quando o notebook rodar
    if celula.cell_type == "code":
        compile(celula.source, "<célula do notebook>", "exec")
nbf.write(nb, sys.argv[1])
print("ok", len(cells), "células")
