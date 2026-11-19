"""Monta notebooks/11a_professor_aberto_maior.ipynb a partir das células abaixo."""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Etapa 11a · O professor aberto sem o limite dos 10 GB

| | |
|---|---|
| **Pergunta** | O cenário B perde por ser aberto, ou porque o professor teve de caber em 10 GB? |
| **Entrada** | `entregas/01_dados/`, o professor aberto da Etapa 5b (`entregas/05b_professor_aberto/` e as respostas dele), o professor fechado da Etapa 5a como referência, e as respostas das versões novas, gravadas no Colab pelos comandos de `especialista/experimentos/etapa11a.py` e `etapa11a_colab.py`. |
| **Entrega** | `entregas/11a_professor_aberto_maior/`: cada versão de professor na sua pasta (respostas, rótulos e logits no formato da Etapa 5b, e um `versao.json`), a escolha pela regra, as tabelas e o custo. |
| **Pronto quando** | as regras estão gravadas antes das rodadas; a troca de máquina, sozinha, foi medida; cada candidato rotulou dev, teste e treino inteiros; a escada foi aplicada em `dev`; os arquivos de cada versão são lidos pelo código da Etapa 6; nada da Etapa 5b mudou. |

Este notebook não roda modelo nenhum. As rodadas foram feitas numa placa alugada (Google Colab), por um condutor que leva o código, traz as respostas e desliga a máquina. Aqui só se lê o que ficou gravado.

**Artigos de apoio** (os dois já usados e conferidos nas Etapas 5a e 5b):

- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, [arXiv 2305.14314](https://arxiv.org/abs/2305.14314): a quantização em 4 bits (NF4) que a Etapa 5b usou para o modelo caber na placa.
- Liu et al., *What Makes Good In-Context Examples for GPT-3?*, [arXiv 2101.06804](https://arxiv.org/abs/2101.06804): a escolha das demonstrações mais parecidas com o pedido, igual em todas as versões.

**Seções**

1. As versões de professor
2. As regras, escritas antes
3. Trocar de máquina, sozinho, muda quanto?
4. A escada, em `dev`
5. A qualidade dos professores
6. O que muda nos rótulos que o aluno recebe
7. Tempo e custo
8. Entrega
9. Resposta
''')

code(r'''
import json
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from especialista import visual
from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.dados import massive
from especialista.experimentos import etapa6, etapa11a

visual.aplicar_estilo()
pd.set_option("display.max_colwidth", 200)

FECHO = etapa11a.PASTA / "fecho"            # gravado por `python -m especialista.experimentos.etapa11a fechar`
ESCADA, CANDIDATOS, CONFERENCIA, FECHADO = etapa11a.ESCADA, etapa11a.CANDIDATOS, etapa11a.CONFERENCIA, etapa11a.FECHADO
NOMES, CURTOS = etapa11a.NOMES, etapa11a.CURTOS

particoes, rotulos = massive.carregar_entrega()
treino, dev, teste = (particoes[p] for p in ("train", "dev", "test"))
intents = rotulos["intents"]

ler = lambda nome: json.loads((FECHO / nome).read_text(encoding="utf-8"))
escolha, custos = ler("escolha.json"), ler("custo.json")
qual, comp = pd.read_csv(FECHO / "qualidade.csv"), pd.read_csv(FECHO / "comparacoes.csv")
mud, diag = pd.read_csv(FECHO / "mudancas_entre_professores.csv"), pd.read_csv(FECHO / "diagnosticos.csv")
regra = json.loads((etapa11a.PASTA / "regra.json").read_text(encoding="utf-8"))
ambiente = json.loads((etapa11a.PASTA / CONFERENCIA / "conferencia_de_ambiente.json").read_text(encoding="utf-8"))
versoes = {v: json.loads((etapa11a.PASTA / v / "versao.json").read_text(encoding="utf-8")) for v in (*ESCADA, CONFERENCIA)}
escolhido = escolha["escolhido"]


def n(x):
    return f"{int(x):,}".replace(",", ".")


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def pts(x, casas=1):
    return num(100 * x, casas)


def dif(x, casas=1):
    return f"{100 * x:+.{casas}f}".replace(".", ",").replace("-", "−")


def com_intervalo(linha):
    return f"{dif(linha['diferenca'])} ({dif(linha['ic_min'])} a {dif(linha['ic_max'])})"


def valor_com_intervalo(linha):
    return f"{pts(linha['valor'])} ({pts(linha['ic_min'])} a {pts(linha['ic_max'])})"


def medida(sistema, particao, metrica="exact_match"):
    return qual[(qual["sistema"] == sistema) & (qual["particao"] == particao) & (qual["metrica"] == metrica)].iloc[0]


def comparacao(b, a, particao, metrica="exact_match"):
    return comp[(comp["b"] == b) & (comp["a"] == a) & (comp["particao"] == particao) & (comp["metrica"] == metrica)].iloc[0]


def horas(segundos):
    return f"{int(segundos // 3600)} h {int(segundos % 3600 // 60):02d} min" if segundos >= 3600 else f"{segundos / 60:.0f} min"
''')

md(r'''
## 1. As versões de professor

A Etapa 5b escolheu o Qwen3-8B **em 4 bits** porque era o que cabia nos 10 GB da placa local. Aqui a mesma escada continua numa placa alugada, com o mesmo prompt, o mesmo código de pontuação e geração e o mesmo lote de 6 pedidos. Mudam duas coisas, uma de cada vez:

1. **sai a compressão**: os mesmos pesos do Qwen3-8B, sem os 4 bits;
2. **cresce o modelo**: o Qwen3-14B, também sem os 4 bits, que não caberia na placa local nem comprimido com folga para o lote.

Cada professor é uma **versão**, com nome `<modelo>__<precisão>__<máquina>`, pasta própria e um `versao.json`: commit dos pesos no Hugging Face, precisão, impressão digital do prompt e das demonstrações, parâmetros de geração, o pacote de código enviado, as bibliotecas que de fato ficaram instaladas, a placa, os tempos e o custo. Nenhuma versão substitui outra: todas ficam guardadas, inclusive a que perde.
''')

code(r'''
def linha_da_versao(v):
    r = versoes[v]
    if v == etapa11a.VERSAO_DA_5B:
        e = r["execucao"]
        placa, do_modelo, pico = r["ambiente"]["gpu"], e["vram_do_modelo_gb"], e["vram_de_pico_gb"]
        segundos, pedidos, unidades = sum(e["segundos_de_gpu"].values()), sum(e["pedidos"].values()), 0.0
    else:
        placa, do_modelo, pico = r["gpu"]["placa"], r["gpu"]["vram_do_modelo_gb"], r["gpu"]["vram_de_pico_gb"]
        segundos, pedidos = sum(r["tempos"]["segundos_de_gpu"].values()), sum(r["execucao"]["pedidos"].values())
        unidades = r["custo"]["unidades_fechadas_na_conta"]
    return {
        "versão": v, "o que é": NOMES[v], "commit dos pesos": r["modelo"]["commit"][:10], "placa": placa,
        "modelo na placa (GB)": num(do_modelo, 1), "pico (GB)": num(pico, 1), "pedidos": n(pedidos),
        "s por pedido": num(segundos / pedidos, 2), "unidades do Colab": num(unidades, 2),
    }

tabela_das_versoes = pd.DataFrame([linha_da_versao(v) for v in (*ESCADA, CONFERENCIA)]).set_index("versão")
tabela_das_versoes
''')

code(r'''
mesmos_prompts = {v: versoes[v]["prompt"]["sha256_das_instrucoes"] == versoes[ESCADA[0]]["prompt"]["sha256_das_instrucoes"]
                  and versoes[v]["prompt"]["sha256_das_demonstracoes"] == versoes[ESCADA[0]]["prompt"]["sha256_das_demonstracoes"] for v in versoes}
mesma_geracao = {v: versoes[v]["geracao"] == versoes[ESCADA[0]]["geracao"] for v in versoes}
bibliotecas = ("python", "torch", "transformers", "bitsandbytes", "accelerate", "tokenizers")
mesmas_bibliotecas = {v: all(versoes[v]["ambiente"][b] == versoes[ESCADA[0]]["ambiente"][b] for b in bibliotecas) for v in versoes}
tipos_dos_4_bits = versoes[CONFERENCIA]["precisao"]["tipos_dos_parametros_carregados"]
display(Markdown(f"""
- **Mesmo prompt em todas as versões:** {'sim' if all(mesmos_prompts.values()) else 'NÃO'} (mesma impressão digital das instruções e das {versoes[ESCADA[0]]['prompt']['demonstracoes']} demonstrações). **Mesmos parâmetros de geração:** {'sim' if all(mesma_geracao.values()) else 'NÃO'} (lote de {versoes[ESCADA[0]]['geracao']['lote']}, decodificação gulosa).
- **Mesmas bibliotecas no Colab e na máquina local:** {'sim' if all(mesmas_bibliotecas.values()) else 'NÃO'} ({', '.join(f"{b} {versoes[CANDIDATOS[0]]['ambiente'][b]}" for b in bibliotecas)}): lá se instalou o `uv.lock` do projeto. Diferem a placa e o driver ({versoes[CANDIDATOS[0]]['ambiente']['nvidia_smi'].split(', ')[-1]} lá, {versoes[ESCADA[0]]['ambiente']['driver']} aqui).
- **Um aviso sobre "só sai a compressão".** Na versão em 4 bits, as camadas comprimidas fazem a conta em float16, mas os tensores que não são comprimidos ficam em bfloat16 ({n(tipos_dos_4_bits['torch.bfloat16'])} tensores, contra {n(tipos_dos_4_bits['torch.uint8'])} comprimidos). Na versão sem os 4 bits tudo está em float16. Entre as duas muda a compressão **e** esse tipo numérico.
"""))
''')

md(r'''
## 2. As regras, escritas antes

Três regras foram gravadas em `regra.json` antes de qualquer rodada no Colab: a da **escada** (a mesma frase da Etapa 5b), a da **precisão** (float16, e bfloat16 só se aparecer NaN ou infinito) e o **limiar da conferência de ambiente**.
''')

code(r'''
primeira_rodada = versoes[CONFERENCIA]["conducoes"][0]["inicio"]
regra_veio_antes = regra["escrita_em"] < primeira_rodada
print(f"Regras gravadas em {regra['escrita_em']}; primeira rodada no Colab iniciada em {primeira_rodada}. Regra antes da rodada: {'sim' if regra_veio_antes else 'NÃO'}")
precisoes = {v: json.loads((etapa11a.PASTA / v / "precisao.json").read_text(encoding="utf-8")) for v in CANDIDATOS}
pd.DataFrame({
    "regra": {"escada": regra["escada"]["regra"], "precisão": regra["precisao"]["regra"], "ambiente": regra["ambiente"]["desenho"]},
    "o que aconteceu": {
        "escada": f"aplicada em dev na seção 4; escolhido: {NOMES[escolhido]}",
        "precisão": "; ".join(f"{CURTOS[v]}: {p['pontuacoes_nao_finitas']} pontuações não finitas em {p['pedidos']} pedidos" for v, p in precisoes.items()) + ": ficou float16",
        "ambiente": "aprovada (seção 3)" if ambiente["aprovado"] else "REPROVADA",
    },
})
''')

md(r'''
## 3. Trocar de máquina, sozinho, muda quanto?

Antes de comparar versões, é preciso saber quanto o resultado muda quando **nada** muda além da máquina. A configuração da Etapa 5b (4 bits) rodou no Colab em 50 lotes inteiros da rodada local de `dev`, com os mesmos pedidos na mesma ordem. As respostas foram comparadas, pedido a pedido, com as gravadas na máquina local.

O limiar está ancorado na conferência dos atalhos da Etapa 5b, que mediu o mesmo tipo de ruído (a mesma conta em ponto flutuante feita em outra ordem).
''')

code(r'''
m, L = ambiente["medido"], ambiente["limiar"]
tabela_do_ambiente = pd.DataFrame([
    {"medida": "mesmos tokens de entrada", "medido": pts(m["mesmos_tokens_de_entrada"]) + "%", "limiar": "100%", "atende": ambiente["criterios"]["mesmos_tokens_de_entrada"]},
    {"medida": "mesma intenção", "medido": pts(m["mesma_intencao"]) + "%", "limiar": "≥ " + pts(L["mesma_intencao_min"], 0) + "%", "atende": ambiente["criterios"]["mesma_intencao"]},
    {"medida": "mesma resposta, depois de lida", "medido": pts(m["mesma_resposta_lida"]) + "%", "limiar": "≥ " + pts(L["mesma_resposta_lida_min"], 2) + "%", "atende": ambiente["criterios"]["mesma_resposta_lida"]},
    {"medida": "diferença de probabilidade, percentil 95", "medido": num(m["p95_da_diferenca_de_probabilidade"], 3), "limiar": "≤ " + num(L["p95_da_diferenca_de_probabilidade_max"], 4), "atende": ambiente["criterios"]["p95_da_diferenca_de_probabilidade"]},
    {"medida": "maior diferença de probabilidade", "medido": num(m["maior_diferenca_de_probabilidade"], 3), "limiar": "só relatada", "atende": True},
]).set_index("medida")
tabela_do_ambiente["atende"] = tabela_do_ambiente["atende"].map({True: "sim", False: "NÃO"})
muda_com_a_maquina = 1 - m["mesma_resposta_lida"]
display(tabela_do_ambiente)
display(Markdown(f"""
**Leitura.** Em {n(m['pedidos'])} pedidos, a mesma configuração em outra máquina deu a mesma intenção em {pts(m['mesma_intencao'])}% e a mesma resposta em {pts(m['mesma_resposta_lida'])}%: trocar de máquina, sozinho, muda cerca de **{pts(muda_com_a_maquina, 0)}% das respostas**. É o piso de ruído desta etapa: diferença entre versões desse tamanho não diz nada sobre a versão.
"""))
''')

md(r'''
## 4. A escada, em `dev`

A regra: do mais barato para o mais caro, um candidato só toma o lugar do titular se o exact match dele em `dev` for distinguivelmente maior (intervalo de 95% da diferença pareada todo acima de zero). O titular de partida é a versão da Etapa 5b.
''')

code(r'''
disputas = escolha["disputas"]
tabela_da_escada = pd.DataFrame([
    {"desafiante": NOMES[d], "titular": CURTOS[x["contra"]], "exact match do titular": pts(x["a"]), "exact match do desafiante": pts(x["b"]),
     "diferença (intervalo de 95%)": com_intervalo(x), "toma o lugar": "sim" if x["ic_min"] > 0 else "não"}
    for d, x in disputas.items()
]).set_index("desafiante")
display(tabela_da_escada)

fig, ax = visual.diferencas_com_intervalo(
    {f"{CURTOS[d]} − {CURTOS[x['contra']]}": (100 * x["diferenca"], 100 * x["ic_min"], 100 * x["ic_max"]) for d, x in disputas.items()},
    "A escada em dev: desafiante menos titular", "diferença de exact match (pontos)",
)
visual.salvar(fig, ENTREGAS / etapa11a.ETAPA, "escada_em_dev")

sem_4_bits, maior = disputas[CANDIDATOS[0]], disputas[CANDIDATOS[1]]
display(Markdown(f"""
**Leitura.** Tirar os 4 bits não muda o exact match em `dev`: {com_intervalo(sem_4_bits)} pontos, com o zero dentro do intervalo. O titular continua o mesmo. O modelo maior acrescenta {com_intervalo(maior)} pontos sobre o titular, com o intervalo todo acima de zero: **o professor aberto passa a ser o {NOMES[escolhido]}**.
"""))
''')

md(r'''
## 5. A qualidade dos professores

A decisão saiu de `dev`. O teste só é relatado. O professor fechado da Etapa 5a entra como referência: é com ele que o cenário B se compara.
''')

code(r'''
def tabela_de_qualidade(particao):
    return pd.DataFrame({
        NOMES[s]: {qualidade.NOMES[metrica]: valor_com_intervalo(medida(s, particao, metrica)) for metrica in qualidade.METRICAS}
        for s in (*ESCADA, FECHADO)
    }).T

for particao, titulo in (("dev", "dev (decide)"), ("test", "teste (só relato)")):
    display(Markdown(f"**Qualidade em {titulo}**, com intervalo de 95%"))
    display(tabela_de_qualidade(particao))
''')

code(r'''
def tabela_de_comparacoes(particao):
    linhas = comp[comp["particao"] == particao]
    return pd.DataFrame({
        qualidade.NOMES[metrica]: {linha["comparacao"]: com_intervalo(linha) for _, linha in linhas[linhas["metrica"] == metrica].iterrows()}
        for metrica in qualidade.METRICAS
    })

display(Markdown("**Comparações pareadas no teste** (B − A, em pontos, com intervalo de 95%)"))
display(tabela_de_comparacoes("test"))

serie = lambda s: {qualidade.NOMES[metrica]: tuple(100 * medida(s, "test", metrica)[k] for k in ("valor", "ic_min", "ic_max")) for metrica in qualidade.METRICAS}
fig, ax = visual.pontos_com_intervalo({CURTOS[ESCADA[0]]: serie(ESCADA[0]), CURTOS[escolhido]: serie(escolhido)}, "O professor aberto no teste: antes e depois da escada", "%")
visual.salvar(fig, ENTREGAS / etapa11a.ETAPA, "professor_aberto_no_teste")

antes, depois = comparacao(ESCADA[0], FECHADO, "test"), comparacao(escolhido, FECHADO, "test")
sem_bits_no_teste, maior_no_teste = comparacao(CANDIDATOS[0], ESCADA[0], "test"), comparacao(CANDIDATOS[1], ESCADA[0], "test")
recuperado = 1 - depois["diferenca"] / antes["diferenca"]
display(Markdown(f"""
**Leitura.** No teste o quadro é o mesmo de `dev`. Sem os 4 bits: {com_intervalo(sem_bits_no_teste)} pontos de exact match. Com o modelo maior: {com_intervalo(maior_no_teste)}. A distância do professor aberto para o fechado cai de {com_intervalo(antes)} para {com_intervalo(depois)} pontos: o modelo maior recupera cerca de {pts(recuperado, 0)}% da distância, e o que sobra continua distinguível de zero.
"""))
''')

md(r'''
## 6. O que muda nos rótulos que o aluno recebe

Descritivo, fora da regra. O aluno aprende do treino rotulado pelo professor. Três perguntas: quão bons são esses rótulos, quanto eles mudam de um professor para outro, e quantas respostas a leitura não aproveita.
''')

code(r'''
display(Markdown("**Qualidade dos rótulos do treino** (contra os rótulos humanos, que o aluno nunca vê), com intervalo de 95%"))
display(tabela_de_qualidade("train"))

quanto_muda = mud[mud["particao"] == "train"].set_index("comparacao")[["mesma_intencao", "mesmas_entidades", "mesma_resposta_lida", "mesmo_texto_bruto"]]
display(Markdown(f"**Quanto as respostas do treino coincidem entre dois professores** ({n(len(treino))} pedidos)"))
display(quanto_muda.map(lambda x: pts(x) + "%").rename(columns={"mesma_intencao": "mesma intenção", "mesmas_entidades": "mesmas entidades", "mesma_resposta_lida": "mesma resposta, depois de lida", "mesmo_texto_bruto": "mesmo texto bruto"}))

leitura = diag.assign(professor=diag["versao"].map(CURTOS)).set_index(["professor", "particao"])[
    ["pedidos", "json_invalido", "sem_rotulo_aproveitavel", "entidades", "tipo_fora_da_lista", "texto_nao_encontrado", "cortadas_no_limite_de_tokens"]
].rename(columns={"json_invalido": "JSON inválido", "sem_rotulo_aproveitavel": "sem rótulo aproveitável", "entidades": "entidades devolvidas",
                  "tipo_fora_da_lista": "tipo fora da lista", "texto_nao_encontrado": "texto não encontrado no pedido", "cortadas_no_limite_de_tokens": "cortadas no limite de tokens"})
display(Markdown("**Diagnósticos de leitura das respostas** (contagens)"))
display(leitura)

par_8b = quanto_muda.loc[f"{CURTOS[CANDIDATOS[0]]} e {CURTOS[ESCADA[0]]}"]
par_14b = quanto_muda.loc[f"{CURTOS[CANDIDATOS[1]]} e {CURTOS[ESCADA[0]]}"]
d_treino = diag[diag["particao"] == "train"].set_index("versao")
display(Markdown(f"""
**Leitura.**

- Tirar os 4 bits muda bastante resposta sem mudar a qualidade: no treino, só {pts(par_8b['mesma_resposta_lida'])}% das respostas das duas versões do 8B coincidem (a troca de máquina, sozinha, explicaria {pts(muda_com_a_maquina, 0)}% de mudança), e o exact match dos rótulos fica em {pts(medida(ESCADA[0], 'train')['valor'])}% e {pts(medida(CANDIDATOS[0], 'train')['valor'])}%. As respostas trocam de lugar: erros somem e outros aparecem na mesma medida.
- O modelo maior concorda com o titular em {pts(par_14b['mesma_resposta_lida'])}% das respostas do treino, e os rótulos dele têm exact match de {pts(medida(CANDIDATOS[1], 'train')['valor'])}%.
- A leitura aproveita mais: entidades com tipo fora da lista no treino caem de {n(d_treino.loc[ESCADA[0], 'tipo_fora_da_lista'])} para {n(d_treino.loc[CANDIDATOS[0], 'tipo_fora_da_lista'])} (sem os 4 bits) e {n(d_treino.loc[CANDIDATOS[1], 'tipo_fora_da_lista'])} (modelo maior); respostas com JSON inválido, de {n(d_treino.loc[ESCADA[0], 'json_invalido'])} para {n(d_treino.loc[CANDIDATOS[0], 'json_invalido'])} e {n(d_treino.loc[CANDIDATOS[1], 'json_invalido'])}.
"""))
''')

md(r'''
## 7. Tempo e custo

O Colab cobra em unidades de computação, por hora de sessão ligada. As duas rodadas longas correram ao mesmo tempo, e o saldo é da conta inteira: o que cada condutor mediu pela diferença de saldo inclui o gasto da outra sessão. Por isso o custo é repartido por **taxa × tempo** de cada sessão e depois posto na escala do que a conta de fato cobrou.
''')

code(r'''
tabela_do_custo = pd.DataFrame([
    {"trecho": p["o_que"], "placa": p["gpu"], "de": p["de"], "até": p["ate"], "horas": num(p["horas"], 2), "unidades por hora": num(p["taxa_por_hora"], 2),
     "taxa × tempo": num(p["unidades"], 2), "fechado na conta": num(p["unidades_fechadas_na_conta"], 2)}
    for p in custos["partes"]
]).set_index("trecho")
display(tabela_do_custo)
conta = custos["conta"]
display(Markdown(f"""
**Leitura.** A soma por taxa × tempo dá {num(custos['soma_por_taxa_e_tempo'], 2)} unidades; a conta foi de {num(conta['saldo_antes'], 2)} para {num(conta['saldo_depois'], 2)}: **{num(conta['gasto'], 2)} unidades gastas na etapa**, de um teto de {custos['teto_da_etapa']}. A diferença de {num(abs(custos['diferenca_entre_a_soma_e_a_conta']), 2)} entre a soma e a conta fica registrada; o número que vale é o da conta. Os condutores tinham medido {', '.join(num(x, 2) for x in list(custos['o_que_os_condutores_mediram_pelo_saldo'].values())[1:])} pelas diferenças de saldo, números que se sobrepõem.

Tempo de GPU para rotular dev, teste e treino ({n(len(dev) + len(teste) + len(treino))} pedidos): {horas(sum(versoes[CANDIDATOS[0]]['tempos']['segundos_de_gpu'].values()))} no {CURTOS[CANDIDATOS[0]]} ({versoes[CANDIDATOS[0]]['gpu']['placa']}) e {horas(sum(versoes[CANDIDATOS[1]]['tempos']['segundos_de_gpu'].values()))} no {CURTOS[CANDIDATOS[1]]} ({versoes[CANDIDATOS[1]]['gpu']['placa']}). Na máquina local, em 4 bits, tinham sido {horas(sum(versoes[ESCADA[0]]['execucao']['segundos_de_gpu'].values()))}.
"""))
''')

md(r'''
## 8. Entrega

A entrega é montada a partir das pastas das versões. Os arquivos grandes entram como links físicos: aparecem na entrega sem ocupar o disco duas vezes, e nada sai do cache.

Para um aluno usar outro professor, basta trocar o caminho: `etapa6.PROFESSORES["aberto"] = etapa11a.como_professor(versão)`. A célula abaixo faz essa troca e lê os arquivos com as funções da Etapa 6, as mesmas que a Etapa 10a usa.
''')

code(r'''
PASTA = etapa11a.entregar()

original = dict(etapa6.PROFESSORES["aberto"])
consumo = {}
try:
    for v in CANDIDATOS:
        etapa6.PROFESSORES["aberto"] = etapa11a.como_professor(v)
        do_treino = etapa6.treino_do_professor("aberto", treino)
        do_dev = etapa6.dev_do_professor("aberto", dev, intents)
        no_teste = etapa6.previsoes_do_professor("aberto", "test", teste)
        logits = etapa6.logits_do_professor("treino", len(treino), intents)
        com_rotulo = do_treino["valido"].to_numpy()
        do_rotulo = np.array([intents.index(i) if i in intents else 0 for i in do_treino["intent"]])
        consumo[v] = {
            "pasta": str(etapa6.pasta_do_professor("aberto").relative_to(RAIZ)),
            "treino relido reproduz as respostas": [Previsao.de_tags(i, list(t)) for i, t in zip(do_treino["intent"], do_treino["tags"])] == etapa11a.previsoes(v, "train")[0],
            "a intenção de maior pontuação é a do rótulo": bool(np.all(logits[com_rotulo, do_rotulo[com_rotulo]] >= logits[com_rotulo].max(axis=1) - 1e-4)),
            "logits finitos": bool(np.isfinite(logits).all()),
            "dev e teste lidos": len(do_dev) == len(dev) and len(no_teste) == len(teste),
        }
finally:
    etapa6.PROFESSORES["aberto"] = original
pd.DataFrame(consumo).T
''')

md(r'''
### Critério de pronto
''')

code(r'''
da_5b = versoes[etapa11a.VERSAO_DA_5B]["saidas"]
intacta = all(etapa11a.sha256_do_arquivo(RAIZ / caminho) == registro["sha256"] for caminho, registro in da_5b.items())
condutor = etapa11a.sha256_do_arquivo(etapa11a.PASTA / etapa11a.CONDUTOR_QUE_RODOU)
completas = all(
    versoes[v]["modelo"]["commit"] and versoes[v]["condutor"]["sha256"] == condutor and versoes[v]["codigo"]["sha256_do_pacote"]
    and versoes[v]["custo"]["unidades"] > 0 and versoes[v]["tempos"]["codigo_de_saida_do_trabalho"] == 0 and versoes[v]["gpu"]["placa"]
    for v in (CONFERENCIA, *CANDIDATOS)
)
inteiros = all(int(linha["pedidos"]) == len(particoes[linha["particao"]]) and linha["sem_registro"] == 0 for _, linha in diag.iterrows())
titular = etapa11a.VERSAO_DA_5B
for desafiante, x in disputas.items():      # a regra, refeita a partir das disputas gravadas
    assert x["contra"] == titular
    titular = desafiante if x["ic_min"] > 0 else titular
pronto = pd.Series({
    "Regras gravadas antes da primeira rodada no Colab": regra_veio_antes,
    f"Troca de máquina medida e dentro do limiar ({n(m['pedidos'])} pedidos)": bool(ambiente["aprovado"]),
    "Precisão decidida pela regra, sem olhar métrica (float16, nenhuma pontuação não finita)": all(p["finito"] for p in precisoes.values()),
    f"Cada candidato rotulou dev, teste e treino inteiros ({n(len(dev) + len(teste) + len(treino))} pedidos)": inteiros,
    "Escada aplicada em dev pela regra gravada": titular == escolhido,
    "Arquivos de cada versão lidos pelo código da Etapa 6 (rótulos, logits, dev e teste)": all(all(v for k, v in c.items() if k != "pasta") for c in consumo.values()),
    "versao.json completo nas versões do Colab (commit, pacote, condutor, placa, tempos, custo)": bool(completas),
    "Nada da Etapa 5b mudou (mesmas impressões digitais das respostas, dos rótulos e dos logits)": intacta,
    f"Gasto dentro do teto ({num(conta['gasto'], 2)} de {custos['teto_da_etapa']} unidades)": conta["gasto"] <= custos["teto_da_etapa"],
}).map({True: "sim", False: "NÃO"}).rename("atendido").to_frame()
assert (pronto["atendido"] == "sim").all()
pronto
''')

md(r'''
## 9. Resposta

O resumo abaixo é gerado a partir dos números calculados neste notebook e gravado em `entregas/11a_professor_aberto_maior/RESUMO.md`.
''')

code(r'''
para_os_alunos = escolha["para_os_alunos"]
linhas_da_escada = "\n".join(
    f"| {NOMES[d]} | {CURTOS[x['contra']]} | {pts(x['a'])} | {pts(x['b'])} | {com_intervalo(x)} | {'sim' if x['ic_min'] > 0 else 'não'} |" for d, x in disputas.items())
linhas_de_qualidade = "\n".join(
    f"| {NOMES[s]} | " + " | ".join(valor_com_intervalo(medida(s, p, metrica)) for p in ("dev", "test") for metrica in ("intent_acc", "slot_f1", "exact_match")) + " |"
    for s in (*ESCADA, FECHADO))
linhas_de_comparacao = "\n".join(
    f"| {linha['comparacao']} | {com_intervalo(comparacao(linha['b'], linha['a'], 'dev'))} | {com_intervalo(linha)} |"
    for _, linha in comp[(comp["particao"] == "test") & (comp["metrica"] == "exact_match")].iterrows())
linhas_de_custo = "\n".join(f"| {p['o_que']} | {p['gpu']} | {num(p['horas'], 2)} | {num(p['unidades'], 2)} | {num(p['unidades_fechadas_na_conta'], 2)} |" for p in custos["partes"])
linhas_de_pronto = "\n".join(f"| {criterio} | {linha['atendido']} |" for criterio, linha in pronto.iterrows())

compressao_pesa = sem_4_bits["ic_min"] > 0 or sem_4_bits["ic_max"] < 0
tamanho_pesa, ainda_sobra = maior["ic_min"] > 0, depois["ic_max"] < 0
veredito = ("As duas coisas." if tamanho_pesa and ainda_sobra else "Porque teve de caber em 10 GB." if tamanho_pesa else "Por ser aberto, até onde esta etapa mediu.")
sobre_a_compressao = ("A compressão em 4 bits pesa" if compressao_pesa else "A compressão em 4 bits não pesa") + f": tirá-la do Qwen3-8B muda o exact match em `dev` em {com_intervalo(sem_4_bits)} pontos."
sobre_o_tamanho = (f"O tamanho do modelo pesa: o **{etapa11a.descrever(CANDIDATOS[1])['nome']}**, que não caberia na placa local, acrescenta {com_intervalo(maior)} pontos em `dev`"
                   + (" e passa a ser o professor aberto." if escolhido == CANDIDATOS[1] else ", sem tomar o lugar do titular."))
sobre_o_resto = (f"A distância para o professor fechado, em exact match no teste, vai de {com_intervalo(antes)} para {com_intervalo(depois)} pontos: pela estimativa pontual, o modelo maior fecha cerca de {pts(recuperado, 0)}% dela"
                 + (", e o que sobra continua distinguível de zero" if ainda_sobra else ", e o que sobra já não se distingue de zero")
                 + f" com um modelo aberto de {num(versoes[escolhido]['modelo']['parametros'] / 1e9, 1)} bilhões de parâmetros.")

resumo = f"""# Etapa 11a · O professor aberto sem o limite dos 10 GB: resumo da entrega

**Pergunta.** O cenário B perde por ser aberto, ou porque o professor teve de caber em 10 GB?

**Resposta.** {veredito} {sobre_a_compressao} {sobre_o_tamanho} No teste, o {NOMES[escolhido]} faz {pts(medida(escolhido, 'test', 'intent_acc')['valor'])}% de acurácia de intenção, {pts(medida(escolhido, 'test', 'slot_f1')['valor'])}% de F1 de entidades e {pts(medida(escolhido, 'test')['valor'])}% de exact match. {sobre_o_resto}

## A escada (dev)

Regra, gravada em {regra['escrita_em'][:16]}, antes da primeira rodada: {regra['escada']['regra']}.

| Desafiante | Titular | Exact match do titular | Exact match do desafiante | Diferença (intervalo de 95%) | Toma o lugar |
|---|---|---|---|---|---|
{linhas_da_escada}

A escada parou no Qwen3-14B. Candidatos maiores não foram rodados.

## Qualidade dos professores

Valor com intervalo de 95%. A decisão saiu de `dev`; o teste só é relatado.

| Professor | Intenção (dev) | Entidades F1 (dev) | Exact match (dev) | Intenção (teste) | Entidades F1 (teste) | Exact match (teste) |
|---|---|---|---|---|---|---|
{linhas_de_qualidade}

Diferenças pareadas de exact match, B − A, em pontos:

| Comparação | dev | teste |
|---|---|---|
{linhas_de_comparacao}

## O que muda nos rótulos do treino

- As duas versões do Qwen3-8B (com e sem os 4 bits) dão a mesma resposta em {pts(par_8b['mesma_resposta_lida'])}% dos pedidos do treino, e a mesma intenção em {pts(par_8b['mesma_intencao'])}%. A qualidade dos rótulos é a mesma ({pts(medida(ESCADA[0], 'train')['valor'])}% e {pts(medida(CANDIDATOS[0], 'train')['valor'])}% de exact match): as respostas mudam, o acerto não.
- O Qwen3-14B dá a mesma resposta que o titular em {pts(par_14b['mesma_resposta_lida'])}% do treino; os rótulos dele têm {pts(medida(CANDIDATOS[1], 'train')['valor'])}% de exact match.
- Leitura das respostas no treino: tipo de entidade fora da lista em {n(d_treino.loc[ESCADA[0], 'tipo_fora_da_lista'])}, {n(d_treino.loc[CANDIDATOS[0], 'tipo_fora_da_lista'])} e {n(d_treino.loc[CANDIDATOS[1], 'tipo_fora_da_lista'])} entidades (4 bits, sem os 4 bits, modelo maior); texto não encontrado no pedido em {n(d_treino.loc[ESCADA[0], 'texto_nao_encontrado'])}, {n(d_treino.loc[CANDIDATOS[0], 'texto_nao_encontrado'])} e {n(d_treino.loc[CANDIDATOS[1], 'texto_nao_encontrado'])}; JSON inválido em {n(d_treino.loc[ESCADA[0], 'json_invalido'])}, {n(d_treino.loc[CANDIDATOS[0], 'json_invalido'])} e {n(d_treino.loc[CANDIDATOS[1], 'json_invalido'])} respostas.

## Como apontar os alunos para um professor desta etapa

Cada versão tem, na sua pasta, os mesmos arquivos da Etapa 5b: `rotulos_treino.parquet`, `logits_treino.npy` (e `dev`, `test`), `intencoes.json` e `previsoes/<versão>_dev.parquet` e `_test.parquet`. O código das Etapas 6 e 10a lê o professor por `etapa6.PROFESSORES`; basta trocar a entrada:

```python
from especialista.experimentos import etapa6, etapa11a
etapa6.PROFESSORES["aberto"] = etapa11a.como_professor("{escolhido}")
# {json.dumps(para_os_alunos, ensure_ascii=False)}
```

Os alunos que saírem daí precisam de pasta de execução própria: os da Etapa 6 e da 10a, treinados com o professor de 4 bits, não podem ser sobrescritos.

## Custo e tempo

| Trecho | Placa | Horas | Taxa × tempo (unidades) | Fechado na conta |
|---|---|---|---|---|
{linhas_de_custo}

A conta do Colab foi de {num(conta['saldo_antes'], 2)} para {num(conta['saldo_depois'], 2)}: **{num(conta['gasto'], 2)} unidades** na etapa (teto: {custos['teto_da_etapa']}). A soma por taxa × tempo dá {num(custos['soma_por_taxa_e_tempo'], 2)}; a última coluna é a mesma repartição na escala da conta.

## Critério de pronto

| Critério | Atendido |
|---|---|
{linhas_de_pronto}

## Limites

- **"Só sai a compressão" não é uma comparação limpa.** Na versão em 4 bits, os tensores não comprimidos ficam em bfloat16; sem os 4 bits, tudo está em float16. Entre as duas versões do 8B mudam a compressão, esse tipo numérico e a máquina.
- **Trocar de máquina, sozinho, muda cerca de {pts(muda_com_a_maquina, 0)}% das respostas** (conferência de ambiente, {n(m['pedidos'])} pedidos de `dev`, mesma configuração). A medida é de uma amostra, numa placa (L4); a A100 não teve conferência própria.
- **A escada parou no 14B.** O Colab tem uma placa de 96 GB, onde caberia um Qwen3-32B. Rodar ou não é decisão de quem conduz o projeto; nada maior foi rodado.
- **Um modelo por tamanho, uma família.** Os dois degraus são do Qwen3; o resultado não diz como outra família de modelos se sairia com o mesmo tamanho.
- **Os alunos não foram refeitos.** Esta etapa entrega o professor; o efeito no aluno do cenário B ainda não foi medido.
- **O que não foi exercitado no condutor:** a retomada depois de uma queda da VM e o modo de relançar o condutor só para acompanhar. As duas rodadas foram do começo ao fim sem queda, e o desligamento automático da sessão funcionou nas duas.
- **A versão retroativa da Etapa 5b** foi registrada depois do fato: dois arquivos de código tinham mudado depois da rodada (`etapa5b.py` e `avaliacao/previsao.py`), e isso está marcado no `versao.json` dela.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `escolha.json` | A regra, as disputas, o escolhido e a linha que aponta os alunos para ele |
| `regra.json` | As três regras, com a hora em que foram gravadas |
| `resultados.csv`, `qualidade.csv`, `comparacoes.csv` | Qualidade com intervalo (dev, teste e rótulos do treino) e as diferenças pareadas |
| `mudancas_entre_professores.csv`, `diagnosticos.csv` | Quanto as respostas coincidem entre professores e o que a leitura não aproveitou |
| `conferencia_de_ambiente.json` | A troca de máquina, medida contra o limiar |
| `custo.json` | As unidades do Colab, repartidas por taxa × tempo e fechadas na conta |
| `sistemas.json` | Os dois professores novos, declarados para o comparativo (formato de `etapa10d.fichas_declaradas`) |
| `professores/<versão>/` | De cada versão: `versao.json`, rótulos, logits, previsões, respostas brutas, ambiente e registro do condutor |
| `codigo/` | O pacote enviado ao Colab, o condutor que rodou e o código da etapa |
| `manifesto.json` | Tudo o que a entrega contém, com impressão digital, e de que arquivos ela foi feita |
| `figuras/`, `11a_professor_aberto_maior.html` | As figuras e este notebook executado |
"""
(PASTA / "RESUMO.md").write_text(resumo, encoding="utf-8")
display(Markdown(resumo))
''')

code(r'''
arquivos = sorted(p for p in PASTA.rglob("*") if p.is_file())
pd.DataFrame(
    {"tamanho (KB)": [round(p.stat().st_size / 1024, 1) for p in arquivos], "link físico": ["sim" if p.stat().st_nlink > 1 else "" for p in arquivos]},
    index=[str(p.relative_to(RAIZ)) for p in arquivos],
)
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("ok", len(cells), "células")
