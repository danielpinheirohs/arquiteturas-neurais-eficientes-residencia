# Etapa 11a · O professor aberto sem o limite dos 10 GB: resumo da entrega

**Pergunta.** O cenário B perde por ser aberto, ou porque o professor teve de caber em 10 GB?

**Resposta.** As duas coisas. A compressão em 4 bits não pesa: tirá-la do Qwen3-8B muda o exact match em `dev` em +0,0 (−1,1 a +1,2) pontos. O tamanho do modelo pesa: o **Qwen3-14B**, que não caberia na placa local, acrescenta +6,0 (+4,2 a +7,7) pontos em `dev` e passa a ser o professor aberto. Numa **extensão decidida depois de ver esse resultado**, o Qwen3-32B desafiou o titular: −0,05 (−1,62 a +1,72) pontos em `dev`, com o zero dentro do intervalo. Não toma o lugar: o professor aberto continua sendo o Qwen3-14B; pela regra, o modelo de 33 bilhões de parâmetros não se distingue do de 15. No teste, o Qwen3-14B sem os 4 bits, em float16 (Colab, A100) faz 80,0% de acurácia de intenção, 61,4% de F1 de entidades e 49,0% de exact match. A distância para o professor fechado, em exact match no teste, vai de −10,8 (−12,4 a −9,1) para −5,1 (−6,7 a −3,5) pontos: pela estimativa pontual, o modelo maior fecha cerca de 53% dela, e o que sobra continua distinguível de zero com um modelo aberto de 14,8 bilhões de parâmetros.

**Versões desta entrega.** Esta é a de 19/11/2026, com a extensão. A de 18/11/2026 (a escada planejada, de três degraus) está guardada inteira em `versoes/2026-11-18_escada_de_tres_degraus/`: resumo, notebook executado, HTML, escolha, regra, tabelas, figuras, manifesto e código. `escolha.json` e `regra.json` são os de 18/11, sem alteração.

## A escada (dev)

Regra, gravada em 2026-11-18 17:16, antes da primeira rodada: do mais barato para o mais caro; um candidato mais caro só toma o lugar se o exact match em dev for distinguivelmente maior (IC95 da diferença pareada acima de zero).

| Desafiante | Titular | Exact match do titular | Exact match do desafiante | Diferença (intervalo de 95%) | Toma o lugar |
|---|---|---|---|---|---|
| Qwen3-8B sem os 4 bits, em float16 (Colab, L4) | 8B, 4 bits | 43,2 | 43,2 | +0,0 (−1,1 a +1,2) | não |
| Qwen3-14B sem os 4 bits, em float16 (Colab, A100) | 8B, 4 bits | 43,2 | 49,1 | +6,0 (+4,2 a +7,7) | sim |

A escada planejada parou no Qwen3-14B. O degrau seguinte foi rodado depois, como extensão.

## Extensão, decidida depois: o Qwen3-32B

**Não estava nas regras de 18/11, 17h16.** Foi decidida depois de ver o resultado da escada acima, e autorizada pelo usuário (a autorização chegou à rodada pela mensagem do orquestrador). A regra, a mesma, foi gravada em 2026-11-18 22:09 em `regra_extensao_32b.json`, antes de a rodada começar; a disputa está em `escolha_extensao_32b.json`. É uma segunda olhada nos mesmos dados de `dev`, e por isso é relatada à parte.

| Desafiante | Titular | Exact match do titular (dev) | Exact match do desafiante (dev) | Diferença em dev (intervalo de 95%) | Toma o lugar | Diferença no teste (só relato) |
|---|---|---|---|---|---|---|
| Qwen3-32B sem os 4 bits, em float16 (Colab, G4) | 14B, float16 | 49,1 | 49,1 | −0,05 (−1,62 a +1,72) | não | +1,04 (−0,30 a +2,46) |

- Os rótulos do treino do 32B têm 48,7% de exact match contra os humanos (48,2% no titular); as respostas dos dois coincidem em 63,3% do treino.
- No teste, o 32B fica a −4,0 (−5,6 a −2,5) pontos do professor fechado.
- Custo da extensão: 24,15 unidades do Colab (2,48 h de G4), para um modelo de 32,8 bilhões de parâmetros e 65,5 GB na placa.
- A versão fica guardada em `professores/qwen3_32b__fp16__colab_g4/`, com os mesmos arquivos das outras, e pode ser usada por um aluno do mesmo jeito.

## Qualidade dos professores

Valor com intervalo de 95%. A decisão saiu de `dev`; o teste só é relatado.

| Professor | Intenção (dev) | Entidades F1 (dev) | Exact match (dev) | Intenção (teste) | Entidades F1 (teste) | Exact match (teste) |
|---|---|---|---|---|---|---|
| Qwen3-8B em 4 bits (RTX 3080 local, Etapa 5b) | 77,4 (75,7 a 79,2) | 58,0 (55,9 a 60,1) | 43,2 (41,0 a 45,3) | 77,6 (76,1 a 79,1) | 55,7 (53,9 a 57,4) | 43,2 (41,5 a 45,0) |
| Qwen3-8B sem os 4 bits, em float16 (Colab, L4) | 77,9 (76,2 a 79,7) | 57,8 (55,7 a 60,0) | 43,2 (41,0 a 45,5) | 77,9 (76,5 a 79,4) | 56,5 (54,8 a 58,2) | 43,4 (41,6 a 45,2) |
| Qwen3-14B sem os 4 bits, em float16 (Colab, A100) | 80,6 (79,0 a 82,3) | 62,4 (60,4 a 64,6) | 49,1 (47,0 a 51,5) | 80,0 (78,6 a 81,4) | 61,4 (59,6 a 63,1) | 49,0 (46,9 a 50,7) |
| Qwen3-32B sem os 4 bits, em float16 (Colab, G4) | 81,0 (79,4 a 82,7) | 63,3 (61,3 a 65,4) | 49,1 (47,0 a 51,4) | 80,9 (79,4 a 82,3) | 62,5 (60,8 a 64,3) | 50,0 (48,2 a 51,8) |
| Claude Haiku 5.5 (professor fechado, Etapa 5a) | 84,1 (82,6 a 85,6) | 68,5 (66,5 a 70,5) | 54,7 (52,6 a 57,0) | 83,5 (82,1 a 84,7) | 66,1 (64,5 a 67,7) | 54,0 (52,3 a 55,7) |

Diferenças pareadas de exact match, B − A, em pontos:

| Comparação | dev | teste |
|---|---|---|
| 8B, float16 − 8B, 4 bits | +0,0 (−1,1 a +1,2) | +0,2 (−0,8 a +1,2) |
| 14B, float16 − 8B, 4 bits | +6,0 (+4,2 a +7,7) | +5,7 (+4,2 a +7,3) |
| 14B, float16 − 8B, float16 | +6,0 (+4,3 a +7,7) | +5,5 (+4,1 a +7,0) |
| 8B, 4 bits − Claude Haiku 5.5 | −11,6 (−13,4 a −9,6) | −10,8 (−12,4 a −9,1) |
| 8B, float16 − Claude Haiku 5.5 | −11,6 (−13,6 a −9,5) | −10,6 (−12,2 a −9,0) |
| 14B, float16 − Claude Haiku 5.5 | −5,6 (−7,6 a −3,6) | −5,1 (−6,7 a −3,5) |
| 32B, float16 − 14B, float16 | −0,0 (−1,6 a +1,7) | +1,0 (−0,3 a +2,5) |
| 32B, float16 − 8B, 4 bits | +5,9 (+4,2 a +7,8) | +6,8 (+5,2 a +8,3) |
| 32B, float16 − Claude Haiku 5.5 | −5,7 (−7,5 a −3,7) | −4,0 (−5,6 a −2,5) |

## O que muda nos rótulos do treino

- As duas versões do Qwen3-8B (com e sem os 4 bits) dão a mesma resposta em 73,7% dos pedidos do treino, e a mesma intenção em 92,7%. A qualidade dos rótulos é a mesma (43,6% e 44,1% de exact match): as respostas mudam, o acerto não.
- O Qwen3-14B dá a mesma resposta que o titular em 55,6% do treino; os rótulos dele têm 48,2% de exact match.
- Leitura das respostas no treino: tipo de entidade fora da lista em 532, 412 e 112 entidades (4 bits, sem os 4 bits, modelo maior); texto não encontrado no pedido em 259, 206 e 104; JSON inválido em 11, 1 e 0 respostas.

## Como apontar os alunos para um professor desta etapa

Cada versão tem, na sua pasta, os mesmos arquivos da Etapa 5b: `rotulos_treino.parquet`, `logits_treino.npy` (e `dev`, `test`), `intencoes.json` e `previsoes/<versão>_dev.parquet` e `_test.parquet`. O código das Etapas 6 e 10a lê o professor por `etapa6.PROFESSORES`; basta trocar a entrada:

```python
from especialista.experimentos import etapa6, etapa11a
etapa6.PROFESSORES["aberto"] = etapa11a.como_professor("qwen3_14b__fp16__colab_a100")
# {"nome": "Qwen3-14B", "etapa": "11a_professor_aberto_maior/professores/qwen3_14b__fp16__colab_a100", "previsoes": "qwen3_14b__fp16__colab_a100"}
```

Os alunos que saírem daí precisam de pasta de execução própria: os da Etapa 6 e da 10a, treinados com o professor de 4 bits, não podem ser sobrescritos.

## Custo e tempo

| Trecho | Placa | Horas | Taxa × tempo (unidades) | Fechado na conta |
|---|---|---|---|---|
| conferência de ambiente (4 bits na L4), do começo da sessão até o candidato 1 assumi-la | L4 | 0,12 | 0,23 | 0,23 |
| candidato 1: precisão, dev, teste e treino | L4 | 3,62 | 7,02 | 6,87 |
| candidato 2: precisão, dev, teste e treino | A100 | 2,29 | 15,39 | 15,06 |
| sonda da A100, até o condutor do candidato 2 assumir a sessão | A100 | 0,07 | 0,50 | 0,49 |
| sonda da L4 (máquina, kernel que segue sem a ligação local, desligamento pela própria VM) | L4 | 0,03 | 0,06 | 0,06 |
| sonda da G4 (placa e taxa) | G4 | 0,01 | 0,13 | 0,13 |

Na escada de três degraus (18/11), a conta do Colab foi de 600,00 para 577,17: **22,83 unidades**. A soma por taxa × tempo dá 23,33; a última coluna é a mesma repartição na escala da conta.

Na extensão, a sessão da G4 ficou ligada 2,48 h a 10,05 unidades por hora: 24,90 por taxa × tempo, **24,15 na conta** (de 577,17 para 553,02; a autorização previa cerca de 40 unidades, com teto de 60).

**Total do Colab no projeto: 46,98 unidades** (de 600,00 para 553,02; teto da etapa: 150).

## Critério de pronto

| Critério | Atendido |
|---|---|
| Regras gravadas antes da primeira rodada no Colab | sim |
| Troca de máquina medida e dentro do limiar (300 pedidos) | sim |
| Precisão decidida pela regra, sem olhar métrica (float16, nenhuma pontuação não finita) | sim |
| Cada candidato rotulou dev, teste e treino inteiros (16.521 pedidos) | sim |
| Escada aplicada em dev pela regra gravada | sim |
| Arquivos de cada versão lidos pelo código da Etapa 6 (rótulos, logits, dev e teste) | sim |
| versao.json completo nas versões do Colab (commit, pacote, condutor, placa, tempos, custo) | sim |
| Nada da Etapa 5b mudou (mesmas impressões digitais das respostas, dos rótulos e dos logits) | sim |
| Extensão: regra própria gravada antes da rodada do 32B, dizendo que veio depois da escada de três degraus | sim |
| Extensão: o 32B rotulou tudo e a disputa com o titular seguiu a regra | sim |
| A escolha de 18/11 não foi alterada, e a entrega de 18/11 está guardada dentro desta | sim |
| Gasto do Colab dentro do teto da etapa (46,98 de 150 unidades, com a extensão) | sim |

## Limites

- **"Só sai a compressão" não é uma comparação limpa.** Na versão em 4 bits, os tensores não comprimidos ficam em bfloat16; sem os 4 bits, tudo está em float16. Entre as duas versões do 8B mudam a compressão, esse tipo numérico e a máquina.
- **Trocar de máquina, sozinho, muda cerca de 4% das respostas** (conferência de ambiente, 300 pedidos de `dev`, mesma configuração). A medida é de uma amostra, numa placa (L4). A A100 e a G4 não tiveram conferência própria: entre o 8B, o 14B e o 32B muda o modelo e muda também a máquina.
- **A extensão não estava nas regras originais.** O degrau do 32B foi decidido depois de ver a escada de três degraus. `dev` serviu para duas decisões seguidas; a regra não mudou, mas quem olha de novo os mesmos dados tem mais chance de achar diferença por acaso. Aqui a segunda olhada não achou diferença, o que não depende desse cuidado.
- **"Não toma o lugar" não é "é igual".** O intervalo da diferença 32B − 14B em `dev` (−0,05 (−1,62 a +1,72)) admite um ganho pequeno ou uma perda pequena. No teste a diferença pontual é positiva (+1,04 (−0,30 a +2,46)), também sem se distinguir de zero; o teste não decide.
- **A escada parou no 32B.** Nada maior foi rodado, e o resultado não diz o que um modelo ainda maior faria.
- **Um modelo por tamanho, uma família.** Os dois degraus são do Qwen3; o resultado não diz como outra família de modelos se sairia com o mesmo tamanho.
- **Os alunos não foram refeitos.** Esta etapa entrega o professor; o efeito no aluno do cenário B ainda não foi medido.
- **O que não foi exercitado no condutor:** a retomada depois de uma queda da VM e o modo de relançar o condutor só para acompanhar. As duas rodadas foram do começo ao fim sem queda, e o desligamento automático da sessão funcionou nas duas.
- **A versão retroativa da Etapa 5b** foi registrada depois do fato: dois arquivos de código tinham mudado depois da rodada (`etapa5b.py` e `avaliacao/previsao.py`), e isso está marcado no `versao.json` dela.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `escolha.json` | A escada planejada (18/11): a regra, as disputas, o escolhido e a linha que aponta os alunos para ele |
| `regra.json` | As três regras de 18/11, com a hora em que foram gravadas |
| `regra_extensao_32b.json`, `escolha_extensao_32b.json` | A extensão: a regra (com o que já se sabia ao escrevê-la) e a disputa do 32B com o titular |
| `versoes/2026-11-18_escada_de_tres_degraus/` | A entrega como foi fechada em 18/11, antes da extensão |
| `resultados.csv`, `qualidade.csv`, `comparacoes.csv` | Qualidade com intervalo (dev, teste e rótulos do treino) e as diferenças pareadas |
| `mudancas_entre_professores.csv`, `diagnosticos.csv` | Quanto as respostas coincidem entre professores e o que a leitura não aproveitou |
| `conferencia_de_ambiente.json` | A troca de máquina, medida contra o limiar |
| `custo.json` | As unidades do Colab, repartidas por taxa × tempo e fechadas na conta |
| `sistemas.json` | Os três professores novos (8B sem os 4 bits, 14B e 32B), declarados para o comparativo (formato de `etapa10d.fichas_declaradas`) |
| `professores/<versão>/` | De cada versão: `versao.json`, rótulos, logits, previsões, respostas brutas, ambiente e registro do condutor |
| `codigo/` | O pacote enviado ao Colab, o condutor que rodou e o código da etapa |
| `manifesto.json` | Tudo o que a entrega contém, com impressão digital, e de que arquivos ela foi feita |
| `figuras/`, `11a_professor_aberto_maior.html` | As figuras e este notebook executado |
