# Etapa 13 · Rodada 2 do piloto de pré-treino (exploratório)

**Pergunta.** Algum pré-treino ou adaptação faz a minha rede melhorar de forma distinguível em exact match, contra o modelo já entregue?

**Resposta.** **Não.** Foram 6 variantes no teto e 2 no aluno A. Contra o modelo entregue, nenhuma tem ganho de exact match com intervalo acima de zero no teste: a melhor do teto faz +0,7 (−0,1 a +1,5) e a do aluno A, −0,3 (−1,1 a +0,5). **O pré-treino tem efeito real, mas pequeno e concentrado nas entidades:** contra a mesma rede sem pré-treino, o teto ganha +1,2 (+0,4 a +2,0) de exact match no teste, e o F1 de entidades do teto sobe em todas as variantes com pré-treino, em dev e no teste (de +0,9 a +1,7 contra o entregue). A intenção não acompanha e, na variante que mais ajuda as entidades, cai −1,1 (−1,9 a −0,3).

Tudo aqui é exploratório: várias variantes, escolhidas em dev, sem correção para comparações múltiplas.

## O que foi testado

| Bloco | Variante | O que muda | Sistemas |
|---|---|---|---|
| A | `r1_lr3e-4` | Pré-treino da rodada 1; ajuste com taxa 3e-4 em tudo | teto |
| A | `r1_pre_x0.1` | Pré-treino da rodada 1; taxa 1e-4 nas partes pré-treinadas e 1e-3 nas cabeças | teto |
| A | `r1_aquec500` | Pré-treino da rodada 1; aquecimento de 500 passos no ajuste | teto |
| B | `vocabg_controle` | Vocabulário grande (105.202 features), **sem** pré-treino | teto, aluno A |
| B | `vocabg_2p` | Vocabulário grande, com pré-treino de 0,89 passada, receita de sempre | teto, aluno A |
| C | `vocabg_2p_pre_x0.1` | O mesmo, com taxa 1e-4 nas partes pré-treinadas | teto |

- **Vocabulário grande:** os 25.202 features do MASSIVE mais 60.000 palavras e 20.000 n-gramas da Wikipédia. A rede vai de 4,9 M para 15,1 M de parâmetros (cerca de 61 MB em 32 bits, contra 20 MB).
- **Pré-treino do bloco B:** 35,0 minutos de GPU, 28.546 passos. Foram pedidas 2 passadas e ele parou em **0,89**, no limite de tempo, porque dividiu a placa com 7 ajustes. O nome `vocabg_2p` não corresponde ao que rodou. A perda de máscara foi de 5,73 para 2,80.

## Dev: todas as variantes

Diferença pareada, média de 3 sementes, intervalo de 95%. Em negrito quando o intervalo não contém zero.

| Sistema | Variante | Contra | Intenção | Entidades (F1) | Exact match |
|---|---|---|---|---|---|
| Teto | `r1_lr3e-4` | entregue | +0,3 (−0,6 a +1,1) | +0,8 (−0,1 a +1,7) | +0,4 (−0,7 a +1,4) |
| Teto | `r1_pre_x0.1` | entregue | −0,4 (−1,4 a +0,6) | **+1,4 (+0,3 a +2,5)** | +0,4 (−0,9 a +1,7) |
| Teto | `r1_aquec500` | entregue | +0,5 (−0,4 a +1,3) | +0,8 (−0,2 a +1,7) | +0,8 (−0,3 a +1,8) |
| Teto | `vocabg_controle` | entregue | +0,2 (−0,4 a +0,9) | −0,7 (−1,4 a +0,1) | −0,4 (−1,2 a +0,5) |
| Teto | `vocabg_2p` | entregue | +0,4 (−0,4 a +1,2) | +0,7 (−0,2 a +1,6) | +1,1 (+0,0 a +2,1) |
| Teto | `vocabg_2p` | mesma rede sem pré-treino | +0,2 (−0,6 a +0,9) | **+1,4 (+0,5 a +2,2)** | **+1,5 (+0,5 a +2,4)** |
| Teto | `vocabg_2p_pre_x0.1` | entregue | +0,0 (−1,1 a +1,1) | **+1,4 (+0,2 a +2,5)** | +1,0 (−0,5 a +2,4) |
| Teto | `vocabg_2p_pre_x0.1` | mesma rede sem pré-treino | −0,2 (−1,2 a +0,9) | **+2,1 (+1,0 a +3,1)** | +1,4 (−0,0 a +2,8) |
| Aluno A | `vocabg_controle` | entregue | −0,3 (−0,9 a +0,3) | −0,5 (−1,1 a +0,1) | **−0,8 (−1,5 a −0,1)** |
| Aluno A | `vocabg_2p` | entregue | +0,5 (−0,3 a +1,2) | +0,0 (−0,8 a +0,8) | +0,1 (−0,8 a +1,1) |
| Aluno A | `vocabg_2p` | mesma rede sem pré-treino | **+0,8 (+0,0 a +1,5)** | +0,5 (−0,2 a +1,2) | +0,9 (+0,0 a +1,7) |

Para comparar, a rodada 1 (pré-treino com o vocabulário do MASSIVE, receita de sempre) deu, contra o entregue: teto +1,1 (+0,1 a +2,1) e aluno A +0,4 (−0,4 a +1,4).

## Teste: só as finalistas, lido uma vez

Finalistas gravadas em `rodada2/finalistas.json` antes da leitura: as de maior exact match em dev contra o entregue.

| Sistema | Variante | Contra | Intenção | Entidades (F1) | Exact match |
|---|---|---|---|---|---|
| Teto | `vocabg_2p` | entregue (62,5) | −0,6 (−1,3 a +0,1) | **+0,9 (+0,2 a +1,7)** | +0,7 (−0,1 a +1,5) |
| Teto | `vocabg_2p` | mesma rede sem pré-treino (61,9) | −0,0 (−0,7 a +0,6) | **+1,1 (+0,3 a +1,8)** | **+1,2 (+0,4 a +2,0)** |
| Teto | `vocabg_2p_pre_x0.1` | entregue (62,5) | **−1,1 (−1,9 a −0,3)** | **+1,7 (+0,7 a +2,6)** | +0,1 (−0,9 a +1,1) |
| Teto | `vocabg_2p_pre_x0.1` | mesma rede sem pré-treino (61,9) | −0,6 (−1,4 a +0,2) | **+1,8 (+0,9 a +2,7)** | +0,7 (−0,3 a +1,7) |
| Aluno A | `vocabg_2p` | entregue (51,2) | −0,6 (−1,2 a +0,0) | +0,4 (−0,4 a +1,1) | −0,3 (−1,1 a +0,5) |
| Aluno A | `vocabg_2p` | mesma rede sem pré-treino (50,5) | +0,2 (−0,4 a +0,8) | +0,5 (−0,2 a +1,2) | +0,4 (−0,3 a +1,2) |

## Bloco D: mais pré-treino (acrescentado às 11:00, depois de fechados os blocos A a C)

A hipótese que faltava: pouco pré-treino. Vocabulário do MASSIVE (a rede continua com 4,9 M de parâmetros), **3 passadas completas** nas 8,2 milhões de frases, com a placa só para o pré-treino (20,8 minutos; perda de máscara de 5,74 para 3,01, contra 3,18 com 1 passada), e a receita de sempre no ajuste. Regras em `rodada2/regras_bloco_D.json`, com um desvio declarado: o teste foi lido uma vez para esta variante qualquer que fosse o resultado em dev. No teto, passam a ser 3 leituras do teste nesta rodada.

| Sistema | Partição | Intenção | Entidades (F1) | Exact match |
|---|---|---|---|---|
| Teto | dev | +0,5 (−0,4 a +1,3) | +0,7 (−0,3 a +1,7) | +1,0 (−0,1 a +2,1) |
| Teto | teste | −0,1 (−0,7 a +0,6) | **+1,2 (+0,4 a +2,0)** | +0,5 (−0,4 a +1,3) |
| Aluno A | dev | +0,3 (−0,4 a +1,1) | −0,3 (−1,1 a +0,5) | −0,2 (−1,1 a +0,7) |
| Aluno A | teste | +0,4 (−0,3 a +1,1) | −0,3 (−0,9 a +0,5) | −0,1 (−0,9 a +0,8) |

Diferença contra o modelo entregue. **Triplicar o pré-treino não mudou o quadro:** o exact match fica onde estava (teto +0,5 no teste, contra +0,1 com 1 passada, os dois com o zero no intervalo), e o único efeito que se repete continua sendo o F1 de entidades do teto. Com isso são 7 variantes no teto e 3 no aluno A.

## O que se aprende

1. **O vocabulário grande sozinho piora.** Triplicar a rede sem pré-treino perde exact match (teto 61,9 contra 62,5 no teste; aluno A 50,5 contra 51,2).
2. **O pré-treino devolve essa perda e um pouco mais, no teto.** Mas contra a rede pequena já entregue o saldo é +0,7 com o zero no intervalo, por 3 vezes o tamanho.
3. **O efeito é nas entidades.** Em todas as variantes com pré-treino o F1 de entidades do teto sobe; baixar a taxa nas partes pré-treinadas aumenta esse ganho e derruba a intenção.
4. **No aluno A o pré-treino não aparece.** Com rótulos do professor, nenhuma variante se distingue do aluno entregue.
5. **Mudar a receita de ajuste não foi a chave.** Taxa menor e aquecimento não superaram a receita de sempre.

## Limites

- **Várias variantes, escolha em dev, sem correção para comparações múltiplas.**
- **Pré-treino curto no bloco B:** menos de uma passada, por contenção na placa. O bloco D testou 3 passadas com a rede pequena; mais pré-treino com o vocabulário grande não foi testado.
- **Bloco A só no teto.** As três receitas de ajuste não rodaram no aluno A; `r1_lr3e-4` no aluno A estava nas regras do bloco C e foi tirada da fila por prazo (`rodada2/fila_cortes.txt`).
- **`r1_lr3e-4` parou entre as épocas 75 e 80 de 80:** provavelmente subtreinada.
- **Corpus enciclopédico, majoritariamente do Brasil;** os pedidos são comandos em português europeu.
- **Sem poda nem quantização** dos modelos novos; o tamanho em MB é a conta de parâmetros em 32 bits, não arquivo medido.
- **A VRAM chegou a 9,75 de 10,24 GB** com os ajustes de vocabulário grande; cada ajuste levou cerca de 15 minutos em vez de 3,5.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `rodada2/regras_bloco_A.json`, `_B`, `_C`, `_D` | O que foi fixado antes de cada bloco, com a hora |
| `rodada2/dev_bloco_A_teto_parcial.csv`, `dev_blocos_A_B.csv`, `dev_bloco_C.csv` | Todas as comparações em dev |
| `rodada2/finalistas.json`, `teste_finalistas.csv` | A escolha das finalistas e a leitura do teste |
| `rodada2/dev_bloco_D.csv`, `teste_finalistas_bloco_D.csv`, `pretreino_massive_3p_meta.json` | O bloco D |
| `rodada2/pretreino_vocabg_meta.json`, `pretreino_vocabg_historico.csv`, `vocab_grande_meta.json` | O pré-treino e o vocabulário do bloco B |
| `rodada2/fila_cortes.txt` | O que saiu da fila, e por quê |
| `rodada2/codigo/etapa13b.py` | O código da rodada |

Os modelos e as respostas estão em `data/execucoes/13_pretreino_piloto/rodada2/`.
