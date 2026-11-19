# Etapa 13 · Piloto: pré-treinar a minha rede só com a máscara (exploratório)

**Pergunta.** Pré-treinar a rede do zero só com a perda de máscara, em texto de português sem rótulo, melhora o teto e o aluno A?

**Resposta.** **Neste piloto, não de forma que dê para afirmar.** Em exact match, o ganho é de +1,1 (+0,1 a +2,1) no teto em `dev`, mas no teste é de +0,1 (−0,7 a +1,0). No aluno A, +0,4 (−0,4 a +1,4) em `dev` e +0,4 (−0,3 a +1,2) no teste. O único efeito que aparece nas duas partições é no F1 de entidades do teto: +1,1 em `dev` e +1,1 no teste, com intervalo acima de zero nas duas.

É uma configuração só, feita numa manhã. O resultado diz que este pré-treino, com esta receita de ajuste, não rende; não diz que pré-treinar a rede não funciona.

## O que foi feito

- **Regras gravadas antes** (`regras.json`, 09:14) e uma emenda antes de qualquer resultado (`regras_emenda_1.json`): o corpus cresceu de 6 para 20 frases por artigo porque o pré-treino roda a 1 milhão de frases por minuto.
- **Corpus:** 8.237.823 frases da Wikipédia em português, de 4 a 25 palavras (14,3 em média), em minúsculas e sem pontuação, como os pedidos do MASSIVE.
- **Pré-treino:** a mesma rede (4,9 M de parâmetros), com o mesmo vocabulário de 25.202 features do MASSIVE, treinada só com a perda de máscara, uma passada, 32.179 passos, 7,6 minutos na RTX 3080. A perda de máscara foi de 5,74 para 3,18 (o acaso é 6,31).
- **Ajuste:** teto (11.514 rótulos humanos) e aluno A (rótulos do Claude Haiku 5.5) com a receita da Etapa 4b sem mudança, 3 sementes cada, partindo dos pesos do pré-treino. As cabeças de intenção e de entidades começam do zero.
- **Controle:** os modelos já entregues nas Etapas 4b e 6. Refeitos aqui das mesmas pastas, dão os números conhecidos (teto 62,5 e aluno A 51,2 de exact match no teste).

## Resultado

Diferença pareada pré-treinado − controle, média de 3 sementes, intervalo de 95%. Em negrito quando o intervalo não contém zero.

| Sistema | Partição | Métrica | Controle | Com pré-treino | Diferença (IC 95%) |
|---|---|---|---|---|---|
| Teto | dev | Intenção | 83,9 | 84,3 | +0,4 (−0,4 a +1,2) |
| Teto | dev | Entidades (F1) | 75,8 | 76,9 | **+1,1 (+0,2 a +2,1)** |
| Teto | dev | Exact match | 62,6 | 63,7 | **+1,1 (+0,1 a +2,1)** |
| Teto | teste | Intenção | 84,4 | 83,7 | **−0,7 (−1,4 a −0,0)** |
| Teto | teste | Entidades (F1) | 73,6 | 74,8 | **+1,1 (+0,4 a +1,9)** |
| Teto | teste | Exact match | 62,5 | 62,6 | +0,1 (−0,7 a +1,0) |
| Aluno A | dev | Intenção | 81,6 | 82,1 | +0,5 (−0,2 a +1,3) |
| Aluno A | dev | Entidades (F1) | 67,0 | 66,7 | −0,2 (−1,0 a +0,5) |
| Aluno A | dev | Exact match | 52,0 | 52,5 | +0,4 (−0,4 a +1,4) |
| Aluno A | teste | Intenção | 80,7 | 80,6 | −0,1 (−0,8 a +0,4) |
| Aluno A | teste | Entidades (F1) | 64,8 | 65,4 | +0,6 (−0,1 a +1,2) |
| Aluno A | teste | Exact match | 51,2 | 51,6 | +0,4 (−0,3 a +1,2) |

## Limites

- **Uma configuração, sem busca.** Uma passada, uma taxa, um corpus.
- **O ajuste usa a taxa cheia (1e-3) por 80 épocas**, a receita de quem parte do zero. É plausível que ela apague boa parte do que o pré-treino deixou; não foi testado.
- **O vocabulário de features é o do MASSIVE.** Palavra da Wikipédia que não está nos 11.514 pedidos entra só pelos n-gramas de caracteres: o pré-treino não ensina palavras novas à rede.
- **Registro diferente:** texto enciclopédico, majoritariamente do Brasil, contra comandos de assistente em português europeu.
- **A rede tem 2 blocos.** A capacidade de aproveitar muito texto é pequena.
- **Sem poda nem quantização** do modelo pré-treinado, e sem os alunos do cenário B.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `regras.json`, `regras_emenda_1.json` | O que foi fixado antes de rodar, com a hora |
| `comparacoes.csv` | Todas as métricas, em dev e no teste |
| `pretreino_meta.json`, `pretreino_historico.csv` | O pré-treino: passos, perda de máscara, tempo |
| `frases.json` | O corpus: quantas frases, de que arquivos |
| `codigo/etapa13.py` | O código do piloto |

Os modelos e as respostas estão em `data/execucoes/13_pretreino_piloto/`.
