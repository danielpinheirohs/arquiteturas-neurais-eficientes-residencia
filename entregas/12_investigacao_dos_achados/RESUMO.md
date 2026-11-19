# Etapa 12 · Investigação dos achados: resumo da entrega

**Pergunta.** Por que acontecem os três efeitos que as etapas mediram e não explicaram, e o que cada um ensina para o artigo?

**Natureza.** É uma investigação **depois dos resultados**: os efeitos já tinham sido vistos, e os testes foram escolhidos depois deles. Tudo aqui é diagnóstico e exploratório. Nenhuma decisão das etapas anteriores muda, e os modelos finais continuam os mesmos (o Aluno A, em cada caminho). O plano, com as hipóteses, os testes, o que contaria como confirmação e a leitura no teste de cada achado, foi gravado em `plano.json` antes de qualquer execução. Diferenças pareadas da média de 3 sementes, com intervalo de 95% por bootstrap de enunciados (2.000 sorteios), em pontos percentuais; tudo em `dev`, salvo a leitura única no teste.

**Resposta curta.**

- **Achado 1 (a poda do XLM-R perde nos alunos B): explicado em parte.** A perda vem na maior parte do **reajuste**, não do corte: o mesmo reajuste de 10 épocas, sem cortar nada, perde −0,7 (−1,4 a −0,1) (B1) e −0,5 (−1,2 a +0,2) (B2) de exact match em dev. O recomeço da taxa derruba o aluno na 1ª época (−2,2 e −2,1 pontos contra os humanos), e as épocas seguintes o levam de volta **para perto do professor**, não dos humanos: no treino, o aluno B1 passa a repetir o rótulo do professor em 98,8% dos pedidos em que o professor erra (eram 83,4%). Como a época é escolhida pelo dev rotulado pelo professor, ganha a que mais concorda com ele. Com a taxa 10 vezes menor o reajuste deixa de perder (corte e reajuste − só o corte: −0,0 (−0,4 a +0,4) e −0,0 (−0,5 a +0,4)). O corte em si pesa só nos 257 pedidos de dev (12,6%) com subtoken fora das 5.196 linhas. **Em aberto:** a diferença entre o Aluno A e os alunos B não é distinguível (efeito do reajuste no B1 menos no A: −0,4 (−1,3 a +0,4)); o Aluno A também encaixa os erros do professor no treino (84,2% → 94,7%). "Passa com o A e não passa com o B" separa dois intervalos que se sobrepõem. **No teste** (a leitura única do plano), o reajuste sem corte também perde: B1 · Qwen3-14B −0,6 (−1,2 a −0,0); B2 · Qwen3-14B −1,2 (−1,7 a −0,7).
- **Achado 3 (a INT8 dinâmica tira cerca de 0,4 ponto de intenção): explicado.** A queda vem das matrizes (−0,38 (−0,72 a −0,02)), não da tabela (−0,07 (−0,15 a +0,00)). É ruído de arredondamento que vira decisões quase empatadas: 86% das 225 trocas de intenção (3 sementes) estão entre os 10% de pedidos de menor margem. As trocas vão 78 vezes de certo para errado e 53 de errado para certo (teste de sinal, p = 0,036), mas essa proporção (0,60) é a que uma troca ao acaso entre as duas primeiras intenções daria (0,62): não há viés além do ruído. O efeito é geral: a diferença de intenção em dev é negativa em 9 dos 9 modelos da 10c. Nenhuma alternativa simples resolve: a cabeça de intenção em 32 bits dá exatamente o mesmo (−0,41 (−0,79 a −0,03)), um degrau por canal piora (−0,62 (−1,02 a −0,23)), a tabela com um degrau por linha fica em −0,23 (−0,64 a +0,16). Nada foi ao teste.
- **Achado 2 (a INT8 estática desaba numa semente): explicado em parte.** A perda vem toda de um grupo de tensores: a entrada da 2ª camada da rede densa (a saída da GELU). Só esse grupo em 8 bits já derruba o teto na semente 1 para 29,0 de F1 (a estática inteira dá 41,9); os outros grupos, sozinhos, mudam no máximo 0,4 ponto. Esse tensor tem valores extremos em poucas dimensões, as mesmas em todas as sementes e modelos (herança do XLM-R pré-treinado), e com a faixa do menor ao maior valor 99% dos valores caem em poucos degraus. Nenhuma camada sozinha derruba o modelo; a perda **se acumula** (camadas 0 a 7 em 8 bits: 74,4; 0 a 9: 29,2, na semente 1). Deixar **um** tensor em 32 bits devolve 70% da perda no teto (camada 8) e 79% no Aluno A (camada 9); deixar os 12 devolve tudo. A semente 1 do teto tem, na camada 8, ativações 3 vezes mais largas que as outras (maior valor 78,9 e percentil 99 de 3,31, contra 26,8 e 0,98 na semente 0). **Em aberto:** a semente 1 do Aluno A desaba sem ter faixas maiores, e a calibração por percentil a conserta (65,6 · 66,9 · 62,8 de F1 nas sementes 0 · 1 · 2) sem consertar a do teto (73,8 · 46,9 · 72,8). **Recomendação:** deixar em 32 bits as 12 multiplicações da saída da rede densa; no teste (a leitura única do plano), essa forma fica a −0,9 (−1,4 a −0,4) no teto; −0,1 (−0,5 a +0,3) no Aluno A de F1 do modelo de 32 bits.

## O plano

`plano.json` (sha256 `577821a434da9a1a…`) foi gravado às 19/11 07:58:48, antes da primeira execução (19/11 08:02:17) e logo depois das impressões digitais das outras etapas (2026-11-19 07:57:40 -0300). Para cada achado: as hipóteses, os testes, o critério de confirmação e de refutação, e a única leitura no teste. Os desvios estão na seção própria.

## Achado 1 · A poda do XLM-R perde cerca de 1 ponto nos alunos do cenário B

**O que se sabia.** A receita "vocabulário do treino" da 10b corta a tabela de subtokens para as 5.196 linhas do texto de treino e reajusta 10 épocas com a receita e o sinal de quem treinou o modelo. Passa com o Aluno A e o teto; perde com os alunos B (B2 antigo −1,0 no teste; B1 e B2 novos −1,0 e −0,9 em dev na 11c).

**Hipóteses.** (a) treino a mais sobre os rótulos do professor aberto: o aluno copia os erros dele; (b) o recomeço da taxa de aprendizado; (c) o corte em si, pelos subtokens de dev e teste fora das 5.196 linhas.

**O que foi testado** (21 reajustes novos, 3 sementes cada; exact match em dev, contra o modelo de partida sem poda):

| Teste | B1 · Qwen3-14B | B2 · Qwen3-14B | Aluno A |
|---|---|---|---|
| Corte e reajuste (a receita, como na 11c e na 10b) | −1,0 (−1,8 a −0,3) | −0,9 (−1,7 a −0,2) | −0,4 (−1,0 a +0,3) |
| T1 · reajuste sem corte (controle) | −0,7 (−1,4 a −0,1) | −0,5 (−1,2 a +0,2) | −0,3 (−0,9 a +0,3) (já na 10b) |
| Corte e reajuste − reajuste sem corte | −0,3 (−0,7 a +0,0) | −0,4 (−0,8 a −0,1) | — |
| Só o corte, sem reajuste | −0,6 (−0,9 a −0,2) | −0,4 (−0,7 a +0,0) | — |
| T4 · corte e reajuste com taxa 10× menor | −0,6 (−1,1 a −0,1) | −0,4 (−0,9 a +0,2) | — |
| T4 − só o corte | −0,0 (−0,4 a +0,4) | −0,0 (−0,5 a +0,4) | — |
| Efeito do reajuste sem corte, B − A (diferença das diferenças) | −0,4 (−1,3 a +0,4) | −0,2 (−1,0 a +0,6) | — |

O teto, com o mesmo reajuste sem corte: +0,0 (−0,4 a +0,4).

- **O corte e o reajuste se separam exatamente.** O texto do treino só usa as 5.196 linhas que ficam; as outras não recebem gradiente. Por isso, com a mesma semente, o reajuste com corte e sem corte dá respostas idênticas em todo o treino (sim) e em todos os pedidos de dev sem subtoken cortado (sim, nas 6 execuções). Toda a diferença entre os dois mora nos 257 pedidos afetados de dev; no teste são 376 de 2.974 (12,6%, contados só pelo texto).
- **T2, a curva.** O reajuste refeito repete o das etapas de origem resposta por resposta (2.033 de 2.033 em cada semente, nos três alunos). Na 1ª época, com a taxa de volta ao pico, o exact match contra os humanos cai −2,2 (B1), −2,1 (B2) e −1,2 (Aluno A). Na 10ª época, contra o professor o aluno termina acima de onde começou (0,43, 0,33 e 0,33) e contra os humanos, abaixo (−0,48, −0,97 e −0,74). A época é escolhida pelo dev do professor. Com a taxa 10 vezes menor não há queda na 1ª época (−0,07 no B1).
- **T3, a cópia do professor.** Nos pedidos de dev em que o professor erra, o aluno passa a repetir a resposta dele com mais frequência (B1 35,6% → 36,6%; B2 36,3% → 36,9%; Aluno A 41,2% → 41,4%) e a acertar menos onde o professor erra (B1 16,3% → 14,8%; B2 16,9% → 15,0%; Aluno A 18,2% → 17,1%). No treino, o acerto contra o rótulo do professor sobe muito mais onde o professor erra do que onde acerta (B1: 83,4% → 98,8% nos pedidos "ruidosos", 98,4% → 99,9% nos "limpos"; Aluno A: 84,2% → 94,7% e 97,8% → 99,3%). O professor aberto discorda do humano (exact match) em 51,8% dos pedidos de treino; o fechado, em 46,4%.
- **T5, o corte.** Nos pedidos afetados, só o corte muda o exact match em −4,4 (−7,1 a −1,7) (B1) e −2,9 (−5,7 a +0,0) (B2); nos outros, em exatamente zero.

**Conclusão: explicado em parte.** (a) **confirmada como mecanismo**: o reajuste encaixa os erros do professor (memorização do ruído, como em Arpit et al. e Zhang et al.), e concordar mais com o professor não melhora o acerto contra os humanos (Stanton et al.). (b) **refutada como causa única** (o teto passa pelo mesmo recomeço sem perder: +0,0 (−0,4 a +0,4)) e **confirmada como gatilho**: o recomeço tira o aluno do ponto de partida, e sem ele (taxa 10× menor) o reajuste não perde nada além do que o corte já perdeu. O reaquecimento da taxa aumentar a perda antes de melhorar é o que Gupta et al. relatam no pré-treino contínuo. (c) **confirmada só para a parte do corte**, que mora inteira nos pedidos com subtoken cortado; em dev ela é menor ou parecida com a do reajuste (−0,3 (−0,7 a +0,0) e −0,4 (−0,8 a −0,1)), e a conta exploratória feita antes desta etapa sobre as respostas gravadas da 11c (não refeita aqui, para não abrir o teste de novo) dava, no teste, +0,1 e −0,1 para só o corte. Os cortes de vocabulário de Abdaoui et al. e Ushio et al. são por língua, não pelo texto do treino, e por isso não deixam pedidos sem linha.

**O que fica em aberto.** (1) Por que o Aluno A "passa": a diferença entre o efeito no B e no A não é distinguível (−0,4 (−1,3 a +0,4) para o B1; −0,2 (−1,0 a +0,6) para o B2), e o A mostra o mesmo mecanismo, um pouco menor. A explicação "rótulos menos ruidosos" é coerente com os números (o fechado discorda menos do humano e o A perde menos) mas não está provada. (2) Por que só o corte custa em dev e não no teste, com a mesma fração de pedidos afetados: não foi investigado (o teste não foi aberto para isso).

**Recomendação (para trabalho futuro; nada foi trocado).** Depois do corte do vocabulário, reajustar com taxa bem menor (5e-6) ou não reajustar (em dev, a taxa menor não perde nada além do corte; isso não foi ao teste): o corte não mexe nos pedidos cobertos pelo treino, e um reajuste com a taxa cheia, num aluno de professor ruidoso, reaprende os erros do professor. Para o artigo: "passa/não passa" com intervalos que se sobrepõem não sustenta dizer que a receita funciona com um aluno e não com outro.

## Achado 3 · A INT8 dinâmica tira cerca de 0,4 ponto de acurácia de intenção do modelo final do XLM-R

**O que se sabia.** Aluno A, vocabulário do treino, INT8 dinâmica − 32 bits: −0,41 (−0,79 a −0,03) de intenção em dev (−0,44 no teste, na 10c). A regra da 10c olhava só as entidades.

**Hipóteses e testes.** h1, a parte da rede (as formas `so_tabela` e `so_matrizes` gravadas na 10c); h2, as trocas se concentram em margem pequena (a pontuação de intenção do ONNX de 32 bits, refeito, em dev); h3, as trocas vão mais para o lado certo → errado; h4, o efeito aparece nos 9 modelos; T4, alternativas.

| Forma (Aluno A, vocabulário do treino) | Intenção − 32 bits, dev | Entidades − 32 bits, dev | MB da rede |
|---|---|---|---|
| INT8 dinâmica (a da 10c, refeita: 6099 de 6099 respostas iguais) | −0,41 (−0,79 a −0,03) | +0,05 (−0,39 a +0,50) | 90,2 |
| Só a tabela em 8 bits | −0,07 (−0,15 a +0,00) | −0,03 (−0,11 a +0,04) | — |
| Só as matrizes em 8 bits (a tabela em 32 bits) | −0,38 (−0,72 a −0,02) | +0,00 (−0,44 a +0,46) | — |
| Dinâmica com a cabeça de intenção em 32 bits | −0,41 (−0,79 a −0,03) | +0,05 (−0,39 a +0,50) | 90,3 |
| Dinâmica com um degrau por canal nas matrizes | −0,62 (−1,02 a −0,23) | −0,14 (−0,59 a +0,30) | 90,6 |
| Dinâmica com a tabela de subtokens com um degrau por linha | −0,23 (−0,64 a +0,16) | −0,16 (−0,62 a +0,31) | 90,2 |

O ONNX de 32 bits do mesmo modelo tem 358,4 MB de rede.

- **h1 confirmada:** a queda vem das matrizes; a tabela sozinha não move a intenção de forma distinguível.
- **h2 confirmada:** a margem mediana dos pedidos que trocam de intenção é 0,29, contra 0,998 em todos os pedidos; 86% das trocas estão abaixo do 10º percentil da margem (0,72).
- **h3, pelo critério do plano, confirmada (p = 0,036), mas sem viés:** a fração certo → errado (0,60) é a esperada se cada troca escolhesse ao acaso entre as duas primeiras intenções do modelo de 32 bits (0,62: nos pedidos de margem baixa, a 1ª intenção está certa mais vezes que a 2ª). As trocas são ruído; elas tiram mais acerto do que dão porque o modelo de 32 bits acerta mais do que erra nos empates.
- **h4 confirmada:** negativa em 9 dos 9 modelos em dev (média −0,32); no teste, a 10c já registrava 7 de 9 negativas (média −0,13).
- **Nenhuma alternativa resolve** pelo critério do plano (diferença com intervalo contendo o zero e ponto ≥ −0,15). A cabeça de intenção em 32 bits muda zero resposta: as trocas nascem antes da cabeça. O degrau por canal piora. A tabela por linha fica com o intervalo contendo o zero, mas o ponto (−0,23) não atinge o critério.

**Conclusão: explicado.** A INT8 dinâmica arredonda as matrizes; o erro, pequeno, só vira a decisão nos pedidos quase empatados, e aí tira um pouco mais de acerto do que dá. É o fenômeno das "trocas" de Dutta et al.: acurácias parecidas escondem respostas que mudam nos dois sentidos. Os −0,4 do modelo final estão dentro da faixa dos 9 modelos (−0,49 a −0,07): o modelo final não tem nada de especial; foi o intervalo dele que não tocou o zero.

**O que fica em aberto.** Que camadas concentram o erro que vira a intenção (não testado); por que o degrau por canal piora (não investigado).

**Recomendação.** A regra de quantização deveria olhar também a intenção (ou o exact match) e contar as trocas, além da média. Nenhuma das alternativas baratas testadas compra a intenção de volta.

## Achado 2 · A INT8 estática desaba numa das três sementes

**O que se sabia.** Na 10c a INT8 estática (MinMax, 512 pedidos do treino, um degrau por tensor) não passa em nenhum dos 9 modelos, e a semente 1 desaba. Conferido nas respostas gravadas: o 41,9 é do teto sem poda (F1 em 32 bits 77,4 / 77,0 / 77,4, por semente) e o 47,4 é do Aluno A **vocabulário do treino** (32 bits: 67,6 / 67,6 / 67,9); a variante sem poda do A dá 47,8. A queda também aparece no B2 antigo (30,7 na semente 1).

**Hipóteses e testes.** h1, valores extremos estouram a faixa (T1: faixa de cada entrada de multiplicação, 512 pedidos de calibração); h2, poucos tensores (T2: um grupo por vez em 8 bits; T4: tensores em 32 bits); h3, outra calibração (percentil 99,999 e entropia); h4, a poda gradual diminui os extremos (descritivo).

- **T2, por tipo** (teto sem poda, só um grupo em 8 bits, queda de F1 em dev): a entrada da saída da rede densa −48,0 na semente 1 e −3,6 na semente 0; todos os outros grupos (consulta, chave e valor; saída da atenção; entrada da rede densa; cabeças; tabelas) entre −0,4 e +0,4.
- **T2, por camada:** uma camada sozinha nunca custa mais que 0,9 ponto (camada 8, semente 1). **Acumulado** (desvio do plano): camadas 0 a 5 em 8 bits, 76,7; 0 a 7, 74,4; 0 a 9, 29,2 (semente 1). Na semente 0: 77,3, 74,7, 73,8.
- **T4, um tensor em 32 bits** (desvio do plano; semente 1): a estática inteira com só a saída da rede densa da camada 8 em 32 bits vai de 41,9 para 66,5 no teto (70% da perda volta, com 1 de 74 multiplicações); no Aluno A, a camada 9 leva de 47,4 para 63,4 (79%).
- **T1, as faixas.** As dimensões com valores extremos se repetem em todos os modelos e sementes (por exemplo a 754 na camada 7, a 1194 e a 3025 na 6), como Kovaleva et al. descrevem para dimensões que vêm do pré-treino. No maior caso, o maior valor é cerca de 60 e o percentil 99 fica perto de 1: com 256 degraus do menor ao maior valor, 99% dos valores ficam em uns 4 degraus. Na semente 1 do teto, a camada 8 tem a distribuição inteira 3 vezes mais larga (percentil 99 de 3,31 contra 0,98 e 1,08); na semente 1 do Aluno A, não (1,08); quem tem a camada 8 larga no A é a semente 2 (3,39), que também perde mais que a 0.

| Forma estática (F1 de entidades − 32 bits, dev) | Teto, sem poda | Aluno A, vocabulário do treino | F1 por semente (0 · 1 · 2) | MB da rede (A) |
|---|---|---|---|---|
| a estática da 10c, refeita (MinMax, todas as multiplicações e tabelas) | −14,9 (−16,3 a −13,6) | −9,8 (−11,1 a −8,4) | teto 73,2 · 41,9 · 71,8; A 66,1 · 47,4 · 60,3 | 90,2 |
| calibração por percentil (99,999) | −12,8 (−14,0 a −11,5) | −2,6 (−3,6 a −1,5) | teto 73,8 · 46,9 · 72,8; A 65,6 · 66,9 · 62,8 | 90,2 |
| calibração por entropia | −13,4 (−14,7 a −12,1) | −2,9 (−3,9 a −1,8) | teto 72,3 · 47,4 · 72,0; A 65,4 · 66,2 · 62,9 | 90,2 |
| MinMax, com as multiplicações de maior razão máximo/percentil em 32 bits (k pelo maior salto, sem olhar dev) | −6,8 (−7,9 a −5,7) | −3,9 (−5,1 a −2,8) | teto 74,6 · 61,7 · 75,0; A 66,3 · 62,6 · 62,3 | 97,3 |
| MinMax, com as 12 multiplicações da saída da rede densa em 32 bits | −0,4 (−0,9 a +0,2) | −0,5 (−1,0 a +0,0) | teto 77,3 · 76,8 · 76,7; A 67,5 · 67,0 · 67,2 | 175,1 |

O ONNX de 32 bits do Aluno A, vocabulário do treino, tem 358,4 MB de rede; a INT8 dinâmica, 90,2.

- **h1 confirmada em parte:** os extremos existem e estão no grupo que perde, mas a semente que desaba nem sempre é a de faixa maior (Aluno A, semente 1).
- **h2 confirmada:** um tensor em 32 bits (1,4% das multiplicações) devolve mais de 2/3 da perda na semente 1 dos dois modelos; os 12 da saída da rede densa (16%) devolvem quase tudo. A regra do plano para escolher os piores só pela faixa (o maior salto da razão máximo/percentil) devolve menos (−6,8 (−7,9 a −5,7) no teto): a razão máximo/percentil aponta camadas que, sozinhas, pesam pouco.
- **h3 confirmada no Aluno A e refutada no teto**, pelo critério do plano (perda abaixo de 1 ponto, ou reduzida à metade ou menos): no Aluno A o percentil leva a perda de −9,8 (−11,1 a −8,4) para −2,6 (−3,6 a −1,5) (a entropia, para −2,9 (−3,9 a −1,8)); no teto, de −14,9 (−16,3 a −13,6) para −12,8 (−14,0 a −11,5). Nenhuma calibração deixa a perda abaixo de 1 ponto.
- **h4 não sustentada:** a faixa do Aluno A com poda gradual de 70% é parecida com a sem poda (maior valor por semente: 58,5, 58,3, 61,9 contra 63,7, 63,9, 66,2); por que a poda gradual sofre menos fica em aberto.

**Conclusão: explicado em parte.** A estática falha porque um tipo de ativação, a saída da GELU que entra na 2ª camada da rede densa, tem valores extremos em poucas dimensões (Bondarenko et al. apontam extremos estruturados como o desafio próprio da quantização de Transformers; Dettmers et al., que separar as poucas dimensões extremas em mais bits resolve), e o erro desse tensor, pequeno em cada camada, se acumula pela pilha até derrubar o modelo. Por que a semente 1 desaba mais fica explicado só no teto, onde ela coincide com uma camada 8 três vezes mais larga; que cada semente fique a uma distância diferente desse ponto de queda é uma hipótese, não testada, e no Aluno A não há diferença de faixa que a sustente.

**O que fica em aberto.** O que torna a semente 1 do Aluno A frágil (as faixas não mostram; a calibração por percentil conserta); por que a poda gradual sofre menos; e se um degrau por grupo de dimensões (o que Bondarenko et al. propõem) resolveria sem deixar nada em 32 bits (não testado: o ONNX Runtime não tem essa forma pronta).

**Recomendação.** Se a estática for necessária, deixar as 12 multiplicações da saída da rede densa em 32 bits; custa 84,9 MB a mais no Aluno A (175,1 contra 90,2 MB). Para o artigo: medir a INT8 estática em 3 sementes, porque uma só pode dar um número qualquer entre "quase sem perda" e "desaba".

## A leitura no teste

Uma por achado, como o plano fixou.

- **Achado 1.** A taxa menor passaria a seguir a regra da 10b em dev nos dois alunos? B1 · Qwen3-14B: −0,6 (−1,1 a −0,1), não passa; B2 · Qwen3-14B: −0,4 (−0,9 a +0,2), passa. Como não passa nos dois, leu-se no teste o reajuste sem corte − modelo de partida: B1 · Qwen3-14B −0,6 (−1,2 a −0,0); B2 · Qwen3-14B −1,2 (−1,7 a −0,7) de exact match.
- **Achado 3.** Nenhuma alternativa resolveu em dev; nada foi ao teste.
- **Achado 2.** A alternativa de menor perda média em dev foi `sem_a_saida_da_rede_densa`; no teste, F1 de entidades − 32 bits: teto sem poda −0,9 (−1,4 a −0,4); Aluno A, vocabulário do treino −0,1 (−0,5 a +0,3).

## Os artigos

Conferidos na API do arXiv (título, primeiro autor e data); só o resumo foi lido.

| Artigo | Sustenta? | O que o resumo diz, e o que não sustenta |
|---|---|---|
| Arpit et al., [arXiv 1706.05394](https://arxiv.org/abs/1706.05394) | em parte | o resumo diz que redes conseguem memorizar ruído, mas aprendem primeiro os padrões simples, e que os dados determinam quanto se memoriza. Sustenta a leitura de que treino a mais sobre rótulos com erro passa a encaixar os erros; não fala de destilação, de encoders pré-treinados nem de reajuste depois de poda |
| Zhang et al., [arXiv 1611.03530](https://arxiv.org/abs/1611.03530) | em parte | o resumo diz que redes treinadas com descida de gradiente encaixam facilmente rótulos sorteados. Sustenta que o aluno tem capacidade de copiar os erros do professor; é sobre redes convolucionais de imagens |
| Stanton et al., [arXiv 2106.05945](https://arxiv.org/abs/2106.05945) | sim | o resumo diz que concordar mais com o professor, paradoxalmente, nem sempre leva a generalizar melhor. É o padrão medido aqui: o reajuste sobe a concordância com o professor em dev e desce o acerto contra os humanos |
| Mosbach, Andriushchenko & Klakow, [arXiv 2006.04884](https://arxiv.org/abs/2006.04884) | em parte | o resumo diz que o ajuste fino de BERT varia muito com a semente e que modelos com a mesma perda de treino generalizam diferente. Sustenta tratar a semente como fonte de variação; não trata de quantização nem de reajuste |
| Gupta et al., [arXiv 2308.04014](https://arxiv.org/abs/2308.04014) | em parte | o resumo diz que reaquecer a taxa de aprendizado primeiro aumenta a perda e só depois melhora. Sustenta a queda da 1ª época do reajuste; é pré-treino contínuo de um modelo de linguagem, não ajuste fino |
| Abdaoui, Pradel & Sigel, [arXiv 2010.05609](https://arxiv.org/abs/2010.05609) | em parte | o resumo diz que versões menores do BERT multilíngue, com o vocabulário das línguas-alvo, mantêm resultados comparáveis. O corte de lá é por língua; o daqui, pelo texto do treino, é mais agressivo e deixa pedidos de dev e do teste com subtokens sem linha |
| Ushio, Zhou & Camacho-Collados, [arXiv 2305.15020](https://arxiv.org/abs/2305.15020) | em parte | o resumo diz que cortar o vocabulário para a língua-alvo mantém o desempenho, e que cerca de 50% do vocabulário costuma bastar. Mesma ressalva: o corte daqui fica com 5.196 linhas (cerca de 2% de 250.002) |
| Bondarenko, Nagel & Blankevoort, [arXiv 2109.12948](https://arxiv.org/abs/2109.12948) | em parte | o resumo diz que Transformers têm ativações de faixa dinâmica alta, difíceis de representar em ponto fixo de poucos bits, com extremos estruturados nas conexões residuais. Sustenta o mecanismo geral; aqui o tensor que desaba é a entrada da 2ª camada da rede densa (saída da GELU), que o resumo não cita |
| Kovaleva et al., [arXiv 2105.06990](https://arxiv.org/abs/2105.06990) | em parte | o resumo diz que as dimensões extremas surgem cedo no pré-treino e aparecem sempre na mesma posição. Sustenta que as dimensões extremas daqui se repetem entre sementes e modelos (vêm do XLM-R pré-treinado); lá elas estão nos parâmetros da normalização, aqui na ativação da rede densa |
| Dettmers et al., [arXiv 2208.07339](https://arxiv.org/abs/2208.07339) | em parte | o resumo diz que Transformers têm atributos extremos sistemáticos, em poucas dimensões, e que isolar essas dimensões numa multiplicação de 16 bits mantém a qualidade com mais de 99,9% dos valores em 8 bits. Sustenta a ideia de deixar em mais bits o pouco que tem extremos; o artigo é sobre modelos de 6,7 bilhões de parâmetros para cima, e a decomposição é por dimensão, não por tensor como aqui |
| Wu et al., [arXiv 2004.09602](https://arxiv.org/abs/2004.09602) | em parte | o resumo diz que um fluxo de 8 bits mantém a acurácia a 1% do ponto flutuante em todas as redes estudadas, BERT-large incluído. Sustenta que a escolha da calibração importa; o resumo não diz qual calibração serve ao BERT |
| Jacob et al., [arXiv 1712.05877](https://arxiv.org/abs/1712.05877) | não lido | — |
| Dutta et al., [arXiv 2407.09141](https://arxiv.org/abs/2407.09141) | sim | o resumo diz que, com acurácia parecida, modelos comprimidos trocam respostas de certas para erradas e o contrário, em proporção. É o que T2 e T3 mediram |

## Desvios do plano

- **Achado 2, modelo.** O 47,4 citado é do Aluno A "vocabulário do treino", não do "sem poda"; usou-se essa variante, como o plano previa ao conferir.
- **Achado 2, testes acrescentados depois de ver resultados:** a sensibilidade por camada também na semente 0 do teto e na 1 do Aluno A; o acumulado (camadas 0 a k); um tensor em 32 bits por vez; e a forma com as 12 multiplicações da saída da rede densa em 32 bits. Todos em dev; a leitura no teste continuou a fixada (a alternativa de menor perda média em dev).
- **Achado 2, calibração por histograma.** A calibração por percentil e por entropia do ONNX Runtime 1.32 falha com pedidos de comprimentos diferentes (ela junta as ativações de todos os pedidos num só arranjo). Ela foi chamada um pedido por vez, somando os histogramas, sem mudar mais nada.
- **Achado 2, execução.** Dois processos de quantização na mesma rede de entrada colidiram (a ferramenta grava e apaga uma cópia ao lado do arquivo de entrada); as duas execuções que falharam foram apagadas e refeitas, e desde então cada quantização lê a rede por um link físico na sua pasta. Nenhuma medida concluída foi afetada: as que terminaram repetem a 10c onde podem ser comparadas.
- **Achado 1.** Nenhum desvio: os 21 reajustes e as contas são os do plano. O Aluno A não teve controle novo sem corte porque a 10b já o tinha.

## Critério de pronto

| Critério | Atendido |
|---|---|
| O plano foi gravado antes de qualquer execução nova, e não mudou depois | sim |
| As 21 execuções do achado 1 estão gravadas, cada uma com as respostas em dev de cada época | sim |
| O reajuste refeito repete, resposta por resposta, o das Etapas 10b e 11c | sim |
| O ONNX de 32 bits exportado de novo repete as respostas da 10c em dev | sim |
| A INT8 dinâmica e a estática refeitas repetem as da 10c, resposta por resposta | sim |
| Cada hipótese do plano tem o seu teste e a sua leitura | sim |
| O teste foi lido uma vez por achado, como o plano fixou (achado 3: nada ao teste, porque nenhuma alternativa resolveu em dev) | sim |
| Cada artigo citado foi conferido na API do arXiv | sim |
| Nada das outras etapas mudou nem sumiu (impressões digitais antes e depois); os arquivos novos fora desta etapa estão listados | sim |

## Limites

- Tudo é exploratório e depois dos resultados; os testes foram escolhidos sabendo dos efeitos. Os intervalos não são corrigidos para o número de comparações.
- Três sementes. Os achados 2 e 3 são sobre o XLM-R base com o ONNX Runtime 1.32 em CPU; a rede do zero não foi investigada.
- Só os resumos dos artigos foram lidos.
- Durante a etapa apareceram pastas que não são dela, `data/execucoes/13_pretreino_piloto/`, `data/execucoes/sintese/` (11 arquivos novos), escritas por outros processos (outro trabalho rodou na máquina ao mesmo tempo); nenhum arquivo das outras etapas mudou nem sumiu. Esta etapa não mede latência, então a máquina dividida não muda nenhum número dela.
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
