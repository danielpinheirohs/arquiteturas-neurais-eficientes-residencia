# Síntese do estudo: de uma LLM a um especialista eficiente

Gerado em 19/11/2026 a partir de `entregas/11c_cenario_b_ate_o_fim/comparativo/` (63 sistemas, 86 comparações pareadas), que é o comparativo mais completo do projeto e refaz, conferindo, os números de todas as etapas anteriores (420 de 420 leituras fecham). Tudo abaixo é no **teste** do MASSIVE pt-PT (2.974 pedidos), contra rótulos humanos, média de 3 sementes nos modelos treinados. Diferenças são pareadas com intervalo de 95%; em **negrito** quando o intervalo não contém zero.

Métricas: **intenção** = acurácia; **entidades** = F1; **exact match** = intenção e todas as entidades certas no mesmo pedido (a métrica principal).

## 1. O que foi comparado

| Eixo | Variantes |
|---|---|
| Cenário (de onde vêm os rótulos do aluno) | **A**: LLM fechada, só texto (Claude Haiku 5.5) · **B**: LLM aberta, com logits (Qwen3-8B em 4 bits; depois Qwen3-14B em float16) |
| Sinal de destilação | **B1**: só os rótulos do professor · **B2**: rótulos + logits de intenção com temperatura |
| Arquitetura do aluno | **Minha rede**: desenhada e treinada do zero neste projeto, no estilo DIET (features de palavra e de caracteres, Transformer de 2 blocos, cabeça de intenção e CRF de entidades), 4,9 M de parâmetros, sem peso pré-treinado · **Checkpoint do XLM-R**: o XLM-R base pré-treinado, 277,6 M, com duas cabeças novas, ajustado (fine-tuning) nos mesmos dados |
| Compressão | aluno destilado → poda com reajuste → ONNX de 32 bits → INT8 (quantização dinâmica) |
| Referências | linhas de base sem rede neural, **piso** (só os 299 rótulos humanos) e **teto** (os 11.514 rótulos humanos) |

Todo sistema fora do teto usa **299 rótulos humanos**: são os exemplos que a LLM vê no prompt. O resto do treino (11.514 pedidos) é rotulado pelo professor.

## 2. Referências: linhas de base, pisos, tetos e professores

| Sistema | Rótulos humanos | Intenção | Entidades (F1) | Exact match (IC 95%) | Tamanho (MB) | Latência p50 (ms) |
|---|---|---|---|---|---|---|
| Linha de base, 299 rótulos | 299 | 47,8 | 35,6 | **19,4** (18,1 a 20,8) | – | – |
| Linha de base, 11.514 rótulos | 11.514 | 84,6 | 47,8 | **45,2** (43,5 a 47,0) | 33,0 | 0,82 |
| Piso · minha rede | 299 | 40,1 | 40,8 | **20,9** (19,6 a 22,2) | 8,73 | 1,12 |
| Piso · XLM-R | 299 | 54,7 | 52,8 | **32,6** (31,1 a 34,2) | 1.127,5 | – |
| **Teto · minha rede** | 11.514 | 84,4 | 73,6 | **62,5** (60,8 a 64,1) | 19,9 | 1,10 |
| **Teto · XLM-R** | 11.514 | 86,9 | 75,5 | **66,3** (64,7 a 67,9) | 1.127,5 | 37,1 |
| Professor fechado · Claude Haiku 5.5 | 299 | 83,5 | 66,1 | **54,0** (52,3 a 55,7) | – | 854 (API) |
| Professor aberto · Qwen3-8B, 4 bits | 299 | 77,6 | 55,7 | **43,2** (41,5 a 45,0) | 16.397,4 | 1.087 (GPU local) |
| Professor aberto · Qwen3-8B, float16 | 299 | 77,9 | 56,5 | **43,4** (41,6 a 45,2) | – | – |
| Professor aberto · Qwen3-14B, float16 | 299 | 80,0 | 61,4 | **49,0** (46,9 a 50,7) | – | – |
| Professor aberto · Qwen3-32B, float16 | 299 | 80,9 | 62,5 | **50,0** (48,2 a 51,8) | – | – |

- Entre os professores: Qwen3-8B em 4 bits − Claude = **−10,8 (−12,4 a −9,1)**; Qwen3-14B − Claude = **−5,1 (−6,7 a −3,5)**; Qwen3-32B − Claude = **−4,0 (−5,6 a −2,5)**.
- Tirar os 4 bits do Qwen3-8B: +0,2 (−0,8 a +1,2). Trocar o 8B pelo 14B: **+5,7 (+4,2 a +7,3)**. Do 14B para o 32B: +1,0 (−0,3 a +2,5) no teste e −0,0 (−1,6 a +1,7) em dev (Etapa 11a), sem diferença distinguível.
- Teto do XLM-R − teto da minha rede: **+3,8 (+2,7 a +5,0)**, com 57 vezes o tamanho.

## 3. Destilação: cada aluno ao lado do seu professor

Alunos como saem do treino, sem compressão (PyTorch, 32 bits).

| Arquitetura | Aluno | Professor | Sinal | Intenção | Entidades (F1) | Exact match (IC 95%) | Exact match do professor | Aluno − professor |
|---|---|---|---|---|---|---|---|---|
| Minha rede | A | Claude Haiku 5.5 | rótulos | 80,7 | 64,8 | **51,2** (49,6 a 52,9) | 54,0 | **−2,8 (−4,3 a −1,5)** |
| Minha rede | B1 | Qwen3-8B, 4 bits | rótulos | 76,3 | 57,1 | **43,6** (41,9 a 45,2) | 43,2 | +0,4 (−1,0 a +1,7) |
| Minha rede | B2 | Qwen3-8B, 4 bits | rótulos + logits (T = 8) | 77,8 | 58,0 | **46,0** (44,2 a 47,6) | 43,2 | **+2,7 (+1,2 a +4,3)** |
| Minha rede | B1 | Qwen3-14B | rótulos | 77,9 | 61,4 | **47,7** (45,9 a 49,3) | 49,0 | −1,3 (−2,7 a +0,2) |
| Minha rede | B2 | Qwen3-14B | rótulos + logits (T = 8) | 78,6 | 60,4 | **48,8** (46,9 a 50,4) | 49,0 | −0,2 (−1,5 a +1,2) |
| Checkpoint do XLM-R | A | Claude Haiku 5.5 | rótulos | 82,6 | 65,5 | **53,8** (52,1 a 55,4) | 54,0 | −0,2 (−1,6 a +1,0) |
| Checkpoint do XLM-R | B1 | Qwen3-8B, 4 bits | rótulos | 78,7 | 58,5 | **45,7** (44,0 a 47,4) | 43,2 | **+2,5 (+1,1 a +3,8)** |
| Checkpoint do XLM-R | B2 | Qwen3-8B, 4 bits | rótulos + logits (T = 2) | 78,9 | 58,8 | **46,3** (44,6 a 48,0) | 43,2 | **+3,0 (+1,7 a +4,3)** |
| Checkpoint do XLM-R | B1 | Qwen3-14B | rótulos | 79,8 | 61,7 | **49,7** (48,0 a 51,4) | 49,0 | +0,8 (−0,4 a +2,0) |
| Checkpoint do XLM-R | B2 | Qwen3-14B | rótulos + logits (T = 8) | 80,6 | 61,8 | **50,5** (48,9 a 52,2) | 49,0 | **+1,6 (+0,3 a +2,9)** |

O mesmo aluno, trocando só uma coisa (exact match):

| O que muda | Minha rede | Checkpoint do XLM-R |
|---|---|---|
| Professor: Claude no lugar do Qwen3-8B (A − B1) | **+7,6 (+6,3 a +8,8)** | **+8,1 (+6,8 a +9,4)** |
| Professor: Claude no lugar do Qwen3-14B (A − B1) | **+3,5 (+2,3 a +4,8)** | **+4,1 (+2,9 a +5,3)** |
| Professor: Qwen3-14B no lugar do 8B, aluno B1 | **+4,0 (+2,9 a +5,2)** | **+4,0 (+2,9 a +5,2)** |
| Professor: Qwen3-14B no lugar do 8B, aluno B2 | **+2,8 (+1,6 a +3,9)** | **+4,3 (+3,1 a +5,5)** |
| Arquitetura: XLM-R no lugar da minha rede, aluno A | **+2,6 (+1,6 a +3,7)** | |
| Arquitetura: XLM-R no lugar da minha rede, B2 do Qwen3-14B | **+1,8 (+0,7 a +2,8)** | |
| Logits (B2 − B1, só os logits mudam), em **acurácia de intenção**, Qwen3-8B | **+2,3 (+1,4 a +3,1)** | +0,2 (−0,3 a +0,8) |
| Logits (B2 − B1), em **acurácia de intenção**, Qwen3-14B | **+1,4 (+0,6 a +2,1)** | **+0,8 (+0,2 a +1,4)** |

As duas linhas de logits vêm das Etapas 6, 10a e 11b (comparação contra o controle sem suavização de rótulos).

## 4. Compressão: cada cadeia, do aluno destilado ao INT8

| Arquitetura | Cenário | Destilado | Podado e reajustado | INT8 | INT8 − destilado | Tamanho (MB) | Latência p50 (ms) |
|---|---|---|---|---|---|---|---|
| Minha rede | A · Claude Haiku 5.5 | 51,2 | 51,3 | 51,2 | +0,0 (−0,8 a +0,8) | 19,9 → 8,06 → 2,11 | 1,11 → 1,10 → 0,26 |
| Minha rede | B2 · Qwen3-8B | 46,0 | 46,2 | 46,2 | +0,2 (−0,7 a +1,2) | 19,9 → 8,06 → 2,11 | 1,11 → 1,10 → 0,26 |
| Minha rede | B2 · Qwen3-14B | 48,8 | 48,6 | 48,7 | −0,1 (−1,0 a +0,8) | 19,9 → 8,06 → 2,11 | 1,10 → 1,07 → 0,25 |
| Minha rede | Teto (rótulos humanos) | 62,5 | 61,3 | 61,3 | **−1,2 (−2,2 a −0,3)** | 19,9 → 8,06 → 2,11 | 1,10 → 1,10 → 0,26 |
| Checkpoint do XLM-R | A · Claude Haiku 5.5 | 53,8 | 53,7 | 53,5 | −0,3 (−0,9 a +0,3) | 1.127,5 → 359,0 → 90,7 | 35,8 → 36,1 → 8,35 |
| Checkpoint do XLM-R | B2 · Qwen3-8B | 46,3 | 45,2 | 45,2 | **−1,1 (−1,6 a −0,5)** | 1.127,5 → 359,0 → 90,7 | 37,6 → 36,8 → 7,94 |
| Checkpoint do XLM-R | B1 · Qwen3-14B | 49,7 | 49,0 | 49,3 | −0,5 (−1,1 a +0,2) | 1.127,5 → 359,0 → 90,7 | 34,4 → 34,4 → 7,69 |
| Checkpoint do XLM-R | B2 · Qwen3-14B | 50,5 | 49,1 | 49,2 | **−1,3 (−2,0 a −0,7)** | 1.127,5 → 359,0 → 90,7 | 36,8 → 34,5 → 7,65 |
| Checkpoint do XLM-R | Teto (rótulos humanos) | 66,3 | 65,8 | 65,7 | **−0,6 (−1,3 a −0,0)** | 1.127,5 → 359,0 → 90,7 | 37,1 → 37,5 → 8,50 |

Colunas de tamanho e latência: destilado (PyTorch) → podado (PyTorch) → INT8 (ONNX Runtime). Latência de uma consulta, 1 thread de CPU, i7-13700K, semente 0. A poda é a principal de cada caminho: tirar 90% das linhas da tabela de features (minha rede) e cortar a tabela de subtokens para o vocabulário do treino (XLM-R).

- Minha rede, aluno A: 9,4 vezes menor e 4,2 vezes mais rápido, sem perda distinguível.
- XLM-R, aluno A: 12,4 vezes menor e 4,3 vezes mais rápido, sem perda distinguível.
- O que cada passo compra, na minha rede: a poda de peças inteiras dá o tamanho (19,9 → 8,06 MB) e não muda a latência; sair do PyTorch para o ONNX dá a maior parte da latência (1,10 → 0,43 ms); os 8 bits dão mais tamanho (8,06 → 2,11 MB) e o resto da latência (0,43 → 0,26 ms).
- No XLM-R: o corte do vocabulário dá o tamanho (1.127,5 → 359,0 MB) e não muda a latência; o ONNX dá 1,44 vezes; os 8 bits dão 3 vezes na latência e 4 vezes no tamanho.

A poda vale a pena depois dos 8 bits? INT8 podado − INT8 sem poda (exact match):

| Arquitetura | Modelo | INT8 podado − INT8 sem poda | Tamanho (MB): sem poda → podado |
|---|---|---|---|
| Minha rede | A · Claude Haiku 5.5 | −0,1 (−0,9 a +0,8) | 5,26 → 2,11 |
| Minha rede | B2 · Qwen3-8B | +0,3 (−0,7 a +1,3) | 5,26 → 2,11 |
| Minha rede | B2 · Qwen3-14B | −0,2 (−1,1 a +0,7) | 5,26 → 2,11 |
| Minha rede | Teto (rótulos humanos) | **−1,2 (−2,1 a −0,3)** | 5,26 → 2,11 |
| Checkpoint do XLM-R | A · Claude Haiku 5.5 | **−1,0 (−1,6 a −0,5)** | 287,3 → 90,7 |
| Checkpoint do XLM-R | B2 · Qwen3-8B | **−1,4 (−1,9 a −1,0)** | 287,3 → 90,7 |
| Checkpoint do XLM-R | B1 · Qwen3-14B | −0,7 (−1,4 a +0,0) | 287,3 → 90,7 |
| Checkpoint do XLM-R | B2 · Qwen3-14B | **−1,4 (−2,0 a −0,8)** | 287,3 → 90,7 |
| Checkpoint do XLM-R | Teto (rótulos humanos) | −0,5 (−0,9 a +0,0) | 287,3 → 90,7 |

## 5. Quem performa melhor: todos os sistemas prontos para uso, lado a lado

Ordenado por exact match. Todos os alunos em INT8 (podado e sem poda), os professores, os tetos e as linhas de base, numa tabela só.

- **Arquitetura:** *Minha rede* é a rede desenhada e treinada do zero neste projeto (estilo DIET, 4,9 M de parâmetros, sem peso pré-treinado); *Checkpoint do XLM-R* é o XLM-R base pré-treinado (277,6 M), com duas cabeças novas, ajustado nos mesmos dados; *LLM fechada* e *LLM aberta* são os professores; *Saco de palavras* é a linha de base sem rede neural.
- **Rótulos humanos:** quantos rótulos feitos por gente o sistema usou. **Rótulos do treino:** quem rotulou os pedidos em que o sistema treinou.
- **Memória para rodar:** pico de RAM do processo que serve o modelo na CPU (Python e ONNX Runtime incluídos, cerca de 50 MB), ou a memória de placa de vídeo ocupada pelo modelo, nas LLMs.
- As linhas **Teto** são referência, não concorrentes: a mesma rede treinada com os 11.514 rótulos humanos em vez dos de uma LLM.

| # | Sistema | Arquitetura | Rótulos humanos | Rótulos do treino | Intenção | Entidades (F1) | Exact match (IC 95%) | Em disco (MB) | Memória para rodar | Onde roda | Latência p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Teto, INT8 sem poda | Checkpoint do XLM-R | 11.514 | humanos, 11.514 pedidos | 86,8 | 74,9 | **66,1** (64,5 a 67,7) | 287,3 | 660 MB de RAM | CPU, 1 thread | 8,39 |
| 2 | Teto, INT8 podado | Checkpoint do XLM-R | 11.514 | humanos, 11.514 pedidos | 86,3 | 74,6 | **65,7** (64,1 a 67,2) | 90,7 | 230 MB de RAM | CPU, 1 thread | 8,50 |
| 3 | Teto, INT8 sem poda | Minha rede | 11.514 | humanos, 11.514 pedidos | 84,3 | 73,7 | **62,5** (60,8 a 64,1) | 5,26 | 72 MB de RAM | CPU, 1 thread | 0,27 |
| 4 | Teto, INT8 podado | Minha rede | 11.514 | humanos, 11.514 pedidos | 83,4 | 73,5 | **61,3** (59,7 a 62,9) | 2,11 | 67 MB de RAM | CPU, 1 thread | 0,26 |
| 5 | Aluno A (Claude Haiku 5.5), INT8 sem poda | Checkpoint do XLM-R | 299 | Claude Haiku 5.5, 11.514 pedidos | 82,8 | 65,6 | **54,5** (52,8 a 56,1) | 287,3 | 660 MB de RAM | CPU, 1 thread | 8,33 |
| 6 | *Professor: Claude Haiku 5.5* | LLM fechada | 299 | não treina (299 exemplos no prompt) | 83,5 | 66,1 | **54,0** (52,3 a 55,7) | – | não se aplica (API) | API, pela rede | 854 (API) |
| 7 | **Aluno A (Claude Haiku 5.5), INT8 podado** (modelo final do caminho) | Checkpoint do XLM-R | 299 | Claude Haiku 5.5, 11.514 pedidos | 82,4 | 65,1 | **53,5** (51,8 a 55,1) | 90,7 | 230 MB de RAM | CPU, 1 thread | 8,35 |
| 8 | Aluno A (Claude Haiku 5.5), INT8 sem poda | Minha rede | 299 | Claude Haiku 5.5, 11.514 pedidos | 80,7 | 64,8 | **51,3** (49,6 a 52,9) | 5,26 | 72 MB de RAM | CPU, 1 thread | 0,26 |
| 9 | **Aluno A (Claude Haiku 5.5), INT8 podado** (modelo final do caminho) | Minha rede | 299 | Claude Haiku 5.5, 11.514 pedidos | 80,2 | 64,9 | **51,2** (49,5 a 52,9) | 2,11 | 67 MB de RAM | CPU, 1 thread | 0,26 |
| 10 | Aluno B2 (Qwen3-14B), INT8 sem poda | Checkpoint do XLM-R | 299 | Qwen3-14B, 11.514 pedidos | 80,2 | 61,6 | **50,6** (48,9 a 52,2) | 287,3 | 660 MB de RAM | CPU, 1 thread | 7,69 |
| 11 | *Professor: Qwen3-32B* | LLM aberta | 299 | não treina (299 exemplos no prompt) | 80,9 | 62,5 | **50,0** (48,2 a 51,8) | – | 65,5 GB de placa | GPU alugada (G4) | não medida |
| 12 | Aluno B1 (Qwen3-14B), INT8 sem poda | Checkpoint do XLM-R | 299 | Qwen3-14B, 11.514 pedidos | 79,7 | 61,9 | **50,0** (48,2 a 51,6) | 287,3 | 660 MB de RAM | CPU, 1 thread | 7,68 |
| 13 | Aluno B1 (Qwen3-14B), INT8 podado | Checkpoint do XLM-R | 299 | Qwen3-14B, 11.514 pedidos | 79,1 | 60,9 | **49,3** (47,5 a 50,9) | 90,7 | 229 MB de RAM | CPU, 1 thread | 7,69 |
| 14 | Aluno B2 (Qwen3-14B), INT8 podado | Checkpoint do XLM-R | 299 | Qwen3-14B, 11.514 pedidos | 79,7 | 60,3 | **49,2** (47,4 a 50,9) | 90,7 | 230 MB de RAM | CPU, 1 thread | 7,65 |
| 15 | *Professor: Qwen3-14B* | LLM aberta | 299 | não treina (299 exemplos no prompt) | 80,0 | 61,4 | **49,0** (46,9 a 50,7) | – | 29,5 GB de placa | GPU alugada (A100) | não medida |
| 16 | Aluno B2 (Qwen3-14B), INT8 sem poda | Minha rede | 299 | Qwen3-14B, 11.514 pedidos | 78,7 | 60,6 | **48,9** (47,1 a 50,5) | 5,26 | 72 MB de RAM | CPU, 1 thread | 0,26 |
| 17 | Aluno B2 (Qwen3-14B), INT8 podado | Minha rede | 299 | Qwen3-14B, 11.514 pedidos | 79,2 | 60,3 | **48,7** (46,9 a 50,4) | 2,11 | 66 MB de RAM | CPU, 1 thread | 0,25 |
| 18 | Aluno B2 (Qwen3-8B), INT8 sem poda | Checkpoint do XLM-R | 299 | Qwen3-8B, 4 bits, 11.514 pedidos | 79,0 | 58,5 | **46,6** (44,9 a 48,3) | 287,3 | 660 MB de RAM | CPU, 1 thread | 8,26 |
| 19 | Aluno B2 (Qwen3-8B), INT8 podado | Minha rede | 299 | Qwen3-8B, 4 bits, 11.514 pedidos | 77,8 | 58,0 | **46,2** (44,5 a 47,9) | 2,11 | 66 MB de RAM | CPU, 1 thread | 0,26 |
| 20 | Aluno B2 (Qwen3-8B), INT8 sem poda | Minha rede | 299 | Qwen3-8B, 4 bits, 11.514 pedidos | 77,9 | 58,1 | **45,9** (44,2 a 47,6) | 5,26 | 72 MB de RAM | CPU, 1 thread | 0,27 |
| 21 | *Linha de base, 11.514 rótulos* | Saco de palavras | 11.514 | humanos, 11.514 pedidos | 84,6 | 47,8 | **45,2** (43,5 a 47,0) | 33,0 | não medida | CPU | 0,82 |
| 22 | Aluno B2 (Qwen3-8B), INT8 podado | Checkpoint do XLM-R | 299 | Qwen3-8B, 4 bits, 11.514 pedidos | 78,2 | 57,6 | **45,2** (43,5 a 46,9) | 90,7 | 230 MB de RAM | CPU, 1 thread | 7,94 |
| 23 | *Professor: Qwen3-8B, 4 bits* | LLM aberta | 299 | não treina (299 exemplos no prompt) | 77,6 | 55,7 | **43,2** (41,5 a 45,0) | 16.397,4 | 6,1 GB de placa | GPU local (RTX 3080) | 1.087 (GPU) |
| 24 | *Linha de base, 299 rótulos* | Saco de palavras | 299 | humanos, 299 pedidos | 47,8 | 35,6 | **19,4** (18,1 a 20,8) | – | não medida | CPU | não medida |

**CPU de um lado, GPU do outro: como ler**

- **Não é uma diferença de protocolo, é o resultado.** Cada sistema está no hardware de que precisa para ser usado: as LLMs abertas não rodam em tempo útil sem placa de vídeo, e a fechada só existe por API. Os alunos dispensam a placa.
- **Em memória:** o modelo final da minha rede ocupa 67 MB de RAM e o do XLM-R, 230 MB; o Qwen3-8B em 4 bits ocupa 6,1 GB de placa, o Qwen3-14B, 29,5 GB, e o Qwen3-32B, 65,5 GB.
- **No XLM-R a poda do vocabulário pesa na memória:** 660 MB de RAM sem poda contra 230 MB podado. É um ganho da poda que as colunas de disco e de latência não mostravam.
- **A latência só é comparável entre os alunos** (mesma CPU, 1 thread, uma consulta por vez). A das LLMs é de outra condição e serve de ordem de grandeza: centenas de milissegundos contra menos de 10.
- **Os alunos não foram medidos em GPU.** Numa placa eles seriam mais rápidos em lote; o estudo mediu o caso mais barato de servir, que é o que interessa à pergunta.
- **A memória foi medida em 19/11**, um processo por modelo, sem PyTorch, respondendo a 256 pedidos de dev, semente 0 (`data/execucoes/sintese/memoria.json`). A dos professores é a registrada nas Etapas 5b e 11a.

**Fronteira de eficiência** (entre os sistemas com 299 rótulos humanos, quem não é dominado por outro que seja ao mesmo tempo melhor e mais barato): só alunos do cenário A.

| Custo | Sistema | Arquitetura | Custo | Exact match |
|---|---|---|---|---|
| Latência | Aluno A, INT8 sem poda | Minha rede | 0,26 ms | 51,3 |
| Latência | Aluno A, INT8 sem poda | Checkpoint do XLM-R | 8,33 ms | 54,5 |
| Tamanho | Aluno A, 95% dos pesos soltos em zero, INT8, compactado | Minha rede | 0,92 MB | 50,3 |
| Tamanho | Aluno A, INT8 podado | Minha rede | 2,11 MB | 51,2 |
| Tamanho | Aluno A, INT8 sem poda | Minha rede | 5,26 MB | 51,3 |
| Tamanho | Aluno A, INT8 podado | Checkpoint do XLM-R | 90,7 MB | 53,5 |
| Tamanho | Aluno A, INT8 sem poda | Checkpoint do XLM-R | 287,3 MB | 54,5 |

## 6. Os dois modelos finais contra as referências

| Comparação (exact match) | Minha rede · aluno A, INT8 podado | XLM-R · aluno A, INT8 podado |
|---|---|---|
| Exact match | **51,2** (49,5 a 52,9) | **53,5** (51,8 a 55,1) |
| − professor (Claude Haiku 5.5) | **−2,8 (−4,3 a −1,5)** | −0,5 (−1,9 a +0,7) |
| − teto do mesmo caminho (11.514 rótulos humanos) | **−11,3 (−12,8 a −9,8)** | **−12,8 (−14,2 a −11,4)** |
| − piso do mesmo caminho (só os 299 rótulos) | **+30,3 (+28,7 a +31,9)** | **+20,9 (+19,3 a +22,3)** |
| − linha de base com 299 rótulos | **+31,8 (+29,9 a +33,5)** | **+34,1 (+32,1 a +35,8)** |
| − linha de base com 11.514 rótulos | **+6,0 (+4,1 a +7,8)** | **+8,3 (+6,3 a +10,1)** |
| − melhor aluno do cenário B em INT8 (Qwen3-14B) | **+2,6 (+1,3 a +3,8)** | **+4,2 (+3,0 a +5,4)** |
| − mesmo aluno do cenário B com o Qwen3-8B (antes) | **+5,0 (+3,8 a +6,3)** | **+8,3 (+7,0 a +9,6)** |
| Tamanho | 2,11 MB | 90,7 MB |
| Latência p50, 1 thread de CPU | 0,26 ms | 8,35 ms |
| Latência do professor ÷ latência do aluno | 3.271 vezes | 102 vezes |
| Custo por mil consultas | zero por consulta (professor: US$ 0,114) | zero por consulta |

XLM-R final − minha rede final: **+2,3 (+1,3 a +3,3)** de exact match, por 42,9 vezes o tamanho e 32,0 vezes a latência (38,9 MB e 3,56 ms por ponto de exact match). Nenhum caminho domina o outro.

## 7. O que descobrimos

1. **O especialista chega perto da LLM e custa ordens de grandeza menos.** O modelo final da minha rede tem 2,11 MB, responde em 0,26 ms numa thread de CPU e fica **−2,8 (−4,3 a −1,5)** do Claude Haiku 5.5 em exact match; o do XLM-R, com 90,7 MB e 8,35 ms, fica −0,5 (−1,9 a +0,7): não se distingue do professor. A LLM responde em 854 ms pela API.
2. **A perda está na destilação, não na compressão.** Na minha rede, o degrau professor → aluno custa **−2,8 (−4,3 a −1,5)**; poda, ONNX e 8 bits somados custam +0,0 (−0,8 a +0,8). No cenário A, poda e quantização fizeram o que a teoria promete: tiraram custo sem tirar qualidade.
3. **A qualidade do professor é o que mais pesa no aluno.** Trocar o Qwen3-8B pelo Claude vale **+7,6 (+6,3 a +8,8)** na minha rede e **+8,1 (+6,8 a +9,4)** no XLM-R: mais que trocar a arquitetura do aluno (**+2,6 (+1,6 a +3,7)**) e mais que qualquer passo de compressão. Um professor aberto melhor (Qwen3-14B) fez os quatro alunos melhores, e entre 49% e 75% do ganho do professor chegou ao aluno.
4. **O cenário B perdia por duas razões, e só uma era o hardware.** A compressão em 4 bits do professor não pesou (+0,2, sem diferença). O tamanho pesou: o 14B, que não cabe nos 10 GB da placa local, sobe +5,7. Mas o 32B não se distingue do 14B, e o melhor professor aberto ainda fica 4 a 5 pontos abaixo do Claude. O cenário B fechou cerca de metade da distância para o A e continua fora da fronteira.
5. **O aluno passa um professor fraco, empata com um médio e não alcança um bom, na minha rede.** Contra o Qwen3-8B o aluno B2 fica **+2,7 (+1,2 a +4,3)**; contra o Qwen3-14B, −0,2 (−1,5 a +1,2); contra o Claude, o aluno A fica **−2,8 (−4,3 a −1,5)**. Com o XLM-R o aluno A empata com o Claude (−0,2 (−1,6 a +1,0)).
6. **Os logits ajudam a rede pequena e quase nada a rede pré-treinada.** Na minha rede, treinar com os logits vale +2,3 pontos de intenção com o Qwen3-8B e +1,4 com o 14B. No XLM-R, +0,2 (sem diferença) e +0,8. Não dá para afirmar que logits de um professor melhor valem mais. E um professor fechado bom, só com texto, ganha de um aberto com logits.
7. **A arquitetura desenhada para a tarefa rende muito por parâmetro.** Com rótulos humanos, a rede DIET do zero (4,9 M de parâmetros, 19,9 MB) faz 62,5 de exact match; o XLM-R base (277,6 M, 1.127,5 MB), 66,3. São 3,8 pontos por 57 vezes o tamanho. O pré-treino ajuda mais quando há poucos rótulos: no piso, com 299 rótulos, a vantagem do XLM-R é de 11,7 pontos.
8. **Poda só vira custo menor quando tira peças inteiras.** Zerar pesos soltos (até 95% na minha rede) não muda o arquivo comum nem a latência; só um formato esparso ou zip aproveita os zeros. Tirar linhas da tabela de entrada encolhe o arquivo, sem perda distinguível no aluno A (no teto e nos alunos B do XLM-R custa cerca de 1 ponto). Só tirar camadas acelera (XLM-R com 6 de 12 camadas: 19,2 ms contra 35,9), e isso perde qualidade.
9. **Depois dos 8 bits, a poda quase não compra latência.** Em latência, o INT8 sem poda domina o INT8 podado nos dois caminhos; a poda só compensa se o que importa é o tamanho do arquivo. No XLM-R ela ainda custa **−1,0 (−1,6 a −0,5)** no aluno A.
10. **Quantização dinâmica é quase de graça; a estática quebrou no XLM-R.** INT8 dinâmica: cerca de 4 vezes menos arquivo, queda de F1 de entidades abaixo de 1 ponto em todos os modelos, sem treino ciente de quantização. A estática não passou em nenhum dos 9 modelos do XLM-R (quedas de 1,7 a 14,9 pontos de F1).
11. **O que falta para o teto é rótulo, não modelo.** O aluno A fica **−11,3 (−12,8 a −9,8)** do teto da mesma rede. O professor concorda inteiramente com os rótulos humanos em só 53,6% do treino: o aluno aprende o professor, erros incluídos. Mesmo assim cobre 73% da distância entre o piso e o teto, e fica **+31,8 (+29,9 a +33,5)** acima do que os 299 rótulos sozinhos dão.

## 8. Limites e o que ficou em aberto

- **Nível absoluto.** O melhor sistema com 299 rótulos faz 54,5% de exact match; o professor, 54,0%; o teto, 66,3%. O resultado é "o especialista substitui esta LLM nesta tarefa", não "a tarefa está resolvida".
- **Intenção contra o saco de palavras.** A linha de base com os 11.514 rótulos humanos faz 84,6% de intenção, acima dos dois modelos finais (80,2% e 82,4%). Em exact match eles ganham (**+6,0 (+4,1 a +7,8)** e **+8,3 (+6,3 a +10,1)**), mas quem tem os rótulos humanos tem uma alternativa barata para a intenção.
- **Um conjunto, uma língua, uma família de professor aberto.** MASSIVE pt-PT, Qwen3, uma rotulagem de cada professor.
- **Latência de uma máquina e uma semente** (i7-13700K, 1 thread). A latência dos professores é de outra condição (API, GPU) e não entra na fronteira como igual.
- **O teste foi aberto muitas vezes** (63 sistemas, 86 comparações, sem correção para comparações múltiplas). As decisões foram tomadas em dev, mas os intervalos não descontam isso.
- **Regras decididas depois de ver resultado:** a extensão do Qwen3-32B.
- **Três efeitos medidos e não explicados**, com plano de investigação já gravado em `data/execucoes/12_investigacao_dos_achados/plano.json` (19/11, 07:58) e nada executado ainda: (1) a poda do vocabulário perde cerca de 1 ponto nos alunos do cenário B do XLM-R e não no aluno A; (2) a INT8 estática desaba no XLM-R, com uma semente muito pior que as outras; (3) a INT8 dinâmica tira 0,4 ponto de intenção do modelo final do XLM-R.

## 9. Onde está cada coisa

| Assunto | Arquivo |
|---|---|
| Tabela final com os 63 sistemas | `entregas/11c_cenario_b_ate_o_fim/comparativo/tabela_final.csv` |
| As 86 comparações pareadas, em dev e no teste | `entregas/11c_cenario_b_ate_o_fim/comparativo/comparacoes.csv` |
| Fronteira e preço de cada ponto | `entregas/10d_comparativo_dos_caminhos/RESUMO.md` |
| Modelo final da minha rede | `entregas/08_quantizacao/decisao.json` |
| Modelo final do XLM-R | `entregas/10c_encoder_quantizacao/decisao.json` |
| Resumo de cada etapa | `entregas/NN_nome/RESUMO.md` |
