# Tabela dos modelos e valores

Tudo no **teste** do MASSIVE pt-PT (2.974 pedidos), contra rótulos humanos; modelos treinados são média de 3 sementes. Os números vêm de `entregas/11c_cenario_b_ate_o_fim/comparativo/tabela_final.csv`; a memória, de `data/execucoes/sintese/memoria.json`. A leitura completa está em [SINTESE.md](SINTESE.md).

## 1. Todos os sistemas prontos para uso

Ordenado por exact match (intenção e todas as entidades certas no mesmo pedido).

- **Arquitetura:** *Minha rede* é a rede desenhada e treinada do zero neste projeto (estilo DIET, 4,9 M de parâmetros, sem peso pré-treinado); *Checkpoint do XLM-R* é o XLM-R base pré-treinado (277,6 M), ajustado nos mesmos dados; *LLM fechada* e *LLM aberta* são os professores; *Saco de palavras* é a linha de base sem rede neural.
- **Rótulos humanos:** quantos rótulos feitos por gente o sistema usou. **Rótulos do treino:** quem rotulou os pedidos em que o sistema treinou.
- **Memória para rodar:** pico de RAM do processo na CPU, ou memória de placa de vídeo, nas LLMs.
- As linhas **Teto** são referência, não concorrentes: a mesma rede treinada com os 11.514 rótulos humanos.

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

## 2. Minha rede com pré-treino (Etapa 13, exploratório)

A mesma rede, antes pré-treinada só com a perda de máscara em 8,2 milhões de frases da Wikipédia em português. Modelos em PyTorch de 32 bits, sem poda nem quantização. A diferença é pareada contra o modelo atual sem pré-treino (teto 62,5; aluno A 51,2), com intervalo de 95%; em negrito quando não contém zero.

| Sistema | Variante | Rótulos do treino | Parâmetros | Intenção | Entidades (F1) | Exact match | Exact match − modelo atual | Tamanho |
|---|---|---|---|---|---|---|---|---|
| Teto | Pré-treino de 1 passada, vocabulário do MASSIVE | humanos, 11.514 pedidos | 4,9 M | 83,7 | 74,8 | **62,6** | +0,1 (−0,7 a +1,0) | 19,9 MB |
| Aluno A (Claude Haiku 5.5) | Pré-treino de 1 passada, vocabulário do MASSIVE | Claude Haiku 5.5, 11.514 pedidos | 4,9 M | 80,6 | 65,4 | **51,6** | +0,4 (−0,3 a +1,2) | 19,9 MB |
| Teto | Pré-treino de 3 passadas, vocabulário do MASSIVE | humanos, 11.514 pedidos | 4,9 M | 84,3 | 74,8 | **63,0** | +0,5 (−0,4 a +1,3) | 19,9 MB |
| Aluno A (Claude Haiku 5.5) | Pré-treino de 3 passadas, vocabulário do MASSIVE | Claude Haiku 5.5, 11.514 pedidos | 4,9 M | 81,1 | 64,6 | **51,1** | −0,1 (−0,9 a +0,8) | 19,9 MB |
| Teto | Vocabulário grande, pré-treino de 0,89 passada | humanos, 11.514 pedidos | 15,1 M | 83,8 | 74,6 | **63,1** | +0,7 (−0,1 a +1,5) | cerca de 61 MB |
| Aluno A (Claude Haiku 5.5) | Vocabulário grande, pré-treino de 0,89 passada | Claude Haiku 5.5, 11.514 pedidos | 15,1 M | 80,1 | 65,2 | **50,9** | −0,3 (−1,1 a +0,5) | cerca de 61 MB |
| Teto | O mesmo, com taxa menor nas partes pré-treinadas | humanos, 11.514 pedidos | 15,1 M | 83,3 | 75,3 | **62,6** | +0,1 (−0,9 a +1,1) | cerca de 61 MB |

Nenhuma variante supera o modelo atual em exact match. Detalhes em [entregas/13_pretreino_piloto/RESUMO.md](entregas/13_pretreino_piloto/RESUMO.md) e [RESUMO_rodada2.md](entregas/13_pretreino_piloto/RESUMO_rodada2.md).
