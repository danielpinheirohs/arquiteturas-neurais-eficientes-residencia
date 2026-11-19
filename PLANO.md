# Plano ponta a ponta: de uma LLM a um especialista eficiente

*16/11/2026. Complementa "Especialista eficiente análise e plano revisados (1).md", que continua sendo a fonte detalhada das evidências.*

## Premissa

**Não existe um artigo único para replicar.** O mais próximo é Ni et al., "From Large to Super-Tiny" ([arXiv 2504.13471](https://arxiv.org/abs/2504.13471)): LLM professor → destilação (rótulos e logits) → poda estruturada → quantização, numa tarefa que nasce como intenção + slots. Ele serve de **esqueleto do workflow**, mas não de receita, porque o aluno é gerativo (Qwen2.5-0.5B), os dados são proprietários e não há código.

Logo: o workflow segue a forma de Ni et al. e a motivação de Belcak et al. ([arXiv 2506.02153](https://arxiv.org/abs/2506.02153)), e **cada etapa é ancorada no artigo que propôs a técnica**. A contribuição é integrativa e empírica: medir quanto cada etapa acrescenta, contra a LLM, num encoder conjunto pequeno.

O preprint que faltava conferir ([arXiv 2611.00464](https://arxiv.org/abs/2611.00464)) foi lido pelo resumo: estuda interação entre quantização e poda em modelos gerativos de 2,7 a 7B, medida por perplexidade. Não ocupa o seu espaço; vira referência para a discussão de ordem das etapas.

## Perguntas que o trabalho responde

1. Um encoder conjunto pequeno, destilado de uma LLM, substitui a LLM em intenção + entidades?
2. Quanto vale ter os logits do professor (LLM aberta) em vez de só os rótulos (LLM fechada)?
3. Depois da arquitetura e da destilação, quanto poda, fine-tuning e quantização ainda acrescentam em qualidade, bytes e milissegundos?

## Etapas e artigo-âncora de cada uma

| # | Etapa | O que fazer | Artigo-âncora | Saída |
|---|---|---|---|---|
| 0 | Dados e avaliação | MASSIVE pt-PT em BIO; métricas: acurácia de intenção, F1 de slots, exact match; teste sempre nos rótulos humanos | FitzGerald et al. (MASSIVE, [2204.08582](https://arxiv.org/abs/2204.08582)) | Arnês de avaliação único |
| 1 | Arquitetura do zero | Transformer de 2 camadas construído e treinado do zero, com cabeça de intenção e cabeça de entidades e perda conjunta (detalhes abaixo) | Bunk et al. (DIET, [2004.09936](https://arxiv.org/abs/2004.09936)) | Aluno "teto" treinado com rótulos humanos |
| 2 | Incumbentes | LLM fechada (API) e LLM aberta (RTX 3080, 4 bits) resolvem a tarefa com o mesmo prompt; as duas rotulam o treino | Belcak et al.; Ni et al. | Duas linhas de LLM na tabela + rótulos de treino |
| 3a | Destilação caixa-preta | Aluno A aprende com rótulos rígidos da LLM fechada | Xu et al. (survey, [2402.13116](https://arxiv.org/abs/2402.13116)); Di Palo et al. ([2411.05045](https://arxiv.org/abs/2411.05045)) | Aluno A |
| 3b | Destilação caixa-branca | Aluno B1: rótulos da LLM aberta. Aluno B2: logits de intenção (distribuição do próximo token restrita aos 60 rótulos, com temperatura) | Hinton et al. ([1503.02531](https://arxiv.org/abs/1503.02531)); Jiao et al. (TinyBERT, [1909.10351](https://arxiv.org/abs/1909.10351)) | Alunos B1 e B2 |
| 4 | Poda | Magnitude gradual em 30/50/70/90% e uma contraparte estruturada (descartar camadas) | Han et al. ([1506.02626](https://arxiv.org/abs/1506.02626)); Zhu & Gupta ([1710.01878](https://arxiv.org/abs/1710.01878)); Xia et al. (CoFi, [2204.00408](https://arxiv.org/abs/2204.00408)) | Curva acurácia × esparsidade |
| 5 | Fine-tuning de recuperação | Reajustar o modelo podado com a máscara travada, mantendo a perda do professor | Han et al.; Zafrir et al. (Prune Once for All, [2111.05754](https://arxiv.org/abs/2111.05754)); Sanh et al. ([2005.07683](https://arxiv.org/abs/2005.07683)) | Antes × depois do ajuste |
| 6 | Quantização | INT8 pós-treino (ONNX Runtime); QAT só se o F1 de slots cair mais de 1 ponto | Jacob et al. ([1712.05877](https://arxiv.org/abs/1712.05877)); Zafrir et al. (Q8BERT, [1910.06188](https://arxiv.org/abs/1910.06188)) | Modelo final INT8 |
| 7 | Comparativo | Uma tabela e um gráfico de Pareto com todos os sistemas | Dehghani et al. ([2110.12894](https://arxiv.org/abs/2110.12894)); Reddi et al. (MLPerf, [1911.02549](https://arxiv.org/abs/1911.02549)) | Resultado do artigo |
| 8 | Encoder pré-treinado | Escolher um encoder de alta adoção e repetir as etapas 3 a 6 com ele, reaproveitando os rótulos e logits das LLMs | Chen et al. (JointBERT, [1902.10909](https://arxiv.org/abs/1902.10909)) | Comparação do zero × pré-treinado |

A ordem destilar → podar → quantizar se apoia em Han et al. (Deep Compression, [1510.00149](https://arxiv.org/abs/1510.00149)) e Harma et al. ([2405.20935](https://arxiv.org/abs/2405.20935)).

## Quando fazer fine-tuning

**Um fine-tuning obrigatório (depois da poda) e um condicional (QAT, na quantização). Nenhum depois da destilação.**

| Momento | Fine-tuning? | Por quê | Âncora |
|---|---|---|---|
| Depois da destilação | Não | A destilação já é o treino do aluno. A distância para o professor é o resultado desse treino, não um dano a consertar. Reajustar com rótulos humanos apagaria a diferença entre os alunos A, B1, B2 e o teto, que é justamente o que o experimento mede | Hinton et al. (rótulo e soft target entram na mesma perda, num treino só) |
| Durante e depois da poda | Sim, sempre | A poda remove pesos de um modelo já treinado; o reajuste deixa os pesos restantes compensarem. Na poda gradual o treino corre junto com a poda, e fecha com algumas épocas de máscara travada | Han et al. (treinar → podar → retreinar); Zhu & Gupta; Prune Once for All |
| Depois da quantização | Só se precisar | INT8 pós-treino não tem reajuste, por definição. Se o F1 de slots cair mais de 1 ponto, troca-se por QAT, que é um fine-tuning com a quantização simulada dentro do treino | Jacob et al.; Q8BERT |

Regras que valem para todo fine-tuning do plano:

- **Mesmo sinal da destilação.** O reajuste do aluno A usa os rótulos da LLM fechada; o do B2 usa os logits da LLM aberta. Rótulos humanos nunca entram, senão a linha muda duas coisas de uma vez.
- **Antes e depois, sempre reportados.** Cada nível de poda tem a linha "podado sem reajuste" e a linha "podado com reajuste". É isso que testa a hipótese "a poda derruba e o fine-tuning recupera".
- **Orçamento fixo.** Mesmo número de épocas e mesma taxa de aprendizado em todos os níveis de esparsidade, declarados no relatório.

Expectativa do documento de análise, a conferir: em esparsidade baixa (30 a 40%) pode não haver queda nenhuma para recuperar; o reajuste importa de 50% para cima.

## A tabela final

Colunas: acurácia de intenção, F1 de slots, exact match, parâmetros, tamanho em disco, latência p50/p95 (batch 1), custo por 1M de requisições.

| Linha | Mostra |
|---|---|
| LLM fechada | Incumbente do cenário 1 |
| LLM aberta | Incumbente do cenário 2 |
| Aluno teto (rótulos humanos) | Limite da arquitetura |
| Aluno A | Destilação só com rótulos |
| Aluno B1 / B2 | Qualidade do professor / ganho dos logits |
| Melhor aluno + poda | Efeito da poda sem reajuste |
| + fine-tuning | Quanto o reajuste recupera |
| + INT8 | Modelo final |

Uma mudança por linha, 3 seeds por modelo treinado. O bloco dos alunos aparece duas vezes: uma para a arquitetura do zero (núcleo) e outra para o encoder pré-treinado (etapa 8).

## Ordem de execução

1. **Fundação:** dados congelados, métricas, arnês das duas LLMs, piloto de 200 enunciados.
2. **Arquitetura:** aluno teto com rótulos humanos. Se ele não chegar perto da linha publicada do MASSIVE, nada adiante vale.
3. **Professores:** avaliar as duas LLMs no teste e rotular o treino (a aberta guarda os logits).
4. **Destilação:** alunos A, B1, B2.
5. **Compressão:** poda → fine-tuning → INT8, no melhor aluno.
6. **Medição:** tabela e Pareto da arquitetura do zero.
7. **Encoder pré-treinado:** escolher o modelo, repetir os passos 4 e 5 com ele, comparar.
8. **Escrita:** ameaças à validade e texto final.

Cada passo só começa com o anterior medido. Early exit, cascata por confiança e esparsidade contextual ficam fora do núcleo, como extensões.

## As duas arquiteturas

As duas rodam o mesmo pipeline, com os mesmos dados, professores e seeds.

**1. Do zero, no estilo DIET (núcleo do trabalho).** Tudo construído e treinado por você, sem pesos pré-treinados:

- Entrada: features esparsas por token (palavra e n-gramas de caracteres) projetadas para a dimensão do modelo.
- Corpo: Transformer de 2 camadas, dimensão 256, como no artigo.
- Cabeça de intenção sobre o token de sentença; cabeça de entidades por token, com CRF.
- Perda conjunta: intenção + entidades (+ a perda de máscara do DIET, como ablação).

O ponto de partida são os hiperparâmetros do artigo; qualquer desvio é declarado. Variação opcional, também prevista no DIET: somar embeddings pré-treinados congelados às features esparsas.

**2. Encoder pré-treinado de alta adoção (etapa final).** A escolha fica para depois, com calma. Critérios combinados: adoção ampla na comunidade, suporte a português, e tamanho que ainda faça sentido como "aluno". Candidato natural a avaliar: XLM-R base, que tem linha publicada exatamente no teste pt-PT do MASSIVE (86,7 / 73,3 / 64,6).

Uma observação: o MASSIVE é multilíngue (51 idiomas), mas o experimento usa só o recorte pt-PT. Um encoder multilíngue é uma opção válida, não uma exigência do dataset.

## O que foi e não foi conferido

- Lidos nesta sessão: resumo do 2611.00464, texto do 2504.13471 (parcial, até o apêndice), resumo do 2407.09943.
- As demais referências vêm do documento de análise e não foram relidas aqui.
- A busca por um artigo único foi curta, não uma revisão sistemática. Antes de submeter, vale uma busca formal.
