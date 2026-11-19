# De uma LLM a um especialista eficiente

Projeto prático da residência em arquiteturas neurais eficientes. A pergunta do trabalho: um modelo pequeno, desenhado para a tarefa e destilado de uma LLM, substitui a LLM em classificação de intenção e extração de entidades? E quanto poda, fine-tuning e quantização ainda acrescentam depois disso?

O plano completo, com o artigo-âncora de cada técnica, está em [PLANO.md](PLANO.md).

## O que está neste repositório

- **Para ler:** a tabela de etapas, mais abaixo, leva ao `RESUMO.md` de cada entrega. O notebook executado de cada etapa está em HTML na pasta da entrega.
- **Para conferir os números:** as tabelas em CSV e as respostas de cada sistema em `dev` e no teste (`previsoes/`) estão nas pastas das entregas.
- **O que não está:** os pesos dos modelos e os tokenizadores (mais de 50 GB) e a pasta `data/` (downloads e caches de treino). Os comandos de cada etapa refazem tudo. A exceção são os sistemas da rede do zero em ONNX, da Etapa 8, que são pequenos e estão em `entregas/08_quantizacao/modelos/`.
- Os caminhos que começam com `/home/...` nos manifestos e nos notebooks são da máquina em que o projeto foi executado.

## Como o projeto anda

O trabalho é feito **uma etapa por vez**. Cada etapa:

1. começa com uma **pergunta**;
2. é contada num **notebook** em `notebooks/`, que mostra entrada, desenvolvimento e saída;
3. termina com uma **entrega** em `entregas/NN_nome/`, que é a resposta à pergunta;
4. só é considerada pronta quando o **critério de pronto** do notebook passa.

A configuração do modelo que as etapas seguintes usam está em [entregas/04b_intencao/decisao.json](entregas/04b_intencao/decisao.json). Os alunos destilados que seguem para a poda, e o sinal de treino de cada um, estão em [entregas/06_destilacao/decisao.json](entregas/06_destilacao/decisao.json). O modelo final do projeto, podado e quantizado, e onde ele está: [entregas/08_quantizacao/decisao.json](entregas/08_quantizacao/decisao.json). O caminho inteiro numa tabela, com o que cada etapa custou e rendeu: [entregas/09_comparativo/RESUMO.md](entregas/09_comparativo/RESUMO.md). Os dois caminhos (rede do zero e encoder pré-treinado) lado a lado, com a fronteira e o preço de cada ponto: [entregas/10d_comparativo_dos_caminhos/RESUMO.md](entregas/10d_comparativo_dos_caminhos/RESUMO.md). O mesmo comparativo com os sistemas da Etapa 11 (os professores abertos maiores, os alunos deles e as formas comprimidas desses alunos): [entregas/11c_cenario_b_ate_o_fim/RESUMO.md](entregas/11c_cenario_b_ate_o_fim/RESUMO.md).

Duas regras mantêm o projeto limpo:

- **O notebook conta, o módulo faz.** A lógica mora em `src/especialista/`; o notebook importa, executa e explica.
- **Uma etapa só lê as entregas das anteriores.** Nenhuma etapa volta ao dado bruto nem depende de variável de outro notebook.

## Como uma etapa é feita

Toda etapa segue o mesmo roteiro. É também o que se pede a quem for executar uma etapa, pessoa ou agente.

1. **Antes de medir:** a pergunta, o critério de pronto e toda regra de decisão ficam escritos antes de olhar o resultado. Decide-se em `dev`, reporta-se no teste. Toda comparação é pareada (`avaliacao/qualidade.py`: `comparar`, `comparar_grupos`), e uma diferença só vale quando o intervalo de 95% não contém zero. Modelo treinado tem 3 sementes.
2. **O que demora roda fora do notebook.** Treino, GPU e API ficam em comandos de `experimentos/etapaNN.py`, que gravam em `data/execucoes/NN_nome/` e não repetem o que já está gravado. O notebook só lê.
3. **A lógica mora em `src/especialista/`, com teste em `tests/`.** Peça nova que pode errar calada ganha teste.
4. **O notebook é gerado por um script** em `notebooks/construtores/build_nbNN.py` (os anteriores servem de modelo). Tem: cabeçalho com Pergunta, Entrada, Entrega e Pronto quando; artigos de apoio com link conferido; seções numeradas; textos de leitura montados a partir dos números calculados, nunca digitados; tabela "Critério de pronto" com `assert`; `RESUMO.md` gerado na última seção.
5. **Números e gráficos:** vírgula decimal e ponto de milhar pelos ajudantes `n`, `num`, `pts` e `dif` do notebook; gráficos pelas funções de `visual.py` (uma cor por série, no máximo duas séries). Toda figura é aberta e conferida depois de gerada.
6. **A entrega** vai para `entregas/NN_nome/`: `RESUMO.md`, `resultados.csv` (por `avaliacao/resultados.py`), `figuras/`, o notebook em HTML e os arquivos que a etapa seguinte lê.
7. **Fecho:** o notebook roda do início ao fim sem erro, os testes passam e a tabela de etapas deste README é atualizada.

Dois cuidados com a máquina. A GPU tem 10 GB e roda **uma tarefa por vez**: quando a memória da placa acaba, o driver não dá erro, passa a usar a RAM e tudo fica dezenas de vezes mais lento. E a chave da API, no arquivo `.env`, nunca é impressa nem copiada para lugar nenhum.

## Etapas

Roteiro proposto. Cada etapa é confirmada antes de começar.

| # | Etapa | Pergunta | Situação |
|---|---|---|---|
| 1 | [Dados](notebooks/01_dados.ipynb) | Em que cenário de teste vamos provar que um especialista pequeno substitui uma LLM? | Entregue: [resumo](entregas/01_dados/RESUMO.md) |
| 2 | [Avaliação](notebooks/02_avaliacao.ipynb) | Como saber se um sistema resolve a tarefa, e a que custo? | Entregue: [resumo](entregas/02_avaliacao/RESUMO.md) |
| 3 | [Arquitetura](notebooks/03_arquitetura.ipynb) | Que rede desenhar, do zero, para este problema? | Entregue: [resumo](entregas/03_arquitetura/RESUMO.md) |
| 4 | [Treino com rótulos humanos](notebooks/04_treino.ipynb) | Até onde essa arquitetura chega? | Entregue: [resumo](entregas/04_treino/RESUMO.md) |
| 4b | [Investigação da intenção](notebooks/04b_intencao.ipynb) | Por que a rede perde em intenção, e dá para recuperar? | Entregue: [resumo](entregas/04b_intencao/RESUMO.md) |
| 5a | [Professor fechado](notebooks/05a_professor_fechado.ipynb) | Quão bem uma LLM fechada resolve a tarefa, a que custo, e que rótulos ela deixa para o aluno? | Entregue: [resumo](entregas/05a_professor_fechado/RESUMO.md) |
| 5b | [Professor aberto](notebooks/05b_professor_aberto.ipynb) | E uma LLM aberta, rodando nesta máquina, que também entrega os logits? | Entregue: [resumo](entregas/05b_professor_aberto/RESUMO.md) |
| 6 | [Destilação](notebooks/06_destilacao.ipynb) | Quanto o aluno aprende só com os rótulos de cada professor, e quanto os logits acrescentam? | Entregue: [resumo](entregas/06_destilacao/RESUMO.md) |
| 7 | [Poda e fine-tuning](notebooks/07_poda.ipynb) | Quanto dá para remover, e quanto o reajuste recupera? | Entregue: [resumo](entregas/07_poda/RESUMO.md) |
| 8 | [Quantização](notebooks/08_quantizacao.ipynb) | Quanto custa, em qualidade, passar a rede para inteiros de 8 bits, e quanto se ganha em tamanho e latência? | Entregue: [resumo](entregas/08_quantizacao/RESUMO.md) |
| 9 | [Comparativo](notebooks/09_comparativo.ipynb) | Somando tudo: o que cada etapa custou e rendeu, e onde fica o modelo final em relação aos professores, ao teto e às linhas de base? | Entregue: [resumo](entregas/09_comparativo/RESUMO.md) |
| 10a | [Encoder pré-treinado: teto e alunos](notebooks/10a_encoder.ipynb) | Com um encoder pré-treinado no lugar da rede do zero, o que muda no teto e no que o aluno aprende de cada professor? | Entregue: [resumo](entregas/10a_encoder/RESUMO.md) |
| 10b | [Encoder pré-treinado: poda e reajuste](notebooks/10b_encoder_poda.ipynb) | Quanto dá para tirar do XLM-R, de que parte, e quanto o reajuste recupera? | Entregue: [resumo](entregas/10b_encoder_poda/RESUMO.md) |
| 10c | [Encoder pré-treinado: quantização](notebooks/10c_encoder_quantizacao.ipynb) | Quanto o XLM-R encolhe e acelera em inteiros de 8 bits, e a qualidade se conserva? | Entregue: [resumo](entregas/10c_encoder_quantizacao/RESUMO.md) |
| 10d | [Comparativo dos dois caminhos](notebooks/10d_comparativo_dos_caminhos.ipynb) | Com os mesmos professores e as mesmas regras, o que o encoder pré-treinado compra em qualidade, e quanto custa em tamanho e latência, em relação à rede feita do zero? | Entregue: [resumo](entregas/10d_comparativo_dos_caminhos/RESUMO.md) |
| 11a | [Professor aberto maior](notebooks/11a_professor_aberto_maior.ipynb) | O cenário B perde por ser aberto, ou porque o professor teve de caber em 10 GB? | Entregue: [resumo](entregas/11a_professor_aberto_maior/RESUMO.md) |
| 11b | [Alunos do professor aberto maior](notebooks/11b_alunos_do_professor_maior.ipynb) | Um professor aberto melhor faz alunos melhores? Quanto do ganho do professor chega ao aluno, e com ele os logits passam a valer mais que os rótulos? | Entregue: [resumo](entregas/11b_alunos_do_professor_maior/RESUMO.md) |
| 11c | [Cenário B com o professor novo até o fim, e comparativo atualizado](notebooks/11c_cenario_b_ate_o_fim.ipynb) | As receitas de poda e de quantização já escolhidas valem para os alunos do professor novo, e onde o cenário B fica na tabela final com ele? | Entregue: [resumo](entregas/11c_cenario_b_ate_o_fim/RESUMO.md) |
| 12 | [Investigação dos achados](notebooks/12_investigacao_dos_achados.ipynb) | Por que acontecem os três efeitos que as etapas mediram e não explicaram (a poda do XLM-R nos alunos B, a semente que desaba na INT8 estática, a intenção na INT8 dinâmica), e o que cada um ensina para o artigo? | Entregue: [resumo](entregas/12_investigacao_dos_achados/RESUMO.md) |

A Etapa 10 é o segundo caminho do plano: o mesmo pipeline com um encoder pré-treinado (XLM-R base) no lugar da rede do zero. Ela tem quatro partes. A **10a** refaz o teto e a destilação (o que as Etapas 4 e 6 fizeram no caminho do zero); a **10b** refaz a poda e o reajuste (Etapa 7); a **10c** refaz a quantização (Etapa 8); a **10d** compara os dois caminhos (Etapa 9). O que a 10b lê está em [entregas/10a_encoder/decisao.json](entregas/10a_encoder/decisao.json), o que a 10c lê, em [entregas/10b_encoder_poda/decisao.json](entregas/10b_encoder_poda/decisao.json), e o que a 10d lê, em [entregas/10c_encoder_quantizacao/decisao.json](entregas/10c_encoder_quantizacao/decisao.json). A resposta da 10d, em dados, está em [entregas/10d_comparativo_dos_caminhos/conclusao.json](entregas/10d_comparativo_dos_caminhos/conclusao.json).

A Etapa 11 volta ao professor aberto, agora com placas alugadas (Google Colab) no lugar da placa local de 10 GB. A **11a** continua a escada da Etapa 5b com a mesma regra: primeiro o mesmo Qwen3-8B sem os 4 bits, depois o Qwen3-14B. Cada professor é uma **versão** com pasta própria (`entregas/11a_professor_aberto_maior/professores/<versão>/`) e um `versao.json`; nenhuma versão substitui outra. Em 19/11 a 11a ganhou uma **extensão, decidida depois de ver a escada de três degraus**: o Qwen3-32B desafiou o Qwen3-14B com a mesma regra (`regra_extensao_32b.json`) e não tomou o lugar (`escolha_extensao_32b.json`); a entrega de 18/11 ficou guardada em `entregas/11a_professor_aberto_maior/versoes/2026-11-18_escada_de_tres_degraus/`. A **11b** treina de novo os alunos do cenário B (B1 e B2, nos dois caminhos) com o professor que a escada escolheu, o Qwen3-14B, com as receitas das Etapas 6 e 10a e sem nova busca: só o professor muda. Os alunos antigos ficam onde estavam; os novos têm pasta e nome de variante próprios, que dizem o professor (`zero__aluno_b1__qwen3_14b__fp16__colab_a100`). A temperatura do B2 em cada caminho, e onde está cada aluno novo, estão em [entregas/11b_alunos_do_professor_maior/decisao.json](entregas/11b_alunos_do_professor_maior/decisao.json). A **11c** leva esses alunos pelo resto do caminho com as receitas que as etapas de origem já escolheram (a poda principal das Etapas 7 e 10b com o reajuste dela, e a forma de 8 bits das Etapas 8 e 10c), **sem decisão nova**: as regras de origem são reaplicadas só como conferência, e a receita que não passa com um aluno novo é relatada, sem procurar outra. Ela também refaz o comparativo com os sistemas da Etapa 11; o da 10d não é tocado. Se cada receita passou, e onde estão os modelos podados e em 8 bits, está em [entregas/11c_cenario_b_ate_o_fim/decisao.json](entregas/11c_cenario_b_ate_o_fim/decisao.json); o comparativo, em `entregas/11c_cenario_b_ate_o_fim/comparativo/`.

A **Etapa 12** é uma investigação **depois dos resultados**: ela procura a causa de três efeitos que as Etapas 10b, 10c e 11c mediram e não explicaram (a poda do XLM-R que perde nos alunos do cenário B, a INT8 estática que desaba numa semente, a INT8 dinâmica que tira um pouco de intenção). O plano, com as hipóteses, os testes, os critérios e uma única leitura no teste por achado, foi gravado antes de qualquer execução (`entregas/12_investigacao_dos_achados/plano.json`). Ela não decide nada e não troca nenhum modelo: as execuções novas (reajustes de controle, quantizações de diagnóstico) ficam só nas pastas dela, e as conclusões, por achado, estão em [entregas/12_investigacao_dos_achados/RESUMO.md](entregas/12_investigacao_dos_achados/RESUMO.md).

## Pastas

```
PLANO.md              plano do trabalho e artigos-âncora
notebooks/            um notebook por etapa (a narrativa)
  construtores/         o script que gera cada notebook
src/especialista/     o código (a lógica)
  dados/massive.py      Etapa 1: download, conversão para BIO, entrega
  avaliacao/            Etapa 2: a régua de todos os sistemas
    previsao.py           o que um sistema devolve
    qualidade.py          métricas, intervalo de confiança, comparação pareada
    eficiencia.py         latência, tamanho, máquina
    resultados.py         a tabela de resultados, uma linha por sistema
    referencia.py         a conta oficial do MASSIVE, para conferir a nossa
    confianca.py          calibração e acerto entre as k mais prováveis (para quem devolve probabilidades)
  modelos/              os sistemas
    linha_de_base.py      piso e linha de base (sem rede neural)
    vocabulario.py        o vocabulário de features da camada A, sem PyTorch
    diet/                 Etapa 3: a rede do zero, no estilo DIET
      features.py           A. texto para ids de features
      modelo.py             B, C, E, G e a perda conjunta
      atencao.py            D. encoder com posição relativa
      crf.py                F. CRF das entidades
      sistema.py            a rede no contrato da régua
      diagrama.py           desenho da arquitetura
    encoder/              Etapa 10a: o encoder pré-treinado (XLM-R base) com cabeça conjunta, no estilo JointBERT
      subtokens.py          A. texto para subtokens, e o primeiro subtoken de cada palavra
      modelo.py             B, C e D: o encoder, as duas cabeças lineares, a perda conjunta e a tabela de subtokens dividida no treino
      sistema.py            o encoder no contrato da régua
  treino.py             Etapa 4: o laço de treino (receita, balanceamento, parada em dev)
  treino_encoder.py     Etapa 10a: o laço de ajuste fino do encoder (AdamW, aquecimento e decaimento linear, parada em dev)
  professores/          Etapa 5: as LLMs que rotulam
    demonstracoes.py      os poucos exemplos com rótulo humano que a LLM vê
    prompt.py             o prompt, igual para as duas LLMs
    resposta.py           do texto da resposta para a previsão
    rotulagem.py          das respostas gravadas para um conjunto rotulado
    claude.py             a LLM fechada, pela API (direto e em lote)
    local.py              a LLM aberta, nesta máquina: resposta e logits de intenção
  experimentos/         as variantes de cada etapa, com cache em disco
    execucao.py           uma execução de treino: treina, grava e não repete
    etapa4.py             treino com rótulos humanos e as suas comparações
    etapa4b.py            candidatas a recuperar a intenção
    etapa5a.py            as chamadas pagas à LLM fechada
    etapa5b.py            as horas de GPU das LLMs abertas e a regra de escolha do professor
    etapa6.py             os alunos da destilação: de onde vêm os rótulos de cada um e as regras da etapa
    etapa7.py             a poda e o reajuste de cada modelo: variantes, grade e a regra do que segue para a quantização
    etapa8.py             exportação e quantização de cada modelo, a medida de latência e a regra da forma final
    etapa9.py             o comparativo: junta o que as etapas gravaram, confere, monta a tabela final, as cadeias e as comparações de ponta a ponta
    etapa10a.py           o teto e os alunos com o XLM-R: a busca da receita, as variantes, as regras da etapa e a onda de treinos
    etapa10b.py           a poda e o reajuste do XLM-R: variantes, grade, a regra do que segue, a onda, os tamanhos e a latência
    etapa10c.py           exportação e quantização do XLM-R: quem é quantizado, a execução de cada modelo e a medida de latência (a regra é a da Etapa 8)
    etapa10d.py           o comparativo dos dois caminhos: o registro de sistemas, as regras gravadas antes das contas, as comparações geradas por regra, a fronteira, o preço de cada ponto, a latência na mesma sessão e a conferência dos artigos, e a lista das entregas que valem para este comparativo (`ENTREGAS_QUE_VALEM`)
    etapa11a.py           o professor aberto no Colab: as versões de professor, as regras da escada, o trabalho que roda na placa alugada e o fecho
    etapa11a_colab.py     o condutor que leva o código ao Colab, traz as respostas e desliga a máquina
    etapa11b.py           os alunos do cenário B com o professor aberto maior: a troca de professor que não vaza, as regras gravadas antes, a rodada de treinos, a temperatura, as comparações, o repasse e a entrega
    etapa11c.py           o cenário B com o professor aberto maior até o fim do caminho: quem é comprimido, as receitas das etapas de origem chamadas sem mudança de código (`emprestar`), a cadeia, as conferências, e o comparativo atualizado (o registro com as entregas da Etapa 11, as famílias novas de comparação e as regras em versões)
    etapa12.py            a investigação dos três achados (depois dos resultados, só diagnóstico): os reajustes de controle do XLM-R chamados sem mudança de código, as faixas das ativações, a quantização por grupo de tensores e com outras calibrações, as alternativas da INT8 dinâmica, a leitura única no teste e a entrega
  compressao/           Etapa 7 em diante: tornar menor a rede já treinada
    poda.py               máscaras por magnitude, poda gradual, peças inteiras, reajuste com máscara travada, arquivo e execução esparsos
    poda_encoder.py       Etapa 10b: a poda do XLM-R (magnitude no corpo, corte do vocabulário com o tokenizador reduzido, camadas de cima) e o seu reajuste
    exportacao.py         a rede em ONNX: o grafo construído a partir do modelo em PyTorch
    sistema_onnx.py       o sistema exportado no contrato da régua e o Viterbi em numpy (roda sem PyTorch)
    quantizacao.py        de 32 para 8 bits com o ONNX Runtime: dinâmica e estática, e a conferência dos pesos e dos zeros (rede do zero e XLM-R)
    exportacao_encoder.py Etapa 10c: o XLM-R com as duas cabeças em ONNX, com o grafo montado à mão (serve a vocabulário cortado e a menos camadas)
    sistema_encoder_onnx.py  Etapa 10c: o XLM-R exportado no contrato da régua, com o pacote tokenizers e o ONNX Runtime (roda sem PyTorch)
  bio.py                entidades como trechos e como rótulos por palavra
  visual.py             padrão de gráficos de todas as etapas
  caminhos.py           onde cada coisa mora
tests/                testes das peças que não podem errar caladas
entregas/             a saída de cada etapa, pronta para a seguinte
data/                 cache, fora do controle de versão: downloads e execuções de treino
```

## Como rodar

O ambiente é gerenciado com [uv](https://docs.astral.sh/uv/). As versões ficam travadas em `uv.lock`.

```bash
uv sync
```

Abrir os notebooks:

```bash
uv run jupyter lab
```

Reexecutar uma etapa do zero e conferir que ela passa (troque o nome do notebook):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04b_intencao.ipynb
```

Gerar de novo um notebook a partir do seu script, depois de mudar o script:

```bash
uv run python notebooks/construtores/build_nb04b.py notebooks/04b_intencao.ipynb
```

Treinar uma variante pela linha de comando (o notebook faz isso sozinho; serve para rodar várias em paralelo):

```bash
uv run python -m especialista.experimentos.etapa4 --variante padrao --semente 0
```

Chamar a LLM fechada (Etapa 5a). São os únicos comandos do projeto que custam dinheiro; precisam de `ANTHROPIC_API_KEY` num arquivo `.env` na raiz (modelo em `.env.example`). O que já foi respondido fica gravado e não é pedido de novo; a entrega da etapa guarda as respostas, então o notebook roda sem nenhuma chamada:

```bash
uv run python -m especialista.experimentos.etapa5a lote train
```

```bash
uv run python -m especialista.experimentos.etapa5a latencia
```

Rodar a LLM aberta (Etapa 5b). Não custa dinheiro, custa horas de GPU; uma execução interrompida continua de onde parou:

```bash
uv run python -m especialista.experimentos.etapa5b rotular --modelo qwen3_8b --conjunto train
```

Treinar um sistema da destilação (Etapa 6): `piso`, `aluno_a`, `aluno_b1`, `aluno_b1_sem_suavizacao` ou `aluno_b2_t1`, `_t2`, `_t4`, `_t8`. O notebook da etapa não treina, só lê o que estes comandos gravaram:

```bash
uv run python -m especialista.experimentos.etapa6 --variante aluno_a --semente 0
```

Podar e reajustar um modelo (Etapa 7). O modelo de partida é `teto` ou um aluno que a Etapa 6 mandou seguir; a variante é `sem_poda`, `magnitude_N`, `gradual_N`, `linhas_N` ou `sem_camada`. `--faltam` lista o que ainda não rodou, uma execução por linha, e serve para rodar a grade inteira em ondas de 7 processos na placa:

```bash
uv run python -m especialista.experimentos.etapa7 --modelo aluno_a --variante magnitude_90 --semente 0
```

```bash
uv run python -m especialista.experimentos.etapa7 --faltam | OMP_NUM_THREADS=2 xargs -P 7 -L 1 uv run python -m especialista.experimentos.etapa7
```

Se um modelo de partida for treinado de novo, o comando é o mesmo: as execuções antigas dele voltam a aparecer em `--faltam` e são refeitas.

Exportar e quantizar os modelos (Etapa 8). Roda na CPU, sem treino, em segundos por modelo. `faltam` lista o que ainda não rodou; `quantizar --tudo` roda tudo em sequência (com `--modelo teto`, só um modelo de partida, para rodar vários em paralelo). Se a Etapa 7 refizer um modelo, apague a pasta dele em `data/execucoes/08_quantizacao/` e rode de novo: o notebook confere se o que está gravado corresponde aos modelos entregues.

```bash
uv run python -m especialista.experimentos.etapa8 quantizar --tudo
```

A latência da Etapa 8 é medida num comando à parte, que precisa da máquina quieta: nada mais pode estar rodando. Ele grava quanto de CPU os outros processos usaram durante a medida, e o notebook recusa a medida se foi demais. `--refazer` mede de novo.

```bash
uv run python -m especialista.experimentos.etapa8 latencia
```

Preparar o comparativo (Etapa 9). Não treina nada, não usa a GPU e não chama a API: só produz, na CPU, em segundos, as respostas no teste que nenhuma etapa tinha gravado (as dos modelos em INT8 e a da linha de base com 299 rótulos) e mede o tamanho em disco do professor aberto, se ele ainda está no cache do Hugging Face. O notebook da etapa lê isso e as entregas das Etapas 2 a 8, e para se a conta dele discordar de um número já gravado. `faltam` lista o que ainda não foi produzido; `--refazer` produz de novo.

```bash
uv run python -m especialista.experimentos.etapa9 preparar
```

Treinar o teto e os alunos com o encoder pré-treinado (Etapa 10a). Na primeira vez, baixa o XLM-R base do Hugging Face (1,1 GB, sem token). `onda` roda a etapa inteira, um tempo depois do outro, e pode ser interrompida e chamada de novo: a busca da receita do teto (6 treinos), a escolha da receita, o teto com 3 sementes, a conferência do teto contra a linha publicada (se ele ficar longe, a onda para aqui) e, por fim, o piso e os alunos, a temperatura do aluno B2 e a abertura do teste. São 30 treinos de cerca de 2 minutos na RTX 3080, **um por vez**: dois juntos na placa rendem menos que um só. `faltam` lista o que ainda pode rodar (`--tudo`: tudo o que falta) e `situacao` diz em que tempo a etapa está:

```bash
uv run python -m especialista.experimentos.etapa10a onda
```

```bash
uv run python -m especialista.experimentos.etapa10a treinar --variante aluno_a --semente 0
```

A latência da Etapa 10a é medida num comando à parte, com a máquina quieta, ao lado do aluno A da rede do zero (`--refazer` mede de novo). O controle de precisão da etapa (o teto todo em 32 bits) é um treino a mais, fora da onda:

```bash
uv run python -m especialista.experimentos.etapa10a latencia
```

```bash
uv run python -m especialista.experimentos.etapa10a treinar --variante teto_em_32_bits --semente 0
```

Podar e reajustar o XLM-R (Etapa 10b). `onda` roda tudo o que falta, uma execução por vez na placa (cerca de 3,5 horas na RTX 3080), apaga do cache os pesos das variantes que já não podem seguir para a quantização e, no fim, mede os tamanhos em disco. Pode ser interrompida e chamada de novo:

```bash
uv run python -m especialista.experimentos.etapa10b onda
uv run python -m especialista.experimentos.etapa10b --faltam      # lista o que ainda não rodou
uv run python -m especialista.experimentos.etapa10b executar --modelo aluno_a --variante magnitude_50 --semente 0
```

A latência da Etapa 10b é medida à parte, com a máquina quieta, em grupos de variantes; o modelo sem poda entra em todo grupo, como referência (`--esparsa` mede com a execução que só faz as contas dos pesos não nulos). Os grupos usados na entrega estão em `entregas/10b_encoder_poda/latencia.sh`:

```bash
uv run python -m especialista.experimentos.etapa10b latencia --modelo aluno_a --grupo pecas --variantes vocabulario_treino camadas_6
```

Comparar os dois caminhos (Etapa 10d). Não treina nada. A ordem dos comandos é a ordem em que as coisas podem ser feitas: `regras` grava a lista fechada de comparações e os critérios de leitura **antes** de qualquer conta (se o registro de sistemas crescer, grava uma versão nova e não toca na anterior); `preparar` produz, na CPU, as respostas que nenhuma etapa gravou; `latencia` mede os dois modelos finais na mesma sessão e precisa da máquina quieta (cerca de 7 minutos; cada medida fica numa pasta com data e hora); `artigos` confere cada referência no arXiv, sem custo; `calcular` faz as contas e não roda sem as regras gravadas. O notebook só lê o que esses comandos gravaram em `data/execucoes/10d_comparativo_dos_caminhos/`.

```bash
uv run python -m especialista.experimentos.etapa10d regras
uv run python -m especialista.experimentos.etapa10d preparar
uv run python -m especialista.experimentos.etapa10d latencia
uv run python -m especialista.experimentos.etapa10d artigos
uv run python -m especialista.experimentos.etapa10d calcular
```

**Cada comparativo diz que entregas valem para ele.** O da Etapa 10d lê só as de `etapa10d.ENTREGAS_QUE_VALEM` (as que existiam quando ele foi entregue) e por isso continua reproduzindo o que entregou, 44 sistemas e 43 comparações, mesmo depois que etapas novas declaram sistemas. Para uma etapa nova entrar num comparativo **novo** sem reescrever tabela: um arquivo `sistemas.json` na entrega dela (o formato está em `etapa10d.fichas_declaradas`) ou uma função nova em `etapa10d.FONTES`; o comparativo novo chama as funções do registro (`etapa10d.registro`, `lista_de_comparacoes`, `cadeias_do_registro`) com a sua lista em `valem`. É o que a Etapa 11c faz (`etapa11c.ENTREGAS_QUE_VALEM`, `etapa11c.registro`): ela lê os professores declarados pela 11a, os alunos declarados pela 11b e as formas comprimidas que ela mesma declara. As regras do comparativo são uma linhagem de versões (1 e 2 na entrega da 10d; 3 em diante na da 11c): quando o registro cresce, grava-se a versão seguinte antes das contas, e nenhuma versão anterior é tocada.

Etapa 11a (professor aberto no Colab). O condutor roda na máquina local e só espera; quem trabalha é a placa alugada. Ele precisa da ferramenta `colab` instalada e com login feito, envia só os arquivos de `etapa11a_colab.ARQUIVOS_DO_PACOTE` e desliga a sessão no fim:

```bash
uv run python -m especialista.experimentos.etapa11a regra        # grava as regras, com a hora, antes de qualquer rodada
uv run python -m especialista.experimentos.etapa11a_colab --versao qwen3_14b__fp16__colab_a100 --gpu A100 --sessao prof11a_a100 \
    --tarefas precisao,dev,test,train --teto-horas 6 --teto-unidades 60
uv run python -m especialista.experimentos.etapa11a fechar       # formato da Etapa 5b, escada, tabelas, custo
```

Treinar os alunos do cenário B com um professor da Etapa 11a (Etapa 11b). A ordem dos comandos é a ordem em que as coisas podem ser feitas: `regras` grava, com a hora, tudo o que a etapa fixa antes de treinar (sem esse arquivo nada treina); `rodada` roda os 33 treinos numa cadeia só, a rede do zero com 7 de cada vez e depois o XLM-R em fila (cerca de uma hora e meia na RTX 3080; pode ser interrompida e chamada de novo, e a última linha que imprime começa com `FIM:` ou `FIM COM ERROS:`); `temperatura` aplica em `dev` a regra da Etapa 6 e abre o teste só para quem a regra prevê; `calcular` faz as comparações; `manifesto` vem depois do notebook. O professor é um parâmetro (`--professor <versão da Etapa 11a>`, por padrão `qwen3_14b__fp16__colab_a100`): cada professor tem as suas regras, as suas pastas e os seus nomes de variante, e uma rodada com outro professor não toca nos alunos dos anteriores. Só ganha alunos o professor que tomou o lugar do titular na escada da Etapa 11a.

```bash
uv run python -m especialista.experimentos.etapa11b regras
uv run python -m especialista.experimentos.etapa11b artigos       # confere as referências no arXiv, sem custo
uv run python -m especialista.experimentos.etapa11b rodada
uv run python -m especialista.experimentos.etapa11b temperatura
uv run python -m especialista.experimentos.etapa11b calcular
uv run python -m especialista.experimentos.etapa11b manifesto     # depois do notebook e do HTML
```

`rodada --fumaca <pasta fora do projeto>` roda quatro treinos curtos, com os dados e o professor de verdade, numa pasta descartável: serve para provar que a cadeia funciona antes de gastar a placa. Dentro do código, a troca de professor vale só dentro de um bloco, e o professor das Etapas 6 e 10a volta ao lugar na saída:

```python
from especialista.experimentos import etapa6, etapa11b

with etapa11b.com_professor("qwen3_14b__fp16__colab_a100"):
    dados = etapa6.dados_da_variante("aluno_b1")      # os rótulos do treino e de dev vêm do Qwen3-14B
```

Levar os alunos da Etapa 11b até o fim do caminho e refazer o comparativo (Etapa 11c). `regras` grava, com a hora, quem é comprimido, com que receita e o que é conferido, e a versão das regras do comparativo que vale nessa hora (sem esse arquivo nada roda). `cadeia` roda as 33 execuções numa ordem só: poda e reajuste (a rede do zero, depois o XLM-R, um por vez), exportação e quantização na CPU, e a latência por último, com a máquina quieta (cerca de 75 minutos na RTX 3080; pode ser interrompida e chamada de novo; a última linha começa com `FIM:` ou `FIM COM ERROS:`). `cadeia --fumaca <pasta fora do projeto>` roda uma versão curta numa pasta descartável. Depois da cadeia: `fechar` faz as contas da compressão (as conferências, os tamanhos, as latências), `entregar` leva essa parte para a entrega (é dela que o registro lê os sistemas novos), `regras-do-comparativo` grava a versão seguinte das regras se o registro cresceu, `calcular` faz o comparativo e `manifesto` vem depois do notebook.

```bash
uv run python -m especialista.experimentos.etapa11c regras
uv run python -m especialista.experimentos.etapa11c cadeia
uv run python -m especialista.experimentos.etapa11c fechar
uv run python -m especialista.experimentos.etapa11c entregar
uv run python -m especialista.experimentos.etapa11c regras-do-comparativo
uv run python -m especialista.experimentos.etapa11c calcular
uv run python -m especialista.experimentos.etapa11c manifesto     # depois do notebook e do HTML
```

A poda e a quantização da Etapa 11c são feitas pelas funções das Etapas 7, 8, 10b e 10c, sem mudança de código. Dentro de um bloco, a etapa de origem enxerga o aluno novo, lê o professor novo e grava na pasta da 11c; na saída, tudo volta ao que era:

```python
from especialista.experimentos import etapa11c

etapa11c.podar("xlmr__aluno_b1__qwen3_14b__fp16__colab_a100", semente=0)      # por dentro: etapa10b.executar, com o aluno e a pasta desta etapa
```

Usar um aluno novo em 8 bits, sem PyTorch (a pasta de cada semente está em `entregas/11c_cenario_b_ate_o_fim/decisao.json`):

```python
from pathlib import Path
from especialista.compressao.sistema_onnx import SistemaOnnx

sistema = SistemaOnnx.carregar(Path("entregas/11c_cenario_b_ate_o_fim/modelos/int8/zero__aluno_b2_t8__qwen3_14b__fp16__colab_a100/linhas_90/dinamica/semente_0"))
sistema.prever(["acorda-me às nove da manhã"])
```

Usar um aluno da Etapa 11b (a pasta de cada semente está em `entregas/11b_alunos_do_professor_maior/decisao.json`):

```python
from pathlib import Path
from especialista.modelos.encoder import SistemaEncoder

sistema = SistemaEncoder.carregar(Path("entregas/11b_alunos_do_professor_maior/modelos/xlmr__aluno_b1__qwen3_14b__fp16__colab_a100/semente_0"))
sistema.prever(["acorda-me às nove da manhã"])
```

Usar um modelo podado da Etapa 10b (a pasta de cada semente está em `entregas/10b_encoder_poda/decisao.json`; a pasta já traz o tokenizador reduzido):

```python
from pathlib import Path
from especialista.modelos.encoder import SistemaEncoder

sistema = SistemaEncoder.carregar(Path("entregas/10b_encoder_poda/modelos/aluno_a/vocabulario_treino/semente_0"))
sistema.prever(["acorda-me às nove da manhã"])
```

Usar um modelo da Etapa 10a (a pasta de cada semente está em `entregas/10a_encoder/decisao.json`):

```python
from pathlib import Path
from especialista.modelos.encoder import SistemaEncoder

sistema = SistemaEncoder.carregar(Path("entregas/10a_encoder/modelos/aluno_a/semente_0"))
sistema.prever(["acorda-me às nove da manhã"])
```

Usar o modelo final, sem PyTorch (a pasta de cada semente está em `entregas/08_quantizacao/decisao.json`). O processo ocupa cerca de 60 MB de RAM, quase tudo do Python e do ONNX Runtime; importar `SistemaOnnx` de `exportacao` também funciona, mas carrega o PyTorch e leva o processo a cerca de 640 MB:

```python
from pathlib import Path
from especialista.compressao.sistema_onnx import SistemaOnnx

sistema = SistemaOnnx.carregar(Path("entregas/08_quantizacao/modelos/aluno_a/linhas_90/semente_0"))
sistema.prever(["acorda-me às nove da manhã"])
```

Experimentar os dois alunos finais lado a lado, numa página local (abre em http://127.0.0.1:8765; para cada pedido mostra a resposta de cada um, a confiança e o tempo de cada trecho):

```bash
uv run python -m especialista.demo
```

O mesmo no terminal, um modelo por vez:

```bash
uv run python -m especialista.experimentar
```

Rodar os testes:

```bash
uv run pytest
```
