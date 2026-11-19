# Etapa 1 · Dados: resumo da entrega

**Pergunta.** Em que cenário de teste vamos provar que um especialista pequeno substitui uma LLM?

**Resposta.** No Amazon MASSIVE, recorte pt-PT: 16.521 comandos de assistente virtual em português europeu, cada um com intenção e entidades anotadas por humanos. As partições são as oficiais e os números conferem com FitzGerald et al. (arXiv 2204.08582).

## Números

| | train | dev | test |
|---|---|---|---|
| Enunciados | 11.514 | 2.033 | 2.974 |
| Sem nenhuma entidade | 33,0% | 32,1% | 33,6% |
| Mediana de palavras | 7 | 7 | 7 |
| Intenção confirmada por 3 de 3 avaliadores | 78,8% | 78,3% | 77,2% |
| Texto que também está no treino | n/a | 143 (7,0%) | 199 (6,7%) |

| Rótulos | |
|---|---|
| Intenções | 60 |
| Tipos de entidade | 55 |
| Tags BIO | 111 |
| Cenários | 18 |

## O que os dados dizem para as próximas etapas

- **Avaliação.** As classes são desbalanceadas (810 exemplos na maior intenção, 4 na menor) e 75,9% das palavras têm tag `O`. Acurácia de intenção e acerto por palavra enganam; medir macro-F1 e F1 por entidade inteira.
- **Arquitetura.** Enunciados curtos (99% com até 18 palavras) e vocabulário pequeno (6.220 palavras no treino), mas 4,0% das palavras do teste são inéditas: a rede precisa olhar dentro da palavra.
- **Leitura dos resultados.** 199 (6,7%) dos enunciados de teste têm texto idêntico a algum de treino, e 31 textos aparecem com mais de uma intenção. Os rótulos têm ruído; a coluna `utt_in_train` permite reportar também só os inéditos.

## Decisões tomadas

| Decisão | Escolha | Motivo |
|---|---|---|
| Unidade de texto | Palavra separada por espaço | É a unidade do MASSIVE; a quebra em pedaços menores fica para a arquitetura |
| Formato das entidades | Tags BIO por palavra | Conversão sem perda (ida e volta idêntica em 100,0% dos enunciados) |
| Inventário de rótulos | Esquema completo, das três partições | O esquema é conhecido de antemão, como a lista de ferramentas de um agente |
| Duplicatas e rótulos ruidosos | Mantidos e marcados | As partições ficam idênticas às publicadas, e os números, comparáveis |
| Normalização do texto | Nenhuma | O texto já vem em minúsculas |

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `train.parquet`, `dev.parquet`, `test.parquet` | Uma linha por enunciado: `utt`, `intent`, `tokens`, `tags` e colunas de apoio |
| `rotulos.json` | Intenções, tipos de entidade, tags BIO e intenções por cenário |
| `manifesto.json` | Fonte, hash, data e versões usadas |
| `figuras/` | Os gráficos deste notebook |
| `01_dados.html` | Este notebook executado, para leitura |

Para usar nas próximas etapas:

```python
from especialista.dados import massive
particoes, rotulos = massive.carregar_entrega()
```
