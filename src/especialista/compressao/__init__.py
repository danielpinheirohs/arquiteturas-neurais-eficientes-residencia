"""Compressão: tornar menor uma rede que já foi treinada.

- `poda`: tirar pesos (Etapa 7). Soltos, pelos de menor valor absoluto, ou em
  peças inteiras (linhas da tabela de features, uma camada do encoder), e
  reajustar o que sobrou.
- `exportacao`: tirar a rede do PyTorch (Etapa 8). O grafo ONNX da tabela de
  features até as pontuações.
- `sistema_onnx`: o sistema exportado no contrato da régua (`SistemaOnnx`), com
  o Viterbi do CRF em numpy. Roda no ONNX Runtime e não carrega o PyTorch.
- `quantizacao`: passar os pesos de 32 para 8 bits (Etapa 8), sem treino. A
  tabela de features e as matrizes viram inteiros; a faixa das ativações é
  medida em cada pedido (dinâmica) ou antes, em pedidos do treino (estática).
"""
