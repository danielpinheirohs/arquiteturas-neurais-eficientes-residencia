"""Encoder pré-treinado com cabeça conjunta, no estilo JointBERT (Chen et al., arXiv 1902.10909).

O segundo caminho do projeto (PLANO.md, "As duas arquiteturas"): em vez de
uma rede desenhada e treinada do zero, um encoder que já leu muito texto, o
XLM-R base (Conneau et al., arXiv 1911.02116), com duas cabeças lineares
por cima. De baixo para cima:

    subtokens.py  A. texto -> subtokens, e onde começa cada palavra
    modelo.py     B. o encoder pré-treinado   C. cabeça de intenção (primeiro token)
                  D. cabeça de entidades (primeiro subtoken de cada palavra)
                  e a perda conjunta
    sistema.py    o modelo no contrato da régua: `prever(textos)`

O laço de ajuste fino está em `especialista/treino_encoder.py`.
"""

from especialista.modelos.encoder.modelo import Config, EncoderConjunto, TabelaDividida
from especialista.modelos.encoder.sistema import SistemaEncoder
from especialista.modelos.encoder.subtokens import Alinhador, Lote

__all__ = ["Alinhador", "Config", "EncoderConjunto", "Lote", "SistemaEncoder", "TabelaDividida"]
