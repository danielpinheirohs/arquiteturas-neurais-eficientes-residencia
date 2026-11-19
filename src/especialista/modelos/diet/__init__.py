"""DIET do zero: reimplementação em PyTorch da arquitetura de Bunk et al. (arXiv 2004.09936).

A rede, de baixo para cima:

    features.py   A. texto -> ids de features (palavra + n-gramas de caracteres)
    modelo.py     B. tabela de features   C. projeção e token CLS
    atencao.py    D. encoder Transformer com posição relativa
    modelo.py     E. cabeça de intenção   G. cabeça de máscara (só no treino)
    crf.py        F. cabeça de entidades (CRF)
    sistema.py    o modelo no contrato da régua: `prever(textos)`
"""

from especialista.modelos.diet.features import Featurizador, Lote, ngramas
from especialista.modelos.diet.modelo import Config, Diet
from especialista.modelos.diet.sistema import SistemaDiet

__all__ = ["Config", "Diet", "Featurizador", "Lote", "SistemaDiet", "ngramas"]
