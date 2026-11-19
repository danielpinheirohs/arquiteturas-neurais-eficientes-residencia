"""O laço de ajuste fino do encoder pré-treinado (Etapa 10a).

A receita de partida junta o que os artigos âncora dizem sobre ajustar um
encoder pré-treinado a uma tarefa:

- **Otimizador:** Adam com decaimento de pesos de 0,01, fora os vieses e as
  normalizações. É o otimizador do BERT (Devlin et al., arXiv 1810.04805,
  apêndice A.2); a forma com o decaimento separado do gradiente é o AdamW
  (Loshchilov & Hutter, arXiv 1711.05101).
- **Taxa de aprendizado:** na casa de 2e-5 a 5e-5. São os três valores que
  o artigo do BERT recomenda no ajuste fino (5e-5, 3e-5, 2e-5; apêndice
  A.3). O JointBERT (Chen et al., arXiv 1902.10909) usa 5e-5; a busca do
  artigo do MASSIVE (FitzGerald et al., arXiv 2204.08582) ficou em 2,8e-5.
- **Agenda da taxa:** sobe em linha reta nos primeiros 10% dos passos e
  depois desce em linha reta até zero. É a agenda do código de ajuste fino
  do BERT e a que Mosbach et al. (arXiv 2006.04884, seção 6) recomendam,
  junto com a correção de viés do Adam, que o PyTorch já faz.
- **Lote de 32:** o maior dos dois tamanhos que o artigo do BERT recomenda.

Três pontos em que este projeto decide por conta própria:

- **Parada:** o modelo é avaliado em `dev` ao fim de cada época e fica o
  melhor ponto. É a regra da Etapa 2 e a mesma do laço da rede do zero.
  "Melhor" é medido contra os rótulos que `dev` trouxer: humanos no treino
  comum, do professor na destilação.
- **Corte da norma do gradiente em 1,0:** os artigos não falam disso; é o
  padrão do código de ajuste fino do BERT e da biblioteca.
- **Orçamento em passos:** uma "época" são `passos_por_epoca` passos. Para
  quem treina com o conjunto inteiro, é uma passada pelo treino, como
  sempre. Para o piso, que só tem 299 exemplos, vale o mesmo número de
  passos dos outros: Mosbach et al. mostram que, com poucos exemplos, o que
  decide se o ajuste fino funciona é o número de passos, não o de épocas.

Duas escolhas de engenharia, que não mudam a conta:

- **Precisão mista** na placa de vídeo: o encoder calcula em 16 bits
  (bfloat16) e as perdas em 32. Muda o arredondamento, não a fórmula.
- **Tabela de subtokens dividida** (`modelos/encoder/modelo.py`,
  `TabelaDividida`): o otimizador só guarda estado para as linhas da tabela
  que aparecem no treino. Um teste confere que o resultado é o mesmo do
  treino com a tabela inteira.

O laço só conhece `sistema.modelo.perda(lote)`, `sistema.lote(dados)` e
`sistema.prever(textos)`, como o da rede do zero. Na destilação nada muda
nele; muda o que ele recebe em `treino` e em `dev`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro


@dataclass(frozen=True)
class Receita:
    epocas: int = 10                 # escolhido na busca da Etapa 10a
    taxa: float = 3e-5               # idem; os três valores do artigo do BERT
    lote: int = 32                   # artigo do BERT
    aquecimento: float = 0.1         # fração dos passos em que a taxa sobe
    decaimento: float = 0.01         # decaimento de pesos (AdamW), fora vieses e normalizações
    norma_maxima: float = 1.0        # corte da norma do gradiente; 0 desliga
    passos_por_epoca: int = 0        # 0: uma passada pelo treino
    criterio: str = "exact_match"    # métrica de dev que escolhe o melhor ponto
    precisao_mista: bool = True      # só vale na placa de vídeo
    tabela_dividida: bool = True     # False: treina a tabela inteira, como a biblioteca faz (mesmo resultado, mais memória)

    def como_dict(self) -> dict:
        return asdict(self)

    def passos_da_epoca(self, exemplos: int) -> int:
        return self.passos_por_epoca or math.ceil(exemplos / self.lote)


def fracao_da_taxa(passo: int, total: int, aquecimento: float) -> float:
    """A taxa de aprendizado no passo dado, como fração da taxa da receita.

    Sobe em linha reta de 0 a 1 nos passos de aquecimento e desce em linha
    reta até 0 no último passo.
    """
    de_aquecimento = int(aquecimento * total)
    if passo < de_aquecimento:
        return passo / max(1, de_aquecimento)
    return max(0.0, (total - passo) / max(1, total - de_aquecimento))


def fila_de_lotes(exemplos: int, tamanho: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Lotes sem fim: o treino é embaralhado e cortado em lotes; quando acaba, é embaralhado de novo."""
    while True:
        ordem = rng.permutation(exemplos)
        for inicio in range(0, exemplos, tamanho):
            yield ordem[inicio : inicio + tamanho]


def treinar(sistema, treino: pd.DataFrame, dev: pd.DataFrame, receita: Receita = Receita(), semente: int = 0) -> pd.DataFrame:
    """Ajusta `sistema` no lugar e o deixa com os pesos do melhor ponto em `dev`.

    Devolve o histórico: uma linha por época, com as perdas médias de treino
    e as métricas em `dev`.
    """
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo = sistema.modelo
    na_placa = str(sistema.dispositivo).startswith("cuda")
    mista = receita.precisao_mista and na_placa

    if receita.tabela_dividida:
        sistema.preparar_treino(treino)
    tabela = modelo.tabela_dividida

    # decaimento só nas matrizes de pesos, não em vieses nem em normalizações
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    grupos = [
        {"params": [p for p in parametros if p.ndim >= 2], "weight_decay": receita.decaimento},
        {"params": [p for p in parametros if p.ndim < 2], "weight_decay": 0.0},
    ]
    otimizador = torch.optim.AdamW(grupos, lr=receita.taxa, fused=na_placa)
    passos_por_epoca = receita.passos_da_epoca(len(treino))
    total = receita.epocas * passos_por_epoca
    agenda = torch.optim.lr_scheduler.LambdaLR(otimizador, lambda passo: fracao_da_taxa(passo, total, receita.aquecimento))

    rotulos_dev, textos_dev = ouro(dev), list(dev["utt"])
    lotes = fila_de_lotes(len(treino), receita.lote, rng)
    historico, melhor_valor, melhor_estado = [], -1.0, None
    inicio = time.perf_counter()
    for epoca in range(receita.epocas):
        modelo.train()
        somas: dict[str, torch.Tensor] = {}
        for _ in range(passos_por_epoca):
            lote = sistema.lote(treino.iloc[next(lotes)])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mista):
                perdas = modelo.perda(lote)
            otimizador.zero_grad(set_to_none=True)
            perdas["total"].backward()
            if receita.norma_maxima > 0:
                torch.nn.utils.clip_grad_norm_(parametros, receita.norma_maxima)
            taxa_do_passo = otimizador.param_groups[0]["lr"]
            otimizador.step()
            if tabela is not None:   # as linhas sem gradiente sofrem só o decaimento de pesos
                tabela.decair(1 - taxa_do_passo * receita.decaimento)
            agenda.step()
            for nome, valor in perdas.items():
                somas[nome] = somas.get(nome, 0.0) + valor.detach()

        metricas = qualidade.avaliar(rotulos_dev, sistema.prever(textos_dev))
        linha = {"epoca": epoca + 1, "lote": receita.lote, "passos": passos_por_epoca, "taxa": taxa_do_passo,
                 **{f"perda_{nome}": float(soma) / passos_por_epoca for nome, soma in somas.items()},
                 **{f"dev_{nome}": valor for nome, valor in metricas.items()}}
        if metricas[receita.criterio] > melhor_valor:
            melhor_valor, melhor_estado = metricas[receita.criterio], modelo.copiar_estado()
            linha["melhor"] = True
        linha["segundos"] = time.perf_counter() - inicio
        historico.append(linha)

    modelo.load_state_dict(melhor_estado, strict=False)
    sistema.concluir_treino()
    modelo.eval()
    historico = pd.DataFrame(historico)
    historico["melhor"] = historico["epoca"] == historico.loc[historico["melhor"].fillna(False).astype(bool), "epoca"].max()
    return historico
