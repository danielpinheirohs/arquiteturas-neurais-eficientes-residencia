"""O laço de treino do projeto.

A receita de partida é a do artigo do DIET (Bunk et al., arXiv 2004.09936,
seções 3 e 4.2): Adam com taxa de aprendizado de 0,001, lote que cresce de
64 para 128 ao longo do treino (Smith et al., "Don't Decay the Learning
Rate, Increase the Batch Size", arXiv 1711.00489) e lotes balanceados por
intenção.

Dois pontos em que este projeto decide por conta própria:

- Parada: o artigo treina um número fixo de épocas. Aqui o modelo é
  avaliado em `dev` a cada poucas épocas e fica o melhor ponto. É a regra
  da Etapa 2: decide-se em `dev`.
- Balanceamento: o artigo cita a estratégia sem detalhar. Aqui, em cada
  época, toda intenção aparece pelo menos `minimo_por_intencao` vezes; as
  mais raras são repetidas até lá.

Uma escolha de engenharia, que só muda a velocidade: cada lote junta
enunciados de comprimento parecido. O CRF anda palavra por palavra até o
maior enunciado do lote; com lotes homogêneos ele anda menos.

O laço só conhece `sistema.modelo.perda(lote)` e `sistema.prever(textos)`,
então serve igual para o treino com rótulos humanos, para a destilação e
para o reajuste depois da poda.

Na destilação (Etapa 6) nada muda no laço; muda o que ele recebe. `treino`
traz os rótulos do professor e, quando a intenção aprende com as
probabilidades dele, a coluna `logits_professor`, que `sistema.lote` leva
até a perda. `dev` traz os rótulos de quem deve escolher a época de parada:
os do mesmo professor, para que nenhum rótulo humano entre no treino do
aluno.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro


@dataclass(frozen=True)
class Receita:
    epocas: int = 80
    taxa: float = 1e-3               # artigo
    lote_inicial: int = 64           # artigo
    lote_final: int = 128            # artigo
    minimo_por_intencao: int = 100   # 0 desliga o balanceamento
    decaimento: float = 0.0          # decaimento de pesos (AdamW); o artigo não usa
    media_dos_pesos: float = 0.0     # > 0: avalia e guarda a média móvel dos pesos, com esse fator (ex.: 0,999)
    avaliar_a_cada: int = 5          # épocas entre avaliações em dev
    criterio: str = "exact_match"    # métrica de dev que escolhe o melhor ponto

    def como_dict(self) -> dict:
        return asdict(self)

    def tamanho_do_lote(self, epoca: int) -> int:
        """Cresce em linha reta do lote inicial ao final."""
        fracao = epoca / max(1, self.epocas - 1)
        return round(self.lote_inicial + fracao * (self.lote_final - self.lote_inicial))


def indices_da_epoca(intents: pd.Series, minimo: int, rng: np.random.Generator) -> np.ndarray:
    """Todos os exemplos uma vez, mais repetições das intenções com menos de `minimo` exemplos."""
    indices = [np.arange(len(intents))]
    if minimo > 0:
        posicoes = pd.Series(np.arange(len(intents))).groupby(intents.to_numpy()).apply(np.array)
        for do_grupo in posicoes:
            if len(do_grupo) < minimo:
                indices.append(rng.choice(do_grupo, minimo - len(do_grupo), replace=True))
    return rng.permutation(np.concatenate(indices))


def lotes_por_comprimento(
    ordem: np.ndarray, comprimentos: np.ndarray, tamanho: int, rng: np.random.Generator, lotes_por_grupo: int = 50
) -> list[np.ndarray]:
    """Corta a ordem da época em lotes de comprimento parecido, sem perder o embaralhamento.

    A ordem (já embaralhada) é dividida em grupos de `lotes_por_grupo` lotes;
    dentro de cada grupo os exemplos são ordenados por comprimento e cortados
    em lotes; no fim, a sequência dos lotes é embaralhada.
    """
    lotes = []
    grupo = tamanho * lotes_por_grupo
    for inicio in range(0, len(ordem), grupo):
        pedaco = ordem[inicio : inicio + grupo]
        pedaco = pedaco[np.argsort(comprimentos[pedaco], kind="stable")]
        lotes.extend(pedaco[i : i + tamanho] for i in range(0, len(pedaco), tamanho))
    return [lotes[i] for i in rng.permutation(len(lotes))]


def treinar(sistema, treino: pd.DataFrame, dev: pd.DataFrame, receita: Receita = Receita(), semente: int = 0) -> pd.DataFrame:
    """Treina `sistema` no lugar e o deixa com os pesos do melhor ponto em `dev`.

    "Melhor" é medido contra os rótulos que `dev` trouxer: humanos no treino
    comum, do professor na destilação.

    Devolve o histórico: uma linha por época, com as perdas médias de treino
    e, nas épocas avaliadas, as métricas em `dev`.
    """
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo = sistema.modelo
    if receita.decaimento > 0:
        # decaimento só nas matrizes de pesos, não em vieses nem em normalizações
        matrizes = [p for p in modelo.parameters() if p.ndim >= 2]
        outros = [p for p in modelo.parameters() if p.ndim < 2]
        otimizador = torch.optim.AdamW(
            [{"params": matrizes, "weight_decay": receita.decaimento}, {"params": outros, "weight_decay": 0.0}],
            lr=receita.taxa,
        )
    else:
        otimizador = torch.optim.Adam(modelo.parameters(), lr=receita.taxa)
    rotulos_dev, textos_dev = ouro(dev), list(dev["utt"])
    comprimentos = treino["tokens"].map(len).to_numpy()

    # Média móvel dos pesos (opcional): em vez dos pesos do último passo, que oscilam, usa-se a média
    # dos passos recentes (Polyak; Izmailov et al., arXiv 1803.05407).
    media = {nome: p.detach().clone() for nome, p in modelo.named_parameters()} if receita.media_dos_pesos > 0 else None
    passos_dados = 0

    historico, melhor_valor, melhores_pesos = [], -1.0, None
    inicio = time.perf_counter()
    for epoca in range(receita.epocas):
        modelo.train()
        ordem = indices_da_epoca(treino["intent"], receita.minimo_por_intencao, rng)
        tamanho = receita.tamanho_do_lote(epoca)
        somas: dict[str, float] = {}
        passos = 0
        for indices in lotes_por_comprimento(ordem, comprimentos, tamanho, rng):
            perdas = modelo.perda(sistema.lote(treino.iloc[indices]))
            otimizador.zero_grad()
            perdas["total"].backward()
            otimizador.step()
            if media is not None:
                passos_dados += 1
                fator = min(receita.media_dos_pesos, (1 + passos_dados) / (10 + passos_dados))  # média curta no começo
                with torch.no_grad():
                    for nome, p in modelo.named_parameters():
                        media[nome].lerp_(p, 1 - fator)
            for nome, valor in perdas.items():
                somas[nome] = somas.get(nome, 0.0) + valor.detach().item()
            passos += 1

        linha = {"epoca": epoca + 1, "lote": tamanho, "passos": passos,
                 **{f"perda_{nome}": soma / passos for nome, soma in somas.items()}}
        if (epoca + 1) % receita.avaliar_a_cada == 0 or epoca + 1 == receita.epocas:
            if media is not None:  # avalia com os pesos médios; os do último passo voltam depois
                do_passo = {nome: p.detach().clone() for nome, p in modelo.named_parameters()}
                modelo.load_state_dict(media, strict=False)
            metricas = qualidade.avaliar(rotulos_dev, sistema.prever(textos_dev))
            linha.update({f"dev_{nome}": valor for nome, valor in metricas.items()})
            if metricas[receita.criterio] > melhor_valor:
                melhor_valor = metricas[receita.criterio]
                melhores_pesos = copy.deepcopy(modelo.state_dict())
                linha["melhor"] = True
            if media is not None:
                modelo.load_state_dict(do_passo, strict=False)
        linha["segundos"] = time.perf_counter() - inicio
        historico.append(linha)

    modelo.load_state_dict(melhores_pesos)
    modelo.eval()
    historico = pd.DataFrame(historico)
    historico["melhor"] = historico["epoca"] == historico.loc[historico["melhor"].fillna(False).astype(bool), "epoca"].max()
    return historico
