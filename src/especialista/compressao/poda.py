"""Poda: tirar pesos de uma rede já treinada e reajustar os que ficaram.

Duas formas de tirar, e o que cada uma muda de verdade:

1. **Pesos soltos, por magnitude** (Han et al., arXiv 1506.02626). Em cada
   matriz, os pesos de menor valor absoluto viram zero. A matriz continua do
   mesmo tamanho: uma "máscara" (um sim ou não por peso) diz quem ficou.
   - De uma vez: corta-se tudo no começo e reajusta-se com a máscara travada.
   - Gradual (Zhu & Gupta, arXiv 1710.01878): a máscara aperta aos poucos,
     enquanto o treino corre, e trava quando chega à esparsidade pedida.

2. **Peças inteiras** (poda estruturada; Xia et al., CoFi, arXiv 2204.00408).
   Aqui, linhas da tabela de features e uma camada do encoder. A rede que sobra
   é uma rede comum, só que menor: não precisa de máscara nem de formato
   especial.

Zerar pesos soltos não diminui o arquivo nem acelera a rede por si só: o zero
ocupa os mesmos 4 bytes e entra nas mesmas contas. O ganho só aparece quando
o armazenamento ou a execução pulam os zeros. Por isso este módulo também
tem um formato de arquivo que só grava os pesos que ficaram
(`salvar_esparso`) e uma execução que só faz as contas deles
(`execucao_esparsa`), para medir uma coisa e outra.

"Esparsidade" é sempre a fração dos pesos podáveis que está em zero.
"""

from __future__ import annotations

import copy
import json
import time
import warnings
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import ouro
from especialista.modelos.diet import Config, Diet, Featurizador, SistemaDiet
from especialista.treino import Receita, indices_da_epoca, lotes_por_comprimento

Mascaras = dict[str, torch.Tensor]  # nome do tensor -> um booleano por peso: True fica, False sai


# --------------------------------------------------------------------------
# Quais pesos entram na poda
# --------------------------------------------------------------------------


def nomes_podaveis(modelo: Diet) -> list[str]:
    """As matrizes que a poda por magnitude pode cortar: a entrada e o encoder.

    São a tabela de features, a projeção e as seis matrizes de cada bloco do
    encoder (quatro da atenção, duas da rede densa). Juntas têm quase 99% dos
    parâmetros da rede. Ficam de fora os vieses, as normalizações, a tabela de
    posição relativa e as cabeças: são poucos pesos, e cada um pesa muito na
    saída. É o recorte habitual: Zhu & Gupta e Zafrir et al. (arXiv 2111.05754)
    também podam só as matrizes das camadas grandes.
    """
    nomes = ["tabela.weight", "projecao.weight"]
    for i in range(len(modelo.blocos)):
        nomes += [f"blocos.{i}.atencao.{matriz}.weight" for matriz in ("consulta", "chave", "valor", "saida")]
        nomes += [f"blocos.{i}.expande.weight", f"blocos.{i}.contrai.weight"]
    return nomes


# --------------------------------------------------------------------------
# Poda por magnitude: máscaras
# --------------------------------------------------------------------------


@torch.no_grad()
def mascaras_por_magnitude(modelo: Diet, esparsidade: float, anteriores: Mascaras | None = None) -> Mascaras:
    """Em cada matriz podável, marca para sair a fração `esparsidade` dos pesos de menor valor absoluto.

    O corte é por matriz: todas ficam com a mesma esparsidade (Zhu & Gupta).
    A conta é exata: saem `round(esparsidade × número de pesos)` pesos de cada
    matriz, nem um a mais.

    `anteriores` são as máscaras de um corte anterior (poda gradual). Quem já
    saiu continua fora: a máscara só aperta.
    """
    if not 0.0 <= esparsidade < 1.0:
        raise ValueError(f"A esparsidade é uma fração entre 0 e 1 (sem chegar a 1); veio {esparsidade}.")
    mascaras = {}
    for nome in nomes_podaveis(modelo):
        peso = modelo.get_parameter(nome)
        tamanho = peso.abs().flatten()
        ja_saiu = None if anteriores is None else ~anteriores[nome].flatten()
        if ja_saiu is not None:
            tamanho = tamanho.masked_fill(ja_saiu, -1.0)   # entram primeiro na fila dos menores
        saem = round(esparsidade * tamanho.numel())
        fica = torch.ones_like(tamanho, dtype=torch.bool)
        fica[tamanho.argsort(stable=True)[:saem]] = False
        if ja_saiu is not None:
            fica &= ~ja_saiu
        mascaras[nome] = fica.view_as(peso)
    return mascaras


@torch.no_grad()
def aplicar(modelo: nn.Module, mascaras: Mascaras) -> None:
    """Zera, no lugar, os pesos que a máscara manda sair."""
    for nome, mascara in mascaras.items():
        modelo.get_parameter(nome).mul_(mascara)


@torch.no_grad()
def mascaras_dos_zeros(modelo: Diet) -> Mascaras:
    """As máscaras de um modelo já podado, lidas dos próprios pesos: fica quem é diferente de zero.

    É o "pattern-lock" de Zafrir et al. (arXiv 2111.05754): serve para
    continuar treinando um modelo podado, em outra sessão, sem perder a
    esparsidade (por exemplo, na quantização com treino).
    """
    return {nome: modelo.get_parameter(nome) != 0 for nome in nomes_podaveis(modelo)}


@dataclass(frozen=True)
class Agenda:
    """A poda gradual de Zhu & Gupta (arXiv 1710.01878, equação 1).

    A esparsidade sobe de zero até `esparsidade` ao longo das primeiras
    `epocas_de_rampa` épocas. A máscara é refeita a cada `passos_entre_cortes`
    passos de treino (o Δt do artigo, que testa de 100 a 1.000 e não vê
    diferença). Depois da rampa a máscara trava.
    """

    esparsidade: float
    epocas_de_rampa: int
    passos_entre_cortes: int = 100


def esparsidade_cubica(passo: int, passos_de_rampa: int, final: float, inicial: float = 0.0) -> float:
    """Quanto da rede deve estar em zero depois de `passo` passos de uma rampa de `passos_de_rampa`.

        s(t) = s_final + (s_inicial − s_final) · (1 − t / T)³

    A curva corta muito no começo, quando sobram pesos pequenos, e cada vez
    menos no fim, quando os pesos que restam já importam.
    """
    progresso = min(max(passo / passos_de_rampa, 0.0), 1.0)
    return final + (inicial - final) * (1.0 - progresso) ** 3


# --------------------------------------------------------------------------
# Contar: o que sobrou
# --------------------------------------------------------------------------


def por_tensor(modelo: Diet) -> pd.DataFrame:
    """Uma linha por tensor de parâmetros: quantos pesos tem, quantos são diferentes de zero e se entra na poda."""
    podaveis = set(nomes_podaveis(modelo))
    linhas = [
        {"tensor": nome, "pesos": p.numel(), "nao_nulos": int(torch.count_nonzero(p)), "podavel": nome in podaveis}
        for nome, p in modelo.named_parameters()
    ]
    return pd.DataFrame(linhas).set_index("tensor")


def contar(modelo: Diet) -> dict[str, int | float]:
    """Os números de tamanho de um modelo, podado ou não.

    - `parametros`: todos os números que o modelo guarda;
    - `nao_nulos`: os que são diferentes de zero;
    - `podaveis` e `nao_nulos_podaveis`: o mesmo, só nas matrizes que entram na poda;
    - `esparsidade`: a fração das matrizes podáveis que está em zero.
    """
    tabela = por_tensor(modelo)
    podaveis = tabela[tabela["podavel"]]
    return {
        "parametros": int(tabela["pesos"].sum()),
        "nao_nulos": int(tabela["nao_nulos"].sum()),
        "podaveis": int(podaveis["pesos"].sum()),
        "nao_nulos_podaveis": int(podaveis["nao_nulos"].sum()),
        "esparsidade": float(1 - podaveis["nao_nulos"].sum() / podaveis["pesos"].sum()),
    }


# --------------------------------------------------------------------------
# Poda estruturada: tirar peças inteiras
# --------------------------------------------------------------------------


@contextmanager
def _sem_a_perda_de_mascara(modelo: Diet):
    """Desliga, por um momento, a tarefa de adivinhar palavras escondidas: ela sorteia palavras, e a medida não pode depender de sorteio."""
    original = modelo.config
    modelo.config = replace(original, peso_mascara=0.0)
    try:
        yield
    finally:
        modelo.config = original


def importancia_das_linhas(sistema: SistemaDiet, dados: pd.DataFrame, tamanho_do_lote: int = 32) -> torch.Tensor:
    """Quanto cada linha da tabela de features (cada feature) pesa na perda de treino -> (n_features,).

    Apagar o feature r é multiplicar a sua linha por zero. Em primeira ordem,
    a perda muda de `w_r · ∂L/∂w_r` (a linha vezes o seu gradiente). A
    importância é o valor absoluto disso, na média dos lotes de `dados`. É a
    medida de Michel et al. (arXiv 1905.10650) para cabeças de atenção, que é
    a expansão de Taylor de Molchanov et al. (arXiv 1611.06440), aplicada a
    linhas.

    Por que não a norma da linha, que seria a "magnitude" de uma linha e é o
    critério de Joulin et al. (FastText.zip, arXiv 1612.03651) numa tabela do
    mesmo tipo: nesta rede todas as linhas nascem com a mesma norma, e a norma
    final quase não acompanha quanto o feature foi visto no treino. O notebook
    da etapa mede os dois critérios lado a lado.

    `dados` é o treino do próprio modelo, com os rótulos com que ele aprendeu
    (os do professor, para um aluno). A perda é a de intenção mais a de
    entidades, com a rede em modo de avaliação (sem dropout).
    """
    modelo = sistema.modelo
    estava_em_treino = modelo.training
    modelo.eval()
    tabela = modelo.tabela.weight
    soma = torch.zeros(tabela.shape[0], device=tabela.device)
    ordem = np.argsort(dados["tokens"].map(len).to_numpy(), kind="stable")   # lotes de comprimento parecido: o CRF anda menos
    lotes = 0
    try:
        with _sem_a_perda_de_mascara(modelo):
            for inicio in range(0, len(ordem), tamanho_do_lote):
                perdas = modelo.perda(sistema.lote(dados.iloc[ordem[inicio : inicio + tamanho_do_lote]]))
                modelo.zero_grad()
                perdas["total"].backward()
                soma += (tabela.detach() * tabela.grad).sum(dim=1).abs()
                lotes += 1
    finally:   # medir não deixa rastro no modelo, nem quando dá erro no meio
        modelo.zero_grad()
        modelo.train(estava_em_treino)
    return soma / lotes


def linhas_que_ficam(importancia: torch.Tensor, fracao_removida: float) -> torch.Tensor:
    """Os índices das linhas que ficam, em ordem, depois de tirar a fração pedida das menos importantes."""
    if not 0.0 <= fracao_removida < 1.0:
        raise ValueError(f"A fração removida fica entre 0 e 1 (sem chegar a 1); veio {fracao_removida}.")
    saem = round(fracao_removida * len(importancia))
    return importancia.argsort(stable=True)[saem:].sort().values.cpu()


def _novo_sistema(sistema: SistemaDiet, config: Config, featurizador: Featurizador, estado: dict) -> SistemaDiet:
    modelo = Diet(config, len(featurizador), len(sistema.intents), sistema.tags)
    modelo.load_state_dict(estado)
    return SistemaDiet(modelo, featurizador, sistema.intents, sistema.tags, sistema.dispositivo, sistema.tamanho_do_lote)


def remover_linhas(sistema: SistemaDiet, ficam) -> SistemaDiet:
    """Um sistema novo só com as linhas `ficam` da tabela de features. O vocabulário encolhe junto.

    Uma palavra passa a ser a soma dos features dela que ficaram. Remover um
    feature que não aparece num pedido não muda nada na resposta a esse pedido.
    """
    ficam = torch.as_tensor(ficam, dtype=torch.long).sort().values
    nomes = list(sistema.featurizador.vocabulario)   # o vocabulário é gravado na ordem dos ids
    featurizador = Featurizador({nomes[i]: novo for novo, i in enumerate(ficam.tolist())}, sistema.featurizador.n_max)
    por_feature = {"tabela.weight"}
    if isinstance(sistema.modelo.atalho, nn.EmbeddingBag):   # o atalho "features" também tem uma linha por feature
        por_feature.add("atalho.weight")
    estado = {
        nome: valor[ficam.to(valor.device)] if nome in por_feature else valor
        for nome, valor in sistema.modelo.state_dict().items()
    }
    return _novo_sistema(sistema, sistema.modelo.config, featurizador, estado)


def remover_camada(sistema: SistemaDiet, qual: int = -1) -> SistemaDiet:
    """Um sistema novo sem um dos blocos do encoder. O padrão é tirar o de cima (o último).

    Sajjad et al. (arXiv 2004.03844) comparam várias formas de descartar
    camadas de um Transformer e ficam com a mais simples: tirar as de cima.
    Cada bloco soma a sua contribuição à sequência que recebe (atalho
    residual), então a rede sem um bloco continua bem formada: a sequência
    passa direto para o que vem depois.
    """
    n = len(sistema.modelo.blocos)
    if n < 2:
        raise ValueError("A rede só tem um bloco: não há camada para tirar.")
    qual = qual % n
    estado = {}
    for nome, valor in sistema.modelo.state_dict().items():
        partes = nome.split(".")
        if partes[0] == "blocos":
            bloco = int(partes[1])
            if bloco == qual:
                continue
            partes[1] = str(bloco - (bloco > qual))   # os blocos de cima descem uma posição
        estado[".".join(partes)] = valor
    config = replace(sistema.modelo.config, n_camadas=n - 1)
    return _novo_sistema(sistema, config, sistema.featurizador, estado)


# --------------------------------------------------------------------------
# O reajuste: treinar com a poda dentro do laço
# --------------------------------------------------------------------------


def passos_por_epoca(treino: pd.DataFrame, receita: Receita) -> list[int]:
    """Quantos passos de treino cada época vai ter. Depende do tamanho do lote, que cresce, e não do sorteio."""
    rng = np.random.default_rng(0)
    comprimentos = treino["tokens"].map(len).to_numpy()
    return [
        len(lotes_por_comprimento(indices_da_epoca(treino["intent"], receita.minimo_por_intencao, rng), comprimentos,
                                  receita.tamanho_do_lote(epoca), rng))
        for epoca in range(receita.epocas)
    ]


def reajustar(
    sistema: SistemaDiet,
    treino: pd.DataFrame,
    dev: pd.DataFrame,
    receita: Receita = Receita(),
    semente: int = 0,
    mascaras: Mascaras | None = None,
    agenda: Agenda | None = None,
    partida_concorre: bool = True,
) -> pd.DataFrame:
    """Treina `sistema` no lugar, com a poda dentro do laço, e o deixa com os pesos do melhor ponto em `dev`.

    É o laço de `treino.treinar`, passo por passo (mesma receita, mesmos
    sorteios), com três usos:

    - `mascaras`: reajuste depois da poda de uma vez (Han et al.). A máscara
      fica travada: depois de cada passo do otimizador, os pesos podados
      voltam a zero.
    - `agenda`: poda gradual (Zhu & Gupta). A máscara é refeita a cada
      `passos_entre_cortes` passos, com a esparsidade da curva cúbica, até o
      fim da rampa; daí em diante fica travada. Só as épocas em que a rede já
      está na esparsidade final concorrem a "melhor ponto".
    - nenhum dos dois: o controle sem poda e as redes da poda estruturada, que
      não precisam de máscara. Com `partida_concorre=False` é exatamente
      `treino.treinar` (um teste confere).

    `partida_concorre`: o modelo como chegou (já podado, antes de qualquer
    passo) também concorre a melhor ponto, como "época 0". Se nenhuma época do
    reajuste for melhor que ele em `dev`, o reajuste não muda o modelo. Um
    reajuste recomeça o treino (otimizador novo, lote pequeno) e pode
    atrapalhar um modelo que não precisava de conserto; assim ele só fica
    quando ajuda. Não vale para a poda gradual: lá o ponto de partida ainda
    não está na esparsidade final.

    A média móvel dos pesos, quando a receita a usa, recebe a mesma máscara:
    o modelo avaliado e guardado é sempre esparso.

    Devolve o histórico: uma linha por época, com as perdas de treino, a
    esparsidade medida no fim da época e, nas épocas avaliadas, as métricas
    em `dev`. Quando a partida concorre, a primeira linha é a época 0.
    """
    if mascaras is not None and agenda is not None:
        raise ValueError("Ou a máscara já vem pronta (`mascaras`) ou é feita aos poucos (`agenda`): não as duas.")
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo = sistema.modelo
    if receita.decaimento > 0:
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

    mascaras = dict(mascaras or {})
    aplicar(modelo, mascaras)
    cortes = fim_da_rampa = 0
    if agenda is not None:
        cortes = sum(passos_por_epoca(treino, receita)[: agenda.epocas_de_rampa]) // agenda.passos_entre_cortes
        fim_da_rampa = cortes * agenda.passos_entre_cortes
        if cortes < 1 or agenda.epocas_de_rampa > receita.epocas:
            raise ValueError("A rampa da poda gradual precisa caber no treino e ter pelo menos um corte.")

    media = {nome: p.detach().clone() for nome, p in modelo.named_parameters()} if receita.media_dos_pesos > 0 else None
    passo = 0

    historico, melhor_valor, melhores_pesos = [], -1.0, None
    inicio = time.perf_counter()
    if partida_concorre and agenda is None:
        metricas = qualidade.avaliar(rotulos_dev, sistema.prever(textos_dev))
        melhor_valor, melhores_pesos = metricas[receita.criterio], copy.deepcopy(modelo.state_dict())
        historico.append({"epoca": 0, "passos": 0, "esparsidade": contar(modelo)["esparsidade"],
                          **{f"dev_{nome}": valor for nome, valor in metricas.items()}, "melhor": True, "segundos": 0.0})
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
            passo += 1
            # --- a poda dentro do laço ---
            if agenda is not None and passo <= fim_da_rampa and passo % agenda.passos_entre_cortes == 0:
                alvo = esparsidade_cubica(passo, fim_da_rampa, agenda.esparsidade)
                mascaras = mascaras_por_magnitude(modelo, alvo, anteriores=mascaras or None)
                if media is not None:   # a média também perde os pesos que acabaram de sair
                    with torch.no_grad():
                        for nome, mascara in mascaras.items():
                            media[nome].mul_(mascara)
            aplicar(modelo, mascaras)   # máscara travada: o que saiu volta a zero depois de cada passo
            # ------------------------------
            if media is not None:
                fator = min(receita.media_dos_pesos, (1 + passo) / (10 + passo))
                with torch.no_grad():
                    for nome, p in modelo.named_parameters():
                        media[nome].lerp_(p, 1 - fator)
            for nome, valor in perdas.items():
                somas[nome] = somas.get(nome, 0.0) + valor.detach().item()
            passos += 1

        linha = {"epoca": epoca + 1, "lote": tamanho, "passos": passos, "esparsidade": contar(modelo)["esparsidade"],
                 **{f"perda_{nome}": soma / passos for nome, soma in somas.items()}}
        if (epoca + 1) % receita.avaliar_a_cada == 0 or epoca + 1 == receita.epocas:
            if media is not None:
                do_passo = {nome: p.detach().clone() for nome, p in modelo.named_parameters()}
                modelo.load_state_dict(media, strict=False)
            metricas = qualidade.avaliar(rotulos_dev, sistema.prever(textos_dev))
            linha.update({f"dev_{nome}": valor for nome, valor in metricas.items()})
            na_esparsidade_final = passo >= fim_da_rampa   # na poda gradual, as épocas da rampa não concorrem
            if na_esparsidade_final and metricas[receita.criterio] > melhor_valor:
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
    primeiras = ["epoca", "lote", "passos", "esparsidade", *(c for c in historico.columns if c.startswith("perda_"))]
    return historico[primeiras + [c for c in historico.columns if c not in primeiras]]


# --------------------------------------------------------------------------
# Armazenamento que aproveita os zeros
# --------------------------------------------------------------------------

ARQUIVO_ESPARSO = "pesos_esparsos.npz"
_BITS, _VALORES, _FORMA = "#bits", "#valores", "#forma"


def salvar_esparso(sistema: SistemaDiet, pasta: Path) -> Path:
    """Grava os pesos em `pasta/pesos_esparsos.npz`, sem gastar 4 bytes com cada zero.

    Para cada tensor, guarda-se um mapa de bits (1 bit por posição: tem peso
    ou é zero) e, em fila, só os pesos diferentes de zero. Custa
    `posições / 8 + 4 × não nulos` bytes, contra `4 × posições` do arquivo
    comum. Um tensor sem zeros bastantes para compensar o mapa é gravado
    inteiro, como sempre.

    É a ideia do armazenamento esparso de Han et al. (Deep Compression, arXiv
    1510.00149), na versão mais simples: eles guardam a distância até o
    próximo peso; aqui, um bit por posição. A leitura devolve exatamente os
    mesmos números.
    """
    arquivos = {}
    for nome, tensor in sistema.modelo.state_dict().items():
        valores = tensor.detach().cpu().numpy()
        fica = valores != 0
        if fica.size / 8 + int(fica.sum()) * valores.itemsize < valores.nbytes:
            arquivos[nome + _BITS] = np.packbits(fica)
            arquivos[nome + _VALORES] = valores[fica]
            arquivos[nome + _FORMA] = np.array(valores.shape, dtype=np.int64)
        else:
            arquivos[nome] = valores
    pasta.mkdir(parents=True, exist_ok=True)
    np.savez(pasta / ARQUIVO_ESPARSO, **arquivos)
    return pasta / ARQUIVO_ESPARSO


def carregar_esparso(pasta: Path, dispositivo: str = "cpu") -> SistemaDiet:
    """Monta o sistema a partir de `pesos_esparsos.npz`, de `config.json` e de `features.json` (não lê `pesos.pt`)."""
    dados = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
    featurizador = Featurizador.carregar(pasta / "features.json")
    estado = {}
    with np.load(pasta / ARQUIVO_ESPARSO) as arquivos:
        for chave in arquivos.files:
            if chave.endswith(_BITS):
                nome = chave[: -len(_BITS)]
                forma = tuple(arquivos[nome + _FORMA])
                fica = np.unpackbits(arquivos[chave], count=int(np.prod(forma))).astype(bool).reshape(forma)
                valores = np.zeros(forma, dtype=arquivos[nome + _VALORES].dtype)
                valores[fica] = arquivos[nome + _VALORES]
                estado[nome] = torch.from_numpy(valores)
            elif not chave.endswith((_VALORES, _FORMA)):
                estado[chave] = torch.from_numpy(arquivos[chave])
    modelo = Diet(Config(**dados["config"]), len(featurizador), len(dados["intents"]), dados["tags"])
    modelo.load_state_dict(estado)
    return SistemaDiet(modelo, featurizador, dados["intents"], dados["tags"], dispositivo)


def ler_esparso(caminho: Path) -> dict[str, torch.Tensor]:
    """Os tensores gravados por `salvar_esparso`, de volta inteiros (com os zeros no lugar). Serve a qualquer rede.

    Acréscimo da Etapa 10b, que grava o encoder no mesmo formato.
    """
    estado = {}
    with np.load(caminho) as arquivos:
        for chave in arquivos.files:
            if chave.endswith(_BITS):
                nome = chave[: -len(_BITS)]
                forma = tuple(arquivos[nome + _FORMA])
                fica = np.unpackbits(arquivos[chave], count=int(np.prod(forma))).astype(bool).reshape(forma)
                valores = np.zeros(forma, dtype=arquivos[nome + _VALORES].dtype)
                valores[fica] = arquivos[nome + _VALORES]
                estado[nome] = torch.from_numpy(valores)
            elif not chave.endswith((_VALORES, _FORMA)):
                estado[chave] = torch.from_numpy(arquivos[chave])
    return estado


def bytes_do_sistema(pasta: Path, pesos: str = "pesos.pt") -> int:
    """Tamanho em disco de um sistema gravado: vocabulário, configuração e o arquivo de pesos escolhido."""
    return sum((pasta / arquivo).stat().st_size for arquivo in ("features.json", "config.json", pesos))


def bytes_com_zip(caminho: Path) -> int:
    """Tamanho do arquivo depois de um compressor comum (deflate, o do zip). Uma sequência de zeros quase some."""
    return len(zlib.compress(caminho.read_bytes(), 6))


# --------------------------------------------------------------------------
# Execução que aproveita os zeros
# --------------------------------------------------------------------------


class LinearEsparsa(nn.Module):
    """Uma camada linear que guarda só os pesos diferentes de zero e só faz as contas deles.

    Os pesos ficam no formato CSR (para cada linha, as colunas ocupadas e os
    seus valores), que é o que as bibliotecas de álgebra esparsa usam.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # o PyTorch avisa que o formato CSR ainda é beta
            self.register_buffer("peso", linear.weight.detach().to_sparse_csr())
        self.register_buffer("vies", linear.bias.detach().clone())
        self.saida = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plano = x.reshape(-1, x.shape[-1])
        return (torch.sparse.mm(self.peso, plano.T).T + self.vies).reshape(*x.shape[:-1], self.saida)


def execucao_esparsa(sistema: SistemaDiet) -> SistemaDiet:
    """Uma cópia do sistema, na CPU, em que a projeção e as matrizes do encoder só fazem as contas dos pesos não nulos.

    Serve para medir a latência que a poda daria com uma execução que
    aproveita os zeros. Só para inferência. A tabela de features não muda: uma
    consulta já lê só as linhas dos features do pedido.
    """
    copia = SistemaDiet(copy.deepcopy(sistema.modelo), sistema.featurizador, sistema.intents, sistema.tags, "cpu",
                        sistema.tamanho_do_lote)
    modelo = copia.modelo.eval()
    modelo.projecao = LinearEsparsa(modelo.projecao)
    for bloco in modelo.blocos:
        for matriz in ("consulta", "chave", "valor", "saida"):
            setattr(bloco.atencao, matriz, LinearEsparsa(getattr(bloco.atencao, matriz)))
        bloco.expande, bloco.contrai = LinearEsparsa(bloco.expande), LinearEsparsa(bloco.contrai)
    return copia
