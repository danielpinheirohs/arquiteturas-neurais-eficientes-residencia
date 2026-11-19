"""O modelo completo: entrada (B, C), encoder (D), três cabeças (E, F, G) e a perda conjunta.

Referência: Bunk et al., "DIET: Lightweight Language Understanding for
Dialogue Systems" (arXiv 2004.09936), seção 3. O que vem do artigo e o que
é escolha deste projeto está marcado campo a campo em `Config`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from especialista.modelos.diet.atencao import Bloco
from especialista.modelos.diet.crf import CRF, restricoes_bio
from especialista.modelos.diet.features import Lote


@dataclass(frozen=True)
class Config:
    # --- do artigo ---
    d_modelo: int = 256            # dimensão do Transformer
    n_camadas: int = 2             # blocos do Transformer
    d_rotulo: int = 20             # espaço em que entrada e rótulos são comparados
    p_mascara: float = 0.15        # fração das palavras escondidas no treino
    # --- o artigo não informa; valor padrão da implementação de referência (Rasa) ---
    n_cabecas: int = 4
    d_features: int = 128          # dimensão em que os features esparsos são somados
    distancia_max: int = 5         # alcance da posição relativa
    dropout: float = 0.2
    dropout_features: float = 0.2  # o artigo aplica dropout nos features esparsos, sem dizer a taxa
    # --- o artigo não informa; escolha deste projeto ---
    d_ffn: int = 1024              # 4 x d_modelo, a proporção de Vaswani et al.
    usar_crf: bool = True          # False: cada palavra escolhe a sua tag sozinha (para medir o que o CRF acrescenta)
    # --- pesos da perda conjunta (no artigo, soma simples; qualquer termo pode ser desligado) ---
    peso_intencao: float = 1.0
    peso_entidades: float = 1.0
    peso_mascara: float = 1.0
    # --- candidatas da Etapa 4b (investigação da intenção); desligadas por padrão ---
    atalho_intencao: str = "nenhum"  # "resumo" ou "features": a intenção também lê direto o saco de features
    suavizacao: float = 0.0          # suavização de rótulos na perda de intenção
    # --- destilação com os logits do professor (Etapa 6; Hinton et al., arXiv 1503.02531); desligada por padrão ---
    temperatura: float = 0.0         # > 0: a intenção aprende com a distribuição do professor, as duas com esta temperatura
    peso_rotulo: float = 0.0         # na destilação, a parte (de 0 a 1) da perda de intenção que continua vindo do rótulo rígido

    def como_dict(self) -> dict:
        return asdict(self)


def perda_de_destilacao(do_aluno: torch.Tensor, do_professor: torch.Tensor, temperatura: float) -> torch.Tensor:
    """Quanto a distribuição do aluno se afasta da do professor, as duas com temperatura.

    Hinton et al. (arXiv 1503.02531, seção 2). Os logits dos dois são divididos
    por T antes do softmax. Com T maior que 1 a distribuição fica mais
    espalhada, e as intenções que o professor deixou em segundo e terceiro
    lugar pesam mais no que o aluno aprende.

        p = softmax(logits do professor / T)
        q = softmax(logits do aluno / T)
        perda = T² · KL(p ‖ q) = T² · Σ p · (log p − log q)

    O fator T² vem do artigo: o gradiente dessa divergência encolhe com 1/T², e
    multiplicar por T² o mantém do mesmo tamanho para qualquer temperatura.
    Assim a perda de intenção continua pesando o mesmo, ao lado das outras
    perdas da rede, quando se troca T.

    `do_professor` só precisa ser proporcional ao log da probabilidade de cada
    intenção: somar uma constante a uma linha não muda o softmax. Uma intenção
    com probabilidade zero entra como −inf.

    As duas entradas têm forma (B, n_intents). Devolve a média do lote.
    """
    alvo = F.softmax(do_professor / temperatura, dim=-1)
    log_do_aluno = F.log_softmax(do_aluno / temperatura, dim=-1)
    return F.kl_div(log_do_aluno, alvo, reduction="batchmean") * temperatura**2


class Diet(nn.Module):
    def __init__(self, config: Config, n_features: int, n_intents: int, tags: list[str]):
        super().__init__()
        self.config = config
        c = config

        # B. Tabela de features: somar as linhas dos features ligados é o mesmo que aplicar
        #    uma camada densa a um vetor esparso de zeros e uns, sem nunca montar o vetor.
        self.tabela = nn.EmbeddingBag(n_features, c.d_features, mode="sum")
        nn.init.normal_(self.tabela.weight, std=0.1)

        # C. Projeção para a dimensão do Transformer e marca do token CLS.
        self.projecao = nn.Linear(c.d_features, c.d_modelo)
        self.marca_cls = nn.Parameter(torch.zeros(c.d_modelo))
        self.dropout = nn.Dropout(c.dropout)

        # D. Encoder.
        self.blocos = nn.ModuleList(
            Bloco(c.d_modelo, c.n_cabecas, c.d_ffn, c.distancia_max, c.dropout) for _ in range(c.n_camadas)
        )
        self.norma_final = nn.LayerNorm(c.d_modelo)

        # E. Intenção: a saída do CLS e cada intenção viram vetores de d_rotulo dimensões;
        #    a pontuação de uma intenção é o produto interno entre os dois.
        self.cls_para_rotulo = nn.Linear(c.d_modelo, c.d_rotulo)
        self.vetor_da_intencao = nn.Parameter(torch.empty(n_intents, c.d_rotulo))
        nn.init.normal_(self.vetor_da_intencao, std=0.1)

        # E'. Atalho (opcional): um caminho linear direto do saco de features do enunciado até as intenções,
        #     somado à pontuação do caminho profundo (Wide & Deep, Cheng et al., arXiv 1606.07792).
        if c.atalho_intencao == "resumo":      # parte do resumo de d_features dimensões que o CLS já usa
            self.atalho = nn.Linear(c.d_features, n_intents)
        elif c.atalho_intencao == "features":  # um peso por (feature, intenção): um classificador linear dentro da rede
            self.atalho = nn.EmbeddingBag(n_features, n_intents, mode="sum")
            nn.init.zeros_(self.atalho.weight)
        else:
            self.atalho = None

        # F. Entidades: uma pontuação por tag para cada palavra, depois o CRF.
        self.emissao = nn.Linear(c.d_modelo, len(tags))
        self.crf = CRF(len(tags), *restricoes_bio(tags))

        # G. Máscara (só no treino): adivinhar a palavra escondida, comparando no mesmo tipo de espaço.
        self.vetor_mascara = nn.Parameter(torch.zeros(c.d_features))
        nn.init.normal_(self.vetor_mascara, std=0.1)
        self.saida_para_token = nn.Linear(c.d_modelo, c.d_rotulo)
        self.entrada_para_token = nn.Linear(c.d_features, c.d_rotulo)

    # ------------------------------------------------------------------
    # B e C: do lote de ids para a sequência que entra no encoder
    # ------------------------------------------------------------------

    def _dropout_de_features(self, lote: Lote) -> torch.Tensor | None:
        """Dropout nos features esparsos: no treino, cada feature ligado é apagado com probabilidade p."""
        if not self.training or self.config.dropout_features == 0:
            return None
        manter = 1 - self.config.dropout_features
        return torch.bernoulli(torch.full(lote.features.shape, manter, device=lote.features.device)) / manter

    def _vetores_das_palavras(self, lote: Lote) -> torch.Tensor:
        """Soma os vetores dos features de cada palavra -> (B, L, d_features)."""
        pesos = self._dropout_de_features(lote)
        return self.tabela(lote.features, lote.inicios, per_sample_weights=pesos).view(*lote.forma, -1)

    def _saco_do_enunciado(self, lote: Lote) -> torch.Tensor:
        """Atalho "features": média dos pesos de todos os features do enunciado -> (B, n_intents)."""
        inicios = lote.inicios.view(*lote.forma)[:, 0]                    # onde começam os features de cada enunciado
        fins = torch.cat([inicios[1:], inicios.new_tensor([len(lote.features)])])
        soma = self.atalho(lote.features, inicios, per_sample_weights=self._dropout_de_features(lote))
        return soma / (fins - inicios).clamp(min=1)[:, None]

    def _esconder(self, x: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Escolhe 15% das palavras; destas, 70% viram o vetor de máscara, 10% viram outra palavra, 20% ficam."""
        escolhidas = (torch.rand(real.shape, device=x.device) < self.config.p_mascara) & real
        sorte = torch.rand(real.shape, device=x.device)
        vira_mascara = escolhidas & (sorte < 0.7)
        vira_outra = escolhidas & (sorte >= 0.7) & (sorte < 0.8)

        reais = x[real]                                                   # (n_palavras, d_features)
        outras = reais[torch.randint(len(reais), real.shape, device=x.device)]
        x = torch.where(vira_mascara[..., None], self.vetor_mascara.to(x.dtype), x)
        x = torch.where(vira_outra[..., None], outras, x)
        return x, escolhidas

    def codificar(self, lote: Lote, esconder: bool = False) -> dict[str, torch.Tensor]:
        """Roda B, C e D. Devolve a saída de cada palavra e a do CLS."""
        linhas = torch.arange(lote.forma[0], device=lote.comprimentos.device)
        posicoes = torch.arange(lote.forma[1] + 1, device=lote.comprimentos.device)
        real = posicoes[None, :-1] < lote.comprimentos[:, None]          # (B, L): palavra de verdade
        valido = posicoes[None, :] <= lote.comprimentos[:, None]         # (B, L+1): palavra ou CLS

        originais = self._vetores_das_palavras(lote)                      # (B, L, d_features)
        x, escolhidas = self._esconder(originais, real) if esconder else (originais, None)

        # CLS: a média dos vetores das palavras do enunciado (um resumo "saco de features"),
        # colocado logo depois da última palavra.
        media = (x * real[..., None]).sum(1) / lote.comprimentos[:, None]
        sequencia = F.pad(self.projecao(x), (0, 0, 0, 1))                 # (B, L+1, d_modelo)
        sequencia[linhas, lote.comprimentos] = self.projecao(media) + self.marca_cls

        sequencia = self.dropout(sequencia)
        for bloco in self.blocos:
            sequencia = bloco(sequencia, valido)
        sequencia = self.norma_final(sequencia)

        return {
            "palavras": sequencia[:, :-1],                                # (B, L, d_modelo)
            "cls": sequencia[linhas, lote.comprimentos],                  # (B, d_modelo)
            "real": real,
            "originais": originais,
            "escolhidas": escolhidas,
            "media": media,
            "saco": self._saco_do_enunciado(lote) if self.config.atalho_intencao == "features" else None,
        }

    # ------------------------------------------------------------------
    # E, F: as duas respostas
    # ------------------------------------------------------------------

    def _pontuar(self, saida: dict) -> tuple[torch.Tensor, torch.Tensor]:
        intencoes = self.cls_para_rotulo(saida["cls"]) @ self.vetor_da_intencao.T   # (B, n_intents)
        if self.config.atalho_intencao == "resumo":
            intencoes = intencoes + self.atalho(saida["media"])
        elif self.config.atalho_intencao == "features":
            intencoes = intencoes + saida["saco"]
        emissoes = self.emissao(saida["palavras"])                                   # (B, L, n_tags)
        return intencoes, emissoes

    def forward(self, lote: Lote) -> tuple[torch.Tensor, torch.Tensor]:
        """Pontuação de cada intenção (B, n_intents) e de cada tag por palavra (B, L, n_tags)."""
        return self._pontuar(self.codificar(lote))

    @torch.no_grad()
    def prever(self, lote: Lote) -> tuple[list[int], list[list[int]]]:
        """A intenção mais pontuada e a melhor sequência de tags de cada enunciado."""
        saida = self.codificar(lote)
        intencoes, emissoes = self._pontuar(saida)
        if self.config.usar_crf:
            caminhos = self.crf.decodificar(emissoes, saida["real"])
        else:
            melhores = emissoes.argmax(dim=-1).tolist()
            caminhos = [linha[:comprimento] for linha, comprimento in zip(melhores, lote.comprimentos.tolist())]
        return intencoes.argmax(dim=1).tolist(), caminhos

    # ------------------------------------------------------------------
    # A perda conjunta: L = L_intenção + L_entidades + L_máscara
    # ------------------------------------------------------------------

    def perda(self, lote: Lote) -> dict[str, torch.Tensor]:
        c = self.config
        saida = self.codificar(lote, esconder=c.peso_mascara > 0)
        intencoes, emissoes = self._pontuar(saida)

        zero = emissoes.new_zeros(())
        l_intencao = self._perda_intencao(intencoes, lote) if c.peso_intencao > 0 else zero
        l_entidades = self._perda_entidades(emissoes, lote.tags, saida["real"]) if c.peso_entidades > 0 else zero
        l_mascara = self._perda_mascara(saida, lote) if c.peso_mascara > 0 else zero

        total = c.peso_intencao * l_intencao + c.peso_entidades * l_entidades + c.peso_mascara * l_mascara
        return {"total": total, "intencao": l_intencao, "entidades": l_entidades, "mascara": l_mascara}

    def _perda_intencao(self, intencoes: torch.Tensor, lote: Lote) -> torch.Tensor:
        """Com rótulo: a intenção certa contra todas as outras. Na destilação: a distribuição do professor.

        Com rótulo, o artigo do DIET sorteia parte das intenções erradas; com poucas
        dezenas de intenções, usar todas é a versão exata da mesma perda.
        """
        c = self.config
        if c.temperatura <= 0:
            return F.cross_entropy(intencoes, lote.intents, label_smoothing=c.suavizacao)
        if lote.logits_professor is None:
            raise ValueError("Com `temperatura` > 0 o lote precisa trazer `logits_professor` (os logits de intenção do professor).")
        suave = perda_de_destilacao(intencoes, lote.logits_professor, c.temperatura)
        if c.peso_rotulo == 0:
            return suave
        # Hinton et al.: soma pesada do alvo suave com o rótulo rígido, com peso menor no rótulo
        rigida = F.cross_entropy(intencoes, lote.intents, label_smoothing=c.suavizacao)
        return (1 - c.peso_rotulo) * suave + c.peso_rotulo * rigida

    def _perda_entidades(self, emissoes: torch.Tensor, tags: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        if self.config.usar_crf:
            return self.crf.perda(emissoes, tags, real).mean()
        # sem CRF: entropia cruzada palavra por palavra, somada no enunciado (mesma escala da perda do CRF)
        return F.cross_entropy(emissoes[real], tags[real], reduction="sum") / len(tags)

    def _perda_mascara(self, saida: dict, lote: Lote) -> torch.Tensor:
        """Para cada palavra escondida, reconhecer a original entre as outras escondidas do lote."""
        escolhidas = saida["escolhidas"]
        if escolhidas.sum() < 2:
            return saida["cls"].new_zeros(())
        previsto = self.saida_para_token(saida["palavras"][escolhidas])     # (M, d_rotulo)
        original = self.entrada_para_token(saida["originais"][escolhidas])  # (M, d_rotulo)
        semelhanca = previsto @ original.T                                  # (M, M); a diagonal é o par certo
        palavra = lote.palavras[escolhidas]
        repetida = palavra[:, None] == palavra[None, :]
        repetida.fill_diagonal_(False)
        semelhanca = semelhanca.masked_fill(repetida, float("-inf"))        # a mesma palavra não conta como errada
        return F.cross_entropy(semelhanca, torch.arange(len(palavra), device=palavra.device))

    # ------------------------------------------------------------------

    def parametros_por_parte(self) -> dict[str, int]:
        """Onde estão os parâmetros. As partes somam o total do modelo."""
        conta = lambda *modulos: sum(p.numel() for m in modulos for p in (m.parameters() if isinstance(m, nn.Module) else [m]))
        return {
            "B. Tabela de features": conta(self.tabela),
            "C. Projeção e CLS": conta(self.projecao, self.marca_cls),
            "D. Encoder": conta(self.blocos, self.norma_final),
            "E. Cabeça de intenção": conta(self.cls_para_rotulo, self.vetor_da_intencao, *([self.atalho] if self.atalho else [])),
            "F. Cabeça de entidades (CRF)": conta(self.emissao, self.crf),
            "G. Cabeça de máscara (só no treino)": conta(self.saida_para_token, self.entrada_para_token, self.vetor_mascara),
        }
