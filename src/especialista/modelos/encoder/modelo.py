"""B, C e D: o encoder pré-treinado, as duas cabeças e a perda conjunta.

Referência: Chen et al., "BERT for Joint Intent Classification and Slot
Filling" (JointBERT, arXiv 1902.10909), seção 3.2. O que vem do artigo:

- a intenção sai de uma camada linear sobre o estado do **primeiro token**
  da sequência (no XLM-R, `<s>`);
- a tag de cada palavra sai de uma camada linear sobre o estado do
  **primeiro subtoken** da palavra;
- as duas respostas são treinadas juntas. O artigo escreve o objetivo como
  o produto p(intenção) · Π p(tag de cada palavra); em logaritmo, isso é a
  entropia cruzada da intenção mais a soma das entropias cruzadas das tags
  do enunciado;
- dropout de 0,1 antes das cabeças;
- sem CRF: o artigo testou um CRF por cima e não mediu ganho.

O encoder é o XLM-R base (Conneau et al., arXiv 1911.02116), o mesmo do
artigo do MASSIVE (FitzGerald et al., arXiv 2204.08582). O que vem do artigo
e o que é escolha deste projeto está marcado campo a campo em `Config`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from especialista.modelos.diet.modelo import perda_de_destilacao   # a mesma perda da Etapa 6
from especialista.modelos.encoder.subtokens import Lote


@dataclass(frozen=True)
class Config:
    # --- o encoder: escolha do projeto (PLANO.md), a mesma do artigo do MASSIVE ---
    modelo_id: str = "FacebookAI/xlm-roberta-base"
    revisao: str = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"  # a versão exata dos pesos no Hugging Face
    max_subtokens: int = 128       # limite de segurança; os pedidos do MASSIVE ficam muito abaixo
    # --- do artigo âncora (JointBERT) ---
    dropout: float = 0.1           # antes das duas cabeças
    # --- pesos da perda conjunta (no artigo, soma simples) ---
    peso_intencao: float = 1.0
    peso_entidades: float = 1.0
    # --- destilação com os logits do professor (Hinton et al., arXiv 1503.02531); desligada por padrão ---
    temperatura: float = 0.0       # > 0: a intenção aprende com a distribuição do professor, as duas com esta temperatura
    peso_rotulo: float = 0.0       # na destilação, a parte (de 0 a 1) da perda de intenção que continua vindo do rótulo rígido

    def como_dict(self) -> dict:
        return asdict(self)


class TabelaDividida(nn.Module):
    """A tabela de subtokens com as linhas que o treino pode mudar separadas das que ele não toca.

    A tabela do XLM-R tem 250.002 linhas (uma por subtoken, em 100 idiomas) e
    é a maior parte do modelo. Num treino só em português, quase todas as
    linhas nunca aparecem: o gradiente delas é zero em todo passo, e o Adam
    não as move. Mesmo assim o otimizador guardaria três cópias da tabela
    inteira (gradiente e duas médias), cerca de 2,3 GB na placa de vídeo.

    Aqui só as linhas dos subtokens que aparecem no treino viram parâmetro. As
    outras ficam num bloco fixo. O resultado é o mesmo do treino com a tabela
    inteira (há um teste que compara os dois), com metade da memória.

    Um detalhe para a conta fechar: o decaimento de pesos encolhe um pouco
    todas as linhas a cada passo, inclusive as que não têm gradiente. Para as
    linhas fixas esse encolhimento é um número só, igual para todas, guardado
    em `escala` e aplicado na leitura. `decair` é chamado pelo laço de treino
    a cada passo.
    """

    def __init__(self, tabela: nn.Embedding, ids_treinaveis: torch.Tensor):
        super().__init__()
        ids_treinaveis = ids_treinaveis.to(tabela.weight.device)
        posicao = torch.full((tabela.num_embeddings,), -1, dtype=torch.long, device=tabela.weight.device)
        posicao[ids_treinaveis] = torch.arange(len(ids_treinaveis), device=tabela.weight.device)
        self.padding_idx = tabela.padding_idx
        self.register_buffer("fixa", tabela.weight.detach())                 # (V, d) todas as linhas, como chegaram
        self.register_buffer("posicao", posicao)                             # (V,)   linha em `treinavel`, ou -1
        self.register_buffer("ids_treinaveis", ids_treinaveis)
        self.register_buffer("escala", torch.ones((), device=tabela.weight.device))  # decaimento acumulado das linhas fixas
        self.treinavel = nn.Parameter(tabela.weight.detach()[ids_treinaveis].clone())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        local = self.posicao[ids]
        fixas = F.embedding(ids, self.fixa) * self.escala
        return torch.where((local >= 0)[..., None], self.treinavel[local.clamp(min=0)], fixas)

    @torch.no_grad()
    def decair(self, fator: float) -> None:
        """O que o decaimento de pesos faz com uma linha sem gradiente: multiplica por `fator`."""
        self.escala.mul_(fator)

    @torch.no_grad()
    def fundir(self) -> nn.Embedding:
        """De volta a uma tabela comum, com as linhas treinadas no lugar."""
        self.fixa.mul_(self.escala)
        self.fixa[self.ids_treinaveis] = self.treinavel.detach().to(self.fixa.dtype)
        return nn.Embedding.from_pretrained(self.fixa, freeze=False, padding_idx=self.padding_idx)


class EncoderConjunto(nn.Module):
    #: partes do estado que não mudam durante o treino com a tabela dividida
    CONSTANTES = ("fixa", "posicao", "ids_treinaveis")

    def __init__(self, config: Config, encoder: nn.Module, n_intents: int, n_tags: int):
        super().__init__()
        self.config = config
        self.encoder = encoder                                   # B. o Transformer pré-treinado, sem cabeça nenhuma
        d = encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.intencao = nn.Linear(d, n_intents)                  # C. sobre o primeiro token
        self.entidades = nn.Linear(d, n_tags)                    # D. sobre o primeiro subtoken de cada palavra
        for cabeca in (self.intencao, self.entidades):           # o mesmo sorteio inicial das camadas do encoder
            nn.init.normal_(cabeca.weight, std=encoder.config.initializer_range)
            nn.init.zeros_(cabeca.bias)

    # ------------------------------------------------------------------
    # B, C e D: do lote às pontuações
    # ------------------------------------------------------------------

    def forward(self, lote: Lote) -> tuple[torch.Tensor, torch.Tensor]:
        """Pontuação de cada intenção (B, n_intents) e de cada tag por palavra (B, L, n_tags)."""
        estados = self.encoder(input_ids=lote.ids, attention_mask=lote.atencao).last_hidden_state   # (B, S, d)
        estados = self.dropout(estados)
        do_inicio = estados[:, 0]                                                                    # (B, d)
        indice = lote.primeiros[..., None].expand(-1, -1, estados.size(-1))
        das_palavras = estados.gather(1, indice)                                                     # (B, L, d)
        # as perdas e a escolha final são sempre em 32 bits, mesmo que o encoder rode em precisão mista
        return self.intencao(do_inicio).float(), self.entidades(das_palavras).float()

    @torch.no_grad()
    def prever(self, lote: Lote) -> tuple[list[int], list[list[int]]]:
        """A intenção mais pontuada e a tag mais pontuada de cada palavra (−1 na palavra sem subtoken)."""
        intencoes, emissoes = self(lote)
        melhores = emissoes.argmax(dim=-1).masked_fill(~lote.real, -1).tolist()
        caminhos = [linha[:comprimento] for linha, comprimento in zip(melhores, lote.comprimentos.tolist())]
        return intencoes.argmax(dim=1).tolist(), caminhos

    # ------------------------------------------------------------------
    # A perda conjunta: L = L_intenção + L_entidades
    # ------------------------------------------------------------------

    def perda(self, lote: Lote) -> dict[str, torch.Tensor]:
        c = self.config
        intencoes, emissoes = self(lote)
        zero = emissoes.new_zeros(())
        l_intencao = self._perda_intencao(intencoes, lote) if c.peso_intencao > 0 else zero
        l_entidades = self._perda_entidades(emissoes, lote) if c.peso_entidades > 0 else zero
        total = c.peso_intencao * l_intencao + c.peso_entidades * l_entidades
        return {"total": total, "intencao": l_intencao, "entidades": l_entidades}

    def _perda_intencao(self, intencoes: torch.Tensor, lote: Lote) -> torch.Tensor:
        """Com rótulo: entropia cruzada. Na destilação: a distribuição do professor (a mesma perda da Etapa 6)."""
        c = self.config
        if c.temperatura <= 0:
            return F.cross_entropy(intencoes, lote.intents)
        if lote.logits_professor is None:
            raise ValueError("Com `temperatura` > 0 o lote precisa trazer `logits_professor` (os logits de intenção do professor).")
        suave = perda_de_destilacao(intencoes, lote.logits_professor, c.temperatura)
        if c.peso_rotulo == 0:
            return suave
        return (1 - c.peso_rotulo) * suave + c.peso_rotulo * F.cross_entropy(intencoes, lote.intents)

    def _perda_entidades(self, emissoes: torch.Tensor, lote: Lote) -> torch.Tensor:
        """Entropia cruzada palavra por palavra, somada no enunciado; depois, a média do lote.

        É o logaritmo do produto do artigo, e a mesma escala da perda de
        entidades da rede do zero. Palavra sem subtoken não entra.
        """
        return F.cross_entropy(emissoes[lote.real], lote.tags[lote.real], reduction="sum") / len(lote.tags)

    # ------------------------------------------------------------------
    # A tabela de subtokens durante o treino
    # ------------------------------------------------------------------

    @property
    def tabela_dividida(self) -> TabelaDividida | None:
        tabela = self.encoder.get_input_embeddings()
        return tabela if isinstance(tabela, TabelaDividida) else None

    def dividir_tabela(self, ids_treinaveis: torch.Tensor) -> None:
        """Só as linhas de `ids_treinaveis` passam a ser parâmetro (ver `TabelaDividida`)."""
        if self.tabela_dividida is not None:
            raise RuntimeError("A tabela já está dividida.")
        self.encoder.set_input_embeddings(TabelaDividida(self.encoder.get_input_embeddings(), ids_treinaveis))

    def fundir_tabela(self) -> None:
        """Desfaz `dividir_tabela`: o encoder volta a ser um modelo comum da biblioteca."""
        if self.tabela_dividida is not None:
            self.encoder.set_input_embeddings(self.tabela_dividida.fundir())

    def copiar_estado(self) -> dict[str, torch.Tensor]:
        """Uma cópia de tudo o que o treino muda, para guardar o melhor ponto. Volta com `load_state_dict(..., strict=False)`.

        A cópia fica na memória comum, não na da placa de vídeo, que é a mais disputada.
        """
        return {
            nome: tensor.detach().to("cpu", copy=True)
            for nome, tensor in self.state_dict().items()
            if nome.rsplit(".", 1)[-1] not in self.CONSTANTES
        }

    # ------------------------------------------------------------------

    def parametros_por_parte(self) -> dict[str, int]:
        """Onde estão os parâmetros (com a tabela inteira, fora do treino). As partes somam o total do modelo."""
        conta = lambda modulo: sum(p.numel() for p in modulo.parameters())
        tabela = conta(self.encoder.get_input_embeddings())
        return {
            "A. Tabela de subtokens": tabela,
            "A. Posições e normalização da entrada": conta(self.encoder.embeddings) - tabela,
            "B. Corpo do Transformer": conta(self.encoder.encoder),
            "C. Cabeça de intenção": conta(self.intencao),
            "D. Cabeça de entidades": conta(self.entidades),
        }
