"""O encoder no contrato da régua: recebe textos, devolve `Previsao`.

Junta as três peças do sistema: o alinhador (texto -> subtokens), a rede
(subtokens -> pontuações) e o inventário de rótulos (índices -> nomes). É o
mesmo contrato de `SistemaDiet`: `lote`, `prever`, `parametros`, `salvar` e
`carregar`. Por isso a régua (`avaliacao/qualidade.py`) e a medida de
latência (`avaliacao/eficiencia.py`) servem sem mudança.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao.previsao import Previsao
from especialista.modelos.encoder.modelo import Config, EncoderConjunto
from especialista.modelos.encoder.subtokens import Alinhador, Lote


def _da_biblioteca(origem, revisao: str | None = None):
    """O encoder e o tokenizador, lidos do Hugging Face (ou de uma pasta), sem a cabeça do pré-treino."""
    import transformers

    falante = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()   # a biblioteca avisa que sobram os pesos da cabeça do pré-treino; é o esperado
    try:
        extra = {"revision": revisao} if revisao else {}
        encoder = transformers.AutoModel.from_pretrained(origem, add_pooling_layer=False, **extra)
        tokenizador = transformers.AutoTokenizer.from_pretrained(origem, **extra)
    finally:
        transformers.logging.set_verbosity(falante)
    return encoder, tokenizador


class SistemaEncoder:
    nome = "XLM-R base com cabeça conjunta"

    def __init__(
        self,
        modelo: EncoderConjunto,
        tokenizador,
        intents: list[str],
        tags: list[str],
        dispositivo: str = "cpu",
        tamanho_do_lote: int = 256,
    ):
        self.modelo = modelo.to(dispositivo)
        self.tokenizador = tokenizador
        self.alinhador = Alinhador(tokenizador, modelo.config.max_subtokens)
        self.intents = intents
        self.tags = tags
        self.dispositivo = dispositivo
        self.tamanho_do_lote = tamanho_do_lote
        self._indice_intent = {nome: i for i, nome in enumerate(intents)}
        self._indice_tag = {nome: i for i, nome in enumerate(tags)}

    @classmethod
    def novo(
        cls,
        rotulos: dict,
        config: Config = Config(),
        semente: int = 0,
        dispositivo: str = "cpu",
        encoder=None,
        tokenizador=None,
    ) -> "SistemaEncoder":
        """O encoder pré-treinado com as duas cabeças recém-sorteadas (ainda sem ajuste fino).

        `encoder` e `tokenizador` servem para os testes, que montam um encoder
        minúsculo em memória. Sem eles, baixa (ou lê do cache) o `config.modelo_id`.
        """
        if encoder is None:
            encoder, tokenizador = _da_biblioteca(config.modelo_id, config.revisao)
        torch.manual_seed(semente)
        modelo = EncoderConjunto(config, encoder, len(rotulos["intents"]), len(rotulos["tags"]))
        return cls(modelo, tokenizador, rotulos["intents"], rotulos["tags"], dispositivo)

    def lote(self, dados: pd.DataFrame) -> Lote:
        """Enunciados com rótulos -> lote de treino.

        Na destilação com logits, `dados` traz também a coluna `logits_professor`:
        um vetor por enunciado, com a pontuação que o professor deu a cada
        intenção, na ordem de `self.intents`. Ela segue junto no lote.
        """
        lote = self.alinhador.montar(
            [list(tokens) for tokens in dados["tokens"]],
            intents=[self._indice_intent[i] for i in dados["intent"]],
            tags=[[self._indice_tag[t] for t in tags] for tags in dados["tags"]],
        )
        if "logits_professor" in dados.columns:
            logits = torch.tensor(np.stack(dados["logits_professor"].to_numpy()), dtype=torch.float32)
            if logits.shape != (len(dados), len(self.intents)):
                raise ValueError(f"`logits_professor` tem forma {tuple(logits.shape)}; o esperado é {(len(dados), len(self.intents))}.")
            lote.logits_professor = logits
        return lote.para(self.dispositivo)

    def prever(self, textos: list[str]) -> list[Previsao]:
        self.modelo.eval()
        previsoes = []
        for inicio in range(0, len(textos), self.tamanho_do_lote):
            enunciados = [texto.split() for texto in textos[inicio : inicio + self.tamanho_do_lote]]
            intents, caminhos = self.modelo.prever(self.alinhador.montar(enunciados).para(self.dispositivo))
            previsoes.extend(
                # palavra sem subtoken (índice −1) fica fora de qualquer entidade
                Previsao.de_tags(self.intents[i], [self.tags[t] if t >= 0 else "O" for t in caminho])
                for i, caminho in zip(intents, caminhos)
            )
        return previsoes

    # ------------------------------------------------------------------
    # A tabela de subtokens durante o treino (ver `TabelaDividida`)
    # ------------------------------------------------------------------

    def preparar_treino(self, treino: pd.DataFrame) -> int:
        """Separa as linhas da tabela que este treino pode mudar. Devolve quantas são."""
        ids = self.alinhador.ids_usados(treino["tokens"])
        self.modelo.dividir_tabela(ids)
        return len(ids)

    def concluir_treino(self) -> None:
        self.modelo.fundir_tabela()

    # ------------------------------------------------------------------

    @property
    def parametros(self) -> int:
        return sum(p.numel() for p in self.modelo.parameters())

    def salvar(self, pasta: Path) -> Path:
        """Grava o encoder e o tokenizador no formato da biblioteca, e as cabeças e os rótulos ao lado."""
        self.modelo.fundir_tabela()
        pasta.mkdir(parents=True, exist_ok=True)
        self.modelo.encoder.save_pretrained(pasta / "encoder")
        self.tokenizador.save_pretrained(pasta / "encoder")   # na mesma pasta, como a biblioteca espera
        cabecas = {nome: tensor for nome, tensor in self.modelo.state_dict().items() if not nome.startswith("encoder.")}
        torch.save(cabecas, pasta / "cabecas.pt")
        (pasta / "config.json").write_text(
            json.dumps({"config": self.modelo.config.como_dict(), "intents": self.intents, "tags": self.tags},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return pasta

    @classmethod
    def carregar(cls, pasta: Path, dispositivo: str = "cpu") -> "SistemaEncoder":
        dados = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
        encoder, tokenizador = _da_biblioteca(pasta / "encoder")
        modelo = EncoderConjunto(Config(**dados["config"]), encoder, len(dados["intents"]), len(dados["tags"]))
        modelo.load_state_dict(torch.load(pasta / "cabecas.pt", map_location="cpu", weights_only=True), strict=False)
        return cls(modelo.eval(), tokenizador, dados["intents"], dados["tags"], dispositivo)


def bytes_em_disco(pasta: Path) -> int:
    """O tamanho do sistema gravado por `SistemaEncoder.salvar`: a soma de todos os arquivos da pasta."""
    return sum(arquivo.stat().st_size for arquivo in pasta.rglob("*") if arquivo.is_file())
