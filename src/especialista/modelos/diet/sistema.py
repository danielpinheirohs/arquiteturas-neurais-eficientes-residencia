"""O modelo no contrato da régua: recebe textos, devolve `Previsao`.

Junta as três peças que formam o sistema: featurizador (texto -> ids),
rede (ids -> pontuações) e inventário de rótulos (índices -> nomes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from especialista.avaliacao.previsao import Previsao
from especialista.modelos.diet.features import Featurizador, Lote
from especialista.modelos.diet.modelo import Config, Diet


class SistemaDiet:
    nome = "DIET do zero"

    def __init__(
        self,
        modelo: Diet,
        featurizador: Featurizador,
        intents: list[str],
        tags: list[str],
        dispositivo: str = "cpu",
        tamanho_do_lote: int = 256,
    ):
        self.modelo = modelo.to(dispositivo)
        self.featurizador = featurizador
        self.intents = intents
        self.tags = tags
        self.dispositivo = dispositivo
        self.tamanho_do_lote = tamanho_do_lote
        self._indice_intent = {nome: i for i, nome in enumerate(intents)}
        self._indice_tag = {nome: i for i, nome in enumerate(tags)}

    @classmethod
    def novo(
        cls,
        treino: pd.DataFrame,
        rotulos: dict,
        config: Config = Config(),
        semente: int = 0,
        contagem_minima: int = 2,
        dispositivo: str = "cpu",
    ) -> "SistemaDiet":
        """Um sistema com pesos iniciais (ainda não treinado). O vocabulário de features vem do treino."""
        torch.manual_seed(semente)
        featurizador = Featurizador.ajustar(treino["tokens"], contagem_minima=contagem_minima)
        modelo = Diet(config, len(featurizador), len(rotulos["intents"]), rotulos["tags"])
        return cls(modelo, featurizador, rotulos["intents"], rotulos["tags"], dispositivo)

    def lote(self, dados: pd.DataFrame) -> Lote:
        """Enunciados com rótulos -> lote de treino.

        Na destilação com logits, `dados` traz também a coluna `logits_professor`:
        um vetor por enunciado, com a pontuação que o professor deu a cada
        intenção, na ordem de `self.intents`. Ela segue junto no lote.
        """
        lote = self.featurizador.montar(
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
            intents, caminhos = self.modelo.prever(self.featurizador.montar(enunciados).para(self.dispositivo))
            previsoes.extend(
                Previsao.de_tags(self.intents[i], [self.tags[t] for t in caminho])
                for i, caminho in zip(intents, caminhos)
            )
        return previsoes

    @property
    def parametros(self) -> int:
        return sum(p.numel() for p in self.modelo.parameters())

    def salvar(self, pasta: Path) -> Path:
        pasta.mkdir(parents=True, exist_ok=True)
        torch.save(self.modelo.state_dict(), pasta / "pesos.pt")
        self.featurizador.salvar(pasta / "features.json")
        (pasta / "config.json").write_text(
            json.dumps({"config": self.modelo.config.como_dict(), "intents": self.intents, "tags": self.tags},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return pasta

    @classmethod
    def carregar(cls, pasta: Path, dispositivo: str = "cpu") -> "SistemaDiet":
        dados = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
        featurizador = Featurizador.carregar(pasta / "features.json")
        modelo = Diet(Config(**dados["config"]), len(featurizador), len(dados["intents"]), dados["tags"])
        modelo.load_state_dict(torch.load(pasta / "pesos.pt", map_location=dispositivo, weights_only=True))
        return cls(modelo, featurizador, dados["intents"], dados["tags"], dispositivo)
