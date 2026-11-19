"""A LLM fechada do projeto: Claude Haiku 5.5, pela API da Anthropic.

Dois jeitos de chamar:

- **Direto** (`ProfessorClaude.chamar`): um pedido por vez, cronometrado.
  É como um agente usaria a LLM, e é o que mede latência e custo.
- **Em lote** (`enviar_com_cuidado`, `coletar`): milhares de pedidos de uma
  vez, sem pressa, pela metade do preço. É como se rotula o treino.

Toda resposta vira um *registro* (um dicionário simples) gravado em disco.
A avaliação lê os registros; a API só é chamada de novo se eles forem
apagados.

Sobre a configuração da chamada:

- O Claude Haiku 5.5 não aceita `temperature`: a saída é controlada pelo
  formato estruturado, não pela amostragem.
- A resposta segue um JSON Schema imposto na geração (`output_config.format`).
- O raciocínio do modelo ("thinking") vem ligado por padrão; aqui ele é
  uma opção do `Ajuste`, porque muda custo e latência.
- As instruções, que são fixas, levam a marca de cache.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from especialista.avaliacao.previsao import Previsao
from especialista.caminhos import RAIZ
from especialista.professores import prompt, resposta
from especialista.professores.demonstracoes import K, Demonstracoes

MODELO = "claude-haiku-5-5"
MAX_TOKENS = 1024
PRAZO_EM_SEGUNDOS = 60.0   # de uma chamada direta
TENTATIVAS = 2             # novas tentativas automáticas do SDK em erro passageiro

# Preço de lista em dólares por milhão de tokens, para prompts de até 100 mil tokens.
# Fonte: tabela de modelos da Anthropic, consultada em 16/11/2026.
PRECO = {"entrada": 0.10, "saida": 0.50, "cache_lido": 0.01, "cache_escrito": 0.125}
DESCONTO_DO_LOTE = 0.5


@dataclass(frozen=True)
class Ajuste:
    """O que pode variar na chamada. O padrão é o de um subagente que precisa responder rápido."""

    k: int = K                  # demonstrações por pedido; 0 = sem exemplos
    raciocinio: bool = False    # False desliga o "thinking" do modelo
    esforco: str = "low"        # low | medium | high

    def como_dict(self) -> dict:
        return asdict(self)


def carregar_chave() -> bool:
    """Lê `ANTHROPIC_API_KEY` de um arquivo `.env` na raiz do projeto, se ela não estiver no ambiente."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    arquivo = RAIZ / ".env"
    if arquivo.exists():
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            nome, _, valor = linha.strip().partition("=")
            if nome.replace("export ", "").strip() == "ANTHROPIC_API_KEY" and valor.strip():
                os.environ["ANTHROPIC_API_KEY"] = valor.strip().strip("'\"")
                return True
    return False


def custo(uso: dict, em_lote: bool = False) -> float:
    """Custo em dólares de uma resposta, a partir dos tokens que a API reportou."""
    dolares = sum(uso.get(chave, 0) * preco for chave, preco in PRECO.items()) / 1e6
    return dolares * (DESCONTO_DO_LOTE if em_lote else 1.0)


def resumir_uso(registros: list[dict], em_lote: bool = False) -> dict:
    """Pedidos, tokens somados por tipo e custo em dólares de um conjunto de registros."""
    tokens = {chave: sum(r["uso"].get(chave, 0) for r in registros) for chave in PRECO}
    return {
        "pedidos": len(registros),
        "com_resposta": sum(r["texto"] is not None for r in registros),
        **tokens,
        "dolares": sum(custo(r["uso"], em_lote) for r in registros),
        "em_lote": em_lote,
    }


def _registro(identificador: str, mensagem=None, ms: float | None = None, erro: str | None = None) -> dict:
    """A parte de uma resposta que o projeto guarda."""
    if mensagem is None:
        return {"id": identificador, "texto": None, "parada": None, "uso": {}, "ms": ms, "erro": erro}
    texto = next((bloco.text for bloco in mensagem.content if bloco.type == "text"), None)
    uso = mensagem.usage
    return {
        "id": identificador,
        "texto": texto if mensagem.stop_reason == "end_turn" else None,  # cortada ou recusada não vale como resposta
        "parada": mensagem.stop_reason,
        "uso": {
            "entrada": uso.input_tokens,
            "saida": uso.output_tokens,
            "cache_lido": uso.cache_read_input_tokens or 0,
            "cache_escrito": uso.cache_creation_input_tokens or 0,
        },
        "ms": ms,
        "erro": erro,
    }


class ProfessorClaude:
    """A LLM fechada no contrato da régua (`prever`) e como rotuladora (`pedido`, `interpretar`)."""

    def __init__(self, demonstracoes: Demonstracoes, rotulos: dict, ajuste: Ajuste = Ajuste(), cliente=None,
                 diario: Path | None = None):
        self.demonstracoes = demonstracoes
        self.diario = diario              # se dado, cada chamada direta é anexada a este arquivo assim que volta
        self.intents = rotulos["intents"]
        self.tipos = rotulos["slot_types"]
        self.ajuste = ajuste
        self.nome = "Claude Haiku 5.5"
        self._cliente = cliente
        self.registros: list[dict] = []   # tudo o que foi respondido em chamadas diretas, na ordem
        self._instrucoes = prompt.instrucoes(self.intents, self.tipos)
        self._esquema = prompt.esquema(self.intents, self.tipos)

    @property
    def cliente(self):
        if self._cliente is None:
            if not carregar_chave():
                raise RuntimeError("Falta a chave: defina ANTHROPIC_API_KEY no ambiente ou num arquivo .env na raiz do projeto.")
            # prazo e tentativas de quem espera uma resposta curta: sem isso, uma conexão caída prende a chamada por 10 minutos
            self._cliente = anthropic.Anthropic(max_retries=TENTATIVAS, timeout=PRAZO_EM_SEGUNDOS)
        return self._cliente

    def pedido(self, texto: str, excluir_id: int | None = None) -> dict:
        """Os parâmetros da chamada para um pedido do usuário."""
        exemplos = self.demonstracoes.mais_parecidos(texto, self.ajuste.k, excluir_id)
        return {
            "model": MODELO,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": self._instrucoes, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt.mensagem(texto, exemplos)}],
            "thinking": {"type": "adaptive"} if self.ajuste.raciocinio else {"type": "disabled"},
            "output_config": {"effort": self.ajuste.esforco, "format": {"type": "json_schema", "schema": self._esquema}},
        }

    def chamar(self, texto: str, excluir_id: int | None = None, identificador: str = "") -> dict:
        """Uma chamada direta, cronometrada do lado de quem chama. Nunca levanta erro: falha vira registro."""
        parametros = self.pedido(texto, excluir_id)
        inicio = time.perf_counter()
        ms = lambda: (time.perf_counter() - inicio) * 1000
        try:
            registro = _registro(identificador, self.cliente.messages.create(**parametros), ms=ms())
        except anthropic.AuthenticationError:
            raise  # chave errada não é uma resposta ruim: é um problema de configuração
        except anthropic.RateLimitError as e:
            registro = _registro(identificador, ms=ms(), erro=f"limite de taxa: {e.message}")
        except anthropic.APIStatusError as e:
            registro = _registro(identificador, ms=ms(), erro=f"{e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            registro = _registro(identificador, ms=ms(), erro=f"conexão: {e}")
        self.registros.append(registro)
        if self.diario is not None:
            self.diario.parent.mkdir(parents=True, exist_ok=True)
            with self.diario.open("a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return registro

    def interpretar(self, registro: dict, texto: str) -> tuple[Previsao, dict]:
        """Do registro de uma resposta para a previsão e o diagnóstico da leitura."""
        return resposta.ler(registro["texto"], texto, self.intents, self.tipos)

    def prever(self, textos: list[str]) -> list[Previsao]:
        """O contrato da régua: um pedido por vez, como um agente faria."""
        return [self.interpretar(self.chamar(texto), texto)[0] for texto in textos]


# --------------------------------------------------------------------------
# Em lote
# --------------------------------------------------------------------------

def enviar_lote(professor: ProfessorClaude, itens: list[tuple[str, str, int | None]]) -> str:
    """Envia um lote. `itens` é uma lista de (identificador, texto, id a excluir das demonstrações)."""
    # Sem nova tentativa automática: se a conexão cair depois de o lote ter sido aceito,
    # tentar de novo criaria um segundo lote, cobrado em dobro. O envio do treino é grande, daí o prazo longo.
    lote = professor.cliente.with_options(max_retries=0, timeout=1800.0).messages.batches.create(
        requests=[
            Request(custom_id=identificador, params=MessageCreateParamsNonStreaming(**professor.pedido(texto, excluir)))
            for identificador, texto, excluir in itens
        ]
    )
    return lote.id


def lote_ja_criado(professor: ProfessorClaude, pedidos: int, anotados: set[str], desde: datetime) -> str | None:
    """Procura na conta um lote com esse número de pedidos, criado depois de `desde` e ainda não anotado.

    Serve para quando o envio falha sem dizer se o lote foi aceito: reenviar
    às cegas poderia criar um segundo lote, cobrado em dobro.

    A conta não mostra o conteúdo de um lote, só o tamanho e a hora. Por isso
    `anotados` precisa trazer os lotes de TODOS os conjuntos, não só os deste:
    sem isso, um lote de outro conjunto com o mesmo tamanho seria confundido
    com o que se procura.
    """
    for lote in professor.cliente.messages.batches.list(limit=50):
        c = lote.request_counts
        total = c.processing + c.succeeded + c.errored + c.canceled + c.expired
        if total == pedidos and lote.id not in anotados and lote.created_at >= desde:
            return lote.id
    return None


def enviar_com_cuidado(professor: ProfessorClaude, itens: list[tuple[str, str, int | None]], anotados: Callable[[], set[str]],
                       tentativas: int = 3, pausa: float = 30.0) -> str:
    """Envia um lote e, se o envio falhar, só tenta de novo depois de conferir que ele não foi criado.

    `anotados` devolve, na hora da conferência, os lotes que já têm dono.
    """
    for tentativa in range(1, tentativas + 1):
        desde = datetime.now(timezone.utc) - timedelta(minutes=2)   # folga para relógios desalinhados
        try:
            return enviar_lote(professor, itens)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            print(f"envio falhou na tentativa {tentativa}: {type(e).__name__}", flush=True)
            time.sleep(pausa)
            if (achado := lote_ja_criado(professor, len(itens), anotados(), desde)) is not None:
                return achado
    raise RuntimeError(f"O envio de {len(itens)} pedidos falhou {tentativas} vezes e nenhum lote foi criado.")


def situacao(professor: ProfessorClaude, lote_id: str) -> dict:
    lote = professor.cliente.messages.batches.retrieve(lote_id)
    contagem = lote.request_counts
    return {
        "terminou": lote.processing_status == "ended",
        "processando": contagem.processing,
        "certas": contagem.succeeded,
        "com_erro": contagem.errored + contagem.expired + contagem.canceled,
    }


def coletar(professor: ProfessorClaude, lote_id: str) -> list[dict]:
    """Os registros de um lote terminado. A ordem não é garantida: cada um traz o seu identificador."""
    registros = []
    for item in professor.cliente.messages.batches.results(lote_id):
        if item.result.type == "succeeded":
            registros.append(_registro(item.custom_id, item.result.message))
        else:
            registros.append(_registro(item.custom_id, erro=item.result.type))
    return registros


# --------------------------------------------------------------------------
# Registros em disco
# --------------------------------------------------------------------------

def gravar(registros: list[dict], caminho: Path) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return caminho


def ler_em_ordem(caminho: Path) -> list[dict]:
    """Os registros de um arquivo, na ordem em que foram gravados.

    É como se leem chamadas diretas: elas não têm identificador próprio, e
    lidas por identificador virariam um registro só.
    """
    with caminho.open(encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def ler_registros(caminho: Path) -> dict[str, dict]:
    """Os registros de um arquivo, por identificador."""
    with caminho.open(encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}
