"""A LLM aberta do projeto: um modelo de pesos abertos rodando na GPU local, em 4 bits.

Recebe exatamente o mesmo prompt da LLM fechada (`prompt.py`) e passa pela
mesma leitura de resposta (`resposta.py`). O modelo é carregado com a
biblioteca `transformers`, quantizado em 4 bits (NF4, Dettmers et al.,
arXiv 2305.14314) para caber na placa.

Ter os pesos em casa dá o que a API não dá: a probabilidade que o modelo
atribui a CADA intenção, e não só a intenção escolhida. Uma resposta sai
em dois passos:

1. **Pontuar as intenções.** A resposta começa sempre por `{"intent": "`.
   Nesse ponto, mede-se a probabilidade de o modelo continuar com cada um
   dos 60 nomes de intenção. Um nome ocupa vários tokens, então a
   probabilidade do nome é o produto das probabilidades dos seus tokens
   (a soma dos logaritmos). O resultado é um vetor de 60 números: os
   "logits de intenção" que o aluno B2 vai usar na Etapa 6.
2. **Gerar as entidades.** A intenção mais provável é escrita na resposta,
   seguida de `", "slots":`, e o modelo continua dali, um token por vez,
   sempre o mais provável (decodificação gulosa), até fechar o JSON.

O texto forçado para em `"slots":`, antes do colchete. Forçar também o `[`
parece inofensivo e não é: para uma lista vazia o modelo escreve `[]` como
um token só, e com o `[` já escrito ele fica fora do que conhece (fecha
errado e inventa entidades para preencher a lista).

A intenção da resposta é sempre a mais provável das 60: o modelo não tem
como inventar uma intenção fora da lista. É o equivalente local do formato
imposto por esquema que a LLM fechada usa.

Três reaproveitamentos tornam isso rápido. Todos se apoiam na memória de
chaves e valores do modelo: o que ele já calculou para um trecho de texto
não precisa ser calculado de novo.

- **As instruções são calculadas uma vez.** Elas são iguais em todos os
  pedidos; a memória delas é copiada para cada lote. É o que o cache de
  prompt faz na API da LLM fechada.
- **O prompt é lido uma vez por pedido.** A pontuação e a geração partem
  da mesma memória.
- **Os 60 nomes são pontuados numa passada só.** Os tokens de todos os
  nomes vão lado a lado, com uma máscara de atenção que deixa cada token
  enxergar o prompt e os tokens anteriores do SEU nome, e nada dos outros.

Para vários pedidos irem juntos num lote, as sequências mais curtas levam
enchimento. O enchimento nunca é enxergado e não conta posição: cada token
recebe a posição que teria se o seu pedido estivesse sozinho.

`pontuar_de_referencia` e `gerar_de_referencia` fazem as mesmas contas do
jeito direto e lento, sem nenhum reaproveitamento. Existem para conferir
que os atalhos não mudam o resultado.
"""

from __future__ import annotations

import copy
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from transformers.utils import logging as registro_do_transformers

from especialista.avaliacao.previsao import Previsao
from especialista.professores import prompt, resposta
from especialista.professores.demonstracoes import K, Demonstracoes

MAX_TOKENS = 200                 # da parte gerada livremente (as entidades)
INICIO = '{"intent": "'          # toda resposta começa assim
DEPOIS = '", "slots":'           # o que vem depois do nome da intenção; a lista quem abre é o modelo


def fechou(resposta: str) -> bool:
    """A resposta já é um objeto JSON completo?"""
    if not resposta.rstrip().endswith("}"):
        return False
    try:
        json.loads(resposta)
        return True
    except ValueError:
        return False


class _Estado:
    """O que o modelo já leu de cada pedido do lote."""

    def __init__(self, memoria: DynamicCache, valido: torch.Tensor, posicao: torch.Tensor):
        self.memoria = memoria      # chaves e valores de tudo o que já foi lido
        self.valido = valido        # (lote, lidos): falso onde há enchimento
        self.posicao = posicao      # (lote, lidos): a posição de cada token no seu pedido
        self.lidos = valido.sum(dim=1)   # (lote,): quantos tokens de verdade cada pedido já tem


class ProfessorLocal:
    """A LLM aberta no contrato da régua (`prever`) e como rotuladora (`responder`)."""

    def __init__(self, modelo_id: str, demonstracoes: Demonstracoes, rotulos: dict, k: int = K, nome: str | None = None,
                 modelo=None, tokenizador=None):
        self.modelo_id = modelo_id
        self.nome = nome or modelo_id.split("/")[-1]
        self.demonstracoes = demonstracoes
        self.intents = rotulos["intents"]
        self.tipos = rotulos["slot_types"]
        self.k = k
        self.registros: list[dict] = []
        self._instrucoes = prompt.instrucoes(self.intents, self.tipos)

        registro_do_transformers.disable_progress_bar()   # a barra de carregamento enche o registro de linhas
        self.tokenizador = tokenizador or AutoTokenizer.from_pretrained(modelo_id)
        self.modelo = modelo or AutoModelForCausalLM.from_pretrained(
            modelo_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
        )
        self.modelo.eval()
        self.dispositivo = next(self.modelo.parameters()).device
        self.vram_do_modelo_gb = torch.cuda.memory_allocated() / 1e9 if self.dispositivo.type == "cuda" else 0.0
        self._tipo = self.modelo.get_input_embeddings().weight.dtype
        self._pad = self.tokenizador.pad_token_id if self.tokenizador.pad_token_id is not None else self.tokenizador.eos_token_id
        fins = self.modelo.generation_config.eos_token_id
        self._fins = {f for f in (fins if isinstance(fins, list) else [fins]) if f is not None} | {self.tokenizador.eos_token_id}
        config = getattr(self.modelo.config, "text_config", None) or self.modelo.config
        tipos_de_camada = set(getattr(config, "layer_types", None) or [])
        # Gemma 3 tem camadas que só enxergam os últimos `janela` tokens
        self._janela = config.sliding_window if "sliding_attention" in tipos_de_camada else None
        self._preparar_intencoes()
        self._memoria_das_instrucoes: DynamicCache | None = None
        self._prefixo: list[int] = []

    # ------------------------------------------------------------------ preparo

    def _preparar_intencoes(self) -> None:
        """Quebra cada resposta possível em tokens e monta o bloco que pontua os 60 nomes de uma vez."""
        tok = self.tokenizador
        ids = lambda texto: tok(texto, add_special_tokens=False).input_ids
        inteiras = [ids(INICIO + intent + DEPOIS) for intent in self.intents]
        self._inicio = ids(INICIO)
        comum = len(self._inicio)
        if any(seq[:comum] != self._inicio for seq in inteiras):
            raise ValueError("O começo da resposta não é tokenizado igual para todas as intenções.")

        # O nome vai até o primeiro token que traz a aspa de fechamento. Sem a aspa,
        # `email_query` seria só o começo de `email_querycontact`.
        self._nomes, self._depois = [], []
        for seq in inteiras:
            resto = seq[comum:]
            fecha = next(j for j, token in enumerate(resto) if '"' in tok.decode([token]))
            self._nomes.append(resto[: fecha + 1])
            self._depois.append(resto[fecha + 1 :])
        if len({tuple(nome) for nome in self._nomes}) != len(self.intents):
            raise ValueError("Duas intenções têm a mesma sequência de tokens.")

        # O bloco: de cada nome, todos os tokens menos o último (a saída do último não é usada).
        # `passo` é a posição do token dentro do seu nome; `dono` é o nome a que pertence.
        bloco, passo, dono, alvo = [], [], [], []
        for i, nome in enumerate(self._nomes):
            for t in range(len(nome) - 1):
                bloco.append(nome[t]); passo.append(t); dono.append(i); alvo.append(nome[t + 1])
        d = self.dispositivo
        self._bloco = torch.tensor(bloco, dtype=torch.long, device=d)
        self._passo = torch.tensor(passo, dtype=torch.long, device=d)
        self._dono = torch.tensor(dono, dtype=torch.long, device=d)
        self._alvo = torch.tensor(alvo, dtype=torch.long, device=d)
        self._primeiro = torch.tensor([nome[0] for nome in self._nomes], dtype=torch.long, device=d)
        # um token do bloco enxerga os tokens do mesmo nome que vêm antes dele, e ele mesmo
        self._enxerga = (self._dono[:, None] == self._dono[None, :]) & (self._passo[None, :] <= self._passo[:, None])

    def texto_do_prompt(self, texto: str, excluir_id: int | None = None) -> str:
        """As mesmas instruções e a mesma mensagem da LLM fechada, no formato de conversa do modelo."""
        return self._conversa(prompt.mensagem(texto, self.demonstracoes.mais_parecidos(texto, self.k, excluir_id)))

    def _conversa(self, mensagem: str) -> str:
        conversa = [{"role": "system", "content": self._instrucoes}, {"role": "user", "content": mensagem}]
        # enable_thinking=False: modelos que "pensam em voz alta" (Qwen3) respondem direto; os outros ignoram
        return self.tokenizador.apply_chat_template(conversa, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _tokens(self, texto_ja_formatado: str) -> list[int]:
        return self.tokenizador(texto_ja_formatado, add_special_tokens=False).input_ids   # o formato de conversa já traz os tokens especiais

    def tokens_do_prompt(self, texto: str, excluir_id: int | None = None) -> list[int]:
        """O prompt mais o começo fixo da resposta, em tokens."""
        return self._tokens(self.texto_do_prompt(texto, excluir_id)) + self._inicio

    # ------------------------------------------------------------------ o motor: ler tokens com máscara e posição explícitas

    def _em_float(self, pode: torch.Tensor) -> torch.Tensor:
        """Máscara booleana (lote, consultas, chaves) para o formato do modelo: zero onde pode olhar, muito negativo onde não."""
        return torch.zeros(pode.shape, dtype=self._tipo, device=pode.device).masked_fill(~pode, torch.finfo(self._tipo).min)[:, None]

    @torch.no_grad()
    def _ler(self, estado: _Estado, ids: torch.Tensor, valido: torch.Tensor, bloco: bool = False) -> torch.Tensor:
        """Passa mais tokens pelo modelo, por cima do que ele já leu, e devolve as saídas.

        No modo normal, `ids` (lote, n) são os próximos tokens de cada pedido,
        com enchimento à esquerda onde `valido` é falso; devolve a saída da
        última posição e o estado passa a incluir esses tokens. No modo
        `bloco`, os tokens são os dos 60 nomes de intenção; devolve a saída de
        todas as posições e o estado NÃO muda.
        """
        lote, n = ids.shape
        if bloco:
            posicao = estado.lidos[:, None] + self._passo[None]                    # cada nome começa logo depois do prompt
            entre_si = self._enxerga[None].expand(lote, n, n)
            do_passado = estado.valido[:, None, :].expand(lote, n, -1)
        else:
            posicao = (estado.lidos[:, None] + valido.cumsum(dim=1) - 1).clamp(min=0)   # o enchimento não conta posição
            causal = torch.ones(n, n, dtype=torch.bool, device=ids.device).tril()
            eu_mesmo = torch.eye(n, dtype=torch.bool, device=ids.device)
            # um token de enchimento olha só para si mesmo: precisa olhar para algo, e ninguém olha para ele
            entre_si = (causal[None] & valido[:, None, :] & valido[:, :, None]) | eu_mesmo[None]
            do_passado = estado.valido[:, None, :] & valido[:, :, None]
        pode = torch.cat([do_passado, entre_si], dim=-1)
        if self._janela is None:
            mascara = self._em_float(pode)
        else:
            chaves = torch.cat([estado.posicao, posicao], dim=1)
            perto = (posicao[:, :, None] - chaves[:, None, :]) < self._janela
            eu = torch.cat([torch.zeros_like(do_passado), torch.eye(n, dtype=torch.bool, device=ids.device)[None].expand(lote, n, n)], dim=-1)
            mascara = {"full_attention": self._em_float(pode), "sliding_attention": self._em_float((pode & perto) | (eu & pode))}
        saida = self.modelo(
            input_ids=ids, attention_mask=mascara, position_ids=posicao, past_key_values=estado.memoria,
            use_cache=True, logits_to_keep=0 if bloco else 1,   # fora do bloco só interessa a última posição
        )
        if bloco:
            estado.memoria.crop(-n)                               # os nomes saem da memória: ela volta ao fim do prompt
            return saida.logits
        estado.valido = torch.cat([estado.valido, valido], dim=1)
        estado.posicao = torch.cat([estado.posicao, posicao], dim=1)
        estado.lidos = estado.lidos + valido.sum(dim=1)
        return saida.logits[:, -1]

    def _empilhar(self, sequencias: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequências de tamanhos diferentes num tensor só, com o enchimento à esquerda (o fim de todas fica alinhado)."""
        largura = max(map(len, sequencias))
        ids = torch.full((len(sequencias), largura), self._pad, dtype=torch.long, device=self.dispositivo)
        valido = torch.zeros((len(sequencias), largura), dtype=torch.bool, device=self.dispositivo)
        for i, seq in enumerate(sequencias):
            if seq:
                ids[i, largura - len(seq) :] = torch.tensor(seq, dtype=torch.long, device=self.dispositivo)
                valido[i, largura - len(seq) :] = True
        return ids, valido

    def _estado_vazio(self, lote: int) -> _Estado:
        vazio = lambda tipo: torch.zeros((lote, 0), dtype=tipo, device=self.dispositivo)
        # DynamicCache() sem configuração guarda todos os tokens em todas as camadas; a janela das camadas
        # deslizantes fica por conta da máscara
        return _Estado(DynamicCache(), vazio(torch.bool), vazio(torch.long))

    def _preparar_instrucoes(self) -> None:
        """Lê uma vez o começo comum a todos os prompts (as instruções) e guarda a memória."""
        a, b = self._tokens(self._conversa("A")), self._tokens(self._conversa("B outro texto"))
        comum = 0
        while comum < min(len(a), len(b)) and a[comum] == b[comum]:
            comum += 1
        self._prefixo = a[: max(comum - 1, 0)]      # um token de folga: o último pode se fundir com o que vem depois
        estado = self._estado_vazio(1)
        if self._prefixo:
            self._ler(estado, *self._empilhar([self._prefixo]))
        self._memoria_das_instrucoes = estado.memoria

    def _estado_inicial(self, lote: int) -> _Estado:
        if self._memoria_das_instrucoes is None:
            self._preparar_instrucoes()
        memoria = copy.deepcopy(self._memoria_das_instrucoes)
        memoria.batch_repeat_interleave(lote)
        n = len(self._prefixo)
        return _Estado(
            memoria,
            torch.ones((lote, n), dtype=torch.bool, device=self.dispositivo),
            torch.arange(n, device=self.dispositivo)[None].expand(lote, n).contiguous(),
        )

    # ------------------------------------------------------------------ os dois passos

    @torch.no_grad()
    def _pontuar_e_gerar(self, sequencias: list[list[int]]) -> tuple[torch.Tensor, list[int], list[tuple[str | None, list[int]]]]:
        """Para cada prompt: as 60 pontuações, a intenção escolhida e as entidades (o texto, ou None se não fechou, e os tokens gerados)."""
        lote = len(sequencias)
        estado = self._estado_inicial(lote)
        n = len(self._prefixo)
        if any(seq[:n] != self._prefixo for seq in sequencias):
            raise ValueError("Um prompt não começa pelas instruções que estão na memória.")

        # o resto do prompt, até `{"intent": "`
        raiz = torch.log_softmax(self._ler(estado, *self._empilhar([seq[n:] for seq in sequencias])).float(), dim=-1)

        # passo 1: os 60 nomes de uma vez
        pontos = raiz[:, self._primeiro]
        if len(self._bloco):
            todos = torch.ones((lote, len(self._bloco)), dtype=torch.bool, device=self.dispositivo)
            saidas = self._ler(estado, self._bloco[None].expand(lote, -1), todos, bloco=True)
            posicoes = torch.arange(len(self._bloco), device=self.dispositivo)
            # um pedido por vez, para a conta em 32 bits não ocupar a placa com o vocabulário inteiro de todo o lote
            passos = torch.stack([torch.log_softmax(s.float(), dim=-1)[posicoes, self._alvo] for s in saidas])
            pontos = pontos.index_add(1, self._dono, passos)                       # soma os tokens de cada nome
        escolhidas = pontos.argmax(dim=1).tolist()

        # passo 2: escreve a intenção escolhida e continua, um token por vez
        proximo = self._ler(estado, *self._empilhar([self._nomes[i] + self._depois[i] for i in escolhidas])).argmax(dim=-1)
        gerados: list[list[int]] = [[] for _ in range(lote)]
        textos: list[str | None] = [None] * lote
        ativo = torch.ones(lote, dtype=torch.bool, device=self.dispositivo)
        comeco = [INICIO + self.intents[i] + DEPOIS for i in escolhidas]
        for _ in range(MAX_TOKENS):
            for i in ativo.nonzero().flatten().tolist():
                token = int(proximo[i])
                if token in self._fins:
                    # o modelo deu a resposta por terminada: vale o que ele escreveu, e a leitura diz se é um JSON
                    textos[i] = self.tokenizador.decode(gerados[i], skip_special_tokens=True)
                    ativo[i] = False
                    continue
                gerados[i].append(token)
                texto = self.tokenizador.decode(gerados[i], skip_special_tokens=True)
                if fechou(comeco[i] + texto):
                    textos[i] = texto
                    ativo[i] = False
            if not ativo.any():
                break
            entrada = torch.where(ativo, proximo, torch.full_like(proximo, self._pad))
            proximo = self._ler(estado, entrada[:, None], ativo[:, None].clone()).argmax(dim=-1)
        return pontos, escolhidas, list(zip(textos, gerados))

    def responder(self, itens: list[tuple[str, str, int | None]]) -> list[dict]:
        """Um lote de pedidos: `itens` é uma lista de (identificador, texto, id a excluir das demonstrações).

        Cada registro traz o texto da resposta (o mesmo JSON que a LLM fechada
        devolve) e `logp`, as 60 log-probabilidades de intenção, na ordem de
        `self.intents`.
        """
        inicio = time.perf_counter()
        sequencias = [self.tokens_do_prompt(texto, excluir) for _, texto, excluir in itens]
        pontos, escolhidas, entidades = self._pontuar_e_gerar(sequencias)
        ms = (time.perf_counter() - inicio) * 1000
        registros = []
        for (identificador, _, _), seq, logp, i, (texto, gerados) in zip(itens, sequencias, pontos.tolist(), escolhidas, entidades):
            registros.append({
                "id": identificador,
                "texto": None if texto is None else INICIO + self.intents[i] + DEPOIS + texto,   # resposta cortada não vale
                "parada": "end_turn" if texto is not None else "max_tokens",
                "uso": {"entrada": len(seq), "saida": len(self._nomes[i]) + len(self._depois[i]) + len(gerados)},
                "ms": ms / len(itens),          # tempo do lote dividido pelos pedidos; só é latência quando o lote tem 1 pedido
                "erro": None,
                "logp": [round(v, 4) for v in logp],
            })
        self.registros.extend(registros)
        return registros

    def chamar(self, texto: str, excluir_id: int | None = None, identificador: str = "") -> dict:
        """Uma chamada, cronometrada: do texto do pedido ao texto da resposta."""
        return self.responder([(identificador, texto, excluir_id)])[0]

    def interpretar(self, registro: dict, texto: str) -> tuple[Previsao, dict]:
        return resposta.ler(registro["texto"], texto, self.intents, self.tipos)

    def prever(self, textos: list[str]) -> list[Previsao]:
        """O contrato da régua: um pedido por vez, como um agente faria."""
        return [self.interpretar(self.chamar(texto), texto)[0] for texto in textos]

    # ------------------------------------------------------------------ as mesmas contas, do jeito direto (para conferir)

    @torch.no_grad()
    def pontuar_de_referencia(self, sequencia: list[int]) -> torch.Tensor:
        """Log-probabilidade de cada nome, com uma passada inteira por nome e nenhum reaproveitamento: tensor (60,)."""
        pontos = []
        for nome in self._nomes:
            ids = torch.tensor([sequencia + nome[:-1]], dtype=torch.long, device=self.dispositivo)
            saidas = torch.log_softmax(self.modelo(input_ids=ids, logits_to_keep=len(nome)).logits[0].float(), dim=-1)
            pontos.append(saidas[torch.arange(len(nome)), torch.tensor(nome, device=self.dispositivo)].sum())
        return torch.stack(pontos)

    @torch.no_grad()
    def gerar_de_referencia(self, sequencia: list[int], escolhida: int) -> str | None:
        """As entidades geradas pela função `generate` da biblioteca, um pedido sozinho, sem nenhum reaproveitamento."""
        ids = torch.tensor([sequencia + self._nomes[escolhida] + self._depois[escolhida]], dtype=torch.long, device=self.dispositivo)
        saida = self.modelo.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=self._pad)
        # o mesmo critério de parada do caminho rápido, aplicado aos tokens que a biblioteca gerou
        comeco, gerados = INICIO + self.intents[escolhida] + DEPOIS, []
        for token in saida[0, ids.shape[1] :].tolist():
            if token in self._fins:
                return self.tokenizador.decode(gerados, skip_special_tokens=True)
            gerados.append(token)
            texto = self.tokenizador.decode(gerados, skip_special_tokens=True)
            if fechou(comeco + texto):
                return texto
        return None
