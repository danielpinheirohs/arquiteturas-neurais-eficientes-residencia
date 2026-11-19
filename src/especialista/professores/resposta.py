"""Da resposta em texto da LLM para uma `Previsao`.

A régua compara entidades por posição no pedido (tipo, início, fim). A LLM
devolve o texto de cada entidade. Esta é a regra única de casamento, igual
para qualquer LLM:

1. O texto da entidade é separado em palavras, como o pedido.
2. Procura-se essa sequência de palavras no pedido, a partir do fim da
   entidade anterior (as entidades vêm em ordem). Se não achar dali em
   diante, procura-se no pedido inteiro.
3. Se a sequência não existe no pedido, ou só existe em cima de uma
   entidade já casada, a entidade é descartada.

Uma entidade descartada vira erro na régua (ela falta na resposta) e é
contada no diagnóstico, para sabermos quanto a LLM perde por não copiar o
texto com exatidão.

Antes disso, a leitura isola o objeto JSON da resposta: uma LLM sem saída
estruturada pode embrulhar o JSON em marcação ou em uma frase. Vale o
trecho do primeiro `{` ao último `}`. Se dali não sair um JSON válido, a
resposta não conta.
"""

from __future__ import annotations

import json
from collections import Counter

from especialista.avaliacao.previsao import Previsao

SEM_RESPOSTA = "(sem resposta)"  # intenção usada quando não há o que ler; nunca coincide com um rótulo


def _achar(agulha: list[str], palheiro: list[str], a_partir_de: int, ocupado: list[bool]) -> int | None:
    for inicio in [*range(a_partir_de, len(palheiro)), *range(0, a_partir_de)]:
        fim = inicio + len(agulha)
        if palheiro[inicio:fim] == agulha and not any(ocupado[inicio:fim]):
            return inicio
    return None


def ler(texto_da_resposta: str | None, pedido: str, intents: list[str], tipos: list[str]) -> tuple[Previsao, Counter]:
    """Devolve a previsão e o diagnóstico do que aconteceu na leitura.

    Chaves possíveis do diagnóstico: `sem_resposta`, `json_invalido`,
    `intencao_fora_da_lista`, `entidades`, `tipo_fora_da_lista`,
    `texto_nao_encontrado`.
    """
    diagnostico: Counter = Counter()
    if not texto_da_resposta:
        diagnostico["sem_resposta"] += 1
        return Previsao(SEM_RESPOSTA), diagnostico
    try:
        inicio, fim = texto_da_resposta.index("{"), texto_da_resposta.rindex("}")
        dados = json.loads(texto_da_resposta[inicio : fim + 1])
        intent = dados["intent"]
        itens = dados.get("slots", [])
        assert isinstance(intent, str) and isinstance(itens, list)
    except (ValueError, KeyError, TypeError, AssertionError):  # ValueError cobre JSON inválido e chave ausente no texto
        diagnostico["json_invalido"] += 1
        return Previsao(SEM_RESPOSTA), diagnostico

    if intent not in intents:
        diagnostico["intencao_fora_da_lista"] += 1

    palavras = pedido.split()
    ocupado = [False] * len(palavras)
    spans, cursor = [], 0
    for item in itens:
        diagnostico["entidades"] += 1
        tipo, texto = (item.get("type"), item.get("text")) if isinstance(item, dict) else (None, None)
        if tipo not in tipos or not isinstance(texto, str) or not texto.split():
            diagnostico["tipo_fora_da_lista"] += 1
            continue
        agulha = texto.split()
        inicio = _achar(agulha, palavras, cursor, ocupado)
        if inicio is None:
            diagnostico["texto_nao_encontrado"] += 1
            continue
        fim = inicio + len(agulha)
        ocupado[inicio:fim] = [True] * len(agulha)
        spans.append((tipo, inicio, fim))
        cursor = fim
    return Previsao(intent, tuple(sorted(spans, key=lambda s: s[1]))), diagnostico
