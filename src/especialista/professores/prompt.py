"""O prompt, igual para as duas LLMs.

Três partes:

1. As instruções (`instrucoes`): a tarefa, as regras e as duas listas de
   rótulos. São fixas, então o provedor pode guardá-las em cache.
2. A mensagem (`mensagem`): as demonstrações mais parecidas e o pedido.
3. O formato da resposta (`esquema`): um JSON com a intenção e as
   entidades. Onde a LLM aceita saída estruturada, o esquema é imposto na
   geração, e a resposta nunca vem malformada nem com rótulo inventado.

A LLM devolve o TEXTO de cada entidade, não a posição: contar palavras é o
tipo de coisa em que LLMs erram. Achar a posição é trabalho de `resposta.py`.
"""

from __future__ import annotations

import json

_REGRAS = """Você é o módulo de compreensão de um assistente virtual. O usuário fala português de Portugal.

Para cada pedido do usuário, identifique:
1. a intenção: a ação que o usuário quer. Escolha exatamente uma da lista de intenções.
2. as entidades: os trechos do pedido que são argumentos dessa ação. Cada entidade tem um tipo da lista de tipos de entidade.

Regras para as entidades:
- Copie o texto de cada entidade exatamente como está no pedido, palavra por palavra. Não corrija, não traduza, não complete.
- Uma entidade é um trecho contínuo do pedido. Não junte palavras que não estão lado a lado.
- Liste as entidades na ordem em que aparecem no pedido.
- Se o pedido não tiver nenhuma entidade, devolva a lista vazia.
- Nem toda palavra importante é uma entidade. Siga o critério dos exemplos resolvidos.

Use só as intenções e os tipos de entidade das listas abaixo.

Responda com um objeto JSON com os campos "intent" e "slots". Cada item de "slots" tem "type" e "text"."""


def instrucoes(intents: list[str], tipos: list[str]) -> str:
    return (
        f"{_REGRAS}\n\n"
        "Intenções:\n" + "\n".join(f"- {i}" for i in intents) + "\n\n"
        "Tipos de entidade:\n" + "\n".join(f"- {t}" for t in tipos)
    )


def resposta_em_json(intent: str, slots: list[dict]) -> str:
    """O formato exato da resposta, usado nas demonstrações."""
    return json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False)


def mensagem(texto: str, exemplos: list[dict]) -> str:
    """As demonstrações (se houver) e o pedido a resolver."""
    partes = []
    if exemplos:
        partes.append("Exemplos resolvidos, parecidos com o pedido:")
        for exemplo in exemplos:
            partes.append(f"Pedido: {exemplo['utt']}\nResposta: {resposta_em_json(exemplo['intent'], exemplo['slots'])}")
        partes.append("Agora resolva este:")
    partes.append(f"Pedido: {texto}")
    return "\n\n".join(partes)


def esquema(intents: list[str], tipos: list[str]) -> dict:
    """JSON Schema da resposta: intenção e tipos de entidade restritos às listas do projeto."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(intents)},
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": list(tipos)},
                        "text": {"type": "string"},
                    },
                    "required": ["type", "text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "slots"],
        "additionalProperties": False,
    }
