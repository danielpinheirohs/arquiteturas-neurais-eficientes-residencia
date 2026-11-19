"""Prompt, demonstrações, leitura da resposta e cliente do Claude, sem rede (com um cliente de mentira)."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from especialista.avaliacao.previsao import Previsao
from especialista.professores import prompt, resposta
from especialista.professores.claude import Ajuste, ProfessorClaude, custo, gravar, ler_registros
from especialista.professores.demonstracoes import Demonstracoes, entidades_em_texto, montar_conjunto

from test_diet import ENUNCIADOS

INTENTS = ["alarm_set", "play_music", "weather_query"]
TIPOS = ["artist", "date", "place", "time"]
ROTULOS = {"intents": INTENTS, "slot_types": TIPOS}


@pytest.fixture
def treino() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": n, "intent": i, "utt": texto, "tokens": texto.split(), "tags": tags.split()}
        for n, (i, texto, tags) in enumerate(ENUNCIADOS)
    )


# ---------------------------------------------------------------- demonstrações

def test_conjunto_tem_no_maximo_n_por_intencao(treino):
    conjunto = montar_conjunto(treino, por_intencao=2, semente=0)
    assert conjunto["intent"].value_counts().to_dict() == {"alarm_set": 2, "play_music": 2, "weather_query": 2}
    assert montar_conjunto(treino, 2, semente=0)["id"].tolist() == conjunto["id"].tolist()   # reproduzível
    assert len(montar_conjunto(treino, por_intencao=99)) == len(treino)                      # classe pequena entra inteira


def test_entidades_em_texto():
    assert entidades_em_texto("acorda-me às nove da manhã".split(), "O O B-time I-time I-time".split()) == [
        {"type": "time", "text": "nove da manhã"}
    ]


def test_busca_traz_os_mais_parecidos_e_nunca_o_proprio(treino):
    demos = Demonstracoes(treino)
    achados = demos.mais_parecidos("toca tony carreira", k=3)
    assert achados[-1]["utt"] == "toca tony carreira"            # o mais parecido fica por último, junto do pedido
    sem_ele = demos.mais_parecidos("toca tony carreira", k=3, excluir_id=3)
    assert all(e["utt"] != "toca tony carreira" for e in sem_ele)
    assert demos.mais_parecidos("qualquer coisa", k=0) == []


# ---------------------------------------------------------------- prompt

def test_prompt_lista_todos_os_rotulos_e_mostra_o_formato(treino):
    instrucoes = prompt.instrucoes(INTENTS, TIPOS)
    assert all(f"- {i}" in instrucoes for i in INTENTS) and all(f"- {t}" in instrucoes for t in TIPOS)

    exemplos = Demonstracoes(treino).mais_parecidos("toca rui veloso", k=2)
    mensagem = prompt.mensagem("toca rui veloso", exemplos)
    assert mensagem.endswith("Pedido: toca rui veloso") and mensagem.count("Resposta: {") == 2
    assert prompt.mensagem("toca rui veloso", []) == "Pedido: toca rui veloso"     # sem exemplos, só o pedido

    esquema = prompt.esquema(INTENTS, TIPOS)
    assert esquema["properties"]["intent"]["enum"] == INTENTS
    assert esquema["properties"]["slots"]["items"]["properties"]["type"]["enum"] == TIPOS
    assert esquema["additionalProperties"] is False


# ---------------------------------------------------------------- leitura da resposta

def ler(dados, pedido):
    return resposta.ler(json.dumps(dados, ensure_ascii=False), pedido, INTENTS, TIPOS)


def test_resposta_certa_vira_spans():
    previsao, diagnostico = ler(
        {"intent": "alarm_set", "slots": [{"type": "time", "text": "nove da manhã"}, {"type": "date", "text": "sexta"}]},
        "acorda-me às nove da manhã na sexta",
    )
    assert previsao == Previsao("alarm_set", (("time", 2, 5), ("date", 6, 7)))
    assert diagnostico == {"entidades": 2}


def test_palavra_repetida_casa_em_ordem():
    previsao, _ = ler(
        {"intent": "weather_query", "slots": [{"type": "place", "text": "porto"}, {"type": "place", "text": "porto"}]},
        "do porto ao porto de leixões",
    )
    assert previsao.slots == (("place", 1, 2), ("place", 3, 4))


def test_texto_que_nao_esta_no_pedido_e_descartado():
    previsao, diagnostico = ler(
        {"intent": "alarm_set", "slots": [{"type": "time", "text": "9 da manhã"}, {"type": "date", "text": "sexta"}]},
        "acorda-me às nove da manhã na sexta",
    )
    assert previsao.slots == (("date", 6, 7),)
    assert diagnostico["texto_nao_encontrado"] == 1 and diagnostico["entidades"] == 2


def test_rotulos_fora_da_lista():
    previsao, diagnostico = ler({"intent": "inventada", "slots": [{"type": "cor", "text": "nove"}]}, "às nove")
    assert previsao.intent == "inventada" and previsao.slots == ()
    assert diagnostico["intencao_fora_da_lista"] == 1 and diagnostico["tipo_fora_da_lista"] == 1


@pytest.mark.parametrize("texto", [None, "", "isto não é json", '{"slots": []}', '{"intent": 3, "slots": []}'])
def test_resposta_ilegivel_conta_como_erro(texto):
    previsao, diagnostico = resposta.ler(texto, "às nove", INTENTS, TIPOS)
    assert previsao == Previsao(resposta.SEM_RESPOSTA)
    assert diagnostico["sem_resposta"] + diagnostico["json_invalido"] == 1


def test_json_embrulhado_em_marcacao_e_lido():
    embrulhado = 'Aqui está:\n```json\n{"intent": "alarm_set", "slots": [{"type": "time", "text": "nove"}]}\n```'
    previsao, diagnostico = resposta.ler(embrulhado, "acorda-me às nove", INTENTS, TIPOS)
    assert previsao == Previsao("alarm_set", (("time", 2, 3),)) and "json_invalido" not in diagnostico


# ---------------------------------------------------------------- cliente do Claude, com um cliente de mentira

class ClienteDeMentira:
    """Responde sempre o mesmo JSON e guarda os parâmetros que recebeu."""

    def __init__(self, texto: str, parada: str = "end_turn"):
        self.recebidos = []
        self._texto, self._parada = texto, parada
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **parametros):
        self.recebidos.append(parametros)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self._texto)],
            stop_reason=self._parada,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=40, cache_read_input_tokens=700, cache_creation_input_tokens=None),
        )


def test_professor_monta_o_pedido_e_le_a_resposta(treino):
    falso = ClienteDeMentira('{"intent": "play_music", "slots": [{"type": "artist", "text": "rui veloso"}]}')
    professor = ProfessorClaude(Demonstracoes(treino), ROTULOS, Ajuste(k=2), cliente=falso)

    assert professor.prever(["toca rui veloso"]) == [Previsao("play_music", (("artist", 1, 3),))]

    enviado = falso.recebidos[0]
    assert enviado["model"] == "claude-haiku-5-5"
    assert "temperature" not in enviado and "top_p" not in enviado            # o modelo rejeita amostragem
    assert enviado["thinking"] == {"type": "disabled"}
    assert enviado["output_config"]["format"]["type"] == "json_schema"
    assert enviado["system"][0]["cache_control"] == {"type": "ephemeral"}     # as instruções fixas vão para o cache
    assert enviado["messages"][-1]["role"] == "user"                          # sem resposta pré-preenchida
    assert enviado["messages"][0]["content"].endswith("Pedido: toca rui veloso")


def test_ajuste_muda_raciocinio_e_demonstracoes(treino):
    falso = ClienteDeMentira('{"intent": "play_music", "slots": []}')
    professor = ProfessorClaude(Demonstracoes(treino), ROTULOS, Ajuste(k=0, raciocinio=True, esforco="medium"), cliente=falso)
    pedido = professor.pedido("toca música")
    assert pedido["thinking"] == {"type": "adaptive"} and pedido["output_config"]["effort"] == "medium"
    assert pedido["messages"][0]["content"] == "Pedido: toca música"


def test_resposta_cortada_ou_recusada_nao_vale(treino):
    for parada in ("max_tokens", "refusal"):
        falso = ClienteDeMentira('{"intent": "play_music", "slots": []}', parada=parada)
        professor = ProfessorClaude(Demonstracoes(treino), ROTULOS, cliente=falso)
        registro = professor.chamar("toca música")
        assert registro["texto"] is None and registro["parada"] == parada
        assert professor.interpretar(registro, "toca música")[0] == Previsao(resposta.SEM_RESPOSTA)


def test_custo_e_registros_em_disco(treino, tmp_path):
    uso = {"entrada": 1_000_000, "saida": 1_000_000, "cache_lido": 1_000_000, "cache_escrito": 0}
    assert custo(uso) == pytest.approx(0.10 + 0.50 + 0.01)
    assert custo(uso, em_lote=True) == pytest.approx((0.10 + 0.50 + 0.01) / 2)

    falso = ClienteDeMentira('{"intent": "play_music", "slots": []}')
    professor = ProfessorClaude(Demonstracoes(treino), ROTULOS, cliente=falso)
    registros = [professor.chamar("toca música", identificador="dev-7")]
    lidos = ler_registros(gravar(registros, tmp_path / "respostas.jsonl"))
    assert lidos["dev-7"]["uso"]["cache_lido"] == 700 and lidos["dev-7"]["ms"] > 0

    # chamadas diretas sem identificador: por identificador viram uma só; em ordem, ficam todas
    from especialista.professores.claude import ler_em_ordem
    sem_nome = gravar([professor.chamar("toca música"), professor.chamar("às nove")], tmp_path / "diretas.jsonl")
    assert len(ler_registros(sem_nome)) == 1 and len(ler_em_ordem(sem_nome)) == 2


# ---------------------------------------------------------------- das respostas gravadas aos rótulos

def test_respostas_gravadas_viram_conjunto_rotulado(treino):
    from especialista.professores.rotulagem import ler_conjunto, rotular

    dados = treino.iloc[:3]
    registros = {
        f"train-{dados.iloc[0]['id']}": {"id": "a", "texto": json.dumps({"intent": dados.iloc[0]["intent"], "slots": []}), "uso": {}},
        f"train-{dados.iloc[1]['id']}": {"id": "b", "texto": None, "uso": {}},          # recusada ou cortada
    }                                                                                   # o terceiro pedido não tem registro
    previsoes, diagnostico, alinhados = ler_conjunto(registros, dados, "train", INTENTS, TIPOS)
    assert [p.intent for p in previsoes] == [dados.iloc[0]["intent"], resposta.SEM_RESPOSTA, resposta.SEM_RESPOSTA]
    assert diagnostico["sem_resposta"] == 2 and diagnostico["sem_registro"] == 1 and len(alinhados) == 3

    rotulado = rotular(dados, previsoes, INTENTS)
    assert rotulado["valido"].tolist() == [True, False, False]
    assert rotulado["tags"].map(len).tolist() == dados["tokens"].map(len).tolist()
    assert set(rotulado.iloc[0]["tags"]) == {"O"}


def test_resumo_de_uso_soma_tokens_e_dolares():
    from especialista.professores.claude import resumir_uso

    registros = [
        {"texto": "{}", "uso": {"entrada": 1_000_000, "saida": 0, "cache_lido": 1_000_000, "cache_escrito": 0}},
        {"texto": None, "uso": {}},
    ]
    resumo = resumir_uso(registros, em_lote=True)
    assert resumo["pedidos"] == 2 and resumo["com_resposta"] == 1 and resumo["cache_lido"] == 1_000_000
    assert resumo["dolares"] == pytest.approx((0.10 + 0.01) / 2)


def test_envio_que_falha_depois_de_aceito_nao_e_repetido(treino):
    from datetime import datetime, timezone

    import anthropic
    import httpx

    from especialista.professores.claude import enviar_com_cuidado

    class Lotes:
        """Todo envio falha, mas a conta mostra um lote com o número certo de pedidos."""

        def __init__(self):
            self.envios = 0

        def create(self, requests):
            self.envios += 1
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://exemplo.invalid"))

        def list(self, limit):
            contagem = SimpleNamespace(processing=2, succeeded=0, errored=0, canceled=0, expired=0)
            return [SimpleNamespace(id="lote_aceito", request_counts=contagem, created_at=datetime.now(timezone.utc))]

    lotes = Lotes()
    falso = SimpleNamespace(messages=SimpleNamespace(batches=lotes))
    falso.with_options = lambda **_: falso
    professor = ProfessorClaude(Demonstracoes(treino), ROTULOS, cliente=falso)
    itens = [("a", "toca música", None), ("b", "às nove", None)]

    assert enviar_com_cuidado(professor, itens, anotados=set, pausa=0) == "lote_aceito"
    assert lotes.envios == 1                                    # achou o lote na conta: não reenviou
    with pytest.raises(RuntimeError):                           # o lote da conta já tem dono (outra parte ou outro conjunto)
        enviar_com_cuidado(professor, itens, anotados=lambda: {"lote_aceito"}, tentativas=2, pausa=0)
    assert lotes.envios == 3


def test_regra_de_escolha_do_professor_aberto():
    """Um candidato mais caro só toma o lugar se for distinguivelmente melhor em exact match."""
    from especialista.avaliacao.previsao import ouro
    from especialista.dados import massive
    from especialista.experimentos import etapa5b

    gabarito = ouro(massive.carregar_entrega()[0]["dev"])
    sem_entidades = [Previsao(o.intent) for o in gabarito]                      # acerta só os pedidos sem entidade
    quase_igual = [*sem_entidades[:-1], gabarito[-1]]                           # um acerto a mais: não distinguível

    escolhido, disputas = etapa5b.escolher({"qwen3_4b": sem_entidades, "gaia_4b": quase_igual, "qwen3_8b": gabarito})
    assert escolhido == "qwen3_8b"
    assert disputas["gaia_4b"]["contra"] == "qwen3_4b" and disputas["gaia_4b"]["ic_min"] <= 0     # não tomou o lugar
    assert disputas["qwen3_8b"]["contra"] == "qwen3_4b" and disputas["qwen3_8b"]["ic_min"] > 0
    assert etapa5b.escolher({"qwen3_4b": gabarito, "qwen3_8b": gabarito})[0] == "qwen3_4b"       # empate fica com o mais barato
