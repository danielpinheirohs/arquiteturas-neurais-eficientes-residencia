"""O motor do professor local, conferido num modelo minúsculo de pesos aleatórios, na CPU.

O que está em jogo: a pontuação dos nomes numa passada só, a memória das
instruções, o enchimento dos lotes e a geração própria precisam dar o mesmo
resultado das contas diretas da biblioteca. Usa os tokenizadores de verdade,
se estiverem em disco; os pesos são sorteados.
"""

import pytest
import torch

from especialista.professores import local
from especialista.professores.demonstracoes import Demonstracoes
from especialista.professores.local import ProfessorLocal

from test_professores import ROTULOS, treino  # noqa: F401  (treino é uma fixture)

transformers = pytest.importorskip("transformers")
PEDIDOS = ["toca rui veloso", "acorda-me às nove da manhã na sexta e depois lembra-me de comprar pão no caminho para o porto", "como está o tempo"]


def _tokenizador(modelo_id: str):
    try:
        return transformers.AutoTokenizer.from_pretrained(modelo_id, local_files_only=True)
    except Exception:
        pytest.skip(f"o tokenizador de {modelo_id} não está em disco")


def _qwen():
    tok = _tokenizador("Qwen/Qwen3-4B")
    config = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    return tok, transformers.Qwen3ForCausalLM(config)


def _gemma():
    """Com janela deslizante de 16 tokens, bem menor que o prompt: a máscara das camadas de janela é exercitada de verdade."""
    tok = _tokenizador("CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it")
    config = transformers.Gemma3TextConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    return tok, transformers.Gemma3ForCausalLM(config)


@pytest.fixture(params=[_qwen, _gemma], ids=["atencao_inteira", "janela_deslizante"])
def professor(request, treino, monkeypatch):  # noqa: F811
    monkeypatch.setattr(local, "MAX_TOKENS", 12)
    torch.manual_seed(0)
    tok, modelo = request.param()
    return ProfessorLocal("teste", Demonstracoes(treino), ROTULOS, k=2, modelo=modelo.eval(), tokenizador=tok)


def test_nomes_terminam_na_aspa_e_sao_distintos(professor):
    for intent, nome in zip(professor.intents, professor._nomes):
        texto = professor.tokenizador.decode(nome)
        assert texto.startswith(intent) and '"' in texto[len(intent):]
    assert len({tuple(n) for n in professor._nomes}) == len(professor.intents)


def test_pontuacao_rapida_igual_a_direta(professor):
    sequencias = [professor.tokens_do_prompt(pedido) for pedido in PEDIDOS]
    assert len({len(s) for s in sequencias}) > 1                       # tamanhos diferentes: o lote tem enchimento
    em_lote, escolhidas, _ = professor._pontuar_e_gerar(sequencias)
    direta = torch.stack([professor.pontuar_de_referencia(s) for s in sequencias])
    um_a_um = torch.cat([professor._pontuar_e_gerar([s])[0] for s in sequencias])

    assert torch.allclose(um_a_um, direta, atol=1e-3)                  # memória das instruções + bloco dos nomes
    assert torch.allclose(em_lote, direta, atol=1e-3)                  # e o enchimento do lote não muda nada
    assert escolhidas == direta.argmax(dim=1).tolist()
    assert len(professor._prefixo) > 20                                # as instruções foram mesmo reaproveitadas


def test_geracao_propria_igual_a_da_biblioteca(professor):
    sequencias = [professor.tokens_do_prompt(pedido) for pedido in PEDIDOS]
    _, escolhidas, entidades = professor._pontuar_e_gerar(sequencias)
    for seq, i, (_, gerados) in zip(sequencias, escolhidas, entidades):
        ids = torch.tensor([seq + professor._nomes[i] + professor._depois[i]])
        esperado = professor.modelo.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, do_sample=False,
                                             pad_token_id=professor._pad)[0, ids.shape[1]:].tolist()
        esperado = esperado[: next((k for k, t in enumerate(esperado) if t in professor._fins), len(esperado))]
        assert gerados == esperado[: len(gerados)] and len(gerados) > 0
        # a geração própria só para antes da biblioteca quando o JSON fecha
        comeco = local.INICIO + professor.intents[i] + local.DEPOIS
        assert len(gerados) == len(esperado) or local.fechou(comeco + professor.tokenizador.decode(gerados))


def test_fechou_reconhece_um_json_completo():
    assert local.fechou('{"intent": "x", "slots": []}') and local.fechou('{"intent": "x", "slots": [\n {"type": "a", "text": "b"}\n]\n}\n')
    assert not local.fechou('{"intent": "x", "slots": [{"type": "a", "text": "b"}')      # a lista ainda está aberta
    assert not local.fechou('{"intent": "x", "slots": [}}')                               # fecha errado


def test_registro_traz_resposta_e_as_60_pontuacoes(professor):
    registros = professor.responder([("dev-1", PEDIDOS[0], None), ("dev-2", PEDIDOS[2], None)])
    assert [r["id"] for r in registros] == ["dev-1", "dev-2"]
    for registro in registros:
        assert len(registro["logp"]) == len(professor.intents)
        assert registro["parada"] in ("end_turn", "max_tokens") and (registro["texto"] is None) == (registro["parada"] == "max_tokens")
        assert registro["uso"]["entrada"] > 0 and registro["uso"]["saida"] > 0
    assert professor.prever([PEDIDOS[0]])[0].intent in [*professor.intents, "(sem resposta)"]
