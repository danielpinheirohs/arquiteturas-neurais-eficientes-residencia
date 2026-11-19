"""O encoder pré-treinado conferido num XLM-R minúsculo de pesos sorteados, na CPU, sem baixar nada.

O que está em jogo: o alinhamento entre subtokens e palavras, o contrato com
a régua, a perda conjunta, o caminho dos logits do professor até o gradiente
e a tabela de subtokens dividida, que tem de dar o mesmo resultado do treino
com a tabela inteira.
"""

from math import ceil

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, ouro
from especialista.modelos.diet.modelo import perda_de_destilacao
from especialista.modelos.encoder import Config, SistemaEncoder, TabelaDividida
from especialista.treino_encoder import Receita, fila_de_lotes, fracao_da_taxa, treinar

from test_diet import ENUNCIADOS, ROTULOS

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

PEQUENO = Config(modelo_id="minusculo", revisao="", max_subtokens=64)
RAPIDA = Receita(epocas=40, taxa=1e-2, lote=4)   # o encoder minúsculo parte de pesos sorteados: precisa de taxa alta


def tokenizador_minusculo():
    """Um tokenizador de subpalavras treinado nos enunciados de brinquedo, com os tokens especiais do XLM-R."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    treinador = trainers.BpeTrainer(vocab_size=90, special_tokens=["<s>", "<pad>", "</s>", "<unk>"], show_progress=False)
    tok.train_from_iterator([texto for _, texto, _ in ENUNCIADOS], treinador)
    tok.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    classe = getattr(transformers, "PreTrainedTokenizerFast", None) or transformers.TokenizersBackend
    return classe(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
                  cls_token="<s>", sep_token="</s>")


def encoder_minusculo(tokenizador, semente: int = 0):
    torch.manual_seed(semente)
    config = transformers.XLMRobertaConfig(
        vocab_size=len(tokenizador), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=66, pad_token_id=tokenizador.pad_token_id, type_vocab_size=1,
    )
    return transformers.XLMRobertaModel(config, add_pooling_layer=False)


def sistema_minusculo(semente: int = 0, config: Config = PEQUENO, rotulos: dict = ROTULOS) -> SistemaEncoder:
    tokenizador = tokenizador_minusculo()
    return SistemaEncoder.novo(rotulos, config, semente=semente, encoder=encoder_minusculo(tokenizador, semente), tokenizador=tokenizador)


def dados() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": i, "intent": intent, "utt": texto, "tokens": texto.split(), "tags": tags.split()}
        for i, (intent, texto, tags) in enumerate(ENUNCIADOS)
    )


@pytest.fixture
def sistema() -> SistemaEncoder:
    return sistema_minusculo()


# --------------------------------------------------------------------------
# A. Subtokens e palavras
# --------------------------------------------------------------------------


def test_cada_palavra_aponta_para_o_seu_primeiro_subtoken(sistema):
    d = dados()
    lote = sistema.alinhador.montar([list(t) for t in d["tokens"]])
    for linha, palavras in enumerate(d["tokens"]):
        subtokens = sistema.tokenizador.convert_ids_to_tokens(lote.ids[linha].tolist())
        inicios = [i for i, s in enumerate(subtokens) if s.startswith("▁")]      # o tokenizador marca o começo de palavra com ▁
        assert lote.primeiros[linha, : len(palavras)].tolist() == inicios
        assert lote.real[linha].tolist() == [True] * len(palavras) + [False] * (lote.real.shape[1] - len(palavras))
        assert lote.comprimentos[linha] == len(palavras)
    assert any(len(sistema.tokenizador.tokenize(p)) > 1 for palavras in d["tokens"] for p in palavras)   # há palavras partidas


def test_alinhamento_com_o_tokenizador_do_xlmr():
    """Com o tokenizador de verdade, se estiver em disco: um caso conferido à mão."""
    try:
        tokenizador = transformers.AutoTokenizer.from_pretrained(Config().modelo_id, revision=Config().revisao, local_files_only=True)
    except Exception:
        pytest.skip("o tokenizador do XLM-R não está em disco")
    from especialista.modelos.encoder import Alinhador

    lote = Alinhador(tokenizador).montar([["acorda-me", "às", "nove", "da", "manhã"], ["põe", "​", "música"]])
    subtokens = tokenizador.convert_ids_to_tokens(lote.ids[0].tolist())
    assert subtokens[:10] == ["<s>", "▁acord", "a", "-", "me", "▁às", "▁nove", "▁da", "▁manhã", "</s>"]
    assert lote.primeiros[0].tolist() == [1, 5, 6, 7, 8]
    # o tokenizador apaga o caractere invisível: a palavra fica sem subtoken, e as vizinhas não mudam de lugar
    assert lote.real[1].tolist()[:3] == [True, False, True]
    assert lote.comprimentos.tolist() == [5, 3]


def test_palavra_sem_subtoken_fica_fora_da_perda_e_recebe_O():
    curto = sistema_minusculo(config=Config(modelo_id="minusculo", revisao="", max_subtokens=6))
    d = dados().iloc[[1]]                                   # "põe um alarme para sexta": não cabe em 6 subtokens
    lote = curto.lote(d)
    cabem = int(lote.real.sum())
    assert 0 < cabem < len(d.iloc[0]["tokens"]) and lote.ids.shape[1] == 6

    curto.modelo.eval()
    tags, palavras = list(d.iloc[0]["tags"]), len(d.iloc[0]["tokens"])
    perda = curto.modelo.perda(lote)["entidades"].item()
    fora = d.assign(tags=[tags[:cabem] + ["B-artist"] * (palavras - cabem)])   # muda só as tags das palavras que não cabem
    dentro = d.assign(tags=[["B-artist"] + tags[1:]])                          # muda a tag de uma palavra que cabe
    assert curto.modelo.perda(curto.lote(fora))["entidades"].item() == pytest.approx(perda)
    assert curto.modelo.perda(curto.lote(dentro))["entidades"].item() != pytest.approx(perda)

    # com a cabeça forçada a dizer B-artist em tudo, só as palavras com subtoken viram entidade
    with torch.no_grad():
        curto.modelo.entidades.weight.zero_()
        curto.modelo.entidades.bias.zero_()
        curto.modelo.entidades.bias[ROTULOS["tags"].index("B-artist")] = 10.0
    previsao = curto.prever([d.iloc[0]["utt"]])[0]
    assert [s[1] for s in previsao.slots] == list(range(cabem))


# --------------------------------------------------------------------------
# O contrato com a régua
# --------------------------------------------------------------------------


def test_formas_e_partes(sistema):
    d = dados()
    intencoes, emissoes = sistema.modelo(sistema.lote(d))
    assert intencoes.shape == (len(d), 3)
    assert emissoes.shape == (len(d), max(len(t) for t in d["tokens"]), 9)
    partes = sistema.modelo.parametros_por_parte()
    assert sum(partes.values()) == sistema.parametros
    assert partes["A. Tabela de subtokens"] == len(sistema.tokenizador) * 32
    assert partes["C. Cabeça de intenção"] == 32 * 3 + 3 and partes["D. Cabeça de entidades"] == 32 * 9 + 9


def test_contrato_de_previsao_serve_a_regua(sistema):
    d = dados()
    previsoes = sistema.prever(list(d["utt"]) + [""])                 # um pedido vazio não pode quebrar
    assert len(previsoes) == len(d) + 1 and all(isinstance(p, Previsao) for p in previsoes)
    assert all(p.intent in ROTULOS["intents"] for p in previsoes) and previsoes[-1].slots == ()
    for p, palavras in zip(previsoes, d["tokens"]):
        assert all(0 <= inicio < fim <= len(palavras) for _, inicio, fim in p.slots)

    gabarito, grupo = ouro(d), [previsoes[:-1], sistema_minusculo(semente=1).prever(list(d["utt"]))]
    assert set(qualidade.avaliar(gabarito, grupo[0])) >= set(qualidade.METRICAS)
    assert list(qualidade.intervalo_do_grupo(gabarito, grupo, n=50).columns) == ["valor", "ic_min", "ic_max", "desvio_sementes"]
    assert "distinguivel" in qualidade.comparar_grupos(gabarito, grupo[:1], grupo[1:], n=50).columns


def test_resposta_nao_depende_de_quem_esta_no_lote(sistema):
    """O preenchimento de um lote não pode vazar para os enunciados curtos."""
    d = dados()
    sistema.modelo.eval()
    with torch.no_grad():
        juntas, emissoes = sistema.modelo(sistema.lote(d))
        for i in range(len(d)):
            sozinha, da_linha = sistema.modelo(sistema.lote(d.iloc[[i]]))
            assert torch.allclose(juntas[i], sozinha[0], atol=1e-5)
            assert torch.allclose(emissoes[i, : len(d.iloc[i]["tokens"])], da_linha[0], atol=1e-5)
    assert sistema.prever(list(d["utt"])) == [sistema.prever([texto])[0] for texto in d["utt"]]


def test_salvar_e_carregar_reproduz_as_respostas(sistema, tmp_path):
    d = dados()
    treinar(sistema, d, d, Receita(epocas=2, taxa=3e-3, lote=4), semente=0)
    antes = sistema.prever(list(d["utt"]))
    pasta = sistema.salvar(tmp_path / "modelo")
    de_volta = SistemaEncoder.carregar(pasta)
    assert de_volta.prever(list(d["utt"])) == antes
    assert de_volta.parametros == sistema.parametros and de_volta.modelo.config == sistema.modelo.config
    with torch.no_grad():
        for a, b in zip(sistema.modelo(sistema.lote(d)), de_volta.modelo(de_volta.lote(d))):
            assert torch.allclose(a, b, atol=1e-6)
    assert sorted(p.name for p in pasta.iterdir()) == ["cabecas.pt", "config.json", "encoder"]


# --------------------------------------------------------------------------
# A perda conjunta e a destilação
# --------------------------------------------------------------------------


def test_perda_conjunta_confere_com_a_conta_a_mao(sistema):
    d = dados()
    sistema.modelo.eval()                                              # sem dropout, para a conta fechar
    lote = sistema.lote(d)
    perdas = sistema.modelo.perda(lote)
    intencoes, emissoes = sistema.modelo(lote)

    da_intencao = F.cross_entropy(intencoes, lote.intents)
    das_entidades = sum(                                               # palavra por palavra, somada no enunciado
        F.cross_entropy(emissoes[i, : len(tags)], lote.tags[i, : len(tags)], reduction="sum") for i, tags in enumerate(d["tags"])
    ) / len(d)
    assert perdas["intencao"].item() == pytest.approx(da_intencao.item(), abs=1e-5)
    assert perdas["entidades"].item() == pytest.approx(das_entidades.item(), abs=1e-4)
    assert perdas["total"].item() == pytest.approx((da_intencao + das_entidades).item(), abs=1e-4)


def logits_de(d: pd.DataFrame, intencoes: list[str], certeza: float = 5.0) -> list[np.ndarray]:
    """Logits de um professor que põe quase toda a probabilidade na intenção dada para cada pedido."""
    linhas = np.full((len(d), len(ROTULOS["intents"])), -certeza, dtype=np.float32)
    linhas[np.arange(len(d)), [ROTULOS["intents"].index(i) for i in intencoes]] = 0.0
    return list(linhas)


def test_com_temperatura_a_perda_e_a_de_destilacao_e_chega_ao_gradiente():
    d = dados()
    aluno = sistema_minusculo(config=Config(modelo_id="minusculo", revisao="", temperatura=2.0))
    aluno.modelo.eval()
    with torch.no_grad():                                              # um aluno com preferência clara por uma intenção
        aluno.modelo.intencao.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
    trocadas = ["weather_query"] * len(d)                              # o professor discorda dos rótulos rígidos e do aluno
    com_logits = d.assign(logits_professor=logits_de(d, trocadas))

    lote = aluno.lote(com_logits)
    assert lote.logits_professor.shape == (len(d), 3)
    intencoes, _ = aluno.modelo(lote)
    perdas = aluno.modelo.perda(lote)
    assert perdas["intencao"].item() == pytest.approx(perda_de_destilacao(intencoes, lote.logits_professor, 2.0).item(), abs=1e-5)

    # o gradiente da perda de intenção chega à cabeça, ao corpo do encoder e à tabela de subtokens
    aluno.modelo.zero_grad()
    perdas["intencao"].backward()
    partes = {
        "cabeça": aluno.modelo.intencao.weight.grad,
        "corpo": aluno.modelo.encoder.encoder.layer[0].attention.self.query.weight.grad,
        "tabela": aluno.modelo.encoder.get_input_embeddings().weight.grad,
    }
    assert all(g is not None and g.abs().sum() > 0 for g in partes.values()), partes
    assert aluno.modelo.entidades.weight.grad is None                  # e não passa pela cabeça de entidades

    # o que ensina é a distribuição do professor: trocar os logits muda a perda, trocar o rótulo rígido não
    outra = aluno.modelo.perda(aluno.lote(d.assign(logits_professor=logits_de(d, ["alarm_set"] * len(d)))))["intencao"].item()
    mesma = aluno.modelo.perda(aluno.lote(com_logits.assign(intent=trocadas)))["intencao"].item()
    assert outra < perdas["intencao"].item() - 1.0                     # o professor que concorda com o aluno custa menos
    assert mesma == pytest.approx(perdas["intencao"].item(), abs=1e-6)

    with pytest.raises(ValueError, match="logits_professor"):
        aluno.modelo.perda(aluno.lote(d))                              # temperatura sem logits é erro, não silêncio


def test_sem_temperatura_os_logits_sao_ignorados(sistema):
    d = dados()
    sistema.modelo.eval()
    sem = sistema.modelo.perda(sistema.lote(d))["total"].item()
    com = sistema.modelo.perda(sistema.lote(d.assign(logits_professor=logits_de(d, list(d["intent"].iloc[::-1])))))["total"].item()
    assert com == pytest.approx(sem, abs=1e-6)


def test_aluno_aprende_a_intencao_so_com_os_logits():
    d = dados()
    do_professor = [{"alarm_set": "play_music", "play_music": "alarm_set", "weather_query": "weather_query"}[i] for i in d["intent"]]
    aluno = sistema_minusculo(config=Config(modelo_id="minusculo", revisao="", temperatura=2.0))
    # o rótulo rígido que o aluno recebe é o humano; a intenção que ele aprende tem de ser a dos logits
    treino = d.assign(logits_professor=logits_de(d, do_professor))
    dev = d.assign(intent=do_professor)
    historico = treinar(aluno, treino, dev, RAPIDA, semente=0)
    assert historico["dev_intent_acc"].max() == 1.0
    assert [p.intent for p in aluno.prever(list(d["utt"]))] == do_professor


# --------------------------------------------------------------------------
# O laço de ajuste fino
# --------------------------------------------------------------------------


def test_agenda_da_taxa_sobe_e_desce_em_linha_reta():
    total = 100
    assert fracao_da_taxa(0, total, 0.1) == 0.0
    assert fracao_da_taxa(5, total, 0.1) == pytest.approx(0.5)
    assert fracao_da_taxa(10, total, 0.1) == 1.0
    assert fracao_da_taxa(55, total, 0.1) == pytest.approx(0.5)
    assert fracao_da_taxa(100, total, 0.1) == 0.0
    assert fracao_da_taxa(3, total, 0.0) == pytest.approx(0.97)        # sem aquecimento, só desce


def test_fila_de_lotes_passa_por_todos_antes_de_repetir():
    fila = fila_de_lotes(10, 4, np.random.default_rng(0))
    primeira = np.concatenate([next(fila) for _ in range(3)])          # 4 + 4 + 2: uma passada
    segunda = np.concatenate([next(fila) for _ in range(3)])
    assert sorted(primeira) == sorted(segunda) == list(range(10))
    assert primeira.tolist() != segunda.tolist()                       # embaralhada de novo


def test_treino_aprende_fica_com_o_melhor_ponto_e_conta_os_passos(sistema):
    d = dados()
    historico = treinar(sistema, d, d, RAPIDA, semente=0)
    assert len(historico) == RAPIDA.epocas and historico["melhor"].sum() == 1
    assert (historico["passos"] == ceil(len(d) / RAPIDA.lote)).all()
    assert historico["perda_total"].iloc[-1] < historico["perda_total"].iloc[0]
    assert historico["taxa"].iloc[-1] < historico["taxa"].iloc[RAPIDA.epocas // 2]      # a taxa desce até o fim

    melhor = historico.loc[historico["melhor"]].iloc[0]
    assert melhor["dev_exact_match"] == historico["dev_exact_match"].max() >= 0.75
    assert melhor["epoca"] == historico.loc[historico["dev_exact_match"].idxmax(), "epoca"]   # a primeira época com o melhor valor
    # o sistema ficou com os pesos daquele ponto, e a tabela voltou a ser a comum da biblioteca
    assert qualidade.avaliar(ouro(d), sistema.prever(list(d["utt"])))["exact_match"] == pytest.approx(melhor["dev_exact_match"])
    assert sistema.modelo.tabela_dividida is None and not sistema.modelo.training


def test_orcamento_em_passos_da_ao_conjunto_pequeno_os_mesmos_passos(sistema):
    d = dados()
    historico = treinar(sistema, d.iloc[:3], d, Receita(epocas=2, taxa=3e-3, lote=4, passos_por_epoca=7), semente=0)
    assert historico["passos"].tolist() == [7, 7]                      # 3 exemplos, 7 passos por época: várias passadas


def test_tabela_dividida_da_o_mesmo_resultado_da_tabela_inteira():
    """O treino só com as linhas usadas tem de terminar igual ao treino com a tabela inteira, linha por linha."""
    d = dados()
    poucos = d.iloc[[3, 4, 5]]                                         # só pedidos de música: muitas linhas da tabela ficam sem uso
    receita = Receita(epocas=4, taxa=3e-3, lote=2, decaimento=0.5)     # decaimento forte, para aparecer nas linhas sem gradiente
    inteira, dividida = sistema_minusculo(), sistema_minusculo()
    inicial = inteira.modelo.encoder.get_input_embeddings().weight.detach().clone()
    h_inteira = treinar(inteira, poucos, d, Receita(**{**receita.como_dict(), "tabela_dividida": False}), semente=0)
    h_dividida = treinar(dividida, poucos, d, receita, semente=0)

    tabela_a = inteira.modelo.encoder.get_input_embeddings().weight
    tabela_b = dividida.modelo.encoder.get_input_embeddings().weight
    assert isinstance(dividida.modelo.encoder.get_input_embeddings(), torch.nn.Embedding)
    assert not isinstance(dividida.modelo.encoder.get_input_embeddings(), TabelaDividida)
    assert torch.allclose(tabela_a, tabela_b, atol=1e-6)
    for (nome, a), (_, b) in zip(inteira.modelo.state_dict().items(), dividida.modelo.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-5), nome
    assert h_inteira["perda_total"].tolist() == pytest.approx(h_dividida["perda_total"].tolist(), abs=1e-5)
    assert inteira.prever(list(d["utt"])) == dividida.prever(list(d["utt"]))

    # as linhas sem uso só sofreram o decaimento de pesos: encolheram todas pelo mesmo fator, e não ficaram paradas
    usadas = dividida.alinhador.ids_usados(poucos["tokens"])
    sem_uso = torch.ones(len(inicial), dtype=torch.bool)
    sem_uso[usadas] = False
    assert 0 < sem_uso.sum() < len(inicial)
    fator = (tabela_b[sem_uso] / inicial[sem_uso]).flatten()
    assert torch.allclose(fator, fator[0].expand_as(fator), atol=1e-5) and 0.9 < fator[0].item() < 0.9999
    assert not torch.allclose(tabela_b[usadas], inicial[usadas], atol=1e-3)     # e as usadas foram treinadas


def test_copia_do_estado_nao_carrega_o_bloco_fixo(sistema):
    d = dados()
    sistema.preparar_treino(d)
    estado = sistema.modelo.copiar_estado()
    assert not any(nome.endswith((".fixa", ".posicao", ".ids_treinaveis")) for nome in estado)
    assert any(nome.endswith(".treinavel") for nome in estado) and any(nome.endswith(".escala") for nome in estado)
    with pytest.raises(RuntimeError, match="já está dividida"):
        sistema.preparar_treino(d)
