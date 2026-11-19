"""A poda do encoder pré-treinado (Etapa 10b), conferida num XLM-R minúsculo de pesos sorteados, na CPU.

O que está em jogo: a esparsidade pedida tem de ser a obtida, a máscara não
pode vazar no reajuste, o corte do vocabulário e o das camadas não podem
mudar o que não cortam, e a etapa tem de retomar de onde parou sem repetir
nem confundir execuções.
"""

import json

import pandas as pd
import pytest
import torch

from especialista.avaliacao.previsao import Previsao, salvar_previsoes
from especialista.compressao import poda, poda_encoder
from especialista.dados import massive
from especialista.experimentos import etapa10b, execucao
from especialista.modelos.encoder import Config, SistemaEncoder
from especialista.treino_encoder import Receita, treinar

from test_diet import ENUNCIADOS, ROTULOS
from test_encoder import dados

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

PEQUENO = Config(modelo_id="minusculo", revisao="", max_subtokens=64)
RAPIDA = Receita(epocas=6, taxa=1e-2, lote=4)
TEXTOS = [texto for _, texto, _ in ENUNCIADOS]
DE_FORA = ["toca jazz no quarto amanhã", "привет мир", "北京 天气"]   # só entram no vocabulário do tokenizador, não nos dados


def tokenizador_unigram():
    """Um tokenizador do tipo do XLM-R (Unigram), com o vocabulário montado à mão para não depender de sorteio.

    As palavras das frases de fora e três palavras dos dados têm peça própria;
    as outras são partidas letra a letra. Assim os dados usam todas as suas
    letras, e as peças das palavras de fora ficam sem uso.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    palavras = sorted({palavra for texto in TEXTOS + DE_FORA for palavra in texto.split()})
    inteiras = sorted({palavra for texto in DE_FORA for palavra in texto.split()} | {"toca", "amanhã", "alarme"})
    letras = sorted({letra for palavra in palavras for letra in palavra})
    vocabulario = [("<s>", 0.0), ("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -2.0)]
    vocabulario += [(letra, -2.0) for letra in letras] + [("▁" + palavra, -1.0) for palavra in inteiras]
    tok = Tokenizer(models.Unigram(vocabulario, unk_id=3, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace()])
    tok.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    classe = getattr(transformers, "PreTrainedTokenizerFast", None) or transformers.TokenizersBackend
    return classe(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>", cls_token="<s>", sep_token="</s>")


def sistema_minusculo(semente: int = 0, camadas: int = 3, config: Config = PEQUENO) -> SistemaEncoder:
    tokenizador = tokenizador_unigram()
    torch.manual_seed(semente)
    do_encoder = transformers.XLMRobertaConfig(
        vocab_size=len(tokenizador), hidden_size=32, num_hidden_layers=camadas, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=66, pad_token_id=tokenizador.pad_token_id, type_vocab_size=1,
    )
    encoder = transformers.XLMRobertaModel(do_encoder, add_pooling_layer=False)
    return SistemaEncoder.novo(ROTULOS, config, semente=semente, encoder=encoder, tokenizador=tokenizador)


def pontuacoes(sistema: SistemaEncoder, textos=TEXTOS) -> tuple[torch.Tensor, torch.Tensor]:
    sistema.modelo.eval()
    with torch.no_grad():
        return sistema.modelo(sistema.alinhador.montar([texto.split() for texto in textos]).para(sistema.dispositivo))


def zeros(sistema: SistemaEncoder) -> dict[str, int]:
    return {nome: int((sistema.modelo.get_parameter(nome) == 0).sum()) for nome in poda_encoder.nomes_podaveis(sistema.modelo)}


def estado(sistema: SistemaEncoder) -> dict[str, torch.Tensor]:
    return {nome: tensor.clone() for nome, tensor in sistema.modelo.state_dict().items()}


# --------------------------------------------------------------------------
# Pesos soltos: máscaras e esparsidade
# --------------------------------------------------------------------------


def test_as_matrizes_podaveis_sao_as_seis_de_cada_camada_do_corpo():
    sistema = sistema_minusculo(camadas=3)
    nomes = poda_encoder.nomes_podaveis(sistema.modelo)
    assert len(nomes) == 6 * 3 and len(set(nomes)) == len(nomes)
    d, densa = 32, 64
    assert sum(sistema.modelo.get_parameter(n).numel() for n in nomes) == 3 * (4 * d * d + 2 * d * densa)
    assert all(sistema.modelo.get_parameter(n).ndim == 2 for n in nomes)
    assert not any("embeddings" in n or n.startswith(("intencao", "entidades")) for n in nomes)   # a tabela e as cabeças ficam fora
    # no XLM-R base: 12 camadas, d = 768 e rede densa de 3.072
    assert 12 * (4 * 768 * 768 + 2 * 768 * 3072) == 84_934_656


@pytest.mark.parametrize("esparsidade", [0.3, 0.5, 0.7, 0.9])
def test_a_esparsidade_pedida_e_a_obtida(esparsidade):
    sistema = sistema_minusculo()
    antes = estado(sistema)
    mascaras = poda_encoder.mascaras_por_magnitude(sistema.modelo, esparsidade)
    poda_encoder.aplicar(sistema.modelo, mascaras)
    for nome, em_zero in zeros(sistema).items():
        peso = sistema.modelo.get_parameter(nome)
        assert em_zero == round(esparsidade * peso.numel())                 # nem um a mais, em cada matriz
        ficou, saiu = antes[nome][mascaras[nome]].abs(), antes[nome][~mascaras[nome]].abs()
        assert saiu.max() <= ficou.min()                                    # saíram os de menor valor absoluto
    podaveis = set(mascaras)
    assert all(torch.equal(tensor, antes[nome]) for nome, tensor in sistema.modelo.state_dict().items() if nome not in podaveis)
    contagem = poda_encoder.contar(sistema.modelo)
    assert contagem["esparsidade"] == pytest.approx(esparsidade, abs=1e-3)
    assert contagem["podaveis"] - contagem["nao_nulos_podaveis"] == sum(zeros(sistema).values())


def test_a_mascara_so_aperta_e_e_lida_de_volta_dos_zeros():
    sistema = sistema_minusculo()
    primeira = poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.3)
    poda_encoder.aplicar(sistema.modelo, primeira)
    with torch.no_grad():   # o treino mexe nos pesos que ficaram; os podados continuam em zero
        for nome in primeira:
            sistema.modelo.get_parameter(nome).add_(torch.randn_like(sistema.modelo.get_parameter(nome)) * primeira[nome])
    segunda = poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.6, anteriores=primeira)
    for nome in primeira:
        assert not (segunda[nome] & ~primeira[nome]).any()                  # quem saiu não volta
        assert int((~segunda[nome]).sum()) == round(0.6 * segunda[nome].numel())
    poda_encoder.aplicar(sistema.modelo, segunda)
    lidas = poda_encoder.mascaras_dos_zeros(sistema.modelo)
    assert all(torch.equal(lidas[nome], segunda[nome]) for nome in segunda)
    with pytest.raises(ValueError):
        poda_encoder.mascaras_por_magnitude(sistema.modelo, 1.0)


# --------------------------------------------------------------------------
# O reajuste
# --------------------------------------------------------------------------


def test_reajustar_sem_poda_e_sem_ponto_de_partida_e_o_treinar():
    d = dados()
    a, b = sistema_minusculo(), sistema_minusculo()
    do_treino = treinar(a, d, d, RAPIDA, semente=3)
    do_reajuste = poda_encoder.reajustar(b, d, d, RAPIDA, semente=3, partida_concorre=False)
    assert all(torch.equal(tensor, b.modelo.state_dict()[nome]) for nome, tensor in a.modelo.state_dict().items())
    comuns = [c for c in do_treino.columns if c != "segundos"]
    pd.testing.assert_frame_equal(do_treino[comuns], do_reajuste[comuns])
    assert (do_reajuste["esparsidade"] == 0).all()


def test_a_mascara_travada_continua_zerada_depois_do_reajuste():
    d = dados()
    sistema = sistema_minusculo()
    mascaras = poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.5)
    poda_encoder.aplicar(sistema.modelo, mascaras)
    antes = estado(sistema)
    historico = poda_encoder.reajustar(sistema, d, d, RAPIDA, semente=0, mascaras=mascaras)
    assert int(historico.loc[historico["melhor"], "epoca"].iloc[0]) > 0     # o reajuste andou: o modelo não é o de partida
    for nome, mascara in mascaras.items():
        peso = sistema.modelo.get_parameter(nome)
        assert not peso[~mascara].any()                                     # tudo o que saiu continua em zero
        assert not torch.equal(peso, antes[nome])                           # e o que ficou foi reajustado
    assert zeros(sistema) == {nome: int((~mascara).sum()) for nome, mascara in mascaras.items()}
    assert historico["esparsidade"].tolist() == pytest.approx([0.5] * len(historico), abs=1e-3)
    assert sistema.modelo.tabela_dividida is None                           # o modelo volta a ser um encoder comum


def test_sem_travar_a_mascara_os_zeros_se_perdem():
    d = dados()
    sistema = sistema_minusculo()
    mascaras = poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.5)
    poda_encoder.aplicar(sistema.modelo, mascaras)
    treinar(sistema, d, d, RAPIDA)                                          # o laço comum não sabe da máscara
    assert sum(zeros(sistema).values()) < 0.1 * sum(int((~m).sum()) for m in mascaras.values())


def test_o_ponto_de_partida_concorre_a_melhor_ponto():
    d = dados()
    sistema = sistema_minusculo()
    treinar(sistema, d, d, Receita(epocas=40, taxa=1e-2, lote=4))
    antes = estado(sistema)
    historico = poda_encoder.reajustar(sistema, d, d, Receita(epocas=2, taxa=50.0, lote=4), semente=0)   # uma taxa que estraga a rede
    assert historico["epoca"].tolist() == [0, 1, 2]
    assert historico.loc[historico["melhor"], "epoca"].tolist() == [0]
    assert historico.loc[0, "dev_exact_match"] > historico.loc[1:, "dev_exact_match"].max()
    assert all(torch.equal(tensor, antes[nome]) for nome, tensor in sistema.modelo.state_dict().items())   # saiu como entrou


def test_poda_gradual_chega_na_esparsidade_e_so_escolhe_epoca_depois_da_rampa():
    d = dados()
    sistema = sistema_minusculo()
    agenda = poda_encoder.Agenda(0.7, epocas_de_rampa=3, passos_entre_cortes=2)   # 2 passos por época: 3 cortes
    historico = poda_encoder.reajustar(sistema, d, d, RAPIDA, semente=0, agenda=agenda)
    assert historico["epoca"].tolist() == [1, 2, 3, 4, 5, 6]                # sem época 0: a partida não está na esparsidade final
    esparsidades = historico["esparsidade"].tolist()
    assert esparsidades[:3] == sorted(esparsidades[:3]) and esparsidades[0] < esparsidades[2]
    assert esparsidades[0] == pytest.approx(poda.esparsidade_cubica(2, 6, 0.7), abs=1e-3)
    assert esparsidades[2:] == pytest.approx([0.7] * 4, abs=1e-3)
    assert int(historico.loc[historico["melhor"], "epoca"].iloc[0]) >= 3
    for nome, em_zero in zeros(sistema).items():
        assert em_zero == round(0.7 * sistema.modelo.get_parameter(nome).numel())


def test_poda_gradual_recusa_o_que_nao_cabe():
    d = dados()
    with pytest.raises(ValueError):
        poda_encoder.reajustar(sistema_minusculo(), d, d, RAPIDA, agenda=poda_encoder.Agenda(0.5, epocas_de_rampa=7))
    with pytest.raises(ValueError):
        poda_encoder.reajustar(sistema_minusculo(), d, d, RAPIDA, agenda=poda_encoder.Agenda(0.5, 1, passos_entre_cortes=100))
    sistema = sistema_minusculo()
    with pytest.raises(ValueError):
        poda_encoder.reajustar(sistema, d, d, RAPIDA, mascaras=poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.5),
                               agenda=poda_encoder.Agenda(0.5, 1, 2))


# --------------------------------------------------------------------------
# Peças inteiras: o vocabulário
# --------------------------------------------------------------------------


def test_cortar_o_vocabulario_preserva_as_respostas_quando_nada_do_texto_e_cortado():
    sistema = sistema_minusculo()
    linhas = len(sistema.tokenizador)
    ficam = poda_encoder.ids_do_texto(sistema, [texto.split() for texto in TEXTOS])
    assert set(sistema.tokenizador.all_special_ids) <= set(ficam.tolist()) and len(ficam) < linhas   # há o que cortar
    antes = pontuacoes(sistema)
    menor = poda_encoder.cortar_vocabulario(sistema, ficam)

    tabela = menor.modelo.encoder.get_input_embeddings()
    assert tabela.num_embeddings == len(menor.tokenizador) == menor.modelo.encoder.config.vocab_size == len(ficam)
    assert torch.equal(tabela.weight, sistema.modelo.encoder.get_input_embeddings().weight[ficam])
    depois = pontuacoes(menor)
    assert torch.equal(antes[0], depois[0]) and torch.equal(antes[1], depois[1])                     # as mesmas pontuações
    assert menor.prever(TEXTOS) == sistema.prever(TEXTOS)
    # os tokens especiais não mudam de lugar; os outros passam a ter a posição que têm em `ficam`
    assert [menor.tokenizador.convert_tokens_to_ids(t) for t in ("<s>", "<pad>", "</s>", "<unk>")] == [0, 1, 2, 3]
    original = sistema.tokenizador([t.split() for t in TEXTOS], is_split_into_words=True)["input_ids"]
    reduzido = menor.tokenizador([t.split() for t in TEXTOS], is_split_into_words=True)["input_ids"]
    novo_id = {antigo: novo for novo, antigo in enumerate(ficam.tolist())}
    assert reduzido == [[novo_id[i] for i in linha] for linha in original]
    # o sistema recebido não muda
    assert sistema.modelo.encoder.get_input_embeddings().num_embeddings == linhas == len(sistema.tokenizador)
    assert menor.parametros == sistema.parametros - (linhas - len(ficam)) * 32


def test_palavra_com_peca_cortada_e_partida_de_novo_ou_vira_desconhecida():
    sistema = sistema_minusculo()
    ficam = poda_encoder.ids_do_texto(sistema, [texto.split() for texto in TEXTOS])
    menor = poda_encoder.cortar_vocabulario(sistema, ficam)
    de_fora = [texto.split() for texto in DE_FORA]

    no_treino = poda_encoder.cobertura(sistema.tokenizador, menor.tokenizador, ficam, [t.split() for t in TEXTOS])
    assert no_treino["subtokens_cortados"] == no_treino["subtokens_desconhecidos"] == 0
    assert no_treino["subtokens_antes"] == no_treino["subtokens_depois"] and no_treino["enunciados"] == len(TEXTOS)

    fora = poda_encoder.cobertura(sistema.tokenizador, menor.tokenizador, ficam, de_fora)
    assert fora["subtokens_cortados"] > 0 and fora["enunciados_com_corte"] == 3
    assert fora["subtokens_desconhecidos"] > 0                               # o cirílico e o chinês não têm mais peça nenhuma
    assert 0 < fora["fracao_desconhecida"] < 1 and 0 < fora["fracao_cortada"] <= 1
    ids = menor.tokenizador(de_fora, is_split_into_words=True)["input_ids"]
    assert max(i for linha in ids for i in linha) < len(ficam)               # nenhum id aponta para fora da tabela
    assert len(menor.prever(DE_FORA)) == 3                                   # e o sistema continua recebendo qualquer texto
    # "quarto" tinha peça própria, que foi cortada; as letras dela estão nos dados: é partida de novo, sem <unk>
    assert len(sistema.tokenizador.tokenize("quarto")) == 1
    de_novo = menor.tokenizador.tokenize("quarto")
    assert de_novo == ["▁", "q", "u", "a", "r", "t", "o"] and "<unk>" not in de_novo
    ids_de = lambda palavra: menor.tokenizador([[palavra]], is_split_into_words=True, add_special_tokens=False)["input_ids"][0]
    assert menor.tokenizador.unk_token_id not in ids_de("quarto")
    assert menor.tokenizador.unk_token_id in ids_de("jazz")                  # já o j e o z não aparecem nos dados


def test_ids_por_alfabeto_fica_com_o_latino_e_tira_os_outros():
    tokenizador = tokenizador_unigram()
    ficam = set(poda_encoder.ids_por_alfabeto(tokenizador).tolist())
    vocabulario = tokenizador.get_vocab()
    assert set(tokenizador.all_special_ids) <= ficam
    fora = {peca for peca, i in vocabulario.items() if i not in ficam}
    assert fora and all(any(c in peca for c in "приветмир北京天气") for peca in fora)     # só saem o cirílico e o chinês
    assert all(i in ficam for peca, i in vocabulario.items() if peca.strip("▁").isascii())
    assert all(vocabulario[peca] in ficam for peca in vocabulario if set(peca) <= set("▁ãõáàçéêíóôú"))
    for sim in "aZ9-,.?!ãçéõü€«»–▁'":
        assert poda_encoder.do_alfabeto_latino(sim), sim
    for nao in "яب北ह한ア":
        assert not poda_encoder.do_alfabeto_latino(nao), nao


def test_o_sistema_cortado_grava_rele_e_o_tokenizador_encolhe(tmp_path):
    sistema = sistema_minusculo()
    inteiro = sistema.salvar(tmp_path / "inteiro")
    menor = poda_encoder.cortar_vocabulario(sistema, poda_encoder.ids_do_texto(sistema, [t.split() for t in TEXTOS]))
    pasta = menor.salvar(tmp_path / "menor")
    relido = SistemaEncoder.carregar(pasta)
    assert relido.prever(TEXTOS + DE_FORA) == menor.prever(TEXTOS + DE_FORA)
    assert (pasta / "encoder" / "tokenizer.json").stat().st_size < (inteiro / "encoder" / "tokenizer.json").stat().st_size
    assert (pasta / "encoder" / "model.safetensors").stat().st_size < (inteiro / "encoder" / "model.safetensors").stat().st_size
    with pytest.raises(ValueError):   # o corte foi escrito para o tipo de tokenizador do XLM-R
        from test_encoder import tokenizador_minusculo
        poda_encoder.reduzir_tokenizador(tokenizador_minusculo(), [0, 1, 2, 3])


# --------------------------------------------------------------------------
# Peças inteiras: as camadas
# --------------------------------------------------------------------------


def test_manter_camadas_fica_com_as_de_baixo_e_as_cabecas_leem_da_ultima_que_ficou():
    sistema = sistema_minusculo(camadas=3)
    menor = poda_encoder.manter_camadas(sistema, 2)
    assert len(menor.modelo.encoder.encoder.layer) == menor.modelo.encoder.config.num_hidden_layers == 2
    assert len(sistema.modelo.encoder.encoder.layer) == sistema.modelo.encoder.config.num_hidden_layers == 3   # o original não muda
    for i in range(2):
        de_la, de_ca = sistema.modelo.encoder.encoder.layer[i].state_dict(), menor.modelo.encoder.encoder.layer[i].state_dict()
        assert all(torch.equal(de_la[nome], de_ca[nome]) for nome in de_la)

    lote = sistema.alinhador.montar([texto.split() for texto in TEXTOS])
    sistema.modelo.eval()
    with torch.no_grad():
        estados = sistema.modelo.encoder(input_ids=lote.ids, attention_mask=lote.atencao, output_hidden_states=True).hidden_states[2]
        esperado = sistema.modelo.intencao(estados[:, 0])
    assert torch.allclose(pontuacoes(menor)[0], esperado, atol=1e-6)
    umas = sum(p.numel() for p in sistema.modelo.encoder.encoder.layer[2].parameters())
    assert menor.parametros == sistema.parametros - umas
    assert len(poda_encoder.nomes_podaveis(menor.modelo)) == 12
    for demais in (0, 4):
        with pytest.raises(ValueError):
            poda_encoder.manter_camadas(sistema, demais)


def test_as_duas_pecas_juntas_gravam_e_relem(tmp_path):
    sistema = sistema_minusculo(camadas=3)
    menor = poda_encoder.cortar_vocabulario(sistema, poda_encoder.ids_do_texto(sistema, [t.split() for t in TEXTOS]))
    menor = poda_encoder.manter_camadas(menor, 1)
    relido = SistemaEncoder.carregar(menor.salvar(tmp_path / "menor"))
    assert relido.prever(TEXTOS) == menor.prever(TEXTOS)
    assert relido.parametros == menor.parametros < sistema.parametros
    assert len(relido.modelo.encoder.encoder.layer) == 1


# --------------------------------------------------------------------------
# Armazenamento e execução que aproveitam os zeros
# --------------------------------------------------------------------------


def test_arquivo_esparso_do_encoder_devolve_os_mesmos_pesos_e_e_menor(tmp_path):
    sistema = sistema_minusculo()
    cheio = poda_encoder.salvar_esparso(sistema, tmp_path / "cheio")
    poda_encoder.aplicar(sistema.modelo, poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.9))
    arquivo = poda_encoder.salvar_esparso(sistema, tmp_path / "esparso")
    relido = poda_encoder.carregar_esparso(tmp_path / "esparso")
    original = sistema.modelo.state_dict()
    assert set(relido.modelo.state_dict()) == set(original)
    assert all(torch.equal(tensor, original[nome]) for nome, tensor in relido.modelo.state_dict().items())
    assert relido.prever(TEXTOS) == sistema.prever(TEXTOS)
    assert arquivo.stat().st_size < cheio.stat().st_size

    pasta = sistema.salvar(tmp_path / "comum")
    medidas = poda_encoder.tamanhos(pasta, arquivo)
    pesos = (pasta / "encoder" / "model.safetensors").stat().st_size + (pasta / "cabecas.pt").stat().st_size
    assert medidas["rede"]["arquivo_comum"] == pesos
    assert medidas["rede"]["arquivo_esparso"] == arquivo.stat().st_size
    assert medidas["rede"]["arquivo_comum_com_zip"] < pesos                  # os zeros quase somem no zip
    assert medidas["tokenizador_e_configuracao"]["arquivo_comum"] >= (pasta / "encoder" / "tokenizer.json").stat().st_size
    total = sum(a.stat().st_size for a in pasta.rglob("*") if a.is_file())
    assert medidas["sistema"]["arquivo_comum"] == total                      # rede + tokenizador = a pasta inteira
    for formato in ("arquivo_comum", "arquivo_comum_com_zip", "arquivo_esparso"):
        assert medidas["sistema"][formato] == medidas["rede"][formato] + medidas["tokenizador_e_configuracao"][formato]
    um = pasta / "encoder" / "model.safetensors"
    assert poda_encoder.bytes_com_zip(um, pedaco=1000) == poda_encoder.bytes_com_zip(um) == poda.bytes_com_zip(um)


def test_execucao_esparsa_da_as_mesmas_respostas():
    sistema = sistema_minusculo()
    poda_encoder.aplicar(sistema.modelo, poda_encoder.mascaras_por_magnitude(sistema.modelo, 0.8))
    esparsa = poda_encoder.execucao_esparsa(sistema)
    trocadas = [m for m in esparsa.modelo.modules() if isinstance(m, poda.LinearEsparsa)]
    assert len(trocadas) == len(poda_encoder.nomes_podaveis(sistema.modelo))
    a, b = pontuacoes(sistema), pontuacoes(esparsa)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)
    assert esparsa.prever(TEXTOS) == sistema.prever(TEXTOS)


# --------------------------------------------------------------------------
# A Etapa 10b em miniatura
# --------------------------------------------------------------------------

MODELOS = {"aluno_a": ("completa", True), "aluno_b2_t2": ("do_principal", False), "teto": ("completa", False)}
RECEITA = {"epocas": 10, "taxa": 1e-2, "lote": 4}


@pytest.fixture
def cenario(tmp_path, monkeypatch):
    """Três modelos de partida minúsculos (12 camadas, 3 sementes), no lugar das entregas da Etapa 10a."""
    d = dados()
    monkeypatch.setattr(massive, "carregar_entrega", lambda: ({"train": d, "dev": d, "test": d}, ROTULOS))
    monkeypatch.setattr(execucao, "EXECUCOES", tmp_path / "execucoes")
    monkeypatch.setattr(etapa10b, "RAIZ", tmp_path)
    monkeypatch.setattr(etapa10b, "EPOCAS", 3)
    monkeypatch.setattr(etapa10b, "EPOCAS_DE_RAMPA", 2)
    monkeypatch.setattr(etapa10b, "PASSOS_ENTRE_CORTES", 2)
    for curta in etapa10b.CURTAS.values():
        monkeypatch.setitem(curta, "epocas", 2)

    fichas = {}
    for i, (chave, (grade, principal)) in enumerate(MODELOS.items()):
        pastas = {}
        for semente in etapa10b.SEMENTES:
            sistema = sistema_minusculo(semente=10 * i + semente, camadas=12)
            pastas[semente] = sistema.salvar(tmp_path / "entregas" / "modelos" / chave / f"semente_{semente}")
            for particao in ("dev", "test"):
                salvar_previsoes(tmp_path / "entregas" / "previsoes" / f"{chave}_semente_{semente}_{particao}.parquet", d["id"],
                                 sistema.prever(list(d["utt"])))
        fichas[chave] = {
            "nome": chave, "curto": chave, "variante_da_10a": chave, "pastas": pastas, "config": PEQUENO.como_dict(), "receita": RECEITA,
            "sinal": {"treino": "de brinquedo", "dev": "de brinquedo"},
            "previsoes": str(tmp_path / "entregas" / "previsoes" / (chave + "_semente_{semente}_{particao}.parquet")),
            "grade": grade, "principal": principal,
        }
    monkeypatch.setattr(etapa10b, "modelos_a_podar", lambda: fichas)
    monkeypatch.setattr(etapa10b, "dados_do_modelo", lambda modelo: {"treino": d, "dev": d, "origem": fichas[modelo]["sinal"]})
    return {"dados": d, "pasta": tmp_path, "fichas": fichas}


def test_o_desenho_e_o_declarado():
    assert list(etapa10b.VARIANTES) == [
        "sem_poda", "vocabulario_treino", "vocabulario_alfabeto", "camadas_8", "camadas_6",
        "vocabulario_treino_camadas_8", "vocabulario_treino_camadas_6",
        "magnitude_90", "gradual_90", "magnitude_70", "gradual_70", "magnitude_50", "gradual_50", "magnitude_30", "gradual_30",
    ]
    assert len(etapa10b.COMBINACOES) <= 2 and len(etapa10b.FORMAS) <= 2
    assert list(etapa10b.CURTAS) == ["magnitude_90_curto", "camadas_6_curto"]
    assert all(etapa10b.definicao(v)["epocas"] == etapa10b.EPOCAS == 10 for v in etapa10b.VARIANTES)   # as do ajuste fino da Etapa 10a
    assert all(etapa10b.definicao(v)["epocas"] == 4 and etapa10b.definicao(v)["familia"] == "curto" for v in etapa10b.CURTAS)
    assert etapa10b.EPOCAS_DE_RAMPA == 5 and etapa10b.PASSOS_ENTRE_CORTES == 100
    assert etapa10b.orcamento("gradual_50") == {"epocas": 10, "epocas_de_rampa": 5, "passos_entre_cortes": 100}
    assert etapa10b.orcamento("camadas_6_curto") == {"epocas": 4, "epocas_de_rampa": None, "passos_entre_cortes": None}
    familias = {v["familia"] for v in etapa10b.VARIANTES.values()}
    assert familias == {"controle", *etapa10b.PESOS_SOLTOS, *etapa10b.PECAS_INTEIRAS}


def test_a_grade_do_cenario(cenario):
    # antes da decisão do aluno principal: as duas grades completas e o controle do orçamento; o outro aluno espera
    assert len(etapa10b.grade()) == 2 * 15 * 3 + 2 * 3
    assert etapa10b.variantes_do_modelo("aluno_b2_t2") is None
    assert etapa10b.faltam()[0] == ("aluno_a", "sem_poda", 0) and etapa10b.faltam()[-1] == ("aluno_a", "camadas_6_curto", 2)
    assert etapa10b.modelo_principal() == "aluno_a"
    assert "espera a grade de aluno_a" in etapa10b.situacao()


@pytest.mark.parametrize("variante", ["sem_poda", "magnitude_50", "gradual_90", "vocabulario_treino", "vocabulario_alfabeto",
                                      "camadas_6", "vocabulario_treino_camadas_8", "magnitude_90_curto"])
def test_cada_variante_roda_e_grava_o_que_promete(cenario, variante):
    pasta = etapa10b.executar("aluno_a", variante, 1, "cpu")
    lido = etapa10b.carregar("aluno_a", variante, 1)
    meta, historico, propria = lido["meta"], lido["historico"], etapa10b.definicao(variante)
    familia = propria.get("familia_de_base", propria["familia"])
    assert meta["familia"] == propria["familia"] and meta["receita"]["epocas"] == propria["epocas"] == (2 if "curto" in variante else 3)
    assert meta["receita"]["taxa"] == RECEITA["taxa"] and meta["receita"]["lote"] == RECEITA["lote"]   # a receita de quem treinou
    assert meta["de_partida"] == "entregas/modelos/aluno_a/semente_1"
    assert historico["epoca"].tolist() == list(range(0 if familia != "gradual" else 1, propria["epocas"] + 1))
    assert set(lido["previsoes"]) == ({"dev", "test"} if familia == "gradual" else {"dev", "test", "dev_antes", "test_antes"})
    assert (meta["dev_antes"] is None) == (familia == "gradual")

    # a esparsidade pedida é a obtida, e só nos pesos soltos
    pedida = propria["nivel"] / 100 if familia in etapa10b.PESOS_SOLTOS else 0.0
    assert meta["contagem"]["esparsidade"] == pytest.approx(pedida, abs=1e-3)
    assert meta["contagem_de_partida"]["esparsidade"] == 0.0

    # as peças que saíram
    assert meta["camadas"] == propria.get("camadas", 12)
    partida = cenario["fichas"]["aluno_a"]["pastas"][1]
    linhas_de_partida = len(SistemaEncoder.carregar(partida).tokenizador)
    if "forma" in propria:
        assert meta["vocabulario"]["linhas_antes"] == linhas_de_partida > meta["vocabulario"]["linhas"]
        assert meta["vocabulario"]["cobertura"]["treino"]["subtokens_cortados"] == 0     # os dados de brinquedo são todos latinos
        assert set(meta["vocabulario"]["cobertura"]) == {"treino", "dev", "test"}
    else:
        assert meta["vocabulario"] is None
    assert (meta["contagem"]["parametros"] < meta["contagem_de_partida"]["parametros"]) == (familia in ("vocabulario", "camadas", "combinacao"))

    # o modelo gravado é uma pasta comum de SistemaEncoder e reproduz as respostas gravadas
    relido = SistemaEncoder.carregar(pasta / "modelo")
    assert relido.prever(list(cenario["dados"]["utt"])) == lido["previsoes"]["test"]
    assert meta["bytes_em_disco"] == sum(a.stat().st_size for a in (pasta / "modelo").rglob("*") if a.is_file())
    assert etapa10b.pronta("aluno_a", variante, 1) and ("aluno_a", variante, 1) not in etapa10b.faltam()


def test_o_controle_sem_poda_que_nao_melhora_devolve_o_modelo_de_partida(cenario, monkeypatch):
    fichas = cenario["fichas"]
    monkeypatch.setitem(fichas["teto"], "receita", {**RECEITA, "taxa": 50.0})   # um reajuste que só estraga
    pasta = etapa10b.executar("teto", "sem_poda", 0, "cpu")
    lido = etapa10b.carregar("teto", "sem_poda", 0)
    assert lido["meta"]["melhor_epoca"] == 0
    assert lido["previsoes"]["test"] == lido["previsoes"]["test_antes"] == etapa10b.previsoes_sem_poda("teto", "test")[0]
    partida, relido = SistemaEncoder.carregar(fichas["teto"]["pastas"][0]), SistemaEncoder.carregar(pasta / "modelo")
    assert all(torch.equal(t, relido.modelo.state_dict()[n]) for n, t in partida.modelo.state_dict().items())


def test_retomada_nao_repete_o_que_esta_pronto_e_refaz_o_que_ficou_pela_metade(cenario):
    pasta = etapa10b.executar("teto", "camadas_6", 0, "cpu")
    gravado_em = (pasta / "meta.json").stat().st_mtime_ns
    assert etapa10b.executar("teto", "camadas_6", 0, "cpu") == pasta
    assert (pasta / "meta.json").stat().st_mtime_ns == gravado_em             # não rodou de novo

    # sem os pesos (apagados para poupar disco), a execução continua pronta: o que conta é o meta.json
    import shutil
    shutil.rmtree(pasta / "modelo")
    assert etapa10b.pronta("teto", "camadas_6", 0)

    # uma pasta sem meta.json é sobra de execução interrompida: é refeita
    (pasta / "meta.json").unlink()
    (pasta / "lixo.txt").write_text("sobra")
    assert ("teto", "camadas_6", 0) in etapa10b.faltam()
    etapa10b.executar("teto", "camadas_6", 0, "cpu")
    assert not (pasta / "lixo.txt").exists() and (pasta / "modelo" / "config.json").exists()

    # se o orçamento do reajuste mudar, a execução antiga deixa de valer; se voltar ao que era, vale de novo
    etapa10b.EPOCAS = 4       # (o `cenario` devolve o valor no fim do teste)
    assert not etapa10b.pronta("teto", "camadas_6", 0)
    etapa10b.EPOCAS = 3
    assert etapa10b.pronta("teto", "camadas_6", 0)

    # se o modelo de partida for treinado de novo, a execução antiga deixa de valer
    partida = cenario["fichas"]["teto"]["pastas"][0]
    sistema = SistemaEncoder.carregar(partida)
    with torch.no_grad():
        sistema.modelo.intencao.bias.add_(1.0)
    sistema.salvar(partida)
    assert not etapa10b.pronta("teto", "camadas_6", 0) and ("teto", "camadas_6", 0) in etapa10b.faltam()


def test_modelo_de_partida_com_outra_configuracao_e_recusado(cenario, monkeypatch):
    monkeypatch.setitem(cenario["fichas"]["teto"], "config", {**PEQUENO.como_dict(), "dropout": 0.5})
    with pytest.raises(ValueError):
        etapa10b.executar("teto", "sem_poda", 0, "cpu")


# --------------------------------------------------------------------------
# A regra, o outro aluno e o disco
# --------------------------------------------------------------------------

GABARITO = [Previsao("alarm_set", (("date", 0, 1),))] * 400


def respostas(erradas: int, desde: int = 0) -> list[Previsao]:
    """O gabarito com `erradas` pedidos errados, a partir do pedido `desde`."""
    return [Previsao("outra", ()) if desde <= i < desde + erradas else p for i, p in enumerate(GABARITO)]


def candidato(erradas: int, tamanho: int = 1000, deslocado: int = 0) -> dict:
    """Um candidato com `erradas` pedidos errados por semente. Com `deslocado`, erra pedidos diferentes dos do modelo sem poda."""
    return {"em_dev": [respostas(erradas, desde=40 * s + deslocado) for s in range(3)], "nao_nulos": 100, "bytes": tamanho}


def test_quem_segue():
    sem_poda = [respostas(40, desde=40 * s) for s in range(3)]   # 90% de exact match
    decisao = etapa10b.quem_segue(GABARITO, sem_poda, {
        "vocabulario_alfabeto": candidato(40, tamanho=600),
        "vocabulario_treino": candidato(44, tamanho=300, deslocado=20),   # acerta 20 que o sem poda errava e erra 24 outros: não distinguível, passa
        "camadas_6": candidato(120, tamanho=900),                         # bem pior: não passa
        "vocabulario_treino_camadas_6": candidato(130, tamanho=150),      # a menor de todas, e não passa
        "magnitude_30": candidato(40), "magnitude_50": candidato(42, deslocado=20), "gradual_50": candidato(41, deslocado=20),
        "magnitude_70": candidato(125), "gradual_70": candidato(130),
    })
    tabela = decisao["tabela"]
    assert tabela["passa"].to_dict() == {
        "vocabulario_alfabeto": True, "vocabulario_treino": True, "camadas_6": False, "vocabulario_treino_camadas_6": False,
        "magnitude_30": True, "magnitude_50": True, "gradual_50": True, "magnitude_70": False, "gradual_70": False,
    }
    assert decisao["principal"] == "vocabulario_treino"       # a que passa com o menor arquivo comum
    assert decisao["secundaria"] == "gradual_50"              # pesos soltos, maior esparsidade que passa; entre as duas, a de maior exact match
    assert decisao["seguem"] == ["vocabulario_treino", "gradual_50"]
    assert tabela.loc["camadas_6", "ic_max"] < 0 <= tabela.loc["vocabulario_treino", "ic_max"]
    assert tabela.loc["vocabulario_treino", "diferenca"] == pytest.approx(-4 / 400)

    # empate entre as duas formas no mesmo nível: fica a de uma vez
    assert etapa10b.quem_segue(GABARITO, sem_poda, {"magnitude_50": candidato(40), "gradual_50": candidato(40)})["secundaria"] == "magnitude_50"
    # nada passa: a principal é o modelo sem poda e não há secundária
    nada = etapa10b.quem_segue(GABARITO, sem_poda, {"camadas_6": candidato(150), "magnitude_90": candidato(150)})
    assert nada["principal"] is None and nada["secundaria"] is None and nada["seguem"] == []
    vazio = etapa10b.quem_segue(GABARITO, sem_poda, {})
    assert vazio["seguem"] == [] and len(vazio["tabela"]) == 0


def fingir(cenario, modelo: str, variante: str, erradas: int, tamanho: int = 1000) -> None:
    """Grava as 3 sementes de uma execução como se tivesse rodado: respostas, meta e uma pasta de pesos."""
    d = cenario["dados"]
    certas = etapa10b.previsoes_sem_poda(modelo, "dev")
    for semente in etapa10b.SEMENTES:
        pasta = etapa10b.pasta_da_execucao(modelo, variante, semente)
        (pasta / "modelo").mkdir(parents=True)
        (pasta / "modelo" / "config.json").write_text("{}")
        previsoes = [Previsao("outra", ()) if i < erradas else p for i, p in enumerate(certas[semente])]
        salvar_previsoes(pasta / "previsoes_dev.parquet", d["id"], previsoes)
        pd.DataFrame({"epoca": [0]}).to_csv(pasta / "historico.csv", index=False)
        etapa10b._gravar_json(pasta / "meta.json", {
            "digital_de_partida": etapa10b.digital_de_partida(modelo, semente), "orcamento": etapa10b.orcamento(variante),
            "contagem": {"nao_nulos": 100}, "bytes_em_disco": tamanho,
        })


def test_o_outro_aluno_recebe_o_que_segue_do_principal_e_o_disco_e_limpo(cenario, monkeypatch):
    # com 8 pedidos de brinquedo o intervalo não separa nada; aqui a regra do "passa" é trocada por uma conta direta
    monkeypatch.setattr(etapa10b, "linha_da_regra", lambda gabarito, sem_poda, em_dev: (lambda erros: {
        "sem_poda": 1.0, "podado": 1 - erros / 8, "diferenca": -erros / 8, "ic_min": -erros / 8, "ic_max": -erros / 8 + 0.2,
        "passa": erros < 2,
    })(sum(p != q for p, q in zip(sem_poda[0], em_dev[0]))))
    erros = {"sem_poda": 0, "vocabulario_treino": 1, "vocabulario_alfabeto": 0, "camadas_8": 0, "camadas_6": 4,
             "vocabulario_treino_camadas_8": 1, "vocabulario_treino_camadas_6": 5,
             "magnitude_90": 6, "gradual_90": 5, "magnitude_70": 1, "gradual_70": 3, "magnitude_50": 0, "gradual_50": 0,
             "magnitude_30": 0, "gradual_30": 0}
    tamanhos = {"vocabulario_treino": 300, "vocabulario_alfabeto": 700, "camadas_8": 900, "vocabulario_treino_camadas_8": 250,
                "vocabulario_treino_camadas_6": 200}
    pesos = lambda m, v, s: (etapa10b.pasta_da_execucao(m, v, s) / "modelo").exists()

    # a grade do principal pela metade: o outro aluno ainda espera, e só sai do disco o que já não pode seguir
    for variante in ("sem_poda", "camadas_6", "magnitude_90", "gradual_90", "magnitude_70"):
        fingir(cenario, "aluno_a", variante, erros[variante], tamanhos.get(variante, 1000))
    assert etapa10b.seguem_do_principal() is None and etapa10b.variantes_do_modelo("aluno_b2_t2") is None
    apagadas = etapa10b.limpar_pesos()
    assert [pesos("aluno_a", "sem_poda", s) for s in (0, 1, 2)] == [False, False, False]        # o controle está fora da regra
    assert [pesos("aluno_a", "camadas_6", s) for s in (0, 1, 2)] == [True, False, False]        # não passa: fica só a semente do custo
    assert [pesos("aluno_a", "magnitude_90", s) for s in (0, 1, 2)] == [True, False, False]
    assert [pesos("aluno_a", "magnitude_70", s) for s in (0, 1, 2)] == [True, True, True]       # passa: pode ser a secundária
    assert len(apagadas) == 3 + 2 + 2 + 2
    assert all(etapa10b.pronta("aluno_a", v, s) for v in ("sem_poda", "camadas_6") for s in (0, 1, 2))   # continuam prontas

    # a grade inteira: a regra é aplicada, fica gravada, e o outro aluno recebe a principal e a secundária
    for variante in etapa10b.VARIANTES:
        if not etapa10b.completa("aluno_a", variante):
            fingir(cenario, "aluno_a", variante, erros[variante], tamanhos.get(variante, 1000))
    decidido = etapa10b.seguem_do_principal()
    assert (decidido["principal"], decidido["secundaria"]) == ("vocabulario_treino_camadas_8", "magnitude_70")
    assert etapa10b.variantes_do_modelo("aluno_b2_t2") == ["vocabulario_treino_camadas_8", "magnitude_70"]
    assert (etapa10b.pasta_da_etapa() / "segue_do_principal.json").exists()
    do_b2 = [e for e in etapa10b.grade() if e[0] == "aluno_b2_t2"]
    assert do_b2 == [("aluno_b2_t2", v, s) for v in ("vocabulario_treino_camadas_8", "magnitude_70") for s in (0, 1, 2)]
    assert len(etapa10b.grade()) == 2 * 15 * 3 + 2 * 3 + 2 * 3 and "principal vocabulario_treino_camadas_8" in etapa10b.situacao()

    etapa10b.limpar_pesos()
    assert [pesos("aluno_a", "magnitude_50", s) for s in (0, 1, 2)] == [True, False, False]     # passa, mas a de 70% também: não pode ser a secundária
    assert [pesos("aluno_a", "gradual_70", s) for s in (0, 1, 2)] == [True, False, False]       # não passa
    assert [pesos("aluno_a", "magnitude_70", s) for s in (0, 1, 2)] == [True, True, True]
    for peca in ("vocabulario_treino", "vocabulario_alfabeto", "camadas_8", "vocabulario_treino_camadas_8"):   # as peças que passam ficam todas
        assert [pesos("aluno_a", peca, s) for s in (0, 1, 2)] == [True, True, True]
    assert etapa10b.limpar_pesos() == []                                                        # nada mais a apagar

    # o outro aluno fica inteiro, passe ou não: segue por desenho
    fingir(cenario, "aluno_b2_t2", "magnitude_70", 8)
    etapa10b.limpar_pesos()
    assert [pesos("aluno_b2_t2", "magnitude_70", s) for s in (0, 1, 2)] == [True, True, True]


def test_depois_da_decisao_so_ficam_no_cache_os_pesos_do_que_segue(cenario, monkeypatch):
    pasta = cenario["pasta"]
    monkeypatch.setattr(etapa10b, "ENTREGAS", pasta / "entregas")
    for variante in ("camadas_6", "magnitude_70", "vocabulario_treino"):
        fingir(cenario, "aluno_a", variante, 0)
    fingir(cenario, "teto", "magnitude_70", 0)
    pesos = lambda m, v: [(etapa10b.pasta_da_execucao(m, v, s) / "modelo").exists() for s in (0, 1, 2)]
    with pytest.raises(FileNotFoundError):                    # sem a decisão gravada, nada é apagado
        etapa10b.limpar_depois_da_decisao()

    entregues = {str(s): f"entregas/{etapa10b.ETAPA}/modelos/aluno_a/magnitude_70/semente_{s}" for s in (0, 1, 2)}
    etapa10b._gravar_json(pasta / "entregas" / etapa10b.ETAPA / "decisao.json", {"modelos": {
        "aluno_a": {"seguem": [{"variante": "magnitude_70", "pastas_por_semente": entregues}]}, "teto": {"seguem": []},
    }})
    with pytest.raises(FileNotFoundError):                    # a decisão aponta para modelos que ainda não estão na entrega
        etapa10b.limpar_depois_da_decisao()
    assert pesos("aluno_a", "camadas_6") == [True, True, True]

    for relativa in entregues.values():
        for arquivo in ("encoder/model.safetensors", "cabecas.pt", "config.json", "encoder/tokenizer.json"):
            (pasta / relativa / arquivo).parent.mkdir(parents=True, exist_ok=True)
            (pasta / relativa / arquivo).write_text("x")
    apagadas = etapa10b.limpar_depois_da_decisao()
    assert len(apagadas) == 9
    assert pesos("aluno_a", "magnitude_70") == [True, True, True]             # o que segue fica
    assert pesos("aluno_a", "camadas_6") == pesos("aluno_a", "vocabulario_treino") == pesos("teto", "magnitude_70") == [False, False, False]
    assert all(etapa10b.pronta(m, v, s) for m, v in (("aluno_a", "camadas_6"), ("teto", "magnitude_70")) for s in (0, 1, 2))   # as execuções continuam prontas
    assert all((pasta / relativa / "cabecas.pt").exists() for relativa in entregues.values())                                 # e a entrega não é tocada


def test_os_tamanhos_sao_medidos_nos_modelos_gravados(cenario):
    etapa10b.executar("teto", "magnitude_90", 0, "cpu")
    etapa10b.executar("teto", "vocabulario_treino_camadas_6", 0, "cpu")
    medidos = etapa10b.medir_tamanhos()
    partida, podado, menor = medidos["teto", "de_partida"], medidos["teto", "magnitude_90"], medidos["teto", "vocabulario_treino_camadas_6"]
    assert set(medidos) == {(m, "de_partida") for m in MODELOS} | {("teto", "magnitude_90"), ("teto", "vocabulario_treino_camadas_6")}
    etapa10b.executar("teto", "sem_poda", 0, "cpu")
    assert set(etapa10b.medir_tamanhos()) == set(medidos)                            # o controle não é medido: tem a rede do modelo de partida
    assert etapa10b.medir_tamanhos(refazer=True, fatia=1, de=2).keys() < medidos.keys()   # uma fatia mede só a sua parte
    assert podado["parametros"] == partida["parametros"] > menor["parametros"]
    assert podado["nao_nulos"] < 0.5 * partida["nao_nulos"] and podado["esparsidade"] == pytest.approx(0.9, abs=1e-3)
    # pesos soltos: o arquivo comum não muda; o zip e o arquivo esparso encolhem
    assert podado["bytes"]["rede"]["arquivo_comum"] == partida["bytes"]["rede"]["arquivo_comum"]
    assert podado["bytes"]["rede"]["arquivo_esparso"] < 0.5 * partida["bytes"]["rede"]["arquivo_esparso"]
    assert podado["bytes"]["rede"]["arquivo_comum_com_zip"] < 0.5 * partida["bytes"]["rede"]["arquivo_comum_com_zip"]
    # peças inteiras: o arquivo comum encolhe, na rede e no tokenizador
    assert menor["bytes"]["rede"]["arquivo_comum"] < partida["bytes"]["rede"]["arquivo_comum"]
    assert menor["bytes"]["tokenizador_e_configuracao"]["arquivo_comum"] < partida["bytes"]["tokenizador_e_configuracao"]["arquivo_comum"]
    assert not list((etapa10b.pasta_da_etapa() / "tamanhos").glob("_esparso_*"))     # a pasta temporária não fica
    assert json.loads((etapa10b.pasta_da_etapa() / "tamanhos" / "teto__magnitude_90.json").read_text())["bytes"] == podado["bytes"]
