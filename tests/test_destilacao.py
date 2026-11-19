"""Etapa 6: a perda com temperatura, o caminho dos logits até o lote, a parada pelo dev do professor e as duas regras."""

import json
from math import log

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from especialista.avaliacao import qualidade
from especialista.avaliacao.previsao import Previsao, carregar_previsoes, ouro, salvar_previsoes
from especialista.dados import massive
from especialista.experimentos import etapa6, execucao
from especialista.modelos.diet import Config, SistemaDiet
from especialista.modelos.diet.modelo import perda_de_destilacao
from especialista.professores.rotulagem import rotular
from especialista.treino import Receita, treinar

from test_diet import ENUNCIADOS, PEQUENO, ROTULOS

INF = float("inf")


def dados() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": i, "intent": intent, "utt": texto, "tokens": texto.split(), "tags": tags.split()}
        for i, (intent, texto, tags) in enumerate(ENUNCIADOS)
    )


def log_de_um_rotulo_so(d: pd.DataFrame) -> list[np.ndarray]:
    """O alvo "100% numa intenção", em log-probabilidades: 0 na intenção do rótulo e −inf nas outras."""
    linhas = np.full((len(d), len(ROTULOS["intents"])), -INF, dtype=np.float32)
    linhas[np.arange(len(d)), [ROTULOS["intents"].index(i) for i in d["intent"]]] = 0.0
    return list(linhas)


# --------------------------------------------------------------------------
# A perda
# --------------------------------------------------------------------------


def test_perda_de_destilacao_confere_com_a_conta_a_mao():
    T = 2.0
    # 1º pedido: professor [2·ln 3, 0] / T -> p = (3/4, 1/4); aluno [0, 0] -> q = (1/2, 1/2)
    # 2º pedido: professor [0, 0] -> p = (1/2, 1/2); aluno [2·ln 2, 0] / T -> q = (2/3, 1/3)
    professor = torch.tensor([[2 * log(3), 0.0], [0.0, 0.0]])
    aluno = torch.tensor([[0.0, 0.0], [2 * log(2), 0.0]])
    kl_1 = 0.75 * log(0.75 / 0.5) + 0.25 * log(0.25 / 0.5)
    kl_2 = 0.5 * log(0.5 / (2 / 3)) + 0.5 * log(0.5 / (1 / 3))
    esperado = T**2 * (kl_1 + kl_2) / 2          # T² vezes a média do lote

    assert perda_de_destilacao(aluno, professor, T).item() == pytest.approx(esperado, abs=1e-6)
    assert esperado == pytest.approx(0.379407, abs=1e-6)   # o número, para quem refizer a conta

    # somar uma constante aos logits do professor não muda nada: só a forma da distribuição conta
    assert perda_de_destilacao(aluno, professor - 17.0, T).item() == pytest.approx(esperado, abs=1e-6)
    # aluno igual ao professor: perda zero
    assert perda_de_destilacao(professor, professor, T).item() == pytest.approx(0.0, abs=1e-7)


def test_com_t_1_e_alvo_de_um_rotulo_so_e_a_perda_de_rotulo_rigido():
    torch.manual_seed(0)
    aluno = torch.randn(5, 4)
    rotulo = torch.tensor([0, 3, 1, 1, 2])
    professor = torch.full((5, 4), -INF)
    professor[torch.arange(5), rotulo] = 0.0
    assert torch.allclose(perda_de_destilacao(aluno, professor, 1.0), F.cross_entropy(aluno, rotulo), atol=1e-6)


def test_o_fator_t2_mantem_o_gradiente_do_mesmo_tamanho():
    """Hinton et al., seção 2.1: em temperatura alta, o gradiente vira (aluno − professor) / N, sem depender de T."""
    sorteio = torch.Generator().manual_seed(0)
    professor = torch.randn(1, 6, generator=sorteio, dtype=torch.float64)
    inicial = torch.randn(1, 6, generator=sorteio, dtype=torch.float64)
    diferenca = inicial - professor
    esperado = (diferenca - diferenca.mean()) / 6
    for T in (500.0, 5000.0):
        aluno = inicial.clone().requires_grad_()
        perda_de_destilacao(aluno, professor, T).backward()
        assert torch.allclose(aluno.grad, esperado, atol=1e-3)
    # sem o T², o mesmo gradiente seria T² vezes menor
    aluno = inicial.clone().requires_grad_()
    (perda_de_destilacao(aluno, professor, 500.0) / 500.0**2).backward()
    assert aluno.grad.abs().max() < esperado.abs().max() / 100_000


def perda_de_intencao_da_rede(intencoes: torch.Tensor, d: pd.DataFrame, **troca) -> float:
    """A perda de intenção que a rede calcula para pontuações dadas, com a configuração trocada em `troca`."""
    sistema = SistemaDiet.novo(d, ROTULOS, Config(**{**PEQUENO.como_dict(), **troca}), contagem_minima=1)
    return sistema.modelo._perda_intencao(intencoes, sistema.lote(d)).item()


def test_a_rede_com_temperatura_usa_os_logits_e_sem_ela_ignora():
    d = dados()
    d["logits_professor"] = log_de_um_rotulo_so(d)
    rotulo = torch.tensor([ROTULOS["intents"].index(i) for i in d["intent"]])
    intencoes = 3 * torch.randn(len(d), 3, generator=torch.Generator().manual_seed(0))
    rigida = F.cross_entropy(intencoes, rotulo).item()
    suavizada = F.cross_entropy(intencoes, rotulo, label_smoothing=0.1).item()
    assert abs(suavizada - rigida) > 0.01

    # sem temperatura os logits do lote não entram: vale o rótulo, com ou sem suavização
    assert perda_de_intencao_da_rede(intencoes, d) == pytest.approx(rigida, abs=1e-6)
    assert perda_de_intencao_da_rede(intencoes, d, suavizacao=0.1) == pytest.approx(suavizada, abs=1e-6)
    # T = 1 e alvo de um rótulo só: a mesma perda de quem treina com o rótulo
    assert perda_de_intencao_da_rede(intencoes, d, temperatura=1.0) == pytest.approx(rigida, abs=1e-5)
    # a suavização de rótulos não se aplica ao alvo que já é uma distribuição
    assert perda_de_intencao_da_rede(intencoes, d, temperatura=1.0, suavizacao=0.1) == pytest.approx(rigida, abs=1e-5)
    # com outra temperatura a perda é outra
    assert abs(perda_de_intencao_da_rede(intencoes, d, temperatura=4.0) - rigida) > 0.01


def test_peso_rotulo_soma_as_duas_partes_com_os_pesos_certos():
    d = dados()
    # um professor que discorda do rótulo: metade da probabilidade na intenção do rótulo, metade na seguinte
    indice = np.array([ROTULOS["intents"].index(i) for i in d["intent"]])
    prob = np.full((len(d), 3), 1e-9)
    prob[np.arange(len(d)), indice] = 0.5
    prob[np.arange(len(d)), (indice + 1) % 3] = 0.5
    d["logits_professor"] = list(np.log(prob).astype(np.float32))
    intencoes = 3 * torch.randn(len(d), 3, generator=torch.Generator().manual_seed(0))

    suave = perda_de_destilacao(intencoes, torch.tensor(np.stack(d["logits_professor"])), 2.0).item()
    rigida = F.cross_entropy(intencoes, torch.tensor(indice)).item()
    suavizada = F.cross_entropy(intencoes, torch.tensor(indice), label_smoothing=0.1).item()
    assert abs(suave - rigida) > 0.01
    assert perda_de_intencao_da_rede(intencoes, d, temperatura=2.0) == pytest.approx(suave, abs=1e-5)
    assert perda_de_intencao_da_rede(intencoes, d, temperatura=2.0, peso_rotulo=0.25) == pytest.approx(0.75 * suave + 0.25 * rigida, abs=1e-5)
    assert perda_de_intencao_da_rede(intencoes, d, temperatura=2.0, peso_rotulo=1.0) == pytest.approx(rigida, abs=1e-5)
    # a parte do rótulo, quando existe, segue a suavização configurada
    assert perda_de_intencao_da_rede(intencoes, d, temperatura=2.0, peso_rotulo=0.25, suavizacao=0.1) == pytest.approx(
        0.75 * suave + 0.25 * suavizada, abs=1e-5)


def test_temperatura_sem_logits_no_lote_e_erro():
    d = dados()
    sistema = SistemaDiet.novo(d, ROTULOS, Config(**{**PEQUENO.como_dict(), "temperatura": 2.0}), contagem_minima=1)
    with pytest.raises(ValueError, match="logits_professor"):
        sistema.modelo.perda(sistema.lote(d))


def test_lote_leva_os_logits_na_ordem_dos_pedidos():
    d = dados()
    sistema = SistemaDiet.novo(d, ROTULOS, PEQUENO, contagem_minima=1)
    assert sistema.lote(d).logits_professor is None

    d["logits_professor"] = [np.array([i, 10 * i, 100 * i], dtype=np.float32) for i in range(len(d))]
    recorte = d.iloc[[5, 2, 7]]
    assert sistema.lote(recorte).logits_professor.tolist() == [[5, 50, 500], [2, 20, 200], [7, 70, 700]]

    d["logits_professor"] = [np.zeros(4, dtype=np.float32)] * len(d)   # uma coluna a mais que as intenções
    with pytest.raises(ValueError, match="forma"):
        sistema.lote(d)


def test_treino_sem_temperatura_nao_muda_com_os_logits_presentes():
    """A coluna de logits, sozinha, não altera o treino: quem liga a destilação é a temperatura."""
    receita = Receita(epocas=6, taxa=3e-3, lote_inicial=4, lote_final=8, minimo_por_intencao=0, avaliar_a_cada=3)
    pesos = []
    for com_logits in (False, True):
        d = dados()
        if com_logits:
            d["logits_professor"] = log_de_um_rotulo_so(d)
        sistema = SistemaDiet.novo(d, ROTULOS, PEQUENO, contagem_minima=1)
        treinar(sistema, d, d, receita, semente=0)
        pesos.append(sistema.modelo.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(pesos[0].values(), pesos[1].values()))


def test_aluno_aprende_a_intencao_so_com_os_logits():
    """Os rótulos de intenção do treino estão todos errados; o aluno acerta porque segue os logits."""
    d = dados()
    indice = np.array([ROTULOS["intents"].index(i) for i in d["intent"]])
    logits = np.full((len(d), 3), -4.0, dtype=np.float32)
    logits[np.arange(len(d)), indice] = 0.0
    errado = d.assign(intent=[ROTULOS["intents"][(i + 1) % 3] for i in indice], logits_professor=list(logits))

    sistema = SistemaDiet.novo(d, ROTULOS, Config(**{**PEQUENO.como_dict(), "temperatura": 2.0}), contagem_minima=1)
    receita = Receita(epocas=60, taxa=3e-3, lote_inicial=4, lote_final=8, minimo_por_intencao=0, avaliar_a_cada=10)
    treinar(sistema, errado, d, receita, semente=0)
    assert qualidade.avaliar(ouro(d), sistema.prever(list(d["utt"])))["intent_acc"] == 1.0


# --------------------------------------------------------------------------
# A execução: de onde vêm os rótulos e quem escolhe a época de parada
# --------------------------------------------------------------------------

TROCA = {"alarm_set": "play_music", "play_music": "alarm_set", "weather_query": "weather_query"}


def professor_que_troca(d: pd.DataFrame, sem_resposta: int | None = None) -> list[Previsao]:
    """Um professor que confunde sempre duas intenções e acerta as entidades. Opcionalmente, deixa um pedido sem resposta."""
    previsoes = [Previsao.de_tags(TROCA[intent], list(tags)) for intent, tags in zip(d["intent"], d["tags"])]
    if sem_resposta is not None:
        previsoes[sem_resposta] = Previsao("(sem resposta)")
    return previsoes


@pytest.fixture
def cenario(tmp_path, monkeypatch):
    """As entregas de que a Etapa 6 depende, em miniatura, numa pasta temporária."""
    treino, dev = dados(), dados().assign(id=lambda t: t["id"] + 100)
    intents = ROTULOS["intents"]
    monkeypatch.setattr(massive, "carregar_entrega", lambda: ({"train": treino, "dev": dev, "test": dev}, ROTULOS))
    monkeypatch.setattr(etapa6, "ENTREGAS", tmp_path / "entregas")
    monkeypatch.setattr(etapa6, "RAIZ", tmp_path)
    monkeypatch.setattr(execucao, "EXECUCOES", tmp_path / "execucoes")

    rede = tmp_path / "entregas" / "04b_intencao"
    rede.mkdir(parents=True)
    (rede / "decisao.json").write_text(json.dumps({
        "config": {**PEQUENO.como_dict(), "suavizacao": 0.1},
        "receita": {"epocas": 40, "taxa": 3e-3, "lote_inicial": 4, "lote_final": 8, "minimo_por_intencao": 0,
                    "avaliar_a_cada": 10, "media_dos_pesos": 0.9},
    }), encoding="utf-8")

    for professor, sem_resposta in (("fechado", None), ("aberto", 3)):
        pasta = etapa6.pasta_do_professor(professor)
        pasta.mkdir(parents=True)
        rotular(treino, professor_que_troca(treino, sem_resposta), intents).to_parquet(pasta / "rotulos_treino.parquet", index=False)
        salvar_previsoes(pasta / "previsoes" / f"{etapa6.PROFESSORES[professor]['previsoes']}_dev.parquet", dev["id"],
                         professor_que_troca(dev, sem_resposta))
    treino.iloc[[0, 3, 6]][["id", "intent", "utt", "tokens", "tags"]].to_parquet(tmp_path / "entregas" / etapa6.DEMONSTRACOES, index=False)

    # os logits do professor aberto: quase toda a probabilidade na intenção que ele escolheu
    escolhida = np.array([intents.index(TROCA[i]) for i in treino["intent"]])
    logits = np.full((len(treino), len(intents)), -5.0, dtype=np.float32)
    logits[np.arange(len(treino)), escolhida] = -0.01
    aberto = etapa6.pasta_do_professor("aberto")
    np.save(aberto / "logits_treino.npy", logits)
    (aberto / "intencoes.json").write_text(json.dumps(intents), encoding="utf-8")
    return {"treino": treino, "dev": dev, "intents": intents, "pasta": tmp_path}


def test_definicao_parte_da_etapa_4b_e_o_b2_tira_a_suavizacao(cenario):
    assert etapa6.definicao("aluno_a")["config"]["suavizacao"] == 0.1
    assert etapa6.definicao("aluno_a")["config"]["temperatura"] == 0.0
    b2 = etapa6.definicao("aluno_b2_t4")
    assert (b2["config"]["temperatura"], b2["config"]["suavizacao"], b2["config"]["peso_rotulo"]) == (4.0, 0.0, 0.0)
    assert b2["receita"] == etapa6.definicao("aluno_a")["receita"]
    assert [etapa6.variante_do_b2(T) for T in etapa6.TEMPERATURAS] == [v for v in etapa6.VARIANTES if v.startswith("aluno_b2")]

    # o controle do B2: o B1 sem a suavização, e mais nada
    b1, controle = etapa6.definicao("aluno_b1"), etapa6.definicao("aluno_b1_sem_suavizacao")
    diferencas = {k for k in b1["config"] if b1["config"][k] != controle["config"][k]}
    assert diferencas == {"suavizacao"} and controle["config"]["suavizacao"] == 0.0 and controle["receita"] == b1["receita"]
    assert {k for k in b2["config"] if b2["config"][k] != controle["config"][k]} == {"temperatura"}   # do controle ao B2, só os logits
    assert not etapa6.VARIANTES["aluno_b1_sem_suavizacao"]["logits"]


def test_aluno_treina_e_para_pelos_rotulos_do_professor(cenario):
    dev = cenario["dev"]
    pasta = etapa6.executar("aluno_a", 0, "cpu")
    execucao_a = etapa6.carregar("aluno_a", 0)
    meta, historico, previsoes = execucao_a["meta"], execucao_a["historico"], execucao_a["previsoes_dev"]

    # a época escolhida é a melhor contra o dev do professor, e é com ele que `meta["dev"]` foi medido
    do_professor = professor_que_troca(dev)
    avaliadas = historico.dropna(subset=["dev_exact_match"])
    assert meta["melhor_epoca"] == int(avaliadas.loc[avaliadas["dev_exact_match"].idxmax(), "epoca"])
    assert meta["dev"]["exact_match"] == pytest.approx(avaliadas["dev_exact_match"].max())
    assert meta["dev"] == pytest.approx(qualidade.avaliar(do_professor, previsoes))
    assert "professor fechado" in meta["origem"]["dev"]

    # o aluno aprendeu a troca do professor: concorda com ele, não com os rótulos humanos
    assert qualidade.avaliar(do_professor, previsoes)["intent_acc"] > 0.8
    assert qualidade.avaliar(ouro(dev), previsoes)["intent_acc"] < 0.5

    # as respostas gravadas cobrem dev inteiro, na ordem da Etapa 1
    assert carregar_previsoes(pasta / "previsoes_dev.parquet")[0] == dev["id"].tolist()
    # o que está gravado não é treinado de novo
    marca = (pasta / "meta.json").stat().st_mtime_ns
    assert etapa6.executar("aluno_a", 0, "cpu") == pasta and (pasta / "meta.json").stat().st_mtime_ns == marca


def test_pedido_sem_rotulo_aproveitavel_fica_fora_do_treino_e_da_parada(cenario):
    lido = etapa6.dados_da_variante("aluno_b1")
    assert len(lido["treino"]) == len(cenario["treino"]) - 1 and len(lido["dev"]) == len(cenario["dev"]) - 1
    assert lido["origem"]["sem_rotulo_aproveitavel"] == {"treino": 1, "dev": 1}
    assert "(sem resposta)" not in set(lido["treino"]["intent"]) | set(lido["dev"]["intent"])
    assert len(lido["avaliar"]) == len(cenario["dev"])            # mas a resposta do aluno fica gravada para todos
    assert len(lido["vocabulario"]) == len(cenario["treino"])     # e o vocabulário vem de todos os pedidos
    assert "logits_professor" not in lido["treino"].columns       # o B1 não vê os logits


def test_b2_recebe_os_logits_alinhados_aos_pedidos(cenario):
    lido = etapa6.dados_da_variante("aluno_b2_t2")
    treino = lido["treino"]
    assert len(treino) == len(cenario["treino"]) - 1
    # depois de tirar o pedido sem rótulo, cada linha continua com os logits do seu pedido
    escolhida = [cenario["intents"][int(np.argmax(linha))] for linha in treino["logits_professor"]]
    assert escolhida == treino["intent"].tolist()
    assert lido["treino"]["tags"].tolist() == etapa6.dados_da_variante("aluno_b1")["treino"]["tags"].tolist()   # mesmas entidades do B1

    pasta = etapa6.executar("aluno_b2_t2", 0, "cpu")
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    assert (meta["config"]["temperatura"], meta["config"]["suavizacao"]) == (2.0, 0.0)
    assert meta["exemplos_de_treino"] == len(cenario["treino"]) - 1
    assert meta["origem"]["logits"].endswith("logits_treino.npy")
    assert meta["dev"]["intent_acc"] > 0.8   # aprendeu a intenção do professor pelos logits


def test_piso_treina_so_com_as_demonstracoes(cenario):
    lido = etapa6.dados_da_variante("piso")
    assert len(lido["treino"]) == 3 and "vocabulario" not in lido and "avaliar" not in lido
    assert lido["dev"]["intent"].tolist() == cenario["dev"]["intent"].tolist()   # o dev humano
    pasta = etapa6.executar("piso", 0, "cpu")
    piso = SistemaDiet.carregar(pasta / "modelo")
    aluno = SistemaDiet.carregar(etapa6.executar("aluno_a", 0, "cpu") / "modelo")
    assert len(piso.featurizador) < len(aluno.featurizador)       # vocabulário só dos seus exemplos
    assert etapa6.faltam(["piso", "aluno_a", "aluno_b1"]) == [("piso", 1), ("piso", 2), ("aluno_a", 1), ("aluno_a", 2),
                                                             ("aluno_b1", 0), ("aluno_b1", 1), ("aluno_b1", 2)]


def test_rotulos_ou_logits_desalinhados_sao_recusados(cenario):
    treino, intents = cenario["treino"], cenario["intents"]
    aberto = etapa6.pasta_do_professor("aberto")

    np.save(aberto / "logits_treino.npy", np.zeros((len(treino) - 1, len(intents)), dtype=np.float32))
    with pytest.raises(ValueError, match="forma"):
        etapa6.dados_da_variante("aluno_b2_t1")
    np.save(aberto / "logits_treino.npy", np.full((len(treino), len(intents)), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="não são números"):
        etapa6.dados_da_variante("aluno_b2_t1")
    np.save(aberto / "logits_treino.npy", np.zeros((len(treino), len(intents)), dtype=np.float32))
    (aberto / "intencoes.json").write_text(json.dumps(intents[::-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="ordem das colunas"):
        etapa6.dados_da_variante("aluno_b2_t1")

    fechado = etapa6.pasta_do_professor("fechado")
    rotular(treino.iloc[::-1], professor_que_troca(treino.iloc[::-1]), intents).to_parquet(fechado / "rotulos_treino.parquet", index=False)
    with pytest.raises(ValueError, match="mesmos pedidos"):
        etapa6.dados_da_variante("aluno_a")


# --------------------------------------------------------------------------
# O aluno ao lado do professor
# --------------------------------------------------------------------------


def test_aluno_e_professor_separa_os_cinco_casos():
    data = (("date", 0, 1),)
    gabarito = [Previsao("a", data), Previsao("a"), Previsao("a"), Previsao("a"), Previsao("a"), Previsao("b", data)]
    professor = [Previsao("a", data), Previsao("a"), Previsao("b"), Previsao("b"), Previsao("b"), Previsao("b")]
    aluno = [Previsao("a"), Previsao("b"), Previsao("a"), Previsao("b"), Previsao("c"), Previsao("b")]

    intencao = etapa6.aluno_e_professor(gabarito, professor, aluno)
    assert list(intencao.index) == list(etapa6.CASOS)
    assert (intencao * 6).round().astype(int).tolist() == [2, 1, 1, 1, 1]
    assert intencao.sum() == pytest.approx(1.0)

    # na resposta inteira, as entidades também contam: o 1º pedido passa a ser erro do aluno, o 6º erro dos dois
    tudo = etapa6.aluno_e_professor(gabarito, professor, aluno, parte="tudo")
    assert (tudo * 6).round().astype(int).tolist() == [0, 2, 1, 2, 1]

    with pytest.raises(ValueError):
        etapa6.aluno_e_professor(gabarito, professor[:-1], aluno)


# --------------------------------------------------------------------------
# As duas regras
# --------------------------------------------------------------------------


def respostas(gabarito: list[Previsao], intencao_errada: int, entidade_errada: int = 0) -> list[Previsao]:
    """Uma cópia do gabarito com a intenção errada nos primeiros pedidos e uma entidade a mais nos últimos."""
    copia = list(gabarito)
    for i in range(intencao_errada):
        copia[i] = Previsao("outra", copia[i].slots)
    for i in range(len(copia) - entidade_errada, len(copia)):
        copia[i] = Previsao(copia[i].intent, (("date", 0, 1),))
    return copia


def test_escolher_temperatura():
    gabarito = [Previsao("alarm_set")] * 100
    em_dev = {
        1: [respostas(gabarito, 30), respostas(gabarito, 30)],
        2: [respostas(gabarito, 10), respostas(gabarito, 30)],       # média de 20 erros de intenção: a melhor
        4: [respostas(gabarito, 25), respostas(gabarito, 25)],
    }
    escolhida, medias = etapa6.escolher_temperatura(gabarito, em_dev)
    assert escolhida == 2
    assert medias.loc[2, "intent_acc"] == pytest.approx(0.80) and list(medias.index) == [1, 2, 4]

    # empate na intenção: decide o exact match; empate nos dois: a menor temperatura
    empate = {4: [respostas(gabarito, 20, 10)], 8: [respostas(gabarito, 20, 5)], 2: [respostas(gabarito, 20, 10)]}
    assert etapa6.escolher_temperatura(gabarito, empate)[0] == 8
    assert etapa6.escolher_temperatura(gabarito, {4: empate[4], 2: empate[2]})[0] == 2


def test_quem_segue():
    gabarito = [Previsao("alarm_set")] * 300
    a = [respostas(gabarito, 60)] * 3
    b1 = [respostas(gabarito, 90)] * 3

    # os logits ajudam a intenção de forma distinguível: segue o B2, e o principal é o de maior exact match
    com_ganho = etapa6.quem_segue(gabarito, {"A": a, "B1": b1, "B2": [respostas(gabarito, 30)] * 3})
    assert com_ganho["seguem"] == ["A", "B2"] and com_ganho["b2_passa"] and com_ganho["principal"] == "B2"

    # diferença pequena demais para distinguir: segue o B1, que não precisa dos logits
    sem_ganho = etapa6.quem_segue(gabarito, {"A": a, "B1": b1, "B2": [respostas(gabarito, 88)] * 3})
    assert sem_ganho["seguem"] == ["A", "B1"] and not sem_ganho["b2_passa"] and sem_ganho["principal"] == "A"

    # a intenção melhora, mas o exact match piora de forma distinguível: os logits não passam
    com_custo = etapa6.quem_segue(gabarito, {"A": a, "B1": b1, "B2": [respostas(gabarito, 30, 200)] * 3})
    assert com_custo["seguem"] == ["A", "B1"]
