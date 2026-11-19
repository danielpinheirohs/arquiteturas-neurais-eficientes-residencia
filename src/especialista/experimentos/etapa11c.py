"""Etapa 11c: o cenário B com o professor novo até o fim do caminho, e o comparativo atualizado.

A pergunta: as receitas de poda e de quantização já escolhidas valem para os
alunos do professor novo, e onde o cenário B fica na tabela final com ele?

A Etapa 11b treinou de novo os alunos do cenário B com o professor aberto
maior (o Qwen3-14B da Etapa 11a). Aqui eles passam pelo resto do caminho,
**com as receitas que as etapas de origem já escolheram e sem nenhuma decisão
nova**:

    caminho   poda e reajuste                           exportação e 8 bits
    zero      a principal da Etapa 7 (`etapa7.executar`)   a forma da Etapa 8 (`etapa8.quantizar_modelo`)
    xlmr      a principal da Etapa 10b (`etapa10b.executar`)   a forma da Etapa 10c (`etapa10c.quantizar_modelo`)

**O código que poda e quantiza é o das etapas de origem, sem mudança.** As
funções delas são chamadas dentro de um bloco (`emprestar`) em que o modelo
de partida, o professor e a pasta de gravação são os desta etapa; na saída
do bloco tudo volta ao que era. Nada das Etapas 5b a 11b é sobrescrito: tudo
o que esta etapa grava fica em `data/execucoes/11c_cenario_b_ate_o_fim/`, em
pastas cujo nome traz o professor:

    poda/<variante da 11b>/<poda>/semente_<n>/             o que `etapa7.executar` ou `etapa10b.executar` grava
    quantizacao/<variante da 11b>/<poda ou sem_poda>/semente_<n>/   o que a quantização grava, e a latência (semente 0)
    comparativo/regras_v<N>.json                           as regras do comparativo, uma versão por arquivo
    regras.json, cadeias/, fumaca/                         as regras, os registros das cadeias e o teste de fumaça

**Quem é comprimido** (`alunos`): em cada caminho, o aluno que a regra de
"quem segue" aponta como representante do cenário B com o professor novo (a
Etapa 11b relata) e, onde o representante não é o B2, também o B2, para a
cadeia ficar comparável com a antiga.

**As regras das etapas de origem são reaplicadas só como conferência**: a
poda não fica distinguivelmente abaixo do modelo sem poda em `dev`; a queda
de F1 de entidades com os 8 bits é de no máximo 1 ponto em `dev`. Se uma
receita não passar com o aluno novo, isso é relatado; não se procura outra.

**O comparativo** é o da Etapa 10d, com o registro lendo também as entregas
da Etapa 11 (`ENTREGAS_QUE_VALEM`) e com as famílias de comparação que a
troca de professor pede (`lista_de_comparacoes`). As regras dele são uma
versão nova, gravada aqui; as da 10d não são tocadas.

A ordem dos comandos é a ordem em que as coisas podem ser feitas:

    uv run python -m especialista.experimentos.etapa11c regras                      # antes de tudo: sem regras nada roda
    uv run python -m especialista.experimentos.etapa11c cadeia --fumaca <pasta>     # teste curto, numa pasta descartável
    uv run python -m especialista.experimentos.etapa11c cadeia                      # poda e reajuste, exportação e 8 bits, latência por último
    uv run python -m especialista.experimentos.etapa11c situacao
    uv run python -m especialista.experimentos.etapa11c fechar                      # depois da cadeia: as contas da parte 1 (conferências, tamanhos, latências)
    uv run python -m especialista.experimentos.etapa11c entregar                    # a parte 1 vai para a entrega: é dela que o registro lê os sistemas novos
    uv run python -m especialista.experimentos.etapa11c regras-do-comparativo       # grava a versão seguinte das regras, se o registro cresceu (um professor novo)
    uv run python -m especialista.experimentos.etapa11c calcular                    # o comparativo, pelas regras gravadas
    uv run python -m especialista.experimentos.etapa11c manifesto                   # depois do notebook e do HTML
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from especialista import caminhos
from especialista.caminhos import ENTREGAS, RAIZ
from especialista.experimentos import etapa9, etapa10d, etapa11a, etapa11b, execucao

ETAPA = "11c_cenario_b_ate_o_fim"
PERGUNTA = ("As receitas de poda e de quantização já escolhidas valem para os alunos do professor novo, "
            "e onde o cenário B fica na tabela final com ele?")
SEMENTES = (0, 1, 2)
SEMENTE_DO_CUSTO = 0                 # tamanho e latência são medidos nos modelos desta semente, como nas Etapas 8, 10b e 10c
PROFESSOR = etapa11b.PROFESSOR_NOVO  # o Qwen3-14B: o professor dos alunos da Etapa 11b
SEM_PODA = "sem_poda"
TAMBEM = ("B2",)                     # além do representante, o aluno que as cadeias antigas do cenário B usam nos dois caminhos
LIMITE_DE_OUTROS = etapa10d.LIMITE_DE_OUTROS   # 0,25 núcleo de CPU dos outros processos: o limite de "máquina quieta" das Etapas 8 e 10c
TENTATIVAS_DE_LATENCIA = 3           # uma medida com a máquina ocupada é guardada como recusada e refeita, até este número de vezes
ESPERA_ENTRE_TENTATIVAS = 120        # segundos
THREADS = 4                          # para responder dev e teste ao quantizar (a latência é sempre com 1), como na Etapa 10c

# As decisões de origem: de onde sai cada receita. Nada aqui é escolhido de novo.
ORIGEM = {
    "zero": {"nome": "Rede do zero", "poda": "07_poda", "quantizacao": "08_quantizacao", "processos": {"poda": 3, "quantizar": 3}},
    "xlmr": {"nome": "XLM-R", "poda": "10b_encoder_poda", "quantizacao": "10c_encoder_quantizacao", "processos": {"poda": 1, "quantizar": 3}},
}
PASSOS = ("poda", "quantizar", "latencia")   # a ordem da cadeia
DECISAO_DOS_ALUNOS = f"{etapa11b.ETAPA}/decisao.json"

# As entregas que valem para o comparativo desta etapa: as da 10d, mais as da Etapa 11. A lista é fechada pelo mesmo motivo
# da 10d (`etapa10d.ENTREGAS_QUE_VALEM`): uma etapa que venha depois não muda o que esta entregou.
ENTREGAS_QUE_VALEM = (*etapa10d.ENTREGAS_QUE_VALEM, etapa11a.ETAPA, etapa11b.ETAPA, ETAPA)
VERSAO_DAS_REGRAS_DO_COMPARATIVO = 3   # as duas primeiras são as da Etapa 10d (regras.json e regras_v2.json)


def pasta_da_etapa() -> Path:
    return execucao.EXECUCOES / ETAPA


def no_cache_de_verdade() -> bool:
    return execucao.EXECUCOES.resolve() == caminhos.EXECUCOES.resolve()


def _json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def _gravar_json(caminho: Path, conteudo: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _agora() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _relativo(caminho: Path) -> str:
    return str(caminho.relative_to(RAIZ)) if caminho.is_relative_to(RAIZ) else str(caminho)


@contextmanager
def emprestar(modulo, **trocas):
    """Dentro do bloco, os nomes dados valem `trocas` no módulo de uma etapa de origem; na saída volta o que estava.

    É o que deixa esta etapa chamar `etapa7.executar` e `etapa10b.executar`
    sem mudar uma linha delas: só o modelo de partida, o professor e a pasta
    de gravação são outros. Como em `etapa11b.com_professor`, a troca vale só
    dentro do bloco, mesmo que ele termine com erro.
    """
    antes = {nome: getattr(modulo, nome) for nome in trocas}
    for nome, valor in trocas.items():
        setattr(modulo, nome, valor)
    try:
        yield modulo
    finally:
        for nome, valor in antes.items():
            setattr(modulo, nome, valor)


# --------------------------------------------------------------------------
# Quem é comprimido, e com que receita
# --------------------------------------------------------------------------

REGRA_DOS_ALUNOS = (
    "em cada caminho, o aluno que a regra de 'quem segue' das etapas de origem (`etapa6.quem_segue`) aponta como representante do cenário B "
    f"com o professor novo, como a Etapa 11b relata em `quem_representa_o_cenario_b` (entregas/{DECISAO_DOS_ALUNOS}); e, onde o representante "
    "não é o B2, também o B2, porque as cadeias antigas do cenário B são de B2 nos dois caminhos"
)


def alunos(professor: str = PROFESSOR) -> dict[str, dict]:
    """Os alunos que esta etapa comprime, na ordem em que rodam: {variante da Etapa 11b: ficha}.

    Sai de `entregas/11b_alunos_do_professor_maior/decisao.json`, por
    `REGRA_DOS_ALUNOS`. Cada ficha traz o caminho, a sigla, de onde o aluno
    parte (a entrega da 11b), a configuração, a receita e o sinal com que foi
    treinado, e a chave dele no registro do comparativo.
    """
    decisao = _json(ENTREGAS / DECISAO_DOS_ALUNOS)
    if decisao["professor"]["versao"] != professor:
        raise ValueError(f"A entrega da Etapa 11b é do professor {decisao['professor']['versao']}, não de {professor}.")
    quem = decisao["professor"]["nome"]
    fichas: dict[str, dict] = {}
    for caminho in ORIGEM:
        representa = decisao["quem_representa_o_cenario_b"][caminho]["representa"]
        for sigla in (representa, *(s for s in TAMBEM if s != representa)):
            aluno = decisao["alunos"][caminho][sigla]
            modelo = f"aluno_{sigla.lower()}@{professor}"
            fichas[aluno["variante"]] = {
                "nome": aluno["nome"], "curto": f"Aluno {sigla} · {quem}", "caminho": caminho, "sigla": sigla, "aluno": aluno["aluno"], "professor": professor,
                "representa": sigla == representa,
                "papel": "representa o cenário B neste caminho" if sigla == representa else "entra para a cadeia ficar comparável com a antiga, que é de B2",
                "modelo": modelo, "chave": etapa10d.CAMINHOS[caminho]["prefixo"] + modelo,
                "modelos": aluno["modelos"], "pastas": {int(s): RAIZ / pasta for s, pasta in aluno["pastas_por_semente"].items()},
                "config": aluno["config"], "receita": aluno["receita"], "sinal": aluno["sinal"],
            }
    return fichas


def receitas() -> dict[str, dict]:
    """A poda principal e a forma de 8 bits de cada caminho, lidas das decisões das etapas de origem. Nada é escolhido aqui.

    Confere que o aluno antigo do cenário B recebeu, na etapa de origem, a
    mesma poda e a mesma forma que o modelo principal: é essa a receita "já
    escolhida" que os alunos novos recebem.
    """
    saida = {}
    for caminho, origem in ORIGEM.items():
        poda, quantizacao = _json(ENTREGAS / origem["poda"] / "decisao.json"), _json(ENTREGAS / origem["quantizacao"] / "decisao.json")
        principal, final = poda["principal"], quantizacao["final"]
        antigo = next(m for m in poda["modelos"] if m not in (principal["modelo"], "teto"))   # o aluno antigo do cenário B que seguiu para a poda
        do_antigo = {"modelo": antigo, "poda": poda["modelos"][antigo]["principal"], "forma": quantizacao["modelos"][f"{antigo}/{principal['variante']}"]["forma"]}
        if (do_antigo["poda"], do_antigo["forma"]) != (principal["variante"], final["forma"]):
            raise ValueError(f"No caminho {caminho}, o aluno antigo do cenário B não tem a poda e a forma do modelo principal: {do_antigo}.")
        nome_da_poda = next(f["nome"] for f in poda["modelos"][principal["modelo"]]["seguem"] if f["variante"] == principal["variante"])
        saida[caminho] = {
            "poda": principal["variante"], "nome_da_poda": nome_da_poda, "reajuste": poda["reajuste"], "regra_da_poda": poda["regra"],
            "forma": final["forma"], "nome_da_forma": final["nome_da_forma"], "regra_da_forma": quantizacao["regra_da_forma"],
            "de": {"poda": f"entregas/{origem['poda']}/decisao.json (principal.variante)", "forma": f"entregas/{origem['quantizacao']}/decisao.json (final.forma)"},
            "no_aluno_antigo_do_cenario_b": do_antigo,
        }
    return saida


def variantes(caminho: str) -> tuple[str, str]:
    """O que é exportado e quantizado em cada aluno: o modelo podado e, como nas Etapas 8 e 10c, o mesmo aluno sem poda (a referência)."""
    return (receitas()[caminho]["poda"], SEM_PODA)


def pasta_da_poda(nome: str, semente: int, professor: str = PROFESSOR) -> Path:
    return pasta_da_etapa() / "poda" / nome / receitas()[alunos(professor)[nome]["caminho"]]["poda"] / f"semente_{semente}"


def base_da_quantizacao() -> Path:
    return pasta_da_etapa() / "quantizacao"


def pasta_da_quantizacao(nome: str, variante: str, semente: int) -> Path:
    return base_da_quantizacao() / nome / variante / f"semente_{semente}"


# --------------------------------------------------------------------------
# As regras, gravadas antes de qualquer execução
# --------------------------------------------------------------------------


def exigir_regras() -> dict:
    """As regras gravadas. Sem elas nada é podado, quantizado nem calculado."""
    arquivo = pasta_da_etapa() / "regras.json"
    if not arquivo.exists():
        raise FileNotFoundError("As regras da Etapa 11c ainda não foram gravadas: rode o comando `regras` antes de qualquer execução.")
    return _json(arquivo)


def grade(professor: str = PROFESSOR) -> list[tuple[str, str, str, int]]:
    """Todas as execuções da cadeia, na ordem em que rodam: (passo, aluno, variante, semente)."""
    daqui = alunos(professor)
    poda = [("poda", nome, receitas()[a["caminho"]]["poda"], s) for nome, a in daqui.items() for s in SEMENTES]
    quantizar = [("quantizar", nome, v, s) for nome, a in daqui.items() for v in variantes(a["caminho"]) for s in SEMENTES]
    latencia = [("latencia", nome, v, SEMENTE_DO_CUSTO) for nome, a in daqui.items() for v in variantes(a["caminho"])]
    return poda + quantizar + latencia


def pronta(passo: str, nome: str, variante: str, semente: int) -> bool:
    if passo == "poda":
        return (pasta_da_poda(nome, semente) / "meta.json").exists()
    return (pasta_da_quantizacao(nome, variante, semente) / ("meta.json" if passo == "quantizar" else "latencia.json")).exists()


def faltam(professor: str = PROFESSOR) -> list[tuple[str, str, str, int]]:
    return [execucao_ for execucao_ in grade(professor) if not pronta(*execucao_)]


def situacao(professor: str = PROFESSOR) -> str:
    feitas, todas = [e for e in grade(professor) if pronta(*e)], grade(professor)
    por_passo = ", ".join(f"{passo} {sum(e[0] == passo for e in feitas)} de {sum(e[0] == passo for e in todas)}" for passo in PASSOS)
    return f"{professor}: {len(feitas)} de {len(todas)} execuções prontas ({por_passo})"


CONFERENCIAS = {
    "poda": {
        "regra_de_origem": "a das Etapas 7 e 10b (`etapa7.quem_segue`, passo 1): passa a poda cujo exact match em dev, depois do reajuste, não é distinguivelmente menor que o do "
                           "modelo sem poda (o intervalo de 95% da diferença pareada, na média de 3 sementes, não fica todo abaixo de zero)",
        "aqui": "aplicada à poda principal de cada aluno novo, contra o mesmo aluno sem poda (as respostas entregues pela Etapa 11b), em dev, com os rótulos humanos",
        "metrica": "exact_match", "passa_se": "ic_max >= 0",
    },
    "quantizacao": {
        "regra_de_origem": "a das Etapas 8 e 10c (`etapa8.escolher_forma`, passo 1; PLANO.md): passa a forma cuja queda de F1 de entidades é de no máximo 1 ponto em dev, contra o "
                           "mesmo modelo em 32 bits no PyTorch (diferença pareada, média de 3 sementes)",
        "aqui": "aplicada à forma escolhida na etapa de origem, no modelo podado de cada aluno novo (e, como registro, no modelo sem poda)",
        "metrica": "slot_f1", "passa_se": "diferenca >= -0.01",
    },
    "onnx": "o ONNX de 32 bits tem de dar a mesma resposta do PyTorch em todos os pedidos de dev e do teste, em toda execução (contagem em meta.json)",
    "se_nao_passar": "a receita que não passar com um aluno novo é relatada como está; não se procura outra poda, outra forma, outro orçamento de reajuste nem treino ciente de quantização",
    "o_teste": "todas as execuções respondem dev e teste, como nas etapas de origem; nenhuma conferência lê o teste",
}


def montar_regras(professor: str = PROFESSOR) -> dict:
    """Tudo o que a etapa fixa antes de rodar: os alunos, as receitas, as conferências, a medida de custo e a grade. Não lê nenhum resultado desta etapa."""
    from especialista.experimentos import etapa7, etapa10b

    daqui, R = alunos(professor), receitas()
    if R["zero"]["reajuste"]["epocas"] != etapa7.EPOCAS or R["xlmr"]["reajuste"]["epocas"] != etapa10b.EPOCAS:
        raise ValueError("O orçamento de reajuste gravado nas decisões das Etapas 7 e 10b não é o do código delas.")
    decisoes = [f"entregas/{DECISAO_DOS_ALUNOS}", *(f"entregas/{o[e]}/decisao.json" for o in ORIGEM.values() for e in ("poda", "quantizacao"))]
    execucoes = grade(professor)
    return {
        "etapa": ETAPA, "pergunta": PERGUNTA,
        "o_que_muda": "nada é escolhido: os alunos da Etapa 11b recebem a poda, o reajuste e a forma de 8 bits que as Etapas 7, 8, 10b e 10c escolheram para os alunos antigos",
        "professor": {"versao": professor, "nome": etapa11b.nome_do_professor(professor), "de_onde": f"entregas/{DECISAO_DOS_ALUNOS}"},
        "regra_dos_alunos": REGRA_DOS_ALUNOS,
        "alunos": {nome: {**{k: a[k] for k in ("nome", "curto", "caminho", "sigla", "aluno", "representa", "papel", "chave", "modelos", "config", "receita", "sinal")},
                          "de_partida": {str(s): _relativo(p) for s, p in a["pastas"].items()}} for nome, a in daqui.items()},
        "receitas": {caminho: {**r, "funcoes": {
            "poda": f"etapa{'7' if caminho == 'zero' else '10b'}.executar, sem mudança, com o modelo de partida, o professor e a pasta desta etapa (`etapa11c.emprestar`)",
            "quantizacao": f"etapa{'8' if caminho == 'zero' else '10c'}.quantizar_modelo, sem mudança",
            "latencia": f"etapa{'8' if caminho == 'zero' else '10c'}.medir_latencia, sem mudança, na pasta desta etapa"}} for caminho, r in R.items()},
        "sinal_do_reajuste": "o de quem treinou o aluno (Etapas 7 e 10b): os rótulos do professor novo no treino e no dev de parada e, no B2, os logits dele; nenhum rótulo humano entra. "
                             "`etapa11b.dados_do_aluno` confere, em cada execução, que os três vieram da pasta desse professor",
        "o_que_e_exportado_e_quantizado": {
            "variantes": "o modelo podado e reajustado e, como as Etapas 8 e 10c fizeram ('como referência'), o mesmo aluno sem poda: é ele que dá a latência do aluno destilado no "
                         "mesmo protocolo e o ponto 'INT8, sem poda' do registro",
            "formas": "a função de origem produz todas as formas dela (ONNX de 32 bits, dinâmica, estática e as duas de diagnóstico). A forma desta etapa é a escolhida na origem; as "
                      "outras são subproduto da função reaproveitada: nenhuma conta desta etapa as lê para decidir, e a estática só participa do rodízio da latência, para o "
                      "protocolo ser o mesmo das Etapas 8 e 10c",
            "fica_de_fora": "a poda de pesos soltos (a secundária das Etapas 7 e 10b), o treino ciente de quantização e qualquer busca nova",
        },
        "conferencias": CONFERENCIAS,
        "custo": {
            "tamanho": "bytes dos arquivos gravados, como nas Etapas 8 e 10c (no XLM-R, com a rede separada do tokenizador)",
            "latencia": "o protocolo das Etapas 8 e 10c: semente 0, 1 thread, uma consulta por vez, do texto à resposta, pedidos de dev, 5 rodadas de 1.024 consultas, sistemas alternados "
                        "em cada rodada (PyTorch, ONNX de 32 bits, dinâmica e estática), por último na cadeia, com a placa e o resto parados",
            "maquina_quieta": f"os outros processos ocupam no máximo {LIMITE_DE_OUTROS} núcleo de CPU durante a medida (`etapa8.nucleos_de_outros`). Uma medida acima disso é guardada como "
                              f"`latencia_recusada_<n>` e refeita, até {TENTATIVAS_DE_LATENCIA} tentativas; se nenhuma passar, a última fica e a cadeia termina com `FIM COM ERROS`",
        },
        "sementes": list(SEMENTES), "semente_do_custo": SEMENTE_DO_CUSTO,
        "execucoes": {"total": len(execucoes), **{passo: sum(e[0] == passo for e in execucoes) for passo in PASSOS},
                      "lista": [{"passo": p, "aluno": n, "variante": v, "semente": s} for p, n, v, s in execucoes]},
        "ordem_da_cadeia": "poda e reajuste (rede do zero, depois XLM-R: as duas famílias nunca dividem a placa; do XLM-R, um processo por vez), depois exportação e quantização "
                           "(na CPU), e a latência por último, um modelo por vez",
        "pastas": f"data/execucoes/{ETAPA}/poda/<variante da 11b>/<poda>/semente_<n>/ e data/execucoes/{ETAPA}/quantizacao/<variante da 11b>/<poda ou sem_poda>/semente_<n>/",
        "decisoes_lidas": {d: etapa10d.sha256(RAIZ / d) for d in decisoes},
        "artigos": [{"quem": a["quem"], "onde": a.get("arxiv") or a.get("acl"), "para_que": a["tecnica"]} for a in etapa10d.ARTIGOS
                    if a["passo"] in ("Poda de vocabulário", "Reajuste", "Quantização") or a["quem"] in ("Reddi et al.", "Koehn", "Berg-Kirkpatrick et al.", "Reimers & Gurevych")],
    }


def gravar_regras(professor: str = PROFESSOR) -> Path:
    """Grava `regras.json`, com a hora, e a versão das regras do comparativo que vale nessa hora. Se já existe, não regrava: a data prova que veio antes das execuções."""
    destino = pasta_da_etapa() / "regras.json"
    if not destino.exists():
        do_comparativo = gravar_regras_do_comparativo(professor)
        comparativo = _json(do_comparativo)
        _gravar_json(destino, {"escritas_em": _agora(), **montar_regras(professor), "comparativo": {
            "regras": _relativo(do_comparativo), "versao": comparativo["versao"], "sha256": etapa10d.sha256(do_comparativo),
            "sistemas": len(comparativo["sistemas_no_registro"]), "comparacoes": len(comparativo["comparacoes"]), "cadeias": list(comparativo["cadeias"]),
            "primarias": comparativo["metodo"]["primarias"],
            "como_ler": "as regras do comparativo são um arquivo com versão: se o registro crescer antes das contas (um professor novo na entrega da Etapa 11a, por exemplo), "
                        "grava-se a versão seguinte e esta não é tocada",
        }})
    return destino


# --------------------------------------------------------------------------
# O comparativo: o registro com os sistemas da Etapa 11 e as regras, numa versão nova
# --------------------------------------------------------------------------

FAMILIAS_NOVAS = {
    "troca_de_professor": "o mesmo aluno, no mesmo caminho e no mesmo ponto da cadeia (destilado, podado, int8), com o professor novo menos com o antigo. Cada B2 está na temperatura "
                          "que a regra escolheu em dev com o seu professor",
    "cenario_a_menos_b": "em cada caminho, o aluno A menos cada aluno do cenário B (com o professor antigo e com o novo), como saiu da destilação e na forma final de 8 bits: "
                         "é a distância entre os cenários, antes e depois da troca de professor",
    "compressao_inteira": "cada modelo em INT8 menos o mesmo modelo como saiu do treino: o custo de toda a compressão (para os sistemas antigos, são as comparações b, d e e2 da Etapa 9)",
    "entre_professores": "cada professor aberto menos o professor fechado, e cada professor aberto da Etapa 11a menos o aberto antigo",
}
PONTOS_DA_TROCA = ("destilado", "podado", "int8")


def _sem_professor(modelo: str) -> str:
    return modelo.split("@")[0]


def montar_sistemas(professor: str = PROFESSOR) -> dict:
    """Os sistemas e as cadeias que esta etapa declara para o comparativo, no formato de `etapa10d.fichas_declaradas`.

    Não lê nenhum resultado: só diz onde cada coisa vai estar na entrega
    desta etapa. Por isso serve antes das execuções, para gravar as regras, e
    depois, para gravar o `sistemas.json` da entrega. `complementos` não é
    lido pela 10d: é o que o registro desta etapa acrescenta às fichas que a
    Etapa 11b declarou (onde ler a latência do aluno destilado).
    """
    poucos = _json(ENTREGAS / "05a_professor_fechado" / "configuracao.json")["conjunto_de_demonstracao"]["exemplos"]
    quem, R = etapa11b.nome_do_professor(professor), receitas()
    declaradas_pela_11b = _json(ENTREGAS / etapa11b.ETAPA / "sistemas.json")["sistemas"]
    sistemas, cadeias, complementos = {}, {}, {}
    cpu_1, onnx_1 = "1 thread, PyTorch", "1 thread, ONNX Runtime"
    for nome, a in alunos(professor).items():
        chave, r, c = a["chave"], R[a["caminho"]], a["caminho"]
        poda, forma = r["poda"], r["forma"]
        de_partida = declaradas_pela_11b[chave]
        etapa_da_poda, etapa_da_forma = (etapa9.etapa_curta(ORIGEM[c][e]) for e in ("poda", "quantizacao"))
        linha = lambda variante, f: [ETAPA, {"modelo": nome, "variante": variante, "forma": f}]
        respostas = lambda molde: {p: [f"entregas/{ETAPA}/previsoes/{molde}_semente_{s}_{p}.parquet" for s in SEMENTES] for p in ("dev", "test")}
        decidido = f"{de_partida['dev_humano'].replace('nenhuma no treino (parada no dev rotulado pelo professor)', 'nenhuma no treino')}; poda da {etapa_da_poda}, sem nova escolha"
        comum = {"bloco": f"Cenário B com o {quem}", "rotulos_humanos": poucos, "rotulos_como": "pelo professor", "condicao": etapa9.CPU, "de_partida": chave, "sementes": len(SEMENTES),
                 "caminho": c, "modelo": a["modelo"], "professor": professor, "variante_da_11b": nome, "papel": f"{a['papel']}; Etapa 11c"}
        nome_da_poda = r["nome_da_poda"][0].lower() + r["nome_da_poda"][1:]
        sistemas[f"{chave}/podado"] = {
            **comum, "nome": f"{a['curto']}, {nome_da_poda}, PyTorch, 32 bits", "curto": f"{a['curto']}, podado", "ponto": "podado", "dev_humano": decidido,
            "resultado": linha(poda, "pytorch"), "previsoes": respostas(f"{nome}__{poda}"), "onde": cpu_1,
            "outras_fontes": [[ETAPA, {"modelo": nome, "variante": poda, "forma": "poda_depois_do_reajuste"}]],
            "degrau_gravado": [ETAPA, {"modelo": nome, "variante": poda, "comparacao": "depois − sem poda", "particao": "test"}],
        }
        sistemas[f"{chave}/onnx"] = {
            **comum, "nome": f"{a['curto']}, {nome_da_poda}, ONNX, 32 bits", "curto": f"{a['curto']}, ONNX 32 bits", "ponto": "onnx", "dev_humano": decidido,
            "resultado": linha(poda, "fp32"), "previsoes": respostas(f"{nome}__{poda}__fp32"), "onde": onnx_1,
            "degrau_gravado": [ETAPA, {"modelo": nome, "variante": poda, "comparacao": "fp32 − pytorch", "particao": "test"}],
        }
        for ponto, variante, rotulo in (("int8", poda, "INT8"), ("int8_sem_poda", SEM_PODA, "INT8 sem poda")):
            sistemas[f"{chave}/{ponto}"] = {
                **comum, "nome": f"{a['curto']}, {nome_da_poda if variante == poda else 'sem poda'}, {r['nome_da_forma']}", "curto": f"{a['curto']}, {rotulo}", "ponto": ponto,
                "dev_humano": f"{decidido if variante == poda else de_partida['dev_humano']}; forma de 8 bits da {etapa_da_forma}, sem nova escolha",
                "resultado": linha(variante, forma), "previsoes": respostas(f"{nome}__{variante}__{forma}"), "onde": onnx_1, "compactado": False,
                **({"bloco": "Quantizado sem poda"} if variante == SEM_PODA else
                   {"degrau_gravado": [ETAPA, {"modelo": nome, "variante": poda, "comparacao": f"{forma} − pytorch", "particao": "test"}]}),
            }
        complementos[chave] = {"eficiencia": linha(SEM_PODA, "pytorch"), "condicao": etapa9.CPU, "onde": cpu_1, "outras_fontes": [linha(SEM_PODA, "pytorch")],
                               "degrau_gravado": [etapa11b.ETAPA, {"id": f"aluno_menos_professor|{c}:{a['sigla']}:novo|prof:novo", "particao": "test"}],
                               "por_que": "a Etapa 11b não mediu latência; a deste aluno, sem poda, é medida aqui, no PyTorch, na mesma sessão das formas comprimidas. "
                                          "O degrau 'aluno − professor' foi gravado pela Etapa 11b, e a qualidade do aluno respondendo na CPU, por esta"}
        cadeias[f"{c}/{a['sigla']}@{professor}"] = {
            "nome": f"{etapa10d.CAMINHOS[c]['curto']} · Cenário B com o {quem} (aluno {a['sigla']})", "caminho": c, "cenario": "B", "professor": professor, "representa": a["representa"],
            "degraus": [["professor", professor], ["destilado", chave], ["podado", f"{chave}/podado"], ["onnx", f"{chave}/onnx"], ["int8", f"{chave}/int8"]],
        }
    return {
        "sistemas": sistemas, "cadeias": cadeias, "complementos": complementos,
        "fora_do_formato": {"a forma estática e as formas de diagnóstico": "subproduto das funções das Etapas 8 e 10c: não são sistemas do comparativo",
                            "o aluno B1 da rede do zero": "não é o representante do cenário B nesse caminho (a regra aponta o B2): não foi comprimido"},
    }


def registro(entregas: Path | None = None, base: Path | None = None, professor: str = PROFESSOR) -> dict[str, dict]:
    """O registro do comparativo desta etapa: o da 10d, lendo também as entregas da Etapa 11 (`ENTREGAS_QUE_VALEM`).

    As respostas que a 10d gerou (os INT8 da rede do zero em `dev`) são lidas
    da entrega dela. Enquanto a entrega desta etapa não tem `sistemas.json`,
    os sistemas dela entram como planejados (`montar_sistemas`): as marcas
    são as mesmas, e é delas que saem as regras.
    """
    entregas = entregas or ENTREGAS
    fichas = etapa10d.registro(entregas, base or entregas / etapa10d.ETAPA, valem=ENTREGAS_QUE_VALEM)
    declarado = entregas / ETAPA / "sistemas.json"
    daqui = _json(declarado) if declarado.exists() else montar_sistemas(professor)
    if not declarado.exists():
        fichas.update(etapa10d.fichas_de(daqui, entregas.parent, "planejado (a entrega da Etapa 11c ainda não existe)")[1])
    for chave, mais in daqui.get("complementos", {}).items():
        fichas[chave] = {**fichas[chave], **{k: (tuple(v) if k in ("eficiencia", "degrau_gravado") else v) for k, v in mais.items() if k != "por_que"}}
    return fichas


def cadeias_do_registro(fichas: dict[str, dict], entregas: Path | None = None, professor: str = PROFESSOR) -> dict[str, dict]:
    """As cadeias da 10d mais as desta etapa (planejadas, enquanto a entrega dela não existe)."""
    entregas = entregas or ENTREGAS
    cadeias = etapa10d.cadeias_do_registro(fichas, entregas, valem=ENTREGAS_QUE_VALEM)
    if not (entregas / ETAPA / "sistemas.json").exists():
        cadeias.update(etapa10d.fichas_de(montar_sistemas(professor), entregas.parent, "planejado")[2])
    return cadeias


def lista_de_comparacoes(fichas: dict[str, dict], entregas: Path | None = None) -> list[dict]:
    """A lista fechada do comparativo: a da 10d (`etapa10d.lista_de_comparacoes`, por regra) mais as famílias de `FAMILIAS_NOVAS`, também geradas das marcas do registro.

    Um par que cai em duas famílias aparece uma vez, com as duas. B − A.
    """
    entregas = entregas or ENTREGAS
    curtos = {c: d["curto"] for c, d in etapa10d.caminhos_do_registro(entregas, ENTREGAS_QUE_VALEM).items()}
    pares = {(item["b"], item["a"]): item for item in etapa10d.lista_de_comparacoes(fichas, entregas, valem=ENTREGAS_QUE_VALEM)}

    def por(familia: str, ident: str, b: str, a: str, pergunta: str) -> None:
        item = pares.setdefault((b, a), {"id": ident, "familias": [], "b": b, "a": a, "pergunta": pergunta, "gravado": None})
        if familia not in item["familias"]:
            item["familias"].append(familia)

    marcas = {(f["caminho"], f["modelo"], f["ponto"]): c for c, f in fichas.items() if f.get("caminho")}
    professores = {c: f for c, f in fichas.items() if f["ponto"] == "professor"}
    abertos = [c for c, f in professores.items() if "aberto" in (f.get("papel") or "")]
    fechados = [c for c, f in professores.items() if "fechado" in (f.get("papel") or "")]
    antigos = [c for c in abertos if "declarado_em" not in professores[c]]

    for chave, f in fichas.items():   # 6. a troca de professor, no mesmo ponto
        if f.get("caminho") and "@" in f["modelo"] and f["ponto"] in PONTOS_DA_TROCA:
            antigo = marcas.get((f["caminho"], _sem_professor(f["modelo"]), f["ponto"]))
            if antigo:
                por("troca_de_professor", f"troca:{chave}", chave, antigo, f"{curtos[f['caminho']]}: {f['curto']} menos o mesmo aluno com o professor antigo")
    for caminho in curtos:            # 7. o cenário A menos o cenário B, em cada caminho
        for ponto in etapa10d.CONTRA_O_PROFESSOR:
            do_a = marcas.get((caminho, "aluno_a", ponto))
            for chave, f in fichas.items():
                if do_a and f.get("caminho") == caminho and f["ponto"] == ponto and f.get("professor") in abertos:
                    por("cenario_a_menos_b", f"AB:{chave}", do_a, chave, f"{curtos[caminho]}: {fichas[do_a]['curto']} menos {f['curto']}")
    for chave, f in fichas.items():   # 8. o custo de toda a compressão
        if f["ponto"] == "int8" and f.get("de_partida") in fichas:
            por("compressao_inteira", f"compressao:{chave}", chave, f["de_partida"], f"{curtos[f['caminho']]}: {f['curto']} menos o mesmo modelo como saiu do treino")
    for aberto in abertos:            # 9. os professores entre si
        for fechado in fechados:
            por("entre_professores", f"professores:{aberto}:{fechado}", aberto, fechado, "professor aberto menos o professor fechado")
        for antigo in antigos:
            if aberto != antigo:
                por("entre_professores", f"professores:{aberto}:{antigo}", aberto, antigo, "professor aberto da Etapa 11a menos o professor aberto antigo")
    return list(pares.values())


def primarias(fichas: dict[str, dict], professor: str = PROFESSOR) -> list[dict]:
    """As leituras principais do comparativo: em cada caminho, o aluno A em INT8 menos o representante do cenário B com o professor novo, em INT8."""
    return [{"b": etapa10d.CAMINHOS[a["caminho"]]["prefixo"] + "aluno_a/int8", "a": f"{a['chave']}/int8", "caminho": a["caminho"]} for a in alunos(professor).values() if a["representa"]]


def montar_regras_do_comparativo(professor: str = PROFESSOR, entregas: Path | None = None) -> dict:
    """As regras do comparativo desta etapa: as da 10d (`etapa10d.montar_regras`), com o registro maior e as famílias novas. Não lê nenhum resultado."""
    entregas = entregas or ENTREGAS
    fichas = registro(entregas, professor=professor)
    lista, cadeias = lista_de_comparacoes(fichas, entregas), cadeias_do_registro(fichas, entregas, professor)
    lidas = tuple(e for e in ENTREGAS_QUE_VALEM if e != ETAPA)   # a decisão desta etapa é saída dela, não entrada
    regras = etapa10d.montar_regras(fichas, entregas, valem=lidas)
    principais = primarias(fichas, professor)
    em_lista = {(c["b"], c["a"]) for c in lista}
    if faltando := [p for p in principais if (p["b"], p["a"]) not in em_lista]:
        raise ValueError(f"As leituras principais não estão na lista de comparações: {faltando}")
    anteriores = sorted((entregas / etapa10d.ETAPA).glob("regras*.json"), key=lambda c: _json(c)["versao"])
    regras.update({
        "etapa": ETAPA, "pergunta": PERGUNTA,
        "continua": {"de": f"entregas/{etapa10d.ETAPA}/", "versoes_anteriores": {str(c.relative_to(entregas.parent)): {"versao": _json(c)["versao"], "sha256": etapa10d.sha256(c)} for c in anteriores},
                     "o_que_muda": "o registro lê também as entregas da Etapa 11 (os professores abertos da 11a, os alunos da 11b e as formas comprimidas da 11c) e a lista ganha "
                                   "as famílias de `familias_novas`; o método, a fronteira e a conferência são os da 10d, sem mudança"},
        "entregas_que_valem": list(ENTREGAS_QUE_VALEM),
        "familias": {**regras["familias"], **FAMILIAS_NOVAS}, "familias_novas": list(FAMILIAS_NOVAS),
        "comparacoes": [{k: v for k, v in item.items() if k != "gravado"} for item in lista],
        "cadeias": {nome: {"nome": c["nome"], "degraus": [list(d) for d in c["degraus"]]} for nome, c in cadeias.items()},
        "sistemas_no_registro": list(fichas),
        "sistemas_desta_etapa": [c for c, f in fichas.items() if str(f.get("declarado_em", "")).startswith(("planejado", f"entregas/{ETAPA}/"))],
    })
    regras["metodo"] = {
        **regras["metodo"],
        "decisoes_nesta_etapa": "nenhuma: os alunos novos recebem as receitas que as Etapas 7, 8, 10b e 10c já escolheram; os modelos finais de cada caminho continuam os da 10d",
        "primarias": {"o_que_sao": "em cada caminho, o aluno A em INT8 menos o representante do cenário B com o professor novo em INT8, nas três métricas, no teste: é o que resta de "
                                   "distância entre os cenários depois da troca de professor. As outras comparações descrevem de onde ela vem",
                      "pares": principais},
    }
    regras["ja_era_conhecido_ao_escrever"] = [
        "tudo o que a Etapa 10d entregou: os 44 sistemas antigos, as 43 comparações dela, a fronteira e o preço de cada ponto",
        "os alunos novos como saíram da destilação: a qualidade de cada um, 'novo − antigo', 'aluno − professor' e 'aluno B − aluno A' no ponto destilado já estão na Etapa 11b",
        "os professores entre si já estão nas Etapas 11a e 11b",
        "novas nesta etapa: tudo o que envolve um aluno novo podado ou em 8 bits (os degraus das cadeias novas, a troca de professor nos pontos podado e INT8, o cenário A menos o "
        "cenário B em INT8 com o professor novo, XLM-R menos rede do zero no B2 novo comprimido) e a fronteira com esses pontos. Nenhum desses números existia quando as regras foram gravadas",
    ]
    return regras


def gravar_regras_do_comparativo(professor: str = PROFESSOR) -> Path:
    """Grava as regras do comparativo como a versão seguinte da linhagem da 10d (`comparativo/regras_v3.json`). Se o registro crescer, grava a próxima e não toca nas anteriores."""
    pasta = pasta_da_etapa() / "comparativo"
    regras = json.loads(json.dumps(montar_regras_do_comparativo(professor)))
    existentes = sorted(pasta.glob("regras_v*.json"), key=lambda c: _json(c)["versao"])
    if existentes and {k: v for k, v in _json(existentes[-1]).items() if k not in ("versao", "gravado_em")} == regras:
        return existentes[-1]
    versao = VERSAO_DAS_REGRAS_DO_COMPARATIVO + len(existentes)
    destino = pasta / f"regras_v{versao}.json"
    _gravar_json(destino, {"versao": versao, "gravado_em": datetime.now().isoformat(timespec="seconds"), **regras})
    return destino


# --------------------------------------------------------------------------
# Uma execução de cada passo
# --------------------------------------------------------------------------

# O teste de fumaça: os dados e o professor de verdade, reajuste de poucas épocas, poucos pedidos e poucas consultas, numa
# pasta descartável. Passa pelo mesmo código da cadeia inteira.
FUMACA = {
    "epocas": {"zero": 2, "xlmr": 1}, "receita": {"zero": {"avaliar_a_cada": 1}, "xlmr": {}},
    "pedidos": 64, "latencia": {"rodadas": 1, "consultas": 32},
    "o_que_roda": "semente 0: a rede do zero e o representante do XLM-R passam pelos três passos; o outro aluno do XLM-R, só pela poda",
}


def podar(nome: str, semente: int, dispositivo: str | None = None, professor: str = PROFESSOR) -> Path:
    """Poda e reajusta um aluno novo com a função da etapa de origem (`etapa7.executar` ou `etapa10b.executar`), sem mudança de código.

    Dentro do bloco, a etapa de origem enxerga um modelo de partida só (o
    aluno da Etapa 11b), lê os dados do professor novo e grava na pasta
    desta etapa. Fora do cache de verdade (teste de fumaça), o reajuste é
    curto.
    """
    exigir_regras()
    a = alunos(professor)[nome]
    caminho, curta = a["caminho"], not no_cache_de_verdade()
    variante = receitas()[caminho]["poda"]
    receita = {**a["receita"], **(FUMACA["receita"][caminho] if curta else {})}
    trocas = {"ETAPA": f"{ETAPA}/poda", "dados_do_modelo": lambda modelo: etapa11b.dados_do_aluno(caminho, a["aluno"], professor)}
    if caminho == "zero":
        from especialista.experimentos import etapa7 as origem
        ficha = {"nome": a["nome"], "curto": a["curto"], "pasta": RAIZ / a["modelos"], "config": a["config"], "receita": receita, "previsoes": None,
                 "variante_da_etapa_6": a["aluno"], "principal": False}
    else:
        from especialista.experimentos import etapa10b as origem
        ficha = {"nome": a["nome"], "curto": a["curto"], "variante_da_10a": a["aluno"], "pastas": a["pastas"], "config": a["config"], "receita": receita, "sinal": a["sinal"],
                 "previsoes": None, "grade": "do_principal", "principal": False}
    if curta:
        trocas["EPOCAS"] = FUMACA["epocas"][caminho]
    with emprestar(origem, modelos_a_podar=lambda: {nome: ficha}, **trocas):
        pasta = origem.pasta_da_execucao(nome, variante, semente)
        if pasta != pasta_da_poda(nome, semente, professor) or not pasta.is_relative_to(pasta_da_etapa()):
            raise RuntimeError(f"A poda de {nome} iria para {pasta}, fora da pasta desta etapa.")
        return origem.executar(nome, variante, semente, dispositivo)


def quantizar(nome: str, variante: str, semente: int, threads: int = THREADS, professor: str = PROFESSOR) -> Path:
    """Exporta para ONNX e quantiza um modelo (o podado ou o sem poda) com a função da etapa de origem, e grava o `meta.json` no formato dela."""
    import onnx
    import onnxruntime
    import torch

    from especialista.dados import massive
    from especialista.experimentos import etapa8, etapa10c

    exigir_regras()
    a = alunos(professor)[nome]
    caminho = a["caminho"]
    if variante not in variantes(caminho):
        raise ValueError(f"{nome} não tem a variante {variante}. As que existem: {', '.join(variantes(caminho))}.")
    destino = pasta_da_quantizacao(nome, variante, semente)
    if (destino / "meta.json").exists():
        return destino
    if destino.exists():   # sobra de uma execução interrompida, dentro da pasta desta etapa
        shutil.rmtree(destino)
    if variante == SEM_PODA:
        de_onde = a["pastas"][semente]
    else:
        if not (pasta_da_poda(nome, semente, professor) / "meta.json").exists():
            raise FileNotFoundError(f"Pode antes: falta {pasta_da_poda(nome, semente, professor) / 'meta.json'}.")
        de_onde = pasta_da_poda(nome, semente, professor) / "modelo"
    particoes, _ = massive.carregar_entrega()
    medir = {"dev": particoes["dev"], "test": particoes["test"]}
    if not no_cache_de_verdade():
        medir = {p: tabela.head(FUMACA["pedidos"]) for p, tabela in medir.items()}
    calibracao = etapa8.textos_de_calibracao(particoes["train"])

    inicio = time.perf_counter()
    if caminho == "zero":
        medido = etapa8.quantizar_modelo(de_onde, destino, medir, calibracao)
    else:
        medido = etapa10c.quantizar_modelo(de_onde, destino, medir, calibracao, threads, guardar_fp32=semente == SEMENTE_DO_CUSTO)
    _gravar_json(destino / "meta.json", {   # por último: marca "pronto"
        "etapa": f"{ETAPA}/quantizacao", "modelo": nome, "variante": variante, "nome": f"{a['nome']}, {'sem poda' if variante == SEM_PODA else receitas()[caminho]['nome_da_poda']}",
        "papel": "sem_poda" if variante == SEM_PODA else "principal", "semente": semente, "caminho": caminho, "professor": professor, "de_partida": _relativo(de_onde),
        **medido,
        "calibracao": {"pedidos": len(calibracao), "de_onde": "pedidos do treino, sem rótulo", "semente_do_sorteio": etapa8.SEMENTE_DA_CALIBRACAO},
        "threads": threads if caminho == "xlmr" else None, "segundos": time.perf_counter() - inicio,
        "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, **etapa8.ambiente(),
    })
    return destino


def medir_latencia(nome: str, variante: str, professor: str = PROFESSOR, tentativas: int | None = None, espera: int = ESPERA_ENTRE_TENTATIVAS) -> tuple[Path, bool]:
    """Mede a latência (semente 0) com a função da etapa de origem. Devolve a pasta e se a máquina estava quieta.

    Uma medida com os outros processos acima de `LIMITE_DE_OUTROS` não é
    apagada: muda de nome (`latencia_recusada_<n>`) e a medida é refeita.
    """
    from especialista.experimentos import etapa8, etapa10c

    exigir_regras()
    caminho, curta = alunos(professor)[nome]["caminho"], not no_cache_de_verdade()
    pasta = pasta_da_quantizacao(nome, variante, SEMENTE_DO_CUSTO)
    tentativas = tentativas or (1 if curta else TENTATIVAS_DE_LATENCIA)
    quieta = False
    for tentativa in range(1, tentativas + 1):
        if caminho == "zero":
            etapa8.medir_latencia(nome, variante, base_da_quantizacao(), SEMENTE_DO_CUSTO, rodadas=FUMACA["latencia"]["rodadas"] if curta else etapa8.RODADAS)
        else:
            mais = FUMACA["latencia"] if curta else {}
            etapa10c.medir_latencia(nome, variante, base_da_quantizacao(), SEMENTE_DO_CUSTO, **mais)
        quieta = _json(pasta / "latencia.json")["nucleos_de_outros"] <= LIMITE_DE_OUTROS
        if quieta or tentativa == tentativas:
            break
        recusadas = len(list(pasta.glob("latencia_recusada_*.json"))) + 1
        for extensao in ("json", "npz"):
            (pasta / f"latencia.{extensao}").rename(pasta / f"latencia_recusada_{recusadas}.{extensao}")
        time.sleep(espera)
    return pasta, quieta


# --------------------------------------------------------------------------
# A cadeia: tudo o que falta, num processo só que pode ficar sozinho
# --------------------------------------------------------------------------


def _da_fumaca(professor: str) -> list[tuple[str, str, str, int]]:
    daqui = alunos(professor)
    inteiros = {next(n for n, a in daqui.items() if a["caminho"] == c and (a["representa"] or c == "zero")) for c in ORIGEM}
    return [e for e in grade(professor) if e[3] == SEMENTE_DO_CUSTO and (e[0] == "poda" or e[1] in inteiros)]


def _rodar(passo: str, nome: str, variante: str, semente: int, professor: str, fumaca: Path | None) -> int:
    """Uma execução num processo próprio: a memória (da placa e da máquina) volta inteira quando ele acaba."""
    comando = [sys.executable, "-m", "especialista.experimentos.etapa11c", "executar", "--passo", passo, "--aluno", nome, "--variante", variante, "--semente", str(semente),
               "--professor", professor]
    if fumaca is not None:
        comando += ["--fumaca", str(fumaca)]
    ambiente = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "OMP_NUM_THREADS": {"poda": "2", "quantizar": str(THREADS), "latencia": "1"}[passo]}
    if passo != "poda":   # a exportação, a quantização e a latência são na CPU: a placa nem é vista
        ambiente["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(comando, env=ambiente).returncode


def conferir_execucoes(feitas: list[tuple[str, str, str, int]], professor: str = PROFESSOR) -> dict:
    """O que a cadeia deixou gravado, conferido sem fazer nenhuma conta de qualidade: o que falta, de que professor cada poda leu, se o ONNX repete o PyTorch e se a latência é válida."""
    daqui = alunos(professor)
    falta = [e for e in feitas if not pronta(*e)]
    de_outro_professor, onnx_diferente, compactado_diferente, pedidos, iguais, outros = [], [], [], 0, 0, {}
    for passo, nome, variante, semente in feitas:
        if not pronta(passo, nome, variante, semente):
            continue
        if passo == "poda":
            meta = _json(pasta_da_poda(nome, semente, professor) / "meta.json")
            origem, campos = meta["origem"], ("treino", "dev", "logits") if "b2" in daqui[nome]["aluno"] else ("treino", "dev")
            if origem.get("professor") != professor or not all(professor in origem.get(campo, "") for campo in campos) or etapa11b.ETAPA not in meta["de_partida"]:
                de_outro_professor.append(f"{nome}/semente_{semente}")
        elif passo == "quantizar":
            meta = _json(pasta_da_quantizacao(nome, variante, semente) / "meta.json")
            forma = receitas()[daqui[nome]["caminho"]]["forma"]
            for particao, quantos in meta["pedidos"].items():
                pedidos += quantos
                iguais += meta["formas"]["fp32"][f"iguais_ao_pytorch_{particao}"]
                if meta["formas"]["fp32"][f"iguais_ao_pytorch_{particao}"] != quantos:
                    onnx_diferente.append(f"{nome}/{variante}/semente_{semente}/{particao}")
            if not meta["formas"][forma].get("compactado_responde_igual", True):
                compactado_diferente.append(f"{nome}/{variante}/semente_{semente}")
        else:
            outros[f"{nome}/{variante}"] = _json(pasta_da_quantizacao(nome, variante, semente) / "latencia.json")["nucleos_de_outros"]
    return {"faltam": [f"{p}:{n}/{v}/semente_{s}" for p, n, v, s in falta], "podas_que_nao_leram_do_professor": de_outro_professor,
            "onnx_de_32_bits": {"pedidos": pedidos, "iguais_ao_pytorch": iguais, "diferentes_em": onnx_diferente}, "compactado_que_responde_diferente": compactado_diferente,
            "outros_processos_na_latencia": outros, "latencias_com_a_maquina_ocupada": [m for m, n in outros.items() if n > LIMITE_DE_OUTROS]}


def cadeia(professor: str = PROFESSOR, fumaca: Path | None = None) -> bool:
    """Roda tudo o que falta, na ordem de `PASSOS`: poda e reajuste, exportação e quantização, latência por último.

    Em cada passo, primeiro a rede do zero e depois o XLM-R: as duas
    famílias nunca dividem a placa, e o XLM-R é podado um processo por vez.
    Pode ser interrompida e chamada de novo: o que já está gravado não é
    refeito. Uma execução com erro não para as outras. A última linha que
    imprime começa com `FIM:` ou com `FIM COM ERROS:`. Com `fumaca`, roda só
    o teste curto, na pasta descartável dada, pelo mesmo código.
    """
    avisar = lambda texto: print(f"[{datetime.now():%d/%m %H:%M:%S}] {texto}", flush=True)
    try:
        if fumaca is not None:
            regras_de_verdade = pasta_da_etapa() / "regras.json"
            etapa11b.usar_pasta_descartavel(fumaca)
            if not (pasta_da_etapa() / "regras.json").exists():
                pasta_da_etapa().mkdir(parents=True, exist_ok=True)
                shutil.copy2(regras_de_verdade, pasta_da_etapa() / "regras.json")
        regras = exigir_regras()
        inicio = datetime.now()
        relato_em = pasta_da_etapa() / "cadeias" / f"cadeia_{inicio:%Y%m%d-%H%M%S}.json"
        codigo = etapa11b.copiar_o_codigo(relato_em.with_name(f"codigo_{inicio:%Y%m%d-%H%M%S}"))
        feitas = _da_fumaca(professor) if fumaca is not None else grade(professor)
        daqui = alunos(professor)
        avisar(f"Etapa 11c, {'teste de fumaça' if fumaca else 'cadeia'} (pid {os.getpid()}): regras de {regras['escritas_em']}; {len(feitas)} execuções previstas; "
               f"{sum(pronta(*e) for e in feitas)} já prontas; carga da máquina: {os.getloadavg()[0]:.2f}")
        relato = {"etapa": ETAPA, "fumaca": fumaca is not None, "pid": os.getpid(), "iniciada_em": _agora(), "regras_escritas_em": regras["escritas_em"], "professor": professor,
                  "ambiente": etapa11b.ambiente(), "codigo": codigo, "copia_do_codigo": str(relato_em.with_name(f"codigo_{inicio:%Y%m%d-%H%M%S}")), "pasta": str(pasta_da_etapa()),
                  "previstas": [list(e) for e in feitas]}
        _gravar_json(relato_em, relato)

        erros: list[str] = []
        for passo in PASSOS:
            if passo == "latencia" and fumaca is None:
                time.sleep(60)   # a máquina assenta antes da medida
            for caminho, origem in ORIGEM.items():
                pendentes = [e for e in feitas if e[0] == passo and daqui[e[1]]["caminho"] == caminho and not pronta(*e)]
                if not pendentes:
                    continue
                processos = origem["processos"].get(passo, 1)
                avisar(f"{passo}, {origem['nome']}: {len(pendentes)} execuções, {processos} de cada vez (carga da máquina: {os.getloadavg()[0]:.2f})")
                with ThreadPoolExecutor(max_workers=processos) as fila:
                    saidas = list(fila.map(lambda e: _rodar(*e, professor, fumaca), pendentes))
                erros += [f"{p}:{n}/{v}/semente_{s} (saída {saida})" for (p, n, v, s), saida in zip(pendentes, saidas) if saida != 0]

        conferido = conferir_execucoes(feitas, professor)
        no_fim = etapa11b.impressao_do_codigo()
        alterados = sorted(r for r in set(codigo) | set(no_fim) if codigo.get(r) != no_fim.get(r))
        minutos = (datetime.now() - inicio).total_seconds() / 60
        relato.update({"terminada_em": _agora(), "minutos": round(minutos, 1), "prontas": len(feitas) - len(conferido["faltam"]), "erros": erros, "conferido": conferido,
                       "codigo_igual_do_inicio_ao_fim": not alterados, "codigo_alterado_durante_a_cadeia": alterados})
        _gravar_json(relato_em, relato)

        onnx = conferido["onnx_de_32_bits"]
        resumo = (f"{relato['prontas']} de {len(feitas)} execuções prontas em {minutos:.0f} min; ONNX de 32 bits igual ao PyTorch em {onnx['iguais_ao_pytorch']} de {onnx['pedidos']} respostas; "
                  f"outros processos durante a latência: no máximo {max(conferido['outros_processos_na_latencia'].values(), default=float('nan')):.2f} núcleo de CPU; registro em {relato_em}")
        if alterados:   # não é erro de execução, mas quem lê precisa saber
            resumo += f"; ATENÇÃO: código alterado durante a cadeia em {', '.join(alterados)}"
        problemas = list(erros)
        problemas += [f"faltam {len(conferido['faltam'])} execuções"] if conferido["faltam"] else []
        problemas += [f"podas que não leram do professor {professor}: {conferido['podas_que_nao_leram_do_professor']}"] if conferido["podas_que_nao_leram_do_professor"] else []
        problemas += [f"o ONNX de 32 bits difere do PyTorch em {onnx['diferentes_em']}"] if onnx["diferentes_em"] else []
        problemas += [f"compactado responde diferente em {conferido['compactado_que_responde_diferente']}"] if conferido["compactado_que_responde_diferente"] else []
        if fumaca is None and conferido["latencias_com_a_maquina_ocupada"]:
            problemas.append(f"latência medida com a máquina ocupada em {conferido['latencias_com_a_maquina_ocupada']}")
        if problemas:
            avisar(f"FIM COM ERROS: {'; '.join(problemas)}. {resumo}")
            return False
        avisar(f"FIM: {resumo}")
        return True
    except Exception as erro:   # a última linha do log tem de dizer como acabou
        avisar(f"FIM COM ERROS: a cadeia parou ({type(erro).__name__}: {erro})")
        return False


# --------------------------------------------------------------------------
# Depois da cadeia · As contas da parte 1: as conferências, os tamanhos e as latências
# --------------------------------------------------------------------------

PARTICOES = ("dev", "test")
REAMOSTRAS = etapa9.REAMOSTRAS
FORMAS_MEDIDAS = ("pytorch", "fp32", "dinamica", "estatica")   # as formas com respostas em dev e no teste e com latência
NOME_DA_FORMA = {"pytorch": "PyTorch, 32 bits", "fp32": "ONNX, 32 bits", "dinamica": "INT8, quantização dinâmica", "estatica": "INT8, quantização estática"}
# as respostas que a execução da poda gravou: o nome da linha em `resultados.csv`, o arquivo e como aparece no texto
MOMENTOS = {"poda_antes_do_reajuste": ("previsoes_{p}_antes.parquet", "podado, antes do reajuste"), "poda_depois_do_reajuste": ("previsoes_{p}.parquet", "podado e reajustado")}
ARQUIVOS_DAS_CONTAS = ("resultados.csv", "comparacoes.csv", "conferencias.csv", "eficiencia.csv", "reajustes.csv", "contas.json")


def pasta_das_contas() -> Path:
    return pasta_da_etapa() / "contas"


def carregar_latencia(nome: str, variante: str) -> dict:
    """A medida de latência de um modelo (semente 0): o resumo gravado e os tempos de cada forma, uma linha por rodada."""
    import numpy as np

    pasta = pasta_da_quantizacao(nome, variante, SEMENTE_DO_CUSTO)
    with np.load(pasta / "latencia.npz") as arquivos:
        tempos = {forma: arquivos[forma] for forma in arquivos.files}
    return {"tempos": tempos, **_json(pasta / "latencia.json")}


def respostas_da_parte_1(nome: str, ids: dict[str, list[int]], professor: str = PROFESSOR) -> dict[str, dict[str, list[list]]]:
    """As respostas de tudo o que a parte 1 compara num aluno: {linha: {partição: [uma lista por semente]}}.

    Linhas: `sem_poda` (o aluno como a Etapa 11b o entregou), as duas de
    `MOMENTOS` (a execução da poda) e `<variante>/<forma>` (a exportação e a
    quantização). A ordem dos pedidos de cada arquivo é conferida.
    """
    caminho = alunos(professor)[nome]["caminho"]
    ler = lambda molde: {p: [etapa11b._ler(Path(str(molde(s)).format(p=p)), ids[p]) for s in SEMENTES] for p in PARTICOES}
    linhas = {SEM_PODA: ler(lambda s: ENTREGAS / etapa11b.ETAPA / "previsoes" / (f"{nome}_semente_{s}_" + "{p}.parquet"))}
    for momento, (arquivo, _) in MOMENTOS.items():
        linhas[momento] = ler(lambda s: pasta_da_poda(nome, s, professor) / arquivo)
    for variante in variantes(caminho):
        for forma in FORMAS_MEDIDAS:
            linhas[f"{variante}/{forma}"] = ler(lambda s: pasta_da_quantizacao(nome, variante, s) / (f"previsoes_{forma}_" + "{p}.parquet"))
    return linhas


def fechar(professor: str = PROFESSOR, n: int = REAMOSTRAS) -> Path:
    """As contas da parte 1, gravadas em `contas/`: a tabela de resultados, as comparações pareadas, as conferências, o custo e os reajustes.

    `conferencias.csv` é a resposta à primeira metade da pergunta: cada
    receita, com cada aluno novo, passa ou não passa na regra da etapa de
    origem, em `dev`. `resultados.csv` e `comparacoes.csv` têm o formato das
    Etapas 8 e 10c: é deles que o registro do comparativo lê a qualidade, o
    tamanho, a latência e os degraus das cadeias novas.
    """
    import pandas as pd

    from especialista.avaliacao import qualidade, resultados
    from especialista.avaliacao.previsao import ouro
    from especialista.dados import massive
    from especialista.experimentos import etapa8

    regras = exigir_regras()
    if falta := faltam(professor):
        raise FileNotFoundError(f"Faltam {len(falta)} execuções da cadeia (a primeira: {falta[0]}). Rode: ... etapa11c cadeia")
    particoes, _ = massive.carregar_entrega()
    gabarito = {p: ouro(particoes[p]) for p in PARTICOES}
    ids = {p: list(particoes[p]["id"]) for p in PARTICOES}
    R = receitas()
    L_res, L_comp, L_conf, L_efi, L_rea = [], [], [], [], []
    for nome, a in alunos(professor).items():
        c = a["caminho"]
        poda, escolhida = R[c]["poda"], R[c]["forma"]
        nome_da_variante = {poda: R[c]["nome_da_poda"][0].lower() + R[c]["nome_da_poda"][1:], SEM_PODA: "sem poda"}
        r = respostas_da_parte_1(nome, ids, professor)
        metas_p = [_json(pasta_da_poda(nome, s, professor) / "meta.json") for s in SEMENTES]
        metas_q = {v: [_json(pasta_da_quantizacao(nome, v, s) / "meta.json") for s in SEMENTES] for v in variantes(c)}
        lat = {v: carregar_latencia(nome, v) for v in variantes(c)}

        def compara(variante: str, rotulo: str, de: str, para: str, papel: str) -> None:   # para − de
            for p in PARTICOES:
                iguais = sum(x == y for ga, gb in zip(r[de][p], r[para][p]) for x, y in zip(ga, gb))
                for metrica, linha in qualidade.comparar_grupos(gabarito[p], r[de][p], r[para][p], n=n).iterrows():
                    L_comp.append({"modelo": nome, "variante": variante, "comparacao": rotulo, "papel": papel, "particao": p, "metrica": metrica,
                                   **{k: float(linha[k]) for k in ("a", "b", "diferenca", "ic_min", "ic_max")}, "distinguivel": bool(linha["distinguivel"]),
                                   "respostas_iguais": iguais, "respostas": len(ids[p]) * len(SEMENTES)})

        gravada = lambda variante, rotulo, metrica: next(l for l in reversed(L_comp) if (l["modelo"], l["variante"], l["comparacao"], l["particao"], l["metrica"]) == (nome, variante, rotulo, "dev", metrica))
        comum = {"aluno": nome, "nome": a["curto"], "caminho": c, "sigla": a["sigla"], "representa": a["representa"]}

        # 1. a poda: o modelo podado e reajustado contra o aluno sem poda; e, para ler o degrau, antes do reajuste
        compara(poda, "depois − sem poda", SEM_PODA, "poda_depois_do_reajuste", "conferência da poda: em dev, exact match")
        compara(poda, "antes − sem poda", SEM_PODA, "poda_antes_do_reajuste", "registro: o que a poda derruba antes do reajuste")
        compara(poda, "depois − antes", "poda_antes_do_reajuste", "poda_depois_do_reajuste", "registro: o que o reajuste recupera")
        da_poda = gravada(poda, "depois − sem poda", CONFERENCIAS["poda"]["metrica"])
        L_conf.append({**comum, "conferencia": "poda", "receita": poda, "variante": poda, "particao": "dev", "metrica": CONFERENCIAS["poda"]["metrica"], "referencia": da_poda["a"],
                       "com_a_receita": da_poda["b"], "diferenca": da_poda["diferenca"], "ic_min": da_poda["ic_min"], "ic_max": da_poda["ic_max"],
                       "criterio": "o intervalo de 95% da diferença não fica todo abaixo de zero", "passa": bool(not da_poda["ic_max"] < 0), "conta_para_a_resposta": True})

        # 2. a exportação e os 8 bits, no modelo podado e no sem poda: cada forma contra o mesmo modelo no PyTorch
        for v in variantes(c):
            origem = SEM_PODA if v == SEM_PODA else "poda_depois_do_reajuste"
            compara(v, "pytorch − respostas de origem", origem, f"{v}/pytorch", "registro: a mesma rede respondendo na CPU, contra as respostas gravadas por quem a treinou")
            for forma in FORMAS_MEDIDAS[1:]:
                papel = ("conferência do ONNX de 32 bits" if forma == "fp32" else "registro: forma que a etapa de origem não escolheu (subproduto da função)" if forma != escolhida
                         else "conferência dos 8 bits: em dev, F1 de entidades" if v == poda else "registro: a forma escolhida, no modelo sem poda")
                compara(v, f"{forma} − pytorch", f"{v}/pytorch", f"{v}/{forma}", papel)
            dos_8 = gravada(v, f"{escolhida} − pytorch", CONFERENCIAS["quantizacao"]["metrica"])
            L_conf.append({**comum, "conferencia": "8 bits", "receita": escolhida, "variante": v, "particao": "dev", "metrica": CONFERENCIAS["quantizacao"]["metrica"], "referencia": dos_8["a"],
                           "com_a_receita": dos_8["b"], "diferenca": dos_8["diferenca"], "ic_min": dos_8["ic_min"], "ic_max": dos_8["ic_max"],
                           "criterio": "a queda é de no máximo 1 ponto", "passa": bool(dos_8["diferenca"] >= -etapa8.LIMITE_DO_PLANO), "conta_para_a_resposta": v == poda})
            pedidos = sum(m["pedidos"][p] for m in metas_q[v] for p in m["pedidos"])
            iguais = sum(m["formas"]["fp32"][f"iguais_ao_pytorch_{p}"] for m in metas_q[v] for p in m["pedidos"])
            L_conf.append({**comum, "conferencia": "ONNX de 32 bits", "receita": "exportação", "variante": v, "particao": "dev e teste", "metrica": "respostas iguais às do PyTorch",
                           "referencia": pedidos, "com_a_receita": iguais, "diferenca": iguais - pedidos, "ic_min": None, "ic_max": None,
                           "criterio": "todas as respostas iguais, nas 3 sementes", "passa": iguais == pedidos, "conta_para_a_resposta": v == poda})

        # 3. a tabela de resultados, no formato das Etapas 8 e 10c (uma linha por modelo, variante, forma e partição)
        extras = {"modelo": nome, "caminho": c, "professor": professor}
        for momento, (_, rotulo) in MOMENTOS.items():
            for p in PARTICOES:
                L_res.append(resultados.registrar_grupo(
                    f"{a['nome']}, {nome_da_variante[poda]}, {rotulo} (respostas da execução da poda), média de {len(SEMENTES)} sementes", p, gabarito[p], r[momento][p],
                    parametros=metas_p[0]["contagem"]["parametros"], bytes_em_disco=metas_q[poda][0]["formas"]["pytorch"]["bytes"], variante=poda, papel="principal", forma=momento, **extras))
        for v in variantes(c):
            quieta = bool(lat[v]["nucleos_de_outros"] <= LIMITE_DE_OUTROS)
            for forma in FORMAS_MEDIDAS:
                do_arquivo, rodadas = metas_q[v][0]["formas"][forma], lat[v]["p50_por_rodada_ms"][forma]
                for p in PARTICOES:
                    L_res.append(resultados.registrar_grupo(
                        f"{a['nome']}, {nome_da_variante[v]}, {NOME_DA_FORMA[forma]}, média de {len(SEMENTES)} sementes", p, gabarito[p], r[f"{v}/{forma}"][p],
                        tempos_ms=lat[v]["tempos"][forma].ravel(), parametros=metas_q[v][0]["parametros"], bytes_em_disco=do_arquivo["bytes"], threads=1, variante=v,
                        papel="sem_poda" if v == SEM_PODA else "principal", forma=forma, bytes_compactado=do_arquivo.get("bytes_compactado"), escolhida=forma == escolhida, final=False,
                        maquina_quieta=quieta, **extras))
                L_efi.append({
                    "modelo": nome, "variante": v, "forma": forma, "nome_da_forma": NOME_DA_FORMA[forma], "caminho": c, "parametros": metas_q[v][0]["parametros"], "bytes": do_arquivo["bytes"],
                    "bytes_da_rede": do_arquivo.get("rede"), "bytes_do_tokenizador_e_configuracao": do_arquivo.get("tokenizador_e_configuracao"), "bytes_compactado": do_arquivo.get("bytes_compactado"),
                    **lat[v]["resumo"][forma], "p50_min_das_rodadas": min(rodadas), "p50_max_das_rodadas": max(rodadas), "threads": lat[v]["threads"], "semente": lat[v]["semente"],
                    "rodadas": lat[v]["rodadas"], "consultas_por_rodada": lat[v]["consultas_por_rodada"], "medido_em": lat[v]["medido_em"], "nucleos_de_outros_processos": lat[v]["nucleos_de_outros"],
                    "maquina_quieta": quieta, "medidas_recusadas": len(list(pasta_da_quantizacao(nome, v, SEMENTE_DO_CUSTO).glob("latencia_recusada_*.json"))),
                    **{k: lat[v][k] for k in ("cpu", "cpus_logicas", "plataforma", "python", "instrucoes_da_cpu", "tem_vnni")},
                })

        # 4. cada reajuste: a época escolhida e o exact match no dev rotulado pelo professor (o que escolhe a época), antes e depois
        for s, m in zip(SEMENTES, metas_p):
            L_rea.append({**comum, "variante": poda, "semente": s, "epocas": m["receita"]["epocas"], "melhor_epoca": m["melhor_epoca"], "segundos": m["segundos"],
                          "exact_match_no_dev_do_professor_antes": m["dev_antes"]["exact_match"], "exact_match_no_dev_do_professor_depois": m["dev"]["exact_match"],
                          "parametros_de_partida": m["contagem_de_partida"]["parametros"], "parametros": m["contagem"]["parametros"], "exemplos_de_treino": m["exemplos_de_treino"],
                          "de_partida": m["de_partida"], "professor": m["origem"]["professor"], "pico_de_vram_gb": m.get("pico_de_vram_gb")})

    saida = pasta_das_contas()
    saida.mkdir(parents=True, exist_ok=True)
    for nome_do_arquivo, linhas in (("resultados.csv", L_res), ("comparacoes.csv", L_comp), ("conferencias.csv", L_conf), ("eficiencia.csv", L_efi), ("reajustes.csv", L_rea)):
        pd.DataFrame(linhas).to_csv(saida / nome_do_arquivo, index=False)
    ultima = _json(sorted((pasta_da_etapa() / "cadeias").glob("cadeia_*.json"))[-1])
    conferidas = [l for l in L_conf if l["conta_para_a_resposta"]]
    _gravar_json(saida / "contas.json", {   # por último: marca "pronto"
        "etapa": ETAPA, "calculado_em": _agora(), "regras_escritas_em": regras["escritas_em"], "professor": professor, "reamostragens": n,
        "cadeia": {k: ultima.get(k) for k in ("pid", "iniciada_em", "terminada_em", "minutos", "prontas", "erros", "codigo_igual_do_inicio_ao_fim", "conferido")},
        "conferencias": {"feitas": len(conferidas), "passam": sum(l["passa"] for l in conferidas), "nao_passam": [f"{l['aluno']}: {l['conferencia']}" for l in conferidas if not l["passa"]]},
        "linhas": {"resultados": len(L_res), "comparacoes": len(L_comp), "conferencias": len(L_conf), "eficiencia": len(L_efi), "reajustes": len(L_rea)},
    })
    return saida


def carregar_contas() -> dict:
    """O que `fechar` gravou: {nome sem extensão: tabela ou dicionário}."""
    import pandas as pd

    saida = pasta_das_contas()
    if not (saida / "contas.json").exists():
        raise FileNotFoundError("Faltam as contas da parte 1. Rode: uv run python -m especialista.experimentos.etapa11c fechar")
    return {nome.rsplit(".", 1)[0]: _json(saida / nome) if nome.endswith(".json") else pd.read_csv(saida / nome) for nome in ARQUIVOS_DAS_CONTAS}


# --------------------------------------------------------------------------
# A entrega: a decisão (que aqui só relata), os sistemas declarados e os arquivos
# --------------------------------------------------------------------------


def pastas_entregues(nome: str, professor: str = PROFESSOR) -> dict:
    """Onde ficam, na entrega, os modelos de um aluno: o podado (PyTorch), os de 8 bits (podado e sem poda) e o ONNX de 32 bits."""
    c = alunos(professor)[nome]["caminho"]
    poda, forma, raiz = receitas()[c]["poda"], receitas()[c]["forma"], f"entregas/{ETAPA}/modelos"
    return {
        "podado": {str(s): f"{raiz}/podados/{nome}/{poda}/semente_{s}" for s in SEMENTES},
        "int8": {str(s): f"{raiz}/int8/{nome}/{poda}/{forma}/semente_{s}" for s in SEMENTES},
        "int8_sem_poda": {str(s): f"{raiz}/int8/{nome}/{SEM_PODA}/{forma}/semente_{s}" for s in SEMENTES},
        "onnx_de_32_bits": {v: f"{raiz}/onnx_32_bits/{nome}/{v}/semente_{SEMENTE_DO_CUSTO}" for v in variantes(c)},
    }


def montar_decisao(professor: str = PROFESSOR) -> dict:
    """O `decisao.json` da etapa. Não há decisão nova: ele diz, para cada aluno, se cada receita passou na sua conferência, o que custou e onde estão os modelos."""
    contas, regras, R = carregar_contas(), exigir_regras(), receitas()
    conf, res, efi = contas["conferencias"], contas["resultados"], contas["eficiencia"]
    do_comparativo = sorted((pasta_da_etapa() / "comparativo").glob("regras_v*.json"), key=lambda c: _json(c)["versao"])[-1]
    por_aluno = {}
    for nome, a in alunos(professor).items():
        c = a["caminho"]
        poda, forma = R[c]["poda"], R[c]["forma"]
        ficha = lambda conferencia, variante: {k: (None if v != v else v.item() if hasattr(v, "item") else v) for k, v in conf[
            (conf["aluno"] == nome) & (conf["conferencia"] == conferencia) & (conf["variante"] == variante)].iloc[0].items()
            if k in ("particao", "metrica", "referencia", "com_a_receita", "diferenca", "ic_min", "ic_max", "criterio", "passa")}
        qualidade_de = lambda variante, f, p: {m: float(res[(res["modelo"] == nome) & (res["variante"] == variante) & (res["forma"] == f) & (res["particao"] == p)].iloc[0][m])
                                               for m in ("intent_acc", "slot_f1", "exact_match")}
        custo_de = lambda variante, f: {k: (None if v != v else float(v)) for k, v in efi[(efi["modelo"] == nome) & (efi["variante"] == variante) & (efi["forma"] == f)].iloc[0].items()
                                        if k in ("parametros", "bytes", "bytes_compactado", "latencia_p50_ms", "latencia_p95_ms")}
        linhas = {"sem_poda": (SEM_PODA, "pytorch"), "podado": (poda, "pytorch"), "onnx": (poda, "fp32"), "int8": (poda, forma), "int8_sem_poda": (SEM_PODA, forma)}
        por_aluno[nome] = {
            "nome": a["nome"], "curto": a["curto"], "caminho": c, "sigla": a["sigla"], "representa_o_cenario_b": a["representa"], "papel": a["papel"], "chave_no_registro": a["chave"],
            "de_partida": a["modelos"],
            "poda": {"variante": poda, "nome": R[c]["nome_da_poda"], "epocas_de_reajuste": R[c]["reajuste"]["epocas"], "conferencia": ficha("poda", poda)},
            "oito_bits": {"forma": forma, "nome_da_forma": R[c]["nome_da_forma"], "conferencia": ficha("8 bits", poda), "no_modelo_sem_poda_registro": ficha("8 bits", SEM_PODA),
                          "onnx_de_32_bits_igual_ao_pytorch": ficha("ONNX de 32 bits", poda)},
            "passa_em_tudo": bool(conf[(conf["aluno"] == nome) & conf["conta_para_a_resposta"]]["passa"].all()),
            "linhas": {linha: {"variante": v, "forma": f, "em_dev": qualidade_de(v, f, "dev"), "no_teste": qualidade_de(v, f, "test"), **custo_de(v, f)} for linha, (v, f) in linhas.items()},
            "pastas_por_semente": pastas_entregues(nome, professor),
        }
    return {
        "etapa": ETAPA, "gerado_em": _agora(), "pergunta": PERGUNTA,
        "como_ler": ("Esta etapa não toma decisão: `alunos` diz, para cada aluno novo, se a poda e a forma de 8 bits que as etapas de origem escolheram passam na conferência (em dev), "
                     "a qualidade, o tamanho e a latência de cada linha, e onde estão os modelos. As pastas `podado` são lidas por `SistemaDiet.carregar` (zero) ou `SistemaEncoder.carregar` (xlmr); "
                     "as `int8`, `int8_sem_poda` e `onnx_de_32_bits`, por `SistemaOnnx.carregar` (zero) ou `SistemaEncoderOnnx.carregar` (xlmr), sem PyTorch. Os modelos finais do projeto "
                     "continuam os das Etapas 8 e 10c"),
        "decisoes_nesta_etapa": "nenhuma",
        "professor": regras["professor"],
        "regras": {"arquivo": f"entregas/{ETAPA}/regras.json", "escritas_em": regras["escritas_em"]},
        "receitas": {c: {k: r[k] for k in ("poda", "nome_da_poda", "reajuste", "forma", "nome_da_forma", "de")} for c, r in R.items()},
        "todas_as_receitas_passam": bool(conf[conf["conta_para_a_resposta"]]["passa"].all()),
        "alunos": por_aluno,
        "latencia": {"protocolo": regras["custo"]["latencia"], "limite_de_outros_processos": LIMITE_DE_OUTROS, "maior_valor_medido": float(efi["nucleos_de_outros_processos"].max()),
                     "maquina_quieta": bool(efi["maquina_quieta"].all()), "medidas_recusadas": int(efi.drop_duplicates(["modelo", "variante"])["medidas_recusadas"].sum())},
        "comparativo": {"regras": f"entregas/{ETAPA}/comparativo/{do_comparativo.name}", "versao": _json(do_comparativo)["versao"], "resultados": f"entregas/{ETAPA}/comparativo/",
                        "sistemas_declarados": f"entregas/{ETAPA}/sistemas.json"},
        "modelos_finais_do_projeto": {"zero": "entregas/08_quantizacao/decisao.json", "xlmr": "entregas/10c_encoder_quantizacao/decisao.json"},
    }


def entregar(professor: str = PROFESSOR) -> Path:
    """Monta `entregas/11c_cenario_b_ate_o_fim/` a partir do cache. Nada é tirado do cache; os modelos entram por link físico. Pode ser chamada de novo."""
    destino, base = ENTREGAS / ETAPA, pasta_da_etapa()
    destino.mkdir(parents=True, exist_ok=True)
    copiar = lambda origem, alvo: (alvo.parent.mkdir(parents=True, exist_ok=True), shutil.copy2(origem, alvo))
    carregar_contas()   # sem as contas da parte 1 não há o que entregar
    for nome_do_arquivo in ARQUIVOS_DAS_CONTAS:
        copiar(pasta_das_contas() / nome_do_arquivo, destino / nome_do_arquivo)
    copiar(base / "regras.json", destino / "regras.json")
    for arquivo in sorted((base / "comparativo").glob("regras_v*.json")):
        copiar(arquivo, destino / "comparativo" / arquivo.name)
    for arquivo in sorted(a for a in (base / "comparativo" / "resultados").glob("*") if a.is_file()):
        copiar(arquivo, destino / "comparativo" / arquivo.name)
    _gravar_json(destino / "sistemas.json", montar_sistemas(professor))
    _gravar_json(destino / "decisao.json", montar_decisao(professor))
    for nome, a in alunos(professor).items():
        c = a["caminho"]
        poda, forma, pastas = receitas()[c]["poda"], receitas()[c]["forma"], pastas_entregues(nome, professor)
        for s in SEMENTES:
            da_poda = pasta_da_poda(nome, s, professor)
            copiar(da_poda / "historico.csv", destino / "historicos" / f"{nome}__{poda}_semente_{s}.csv")
            for p in PARTICOES:   # as respostas, com os nomes que `sistemas.json` declara
                copiar(da_poda / f"previsoes_{p}.parquet", destino / "previsoes" / f"{nome}__{poda}_semente_{s}_{p}.parquet")
                copiar(da_poda / f"previsoes_{p}_antes.parquet", destino / "previsoes" / f"{nome}__{poda}__antes_do_reajuste_semente_{s}_{p}.parquet")
                for v in variantes(c):
                    for f in ("pytorch", "fp32", forma):
                        copiar(pasta_da_quantizacao(nome, v, s) / f"previsoes_{f}_{p}.parquet", destino / "previsoes" / f"{nome}__{v}__{f}_semente_{s}_{p}.parquet")
            ligacoes = [(da_poda / "modelo", RAIZ / pastas["podado"][str(s)])]
            ligacoes += [(pasta_da_quantizacao(nome, v, s) / "sistemas" / forma, RAIZ / pastas[chave][str(s)]) for v, chave in ((poda, "int8"), (SEM_PODA, "int8_sem_poda"))]
            if s == SEMENTE_DO_CUSTO:
                ligacoes += [(pasta_da_quantizacao(nome, v, s) / "sistemas" / "fp32", RAIZ / pastas["onnx_de_32_bits"][v]) for v in variantes(c)]
            for origem, alvo in ligacoes:
                for arquivo in sorted(x for x in origem.rglob("*") if x.is_file()):
                    etapa11a.ligar(arquivo, alvo / arquivo.relative_to(origem))
        for v in variantes(c):   # a medida de latência como foi gravada
            copiar(pasta_da_quantizacao(nome, v, SEMENTE_DO_CUSTO) / "latencia.json", destino / "latencia" / f"{nome}__{v}.json")
    for registro_ in sorted((base / "cadeias").glob("cadeia_*.json")):
        copiar(registro_, destino / "cadeia" / registro_.name)
    for nome_do_arquivo in ("cadeia.log", "cadeia.sh"):
        if (base / nome_do_arquivo).exists():
            copiar(base / nome_do_arquivo, destino / "cadeia" / nome_do_arquivo)
    for sobra in sorted(a for a in (base / "fumaca").rglob("*") if a.is_file()):   # o teste de fumaça, as impressões digitais e a prova da 10d
        copiar(sobra, destino / "fumaca" / sobra.relative_to(base / "fumaca"))
    for nome_do_arquivo in ("src/especialista/experimentos/etapa11c.py", "src/especialista/experimentos/etapa10d.py", "notebooks/construtores/build_nb11c.py", "tests/test_etapa11c.py"):
        if (RAIZ / nome_do_arquivo).exists():
            copiar(RAIZ / nome_do_arquivo, destino / "codigo" / Path(nome_do_arquivo).name)
    return destino


# --------------------------------------------------------------------------
# O comparativo: as contas, pelas funções da Etapa 10d, com o registro e a lista desta etapa
# --------------------------------------------------------------------------

ARQUIVOS_DO_COMPARATIVO = ("tabela_final.csv", "conferencia.csv", "cadeias.csv", "comparacoes.csv", "fronteira.csv", "precos.csv", "calculo.json")


def carregar_regras_do_comparativo() -> tuple[dict, Path]:
    """A versão mais nova das regras do comparativo gravadas no cache desta etapa, e o arquivo."""
    existentes = sorted((pasta_da_etapa() / "comparativo").glob("regras_v*.json"), key=lambda c: _json(c)["versao"])
    if not existentes:
        raise FileNotFoundError("Grave as regras do comparativo antes de calcular: ... etapa11c regras-do-comparativo")
    return _json(existentes[-1]), existentes[-1]


def gravadas_na_etapa_11(fichas: dict[str, dict], professor: str = PROFESSOR) -> list[tuple[str, str, str, dict, int]]:
    """As comparações deste comparativo que as Etapas 11a e 11b já tinham gravado: (B, A, etapa, filtro no `comparacoes.csv` dela, sinal).

    Sinal −1: lá a diferença está escrita ao contrário (aluno B − aluno A).
    """
    saida = []
    for c in ORIGEM:
        P = etapa10d.CAMINHOS[c]["prefixo"]
        for sigla in ("B1", "B2"):
            novo, antigo, do_a = f"{P}aluno_{sigla.lower()}@{professor}", f"{P}aluno_{sigla.lower()}", f"{P}aluno_a"
            saida += [
                (novo, antigo, etapa11b.ETAPA, {"id": f"novo_menos_antigo|{c}:{sigla}:novo|{c}:{sigla}:antigo"}, 1),
                (novo, professor, etapa11b.ETAPA, {"id": f"aluno_menos_professor|{c}:{sigla}:novo|prof:novo"}, 1),
                (do_a, novo, etapa11b.ETAPA, {"id": f"b_menos_a|{c}:{sigla}:novo|{c}:A"}, -1),
                (do_a, antigo, etapa11b.ETAPA, {"id": f"b_menos_a|{c}:{sigla}:antigo|{c}:A"}, -1),
            ]
    na_11a = lambda chave: etapa11a.VERSAO_DA_5B if chave == "qwen" else chave
    professores = [c for c, f in fichas.items() if f["ponto"] == "professor"]
    return saida + [(b, a, etapa11a.ETAPA, {"b": na_11a(b), "a": na_11a(a)}, 1) for b in professores for a in professores if b != a]


def conferir_com_a_etapa_11(comparacoes, fichas: dict[str, dict], professor: str = PROFESSOR, entregas: Path | None = None):
    """Cada comparação que as Etapas 11a e 11b já tinham gravado, refeita aqui das respostas, contra o número de lá (com o sinal trocado onde lá está ao contrário)."""
    import numpy as np
    import pandas as pd

    from especialista.avaliacao import qualidade

    ler = etapa9._Tabelas(entregas or ENTREGAS)
    daqui = comparacoes.set_index(["chave_b", "chave_a", "particao", "metrica"])
    linhas = []
    for b, a, etapa, filtro, sinal in gravadas_na_etapa_11(fichas, professor):
        for p in PARTICOES:
            if (b, a, p, "exact_match") not in daqui.index:
                continue
            gravado = etapa10d._gravada(ler, etapa, "comparacoes.csv", {**filtro, "particao": p})
            linha = {"o_que": "comparação entre sistemas", "chave": b, "anterior": a, "particao": p, "gravado_em": f"entregas/{etapa}/comparacoes.csv"}
            if gravado is None:
                linhas.append({**linha, "gravado": False, "maior_diferenca": np.nan, "confere": None})
                continue
            diferencas = []
            for m in qualidade.METRICAS:
                la = gravado.loc[m]
                esperado = (la["diferenca"], la["ic_min"], la["ic_max"]) if sinal > 0 else (-la["diferenca"], -la["ic_max"], -la["ic_min"])
                diferencas += [abs(daqui.loc[(b, a, p, m), coluna] - valor) for coluna, valor in zip(("diferenca", "ic_min", "ic_max"), esperado)]
            linhas.append({**linha, "gravado": True, "maior_diferenca": float(max(diferencas)), "confere": bool(max(diferencas) <= etapa10d.TOLERANCIA)})
    return pd.DataFrame(linhas)


def conferir_com_a_10d(pasta_do_calculo: Path, entregas: Path | None = None):
    """O que a Etapa 10d entregou, linha a linha, contra o mesmo aqui: as 44 linhas da tabela final e cada uma das comparações dela têm de sair iguais."""
    import numpy as np
    import pandas as pd

    da_10d = (entregas or ENTREGAS) / etapa10d.ETAPA
    antiga, nova = (pd.read_csv(pasta / "tabela_final.csv").set_index("chave") for pasta in (da_10d, pasta_do_calculo))
    linhas = []
    for chave, la in antiga.iterrows():
        aqui = nova.loc[chave] if chave in nova.index else None
        diferentes = list(antiga.columns) if aqui is None else [c for c in antiga.columns if not (pd.isna(la[c]) and pd.isna(aqui[c])) and not etapa10d._igual(la[c], aqui[c])]
        linhas.append({"o_que": "linha da tabela final da Etapa 10d", "chave": chave, "particao": "dev e test", "gravado_em": f"entregas/{etapa10d.ETAPA}/tabela_final.csv", "gravado": True,
                       "maior_diferenca": 0.0 if not diferentes else np.nan, "confere": not diferentes, "colunas_diferentes": ", ".join(diferentes)})
    chaves = ["chave_b", "chave_a", "particao", "metrica"]
    antigas, novas = (pd.read_csv(pasta / "comparacoes.csv").set_index(chaves) for pasta in (da_10d, pasta_do_calculo))
    for (b, a, p), grupo in antigas.groupby(level=[0, 1, 2], sort=False):
        linha = {"o_que": "comparação da Etapa 10d", "chave": b, "anterior": a, "particao": p, "gravado_em": f"entregas/{etapa10d.ETAPA}/comparacoes.csv", "gravado": True}
        if not all(indice in novas.index for indice in grupo.index):
            linhas.append({**linha, "maior_diferenca": np.nan, "confere": False})
            continue
        maior = float(max(abs(novas.loc[indice, coluna] - valores[coluna]) for indice, valores in grupo.iterrows() for coluna in ("diferenca", "ic_min", "ic_max")))
        linhas.append({**linha, "maior_diferenca": maior, "confere": bool(maior <= etapa10d.TOLERANCIA)})
    return pd.DataFrame(linhas)


def calcular(professor: str = PROFESSOR) -> Path:
    """O comparativo: a tabela final, a conferência, as cadeias, as comparações, a fronteira e o preço de cada ponto, gravados em `comparativo/resultados/`.

    As contas são as da Etapa 10d, pelas funções dela; o que muda é o
    registro (com as entregas da Etapa 11) e a lista de comparações (com as
    famílias novas). Não roda sem as regras gravadas, nem se o registro ou a
    lista de hoje não forem os das regras: nesse caso grava-se antes a versão
    seguinte (`regras-do-comparativo`).
    """
    import pandas as pd

    from especialista.avaliacao.previsao import ouro
    from especialista.dados import massive

    entregas = ENTREGAS
    if not (entregas / ETAPA / "sistemas.json").exists() or not (entregas / ETAPA / "resultados.csv").exists():
        raise FileNotFoundError("A parte 1 ainda não está na entrega (é dela que o registro lê os sistemas novos). Rode: ... etapa11c fechar e ... etapa11c entregar")
    regras, arquivo_das_regras = carregar_regras_do_comparativo()
    fichas = registro(entregas, professor=professor)
    if ausentes := etapa10d.faltam(fichas):
        raise FileNotFoundError(f"Faltam {len(ausentes)} arquivo(s) de respostas (o primeiro: {ausentes[0]}).")
    lista, cadeias = lista_de_comparacoes(fichas, entregas), cadeias_do_registro(fichas, entregas, professor)
    da_regra = [(c["id"], c["b"], c["a"], c["familias"]) for c in regras["comparacoes"]]
    if da_regra != [(c["id"], c["b"], c["a"], c["familias"]) for c in lista] or list(regras["cadeias"]) != list(cadeias) or regras["sistemas_no_registro"] != list(fichas):
        raise RuntimeError("O registro ou a lista de comparações mudou depois das regras gravadas. Grave uma versão nova antes de calcular: ... etapa11c regras-do-comparativo")

    particoes, _ = massive.carregar_entrega(entregas / "01_dados")
    gabaritos = {p: ouro(particoes[p]) for p in PARTICOES}
    grupos = {p: etapa10d.respostas(fichas, p, list(particoes[p]["id"])) for p in PARTICOES}
    T = etapa10d.tabela_final(fichas, entregas)
    tabelas = {"test": T, "dev": etapa10d.em_dev(T)}

    confere = [etapa10d.conferir_qualidade(fichas, gabaritos, grupos, entregas)]
    CAD, PP = [], []
    for p in PARTICOES:
        cad = etapa9.cadeias(tabelas[p], gabaritos[p], grupos[p], cadeias)
        cad.insert(1, "particao", p)
        CAD.append(cad)
        confere.append(etapa10d.conferir_degraus(fichas, cad, p, entregas))
        PP.append(etapa10d.comparar_lista(lista, tabelas[p], gabaritos[p], grupos[p], p))
    CAD, PP = pd.concat(CAD, ignore_index=True), pd.concat(PP, ignore_index=True)
    confere += [etapa10d.conferir_comparacoes(lista, PP, entregas), conferir_com_a_etapa_11(PP, fichas, professor, entregas)]

    folga = etapa10d.variacao_da_latencia(entregas)
    F = etapa10d.fronteiras(T, folga).assign(latencia_usada="etapas de origem")
    PR = etapa10d.precos(T, PP, fichas, None)

    saida = pasta_da_etapa() / "comparativo" / "resultados"
    saida.mkdir(parents=True, exist_ok=True)
    T.reset_index().to_csv(saida / "tabela_final.csv", index=False)
    CAD.to_csv(saida / "cadeias.csv", index=False)
    PP.to_csv(saida / "comparacoes.csv", index=False)
    F.to_csv(saida / "fronteira.csv", index=False)
    PR.to_csv(saida / "precos.csv", index=False)
    confere.append(conferir_com_a_10d(saida, entregas))   # depois de gravar: as duas tabelas são comparadas como estão em disco
    pd.concat(confere, ignore_index=True).to_csv(saida / "conferencia.csv", index=False)
    entradas = sorted({c for f in fichas.values() for lista_ in f["previsoes"].values() for c in lista_})
    _gravar_json(saida / "calculo.json", {   # por último: marca "pronto"
        "etapa": ETAPA, "calculado_em": datetime.now().isoformat(timespec="seconds"),
        "regras": {"arquivo": arquivo_das_regras.name, "versao": regras["versao"], "gravado_em": regras["gravado_em"], "sha256": etapa10d.sha256(arquivo_das_regras)},
        "sistemas": len(fichas), "comparacoes": len(lista), "cadeias": list(cadeias), "folga_da_latencia": folga, "entregas_que_valem": list(ENTREGAS_QUE_VALEM),
        "primarias": regras["metodo"]["primarias"]["pares"],
        "respostas_lidas": {"arquivos": len(entradas), "sha256": {etapa9._relativo(c, entregas.parent): etapa10d.sha256(c) for c in entradas}},
    })
    return saida


def carregar_calculo() -> dict:
    """O que `calcular` gravou: {nome sem extensão: tabela ou dicionário}."""
    import pandas as pd

    saida = pasta_da_etapa() / "comparativo" / "resultados"
    if not (saida / "calculo.json").exists():
        raise FileNotFoundError("Falta o cálculo do comparativo. Rode: uv run python -m especialista.experimentos.etapa11c calcular")
    lido = {nome.rsplit(".", 1)[0]: _json(saida / nome) if nome.endswith(".json") else pd.read_csv(saida / nome) for nome in ARQUIVOS_DO_COMPARATIVO}
    lido["tabela_final"] = lido["tabela_final"].set_index("chave")
    lido["fronteira"]["dominado_por"] = lido["fronteira"]["dominado_por"].fillna("")
    return lido


def manifesto(professor: str = PROFESSOR) -> Path:
    """O que a entrega contém (com a impressão digital de cada arquivo), do que ela foi feita, e que código mudou desde a cadeia."""
    destino, regras = ENTREGAS / ETAPA, exigir_regras()
    ultima = _json(sorted((pasta_da_etapa() / "cadeias").glob("cadeia_*.json"))[-1])
    calculo = _json(pasta_da_etapa() / "comparativo" / "resultados" / "calculo.json")
    entradas = [RAIZ / d for d in regras["decisoes_lidas"]]
    entradas += [ENTREGAS / n for n in (f"{etapa11a.ETAPA}/sistemas.json", f"{etapa11a.ETAPA}/resultados.csv", f"{etapa11a.ETAPA}/comparacoes.csv", f"{etapa11b.ETAPA}/sistemas.json",
                                        f"{etapa11b.ETAPA}/resultados.csv", f"{etapa11b.ETAPA}/comparacoes.csv", f"{etapa10d.ETAPA}/tabela_final.csv", f"{etapa10d.ETAPA}/comparacoes.csv",
                                        f"{etapa10d.ETAPA}/regras_v2.json")]
    entradas += [arquivo for a in alunos(professor).values() for pasta in a["pastas"].values() for arquivo in sorted(x for x in pasta.rglob("*") if x.is_file())]
    hoje = etapa11b.impressao_do_codigo()
    alterados = sorted(r for r in set(ultima["codigo"]) | set(hoje) if ultima["codigo"].get(r) != hoje.get(r))
    arquivos = sorted(a for a in destino.rglob("*") if a.is_file() and a.name != "manifesto.json")
    arquivo = destino / "manifesto.json"
    _gravar_json(arquivo, {
        "etapa": ETAPA, "gerado_em": _agora(),
        "o_que_e": "O que a Etapa 11c leu (com a impressão digital de cada arquivo), as regras, a cadeia de execuções, o cálculo do comparativo e o que ela entrega.",
        "professor": professor,
        "regras": {"arquivo": f"entregas/{ETAPA}/regras.json", "escritas_em": regras["escritas_em"], "do_comparativo": calculo["regras"]},
        "cadeia": {k: ultima.get(k) for k in ("pid", "iniciada_em", "terminada_em", "minutos", "prontas", "erros", "codigo_igual_do_inicio_ao_fim", "codigo_alterado_durante_a_cadeia",
                                              "ambiente", "copia_do_codigo")},
        "codigo_da_cadeia": ultima["codigo"],
        "codigo_alterado_depois_da_cadeia": {"arquivos": alterados, "como_ler": "arquivos do pacote cuja impressão digital de hoje não é a do começo da cadeia. As execuções da cadeia rodaram com o "
                                             "código da cópia; as contas e o comparativo, com o de hoje (copiado em codigo/)"},
        "entradas": {_relativo(a): (etapa11a.sha256_do_arquivo(a) if a.stat().st_size <= 64 * 2**20 else f"{a.stat().st_size} bytes (acima de 64 MB: sem sha256)") for a in entradas},
        "respostas_lidas_pelo_comparativo": calculo["respostas_lidas"]["arquivos"],
        "saidas": {str(a.relative_to(destino)): {"sha256": etapa11a.sha256_do_arquivo(a) if a.stat().st_size <= 64 * 2**20 else None, "bytes": a.stat().st_size, "link_fisico": a.stat().st_nlink > 1}
                   for a in arquivos},
        "cache": f"data/execucoes/{ETAPA}/ (as 33 execuções, a forma estática e as de diagnóstico, as contas e a cópia do código da cadeia; os modelos da entrega são links físicos para lá)",
    })
    return arquivo


# --------------------------------------------------------------------------


def _linha_de_comando() -> None:
    argumentos = argparse.ArgumentParser(description="Etapa 11c: o cenário B com o professor novo até o fim do caminho, e o comparativo atualizado.")
    comandos = argumentos.add_subparsers(dest="comando", required=True)

    def comando(nome: str, ajuda: str) -> argparse.ArgumentParser:
        novo = comandos.add_parser(nome, help=ajuda)
        novo.add_argument("--professor", default=PROFESSOR, help=f"a versão do professor, como a Etapa 11a a nomeia (padrão: {PROFESSOR})")
        return novo

    comando("regras", "grava as regras, com a hora (antes de qualquer execução), e a versão das regras do comparativo")
    comando("regras-do-comparativo", "grava uma versão nova das regras do comparativo, se o registro cresceu")
    uma = comando("executar", "um passo de um aluno com uma semente (é o que a cadeia chama)")
    uma.add_argument("--passo", required=True, choices=PASSOS)
    uma.add_argument("--aluno", required=True, help="a variante da Etapa 11b")
    uma.add_argument("--variante", required=True, help="a poda principal do caminho ou sem_poda")
    uma.add_argument("--semente", type=int, required=True)
    uma.add_argument("--fumaca", type=Path, help="pasta descartável: roda a versão curta do teste de fumaça")
    da_cadeia = comando("cadeia", "roda tudo o que falta: poda e reajuste, exportação e quantização, latência por último")
    da_cadeia.add_argument("--fumaca", type=Path, help="pasta descartável: roda só o teste de fumaça")
    comando("faltam", "lista o que ainda não rodou, uma execução por linha")
    comando("situacao", "quantas execuções estão prontas")
    comando("fechar", "depois da cadeia: as contas da parte 1 (conferências, resultados, tamanhos e latências)")
    comando("entregar", "monta a entrega a partir do cache (o notebook também chama)")
    comando("calcular", "o comparativo: tabela final, cadeias, comparações e fronteira (precisa da parte 1 entregue e das regras gravadas)")
    comando("manifesto", "depois do notebook e do HTML: o que a entrega contém, com impressão digital")
    a = argumentos.parse_args()

    if a.comando == "regras":
        destino = gravar_regras(a.professor)
        regras = _json(destino)
        print(destino, "escritas em", regras["escritas_em"], "|", regras["execucoes"]["total"], "execuções |", "comparativo:", regras["comparativo"]["regras"],
              f"({regras['comparativo']['sistemas']} sistemas, {regras['comparativo']['comparacoes']} comparações, {len(regras['comparativo']['cadeias'])} cadeias)")
    elif a.comando == "regras-do-comparativo":
        destino = gravar_regras_do_comparativo(a.professor)
        regras = _json(destino)
        print(f"{destino}: versão {regras['versao']}, gravada em {regras['gravado_em']}; {len(regras['comparacoes'])} comparações, {len(regras['cadeias'])} cadeias, {len(regras['sistemas_no_registro'])} sistemas.")
    elif a.comando == "executar":
        if a.fumaca is not None:
            etapa11b.usar_pasta_descartavel(a.fumaca)
        agora = lambda: f"[{datetime.now():%d/%m %H:%M:%S}]"
        if a.passo == "poda":
            import torch
            torch.set_num_threads(2)
            meta = _json(podar(a.aluno, a.semente, professor=a.professor) / "meta.json")
            antes = "" if meta["dev_antes"] is None else f"exact match no dev do professor antes {meta['dev_antes']['exact_match']:.4f}, "
            print(agora(), "poda", a.aluno, meta["variante"], a.semente, f"época {meta['melhor_epoca']} de {meta['receita']['epocas']};", f"{antes}depois {meta['dev']['exact_match']:.4f};",
                  f"{meta['segundos']:.0f}s", flush=True)
        elif a.passo == "quantizar":
            meta = _json(quantizar(a.aluno, a.variante, a.semente, professor=a.professor) / "meta.json")
            f, forma = meta["formas"], receitas()[meta["caminho"]]["forma"]
            print(agora(), "quantizar", a.aluno, a.variante, a.semente, f"ONNX igual ao PyTorch em {f['fp32']['iguais_ao_pytorch_dev']} de {meta['pedidos']['dev']} pedidos de dev;",
                  f"{forma}: {f[forma]['bytes'] / 1e6:.2f} MB, {f[forma]['iguais_ao_pytorch_dev']} iguais ({f['pytorch']['bytes'] / 1e6:.2f} MB em 32 bits); {meta['segundos']:.0f}s", flush=True)
        else:
            pasta, quieta = medir_latencia(a.aluno, a.variante, a.professor)
            lido = _json(pasta / "latencia.json")
            print(agora(), "latencia", a.aluno, a.variante, "p50 (ms):", ", ".join(f"{forma} {r['latencia_p50_ms']:.3f}" for forma, r in lido["resumo"].items()),
                  f"| outros processos: {lido['nucleos_de_outros']:.2f} núcleo(s) de CPU | quieta: {quieta}", flush=True)
    elif a.comando == "cadeia":
        sys.exit(0 if cadeia(a.professor, a.fumaca) else 1)
    elif a.comando == "faltam":
        for passo, nome, variante, semente in faltam(a.professor):
            print(f"--passo {passo} --aluno {nome} --variante {variante} --semente {semente}")
    elif a.comando == "situacao":
        print(situacao(a.professor))
    elif a.comando == "fechar":
        saida = fechar(a.professor)
        contas = _json(saida / "contas.json")
        print(saida, "|", contas["linhas"], "| conferências:", f"{contas['conferencias']['passam']} de {contas['conferencias']['feitas']} passam", contas["conferencias"]["nao_passam"] or "")
    elif a.comando == "entregar":
        print(entregar(a.professor))
    elif a.comando == "calcular":
        import pandas as pd

        saida = calcular(a.professor)
        confere = pd.read_csv(saida / "conferencia.csv")
        print(saida, "| conferência:", confere.groupby("o_que")["confere"].agg(lambda c: f"{int((c == True).sum())} de {int(c.notna().sum())}").to_dict())   # noqa: E712
    elif a.comando == "manifesto":
        print(manifesto(a.professor))


if __name__ == "__main__":
    _linha_de_comando()
