"""Prova de que a Etapa 10d reproduz o que entregou, com as entregas da Etapa 11 em disco: `calcular` numa pasta descartável.

A pasta descartável recebe só o que a 10d tinha no cache antes das contas (as regras, as respostas que ela gerou e a
medida de latência na mesma sessão), copiado da ENTREGA dela. As contas são refeitas com o código de hoje, lendo as
entregas de hoje, e cada tabela é comparada com a entregue, byte a byte e, se preciso, número a número.
"""
import hashlib, json, shutil, sys, time
from pathlib import Path

import pandas as pd

from especialista.caminhos import ENTREGAS
from especialista.experimentos import etapa10d

base = Path(sys.argv[1]) / etapa10d.ETAPA      # o nome da pasta entra numa coluna de conferencia.csv
entrega = ENTREGAS / etapa10d.ETAPA
if base.exists():
    shutil.rmtree(base)
base.mkdir(parents=True)
for nome in ("regras.json", "regras_v2.json"):
    shutil.copy2(entrega / nome, base / nome)
shutil.copytree(entrega / "previsoes", base / "previsoes")
shutil.copytree(entrega / "latencia", base / "latencia")
sha = lambda c: hashlib.sha256(c.read_bytes()).hexdigest()
inicio = time.time()
fichas = etapa10d.registro(base=base)
regras_de_hoje = json.loads(json.dumps(etapa10d.montar_regras(fichas)))
entregues = json.loads((entrega / "regras_v2.json").read_text(encoding="utf-8"))
saida = etapa10d.calcular(base)
linhas = []
for nome in etapa10d.ARQUIVOS:
    if nome.endswith(".json"):
        continue
    novo, antigo = saida / nome, entrega / nome
    identico = sha(novo) == sha(antigo)
    maior = 0.0
    if not identico:
        a, b = pd.read_csv(antigo), pd.read_csv(novo)
        mesma_forma = a.shape == b.shape and list(a.columns) == list(b.columns)
        numeros = a.select_dtypes("number").columns if mesma_forma else []
        maior = float((a[numeros] - b[numeros]).abs().max().max()) if mesma_forma and len(numeros) else float("nan")
    linhas.append({"arquivo": nome, "linhas": len(pd.read_csv(novo)) if novo.stat().st_size > 1 else 0, "identico_byte_a_byte": identico, "maior_diferenca_numerica": maior})
calculo, entregue = json.loads((saida / "calculo.json").read_text(encoding="utf-8")), json.loads((entrega / "calculo.json").read_text(encoding="utf-8"))
relato = {
    "feito_em": time.strftime("%Y-%m-%d %H:%M:%S %z"), "pasta_descartavel": str(base), "segundos": round(time.time() - inicio),
    "sistemas_json_em_disco": sorted(str(c.relative_to(ENTREGAS.parent)) for c in ENTREGAS.glob("*/sistemas.json")),
    "registro": {"sistemas": len(fichas), "entregue": entregue["sistemas"], "mesmas_chaves_na_mesma_ordem": list(fichas) == entregues["sistemas_no_registro"]},
    "regras_iguais_as_entregues": {k: v for k, v in entregues.items() if k not in ("versao", "gravado_em")} == regras_de_hoje,
    "comparacoes": {"hoje": calculo["comparacoes"], "entregue": entregue["comparacoes"]}, "cadeias_iguais": calculo["cadeias"] == entregue["cadeias"],
    "respostas_lidas_iguais": calculo["respostas_lidas"]["arquivos"] == entregue["respostas_lidas"]["arquivos"] and
        {Path(k).name: v for k, v in calculo["respostas_lidas"]["sha256"].items()} == {Path(k).name: v for k, v in entregue["respostas_lidas"]["sha256"].items()},
    "tabelas": linhas, "tudo_identico": all(l["identico_byte_a_byte"] for l in linhas),
    "sem_a_lista_o_registro_teria": len(etapa10d.registro(base=base, valem=None)),
}
(base.parent / "prova.json").write_text(json.dumps(relato, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(relato, ensure_ascii=False, indent=1))
