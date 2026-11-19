#!/usr/bin/env bash
# Etapa 10c, fase 2: a cadeia oficial, desacoplada de qualquer agente.
#
#   (a) quantização: três processos em paralelo, um por modelo de partida (aluno_a, teto, aluno_b2_t2), 4 threads cada.
#       Em cada um, primeiro a variante principal da semente 0 (o caminho novo: vocabulário cortado, com o tokenizador
#       reduzido da pasta), depois tudo o que falta. No fim, uma repescagem em sequência do que tiver falhado.
#   (b) só depois de (a) terminar: a latência OFICIAL, sozinha na máquina, um modelo por vez, 1 thread, 5 rodadas de 1.024 consultas.
#
# Log único (a saída deste script). Última linha: "FIM: ..." ou "FIM COM ERROS: ...". O PID fica em cadeia.pid.
# Tudo na CPU; nada é instalado (uv run --no-sync). O que já está gravado não é refeito.

cd /home/danie/project/residencia || { echo "FIM COM ERROS: não achei a pasta do projeto"; exit 1; }
BASE=data/execucoes/10c_encoder_quantizacao
echo $$ > "$BASE/cadeia.pid"
export CUDA_VISIBLE_DEVICES=
ETAPA="uv run --no-sync python -m especialista.experimentos.etapa10c"
MODELOS="aluno_a teto aluno_b2_t2"
PRINCIPAL=vocabulario_treino   # a variante principal dos três modelos em entregas/10b_encoder_poda/decisao.json
THREADS=4

agora() { date '+%d/%m %H:%M:%S'; }

rodar() {   # rodar <rótulo> <comando...>: roda o comando, põe o rótulo em cada linha e devolve o código de saída do comando
  local rotulo=$1; shift
  "$@" 2>&1 | grep --line-buffered -v 'Loading weights\|Writing model shards' | sed -u "s/^/[$rotulo] /"
  return "${PIPESTATUS[0]}"
}

quantizar_modelo() {
  local m=$1 erros=0
  OMP_NUM_THREADS=$THREADS rodar "$m" $ETAPA quantizar --modelo "$m" --variante "$PRINCIPAL" --semente 0 --threads $THREADS || erros=$((erros + 1))
  OMP_NUM_THREADS=$THREADS rodar "$m" $ETAPA quantizar --tudo --modelo "$m" --threads $THREADS || erros=$((erros + 1))
  echo "[$(agora)] [$m] quantização do modelo terminou; comandos com erro: $erros"
  echo "$erros" > "$BASE/.cadeia_erros_$m"
}

echo "[$(agora)] INÍCIO da cadeia da Etapa 10c (PID $$). Carga da máquina: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "[$(agora)] (a) quantização: um processo por modelo ($MODELOS), $THREADS threads cada"
for m in $MODELOS; do
  rm -f "$BASE/.cadeia_erros_$m"
  quantizar_modelo "$m" &
done
wait
ERROS_A=0
for m in $MODELOS; do
  ERROS_A=$((ERROS_A + $(cat "$BASE/.cadeia_erros_$m" 2>/dev/null || echo 1)))
  rm -f "$BASE/.cadeia_erros_$m"
done
echo "[$(agora)] (a) os três processos terminaram; comandos com erro: $ERROS_A. Repescagem em sequência do que faltar:"
OMP_NUM_THREADS=$THREADS rodar repescagem $ETAPA quantizar --tudo --threads $THREADS
ERROS_REPESCAGEM=$?
echo "[$(agora)] (a) terminou. O que ainda falta (vazio = nada):"
rodar faltam $ETAPA faltam

sleep 60   # a máquina assenta antes da medida
echo "[$(agora)] (b) latência oficial, sozinha. Carga da máquina: $(cut -d' ' -f1-3 /proc/loadavg)"
OMP_NUM_THREADS=1 rodar latencia $ETAPA latencia
ERROS_B=$?
echo "[$(agora)] (b) terminou; código de saída: $ERROS_B"

ERROS_A=$ERROS_A ERROS_REPESCAGEM=$ERROS_REPESCAGEM ERROS_B=$ERROS_B uv run --no-sync python - <<'EOF' 2>&1 | grep -v 'Loading weights'
import json
import os
from datetime import datetime

from especialista.experimentos import etapa10c

try:
    fichas = etapa10c.modelos_a_quantizar()
    grade, falta = etapa10c.grade(), etapa10c.faltam()
    metas = [json.loads((etapa10c.pasta_da_execucao(m, v, s) / "meta.json").read_text(encoding="utf-8")) for m, v, s in grade if (m, v, s) not in falta]
    pedidos = sum(meta["pedidos"][p] for meta in metas for p in ("dev", "test"))
    iguais = sum(meta["formas"]["fp32"][f"iguais_ao_pytorch_{p}"] for meta in metas for p in ("dev", "test"))
    zeros = all(meta["formas"][forma]["conservados"] for meta in metas for forma in etapa10c.FORMAS)
    compactado = all(meta["formas"][forma]["compactado_responde_igual"] for meta in metas for forma in etapa10c.CANDIDATAS)
    latencias = {chave: etapa10c.carregar_latencia(*chave) for chave in fichas}
    medidas = [l for l in latencias.values() if l is not None]
    completas = [l for l in medidas if all(len(l["p50_por_rodada_ms"][forma]) == etapa10c.RODADAS for forma in etapa10c.COM_LATENCIA)]
    outros = max((l["nucleos_de_outros"] for l in medidas), default=float("nan"))
    comandos = {nome: int(os.environ[nome]) for nome in ("ERROS_A", "ERROS_REPESCAGEM", "ERROS_B")}
    problemas = []
    if falta:
        problemas.append(f"faltam {len(falta)} execuções: " + ", ".join(f"{m}/{v}/{s}" for m, v, s in falta))
    if len(completas) < len(fichas):
        problemas.append("falta latência de " + ", ".join(f"{m}/{v}" for (m, v), l in latencias.items() if l is None or l not in completas))
    if iguais != pedidos:
        problemas.append(f"o ONNX de 32 bits difere do PyTorch em {pedidos - iguais} respostas")
    if not zeros:
        problemas.append("há zeros da poda perdidos")
    if not compactado:
        problemas.append("há sistema compactado que responde diferente")
    if comandos["ERROS_REPESCAGEM"] or comandos["ERROS_B"]:
        problemas.append(f"códigos de saída: {comandos}")
    resumo = (f"{len(metas)} de {len(grade)} execuções gravadas; ONNX de 32 bits igual ao PyTorch em {iguais} de {pedidos} respostas; "
              f"zeros conservados: {'sim' if zeros else 'NÃO'}; latência medida em {len(completas)} de {len(fichas)} modelos; "
              f"outros processos durante a latência: no máximo {outros:.2f} núcleo de CPU; comandos com erro na parte (a), antes da repescagem: {comandos['ERROS_A']}; "
              f"terminou em {datetime.now():%d/%m %H:%M}")
    print(("FIM COM ERROS: " + "; ".join(problemas) + ". " + resumo) if problemas else ("FIM: " + resumo))
except Exception as erro:   # a última linha do log tem de dizer como acabou
    print(f"FIM COM ERROS: o resumo final falhou ({type(erro).__name__}: {erro})")
EOF
