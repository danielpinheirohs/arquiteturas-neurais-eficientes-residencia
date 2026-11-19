#!/usr/bin/env bash
# Etapa 10b: a medida oficial de latência em CPU (1 thread), um grupo depois do outro, nunca em paralelo.
# Cada grupo inclui sempre o modelo de partida (sem poda) como referência. Precisa da máquina quieta.
cd /home/danie/project/residencia
B=data/execucoes/10b_encoder_poda
echo $$ > $B/latencia.pid
export CUDA_VISIBLE_DEVICES= TQDM_DISABLE=1 PYTHONUNBUFFERED=1
erros=0
medir() {   # modelo grupo variantes... [--esparsa]
  echo "[$(date +%H:%M:%S)] grupo $1/$2: ${*:3} (carga da máquina: $(cut -d' ' -f1-3 /proc/loadavg))"
  .venv/bin/python -m especialista.experimentos.etapa10b latencia --modelo "$1" --grupo "$2" --variantes "${@:3}" || { erros=$((erros+1)); echo "[$(date +%H:%M:%S)] ERRO no grupo $1/$2"; }
}
echo "[$(date +%H:%M:%S)] INÍCIO LATENCIA (processo $$)"
medir aluno_a pecas vocabulario_treino vocabulario_alfabeto camadas_8 camadas_6 vocabulario_treino_camadas_8 vocabulario_treino_camadas_6
medir aluno_a gradual gradual_30 gradual_50 gradual_70 gradual_90
medir aluno_a magnitude magnitude_30 magnitude_50 magnitude_70 magnitude_90
medir aluno_a esparsa magnitude_30 magnitude_50 magnitude_70 magnitude_90 gradual_70 --esparsa
medir aluno_b2_t2 seguem vocabulario_treino gradual_70
medir teto seguem vocabulario_treino magnitude_50
echo "[$(date +%H:%M:%S)] grupos gravados: $(ls $B/latencia/*.json 2>/dev/null | wc -l) de 6"
if [ "$erros" -eq 0 ]; then echo "FIM LATENCIA"; else echo "FIM COM ERROS"; fi
