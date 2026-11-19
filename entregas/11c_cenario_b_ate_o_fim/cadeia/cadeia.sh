#!/usr/bin/env bash
# Etapa 11c, parte 1: a cadeia inteira, desacoplada de qualquer agente.
#
#   poda e reajuste (rede do zero, 3 processos; depois XLM-R, um por vez)
#   -> exportação para ONNX e quantização (na CPU; XLM-R com 3 processos de 4 threads)
#   -> latência por último, um modelo por vez, 1 thread, 5 rodadas de 1.024 consultas
#
# Quem faz a ordem é `etapa11c.cadeia`; cada execução é um processo próprio. O que já está gravado não é refeito:
# pode ser chamada de novo. Log único (a saída deste script). Última linha: "FIM: ..." ou "FIM COM ERROS: ...".
# O PID fica em cadeia.pid. Nada é instalado (uv run --no-sync).
cd /home/danie/project/residencia || { echo "FIM COM ERROS: não achei a pasta do projeto"; exit 1; }
BASE=data/execucoes/11c_cenario_b_ate_o_fim
echo $$ > "$BASE/cadeia.pid"
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error
exec uv run --no-sync python -m especialista.experimentos.etapa11c cadeia
