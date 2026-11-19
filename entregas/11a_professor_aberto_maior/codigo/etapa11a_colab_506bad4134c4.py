"""Etapa 11a, o condutor: leva uma versão de professor ao Colab e traz as respostas.

Roda na máquina local, como um processo só e leve (ele espera; quem trabalha
é a placa alugada):

    PYTHONPATH=src .venv/bin/python -m especialista.experimentos.etapa11a_colab \
        --versao qwen3_8b__fp16__colab_l4 --gpu L4 --sessao prof11a_l4 --tarefas precisao,dev,test,train

O que ele garante:

- **Pacote por lista explícita.** Só vai ao Colab o que está em
  `ARQUIVOS_DO_PACOTE`. O `.tar.gz` enviado fica guardado, com o sha256.
- **O ambiente do projeto.** Lá se instala o `uv.lock` do projeto
  (`uv sync --frozen`), num Python 3.12 à parte; o que ficou instalado é gravado.
- **Retomável.** As respostas são baixadas a cada poucos minutos. Se a VM
  cair, o condutor cria outra, devolve as parciais e o trabalho continua do
  último lote inteiro.
- **A VM se desliga.** O condutor para a sessão no fim e na falha. Se o
  condutor morrer, a própria VM se desliga: a célula que roda o trabalho
  espera um prazo e pede o desligamento, e um vigia solto na VM faz o mesmo
  no teto de tempo.
- **Tetos.** De tempo e de unidades de computação, por versão, e o da etapa
  (`SALDO_MINIMO`).
- **`versao.json`.** Tudo o que identifica a versão, atualizado a cada ciclo.

A última linha da saída é `FIM: ...` ou `FIM COM ERROS: ...`.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from especialista.caminhos import RAIZ
from especialista.experimentos import etapa11a
from especialista.experimentos.etapa11a import ETAPA, PASTA, descrever, sha256_do_arquivo

COLAB = shutil.which("colab") or str(Path.home() / ".local" / "bin" / "colab")
REMOTO = "/content/pacote"
PYTHON_DO_PROJETO = "3.12.13"
CICLO_S = 240                 # de quanto em quanto tempo as parciais são baixadas
GRACA_S = 20 * 60             # depois do fim, quanto a VM espera o condutor antes de se desligar sozinha
SALDO_INICIAL = 600.0         # medido em 18/11, antes da etapa
SALDO_MINIMO = 450.0          # teto da etapa inteira: 150 unidades
TENTATIVAS = 4

# Só isto vai ao Colab. Nada de `.env`, de chaves, de entregas de outras etapas.
ARQUIVOS_DO_PACOTE = (
    "pyproject.toml", "uv.lock",
    "src/especialista/__init__.py", "src/especialista/caminhos.py", "src/especialista/bio.py",
    "src/especialista/dados/__init__.py", "src/especialista/dados/massive.py",
    "src/especialista/avaliacao/__init__.py", "src/especialista/avaliacao/previsao.py",
    "src/especialista/professores/__init__.py", "src/especialista/professores/prompt.py",
    "src/especialista/professores/demonstracoes.py", "src/especialista/professores/resposta.py",
    "src/especialista/professores/local.py",
    "src/especialista/experimentos/__init__.py", "src/especialista/experimentos/etapa11a.py",
    "entregas/01_dados/train.parquet", "entregas/01_dados/dev.parquet", "entregas/01_dados/test.parquet",
    "entregas/01_dados/rotulos.json",
    f"data/execucoes/{ETAPA}/lotes_da_conferencia.json", f"data/execucoes/{ETAPA}/regra.json",
)


def agora() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def diz(texto: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {texto}", flush=True)


# ---------------------------------------------------------------------------
# O pacote
# ---------------------------------------------------------------------------

def empacotar() -> dict:
    """Monta o `.tar.gz` (sempre o mesmo para os mesmos arquivos) e devolve o registro dele."""
    memoria = io.BytesIO()
    with gzip.GzipFile(fileobj=memoria, mode="wb", mtime=0) as comprimido, tarfile.open(fileobj=comprimido, mode="w") as tar:
        for nome in ARQUIVOS_DO_PACOTE:
            dados = (RAIZ / nome).read_bytes()
            info = tarfile.TarInfo(nome)
            info.size, info.mode = len(dados), 0o644
            tar.addfile(info, io.BytesIO(dados))
    conteudo = memoria.getvalue()
    import hashlib

    sha = hashlib.sha256(conteudo).hexdigest()
    arquivo = PASTA / "pacotes" / f"pacote_{sha[:12]}.tar.gz"
    if not arquivo.exists():
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        arquivo.write_bytes(conteudo)
    return {
        "pacote": str(arquivo.relative_to(RAIZ)), "sha256_do_pacote": sha, "bytes": len(conteudo),
        "arquivos": {nome: sha256_do_arquivo(RAIZ / nome) for nome in ARQUIVOS_DO_PACOTE},
    }


# ---------------------------------------------------------------------------
# A ferramenta `colab`
# ---------------------------------------------------------------------------

def colab(*argumentos: str, limite: int = 600) -> tuple[int, str]:
    try:
        r = subprocess.run([COLAB, *argumentos], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=limite)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "tempo esgotado"


def versao_do_colab() -> str | None:
    """A versão da ferramenta, lida de onde ela está instalada (ela não tem `--version`)."""
    achados = sorted(Path(COLAB).resolve().parents[1].glob("lib/python*/site-packages/google_colab_cli-*.dist-info"))
    return achados[-1].name.removesuffix(".dist-info").replace("_", "-") if achados else None


def uso() -> tuple[float | None, float | None]:
    """Saldo de unidades de computação e taxa por hora da conta."""
    _, saida = colab("usage", limite=60)
    saldo = re.search(r"balance:\s*([\d.]+)", saida)
    taxa = re.search(r"rate:\s*([\d.]+)", saida)
    return (float(saldo.group(1)) if saldo else None, float(taxa.group(1)) if taxa else None)


def sessao_existe(nome: str) -> bool | None:
    """A sessão está na lista do servidor? None quando a consulta falhou (não se conclui nada de uma falha de rede)."""
    codigo, saida = colab("sessions", limite=60)
    return None if codigo != 0 else nome in saida


def ocupada(nome: str) -> bool:
    _, saida = colab("status", "-s", nome, limite=60)
    return "BUSY" in saida


def baixar(sessao: str, remoto: str, local: Path) -> bool:
    """Baixa um arquivo; só troca a cópia local se o que veio não for menor (uma parcial nunca encolhe)."""
    provisorio = local.with_name(local.name + ".baixando")
    local.parent.mkdir(parents=True, exist_ok=True)
    codigo, _ = colab("download", "-s", sessao, remoto, str(provisorio), limite=300)
    if codigo != 0 or not provisorio.exists():
        provisorio.unlink(missing_ok=True)
        return False
    if local.exists() and local.name.endswith(".jsonl") and provisorio.stat().st_size < local.stat().st_size:
        provisorio.unlink()
        return False
    provisorio.replace(local)
    return True


# ---------------------------------------------------------------------------
# As células que rodam no kernel do Colab
# ---------------------------------------------------------------------------

PREPARO = r'''
import hashlib, os, subprocess, sys, tarfile, time
SHA, PY, VERSAO, VIGIA = "@SHA@", "@PY@", "@VERSAO@", @VIGIA@
def sh(comando, pode_falhar=False):
    print("$", comando, flush=True)
    r = subprocess.run(comando, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-2500:], flush=True)
    if r.returncode and not pode_falhar:
        raise SystemExit(f"falhou ({r.returncode}): {comando}")
    return r.returncode
marca = f"/content/pacote/PRONTO_{SHA}"
if not os.path.exists(marca):
    assert hashlib.sha256(open("/content/pacote.tar.gz", "rb").read()).hexdigest() == SHA, "o pacote chegou diferente do que saiu"
    os.makedirs("/content/pacote", exist_ok=True)
    with tarfile.open("/content/pacote.tar.gz") as tar:
        tar.extractall("/content/pacote", filter="data")
    sincronizar = "cd /content/pacote && uv sync --frozen --no-dev --no-install-project --python "
    if sh(sincronizar + PY, pode_falhar=True):
        sh(sincronizar + "3.12")
    sh("cd /content/pacote && uv pip freeze --python .venv/bin/python > pacotes_instalados.txt && wc -l pacotes_instalados.txt")
    open(marca, "w").write(time.strftime("%F %T"))
for parte in ("conferencia", "dev", "test", "train"):
    os.makedirs(f"/content/pacote/data/execucoes/11a_professor_aberto_maior/{VERSAO}/{parte}", exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; /content/pacote/.venv/bin/python --version")
# o vigia: desliga a VM no teto de tempo, aconteça o que acontecer com o kernel e com o condutor
if os.path.exists("/content/vigia.pid"):
    sh("kill " + open("/content/vigia.pid").read(), pode_falhar=True)
vigia = subprocess.Popen(
    [sys.executable, "-c", "import os, time, urllib.request\ntime.sleep(%d)\nurllib.request.urlopen(urllib.request.Request('http://' + os.environ['TBE_RUNTIME_ADDR'] + '/unassign', method='POST'))" % VIGIA],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
open("/content/vigia.pid", "w").write(str(vigia.pid))
print("PREPARO OK", flush=True)
'''

LANCAMENTO = r'''
import json, os, subprocess, threading, time, urllib.request
VERSAO, TAREFAS, TETO, GRACA = "@VERSAO@", "@TAREFAS@", @TETO@, @GRACA@
pasta = f"/content/pacote/data/execucoes/11a_professor_aberto_maior/{VERSAO}"
if os.path.exists(pasta + "/FIM_REMOTO.json"):
    os.remove(pasta + "/FIM_REMOTO.json")
ambiente = dict(os.environ, PYTHONPATH="/content/pacote/src", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
inicio = time.time()
processo = subprocess.Popen(
    ["/content/pacote/.venv/bin/python", "-m", "especialista.experimentos.etapa11a", "trabalho", "--versao", VERSAO, "--tarefas", TAREFAS],
    cwd="/content/pacote", env=ambiente, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
relogio = threading.Timer(TETO, processo.kill)      # o teto de tempo do trabalho
relogio.start()
def pulso():
    while processo.poll() is None:
        time.sleep(120)
        print(f"[pulso: {int(time.time() - inicio)} s]", flush=True)
threading.Thread(target=pulso, daemon=True).start()
with open(pasta + "/saida_do_trabalho.log", "a") as registro:
    for linha in processo.stdout:
        print(linha, end="", flush=True)
        registro.write(linha)
        registro.flush()
codigo = processo.wait()
relogio.cancel()
json.dump({"codigo": codigo, "segundos": round(time.time() - inicio), "tarefas": TAREFAS, "quando_utc": time.strftime("%F %T", time.gmtime())},
          open(pasta + "/FIM_REMOTO.json", "w"))
print(f"TRABALHO REMOTO TERMINOU codigo={codigo}", flush=True)
if GRACA:
    # o condutor baixa tudo e para a sessão; se ele não aparecer no prazo, a VM se desliga sozinha
    time.sleep(GRACA)
    urllib.request.urlopen(urllib.request.Request("http://" + os.environ["TBE_RUNTIME_ADDR"] + "/unassign", method="POST"))
'''


def preencher(molde: str, **valores) -> str:
    for nome, valor in valores.items():
        molde = molde.replace(f"@{nome}@", str(valor))
    return molde


# ---------------------------------------------------------------------------
# A condução
# ---------------------------------------------------------------------------

class Conducao:
    def __init__(self, versao: str, gpu: str, sessao: str, tarefas: list[str], teto_horas: float, teto_unidades: float, manter: bool):
        self.versao, self.gpu, self.sessao, self.tarefas = versao, gpu, sessao, tarefas
        self.teto_s, self.teto_unidades, self.manter = teto_horas * 3600, teto_unidades, manter
        self.pasta = PASTA / versao
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.remota = f"{REMOTO}/data/execucoes/{ETAPA}/{versao}"
        self.pacote = empacotar()
        self.inicio, self.inicio_txt = time.time(), agora()
        self.saldo_antes, _ = uso()
        self.saldo, self.taxa, self.sessoes_criadas, self.situacao = self.saldo_antes, None, 0, "rodando"
        self.partes = [t for t in tarefas if t in ("conferencia", "dev", "test", "train")]
        self.prompt_local = etapa11a.impressao_do_prompt()
        self.devolver_parciais = False

    # -- arquivos

    def arquivos(self) -> list[str]:
        nomes = ["ambiente.json", "saida_do_trabalho.log", "FIM_REMOTO.json", "commit.txt"]
        if "precisao" in self.tarefas:
            nomes.append("precisao.json")
        for parte in self.partes:
            nomes += [f"{parte}/respostas.jsonl", f"{parte}/meta.json"]
        return nomes

    def baixar_parciais(self) -> None:
        for nome in self.arquivos():
            if nome == "FIM_REMOTO.json":
                continue
            baixar(self.sessao, f"{self.remota}/{nome}", self.pasta / nome)
        baixar(self.sessao, f"{REMOTO}/pacotes_instalados.txt", self.pasta / "pacotes_instalados.txt")

    def linhas(self) -> dict[str, int]:
        return {parte: len(etapa11a.ler_registros(self.pasta / parte / "respostas.jsonl")) for parte in self.partes}

    def fim_remoto(self) -> dict | None:
        """O trabalho remoto terminou? Baixa o sinal de fim, se já existir."""
        local = self.pasta / "FIM_REMOTO.json"
        local.unlink(missing_ok=True)
        if baixar(self.sessao, f"{self.remota}/FIM_REMOTO.json", local):
            return json.loads(local.read_text(encoding="utf-8"))
        return None

    # -- o registro da versão

    def gravar_versao(self, fim: str | None = None) -> None:
        v = descrever(self.versao)
        onde = self.pasta / "ambiente.json"
        remoto = json.loads(onde.read_text(encoding="utf-8")) if onde.exists() else {}
        local = self.prompt_local
        metas = {p: json.loads((self.pasta / p / "meta.json").read_text(encoding="utf-8")) for p in self.partes if (self.pasta / p / "meta.json").exists()}
        saidas = {}
        for parte in self.partes:
            arquivo = self.pasta / parte / "respostas.jsonl"
            if arquivo.exists():
                saidas[parte] = {"arquivo": f"{parte}/respostas.jsonl", "sha256": sha256_do_arquivo(arquivo), "pedidos": len(etapa11a.ler_registros(arquivo))}
        destino = self.pasta / "versao.json"
        anterior = json.loads(destino.read_text(encoding="utf-8")) if destino.exists() else {}
        conducoes = [c for c in anterior.get("conducoes", []) if c["inicio"] != self.inicio_txt]
        conducoes.append({
            "inicio": self.inicio_txt, "fim": fim, "tarefas": self.tarefas, "situacao": self.situacao,
            "sessao": self.sessao, "gpu_pedida": self.gpu, "sessoes_criadas": self.sessoes_criadas,
            "sha256_do_pacote": self.pacote["sha256_do_pacote"],
            "unidades": {
                "saldo_antes": self.saldo_antes, "saldo_depois": self.saldo, "taxa_por_hora": self.taxa,
                "gastas_pelo_saldo": None if None in (self.saldo_antes, self.saldo) else round(self.saldo_antes - self.saldo, 2),
                "observacao": "o saldo é da conta inteira: se outra sessão rodou ao mesmo tempo, o gasto dela está aqui dentro",
            },
        })
        destino.write_text(json.dumps({
            "id": self.versao, "etapa": ETAPA,
            "modelo": {"repositorio": v["repositorio"], "nome": v["nome"], "commit": remoto.get("commit_do_modelo", v["commit"])},
            "precisao": {"id": v["precisao"], "descricao": v["descricao"], "tipo_de_calculo": v["tipo_de_calculo"],
                         "tipos_dos_parametros_carregados": remoto.get("tipos_dos_parametros"),
                         "tipo_das_contas_do_professor": remoto.get("tipo_das_contas_do_professor")},
            "prompt": {**local, "igual_no_colab": (remoto.get("prompt") == local) if remoto else None,
                       "tokens_das_instrucoes_em_memoria": remoto.get("tokens_das_instrucoes_em_memoria"),
                       "sha256_das_instrucoes_no_formato_do_modelo": remoto.get("sha256_das_instrucoes_no_formato_do_modelo")},
            "geracao": etapa11a.geracao(),
            "codigo": self.pacote,
            "ambiente": {"maquina": f"Colab, GPU pedida: {self.gpu}", "colab_cli": versao_do_colab(),
                         "python_pedido": PYTHON_DO_PROJETO, "instalacao": "uv sync --frozen --no-dev (o uv.lock do projeto)",
                         "pacotes_instalados": "pacotes_instalados.txt",
                         **{k: remoto[k] for k in remoto if k not in ("prompt", "geracao", "tipos_dos_parametros", "tipo_das_contas_do_professor")}},
            "execucao": {
                "pedidos": {p: s["pedidos"] for p, s in saidas.items()},
                "segundos_de_gpu": {p: m.get("segundos_de_gpu") for p, m in metas.items()},
                "vram_de_pico_gb": {p: m.get("vram_de_pico_gb") for p, m in metas.items()},
                "trechos": {p: m.get("trechos") for p, m in metas.items()},
            },
            "conducoes": conducoes,
            "saidas": saidas,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- os passos

    def preparar(self) -> bool:
        """Garante uma sessão com o pacote, o ambiente e as parciais. Devolve se deu certo."""
        existe = sessao_existe(self.sessao)
        if existe is None:
            diz("não deu para consultar as sessões; tenta de novo")
            return False
        if not existe:
            codigo, saida = colab("new", "-s", self.sessao, "--gpu", self.gpu, limite=600)
            diz(f"sessão {self.sessao} ({self.gpu}): {saida.strip().splitlines()[-1] if saida.strip() else codigo}")
            if codigo != 0 or not sessao_existe(self.sessao):
                return False
            self.sessoes_criadas += 1
            self.devolver_parciais = True
        codigo, saida = colab("upload", "-s", self.sessao, str(RAIZ / self.pacote["pacote"]), "/content/pacote.tar.gz", limite=300)
        if codigo != 0:
            diz(f"envio do pacote falhou: {saida.strip()[-300:]}")
            return False
        celula = self.pasta / "celulas" / "preparo.py"
        celula.parent.mkdir(parents=True, exist_ok=True)
        celula.write_text(preencher(PREPARO, SHA=self.pacote["sha256_do_pacote"], PY=PYTHON_DO_PROJETO, VERSAO=self.versao,
                                    VIGIA=int(self.teto_s + GRACA_S + 900)), encoding="utf-8")
        codigo, saida = colab("exec", "-s", self.sessao, "-f", str(celula), "--timeout", "900", limite=1800)
        (self.pasta / "celulas" / "preparo.log").write_text(saida, encoding="utf-8")
        if "PREPARO OK" not in saida:
            diz(f"preparo falhou: {saida.strip()[-600:]}")
            return False
        placa = [linha for linha in saida.splitlines() if "MiB" in linha]
        diz(f"preparo pronto; placa: {placa[-1].strip() if placa else '?'}")
        if self.devolver_parciais:   # a VM é nova: as parciais que já estão aqui voltam para lá
            self.devolver_parciais = False
            for nome in ["commit.txt", "precisao.json"] + [f"{p}/{a}" for p in self.partes for a in ("respostas.jsonl", "meta.json")]:
                if (self.pasta / nome).exists():
                    colab("upload", "-s", self.sessao, str(self.pasta / nome), f"{self.remota}/{nome}", limite=300)
        return True

    def lancar(self, numero: int) -> subprocess.Popen:
        celula = self.pasta / "celulas" / f"lancamento_{numero}.py"
        restante = max(600, int(self.teto_s - (time.time() - self.inicio)))
        celula.write_text(preencher(LANCAMENTO, VERSAO=self.versao, TAREFAS=",".join(self.tarefas), TETO=restante,
                                    GRACA=0 if self.manter else GRACA_S), encoding="utf-8")
        registro = (self.pasta / "celulas" / f"lancamento_{numero}.log").open("w", encoding="utf-8")
        return subprocess.Popen([COLAB, "exec", "-s", self.sessao, "-f", str(celula), "--timeout", "1500"],
                                stdin=subprocess.DEVNULL, stdout=registro, stderr=subprocess.STDOUT)

    def acompanhar(self, processo: subprocess.Popen, numero: int) -> dict | None:
        """Espera o trabalho remoto. Devolve o sinal de fim, ou None se a sessão se perdeu ou um teto foi atingido."""
        registro, ultimo_ciclo = self.pasta / "celulas" / f"lancamento_{numero}.log", time.time()
        while True:
            time.sleep(20)
            texto = registro.read_text(encoding="utf-8", errors="replace") if registro.exists() else ""
            terminou = "TRABALHO REMOTO TERMINOU" in texto
            morreu = processo.poll() is not None
            if terminou or morreu or time.time() - ultimo_ciclo >= CICLO_S:
                ultimo_ciclo = time.time()
                self.baixar_parciais()
                self.saldo, taxa = uso()
                self.taxa = taxa or self.taxa
                self.gravar_versao()
                ultima = next((linha for linha in reversed(texto.splitlines()) if linha.strip() and "pulso" not in linha), "")
                diz(f"{self.linhas()} | saldo {self.saldo} ({self.taxa}/h) | {ultima.strip()[:110]}")
                fim = self.fim_remoto()
                if fim is not None:
                    return fim
                if time.time() - self.inicio > self.teto_s:
                    self.situacao = "teto de tempo"
                    return None
                if self.saldo is not None and (self.saldo < SALDO_MINIMO or (self.saldo_antes or 0) - self.saldo > self.teto_unidades):
                    self.situacao = "teto de unidades"
                    return None
                existe = sessao_existe(self.sessao)
                if existe is False or (morreu and existe and not ocupada(self.sessao)):
                    return None        # a célula não está mais rodando: a sessão caiu, ou o kernel parou
                # se a ligação local caiu mas o kernel segue ocupado, o trabalho continua lá: é só seguir baixando

    def conduzir(self) -> tuple[bool, str]:
        etapa11a.gravar_regra()
        diz(f"versão {self.versao} | tarefas {','.join(self.tarefas)} | pacote {self.pacote['sha256_do_pacote'][:12]} | saldo {self.saldo_antes}")
        self.gravar_versao()
        fim, processo = None, None
        try:
            for numero in range(1, TENTATIVAS + 1):
                if not self.preparar():
                    self.situacao = "preparo falhou"
                    time.sleep(60)
                    continue
                self.situacao = "rodando"
                processo = self.lancar(numero)
                fim = self.acompanhar(processo, numero)
                if processo.poll() is None:
                    processo.terminate()
                if fim is not None or self.situacao.startswith("teto"):
                    break
                diz(f"tentativa {numero}: a sessão se perdeu antes do fim; recomeçando de onde parou")
                self.situacao = "sessão perdida"
        finally:
            if processo is not None and processo.poll() is None:
                processo.terminate()
            terminou_bem = fim is not None and fim["codigo"] == 0
            if not (self.manter and terminou_bem):
                diz(f"parando a sessão {self.sessao}: {colab('stop', '-s', self.sessao, limite=120)[1].strip()[-120:]}")
            self.saldo, _ = uso()
            if fim is not None:
                self.situacao = {0: "concluída", 3: "precisão insuficiente (NaN ou infinito)"}.get(fim["codigo"], f"o trabalho remoto falhou (código {fim['codigo']})")
            self.gravar_versao(fim=agora())
        gasto = None if None in (self.saldo_antes, self.saldo) else round(self.saldo_antes - self.saldo, 2)
        return terminou_bem, f"{self.versao}: {self.situacao}; pedidos {self.linhas()}; unidades gastas {gasto}; saldo {self.saldo}"


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argumentos.add_argument("--versao", required=True)
    argumentos.add_argument("--gpu", required=True, help="L4, A100, G4, T4")
    argumentos.add_argument("--sessao", required=True)
    argumentos.add_argument("--tarefas", required=True, help="separadas por vírgula: conferencia, precisao, dev, test, train")
    argumentos.add_argument("--teto-horas", type=float, required=True)
    argumentos.add_argument("--teto-unidades", type=float, required=True)
    argumentos.add_argument("--manter-sessao", action="store_true", help="não para a sessão se terminar bem (para o passo seguinte usar a mesma VM)")
    argumentos.add_argument("--reserva", default=None, help="versão que roda no lugar se esta parar por precisão insuficiente")
    a = argumentos.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))     # um `kill` passa pelo `finally`: a sessão é parada
    try:
        bem, resumo = Conducao(a.versao, a.gpu, a.sessao, a.tarefas.split(","), a.teto_horas, a.teto_unidades, a.manter_sessao).conduzir()
        if not bem and "precisão insuficiente" in resumo and a.reserva:
            diz(resumo)
            diz(f"regra de precisão: refazendo inteira na versão de reserva {a.reserva}")
            bem, resumo = Conducao(a.reserva, a.gpu, a.sessao, a.tarefas.split(","), a.teto_horas, a.teto_unidades, a.manter_sessao).conduzir()
    except BaseException as erro:      # a última linha do registro tem de dizer o que houve
        import traceback

        traceback.print_exc()
        colab("stop", "-s", a.sessao, limite=120)
        bem, resumo = False, f"o condutor falhou: {erro!r}"
    print(("FIM: " if bem else "FIM COM ERROS: ") + resumo, flush=True)
    sys.exit(0 if bem else 1)
