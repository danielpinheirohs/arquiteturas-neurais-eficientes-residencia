"""Impressão digital de tudo o que as etapas anteriores gravaram, para provar que a Etapa 12 não mexeu em nada.

    python impressoes.py tirar SAIDA.json        # tamanho e data de todo arquivo; sha256 dos de até 64 MB
    python impressoes.py comparar ANTES.json DEPOIS.json

Cobre `data/execucoes/` e `entregas/` inteiros, menos as pastas da própria Etapa 12.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

RAIZ = Path("/home/danie/project/residencia")
PASTAS = (RAIZ / "data" / "execucoes", RAIZ / "entregas")
FORA = "12_investigacao_dos_achados"
LIMITE = 64 * 2**20


def sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def tirar() -> dict:
    arquivos = {}
    for pasta in PASTAS:
        for caminho in sorted(pasta.rglob("*")):
            relativo = str(caminho.relative_to(RAIZ))
            if not caminho.is_file() or relativo.split("/")[2].startswith(FORA):
                continue
            estado = caminho.stat()
            arquivos[relativo] = {"bytes": estado.st_size, "mtime_ns": estado.st_mtime_ns,
                                  "sha256": sha256(caminho) if estado.st_size <= LIMITE else None}
    return {"tirada_em": time.strftime("%Y-%m-%d %H:%M:%S %z"), "arquivos": arquivos}


def comparar(antes: dict, depois: dict) -> dict:
    a, d = antes["arquivos"], depois["arquivos"]
    por_etapa: dict[str, dict] = {}
    for relativo in sorted(set(a) | set(d)):
        etapa = "/".join(relativo.split("/")[:3]) if relativo.startswith("data/") else "/".join(relativo.split("/")[:2])
        conta = por_etapa.setdefault(etapa, {"arquivos": 0, "com_sha256": 0, "iguais": 0, "mudaram": [], "sumiram": [], "novos": []})
        if relativo not in d:
            conta["sumiram"].append(relativo)
        elif relativo not in a:
            conta["novos"].append(relativo)
        else:
            conta["arquivos"] += 1
            conta["com_sha256"] += a[relativo]["sha256"] is not None
            if a[relativo] == d[relativo]:
                conta["iguais"] += 1
            else:
                conta["mudaram"].append(relativo)
    return por_etapa


if __name__ == "__main__":
    if sys.argv[1] == "tirar":
        tirada = tirar()
        Path(sys.argv[2]).write_text(json.dumps(tirada, indent=1), encoding="utf-8")
        grandes = sum(v["sha256"] is None for v in tirada["arquivos"].values())
        print(f"{len(tirada['arquivos'])} arquivos ({grandes} acima de 64 MB, só com tamanho e data) em {tirada['tirada_em']}")
    else:
        antes, depois = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:4])
        por_etapa = comparar(antes, depois)
        print(f"antes: {antes['tirada_em']}; depois: {depois['tirada_em']}")
        for etapa, c in por_etapa.items():
            sinal = "igual" if not (c["mudaram"] or c["sumiram"] or c["novos"]) else "MUDOU"
            print(f"{sinal:6} {etapa}: {c['iguais']} de {c['arquivos']} arquivos iguais ({c['com_sha256']} com sha256); "
                  f"mudaram {len(c['mudaram'])}, sumiram {len(c['sumiram'])}, novos {len(c['novos'])}")
            for nome in ("mudaram", "sumiram", "novos"):
                for relativo in c[nome][:10]:
                    print(f"        {nome}: {relativo}")
        total = sum(len(c["mudaram"]) + len(c["sumiram"]) + len(c["novos"]) for c in por_etapa.values())
        print("NADA MUDOU" if total == 0 else f"{total} DIFERENÇAS")
