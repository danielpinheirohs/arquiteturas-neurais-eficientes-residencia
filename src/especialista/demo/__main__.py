"""Uma página local para experimentar os dois alunos finais lado a lado.

    uv run python -m especialista.demo            # abre em http://127.0.0.1:8765
    uv run python -m especialista.demo --porta 9000

Para cada pedido, os dois modelos respondem (aluno A, do professor fechado, e
aluno B2, do professor aberto), e a página mostra a resposta de cada um, a
confiança na intenção e o tempo de cada trecho da consulta.

O servidor usa só a biblioteca padrão e o sistema exportado em 8 bits
(`compressao/sistema_onnx.py`): roda na CPU, sem PyTorch. Só atende esta
máquina (127.0.0.1). Os números fixos de cada modelo (qualidade no teste,
tamanho, latência do professor) vêm de `entregas/09_comparativo/tabela_final.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from especialista.bio import bio_para_spans
from especialista.compressao.sistema_onnx import SistemaOnnx

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = RAIZ / "entregas" / "08_quantizacao" / "modelos"
TABELA = RAIZ / "entregas" / "09_comparativo" / "tabela_final.csv"
PAGINA = Path(__file__).with_name("pagina.html")

# chave na página: (pasta do modelo, linha do aluno na tabela final, linha do professor, como aprendeu)
ALUNOS = {
    "a": ("aluno_a", "aluno_a/int8", "claude", "Aluno A", "caixa-preta: só os rótulos do professor"),
    "b2": ("aluno_b2_t8", "aluno_b2/int8", "qwen", "Aluno B2", "caixa-branca: rótulos e logits do professor"),
}
MAXIMO_DE_REPETICOES = 500


def _tabela() -> dict[str, dict]:
    with TABELA.open(encoding="utf-8") as arquivo:
        return {linha["chave"]: linha for linha in csv.DictReader(arquivo)}


def _numero(texto: str) -> float | None:
    return float(texto) if texto not in ("", None) else None


class Demo:
    def __init__(self, semente: int = 0):
        tabela = _tabela()
        self.sistemas: dict[str, SistemaOnnx] = {}
        self.fichas: dict[str, dict] = {}
        for chave, (pasta, linha_do_aluno, linha_do_professor, nome, metodo) in ALUNOS.items():
            caminho = MODELOS / pasta / "linhas_90" / f"semente_{semente}"
            sistema = SistemaOnnx.carregar(caminho)
            sistema.prever(["aquecimento do modelo"])   # a primeira consulta de um processo é mais lenta
            self.sistemas[chave] = sistema
            aluno, professor = tabela[linha_do_aluno], tabela[linha_do_professor]
            self.fichas[chave] = {
                "nome": nome,
                "metodo": metodo,
                "forma": aluno["sistema"],
                "bytes": sum(arquivo.stat().st_size for arquivo in caminho.iterdir()),
                "parametros": sistema.parametros,
                "teste": {m: _numero(aluno[m]) for m in ("intent_acc", "slot_f1", "exact_match")},
                "latencia_p50_ms": _numero(aluno["latencia_p50_ms"]),
                "professor": {
                    "nome": professor["curto"],
                    "teste": {m: _numero(professor[m]) for m in ("intent_acc", "slot_f1", "exact_match")},
                    "latencia_p50_ms": _numero(professor["latencia_p50_ms"]),
                    "onde": professor["condicao_da_latencia"],
                    "bytes": _numero(professor["bytes"]),
                },
            }

    def _uma_consulta(self, sistema: SistemaOnnx, enunciado: list[str]):
        t0 = time.perf_counter()
        entradas = sistema.entradas([enunciado])
        t1 = time.perf_counter()
        intencoes, emissoes = sistema.sessao.run(["intencoes", "emissoes"], entradas)
        t2 = time.perf_counter()
        caminho = sistema.decodificar(emissoes, [len(enunciado)])[0]
        t3 = time.perf_counter()
        return intencoes[0], caminho, ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)

    def responder(self, texto: str, repeticoes: int = 1) -> dict:
        enunciado = texto.lower().split()
        if not enunciado:
            raise ValueError("pedido vazio")
        repeticoes = max(1, min(int(repeticoes), MAXIMO_DE_REPETICOES))
        resposta = {"texto": " ".join(enunciado), "palavras": enunciado, "repeticoes": repeticoes, "alunos": {}}
        for chave, sistema in self.sistemas.items():
            pontos, caminho, tempos = self._uma_consulta(sistema, enunciado)
            totais = [sum(tempos)]
            for _ in range(repeticoes - 1):
                totais.append(sum(self._uma_consulta(sistema, enunciado)[2]))
            probabilidades = np.exp(pontos - pontos.max())
            probabilidades /= probabilidades.sum()
            ordem = np.argsort(-probabilidades)[:3]
            tags = [sistema.tags[t] for t in caminho]
            conhecidas = [bool(sistema.featurizador.vocabulario.get(f"p:{palavra}") is not None) for palavra in enunciado]
            resposta["alunos"][chave] = {
                "intent": sistema.intents[int(ordem[0])],
                "confianca": float(probabilidades[ordem[0]]),
                "mais_provaveis": [{"intent": sistema.intents[int(i)], "p": float(probabilidades[i])} for i in ordem],
                "tags": tags,
                "slots": [{"tipo": tipo, "inicio": a, "fim": b, "texto": " ".join(enunciado[a:b])} for tipo, a, b in bio_para_spans(tags)],
                "palavras_conhecidas": conhecidas,
                "ms": {"texto_para_ids": tempos[0], "rede": tempos[1], "viterbi": tempos[2], "total": sum(tempos)},
                "ms_repetido": None if repeticoes == 1 else {
                    "p50": float(np.percentile(totais, 50)), "p95": float(np.percentile(totais, 95)), "minimo": float(min(totais)),
                },
            }
        return resposta


def tratador(demo: Demo):
    class Tratador(BaseHTTPRequestHandler):
        def _enviar(self, codigo: int, corpo: bytes, tipo: str) -> None:
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corpo)

        def _json(self, codigo: int, dados) -> None:
            self._enviar(codigo, json.dumps(dados, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._enviar(200, PAGINA.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/modelos":
                self._json(200, demo.fichas)
            else:
                self._json(404, {"erro": "não encontrado"})

        def do_POST(self) -> None:
            if self.path != "/api/prever":
                return self._json(404, {"erro": "não encontrado"})
            try:
                tamanho = int(self.headers.get("Content-Length", 0))
                if tamanho > 10_000:
                    return self._json(413, {"erro": "pedido grande demais"})
                pedido = json.loads(self.rfile.read(tamanho) or b"{}")
                self._json(200, demo.responder(str(pedido.get("texto", ""))[:500], pedido.get("repeticoes", 1)))
            except (ValueError, TypeError) as erro:
                self._json(400, {"erro": str(erro)})

        def log_message(self, *args) -> None:   # sem uma linha no terminal a cada pedido
            pass

    return Tratador


def main() -> None:
    argumentos = argparse.ArgumentParser(description="Página local para experimentar os dois alunos finais.")
    argumentos.add_argument("--porta", type=int, default=8765)
    argumentos.add_argument("--semente", type=int, default=0, choices=[0, 1, 2])
    opcoes = argumentos.parse_args()
    servidor = ThreadingHTTPServer(("127.0.0.1", opcoes.porta), tratador(Demo(opcoes.semente)))
    print(f"Os dois alunos estão prontos em http://127.0.0.1:{opcoes.porta}  (Ctrl+C para parar)", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
