"""Experimentar o modelo final à mão: digite um pedido, veja a intenção, as entidades e o tempo.

    uv run python -m especialista.experimentar                       # conversa: um pedido por linha
    uv run python -m especialista.experimentar "toca tony carreira"  # um ou mais pedidos, e sai
    uv run python -m especialista.experimentar --modelo aluno_b2_t8  # o aluno do professor aberto

Roda o sistema exportado em 8 bits (`compressao/sistema_onnx.py`), na CPU,
sem PyTorch. O texto é passado para minúsculas, como no MASSIVE.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from especialista.compressao.sistema_onnx import SistemaOnnx

MODELOS = Path(__file__).resolve().parents[2] / "entregas" / "08_quantizacao" / "modelos"


def responder(sistema: SistemaOnnx, texto: str) -> str:
    texto = " ".join(texto.lower().split())
    inicio = time.perf_counter()
    previsao = sistema.prever([texto])[0]
    ms = (time.perf_counter() - inicio) * 1000
    palavras = texto.split()
    entidades = [f"{tipo} = {' '.join(palavras[a:b])}" for tipo, a, b in previsao.slots]
    return (
        f"  intenção:  {previsao.intent}\n"
        f"  entidades: {'; '.join(entidades) if entidades else 'nenhuma'}\n"
        f"  tempo:     {ms:.2f} ms".replace(".", ",")
    )


def main() -> None:
    argumentos = argparse.ArgumentParser(description="Experimentar o modelo final à mão.")
    argumentos.add_argument("pedidos", nargs="*", help="pedidos a responder; sem nenhum, abre a conversa")
    argumentos.add_argument("--modelo", default="aluno_a", choices=["aluno_a", "aluno_b2_t8", "teto"])
    argumentos.add_argument("--semente", type=int, default=0, choices=[0, 1, 2])
    opcoes = argumentos.parse_args()

    pasta = MODELOS / opcoes.modelo / "linhas_90" / f"semente_{opcoes.semente}"
    sistema = SistemaOnnx.carregar(pasta)
    sistema.prever(["aquecimento"])   # a primeira consulta de um processo é mais lenta
    tamanho = sum(arquivo.stat().st_size for arquivo in pasta.iterdir()) / 1e6
    print(f"{sistema.nome} ({f'{tamanho:.2f}'.replace('.', ',')} MB em disco)\n")

    if opcoes.pedidos:
        for pedido in opcoes.pedidos:
            print(f"> {pedido}\n{responder(sistema, pedido)}\n")
        return
    print("Digite um pedido e Enter. Linha vazia ou Ctrl+D para sair.\n")
    while True:
        try:
            pedido = input("> ").strip()
        except EOFError:
            break
        if not pedido:
            break
        print(responder(sistema, pedido) + "\n")


if __name__ == "__main__":
    main()
