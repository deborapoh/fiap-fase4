#!/usr/bin/env python
"""Reescreve arquivos de texto do repositorio em UTF-8.

O editor usado no desenvolvimento gravou alguns arquivos em UTF-16, o que o
interpretador do Python rejeita ("source code string cannot contain null
bytes") e o git trata como binario. Este script detecta e corrige.

Uso:
    python scripts/normalizar_encoding.py            # verifica e corrige
    python scripts/normalizar_encoding.py --verificar # apenas aponta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXTENSOES = {".py", ".md", ".txt", ".sh", ".csv", ".yml", ".yaml", ".json",
             ".toml", ".cfg"}
IGNORAR = {".git", ".venv", "data", "models", "__pycache__"}


def detectar_utf16(dados: bytes) -> str | None:
    """Devolve o nome da codificacao UTF-16 do conteudo, ou None se ja for UTF-8.

    Sem BOM a ordem dos bytes precisa ser inferida: em texto ASCII gravado
    como UTF-16LE o byte nulo cai nas posicoes impares, e em UTF-16BE nas
    pares.
    """
    if dados.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if b"\x00" not in dados[:4096]:
        return None

    amostra = dados[:4096]
    nulos_impares = sum(1 for i in range(1, len(amostra), 2) if amostra[i] == 0)
    nulos_pares = sum(1 for i in range(0, len(amostra), 2) if amostra[i] == 0)
    return "utf-16-le" if nulos_impares >= nulos_pares else "utf-16-be"


def arquivos_candidatos() -> list[Path]:
    return sorted(
        caminho for caminho in RAIZ.rglob("*")
        if caminho.is_file()
        and caminho.suffix in EXTENSOES
        and not any(parte in IGNORAR for parte in caminho.parts)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar", action="store_true",
                        help="nao altera nada, so lista o que esta errado")
    args = parser.parse_args()

    problemas = []
    for caminho in arquivos_candidatos():
        dados = caminho.read_bytes()
        codificacao = detectar_utf16(dados)
        if codificacao is None:
            continue

        relativo = caminho.relative_to(RAIZ)
        problemas.append(relativo)
        if args.verificar:
            print(f"UTF-16 ({codificacao}): {relativo}")
            continue

        texto = dados.decode(codificacao)
        caminho.write_text(texto, encoding="utf-8")
        print(f"convertido de {codificacao}: {relativo}")

    if not problemas:
        print("Todos os arquivos de texto estao em UTF-8.")
        return 0
    return 1 if args.verificar else 0


if __name__ == "__main__":
    sys.exit(main())
