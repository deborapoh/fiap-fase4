#!/usr/bin/env python
"""Monta pacientes sinteticos e funde os escores das tres frentes.

Uso:
    python scripts/fundir_risco.py
    python scripts/fundir_risco.py --n 30 --semente 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.fusion import risco  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()

    print("Fundindo escores das frentes em pacientes sinteticos...")
    df = risco.gerar(n=args.n, semente=args.semente)
    if df.empty:
        print("Nenhum CSV de frente encontrado em data/processed/.")
        print("Rode antes: analisar_audio, detectar_anomalias, analisar_video.")
        sys.exit(1)

    print(f"  {len(df)} pacientes sinteticos")
    print(f"  alertas: {int(df['alerta'].sum())}")
    print(f"  escore medio: {df['escore_risco'].mean():.3f}")
    print(f"  por severidade:\n{df['severidade'].value_counts().to_string()}")
    print("\nTop alertas:")
    top = df.sort_values("escore_risco", ascending=False).head(5)
    print(top[["id_sintetico", "escore_risco", "severidade",
               "escore_audio", "escore_vitais", "escore_video",
               "escore_prescricoes"]].to_string(index=False))
    print("\nEscrito em data/processed/fusao_risco.csv e fusao_alertas.csv")


if __name__ == "__main__":
    main()
