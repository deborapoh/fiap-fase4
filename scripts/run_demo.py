#!/usr/bin/env python3
"""Demo end-to-end do Tech Challenge Fase 4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(description="FIAP Fase 4 — demo multimodal")
    parser.add_argument("--no-yolo", action="store_true", help="Pula YOLOv8 (mais rápido)")
    parser.add_argument("--output", type=str, default="output/demo_report.json")
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    if not args.skip_synthetic:
        from scripts.generate_synthetic import main as gen_main
        gen_main()

    from src.fusion.orchestrator import run_full_analysis

    print("Executando análise multimodal...")
    result = run_full_analysis(run_yolo=not args.no_yolo)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    alert = result["alert"]
    print("\n=== ALERTA ===")
    print(f"Severidade: {alert['severity']}")
    print(f"Score:      {alert['risk_score']}")
    print(f"Mensagem:   {alert['message']}")
    print(f"\nEvidências ({len(alert['evidence'])}):")
    for ev in alert["evidence"]:
        first = ev["alerts"][0] if ev["alerts"] else "-"
        print(f"  - [{ev['modality']}] score={ev['score']}: {first}")

    print(f"\nRelatório completo: {out_path}")


if __name__ == "__main__":
    main()
