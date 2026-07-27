#!/usr/bin/env python
"""Roda a analise de video sobre o Keraal (OpenPose + YOLOv8 + anotacoes).

Passos:
  1. Parseia os pares Anvil UTF-16 dos dois medicos (consenso / mais grave).
  2. Extrai angulos articulares dos JSONs OpenPose e monta o perfil saudavel.
  3. Pontua desvio postural de cada gravacao contra o perfil do exercicio.
  4. Amostra frames dos MP4 com YOLOv8 para objetos e areas criticas.
  5. Escreve o relatorio de desvios e as metricas de validacao contra os medicos.

Uso:
    python scripts/analisar_video.py
    python scripts/analisar_video.py --sem-yolo          # so esqueleto, ~20 s
    python scripts/analisar_video.py --limite 30         # amostra para teste
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.video import anotacoes, desvios, esqueleto, objetos  # noqa: E402

SAIDA = RAIZ / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, default=None,
                        help="processa apenas N gravacoes (teste rapido)")
    parser.add_argument("--sem-yolo", action="store_true",
                        help="pula a deteccao de objetos (bem mais rapido)")
    parser.add_argument("--frames-yolo", type=int, default=objetos.FRAMES_POR_VIDEO)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    print("=== 1. Anotacoes dos medicos ===")
    gravacoes = anotacoes.listar_gravacoes()
    df_anot = anotacoes.para_dataframe(gravacoes)
    if args.limite:
        # Mantem proporcao aproximada de grupos ao cortar.
        df_anot = (df_anot.groupby("grupo", group_keys=False)
                   .head(max(1, args.limite // 2)))
        ids = set(df_anot["identificador"])
        gravacoes = [g for g in gravacoes if g.identificador in ids]
    print(f"  {len(df_anot)} gravacoes com par A/B")
    print(f"  consenso: {df_anot['consenso'].mean():.1%}")
    print(f"  por avaliacao:\n{df_anot['avaliacao'].value_counts().to_string()}")
    df_anot.to_csv(args.saida / "video_anotacoes.csv", index=False)

    print("\n=== 2. Features OpenPose ===")
    df_feat = esqueleto.features_todas(list(df_anot["identificador"]))
    print(f"  {len(df_feat)} series com angulos")
    df_feat.to_csv(args.saida / "video_features.csv", index=False)

    print("\n=== 3. Perfil saudavel + classificador ===")
    perfis = desvios.construir_perfis(df_feat)
    for ex, perfil in sorted(perfis.items()):
        print(f"  {ex}: {perfil.n_controle} controles")
    pontuacao, met_cv = desvios.pontuar(df_feat, df_anot, perfis)

    print("\n=== 4. Validacao cruzada (consenso G1A) ===")
    _imprimir_metricas(met_cv)
    pd.DataFrame([{"recorte": "cv_consenso_g1a", **met_cv}]).to_csv(
        args.saida / "video_validacao.csv", index=False
    )

    df_objetos = pd.DataFrame()
    if not args.sem_yolo:
        print("\n=== 5. YOLOv8 em frames amostrados ===")
        pares = []
        for ident in df_anot["identificador"]:
            vid = esqueleto.caminho_video(ident)
            if vid is not None:
                pares.append((ident, vid))
        df_objetos = objetos.analisar_varios(pares, frames_por_video=args.frames_yolo)
        df_objetos.to_csv(args.saida / "video_objetos.csv", index=False)
        print(f"  fracao media com pessoa: "
              f"{df_objetos['fracao_com_pessoa'].mean():.1%}")
    else:
        print("\n=== 5. YOLOv8 pulado (--sem-yolo) ===")

    print("\n=== 6. Relatorio de desvios ===")
    relatorio = desvios.relatorio_desvios(pontuacao, df_anot, df_objetos)
    caminho_rel = args.saida / "video_desvios.csv"
    relatorio.to_csv(caminho_rel, index=False)

    # Resumo por gravacao para a fusao: uma linha, escore_risco 0-1.
    resumo = relatorio[["identificador", "grupo", "exercicio", "avaliacao",
                        "escore_risco", "alerta", "desvio_detectado"]].copy()
    resumo.to_csv(args.saida / "video_desvios_resumo.csv", index=False)

    n_alerta = int(relatorio["alerta"].sum()) if "alerta" in relatorio else 0
    print(f"  {len(relatorio)} linhas em {caminho_rel}")
    print(f"  alertas: {n_alerta}")
    print("\nTop desvios (pacientes):")
    top = relatorio[relatorio["grupo"] == "G1A"].head(8)
    cols = [c for c in ("identificador", "exercicio", "avaliacao",
                        "escore_desvio", "escore_risco") if c in top.columns]
    print(top[cols].to_string(index=False))
    print("\nConcluido.")


def _imprimir_metricas(m: dict) -> None:
    if "aviso" in m:
        print(f"    n={m.get('n', 0)} ({m['aviso']})")
        return
    print(
        f"    n={m['n']} (erro={m['n_com_erro']}, ok={m['n_sem_erro']})  "
        f"AUC={m.get('auc')}  P={m['precisao']}  R={m['revocacao']}  "
        f"F1={m['f1']}  Acc={m['acuracia']}"
    )


if __name__ == "__main__":
    main()
