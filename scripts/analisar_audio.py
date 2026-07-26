#!/usr/bin/env python
"""Roda a analise de audio sobre a amostra do TORGO.

Para cada arquivo: transcreve, extrai as metricas acusticas e analisa o texto
(sentimento e termos criticos). No fim compara os dois grupos, fala disartrica
contra controle, com teste de Mann-Whitney. O resultado sustenta o requisito
de detectar alteracoes vocais indicativas de condicoes medicas.

Uso:
    python scripts/analisar_audio.py
    python scripts/analisar_audio.py --modelo small --limite 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jiwer
import pandas as pd
from scipy.stats import mannwhitneyu

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.audio.analise_texto import analisar as analisar_texto  # noqa: E402
from src.audio.metricas_vocais import extrair_metricas  # noqa: E402
from src.audio.transcricao import MODELO_PADRAO, carregar_modelo, transcrever  # noqa: E402

ENTRADA = RAIZ / "data" / "raw" / "torgo"
SAIDA = RAIZ / "data" / "processed"

# jiwer aplica estas transformacoes antes de comparar, para que a taxa de erro
# reflita diferenca de palavras e nao de pontuacao ou caixa.
NORMALIZACAO = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

COLUNAS_METRICAS = [
    "jitter_local", "shimmer_local", "hnr_db", "f0_media_hz", "f0_desvio_hz",
    "proporcao_pausa", "taxa_fala_silabas_s", "wer", "logprob_media",
]

GRUPO_CASO = "dysarthria"
GRUPO_CONTROLE = "healthy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", default=MODELO_PADRAO,
                        help=f"tamanho do modelo Whisper (default: {MODELO_PADRAO})")
    parser.add_argument("--limite", type=int, default=None,
                        help="processa apenas os N primeiros arquivos")
    parser.add_argument("--entrada", type=Path, default=ENTRADA)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()

    manifesto = pd.read_csv(args.entrada / "manifesto.csv")
    if args.limite:
        manifesto = manifesto.head(args.limite)
    pasta_audio = args.entrada / "audio"
    args.saida.mkdir(parents=True, exist_ok=True)

    print(f"Carregando Whisper '{args.modelo}'...")
    modelo = carregar_modelo(args.modelo)

    linhas = []
    for i, registro in enumerate(manifesto.itertuples(), start=1):
        caminho = pasta_audio / registro.arquivo
        transcricao = transcrever(caminho, modelo=modelo)
        metricas = extrair_metricas(caminho).to_dict()
        texto = analisar_texto(transcricao.texto)

        referencia = str(registro.transcricao)
        hipotese = transcricao.texto

        # jiwer falha se a normalizacao esvaziar a referencia.
        try:
            wer = jiwer.wer(referencia, hipotese,
                            reference_transform=NORMALIZACAO,
                            hypothesis_transform=NORMALIZACAO)
        except ValueError:
            wer = float("nan")

        metricas.update({
            "grupo": registro.grupo,
            "genero": registro.genero,
            # O locutor entra no CSV para permitir analise por sujeito depois:
            # sao 15 locutores, e efeito individual e forte em medida vocal.
            "falante": registro.falante,
            "referencia": referencia,
            "transcricao": hipotese,
            "wer": round(float(wer), 4),
            "logprob_media": transcricao.logprob_media,
            "proporcao_sem_fala": transcricao.proporcao_sem_fala,
        })
        metricas.update(texto.to_dict())
        linhas.append(metricas)
        print(f"  [{i:3d}/{len(manifesto)}] {registro.arquivo}  "
              f"WER={wer:.2f}  \"{hipotese[:40]}\"")

    df = pd.DataFrame(linhas)
    destino = args.saida / "audio_metricas.csv"
    df.to_csv(destino, index=False)

    _resumir(df)
    print(f"\nResultados em {destino}")


def _resumir(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("COMPARACAO ENTRE GRUPOS")
    print("=" * 78)

    medias = df.groupby("grupo")[COLUNAS_METRICAS].mean().round(4)
    print(medias.to_string())

    if not {GRUPO_CASO, GRUPO_CONTROLE} <= set(medias.index):
        return

    print(f"\n{'metrica':24s} {'variacao':>10s} {'p-valor':>10s}")
    print("-" * 78)
    for coluna in COLUNAS_METRICAS:
        caso = df.loc[df.grupo == GRUPO_CASO, coluna].dropna()
        controle = df.loc[df.grupo == GRUPO_CONTROLE, coluna].dropna()
        if caso.empty or controle.empty or controle.mean() == 0:
            continue

        variacao = (caso.mean() - controle.mean()) / abs(controle.mean()) * 100
        # Mann-Whitney nao assume normalidade, o que importa aqui: jitter e
        # shimmer tem distribuicao assimetrica e a amostra e pequena.
        _, p = mannwhitneyu(caso, controle, alternative="two-sided")
        marca = " *" if p < 0.05 else ""
        print(f"{coluna:24s} {variacao:+9.1f}% {p:10.4f}{marca}")

    print("\n* diferenca significativa a 5%. Variacao e do grupo disartrico "
          "em relacao ao controle.")
    _resumir_texto(df)


def _resumir_texto(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("ANALISE DE TEXTO DAS TRANSCRICOES")
    print("=" * 78)

    print("Sentimento por grupo:")
    print(pd.crosstab(df.grupo, df.sentimento).to_string())

    com_termo = df[df.n_termos_criticos > 0]
    print(f"\nArquivos com termo critico: {len(com_termo)} de {len(df)}")
    if not com_termo.empty:
        print(com_termo[["arquivo", "grupo", "transcricao",
                         "termos_criticos", "escore_criticidade"]]
              .to_string(index=False, max_colwidth=38))

    # As frases do TORGO vem do TIMIT e quase nao tem vocabulario clinico,
    # entao um numero baixo aqui e o esperado, nao falha do detector.
    print("\nLembrete: o corpus TORGO nao e clinico. O detector de termos e "
          "exercitado em src/audio/analise_texto.py (EXEMPLOS_CLINICOS).")


if __name__ == "__main__":
    main()
