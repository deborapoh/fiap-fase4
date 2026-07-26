#!/usr/bin/env python
"""Extrai uma amostra balanceada do TORGO (fala disartrica x fala saudavel).

O dataset esta publicado como quatro arquivos parquet (1,5 GB no total) e vem
ordenado por locutor: todos os controles saudaveis aparecem antes dos falantes
disartricos. Por isso o modo streaming e ruim aqui, ele teria que percorrer
quase o dataset inteiro para achar o segundo grupo. Baixamos os shards (o cache
do huggingface_hub retoma downloads interrompidos) e amostramos localmente.

Uso:
    python scripts/download_torgo.py                # 40 audios por grupo
    python scripts/download_torgo.py --por-grupo 80
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET = "abnerh/TORGO-database"
SHARDS = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "raw" / "torgo"

# Amostras muito curtas (digitos isolados, "yes"/"no") nao sustentam analise de
# fadiga ou prosodia, entao exigimos um minimo de duracao.
DURACAO_MINIMA_S = 1.5

COLUNAS = ["audio", "transcription", "speech_status", "gender", "duration"]


def baixar_shards() -> list[Path]:
    caminhos = []
    for shard in SHARDS:
        print(f"  baixando {shard} ...")
        caminho = hf_hub_download(DATASET, shard, repo_type="dataset")
        caminhos.append(Path(caminho))
    return caminhos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--por-grupo", type=int, default=40,
                        help="quantos audios salvar de cada grupo (default: 40)")
    parser.add_argument("--destino", type=Path, default=DESTINO)
    args = parser.parse_args()

    pasta_audio = args.destino / "audio"
    pasta_audio.mkdir(parents=True, exist_ok=True)

    print(f"Obtendo shards de {DATASET} (cache do huggingface_hub)...")
    shards = baixar_shards()

    alvo = {"dysarthria": args.por_grupo, "healthy": args.por_grupo}
    salvos: Counter[str] = Counter()
    linhas = []

    print("\nAmostrando...")
    for shard in shards:
        if all(salvos[g] >= n for g, n in alvo.items()):
            break

        arquivo_parquet = pq.ParquetFile(shard)
        for lote in arquivo_parquet.iter_batches(batch_size=256, columns=COLUNAS):
            if all(salvos[g] >= n for g, n in alvo.items()):
                break

            for registro in lote.to_pylist():
                grupo = registro["speech_status"]
                if grupo not in alvo or salvos[grupo] >= alvo[grupo]:
                    continue
                if registro["duration"] < DURACAO_MINIMA_S:
                    continue

                audio = registro["audio"]
                nome = Path(audio["path"]).stem
                caminho = pasta_audio / f"{grupo}__{nome}.wav"
                caminho.write_bytes(audio["bytes"])

                linhas.append({
                    "arquivo": caminho.name,
                    "grupo": grupo,
                    "genero": registro["gender"],
                    "duracao_s": round(float(registro["duration"]), 3),
                    "transcricao": registro["transcription"],
                })
                salvos[grupo] += 1
                print(f"  [{sum(salvos.values()):4d}] {caminho.name}")

    if not linhas:
        raise SystemExit("Nenhum audio extraido - verifique o formato do dataset.")

    manifesto = args.destino / "manifesto.csv"
    with manifesto.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"\n{len(linhas)} audios em {pasta_audio}")
    for grupo, n in sorted(salvos.items()):
        print(f"  {grupo}: {n}")
    print(f"Manifesto: {manifesto}")


if __name__ == "__main__":
    main()
