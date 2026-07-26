#!/usr/bin/env python
"""Extrai uma amostra pareada do TORGO (fala disartrica x fala saudavel).

O dataset esta publicado como quatro arquivos parquet (1,5 GB no total) e vem
ordenado por locutor: todos os controles saudaveis aparecem antes dos falantes
disartricos. Por isso o modo streaming e ruim aqui, ele teria que percorrer
quase o dataset inteiro para achar o segundo grupo. Baixamos os shards (o cache
do huggingface_hub retoma downloads interrompidos) e amostramos localmente.

A amostragem e pareada de proposito. Pegar simplesmente os N primeiros de cada
grupo produz uma comparacao invalida: o TORGO fica ordenado de tal forma que
isso da uma unica falante saudavel gravada em microfone de array contra um
unico falante disartrico gravado em microfone de cabeca. Nessa configuracao a
diferenca de captacao e de sexo domina qualquer efeito da patologia, e as
metricas saem invertidas (o grupo saudavel aparenta ter voz pior porque o
microfone distante capta mais ruido de sala).

O criterio aqui controla as tres fontes de confusao:

microfone   so `headMic`, presente nos dois grupos
frase       so frases ditas pelos dois grupos, e cada frase entra com um audio
            de cada lado, entao o conteudo fonetico e identico na comparacao
falante     cota por sexo e rodizio entre os locutores, para que nenhum
            domine a amostra

Uso:
    python scripts/download_torgo.py                # 40 pares (80 audios)
    python scripts/download_torgo.py --pares 80
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.common.config import DATA_RAW  # noqa: E402  (define HF_HOME)
from huggingface_hub import hf_hub_download  # noqa: E402

DATASET = "abnerh/TORGO-database"
SHARDS = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
DESTINO = DATA_RAW / "torgo"

# Amostras muito curtas (digitos isolados, "yes"/"no") nao sustentam analise de
# fadiga ou prosodia, entao exigimos um minimo de duracao.
DURACAO_MINIMA_S = 1.5

MICROFONE = "headMic"
GRUPOS = ("dysarthria", "healthy")
SEXOS = ("female", "male")

COLUNAS_META = ["audio.path", "transcription", "speech_status", "gender",
                "duration"]
COLUNAS_AUDIO = ["audio", "transcription", "speech_status", "gender",
                 "duration"]


def baixar_shards() -> list[Path]:
    caminhos = []
    for shard in SHARDS:
        print(f"  baixando {shard} ...")
        caminhos.append(Path(hf_hub_download(DATASET, shard, repo_type="dataset")))
    return caminhos


def _falante(caminho: str) -> str:
    achado = re.match(r"([A-Z]+\d+)_", caminho)
    return achado.group(1) if achado else "?"


def ler_metadados(shards: list[Path]) -> list[dict]:
    """Le so as colunas leves, sem os bytes de audio."""
    registros = []
    for shard in shards:
        # Selecionar "audio.path" traz a subcoluna achatada como "path", sem
        # carregar os bytes do audio junto.
        tabela = pq.read_table(shard, columns=COLUNAS_META)
        for linha in tabela.to_pylist():
            caminho = linha["path"]
            if MICROFONE not in caminho or linha["duration"] < DURACAO_MINIMA_S:
                continue
            registros.append({
                "path": caminho,
                "falante": _falante(caminho),
                "grupo": linha["speech_status"],
                "genero": linha["gender"],
                "duracao": float(linha["duration"]),
                # Normalizado so para casar as frases entre os grupos; o texto
                # original vai para o manifesto.
                "chave": linha["transcription"].lower().strip().rstrip("."),
                "texto": linha["transcription"],
            })
    return registros


def selecionar_pares(registros: list[dict], pares: int) -> list[dict]:
    """Escolhe pares de audios que dizem a mesma frase, um de cada grupo.

    Alterna o sexo alvo a cada par, e dentro do sexo escolhe sempre o locutor
    menos usado ate aqui. Frases que nao tenham candidato nos dois grupos com
    o sexo da vez sao puladas.
    """
    por_frase: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for registro in registros:
        chave = (registro["grupo"], registro["genero"])
        por_frase[registro["chave"]][chave].append(registro)

    # Frases mais longas primeiro: pausa, taxa de fala e prosodia so fazem
    # sentido em enunciado com mais de uma palavra.
    frases = sorted(por_frase, key=lambda f: (-len(f.split()), f))

    usos: Counter[str] = Counter()
    escolhidos = []
    for indice, frase in enumerate(frases):
        if len(escolhidos) >= pares * len(GRUPOS):
            break

        sexo = SEXOS[indice % len(SEXOS)]
        candidatos = [por_frase[frase].get((grupo, sexo), []) for grupo in GRUPOS]
        if not all(candidatos):
            continue

        for opcoes in candidatos:
            escolhido = min(opcoes, key=lambda r: (usos[r["falante"]], r["path"]))
            usos[escolhido["falante"]] += 1
            escolhidos.append(escolhido)

    return escolhidos


def salvar_audios(shards: list[Path], selecionados: list[dict],
                  pasta_audio: Path) -> dict[str, str]:
    """Segunda passada nos shards, agora lendo os bytes so do que foi escolhido."""
    alvos = {registro["path"]: registro for registro in selecionados}
    nomes = {}

    for shard in shards:
        arquivo = pq.ParquetFile(shard)
        for lote in arquivo.iter_batches(batch_size=256, columns=COLUNAS_AUDIO):
            for linha in lote.to_pylist():
                caminho = linha["audio"]["path"]
                if caminho not in alvos:
                    continue

                grupo = linha["speech_status"]
                destino = pasta_audio / f"{grupo}__{Path(caminho).stem}.wav"
                destino.write_bytes(linha["audio"]["bytes"])
                nomes[caminho] = destino.name

    return nomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pares", type=int, default=40,
                        help="quantos pares de audio salvar (default: 40)")
    parser.add_argument("--destino", type=Path, default=DESTINO)
    args = parser.parse_args()

    pasta_audio = args.destino / "audio"
    pasta_audio.mkdir(parents=True, exist_ok=True)

    print(f"Obtendo shards de {DATASET} (cache do huggingface_hub)...")
    shards = baixar_shards()

    print("\nLendo metadados...")
    registros = ler_metadados(shards)
    print(f"  {len(registros)} audios em {MICROFONE} com {DURACAO_MINIMA_S}s ou mais")

    selecionados = selecionar_pares(registros, args.pares)
    if not selecionados:
        raise SystemExit("Nenhum par encontrado - verifique o formato do dataset.")
    print(f"  {len(selecionados) // 2} pares selecionados")

    print("\nGravando audios...")
    nomes = salvar_audios(shards, selecionados, pasta_audio)

    linhas = [{
        "arquivo": nomes[registro["path"]],
        "grupo": registro["grupo"],
        "genero": registro["genero"],
        "falante": registro["falante"],
        "microfone": MICROFONE,
        "duracao_s": round(registro["duracao"], 3),
        "transcricao": registro["texto"],
    } for registro in selecionados if registro["path"] in nomes]

    manifesto = args.destino / "manifesto.csv"
    with manifesto.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"\n{len(linhas)} audios em {pasta_audio}")
    _resumir(linhas)
    print(f"Manifesto: {manifesto}")


def _resumir(linhas: list[dict]) -> None:
    por_grupo: Counter[str] = Counter()
    por_sexo: Counter[tuple[str, str]] = Counter()
    por_falante: Counter[str] = Counter()
    for linha in linhas:
        por_grupo[linha["grupo"]] += 1
        por_sexo[(linha["grupo"], linha["genero"])] += 1
        por_falante[linha["falante"]] += 1

    for grupo in sorted(por_grupo):
        detalhe = ", ".join(f"{sexo} {por_sexo[(grupo, sexo)]}" for sexo in SEXOS)
        print(f"  {grupo}: {por_grupo[grupo]} ({detalhe})")
    print(f"  locutores: {dict(sorted(por_falante.items()))}")


if __name__ == "__main__":
    main()
