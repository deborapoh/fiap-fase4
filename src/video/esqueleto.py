"""Consumo dos JSONs OpenPose do Keraal e angulos articulares.

O enunciado pede OpenPose. O Keraal ja traz os keypoints por frame no formato
nomeado (Head, lShoulder, rElbow, ...), derivados do OpenPose COCO. Consumimos
esses JSONs diretamente, sem compilar o OpenPose — decisao documentada no
handoff. Coordenadas vem normalizadas em [0, 1] relativamente ao frame.

Os angulos que sustentam a comparacao paciente vs saudavel:

cotovelo     ombro-cotovelo-pulso, esquerda e direita
ombro        quadril-ombro-cotovelo (elevacao do braco)
tronco       inclinacao da reta mShoulder -> ponto medio dos quadris
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import DATA_RAW

PASTA_OPENPOSE = DATA_RAW / "keraal" / "openpose"
PASTA_VIDEOS = DATA_RAW / "keraal" / "videos"
FPS = 10.0

# Triplas (proximal, articulacao, distal) para angulo interno em graus.
ARTICULACOES = {
    "cotovelo_esq": ("lShoulder", "lElbow", "lWrist"),
    "cotovelo_dir": ("rShoulder", "rElbow", "rWrist"),
    "ombro_esq": ("lHip", "lShoulder", "lElbow"),
    "ombro_dir": ("rHip", "rShoulder", "rElbow"),
    "joelho_esq": ("lHip", "lKnee", "lAnkle"),
    "joelho_dir": ("rHip", "rKnee", "rAnkle"),
}

PADRAO_OP = re.compile(
    r"^(?P<grupo>G[12]A)-OP-(?P<exercicio>CTK|ELK|RTK)-(?P<resto>.+)\.json$"
)


@dataclass
class SerieEsqueleto:
    identificador: str
    grupo: str
    exercicio: str
    tempos_s: np.ndarray
    angulos: dict[str, np.ndarray]  # nome -> serie em graus, NaN se junta ausente
    caminho_video: Path | None


def identificador_de_openpose(nome: str) -> str | None:
    """G1A-OP-CTK-R1-Brest-022.json -> G1A-CTK-R1-Brest-022."""
    match = PADRAO_OP.match(nome)
    if not match:
        return None
    return f"{match.group('grupo')}-{match.group('exercicio')}-{match.group('resto')}"


def caminho_video(identificador: str) -> Path | None:
    """G1A-CTK-R1-Brest-022 -> videos/G1A-Anon-CTK-R1-Brest-022.mp4."""
    match = re.match(r"^(G[12]A)-(CTK|ELK|RTK)-(.+)$", identificador)
    if not match:
        return None
    nome = f"{match.group(1)}-Anon-{match.group(2)}-{match.group(3)}.mp4"
    caminho = PASTA_VIDEOS / nome
    return caminho if caminho.exists() else None


def caminho_openpose(identificador: str) -> Path | None:
    match = re.match(r"^(G[12]A)-(CTK|ELK|RTK)-(.+)$", identificador)
    if not match:
        return None
    nome = f"{match.group(1)}-OP-{match.group(2)}-{match.group(3)}.json"
    caminho = PASTA_OPENPOSE / nome
    return caminho if caminho.exists() else None


def _angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angulo interno ABC em graus. NaN se algum ponto e invalido."""
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return float("nan")
    ba = a - b
    bc = c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return float("nan")
    cos = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _ponto(frame: dict, nome: str) -> np.ndarray:
    xy = frame.get(nome)
    if xy is None or len(xy) < 2:
        return np.array([np.nan, np.nan])
    return np.asarray(xy[:2], dtype=float)


def _inclinacao_tronco(frame: dict) -> float:
    """Angulo do tronco com a vertical, em graus (0 = ereto)."""
    ombro = _ponto(frame, "mShoulder")
    l_hip, r_hip = _ponto(frame, "lHip"), _ponto(frame, "rHip")
    if not (np.isfinite(ombro).all() and np.isfinite(l_hip).all()
            and np.isfinite(r_hip).all()):
        return float("nan")
    quadril = (l_hip + r_hip) / 2.0
    vetor = ombro - quadril
    # Vertical da imagem aponta para cima (y diminui); usamos -y.
    vertical = np.array([0.0, -1.0])
    n = np.linalg.norm(vetor)
    if n < 1e-8:
        return float("nan")
    cos = float(np.clip(np.dot(vetor / n, vertical), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def carregar(identificador: str) -> SerieEsqueleto | None:
    caminho = caminho_openpose(identificador)
    if caminho is None:
        return None
    match = re.match(r"^(G[12]A)-(CTK|ELK|RTK)-", identificador)
    if not match:
        return None
    bruto = json.loads(caminho.read_text(encoding="utf-8"))
    posicoes = bruto.get("positions") or {}
    if not posicoes:
        return None

    chaves = sorted(posicoes.keys(), key=float)
    # Indices do Keraal comecam em 1.0; tempo = (frame-1)/fps.
    tempos = np.array([(float(k) - 1.0) / FPS for k in chaves], dtype=float)
    angulos: dict[str, list[float]] = {n: [] for n in ARTICULACOES}
    angulos["tronco"] = []

    for chave in chaves:
        frame = posicoes[chave]
        for nome, (p, a, d) in ARTICULACOES.items():
            angulos[nome].append(_angulo(_ponto(frame, p), _ponto(frame, a),
                                         _ponto(frame, d)))
        angulos["tronco"].append(_inclinacao_tronco(frame))

    return SerieEsqueleto(
        identificador=identificador,
        grupo=match.group(1),
        exercicio=match.group(2),
        tempos_s=tempos,
        angulos={k: np.asarray(v, dtype=float) for k, v in angulos.items()},
        caminho_video=caminho_video(identificador),
    )


def resumo_features(serie: SerieEsqueleto) -> dict:
    """Estatisticas por angulo: media, desvio, amplitude e velocidade media."""
    out = {
        "identificador": serie.identificador,
        "grupo": serie.grupo,
        "exercicio": serie.exercicio,
        "duracao_s": float(serie.tempos_s[-1]) if len(serie.tempos_s) else 0.0,
        "n_frames": int(len(serie.tempos_s)),
    }
    for nome, serie_ang in serie.angulos.items():
        validos = serie_ang[np.isfinite(serie_ang)]
        if len(validos) < 3:
            out[f"{nome}_media"] = float("nan")
            out[f"{nome}_desvio"] = float("nan")
            out[f"{nome}_amplitude"] = float("nan")
            out[f"{nome}_velocidade"] = float("nan")
            continue
        out[f"{nome}_media"] = float(np.mean(validos))
        out[f"{nome}_desvio"] = float(np.std(validos))
        out[f"{nome}_amplitude"] = float(np.ptp(validos))
        # Velocidade media absoluta entre frames consecutivos validos.
        diffs = np.abs(np.diff(serie_ang))
        diffs = diffs[np.isfinite(diffs)]
        out[f"{nome}_velocidade"] = float(np.mean(diffs) * FPS) if len(diffs) else float("nan")
    return out


def features_todas(identificadores: list[str] | None = None) -> pd.DataFrame:
    if identificadores is None:
        identificadores = []
        for caminho in sorted(PASTA_OPENPOSE.glob("*.json")):
            ident = identificador_de_openpose(caminho.name)
            if ident:
                identificadores.append(ident)
    linhas = []
    for ident in identificadores:
        serie = carregar(ident)
        if serie is not None:
            linhas.append(resumo_features(serie))
    return pd.DataFrame(linhas)
