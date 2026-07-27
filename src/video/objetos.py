"""Deteccao de objetos e areas criticas com YOLOv8 nos MP4 do Keraal.

OpenPose cobre a postura; o YOLOv8 cobre o contexto visual da cena — pessoa,
cadeira, e quaisquer objetos que o modelo COCO reconheca. Em sessao de
fisioterapia o que interessa e: ha pessoa em cena? a pessoa permanece no
quadro? aparece objeto inesperado?

Nao rodamos o detector em todos os frames (10 fps x 30 s x 301 videos).
Amostramos `FRAMES_POR_VIDEO` frames igualmente espacados; para o requisito
de "areas criticas" isso basta e mantem a execucao em minutos, nao horas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Classes COCO que, neste contexto clinico de fisioterapia, sao relevantes.
CLASSES_CRITICAS = {
    0: "pessoa",
    56: "cadeira",
    57: "sofa",
    60: "mesa",
    63: "notebook",
    67: "celular",
    39: "garrafa",
    41: "copa",
}

FRAMES_POR_VIDEO = 5
MODELO_PADRAO = "yolov8n.pt"
CONF_MIN = 0.35


@dataclass
class DeteccaoFrame:
    tempo_s: float
    classes: list[str] = field(default_factory=list)
    n_pessoas: int = 0
    conf_pessoa_max: float = 0.0


@dataclass
class ResumoObjetos:
    identificador: str
    n_frames: int
    fracao_com_pessoa: float
    conf_pessoa_media: float
    objetos_unicos: list[str]
    sem_pessoa_em_algum_frame: bool
    escore_cena: float  # 0 = cena estavel com pessoa; 1 = problema


def _carregar_modelo(pesos: str = MODELO_PADRAO):
    from ultralytics import YOLO
    return YOLO(pesos)


def _amostrar_indices(n_frames: int, quantos: int) -> list[int]:
    if n_frames <= 0:
        return []
    if n_frames <= quantos:
        return list(range(n_frames))
    return [int(round(i)) for i in np.linspace(0, n_frames - 1, quantos)]


def analisar_video(caminho: Path, modelo=None,
                   frames_por_video: int = FRAMES_POR_VIDEO) -> list[DeteccaoFrame]:
    """Roda YOLOv8 em frames amostrados de um MP4."""
    if modelo is None:
        modelo = _carregar_modelo()
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _amostrar_indices(total, frames_por_video)
    saida: list[DeteccaoFrame] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        resultado = modelo.predict(frame, verbose=False, conf=CONF_MIN)[0]
        classes: list[str] = []
        n_pessoas = 0
        conf_pessoa = 0.0
        if resultado.boxes is not None and len(resultado.boxes):
            for box in resultado.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                nome = CLASSES_CRITICAS.get(cls_id)
                if nome is None:
                    # Mantem o nome COCO generico so para contagem de "outros".
                    nome = resultado.names.get(cls_id, str(cls_id))
                    if nome not in classes:
                        classes.append(f"outro:{nome}")
                    continue
                if nome not in classes:
                    classes.append(nome)
                if nome == "pessoa":
                    n_pessoas += 1
                    conf_pessoa = max(conf_pessoa, conf)
        saida.append(DeteccaoFrame(
            tempo_s=idx / fps,
            classes=classes,
            n_pessoas=n_pessoas,
            conf_pessoa_max=conf_pessoa,
        ))
    cap.release()
    return saida


def resumir(identificador: str, deteccoes: list[DeteccaoFrame]) -> ResumoObjetos:
    if not deteccoes:
        return ResumoObjetos(identificador, 0, 0.0, 0.0, [], True, 1.0)
    com_pessoa = [d for d in deteccoes if d.n_pessoas > 0]
    fracao = len(com_pessoa) / len(deteccoes)
    conf_media = float(np.mean([d.conf_pessoa_max for d in com_pessoa])) if com_pessoa else 0.0
    objetos = sorted({c for d in deteccoes for c in d.classes if c != "pessoa"})
    # Escore: falta de pessoa pesa; objetos "outro" somam pouco.
    outros = sum(1 for o in objetos if o.startswith("outro:"))
    escore = min(1.0, (1.0 - fracao) * 0.8 + min(outros, 3) * 0.05)
    return ResumoObjetos(
        identificador=identificador,
        n_frames=len(deteccoes),
        fracao_com_pessoa=fracao,
        conf_pessoa_media=conf_media,
        objetos_unicos=objetos,
        sem_pessoa_em_algum_frame=fracao < 1.0,
        escore_cena=escore,
    )


def analisar_varios(pares: list[tuple[str, Path]], pesos: str = MODELO_PADRAO,
                    frames_por_video: int = FRAMES_POR_VIDEO) -> pd.DataFrame:
    modelo = _carregar_modelo(pesos)
    linhas = []
    for i, (ident, caminho) in enumerate(pares, start=1):
        det = analisar_video(caminho, modelo=modelo,
                             frames_por_video=frames_por_video)
        resumo = resumir(ident, det)
        linhas.append({
            "identificador": resumo.identificador,
            "n_frames_yolo": resumo.n_frames,
            "fracao_com_pessoa": resumo.fracao_com_pessoa,
            "conf_pessoa_media": resumo.conf_pessoa_media,
            "objetos": "|".join(resumo.objetos_unicos),
            "sem_pessoa_em_algum_frame": resumo.sem_pessoa_em_algum_frame,
            "escore_cena": resumo.escore_cena,
        })
        if i % 20 == 0 or i == len(pares):
            print(f"  YOLO {i}/{len(pares)}")
    return pd.DataFrame(linhas)
