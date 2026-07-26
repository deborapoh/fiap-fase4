"""Transcricao de audio com faster-whisper.

Substitui o Azure Speech to Text previsto no enunciado. O faster-whisper e uma
reimplementacao do Whisper sobre o CTranslate2, com a mesma acuracia do modelo
original e ate 4x mais rapido.

Alem do texto, guardamos a probabilidade media de log dos segmentos. Ela cai
quando o modelo tem dificuldade de reconhecer a fala, o que por si so ja e um
indicador de inteligibilidade reduzida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# A configuracao vem antes do faster_whisper de proposito: ela define HF_HOME,
# e o huggingface_hub le essa variavel no momento em que e importado. Invertida
# a ordem, os pesos iriam para o cache padrao do sistema.
from src.common.config import (IDIOMA_WHISPER, MODELO_WHISPER,  # isort: skip
                               dispositivo_whisper)
from faster_whisper import WhisperModel  # noqa: E402

MODELO_PADRAO = MODELO_WHISPER
IDIOMA_PADRAO = IDIOMA_WHISPER


@dataclass
class Transcricao:
    arquivo: str
    texto: str
    idioma: str
    duracao_s: float
    logprob_media: float
    proporcao_sem_fala: float
    segmentos: list[dict] = field(default_factory=list)


@lru_cache(maxsize=4)
def carregar_modelo(nome: str = MODELO_PADRAO,
                    dispositivo: str | None = None,
                    tipo_computacao: str = "int8") -> WhisperModel:
    """Carrega e memoriza o modelo.

    O padrao e CPU com int8: o CTranslate2 ainda nao tem backend para o MPS do
    Apple Silicon, e nesta escala (audios de poucos segundos) a CPU quantizada
    resolve em cerca de 1 segundo por arquivo no modelo small.
    """
    return WhisperModel(nome, device=dispositivo or dispositivo_whisper(),
                        compute_type=tipo_computacao)


def transcrever(caminho: Path | str,
                modelo: WhisperModel | None = None,
                idioma: str = IDIOMA_PADRAO) -> Transcricao:
    caminho = Path(caminho)
    modelo = modelo or carregar_modelo()

    segmentos, info = modelo.transcribe(str(caminho), language=idioma, beam_size=5)

    dados = []
    for s in segmentos:
        dados.append({
            "inicio": round(s.start, 3),
            "fim": round(s.end, 3),
            "texto": s.text.strip(),
            "logprob": round(s.avg_logprob, 4),
            "sem_fala": round(s.no_speech_prob, 4),
        })

    texto = " ".join(d["texto"] for d in dados).strip()
    if dados:
        logprob = sum(d["logprob"] for d in dados) / len(dados)
        sem_fala = sum(d["sem_fala"] for d in dados) / len(dados)
    else:
        # Audio sem fala reconhecida: o valor neutro evita poluir as medias.
        logprob, sem_fala = float("nan"), 1.0

    return Transcricao(
        arquivo=caminho.name,
        texto=texto,
        idioma=info.language,
        duracao_s=round(info.duration, 3),
        logprob_media=round(logprob, 4),
        proporcao_sem_fala=round(sem_fala, 4),
        segmentos=dados,
    )
