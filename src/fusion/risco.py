"""Fusao multimodal: um escore de risco por paciente sintetico.

As tres frentes do trabalho nao compartilham identificador de pessoa:

  audio     locutor do TORGO (disartria vs controle)
  vitais    paciente do BIDMC
  video     gravacao do Keraal (lombalgia vs saudavel)
  prescricoes  internacao do MIMIC-IV Demo

A saida honesta e montar um *paciente sintetico* que combina uma amostra de
cada frente, declarar isso no relatorio, e so entao fundir os escores. Nao ha
como inventar um join real entre essas bases publicas.

Cada frente ja expoe um escore em [0, 1] na mesma escala de severidade
(pesos alta/media/baixa e saturacao compartilhados entre audio e anomalias).
A fusao e uma media ponderada + regra de alerta: qualquer frente em alerta
alto dispara, mesmo que as outras estejam baixas — e o que a equipe medica
precisa, nao a media diluida.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import DATA_PROCESSED

# Pesos da fusao. Video e vitais pesam mais porque sao as frentes com
# validacao quantitativa mais forte neste trabalho; audio entra como
# proxy de alteracao vocal; prescricoes como sinal de evolucao do tratamento.
PESOS = {
    "audio": 0.20,
    "vitais": 0.30,
    "video": 0.30,
    "prescricoes": 0.20,
}

LIMIAR_ALERTA = 0.55
LIMIAR_ALTO = 0.75


@dataclass
class PacienteSintetico:
    id_sintetico: str
    id_audio: str | None
    id_vitais: str | None
    id_video: str | None
    id_prescricao: str | None
    escore_audio: float
    escore_vitais: float
    escore_video: float
    escore_prescricoes: float
    escore_risco: float
    alerta: bool
    severidade: str
    motivos: list[str]


def _carregar_csv(nome: str) -> pd.DataFrame:
    caminho = DATA_PROCESSED / nome
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho)


def carregar_frentes() -> dict[str, pd.DataFrame]:
    """Le os CSVs que cada pipeline escreve como interface da fusao."""
    return {
        "audio": _carregar_csv("audio_metricas.csv"),
        "vitais": _carregar_csv("anomalias_vitais_resumo.csv"),
        "video": _carregar_csv("video_desvios_resumo.csv"),
        "prescricoes": _carregar_csv("anomalias_prescricoes_resumo.csv"),
    }


def _escore_audio(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["id", "escore", "alerta", "detalhe"])
    out = pd.DataFrame({
        "id": df["arquivo"].astype(str),
        "escore": df["escore_criticidade"].fillna(0).clip(0, 1),
        # No TORGO o sinal clinico forte e a disartria (WER alto + grupo).
        # Criticidade de termos quase nao dispara; usamos tambem um proxy
        # acustico: WER normalizado + grupo dysarthria.
        "grupo": df.get("grupo", pd.Series("", index=df.index)),
        "wer": df.get("wer", pd.Series(0.0, index=df.index)).fillna(0),
    })
    # Combina criticidade textual com proxy vocal (WER saturado em 0.5 = 1.0).
    proxy_vocal = (out["wer"] / 0.5).clip(0, 1)
    caso = (out["grupo"].astype(str) == "dysarthria").astype(float) * 0.3
    out["escore"] = (0.4 * out["escore"] + 0.4 * proxy_vocal + 0.2 * caso).clip(0, 1)
    out["alerta"] = out["escore"] >= LIMIAR_ALERTA
    out["detalhe"] = out.apply(
        lambda r: f"grupo={r['grupo']} wer={r['wer']:.2f}", axis=1
    )
    return out[["id", "escore", "alerta", "detalhe"]]


def _escore_vitais(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["id", "escore", "alerta", "detalhe"])
    id_col = "paciente" if "paciente" in df.columns else df.columns[0]
    out = pd.DataFrame({
        "id": df[id_col].astype(str),
        "escore": df["escore_risco"].fillna(0).clip(0, 1),
    })
    out["alerta"] = out["escore"] >= LIMIAR_ALERTA
    n = df["n_alertas"] if "n_alertas" in df.columns else 0
    out["detalhe"] = [f"n_alertas={v}" for v in n] if not isinstance(n, int) else "vitais"
    return out


def _escore_video(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["id", "escore", "alerta", "detalhe"])
    # So pacientes (G1A) entram na fusao como sinal clinico; controle e baseline.
    if "grupo" in df.columns:
        df = df[df["grupo"] == "G1A"].copy()
    out = pd.DataFrame({
        "id": df["identificador"].astype(str),
        "escore": df["escore_risco"].fillna(0).clip(0, 1),
        "alerta": df.get("alerta", df["escore_risco"] >= LIMIAR_ALERTA),
        "detalhe": df.apply(
            lambda r: f"{r.get('exercicio','')} eval={r.get('avaliacao','')}",
            axis=1,
        ),
    })
    return out


def _escore_prescricoes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["id", "escore", "alerta", "detalhe"])
    id_col = "hadm_id" if "hadm_id" in df.columns else df.columns[0]
    out = pd.DataFrame({
        "id": df[id_col].astype(str),
        "escore": df["escore_risco"].fillna(0).clip(0, 1),
    })
    out["alerta"] = out["escore"] >= LIMIAR_ALERTA
    out["detalhe"] = "prescricoes"
    return out


def _severidade(escore: float, alertas_frente: list[str]) -> str:
    if escore >= LIMIAR_ALTO or len(alertas_frente) >= 2:
        return "alta"
    if escore >= LIMIAR_ALERTA or alertas_frente:
        return "media"
    return "baixa"


def montar_pacientes(n: int = 20, semente: int = 42) -> list[PacienteSintetico]:
    """Emparelha N amostras de cada frente num paciente sintetico."""
    frentes = carregar_frentes()
    audio = _escore_audio(frentes["audio"])
    vitais = _escore_vitais(frentes["vitais"])
    video = _escore_video(frentes["video"])
    presc = _escore_prescricoes(frentes["prescricoes"])

    rng = np.random.default_rng(semente)
    n = int(min(n, len(audio) or n, len(vitais) or n,
                len(video) or n, len(presc) or n))
    if n == 0:
        return []

    def amostra(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        idx = rng.choice(len(df), size=n, replace=len(df) < n)
        return df.iloc[idx].reset_index(drop=True)

    a, v, vd, p = amostra(audio), amostra(vitais), amostra(video), amostra(presc)
    pacientes = []
    for i in range(n):
        escores = {
            "audio": float(a.iloc[i]["escore"]) if len(a) else 0.0,
            "vitais": float(v.iloc[i]["escore"]) if len(v) else 0.0,
            "video": float(vd.iloc[i]["escore"]) if len(vd) else 0.0,
            "prescricoes": float(p.iloc[i]["escore"]) if len(p) else 0.0,
        }
        alertas = []
        if len(a) and bool(a.iloc[i]["alerta"]):
            alertas.append("audio")
        if len(v) and bool(v.iloc[i]["alerta"]):
            alertas.append("vitais")
        if len(vd) and bool(vd.iloc[i]["alerta"]):
            alertas.append("video")
        if len(p) and bool(p.iloc[i]["alerta"]):
            alertas.append("prescricoes")

        risco = sum(PESOS[k] * escores[k] for k in PESOS)
        # Regra de alerta: media ponderada alta OU qualquer frente em alerta
        # com escore individual alto — evita diluir um sinal critico.
        alerta = risco >= LIMIAR_ALERTA or any(
            escores[k] >= LIMIAR_ALTO for k in alertas
        ) or len(alertas) >= 2

        motivos = []
        for nome, df_i, key in (
            ("audio", a, "audio"), ("vitais", v, "vitais"),
            ("video", vd, "video"), ("prescricoes", p, "prescricoes"),
        ):
            if not len(df_i):
                continue
            if nome in alertas or escores[key] >= LIMIAR_ALERTA:
                motivos.append(f"{nome}:{escores[key]:.2f} ({df_i.iloc[i]['detalhe']})")

        pacientes.append(PacienteSintetico(
            id_sintetico=f"SYN-{i+1:03d}",
            id_audio=str(a.iloc[i]["id"]) if len(a) else None,
            id_vitais=str(v.iloc[i]["id"]) if len(v) else None,
            id_video=str(vd.iloc[i]["id"]) if len(vd) else None,
            id_prescricao=str(p.iloc[i]["id"]) if len(p) else None,
            escore_audio=escores["audio"],
            escore_vitais=escores["vitais"],
            escore_video=escores["video"],
            escore_prescricoes=escores["prescricoes"],
            escore_risco=float(risco),
            alerta=bool(alerta),
            severidade=_severidade(risco, alertas),
            motivos=motivos,
        ))
    return pacientes


def para_dataframe(pacientes: list[PacienteSintetico]) -> pd.DataFrame:
    linhas = []
    for p in pacientes:
        linhas.append({
            "id_sintetico": p.id_sintetico,
            "id_audio": p.id_audio,
            "id_vitais": p.id_vitais,
            "id_video": p.id_video,
            "id_prescricao": p.id_prescricao,
            "escore_audio": round(p.escore_audio, 4),
            "escore_vitais": round(p.escore_vitais, 4),
            "escore_video": round(p.escore_video, 4),
            "escore_prescricoes": round(p.escore_prescricoes, 4),
            "escore_risco": round(p.escore_risco, 4),
            "alerta": p.alerta,
            "severidade": p.severidade,
            "motivos": " | ".join(p.motivos),
        })
    return pd.DataFrame(linhas)


def gerar(saida: Path | None = None, n: int = 20, semente: int = 42) -> pd.DataFrame:
    saida = saida or DATA_PROCESSED
    saida.mkdir(parents=True, exist_ok=True)
    df = para_dataframe(montar_pacientes(n=n, semente=semente))
    df.to_csv(saida / "fusao_risco.csv", index=False)
    alertas = df[df["alerta"]].copy()
    alertas.to_csv(saida / "fusao_alertas.csv", index=False)
    return df
