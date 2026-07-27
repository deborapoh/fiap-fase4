"""Deteccao de movimento fora do padrao no Keraal.

Duas sinais entram no escore:

1. Distancia ao perfil saudavel (G2A) por exercicio — media dos |z-scores|
   das features angulares. E o componente "fora do padrao esperado".
2. Classificador logistico treinado no consenso dos medicos (G1A) — o Keraal
   tem rotulo verdadeiro, ao contrario do BIDMC, entao a validacao e
   supervisionada com validacao cruzada estratificada.

O escore final em [0, 1] e a probabilidade do classificador. O limiar de
alerta e 0,5. Metricas reportadas sao sempre da validacao cruzada, nunca do
ajuste in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Features que discriminam Incorrect vs Correct dentro do G1A (Mann-Whitney
# exploratorio) e as que separam G1A de G2A. Amplitude/velocidade de cotovelo
# caem no erro; tronco oscila mais.
FEATURES = [
    "ombro_esq_media",
    "cotovelo_dir_velocidade",
    "cotovelo_esq_velocidade",
    "cotovelo_dir_amplitude",
    "tronco_amplitude",
    "tronco_desvio",
    "tronco_velocidade",
    "ombro_dir_desvio",
    "ombro_dir_velocidade",
    "cotovelo_esq_media",
    "ombro_esq_amplitude",
    "ombro_dir_amplitude",
]

LIMIAR_ALERTA = 0.5


@dataclass
class PerfilExercicio:
    exercicio: str
    medias: pd.Series
    desvios: pd.Series
    n_controle: int


def construir_perfis(features: pd.DataFrame) -> dict[str, PerfilExercicio]:
    perfis = {}
    controle = features[features["grupo"] == "G2A"]
    for exercicio, lote in controle.groupby("exercicio"):
        cols = [c for c in FEATURES if c in lote.columns]
        bloco = lote[cols].apply(pd.to_numeric, errors="coerce")
        medias = bloco.mean()
        desvios = bloco.std(ddof=0).replace(0, np.nan)
        perfis[exercicio] = PerfilExercicio(
            exercicio=exercicio, medias=medias, desvios=desvios,
            n_controle=len(lote),
        )
    return perfis


def distancia_controle(linha: pd.Series, perfil: PerfilExercicio) -> float:
    zs = []
    for col in perfil.medias.index:
        valor = linha.get(col)
        media, desvio = perfil.medias[col], perfil.desvios[col]
        if pd.isna(valor) or pd.isna(media) or pd.isna(desvio) or desvio < 1e-6:
            continue
        zs.append(abs((float(valor) - float(media)) / float(desvio)))
    return float(np.mean(zs)) if zs else 0.0


def _matriz(features: pd.DataFrame, perfis: dict[str, PerfilExercicio]
            ) -> tuple[pd.DataFrame, list[str]]:
    """Monta a matriz de entrada: features angulares + distancia ao G2A."""
    dist = []
    for _, linha in features.iterrows():
        perfil = perfis.get(linha["exercicio"])
        dist.append(distancia_controle(linha, perfil) if perfil else 0.0)
    cols = [c for c in FEATURES if c in features.columns]
    X = features[cols].apply(pd.to_numeric, errors="coerce").copy()
    X["distancia_controle"] = dist
    X = X.fillna(X.median(numeric_only=True))
    return X, cols + ["distancia_controle"]


def _modelo() -> Pipeline:
    return Pipeline([
        ("escala", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def pontuar(features: pd.DataFrame, anotacoes: pd.DataFrame,
            perfis: dict[str, PerfilExercicio] | None = None
            ) -> tuple[pd.DataFrame, dict]:
    """Treina no consenso G1A e devolve escore para todas as gravacoes.

    Retorna (tabela com escores, metricas da validacao cruzada).
    """
    if perfis is None:
        perfis = construir_perfis(features)

    X_all, _ = _matriz(features, perfis)
    base = features[["identificador", "grupo", "exercicio"]].copy()
    base = base.merge(
        anotacoes[["identificador", "tem_erro", "consenso", "avaliacao"]],
        on="identificador", how="left",
    )
    base["distancia_controle"] = X_all["distancia_controle"].to_numpy()

    # Treino: so G1A com consenso (rotulo confiavel).
    mask_treino = (base["grupo"] == "G1A") & base["consenso"].fillna(False)
    X_tr = X_all.loc[mask_treino]
    y_tr = base.loc[mask_treino, "tem_erro"].astype(int)

    metricas = _validacao_cruzada(X_tr, y_tr)

    modelo = _modelo()
    modelo.fit(X_tr, y_tr)
    proba = modelo.predict_proba(X_all)[:, 1]

    base["escore_desvio"] = proba
    # Distancia bruta normalizada so para inspecao/relatorio.
    d = base["distancia_controle"]
    base["distancia_norm"] = (d / d.quantile(0.95)).clip(0, 1)
    base["desvio_detectado"] = base["escore_desvio"] >= LIMIAR_ALERTA
    return base, metricas


def _validacao_cruzada(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict:
    if len(X) < 20 or y.nunique() < 2:
        return {"n": int(len(X)), "aviso": "amostra insuficiente"}
    n_splits = min(folds, int(y.value_counts().min()))
    if n_splits < 2:
        return {"n": int(len(X)), "aviso": "classe rara demais para CV"}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = np.zeros(len(X))
    preds = np.zeros(len(X), dtype=int)
    for treino_idx, teste_idx in cv.split(X, y):
        modelo = _modelo()
        modelo.fit(X.iloc[treino_idx], y.iloc[treino_idx])
        probs[teste_idx] = modelo.predict_proba(X.iloc[teste_idx])[:, 1]
        preds[teste_idx] = (probs[teste_idx] >= LIMIAR_ALERTA).astype(int)

    y_np = y.to_numpy()
    out = {
        "n": int(len(X)),
        "n_com_erro": int(y_np.sum()),
        "n_sem_erro": int((1 - y_np).sum()),
        "folds": n_splits,
        "acuracia": round(float(accuracy_score(y_np, preds)), 3),
        "precisao": round(float(precision_score(y_np, preds, zero_division=0)), 3),
        "revocacao": round(float(recall_score(y_np, preds, zero_division=0)), 3),
        "f1": round(float(f1_score(y_np, preds, zero_division=0)), 3),
    }
    try:
        out["auc"] = round(float(roc_auc_score(y_np, probs)), 3)
    except ValueError:
        out["auc"] = float("nan")
    return out


def relatorio_desvios(pontuacao: pd.DataFrame, anotacoes: pd.DataFrame,
                      objetos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por gravacao para o CSV de saida e para a fusao."""
    cols_anot = ["identificador", "avaliacao_a", "avaliacao_b",
                 "n_erros_temporais", "duracao_erro_s"]
    base = pontuacao.merge(
        anotacoes[[c for c in cols_anot if c in anotacoes.columns]],
        on="identificador", how="left",
    )
    if objetos is not None and not objetos.empty:
        base = base.merge(objetos, on="identificador", how="left")
    else:
        base["escore_cena"] = 0.0

    cena = base.get("escore_cena", pd.Series(0.0, index=base.index)).fillna(0.0)
    # Probabilidade do classificador domina; cena (YOLO) entra leve.
    base["escore_risco"] = (0.9 * base["escore_desvio"] + 0.1 * cena).clip(0, 1)
    base["alerta"] = (base["escore_risco"] >= LIMIAR_ALERTA) | (cena >= 0.5)

    cols = [
        "identificador", "grupo", "exercicio", "avaliacao", "avaliacao_a",
        "avaliacao_b", "consenso", "tem_erro", "escore_desvio",
        "distancia_controle", "desvio_detectado", "escore_cena",
        "escore_risco", "alerta", "fracao_com_pessoa", "objetos",
        "n_erros_temporais", "duracao_erro_s",
    ]
    presentes = [c for c in cols if c in base.columns]
    return base[presentes].sort_values(
        ["grupo", "exercicio", "escore_risco"], ascending=[True, True, False]
    )
