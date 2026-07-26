"""Metricas de validacao dos detectores contra as anomalias injetadas.

Duas leituras, que respondem perguntas diferentes:

por janela  ROC AUC, precisao, revocacao e F1 sobre cada janela de 30 s.
            Mede a qualidade do escore e a taxa de alarme.
por evento  fracao dos eventos injetados que produziram pelo menos um alerta.
            E o que importa na pratica: o evento precisa ser pego uma vez, nao
            em todas as janelas que ele ocupa.

O falso positivo aparece tambem como janelas falsas por hora de monitoramento,
que e como a equipe percebe o custo do detector. Precisao de 80% nao diz nada a
quem esta de plantao; tres avisos falsos por hora dizem.

Esse numero por janela e pessimista de proposito, e nao deve ser lido como
alarme na cabeceira: janela sinalizada logo depois de um evento conta como
falso positivo aqui, mas na saida real ela e absorvida pelo alerta do proprio
evento. O custo que a equipe sente esta em `alertas_fora_de_evento`, na
avaliacao por evento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.anomaly.alertas import Alerta
from src.anomaly.injecao import EventoInjetado
from src.anomaly.sinais_vitais import JANELA_S, PASSO_S

SEGUNDOS_POR_HORA = 3600


def avaliar_por_janela(escores: np.ndarray, rotulos: np.ndarray,
                       limiar: float, passo_s: int = PASSO_S) -> dict:
    escores = np.asarray(escores, dtype=float)
    rotulos = np.asarray(rotulos, dtype=int)

    previsto = (escores >= limiar).astype(int)
    precisao, revocacao, f1, _ = precision_recall_fscore_support(
        rotulos, previsto, average="binary", zero_division=0)

    negativas = int((rotulos == 0).sum())
    falsos = int(((previsto == 1) & (rotulos == 0)).sum())
    # Cada janela avanca `passo_s` segundos, entao o tempo monitorado sem
    # evento e o numero de janelas negativas vezes o passo.
    horas_normais = negativas * passo_s / SEGUNDOS_POR_HORA

    return {
        "auc": round(float(roc_auc_score(rotulos, escores)), 4)
        if len(set(rotulos)) > 1 else float("nan"),
        "precisao": round(float(precisao), 4),
        "revocacao": round(float(revocacao), 4),
        "f1": round(float(f1), 4),
        "janelas_falsas_por_hora": round(falsos / horas_normais, 2)
        if horas_normais else float("nan"),
        "n_janelas": len(rotulos),
        "n_anomalas": int(rotulos.sum()),
    }


def revocacao_por_tipo(escores: np.ndarray, tipos: list[str],
                       limiar: float) -> pd.DataFrame:
    """Fracao de janelas detectadas para cada tipo de anomalia injetada."""
    tabela = pd.DataFrame({"tipo": tipos,
                           "detectada": np.asarray(escores) >= limiar})
    tabela = tabela[tabela.tipo != ""]
    if tabela.empty:
        return pd.DataFrame(columns=["tipo", "janelas", "revocacao"])

    resumo = (tabela.groupby("tipo")
              .agg(janelas=("detectada", "size"),
                   revocacao=("detectada", "mean"))
              .reset_index())
    resumo["revocacao"] = resumo["revocacao"].round(4)
    return resumo.sort_values("revocacao", ascending=False)


def avaliar_por_evento(alertas: list[Alerta],
                       eventos: list[EventoInjetado]) -> dict:
    """Cobertura dos eventos injetados e quanto do alerta e fora deles."""
    if not eventos:
        return {}

    pegos = [_tem_alerta(evento, alertas) for evento in eventos]
    atraso = [d for d in (_atraso(e, alertas) for e in eventos) if d is not None]
    espurios = [a for a in alertas
                if not any(_sobrepoe(a, e) for e in eventos)]

    return {
        "eventos": len(eventos),
        "eventos_detectados": int(sum(pegos)),
        "revocacao_evento": round(float(np.mean(pegos)), 4),
        "atraso_mediano_s": float(np.median(atraso)) if atraso else float("nan"),
        "alertas": len(alertas),
        "alertas_fora_de_evento": len(espurios),
    }


def revocacao_evento_por_tipo(detalhe: pd.DataFrame) -> pd.DataFrame:
    """Fracao dos eventos de cada tipo que geraram pelo menos um alerta."""
    if detalhe.empty:
        return pd.DataFrame(columns=["tipo", "eventos", "detectados", "revocacao"])

    resumo = (detalhe.groupby("tipo")
              .agg(eventos=("detectado", "size"),
                   detectados=("detectado", "sum"),
                   revocacao=("detectado", "mean"))
              .reset_index())
    resumo["revocacao"] = resumo["revocacao"].round(4)
    return resumo.sort_values("revocacao", ascending=False)


def detalhar_eventos(alertas: list[Alerta],
                     eventos: list[EventoInjetado]) -> pd.DataFrame:
    linhas = []
    for evento in eventos:
        cobrindo = [a for a in alertas if _sobrepoe(a, evento)]
        linhas.append({
            **evento.to_dict(),
            "detectado": bool(cobrindo),
            "tipo_alertado": "|".join(sorted({a.tipo for a in cobrindo})),
            "escore_maximo": round(max((a.escore for a in cobrindo), default=0.0), 4),
        })
    return pd.DataFrame(linhas)


def _sobrepoe(alerta: Alerta, evento: EventoInjetado) -> bool:
    return (alerta.paciente == evento.paciente
            and alerta.inicio_s < evento.fim_s
            and evento.inicio_s < alerta.fim_s)


def _tem_alerta(evento: EventoInjetado, alertas: list[Alerta]) -> bool:
    return any(_sobrepoe(a, evento) for a in alertas)


def _atraso(evento: EventoInjetado, alertas: list[Alerta],
            janela_s: int = JANELA_S) -> float | None:
    """Quando o alerta poderia sair, contado a partir do inicio do evento.

    A janela que cobre o evento so fecha `janela_s` depois de comecar, e o
    alerta so pode ser emitido ai. Por isso o atraso conta do fim da primeira
    janela sinalizada, e nao do inicio dela, que costuma ser anterior ao
    proprio evento.
    """
    fechamentos = [a.inicio_s + janela_s for a in alertas if _sobrepoe(a, evento)]
    return min(fechamentos) - evento.inicio_s if fechamentos else None
