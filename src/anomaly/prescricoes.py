"""Alteracoes inesperadas na evolucao das prescricoes do MIMIC-IV Demo.

O enunciado pede deteccao de anomalia na evolucao do tratamento, nao so nos
sinais vitais. A tabela `prescriptions` do MIMIC-IV Demo tem 18.087 ordens de
250 internacoes de 100 pacientes, com medicamento, dose, unidade, via e janela
de vigencia.

Aqui a abordagem e deliberadamente diferente da usada nos sinais vitais. La o
padrao normal e uma forma de onda, que um modelo aprende bem. Aqui o evento
que interessa e definido por regra clinica conhecida de antemao (salto de
dose, troca de via, suspensao), e a saida precisa ser auditavel: a equipe tem
que saber por que a ordem foi sinalizada. E o mesmo raciocinio que levou o
detector de termos criticos do audio para dicionario curado em vez de NER
estatistico.

As regras se dividem em duas familias. Tres sao limiares fixos com origem
clinica (salto de dose, escalonamento de via, inconsistencia temporal) e duas
sao estatisticas, comparando cada ordem com a distribuicao da propria coorte
(dose atipica pelo desvio absoluto mediano, rajada de novas prescricoes pelo
percentil). As estatisticas se recalibram sozinhas se o pipeline rodar sobre
outro hospital, o que as tres primeiras nao fazem.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.regras import PESOS, SATURACAO_ESCORE
from src.common.config import DATA_RAW

PASTA_MIMIC = (DATA_RAW / "mimic-iv-demo"
               / "mimic-iv-clinical-database-demo-2.2" / "hosp")

# Salto de dose: cinco vezes para cima ou para baixo entre ordens consecutivas
# do mesmo medicamento e na mesma unidade. Abaixo disso a maior parte do que
# aparece e titulacao de rotina (dobrar a dose responde por um terco dos pares
# consecutivos desta coorte, o que nao e evento).
LIMIAR_SALTO_DOSE = 5.0
JANELA_SALTO_H = 48

# Dose atipica pelo desvio absoluto mediano, que nao e arrastado pelo proprio
# outlier como a media e o desvio padrao seriam.
LIMIAR_Z_ROBUSTO = 5.0
MINIMO_ORDENS_REFERENCIA = 20

JANELA_RAJADA = "6h"
PERCENTIL_RAJADA = 99

VIAS_ORAIS = {"PO", "PO/NG", "ORAL", "SL", "PO/OG", "NG", "PO/PR"}
VIAS_PARENTERAIS = {"IV", "IV DRIP", "IV BOLUS", "IM", "IVPCA"}
JANELA_TROCA_VIA_H = 24

# Lista de alto risco do ISMP, reduzida ao que aparece nesta coorte. Nao muda
# a deteccao: agrava o evento ja detectado, porque erro de dose nestes
# medicamentos tem consequencia desproporcional.
MEDICAMENTOS_ALTO_RISCO = (
    "insulin", "heparin", "warfarin", "morphine", "fentanyl", "hydromorphone",
    "midazolam", "propofol", "potassium chloride", "digoxin", "amiodarone",
    "norepinephrine", "epinephrine", "dopamine", "dobutamine", "vasopressin",
    "argatroban", "enoxaparin", "methadone", "oxycodone",
)

SEVERIDADE_POR_TIPO = {
    "salto_de_dose": "media",
    "dose_atipica": "media",
    "escalonamento_de_via": "media",
    "rajada_de_prescricoes": "media",
    "inconsistencia_temporal": "baixa",
}

# Peso extra do medicamento de alto risco no escore do evento.
FATOR_ALTO_RISCO = 1.5


@dataclass
class EventoPrescricao:
    subject_id: int
    hadm_id: int
    momento: str
    medicamento: str
    tipo: str
    detalhe: str
    severidade: str
    alto_risco: bool
    escore: float

    def to_dict(self) -> dict:
        return asdict(self)


def carregar(pasta: Path | None = None) -> pd.DataFrame:
    """Le a tabela de prescricoes e acrescenta dose numerica e marca de risco.

    O pandas abre o `.csv.gz` direto, sem descompactar em disco.
    """
    caminho = (pasta or PASTA_MIMIC) / "prescriptions.csv.gz"
    dados = pd.read_csv(caminho, parse_dates=["starttime", "stoptime"])

    dados["dose"] = dados["dose_val_rx"].map(_primeiro_numero)
    dados["alto_risco"] = dados["drug"].str.lower().str.contains(
        "|".join(MEDICAMENTOS_ALTO_RISCO), na=False)
    return dados.sort_values("starttime").reset_index(drop=True)


def detectar(prescricoes: pd.DataFrame) -> list[EventoPrescricao]:
    eventos: list[EventoPrescricao] = []
    eventos += _saltos_de_dose(prescricoes)
    eventos += _doses_atipicas(prescricoes)
    eventos += _escalonamentos_de_via(prescricoes)
    eventos += _rajadas(prescricoes)
    eventos += _inconsistencias_temporais(prescricoes)
    return sorted(eventos, key=lambda e: (e.hadm_id, e.momento))


def _primeiro_numero(valor) -> float:
    """Extrai o numero da dose, que vem como texto e as vezes como faixa."""
    if pd.isna(valor):
        return float("nan")
    achado = re.match(r"\s*([0-9]*\.?[0-9]+)", str(valor))
    return float(achado.group(1)) if achado else float("nan")


def _evento(linha, tipo: str, detalhe: str) -> EventoPrescricao:
    severidade = SEVERIDADE_POR_TIPO[tipo]
    alto_risco = bool(linha.alto_risco)
    peso = PESOS[severidade] * (FATOR_ALTO_RISCO if alto_risco else 1.0)

    return EventoPrescricao(
        subject_id=int(linha.subject_id),
        hadm_id=int(linha.hadm_id),
        momento=str(linha.starttime),
        medicamento=str(linha.drug),
        tipo=tipo,
        detalhe=detalhe,
        severidade="alta" if (alto_risco and severidade == "media") else severidade,
        alto_risco=alto_risco,
        escore=round(min(1.0, peso / SATURACAO_ESCORE), 4),
    )


def _saltos_de_dose(prescricoes: pd.DataFrame) -> list[EventoPrescricao]:
    """Mudanca abrupta de dose entre ordens consecutivas do mesmo medicamento."""
    eventos = []
    validas = prescricoes.dropna(subset=["dose"])

    for _, grupo in validas.groupby(["hadm_id", "drug", "dose_unit_rx"],
                                    observed=True):
        if len(grupo) < 2:
            continue

        grupo = grupo.sort_values("starttime")
        anterior = None
        for linha in grupo.itertuples():
            if anterior is not None and anterior.dose > 0:
                razao = linha.dose / anterior.dose
                horas = (linha.starttime - anterior.starttime).total_seconds() / 3600
                if (razao >= LIMIAR_SALTO_DOSE or razao <= 1 / LIMIAR_SALTO_DOSE) \
                        and horas <= JANELA_SALTO_H:
                    sentido = "aumento" if razao > 1 else "reducao"
                    eventos.append(_evento(
                        linha, "salto_de_dose",
                        f"{sentido} de {anterior.dose:g} para {linha.dose:g} "
                        f"{linha.dose_unit_rx} em {horas:.0f}h"))
            anterior = linha

    return eventos


def _doses_atipicas(prescricoes: pd.DataFrame) -> list[EventoPrescricao]:
    """Dose distante do que a coorte usa daquele medicamento naquela unidade."""
    eventos = []
    validas = prescricoes.dropna(subset=["dose"])

    for _, grupo in validas.groupby(["drug", "dose_unit_rx"], observed=True):
        if len(grupo) < MINIMO_ORDENS_REFERENCIA:
            continue

        doses = grupo["dose"].to_numpy()
        mediana = float(np.median(doses))
        desvio = float(np.median(np.abs(doses - mediana)))
        if desvio == 0:
            continue

        # 0.6745 converte o desvio absoluto mediano na escala do desvio padrao
        # de uma normal, o que mantem o limiar com a leitura usual de z.
        z = 0.6745 * (doses - mediana) / desvio
        for linha, escore_z in zip(grupo.itertuples(), z):
            if abs(escore_z) >= LIMIAR_Z_ROBUSTO:
                eventos.append(_evento(
                    linha, "dose_atipica",
                    f"{linha.dose:g} {linha.dose_unit_rx} contra mediana de "
                    f"{mediana:g} na coorte (z robusto {escore_z:+.1f})"))

    return eventos


def _escalonamentos_de_via(prescricoes: pd.DataFrame) -> list[EventoPrescricao]:
    """Troca de via oral para parenteral no mesmo medicamento.

    Costuma acompanhar piora: o paciente deixou de absorver, deixou de aceitar
    via oral ou passou a precisar de efeito mais rapido.
    """
    eventos = []
    com_via = prescricoes.dropna(subset=["route"])

    for _, grupo in com_via.groupby(["hadm_id", "drug"], observed=True):
        if len(grupo) < 2:
            continue

        grupo = grupo.sort_values("starttime")
        anterior = None
        for linha in grupo.itertuples():
            if anterior is not None:
                horas = (linha.starttime - anterior.starttime).total_seconds() / 3600
                if (anterior.route in VIAS_ORAIS and linha.route in VIAS_PARENTERAIS
                        and horas <= JANELA_TROCA_VIA_H):
                    eventos.append(_evento(
                        linha, "escalonamento_de_via",
                        f"{anterior.route} para {linha.route} em {horas:.0f}h"))
            anterior = linha

    return eventos


def _rajadas(prescricoes: pd.DataFrame) -> list[EventoPrescricao]:
    """Concentracao incomum de medicamentos novos numa janela de 6 horas.

    Conta a primeira ordem de cada medicamento na internacao, e nao todas as
    ordens: represcricao de rotina do que ja estava em uso nao e mudanca de
    tratamento. O limiar e o percentil da propria coorte, entao o que se
    sinaliza e a internacao que destoa das outras, nao um numero arbitrario.
    """
    primeiras = (prescricoes.groupby(["hadm_id", "drug"], observed=True)
                 .agg(starttime=("starttime", "min"),
                      subject_id=("subject_id", "first"),
                      alto_risco=("alto_risco", "first"))
                 .reset_index())

    contagens = {}
    for hadm, grupo in primeiras.groupby("hadm_id", observed=True):
        serie = grupo.set_index("starttime").sort_index()["drug"]
        contagens[hadm] = serie.rolling(JANELA_RAJADA).count()

    if not contagens:
        return []

    todas = np.concatenate([c.to_numpy() for c in contagens.values()])
    limiar = float(np.percentile(todas, PERCENTIL_RAJADA))

    eventos = []
    for hadm, contagem in contagens.items():
        if contagem.max() < limiar:
            continue

        # Um evento por internacao, no pico da janela: a mesma rajada aparece
        # em varias janelas consecutivas e viraria alerta repetido.
        momento = contagem.idxmax()
        pico = int(contagem.max())
        linha = primeiras[(primeiras.hadm_id == hadm)
                          & (primeiras.starttime == momento)].iloc[0]
        eventos.append(_evento(
            linha, "rajada_de_prescricoes",
            f"{pico} medicamentos novos em {JANELA_RAJADA} "
            f"(percentil {PERCENTIL_RAJADA} da coorte: {limiar:.0f})"))

    return eventos


def _inconsistencias_temporais(prescricoes: pd.DataFrame) -> list[EventoPrescricao]:
    """Ordem com termino anterior ao inicio.

    Aparece quando a prescricao foi revista ou cancelada depois de emitida. Nao
    e deterioracao clinica, e por isso entra com severidade baixa, mas e
    exatamente o tipo de alteracao inesperada de tratamento que o enunciado
    pede para rastrear, e vale como controle de qualidade da prescricao.
    """
    invertidas = prescricoes[prescricoes.stoptime < prescricoes.starttime]
    eventos = []
    for linha in invertidas.itertuples():
        horas = (linha.starttime - linha.stoptime).total_seconds() / 3600
        eventos.append(_evento(
            linha, "inconsistencia_temporal",
            f"termino {horas:.0f}h antes do inicio"))
    return eventos


def para_dataframe(eventos: list[EventoPrescricao]) -> pd.DataFrame:
    if not eventos:
        return pd.DataFrame(columns=list(EventoPrescricao.__annotations__))
    return pd.DataFrame([e.to_dict() for e in eventos])


def resumir_por_internacao(eventos: list[EventoPrescricao],
                           prescricoes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por internacao, no formato que a fusao multimodal consome.

    O escore da internacao e o maior entre a gravidade do pior evento e a
    densidade de eventos por dia internado. Somar todos os eventos, que era o
    caminho obvio, satura em 1 para dois tercos da coorte: internacao longa
    acumula ocorrencia por tempo de exposicao e nao por gravidade. Com as duas
    parcelas, um unico salto de dose de insulina pesa mesmo isolado, e a
    internacao cheia de eventos leves so sobe se eles forem frequentes.
    """
    dias = _dias_internado(prescricoes)

    por_internacao: dict[tuple[int, int], list[EventoPrescricao]] = {}
    for evento in eventos:
        por_internacao.setdefault((evento.subject_id, evento.hadm_id), []).append(evento)

    linhas = []
    for (subject_id, hadm_id), lista in sorted(por_internacao.items()):
        tipos = Counter(e.tipo for e in lista)
        pesos = [PESOS[e.severidade] * (FATOR_ALTO_RISCO if e.alto_risco else 1.0)
                 for e in lista]
        internado = max(1.0, dias.get(hadm_id, 1.0))
        escore = max(max(pesos) / SATURACAO_ESCORE,
                     sum(pesos) / (SATURACAO_ESCORE * internado))

        linhas.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "n_eventos": len(lista),
            "dias_internado": round(internado, 1),
            "escore_risco": round(min(1.0, escore), 4),
            "severidade_maxima": _pior_severidade(lista),
            "tipos": "|".join(t for t, _ in tipos.most_common()),
        })
    return pd.DataFrame(linhas)


def _dias_internado(prescricoes: pd.DataFrame | None) -> dict[int, float]:
    """Duracao da internacao pelo intervalo coberto pelas proprias ordens."""
    if prescricoes is None:
        return {}

    limites = prescricoes.groupby("hadm_id")["starttime"].agg(["min", "max"])
    duracao = (limites["max"] - limites["min"]).dt.total_seconds() / 86400
    return duracao.to_dict()


def _pior_severidade(eventos: list[EventoPrescricao]) -> str:
    severidades = {e.severidade for e in eventos}
    for nivel in ("alta", "media", "baixa"):
        if nivel in severidades:
            return nivel
    return "nenhuma"
