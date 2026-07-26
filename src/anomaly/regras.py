"""Limiares clinicos de alarme aplicados a cada janela de sinais vitais.

Os dois detectores estatisticos dizem que uma janela e atipica, mas nao dizem
o que ha de errado nela. As regras daqui cumprem tres papeis que o modelo nao
cumpre:

nomear      o alerta chega a equipe como "hipoxemia" e nao como "escore 0,83".
garantir    valor perigoso e sinalizado mesmo quando e comum no paciente e por
            isso passaria por normal para um detector nao supervisionado.
comparar    servem de baseline: o ganho do modelo e o que ele acha alem delas.

Os limiares seguem faixas usuais de alarme de monitor de cabeceira para
adulto. Nao substituem protocolo institucional, e o relatorio deve dizer isso:
num uso real eles viriam da configuracao da propria UTI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.anomaly.sinais_vitais import DISPOSITIVOS, VITAIS

# Mesma escala do detector de termos criticos do audio, para que os escores das
# tres frentes cheguem comparaveis a fusao multimodal.
PESOS = {"alta": 3, "media": 2, "baixa": 1}
SATURACAO_ESCORE = 6

SPO2_HIPOXEMIA = 90.0
SPO2_HIPOXEMIA_GRAVE = 85.0
SPO2_QUEDA_RAPIDA = 4.0

HR_BRADICARDIA = 50.0
HR_TAQUICARDIA = 130.0

RESP_BRADIPNEIA = 8.0
RESP_TAQUIPNEIA = 30.0

# Frequencia de pulso e frequencia cardiaca medem o mesmo batimento por vias
# diferentes. Separadas, indicam perfusao periferica ruim ou sensor solto.
DISCREPANCIA_HR_PULSE = 15.0


@dataclass
class Achado:
    regra: str
    vital: str
    severidade: str
    valor: float

    def to_dict(self) -> dict:
        return asdict(self)


def avaliar(janela: np.ndarray) -> list[Achado]:
    """Regras violadas numa janela de forma (duracao, len(VITAIS))."""
    canal = {vital: janela[:, i] for i, vital in enumerate(VITAIS)}
    medias = {vital: float(serie.mean()) for vital, serie in canal.items()}
    achados: list[Achado] = []

    if medias["SpO2"] < SPO2_HIPOXEMIA_GRAVE:
        achados.append(Achado("hipoxemia_grave", "SpO2", "alta", medias["SpO2"]))
    elif medias["SpO2"] < SPO2_HIPOXEMIA:
        achados.append(Achado("hipoxemia", "SpO2", "alta", medias["SpO2"]))

    queda = float(canal["SpO2"][0] - canal["SpO2"].min())
    if queda >= SPO2_QUEDA_RAPIDA and medias["SpO2"] >= SPO2_HIPOXEMIA:
        # Queda em curso dentro de faixa ainda normal: o valor absoluto so
        # cruzaria o limiar depois, e o objetivo e avisar antes disso.
        achados.append(Achado("queda_de_saturacao", "SpO2", "media", queda))

    if medias["HR"] < HR_BRADICARDIA:
        achados.append(Achado("bradicardia", "HR", "alta", medias["HR"]))
    elif medias["HR"] > HR_TAQUICARDIA:
        achados.append(Achado("taquicardia", "HR", "alta", medias["HR"]))

    if medias["RESP"] < RESP_BRADIPNEIA:
        achados.append(Achado("bradipneia", "RESP", "alta", medias["RESP"]))
    elif medias["RESP"] > RESP_TAQUIPNEIA:
        achados.append(Achado("taquipneia", "RESP", "media", medias["RESP"]))

    diferenca = float(np.abs(canal["HR"] - canal["PULSE"]).mean())
    if diferenca > DISCREPANCIA_HR_PULSE:
        achados.append(Achado("discrepancia_hr_pulse", "HR/PULSE", "baixa",
                              diferenca))

    # Canal isolado constante nao e evidencia de nada: nesta coorte o SpO2 fica
    # constante por 30 s em 49% das janelas e o RESP em 20%, so por estabilidade
    # do paciente. Os dois canais do mesmo aparelho parados juntos caem para 5%.
    for canais in DISPOSITIVOS.values():
        if all(float(canal[c].std()) == 0.0 for c in canais):
            achados.append(Achado("sinal_congelado", "+".join(canais), "media",
                                  float(canal[canais[0]][0])))

    return achados


def escore(achados: list[Achado]) -> float:
    """Gravidade agregada da janela, de 0 a 1."""
    if not achados:
        return 0.0
    soma = sum(PESOS[a.severidade] for a in achados)
    return round(min(1.0, soma / SATURACAO_ESCORE), 4)


def severidade_maxima(achados: list[Achado]) -> str:
    severidades = {a.severidade for a in achados}
    for nivel in ("alta", "media", "baixa"):
        if nivel in severidades:
            return nivel
    return "nenhuma"


def avaliar_lote(matriz: np.ndarray) -> list[list[Achado]]:
    return [avaliar(janela) for janela in matriz]
