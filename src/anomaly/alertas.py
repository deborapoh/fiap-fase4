"""Consolidacao das janelas sinalizadas em alertas para a equipe medica.

Janela nao e alerta. Um evento de dessaturacao de 80 segundos cai em varias
janelas consecutivas e chegaria a equipe como uma dezena de avisos repetidos,
que e o mecanismo classico de fadiga de alarme. Aqui as janelas vizinhas
sinalizadas viram um unico alerta com hora de inicio, hora de fim, motivo e
gravidade.

O escore final do alerta e o maior entre o escore das regras clinicas e o
escore dos modelos, normalizados na mesma escala de 0 a 1. Vale o maior porque
os dois erram para lados opostos: a regra ignora o que nao esta na lista, o
modelo ignora o que e perigoso mas frequente. Esse escore e a contribuicao da
frente de anomalias ao score de risco multimodal, junto do escore de
criticidade do audio.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.anomaly import regras
from src.anomaly.regras import Achado
from src.anomaly.sinais_vitais import JANELA_S, Janelas

# Escore do modelo saturado em duas vezes o limiar: acima disso a diferenca nao
# muda a conduta, todo mundo ja e prioridade maxima.
FATOR_SATURACAO = 2.0

# Duas janelas sinalizadas separadas por ate este intervalo pertencem ao mesmo
# evento. Serve para nao quebrar um evento em dois quando uma janela do meio
# fica logo abaixo do limiar.
TOLERANCIA_UNIAO_S = 15

# Sinal congelado so vira alerta se durar. Numa janela isolada ele e comum: o
# oximetro repete o mesmo par PULSE/SpO2 por 30 s em 5% das janelas normais
# desta coorte, o que e paciente estavel e nao aparelho parado. Exigindo seis
# janelas seguidas, cerca de 55 s de sinal identico, os 63 trechos candidatos
# do BIDMC caem para 18, em 10 dos 53 pacientes.
PERSISTENCIA_CONGELAMENTO = 6


@dataclass
class Alerta:
    paciente: str
    inicio_s: int
    fim_s: int
    duracao_s: int
    tipo: str
    severidade: str
    escore: float
    detectores: str
    regras: str
    n_janelas: int

    def to_dict(self) -> dict:
        return asdict(self)


def pontuar_janelas(escores: dict[str, np.ndarray],
                    limiares: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """Razao ao limiar por janela e quais detectores dispararam nela."""
    tamanho = len(next(iter(escores.values())))
    razoes = np.zeros(tamanho)
    disparos: list[list[str]] = [[] for _ in range(tamanho)]

    for nome, valores in escores.items():
        limiar = limiares[nome]
        if not np.isfinite(limiar) or limiar <= 0:
            continue
        razao = np.asarray(valores) / limiar
        razoes = np.maximum(razoes, razao)
        for indice in np.flatnonzero(razao >= 1.0):
            disparos[indice].append(nome)

    return razoes, ["|".join(d) for d in disparos]


def consolidar(janelas: Janelas,
               escores: dict[str, np.ndarray],
               limiares: dict[str, float],
               achados: list[list[Achado]] | None = None,
               janela_s: int = JANELA_S) -> list[Alerta]:
    if len(janelas) == 0:
        return []

    achados = achados or regras.avaliar_lote(janelas.matriz)
    razoes, detectores = pontuar_janelas(escores, limiares)

    escore_modelo = np.minimum(1.0, razoes / FATOR_SATURACAO)
    escore_regra = np.array([regras.escore(a) for a in achados])
    escore_janela = np.maximum(escore_modelo, escore_regra)

    # Regra de severidade alta entra mesmo sem o modelo concordar: valor
    # perigoso que se repete no paciente e justamente o que o detector nao
    # supervisionado aprende como normal.
    grave = np.array([regras.severidade_maxima(a) == "alta" for a in achados])
    congelado = np.array([any(x.regra == "sinal_congelado" for x in a)
                          for a in achados])
    sustentado = _sequencias_longas(congelado, janelas.inicios,
                                    PERSISTENCIA_CONGELAMENTO)
    sinalizadas = np.flatnonzero((razoes >= 1.0) | grave | sustentado)
    if sinalizadas.size == 0:
        return []

    return [_montar_alerta(janelas, grupo, escore_janela, detectores,
                           achados, janela_s)
            for grupo in _agrupar(janelas.inicios[sinalizadas], sinalizadas,
                                  janela_s)]


def _sequencias_longas(marcas: np.ndarray, inicios: np.ndarray,
                       minimo: int) -> np.ndarray:
    """Mantem so as marcas que pertencem a uma sequencia continua longa.

    A continuidade e checada no tempo, e nao no indice: janela descartada por
    faltante deixa buraco na serie e nao pode emendar dois trechos distantes.
    """
    sustentado = np.zeros(len(marcas), dtype=bool)
    if len(marcas) == 0:
        return sustentado

    passo = int(np.median(np.diff(inicios))) if len(inicios) > 1 else 1
    comeco = None
    for indice in range(len(marcas) + 1):
        continua = (indice < len(marcas) and marcas[indice]
                    and (comeco is None
                         or inicios[indice] - inicios[indice - 1] <= passo))
        if continua:
            comeco = indice if comeco is None else comeco
            continue
        if comeco is not None and indice - comeco >= minimo:
            sustentado[comeco:indice] = True
        comeco = indice if (indice < len(marcas) and marcas[indice]) else None

    return sustentado


def _agrupar(inicios: np.ndarray, indices: np.ndarray,
             janela_s: int) -> list[list[int]]:
    """Une janelas sinalizadas que se sobrepoem ou quase, em blocos."""
    grupos: list[list[int]] = []
    fim_do_grupo = -np.inf

    for inicio, indice in zip(inicios, indices):
        if grupos and inicio <= fim_do_grupo + TOLERANCIA_UNIAO_S:
            grupos[-1].append(int(indice))
        else:
            grupos.append([int(indice)])
        fim_do_grupo = max(fim_do_grupo, inicio + janela_s)

    return grupos


def _montar_alerta(janelas: Janelas, grupo: list[int], escore_janela: np.ndarray,
                   detectores: list[str], achados: list[list[Achado]],
                   janela_s: int) -> Alerta:
    inicio = int(janelas.inicios[grupo[0]])
    fim = int(janelas.inicios[grupo[-1]] + janela_s)

    do_grupo = [a for indice in grupo for a in achados[indice]]
    nomes_regras = Counter(a.regra for a in do_grupo)
    escore = float(escore_janela[grupo].max())

    ferramentas = sorted({d for indice in grupo
                          for d in detectores[indice].split("|") if d})

    return Alerta(
        paciente=janelas.paciente,
        inicio_s=inicio,
        fim_s=fim,
        duracao_s=fim - inicio,
        # O nome do alerta vem da regra mais frequente no trecho. Sem regra, o
        # que restou foi um padrao que o modelo achou atipico e a equipe
        # precisa olhar o sinal para dizer o que e.
        tipo=nomes_regras.most_common(1)[0][0] if nomes_regras else "padrao_atipico",
        severidade=(regras.severidade_maxima(do_grupo) if do_grupo
                    else _severidade_por_escore(escore)),
        escore=round(escore, 4),
        detectores="|".join(ferramentas),
        regras="|".join(nomes_regras),
        n_janelas=len(grupo),
    )


def _severidade_por_escore(escore: float) -> str:
    if escore >= 0.8:
        return "alta"
    return "media" if escore >= 0.5 else "baixa"


def resumir_por_paciente(alertas: list[Alerta],
                         pacientes: list[str] | None = None) -> pd.DataFrame:
    """Uma linha por paciente, no formato que a fusao multimodal consome."""
    por_paciente: dict[str, list[Alerta]] = {p: [] for p in (pacientes or [])}
    for alerta in alertas:
        por_paciente.setdefault(alerta.paciente, []).append(alerta)

    linhas = []
    for paciente, lista in sorted(por_paciente.items()):
        tipos = Counter(a.tipo for a in lista)
        linhas.append({
            "paciente": paciente,
            "n_alertas": len(lista),
            "segundos_em_alerta": sum(a.duracao_s for a in lista),
            "escore_risco": round(max((a.escore for a in lista), default=0.0), 4),
            "severidade_maxima": _pior_severidade(lista),
            "tipos": "|".join(t for t, _ in tipos.most_common()),
        })
    return pd.DataFrame(linhas)


def _pior_severidade(alertas: list[Alerta]) -> str:
    severidades = {a.severidade for a in alertas}
    for nivel in ("alta", "media", "baixa"):
        if nivel in severidades:
            return nivel
    return "nenhuma"


def para_dataframe(alertas: list[Alerta]) -> pd.DataFrame:
    if not alertas:
        return pd.DataFrame(columns=list(Alerta.__annotations__))
    return pd.DataFrame([a.to_dict() for a in alertas])
