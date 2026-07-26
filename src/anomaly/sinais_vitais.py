"""Carga e preparacao das series de sinais vitais do BIDMC.

O BIDMC traz 53 pacientes de UTI com HR, PULSE, RESP e SpO2 amostrados a 1 Hz
por 8 minutos (481 amostras). A deteccao trabalha sobre janelas deslizantes e
nao sobre amostras isoladas: um valor fora da faixa pode ser artefato de
sensor, enquanto uma alteracao sustentada por dezenas de segundos e o que
interessa clinicamente.

Duas escolhas de tratamento dos dados, que precisam constar da metodologia do
relatorio:

faltantes    concentrados em PULSE e SpO2 (13 de 481 no paciente 01) e sempre
             em trechos curtos, tipicos de perda momentanea do oximetro.
             Interpolamos linearmente ate LIMITE_INTERPOLACAO_S segundos, que
             preserva a grade regular exigida pelo modelo de sequencia. Buraco
             maior que isso nao e inventado: a janela que o contem e
             descartada.
janelamento  30 segundos com passo de 5. A janela precisa ser longa o
             bastante para conter a dinamica de um evento (uma dessaturacao
             leva dezenas de segundos) e o passo curto o bastante para nao
             perder o inicio dele.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import DATA_RAW

PASTA_BIDMC = (DATA_RAW / "bidmc" / "bidmc-ppg-and-respiration-dataset-1.0.0"
               / "bidmc_csv")

VITAIS = ("HR", "PULSE", "RESP", "SpO2")

# Os quatro numericos vem de dois aparelhos: o oximetro de dedo entrega PULSE e
# SpO2, e os eletrodos de ECG entregam HR e RESP, esta por impedancia toracica.
# Perda de sensor congela os dois canais do mesmo aparelho ao mesmo tempo, e e
# esse par que distingue falha de instrumentacao de paciente estavel.
DISPOSITIVOS = {"oximetro": ("PULSE", "SpO2"), "ecg": ("HR", "RESP")}

FREQUENCIA_HZ = 1.0
LIMITE_INTERPOLACAO_S = 5
JANELA_S = 30
PASSO_S = 5

# Uma estatistica por vital dentro da janela. As tres primeiras descrevem o
# nivel, as tres ultimas a dinamica: e a dinamica que separa uma taquicardia
# instalada de uma que esta se instalando.
ESTATISTICAS = ("media", "desvio", "minimo", "maximo", "amplitude",
                "inclinacao", "delta_max")


@dataclass
class Paciente:
    """Serie completa de um paciente, ja limpa."""

    identificador: str
    idade: str
    genero: str
    local: str
    sinais: pd.DataFrame
    faltantes: dict[str, int]

    @property
    def duracao_s(self) -> int:
        return len(self.sinais)


@dataclass
class Janelas:
    """Janelas deslizantes de um paciente.

    `matriz` tem forma (n_janelas, JANELA_S, len(VITAIS)) e alimenta o
    autoencoder; `features` e a versao achatada em estatisticas que alimenta a
    Isolation Forest. `inicios` guarda o segundo em que cada janela comeca, e e
    o que permite devolver o alerta com hora de inicio e fim.
    """

    paciente: str
    matriz: np.ndarray
    features: pd.DataFrame
    inicios: np.ndarray

    def __len__(self) -> int:
        return len(self.inicios)


def listar_pacientes(pasta: Path | None = None) -> list[str]:
    pasta = pasta or PASTA_BIDMC
    arquivos = sorted(pasta.glob("bidmc_*_Numerics.csv"))
    return [a.name.split("_")[1] for a in arquivos]


def carregar_paciente(identificador: str, pasta: Path | None = None) -> Paciente:
    pasta = pasta or PASTA_BIDMC

    # skipinitialspace porque o cabecalho do BIDMC vem como " HR", " PULSE":
    # sem isso o acesso por nome de coluna falha.
    bruto = pd.read_csv(pasta / f"bidmc_{identificador}_Numerics.csv",
                        skipinitialspace=True)
    sinais = bruto[list(VITAIS)].astype("float64")
    faltantes = {vital: int(sinais[vital].isna().sum()) for vital in VITAIS}

    sinais = sinais.interpolate(method="linear",
                                limit=LIMITE_INTERPOLACAO_S,
                                limit_direction="both")

    return Paciente(identificador=identificador,
                    sinais=sinais,
                    faltantes=faltantes,
                    **_metadados(pasta / f"bidmc_{identificador}_Fix.txt"))


def carregar_pacientes(identificadores: list[str] | None = None,
                       pasta: Path | None = None) -> list[Paciente]:
    pasta = pasta or PASTA_BIDMC
    identificadores = identificadores or listar_pacientes(pasta)
    return [carregar_paciente(i, pasta) for i in identificadores]


def _metadados(caminho: Path) -> dict[str, str]:
    """Idade, sexo e unidade de internacao, do arquivo Fix.txt do paciente."""
    padrao = {"idade": "", "genero": "", "local": ""}
    if not caminho.exists():
        return padrao

    chaves = {"Age": "idade", "Gender": "genero", "Location": "local"}
    for linha in caminho.read_text(errors="ignore").splitlines():
        campo, _, valor = linha.partition(":")
        if campo.strip() in chaves:
            padrao[chaves[campo.strip()]] = valor.strip()
    return padrao


def janelar(paciente: Paciente,
            janela_s: int = JANELA_S,
            passo_s: int = PASSO_S) -> Janelas:
    """Recorta a serie em janelas deslizantes, descartando as que tem buraco.

    A janela com faltante remanescente e descartada em vez de preenchida com
    zero ou media: o detector aprenderia o valor de preenchimento como padrao
    normal e deixaria de sinalizar a propria perda de sinal, que e um evento
    relevante para a equipe.
    """
    valores = paciente.sinais[list(VITAIS)].to_numpy(dtype="float64")
    total = len(valores)

    blocos, inicios = [], []
    for inicio in range(0, total - janela_s + 1, passo_s):
        bloco = valores[inicio:inicio + janela_s]
        if np.isnan(bloco).any():
            continue
        blocos.append(bloco)
        inicios.append(inicio)

    if not blocos:
        vazio = np.empty((0, janela_s, len(VITAIS)))
        return Janelas(paciente.identificador, vazio,
                       pd.DataFrame(columns=nomes_features()),
                       np.array([], dtype=int))

    matriz = np.stack(blocos)
    return Janelas(paciente=paciente.identificador,
                   matriz=matriz,
                   features=extrair_features(matriz),
                   inicios=np.array(inicios, dtype=int))


def nomes_features() -> list[str]:
    return [f"{vital}_{est}" for vital in VITAIS for est in ESTATISTICAS]


def extrair_features(matriz: np.ndarray) -> pd.DataFrame:
    """Estatisticas por janela e por vital, entrada da Isolation Forest.

    A floresta nao le sequencia, so vetor de tamanho fixo, entao a informacao
    temporal precisa entrar resumida: inclinacao para a tendencia dentro da
    janela e maior salto entre amostras consecutivas para a variacao abrupta.
    """
    if matriz.size == 0:
        return pd.DataFrame(columns=nomes_features())

    passos = np.arange(matriz.shape[1], dtype="float64")
    centrado = passos - passos.mean()
    # Coeficiente angular da reta de minimos quadrados, na forma fechada, para
    # evitar um polyfit por janela e por canal.
    denominador = float((centrado ** 2).sum())

    colunas = {}
    for indice, vital in enumerate(VITAIS):
        canal = matriz[:, :, indice]
        diferencas = np.diff(canal, axis=1)
        colunas[f"{vital}_media"] = canal.mean(axis=1)
        colunas[f"{vital}_desvio"] = canal.std(axis=1)
        colunas[f"{vital}_minimo"] = canal.min(axis=1)
        colunas[f"{vital}_maximo"] = canal.max(axis=1)
        colunas[f"{vital}_amplitude"] = canal.max(axis=1) - canal.min(axis=1)
        colunas[f"{vital}_inclinacao"] = (
            (canal - canal.mean(axis=1, keepdims=True)) @ centrado / denominador)
        colunas[f"{vital}_delta_max"] = np.abs(diferencas).max(axis=1)

    return pd.DataFrame(colunas, columns=nomes_features())


def resumir_faltantes(pacientes: list[Paciente]) -> pd.DataFrame:
    linhas = []
    for paciente in pacientes:
        linha = {"paciente": paciente.identificador,
                 "amostras": paciente.duracao_s}
        linha.update({f"faltantes_{k}": v for k, v in paciente.faltantes.items()})
        linha["faltantes_apos_interpolacao"] = int(
            paciente.sinais.isna().sum().sum())
        linhas.append(linha)
    return pd.DataFrame(linhas)
