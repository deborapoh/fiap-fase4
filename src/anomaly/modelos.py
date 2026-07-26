"""Detectores de anomalia em series de sinais vitais.

Dois modelos com pontos cegos diferentes, treinados so em janelas de pacientes
que ficam de fora da avaliacao:

Isolation Forest      baseline nao supervisionado sobre as estatisticas da
                      janela. Barato, sem hiperparametro sensivel e facil de
                      justificar, mas enxerga a janela como saco de numeros:
                      perde a ordem temporal.
autoencoder LSTM      le a sequencia inteira e e pontuado pelo erro de
                      reconstrucao. Aprende a forma tipica da variacao de cada
                      vital, entao reage a padrao temporal estranho mesmo com
                      todas as estatisticas dentro da faixa.

Os dois devolvem escore em que **maior significa mais anomalo**, e guardam um
limiar fixado como quantil dos escores. Por quantil, e nao por valor absoluto,
para manter a taxa de alarme sob controle: a fracao esperada de janelas
sinalizadas em dados normais e 1 - QUANTIL_LIMIAR.

O limiar deve ser fixado com `calibrar`, em pacientes que nao entraram no
ajuste. Calibrado nos proprios dados de treino, ele sai apertado demais: o
modelo reconstroi bem quem ele ja viu, o quantil cai junto e a taxa de alarme
em paciente novo estoura. `treinar` deixa um limiar provisorio so para o
detector nao ficar inutilizavel se a calibracao for esquecida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

QUANTIL_LIMIAR = 0.99
SEMENTE_PADRAO = 7


class DetectorFloresta:
    """Isolation Forest sobre as estatisticas da janela."""

    def __init__(self, n_arvores: int = 300,
                 contaminacao: float | str = "auto",
                 semente: int = SEMENTE_PADRAO) -> None:
        self.escalador = StandardScaler()
        self.modelo = IsolationForest(n_estimators=n_arvores,
                                      contamination=contaminacao,
                                      random_state=semente,
                                      n_jobs=-1)
        self.limiar = float("inf")
        self.colunas: list[str] = []

    def treinar(self, features: pd.DataFrame,
                quantil: float = QUANTIL_LIMIAR) -> "DetectorFloresta":
        self.colunas = list(features.columns)
        matriz = self.escalador.fit_transform(features.to_numpy())
        self.modelo.fit(matriz)
        self.limiar = float(np.quantile(self._pontuar(matriz), quantil))
        return self

    def calibrar(self, features: pd.DataFrame,
                 quantil: float = QUANTIL_LIMIAR) -> float:
        self.limiar = float(np.quantile(self.pontuar(features), quantil))
        return self.limiar

    def pontuar(self, features: pd.DataFrame) -> np.ndarray:
        matriz = self.escalador.transform(features[self.colunas].to_numpy())
        return self._pontuar(matriz)

    def _pontuar(self, matriz: np.ndarray) -> np.ndarray:
        # score_samples cresce com a normalidade; invertido, fica na mesma
        # direcao do erro de reconstrucao do autoencoder.
        return -self.modelo.score_samples(matriz)


class DetectorAutoencoder:
    """Autoencoder LSTM pontuado pelo erro de reconstrucao da janela."""

    def __init__(self, unidades: int = 32, latente: int = 16,
                 epocas: int = 40, lote: int = 64, taxa: float = 1e-3,
                 semente: int = SEMENTE_PADRAO,
                 dispositivo: str = "cpu") -> None:
        self.unidades = unidades
        self.latente = latente
        self.epocas = epocas
        self.lote = lote
        self.taxa = taxa
        self.semente = semente
        # CPU por padrao: as janelas sao 30x4 e o custo de transferir lote a
        # lote para o MPS supera o ganho de calculo nesta escala.
        self.dispositivo = dispositivo
        self.modelo = None
        self.media = None
        self.desvio = None
        self.limiar = float("inf")

    def treinar(self, matriz: np.ndarray,
                quantil: float = QUANTIL_LIMIAR,
                verboso: bool = False) -> "DetectorAutoencoder":
        import torch

        torch.manual_seed(self.semente)
        self.media = matriz.mean(axis=(0, 1))
        # O desvio zero aparece em canal constante no treino e dividiria por
        # zero na normalizacao.
        self.desvio = np.where(matriz.std(axis=(0, 1)) == 0, 1.0,
                               matriz.std(axis=(0, 1)))

        dados = torch.tensor(self._normalizar(matriz), dtype=torch.float32,
                             device=self.dispositivo)
        self.modelo = _criar_autoencoder(matriz.shape[2], self.unidades,
                                         self.latente).to(self.dispositivo)
        otimizador = torch.optim.Adam(self.modelo.parameters(), lr=self.taxa)
        criterio = torch.nn.MSELoss()
        gerador = torch.Generator().manual_seed(self.semente)

        self.modelo.train()
        for epoca in range(self.epocas):
            ordem = torch.randperm(len(dados), generator=gerador)
            perda_total = 0.0
            for comeco in range(0, len(dados), self.lote):
                lote = dados[ordem[comeco:comeco + self.lote]]
                otimizador.zero_grad()
                perda = criterio(self.modelo(lote), lote)
                perda.backward()
                otimizador.step()
                perda_total += float(perda.detach()) * len(lote)

            if verboso and (epoca + 1) % 10 == 0:
                print(f"    epoca {epoca + 1:3d}/{self.epocas}  "
                      f"perda={perda_total / len(dados):.5f}")

        self.limiar = float(np.quantile(self.pontuar(matriz), quantil))
        return self

    def calibrar(self, matriz: np.ndarray,
                 quantil: float = QUANTIL_LIMIAR) -> float:
        self.limiar = float(np.quantile(self.pontuar(matriz), quantil))
        return self.limiar

    def pontuar(self, matriz: np.ndarray) -> np.ndarray:
        import torch

        if self.modelo is None:
            raise RuntimeError("treinar() precisa ser chamado antes de pontuar()")
        if len(matriz) == 0:
            return np.array([])

        self.modelo.eval()
        dados = torch.tensor(self._normalizar(matriz), dtype=torch.float32,
                             device=self.dispositivo)
        with torch.no_grad():
            erros = []
            for comeco in range(0, len(dados), 256):
                lote = dados[comeco:comeco + 256]
                reconstruido = self.modelo(lote)
                erros.append(((reconstruido - lote) ** 2).mean(dim=(1, 2)).cpu())
            return torch.cat(erros).numpy()

    def _normalizar(self, matriz: np.ndarray) -> np.ndarray:
        return (matriz - self.media) / self.desvio


def _criar_autoencoder(n_canais: int, unidades: int, latente: int):
    """Codifica a janela num vetor latente e a reconstroi a partir dele.

    A classe nasce aqui dentro porque o torch so e importado quando o
    autoencoder e usado: o script pode rodar so com a Isolation Forest, e nesse
    caso nao paga o custo de carregar o torch.
    """
    from torch import Tensor, nn

    class AutoencoderLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codificador = nn.LSTM(n_canais, unidades, batch_first=True)
            self.para_latente = nn.Linear(unidades, latente)
            self.do_latente = nn.Linear(latente, unidades)
            self.decodificador = nn.LSTM(unidades, unidades, batch_first=True)
            self.saida = nn.Linear(unidades, n_canais)

        def forward(self, x: Tensor) -> Tensor:
            _, (estado, _) = self.codificador(x)
            # O gargalo e o estado final do codificador comprimido em `latente`
            # dimensoes: a janela inteira precisa caber ai, o que forca o
            # modelo a aprender a forma tipica em vez de copiar a entrada.
            comprimido = self.para_latente(estado[-1])
            semente = self.do_latente(comprimido)
            repetido = semente.unsqueeze(1).repeat(1, x.shape[1], 1)
            reconstruido, _ = self.decodificador(repetido)
            return self.saida(reconstruido)

    return AutoencoderLSTM()
