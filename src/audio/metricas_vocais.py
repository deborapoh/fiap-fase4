"""Metricas acusticas para deteccao de alteracoes vocais.

O requisito do desafio e detectar alteracoes vocais indicativas de condicoes
medicas (cansaco, dificuldade respiratoria). As metricas escolhidas aqui sao as
usadas na pratica clinica de fonoaudiologia, calculadas pelo Praat atraves do
parselmouth:

jitter e shimmer  instabilidade de frequencia e de amplitude entre ciclos
                  glotais consecutivos. Sobem quando o controle neuromuscular
                  da laringe esta comprometido.
HNR               razao harmonico-ruido. Cai quando a voz fica soprosa ou
                  rouca, o que acontece com fechamento glotal incompleto.
f0                frequencia fundamental e sua variabilidade, ligadas ao
                  controle prosodico.
pausas            proporcao de silencio e taxa de fala. Fadiga e disartria
                  aumentam o tempo de pausa e reduzem a velocidade.

Nenhuma metrica isolada diagnostica. O que sustenta a deteccao e o conjunto
delas comparado contra a distribuicao do grupo controle.

Ressalva importante para a leitura dos resultados: jitter, shimmer e HNR foram
validados clinicamente em **vogal sustentada**, nao em fala corrida. Medidos
sobre frase inteira, eles acompanham o estilo de fala. Nos dados do TORGO isso
aparece com clareza: a taxa de fala se correlaciona com o jitter (rho de 0,28 a
0,36) e, invertida, com o HNR (rho de -0,23 a -0,41), inclusive dentro de um
mesmo grupo. Como o grupo controle fala mais rapido, ele acaba medindo pior
nessas tres metricas do que o grupo disartrico. O sinal confiavel de patologia
neste corpus esta na proporcao de pausa, no WER e na confianca do
reconhecedor - todos medidos sobre fala corrida, que e o que se tem aqui.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

# Faixa de f0 ampla o suficiente para cobrir vozes masculinas e femininas.
F0_MINIMO_HZ = 75.0
F0_MAXIMO_HZ = 500.0

# Um quadro abaixo deste nivel relativo ao pico e tratado como silencio.
LIMIAR_SILENCIO_DB = 25.0


@dataclass
class MetricasVocais:
    """Metricas acusticas de um unico audio."""

    arquivo: str
    duracao_s: float
    f0_media_hz: float
    f0_desvio_hz: float
    jitter_local: float
    shimmer_local: float
    hnr_db: float
    proporcao_pausa: float
    taxa_fala_silabas_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def _seguro(funcao, padrao: float = float("nan")) -> float:
    """Praat lanca excecao ou devolve NaN em audio curto ou sem voz detectada."""
    try:
        valor = float(funcao())
    except Exception:
        return padrao
    return padrao if np.isnan(valor) else valor


def extrair_metricas(caminho: Path | str) -> MetricasVocais:
    caminho = Path(caminho)
    som = parselmouth.Sound(str(caminho))

    ponto_processo = call(som, "To PointProcess (periodic, cc)",
                          F0_MINIMO_HZ, F0_MAXIMO_HZ)

    pitch = call(som, "To Pitch", 0.0, F0_MINIMO_HZ, F0_MAXIMO_HZ)
    f0_media = _seguro(lambda: call(pitch, "Get mean", 0, 0, "Hertz"))
    f0_desvio = _seguro(lambda: call(pitch, "Get standard deviation", 0, 0, "Hertz"))

    # Os parametros posicionais seguem a assinatura do Praat: janela de analise,
    # periodo minimo e maximo, e fator maximo de variacao entre periodos.
    jitter = _seguro(lambda: call(ponto_processo, "Get jitter (local)",
                                  0, 0, 0.0001, 0.02, 1.3))
    shimmer = _seguro(lambda: call([som, ponto_processo], "Get shimmer (local)",
                                   0, 0, 0.0001, 0.02, 1.3, 1.6))

    harmonicidade = call(som, "To Harmonicity (cc)", 0.01, F0_MINIMO_HZ, 0.1, 1.0)
    hnr = _seguro(lambda: call(harmonicidade, "Get mean", 0, 0))

    proporcao_pausa, taxa_fala = _metricas_temporais(som, pitch)

    return MetricasVocais(
        arquivo=caminho.name,
        duracao_s=round(som.duration, 3),
        f0_media_hz=round(f0_media, 2),
        f0_desvio_hz=round(f0_desvio, 2),
        jitter_local=round(jitter, 5),
        shimmer_local=round(shimmer, 5),
        hnr_db=round(hnr, 2),
        proporcao_pausa=round(proporcao_pausa, 4),
        taxa_fala_silabas_s=round(taxa_fala, 3),
    )


def _metricas_temporais(som: parselmouth.Sound,
                        pitch: parselmouth.Pitch) -> tuple[float, float]:
    """Proporcao de silencio e uma aproximacao da taxa de fala.

    A taxa usa os nucleos de intensidade como proxy de silabas. Nao e tao
    precisa quanto uma segmentacao fonetica, mas e estavel o bastante para
    comparar grupos, que e o uso aqui.
    """
    intensidade = som.to_intensity(minimum_pitch=F0_MINIMO_HZ)
    valores = intensidade.values[0]
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return float("nan"), float("nan")

    limiar = valores.max() - LIMIAR_SILENCIO_DB
    quadros_com_voz = valores > limiar
    proporcao_pausa = 1.0 - float(quadros_com_voz.mean())

    # Cada transicao de silencio para som conta como um nucleo silabico.
    inicios = int(np.count_nonzero(np.diff(quadros_com_voz.astype(int)) == 1))
    inicios += int(quadros_com_voz[0])
    taxa_fala = inicios / som.duration if som.duration > 0 else float("nan")

    return proporcao_pausa, taxa_fala
