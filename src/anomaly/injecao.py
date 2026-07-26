"""Anomalias sinteticas injetadas nas series do BIDMC, para validacao.

O BIDMC nao tem rotulo de anomalia: e um registro de 8 minutos por paciente,
sem anotacao de evento. Sem rotulo nao ha como afirmar que o detector acerta,
so que ele sinaliza alguma coisa. A saida adotada e injetar eventos de
comportamento conhecido em copias das series e medir a deteccao contra eles.
Essa e a decisao de validacao do pipeline de anomalias e precisa aparecer no
relatorio.

O que essa validacao mede e o que ela nao mede:

mede      se o detector reage a desvios com forma fisiologica plausivel e a
          que taxa de falso positivo, com a serie real como plano de fundo.
nao mede  acuracia clinica em evento espontaneo real. Para isso seria preciso
          um dataset anotado por medico, que o BIDMC nao e.

Os cinco tipos cobrem as familias de falha que a equipe de UTI monitora:
deterioracao respiratoria (dessaturacao, apneia), instabilidade cardiaca
(bradicardia, taquicardia) e perda de instrumentacao (falha de sensor). Os
alvos numericos seguem faixas de alarme usuais de monitor de cabeceira.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.anomaly.sinais_vitais import DISPOSITIVOS, Paciente

TIPOS = ("dessaturacao", "bradicardia", "taquicardia", "apneia", "falha_sensor")

DURACAO_MIN_S = 40
DURACAO_MAX_S = 90

# Margem nas bordas da serie: um evento colado no inicio ou no fim entraria em
# poucas janelas e distorceria a taxa de deteccao para baixo.
MARGEM_S = 20

# Distancia minima entre dois eventos no mesmo paciente, para que cada janela
# rotulada como anomala pertenca a um evento so.
SEPARACAO_S = 60


@dataclass
class EventoInjetado:
    paciente: str
    tipo: str
    inicio_s: int
    fim_s: int

    def to_dict(self) -> dict:
        return asdict(self)


def injetar(sinais: pd.DataFrame, tipo: str, inicio_s: int, duracao_s: int,
            rng: np.random.Generator) -> pd.DataFrame:
    """Devolve uma copia da serie com o evento aplicado no trecho indicado."""
    if tipo not in TIPOS:
        raise ValueError(f"tipo desconhecido: {tipo}")

    saida = sinais.copy()
    trecho = slice(inicio_s, inicio_s + duracao_s)
    envelope = _envelope(duracao_s, subida=0.1 if tipo == "apneia" else 0.3)

    if tipo == "dessaturacao":
        # Queda de saturacao acompanhada de taquipneia e de aumento da
        # frequencia cardiaca, que e a resposta compensatoria esperada.
        _deslocar(saida, "SpO2", trecho, envelope, -rng.uniform(8, 18), rng)
        _escalar(saida, "RESP", trecho, envelope, rng.uniform(1.3, 1.6), rng)
        _escalar(saida, "HR", trecho, envelope, rng.uniform(1.1, 1.25), rng)
        _escalar(saida, "PULSE", trecho, envelope, rng.uniform(1.1, 1.25), rng)

    elif tipo == "bradicardia":
        alvo = rng.uniform(35, 45)
        _puxar_para(saida, "HR", trecho, envelope, alvo, rng)
        _puxar_para(saida, "PULSE", trecho, envelope, alvo, rng)

    elif tipo == "taquicardia":
        alvo = rng.uniform(130, 165)
        _puxar_para(saida, "HR", trecho, envelope, alvo, rng)
        _puxar_para(saida, "PULSE", trecho, envelope, alvo, rng)

    elif tipo == "apneia":
        # Parada respiratoria: a frequencia cai quase a zero de imediato e a
        # saturacao so desce depois, pelo tempo de reserva de oxigenio.
        _puxar_para(saida, "RESP", trecho, envelope, rng.uniform(0, 3), rng)
        atraso = min(10, max(0, duracao_s - 10))
        trecho_spo2 = slice(inicio_s + atraso, inicio_s + duracao_s)
        _deslocar(saida, "SpO2", trecho_spo2,
                  _envelope(duracao_s - atraso, subida=0.5),
                  -rng.uniform(6, 14), rng)

    elif tipo == "falha_sensor":
        # O monitor congela o ultimo valor lido em vez de reportar vazio, que e
        # como a perda de contato costuma aparecer no registro. Congela o
        # aparelho inteiro, nao um canal: oximetro solto para PULSE e SpO2 na
        # mesma hora.
        dispositivo = str(rng.choice(list(DISPOSITIVOS)))
        for canal in DISPOSITIVOS[dispositivo]:
            saida.iloc[trecho, saida.columns.get_loc(canal)] = float(
                saida[canal].iloc[max(inicio_s - 1, 0)])

    _limitar_faixas(saida)
    return saida


def injetar_em_paciente(paciente: Paciente,
                        rng: np.random.Generator,
                        n_eventos: int = 2,
                        tipos: tuple[str, ...] = TIPOS,
                        ) -> tuple[Paciente, list[EventoInjetado]]:
    """Aplica n eventos de tipos sorteados em posicoes que nao se sobrepoem."""
    sinais = paciente.sinais
    eventos: list[EventoInjetado] = []

    for _ in range(n_eventos):
        tipo = str(rng.choice(tipos))
        duracao = int(rng.integers(DURACAO_MIN_S, DURACAO_MAX_S + 1))
        inicio = _sortear_inicio(len(sinais), duracao, eventos, rng)
        if inicio is None:
            continue

        sinais = injetar(sinais, tipo, inicio, duracao, rng)
        eventos.append(EventoInjetado(paciente=paciente.identificador,
                                      tipo=tipo,
                                      inicio_s=inicio,
                                      fim_s=inicio + duracao))

    alterado = Paciente(identificador=paciente.identificador,
                        idade=paciente.idade,
                        genero=paciente.genero,
                        local=paciente.local,
                        sinais=sinais,
                        faltantes=paciente.faltantes)
    return alterado, eventos


def rotular_janelas(inicios: np.ndarray, janela_s: int,
                    eventos: list[EventoInjetado],
                    sobreposicao_minima: float = 0.5) -> np.ndarray:
    """Marca como anomala a janela que cobre metade ou mais de algum evento.

    O criterio de metade evita contar como acerto uma janela que so encosta na
    borda do evento, onde o sinal ainda esta praticamente normal.
    """
    rotulos = np.zeros(len(inicios), dtype=int)
    for indice, inicio in enumerate(inicios):
        fim = inicio + janela_s
        for evento in eventos:
            coberto = min(fim, evento.fim_s) - max(inicio, evento.inicio_s)
            if coberto >= sobreposicao_minima * janela_s:
                rotulos[indice] = 1
                break
    return rotulos


def marcar_ambiguas(inicios: np.ndarray, janela_s: int,
                    eventos: list[EventoInjetado],
                    sobreposicao_minima: float = 0.5) -> np.ndarray:
    """Janelas que encostam num evento sem cobrir metade dele.

    Sao contadas como normais pelo rotulo e como anomalas pelo detector, que
    ali ja ve parte do evento. Contar essas janelas como erro inflaria o
    numero de falso positivo sem que exista alarme extra: elas caem dentro do
    mesmo alerta do evento vizinho. Ficam de fora da metrica por janela.
    """
    ambiguas = np.zeros(len(inicios), dtype=bool)
    for indice, inicio in enumerate(inicios):
        fim = inicio + janela_s
        for evento in eventos:
            coberto = min(fim, evento.fim_s) - max(inicio, evento.inicio_s)
            if 0 < coberto < sobreposicao_minima * janela_s:
                ambiguas[indice] = True
                break
    return ambiguas


def tipo_por_janela(inicios: np.ndarray, janela_s: int,
                    eventos: list[EventoInjetado]) -> list[str]:
    """Tipo do evento que domina cada janela, ou vazio se ela for normal."""
    tipos = []
    for inicio in inicios:
        fim = inicio + janela_s
        melhor, cobertura_maxima = "", 0
        for evento in eventos:
            coberto = min(fim, evento.fim_s) - max(inicio, evento.inicio_s)
            if coberto > cobertura_maxima:
                melhor, cobertura_maxima = evento.tipo, coberto
        tipos.append(melhor if cobertura_maxima >= janela_s / 2 else "")
    return tipos


def _envelope(duracao_s: int, subida: float = 0.3) -> np.ndarray:
    """Rampa de subida, patamar e rampa de descida, entre 0 e 1.

    Evento fisiologico nao aparece e some em degrau. A transicao suave tambem
    torna a validacao mais honesta: um degrau seria trivial de detectar pela
    feature de maior salto entre amostras.
    """
    n_subida = max(1, int(duracao_s * subida))
    n_descida = max(1, int(duracao_s * subida))
    n_patamar = max(0, duracao_s - n_subida - n_descida)
    return np.concatenate([
        np.linspace(0, 1, n_subida, endpoint=False),
        np.ones(n_patamar),
        np.linspace(1, 0, n_descida),
    ])[:duracao_s]


def _ruido(rng: np.random.Generator, tamanho: int, escala: float) -> np.ndarray:
    return rng.normal(0, escala, tamanho)


def _coluna(sinais: pd.DataFrame, nome: str, trecho: slice) -> np.ndarray:
    return sinais[nome].to_numpy()[trecho]


def _gravar(sinais: pd.DataFrame, nome: str, trecho: slice,
            valores: np.ndarray) -> None:
    sinais.iloc[trecho, sinais.columns.get_loc(nome)] = valores


def _deslocar(sinais: pd.DataFrame, nome: str, trecho: slice,
              envelope: np.ndarray, delta: float,
              rng: np.random.Generator) -> None:
    base = _coluna(sinais, nome, trecho)
    envelope = envelope[:len(base)]
    _gravar(sinais, nome, trecho,
            base + envelope * delta + _ruido(rng, len(base), 0.3))


def _escalar(sinais: pd.DataFrame, nome: str, trecho: slice,
             envelope: np.ndarray, fator: float,
             rng: np.random.Generator) -> None:
    base = _coluna(sinais, nome, trecho)
    envelope = envelope[:len(base)]
    _gravar(sinais, nome, trecho,
            base * (1 + envelope * (fator - 1)) + _ruido(rng, len(base), 0.3))


def _puxar_para(sinais: pd.DataFrame, nome: str, trecho: slice,
                envelope: np.ndarray, alvo: float,
                rng: np.random.Generator) -> None:
    """Interpola o canal em direcao a um valor alvo, na forma do envelope."""
    base = _coluna(sinais, nome, trecho)
    envelope = envelope[:len(base)]
    _gravar(sinais, nome, trecho,
            base + envelope * (alvo - base) + _ruido(rng, len(base), 0.3))


def _limitar_faixas(sinais: pd.DataFrame) -> None:
    """Mantem os valores dentro do que um monitor consegue reportar."""
    limites = {"HR": (0, 300), "PULSE": (0, 300), "RESP": (0, 80),
               "SpO2": (50, 100)}
    for nome, (minimo, maximo) in limites.items():
        if nome in sinais.columns:
            sinais[nome] = sinais[nome].clip(minimo, maximo)


def _sortear_inicio(total_s: int, duracao_s: int,
                    eventos: list[EventoInjetado],
                    rng: np.random.Generator) -> int | None:
    limite = total_s - duracao_s - MARGEM_S
    if limite <= MARGEM_S:
        return None

    for _ in range(20):
        inicio = int(rng.integers(MARGEM_S, limite))
        fim = inicio + duracao_s
        conflito = any(inicio < e.fim_s + SEPARACAO_S
                       and e.inicio_s - SEPARACAO_S < fim
                       for e in eventos)
        if not conflito:
            return inicio
    return None
