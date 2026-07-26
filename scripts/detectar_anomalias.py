#!/usr/bin/env python
"""Roda a deteccao de anomalias nas duas fontes tabulares do projeto.

Sinais vitais (BIDMC) e evolucao de prescricoes (MIMIC-IV Demo). A execucao
tem tres partes:

1. treino    Isolation Forest e autoencoder LSTM aprendem o padrao normal em
             janelas de 30 s de um subconjunto de pacientes.
2. validacao os pacientes que ficaram de fora recebem anomalias sinteticas de
             forma conhecida, e o detector e medido contra elas. O BIDMC nao
             tem rotulo de evento, entao essa e a unica forma de quantificar
             acerto aqui.
3. producao  os mesmos detectores rodam sobre as series reais, sem injecao, e
             o que passa do limiar vira alerta com hora de inicio, motivo e
             gravidade.

Uso:
    python scripts/detectar_anomalias.py
    python scripts/detectar_anomalias.py --sem-autoencoder --epocas 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.anomaly import alertas as mod_alertas  # noqa: E402
from src.anomaly import injecao, prescricoes, regras, validacao  # noqa: E402
from src.anomaly import sinais_vitais as mod_sinais  # noqa: E402
from src.anomaly.modelos import (QUANTIL_LIMIAR, DetectorAutoencoder,  # noqa: E402
                                 DetectorFloresta)
from src.common.config import DATA_PROCESSED  # noqa: E402

PROPORCAO_TREINO = 0.6
PROPORCAO_CALIBRACAO = 0.25
EVENTOS_POR_PACIENTE = 2

# Qualquer regra clinica violada ja e deteccao, entao o limiar do baseline e o
# menor escore positivo possivel.
LIMIAR_REGRAS = 1e-9

LARGURA = 78


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semente", type=int, default=7)
    parser.add_argument("--epocas", type=int, default=40,
                        help="epocas de treino do autoencoder (default: 40)")
    parser.add_argument("--sem-autoencoder", action="store_true",
                        help="roda apenas com a Isolation Forest e as regras")
    parser.add_argument("--eventos", type=int, default=EVENTOS_POR_PACIENTE,
                        help="anomalias injetadas por paciente de teste")
    parser.add_argument("--saida", type=Path, default=DATA_PROCESSED)
    args = parser.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.semente)

    print(f"Carregando pacientes do BIDMC de {mod_sinais.PASTA_BIDMC}...")
    pacientes = mod_sinais.carregar_pacientes()
    treino, teste = _dividir(pacientes, rng)
    print(f"  {len(pacientes)} pacientes: {len(treino)} para treino, "
          f"{len(teste)} para validacao")
    _relatar_dados(pacientes)

    detectores = _treinar(treino, args)
    _validar(detectores, teste, rng, args)
    _detectar_series_reais(detectores, pacientes, args.saida)
    _detectar_prescricoes(args.saida)


def _dividir(pacientes: list[mod_sinais.Paciente],
             rng: np.random.Generator,
             ) -> tuple[list[mod_sinais.Paciente], list[mod_sinais.Paciente]]:
    """Separa por paciente, e nao por janela.

    Janelas do mesmo paciente sao parecidas entre si. Divididas ao acaso, o
    modelo veria em treino o proprio paciente que vai avaliar e a metrica sairia
    otimista.
    """
    ordem = rng.permutation(len(pacientes))
    corte = int(len(pacientes) * PROPORCAO_TREINO)
    return ([pacientes[i] for i in ordem[:corte]],
            [pacientes[i] for i in ordem[corte:]])


def _janelar_varios(pacientes: list[mod_sinais.Paciente]) -> mod_sinais.Janelas:
    lotes = [mod_sinais.janelar(p) for p in pacientes]
    return mod_sinais.Janelas(
        paciente="+".join(p.identificador for p in pacientes),
        matriz=np.concatenate([lote.matriz for lote in lotes]),
        features=pd.concat([lote.features for lote in lotes], ignore_index=True),
        inicios=np.concatenate([lote.inicios for lote in lotes]),
    )


def _relatar_dados(pacientes: list[mod_sinais.Paciente]) -> None:
    """Quanto do sinal foi interpolado e quanto foi descartado."""
    faltantes = mod_sinais.resumir_faltantes(pacientes)
    colunas = [c for c in faltantes.columns if c.startswith("faltantes_")
               and c != "faltantes_apos_interpolacao"]

    amostras = int(faltantes.amostras.sum())
    buracos = int(faltantes[colunas].to_numpy().sum())
    esperadas = sum((p.duracao_s - mod_sinais.JANELA_S) // mod_sinais.PASSO_S + 1
                    for p in pacientes)
    obtidas = sum(len(mod_sinais.janelar(p)) for p in pacientes)

    print(f"  {buracos} valores faltantes em {amostras * len(mod_sinais.VITAIS)} "
          f"leituras, interpolados ate {mod_sinais.LIMITE_INTERPOLACAO_S}s")
    print(f"  {esperadas - obtidas} de {esperadas} janelas descartadas por "
          f"buraco maior que isso")


def _treinar(treino: list[mod_sinais.Paciente], args) -> dict:
    # O limiar sai de pacientes que o modelo nao viu no ajuste. Calibrado nos
    # proprios dados de ajuste ele fica apertado, porque o modelo reconstroi bem
    # quem ele ja conhece, e a taxa de alarme estoura em paciente novo.
    corte = max(1, int(len(treino) * (1 - PROPORCAO_CALIBRACAO)))
    ajuste, calibracao = treino[:corte], treino[corte:]

    janelas = _janelar_varios(ajuste)
    janelas_calibracao = _janelar_varios(calibracao)
    print(f"\nAjuste com {len(janelas)} janelas de {mod_sinais.JANELA_S}s "
          f"(passo {mod_sinais.PASSO_S}s) de {len(ajuste)} pacientes")
    print(f"Limiar no quantil {QUANTIL_LIMIAR} de {len(janelas_calibracao)} "
          f"janelas de outros {len(calibracao)} pacientes")

    floresta = DetectorFloresta(semente=args.semente).treinar(janelas.features)
    floresta.calibrar(janelas_calibracao.features)
    detectores = {"floresta": floresta}
    print(f"  floresta     limiar {floresta.limiar:.4f}")

    if not args.sem_autoencoder:
        print("  autoencoder  treinando...")
        autoencoder = DetectorAutoencoder(epocas=args.epocas,
                                          semente=args.semente)
        autoencoder.treinar(janelas.matriz, verboso=True)
        autoencoder.calibrar(janelas_calibracao.matriz)
        detectores["autoencoder"] = autoencoder
        print(f"  autoencoder  limiar {autoencoder.limiar:.4f}")

    return detectores


def _pontuar(detectores: dict, janelas: mod_sinais.Janelas) -> dict[str, np.ndarray]:
    escores = {}
    for nome, detector in detectores.items():
        entrada = (janelas.features if nome == "floresta" else janelas.matriz)
        escores[nome] = detector.pontuar(entrada)
    return escores


def _limiares(detectores: dict) -> dict[str, float]:
    return {nome: detector.limiar for nome, detector in detectores.items()}


def _validar(detectores: dict, teste: list[mod_sinais.Paciente],
             rng: np.random.Generator, args) -> None:
    _titulo("VALIDACAO COM ANOMALIAS INJETADAS")

    escores: dict[str, list[np.ndarray]] = {n: [] for n in detectores}
    escores["regras"] = []
    rotulos, ambiguas, tipos = [], [], []
    todos_alertas, todos_eventos = [], []

    for paciente in teste:
        alterado, eventos = injecao.injetar_em_paciente(
            paciente, rng, n_eventos=args.eventos)
        janelas = mod_sinais.janelar(alterado)
        if len(janelas) == 0:
            continue

        achados = regras.avaliar_lote(janelas.matriz)
        pontuacoes = _pontuar(detectores, janelas)

        for nome, valores in pontuacoes.items():
            escores[nome].append(valores)
        escores["regras"].append(np.array([regras.escore(a) for a in achados]))
        rotulos.append(injecao.rotular_janelas(janelas.inicios,
                                               mod_sinais.JANELA_S, eventos))
        ambiguas.append(injecao.marcar_ambiguas(janelas.inicios,
                                                mod_sinais.JANELA_S, eventos))
        tipos += injecao.tipo_por_janela(janelas.inicios, mod_sinais.JANELA_S,
                                         eventos)

        todos_alertas += mod_alertas.consolidar(janelas, pontuacoes,
                                                _limiares(detectores), achados)
        todos_eventos += eventos

    rotulos = np.concatenate(rotulos)
    firmes = ~np.concatenate(ambiguas)
    limiares = {**_limiares(detectores), "regras": LIMIAR_REGRAS}

    print(f"{len(todos_eventos)} eventos injetados em {len(teste)} pacientes, "
          f"{len(rotulos)} janelas ({int(rotulos.sum())} anomalas, "
          f"{int((~firmes).sum())} de borda descartadas)\n")

    linhas = []
    for nome, partes in escores.items():
        juntos = np.concatenate(partes)[firmes]
        linhas.append({"detector": nome,
                       **validacao.avaliar_por_janela(juntos, rotulos[firmes],
                                                      limiares[nome])})
    print("Por janela de 30 segundos:")
    print(pd.DataFrame(linhas).to_string(index=False))

    print("\nPor evento, com os detectores combinados em alerta:")
    por_evento = validacao.avaliar_por_evento(todos_alertas, todos_eventos)
    print(pd.Series(por_evento).to_string())

    detalhe = validacao.detalhar_eventos(todos_alertas, todos_eventos)
    print("\nPor tipo de anomalia, no nivel de evento:")
    print(validacao.revocacao_evento_por_tipo(detalhe).to_string(index=False))

    print("\nRevocacao por janela e por tipo, que mostra onde cada detector "
          "e cego:")
    print(_revocacao_cruzada(escores, tipos, limiares).to_string(index=False))

    destino = args.saida / "anomalias_validacao.csv"
    detalhe.to_csv(destino, index=False)
    print(f"\nDetalhe evento a evento em {destino}")


def _revocacao_cruzada(escores: dict[str, list[np.ndarray]], tipos: list[str],
                       limiares: dict[str, float]) -> pd.DataFrame:
    """Uma coluna por detector, uma linha por tipo de anomalia injetada."""
    tabela = None
    for nome, partes in escores.items():
        parcial = validacao.revocacao_por_tipo(np.concatenate(partes), tipos,
                                               limiares[nome])
        parcial = parcial.rename(columns={"revocacao": nome})
        tabela = (parcial if tabela is None
                  else tabela.merge(parcial[["tipo", nome]], on="tipo"))
    return tabela


def _detectar_series_reais(detectores: dict,
                           pacientes: list[mod_sinais.Paciente],
                           saida: Path) -> None:
    _titulo("ALERTAS NAS SERIES REAIS (SEM INJECAO)")

    encontrados = []
    for paciente in pacientes:
        janelas = mod_sinais.janelar(paciente)
        if len(janelas) == 0:
            continue
        encontrados += mod_alertas.consolidar(janelas, _pontuar(detectores, janelas),
                                              _limiares(detectores))

    tabela = mod_alertas.para_dataframe(encontrados)
    resumo = mod_alertas.resumir_por_paciente(
        encontrados, [p.identificador for p in pacientes])

    tabela.to_csv(saida / "anomalias_vitais.csv", index=False)
    resumo.to_csv(saida / "anomalias_vitais_resumo.csv", index=False)

    minutos = sum(p.duracao_s for p in pacientes) / 60
    print(f"{len(tabela)} alertas em {len(pacientes)} pacientes "
          f"({minutos:.0f} minutos de monitoramento)")

    if not tabela.empty:
        print("\nPor tipo:")
        print(tabela.groupby(["tipo", "severidade"]).size()
              .rename("alertas").reset_index().to_string(index=False))
        print("\nPacientes com maior escore de risco:")
        print(resumo.sort_values("escore_risco", ascending=False)
              .head(8).to_string(index=False))

    print(f"\nAlertas em {saida / 'anomalias_vitais.csv'}")
    print(f"Resumo por paciente em {saida / 'anomalias_vitais_resumo.csv'}")


def _detectar_prescricoes(saida: Path) -> None:
    _titulo("ANOMALIAS NA EVOLUCAO DAS PRESCRICOES (MIMIC-IV DEMO)")

    tabela_bruta = prescricoes.carregar()
    eventos = prescricoes.detectar(tabela_bruta)
    tabela = prescricoes.para_dataframe(eventos)
    resumo = prescricoes.resumir_por_internacao(eventos, tabela_bruta)

    tabela.to_csv(saida / "anomalias_prescricoes.csv", index=False)
    resumo.to_csv(saida / "anomalias_prescricoes_resumo.csv", index=False)

    print(f"{len(tabela_bruta)} ordens de {tabela_bruta.hadm_id.nunique()} "
          f"internacoes; {len(tabela)} eventos sinalizados")

    if not tabela.empty:
        print("\nPor tipo:")
        print(tabela.groupby(["tipo", "severidade"]).size()
              .rename("eventos").reset_index().to_string(index=False))
        print(f"\nEm medicamento de alto risco: "
              f"{int(tabela.alto_risco.sum())} de {len(tabela)}")
        print("\nInternacoes com maior escore de risco:")
        print(resumo.sort_values("escore_risco", ascending=False)
              .head(8).to_string(index=False))
        print("\nExemplos:")
        exemplos = (tabela.sort_values("escore", ascending=False)
                    .groupby("tipo").head(1))
        print(exemplos[["hadm_id", "medicamento", "tipo", "detalhe"]]
              .to_string(index=False, max_colwidth=46))

    print(f"\nEventos em {saida / 'anomalias_prescricoes.csv'}")
    print(f"Resumo por internacao em {saida / 'anomalias_prescricoes_resumo.csv'}")


def _titulo(texto: str) -> None:
    print("\n" + "=" * LARGURA)
    print(texto)
    print("=" * LARGURA)


if __name__ == "__main__":
    main()
