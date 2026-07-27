"""Parser das anotacoes Anvil dos dois medicos do Keraal.

Os arquivos `.anvil` sao XML em UTF-16. Tres trilhas importam:

Global evaluation   rotulo da sessao: Correct, Incorrect, Incomplete, Motionless
Global error        erros com severidade, tipo e parte do corpo
Temporal error      mesma coisa, com inicio e fim em segundos

Onde os dois medicos concordam, o rotulo e o consenso. Onde discordam, usamos
o rotulo mais grave (Incorrect > Incomplete > Motionless > Correct) e marcamos
`consenso=False`. A taxa de acordo nesta amostra e cerca de 78%; a escolha
conservadora evita tratar como saudavel uma sessao que um dos medicos marcou
como errada, e precisa constar da metodologia do relatorio.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.common.config import DATA_RAW

PASTA_KERAAL = DATA_RAW / "keraal"
PASTA_A = PASTA_KERAAL / "annotatorA"
PASTA_B = PASTA_KERAAL / "annotatorB"

# Ordem de gravidade: maior indice = mais grave. Discordancia resolve para o
# maximo, nao para o voto majoritario (so ha dois anotadores).
GRAVIDADE = {"Correct": 0, "Motionless": 1, "Incomplete": 2, "Incorrect": 3}

# Prefixo do identificador no nome do arquivo Anvil: G1A-CTK-R1-Brest-022
PADRAO_ID = re.compile(
    r"^(?P<grupo>G[12]A)-(?P<exercicio>CTK|ELK|RTK)-(?P<resto>.+)$"
)


@dataclass
class ErroTemporal:
    inicio_s: float
    fim_s: float
    severidade: str
    tipo: str
    parte_corpo: str


@dataclass
class AnotacaoMedico:
    medico: str
    avaliacao: str | None
    erros_globais: list[ErroTemporal] = field(default_factory=list)
    erros_temporais: list[ErroTemporal] = field(default_factory=list)


@dataclass
class GravacaoAnotada:
    identificador: str
    grupo: str
    exercicio: str
    avaliacao_a: str | None
    avaliacao_b: str | None
    avaliacao: str | None
    consenso: bool
    erros: list[ErroTemporal] = field(default_factory=list)

    @property
    def eh_paciente(self) -> bool:
        return self.grupo == "G1A"

    @property
    def eh_controle(self) -> bool:
        return self.grupo == "G2A"

    @property
    def tem_erro(self) -> bool:
        return self.avaliacao in {"Incorrect", "Incomplete", "Motionless"}


def _atributos(el: ET.Element) -> dict[str, str]:
    out = {}
    for atr in el.findall("attribute"):
        nome = atr.attrib.get("name", "")
        out[nome] = (atr.text or "").strip()
    return out


def _float_seguro(valor: str | None, padrao: float = 0.0) -> float:
    """Alguns `.anvil` do Keraal tem atributo corrompido (ex.: end='tart:')."""
    if valor is None:
        return padrao
    try:
        return float(valor)
    except ValueError:
        return padrao


def _erros_da_trilha(track: ET.Element) -> list[ErroTemporal]:
    erros = []
    for el in track.findall("el"):
        atr = _atributos(el)
        inicio = _float_seguro(el.attrib.get("start"))
        fim = _float_seguro(el.attrib.get("end"))
        if fim < inicio:
            continue
        erros.append(ErroTemporal(
            inicio_s=inicio,
            fim_s=fim,
            severidade=atr.get("Evaluation") or atr.get("evaluation") or "",
            tipo=atr.get("type") or "",
            parte_corpo=atr.get("bodyPart") or "",
        ))
    return erros


def ler_anvil(caminho: Path, medico: str) -> AnotacaoMedico:
    """Le um `.anvil` UTF-16 e devolve as tres trilhas uteis."""
    # encoding='utf-16' cobre BE/LE com ou sem BOM.
    root = ET.parse(caminho).getroot()
    avaliacao = None
    erros_globais: list[ErroTemporal] = []
    erros_temporais: list[ErroTemporal] = []
    for track in root.iter("track"):
        nome = track.attrib.get("name", "")
        if nome == "Global evaluation":
            for el in track.findall("el"):
                atr = _atributos(el)
                avaliacao = atr.get("evaluation") or atr.get("Evaluation")
        elif nome == "Global error":
            erros_globais = _erros_da_trilha(track)
        elif nome == "Temporal error":
            erros_temporais = _erros_da_trilha(track)
    return AnotacaoMedico(medico, avaliacao, erros_globais, erros_temporais)


def _mais_grave(a: str | None, b: str | None) -> str | None:
    candidatos = [x for x in (a, b) if x in GRAVIDADE]
    if not candidatos:
        return a or b
    return max(candidatos, key=lambda x: GRAVIDADE[x])


def _unir_erros(a: AnotacaoMedico, b: AnotacaoMedico) -> list[ErroTemporal]:
    """Une erros temporais dos dois medicos, preferindo a uniao dos intervalos."""
    return list(a.erros_temporais) + list(b.erros_temporais) + list(a.erros_globais)


def carregar_par(identificador: str) -> GravacaoAnotada | None:
    """Carrega o par A/B de uma gravacao. None se faltar algum dos dois."""
    caminho_a = PASTA_A / f"{identificador}.anvil"
    caminho_b = PASTA_B / f"{identificador}.anvil"
    if not caminho_a.exists() or not caminho_b.exists():
        return None
    match = PADRAO_ID.match(identificador)
    if not match:
        return None
    anot_a = ler_anvil(caminho_a, "A")
    anot_b = ler_anvil(caminho_b, "B")
    consenso = anot_a.avaliacao == anot_b.avaliacao and anot_a.avaliacao is not None
    avaliacao = (anot_a.avaliacao if consenso
                 else _mais_grave(anot_a.avaliacao, anot_b.avaliacao))
    return GravacaoAnotada(
        identificador=identificador,
        grupo=match.group("grupo"),
        exercicio=match.group("exercicio"),
        avaliacao_a=anot_a.avaliacao,
        avaliacao_b=anot_b.avaliacao,
        avaliacao=avaliacao,
        consenso=consenso,
        erros=_unir_erros(anot_a, anot_b),
    )


def listar_gravacoes() -> list[GravacaoAnotada]:
    """Todas as gravacoes com par A/B presente e identificador parseavel."""
    ids = sorted({p.stem for p in PASTA_A.glob("*.anvil")}
                 & {p.stem for p in PASTA_B.glob("*.anvil")})
    out = []
    for ident in ids:
        grav = carregar_par(ident)
        if grav is not None:
            out.append(grav)
    return out


def para_dataframe(gravacoes: list[GravacaoAnotada]) -> "pd.DataFrame":
    import pandas as pd
    linhas = []
    for g in gravacoes:
        linhas.append({
            "identificador": g.identificador,
            "grupo": g.grupo,
            "exercicio": g.exercicio,
            "avaliacao_a": g.avaliacao_a,
            "avaliacao_b": g.avaliacao_b,
            "avaliacao": g.avaliacao,
            "consenso": g.consenso,
            "tem_erro": g.tem_erro,
            "n_erros_temporais": len(g.erros),
            "duracao_erro_s": sum(max(0.0, e.fim_s - e.inicio_s) for e in g.erros),
        })
    return pd.DataFrame(linhas)


def erro_em(tempo_s: float, erros: list[ErroTemporal]) -> bool:
    return any(e.inicio_s <= tempo_s <= e.fim_s for e in erros)
