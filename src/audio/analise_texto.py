"""Analise das transcricoes: sentimento e termos criticos.

Sao duas tarefas distintas:

sentimento       modelo de classificacao do transformers (distilbert). Serve de
                 proxy para desconforto ou angustia na fala do paciente.
termos criticos  dicionario clinico curado, casado com o `EntityRuler` do
                 spaCy. Um modelo estatistico de NER erraria aqui, porque as
                 expressoes que importam sao fechadas e conhecidas de antemao;
                 regra da resultado deterministico e auditavel, que e o que se
                 espera de um sistema que dispara alerta medico.

Aviso sobre os dados: as frases do TORGO vem do TIMIT (frases foneticamente
balanceadas, digitos, alfabeto radiofonico) e quase nao tem vocabulario
clinico. O dicionario acha pouca coisa nelas, e isso e esperado. O modulo e
exercitado de verdade pelas frases de `EXEMPLOS_CLINICOS`, que simulam falas
de paciente, e mais adiante pelos textos do MIMIC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from src.common.config import MODELO_SENTIMENTO

MODELO_SPACY = "en_core_web_sm"

# Peso de cada categoria no escore de criticidade. Sintomas que podem indicar
# deterioracao aguda pesam mais do que a simples mencao de uma parte do corpo.
PESOS = {"alta": 3, "media": 2, "baixa": 1}

SEVERIDADE_POR_CATEGORIA = {
    "DOR": "alta",
    "RESPIRACAO": "alta",
    "CARDIACO": "alta",
    "NEUROLOGICO": "alta",
    "URGENCIA": "alta",
    "LESAO": "media",
    "MEDICACAO": "media",
    "SINTOMA_GERAL": "media",
    "PARTE_CORPO": "baixa",
}

# Dois termos de severidade alta ja saturam o escore normalizado. O objetivo
# nao e ranquear gravidade com precisao, e sim separar "tem sinal" de "nao tem
# sinal" para a fusao multimodal.
SATURACAO_ESCORE = 6

# Marcadores de negacao. Uma janela de tres tokens antes da entidade cobre os
# casos comuns ("no pain", "not in any pain", "denies chest pain") sem alcancar
# a oracao anterior.
NEGACOES = {"no", "not", "n't", "never", "without", "deny", "denies", "denied"}
JANELA_NEGACAO = 3

# Dicionario curado. Termos de uma palavra usam lema, para casar plural e
# flexao verbal ("injuries" casa com "injury", "fell" com "fall"). Expressoes
# de mais de uma palavra sao listas de lemas, o que resolve as contracoes: o
# spaCy quebra "can't" em "ca" + "n't" com lemas "can" + "not", de modo que o
# mesmo padrao cobre "can't breathe" e "cannot breathe".
TERMOS_POR_CATEGORIA: dict[str, dict[str, list]] = {
    "DOR": {
        "lemas": ["pain", "painful", "ache", "hurt", "sore", "cramp",
                  "headache", "migraine", "burning"],
        "expressoes": ["chest pain", "back pain", "low back pain",
                       "sharp pain", "it hurt"],
    },
    "RESPIRACAO": {
        "lemas": ["breathe", "breath", "breathless", "wheeze", "choke",
                  "suffocate", "gasp", "cough"],
        "expressoes": ["shortness of breath", "short of breath",
                       "can not breathe", "hard to breathe", "out of breath"],
    },
    "CARDIACO": {
        "lemas": ["palpitation", "tachycardia", "tightness"],
        "expressoes": ["heart attack", "chest tightness", "chest be tight",
                       "chest feel tight", "racing heart",
                       "irregular heartbeat"],
    },
    "NEUROLOGICO": {
        "lemas": ["dizzy", "dizziness", "faint", "numb", "numbness",
                  "seizure", "tremor", "paralysis", "confuse", "confused",
                  "stroke"],
        "expressoes": ["blurred vision", "can not move", "can not feel",
                       "pins and needles"],
    },
    "URGENCIA": {
        "lemas": ["emergency", "urgent"],
        "expressoes": ["help me", "i need help", "call the nurse",
                       "call a doctor", "call the doctor", "right now"],
    },
    "LESAO": {
        "lemas": ["injury", "injure", "wound", "fracture", "bruise", "bleed",
                  "bleeding", "blood", "fall"],
        "expressoes": ["i fall", "fall down"],
    },
    "MEDICACAO": {
        "lemas": ["medication", "medicine", "pill", "dose", "overdose",
                  "insulin", "morphine", "prescription", "antibiotic"],
        "expressoes": ["miss dose", "wrong pill", "too much medicine"],
    },
    "SINTOMA_GERAL": {
        "lemas": ["nausea", "nauseous", "vomit", "fever", "chill", "fatigue",
                  "exhaust", "exhausted", "weak", "weakness", "swell",
                  "swelling", "swollen", "dehydrate", "dehydrated"],
        "expressoes": ["throw up", "very tired", "no energy"],
    },
    "PARTE_CORPO": {
        "lemas": ["leg", "arm", "chest", "knee", "shoulder", "hip", "neck",
                  "spine", "abdomen", "stomach"],
        "expressoes": ["lower back", "left leg", "right leg"],
    },
}

# Frases sinteticas de paciente, usadas para demonstrar o detector num texto
# com vocabulario clinico de verdade, que o TORGO nao tem.
EXEMPLOS_CLINICOS = [
    "I have a sharp pain in my lower back and my left leg is numb",
    "Nurse, I can't breathe well and my chest feels tight",
    "I fell this morning and my knee is swollen",
    "I feel dizzy and I think I missed dose of my medication",
    "I am fine today, no pain at all",
]


@dataclass
class TermoCritico:
    texto: str
    categoria: str
    severidade: str
    inicio_char: int
    negado: bool = False


@dataclass
class AnaliseTexto:
    """Resultado da analise de uma transcricao."""

    sentimento: str
    sentimento_confianca: float
    escore_criticidade: float
    severidade_maxima: str
    categorias: list[str] = field(default_factory=list)
    termos_criticos: list[TermoCritico] = field(default_factory=list)

    @property
    def termos_ativos(self) -> list[TermoCritico]:
        return [t for t in self.termos_criticos if not t.negado]

    def to_dict(self) -> dict:
        """Achatado para caber numa linha de CSV junto das metricas acusticas."""
        dados = asdict(self)
        ativos = self.termos_ativos
        dados["categorias"] = "|".join(self.categorias)
        dados["termos_criticos"] = "|".join(t.texto for t in ativos)
        dados["n_termos_criticos"] = len(ativos)
        dados["n_termos_negados"] = len(self.termos_criticos) - len(ativos)
        return dados


def _montar_padroes() -> list[dict]:
    """Um padrao por lema e dois por expressao.

    A expressao entra em duas versoes, por lema e literal, porque nenhuma das
    duas cobre tudo sozinha: por lema "can't breathe" casa (o spaCy quebra a
    contracao em "ca" + "n't", com lemas "can" + "not"), mas "lower back" nao,
    porque o lema de "lower" e "low".
    """
    padroes = []
    for categoria, grupos in TERMOS_POR_CATEGORIA.items():
        for lema in grupos["lemas"]:
            padroes.append({"label": categoria, "pattern": [{"LEMMA": lema}]})
        for expressao in grupos["expressoes"]:
            palavras = expressao.split()
            for atributo in ("LEMMA", "LOWER"):
                tokens = [{atributo: palavra} for palavra in palavras]
                padroes.append({"label": categoria, "pattern": tokens})
    return padroes


@lru_cache(maxsize=1)
def carregar_detector() -> Language:
    """Pipeline do spaCy com o dicionario clinico.

    O NER estatistico e desligado: ele so competiria com as regras e marcaria
    entidades genericas (pessoa, lugar) que nao interessam aqui.
    """
    nlp = spacy.load(MODELO_SPACY, exclude=["ner"])
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(_montar_padroes())
    return nlp


@lru_cache(maxsize=1)
def carregar_classificador_sentimento():
    """Classificador de sentimento.

    Fica em CPU pelo mesmo motivo da transcricao: sao textos de poucas
    palavras, e o custo de mover tensores para o MPS supera o ganho.

    Os pesos precisam ser copiados para fora do arquivo mapeado em memoria.
    O `from_pretrained` deixa os tensores como vistas do `.safetensors`, e
    nesta combinacao de torch com Apple Silicon a multiplicacao de matrizes
    sobre essas vistas aborta o processo com bus error. Ler os pesos funciona;
    so o kernel de GEMM falha. `low_cpu_mem_usage=False` e `dtype` explicito
    nao evitam o problema, entao o `clone` e a saida.
    """
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, pipeline)

    tokenizador = AutoTokenizer.from_pretrained(MODELO_SENTIMENTO)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_SENTIMENTO)
    for tensor in list(modelo.parameters()) + list(modelo.buffers()):
        tensor.data = tensor.data.clone()

    return pipeline("sentiment-analysis", model=modelo.eval(),
                    tokenizer=tokenizador, device="cpu")


def _negado(documento: Doc, entidade: Span) -> bool:
    """Procura marcador de negacao imediatamente antes da entidade.

    Vale para negacao que precede o termo. Quando a negacao faz parte da
    queixa ("can't breathe"), ela esta dentro do proprio padrao e a entidade
    comeca no "ca", entao a janela olha para tras dela e nao se confunde.
    """
    janela = documento[max(0, entidade.start - JANELA_NEGACAO):entidade.start]
    return any(token.lower_ in NEGACOES for token in janela)


def detectar_termos_criticos(texto: str) -> list[TermoCritico]:
    if not texto.strip():
        return []

    documento = carregar_detector()(texto)
    return [
        TermoCritico(
            texto=entidade.text,
            categoria=entidade.label_,
            severidade=SEVERIDADE_POR_CATEGORIA[entidade.label_],
            inicio_char=entidade.start_char,
            negado=_negado(documento, entidade),
        )
        for entidade in documento.ents
        if entidade.label_ in SEVERIDADE_POR_CATEGORIA
    ]


def analisar_sentimento(texto: str) -> tuple[str, float]:
    if not texto.strip():
        return "neutro", 0.0

    resultado = carregar_classificador_sentimento()(texto, truncation=True)[0]
    rotulo = {"POSITIVE": "positivo", "NEGATIVE": "negativo"}.get(
        resultado["label"].upper(), resultado["label"].lower()
    )
    return rotulo, round(float(resultado["score"]), 4)


def analisar(texto: str) -> AnaliseTexto:
    termos = detectar_termos_criticos(texto)
    rotulo, confianca = analisar_sentimento(texto)

    ativos = [t for t in termos if not t.negado]
    escore = min(1.0, sum(PESOS[t.severidade] for t in ativos) / SATURACAO_ESCORE)

    severidades = {t.severidade for t in ativos}
    severidade_maxima = next(
        (nivel for nivel in ("alta", "media", "baixa") if nivel in severidades),
        "nenhuma",
    )

    # dict.fromkeys preserva a ordem de aparicao ao remover duplicatas.
    categorias = list(dict.fromkeys(t.categoria for t in ativos))

    return AnaliseTexto(
        sentimento=rotulo,
        sentimento_confianca=confianca,
        escore_criticidade=round(escore, 4),
        severidade_maxima=severidade_maxima,
        categorias=categorias,
        termos_criticos=termos,
    )
