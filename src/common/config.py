"""Configuracao central dos pipelines, lida do .env.

Importar este modulo carrega o .env. Os modulos que dependem de variavel de
ambiente devem importar as constantes daqui em vez de chamar `os.getenv`
direto, senao o valor do .env e ignorado quando o modulo e importado antes do
carregamento.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent.parent

# override=False: variavel ja exportada no shell tem prioridade sobre o
# arquivo, o que permite sobrescrever pontualmente sem editar o .env.
load_dotenv(RAIZ / ".env", override=False)


def _caminho(nome: str, padrao: str) -> Path:
    valor = Path(os.getenv(nome, padrao))
    return valor if valor.is_absolute() else (RAIZ / valor).resolve()


DATA_RAW = _caminho("DATA_RAW", "./data/raw")
DATA_PROCESSED = _caminho("DATA_PROCESSED", "./data/processed")

# O cache do Hugging Face precisa estar no ambiente antes de qualquer import
# de transformers ou huggingface_hub, que leem a variavel na importacao. Volta
# para o ambiente ja resolvido em caminho absoluto, senao o local do cache
# mudaria conforme o diretorio de onde o script e chamado.
HF_HOME = _caminho("HF_HOME", "./models/hf")
os.environ["HF_HOME"] = str(HF_HOME)

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "")

MODELO_WHISPER = os.getenv("WHISPER_MODEL", "small")
IDIOMA_WHISPER = os.getenv("WHISPER_LANGUAGE", "en")

MODELO_SENTIMENTO = os.getenv(
    "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
)


def dispositivo_whisper() -> str:
    """Resolve WHISPER_DEVICE para um valor que o CTranslate2 aceite.

    O CTranslate2, que roda por baixo do faster-whisper, so tem backend de CPU
    e CUDA. Nao existe MPS: em Apple Silicon, "auto" e sempre CPU.
    """
    escolhido = os.getenv("WHISPER_DEVICE", "auto").lower()
    if escolhido != "auto":
        return escolhido

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
