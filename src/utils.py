"""Utilitários compartilhados."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_critical_terms(config_path: Path | None = None) -> list[str]:
    path = config_path or project_root() / "config" / "critical_terms_pt.txt"
    if not path.exists():
        return []
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
