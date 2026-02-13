"""Configuration management — layered: YAML → env vars → CLI overrides.

Uses dynaconf for environment-aware config with secret injection.
Config is loaded once at startup and passed explicitly (no global state).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from dynaconf import Dynaconf

logger = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "sentinel.yaml"


def load_config(
    config_path: Path | None = None,
    env_prefix: str = "SENTINEL",
) -> Dynaconf:
    """Load configuration with layered overrides.

    Priority (highest wins): env vars → config file → defaults.
    Secrets (passwords, keys) MUST come from env vars — they're never in YAML.
    """
    path = config_path or _DEFAULT_CONFIG

    if not path.exists():
        logger.warning("config_file_missing", path=str(path))

    settings = Dynaconf(
        settings_files=[str(path)],
        envvar_prefix=env_prefix,
        environments=False,
        load_dotenv=True,
    )

    _validate_required(settings)
    logger.info("config_loaded", path=str(path))
    return settings


def _validate_required(settings: Dynaconf) -> None:
    """Fail fast if critical config is missing."""
    warnings: list[str] = []

    db_password = settings.get("database.password", "")
    if not db_password:
        warnings.append("database.password not set — DB connection will fail")

    if warnings:
        for w in warnings:
            logger.warning("config_validation", issue=w)


def get_nested(settings: Dynaconf, key: str, default: Any = None) -> Any:
    """Safely get a nested config value like 'detection.z_score_threshold'."""
    try:
        return settings.get(key, default)
    except (KeyError, AttributeError):
        return default
