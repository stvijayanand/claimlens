"""Central configuration for ClaimLens.

Every tunable lives here: model IDs, effort levels, thresholds, budgets,
feature flags, and the price table. Each value can be overridden with an
environment variable of the same name.

Values marked PROVISIONAL come from the submission doc and must be confirmed
against the live Meta Model API at the Issue #1 gate.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
load_dotenv(ROOT / ".env")  # real environment variables take precedence
DATA_DIR = Path(os.environ.get("CLAIMLENS_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("CLAIMLENS_DB_PATH", DATA_DIR / "claimlens.duckdb"))
TAXONOMY_CSV = Path(
    os.environ.get("CLAIMLENS_TAXONOMY_CSV", DATA_DIR / "taxonomy.csv")
)
LABOR_CSV = Path(os.environ.get("CLAIMLENS_LABOR_CSV", DATA_DIR / "labor_ranges.csv"))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Meta Model API -------------------------------------------------------
# Accepts MODEL_API_KEY as an alias; META_MODEL_API_KEY wins when both are set.
META_MODEL_API_KEY = os.environ.get("META_MODEL_API_KEY", "") or os.environ.get(
    "MODEL_API_KEY", ""
)
META_MODEL_BASE_URL = _env_str("META_MODEL_BASE_URL", "")  # required; no default
SPARK_MODEL = _env_str(
    "CLAIMLENS_SPARK_MODEL", "muse-spark-1.3"
)  # verified live 2026-10-30 (chat-completions + strict json_schema)
SAM_MODEL = _env_str("CLAIMLENS_SAM_MODEL", "sam-3.1")  # PROVISIONAL id
IMAGE_MODEL = _env_str("CLAIMLENS_IMAGE_MODEL", "muse-image")  # PROVISIONAL id

EFFORT_LOW = _env_str("CLAIMLENS_EFFORT_LOW", "low")
EFFORT_HIGH = _env_str("CLAIMLENS_EFFORT_HIGH", "high")

# --- Price table (USD; PROVISIONAL doc figures, confirm live) -------------
SAM_USD_PER_IMAGE = _env_float("CLAIMLENS_SAM_USD_PER_IMAGE", 2.50 / 1000)
SYNTH_IMAGE_USD = _env_float("CLAIMLENS_SYNTH_IMAGE_USD", 0.01)
# Spark pricing unknown until the live check; per-call fallback used only when
# the API returns no usage payload (figures then marked estimated).
SPARK_USD_PER_CALL = _env_float("CLAIMLENS_SPARK_USD_PER_CALL", 0.0)

# --- Pipeline tunables ----------------------------------------------------
DAMAGE_PHRASES = ("dent", "scratch", "crack")
MIN_PART_CONFIDENCE = _env_float("CLAIMLENS_MIN_PART_CONFIDENCE", 0.3)
MIN_ASSESS_CONFIDENCE = _env_float("CLAIMLENS_MIN_ASSESS_CONFIDENCE", 0.4)
MATCH_MIN_SCORE = _env_float("CLAIMLENS_MATCH_MIN_SCORE", 0.6)
OVERLAP_MIN_INTERSECT_PX = _env_int("CLAIMLENS_OVERLAP_MIN_INTERSECT_PX", 1)
CROP_PADDING_FRAC = _env_float("CLAIMLENS_CROP_PADDING_FRAC", 0.15)
MAX_AGENT_ROUNDS = _env_int("CLAIMLENS_MAX_AGENT_ROUNDS", 3)

# --- Feature flags (binding cut order: labor first, then repair-vs-replace)
FEATURE_LABOR_CHECK = _env_bool("CLAIMLENS_FEATURE_LABOR_CHECK", True)
FEATURE_REPAIR_REPLACE = _env_bool("CLAIMLENS_FEATURE_REPAIR_REPLACE", True)

# --- Budgets ---------------------------------------------------------------
BUDGET_SECONDS_PER_CAR = _env_float("CLAIMLENS_BUDGET_SECONDS_PER_CAR", 120.0)
BUDGET_USD_PER_CAR = _env_float("CLAIMLENS_BUDGET_USD_PER_CAR", 0.50)
