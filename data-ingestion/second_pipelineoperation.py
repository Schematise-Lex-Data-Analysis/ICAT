"""
second_pipelineoperation.py

Classification pipeline driven by the CLASSIFIER_BACKEND environment variable
(matches .env.example):

  CLASSIFIER_BACKEND=huggingface  (default)
      Primary:  local HuggingFace zero-shot NLI model  (HF_TOKEN for private repos)
      Fallback: Azure AI inference endpoint

  CLASSIFIER_BACKEND=azure
      Only Azure AI is used (no local model loaded).

All credential env vars mirror .env.example exactly:
  API_KEY                   — IndianKanoon (used in app.py)
  HF_TOKEN                  — HuggingFace token (private model repos)
  DB_HOST / DB_NAME / DB_USER / DB_PASS / SSLMODE  — Postgres
  AZURE_INFERENCE_ENDPOINT  — Azure AI endpoint URL
  AZURE_INFERENCE_API_KEY   — Azure AI API key
  AZURE_INFERENCE_MODEL     — model name (default: Llama-3.3-70B-Instruct)
  CLASSIFIER_BACKEND        — "huggingface" | "azure"
"""

import os
import logging
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Read CLASSIFIER_BACKEND from env (.env.example key)
# ──────────────────────────────────────────────

CLASSIFIER_BACKEND = os.environ.get("CLASSIFIER_BACKEND", "huggingface").strip().lower()

# ──────────────────────────────────────────────
# HuggingFace configuration  (only when backend = huggingface)
# ──────────────────────────────────────────────

HF_MODEL = os.environ.get(
    "HF_CLASSIFIER_MODEL",
    "cross-encoder/nli-MiniLM2-L6-H768",   # ~90 MB, fast on CPU
)
HF_TOKEN = os.environ.get("HF_TOKEN") or None          # for private HF repos
HF_CANDIDATE_LABELS = ["contract clause", "other text"]
HF_CLAUSE_LABEL = "contract clause"
# Confidence below this → fall back to Azure
HF_CONFIDENCE_THRESHOLD = float(os.environ.get("HF_CONFIDENCE_THRESHOLD", "0.55"))

# ──────────────────────────────────────────────
# Azure configuration  (mirroring .env.example keys exactly)
# ──────────────────────────────────────────────

AZURE_INFERENCE_ENDPOINT  = os.environ.get("AZURE_INFERENCE_ENDPOINT", "")
AZURE_INFERENCE_API_KEY   = os.environ.get("AZURE_INFERENCE_API_KEY", "")
AZURE_INFERENCE_MODEL     = os.environ.get("AZURE_INFERENCE_MODEL", "Llama-3.3-70B-Instruct")

# ──────────────────────────────────────────────
# Lazy-loaded HuggingFace pipeline
# ──────────────────────────────────────────────

_hf_classifier = None
_hf_load_failed = False


def _get_hf_classifier():
    """Load (once) a zero-shot classification pipeline from HuggingFace."""
    global _hf_classifier, _hf_load_failed
    if _hf_classifier is not None:
        return _hf_classifier
    if _hf_load_failed:
        return None
    try:
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading local HF classifier: {HF_MODEL}")
        _hf_classifier = hf_pipeline(
            "zero-shot-classification",
            model=HF_MODEL,
            token=HF_TOKEN,         # HF_TOKEN from .env.example
            device=-1,              # CPU
        )
        logger.info("Local HF classifier loaded.")
        return _hf_classifier
    except Exception as e:
        logger.warning(
            f"Could not load local HF classifier ({HF_MODEL}): {e}. "
            "Will use Azure for all classifications."
        )
        _hf_load_failed = True
        return None


# ──────────────────────────────────────────────
# Lazy-loaded Azure client
# ──────────────────────────────────────────────

_azure_client = None


def _get_azure_client():
    """Return an OpenAI-compatible client for the Azure AI endpoint."""
    global _azure_client
    if _azure_client is not None:
        return _azure_client
    if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_API_KEY:
        return None
    _azure_client = OpenAI(
        base_url=AZURE_INFERENCE_ENDPOINT.rstrip("/"),
        api_key=AZURE_INFERENCE_API_KEY,
        http_client=httpx.Client(limits=httpx.Limits(
            max_keepalive_connections=2,
            max_connections=5,
            keepalive_expiry=30,
        )),
    )
    return _azure_client


# ──────────────────────────────────────────────
# Individual backend classifiers
# ──────────────────────────────────────────────

def _classify_local(text: str) -> tuple:
    """
    Run zero-shot classification with the local HF model.
    Returns (is_clause: bool, confidence: float).
    Raises RuntimeError if unavailable.
    """
    clf = _get_hf_classifier()
    if clf is None:
        raise RuntimeError("Local HF classifier unavailable.")
    snippet = text[:1800] if len(text) > 1800 else text
    result = clf(snippet, candidate_labels=HF_CANDIDATE_LABELS)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return (top_label == HF_CLAUSE_LABEL), top_score


def _classify_azure(text: str) -> tuple:
    """
    Classify via Azure AI (Llama). Uses AZURE_INFERENCE_* env vars.
    Returns (is_clause: bool, confidence: float).
    Raises RuntimeError if Azure is not configured.
    """
    client = _get_azure_client()
    if client is None:
        raise RuntimeError(
            "Azure AI not configured. Set AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_API_KEY."
        )
    response = client.chat.completions.create(
        model=AZURE_INFERENCE_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a legal text classifier. "
                "Determine whether the text the user provides is a contract clause. "
                "Reply with exactly one word: 'contractclause' if it is, or 'other' if it is not."
            )},
            {"role": "user", "content": text},
        ],
        max_tokens=10,
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()
    del response
    is_clause = (label == "contractclause")
    return is_clause, (0.9 if is_clause else 0.85)


# ──────────────────────────────────────────────
# Combined entry point: respects CLASSIFIER_BACKEND
# ──────────────────────────────────────────────

def classify_text(text: str) -> tuple:
    """
    Classify a text snippet as a contract clause or not.

    When CLASSIFIER_BACKEND=huggingface (default):
      - Tries local HF model first.
      - Falls back to Azure if local fails or confidence < HF_CONFIDENCE_THRESHOLD.

    When CLASSIFIER_BACKEND=azure:
      - Goes straight to Azure AI.

    Returns (is_clause: bool, confidence: float, backend: str)
    """
    if CLASSIFIER_BACKEND == "azure":
        try:
            is_clause, conf = _classify_azure(text)
            return is_clause, conf, "azure"
        except Exception as e:
            logger.warning(f"Azure classifier failed: {e}. Defaulting to False.")
            return False, 0.0, "error"

    # --- CLASSIFIER_BACKEND == "huggingface" ---
    try:
        is_clause, confidence = _classify_local(text)
        if confidence >= HF_CONFIDENCE_THRESHOLD:
            return is_clause, confidence, "local"
        logger.debug(
            f"Local confidence {confidence:.2f} < threshold {HF_CONFIDENCE_THRESHOLD}; "
            "falling back to Azure."
        )
    except Exception as e:
        logger.warning(f"Local HF classifier failed: {e}. Falling back to Azure.")

    try:
        is_clause, conf = _classify_azure(text)
        return is_clause, conf, "azure"
    except Exception as e:
        logger.warning(f"Azure fallback also failed: {e}. Defaulting to False.")
        return False, 0.0, "error"


# ──────────────────────────────────────────────
# Public pipeline interface  (same signature as pipelineoperation.py)
# ──────────────────────────────────────────────

def pipeline_operations(results: list) -> list:
    """
    Classify each result's matched snippets (columns + indents, raw + expanded).

    Reads CLASSIFIER_BACKEND from env to pick the classification strategy:
      huggingface → local NLI first, Azure fallback
      azure       → Azure only

    Adds four classification keys per result (same as pipelineoperation.py):
      matching_columns_after_classification
      matching_indents_after_classification
      expanded_columns_after_classification
      expanded_indents_after_classification

    Also adds:
      classification_backend  — 'local' | 'azure' | 'mixed' | 'error'
    """
    for result in results:
        backends_used = set()

        def keep(text: str) -> bool:
            is_clause, _conf, backend = classify_text(text)
            backends_used.add(backend)
            return is_clause

        result['matching_columns_after_classification'] = [
            col for col in result.get('matching_columns', []) if keep(col)
        ]
        result['matching_indents_after_classification'] = [
            ind for ind in result.get('matching_indents', []) if keep(ind)
        ]
        result['expanded_columns_after_classification'] = [
            col for col in result.get('expanded_columns', []) if keep(col)
        ]
        result['expanded_indents_after_classification'] = [
            ind for ind in result.get('expanded_indents', []) if keep(ind)
        ]

        if len(backends_used) > 1:
            result['classification_backend'] = 'mixed'
        else:
            result['classification_backend'] = next(iter(backends_used), 'error')

    return results
