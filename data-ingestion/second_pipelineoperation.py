"""
second_pipelineoperation.py

Classification pipeline that uses a local HuggingFace NLI model as the primary
classifier, with Azure AI (Llama) as a fallback if the local model fails or is
unavailable.

Primary:  zero-shot classification via transformers pipeline
           (default model: cross-encoder/nli-MiniLM2-L6-H768)
Fallback: Azure AI inference endpoint (same as pipelineoperation.py)
"""

import os
import logging
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

HF_MODEL = os.environ.get(
    "HF_CLASSIFIER_MODEL",
    "cross-encoder/nli-MiniLM2-L6-H768",   # ~90 MB, fast on CPU
)
HF_CANDIDATE_LABELS = ["contract clause", "other text"]
HF_CLAUSE_LABEL = "contract clause"
# Confidence threshold below which we fall back to Azure
HF_CONFIDENCE_THRESHOLD = float(os.environ.get("HF_CONFIDENCE_THRESHOLD", "0.55"))

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
        hf_token = os.environ.get("HF_TOKEN") or None
        logger.info(f"Loading local HF classifier: {HF_MODEL}")
        _hf_classifier = hf_pipeline(
            "zero-shot-classification",
            model=HF_MODEL,
            token=hf_token,
            device=-1,          # CPU
        )
        logger.info("Local HF classifier loaded successfully.")
        return _hf_classifier
    except Exception as e:
        logger.warning(f"Could not load local HF classifier ({HF_MODEL}): {e}. "
                       "Will use Azure fallback for all classifications.")
        _hf_load_failed = True
        return None


# ──────────────────────────────────────────────
# Lazy-loaded Azure client (fallback)
# ──────────────────────────────────────────────

_azure_client = None


def _get_azure_client():
    """Return an OpenAI-compatible client pointing at the Azure AI endpoint."""
    global _azure_client
    if _azure_client is not None:
        return _azure_client
    endpoint = os.environ.get("AZURE_INFERENCE_ENDPOINT", "")
    api_key = os.environ.get("AZURE_INFERENCE_API_KEY", "")
    if not endpoint or not api_key:
        return None
    _azure_client = OpenAI(
        base_url=endpoint.rstrip("/"),
        api_key=api_key,
        http_client=httpx.Client(limits=httpx.Limits(
            max_keepalive_connections=2,
            max_connections=5,
            keepalive_expiry=30,
        )),
    )
    return _azure_client


# ──────────────────────────────────────────────
# Individual classifiers
# ──────────────────────────────────────────────

def _classify_local(text: str) -> tuple[bool, float]:
    """
    Run zero-shot classification locally.
    Returns (is_contract_clause: bool, confidence: float).
    Raises RuntimeError if local model is unavailable.
    """
    clf = _get_hf_classifier()
    if clf is None:
        raise RuntimeError("Local HF classifier unavailable.")

    # Truncate to avoid hitting model token limits (~512 tokens ≈ 1800 chars)
    snippet = text[:1800] if len(text) > 1800 else text

    result = clf(snippet, candidate_labels=HF_CANDIDATE_LABELS)
    # result["labels"] is sorted by score descending
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    is_clause = (top_label == HF_CLAUSE_LABEL)
    return is_clause, top_score


def _classify_azure(text: str) -> tuple[bool, float]:
    """
    Classify via Azure AI (Llama).
    Returns (is_contract_clause: bool, confidence: float).
    Raises RuntimeError if Azure is unavailable.
    """
    client = _get_azure_client()
    if client is None:
        raise RuntimeError("Azure AI client unavailable (check AZURE_INFERENCE_ENDPOINT / AZURE_INFERENCE_API_KEY).")

    model = os.environ.get("AZURE_INFERENCE_MODEL", "Llama-3.3-70B-Instruct")
    response = client.chat.completions.create(
        model=model,
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
    # Azure gives a hard yes/no; assign a nominal confidence
    return is_clause, 0.9 if is_clause else 0.85


# ──────────────────────────────────────────────
# Combined classifier: local first, Azure fallback
# ──────────────────────────────────────────────

def classify_text(text: str) -> tuple[bool, float, str]:
    """
    Classify a text snippet as a contract clause or not.
    Tries local HuggingFace model first; falls back to Azure AI.

    Returns:
        (is_contract_clause: bool, confidence: float, backend: str)
        where backend is 'local' or 'azure'.
    """
    # --- Try local ---
    try:
        is_clause, confidence = _classify_local(text)
        # If the local model is uncertain (score close to 50/50), fall through to Azure
        if confidence >= HF_CONFIDENCE_THRESHOLD:
            return is_clause, confidence, "local"
        logger.debug(f"Local confidence {confidence:.2f} below threshold {HF_CONFIDENCE_THRESHOLD}; trying Azure.")
    except Exception as e:
        logger.warning(f"Local classifier failed: {e}. Falling back to Azure.")

    # --- Fallback: Azure ---
    try:
        is_clause, confidence = _classify_azure(text)
        return is_clause, confidence, "azure"
    except Exception as e:
        logger.warning(f"Azure classifier also failed: {e}. Defaulting to False.")
        return False, 0.0, "error"


# ──────────────────────────────────────────────
# Public pipeline interface (same as pipelineoperation.py)
# ──────────────────────────────────────────────

def pipeline_operations(results: list) -> list:
    """
    Run classification over each result's matching_columns and matching_indents,
    plus their expanded counterparts.

    Uses local HuggingFace model as primary classifier, Azure AI as fallback.

    Returns the same list with four new keys added per result:
      - matching_columns_after_classification
      - matching_indents_after_classification
      - expanded_columns_after_classification
      - expanded_indents_after_classification

    Also adds per-result metadata:
      - classification_backend  ('local' | 'azure' | 'error' | 'mixed')
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
