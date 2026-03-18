import os
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

_classifier = None
_azure_client = None


def get_classifier():
    global _classifier
    if _classifier is None:
        token = os.environ.get("HF_TOKEN")
        _classifier = hf_pipeline(
            task="text-classification",
            truncation=True,
            model="sankalps/NonCompete-Test",
            token=token,
            use_auth_token=token,
        )
    return _classifier


def get_azure_client():
    global _azure_client
    if _azure_client is None:
        _azure_client = ChatCompletionsClient(
            endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_INFERENCE_API_KEY"]),
        )
    return _azure_client


def classify_with_azure(text: str) -> bool:
    """Returns True if the Azure direct model labels the text as a contract clause."""
    client = get_azure_client()
    response = client.complete(
        model=os.environ["AZURE_INFERENCE_MODEL"],
        messages=[
            SystemMessage(content=(
                "You are a legal text classifier. "
                "Determine whether the text the user provides is a contract clause. "
                "Reply with exactly one word: 'contractclause' if it is, or 'other' if it is not."
            )),
            UserMessage(content=text),
        ],
        max_tokens=10,
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()
    return label == "contractclause"


def expand_and_classify_with_azure(doc_text: str, snippet: str,
                                   context_before: int = 2000,
                                   context_after: int = 3000) -> dict:
    """Expand a snippet to its full clause and classify it in a single LLM call.

    Instead of using regex-based boundary detection, sends a context window
    around the snippet to the LLM and lets it determine where the clause
    naturally begins and ends, while also classifying it.

    Returns dict with:
        clause_text (str): the full clause as extracted by the LLM
        is_contract_clause (bool): True if the LLM considers it a contract clause
    """
    import json as _json
    from insert_data import _find_snippet_position

    fallback = {"clause_text": snippet or "", "is_contract_clause": False}

    if not snippet:
        return fallback

    # Build context window around the snippet
    context_window = None
    if doc_text:
        pos = _find_snippet_position(doc_text, snippet)
        if pos != -1:
            start = max(0, pos - context_before)
            end = min(len(doc_text), pos + len(snippet) + context_after)
            context_window = doc_text[start:end]

    # Build user message
    if context_window:
        user_content = (
            f"CONTEXT WINDOW:\n---\n{context_window}\n---\n\n"
            f"MATCHED SNIPPET:\n---\n{snippet}\n---"
        )
    else:
        user_content = (
            f"MATCHED SNIPPET (no surrounding context available):\n---\n{snippet}\n---"
        )

    client = get_azure_client()
    response = client.complete(
        model=os.environ["AZURE_INFERENCE_MODEL"],
        messages=[
            SystemMessage(content=(
                "You are a legal document analyst. You will receive a matched snippet "
                "from a legal document, and optionally a context window surrounding it.\n\n"
                "Your tasks:\n"
                "1. If a context window is provided, identify the full clause or section "
                "that contains the snippet. Use your judgment to determine where the clause "
                "naturally begins and ends \u2014 look for structural markers like section numbers, "
                "article headings, defined-term introductions, or natural paragraph boundaries, "
                "but rely primarily on semantic completeness of the legal provision.\n"
                "2. If no context window is provided, use the snippet as-is.\n"
                "3. Classify whether the extracted clause is a contract clause (a binding "
                "provision such as non-compete, non-solicitation, confidentiality, termination, "
                "indemnification, intellectual property assignment, restrictive covenant, etc.) "
                "or not.\n\n"
                "Respond with JSON only, no markdown fencing:\n"
                "{\"clause_text\": \"the complete clause text exactly as it appears\", "
                "\"is_contract_clause\": true or false}"
            )),
            UserMessage(content=user_content),
        ],
        max_tokens=2000,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = _json.loads(raw)
        return {
            "clause_text": result.get("clause_text", snippet),
            "is_contract_clause": bool(result.get("is_contract_clause", False)),
        }
    except (_json.JSONDecodeError, KeyError, TypeError):
        is_clause = (
            "contractclause" in raw.replace(" ", "").lower()
            or '"is_contract_clause": true' in raw.lower()
        )
        return {"clause_text": snippet, "is_contract_clause": is_clause}


def extract_discussion_with_azure(doc_text: str, contract_clause: str) -> dict:
    """Extract the most succinct court discussion of a contractual clause from a judgment.

    Sends the full judgment text and the identified contract clause to the LLM,
    which locates the court's ratio/conclusion discussing that clause and
    determines sentiment, following annotation guidelines.

    Returns dict with:
        discussion (str): the most succinct discussion extracted from the judgment
        sentiment (str): "neutral or positive" or "negative or struck down"
    """
    import json as _json

    fallback = {"discussion": "", "sentiment": ""}

    if not doc_text or not contract_clause:
        return fallback

    client = get_azure_client()
    response = client.complete(
        model=os.environ["AZURE_INFERENCE_MODEL"],
        messages=[
            SystemMessage(content=(
                "You are a legal judgment analyst. You will receive the full text of "
                "an Indian court judgment and a contractual clause that was identified "
                "within it.\n\n"
                "Your task is to extract the court's most succinct discussion of that "
                "contractual clause from the judgment. Follow these guidelines strictly:\n\n"
                "WHAT TO INCLUDE:\n"
                "- The ratio decidendi or conclusion where the court discusses the "
                "contractual clause.\n"
                "- The most succinct discussion is usually found towards the end of the "
                "relevant section of the judgment.\n"
                "- The discussion may span more than one paragraph and may appear in "
                "different places in the judgment \u2014 include all relevant parts.\n\n"
                "WHAT TO EXCLUDE:\n"
                "- Arguments made by the petitioner or appellant.\n"
                "- The law governing the issue (e.g. Contract Act, Sale of Goods Act, "
                "Arbitration Act) and specifically any quotations from statutes.\n"
                "- Lengthy discussions on case law cited by the court (shepherding).\n"
                "- What lower courts or tribunals discussed \u2014 only include the "
                "current court's own analysis.\n\n"
                "SENTIMENT:\n"
                "- \"neutral or positive\" if the challenge to the clause did not succeed "
                "(clause upheld, enforced, or not invalidated).\n"
                "- \"negative or struck down\" if the clause was struck down, declared "
                "invalid, void, or substantially reinterpreted against the drafter's intent.\n\n"
                "Respond with JSON only, no markdown fencing:\n"
                "{\"discussion\": \"the extracted discussion text exactly as it appears "
                "in the judgment\", \"sentiment\": \"neutral or positive\" or "
                "\"negative or struck down\"}"
            )),
            UserMessage(content=(
                f"CONTRACT CLAUSE:\n---\n{contract_clause}\n---\n\n"
                f"FULL JUDGMENT TEXT:\n---\n{doc_text}\n---"
            )),
        ],
        max_tokens=4000,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = _json.loads(raw)
        return {
            "discussion": result.get("discussion", ""),
            "sentiment": result.get("sentiment", ""),
        }
    except (_json.JSONDecodeError, KeyError, TypeError):
        return {"discussion": raw, "sentiment": ""}


def pipeline_operations(results):
    """
    Run each result's matching_columns and matching_indents through the
    classifier. Only items labelled 'contractclause' are kept.
    Returns the same list with two new keys added per result:
      - matching_columns_after_classification
      - matching_indents_after_classification

    Set CLASSIFIER_BACKEND=azure to use Azure OpenAI; defaults to huggingface.
    """
    backend = os.environ.get("CLASSIFIER_BACKEND", "huggingface").lower()

    if backend == "azure":
        is_clause = classify_with_azure
    else:
        classifier = get_classifier()
        is_clause = lambda text: classifier(text)[0]['label'] == "contractclause"

    for result in results:
        # Keep original snippet-based classification for backwards compatibility
        matching_columns = result.get('matching_columns', [])
        result['matching_columns_after_classification'] = [
            col for col in matching_columns if is_clause(col)
        ]

        matching_indents = result.get('matching_indents', [])
        result['matching_indents_after_classification'] = [
            indent for indent in matching_indents if is_clause(indent)
        ]

        # Classify expanded full clause texts (the primary output)
        expanded_columns = result.get('expanded_columns', [])
        result['expanded_columns_after_classification'] = [
            col for col in expanded_columns if is_clause(col)
        ]

        expanded_indents = result.get('expanded_indents', [])
        result['expanded_indents_after_classification'] = [
            indent for indent in expanded_indents if is_clause(indent)
        ]

    return results
