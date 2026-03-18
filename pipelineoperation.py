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
