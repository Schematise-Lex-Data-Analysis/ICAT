import os
from transformers import pipeline as hf_pipeline

token = os.environ.get("HF_TOKEN")
if not token:
    print("Warning: HF_TOKEN not set — download may fail for private models")
print("Pre-downloading model: sankalps/NonCompete-Test")
hf_pipeline(
    task="text-classification",
    model="sankalps/NonCompete-Test",
    token=token,
    truncation=True,
)
print("Model cached successfully.")
