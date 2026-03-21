# NonCompeteTestRelease

A Flask web application for searching Indian court judgments for contractual clauses via the IndianKanoon API.

## Architecture

- **Backend**: Python 3.12 / Flask
- **Database**: PostgreSQL (Replit managed, via psycopg)
- **ML Pipeline**: Azure AI inference (Llama-3.3-70B-Instruct) for clause classification
- **External API**: IndianKanoon API for document retrieval

## Project Structure

```
app.py                  - Flask application factory
main.py                 - Entry point (runs Flask on port 5000)
data-ingestion/
  insert_data.py        - Database operations (symlinked to root as insert_data.py)
  pipelineoperation.py  - Azure AI classification pipeline (symlinked to root)
templates/              - Jinja2 HTML templates
static/                 - Static assets (images, etc.)
```

## Module Symlinks

`insert_data.py` and `pipelineoperation.py` in the project root are symlinks to their counterparts in `data-ingestion/`. This is required because `app.py` imports them directly.

## Environment Variables

The app uses these environment variables (set in Replit Secrets):
- `API_KEY` - IndianKanoon API token
- `HF_TOKEN` - HuggingFace token (optional)
- `DB_HOST` / `PGHOST` - PostgreSQL host
- `DB_NAME` / `PGDATABASE` - PostgreSQL database name
- `DB_USER` / `PGUSER` - PostgreSQL user
- `DB_PASS` / `PGPASSWORD` - PostgreSQL password
- `SSLMODE` - PostgreSQL SSL mode (default: prefer)
- `AZURE_INFERENCE_ENDPOINT` - Azure AI endpoint URL
- `AZURE_INFERENCE_API_KEY` - Azure AI API key
- `AZURE_INFERENCE_MODEL` - Azure AI model (default: Llama-3.3-70B-Instruct)

The DB connection falls back to Replit-provided `PG*` vars if `DB_*` vars are not set.

## Running

- **Development**: `python main.py` (port 5000)
- **Production**: `gunicorn --bind=0.0.0.0:5000 --reuse-port main:app`

## Key Features

1. Search for contractual clauses in Indian court judgments
2. Caches document text in PostgreSQL to avoid redundant API calls
3. ML classification via Azure AI to filter true contract clauses
4. Clause expansion to extract full context around matched snippets
5. Search history browsing
