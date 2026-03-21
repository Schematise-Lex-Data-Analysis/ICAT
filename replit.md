# NonCompeteTestRelease

A Flask web application for searching Indian court judgments for contractual clauses via the IndianKanoon API.

## Architecture

- **Backend**: Python 3.12 / Flask
- **Database**: PostgreSQL (Replit managed, via psycopg)
- **ML Pipeline**: Azure AI inference (Llama-3.3-70B-Instruct) for clause classification
- **External API**: IndianKanoon API for document retrieval

## Project Structure

```
app.py                          - Flask application factory
main.py                         - Entry point (runs Flask on port 5000)
data-ingestion/
  insert_data.py                - Database operations (symlinked to root)
  pipelineoperation.py          - Azure-only classification pipeline (reference)
  second_pipelineoperation.py   - HuggingFace-primary + Azure-fallback pipeline
                                  (symlinked to root, used by app.py)
templates/              - Jinja2 HTML templates
static/                 - Static assets (images, etc.)
```

## Module Symlinks

`insert_data.py`, `pipelineoperation.py`, and `second_pipelineoperation.py` in the project root are symlinks to their counterparts in `data-ingestion/`. This is required because `app.py` imports them directly.

## Environment Variables  (mirror of .env.example)

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

## Routes

- `GET /` — Dashboard (main search page)
- `GET /confirm?shortcode=&suffixes[]=&page_max=&classifier=` — Run a search
- `GET /history` — Past searches list
- `GET /history/results?query=` — View stored results for a past query

## Key Features

1. Full search dashboard with configurable query suffixes, page range (1–3), and classifier backend
2. IndianKanoon API integration for document retrieval
3. Regex + blockquote matching to find clause text in documents
4. Clause expansion using boundary detection to extract full clause context
5. Azure AI (Llama 3.3 70B) classification pipeline to filter genuine contract clauses
6. Regex-only mode for fast, LLM-free results
7. PostgreSQL caching — already-seen documents are not re-fetched
8. Search history with stored classified results
