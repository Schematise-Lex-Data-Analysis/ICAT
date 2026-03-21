# ICAT — Indian Contract Analysis Tool

A Flask-based legal tech application that searches, extracts, classifies, and
analyzes contractual clauses from Indian court judgments via the IndianKanoon
API, backed by a hybrid ML pipeline (HuggingFace + Azure AI).

---

## Architecture

```
                         ┌──────────────────────────────────┐
                         │          User (Browser)          │
                         └────────────────┬─────────────────┘
                                          │ HTTP
                         ┌────────────────▼─────────────────┐
                         │       Gunicorn / Flask            │
                         │  (main.py → app.py factory)       │
                         │                                   │
                         │   GET /          → dashboard.html │
                         │   GET /confirm   → search flow    │
                         │   GET /history   → query log      │
                         └─────┬───────────────────┬─────────┘
                               │                   │
               ┌───────────────▼──┐   ┌────────────▼──────────────┐
               │  insert_data.py  │   │   second_pipelineoperation │
               │  (DB operations) │   │   (classification pipeline)│
               └───────┬──────────┘   └──────────┬────────────────┘
                       │                          │
          ┌────────────▼──────────┐    ┌──────────▼──────────────────┐
          │   PostgreSQL (Azure)  │    │  CLASSIFIER_BACKEND selector │
          │                       │    │                              │
          │  • tasks              │    │  ┌───────────────────────┐   │
          │  • stored_results     │    │  │  "huggingface"        │   │
          │  • classified_index   │    │  │  HuggingFace          │   │
          │  • search_queries     │    │  │  Transformers         │   │
          └───────────────────────┘    │  │  (sankalps/           │   │
                                       │  │   NonCompete-Test)    │   │
          ┌────────────────────────┐   │  └───────────────────────┘   │
          │   IndianKanoon API     │   │  ┌───────────────────────┐   │
          │                        │   │  │  "azure"              │   │
          │  /search/   (query)    │◄──┘  │  Azure AI Inference   │   │
          │  /doc/{id}/ (text)     │      │  Llama-3.3-70B-Instruct│  │
          │  /docmeta/{id}/        │      └───────────────────────┘   │
          └────────────────────────┘    └──────────────────────────────┘
```

### Request / Data Flow

```
  Browser search form
        │
        ▼
  /confirm route
        │
        ├─► Fetch doc IDs from IndianKanoon API (up to 3 query suffix variants)
        │
        ├─► Check stored_results table
        │       ├── Cache HIT  → skip fetch
        │       └── Cache MISS → fetch full text via API → store in DB
        │
        ├─► Regex match: clause boundary detection on doc text & blockquotes
        │
        ├─► Classification pipeline
        │       ├── HuggingFace local transformer  (default)
        │       └── Azure Llama-3.3-70B-Instruct   (via OpenAI-compat SDK)
        │               ├── expand snippet → full clause
        │               ├── binary classify (contract clause Y/N)
        │               ├── extract court discussion
        │               └── sentiment (positive / negative / struck down)
        │
        ├─► Write results → classified_index table
        ├─► Log query    → search_queries table
        │
        └─► Render results.html
```

---

## Tech Stack

| Layer              | Technology                                      |
|--------------------|-------------------------------------------------|
| Web framework      | Flask 3.x (Python 3.12)                         |
| WSGI server        | Gunicorn                                        |
| Database           | PostgreSQL (Azure-hosted), psycopg driver       |
| ML — local         | HuggingFace Transformers + PyTorch              |
| ML — cloud         | Azure AI Inference (Llama-3.3-70B-Instruct)     |
| Azure SDK compat   | OpenAI Python SDK (`openai` package)            |
| HTTP clients       | httpx, requests                                 |
| HTML parsing       | BeautifulSoup4                                  |
| Data processing    | pandas                                          |
| Template engine    | Jinja2 (bundled with Flask)                     |
| External data API  | IndianKanoon REST API                           |
| Hosting target     | Replit (autoscale, managed secrets)             |

---

## Directory Tree

```
ICAT/
├── main.py                          # Entry point — creates and runs Flask app
├── app.py                           # Flask application factory & all routes
├── requirements.txt                 # Python dependencies
├── .env                             # Runtime secrets (DO NOT COMMIT)
├── .env.example                     # Secret key template (safe to commit)
├── .gitignore
├── .replit                          # Replit run/deploy configuration
├── replit.md                        # Replit-specific notes
├── pythonsqlite.db                  # Legacy SQLite database (superseded by PG)
├── remover.py                       # Utility: remove records from DB
│
├── data-ingestion/                  # Core pipeline modules (canonical source)
│   ├── app.py                       # Standalone ingestion app variant
│   ├── insert_data.py               # DB schema init, caching, regex matching
│   ├── pipelineoperation.py         # Azure-only classification pipeline
│   ├── second_pipelineoperation.py  # HuggingFace + Azure hybrid pipeline
│   ├── remover.py
│   ├── requirements.txt
│   └── batch_expand_classify.ipynb  # Jupyter notebook for batch processing
│
├── templates/                       # Jinja2 HTML templates
│   ├── dashboard.html               # Main search UI
│   ├── results.html                 # Results display (cards + metadata)
│   ├── history.html                 # Past query log
│   ├── noresults.html               # Empty results state
│   ├── error.html                   # Error page
│   ├── home.html
│   ├── testprint.html
│   └── testResults3.html
│
├── static/
│   └── images/
│       ├── ikanoon6_powered_transparent.png
│       └── ikanoon_mobile_powered_transparent.png
│
└── [data files]
    ├── batch1_1.csv
    ├── batch1_2.csv
    ├── batch2.csv
    ├── batch2_discussions.csv
    ├── Matching_rows_Format.txt
    ├── relevant extract of guidelines.txt
    └── Revised Annotation Process Guidelines.pdf
```

> **Symlinks**: `insert_data.py`, `pipelineoperation.py`, `second_pipelineoperation.py`,
> and `remover.py` in the root are symlinks pointing into `data-ingestion/`.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in every value before running.

```
# .env
API_KEY=                          # IndianKanoon API token
HF_TOKEN=                         # HuggingFace access token
DB_HOST=                          # PostgreSQL hostname (e.g. *.postgres.database.azure.com)
DB_NAME=                          # Database name
DB_USER=                          # Database user
DB_PASS=                          # Database password
SSLMODE=require                   # Keep as "require" for Azure PostgreSQL

CLASSIFIER_BACKEND=huggingface    # "huggingface" or "azure"

AZURE_INFERENCE_ENDPOINT=         # Azure AI endpoint URL
AZURE_INFERENCE_API_KEY=          # Azure AI API key
AZURE_INFERENCE_MODEL=Llama-3.3-70B-Instruct  # Model deployment name
HF_CLASSIFIER_MODEL=sankalps/NonCompete-Test  # HuggingFace model repo
```

**Obtaining secrets:**

| Variable | Where to get it |
|---|---|
| `API_KEY` | IndianKanoon developer portal → your account token |
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens |
| `DB_*` | Azure Database for PostgreSQL → Connection strings |
| `AZURE_INFERENCE_*` | Azure AI Studio → your deployed model endpoint |

**Never commit `.env`** — it is already listed in `.gitignore`.
On Replit, add each key under **Tools → Secrets** instead of a `.env` file.

---

## Local Setup & Build

### Prerequisites

- Python 3.12+
- PostgreSQL client libraries (`libpq-dev` on Debian/Ubuntu)
- A filled-in `.env` file (see above)

### Install

```bash
# 1. Clone the repo
git clone <repo-url>
cd ICAT

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the env template and populate it
cp .env.example .env
# Edit .env with your credentials
```

### Run (development)

```bash
python main.py
# → Flask dev server on http://0.0.0.0:5000
```

### Run (production)

```bash
gunicorn --bind=0.0.0.0:5000 --reuse-port main:app
```

### Replit deployment

The `.replit` file handles everything automatically:

- **Run button** → `python main.py`
- **Deploy** → `gunicorn --bind=0.0.0.0:5000 --reuse-port main:app` (port 5000 mapped to external port 80)
- Secrets are injected from Replit's Secrets manager — no `.env` file needed in production.

---

## Database Schema

Tables are created automatically on first startup via `insert_data.py`.

| Table              | Purpose                                                         |
|--------------------|-----------------------------------------------------------------|
| `tasks`            | Temporary working set for the current search session            |
| `stored_results`   | Persistent document cache (text, blockquotes, court metadata)   |
| `classified_index` | Classification results keyed by `(Doc_Id, searchquery)`         |
| `search_queries`   | Timestamped log of every query (IST timezone)                   |

---

## Query Suffix Variants

Each search is executed with up to three phrase suffixes that can be toggled
from the dashboard:

1. `"clause which reads as"`
2. `" mutually agreed"`
3. `"clause states the following"`

Results from all active suffixes are merged and deduplicated before
classification.

---

## Classifier Backends

Set `CLASSIFIER_BACKEND` in `.env` to choose:

| Value | Model | Notes |
|---|---|---|
| `huggingface` | `sankalps/NonCompete-Test` (default) | Runs locally; requires `HF_TOKEN` |
| `azure` | `Llama-3.3-70B-Instruct` | Cloud inference; requires Azure credentials; also used for clause expansion, discussion extraction, and sentiment analysis regardless of backend |

---

## Key Files Quick Reference

| File | What it does |
|---|---|
| `main.py` | Calls `create_app()` and starts the server |
| `app.py` | Flask factory, all route handlers, IndianKanoon API calls |
| `data-ingestion/insert_data.py` | DB init, document caching, regex clause matching |
| `data-ingestion/second_pipelineoperation.py` | Full ML pipeline: expand → classify → extract → sentiment |
| `data-ingestion/pipelineoperation.py` | Azure-only classification pipeline (simpler variant) |
| `templates/dashboard.html` | Search form UI |
| `templates/results.html` | Results card rendering |
