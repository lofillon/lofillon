# ImpactLens

ImpactLens is a local analytics and reporting module designed to transform messy, mixed-format program data into decision-ready outputs. It combines structured metadata (e.g., project records) with unstructured evidence (PDF reports, annexes, and scanned documents) to extract key facts, summarize outcomes, and produce standardized reports—while enforcing quality assurance and evidence grounding to minimize inconsistencies and hallucinations.

## How to use this project is found [here](./How_to_analyse_and_interpret.md).

## What you get

- A bronze/silver/gold-style local data layout (Parquet + manifests)
- PDF to text extraction with OCR fallback
- JSON extraction with citations
- A small FastAPI service:
  - `POST /projects/{project_id}/extract`
  - `GET /projects/{project_id}/report`



## Datasets / Sources

- World Bank Projects & Operations (Projects Search API v2): `https://search.worldbank.org/api/v2/projects`
- World Bank Documents & Reports (WDS API v3): `https://search.worldbank.org/api/v3/wds`
- World Bank Indicators API (v2): `https://api.worldbank.org/v2/`

## Quickstart (Streamlit UI)

### 1) Install Ollama

Install from `https://ollama.com` and start it (it runs on `http://localhost:11434`).

Pull a model (example):

```bash
ollama pull llama3.1:8b
```

### 2) Install ImpactLens with the UI extra

Create a virtual environment, then:

```bash
pip install -e ".[ui]"
```

### 3) Run the UI (one command)

```bash
impactlens ui
```

Then open `http://127.0.0.1:8501`.


## Quickstart (no UI)

### 1) Install Ollama

Install from `https://ollama.com` and start it (runs on `http://localhost:11434`).

```bash
ollama pull llama3.1:8b
```

### 2) Install ImpactLens

Create a virtual environment, then:

```bash
pip install -e "."
```

If you are working on the codebase (tests, linting), install the dev extra instead:

```bash
pip install -e ".[dev]"
```

### 3) Run an end-to-end demo (one command)

```bash
impactlens demo --project-id P131765
```

### 4) Run the API (optional)

```bash
impactlens api
```

Open `http://127.0.0.1:8000/docs`.

## Run the pipeline (CLI, step-by-step)

Use this if you want to run or debug individual stages, or run batches.

Initialize local folders:

```bash
impactlens init
```

Ingest a small sample of projects (1 page) and build silver tables:

```bash
impactlens ingest-projects --max-pages 1 --rows 100
impactlens build-silver
```

Ingest document metadata for one project (example):

```bash
impactlens ingest-docs --project-id P505244
impactlens build-silver --projects false --documents true
```

Download PDFs and extract text chunks (OCR fallback):

```bash
impactlens process-docs --limit 5
```

Process only the selected documents for one project:

```bash
impactlens select-docs --project-id P505244 --top-k-per-project 5
impactlens process-docs --project-id P505244
```

Run local LLM extraction (Ollama) and build a report:

```bash
impactlens extract-project --project-id P505244
impactlens build-report --project-id P505244
```

Evaluate against a gold facts file (starter in `assets/goldset/`):

```bash
impactlens evaluate --gold-facts-path assets/goldset/gold_facts.sample.jsonl
```

## Configuration

Environment variables (optional):

- `IMPACTLENS_DATA_DIR` (default: `./data`)
- `IMPACTLENS_OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `IMPACTLENS_OLLAMA_MODEL` (default: `llama3.1:8b`)

You can put them in a `.env` file.

## Repo structure

- `src/impactlens/` core library
- `src/impactlens/cli.py` CLI entrypoint
- `src/impactlens/api.py` FastAPI app
- `data/` local datasets (created at runtime)
