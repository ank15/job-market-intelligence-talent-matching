# Job Market Intelligence & Talent Matching

An end-to-end machine learning system that transforms raw job-market data into
actionable market intelligence and an explainable talent-matching score.

The system analyzes ~12,000 real job postings to:

- Identify in-demand skills across the market
- Analyze skill demand by location and hiring trends by seniority / work type
- Compute a job–candidate fit score using NLP — semantic sentence-transformer
  embeddings, with a TF-IDF baseline to compare against
- Highlight a candidate's skill gaps for a given role

It is structured as a modular ML pipeline (data validation, preprocessing,
feature engineering, analytics, and a Streamlit app) with reproducibility in mind.

---

## Business problem

Recruiters and job seekers often lack data-driven visibility into:

- Which skills are currently in demand
- Which skills are concentrated in which locations
- How well a candidate matches a specific job, and what they're missing

This system converts unstructured job data into structured market intelligence
and a quantitative, explainable talent-matching score.

---

## System architecture

```
Raw Data → Validation → Preprocessing → Feature Engineering →
   ├── Market Intelligence Analytics      (src/analytics)
   └── Semantic Talent Matching Engine    (src/matching + src/app) → Streamlit Dashboard
```

---

## Modules

### Module 1 — Job Market Intelligence

`src/analytics/market_intelligence.py`

- **Top in-demand skills** across all postings
- **Skill demand by location** (top skills within the busiest cities)
- **Hiring trends** by `job_level` (e.g. Mid-senior vs. Associate) and
  `job_type` (Onsite / Remote / Hybrid)
- Generates CSV reports under `artifacts/reports/`

Techniques: exploratory data analysis, grouped aggregation, rule-based skill
cleaning, and missing-value handling.

### Module 2 — Talent Matching Engine

`src/app/app.py`, `src/matching/semantic_matcher.py`, `src/utils/preprocessing.py`

Computes similarity between a candidate's skills and job descriptions, with a
**switchable engine** so you can compare a semantic model against a lexical
baseline:

| Engine | Technique | Strength |
|--------|-----------|----------|
| **Semantic (default)** | sentence-transformer embeddings (`all-MiniLM-L6-v2`) + cosine | Matches *meaning* — "ML" ≈ "machine learning" even with no shared words |
| **TF-IDF** | TF-IDF (`max_features=5000`) + cosine | Fast lexical baseline; only matches overlapping words |

Why it matters: for a resume reading *"ML and neural networks"*, the semantic
engine surfaces machine-learning / deep-learning jobs (~35% similarity) while
TF-IDF scores them all **0%** — it can't tell that "ML" means "machine learning".

**Which is best?** **Semantic is the default** — real resumes are full of synonyms and
abbreviations a word-only model misses, so it gives better matches on typical queries.
TF-IDF stays available as a fast, fully-interpretable baseline (and is occasionally more
robust on very short, exact terms). The honest answer is to pick per query type — and to
evaluate both on labeled data (precision@k) rather than assume a universal winner.

Pipeline:

1. Clean skills and summaries (shared functions in `src/utils/preprocessing.py`,
   so the app mirrors the notebook exactly — no train/serve skew)
2. Embed the job corpus **once** and persist it (`.npy` for embeddings,
   `joblib` for the TF-IDF vectorizer), so only the short query is processed
   per request
3. **Cosine similarity** between the resume vector and every job, returning the
   top 5 matches with a **match score (0–100)**
4. **Skill-level gap analysis** via `partial_match` — exact, token-subset, and
   fuzzy (`difflib`) matching, so `node` matches `node.js` and `react` ~
   `reactjs`, while avoiding substring false positives like `java` → `javascript`

This provides a practical, explainable, and semantically-aware job-fit system.

---

## Testing & validation

- Unit tests for preprocessing functions (`clean_skills`, `clean_text`, `partial_match`)
- Unit tests for the market-intelligence analytics
- Unit tests for the semantic matcher's ranking & cache logic (no model download)
- Run with `pytest` (24 tests):

```bash
pytest -q
```

---

## Tech stack

Python · pandas · numpy · **sentence-transformers** (`all-MiniLM-L6-v2` embeddings) ·
scikit-learn (TF-IDF, cosine similarity) · joblib / numpy (model & embedding
persistence) · matplotlib / seaborn (EDA) · Streamlit (UI) · pytest

---

## How to run locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate market-intelligence reports (writes to artifacts/reports/)
python -m src.analytics.market_intelligence

# 4. Launch the talent-matching app
streamlit run src/app/app.py

# 5. Run the tests
pytest -q
```

> **Data:** the raw/processed CSVs (~120 MB) are not committed to git. Place the
> source datasets under `data/raw/` and the merged `final_jobs.csv` under
> `data/processed/`. The merge/cleaning steps are in
> `notebooks/01_data_exploration.ipynb`.

> **First run (semantic engine):** the first semantic search downloads the
> `all-MiniLM-L6-v2` model (~80 MB) and embeds all ~12k job summaries once
> (~1–2 min on CPU), caching the result to
> `artifacts/trained_models/job_embeddings.npy`. Every run after that loads the
> cache instantly.
>
> **Why cache the embeddings?** Encoding the whole corpus is the expensive part,
> and the corpus doesn't change between queries — so we do it once and persist it.
> At request time the model then only has to embed the user's short query and take
> a dot product against the cached matrix, which keeps each search fast. (The cache
> is gitignored and rebuilt automatically if the row count changes.)

---

## Example use case

A job seeker enters their skills. The system:

1. Encodes the skills with the selected engine — semantic embeddings (default) or
   TF-IDF — into the same vector space as the job corpus
2. Compares against thousands of job descriptions via cosine similarity
3. Returns the top-matching jobs with a similarity score
4. Highlights matched vs. missing skills (skill-gap analysis)

---

## Project structure

```
job-market-intelligence-talent-matching/
├── data/
│   ├── raw/            # source datasets (gitignored; structure kept via .gitkeep)
│   └── processed/      # cleaned, feature-ready data (gitignored)
├── notebooks/
│   └── 01_data_exploration.ipynb   # data merge, cleaning, TF-IDF fitting
├── src/
│   ├── analytics/      # market intelligence analytics
│   ├── matching/       # semantic (embedding) matching engine
│   ├── utils/          # shared preprocessing helpers
│   └── app/            # Streamlit talent-matching app
├── tests/              # unit tests
├── artifacts/
│   ├── trained_models/ # persisted TF-IDF vectorizer + job embeddings (gitignored)
│   └── reports/        # generated analytics CSVs
├── requirements.txt
└── README.md
```

---

## Roadmap

- **Scale the semantic search** — add an ANN index.
- **Resume upload + skill extraction** and a market-intelligence dashboard.
