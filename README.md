# Job Market Intelligence & Talent Matching

An end-to-end machine learning system that transforms raw job-market data into
actionable market intelligence and an explainable talent-matching score.

The system analyzes ~12,000 real job postings to:

- Identify in-demand skills across the market
- Analyze skill demand by location and hiring trends by seniority / work type
- Compute a job–candidate fit score using NLP (TF-IDF + cosine similarity)
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
and a quantitative, **explainable** talent-matching score.

---

## System architecture

```
Raw Data → Validation → Preprocessing → Feature Engineering →
   ├── Market Intelligence Analytics   (src/analytics)
   └── NLP Talent Matching Engine      (src/app) → Streamlit Dashboard
```

---

## Modules

### Module 1 — Job Market Intelligence ✅ implemented

`src/analytics/market_intelligence.py`

- **Top in-demand skills** across all postings
- **Skill demand by location** (top skills within the busiest cities)
- **Hiring trends** by `job_level` (e.g. Mid-senior vs. Associate) and
  `job_type` (Onsite / Remote / Hybrid)
- Generates CSV reports under `artifacts/reports/`

Techniques: exploratory data analysis, grouped aggregation, rule-based skill
cleaning, and missing-value handling.

### Module 2 — NLP Talent Matching Engine ✅ implemented

`src/app/app.py`, `src/utils/preprocessing.py`

Computes similarity between a candidate's skills and job descriptions.

Pipeline:

1. Clean skills and job summaries (shared functions in `src/utils/preprocessing.py`,
   so the app mirrors the notebook exactly — no train/serve skew)
2. **TF-IDF** vectorization of job summaries (`max_features=5000`, English stop words),
   persisted with `joblib` so it isn't refit on every run
3. **Cosine similarity** between the resume vector and every job
4. Return the top 5 matches with a **match score (0–100)**, matched skills, and
   missing skills

This provides a practical and explainable job-fit scoring system.

---

## Testing & validation

- Unit tests for preprocessing functions (`clean_skills`, `clean_text`, `partial_match`)
- Unit tests for the market-intelligence analytics
- Run with `pytest` (16 tests):

```bash
pytest -q
```

---

## Tech stack

Python · pandas · numpy · scikit-learn (TF-IDF, cosine similarity) ·
joblib (model persistence) · matplotlib / seaborn (EDA) · Streamlit (UI) · pytest

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

---

## Example use case

A job seeker enters their skills. The system:

1. Vectorizes the skills in the same TF-IDF space as the job corpus
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
│   ├── utils/          # shared preprocessing helpers
│   └── app/            # Streamlit talent-matching app
├── tests/              # unit tests
├── artifacts/
│   ├── trained_models/ # persisted TF-IDF vectorizer (gitignored)
│   └── reports/        # generated analytics CSVs
├── requirements.txt
└── README.md
```

---

## Roadmap

- **Semantic matching** — replace TF-IDF with sentence embeddings so "ML" ≈
  "machine learning"; add an ANN index (FAISS) to scale beyond brute-force cosine.
- **Stronger skill matching** — replace substring containment in `partial_match`
  with token/fuzzy matching to avoid false positives.
- **Resume upload + skill extraction** (NER) and a market-intelligence dashboard.
