import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics.pairwise import cosine_similarity

from src.matching import semantic_matcher as sm
from src.utils.preprocessing import (
    clean_skills,
    clean_text,
    load_or_fit_vectorizer,
    partial_match,
)

PROCESSED_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "final_jobs.csv"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "trained_models"
TOP_N = 5


# -------------------------------
# Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DATA_FILE)

    # Parse the comma-separated skills, then filter noise (shared with the notebook).
    df["job_skills_list"] = df["job_skills"].fillna("").apply(
        lambda x: [i.strip().lower() for i in x.split(",") if i.strip() != ""]
    )
    df["clean_skills"] = df["job_skills_list"].apply(clean_skills)

    # Lexical text for TF-IDF mirrors the notebook preprocessing.
    if "clean_summary" not in df.columns:
        df["clean_summary"] = df["job_summary"].fillna("").apply(clean_text)
    else:
        df["clean_summary"] = df["clean_summary"].fillna("").apply(clean_text)

    # Raw summary is kept for the semantic engine (embeddings read natural text).
    df["raw_summary"] = df["job_summary"].fillna("")
    return df


# -------------------------------
# Engines (cached so models/embeddings build once)
# -------------------------------
# Leading underscores tell Streamlit not to hash these large inputs; the corpus
# is static within a session, so the cache is keyed on the function alone.
@st.cache_resource
def get_tfidf_engine(_clean_summaries):
    vectorizer, tfidf_matrix = load_or_fit_vectorizer(
        _clean_summaries,
        MODELS_DIR,
        "tfidf_vectorizer.pkl",
        stop_words="english",
        max_features=5000,
    )
    return vectorizer, tfidf_matrix


@st.cache_resource
def get_semantic_embeddings(_raw_summaries):
    return sm.load_or_build_embeddings(
        _raw_summaries, MODELS_DIR / "job_embeddings.npy"
    )


def tfidf_scores(resume_text, engine):
    vectorizer, tfidf_matrix = engine
    candidate_vector = vectorizer.transform([resume_text])
    return cosine_similarity(candidate_vector, tfidf_matrix).flatten()


def semantic_scores(resume_text, corpus_embeddings):
    query = sm.embed_texts([resume_text])
    # Cosine via dot product (embeddings are L2-normalized).
    return (corpus_embeddings @ query.reshape(-1)).astype(float)


def render_matches(df, ranked_idx, ranked_scores, resume_skills):
    st.subheader("Top Job Matches")
    for rank, idx in enumerate(ranked_idx):
        job = df.iloc[idx]
        job_skills = set(job["clean_skills"])
        matched = partial_match(resume_skills, job_skills)
        missing = job_skills - matched

        st.markdown("---")
        st.markdown(f"### {job['job_title']}")
        st.write(f" {job['company']}")
        st.write(f" {job['job_location']}")
        st.write(f" Match Score: **{round(ranked_scores[rank] * 100, 2)}%**")
        st.write(f" Matched Skills: {', '.join(matched) if matched else 'None'}")
        st.write(f" Missing Skills: {', '.join(list(missing)[:5])}")


# -------------------------------
# UI
# -------------------------------
st.title("AI Job Matcher & Skill Gap Analyzer")

df = load_data()

st.sidebar.header("Matching engine")
engine_choice = st.sidebar.radio(
    "How should jobs be ranked?",
    ("Semantic (AI embeddings)", "TF-IDF (lexical)"),
    help=(
        "Semantic uses sentence-transformer embeddings, so related terms match "
        "even without the same words (e.g. 'ML' is close to 'machine learning'). "
        "TF-IDF is faster but only matches overlapping words."
    ),
)
use_semantic = engine_choice.startswith("Semantic")

st.write("Paste your resume skills below:")
user_input = st.text_area("Enter your skills (comma separated):")

if st.button("Find Matching Jobs"):
    if user_input.strip() == "":
        st.warning("Please enter your skills")
    else:
        resume_skills = {s.strip().lower() for s in user_input.split(",") if s.strip()}
        resume_text = " ".join(resume_skills)

        if use_semantic:
            with st.spinner("Embedding jobs and scoring semantically..."):
                corpus_embeddings = get_semantic_embeddings(df["raw_summary"].tolist())
                scores = semantic_scores(resume_text, corpus_embeddings)
        else:
            engine = get_tfidf_engine(df["clean_summary"])
            scores = tfidf_scores(resume_text, engine)

        # Only keep positive-similarity jobs, then take the top N.
        valid_mask = scores > 0
        if not valid_mask.any():
            st.warning(" No matching jobs found. Try different skills.")
        else:
            valid_indices = df.index[valid_mask].to_numpy()
            valid_scores = scores[valid_mask]
            order = valid_scores.argsort()[::-1][:TOP_N]
            render_matches(df, valid_indices[order], valid_scores[order], resume_skills)
