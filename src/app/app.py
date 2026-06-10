import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.preprocessing import clean_skills, clean_text, partial_match, load_or_fit_vectorizer

# -------------------------------
# Load Data
# -------------------------------
PROCESSED_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "final_jobs.csv"

@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_DATA_FILE)

df = load_data()

# -------------------------------
# Skill Cleaning Function
# -------------------------------
# Functions imported from src.utils.preprocessing

# Convert string to list safely
df["job_skills_list"] = df["job_skills"].fillna("").apply(
    lambda x: [i.strip().lower() for i in x.split(",") if i.strip() != ""]
)

df["clean_skills"] = df["job_skills_list"].apply(clean_skills)

# Prepare text data exactly like the notebook pipeline
# Use clean_summary to mirror notebook preprocessing
if "clean_summary" not in df.columns:
    df["clean_summary"] = df["job_summary"].fillna("").apply(clean_text)
else:
    df["clean_summary"] = df["clean_summary"].fillna("").apply(clean_text)

# -------------------------------
# NLP Model
# -------------------------------
vectorizer, tfidf_matrix = load_or_fit_vectorizer(
    df["clean_summary"],
    PROJECT_ROOT / "artifacts" / "trained_models",
    "tfidf_vectorizer.pkl",
    stop_words='english',
    max_features=5000
)


# -------------------------------
# Partial Matching Function
# -------------------------------
# Function imported from src.utils.preprocessing

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💼 AI Job Matcher & Skill Gap Analyzer")

st.write("Paste your resume skills below:")

user_input = st.text_area("Enter your skills (comma separated):")

if st.button("Find Matching Jobs"):

    if user_input.strip() == "":
        st.warning("Please enter your skills")
    
    else:
        # Process resume
        resume_skills = set([s.strip().lower() for s in user_input.split(",")])

        # Convert to string for TF-IDF
        resume_text = " ".join(resume_skills)

        candidate_vector = vectorizer.transform([resume_text])
        similarity_scores = cosine_similarity(candidate_vector, tfidf_matrix)

        scores = similarity_scores.flatten()

        #  FILTER: only scores > 0
        valid_mask = scores > 0

        if not valid_mask.any():
            st.warning(" No matching jobs found. Try different skills.")

        else:
            # Get valid scores and indices
            valid_scores = scores[valid_mask]
            valid_indices = df.index[valid_mask]

            # Get top 5 from valid ones
            top_idx = valid_scores.argsort()[-5:][::-1]
            top_matches = valid_indices[top_idx]
            top_scores = valid_scores[top_idx]

            st.subheader("🔍 Top Job Matches")

            for i, idx in enumerate(top_matches):

                job = df.iloc[idx]

                job_skills = set(job["clean_skills"])
                matched = partial_match(resume_skills, job_skills)
                missing = job_skills - matched

                st.markdown("---")
                st.markdown(f"### 🔹 {job['job_title']}")
                st.write(f" {job['company']}")
                st.write(f" {job['job_location']}")
                st.write(f" Match Score: **{round(top_scores[i]*100, 2)}%**")

                st.write(f" Matched Skills: {', '.join(matched) if matched else 'None'}")
                st.write(f" Missing Skills: {', '.join(list(missing)[:5])}")