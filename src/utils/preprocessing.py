import re
import joblib
from difflib import SequenceMatcher
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_skills(skills_list):
    """Clean and filter job skills list."""
    clean = []
    for skill in skills_list:
        skill = skill.lower().strip()
        if len(skill.split()) > 3:
            continue
        if any(word in skill for word in ["years", "experience", "degree", "bachelor", "master"]):
            continue
        clean.append(skill)
    return clean

def clean_text(text):
    """Clean text by lowercasing and removing non-alphabetic characters."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def _tokenize_skill(skill):
    """Split a skill into normalized word tokens (keeping +/# for c++, c#)."""
    return [t for t in re.split(r"[^a-z0-9+#]+", skill.lower().strip()) if t]


def partial_match(resume_skills, job_skills, fuzzy_threshold=0.8):
    """Match resume skills against job skills.

    Uses three strategies, from strict to lenient, instead of naive substring
    containment (which wrongly matched e.g. "java" → "javascript" or
    "go" → "google"):

    1. Exact match (``python`` == ``python``).
    2. Token-subset match, so a skill matches when all of its tokens appear as
       whole tokens in the job skill (``node`` matches ``node.js``;
       ``machine learning`` matches ``machine learning engineer``).
    3. Fuzzy match via ``difflib`` similarity ratio for near-spellings
       (``react`` ~ ``reactjs``), gated by ``fuzzy_threshold``.
    """
    matched = set()
    for r in resume_skills:
        r = r.lower().strip()
        if not r:
            continue
        r_tokens = set(_tokenize_skill(r))
        for j in job_skills:
            j = j.lower().strip()
            if r == j:
                matched.add(r)
                break
            j_tokens = set(_tokenize_skill(j))
            if r_tokens and r_tokens <= j_tokens:
                matched.add(r)
                break
            if SequenceMatcher(None, r, j).ratio() >= fuzzy_threshold:
                matched.add(r)
                break
    return matched

def load_or_fit_vectorizer(texts, model_dir, vectorizer_path, **kwargs):
    """Load existing TF-IDF vectorizer or fit a new one."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = model_dir / vectorizer_path

    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = vectorizer.transform(texts)
    else:
        vectorizer = TfidfVectorizer(**kwargs)
        tfidf_matrix = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, vectorizer_path)

    return vectorizer, tfidf_matrix