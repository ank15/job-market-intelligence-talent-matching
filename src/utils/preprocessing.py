import re
import joblib
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

def partial_match(resume_skills, job_skills):
    """Find partial matches between resume and job skills."""
    matched = set()
    for r in resume_skills:
        for j in job_skills:
            if r in j or j in r:
                matched.add(r)
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