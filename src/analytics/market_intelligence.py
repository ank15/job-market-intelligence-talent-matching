"""Job Market Intelligence analytics.

Turns the processed job-postings dataset into market intelligence:
top in-demand skills, skill demand by location, and hiring trends by
job level and work type.

Note: the dataset has no salary field, so salary-based analysis is not
implemented here (see README roadmap).

Run as a script to generate report files:
    python -m src.analytics.market_intelligence
"""

from pathlib import Path

import pandas as pd

from src.utils.preprocessing import clean_skills

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "final_jobs.csv"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"


def load_jobs(path=PROCESSED_DATA_FILE, dedupe=True):
    """Load processed jobs. Deduplicate on job_link so each posting counts once."""
    df = pd.read_csv(path)
    if dedupe and "job_link" in df.columns:
        df = df.drop_duplicates("job_link").reset_index(drop=True)
    return df


def explode_skills(df):
    """Return a long DataFrame of (row index, skill) pairs.

    Parses the comma-separated ``job_skills`` column using the same
    convention as the matching pipeline, then applies ``clean_skills`` to
    drop noise (long phrases, experience/degree requirements).
    """
    skills_list = df["job_skills"].fillna("").apply(
        lambda x: [s.strip().lower() for s in x.split(",") if s.strip()]
    )
    cleaned = skills_list.apply(clean_skills)
    exploded = cleaned.explode().dropna()
    exploded = exploded[exploded != ""]
    return exploded


def top_skills(df, top_n=20):
    """Most in-demand skills across all postings, as a count Series."""
    return explode_skills(df).value_counts().head(top_n)


def skill_demand_by_location(df, location_col="search_city", top_n_locations=5, skills_per_location=10):
    """Top skills within each of the busiest locations.

    Returns a dict mapping location -> count Series of its top skills.
    """
    if location_col not in df.columns:
        raise KeyError(f"{location_col!r} not in dataframe columns")

    locations = df[location_col].value_counts().head(top_n_locations).index
    result = {}
    for loc in locations:
        subset = df[df[location_col] == loc]
        result[loc] = explode_skills(subset).value_counts().head(skills_per_location)
    return result


def hiring_trends(df, by):
    """Posting counts grouped by a categorical column (e.g. job_level, job_type)."""
    if by not in df.columns:
        raise KeyError(f"{by!r} not in dataframe columns")
    return df[by].value_counts(dropna=False)


def generate_report(df=None, out_dir=REPORTS_DIR, top_n=25):
    """Compute all analytics and write them to CSV files under ``out_dir``.

    Returns the list of written file paths.
    """
    if df is None:
        df = load_jobs()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    skills = top_skills(df, top_n=top_n).rename_axis("skill").reset_index(name="count")
    path = out_dir / "top_skills.csv"
    skills.to_csv(path, index=False)
    written.append(path)

    for col in ("job_level", "job_type", "search_country"):
        if col in df.columns:
            trend = hiring_trends(df, col).rename_axis(col).reset_index(name="count")
            path = out_dir / f"hiring_trends_by_{col}.csv"
            trend.to_csv(path, index=False)
            written.append(path)

    if "search_city" in df.columns:
        rows = []
        for loc, series in skill_demand_by_location(df).items():
            for skill, count in series.items():
                rows.append({"location": loc, "skill": skill, "count": count})
        path = out_dir / "skill_demand_by_location.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        written.append(path)

    return written


def main():
    df = load_jobs()
    print(f"Loaded {len(df):,} unique postings\n")

    print("=== Top 15 in-demand skills ===")
    print(top_skills(df, top_n=15).to_string())

    for col in ("job_level", "job_type"):
        print(f"\n=== Hiring trends by {col} ===")
        print(hiring_trends(df, col).to_string())

    written = generate_report(df)
    print(f"\nWrote {len(written)} report files to {REPORTS_DIR}:")
    for p in written:
        print(f"  - {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
