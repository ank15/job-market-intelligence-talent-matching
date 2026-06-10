"""Unit tests for the market intelligence analytics."""

import pandas as pd
import pytest

from src.analytics import market_intelligence as mi


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "job_link": ["a", "b", "c"],
            "job_skills": [
                "Python, SQL, 5 years experience",
                "python, machine learning",
                "SQL, Tableau",
            ],
            "search_city": ["NYC", "NYC", "London"],
            "job_level": ["Mid senior", "Associate", "Mid senior"],
            "job_type": ["Onsite", "Remote", "Onsite"],
        }
    )


def test_explode_skills_filters_noise(sample_df):
    skills = list(mi.explode_skills(sample_df))
    assert "python" in skills
    assert "sql" in skills
    # The experience phrase is dropped by clean_skills.
    assert "5 years experience" not in skills


def test_top_skills_counts(sample_df):
    counts = mi.top_skills(sample_df, top_n=10)
    assert counts["python"] == 2
    assert counts["sql"] == 2
    assert counts["tableau"] == 1


def test_hiring_trends(sample_df):
    trends = mi.hiring_trends(sample_df, "job_level")
    assert trends["Mid senior"] == 2
    assert trends["Associate"] == 1


def test_hiring_trends_unknown_column(sample_df):
    with pytest.raises(KeyError):
        mi.hiring_trends(sample_df, "salary")


def test_skill_demand_by_location(sample_df):
    result = mi.skill_demand_by_location(sample_df, top_n_locations=2, skills_per_location=5)
    assert "NYC" in result
    assert result["NYC"]["python"] == 2


def test_generate_report_writes_files(sample_df, tmp_path):
    written = mi.generate_report(sample_df, out_dir=tmp_path)
    assert len(written) >= 1
    assert (tmp_path / "top_skills.csv").exists()
