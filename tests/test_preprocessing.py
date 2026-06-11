"""Unit tests for text and skill preprocessing helpers."""

from src.utils.preprocessing import clean_skills, clean_text, partial_match


class TestCleanSkills:
    def test_lowercases_and_strips(self):
        assert clean_skills(["  Python ", "SQL"]) == ["python", "sql"]

    def test_drops_long_phrases(self):
        # More than 3 words is treated as noise, not a skill.
        assert clean_skills(["a b c d"]) == []
        assert clean_skills(["a b c"]) == ["a b c"]

    def test_drops_experience_and_education_noise(self):
        noisy = ["5 years experience", "bachelor degree", "master of science"]
        assert clean_skills(noisy) == []

    def test_keeps_real_skills(self):
        assert clean_skills(["python", "machine learning"]) == ["python", "machine learning"]


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_removes_non_alpha(self):
        assert clean_text("C++ & Python 3.0!") == "c  python "

    def test_handles_non_string(self):
        assert clean_text(123) == ""


class TestPartialMatch:
    def test_exact_overlap(self):
        assert partial_match({"python"}, {"python", "sql"}) == {"python"}

    def test_token_subset_match(self):
        # All tokens of the skill appear as whole tokens in the job skill.
        assert partial_match({"node"}, {"node.js"}) == {"node"}
        assert partial_match({"machine learning"}, {"machine learning engineer"}) == {
            "machine learning"
        }

    def test_fuzzy_match_near_spelling(self):
        # "react" ~ "reactjs" is close enough to count as a match.
        assert partial_match({"react"}, {"reactjs"}) == {"react"}

    def test_no_match(self):
        assert partial_match({"golang"}, {"python", "sql"}) == set()

    def test_no_substring_false_positives(self):
        # The old substring logic wrongly matched these; the new logic must not.
        assert partial_match({"java"}, {"javascript"}) == set()
        assert partial_match({"go"}, {"google"}) == set()
