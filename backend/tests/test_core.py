import unittest
from backend.core.chat import answer_question, store_session
from backend.core.parser import clean_text, extract_candidate_name, parse_resume_sections
from backend.core.skills import skill_evidence
from backend.core.stage1_bm25 import score_resumes_bm25
from backend.core.stage2_embeddings import score_resumes_semantic
from backend.core.stage3_llm import _local_score


class ParserTests(unittest.TestCase):
    def test_clean_text_preserves_lines_for_name_detection(self):
        text = clean_text("Jane Doe\n jane@example.com \nPython Developer")
        self.assertEqual(extract_candidate_name(text), "Jane Doe")
        self.assertIn("\n", text)

    def test_parse_resume_sections_detects_experience_and_education(self):
        text = "Jane Doe\n\nExperience\nPython dev at Google 2020-2024\n\nEducation\nB.Tech CS\n\nSkills\nPython, FastAPI, Docker"
        sections = parse_resume_sections(text)
        self.assertIn("experience", sections)
        self.assertIn("education", sections)
        self.assertIn("skill", sections)
        self.assertIn("Python", sections["experience"])


class ScoringTests(unittest.TestCase):
    JD = "We need a Python FastAPI engineer with SQL, Docker, REST API skills and 3 years experience."

    def test_skill_evidence_is_explainable(self):
        matched, missing = skill_evidence(self.JD, "Python developer using FastAPI and Docker for 4 years")
        self.assertEqual(matched, ["Python", "FastAPI", "Docker"])
        self.assertIn("SQL", missing)
        self.assertIn("REST APIs", missing)

    def test_single_resume_does_not_fail_bm25_normalization(self):
        resumes = [{"filename": "jane.txt", "text": "Jane Doe\nPython FastAPI SQL Docker REST API engineer", "name": "Jane Doe"}]
        passed = score_resumes_bm25(self.JD, resumes)
        self.assertEqual(len(passed), 1)
        self.assertGreater(passed[0]["bm25_score"], 25)

    def test_offline_semantic_scoring(self):
        resumes = [
            {"filename": "a.txt", "text": "Python FastAPI SQL backend APIs", "bm25_score": 80},
            {"filename": "b.txt", "text": "Junior developer with Python and basic SQL", "bm25_score": 30},
        ]
        results = score_resumes_semantic(self.JD, resumes)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["filename"], "a.txt")
        self.assertGreater(resumes[0]["semantic_score"], resumes[1]["semantic_score"])

    def test_local_deep_score_has_required_breakdown(self):
        resume = {"filename": "a.txt", "text": "Jane Doe\nSenior Python FastAPI Docker engineer at Google with 5 years of experience\nB.Tech Computer Science, IIT Bombay", "bm25_score": 70, "semantic_score": .6}
        result = _local_score(self.JD, resume)
        self.assertTrue(result["matched_skills"])
        self.assertTrue(result["missing_skills"])
        self.assertIn("experience_relevance", result)
        self.assertIn("role_axes", result)
        self.assertIn("companies", result)
        self.assertIn("education_detail", result)
        self.assertGreaterEqual(result["llm_score"], 0)
        self.assertLessEqual(result["llm_score"], 100)
        # Education should detect bachelors from B.Tech in resume
        self.assertIn("education", result.get("breakdown", {}))
        self.assertGreaterEqual(result["breakdown"]["education"], 40)


class ChatTests(unittest.TestCase):
    def test_chat_uses_stored_candidate_evidence(self):
        candidates = [
            {"rank": 1, "candidate_name": "Jane Doe", "final_score": 88, "matched_skills": ["Python", "FastAPI"], "missing_skills": ["SQL"], "experience_relevance": 90, "reason": "Strong backend evidence.", "text": "Python FastAPI backend", "role_axes": {"backend_depth": 8, "frontend_depth": 0, "data_depth": 2, "leadership_depth": 1}},
            {"rank": 2, "candidate_name": "John Roe", "final_score": 70, "matched_skills": ["React"], "missing_skills": ["Python", "SQL"], "experience_relevance": 65, "reason": "Frontend profile.", "text": "React frontend", "role_axes": {"backend_depth": 0, "frontend_depth": 7, "data_depth": 0, "leadership_depth": 0}},
        ]
        store_session("test-session", "Python backend role", candidates)
        response = answer_question("test-session", "Who has the strongest backend experience?")
        self.assertIn("Jane Doe", response["answer"])


if __name__ == "__main__":
    unittest.main()
