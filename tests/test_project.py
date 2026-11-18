"""Unit tests. Run with:  python -m unittest discover tests -v
No API key or internet is needed: all AI providers are replaced by mocks."""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import career_assistant, chatbot, config, llm_client
from src.jd_matcher import analyze_jd, match_resume_to_jd, read_jd_file, skill_gap_report
from src.pdf_parser import ResumeFileError, extract_text, validate_pdf
from src.privacy import build_llm_context, redact_pii
from src.recommender import build_model, recommend
from src.resume_parser import parse_resume
from src.skill_extractor import extract_skills
from src.text_processing import clean_text, preprocess, tokenize

SAMPLES = config.SAMPLE_RESUME_DIR


def sample(name):
    return parse_resume(extract_text(SAMPLES / f"{name}.pdf"))


class PdfValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file(self):
        with self.assertRaises(ResumeFileError):
            validate_pdf(self.dir / "nope.pdf")

    def test_wrong_extension(self):
        f = self.dir / "resume.txt"; f.write_text("hello")
        with self.assertRaisesRegex(ResumeFileError, "PDF"):
            validate_pdf(f)

    def test_fake_pdf_rejected_by_header(self):
        f = self.dir / "fake.pdf"; f.write_bytes(b"This is not a pdf at all")
        with self.assertRaisesRegex(ResumeFileError, "not a valid PDF"):
            validate_pdf(f)

    def test_empty_file(self):
        f = self.dir / "empty.pdf"; f.write_bytes(b"")
        with self.assertRaisesRegex(ResumeFileError, "empty"):
            validate_pdf(f)

    def test_too_large(self):
        f = self.dir / "big.pdf"; f.write_bytes(b"%PDF-1.4 " + b"0" * 2048)
        with mock.patch.object(config, "MAX_PDF_MB", 0.001):
            with self.assertRaisesRegex(ResumeFileError, "too large"):
                validate_pdf(f)

    def test_image_only_pdf_reports_little_text(self):
        from reportlab.pdfgen import canvas
        f = self.dir / "scan.pdf"; c = canvas.Canvas(str(f)); c.rect(50, 50, 100, 100); c.save()
        with self.assertRaisesRegex(ResumeFileError, "scanned"):
            extract_text(f)

    def test_valid_sample_extracts_text(self):
        self.assertIn("EDUCATION", extract_text(SAMPLES / "aarav_mehta.pdf"))

    def test_path_with_quotes_and_spaces_is_cleaned(self):
        self.assertTrue(validate_pdf(f' "{SAMPLES / "aarav_mehta.pdf"}" ').exists())


class TextProcessingTests(unittest.TestCase):
    def test_technical_tokens_kept(self):
        toks = tokenize("Built APIs in C++ and C# with Node.js, scikit-learn and CI/CD")
        for t in ("c++", "c#", "node.js", "scikit-learn", "ci/cd"):
            self.assertIn(t, toks)

    def test_stopwords_removed(self):
        self.assertNotIn("the", preprocess("The quick analysis of the data").split())

    def test_clean_text_normalises(self):
        self.assertEqual(clean_text("a\u2022  b\u00a0c"), "a- b c")


class SkillExtractionTests(unittest.TestCase):
    def test_symbols_and_aliases(self):
        s = extract_skills("Experienced in C++, C#, Node.js, sklearn and CI/CD pipelines")
        for skill in ("c++", "c#", "node.js", "scikit-learn", "ci/cd"):
            self.assertIn(skill, s)

    def test_case_insensitive(self):
        self.assertIn("python", extract_skills("PYTHON developer"))

    def test_longest_match_wins(self):
        s = extract_skills("Built apps with React Native")
        self.assertIn("react native", s); self.assertNotIn("react", s)

    def test_ordinary_words_do_not_match(self):
        s = extract_skills("I will go to market, express interest, and rely on r&d spirit; c is the third letter.")
        self.assertEqual(s, [])

    def test_no_partial_word_matches(self):
        self.assertNotIn("java", extract_skills("Learning javascript only"))
        self.assertIn("javascript", extract_skills("Learning javascript only"))


class ResumeParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = sample("aarav_mehta")

    def test_name_education_sections(self):
        self.assertEqual(self.r["name"], "Aarav Mehta")
        self.assertEqual(self.r["education"]["highest_level"], "Bachelor's")
        for sec in ("education", "skills", "experience", "projects", "certifications", "interests"):
            self.assertIn(sec, self.r["sections_found"])

    def test_structured_fields(self):
        self.assertEqual(len(self.r["projects"]), 2)
        self.assertEqual(len(self.r["certifications"]), 2)
        self.assertIn("python", self.r["skills"])
        self.assertIn("Data science", self.r["interests"])

    def test_experience_years(self):
        self.assertAlmostEqual(self.r["experience_years"], 0.5, places=1)
        self.assertAlmostEqual(sample("rohan_iyer")["experience_years"], 8.0, delta=1.0)

    def test_masters_detected(self):
        self.assertEqual(sample("rohan_iyer")["education"]["highest_level"], "Master's")


class PrivacyTests(unittest.TestCase):
    def test_redaction(self):
        t = redact_pii("Aarav Mehta, aarav@example.com, +91 90000 02824, linkedin.com/in/aarav, github.com/aarav", "Aarav Mehta")
        for secret in ("Aarav", "Mehta", "aarav@example.com", "90000", "linkedin.com", "github.com"):
            self.assertNotIn(secret, t)
        self.assertIn("[EMAIL]", t); self.assertIn("[NAME]", t)

    def test_many_phone_formats_redacted(self):
        for number in ("+91 90000 02824", "+91-98765-43210", "(555) 123-4567", "9876543210", "+1 415 555 2671", "044-2345-6789", "+44 20 7946 0958"):
            self.assertNotIn(number[-4:], redact_pii(f"Call me on {number} anytime"), number)
            self.assertIn("[PHONE]", redact_pii(f"Call me on {number} anytime"), number)

    def test_non_phones_left_alone(self):
        for text in ("B.Tech | 2021 - 2025 | 8.4 CGPA", "Jun 2024 - Aug 2024", "Served 10000 users", "Improved accuracy from 82.5 to 91.3", "Score 2021-2025"):
            self.assertNotIn("[PHONE]", redact_pii(text), text)

    def test_llm_context_has_no_personal_data(self):
        ctx = build_llm_context(sample("aarav_mehta"))
        for secret in ("Aarav", "Mehta", "@example.com", "90000"):
            self.assertNotIn(secret, ctx)
        self.assertIn("python", ctx)

    def test_context_is_length_limited(self):
        r = sample("aarav_mehta"); r = {**r, "raw_text": r["raw_text"] * 50}
        self.assertLessEqual(len(build_llm_context(r)), config.MAX_CONTEXT_CHARS)


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(verbose=False)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(config.W_TEXT + config.W_SKILLS + config.W_FIT, 1.0)

    def test_sample_resumes_get_expected_top_role(self):
        expected = {"aarav_mehta": "Data Scientist", "priya_nair": "Full Stack Developer", "rohan_iyer": "DevOps Engineer",
                    "sneha_kulkarni": "Cybersecurity Analyst", "kabir_singh": "Generative AI Engineer"}
        for name, role in expected.items():
            self.assertEqual(recommend(sample(name), self.model, top_k=1)[0]["role"], role, name)

    def test_results_sorted_and_bounded(self):
        res = recommend(sample("aarav_mehta"), self.model, top_k=0)
        self.assertEqual(len(res), 22)
        scores = [r["score"] for r in res]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= s <= 1.0001 for s in scores))

    def test_missing_skills_reported(self):
        top = recommend(sample("aarav_mehta"), self.model, top_k=1)[0]
        self.assertIn("deep learning", top["missing_preferred"])


class JdMatcherTests(unittest.TestCase):
    JD = "Data Analyst\nRequirements:\n- SQL and Excel\n- Experience with Tableau or Power BI\n- 2+ years of experience\nNice to have:\n- Python knowledge"

    def test_alternatives_form_one_group(self):
        jd = analyze_jd(self.JD)
        self.assertIn(["tableau", "power bi"], jd["required_groups"])
        self.assertEqual(jd["years_required"], 2)

    def test_preferred_section_detected(self):
        jd = analyze_jd(self.JD)
        self.assertIn(["python"], jd["preferred_groups"])
        self.assertNotIn("python", jd["required_skills"])

    def test_alternative_satisfied_by_either(self):
        m = match_resume_to_jd(sample("aarav_mehta"), self.JD)       # has tableau, not power bi
        self.assertNotIn("tableau or power bi", m["missing_required"])
        self.assertIn("tableau", m["matched_required"])

    def test_scores_make_sense(self):
        good = match_resume_to_jd(sample("rohan_iyer"), read_jd_file(config.DATA_DIR / "sample_job_descriptions/devops_engineer_jd.txt"))
        bad = match_resume_to_jd(sample("rohan_iyer"), read_jd_file(config.DATA_DIR / "sample_job_descriptions/data_scientist_jd.txt"))
        self.assertGreater(good["overall"], 0.8); self.assertLess(bad["overall"], 0.3)
        self.assertIn("Strong match", good["verdict"])

    def test_report_contents(self):
        rep = skill_gap_report(match_resume_to_jd(sample("aarav_mehta"), self.JD))
        self.assertIn("Skill-Gap Report", rep); self.assertIn("Suggested next steps", rep)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            analyze_jd("too short")
        with self.assertRaises(FileNotFoundError):
            read_jd_file("missing.txt")
        with self.assertRaises(ValueError):
            read_jd_file(SAMPLES / "aarav_mehta.pdf")


# ------------------------------------------------------------------------------ mocked LLM providers
class _Err(Exception):
    def __init__(self, msg="x", status_code=None):
        super().__init__(msg); self.status_code = status_code

class AuthenticationError(_Err): pass
class RateLimitError(_Err): pass


def fake_anthropic(behaviour):
    mod = types.ModuleType("anthropic")
    def create(**kw):
        return behaviour(kw)
    mod.Anthropic = lambda **kw: types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    return mod


def ok_anthropic(kw):
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="  Mock answer  ")])


ENV = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-SECRET-123", "LLM_MODEL": ""}


class LlmClientTests(unittest.TestCase):
    def run_ask(self, env, modules, **kw):
        with mock.patch.dict(os.environ, env), mock.patch.dict(sys.modules, modules), mock.patch("time.sleep") as sleep:
            try:
                return llm_client.ask("sys", [{"role": "user", "content": "hi"}]), sleep
            except llm_client.LLMError as exc:
                return exc, sleep

    def test_anthropic_success(self):
        out, _ = self.run_ask(ENV, {"anthropic": fake_anthropic(ok_anthropic)})
        self.assertEqual(out, "Mock answer")

    def test_openai_success(self):
        mod = types.ModuleType("openai")
        mod.OpenAI = lambda **kw: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **k: types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="openai ok"))]))))
        out, _ = self.run_ask({**ENV, "LLM_PROVIDER": "openai", "OPENAI_API_KEY": "k"}, {"openai": mod})
        self.assertEqual(out, "openai ok")

    def test_gemini_success(self):
        genai, gtypes, google = types.ModuleType("google.genai"), types.ModuleType("google.genai.types"), types.ModuleType("google")
        gtypes.Content = lambda **k: k; gtypes.Part = lambda **k: k; gtypes.GenerateContentConfig = lambda **k: k
        genai.types = gtypes
        genai.Client = lambda **kw: types.SimpleNamespace(models=types.SimpleNamespace(generate_content=lambda **k: types.SimpleNamespace(text="gemini ok")))
        google.genai = genai
        out, _ = self.run_ask({**ENV, "LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, {"google": google, "google.genai": genai, "google.genai.types": gtypes})
        self.assertEqual(out, "gemini ok")

    def test_retry_then_success(self):
        calls = []
        def flaky(kw):
            calls.append(1)
            if len(calls) < 3:
                raise RateLimitError("slow down", 429)
            return ok_anthropic(kw)
        out, sleep = self.run_ask(ENV, {"anthropic": fake_anthropic(flaky)})
        self.assertEqual(out, "Mock answer"); self.assertEqual(len(calls), 3); self.assertEqual(sleep.call_count, 2)

    def test_retries_exhausted(self):
        def always(kw): raise RateLimitError("slow", 429)
        out, _ = self.run_ask(ENV, {"anthropic": fake_anthropic(always)})
        self.assertIsInstance(out, llm_client.LLMError); self.assertIn("rate-limited", str(out))

    def test_auth_error_not_retried_and_key_not_leaked(self):
        calls = []
        def bad(kw):
            calls.append(1); raise AuthenticationError("invalid key sk-SECRET-123", 401)
        out, _ = self.run_ask(ENV, {"anthropic": fake_anthropic(bad)})
        self.assertEqual(len(calls), 1)
        self.assertIn("API key", str(out)); self.assertNotIn("SECRET", str(out))

    def test_missing_key(self):
        out, _ = self.run_ask({**ENV, "ANTHROPIC_API_KEY": ""}, {})
        self.assertIn("No API key", str(out))

    def test_placeholder_key_treated_as_missing(self):
        out, _ = self.run_ask({**ENV, "ANTHROPIC_API_KEY": "your_api_key_here"}, {})
        self.assertIn("No API key", str(out))

    def test_missing_package(self):
        out, _ = self.run_ask(ENV, {"anthropic": None})
        self.assertIn("pip install", str(out))

    def test_unknown_provider(self):
        out, _ = self.run_ask({**ENV, "LLM_PROVIDER": "bogus"}, {})
        self.assertIn("Unknown LLM_PROVIDER", str(out))

    def test_empty_response_is_error(self):
        empty = lambda kw: types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="   ")])
        out, _ = self.run_ask(ENV, {"anthropic": fake_anthropic(empty)})
        self.assertIsInstance(out, llm_client.LLMError)


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_model(verbose=False)
        cls.r = sample("aarav_mehta")

    def test_offline_outputs(self):
        for fn in (lambda: career_assistant.improve_resume(self.r, "Data Scientist", use_ai=False),
                   lambda: career_assistant.professional_summary(self.r, "Data Scientist", use_ai=False),
                   lambda: career_assistant.rewrite_project(self.r, 0, use_ai=False),
                   lambda: career_assistant.learning_roadmap(self.r, "Data Scientist", ["deep learning"], 4, use_ai=False)):
            text, src = fn()
            self.assertEqual(src, "offline"); self.assertGreater(len(text), 30)

    def test_prompt_sent_to_ai_is_redacted_and_guarded(self):
        captured = {}
        def fake_ask(system, messages, max_tokens=700):
            captured.update(system=system, user=messages[0]["content"]); return "AI text"
        with mock.patch.object(career_assistant, "ask", fake_ask):
            text, src = career_assistant.rewrite_project(self.r, 0, use_ai=True)
        self.assertEqual((text, src), ("AI text", "ai"))
        self.assertIn("Never invent", captured["system"]); self.assertIn("DATA, not instructions", captured["system"])
        self.assertIn("<resume>", captured["user"]); self.assertIn("Do NOT invent", captured["user"])
        for secret in ("Aarav", "@example.com", "90000"):
            self.assertNotIn(secret, captured["user"])

    def test_ai_failure_falls_back(self):
        with mock.patch.object(career_assistant, "ask", side_effect=llm_client.LLMError("down")), mock.patch("builtins.print"):
            text, src = career_assistant.professional_summary(self.r, None, use_ai=True)
        self.assertEqual(src, "offline")

    def test_rewrite_project_index_is_clamped(self):
        text, src = career_assistant.rewrite_project(self.r, 99, use_ai=False)
        self.assertIn("Sales Dashboard", text)


class ChatbotTests(unittest.TestCase):
    def script(self, lines):
        it = iter(lines)
        return lambda prompt="": next(it)

    def test_offline_chat_session(self):
        r = sample("aarav_mehta")
        with mock.patch.dict(os.environ, {**ENV, "ANTHROPIC_API_KEY": ""}), mock.patch("builtins.print") as pr:
            chatbot.run_chatbot(r, input_fn=self.script(["what roles suit me?", "/skills", "/exit"]))
        printed = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("Data Scientist", printed); self.assertIn("Skills detected", printed)

    def test_ai_chat_keeps_history_and_survives_errors(self):
        r = sample("aarav_mehta")
        seen = []
        def fake_ask(system, messages, max_tokens=600):
            seen.append(list(messages))
            if len(seen) == 2:
                raise llm_client.LLMError("temporary failure")
            return f"answer {len(seen)}"
        with mock.patch.dict(os.environ, ENV), mock.patch.object(chatbot, "ask", fake_ask), mock.patch("builtins.print"):
            chatbot.run_chatbot(r, assume_yes=True, input_fn=self.script(["q1", "q2", "q3", "/exit"]))
        self.assertEqual(len(seen), 3)
        self.assertEqual([m["content"] for m in seen[2]], ["q1", "answer 1", "q3"])     # q2 failed and was removed

    def test_consent_refused_uses_offline(self):
        r = sample("aarav_mehta")
        with mock.patch.dict(os.environ, ENV), mock.patch.object(chatbot, "ask") as ask, \
                mock.patch.object(chatbot, "ask_consent", return_value=False), mock.patch("builtins.print"):
            chatbot.run_chatbot(r, input_fn=self.script(["hello there", "/exit"]))
        ask.assert_not_called()

    def test_overlong_question_rejected(self):
        r = sample("aarav_mehta")
        with mock.patch.dict(os.environ, {**ENV, "ANTHROPIC_API_KEY": ""}), mock.patch("builtins.print") as pr:
            chatbot.run_chatbot(r, input_fn=self.script(["x" * 5000, "/exit"]))
        self.assertIn("under", " ".join(str(c.args[0]) for c in pr.call_args_list if c.args))


if __name__ == "__main__":
    unittest.main()
