"""Central configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

TAXONOMY_JSON = DATA_DIR / "skills_taxonomy.json"
JOB_ROLES_CSV = DATA_DIR / "job_roles.csv"
EVAL_CSV = DATA_DIR / "eval" / "synthetic_resumes.csv"
SAMPLE_RESUME_DIR = DATA_DIR / "sample_resumes"
MODEL_PATH = MODEL_DIR / "career_recommender.joblib"

# --- file validation -------------------------------------------------------
MAX_PDF_MB = 10
MAX_PDF_PAGES = 10
MIN_TEXT_CHARS = 100          # fewer characters than this usually means a scanned/image-only PDF

# --- recommender weights (must sum to 1) -----------------------------------
W_TEXT = 0.50                 # TF-IDF cosine similarity between resume and job description
W_SKILLS = 0.40               # weighted skill coverage (required = 1.0, preferred = 0.5)
W_FIT = 0.10                  # education + experience fit
PREFERRED_WEIGHT = 0.5

# --- LLM -------------------------------------------------------------------
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini", "gemini": "gemini-3.6-flash"}
MAX_LLM_RETRIES = 3
LLM_TIMEOUT_S = 40
MAX_CONTEXT_CHARS = 6000      # resume text sent to an LLM is truncated to this length
MAX_QUESTION_CHARS = 1500
