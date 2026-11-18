"""Privacy helpers: personal data is redacted BEFORE anything is sent to an external AI service.

Principles used by this project:
  1. The resume stays on the user's computer; only redacted text is ever sent to an LLM.
  2. The user must give consent before the first LLM call (or pass --yes).
  3. Resume text is never written to disk or logs by the program; reports contain analysis only.
  4. Real resumes should be kept in the git-ignored `private/` folder.
"""
import re

from src import config
from src.resume_parser import EMAIL_RE, PHONE_RE, is_phone_number

URL_RE = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github|gitlab|twitter|x)\.com/\S+", re.I)
ADDRESS_RE = re.compile(r"\b\d{1,5}[, ]+[A-Za-z0-9 .,-]{3,40}\b(?:street|st\.|road|rd\.|lane|nagar|colony|avenue|ave\.)[^\n]*", re.I)


def redact_pii(text: str, name: str = "") -> str:
    """Replace emails, phone numbers, URLs, addresses, postal codes and the candidate's name."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = PHONE_RE.sub(lambda m: "[PHONE]" if is_phone_number(m.group()) else m.group(), text)
    text = ADDRESS_RE.sub("[ADDRESS]", text)
    if name:
        for part in {name, *name.split()}:
            if len(part) > 2:
                text = re.sub(re.escape(part), "[NAME]", text, flags=re.I)
    return text


def build_llm_context(resume: dict) -> str:
    """Compact, redacted summary of the resume that is safe to send to an LLM."""
    redacted = redact_pii(resume["raw_text"], resume.get("name", ""))
    header = (f"Detected skills: {', '.join(resume['skills']) or 'none'}\n"
              f"Highest education: {resume['education']['highest_level']}\n"
              f"Estimated experience: {resume['experience_years']} years\n"
              f"Certifications: {'; '.join(resume['certifications']) or 'none'}\n---\nRESUME TEXT (personal details removed):\n")
    return (header + redacted)[: config.MAX_CONTEXT_CHARS]


def ask_consent(provider: str, assume_yes: bool = False) -> bool:
    """Explain what will be sent and ask the user to confirm."""
    if assume_yes:
        return True
    print(f"\nPrivacy notice: a version of your resume with personal details (name, email, phone, links, address) removed "
          f"will be sent to the {provider} API to generate the answer.")
    try:
        return input("Do you agree to continue? (y/N): ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False
