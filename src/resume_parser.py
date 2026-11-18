"""Convert resume text into structured, machine-readable data.

Extracts: name, contact flags, summary, education, skills, certifications, projects,
work experience (with an estimate of years) and professional interests.
"""
import re
from datetime import date

from src.skill_extractor import extract_skills, group_by_category
from src.text_processing import clean_text

SECTION_HEADINGS = {
    "summary": ["summary", "profile", "objective", "career objective", "professional summary", "about me"],
    "education": ["education", "academic background", "qualifications", "educational qualifications"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "technologies"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "internships", "internship experience"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "certifications": ["certifications", "certificates", "courses", "licenses", "certifications and courses", "training"],
    "interests": ["interests", "professional interests", "areas of interest", "hobbies"],
}
HEADING_LOOKUP = {h: sec for sec, hs in SECTION_HEADINGS.items() for h in hs}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Phone numbers: a run of digits with spaces/dashes/dots/brackets that contains 9-15 digits in total.
# The predicate below rejects year ranges ("2021 - 2025") and decimals ("8.4 CGPA") that could look similar.
PHONE_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().-]{7,20}\d(?![\w])")
YEAR_RANGE_ONLY = re.compile(r"(19|20)\d{2}\s*[-/]\s*(19|20)\d{2}")


def is_phone_number(candidate: str) -> bool:
    digits = sum(c.isdigit() for c in candidate)
    return 9 <= digits <= 15 and not YEAR_RANGE_ONLY.search(candidate) and candidate.count(".") <= 1 and "\n" not in candidate


def find_phone_numbers(text: str) -> list:
    return [m.group().strip() for m in PHONE_RE.finditer(text) if is_phone_number(m.group())]


YEAR_RANGE_RE = re.compile(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(19|20)(\d{2})\s*(?:-|to)\s*"
                           r"(?:((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(19|20)(\d{2})|(present|current|now|ongoing))", re.I)
DEGREE_LEVELS = [
    (5, "PhD", r"\b(ph\.?d|doctorate|doctor of philosophy)\b"),
    (4, "Master's", r"\b(m\.?tech|m\.?e\.?|m\.?sc|m\.?s\.?|m\.?c\.?a|mba|master(?:'s|s)?(?: of)?|post[- ]?graduate)\b"),
    (3, "Bachelor's", r"\b(b\.?tech|b\.?e\.?|b\.?sc|b\.?s\.?|b\.?c\.?a|bba|b\.?com|bachelor(?:'s|s)?(?: of)?|undergraduate|graduate in)\b"),
    (2, "Diploma", r"\b(diploma|polytechnic)\b"),
    (1, "High school", r"\b(high school|higher secondary|hsc|12th|senior secondary)\b"),
]
FIELD_WORDS = ["computer science", "information technology", "artificial intelligence", "data science", "electronics", "electrical",
               "mechanical", "statistics", "mathematics", "business", "commerce", "economics", "design", "engineering", "software"]
ACTION_VERBS = {"built", "developed", "created", "designed", "implemented", "led", "managed", "automated", "analysed", "analyzed", "trained", "deployed",
                "improved", "reduced", "increased", "wrote", "optimised", "optimized", "launched", "configured", "monitored", "provisioned", "ran",
                "containerised", "containerized", "collaborated", "organised", "organized", "tracked", "fine-tuned", "explored", "exposed", "administered", "scanned"}


def split_sections(text: str) -> dict:
    """Split text into sections by recognising heading lines (e.g. 'EDUCATION', 'Work Experience:')."""
    sections, current = {"header": []}, "header"
    for line in text.split("\n"):
        key = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if key in HEADING_LOOKUP and len(line.strip()) <= 40:
            current = HEADING_LOOKUP[key]
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line.strip())
    return {k: [l for l in v if l] for k, v in sections.items()}


def guess_name(lines) -> str:
    for line in lines[:5]:
        if EMAIL_RE.search(line) or re.search(r"\d", line) or "|" in line:
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w.replace(".", "").replace("-", "").isalpha() for w in words):
            return line.strip()
    return ""


def extract_education(lines, whole_text) -> dict:
    block = " ".join(lines) if lines else whole_text
    level, label = 0, "Not detected"
    for lvl, name, pattern in DEGREE_LEVELS:
        if re.search(pattern, block, re.I):
            level, label = lvl, name
            break
    fields = [f for f in FIELD_WORDS if f in block.lower()]
    return {"highest_level": label, "level_rank": level, "fields": fields, "details": lines[:6]}


def estimate_experience_years(lines) -> float:
    """Sum of date ranges (merged if overlapping) found in the experience section; also honours '3 years of experience'."""
    this_year = date.today().year
    spans = []
    for m in YEAR_RANGE_RE.finditer(" ".join(lines)):
        start = int(m.group(2) + m.group(3)) + (0.5 if m.group(1) and m.group(1).lower().startswith(("jul", "aug", "sep", "oct", "nov", "dec")) else 0)
        end = this_year + 0.5 if m.group(7) else int(m.group(5) + m.group(6)) + 0.5
        if end >= start:
            spans.append((start, end))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    total = sum(e - s for s, e in merged)
    stated = re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)\s+of\s+(?:professional\s+)?experience", " ".join(lines), re.I)
    if stated:
        total = max(total, float(max(stated, key=int)))
    return round(min(total, 40), 1)


def split_entries(lines) -> list:
    """Group section lines into entries: a new entry starts at a non-bullet line that follows bullet lines."""
    entries, cur = [], []
    for line in lines:
        is_bullet = line.startswith(("-", "*"))
        if cur and not is_bullet and cur[-1].startswith(("-", "*")):
            entries.append(cur)
            cur = []
        cur.append(line)
    if cur:
        entries.append(cur)
    return entries


def parse_projects(lines) -> list:
    projects = []
    for entry in split_entries(lines):
        title = entry[0].lstrip("-* ").strip()
        desc = " ".join(l.lstrip("-* ").strip() for l in entry[1:]) if len(entry) > 1 else ""
        projects.append({"title": title, "description": desc, "skills": extract_skills(title + " " + desc)})
    return projects


def parse_experience(lines) -> list:
    jobs = []
    for entry in split_entries(lines):
        header = entry[0].lstrip("-* ").strip()
        bullets = [l.lstrip("-* ").strip() for l in entry[1:]]
        jobs.append({"title_and_company": header, "bullets": bullets, "skills": extract_skills(" ".join(entry))})
    return jobs


def parse_list_section(lines) -> list:
    items = []
    for l in lines:
        l = l.lstrip("-* ").strip()
        items += [p.strip() for p in re.split(r"[;,]", l) if p.strip()] if "," in l and len(l) < 120 and not re.search(r"\(", l) else [l]
    return items


def parse_resume(text: str) -> dict:
    """Main entry point: resume text -> structured dictionary."""
    text = clean_text(text)
    sec = split_sections(text)
    body = "\n".join(l for k, v in sec.items() if k != "header" for l in v)    # text without the contact header
    skills = extract_skills(text)
    exp_lines = sec.get("experience", [])
    certs = [l.lstrip("-* ").strip() for l in sec.get("certifications", [])]
    return {
        "name": guess_name(sec.get("header", []) or text.split("\n")),
        "has_email": bool(EMAIL_RE.search(text)),
        "has_phone": bool(find_phone_numbers(text)),
        "summary": " ".join(sec.get("summary", [])),
        "education": extract_education(sec.get("education", []), text),
        "skills": skills,
        "skills_by_category": group_by_category(skills),
        "certifications": certs,
        "projects": parse_projects(sec.get("projects", [])),
        "experience": parse_experience(exp_lines),
        "experience_years": estimate_experience_years(exp_lines or text.split("\n")),
        "interests": parse_list_section(sec.get("interests", [])),
        "sections_found": [k for k in sec if k != "header"],
        "raw_text": text,
        "body_text": body or text,
    }
