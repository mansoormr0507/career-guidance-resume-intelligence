"""Job-description analysis, resume-to-JD matching and skill-gap reporting."""
import re
import re as _re
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.recommender import load_model, resume_text_for_matching
from src.skill_extractor import extract_skill_spans, extract_skills, group_by_category
from src.text_processing import clean_text

PREFERRED_HEAD = re.compile(r"(nice to have|good to have|preferred|bonus|desirable|plus points|added advantage)", re.I)
REQUIRED_HEAD = re.compile(r"(requirements?|qualifications?|must have|responsibilities|what you|you will|skills required|required)", re.I)
INLINE_PREFERRED = re.compile(r"(is a plus|are a plus|a plus\b|nice to have|preferred|bonus|advantage)", re.I)
MAX_JD_CHARS = 20000


def read_jd_file(path) -> str:
    path = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Job description file not found: {path}")
    if path.suffix.lower() not in (".txt", ".md"):
        raise ValueError("Job description files must be plain text (.txt or .md).")
    if path.stat().st_size > MAX_JD_CHARS * 4:
        raise ValueError("Job description file is too large.")
    return path.read_text(encoding="utf-8", errors="ignore")


def read_jd_terminal() -> str:
    print("Paste the job description. Finish with a line containing only END:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


ALT_GAP = _re.compile(r"^\s*(,\s*)?(or|/)\s*$", _re.I)


def skill_groups(line: str) -> list:
    """Skills in a line as requirement groups. 'Tableau or Power BI' becomes ONE group [tableau, power bi]."""
    spans = extract_skill_spans(line)
    groups = []
    for i, (s, e, skill) in enumerate(spans):
        if i and ALT_GAP.match(line[spans[i - 1][1]:s]):
            groups[-1].append(skill)
        else:
            groups.append([skill])
    return groups


def analyze_jd(text: str) -> dict:
    """Extract title, required/preferred skill groups and years of experience from a job description."""
    text = clean_text(text)[:MAX_JD_CHARS]
    if len(text) < 30:
        raise ValueError("The job description is too short to analyse.")
    required, preferred, mode = [], [], "required"
    for line in text.split("\n"):
        if not line.strip():
            continue
        is_head = line.strip().endswith(":") or len(line.strip()) < 40
        if is_head and PREFERRED_HEAD.search(line):
            mode = "preferred"
        elif is_head and REQUIRED_HEAD.search(line):
            mode = "required"
        target = preferred if (mode == "preferred" or INLINE_PREFERRED.search(line)) else required
        for g in skill_groups(line):
            if g not in target:
                target.append(g)
    req_flat = {s for g in required for s in g}
    preferred = [g for g in preferred if not (set(g) & req_flat)]
    years = re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)", text, re.I)
    return {"title": text.split("\n")[0].strip()[:90], "required_groups": required, "preferred_groups": preferred,
            "required_skills": sorted(req_flat), "preferred_skills": sorted({s for g in preferred for s in g}),
            "years_required": int(max(years, key=int)) if years else None, "text": text}


def _split_groups(groups, have):
    """-> (matched labels, missing labels). A group is satisfied if the candidate has any skill in it."""
    matched, missing = [], []
    for g in groups:
        hit = [s for s in g if s in have]
        (matched if hit else missing).append(", ".join(hit) if hit else " or ".join(g))
    return matched, missing


def match_resume_to_jd(resume: dict, jd_text: str) -> dict:
    jd = analyze_jd(jd_text)
    have = set(resume["skills"])
    matched_req, missing_req = _split_groups(jd["required_groups"], have)
    matched_pref, missing_pref = _split_groups(jd["preferred_groups"], have)
    n_req, n_pref = len(jd["required_groups"]), len(jd["preferred_groups"])
    total = n_req + config.PREFERRED_WEIGHT * n_pref
    got = len(matched_req) + config.PREFERRED_WEIGHT * len(matched_pref)
    coverage = got / total if total else 0.0
    vec = load_model()["vectorizer"]
    sim = float(cosine_similarity(vec.transform([resume_text_for_matching(resume)]), vec.transform([jd["text"]]))[0, 0])
    overall = 0.7 * coverage + 0.3 * min(sim / 0.5, 1.0)      # cosine of ~0.5 or more already counts as a full text match
    verdict = "Strong match" if overall >= 0.75 else "Moderate match" if overall >= 0.5 else "Low match"
    exp_note = None
    if jd["years_required"] is not None:
        gap = jd["years_required"] - resume["experience_years"]
        exp_note = (f"The job asks for {jd['years_required']}+ years; your resume shows about {resume['experience_years']}."
                    if gap > 0 else f"Your experience ({resume['experience_years']} years) meets the {jd['years_required']}+ years mentioned.")
    mentioned = set(jd["required_skills"]) | set(jd["preferred_skills"])
    return {"jd": jd, "overall": overall, "verdict": verdict, "text_similarity": sim, "skill_coverage": coverage,
            "matched_required": matched_req, "missing_required": missing_req, "matched_preferred": matched_pref, "missing_preferred": missing_pref,
            "extra_skills": sorted(have - mentioned), "experience_note": exp_note}


def skill_gap_report(m: dict) -> str:
    """Readable (Markdown) skill-gap report."""
    L = [f"# Skill-Gap Report: {m['jd']['title']}", "",
         f"**Overall match: {m['overall']:.0%} ({m['verdict']})**", "",
         f"- Skill coverage (required + preferred): {m['skill_coverage']:.0%}",
         f"- Text similarity (TF-IDF cosine): {m['text_similarity']:.2f}"]
    if m["experience_note"]:
        L.append(f"- Experience: {m['experience_note']}")
    L += ["", f"## Required skills you have ({len(m['matched_required'])}/{len(m['jd']['required_groups'])})",
          ", ".join(m["matched_required"]) or "None", "",
          f"## Required skills missing - top priority ({len(m['missing_required'])})",
          ", ".join(m["missing_required"]) or "None - you meet every required skill.", ""]
    if m["jd"]["preferred_groups"]:
        L += [f"## Preferred skills you have ({len(m['matched_preferred'])}/{len(m['jd']['preferred_skills'])})", ", ".join(m["matched_preferred"]) or "None", "",
              "## Preferred skills missing - nice to have", ", ".join(m["missing_preferred"]) or "None", ""]
    L += ["## Other skills on your resume not mentioned in the job", ", ".join(m["extra_skills"]) or "None", "", "## Suggested next steps"]
    if m["missing_required"]:
        for label in m["missing_required"]:
            cat = next(iter(group_by_category([label.split(" or ")[0]])))
            if cat == "Soft Skills":
                L.append(f"- **{label}** ({cat}): show it with a concrete example (a team project, presentation or leadership role) and describe it in a resume bullet.")
            else:
                L.append(f"- Learn **{label}** ({cat}): start with official documentation, then build a small project that uses it and add it to your resume.")
    else:
        L.append("- You meet all required skills: tailor your summary and project descriptions to the job's wording.")
    if m["missing_preferred"]:
        L.append(f"- After the required skills, consider: {', '.join(m['missing_preferred'][:4])}.")
    L.append("- Only list a skill on your resume once you can discuss it in an interview.")
    return "\n".join(L)
