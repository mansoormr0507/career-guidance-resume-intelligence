"""Content-based career recommendation engine (TF-IDF + cosine similarity + skill coverage).

For every job role we compute three things and combine them with the weights in config.py:
  1. text similarity   - cosine similarity between TF-IDF vectors of the resume and the role description
  2. skill coverage    - share of the role's required (weight 1) and preferred (weight 0.5) skills the candidate has
  3. fit               - education level / field and experience level compared with the role
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.text_processing import clean_text, remove_stopwords, tokenize


def analyzer_tokens(text: str) -> list:
    """Module-level function (so the saved model can be loaded anywhere): text -> uni/bi-gram tokens."""
    toks = remove_stopwords(tokenize(clean_text(text)))
    return toks + [" ".join(p) for p in zip(toks, toks[1:])]


def load_roles() -> pd.DataFrame:
    df = pd.read_csv(config.JOB_ROLES_CSV)
    for col in ("required_skills", "preferred_skills", "relevant_fields"):
        df[col] = df[col].apply(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])
    return df


def job_document(row) -> str:
    """Text representation of a role; required skills are repeated so they count more."""
    return " ".join([row["role"], row["description"]] + row["required_skills"] * 2 + row["preferred_skills"])


def build_model(verbose: bool = True) -> dict:
    roles = load_roles()
    docs = [job_document(r) for _, r in roles.iterrows()]
    vec = TfidfVectorizer(analyzer=analyzer_tokens, sublinear_tf=True)
    matrix = vec.fit_transform(docs)
    model = {"vectorizer": vec, "roles": roles, "job_matrix": matrix}
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    if verbose:
        print(f"Recommendation model built: {len(roles)} job roles, {len(vec.vocabulary_)} TF-IDF features")
        print(f"Saved to {config.MODEL_PATH.relative_to(config.ROOT)}")
    return model


def load_model() -> dict:
    if not config.MODEL_PATH.exists():
        return build_model(verbose=False)
    return joblib.load(config.MODEL_PATH)


def resume_text_for_matching(resume: dict) -> str:
    """Body text plus canonical skill names (so aliases such as 'sklearn' map to 'scikit-learn')."""
    return resume["body_text"] + " " + " ".join(resume["skills"])


def _fit_score(resume: dict, role) -> float:
    rank = resume["education"]["level_rank"]
    level = 1.0 if rank >= 3 else 0.7 if rank == 2 else 0.3 if rank == 1 else 0.5
    edu_text = (" ".join(resume["education"]["details"]) + " " + " ".join(resume["education"]["fields"])).lower()
    field = 1.0 if any(f.lower() in edu_text for f in role["relevant_fields"]) else 0.4
    needs_exp = "mid" in role["experience_level"]
    exp = 1.0 if (not needs_exp or resume["experience_years"] >= 1) else 0.6
    return 0.6 * (0.6 * level + 0.4 * field) + 0.4 * exp


def recommend(resume: dict, model: dict = None, top_k: int = 5, weights=None) -> list:
    """Rank all roles for a parsed resume. Returns a list of dicts sorted by score (best first)."""
    model = model or load_model()
    w_text, w_skill, w_fit = weights or (config.W_TEXT, config.W_SKILLS, config.W_FIT)
    roles = model["roles"]
    vec = model["vectorizer"].transform([resume_text_for_matching(resume)])
    cos = cosine_similarity(vec, model["job_matrix"]).ravel()
    cos_norm = cos / cos.max() if cos.max() > 0 else cos
    have = set(resume["skills"])
    results = []
    for i, (_, role) in enumerate(roles.iterrows()):
        req, pref = role["required_skills"], role["preferred_skills"]
        total = len(req) + config.PREFERRED_WEIGHT * len(pref)
        got = sum(s in have for s in req) + config.PREFERRED_WEIGHT * sum(s in have for s in pref)
        coverage = got / total if total else 0.0
        fit = _fit_score(resume, role)
        score = w_text * cos_norm[i] + w_skill * coverage + w_fit * fit
        results.append({
            "role": role["role"], "score": float(score), "text_similarity": float(cos[i]), "skill_coverage": float(coverage), "fit": float(fit),
            "matched_required": [s for s in req if s in have], "missing_required": [s for s in req if s not in have],
            "matched_preferred": [s for s in pref if s in have], "missing_preferred": [s for s in pref if s not in have],
            "description": role["description"],
        })
    results.sort(key=lambda r: -r["score"])
    return results[:top_k] if top_k else results


def format_recommendations(results: list) -> str:
    lines = ["", "Top career recommendations", "=" * 60]
    for i, r in enumerate(results, 1):
        lines += [f"{i}. {r['role']}  -  match {r['score']:.0%}",
                  f"   text similarity {r['text_similarity']:.2f} | skill coverage {r['skill_coverage']:.0%} | profile fit {r['fit']:.0%}",
                  f"   You have : {', '.join(r['matched_required'] + r['matched_preferred']) or '-'}",
                  f"   To learn : {', '.join(r['missing_required'][:5]) or 'nothing required - you meet all core skills'}", ""]
    return "\n".join(lines)
