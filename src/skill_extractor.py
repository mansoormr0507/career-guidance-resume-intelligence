"""Dictionary-based skill identification with alias handling.

Every alias is matched as a whole word/phrase (case-insensitive). When two matches overlap,
the longer one wins, so "react native" is not also counted as "react".
"""
import json
import re
from functools import lru_cache

from src import config


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    return json.loads(config.TAXONOMY_JSON.read_text())


@lru_cache(maxsize=1)
def _patterns():
    """List of (compiled_regex, canonical_skill, category), longest aliases first."""
    items = []
    for category, skills in load_taxonomy().items():
        for skill, aliases in skills.items():
            for name in {skill, *aliases}:
                # (?<![\w+#.]) / (?![\w+#]) act as word boundaries that also respect c++, c#, node.js
                items.append((re.compile(r"(?<![\w+#.\-])" + re.escape(name) + r"(?![\w+#\-])(?!\.\w)", re.I), skill, category, len(name)))
    items.sort(key=lambda x: -x[3])
    return items


def skill_category(skill: str) -> str:
    for category, skills in load_taxonomy().items():
        if skill in skills:
            return category
    return "Other"


def extract_skill_spans(text: str) -> list:
    """Return [(start, end, canonical_skill)] for all non-overlapping matches, in text order."""
    taken, spans = [], []
    for regex, skill, _, _ in _patterns():
        for m in regex.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            spans.append((s, e, skill))
    return sorted(spans)


def extract_skills(text: str) -> list:
    """Return the sorted list of canonical skill names found in the text."""
    return sorted({skill for _, _, skill in extract_skill_spans(text)})


def group_by_category(skills) -> dict:
    grouped = {}
    for s in skills:
        grouped.setdefault(skill_category(s), []).append(s)
    return {k: sorted(v) for k, v in sorted(grouped.items())}
