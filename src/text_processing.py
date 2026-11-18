"""NLP preprocessing: cleaning, tokenisation, stop-word removal and keyword extraction.

No external NLP data downloads are needed: scikit-learn provides the English stop-word list.
"""
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# keeps tokens such as c++, c#, node.js, scikit-learn, ci/cd intact
TOKEN_RE = re.compile(r"[a-z][a-z0-9+#]*(?:[./\-][a-z0-9+#]+)*")
STOP = set(ENGLISH_STOP_WORDS) | {"experience", "work", "working", "project", "projects", "using", "used", "skills", "years", "year", "also"}


def clean_text(text: str) -> str:
    """Normalise whitespace and bullet characters; remove non-printable characters."""
    text = text.replace("\u2022", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    text = re.sub(r"[^\x09\x0a\x20-\x7e]", " ", text)       # drop anything non-ASCII/non-printable
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list:
    """Lower-case word tokens (technical tokens like c++ and node.js are kept whole)."""
    return TOKEN_RE.findall(text.lower())


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP and len(t) > 1]


def preprocess(text: str) -> str:
    """Clean -> tokenise -> remove stop words -> join. Used before TF-IDF."""
    return " ".join(remove_stopwords(tokenize(clean_text(text))))


def extract_keywords(text: str, top_n: int = 15) -> list:
    """Most frequent meaningful words and two-word phrases (simple frequency-based keyword extraction)."""
    toks = remove_stopwords(tokenize(text))
    counts = Counter(toks)
    counts.update(" ".join(p) for p in zip(toks, toks[1:]))
    return [w for w, c in counts.most_common(top_n * 3) if c > 1 or len(counts) < top_n][:top_n]
