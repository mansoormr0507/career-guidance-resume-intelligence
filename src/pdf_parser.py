"""PDF validation and text extraction.

PyMuPDF (imported as `fitz`) is used when installed; `pypdf` is the automatic fallback.
"""
from pathlib import Path

from src import config


class ResumeFileError(Exception):
    """Raised with a user-friendly message when a file cannot be used."""


def validate_pdf(path) -> Path:
    """Check existence, extension, size and the PDF 'magic bytes' before reading."""
    path = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if not path.exists():
        raise ResumeFileError(f"File not found: {path}")
    if not path.is_file():
        raise ResumeFileError(f"Not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ResumeFileError("Only PDF files are supported (the file name must end with .pdf).")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.MAX_PDF_MB:
        raise ResumeFileError(f"File is too large ({size_mb:.1f} MB). The limit is {config.MAX_PDF_MB} MB.")
    if path.stat().st_size == 0:
        raise ResumeFileError("The file is empty.")
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise ResumeFileError("This file is not a valid PDF (wrong file header).")
    return path


def _extract_pymupdf(path: Path) -> list:
    import fitz  # PyMuPDF
    with fitz.open(path) as doc:
        if doc.needs_pass:
            raise ResumeFileError("The PDF is password protected. Please remove the password first.")
        if len(doc) > config.MAX_PDF_PAGES:
            raise ResumeFileError(f"The PDF has {len(doc)} pages; the limit is {config.MAX_PDF_PAGES}.")
        return [page.get_text("text") for page in doc]


def _extract_pypdf(path: Path) -> list:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ResumeFileError("The PDF is password protected. Please remove the password first.")
    if len(reader.pages) > config.MAX_PDF_PAGES:
        raise ResumeFileError(f"The PDF has {len(reader.pages)} pages; the limit is {config.MAX_PDF_PAGES}.")
    return [page.extract_text() or "" for page in reader.pages]


def extract_text(path) -> str:
    """Validate the file and return its text (pages joined by newlines)."""
    path = validate_pdf(path)
    try:
        try:
            pages = _extract_pymupdf(path)
        except ImportError:
            pages = _extract_pypdf(path)
    except ResumeFileError:
        raise
    except Exception as exc:  # corrupted file, unsupported encoding, ...
        raise ResumeFileError(f"Could not read the PDF ({type(exc).__name__}). The file may be corrupted.") from exc
    text = "\n".join(pages).strip()
    if len(text) < config.MIN_TEXT_CHARS:
        raise ResumeFileError("Very little text could be extracted. The PDF may be a scanned image; "
                              "please provide a text-based PDF (OCR is not supported).")
    return text
