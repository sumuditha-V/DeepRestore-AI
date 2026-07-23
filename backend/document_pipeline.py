"""
document_pipeline.py
====================
The NLP half of DeepRestore AI. After the Computer-Vision side has *denoised* a
scanned document image, this module turns that image into clean, structured text:

    denoised image
        -> OCR            (Tesseract, via pytesseract)      -> raw text
        -> spell-correct  (pyspellchecker)                  -> corrected text
        -> summarise      (Hugging Face Transformers)       -> summary
        -> keywords       (frequency-based)                 -> keywords

Every heavy dependency (Tesseract binary, transformers model) is optional and
**lazy-loaded**, so the API starts instantly and degrades gracefully: if a
component is unavailable, its field is returned empty with an explanatory note
instead of crashing the request.
"""

import os
import re
from collections import Counter

# --- Lazy singletons (loaded on first use, not at import) ------------------- #
_summarizer = None
_spell = None

# A small stop-word list keeps keyword extraction dependency-free.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "them", "his", "her", "their", "we", "you", "your", "our", "i", "not", "no",
    "so", "than", "then", "there", "here", "which", "who", "whom", "will",
    "would", "can", "could", "should", "may", "might", "do", "does", "did",
    "have", "has", "had", "about", "into", "over", "after", "before", "up",
    "down", "out", "all", "any", "each", "more", "most", "some", "such",
}


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def extract_text(image_bgr):
    """Run OCR on a denoised BGR image. Returns (text, error_or_None)."""
    try:
        import cv2
        import pytesseract
    except ImportError as e:
        return "", f"OCR unavailable (missing dependency: {e.name})."

    # Allow pointing at the Tesseract binary via env var on Windows, e.g.
    #   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
    tess_cmd = os.environ.get("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Otsu threshold improves OCR on document-like images.
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary)
        return text.strip(), None
    except pytesseract.TesseractNotFoundError:
        return "", ("Tesseract binary not found. Install it and/or set the "
                    "TESSERACT_CMD environment variable.")
    except Exception as e:
        return "", f"OCR failed: {e}"


# --------------------------------------------------------------------------- #
# Spell correction
# --------------------------------------------------------------------------- #
def _get_spellchecker():
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker
        _spell = SpellChecker(distance=1)  # distance 1 = fast, OCR-friendly
    return _spell


def correct_spelling(text):
    """Fix likely OCR spelling errors word-by-word. Returns (corrected, note)."""
    if not text.strip():
        return text, None
    try:
        spell = _get_spellchecker()
    except ImportError:
        return text, "Spell-check unavailable (pyspellchecker not installed)."

    def fix(match):
        word = match.group(0)
        # Only touch plain alphabetic words; leave numbers/IDs/casing alone.
        if len(word) < 3 or not word.isalpha() or not word.islower():
            return word
        if word in spell:
            return word
        suggestion = spell.correction(word)
        return suggestion if suggestion else word

    corrected = re.sub(r"[A-Za-z]+", fix, text)
    return corrected, None


# --------------------------------------------------------------------------- #
# Summarisation
# --------------------------------------------------------------------------- #
def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        # distilbart is a good speed/quality trade-off (~300 MB, downloaded once).
        _summarizer = pipeline("summarization",
                               model="sshleifer/distilbart-cnn-12-6")
    return _summarizer


def summarize(text, min_words=40):
    """Summarise text if it is long enough. Returns (summary, note)."""
    word_count = len(text.split())
    if word_count < min_words:
        return "", f"Text too short to summarise ({word_count} words)."
    try:
        summarizer = _get_summarizer()
    except ImportError:
        return "", "Summarisation unavailable (transformers not installed)."
    except Exception as e:
        return "", f"Could not load summariser: {e}"

    try:
        # BART handles ~1024 tokens; trim very long OCR dumps.
        snippet = " ".join(text.split()[:800])
        out = summarizer(snippet, max_length=130, min_length=30,
                         do_sample=False)
        return out[0]["summary_text"].strip(), None
    except Exception as e:
        return "", f"Summarisation failed: {e}"


# --------------------------------------------------------------------------- #
# Keywords
# --------------------------------------------------------------------------- #
def extract_keywords(text, top_n=8):
    """Return the most frequent meaningful words (simple, dependency-free)."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    words = [w for w in words if w not in _STOPWORDS]
    if not words:
        return []
    return [w for w, _ in Counter(words).most_common(top_n)]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def restore_document_text(image_bgr):
    """Full OCR + NLP pipeline over a denoised document image.

    Returns a dict ready to serialise as JSON. Individual stages fail softly and
    report their status in `notes` rather than raising.
    """
    notes = []

    raw_text, ocr_err = extract_text(image_bgr)
    if ocr_err:
        notes.append(ocr_err)

    corrected_text, corr_note = correct_spelling(raw_text)
    if corr_note:
        notes.append(corr_note)

    summary, sum_note = summarize(corrected_text)
    if sum_note:
        notes.append(sum_note)

    keywords = extract_keywords(corrected_text)

    return {
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "summary": summary,
        "keywords": keywords,
        "stats": {
            "word_count": len(corrected_text.split()),
            "char_count": len(corrected_text),
        },
        "notes": notes,
    }
