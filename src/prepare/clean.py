"""Light text cleaning. We keep the words as they are, because the models read them."""
import html
import re

from src import config

WHITESPACE = re.compile(r"\s+")


def clean_text(text):
    """Unescape HTML entities and collapse all runs of whitespace into single spaces."""
    if not isinstance(text, str):
        return ""
    return WHITESPACE.sub(" ", html.unescape(text)).strip()


def normalized_key(text):
    """The form used to find exact duplicates: lower case, single spaces."""
    return clean_text(text).lower()


def clean_frame(frame):
    """Clean the text column and drop complaints without enough text. Returns (frame, dropped)."""
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean_text)
    keep = frame["text"].str.len() >= config.MIN_TEXT_CHARS
    return frame[keep], int((~keep).sum())
