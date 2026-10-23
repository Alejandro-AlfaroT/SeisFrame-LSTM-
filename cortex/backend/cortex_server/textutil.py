"""Text helpers shared by search, similarity, conflict detection and import."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself nor of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when where which while who whom
    why will with would you your yours yourself yourselves also may might must shall via per each
    """.split()
)

NEGATIONS = frozenset(
    """
    not no never none cannot can't without neither nor isn't aren't wasn't weren't doesn't don't
    didn't won't shouldn't mustn't hasn't haven't hadn't couldn't wouldn't
    """.split()
)

WIKILINK_RE = re.compile(r"!?\[\[([^\]\|#]*)(#[^\]\|]*)?(?:\|([^\]]*))?\]\]")
MDLINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str) -> str:
    """Drop control characters (other than tab/newline/CR) that have no place in notes."""
    return CONTROL_RE.sub("", text)


def plain_text(markdown: str) -> str:
    """Markdown to readable plain text: link targets dropped, aliases kept, markup removed."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = WIKILINK_RE.sub(lambda m: m.group(3) or m.group(1).rsplit("/", 1)[-1], text)
    text = MDLINK_RE.sub(r"\1", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_`>]+", "", text)
    text = re.sub(r"^\s*[-+]\s+\[[ xX]\]\s*", "", text, flags=re.M)
    return text


def excerpt(markdown: str, length: int = 240) -> str:
    """Opening prose of a note; headings are skipped (the first usually repeats the title)."""
    body = re.sub(r"^\s{0,3}#{1,6}\s.*$", " ", markdown, flags=re.M)
    text = re.sub(r"\s+", " ", plain_text(body)).strip()
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


WORD_RE = re.compile(r"[a-z][a-z0-9_]{2,}")


def index_tokens(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS]
