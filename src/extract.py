"""
Stage 1 / step B: final-answer extraction + Part 1 category labelling.

The category labels are the *conditioning variable* of the whole paper, so
they are attached at write time, inside the group record, never later.

>>> SWAP-IN POINT <<<
`categorise()` below is a faithful re-implementation of the Part 1 surface-form
taxonomy (verifier-error-budget). If your Part 1 repo exposes the canonical
classifier, import it here instead and delete this function. Do NOT let the two
implementations drift -- the join between Part 1 and Part 2 depends on identical
labels.

Categories are NOT mutually exclusive: an answer can be
{trailing_period, boxed_whitespace, latex_frac}. We store a sorted list plus a
single `primary_cat` (highest-precedence label) for bar charts and routing.
"""

import re

# precedence order == Part 1 error-budget ordering (whitespace/punctuation first)
PRECEDENCE = [
    "trailing_newline",
    "trailing_period",
    "trailing_whitespace",
    "internal_whitespace",
    "boxed_whitespace",
    "latex_frac",
    "latex_sqrt",
    "latex_text_unit",
    "degree_percent",
    "interval_or_tuple",
    "set_or_list",
    "mixed_number",
    "scientific_notation",
    "large_numeric",
    "small_numeric",
    "plain_integer",
    "symbolic_other",
]

_BOXED = re.compile(r"\\boxed\s*\{")
_NUM = re.compile(r"^[-+]?\d[\d,]*(\.\d+)?$")


def extract_boxed(text: str):
    """Last \\boxed{...} with brace matching. Returns raw inner string or None."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None


def extract_final_answer(completion: str):
    """
    Returns (raw_answer, extraction_mode).

    RAW is deliberately un-normalised: trailing periods, newlines and stray
    spaces ARE the object of study. Never strip here.
    """
    b = extract_boxed(completion)
    if b is not None:
        return b, "boxed"

    m = re.search(
        r"(?:final answer|answer)\s*(?:is)?\s*[:=]?\s*(.+)$",
        completion.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).split("\n")[0], "answer_phrase"

    lines = [l for l in completion.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1], "last_line"
    return "", "none"


def categorise(raw: str, completion: str = ""):
    """Return (sorted_category_list, primary_cat) for a RAW answer string."""
    cats = set()
    if raw is None:
        return [], "none"

    if raw.endswith("\n"):
        cats.add("trailing_newline")
    if raw.rstrip().endswith("."):
        cats.add("trailing_period")
    if raw != raw.strip():
        cats.add("trailing_whitespace")
    if re.search(r"\S\s{2,}\S", raw):
        cats.add("internal_whitespace")

    if completion:
        m = _BOXED.search(completion)
        if m and re.match(r"\\boxed\s+\{", completion[m.start() : m.end()]):
            cats.add("boxed_whitespace")
    s = raw.strip()
    if s != s.strip(" \t"):
        cats.add("boxed_whitespace")

    if r"\frac" in s or r"\dfrac" in s or r"\tfrac" in s:
        cats.add("latex_frac")
    if r"\sqrt" in s:
        cats.add("latex_sqrt")
    if r"\text" in s or r"\mbox" in s:
        cats.add("latex_text_unit")
    if "^\\circ" in s or r"\%" in s or s.endswith("%"):
        cats.add("degree_percent")
    if re.match(r"^[\(\[].+,.+[\)\]]$", s):
        cats.add("interval_or_tuple")
    if s.startswith("\\{") or (s.startswith("{") and "," in s):
        cats.add("set_or_list")
    if re.match(r"^\d+\s+\\?[dt]?frac", s):
        cats.add("mixed_number")
    if re.search(r"\d\s*(\\times\s*10\^|e[-+]?\d)", s):
        cats.add("scientific_notation")

    core = s.rstrip(".").replace(",", "").strip()
    if _NUM.match(core):
        try:
            v = abs(float(core))
            if v >= 1e4:
                cats.add("large_numeric")
            else:
                cats.add("small_numeric")
            if float(core).is_integer():
                cats.add("plain_integer")
        except ValueError:
            pass
    elif not cats:
        cats.add("symbolic_other")

    if not cats:
        cats.add("symbolic_other")

    primary = next((c for c in PRECEDENCE if c in cats), "symbolic_other")
    return sorted(cats), primary
