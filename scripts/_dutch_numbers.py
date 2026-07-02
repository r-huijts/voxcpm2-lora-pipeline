"""normalize_dutch() — deterministic Dutch number/abbreviation expansion for TTS.

Conservative by design: unknown or ambiguous numeric tokens (route/stage
codes, model numbers, phone numbers, odd alphanumerics, thousands-grouped
numbers like "2.500") are left UNCHANGED rather than guessed. A bad
deterministic normalization is worse than none. Proper nouns are never
touched here -- that's the pronunciation lexicon's job (see 01_chunk.py).

Covers: cardinals, comma-decimals, ordinals (1e/3e/...), times (14:30),
years (1900-2099, using the Dutch century-grouped reading for the 1900s and
the "tweeduizend X" reading for the 2000s), percentages, km/u speeds, km
distances, and a small set of fixed abbreviations (UCI, ASO, nr., ca., bv.).
"""
import re
import sys

try:
    from num2words import num2words
except ImportError:  # pragma: no cover - exercised only when dep is missing
    num2words = None

_warned_missing_dep = False


def _num_nl(n: int) -> str:
    return num2words(n, lang="nl")


_ORDINAL_IRREGULAR_NL = {1: "eerste", 3: "derde", 8: "achtste"}


def _ordinal_tail_0_99(n: int) -> str:
    """Ordinal form for 0-99, also used as the tail of larger numbers
    (e.g. 118 -> "honderd" + ordinal_tail(18))."""
    if n in _ORDINAL_IRREGULAR_NL:
        return _ORDINAL_IRREGULAR_NL[n]
    base = _num_nl(n)
    if n == 0:
        return base
    if n >= 20:
        return base + "ste"  # 20-99 all end in a "-tig" word -> -ste
    return base + "de"       # 2,4-7,9-19 -> -de (1,3,8 handled above)


def _ordinal_nl(n: int) -> str:
    if n < 100:
        return _ordinal_tail_0_99(n)
    remainder = n % 100
    if remainder == 0:
        return _num_nl(n) + "ste"  # honderdste, tweehonderdste, duizendste
    root = _num_nl(n - remainder)
    return root + _ordinal_tail_0_99(remainder)


def _decimal_or_int_nl(s: str) -> str:
    """s is a plain int string or a comma-decimal string, e.g. "8" or "8,5"."""
    if "," in s:
        whole, frac = s.split(",", 1)
        return f"{_num_nl(int(whole))} komma " + " ".join(_num_nl(int(d)) for d in frac)
    return _num_nl(int(s))


def _year_nl(n: int) -> str:
    """Dutch spoken-year convention, which differs from a straight cardinal
    reading: 1900-1999 groups as two two-digit halves ("negentienhonderd
    zevenennegentig" for 1997, not "duizend negenhonderd zevenennegentig"),
    2000-2099 reads as "tweeduizend <rest>"."""
    if 2000 <= n <= 2099:
        rest = n - 2000
        base = _num_nl(2000)
        return base if rest == 0 else f"{base} {_num_nl(rest)}"
    hi, lo = divmod(n, 100)
    base = f"{_num_nl(hi)}honderd"
    return base if lo == 0 else f"{base} {_num_nl(lo)}"


# Order matters: consume the most specific numeric shapes first so the
# generic cardinal pass at the end never re-touches already-converted text.
_PROTECTED_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")  # e.g. "2.500" -- ambiguous
_SPEED_RE = re.compile(r"\b(\d+(?:,\d+)?)\s?km/u\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b(\d+(?:,\d+)?)\s?%")
# Trailing "uur" is consumed too (Dutch often writes "14:30 uur"), so the
# expansion -- which already ends in "uur" -- doesn't double up.
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?:\s?uur\b)?", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"\b(\d+)e\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_KM_DISTANCE_RE = re.compile(r"\b(\d+(?:,\d+)?)\s?km\b", re.IGNORECASE)
_KM_WORD_RE = re.compile(r"\bkm\b", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\b(\d+),(\d+)\b")
_CARDINAL_RE = re.compile(r"(?<![A-Za-zÀ-ÿ])\d+(?![A-Za-zÀ-ÿ])")
# Any whitespace-delimited token still containing BOTH a letter and a digit
# at this point is a code, not a number (route/stage codes, model numbers
# like "T-3000", bib numbers like "Bib124") -- protect it wholesale rather
# than partially convert just the digit run inside it.
_TOKEN_RE = re.compile(r"\S+")

_ABBREVIATIONS = [
    (re.compile(r"\bUCI\b"), "U-C-I"),
    (re.compile(r"\bASO\b"), "A-S-O"),
    (re.compile(r"\bnr\."), lambda m: "Nummer"),
    (re.compile(r"\bNr\."), lambda m: "Nummer"),
    (re.compile(r"\bca\."), lambda m: "circa"),
    (re.compile(r"\bCa\."), lambda m: "Circa"),
    (re.compile(r"\bbv\."), lambda m: "bijvoorbeeld"),
    (re.compile(r"\bBv\."), lambda m: "Bijvoorbeeld"),
]


def normalize_dutch(text: str) -> str:
    """Deterministically expand numbers/abbreviations for TTS pronunciation.
    Returns text unchanged (with a one-time warning) if num2words isn't
    installed, rather than crashing."""
    global _warned_missing_dep
    if num2words is None:
        if not _warned_missing_dep:
            print("WARNING: num2words not installed -- normalize_dutch() is a "
                  "no-op (pip install num2words). Numbers will not be spelled out.",
                  file=sys.stderr)
            _warned_missing_dep = True
        return text

    # Protect ambiguous thousands-grouped numbers (e.g. "2.500") from every
    # later pass -- distinguishing a thousands separator from a decimal or
    # sentence-final period is genuinely ambiguous, so leave them as-is.
    protected = {}

    def _protect(m):
        # A single Private Use Area character -- contains no digits or
        # letters, so it's invisible to every digit-/letter-based regex below.
        key = chr(0xE000 + len(protected))
        protected[key] = m.group(0)
        return key

    out = _PROTECTED_THOUSANDS_RE.sub(_protect, text)

    out = _SPEED_RE.sub(lambda m: f"{_decimal_or_int_nl(m.group(1))} kilometer per uur", out)
    out = _PERCENT_RE.sub(lambda m: f"{_decimal_or_int_nl(m.group(1))} procent", out)
    out = _TIME_RE.sub(
        lambda m: f"{_num_nl(int(m.group(1)))} uur"
                  + (f" {_num_nl(int(m.group(2)))}" if int(m.group(2)) != 0 else ""),
        out,
    )
    out = _ORDINAL_RE.sub(lambda m: _ordinal_nl(int(m.group(1))), out)
    out = _YEAR_RE.sub(lambda m: _year_nl(int(m.group(0))), out)
    out = _KM_DISTANCE_RE.sub(lambda m: f"{_decimal_or_int_nl(m.group(1))} kilometer", out)
    out = _KM_WORD_RE.sub("kilometer", out)
    out = _DECIMAL_RE.sub(
        lambda m: f"{_num_nl(int(m.group(1)))} komma "
                  + " ".join(_num_nl(int(d)) for d in m.group(2)),
        out,
    )

    def _protect_mixed_alnum(m):
        tok = m.group(0)
        if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
            return _protect(m)
        return tok

    out = _TOKEN_RE.sub(_protect_mixed_alnum, out)
    out = _CARDINAL_RE.sub(lambda m: _num_nl(int(m.group(0))), out)

    for pattern, repl in _ABBREVIATIONS:
        out = pattern.sub(repl, out)

    for key, original in protected.items():
        out = out.replace(key, original)

    # A fragment that now starts with a spelled-out number (or otherwise
    # lowercase) should still read as a proper sentence-initial capital.
    if out and out[0].islower():
        out = out[0].upper() + out[1:]

    return out
