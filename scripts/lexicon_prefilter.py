#!/usr/bin/env python3
"""
lexicon_prefilter.py — offline phoneme-sanity screening for candidate
pronunciation respellings, BEFORE any of them cost a generation cycle.

SCOPE — read this before trusting the output.

This checks whether a candidate respelling produces a SANE PHONEME SEQUENCE,
via espeak-ng -- which has its own internal model of Dutch phonology, a
DIFFERENT model than VoxCPM2's LoRA-tuned weights. The two do not share a
brain.

What this catches well: consonant-cluster collapse and gross garbage -- the
"klassementsman" -> "klassemensman" class of bug, where a respelling produces
a phoneme sequence that's obviously wrong or unstable regardless of which TTS
model reads it. This is the majority of current failures.

What this does NOT catch: stress/emphasis placement. espeak's stress model is
not VoxCPM's stress behavior. A candidate can pass this filter cleanly and
still come out with wrong emphasis in VoxCPM. Stress-focused respellings
(e.g. an accent-mark approach for "klassemENTSman") still go straight to the
ear test -- this filter deliberately does NOT score stress. That's separate,
not-yet-built work.

This is a coarse net for a cheap, common failure mode, not a general
pronunciation oracle. It does not replace the ear test, and it does not
auto-write to lexicon.json -- a human confirms every entry by ear on real
generated audio (see 02_generate_nanovllm.py --only-chunks) before it's kept.

Usage (diagnostic, run by hand -- not wired into the generation pipeline):
    python scripts/lexicon_prefilter.py --word klassementsman \\
        --candidates "klassements-man" "klassement man" "klassemensman"

Requires (optional): espeak-ng (system binary) with Dutch voice support, or
the `phonemizer` package as a wrapper around it. Neither is a hard
dependency -- if unavailable, every candidate is flagged "espeak_unavailable"
and returned unreordered. The rest of the pipeline is unaffected either way.
"""
import argparse
import re
import shutil
import subprocess
import sys

try:
    from phonemizer import phonemize as _phonemizer_phonemize
    _HAS_PHONEMIZER = True
except ImportError:
    _HAS_PHONEMIZER = False

# Stress marks (primary/secondary) and the length mark. Stripped before
# tokenizing -- this tool explicitly does not judge stress (see module
# docstring), and length is a coarse-enough net that we fold it in too.
_STRESS_RE = re.compile(r"[ˈˌ]")
_LENGTH_RE = re.compile(r"ː")


def phonemize_nl(text: str) -> str | None:
    """
    Phonemize Dutch text to IPA via espeak-ng. Tries the `phonemizer`
    package first, falls back to a direct `espeak-ng --ipa` subprocess call.
    Returns None if neither is available or the call fails for any reason --
    callers must degrade gracefully, this must never crash the caller.
    """
    if _HAS_PHONEMIZER:
        try:
            result = _phonemizer_phonemize(
                text, language="nl", backend="espeak", strip=True, with_stress=True,
            )
            result = (result or "").strip()
            if result:
                return result
        except Exception:
            pass  # fall through to a direct subprocess attempt

    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak_bin:
        return None
    try:
        proc = subprocess.run(
            [espeak_bin, "--ipa", "-q", "-v", "nl", text],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        result = proc.stdout.strip()
        return result or None
    except Exception:
        return None


def _tokenize_phonemes(ipa: str) -> list[str]:
    """
    Coarse phoneme tokenization: strip stress/length marks (out of scope for
    this tool -- see module docstring) and whitespace, then treat each
    remaining character as one symbol. Not linguistically rigorous -- it
    only needs to separate "plausible" from "obviously broken" per this
    tool's stated scope.
    """
    cleaned = _STRESS_RE.sub("", ipa)
    cleaned = _LENGTH_RE.sub("", cleaned)
    cleaned = cleaned.replace(" ", "")
    return list(cleaned)


def _levenshtein(a: list, b: list) -> int:
    """Edit distance over two symbol sequences (any hashable tokens)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def screen_candidates(original: str, candidates: list[str],
                       phonemize_fn=phonemize_nl) -> list[dict]:
    """
    For a target word and a list of candidate respellings, phonemize the
    original once and each candidate, then score each candidate's phoneme
    sequence against the original.

    phonemize_fn is injectable for testing without a real espeak-ng install;
    real callers should leave it at the default.

    Returns a list of dicts:
        {candidate, ipa, phoneme_edit_distance, phoneme_count_ratio, flag}
    flag is one of "ok" | "suspect" | "espeak_unavailable".
    "suspect" means the candidate's phoneme count differs from the
    original's by more than 50% -- catches an added vowel / a collapsed
    cluster directly. Sorted best-first by edit distance, EXCEPT when
    espeak is unavailable entirely, in which case the input order is
    preserved (no reordering, no false confidence).
    """
    original_ipa = phonemize_fn(original)

    if original_ipa is None:
        return [
            {
                "candidate": c,
                "ipa": None,
                "phoneme_edit_distance": None,
                "phoneme_count_ratio": None,
                "flag": "espeak_unavailable",
            }
            for c in candidates
        ]

    original_tokens = _tokenize_phonemes(original_ipa)
    results = []
    for c in candidates:
        c_ipa = phonemize_fn(c)
        if c_ipa is None:
            results.append({
                "candidate": c,
                "ipa": None,
                "phoneme_edit_distance": None,
                "phoneme_count_ratio": None,
                "flag": "espeak_unavailable",
            })
            continue

        c_tokens = _tokenize_phonemes(c_ipa)
        distance = _levenshtein(original_tokens, c_tokens)
        ratio = (len(c_tokens) / len(original_tokens)) if original_tokens else 1.0
        flag = "suspect" if (ratio > 1.5 or ratio < 0.5) else "ok"

        results.append({
            "candidate": c,
            "ipa": c_ipa,
            "phoneme_edit_distance": distance,
            "phoneme_count_ratio": round(ratio, 3),
            "flag": flag,
        })

    # Preserve input order only when nothing could be phonemized at all;
    # otherwise sort successes best-first, unavailable ones last.
    if all(r["flag"] == "espeak_unavailable" for r in results):
        return results
    return sorted(
        results,
        key=lambda r: (r["phoneme_edit_distance"] is None, r["phoneme_edit_distance"] or 0),
    )


def _print_table(word: str, results: list[dict]) -> None:
    print(f"Original: {word!r}")
    print()
    header = f"{'candidate':<24} {'IPA':<24} {'dist':>5} {'ratio':>6}  flag"
    print(header)
    print("-" * len(header))
    for r in results:
        ipa_str = r["ipa"] if r["ipa"] is not None else "(unavailable)"
        dist_str = str(r["phoneme_edit_distance"]) if r["phoneme_edit_distance"] is not None else "n/a"
        ratio_str = f"{r['phoneme_count_ratio']:.2f}" if r["phoneme_count_ratio"] is not None else "n/a"
        print(f"{r['candidate']:<24} {ipa_str:<24} {dist_str:>5} {ratio_str:>6}  {r['flag']}")
    print()
    if results and results[0]["flag"] == "espeak_unavailable":
        print("espeak-ng / phonemizer not available -- install to get real screening "
              "(see module docstring). No candidates were reordered.", file=sys.stderr)
    else:
        print("This is a coarse phoneme-sanity net, not a pronunciation oracle -- it does "
              "NOT judge stress/emphasis. Confirm the top candidate(s) by ear via "
              "02_generate_nanovllm.py --only-chunks before adding to lexicon.json.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word", required=True, help="The original word as it appears in the text.")
    ap.add_argument("--candidates", required=True, nargs="+",
                    help="One or more candidate respellings to screen, space-separated "
                         "(quote multi-word candidates).")
    args = ap.parse_args()

    results = screen_candidates(args.word, args.candidates)
    _print_table(args.word, results)


if __name__ == "__main__":
    main()
