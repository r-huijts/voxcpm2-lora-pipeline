#!/usr/bin/env python3
"""
test_lexicon_prefilter.py — regression test for lexicon_prefilter.py.

Two groups:
1. Pure logic tests against an INJECTED fake phonemizer (no espeak-ng
   needed) -- tokenization, edit distance, ratio/flag thresholds, and the
   graceful-degradation path.
2. A real-binary integration test using actual espeak-ng, skipped (not
   failed) if it isn't installed -- asserts ROBUST relative properties
   (X ranks better than Y, Z is flagged suspect) rather than exact IPA
   strings, so it doesn't flake across espeak-ng versions.

Run after touching lexicon_prefilter.py:
    python scripts/test_lexicon_prefilter.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexicon_prefilter import _levenshtein, _tokenize_phonemes, phonemize_nl, screen_candidates

# A small fake Dutch phoneme "dictionary" for injected tests -- hand-built,
# not from a real espeak-ng run (that's the second test group below).
_FAKE_IPA = {
    "klassementsman": "klˈɑsəmˌɛntsmɑn",   # 13 phoneme symbols after stripping stress
    "klassements-man": "klˌɑsəmˈɛntsmˈɑn",  # same symbols, different stress only
    "klassemensman": "klˈɑsəmˌɛnsmɑn",      # missing the 't' -> 12 symbols
    "man": "mˈɑn",                          # 3 symbols -- wildly shorter
    "unphonemizable": None,                 # simulates a per-candidate failure
}


def _fake_phonemize(text: str):
    return _FAKE_IPA.get(text)


def test_tokenize_strips_stress_and_length_marks_and_whitespace():
    assert _tokenize_phonemes("klˈɑsəmˌɛntsmɑn") == list("klɑsəmɛntsmɑn")
    assert _tokenize_phonemes("a b") == ["a", "b"]
    assert _tokenize_phonemes("aːb") == ["a", "b"]


def test_levenshtein_basic():
    assert _levenshtein(list("abc"), list("abc")) == 0
    assert _levenshtein(list("abc"), list("ab")) == 1
    assert _levenshtein([], list("abc")) == 3
    assert _levenshtein(list("abc"), []) == 3


def test_cluster_preserving_candidate_ranks_best_with_zero_distance():
    results = screen_candidates("klassementsman", ["klassements-man", "klassemensman", "man"],
                                 phonemize_fn=_fake_phonemize)
    assert results[0]["candidate"] == "klassements-man"
    assert results[0]["phoneme_edit_distance"] == 0
    assert results[0]["flag"] == "ok"


def test_dropped_consonant_candidate_ranks_worse_but_not_suspect():
    results = screen_candidates("klassementsman", ["klassements-man", "klassemensman"],
                                 phonemize_fn=_fake_phonemize)
    by_candidate = {r["candidate"]: r for r in results}
    dropped = by_candidate["klassemensman"]
    assert dropped["phoneme_edit_distance"] == 1
    assert dropped["flag"] == "ok"  # only 1/13 missing -- not "wildly" different


def test_wildly_shorter_candidate_flagged_suspect():
    results = screen_candidates("klassementsman", ["man"], phonemize_fn=_fake_phonemize)
    assert results[0]["flag"] == "suspect"
    assert results[0]["phoneme_count_ratio"] < 0.5


def test_ranking_order_matches_severity():
    results = screen_candidates(
        "klassementsman", ["man", "klassemensman", "klassements-man"],
        phonemize_fn=_fake_phonemize,
    )
    order = [r["candidate"] for r in results]
    assert order == ["klassements-man", "klassemensman", "man"]


def test_original_unavailable_flags_all_and_preserves_order():
    results = screen_candidates("klassementsman", ["man", "klassements-man"],
                                 phonemize_fn=lambda t: None)
    assert all(r["flag"] == "espeak_unavailable" for r in results)
    assert [r["candidate"] for r in results] == ["man", "klassements-man"]
    for r in results:
        assert r["ipa"] is None
        assert r["phoneme_edit_distance"] is None


def test_single_candidate_unavailable_flagged_individually():
    results = screen_candidates("klassementsman", ["klassements-man", "unphonemizable"],
                                 phonemize_fn=_fake_phonemize)
    by_candidate = {r["candidate"]: r for r in results}
    assert by_candidate["klassements-man"]["flag"] == "ok"
    assert by_candidate["unphonemizable"]["flag"] == "espeak_unavailable"


def test_phonemize_nl_never_crashes_with_no_espeak_on_path():
    import os
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = "/nonexistent"
    try:
        result = phonemize_nl("klassementsman")
        assert result is None
    finally:
        os.environ["PATH"] = old_path


def test_real_espeak_ng_ranks_confirmed_lexicon_entry_best():
    """Integration test against the ACTUAL espeak-ng binary (skipped, not
    failed, if unavailable). Asserts robust relative properties rather than
    exact IPA strings, so it survives espeak-ng version differences."""
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        print("SKIP test_real_espeak_ng_ranks_confirmed_lexicon_entry_best "
              "(espeak-ng not installed)")
        return
    results = screen_candidates(
        "klassementsman",
        ["klassements-man", "klassemensman", "man"],
    )
    assert results[0]["flag"] != "espeak_unavailable", "espeak-ng found but produced no IPA"
    by_candidate = {r["candidate"]: r for r in results}
    # klassements-man is the CONFIRMED (ear-tested) lexicon.json entry --
    # must not be flagged suspect and must rank at least as well as the
    # known dropped-consonant candidate.
    assert by_candidate["klassements-man"]["flag"] == "ok"
    assert (by_candidate["klassements-man"]["phoneme_edit_distance"]
            <= by_candidate["klassemensman"]["phoneme_edit_distance"])
    # "man" alone is wildly shorter than the original -- must be suspect.
    assert by_candidate["man"]["flag"] == "suspect"


def main():
    tests = [
        test_tokenize_strips_stress_and_length_marks_and_whitespace,
        test_levenshtein_basic,
        test_cluster_preserving_candidate_ranks_best_with_zero_distance,
        test_dropped_consonant_candidate_ranks_worse_but_not_suspect,
        test_wildly_shorter_candidate_flagged_suspect,
        test_ranking_order_matches_severity,
        test_original_unavailable_flags_all_and_preserves_order,
        test_single_candidate_unavailable_flagged_individually,
        test_phonemize_nl_never_crashes_with_no_espeak_on_path,
        test_real_espeak_ng_ranks_confirmed_lexicon_entry_best,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    if failures:
        sys.exit(f"\n{failures}/{len(tests)} test(s) failed.")
    print(f"\nAll {len(tests)} tests passed (or skipped where noted).")


if __name__ == "__main__":
    main()
