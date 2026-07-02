#!/usr/bin/env python3
"""
test_dutch_numbers.py — regression test for normalize_dutch() (_dutch_numbers.py).

Standalone: only depends on num2words. Run after touching _dutch_numbers.py:
    python scripts/test_dutch_numbers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dutch_numbers import normalize_dutch

CASES = [
    ("214 renners kwamen aan.", "Tweehonderdveertien renners kwamen aan."),
    ("Hij reed 420,3 kilometer.", "Hij reed vierhonderdtwintig komma drie kilometer."),
    ("Dit is de 1e etappe, hij werd 3e.", "Dit is de eerste etappe, hij werd derde."),
    ("De start is om 14:30 uur.", "De start is om veertien uur dertig."),
    ("De start is om 14:30.", "De start is om veertien uur dertig."),
    ("In 2026 wordt het spannend.", "In tweeduizend zesentwintig wordt het spannend."),
    ("Terug naar 1997, een historisch jaar.",
     "Terug naar negentienhonderd zevenennegentig, een historisch jaar."),
    ("Hij had 8% voorsprong.", "Hij had acht procent voorsprong."),
    ("Met 45 km/u ging hij de bocht in.",
     "Met vijfenveertig kilometer per uur ging hij de bocht in."),
    ("De afstand was 12 km lang.", "De afstand was twaalf kilometer lang."),
    ("Gemeten in km is dat een record.", "Gemeten in kilometer is dat een record."),
    ("De UCI en de ASO grepen in.", "De U-C-I en de A-S-O grepen in."),
    ("Zie nr. 4, ca. 10 renners, bv. deze groep.",
     "Zie Nummer vier, circa tien renners, bijvoorbeeld deze groep."),
    # Alphanumeric codes must survive UNCHANGED -- a bad guess is worse than none.
    ("Renner N7 en model T-3000 staan genoteerd.",
     "Renner N7 en model T-3000 staan genoteerd."),
    ("Bib124 finishte als 8e.", "Bib124 finishte als achtste."),
    # Thousands-grouped (dot) numbers are ambiguous -- left untouched.
    ("Er stonden 2.500 man langs de kant.", "Er stonden 2.500 man langs de kant."),
    ("45 renners bereikten de finish.", "Vijfenveertig renners bereikten de finish."),
    ("De 18e editie was de honderdste.", "De achttiende editie was de honderdste."),
    ("Dit was de 21e keer.", "Dit was de eenentwintigste keer."),
]


def main():
    failures = 0
    for text, expected in CASES:
        got = normalize_dutch(text)
        if got == expected:
            print(f"PASS {text!r}")
        else:
            failures += 1
            print(f"FAIL {text!r}\n     expected: {expected!r}\n     got:      {got!r}")
    if failures:
        sys.exit(f"\n{failures}/{len(CASES)} case(s) failed.")
    print(f"\nAll {len(CASES)} cases passed.")


if __name__ == "__main__":
    main()
