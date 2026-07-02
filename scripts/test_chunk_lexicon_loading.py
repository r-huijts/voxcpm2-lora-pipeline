#!/usr/bin/env python3
"""
test_chunk_lexicon_loading.py — regression test for a real production bug:
a `voice.json`-supplied "lexicon" path that doesn't exist (e.g. a relative
path that's only valid from a different cwd) used to hard-crash the whole
chunking run via load_lexicon()'s sys.exit. It must now be treated like any
other default -- skipped with a warning -- and only an EXPLICIT --lexicon
CLI flag pointing at a missing file should still be a hard error.

Needs pysbd + a portkey_ai stub importable (01_chunk.py imports both at
module level) but makes no network calls -- call_llm is monkeypatched.

Run after touching 01_chunk.py's lexicon-loading logic:
    python scripts/test_chunk_lexicon_loading.py
"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAKE_LLM_PLAN = json.dumps({
    "register": "test register",
    "chunks": [
        {"id": 1, "sentences": ["P1S1"], "position": "opening",
         "control": "dry, measured", "gap_after_ms": 300},
        {"id": 2, "sentences": ["P1S2"], "position": "final",
         "control": "slow, dry, settled", "gap_after_ms": 0},
    ],
})


def _load_chunk_module():
    spec = importlib.util.spec_from_file_location("chunk01_lexicon_test", HERE / "01_chunk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.call_llm = lambda client, model, sentences_block, style_profile: FAKE_LLM_PLAN
    return mod


def _run_main(mod, argv, tmpdir):
    column_path = Path(tmpdir) / "column.txt"
    column_path.write_text("Eerste zin hier. Tweede zin hier.", encoding="utf-8")
    output_path = Path(tmpdir) / "plan.json"

    real_argv = sys.argv
    stdout, stderr = io.StringIO(), io.StringIO()
    sys.argv = ["01_chunk.py", "--input", str(column_path), "--output", str(output_path),
                "--api-key", "fake-key-not-used"] + argv
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            mod.main()
    finally:
        sys.argv = real_argv
    return output_path, stdout.getvalue(), stderr.getvalue()


def test_missing_lexicon_from_voice_json_warns_and_continues():
    # Reproduces the real bug: a relative voice.json lexicon path that's
    # only valid from a different cwd than the one the script is actually
    # run from (e.g. "scripts/lexicon.json" when already inside scripts/).
    # Use a path that's unambiguously missing regardless of cwd.
    mod = _load_chunk_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        voice_json = Path(tmpdir) / "voice.json"
        voice_json.write_text(
            json.dumps({"lexicon": "scripts/definitely_missing_lexicon.json"}),
            encoding="utf-8",
        )
        output_path, out, err = _run_main(
            mod, ["--config", str(voice_json)], tmpdir,
        )
        assert output_path.exists(), "run must complete and write the plan despite the missing lexicon"
        assert "NOTE: lexicon not found" in err
        assert "(voice.json)" in err


def test_missing_lexicon_explicit_cli_flag_is_a_hard_error():
    mod = _load_chunk_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "does_not_exist.json"
        try:
            _run_main(mod, ["--lexicon", str(missing)], tmpdir)
            raise AssertionError("expected SystemExit for an explicitly-passed missing --lexicon")
        except SystemExit:
            pass


def test_no_config_at_all_still_works():
    # No --config, no voice.json -- falls back to the hardcoded default next
    # to the script (this repo's real scripts/lexicon.json, which exists),
    # exercising the plain no-config path end to end without crashing.
    mod = _load_chunk_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path, out, err = _run_main(mod, [], tmpdir)
        assert output_path.exists()


def main():
    tests = [
        test_missing_lexicon_from_voice_json_warns_and_continues,
        test_missing_lexicon_explicit_cli_flag_is_a_hard_error,
        test_no_config_at_all_still_works,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"FAIL {test.__name__}: unexpected {type(e).__name__}: {e}")
    if failures:
        sys.exit(f"\n{failures}/{len(tests)} test(s) failed.")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
