#!/usr/bin/env python3
"""
test_style_profile.py — regression test for 01_chunk.py's configurable style
profile (Task 4: ship a generic persona, never a real name, in any committed
file; allow local override via env var or voice.json).

Needs pysbd + a portkey_ai stub importable (01_chunk.py imports both at
module level) but makes no network calls -- pure prompt-construction checks.

Run after touching 01_chunk.py's style-profile code:
    python scripts/test_style_profile.py
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_chunk_module():
    spec = importlib.util.spec_from_file_location("chunk01_style_test", HERE / "01_chunk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_no_real_name_in_template():
    mod = _load_chunk_module()
    assert "Mart Smeets" not in mod.SYSTEM_PROMPT_TEMPLATE
    assert "Smeets" not in mod.SYSTEM_PROMPT_TEMPLATE


def test_fresh_environment_produces_generic_profile():
    os.environ.pop("NARRATE_STYLE_PROFILE", None)
    mod = _load_chunk_module()
    assert mod.default_style_profile() == mod.GENERIC_STYLE_PROFILE
    prompt = mod.build_system_prompt(mod.default_style_profile())
    assert mod.GENERIC_STYLE_PROFILE in prompt
    assert "Mart Smeets" not in prompt


def test_env_var_overrides_generic_default():
    mod = _load_chunk_module()
    os.environ["NARRATE_STYLE_PROFILE"] = "a warm late-night radio host"
    try:
        assert mod.default_style_profile() == "a warm late-night radio host"
        prompt = mod.build_system_prompt(mod.default_style_profile())
        assert "a warm late-night radio host" in prompt
    finally:
        del os.environ["NARRATE_STYLE_PROFILE"]


def test_explicit_style_profile_flows_into_prompt():
    mod = _load_chunk_module()
    prompt = mod.build_system_prompt("a specific local project's own descriptor")
    assert "a specific local project's own descriptor" in prompt
    assert mod.GENERIC_STYLE_PROFILE not in prompt


def main():
    tests = [
        test_no_real_name_in_template,
        test_fresh_environment_produces_generic_profile,
        test_env_var_overrides_generic_default,
        test_explicit_style_profile_flows_into_prompt,
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
