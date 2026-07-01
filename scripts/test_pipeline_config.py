#!/usr/bin/env python3
"""
test_pipeline_config.py — regression test for the voice.json config mechanism
shared by 01_chunk.py / 02_generate_nanovllm.py / 03_stitch.py.

Standalone: only depends on argparse + _pipeline_config, not on portkey_ai /
pysbd / nano-vllm-voxcpm / numpy / soundfile, so it runs without any of the
heavy pipeline dependencies installed.

Run after touching any of the three scripts' argparse blocks:
    python scripts/test_pipeline_config.py

This exists because a config-loading bug is easy to introduce silently: the
--interactive flag in 02_generate_nanovllm.py referenced a bare `interactive`
name instead of `args.interactive` for a long time, and nothing caught it
until someone happened to grep for it.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _pipeline_config import apply_config_defaults


def _build_parser():
    """Mirrors the required=True + CONFIGURABLE shape used by the real
    scripts (--lora/--reference are required, like in 02_generate_nanovllm.py)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    lora_action = ap.add_argument("--lora", required=True, type=Path)
    reference_action = ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cfg", type=float, default=1.6)
    return ap, lora_action, reference_action


CONFIGURABLE = {"lora", "reference", "cfg"}


def _parse(argv, config):
    ap, lora_action, reference_action = _build_parser()
    applied = apply_config_defaults(ap, config, CONFIGURABLE)
    if "lora" in applied:
        lora_action.required = False
    if "reference" in applied:
        reference_action.required = False
    return ap.parse_args(argv)


def test_missing_required_without_config():
    try:
        _parse(["--plan", "p.json", "--out-dir", "out"], {})
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for missing --lora/--reference")


def test_config_supplies_required_args():
    config = {"lora": "/from/config/lora", "reference": "/from/config/ref.wav", "cfg": 1.6}
    args = _parse(["--plan", "p.json", "--out-dir", "out"], config)
    assert str(args.lora) == "/from/config/lora", args.lora
    assert str(args.reference) == "/from/config/ref.wav", args.reference
    assert args.cfg == 1.6, args.cfg


def test_cli_overrides_config():
    config = {"lora": "/from/config/lora", "reference": "/from/config/ref.wav", "cfg": 1.6}
    args = _parse(
        ["--plan", "p.json", "--out-dir", "out", "--lora", "/cli/lora", "--cfg", "2.0"],
        config,
    )
    assert str(args.lora) == "/cli/lora", args.lora
    assert args.cfg == 2.0, args.cfg
    assert str(args.reference) == "/from/config/ref.wav", args.reference  # untouched


def test_boolean_optional_action_override():
    """--controllable/--loudnorm use BooleanOptionalAction so a voice.json
    `true` can still be forced off for one run via --no-controllable, unlike
    plain store_true flags which have no CLI token to turn a sticky default
    back off."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllable", action=argparse.BooleanOptionalAction, default=False)
    apply_config_defaults(ap, {"controllable": True}, {"controllable"})

    assert ap.parse_args([]).controllable is True  # config default applies
    assert ap.parse_args(["--no-controllable"]).controllable is False  # CLI forces off
    assert ap.parse_args(["--controllable"]).controllable is True  # CLI forces on


def test_unrelated_keys_ignored():
    config = {
        "lora": "/from/config/lora",
        "reference": "/from/config/ref.wav",
        "out_dir": "/should/not/apply",
        "unexpected_key": 123,
    }
    args = _parse(["--plan", "p.json", "--out-dir", "out"], config)
    assert str(args.out_dir) == "out", args.out_dir  # per-run value, never from config


def test_generate_and_stitch_routing():
    """generate_and_stitch.py must route flags to the right subprocess call
    (stitch-only flags never leak into the generate call and vice versa),
    compute --run-dir/--output defaults from --out-dir, forward --config to
    both calls, and abort before stitching if generation fails."""
    calls = []

    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        class Result:
            returncode = 0
        return Result()

    real_run = subprocess.run
    real_argv = sys.argv
    try:
        subprocess.run = fake_run
        sys.argv = [
            "generate_and_stitch.py",
            "--plan", "plan.json", "--out-dir", "run01", "--config", "custom.json",
            "--lora", "/x/lora", "--cfg", "1.8",
            "--loudnorm", "--lufs", "-14", "--gap-scale", "1.2",
        ]
        spec = importlib.util.spec_from_file_location(
            "generate_and_stitch", HERE / "generate_and_stitch.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

        assert len(calls) == 2, f"expected 2 subprocess calls, got {len(calls)}"
        gen, stitch = calls
        assert "--lora" in gen and "/x/lora" in gen
        assert "--cfg" in gen and "1.8" in gen
        assert "--loudnorm" not in gen, "stitch-only flag leaked into generate call"
        assert "--run-dir" in stitch and "run01" in stitch
        assert "--output" in stitch
        assert str(Path("run01/final.wav")) in stitch, "default --output not derived from --out-dir"
        assert "--config" in stitch and "custom.json" in stitch
        assert "--loudnorm" in stitch
        assert float(stitch[stitch.index("--lufs") + 1]) == -14.0
        assert float(stitch[stitch.index("--gap-scale") + 1]) == 1.2

        # Generation failure must abort before stitching.
        calls.clear()
        def failing_run(cmd, *a, **kw):
            calls.append(cmd)
            class Result:
                returncode = 1
            return Result()
        subprocess.run = failing_run
        try:
            mod.main()
            raise AssertionError("expected SystemExit on generation failure")
        except SystemExit:
            pass
        assert len(calls) == 1, "stitch must not run after a failed generate"
    finally:
        subprocess.run = real_run
        sys.argv = real_argv


def main():
    tests = [
        test_missing_required_without_config,
        test_config_supplies_required_args,
        test_cli_overrides_config,
        test_boolean_optional_action_override,
        test_unrelated_keys_ignored,
        test_generate_and_stitch_routing,
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
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
