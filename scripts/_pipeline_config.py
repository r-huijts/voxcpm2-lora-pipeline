"""Shared voice.json config loader for the narration pipeline scripts.

Each of 01_chunk.py / 02_generate_nanovllm.py / 03_stitch.py can load a
per-project voice.json to supply defaults for flags that are stable across
runs (--lora, --reference, --cfg, --gap-scale, etc.) so they don't need to
be retyped on every invocation. Command-line flags always override the
config file, and the config file always overrides a script's own
hardcoded default. Per-run values (input/output paths, --plan, --out-dir,
--run-dir) and secrets (--api-key) are intentionally never read from here
-- see each script's own CONFIGURABLE set.
"""

import json
import sys
from pathlib import Path


def default_voice_config_path(script_file: str) -> Path:
    """
    Where to look for voice.json when --config isn't passed: the current
    working directory first (per-project, most specific -- lets you run
    several voices from different cwds with one shared script tree), then
    next to the calling script itself (lets you keep one voice.json in
    scripts/ and run commands from anywhere else). Mirrors 01_chunk.py's
    existing .env lookup (repo root, then scripts/). Returns whichever
    exists; if neither does, returns the cwd-relative path so "not found"
    messages still point somewhere predictable.
    """
    cwd_path = Path("voice.json")
    if cwd_path.exists():
        return cwd_path
    script_dir_path = Path(script_file).resolve().parent / "voice.json"
    if script_dir_path.exists():
        return script_dir_path
    return cwd_path


def load_voice_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_config_defaults(parser, config: dict, allowed: set) -> dict:
    filtered = {k: v for k, v in config.items() if k in allowed}
    ignored = set(config) - allowed
    if ignored:
        print(f"[voice.json] ignoring keys not used by this script: {sorted(ignored)}",
              file=sys.stderr)
    if filtered:
        parser.set_defaults(**filtered)
        print(f"[voice.json] applied defaults: {sorted(filtered)}", file=sys.stderr)
    return filtered
