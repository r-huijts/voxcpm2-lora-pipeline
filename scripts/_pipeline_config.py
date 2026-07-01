"""Shared voice.json config loader for the narration pipeline scripts.

Each of 01_chunk.py / 02_generate_nanovllm.py / 03_stitch.py can load a
per-project voice.json (default: ./voice.json in the current working
directory) to supply defaults for flags that are stable across runs
(--lora, --reference, --cfg, --gap-scale, etc.) so they don't need to be
retyped on every invocation. Command-line flags always override the
config file, and the config file always overrides a script's own
hardcoded default. Per-run values (input/output paths, --plan, --out-dir,
--run-dir) and secrets (--api-key) are intentionally never read from here
-- see each script's own CONFIGURABLE set.
"""

import json
import sys
from pathlib import Path


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
