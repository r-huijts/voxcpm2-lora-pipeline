from pathlib import Path
import re
import subprocess
import torch
import soundfile as sf
from voxcpm import VoxCPM
import json
from voxcpm.model.voxcpm2 import LoRAConfig

torch.set_float32_matmul_precision("high")

PROJECT = Path("/workspace/voxcpm_project")
TEXTS = PROJECT / "texts"
REFS = PROJECT / "references"
OUTPUTS = PROJECT / "outputs"

TEXTS.mkdir(parents=True, exist_ok=True)
REFS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Generation modes
# ---------------------------------------------------------------------------
MODES = {
    "1": {
        "name": "Controllable Cloning",
        "description": "Reference audio sets timbre. Control tag steers pace/style. Recommended.",
        "needs_transcript": False,
        "supports_control": True,
    },
    "2": {
        "name": "Hi-Fi Cloning",
        "description": "Reference audio + transcript for tightest timbre match. Control tag is ignored.",
        "needs_transcript": True,
        "supports_control": False,
    },
    "3": {
        "name": "Voice Design",
        "description": "No reference audio. Control tag invents a voice from scratch.",
        "needs_transcript": False,
        "supports_control": True,
        "no_reference": True,
    },
    "4": {
        "name": "Chained Continuation (Hi-Fi)",
        "description": "Each chunk continues the previous chunk's audio for cross-seam glue. "
                       "Control tag ignored. Re-anchors to reference periodically to limit drift.",
        "needs_transcript": True,
        "supports_control": False,
        "chained": True,
    },
}


def split_text(text: str, max_chars: int) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"\n{2,}", "\n\n", text)

    def split_long_sentence(sentence: str) -> list[str]:
        parts = re.split(r"(?<=[,;:])\s+", sentence)
        chunks = []
        current = ""

        for part in parts:
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) <= max_chars:
                    current = part
                else:
                    words = part.split()
                    word_chunk = ""
                    for word in words:
                        word_candidate = f"{word_chunk} {word}".strip() if word_chunk else word
                        if len(word_candidate) <= max_chars:
                            word_chunk = word_candidate
                        else:
                            if word_chunk:
                                chunks.append(word_chunk)
                            word_chunk = word
                    current = word_chunk

        if current:
            chunks.append(current)
        return chunks

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        current = ""

        for sentence in sentences:
            if len(sentence) <= max_chars:
                candidate = f"{current} {sentence}".strip() if current else sentence
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
            else:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(split_long_sentence(sentence))

        if current:
            chunks.append(current)

    return chunks


def prepend_control(chunk: str, control: str) -> str:
    if not control.strip():
        return chunk
    return f"({control.strip()}){chunk}"


def convert_reference_to_wav(input_audio: Path, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_audio),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_wav)],
        check=True,
    )
    return output_wav


def combine_wavs(wav_paths: list[Path], output_path: Path) -> None:
    concat_file = output_path.parent / "concat_list.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for wav_path in wav_paths:
            f.write(f"file '{wav_path}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-ar", "48000", "-ac", "1", str(output_path)],
        check=True,
    )


def get_wav_duration(wav_path: Path) -> float:
    """Return duration of a WAV file in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_audio_tail(src_wav: Path, dst_wav: Path, tail_seconds: float) -> Path:
    """
    Extract the last `tail_seconds` of src_wav, converted to 16 kHz mono,
    for use as a continuation prompt. If the source is shorter than the
    requested tail, the whole file is used.
    """
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    duration = get_wav_duration(src_wav)
    start = max(0.0, duration - tail_seconds)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src_wav),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst_wav)],
        check=True,
    )
    return dst_wav


def text_tail(chunk_text: str, tail_seconds: float, chars_per_second: float = 15.0) -> str:
    """
    Approximate the transcript of the audio tail by taking trailing sentences
    of `chunk_text` whose combined length roughly matches `tail_seconds` of
    speech. The transcript must correspond to the audio tail, or the
    continuation prompt will produce edge artifacts.

    chars_per_second is a rough speaking-rate estimate (~15 chars/s for
    relaxed narration); tune if tails feel mismatched.
    """
    target_chars = int(tail_seconds * chars_per_second)
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    tail = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {tail}".strip() if tail else sentence
        if len(candidate) <= target_chars or not tail:
            tail = candidate
        else:
            break
    return tail


def ask(prompt: str, default: str | None = None) -> str:
    if default is None:
        value = input(f"{prompt}: ").strip()
    else:
        value = input(f"{prompt} [{default}]: ").strip()
    return value if value else (default or "")


def ask_float(prompt: str, default: float) -> float:
    return float(ask(prompt, str(default)))


def ask_int(prompt: str, default: int) -> int:
    return int(ask(prompt, str(default)))


def ask_bool(prompt: str, default: bool) -> bool:
    default_str = "yes" if default else "no"
    return ask(prompt, default_str).lower() in {"yes", "y", "true", "1"}


def list_files(folder: Path, suffixes: tuple[str, ...]) -> None:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in suffixes])
    if not files:
        print(f"  (no files found in {folder})")
        return
    for file in files:
        print(f"  - {file.name}")


def pick_mode(last_mode: str) -> str:
    print()
    print("Generation mode:")
    for key, mode in MODES.items():
        marker = " *" if key == last_mode else "  "
        print(f"{marker} {key} = {mode['name']}")
        print(f"       {mode['description']}")
    choice = ask("Mode", last_mode)
    if choice not in MODES:
        print(f"Unknown mode '{choice}', falling back to {last_mode}.")
        return last_mode
    return choice


def main():
    torch.set_float32_matmul_precision("high")

    print("VoxCPM2 interactive long-form synthesis")
    print()
    print(f"Texts folder:      {TEXTS}")
    print(f"References folder: {REFS}")
    print(f"Outputs folder:    {OUTPUTS}")
    print()

    print("Loading VoxCPM2. This can take a few minutes...")
    
    _lora_path = "/workspace/voxcpm2-lora-pipeline/checkpoints/lora/step_0000999"
    _cfg = json.loads(open(f"{_lora_path}/lora_config.json").read())
    _lora_config = LoRAConfig(**_cfg.get("lora_config", _cfg))

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        lora_config=_lora_config,
        lora_weights_path=_lora_path,
        load_denoiser=False,
    )
    print("Model loaded.")
    print()

    last_mode = "1"
    last_text_file = "text.txt"
    last_ref_audio = "ref_voice.mp3"
    last_ref_transcript = "ref_voice.txt"
    last_control = "rustig tempo, ontspannen en helder, duidelijke pauzes"
    last_max_chars = 270
    last_cfg_value = 1.6
    last_timesteps = 20
    last_normalize = True
    last_tail_seconds = 8.0
    last_reanchor_every = 4

    while True:
        print()
        print("Choose:")
        print("  1 = generate audio")
        print("  2 = list files")
        print("  q = quit")
        choice = ask("Choice", "1").lower()

        if choice in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if choice == "2":
            print()
            print("Text files:")
            list_files(TEXTS, (".txt",))
            print()
            print("Reference audio files:")
            list_files(REFS, (".wav", ".mp3", ".flac", ".m4a"))
            print()
            print("Reference transcript files:")
            list_files(REFS, (".txt",))
            continue

        # --- Mode selection ---
        last_mode = pick_mode(last_mode)
        mode = MODES[last_mode]

        # --- File inputs ---
        text_file_name = ask("Long text file in /texts", last_text_file)

        ref_audio = None
        converted_ref = None
        if not mode.get("no_reference"):
            ref_audio_name = ask("Reference audio file in /references", last_ref_audio)
            last_ref_audio = ref_audio_name
            ref_audio = REFS / ref_audio_name

        prompt_text = None
        ref_transcript = None
        if mode["needs_transcript"]:
            ref_transcript_name = ask("Reference transcript file in /references", last_ref_transcript)
            last_ref_transcript = ref_transcript_name
            ref_transcript = REFS / ref_transcript_name

        control = ""
        if mode["supports_control"]:
            control = ask("Control instruction", last_control)
            last_control = control
        else:
            print("  (control instruction not used in Hi-Fi mode)")

        max_chars = ask_int("Max characters per chunk", last_max_chars)
        cfg_value = ask_float("cfg_value", last_cfg_value)
        timesteps = ask_int("inference_timesteps", last_timesteps)
        normalize = ask_bool("Normalize text (expand numbers/dates)?", last_normalize)

        tail_seconds = last_tail_seconds
        reanchor_every = last_reanchor_every
        if mode.get("chained"):
            tail_seconds = ask_float("Continuation tail length (seconds)", last_tail_seconds)
            reanchor_every = ask_int(
                "Re-anchor to reference every N chunks (0 = never)", last_reanchor_every
            )
            last_tail_seconds = tail_seconds
            last_reanchor_every = reanchor_every
        output_name = ask("Output run name", "test_run")

        # --- Validate files ---
        text_file = TEXTS / text_file_name
        output_dir = OUTPUTS / output_name
        chunks_dir = output_dir / "chunks"

        if not text_file.exists():
            print(f"Missing text file: {text_file}")
            continue
        if ref_audio and not ref_audio.exists():
            print(f"Missing reference audio: {ref_audio}")
            continue
        if mode["needs_transcript"] and not ref_transcript.exists():
            print(f"Missing transcript: {ref_transcript}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # --- Load inputs ---
        long_text = text_file.read_text(encoding="utf-8").strip()
        if mode["needs_transcript"]:
            prompt_text = ref_transcript.read_text(encoding="utf-8").strip()

        chunks = split_text(long_text, max_chars)

        # --- Summary ---
        print()
        print(f"Mode:            {mode['name']}")
        print(f"Text file:       {text_file}")
        if ref_audio:
            print(f"Reference audio: {ref_audio}")
        if mode["needs_transcript"]:
            print(f"Transcript:      {ref_transcript}")
        if mode["supports_control"] and control:
            print(f"Control tag:     ({control})")
        print(f"Output folder:   {output_dir}")
        print(f"Chunks:          {len(chunks)}")
        print(f"cfg_value:       {cfg_value}")
        print(f"timesteps:       {timesteps}")
        print(f"normalize:       {normalize}")
        if mode.get("chained"):
            print(f"tail seconds:    {tail_seconds}")
            print(f"re-anchor every: {reanchor_every if reanchor_every else 'never'}")
        print()
        print("Preview (first 10 chunks as sent to model):")
        for i, chunk in enumerate(chunks[:10], start=1):
            display = prepend_control(chunk, control) if mode["supports_control"] else chunk
            print(f"  {i:03d}: {display}")
        if len(chunks) > 10:
            print(f"  ... plus {len(chunks) - 10} more chunks")

        print()
        confirm = ask("Type yes to generate", "no").lower()
        if confirm != "yes":
            print("Cancelled.")
            continue

        # --- Convert reference ---
        if ref_audio:
            converted_ref = output_dir / "reference_16k.wav"
            print("Converting reference audio to 16 kHz mono WAV...")
            convert_reference_to_wav(ref_audio, converted_ref)

        # --- Generate ---
        wav_paths = []

        # Chained-mode state: prompt audio/text carried from the previous chunk.
        # None means "use the clean reference" (chunk 1, or a re-anchor point).
        tails_dir = output_dir / "tails"
        if mode.get("chained"):
            tails_dir.mkdir(parents=True, exist_ok=True)
        chain_prompt_wav: Path | None = None
        chain_prompt_text: str | None = None

        for i, chunk in enumerate(chunks, start=1):
            text_input = prepend_control(chunk, control) if mode["supports_control"] else chunk

            print()
            print(f"Generating chunk {i}/{len(chunks)}")
            print(f"  {text_input}")

            if last_mode == "1":
                # Controllable Cloning: reference sets timbre, control tag steers style
                wav = model.generate(
                    text=text_input,
                    reference_wav_path=str(converted_ref),
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                    normalize=normalize,
                )
            elif last_mode == "2":
                # Hi-Fi Cloning: tightest timbre match, control tag ignored
                wav = model.generate(
                    text=text_input,
                    prompt_wav_path=str(converted_ref),
                    prompt_text=prompt_text,
                    reference_wav_path=str(converted_ref),
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                    normalize=normalize,
                )
            elif last_mode == "4":
                # Chained Continuation: condition on the previous chunk's tail so
                # prosody carries across the seam. Re-anchor to the clean reference
                # on chunk 1 and every `reanchor_every` chunks to limit drift.
                is_reanchor = (
                    chain_prompt_wav is None
                    or (reanchor_every and (i - 1) % reanchor_every == 0)
                )

                if is_reanchor:
                    # Use the clean reference (its transcript is the ref transcript)
                    prompt_wav_for_chunk = str(converted_ref)
                    prompt_text_for_chunk = prompt_text
                    print("  [re-anchor to reference]")
                else:
                    prompt_wav_for_chunk = str(chain_prompt_wav)
                    prompt_text_for_chunk = chain_prompt_text
                    print(f"  [continuing from prev tail: \"{chain_prompt_text}\"]")

                wav = model.generate(
                    text=chunk,  # no control tag in Hi-Fi/continuation
                    prompt_wav_path=prompt_wav_for_chunk,
                    prompt_text=prompt_text_for_chunk,
                    reference_wav_path=str(converted_ref),
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                    normalize=normalize,
                )
            else:
                # Voice Design: no reference, control tag invents the voice
                wav = model.generate(
                    text=text_input,
                    cfg_value=cfg_value,
                    inference_timesteps=timesteps,
                    normalize=normalize,
                )

            chunk_path = chunks_dir / f"chunk_{i:03d}.wav"
            sf.write(chunk_path, wav, model.tts_model.sample_rate)
            wav_paths.append(chunk_path)
            print(f"  Saved: {chunk_path}")

            # Prepare continuation prompt for the next chunk (chained mode only)
            if mode.get("chained"):
                tail_wav = tails_dir / f"tail_{i:03d}.wav"
                extract_audio_tail(chunk_path, tail_wav, tail_seconds)
                chain_prompt_wav = tail_wav
                chain_prompt_text = text_tail(chunk, tail_seconds)

        final_path = output_dir / "combined.wav"
        print()
        print("Combining chunks...")
        combine_wavs(wav_paths, final_path)
        print()
        print(f"Done: {final_path}")

        last_text_file = text_file_name
        last_max_chars = max_chars
        last_cfg_value = cfg_value
        last_timesteps = timesteps
        last_normalize = normalize


if __name__ == "__main__":
    main()
