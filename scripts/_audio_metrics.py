"""
_audio_metrics.py — advisory waveform quality metrics for generated chunks.

Waveform analysis measures technical and prosodic quality — it says nothing
about intelligibility or pronunciation (the ASR/WER gate owns those) and it
cannot judge "sounds human". So everything here is ADVISORY: flags surface in
the console, wer_log.json/journal and the run report, but never trigger a
retry. Used as a hard gate these thresholds would over-reject; used as triage
they tell you which chunks (or candidate takes) to listen to first.

What it covers that the WER + duration gates don't:
  - clipping / suspiciously low level          (technical sanity)
  - leading/trailing silence and abrupt edges  (boundary quality; a take cut
    mid-decay can pass WER — Whisper still hears the word)
  - flat pitch / flat energy contour           (monotone delivery)
  - chunk-vs-neighbours loudness outliers      (stitch-time --loudnorm
    normalizes the WHOLE timeline, so a chunk generated 5 dB quieter than
    its neighbours stays 5 dB quieter relative to them)
  - candidate ranking                          (a pre-listening sort order
    for --candidates takes, not a verdict)

numpy-only (pyloudnorm is used for LUFS when installed, else LUFS is None),
so it runs both on the GPU pod and in the local test environment.
"""
import math
import statistics

import numpy as np

# Framing (shared by the energy and pitch tracks).
FRAME_MS = 40.0            # >= 2 periods of F0_MIN_HZ, so ACF can see a cycle
HOP_MS = 10.0
SILENCE_THRESH_DB = -40.0  # frame considered silent below this (matches
                           # 03_stitch.py's trim_silence threshold)
EDGE_MS = 60.0             # window for the start/end abruptness check

# Pitch search range — generous band for adult speech.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
VOICED_ACF_MIN = 0.5       # normalized ACF peak below this = unvoiced frame

# Advisory flag thresholds. Deliberately conservative: a flag should mean
# "listen to this one", not fire on every chunk.
CLIP_SAMPLE_LEVEL = 0.999
CLIP_RATIO_FLAG = 1e-4        # fraction of samples at/above CLIP_SAMPLE_LEVEL
LOW_LEVEL_DBFS = -38.0        # active-region RMS below this = suspiciously quiet
FLAT_PITCH_ST = 1.0           # f0 std (semitones) below this = monotone
FLAT_PITCH_MIN_VOICED = 0.3   # ...but only judged with enough voiced frames
FLAT_ENERGY_CV = 0.25         # natural speech energy CV is typically > 0.5
ABRUPT_END_RATIO = 0.9        # ending at >= 90% of median energy = likely cut
ABRUPT_START_RATIO = 1.4      # starts may legitimately be plosive — higher bar
LOUDNESS_OUTLIER_DB = 3.0     # deviation from neighbour median that gets flagged


def _dbfs(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-10))


def rms_dbfs(audio: np.ndarray) -> float:
    """Whole-array RMS in dBFS (silence floor at -200)."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return -200.0
    return round(_dbfs(math.sqrt(float(np.mean(audio ** 2)))), 2)


def _frame(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int, int]:
    """Slice audio into overlapping frames: (frames[m, L], hop, frame_len)."""
    frame_len = max(1, int(round(sr * FRAME_MS / 1000.0)))
    hop = max(1, int(round(sr * HOP_MS / 1000.0)))
    if audio.size < frame_len:
        return audio[None, :].copy(), hop, audio.size
    n_frames = 1 + (audio.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return audio[idx], hop, frame_len


def _pitch_track(frames: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame F0 via normalized autocorrelation (FFT-based, vectorized).
    Returns (f0_hz[m], voiced[m]). Coarse — no sub-lag interpolation — but at
    typical sample rates the lag quantization error is well under 0.1
    semitone, plenty for a "does the contour move" variance estimate.
    """
    m, L = frames.shape
    lag_min = max(1, int(sr / F0_MAX_HZ))
    lag_max = min(L - 1, int(sr / F0_MIN_HZ))
    if lag_max <= lag_min:
        return np.zeros(m), np.zeros(m, dtype=bool)
    frames = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * L - 1).bit_length()
    spec = np.fft.rfft(frames, n=nfft, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :L]
    lag0 = acf[:, 0].copy()
    lag0[lag0 <= 0.0] = 1.0
    acf = acf / lag0[:, None]
    band = acf[:, lag_min:lag_max + 1]
    best = band.argmax(axis=1)
    strength = band[np.arange(m), best]
    f0 = sr / (lag_min + best).astype(np.float64)
    voiced = strength >= VOICED_ACF_MIN
    return f0, voiced


def compute_metrics(audio: np.ndarray, sr: int) -> dict:
    """
    Measure one chunk's waveform. Returns a JSON-serializable dict; fields
    that need speech to measure (contour, edges) are None when the audio is
    empty or entirely below the silence threshold.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    metrics = {
        "duration_s": round(audio.size / sr, 3) if sr else 0.0,
        "peak_dbfs": None, "rms_dbfs": None, "active_rms_dbfs": None,
        "lufs": None, "clip_ratio": 0.0,
        "lead_silence_ms": None, "trail_silence_ms": None,
        "edge_start_ratio": None, "edge_end_ratio": None,
        "energy_cv": None,
        "f0_median_hz": None, "f0_std_semitones": None, "voiced_fraction": None,
    }
    if audio.size == 0:
        return metrics

    abs_a = np.abs(audio)
    metrics["peak_dbfs"] = round(_dbfs(abs_a.max()), 2)
    metrics["rms_dbfs"] = rms_dbfs(audio)
    metrics["clip_ratio"] = round(float(np.mean(abs_a >= CLIP_SAMPLE_LEVEL)), 6)
    metrics["lufs"] = _lufs(audio, sr)

    frames, hop, frame_len = _frame(audio, sr)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    active = frame_rms > 10.0 ** (SILENCE_THRESH_DB / 20.0)
    if not active.any():
        return metrics  # all-silent file: level fields are enough to flag it

    first_f = int(np.argmax(active))
    last_f = int(len(active) - 1 - np.argmax(active[::-1]))
    start_s = first_f * hop
    end_s = min(audio.size, last_f * hop + frame_len)
    metrics["lead_silence_ms"] = round(start_s / sr * 1000.0, 1)
    metrics["trail_silence_ms"] = round((audio.size - end_s) / sr * 1000.0, 1)

    act_rms = frame_rms[active]
    metrics["active_rms_dbfs"] = round(_dbfs(math.sqrt(float(np.mean(act_rms ** 2)))), 2)
    mean_rms = float(act_rms.mean())
    if mean_rms > 0:
        metrics["energy_cv"] = round(float(act_rms.std() / mean_rms), 3)

    # Edge abruptness: energy of the first/last EDGE_MS of the ACTIVE span
    # (i.e. after any TTS-added edge silence) vs the median active frame.
    # Natural utterances decay into their ending; a take cut mid-vowel ends
    # near full energy — and survives both the WER gate and stitch trimming.
    edge_n = max(1, int(round(sr * EDGE_MS / 1000.0)))
    median_rms = float(np.median(act_rms))
    if median_rms > 0 and end_s - start_s >= edge_n:
        start_edge = math.sqrt(float(np.mean(audio[start_s:start_s + edge_n] ** 2)))
        end_edge = math.sqrt(float(np.mean(audio[end_s - edge_n:end_s] ** 2)))
        metrics["edge_start_ratio"] = round(start_edge / median_rms, 3)
        metrics["edge_end_ratio"] = round(end_edge / median_rms, 3)

    f0, voiced = _pitch_track(frames, sr)
    voiced &= active
    metrics["voiced_fraction"] = round(float(voiced.mean()), 3)
    v_f0 = f0[voiced]
    if v_f0.size >= 3:
        f0_med = float(np.median(v_f0))
        metrics["f0_median_hz"] = round(f0_med, 1)
        semitones = 12.0 * np.log2(v_f0 / f0_med)
        metrics["f0_std_semitones"] = round(float(semitones.std()), 2)

    return metrics


def _lufs(audio: np.ndarray, sr: int):
    """Integrated LUFS via pyloudnorm when available; None otherwise.
    The BS.1770 meter needs ~400 ms blocks, so very short chunks skip it."""
    if audio.size < int(0.5 * sr):
        return None
    try:
        import pyloudnorm as pyln
    except ImportError:
        return None
    try:
        loudness = pyln.Meter(sr).integrated_loudness(audio)
    except Exception:
        return None
    return round(float(loudness), 2) if math.isfinite(loudness) else None


def derive_flags(metrics: dict) -> list[str]:
    """Advisory flags from one chunk's metrics — triage, never a retry gate."""
    flags = []
    if metrics.get("clip_ratio", 0.0) > CLIP_RATIO_FLAG:
        flags.append("clipping")
    active_rms = metrics.get("active_rms_dbfs")
    level = active_rms if active_rms is not None else metrics.get("rms_dbfs")
    if level is not None and level < LOW_LEVEL_DBFS:
        flags.append("low_level")
    f0_std = metrics.get("f0_std_semitones")
    voiced = metrics.get("voiced_fraction")
    if (f0_std is not None and voiced is not None
            and voiced >= FLAT_PITCH_MIN_VOICED and f0_std < FLAT_PITCH_ST):
        flags.append("flat_pitch")
    energy_cv = metrics.get("energy_cv")
    if energy_cv is not None and energy_cv < FLAT_ENERGY_CV:
        flags.append("flat_energy")
    start_ratio = metrics.get("edge_start_ratio")
    if start_ratio is not None and start_ratio > ABRUPT_START_RATIO:
        flags.append("abrupt_start")
    end_ratio = metrics.get("edge_end_ratio")
    if end_ratio is not None and end_ratio > ABRUPT_END_RATIO:
        flags.append("abrupt_end")
    return flags


def loudness_outliers(loudness_by_id: dict, threshold_db: float = LOUDNESS_OUTLIER_DB,
                      window: int = 2) -> dict:
    """
    Chunk-vs-neighbours loudness consistency: {id: deviation_db} for chunks
    whose loudness deviates more than threshold_db from the median of their
    nearest 2*window neighbours (self excluded). A neighbour window — not one
    global median — so intended long-arc dynamics don't false-flag; nearest-N
    rather than N-per-side so an edge chunk is still judged against a full
    window (against a half window, one genuinely bad neighbour would drag the
    median and false-flag the edge chunk itself). Ids are compared in sorted
    order; None values neither judge nor get judged.
    """
    ids = sorted(loudness_by_id)
    vals = [loudness_by_id[i] for i in ids]
    out = {}
    for idx, cid in enumerate(ids):
        if vals[idx] is None:
            continue
        by_distance = sorted((abs(j - idx), j) for j in range(len(ids))
                             if j != idx and vals[j] is not None)
        neigh = [vals[j] for _, j in by_distance[:2 * window]]
        if len(neigh) < 2:
            continue
        dev = vals[idx] - statistics.median(neigh)
        if abs(dev) > threshold_db:
            out[cid] = round(dev, 2)
    return out


def rank_candidates(entries: list) -> list:
    """
    Pre-listening sort order for candidate takes, best first. Not a verdict —
    a filter for where to start listening. Order of concerns: passing the
    duration gate, then fewest advisory flags, then lowest WER, then the
    LIVELIEST pitch contour (prosody is why you generate candidates at all).
    Each entry: {"version", "wer", "duration_ok", "flags", "metrics", ...}.
    """
    def key(e):
        wer = e.get("wer")
        wer = float(wer) if isinstance(wer, (int, float)) and wer >= 0 else 1.0
        flags = e.get("flags") or []
        m = e.get("metrics") or {}
        f0_std = m.get("f0_std_semitones")
        f0_std = float(f0_std) if f0_std is not None else 0.0
        return (0 if e.get("duration_ok", True) else 1, len(flags),
                round(wer, 4), -f0_std)
    return sorted(entries, key=key)
