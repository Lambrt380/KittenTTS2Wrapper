import numpy as np
import soundfile as sf


def rate_voice(path):
    # These signal checks estimate reference suitability, not cloning fidelity.
    try:
        with sf.SoundFile(path) as recording:
            seconds = len(recording) / recording.samplerate
            rate = recording.samplerate
            samples = recording.read(rate * 30, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        return 0, f"Unreadable WAV: {error}"
    if not samples.size or not np.isfinite(samples).all():
        return 0, "Empty or invalid audio"

    score = 5
    notes = []
    if seconds < 5:
        score -= 2
        notes.append("too short; aim for 5–30s")
    elif seconds > 30:
        score -= 1
        notes.append("long clip; trim to 5–30s")
    if rate < 16000:
        score -= 1
        notes.append("low sample rate")
    if np.mean(np.abs(samples) >= 0.999) > 0.001:
        score -= 1
        notes.append("possible clipping")
    mono = samples.mean(axis=1)
    if np.sqrt(np.mean(mono**2)) < 0.015:
        score -= 3
        notes.append("very quiet")
    frame_size = max(1, rate // 50)
    frames = mono[: len(mono) // frame_size * frame_size]
    if frames.size:
        levels = np.sqrt(np.mean(frames.reshape(-1, frame_size) ** 2, axis=1))
        if np.mean(levels < 0.003) > 0.4:
            score -= 1
            notes.append("lots of silence")
    details = "; ".join(notes) or "good length and recording level"
    return max(1, score), f"{seconds:.1f}s · {rate / 1000:g} kHz — {details}"
