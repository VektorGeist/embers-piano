"""piano.py - the model's instrument on the composing side.

Plays a score into a WAV file.  Additive stretched-partial
synthesis, 88 keys (A0 to C8).  This file lives at the project
root; nothing under stage2 plays or synthesizes sound.
"""

import math
import os
import wave

import numpy as np

__all__ = ["render_score", "SAMPLE_RATE"]

SAMPLE_RATE = 44100
_FAST_WEIGHT = 0.85
_SLOW_WEIGHT = 0.15
_FAST_TAU_FRACTION = 0.22
_INAUDIBLE_LEVEL = 1.0e-3
_MAX_RING_SECONDS = 30.0
_TAIL_SECONDS = 0.1
_DAMPER_FADE_SECONDS = 0.03
_MAX_PARTIAL_FREQ = 8000.0
_MIN_PARTIAL_REL_AMP = 0.004
_MAX_PARTIALS = 32


def _midi_to_freq(midi):
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def _inharmonicity(f0):
    return float(np.clip(4.0e-4 * (261.63 / f0) ** 1.2, 2.0e-5, 1.5e-3))


def _slow_tau(f0):
    return float(np.clip(20.0 * (27.5 / f0) ** 0.65, 0.4, 20.0))


def _string_count(f0):
    if f0 < 65.0:
        return 1
    if f0 < 130.0:
        return 2
    return 3


def _tilt_db_per_100hz(velocity):
    return max(0.35, 2.4 - 1.9 * float(velocity))


def _partial_envelope(t, tau_slow):
    tau_fast = _FAST_TAU_FRACTION * tau_slow
    return _FAST_WEIGHT * np.exp(-t / tau_fast) + _SLOW_WEIGHT * np.exp(-t / tau_slow)


def _ring_seconds(tau_slow):
    if _SLOW_WEIGHT <= _INAUDIBLE_LEVEL:
        return _MAX_RING_SECONDS
    return float(min(max(tau_slow * math.log(_SLOW_WEIGHT / _INAUDIBLE_LEVEL), 0.4), _MAX_RING_SECONDS))


def _sounding_seconds(f0, duration):
    if duration is not None:
        return max(0.0, float(duration))
    return _ring_seconds(_slow_tau(f0))


def render_score(events, out_wav, humanize_ms=0.0):
    """Render (start, midi, velocity, duration_or_None) events to WAV.

    Returns (duration_seconds, peak).  Deterministic; at
    humanize_ms=0.0 there is no jitter.
    """
    events = list(events)
    prepared = []
    for start, midi, velocity, duration in events:
        start = float(start)
        midi = int(midi)
        if midi < 21 or midi > 108:
            continue
        velocity = float(np.clip(float(velocity), 0.0, 1.0))
        if velocity <= 0.0:
            continue
        duration = None if duration is None else float(duration)
        f0 = _midi_to_freq(midi)
        sounding = _sounding_seconds(f0, duration)
        if sounding > 0.0:
            prepared.append((start, midi, velocity, duration, f0, sounding))
    parent = os.path.dirname(os.path.abspath(out_wav))
    os.makedirs(parent, exist_ok=True)
    if not prepared:
        with wave.open(out_wav, "wb") as h:
            h.setnchannels(1); h.setsampwidth(2); h.setframerate(SAMPLE_RATE)
            h.writeframes(b"\x00\x00" * int(round(_TAIL_SECONDS * SAMPLE_RATE)))
        return _TAIL_SECONDS, 0.0
    last_end = max(s + d for s, _, _, _, _, d in prepared)
    duration_seconds = last_end + _TAIL_SECONDS
    n_samples = int(round(duration_seconds * SAMPLE_RATE))
    mix = np.zeros(n_samples, dtype=np.float64)
    for start, midi, velocity, duration, f0, sounding in prepared:
        b = _inharmonicity(f0)
        tau_fund = _slow_tau(f0)
        n_strings = _string_count(f0)
        offsets = [0.0] if n_strings == 1 else [(i - (n_strings - 1) / 2.0) * 1.0 for i in range(n_strings)]
        start_sample = int(round(start * SAMPLE_RATE))
        note_samples = min(int(round(sounding * SAMPLE_RATE)), n_samples - start_sample)
        if note_samples <= 0:
            continue
        t = np.arange(note_samples, dtype=np.float64) / SAMPLE_RATE
        slope = _tilt_db_per_100hz(velocity)
        partials = []
        m = 1
        while m <= _MAX_PARTIALS:
            fm = m * f0 * math.sqrt(1.0 + b * m * m)
            if fm > _MAX_PARTIAL_FREQ and m > 1:
                break
            rel = 10.0 ** (-slope * (fm - f0) / 100.0 / 20.0)
            if m > 1 and rel < _MIN_PARTIAL_REL_AMP:
                break
            partials.append((m, rel, tau_fund * (f0 / fm) ** 0.45))
            m += 1
        amp_sum = sum(r for _, r, _ in partials) or 1.0
        wave_note = np.zeros(note_samples, dtype=np.float64)
        for off in offsets:
            fs = f0 + off
            if fs <= 0:
                continue
            for m, rel, tau in partials:
                fm = m * fs * math.sqrt(1.0 + b * m * m)
                if fm > _MAX_PARTIAL_FREQ * 1.5:
                    continue
                wave_note += (rel / amp_sum) * _partial_envelope(t, tau) * np.sin(2.0 * math.pi * fm * t)
            wave_note /= n_strings
        if duration is not None and duration > 0.0:
            fade = min(_DAMPER_FADE_SECONDS, duration * 0.5)
            if fade > 0.0:
                fs_ = duration - fade
                mask = t >= fs_
                if np.any(mask):
                    wave_note[mask] *= 0.5 * (1.0 + np.cos(math.pi * (t[mask] - fs_) / fade))
        mix[start_sample:start_sample + note_samples] += 0.85 * velocity * wave_note
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.89:
        mix *= 0.89 / peak
        peak = 0.89
    pcm = np.clip(mix, -1.0, 1.0)
    pcm16 = np.round(pcm * 32767.0).astype(np.int16)
    with wave.open(out_wav, "wb") as h:
        h.setnchannels(1); h.setsampwidth(2); h.setframerate(SAMPLE_RATE)
        h.writeframes(pcm16.tobytes())
    return float(duration_seconds), float(np.max(np.abs(pcm16.astype(np.float64) / 32767.0))) if pcm16.size else 0.0
