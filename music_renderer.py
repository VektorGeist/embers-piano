"""music_renderer.py - composing-side music renderer.

renderer(artifact, out_path, faults): plays the score through
the root piano, applies faults in the playing, writes the WAV
to out_path, returns {"path", "kind": "audio", "duration", "peak"}.
"""

import os

from piano import render_score


def _clamp01(v):
    return float(min(1.0, max(0.0, float(v))))


def _flatten(score):
    events = []
    for part in score.get("parts", []):
        gain = float(part.get("gain", 1.0))
        for n in part.get("notes", []):
            events.append({"part": part.get("name"), "start": float(n["start"]),
                           "midi": int(n["midi"]), "vel": _clamp01(float(n["vel"]) * gain),
                           "dur": n.get("dur")})
    return events


def _apply_faults(events, faults):
    result = [dict(e) for e in events]
    for fault in faults or []:
        kind = fault.get("kind")
        if kind == "part_gain":
            for e in result:
                if e["part"] == fault.get("part"):
                    e["vel"] = _clamp01(e["vel"] * float(fault["factor"]))
        elif kind == "swell":
            for e in result:
                if float(fault["from"]) <= e["start"] < float(fault["to"]):
                    e["vel"] = _clamp01(e["vel"] * float(fault["factor"]))
        elif kind == "drop_note":
            for i, e in enumerate(result):
                if e["part"] == fault.get("part") and float(e["start"]) == float(fault["start"]) and int(e["midi"]) == int(fault["midi"]):
                    del result[i]
                    break
        else:
            raise ValueError("unknown fault kind: %r" % (kind,))
    return result


def renderer(artifact, out_path, faults):
    events = _apply_faults(_flatten(artifact), faults)
    tuples = [(e["start"], e["midi"], e["vel"], e["dur"]) for e in events]
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    duration, peak = render_score(tuples, out_path, humanize_ms=0.0)
    return {"path": str(out_path), "kind": "audio", "duration": float(duration),
            "peak": float(peak), "events_rendered": len(tuples),
            "faults_applied": list(faults or [])}
