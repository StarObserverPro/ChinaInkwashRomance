"""Read recurring ink structure separately from time-varying motion and lesson terms."""

import json

import numpy as np
from scipy.ndimage import distance_transform_edt

from metrics import envelope, measure
from paired_pose_eval import library
from timed_deposition import OBSERVATIONS, clock
from width_model import ROOT


OUT = ROOT / "results" / "stroke-end"
WHITE_LEVELS = (.035, .05, .075)


def continued(white, inside, lag):
    edge = inside[:, :-lag] & inside[:, lag:]
    starts = (white[:, :-lag] & edge).sum()+(white[:, lag:] & edge).sum()
    pairs = (white[:, :-lag] & white[:, lag:] & edge).sum()
    return dict(pairs=int(pairs), starts=int(starts), probability=float(2*pairs/starts) if starts >= 16 else None)


def differences(ink, inside, lag):
    valid = inside[:, :-lag] & inside[:, lag:]
    if valid.sum() < 32:
        return None
    contrast = ink[:, :-lag]-ink[:, lag:]
    return float(np.mean(np.square(contrast[valid])))


def interior_direction(ink, inside, lag, width):
    core = distance_transform_edt(inside) >= max(2., .075*width)
    along, across = differences(ink, core, lag), differences(ink.T, core.T, lag)
    if along is None or across is None or along <= 1e-8:
        return None
    rng = np.random.default_rng(1223)
    null = []
    for _ in range(6):
        shuffled = ink.copy()
        for x in range(ink.shape[1]):
            y = core[:, x]
            shuffled[y, x] = rng.permutation(ink[y, x])
        a, b = differences(shuffled, core, lag), differences(shuffled.T, core.T, lag)
        if a is not None and b is not None and a > 1e-8:
            null.append(b/a)
    return dict(across_over_along=across/along, column_permutation=null)


def morphology(ink):
    span, upper, lower, interior = envelope(ink)
    mark = ink[:, span]
    width = np.median(lower-upper+1.)
    lag = max(2, round(.12*width))
    readings = []
    for limit in WHITE_LEVELS:
        white = (mark <= limit) & interior
        readings.append(dict(level=limit, fraction=float(white.sum()/interior.sum()),
                             along=continued(white, interior, lag), across=continued(white.T, interior.T, lag)))
    stats = measure(ink)
    return dict(width_px=float(width), lag_px=lag, lag_in_widths=float(lag/width),
                ink_depth=stats["mean_density"], eccentricity=stats["eccentricity"],
                roughness=stats["roughness"], spatial=readings, internal_direction=interior_direction(mark, interior, lag, width))


def observed_motion():
    traces = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    summary = {}
    for name, trace in traces.items():
        contact = clock(trace)
        times, progress = contact["stations_s"], contact["stations_progress"]
        moving = [f for f in trace["frames"] if f["state"] == "travel"]
        tangent = np.asarray(moving[-1]["path"])-moving[0]["path"]
        tangent = tangent/np.linalg.norm(tangent)
        right = np.array((-tangent[1], tangent[0]))
        stations = []
        for frame in trace["frames"]:
            if "tip" not in frame or frame["state"] not in ("travel", "hold"):
                continue
            q = np.asarray(frame["tip"])-frame["root"]
            projection = np.array((q @ tangent, q @ right))/np.linalg.norm(q)
            stations.append(dict(time_s=frame["s"], contact_state=frame["state"], signed_tip_TN=projection.tolist()))
        hold = contact["hold_end"]-contact["arrival"] if any(frame["state"] == "hold" for frame in trace["frames"]) else None
        summary[name] = dict(source=trace["source"], observed_hold_s=hold,
                             speeds_in_projected_stroke_lengths_per_s=(np.diff(progress)/np.diff(times)).tolist(),
                             signed_tip_by_frame=stations, normal_force=None, calibrated_mm=None,
                             time_bracket=trace["lift_bracket_s"])
    return summary


def main():
    bank, _ = library()
    course = {source: {model.name: dict(teaching_label=model.label, nominal_posture=int(model.pose),
                                      label_evidence=model.label_quality, **morphology(model.ink))
                       for model in rows} for source, rows in bank.items()}
    report = dict(observed_ink=course, observed_action=observed_motion(),
                  controls=dict(lag_in_brush_widths=.12, white_levels=WHITE_LEVELS,
                                interpretation="Static ink measures geometry. Time-varying action needs independent observation."))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"brush-state.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print({source: len(samples) for source, samples in course.items()}, flush=True)


if __name__ == "__main__":
    main()
