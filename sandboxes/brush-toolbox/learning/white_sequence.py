"""Learn the conditional sequence of exposed paper within a dry side-brush contact."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import expit
import torch

import conservative_detail as detail
import gap_deposition as gaps
from gap_sampling import deposit_with_gaps, shapes
from metrics import CORE, PAPER
from width_model import NAME, PREPARED, ROOT


OUT = ROOT / "results" / "white-sequence"
COURSES = {"yixue": ("lesson-141", "lesson-upper", "lesson-lower"),
           "xingjie": (NAME,), "textbook": ("textbook",)}
MODES = ("independent", "paper-history", "edge-history", "edge-and-transverse", "marginal-history")
LAGS = ((1, 0), (1, -1), (1, 1), (2, 0), (4, 0), (8, 0))
HISTORY_COLUMNS = max(lag for lag, _ in LAGS)
CONTEXT = ("intercept", "phase", "phase_squared", "signed_position", "from_midline",
           "near_edge_px", "signed_near_edge", "phase_signed", "phase_near_edge",
           "ink_per_width_change", "ink_near_edge", "relative_contact_width")
HISTORY = ("previous", "previous_negative_row", "previous_positive_row", "two_back",
           "four_back", "eight_back", "previous_near_edge", "previous_by_phase")
RIDGE = 0.001


@dataclass
class Frame:
    contact: np.ndarray
    domain: np.ndarray
    low: np.ndarray
    high: np.ndarray
    phase: np.ndarray
    across: np.ndarray
    context: np.ndarray


def envelope(ink):
    occupied = np.asarray(ink) >= CORE
    return np.maximum.accumulate(occupied, axis=0) & np.maximum.accumulate(occupied[::-1], axis=0)[::-1]


def interior(contact):
    pad = np.pad(contact, ((1, 1), (0, 0)))
    return pad[:-2] & pad[1:-1] & pad[2:]


def frame(contact, amount, *, opening_columns, allowed=None):
    """Express positions in the signed local cross-section and in registered paper pixels."""
    contact = np.asarray(contact, dtype=bool)
    amount = np.asarray(amount, dtype=float).reshape(-1)
    if contact.ndim != 2 or amount.shape != (contact.shape[1],) or not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("Provide the contact raster and nonnegative optical amount per column")
    if not 1 <= opening_columns <= contact.shape[1]:
        raise ValueError("Provide the length of the visible opening")
    domain = interior(contact) if allowed is None else np.asarray(allowed, dtype=bool)
    if domain.shape != contact.shape or (domain & ~contact).any():
        raise ValueError("Candidate paper must lie within the contact raster")
    h, w = contact.shape
    rows = np.arange(h)[:, None]
    stations = np.arange(w)
    observed = np.flatnonzero(contact.sum(axis=0) >= 3)
    if observed.size:
        first = np.argmax(contact[:, observed], axis=0)
        last = h - 1 - np.argmax(contact[::-1, observed], axis=0)
        low, high = (np.interp(stations, observed, values) for values in (first, last))
        phase = (2. * (stations - observed[0]) / max(1, observed[-1] - observed[0]) - 1.).clip(-1., 1.)
    else:
        low = high = np.full(w, .5 * (h - 1))
        phase = np.zeros(w)
    span = np.maximum(1., high - low)
    across = (2. * (rows - low) / span - 1.).clip(-1., 1.)
    on_edge = np.exp(-np.maximum(0., np.minimum(rows - low, high - rows)) / 4.)
    blur = gaussian_filter1d(amount, sigma=6., mode="nearest")
    ink = np.log(np.maximum(.035, blur / (span + 1.)))
    early = observed[observed < opening_columns]
    initial_width = np.median(span[early]) if early.size else max(1., np.median(span))
    initial_ink = np.median(ink[early]) if early.size else np.median(ink)
    density = (ink - initial_ink).clip(-2.5, 2.5)
    width = np.log((span + 1.) / (initial_width + 1.)).clip(-1.5, 1.5)
    fields = (1., phase, phase ** 2, across, np.abs(across), on_edge, across * on_edge,
              across * phase, phase * on_edge, density, density * on_edge, width)
    context = np.stack(np.broadcast_arrays(*fields, np.zeros_like(across))[:-1], axis=-1)
    return Frame(contact, domain, low, high, phase, across, context)


def links(view, mode):
    """Map past-column neighbours along either the paper grid or the changing brush band."""
    h, w = view.domain.shape
    stations = np.arange(w)[None]
    rows = np.arange(h)[:, None]
    result = []
    for lag, offset in LAGS:
        previous = (stations - lag).clip(0)
        y = rows if mode == "paper-history" else (
            view.low[previous] + .5 * (1. + view.across) * (view.high - view.low)[previous])
        y = np.rint(y).astype(int) + offset
        valid = (stations >= lag) & (y >= 0) & (y < h)
        result.append(np.where(valid, y * w + previous, h * w))
    return np.stack(result, axis=-1)


def history(raw, edge, phase):
    return np.concatenate((raw, raw[..., :1] * edge[..., None], raw[..., :1] * phase[..., None]), axis=-1)


def design(view, white, mode):
    features = view.context
    if mode == "independent":
        return features
    values = np.append(white.ravel(), False)
    adjacent = values[links(view, mode)].astype(float)
    temporal = history(adjacent, features[..., 5], np.broadcast_to(view.phase, white.shape))
    blocks = [features, temporal]
    if mode == "edge-and-transverse":
        blocks.append(np.pad(white[:-1], ((1, 0), (0, 0)))[..., None])
    return np.concatenate(blocks, axis=-1)


def training_fields(ink, cutoff):
    contact = envelope(ink)
    amount = -np.log(np.maximum(1. - ink, 1. / 255.)).sum(axis=0)
    view = frame(contact, amount, opening_columns=cutoff)
    white = view.domain & (ink <= PAPER)
    return view, white


def source_fields(natives, course, cutoff):
    cases = []
    for name in COURSES[course]:
        ink = natives[name]
        measured = detail.fitted_side(name)
        for sign in (-1, 1) if measured is None else (measured,):
            units = ink[::sign]
            cases.append((name, sign, *training_fields(units, cutoff)))
    return cases


def paired_fields(view, rate_weights):
    """Conditional deviations at the upper Bernoulli coupling bound."""
    h, w = view.domain.shape
    rate = expit(view.context @ rate_weights) * view.domain
    lateral = np.pad(np.arange(h * w).reshape(h, w)[:-1], ((1, 0), (0, 0)), constant_values=h * w)
    parents = np.concatenate((links(view, "paper-history"), lateral[..., None]), axis=-1)
    parent_rate = np.append(rate.ravel(), 0.)[parents]
    target = np.broadcast_to(rate[..., None], parent_rate.shape)
    low = np.divide(np.maximum(0., target - parent_rate), 1. - parent_rate,
                    out=target.copy(), where=parent_rate < 1.)
    high = np.divide(np.minimum(target, parent_rate), parent_rate,
                     out=target.copy(), where=parent_rate > 0.)
    return rate, parents, low - target, high - low


def fit_marginal_history(records, cutoff):
    rate_model = fit(records, "independent", cutoff)
    weighted_rate, x, y, weights = [], [], [], []
    for cases in records.values():
        area = sum(int(view.domain.sum()) for _, _, view, _ in cases)
        for _, _, view, white in cases:
            rate, parents, baseline, delta = paired_fields(view, rate_model["coefficients"])
            dev = baseline + delta * np.append(white.ravel(), False)[parents]
            phase = (1. + view.phase[None, :, None]) * .5
            features = np.concatenate((dev * (1. - phase), dev * phase), axis=-1)
            x.append(features[view.domain])
            weighted_rate.append(rate[view.domain])
            y.append(white[view.domain])
            weights.append(np.full(int(view.domain.sum()), 1. / (len(records) * area)))
    x, y, rate, weights = (np.concatenate(a) for a in (x, y, weighted_rate, weights))
    size = x.shape[1] // 2
    groups = np.repeat(np.eye(2), size, axis=1)

    def objective(coef):
        prob = (rate + x @ coef).clip(1e-7, 1. - 1e-7)
        value = -(y * np.log(prob) + (1. - y) * np.log1p(-prob))
        slope = (prob - y) / (prob * (1. - prob))
        return float(weights @ value + .5 * RIDGE * (coef @ coef)), x.T @ (weights * slope) + RIDGE * coef

    optimum = minimize(objective, np.full(2 * size, .05), jac=True, method="SLSQP",
                       bounds=[(0., 1.)] * (2 * size),
                       constraints={"type": "ineq", "fun": lambda w: 1. - groups @ w, "jac": lambda w: -groups},
                       options={"ftol": 1e-9, "maxiter": 120})
    if not optimum.success:
        raise ValueError(f"Marginal-preserving fit failed: {optimum.message}")
    return dict(rate_model, mode="marginal-history",
                coefficients=np.concatenate((rate_model["coefficients"], optimum.x)),
                weighted_train_objective=float(optimum.fun))


def fit(records, mode, cutoff):
    """Estimate a shared conditional likelihood with equal total weight for each source."""
    if mode == "marginal-history":
        return fit_marginal_history(records, cutoff)
    x, y, weights = [], [], []
    for cases in records.values():
        total = sum(int(v.domain.sum()) for _, _, v, _ in cases)
        if not total:
            raise ValueError("Each training source needs contact pixels")
        for _, _, view, label in cases:
            selected = view.domain
            x.append(design(view, label, mode)[selected])
            y.append(label[selected])
            weights.append(np.full(int(selected.sum()), 1. / (len(records) * total)))
    x, y, weights = (np.concatenate(a) for a in (x, y, weights))
    penalty = np.full(x.shape[-1], RIDGE)
    penalty[0] = 0.
    # Distance coordinates describe exposure. The larger recurrence features use the
    # same penalty, so rare continuations still require evidence in the training marks.
    initial = np.zeros(x.shape[-1])
    rate = np.average(y, weights=weights).clip(1e-4, 1. - 1e-4)
    initial[0] = np.log(rate / (1. - rate))

    def objective(coef):
        logits = x @ coef
        error = np.logaddexp(0., logits) - y * logits
        l2 = penalty * coef
        return float(weights @ error + .5 * (l2 @ coef)), x.T @ (weights * (expit(logits) - y)) + l2

    bounds = [(None, None)] * len(initial)
    if mode == "edge-and-transverse":
        bounds[-1] = (0., None)
    found = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                     options={"maxiter": 220, "ftol": 1e-11})
    if not found.success:
        raise ValueError(f"Fit failed for {mode}: {found.message}")
    return dict(mode=mode, coefficients=found.x, sources=sorted(records), opening_columns=cutoff,
                weighted_train_objective=float(found.fun))


def mixed_fields(view, weights):
    rate, indexes, baseline, delta = paired_fields(view, weights[:len(CONTEXT)])
    phase = (1. + view.phase[:, None]) * .5
    early, late = weights[len(CONTEXT):].reshape(2, -1)
    associations = early * (1. - phase) + late * phase
    return rate + (baseline * associations).sum(axis=-1), indexes, delta * associations


def conditional_score(model, view, actual):
    if model["mode"] == "marginal-history":
        base, parents, delta = mixed_fields(view, model["coefficients"])
        p = (base + (delta * np.append(actual.ravel(), False)[parents]).sum(axis=-1))[view.domain]
    else:
        x = design(view, actual, model["mode"])[view.domain]
        p = expit(x @ model["coefficients"])
    p = p.clip(1e-7, 1. - 1e-7)
    y = actual[view.domain]
    error = -(y * np.log2(p) + (1. - y) * np.log2(1. - p))
    return dict(bits=float(error.mean()), white_bits=float(error[y].mean()) if y.any() else None,
                brier=float(np.square(p - y).mean()), predicted_rate=float(p.mean()), observed_rate=float(y.mean()))


def linked_column(p0, p1, draw, legal):
    births = (draw < p0) & legal
    possible = (draw < p1) & legal
    rows = np.arange(legal.shape[-1])
    change = births | ~possible
    decisive = np.maximum.accumulate(np.where(change, rows, 0), axis=-1)
    return np.take_along_axis(births, decisive, axis=-1)


def column_sample(paper_logits, draw, legal, transverse):
    p0 = expit(paper_logits)
    if transverse <= 0.:
        return (draw < p0) & legal
    return linked_column(p0, expit(paper_logits + transverse), draw, legal)


def mask_sample(model, view, seeds, *, opening=None):
    """Draw fresh masks one column at a time with optional observed opening state."""
    if not seeds:
        raise ValueError("At least one seed is required")
    mode, weights = (model[k] for k in ("mode", "coefficients"))
    h, w = view.domain.shape
    draws = np.stack([np.random.default_rng(seed).random((h, w)) for seed in seeds])
    if mode == "independent":
        return (draws < expit(view.context @ weights)) & view.domain
    if mode == "marginal-history":
        base, lookup, pair_delta = mixed_fields(view, weights)
    else:
        base = view.context @ weights[:len(CONTEXT)]
        downstream = weights[len(CONTEXT):len(CONTEXT) + len(HISTORY)]
        y_effect = float(weights[-1]) if mode == "edge-and-transverse" else 0.
        lookup = links(view, mode)
    paper = np.zeros((len(seeds), h * w + 1), dtype=bool)
    rows = np.arange(h) * w
    if opening is not None:
        size = int(model["opening_columns"])
        opening = np.asarray(opening)
        if opening.ndim != 2 or opening.shape[0] != h or not 1 <= size <= min(opening.shape[1], w):
            raise ValueError("Provide the observed opening on the same paper raster")
        sample = opening[:, :size]
        if not np.isfinite(sample).all() or (sample < 0.).any() or (sample > 1.).any():
            raise ValueError("The visible opening must be a relative opacity in [0, 1]")
        known = envelope(sample)
        measured = known & (sample <= PAPER) & view.domain[:, :size]
    for station in range(w):
        if opening is not None and station == size:
            # Condition the later sequence on the actually visible gaps. Earlier
            # synthetic marks stay in the returned image and have no later influence.
            hist = slice(max(0, station - HISTORY_COLUMNS), station)
            before = np.arange(hist.start, hist.stop)
            cells = rows[:, None] + before[None]
            remembered = paper[:, cells].copy()
            paper[:, cells] = np.where(known[:, hist], measured[:, hist], remembered)
        draw, legal = draws[:, :, station], view.domain[:, station]
        if mode == "marginal-history":
            parents = paper[:, lookup[:, station, :-1]]
            strengths = pair_delta[:, station]
            p0 = base[:, station] + (parents * strengths[:, :-1]).sum(axis=-1)
            column = linked_column(p0, p0 + strengths[:, -1], draw, legal)
        else:
            raw = paper[:, lookup[:, station]].astype(float)
            features = history(raw, view.context[:, station, 5], view.phase[station])
            logits = base[:, station] + features @ downstream
            column = column_sample(logits, draw, legal, y_effect)
        paper[:, rows + station] = column
    if opening is not None and size < w:
        paper[:, cells] = remembered
    return paper[:, :-1].reshape(-1, h, w)


@torch.inference_mode()
def render(model, guide, optical, contact, seeds, *, side, opening=None):
    """Draw optical depth and its sampled gaps on the native paper grid."""
    if side not in (-1, 1):
        raise ValueError("Give the signed tip side")
    if guide.shape != detail.LOW or optical.shape != detail.HIGH or contact.shape != detail.HIGH:
        raise ValueError("Use the transport grid and its registered native paper grid")
    if side < 0:
        guide, optical, contact = (value.flip(0) for value in (guide, optical, contact))
        opening = opening[::-1] if opening is not None else None
    total = gaps.smooth_mass(guide[None])
    eligible, extent, brush, floor = shapes(optical, contact, total)
    view = frame(extent.numpy(), total[0, 0].numpy(), allowed=eligible.numpy(),
                 opening_columns=model["opening_columns"])
    mask = torch.from_numpy(mask_sample(model, view, seeds, opening=opening))
    ink = deposit_with_gaps(brush, extent, mask, total, floor)
    return (ink, mask) if side > 0 else (ink.flip(1), mask.flip(1))


def save(model, file):
    info = {k: v for k, v in model.items() if k != "coefficients"}
    np.savez_compressed(file, coefficients=model["coefficients"], metadata=json.dumps(info))


def load(file):
    with np.load(file, allow_pickle=False) as bank:
        state = json.loads(bank["metadata"].item())
        return dict(state, coefficients=bank["coefficients"])


def course_data():
    with np.load(PREPARED / "fine.npz", allow_pickle=False) as bank:
        return {name: bank[name].copy() for names in COURSES.values() for name in names}


def fit_all(cutoff=192):
    natives = course_data()
    records = {course: source_fields(natives, course, cutoff) for course in COURSES}
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for holdout in COURSES:
        condition = {key: item for key, item in records.items() if key != holdout}
        inner_scores = {}
        for validation, cases in condition.items():
            train = {key: item for key, item in condition.items() if key != validation}
            for mode in MODES:
                model = fit(train, mode, cutoff)
                inner_scores.setdefault(mode, {})[validation] = [
                    dict(stroke=name, tip_side=side, **conditional_score(model, view, truth))
                    for name, side, view, truth in cases]
        means = {mode: float(np.mean([np.mean([s["bits"] for s in case_scores])
                                     for case_scores in val.values()]))
                 for mode, val in inner_scores.items()}
        chosen = min(means, key=means.get)
        outer = {}
        for mode in MODES:
            model = fit(condition, mode, cutoff)
            model.update(holdout=holdout, selected=chosen, validation_bits=means[mode])
            save(model, OUT / f"{holdout}-{mode}.npz")
            outer[mode] = [dict(stroke=name, tip_side=side, **conditional_score(model, view, truth))
                           for name, side, view, truth in records[holdout]]
        results[holdout] = dict(selected=chosen, validation=inner_scores, mean_validation_bits=means,
                                likelihood_known_previous_column=outer)
        print(json.dumps(dict(holdout=holdout, primary=chosen, validation=means,
                              heldout={mode: [s["bits"] for s in case] for mode, case in outer.items()})), flush=True)
    (OUT / "training.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    train = actions.add_parser("train")
    train.add_argument("--opening-columns", type=int, default=192)
    draw = actions.add_parser("mask")
    draw.add_argument("model", type=Path)
    draw.add_argument("contact", type=Path)
    draw.add_argument("ink_amount", type=Path)
    draw.add_argument("output", type=Path)
    draw.add_argument("--side", type=int, required=True, choices=(-1, 1))
    draw.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.action == "train":
        fit_all(args.opening_columns)
    else:
        model = load(args.model)
        contact = np.load(args.contact, allow_pickle=False).astype(bool)
        amount = np.load(args.ink_amount, allow_pickle=False)
        view = frame(contact[::args.side], amount, opening_columns=model["opening_columns"])
        mask = mask_sample(model, view, (args.seed,))[0][::args.side]
        np.save(args.output, mask, allow_pickle=False)


if __name__ == "__main__":
    main()
