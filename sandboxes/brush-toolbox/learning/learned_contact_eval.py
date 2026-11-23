"""Train conservative transverse contact distributions from independent Chinese painting courses."""

import argparse
from dataclasses import dataclass
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import nnls
import torch
from torch import nn

import contact_projection_eval as signed
from learned_contact import TransverseContact
from metrics import envelope
import paper_appearance_eval as cases
from shared_brush import Contact, measured_contact, optical, place
from timed_deposition import OBSERVATIONS
from wet_dry import load_engine
from width_model import NAME


OUT = cases.ROOT / "results" / "learned-contact"
CALIBRATION = cases.ROOT / "results" / "wet-dry" / "calibration.json"
AXIS = np.linspace(-1.36, 1.36, 73)
FOLDS = ("xingjie", "yixue", "aam", "gathered")
INKING = {
    "aam-f": (.63, .185, .185), "gathered-vi": (.63, .185, .185),
    "aam-g": (.19, .405, .405), "gathered-vii": (.19, .405, .405),
}


@dataclass(frozen=True)
class Reference:
    name: str
    label: str
    ink: np.ndarray
    guide: Contact
    shaft: np.ndarray


def references():
    trace = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))[NAME]
    signed_records = signed.case_bank(trace)
    metadata = {record.name: record for rows in signed_records.values() for record in rows}
    grouped = {}
    for course, samples in cases.examples().items():
        entries = []
        for sample in samples:
            known = metadata.get(sample.name)
            if known is None:
                ink = cases.native(sample)
                crop = envelope(ink)[0]
                ink = ink[:, crop]
                entries.append(Reference(sample.name, sample.label, ink, measured_contact(ink),
                                         np.zeros(ink.shape[1])))
            else:
                entries.append(Reference(sample.name, sample.label, known.ink, known.guide, known.axes[:, 1]))
        grouped[course] = entries
    captured = signed_records["wu"][0]
    grouped["wu"] = [Reference(captured.name, "吴作成，实拍锋位", captured.ink, captured.guide, captured.axes[:, 1])]
    return grouped


def profile_samples(reference):
    width = reference.guide.width
    center = reference.guide.center
    x = np.arange(len(width))
    interior = (width >= .52 * reference.guide.settled_width) & np.isfinite(reference.shaft)
    visible = np.flatnonzero(interior)
    if len(visible) < 5:
        raise ValueError(f"Too little observed contact in {reference.name}")
    use = visible[np.linspace(0, len(visible) - 1, min(72, len(visible))).round().astype(int)]
    ink = gaussian_filter1d(optical(reference.ink), max(.65, .025 * reference.guide.settled_width), axis=1)
    sampling_x = np.broadcast_to(x[use][None], (len(AXIS), len(use)))
    sampling_y = center[use][None] + .5 * width[use][None] * AXIS[:, None]
    measured = map_coordinates(ink, (sampling_y, sampling_x), order=1, mode="constant").T
    selected = measured.sum(axis=1) > 1e-6
    measured = measured[selected]
    return use[selected], (measured / measured.sum(axis=1, keepdims=True)).astype(np.float32)


class KernelModel(nn.Module):
    """Learn the widthwise shape of all three stock zones from deposited profiles."""

    def __init__(self):
        super().__init__()
        self.center = nn.Sequential(nn.Linear(4, 26), nn.SiLU(), nn.Linear(26, 26), nn.SiLU(), nn.Linear(26, 1))
        self.flank = nn.Sequential(nn.Linear(4, 26), nn.SiLU(), nn.Linear(26, 26), nn.SiLU(), nn.Linear(26, 1))
        for component in (self.center, self.flank):
            nn.init.zeros_(component[-1].weight)
            nn.init.zeros_(component[-1].bias)
        self.register_buffer("u", torch.from_numpy(AXIS).float()[None])

    def forward(self, shaft):
        u = self.u.expand(len(shaft), -1)
        s = shaft[:, None].expand_as(u)
        mask = (1. - (u / AXIS[-1]).square()).clamp_min(1e-5)

        def solve(p, a):
            xy = torch.stack((p, p.square(), a, p * a), dim=-1)
            return self.flank(xy)[..., 0] - 1.8 * (p - .42).square()

        core = torch.stack((u.square(), u * s, s.square(), u.square() * s.square()), dim=-1)
        center = self.center(core)[..., 0] - 2.0 * (u - .23 * s).square()
        logits = torch.stack((center, solve(-u, -s), solve(u, s)), dim=1)
        return (logits + 1.7 * mask[:, None].log()).softmax(dim=-1)


def fit(groups, first, second):
    targets, angles, marks, weights, priors = [], [], [], [], []
    for group in groups.values():
        for record in group:
            indices, profile = profile_samples(record)
            targets.append(profile)
            angles.append(record.shaft[indices])
            marks.append(np.full(len(profile), len(priors)))
            weights.append(np.full(len(profile), 1. / (len(profile) * len(group) * len(groups))))
            priors.append(INKING.get(record.name, (.44, .28, .28)))
    target, angle, weight = (torch.from_numpy(np.concatenate(value)).float() for value in (targets, angles, weights))
    index = torch.from_numpy(np.concatenate(marks)).long()
    preferred = torch.tensor(priors).log()
    latent = nn.Parameter(torch.zeros_like(preferred))
    model = KernelModel()
    opt = torch.optim.AdamW([*model.parameters(), latent], lr=.008, weight_decay=.0004)
    report = []
    for step in range(first + second):
        normalized = (preferred + latent).softmax(dim=-1)
        stock = preferred.softmax(dim=-1) if step < first else normalized
        parts = model(angle)
        mixture = (parts * stock[index, :, None]).sum(dim=1)
        cross = -(target * mixture.clamp_min(1e-9).log()).sum(dim=1)
        transport = (target.cumsum(dim=1) - mixture.cumsum(dim=1)).abs().mean(dim=1)
        second_derivative = parts.diff(n=2, dim=2).square().mean()
        loss = weight @ (cross + 2. * transport) + .16 * latent.square().mean() + .25 * second_derivative
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        report.append(float(loss.detach()))
        if step + 1 in (first, first + second):
            print(f"  contact training {step+1}/{first+second}, loss {np.mean(report[-12:]):.5f}", flush=True)
    return model.eval(), report


def analytic(shaft, calibration):
    u = AXIS
    shifted = u[None] - .48 * shaft[:, None]
    center = np.maximum(0., 1. - shifted ** 2) ** 2.5
    center *= np.interp(abs(shifted), calibration.radial, calibration.tip_profile)
    wings = np.maximum(0., 1. - u ** 2) ** 2.5 * np.interp(abs(u), calibration.radial, calibration.flank_profile)
    left = np.broadcast_to(wings * np.maximum(0., -u), center.shape)
    right = np.broadcast_to(wings * np.maximum(0., u), center.shape)
    fields = np.stack((center, left, right), axis=1)
    return fields / fields.sum(axis=-1, keepdims=True)


def from_opening(table, target, seen, preferred):
    penalty = .3
    samples = table[seen].transpose(0, 2, 1).reshape(-1, 3)
    start = target[seen].ravel()
    loading, _ = nnls(np.vstack((samples, penalty * np.eye(3))), np.r_[start, penalty * np.asarray(preferred)])
    return loading / loading.sum()


def grade(reference, model, base):
    selected, truth = profile_samples(reference)
    shaft = reference.shaft[selected]
    first = max(2, round(.30 * len(selected)))
    prior = INKING.get(reference.name, (.44, .28, .28))
    records, photos = {}, [reference.ink]
    all_shaft = np.nan_to_num(reference.shaft)
    with torch.inference_mode():
        learned = model(torch.from_numpy(all_shaft).float()).numpy()
    for name, table in (("physical", analytic(all_shaft, base)),
                        ("fixed_loading", learned), ("estimated_loading", learned)):
        if name == "fixed_loading":
            stocks = np.asarray(prior)
        else:
            stocks = from_opening(table[selected], truth, slice(0, first), prior)
        profiles = (table * stocks[None, :, None]).sum(axis=1)
        drawn, expected = profiles[selected][first:], truth[first:]
        area = (drawn.cumsum(axis=1) - expected.cumsum(axis=1))
        records[name] = dict(relative_stock=stocks.tolist(), suffix_columns=int(len(drawn)),
                             transverse_distance_half_width=float(abs(area).mean() * (AXIS[-1] - AXIS[0])),
                             signed_centroid_error=float(np.mean(abs((drawn - expected) @ AXIS))),
                             center_share_mae=float(abs(drawn[:, abs(AXIS) < .28].sum(axis=1)
                                                        - expected[:, abs(AXIS) < .28].sum(axis=1)).mean()))
        n = (np.arange(len(reference.ink))[:, None] - reference.guide.center[None]) / (.5 * reference.guide.width[None])
        predicted = np.column_stack([np.interp(q, AXIS, p, left=0., right=0.) for q, p in zip(n.T, profiles, strict=True)])
        photos.append(-np.expm1(-place(predicted, optical(reference.ink).sum(axis=0))))
    return dict(first_sample_columns=first, observed_profile_columns=selected.tolist(), models=records), photos


def replay(mode):
    from ink_coverage_eval import OUT as older, recorded_suffix, replay_recorded

    if mode == "compare":
        recorded_suffix({"recorded-prefix": older, "fixed-stock": OUT, "refitted-stock": OUT}, OUT)
        return
    previous = json.loads((older / "recorded-prefix.json").read_text(encoding="utf-8"))
    frozen = previous["prefix"] if mode == "fixed" else None
    name = "fixed-stock" if frozen else "refitted-stock"
    replay_recorded(140, 120, reuse_checkpoint=True, fit_prefix=(frozen is None), frozen_prefix=frozen,
                    contact_model=OUT / "without-xingjie.npz", destination=OUT, output_name=name)
    if frozen is not None:
        latest = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        if not np.isclose(latest["new_pigment_mg"], previous["new_pigment_mg"], rtol=1e-9):
            raise AssertionError("The fixed-stock replay must lay down the same pigment mass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=160)
    parser.add_argument("--second", type=int, default=190)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=FOLDS)
    parser.add_argument("--replay", choices=("fixed", "refit", "compare"))
    args = parser.parse_args()
    if min(args.first, args.second) < 1:
        parser.error("Both training phases require a positive number of steps")
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.replay is not None:
        replay(args.replay)
        return
    bank = references()
    base = load_engine(CALIBRATION).calibration
    report, gallery = {}, []
    focus = dict(xingjie="xingjie-side", yixue="lesson-upper", aam="aam-g", gathered="gathered-vii")
    for course in args.folds:
        torch.manual_seed(812)
        print(f"Withheld {course}", flush=True)
        training = {source: references for source, references in bank.items() if source != course}
        model, loss = fit(training, args.first, args.second)
        shaft = np.linspace(-1., 1., 41)
        with torch.inference_mode():
            table = model(torch.from_numpy(shaft).float()).numpy()
        interface = TransverseContact(AXIS, shaft, table)
        np.testing.assert_allclose(interface.transverse(AXIS, .47)[[0, 2, 1], ::-1],
                                   interface.transverse(AXIS, -.47), atol=5e-8)
        np.savez_compressed(OUT / f"without-{course}.npz", coordinate=AXIS, projection=shaft,
                            tables=table, sources=np.array(list(training)))
        tests = {}
        for real in bank[course]:
            readings, examples = grade(real, model, base)
            tests[real.name] = readings
            if real.name == focus[course]:
                gallery.append((real.label, .57, examples))
        report[course] = dict(training=list(training), objective=loss, marks=tests)
        (OUT / "profiles.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        cases.board(gallery, OUT / "transverse.png", titles=("实写", "原接纸模型", "学习横截面", "加起段载墨估计"))


if __name__ == "__main__":
    main()
