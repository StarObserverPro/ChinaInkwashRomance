"""Train paired rail models with entire Chinese-painting courses held out."""

import argparse
from dataclasses import dataclass
import json
from statistics import median

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F

import brush_patch_eval as fine
from brush_patch import TextureStrength
import conservative_detail as detail
from frequency import spectrum
import ink_layout as layout
from ink_layout_eval import native_textbook_crop
import linked_apertures as channels
from metrics import envelope, measure
from paired_pose_eval import library
import paired_rails as rails
from shared_brush import measured_contact, optical
import stroke_distribution_eval as distribution
from timed_deposition_eval import native_comparison
import transition_release as release
import transition_release_eval as replay
from visible_transition_eval import OUT as CLOCK_RESULTS
from whole_ink import PatchBatch


OUT = fine.ROOT/"results"/"paired-rails"
ROUNDS = {"adapt": 60, "local": 84, "paired": 84}
READOUTS = ("grey_l1", "outline_distance", "ink_distance", "white_fraction", "along_run_error",
            "fourier_brush_relative_grid", "same_grid_frequency", "marked_energy")
SHAPE = ("eccentricity", "mean_density", "density_p90", "white_fraction", "roughness_by_scale_and_edge")


@dataclass(frozen=True)
class Practice:
    name: str
    ink: np.ndarray
    signed: bool


def examples():
    sources, _ = library()
    groups = {}
    for nominal in rails.WIDTHS:
        for course, marks in sources.items():
            admitted = []
            for mark in marks:
                image, contact, _, signed = mark.supervision()
                if contact.settled_width >= nominal:
                    admitted.append(Practice(mark.name, fine.sampling(image, contact.settled_width, nominal=nominal),
                                               signed))
            groups.setdefault(course, {})[nominal] = admitted
    return groups


class Batches:
    def __init__(self, groups):
        self.random = np.random.default_rng(9217)
        self.windows = {width: {} for width in rails.WIDTHS}
        for course_index, (source, scales) in enumerate(groups.items()):
            for width, marks in scales.items():
                if not marks:
                    continue
                prepared = []
                for mark_index, mark in enumerate(marks):
                    phase = layout.stroke_phase(torch.from_numpy(mark.ink)[None, None]).numpy()
                    patches = PatchBatch(mark.ink[None], mark.ink[None], 5237+course_index*37+mark_index,
                                         phase, patch=detail.PATCH)
                    prepared.append((patches, mark.signed))
                self.windows[width][source] = prepared

    def sample(self, width):
        bank = self.windows[width]
        sources = self.random.choice(list(bank), size=min(3, len(bank)), replace=False)
        ink, phases = [], []
        for source in sources:
            patches, signed = bank[source][self.random.integers(len(bank[source]))]
            actual, _, phase = patches.sample(1)
            if not signed and self.random.integers(2):
                actual, phase = actual.flip(-2), phase.flip(-2)
            ink.append(actual)
            phases.append(phase)
        return torch.cat(ink), torch.cat(phases)


def load(held, stage):
    return detail.load(OUT/f"{stage}-without-{held}.pt")


def learn(groups, held, *, reuse=False):
    groups = {source: marks for source, marks in groups.items() if source not in (held, "xingjie")}
    source_marks = {source: {str(width): [mark.name for mark in marks] for width, marks in scales.items()}
                    for source, scales in groups.items()}
    model, init = detail.load(fine.OUT/f"joint-without-{held}.pt")
    if set(init["training"]) != set(groups):
        raise AssertionError("The warm start must have exactly the declared training courses")
    common = model.state_dict()
    counts = {}
    for stage, steps in ROUNDS.items():
        path = OUT/f"{stage}-without-{held}.pt"
        if reuse and path.is_file():
            trained, state = load(held, stage)
            if state["sources"] != source_marks or len(state["loss"]) != steps:
                raise ValueError("A reused checkpoint must match the current source groups and training schedule")
            if stage == "adapt":
                common = trained.state_dict()
            counts[stage] = steps
            continue
        model = detail.TextureVelocity()
        model.load_state_dict(common)
        samples = Batches(groups)
        rng = torch.Generator().manual_seed(2192 if stage == "adapt" else 4751)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0012 if stage == "adapt" else .0008)
        history = []
        for step in range(steps):
            rows = rails.BAND_ROWS[step%len(rails.BAND_ROWS)]
            width = rails.WIDTHS[(step//len(rails.BAND_ROWS))%len(rails.WIDTHS)]
            ink, phase = samples.sample(width)
            loss = rails.objective(model, ink, phase, (rows, 1), rng, structure=stage != "adapt",
                                   paired=stage == "paired")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("The learned update must remain finite")
            optimizer.step()
            history.append(float(loss.detach()))
        checkpoint = dict(weights=model.state_dict(), direction=True, mode=stage, loss=history,
                          training=list(groups), sources=source_marks, band_rows=list(rails.BAND_ROWS),
                          train_widths=list(rails.WIDTHS), warm_start="joint-patch")
        torch.save(checkpoint, path)
        if stage == "adapt":
            common = model.state_dict()
        counts[stage] = len(history)
        print(held, stage, len(history), round(median(history[:10]), 5), round(median(history[-10:]), 5), flush=True)
    return counts


def optical_budget(actual, candidate, photo_plan):
    actual_dose = optical(actual).sum(axis=0, dtype=float)
    drawn_dose = optical(candidate).sum(axis=0, dtype=float)
    reference = photo_plan[0, 0].sum(dim=-2).numpy()
    return dict(relative_column_error=float(np.abs(drawn_dose-actual_dose).sum()/actual_dose.sum()),
                change_from_offered_max_column=float(np.abs(drawn_dose-reference).max()))


def rail_reads(field, contact, *, strict=True):
    tracks = channels.observe(field, contact, strict=strict)
    signed = [float(np.median(row.centers-contact.center[row.columns])/contact.settled_width) for row in tracks]
    by_side = {"upper": [row for row, key in zip(tracks, signed, strict=True) if key < -.12],
               "lower": [row for row, key in zip(tracks, signed, strict=True) if key > .12],
               "interior": [row for row, key in zip(tracks, signed, strict=True) if abs(key) <= .12]}
    return {key: dict(count=len(group), longest_px=int(max((row.length for row in group), default=0)),
                      channel_area=int(sum(np.diff(row.bands[:, 1], axis=1).sum() for row in group)),
                      mean_width_px=float(np.concatenate([np.diff(row.bands[:, 1], axis=1) for row in group]).mean())
                      if group else None)
            for key, group in by_side.items()}


def capacity(actual, plan, span):
    """Use the observed lane as an oracle for the feasibility of fixed optical budgets."""
    contact = measured_contact(actual[:, span])
    original = plan[0, 0].numpy().astype(float)
    diagnostics = {}
    configurations = [(4, 4, 0)]+[(rows, 1, origin) for rows in rails.BAND_ROWS for origin in (0, rows//2)]
    for strict in (True, False):
        records = channels.observe(actual[:, span], contact, strict=strict)
        lower = [row for row in records if np.median(row.centers-contact.center[row.columns]) > .12*contact.settled_width]
        track = max(lower, key=lambda row: row.length)
        required = np.zeros_like(original)
        allowed = np.full_like(original, rails.CEILING)
        columns = track.columns+span.start
        for x, (inner, paper, outer) in zip(columns, track.bands, strict=True):
            required[slice(*inner), x] = -np.log1p(-channels.INK)
            required[slice(*outer), x] = -np.log1p(-channels.INK)
            allowed[slice(*paper), x] = -np.log1p(-channels.PAPER)
        entries = {}
        for rows, cols, origin in configurations:
            def pooled(value, fill):
                padded = np.pad(value, ((origin, (-len(value)-origin)%rows), (0, (-value.shape[1])%cols)),
                                constant_values=fill)
                return padded.reshape(len(padded)//rows, rows, padded.shape[1]//cols, cols).sum(axis=(1, 3))

            allocated = pooled(original, 0.)
            feasible = (allocated >= pooled(required, 0.)-1e-7)&(allocated <= pooled(allowed, rails.CEILING)+1e-7)
            compatible = []
            for x, (inner, _, outer) in zip(columns, track.bands, strict=True):
                start = (inner[0]+origin)//rows
                stop = (outer[-1]-1+origin)//rows+1
                compatible.append(feasible[start:stop, x//cols].all())
            entries[f"{rows}x{cols}+{origin}"] = int(sum(compatible))
        diagnostics["strict" if strict else "linked"] = dict(span=[int(columns[0]), int(columns[-1]+1)],
            observed_columns=len(columns), feasible_columns=entries)
    return diagnostics


def filmed_sweep():
    trace = json.loads(replay.OBSERVATIONS.read_text(encoding="utf-8"))["xingjie-side"]
    with np.load(replay.POINTS, allow_pickle=False) as footprint:
        baseline, actual, cross, registration, _ = replay.registered_xingjie(footprint, trace)
        stroke = baseline.arrival
        record = stroke.stroke.coarse[2]
        track = release.stations(record, reservoir=True)
        weights = np.array(json.loads((CLOCK_RESULTS/"metrics.json").read_text(encoding="utf-8"))
                           ["given_event"]["profiles"]["0.15"]["weights"])
        amounts = release.press_schedule(track, weights[release.TRAVEL_BASIS:])
        swept = release.follow(stroke, track, amounts, cross)
    if not np.isclose(swept.final.sum(), baseline.accepted.sum(), rtol=0., atol=1e-6):
        raise AssertionError("The fixed film plan must repeat the previous full optical budget")
    return actual, swept, registration, weights


def filmed_guide():
    actual, swept, registration, weights = filmed_sweep()
    return (actual.astype(np.float32), torch.from_numpy(swept.final.astype(np.float32))[None, None],
            dict(arrival_s=swept.move.stroke.coarse[2]["arrival"], extra_dose_weights=weights.tolist(),
                 registration=registration))


def rendered(source, plan, *, verify_shift):
    init = detail.load(fine.OUT/f"joint-without-{source}.pt")
    data = json.loads((fine.OUT/"metrics.json").read_text(encoding="utf-8"))["held_out"][source]["texture_strength"]
    texture = TextureStrength(**{key: np.asarray(value) for key, value in data.items()})
    before = detail.ink_from_density(plan)[0, 0].numpy()
    local, local_data = load(source, "local")
    paired, paired_data = load(source, "paired")
    if any(source in state["training"] for state in (init[1], local_data, paired_data)):
        raise AssertionError("Every model must exclude the source of the compared paper")
    old = fine.draw({"joint": init}, before, texture)["adaptive"]
    draws = {"offered": [before], "previous": old, "local": rails.draw(local, plan, fine.SEEDS),
             "paired4": rails.draw(paired, plan, fine.SEEDS, rows=4),
             "paired8": rails.draw(paired, plan, fine.SEEDS)}
    if verify_shift:
        draws["paired8_shifted"] = rails.draw(paired, plan, fine.SEEDS, origin=4)
    return draws


def lower_share(field, center):
    dose = optical(field)
    rows = np.arange(len(field))[:, None]
    lower = dose*(rows > center)
    return [float(part.sum()/max(total.sum(), 1e-7))
            for part, total in zip(np.array_split(lower, 3, axis=1), np.array_split(dose, 3, axis=1), strict=True)]


def scores(actual, plan, rendered_, span=None, *, linked=False):
    if span is None:
        span = envelope(actual)[0]
    truth, power = distribution.features(actual), spectrum(actual)
    contact = measured_contact(actual[:, span])
    result = dict(reference={key: truth["profile"][key] for key in SHAPE},
                  reference_rails=rail_reads(actual[:, span], contact),
                  reference_lower_share=lower_share(actual[:, span], contact.center),
                  section=[int(span.start), int(span.stop)], section_width=float(contact.settled_width))
    if linked:
        result["reference_linked_rails"] = rail_reads(actual[:, span], contact, strict=False)
    outcomes = {}
    for name, candidates in rendered_.items():
        rows = []
        for picture in candidates:
            row = fine.compare(actual, picture, truth, power)
            row["optical"] = optical_budget(actual, picture, plan)
            profile = measure(picture)
            row["shape"] = {key: profile[key] for key in SHAPE}
            row["rails"] = rail_reads(picture[:, span], contact)
            row["lower_share"] = lower_share(picture[:, span], contact.center)
            if linked:
                row["linked_rails"] = rail_reads(picture[:, span], contact, strict=False)
            rows.append(row)
        outcomes[name] = rows
    return dict(result, drawings=outcomes)


def paired_panel(actual, fields, output, source_label):
    native_comparison([(source_label, actual), ("基线 · 固定局部墨量", fields["previous"][0]),
                       ("横移 · 通用纹理", fields["local"][0]),
                       ("横移 · 白道和墨带", fields["paired8"][0])], output)


def controlled_guide(actual, seed):
    tensor = torch.from_numpy(actual)[None, None]
    height = len(actual)
    dose = F.pad(detail.density(tensor), (0, 0, 0, (-height)%8))
    offered = layout.uncertain_contact(dose, (8, 1), torch.Generator().manual_seed(seed))
    return offered[..., :height, :]


def observed_cross_guide(actual, plan, width):
    """Diagnostic only: expose the photographed low-frequency shape but no fine raster."""
    broad = gaussian_filter(optical(actual), (.095*width, .21*width), mode="constant")
    motion = plan.sum(dim=-2, keepdim=True)
    flow = torch.from_numpy(broad)[None, None]
    support = flow > torch.maximum(.006*flow.amax(dim=-2, keepdim=True), flow.new_tensor(.002))
    flow = flow*support
    total = flow.sum(dim=-2, keepdim=True)
    proposal = flow*(motion/total.clamp_min(1e-7))
    proposal = torch.where(support, proposal, -2.*rails.CEILING)
    region = (len(actual), 1)
    restored = layout.within_regions(proposal, motion/region[0], region)
    feasible = support.sum(dim=-2, keepdim=True)*rails.CEILING >= motion
    return torch.where(feasible, restored, plan)


def medians(drawings):
    summary = {}
    for name, rows in drawings.items():
        summary[name] = {}
        for key in READOUTS:
            values = [row[key] for row in rows if row[key] is not None]
            summary[name][key] = median(values) if values else None
        summary[name]["longest_lower_px"] = [row["rails"]["lower"]["longest_px"] for row in rows]
        if "linked_rails" in rows[0]:
            summary[name]["longest_lower_linked_px"] = [row["linked_rails"]["lower"]["longest_px"] for row in rows]
    return summary


def camera_diagnostics(actual, plan, span, width, ordinary):
    fitted = load("xingjie", "paired")[0]
    observed_shape = observed_cross_guide(actual, plan, width)
    controls = {"reversed_tip": rails.draw(fitted, plan, fine.SEEDS, tip_side=-1),
                "provided_cross_shape": rails.draw(fitted, observed_shape, fine.SEEDS)}
    native_comparison([("兴杰实拍 · 整课留出", actual), ("预测接纸和笔尖方向", ordinary),
                       ("仅反转笔尖方向", controls["reversed_tip"][0]),
                       ("给成迹的低频横截面", controls["provided_cross_shape"][0])], OUT/"cross-support.png")
    return scores(actual, plan, controls, span, linked=True)["drawings"]


def textbook_diagnostics(actual, plan, width):
    lower_frequency = observed_cross_guide(actual, plan, width)
    fields = rendered("tup", lower_frequency, verify_shift=False)
    paired_panel(actual, fields, OUT/"textbook-coarse.png", "清华原嵌图 · 原像素")
    return scores(actual, lower_frequency, fields)


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def assess(groups):
    actual, plan, film_meta = filmed_guide()
    fields = rendered("xingjie", plan, verify_shift=True)
    span = slice(213, 444)
    camera = scores(actual, plan, fields, span, linked=True)
    camera["capacity"] = capacity(actual, plan, span)
    camera["film"] = film_meta
    paired_panel(actual, fields, OUT/"xingjie.png", "兴杰实拍 · 整课留出")
    # The crop is registered to the same finished stroke for all four panels.
    _, _, lower, _ = envelope(actual[:, span])
    edge = int(round(float(np.median(lower))))
    view = np.s_[edge-51:edge+19, 263:453]
    paired_panel(actual[view], {k: [v[0][view]] for k, v in fields.items()}, OUT/"xingjie-detail.png", "兴杰下缘原像素")
    camera["diagnostic_controls"] = camera_diagnostics(actual, plan, span, camera["section_width"], fields["paired8"][0])
    print("xingjie", medians(camera["drawings"]), flush=True)
    print("diagnostic", medians(camera["diagnostic_controls"]), flush=True)

    actual, width, crop = native_textbook_crop()
    plan = controlled_guide(actual, 1172)
    fields = rendered("tup", plan, verify_shift=True)
    textbook = scores(actual, plan, fields, linked=True)
    textbook["crop_xywh"] = crop
    textbook["source_brush_width_px"] = width
    paired_panel(actual, fields, OUT/"textbook.png", "清华教材原嵌图 · 整课留出")
    textbook["lower_frequency_control"] = textbook_diagnostics(actual, plan, width)
    print("native textbook", {key: row["same_grid_frequency"]
                              for key, row in medians(textbook["lower_frequency_control"]["drawings"]).items()}, flush=True)

    normalized = {}
    for course, scales in groups.items():
        cases = {}
        for mark in scales[rails.WIDTHS[0]]:
            plan = controlled_guide(mark.ink, 7234)
            fields = rendered(course, plan, verify_shift=False)
            cases[mark.name] = scores(mark.ink, plan, fields)
        normalized[course] = cases
        print(course, len(cases), flush=True)
    natural = fine.statistic_range({name: scales[rails.WIDTHS[0]] for name, scales in groups.items()})
    summary = {}
    for course, baseline in natural.items():
        method_counts = {}
        for case in normalized[course].values():
            for method, values in medians(case["drawings"]).items():
                accepted = all(values[key] <= threshold for key, threshold in baseline["quantile_90"].items())
                method_counts[method] = method_counts.get(method, 0)+accepted
        summary[course] = dict(within_all=method_counts, tested=len(normalized[course]))
    data = dict(rounds=ROUNDS, seeds=list(fine.SEEDS), width_scales=rails.WIDTHS, camera=camera, textbook=textbook,
                normalized=normalized, natural_variation=natural, within_course_radius=summary)
    (OUT/"metrics.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=json_scalar), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("courses", nargs="*")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--assess", action="store_true", help="Run the native and normalized comparisons")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = examples()
    if args.assess:
        assess(groups)
        return
    for held in args.courses or groups:
        if held not in groups:
            parser.error(f"Unknown course: {held}")
        learn(groups, held, reuse=args.reuse)


if __name__ == "__main__":
    main()
