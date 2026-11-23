"""Hold out each lesson to test whether its labelled action transfers between courses."""

import json

import numpy as np
from PIL import Image

import cross_section_states as states
from cross_section_states_eval import OUT, SEEDS, aperture_spans, gathered, local_sheet, score
import double_loading as ink_sources
import shared_brush_eval as study


ACTIONS = {"aam-b": "regular", "aam-c": "regular", "aam-d": "regular", "aam-e": "fast-low",
           "gathered-ii": "regular", "gathered-iii": "regular", "gathered-iv": "regular", "gathered-v": "fast-low"}
PAIRS = {"aam": ("aam-d", "aam-e"), "gathered": ("gathered-iv", "gathered-v")}
THRESHOLDS = ((.035, .08), (.05, .13), (.075, .18))


def samples():
    bank = study.examples()
    bank["gathered"] = gathered()
    with Image.open(ink_sources.OUT / "gathered-original.jpg") as photo:
        for key in ("ii", "iii", "iv", "v"):
            demo = ink_sources.Mark.read(photo, ink_sources.GATHERED[key])
            bank["gathered"].append(study.example(f"gathered-{key}", key, "gathered", demo.photo[:, demo.span],
                                                  .5, None, "action explicit, posture and time direction unknown", True))
    return bank


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    bank = samples()
    stats, pictures = {}, {}
    for holdout, names in PAIRS.items():
        learn = {course: marks for course, marks in bank.items() if course != holdout}
        baseline = states.fit(learn, 6)
        trained = states.fit(learn, 6, actions=ACTIONS)
        if holdout in trained["sources"]:
            raise AssertionError("The scored course must be wholly held out")
        stats[holdout] = {"training": {k: len(v) for k, v in learn.items()}, "cases": {}}
        for mark in bank[holdout]:
            if mark.name not in names:
                continue
            action = ACTIONS[mark.name]
            inverse = "regular" if action == "fast-low" else "fast-low"
            target = study.targets(mark.ink)
            width = mark.contact.settled_width
            pictures[mark.name] = {"actual": mark.ink}
            trials = {}
            options = (("base", baseline, "unlabelled"), ("true", trained, action), ("swapped", trained, inverse))
            for name, model, label in options:
                values = []
                for i, seed in enumerate(SEEDS):
                    generated, _ = states.render(model, mark.contact, pose=mark.pose, seed=seed, action=label)
                    values.append(score(mark.ink, generated, target, width, signed=False))
                    if not i:
                        pictures[mark.name][name] = generated
                trials[name] = study.average(values)
                trials[name]["profile_nll"] = states.read_likelihood(model, mark.ink, mark.contact, mark.pose, action=label)
            stats[holdout]["cases"][mark.name] = dict(action=action, width_px=width, simulations=trials,
                                                      reference=[aperture_spans(mark.ink, width, *limits).tolist()
                                                                 for limits in THRESHOLDS])
        for name, model in (("without-action", baseline), ("with-action", trained)):
            np.savez_compressed(OUT / f"{name}-without-{holdout}.npz", **model)
    picked = (("aam-d", "馆方·变压立笔", .54), ("aam-e", "馆方·快行少水", .36),
              ("gathered-iv", "Gathered·变压中速", .47), ("gathered-v", "Gathered·轻压快行少墨", .52))
    columns = (("actual", "教学实写"), ("base", "只给笔宽和光学量"), ("true", "再给课程的动作标签"),
               ("swapped", "交换两类动作标签"))
    by_name = {name: row for split in stats.values() for name, row in split["cases"].items()}
    local_sheet(pictures, by_name, picked, columns, headline="快行低含水或低载墨的动作能否跨课迁移",
                destination=OUT / "action.png")
    evidence = dict(held_out=stats, seeds=list(SEEDS), thresholds=THRESHOLDS,
                    scope="Fast with little moisture and fast with little ink are the two joint teaching labels. "
                          "Pressure, speed and brush loading are not separately controlled.")
    (OUT / "action.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for name, row in by_name.items():
        print(name, row["reference"], [(key, round(a["separation_fraction"], 3),
                                       round(a["separation_longest_widths"], 3), round(a["optical_l1"], 3))
                                      for key, a in row["simulations"].items()])


if __name__ == "__main__":
    main()
