"""Test transferable optical dose on the filmed dry side-brush stroke."""

import json

import numpy as np
import torch

import ink_volume as volume
import paired_brush_eval as paired
from ink_volume_eval import OUT, Sample, assess, broad_sources, training
from lane_transport import SectionGuide
from lane_transport_eval import sources, take
from shared_brush import optical


def simulate(data, demand, amount, hold, span):
    length = data.supported.support.shape[1]
    deposits, charges = [], []
    for white in data.white:
        capacity = (data.supported.support & ~white).sum(dim=0).numpy() * paired.uptake.CEILING
        intended = np.zeros(length)
        intended[span] = demand
        fraction = np.divide(intended, capacity, out=np.zeros(length), where=capacity > 0.).clip(0., .997)
        offer = -capacity * np.log1p(-fraction)
        laid, account = paired.uptake.deposit(data.supported, white[None], stock=amount,
                                               demand=offer, hold=hold, saturating=True)
        deposits.append(laid[0])
        charges.append(account)
    ink = torch.stack(deposits)
    if torch.count_nonzero(ink[data.white | ~data.supported.support[None]]):
        raise ValueError("Ink has entered a non-contact pixel or filled a flying-white gap")
    return ink, charges


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    bank = {key: [Sample(*take(mark), (1, -1))] for key, mark in sources().items()}
    donors = training(bank, broad_sources(), paired.SOURCE)
    geometry = volume.fit(donors, finite=False)
    reservoir = volume.fit(donors, finite=True)
    profile = volume.fit_profile(donors, geometry, kind="arc")
    conditional = volume.fit_opening_change(donors, conditional=True)
    np.savez_compressed(OUT / "shared-finite-without-xingjie.npz", **reservoir)
    np.savez_compressed(OUT / "arc-without-xingjie.npz", **profile)
    np.savez_compressed(OUT / "opening-change-without-xingjie.npz", **conditional)
    data = paired.inputs()
    actual = data.case["image"]
    moving, account = paired.uptake.deposit(data.supported, data.white, demand=data.travel, hold=data.dwell,
                                            saturating=True)
    coarse, support, span = paired.forward.from_plan(data.supported, account["by_column"][0])
    upper = coarse.center - .5 * (coarse.width - 1.)
    lower = coarse.center + .5 * (coarse.width - 1.)
    frame = SectionGuide(upper, lower, support, coarse.settled_width)
    opening = data.case["opening"][:, span]
    cut = paired.uptake.CUTOFF - span.start
    whole = {"old": paired.old_texture(data, moving)}
    optical_volume = {"old": np.median(account["by_column"], axis=0)}
    variants = (("geometry", geometry, False, True),
                ("finite", reservoir, True, False),
                ("finite_pause", reservoir, True, True))
    details = {}
    for name, model, adaptive, has_hold in variants:
        dose, state = volume.forecast(model, frame, opening, cut, adaptive=adaptive)
        available = state["initial"] if state["initial"] is not None else float(account["initial"][0])
        solved, history = simulate(data, dose, available, data.dwell if has_hold else None, span)
        whole[name] = paired.old_texture(data, solved)
        optical_volume[name] = np.median(np.concatenate([row["by_column"] for row in history]), axis=0)
        details[name] = dict(incoming=available, rate=state["rate"], remaining=[float(row["remaining"][0]) for row in history],
                             realized_pause=[float(row["pause_applied"][0].sum()) for row in history])
    target, coefficients = volume.predict_profile(profile, frame, opening, cut)
    painted, history = simulate(data, target, float(account["initial"][0]), data.dwell, span)
    whole["profile_pause"] = paired.old_texture(data, painted)
    optical_volume["profile_pause"] = np.median(np.concatenate([row["by_column"] for row in history]), axis=0)
    details["profile_pause"] = {"coefficients": coefficients.tolist()}
    target = volume.predict_opening_change(conditional, frame, opening, cut)
    painted, history = simulate(data, target, float(account["initial"][0]), data.dwell, span)
    whole["opening_pause"] = paired.old_texture(data, painted)
    optical_volume["opening_pause"] = np.median(np.concatenate([row["by_column"] for row in history]), axis=0)
    names = {"old": "原来由动作预测", "geometry": "学接纸与落墨量，加实拍停驻",
             "finite": "学习有限墨量", "finite_pause": "学习有限墨量，加实拍停驻",
             "profile_pause": "学沿程墨深，加实拍停驻", "opening_pause": "学起段变化，加实拍停驻"}
    paired.board("兴杰干侧锋的整课留出", "绿线右边评分。保持同一接纸范围与飞白，比较落墨总量。",
                 [("实写", actual)] + [(names[k], values[0]) for k, values in whole.items()], OUT / "filmed.png")
    direct = paired.measures(actual, whole, side=data.case["side"])
    amount = {label: assess(optical(actual).sum(axis=0), estimate, paired.prior.FIRST_SCORE)
              for label, estimate in optical_volume.items()}
    report = dict(trained_courses=reservoir["training"].tolist(), cutoff=paired.uptake.CUTOFF,
                  scoring_from=paired.prior.FIRST_SCORE, amounts=amount, appearance=direct, estimated_loads=details)
    (OUT / "filmed.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Recorded dry side brush", {k: (round(amount[k]["relative_l1"], 4),
          round(v["native_suffix"]["relative_opacity_l1"], 4), round(v["native_suffix"]["f1"], 4)) for k, v in direct.items()})


if __name__ == "__main__":
    main()
