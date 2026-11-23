"""Extract the real before/after pair in Waseda's public Fig. 12 appendix."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader


OUT = Path(__file__).parent / "references"
RECT = (1003, 1320, 1868, 1858)


def extract(pub_pdf, output=OUT):
    page = PdfReader(pub_pdf).pages[4]
    chosen = {}
    for asset in page.images:
        key = asset.name.split(".", 1)[0]
        if key in ("Im35", "Im36", "Im37"):
            rgb = asset.image.convert("RGB")
            if rgb.size != (2800, 2800):
                raise ValueError("The reference should be the 2800 × 2800 published version")
            chosen[key] = rgb.crop(RECT)
    if len(chosen) != 3:
        raise ValueError("Figure 12 source images were not found")

    before, after = chosen["Im36"], chosen["Im37"]
    before_l, after_l = (np.asarray(image.convert("L"), dtype=float) for image in (before, after))
    support = np.asarray(chosen["Im35"].convert("L")) > 62
    diff = before_l - after_l
    blank = ~support & (before_l > 165) & (after_l > 165)
    offset = np.median(diff[blank])
    # A robust floor rejects small lighting changes in the unpainted paper.
    floor = max(0., float(np.percentile(diff[blank] - offset, 95)))
    delta = np.clip((diff - offset - floor) / np.maximum(40., before_l), 0., 1.)
    if np.median(diff[support]) <= floor * 3:
        raise ValueError("The stroke does not darken the expected area")

    output.mkdir(parents=True, exist_ok=True)
    before.save(output / "waseda-before.png", optimize=True)
    after.save(output / "waseda-after.png", optimize=True)
    Image.fromarray(np.rint(255 * (1 - delta)).astype("uint8")).save(
        output / "waseda-addition.png", optimize=True)
    info = dict(
        source="Madono and Simo-Serra, PG23 supplementary PDF, p. 5, Fig. 12",
        source_url="https://diglib.eg.org/items/194d937f-f7c2-4980-bea7-0cd771e103ec",
        source_pixels=[2800, 2800], crop_xyxy=RECT, measured_blank_level_shift=float(offset),
        blank_difference_floor=floor, median_darkening_in_stroke=float(np.median(diff[support])),
    )
    (output / "waseda-pair.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("appendix", type=Path)
    print(json.dumps(extract(parser.parse_args().appendix)))
