"""Prepare a separately filmed Chinese dry side-brush mark at its supported scale."""

import argparse
import json

import numpy as np
from PIL import Image
import torch

from contact_data import DEST, TRAIN_CONTACT, late_contact
from contact_model import fit
from sequence_eval import low_pass
from stroke_data import ROOT, SOURCES, image_file, orient, relative_reflectance


NAME = "shanyang-side"
SOURCE_URL = "https://www.bilibili.com/video/BV1yd4y197NE/?t=456"
CAPTURE = "shanyang-side-final-466.jpg"
MEDIUM = (192, 84)
PREPARED = ROOT / ".build" / "instructor-sample"
TRAINING = (*TRAIN_CONTACT, NAME)
# Coordinates on the captured 2x display raster. The polygon separates the bottom stroke.
BOUNDARY = [(108, 217), (147, 205), (186, 184), (217, 168), (248, 151), (280, 140), (308, 124),
            (342, 111), (373, 91), (410, 73), (446, 55), (478, 46), (493, 54), (490, 81),
            (449, 92), (410, 109), (375, 136), (342, 145), (302, 173), (265, 198), (219, 225),
            (184, 247), (108, 247)]


def prepare():
    PREPARED.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCES / CAPTURE) as frame:
        native = frame.resize((frame.width // 2, frame.height // 2), Image.Resampling.BOX)
        captured_size = frame.size
    polygon = [(x / 2, y / 2) for x, y in BOUNDARY]
    ink, exposure = relative_reflectance(native, polygon, dark_floor=.43)
    field, geometry = orient(ink, -25.2, size=MEDIUM, ink_width=162)
    guide = low_pass(field)
    with np.load(DEST / "fields.npz", allow_pickle=False) as old:
        np.savez_compressed(PREPARED / "fields.npz", **{id_: old[id_] for id_ in old.files}, **{NAME: guide})
    np.savez_compressed(PREPARED / "medium.npz", **{NAME: field})
    image_file(field, PREPARED / "reference.png")
    manifest = dict(source=SOURCE_URL, completed_sec=466.700503, observed_motion_secs=[460.22, 463.71],
                    technique="demonstrated side brush, dry separated tail, travel from bottom-left to top-right",
                    sheet_mark="lower of three, brush held in right hand", source_crop=CAPTURE,
                    video_decoded_px=[640, 360], full_display_video_px=[1280, 720],
                    displayed_window_px=[366, 244, *captured_size],
                    measured_window_px=list(native.size), measured_mm_per_px=None, rotation=-25.2,
                    medium_grid_px=list(MEDIUM), isolated_polygon_px=polygon, q=late_contact(guide),
                    **exposure, **geometry)
    (PREPARED / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "control", "expanded"))
    parser.add_argument("--steps", type=int, default=550)
    opts = parser.parse_args()
    if opts.mode == "prepare":
        prepare()
        return
    torch.set_num_threads(3)
    fit(PREPARED / opts.mode, opts.steps, resume=ROOT / "results" / "contact" / "contact.pt",
        with_contact=True, train_prior=True, dataset=PREPARED / "fields.npz",
        training_names=TRAINING if opts.mode == "expanded" else TRAIN_CONTACT, select_after=1)


if __name__ == "__main__":
    main()
