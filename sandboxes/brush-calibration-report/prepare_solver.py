"""Prepare an isolated copy of the original kernels for the stroke exercise."""

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build"
MODEL = BUILD / "model"


def prepare(zig):
    source = ROOT.parents[1] / "brush-lab"
    if not (source / "selected_v016.json").is_file():
        raise FileNotFoundError(source)
    target = MODEL / "inkscape_lab"
    shutil.copytree(source / "inkscape_lab", target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    shutil.copy2(source / "selected_v016.json", MODEL)
    if zig is None:
        return  # Linux uses the project's normal g++ lazy compiler.

    env = os.environ.copy()
    env["ZIG_GLOBAL_CACHE_DIR"] = str(BUILD / "zig-global-cache")
    env["ZIG_LOCAL_CACHE_DIR"] = str(BUILD / "zig-local-cache")
    kernels = {
        "meniscus_film": ("_meniscus_film", ("film_step",), False),
        "paper_transport": ("_paper_transport", ("paper_step",), True),
        "technique_paper": ("_technique_paper", ("paper_step",), True),
        "mountain_paper_fast": ("_mountain_paper", ("make_cache", "free_cache", "paper_step"), True),
    }
    for stem, (prefix, symbols, hashed) in kernels.items():
        cpp = target / f"{stem}.cpp"
        suffix = "_" + hashlib.sha256(cpp.read_bytes()).hexdigest()[:12] if hashed else ""
        lib = target / f"{prefix}{suffix}.so"
        if not lib.exists():
            print(f"Compiling {cpp.name}", flush=True)
            subprocess.run([str(zig), "c++", "-target", "x86_64-windows-gnu", "-O3", "-std=c++17",
                            "-shared", "-Wl,--export-all-symbols", str(cpp), "-o", str(lib)],
                           env=env, cwd=BUILD, check=True)
        dll = ctypes.CDLL(str(lib))
        for symbol in symbols:
            getattr(dll, symbol)
        print(f"Verified kernel {lib.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zig", type=Path, help="Absolute path to Zig for Windows")
    prepare(parser.parse_args().zig)
