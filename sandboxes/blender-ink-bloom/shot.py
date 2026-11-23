"""Authoring data for one shot. Image field axes are U right, V up."""

from dataclasses import dataclass
from math import pi, sin


WIDTH = 960
HEIGHT = 540
FPS = 24
FRAME_COUNT = 204
FIELD_WIDTH = 416
FIELD_HEIGHT = 256

PAPER = (244, 240, 228)
INK = (28, 37, 34)
DROP_UV = (0.40, 0.688)

RIDGE = (
    (.075, .245), (.124, .312), (.177, .365), (.224, .379), (.252, .433),
    (.278, .462), (.302, .541), (.341, .578), (.365, .647), (.408, .716),
    (.429, .722), (.450, .698), (.490, .686), (.501, .648), (.549, .619),
    (.585, .627), (.615, .583), (.638, .559), (.678, .536), (.696, .497),
    (.738, .464), (.769, .461), (.789, .424), (.832, .403), (.855, .369),
    (.888, .305), (.937, .248),
)


@dataclass(frozen=True)
class Stroke:
    label: str
    start: float
    stop: float
    family: str
    points: tuple
    width: float
    load: float
    water: float
    split: bool = False
    dry_catch: float = 0.0


STROKES = (
    Stroke("湿染", .84, 2.76, "axe",
           (DROP_UV, (.442, .663), (.454, .608), (.479, .563), (.524, .541),
            (.540, .475), (.575, .461), (.607, .401), (.682, .370), (.759, .300)),
           width=10.4, load=13.0, water=10.0),
    Stroke("勾线", 2.92, 4.80, "turn", RIDGE,
           width=1.19, load=2.94, water=.28, dry_catch=.61),
    Stroke("侧锋", 4.96, 6.11, "axe",
           ((.458, .641), (.409, .562), (.431, .511), (.398, .482),
            (.442, .43), (.462, .407), (.481, .349), (.535, .303)),
           width=3.95, load=5.1, water=.36, split=True, dry_catch=.52),
    Stroke("余笔", 6.20, 6.98, "axe",
           ((.614, .501), (.606, .468), (.646, .440), (.664, .412), (.711, .379), (.744, .355)),
           width=2.65, load=3.2, water=.28, split=True, dry_catch=.51),
)

MARKERS = (
    (1, "留白"),
    (9, "墨滴"),
    (21, "接到坡面"),
    (71, "勾线"),
    (120, "侧锋"),
    (170, "停笔 · 横移看景"),
)


def slope_point(u, v):
    """Curved, orientable paper carrier in Blender XYZ (Z up)."""
    x = (u - .5) * 10.10 + .30
    y = 2.50 + (v - .5) * 1.36
    y -= .68 * sin(pi * u) * sin(pi * v)
    y += .19 * sin(2 * pi * u + .3) * sin(pi * v)
    z = .16 + v * 5.53
    return x, y, z
