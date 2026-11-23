"""Build and render the ink-bloom shot with Blender. Run in background mode."""

from pathlib import Path
import argparse
import json
import math
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dry_shot
import shot
from shot import FPS, FRAME_COUNT, HEIGHT, PAPER, WIDTH, slope_point


def linear_color(rgb):
    return tuple((c / 255 + .055) ** 2.4 / 1.055 ** 2.4 if c > 10 else c / 3294.6
                 for c in rgb) + (1.0,)


def new_material(name, method=None):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    if method:
        material.surface_render_method = method
    graph = material.node_tree
    graph.nodes.clear()
    return material, graph, graph.nodes.new("ShaderNodeOutputMaterial")


def emission(graph, color):
    node = graph.nodes.new("ShaderNodeEmission")
    if isinstance(color, tuple):
        node.inputs["Color"].default_value = color
    else:
        graph.links.new(color, node.inputs["Color"])
    return node.outputs[0]


def blend_shader(graph, factor, background, foreground, label):
    node = graph.nodes.new("ShaderNodeMixShader")
    node.label = label
    for src, socket in zip((factor, background, foreground), node.inputs):
        graph.links.new(src, socket)
    return node.outputs[0]


def scalar(graph, operation, x, y, label=""):
    node = graph.nodes.new("ShaderNodeMath")
    node.operation = operation
    node.label = label
    for value, socket in zip((x, y), node.inputs):
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            graph.links.new(value, socket)
    return node.outputs[0]


def grid(collection, name, point, nx, ny, material):
    uv = [(ix / nx, iy / ny) for iy in range(ny + 1) for ix in range(nx + 1)]
    width = nx + 1
    faces = [(p, p+1, p+width+1, p+width)
             for row in range(ny) for p in range(row*width, row*width + nx)]
    mesh = bpy.data.meshes.new(name)
    center = Vector(point(.5, .5))
    mesh.from_pydata([Vector(point(u, v)) - center for u, v in uv], [], faces)
    mesh.update()
    layer = mesh.uv_layers.new(name="Paint UV")
    for loop in mesh.loops:
        layer.data[loop.index].uv = uv[loop.vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = center
    obj.data.materials.append(material)
    return obj, uv


def paper_backdrop(collection):
    material, graph, output = new_material("宣纸 · 微弱的纤维明暗")
    nodes, links = graph.nodes, graph.links
    coordinates = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 465
    noise.inputs["Detail"].default_value = 1.8
    links.new(coordinates.outputs["UV"], noise.inputs["Vector"])
    color = nodes.new("ShaderNodeMixRGB")
    color.inputs[1].default_value = linear_color((240, 235, 222))
    color.inputs[2].default_value = linear_color((248, 245, 234))
    links.new(noise.outputs["Fac"], color.inputs[0])
    links.new(emission(graph, color.outputs[0]), output.inputs["Surface"])
    grid(collection, "宣纸底", lambda u, v: ((u-.5)*63, 40, (v-.5)*37 + 3), 1, 1, material)


def landscape_plate(collection, asset, label, center, dimensions, opacity, bend):
    material, graph, output = new_material(label, "BLENDED")
    tex = graph.nodes.new("ShaderNodeTexImage")
    tex.extension = "CLIP"
    tex.image = bpy.data.images.load(str(ROOT / "assets" / f"{asset}.png"))
    tex.image.pack()
    alpha = scalar(graph, "MULTIPLY", tex.outputs["Alpha"], opacity, "景片浓淡")
    transparent = graph.nodes.new("ShaderNodeBsdfTransparent").outputs[0]
    fg = emission(graph, tex.outputs["Color"])
    graph.links.new(blend_shader(graph, alpha, transparent, fg, "景片与纸底"), output.inputs["Surface"])

    def position(u, v):
        x, y, z = center
        width, height = dimensions
        depth = y + bend * math.sin(math.pi * u) * math.sin(math.pi * v)
        return x + (u - .5) * width, depth + .39 * (v - .5), z + (v - .5) * height

    grid(collection, label, position, 18, 10, material)


def paint_material(directory, ink_color):
    material, graph, output = new_material("活墨 · 墨迹 / 内部留白", "BLENDED")
    nodes, links = graph.nodes, graph.links
    fields = nodes.new("ShaderNodeTexImage")
    fields.label = "R 墨迹 A / G 内部覆盖 C / B 湿度"
    fields.extension = "CLIP"
    fields.image = bpy.data.images.load(str(directory / "fields" / "ink-0001.png"))
    fields.image.source = "SEQUENCE"
    fields.image.colorspace_settings.name = "Non-Color"
    fields.image_user.frame_duration = FRAME_COUNT
    fields.image_user.frame_start = 1
    fields.image_user.use_auto_refresh = True
    channels = nodes.new("ShaderNodeSeparateColor")
    links.new(fields.outputs["Color"], channels.inputs[0])

    transparent = nodes.new("ShaderNodeBsdfTransparent").outputs[0]
    paper_shader = emission(graph, linear_color(PAPER))
    ink_shader = emission(graph, linear_color(ink_color))
    interior = blend_shader(graph, channels.outputs["Green"], transparent, paper_shader,
                            "C · 留白遮住后景")
    composite = blend_shader(graph, channels.outputs["Red"], interior, ink_shader,
                             "A · 叠墨，保留渗到轮廓外的边缘")
    links.new(composite, output.inputs["Surface"])

    for node, pos in zip(graph.nodes, (
        (570, 110), (-570, 250), (-300, 250), (-300, -30), (-300, -165),
        (0, -185), (0, 65), (290, 110),
    )):
        node.location = pos
    return material, fields.image


def airy_ink(collection, drop_uv, dry):
    """An animatable, bounded 3D absorbing density. All noise is object-local."""
    material, graph, output = new_material("入墨 · 空间吸收")
    nodes, links = graph.nodes, graph.links
    coordinates = nodes.new("ShaderNodeTexCoord").outputs["Generated"]

    def vector(operation, x, y=None):
        node = nodes.new("ShaderNodeVectorMath")
        node.operation = operation
        for value, socket in zip((x, y), node.inputs):
            if value is None:
                continue
            if isinstance(value, tuple):
                socket.default_value = value
            else:
                links.new(value, socket)
        return node.outputs["Value" if operation == "LENGTH" else "Vector"]

    def noise(scale, detail):
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = detail
        tex.inputs["Roughness"].default_value = .70
        links.new(coordinates, tex.inputs["Vector"])
        return tex

    def remap(src, lo, hi, end0, end1):
        node = nodes.new("ShaderNodeMapRange")
        node.interpolation_type = "SMOOTHERSTEP"
        node.clamp = True
        for socket, value in zip(node.inputs[1:5], (lo, hi, end0, end1)):
            socket.default_value = value
        links.new(src, node.inputs["Value"])
        return node.outputs[0]

    center = vector("SUBTRACT", coordinates, (.5, .52, .49))
    center = vector("MULTIPLY", center, (2.35, 2.18, 2.18))
    curl = vector("SUBTRACT", noise(2.65, 3.0).outputs["Color"], (.5, .5, .5))
    curl = vector("MULTIPLY", curl, (1.45, 1.48, 1.58))
    radius = vector("LENGTH", vector("ADD", center, curl))
    body = remap(radius, .12, 1.02, 1.0, 0.0)
    threads = remap(noise(6.8, 2.7).outputs["Fac"], .37, .69, .004, 1.0)
    strength = nodes.new("ShaderNodeValue")
    strength.label = "空气中的墨 · 浓度随扩散减弱"
    turbidity = scalar(graph, "MULTIPLY", body, threads)
    density = scalar(graph, "MULTIPLY", turbidity, strength.outputs[0])
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = linear_color((84, 104, 97))
    links.new(density, absorption.inputs["Density"])
    links.new(absorption.outputs[0], output.inputs["Volume"])
    for i, node in enumerate(nodes):
        node.location = (180 * (i // 4), -165 * (i % 4))

    vertices = [(x, y, z) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
    faces = [(2, 3, 1, 0), (4, 5, 7, 6), (0, 1, 5, 4), (1, 3, 7, 5), (2, 6, 7, 3), (0, 4, 6, 2)]
    mesh = bpy.data.meshes.new("局部墨晕边界")
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new("空中墨晕", mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)

    origin = Vector(slope_point(*drop_uv))
    wet_keys = (
        (1, (0, -.82, .83), (.051, .056, .068), 0),
        (9, (-.035, -.81, .80), (.052, .059, .072), 0),
        (12, (-.018, -.78, .73), (.081, .094, .102), 94),
        (17, (.014, -.66, .53), (.21, .24, .30), 48),
        (23, (.064, -.40, .32), (.46, .38, .55), 26),
        (31, (.16, -.23, .15), (.75, .47, .57), 11.4),
        (40, (.25, -.08, .08), (.92, .52, .51), 3.8),
        (55, (.40, -.02, .05), (1.04, .58, .41), 0),
    )
    dry_keys = (
        (1, (0, -.58, .58), (.044, .050, .058), 0),
        (9, (0, -.57, .57), (.044, .050, .058), 0),
        (12, (-.012, -.49, .49), (.075, .081, .100), 67),
        (17, (.018, -.26, .27), (.13, .14, .20), 34),
        (21, (.021, -.09, .10), (.20, .19, .19), 17),
        (26, (.032, -.02, .025), (.26, .17, .16), 4.2),
        (33, (.067, -.02, .015), (.32, .17, .12), 0),
    )
    keys = dry_keys if dry else wet_keys
    for frame, offset, scale, amount in keys:
        obj.location = origin + Vector(offset)
        obj.scale = scale
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("scale", frame=frame)
        strength.outputs[0].default_value = amount
        strength.outputs[0].keyframe_insert("default_value", frame=frame)
    for frame, hidden in ((1, True), (9, False), (keys[-1][0], True)):
        obj.hide_render = hidden
        obj.keyframe_insert("hide_render", frame=frame)
    return obj


def camera_at(collection, name, points):
    data = bpy.data.cameras.new(name)
    data.lens = 40
    data.sensor_width = 36
    data.clip_end = 90
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    for frame, loc, target in points:
        obj.location = loc
        obj.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_euler", frame=frame)
    return obj


def create_shot(directory, config):
    scene = bpy.context.scene
    if not bpy.app.background or bpy.data.filepath:
        raise RuntimeError("Build in a separate Blender background process with factory startup.")
    dry = config is dry_shot
    scene.name = "墨落成山 · 干笔皴岩" if dry else "墨落成山"
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    stages = bpy.data.collections.new("景片")
    active = bpy.data.collections.new("落墨实验")
    cameras = bpy.data.collections.new("机位")
    for collection in (stages, active, cameras):
        scene.collection.children.link(collection)
    paper_backdrop(stages)
    landscape_plate(stages, "far-left", "远山 · 左", (-5.2, 13.5, 3.64), (17.7, 4.65), .80, -.49)
    landscape_plate(stages, "far-right", "远山 · 右", (5.45, 10.3, 3.77), (13.40, 5.08), .78, -.65)
    landscape_plate(stages, "near-bank", "近岸 · 右", (7.60, -.65, .67), (8.45, 2.80), .60, -.38)

    material, sequence = paint_material(directory, config.INK)
    surface_name = "中景 · 干纸皴岩" if dry else "中景 · 可渗化坡面"
    carrier, uv = grid(active, surface_name, slope_point, 32, 20, material)
    airy_ink(active, config.DROP_UV, dry)
    reference = (-.66, -13.9, 3.39)
    target = (-.14, 2.25, 2.66)
    camera = camera_at(cameras, "主镜头 · 停笔后横移", [
        (1, reference, target), (9, reference, target),
        (58, (.03, -13.9, 3.37), target), (155, (.03, -13.9, 3.37), target),
        (204, (2.49, -13.65, 3.16), (.13, 2.25, 2.67)),
    ])
    scene.camera = camera
    scene.world.color = (.35, .35, .35)
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 24
    scene.eevee.volumetric_samples = 48
    scene.eevee.volumetric_tile_size = "4"
    scene.eevee.use_volume_custom_range = True
    scene.eevee.volumetric_start = 11.5
    scene.eevee.volumetric_end = 19.5
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT
    scene.view_settings.view_transform = "Standard"
    for frame, name in config.MARKERS:
        scene.timeline_markers.new(name, frame=frame)
    render = scene.render
    render.image_settings.file_format = "PNG"
    render.filepath = "//.build/frame-"

    sequence.filepath = "//fields/ink-0001.png"
    scene.frame_set(1)
    bpy.context.view_layer.update()
    export_carrier(scene, carrier, uv, directory, config)
    scene.frame_set(177 if dry else 165)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                view = area.spaces.active
                view.region_3d.view_perspective = "CAMERA"
                view.region_3d.view_camera_zoom = 5
                view.overlay.show_overlays = False
                view.shading.type = "RENDERED"

    bpy.context.preferences.filepaths.save_version = 0
    blend = directory / "ink-bloom.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True, relative_remap=False)
    print(f"Saved scene {scene.name}: {blend}", flush=True)


def export_carrier(scene, carrier, uv, directory, config):
    carrier.data.calc_loop_triangles()
    camera = scene.camera
    anchors = [Vector(slope_point(*config.DROP_UV)), Vector((5.45, 10.3, 3.77))]
    projected = [[round(v, 6) for v in world_to_camera_view(scene, camera, point)] for point in anchors]
    descriptor = {
        "version": "ink-bloom-02-dry" if config is dry_shot else "ink-bloom-01",
        "mountain_id": "sandbox-middle", "region_id": "painted-slope",
        "coordinates": "Blender XYZ, Z up, UV origin bottom-left",
        "uv_domain": [0, 0, 1, 1],
        "reference_camera": {
            "frame": 1, "matrix_world": [list(row) for row in camera.matrix_world],
            "lens_mm": camera.data.lens, "sensor_width_mm": camera.data.sensor_width,
        },
        "view_domain": {"camera": camera.name, "tested_path": [1, FRAME_COUNT]},
        "reference_anchors": {"world": [list(v) for v in anchors], "projected": projected},
        "proxy_mesh": {
            "xyz": [[round(v, 6) for v in carrier.matrix_world @ vertex.co]
                    for vertex in carrier.data.vertices],
            "uv": [[round(u, 6), round(v, 6)] for u, v in uv],
            "triangles": [list(triangle.vertices) for triangle in carrier.data.loop_triangles],
        },
        "sequence": {
            "path": "fields/ink-%04d.png", "fps": FPS, "frame_range": [1, FRAME_COUNT],
            "dimensions": [config.FIELD_WIDTH, config.FIELD_HEIGHT], "data_is_linear": True,
            "channels": {"R": "ink alpha A", "G": "interior coverage C", "B": "wetness"},
            "paper_color_srgb8": PAPER, "ink_color_srgb8": config.INK,
            "ink_rgba_premultiplied": "RGB = linear(ink_color_srgb8) * R; alpha = R",
            "interior_coverage": "G",
            "stroke_plan": [{"label": s.label, "family": s.family, "start_s": s.start,
                             "stop_s": s.stop, "points_uv": s.points} for s in config.STROKES],
        },
    }
    (directory / "carrier.json").write_text(json.dumps(descriptor, ensure_ascii=False), encoding="utf-8")


def render_stills(frames, build):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    for frame in frames:
        scene.frame_set(frame)
        scene.render.filepath = str(build / f"frame-{frame:04d}.png")
        bpy.ops.render.render(write_still=True)


def render_movie(directory):
    scene = bpy.context.scene
    scene.eevee.taa_render_samples = 16
    settings = scene.render.image_settings
    if hasattr(settings, "media_type"):
        settings.media_type = "VIDEO"
    settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = str(directory / "ink-bloom.mp4")
    scene.render.use_file_extension = False
    bpy.ops.render.render(animation=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--stills", type=int, nargs="+")
    parser.add_argument("--movie", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    directory = ROOT / "dry" if args.dry else ROOT
    config = dry_shot if args.dry else shot
    build = directory / ".build"
    build.mkdir(parents=True, exist_ok=True)
    if args.build:
        create_shot(directory, config)
    if args.stills:
        render_stills(args.stills, build)
    if args.movie:
        render_movie(directory)


if __name__ == "__main__":
    main()
