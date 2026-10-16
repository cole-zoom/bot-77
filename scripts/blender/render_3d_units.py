"""Render generator sprites for the units the game draws live from 3D models (decisions.md, 2026-10-10): all hero
forms, Minion Giant, Ronin, evo Electro Giant, evo Elite Barbarians. Their .sc files hold only effects, so
extract_game_sprites.py can't cover them.

Runs inside Blender 5.2 (background), with the gltf-Supercell-IO extension (Daniil-SV, GPL-3.0; used as a tool, not
vendored) to read Supercell's "Odin" .glb. Textures are the pack's .ktx files decoded locally to PNG
(~/tools/bluestacks/tex, by KTX-Software); the extension's network texture service is stubbed out so nothing leaves
the machine.

Look: the extension's shader nodes don't render in background mode, so each material is rebuilt from the parameters
the extension reads out of the file (Supercell's uber_pbr): diffuse (the "_battle" variant when there is one),
normal map, and the team colour mask (blue channel of the "_spec" texture) filled with red or blue. Camera: orthographic,
47 deg elevation, 66 px per model unit, calibrated against the official 2D Giant sprite rendered from the same model
(silhouette IoU 0.83 to 0.90). Directions follow the 2D sprites: d01 faces away, d09 faces the camera.

  /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/blender/render_3d_units.py -- [--only giant-hero] [--frames 3]
Writes ~/tools/bluestacks/sprites3d/<class>/<class>_<team>_r3d-<anim>-dNN-fNNN.png + manifest.json.
"""
import json, math, os, re, sys
from pathlib import Path

import addon_utils
import bpy
import mathutils

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ONLY = ARGS[ARGS.index("--only") + 1].split(",") if "--only" in ARGS else None
NFRAMES = int(ARGS[ARGS.index("--frames") + 1]) if "--frames" in ARGS else 3
HOME = Path.home()
PACK, TEX, OUT = HOME / "tools/bluestacks/pack/assets", HOME / "tools/bluestacks/tex", HOME / "tools/bluestacks/sprites3d"
CH = "sc3d/characters/"
ELEV, PX_PER_UNIT, RES = 47.0, 66.0, 320
TEAM = {1: (0.86, 0.17, 0.20), 0: (0.20, 0.47, 0.95)}  # red (opponent), blue
SKIP_ANIM = re.compile(r"meta|victory|defeat|death|effect|intro|pose|lookdev|break_effect|destroyed|spawn", re.I)
KEEP_ANIM = re.compile(r"walk|run|attack|idle|deploy|drop|ability|charge|jump", re.I)

# class -> folder, models (lod1 rigs; all parts drawn together), optional animation filter
UNITS = {
    "giant-hero": ("hero/hero_giant", ["hero_giant_lod1_rig"]),
    "knight-hero": ("hero/hero_knight", None),
    "mini-pekka-hero": ("hero/hero_minipekka", ["hero_minipekka_lod1_rig"]),
    "valkyrie-hero": ("hero/hero_valkyrie", None),
    "musketeer-hero": ("hero/hero_musketeer", None),
    "wizard-hero": ("hero/hero_wizard", None),
    "ice-wizard-hero": ("hero/hero_icewizard", None),
    "electro-wizard-hero": ("hero/hero_electrowizard", None),
    "magic-archer-hero": ("hero/hero_magic_archer", None),
    "mega-minion-hero": ("hero/hero_megaminion", None),
    "balloon-hero": ("hero/hero_balloon", None),
    "bowler-hero": ("hero/hero_bowler", None),
    "berserker-hero": ("hero/hero_berserker", None),
    "barbarian-barrel-hero": ("hero/hero_barbarian_barrel", None),
    "ice-golem-hero": ("hero/hero_icegolem", None),
    "goblins-hero": ("hero/hero_goblins", ["hero_goblins_lod1_rig"]),
    "tombstone-hero": ("hero/hero_tombstone", ["hero_tombstone_lod1_rig"]),
    "tomb-queen": ("hero/hero_tombstone", ["hero_tombstone_skeletonqueen_lod1_rig"], "skeletonqueen"),
    "dark-prince-hero": ("hero/hero_darkprince", ["hero_darkprince_lod1_rig", "hero_darkprince_rhino_lod1_rig",
                                                  "hero_darkprince_shield_lod1", "hero_darkprince_weapon_lod1"], "mounted|rhino"),
    "dark-prince-hero-dismounted": ("hero/hero_darkprince", ["hero_darkprince_lod1_rig", "hero_darkprince_weapon_lod1"], "dismount|noshield|unmount"),
    "minion-giant": ("miniongiant", None),
    "ronin": ("ronin_vanilla", None),
    "electro-giant-evolution": ("evolution/evo_electrogiant", None),
    "elite-barbarian-evolution": ("evolution/evo_elite_barbarians", ["evo_elitebarbarians_lod1_rig"]),
}


def setup_extension():
    addon_utils.enable("bl_ext.user_default.gltf_supercell_io", default_set=True)
    import bl_ext.user_default.gltf_supercell_io.com.net.texture_loader as tl
    for fn in [n for n in dir(tl) if not n.startswith("_") and callable(getattr(tl, n))]:
        setattr(tl, fn, lambda *a, **k: None)  # no network: textures come from TEX only
    os.chdir(TEX)  # the extension resolves texture paths ("sc3d/...") relative to the working directory


def clear():
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.materials, bpy.data.images):
        for x in list(coll):
            if x.users == 0 or coll is bpy.data.objects:
                coll.remove(x)


def png(ktx_name):
    """'sc3d/.../x_diff.ktx(.001)' -> decoded PNG path, preferring the battle diffuse."""
    p = TEX / re.sub(r"\.ktx(\.\d+)?$", ".png", ktx_name)
    battle = p.with_name(p.stem + "_battle.png")
    return battle if battle.exists() else p


def build_material(params, team):
    m = bpy.data.materials.new("unit")
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.65
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    diff = nt.nodes.new("ShaderNodeTexImage"); diff.image = bpy.data.images.load(str(png(params["diffuseTex"])), check_existing=True)
    color = diff.outputs["Color"]
    spec_p = TEX / re.sub(r"\.ktx(\.\d+)?$", ".png", params.get("teamColorTex", ""))
    if params.get("teamColorTex") and spec_p.exists():
        spec = nt.nodes.new("ShaderNodeTexImage"); spec.image = bpy.data.images.load(str(spec_p), check_existing=True)
        spec.image.colorspace_settings.name = "Non-Color"
        sep = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(sep.inputs[0], spec.outputs["Color"])
        mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
        nt.links.new(mix.inputs["Factor"], sep.outputs["Blue"])  # team mask
        nt.links.new(mix.inputs["A"], color)
        mix.inputs["B"].default_value = (*[c * 2.2 for c in TEAM[team]], 1.0)  # battle textures leave the team areas dark
        mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
        nt.links.new(mx.inputs["Factor"], sep.outputs["Blue"]); nt.links.new(mx.inputs["A"], color)
        mx.inputs["B"].default_value = (*TEAM[team], 1.0)
        # dark (battle) team areas take the flat team colour, textured ones a multiply
        add = nt.nodes.new("ShaderNodeMix"); add.data_type = "RGBA"; add.blend_type = "LIGHTEN"; add.inputs["Factor"].default_value = 1.0
        nt.links.new(add.inputs["A"], mix.outputs["Result"]); nt.links.new(add.inputs["B"], mx.outputs["Result"])
        color = add.outputs["Result"]
    nt.links.new(bsdf.inputs["Base Color"], color)
    if diff.image.alpha_mode != "NONE" and diff.image.depth in (32, 64):
        nt.links.new(bsdf.inputs["Alpha"], diff.outputs["Alpha"])
    n_p = TEX / re.sub(r"\.ktx(\.\d+)?$", ".png", params.get("normalTex", ""))
    if params.get("normalTex") and n_p.exists():
        nt_ = nt.nodes.new("ShaderNodeTexImage"); nt_.image = bpy.data.images.load(str(n_p), check_existing=True)
        nt_.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.6
        nt.links.new(nm.inputs["Color"], nt_.outputs["Color"]); nt.links.new(bsdf.inputs["Normal"], nm.outputs["Normal"])
    return m


def sc_params(mat):
    for n in mat.node_tree.nodes if mat.node_tree else []:
        if n.bl_idname == "ShaderNodeScShader":
            return {k: (v.name if hasattr(v, "name") else v) for k, v in n.items()}
    return None


def action_bones(act):
    curves = list(getattr(act, "fcurves", []) or [])
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                curves += list(cb.fcurves)
    return {m[1] for fc in curves if (m := re.match(r'pose\.bones\["(.+?)"\]', fc.data_path))}


def load_unit(folder, models):
    d = PACK / CH / folder
    files = sorted((d / "model").glob("*.glb"))
    if models is None:
        lod1 = [f for f in files if "lod1_rig" in f.stem and not re.search(r"lookdev|broken|noarmor|projectile", f.stem)]
        # the main body is the shortest name (hero_icewizard_lod1_rig, not hero_icewizard_icechunks_lod1_rig)
        files = sorted(lod1, key=lambda f: len(f.stem))[:1] or [f for f in files if "rig" in f.stem][:1]
    else:
        files = [d / "model" / f"{m}.glb" for m in models]
    before = set(bpy.data.objects)
    for f in files:
        bpy.ops.import_scene.gltf(filepath=str(f))
    parts = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new("unit_root", None); bpy.context.scene.collection.objects.link(root)
    for o in parts:
        if o.parent is None:
            o.parent = root
    root.rotation_euler = (math.radians(-90), 0, 0)  # Odin files are Y-up with the body along -Y
    return root, [o for o in parts if o.type == "ARMATURE"]


def fit_camera(root, meshes):
    """Stand the unit up (feet at z=0) and frame it; returns the unit's height in model units."""
    import numpy as np
    bpy.context.view_layer.update()
    pts = np.array([(o.matrix_world @ mathutils.Vector(c))[:] for o in meshes for c in o.bound_box])
    lo, hi = pts.min(0), pts.max(0)
    root.location.z -= lo[2]
    size = max(hi - lo)
    cam = bpy.context.scene.camera
    cam.data.ortho_scale = size * 1.9
    e = math.radians(ELEV); mid = (hi[2] - lo[2]) / 2
    cam.location = (0, -20 * math.cos(e), 20 * math.sin(e) + mid)
    cam.rotation_euler = (mathutils.Vector((0, 0, mid)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    return hi[2] - lo[2]


def animations(folder, pattern):
    out = []
    for f in sorted(x for d in (PACK / CH / folder).glob("animation*") for x in d.rglob("*.glb")):
        if SKIP_ANIM.search(f.stem) or not KEEP_ANIM.search(f.stem):
            continue
        if pattern and not re.search(pattern, str(f.relative_to(PACK / CH / folder))):
            continue
        out.append(f)
    return out


HELPER_BONES = ("toEngineGrp", "EngineJoints")


def apply_animation(f, rigs, root, state):
    """Import a pure-animation .glb onto the matching rig (the extension retargets onto the selected armature).
    The meshes hang off the rig's top helper bone *and* are skinned to the rig, so the 90 deg up-axis rotation the
    animations key on the helper bones would be applied twice (the unit ends up lying down); those channels are dropped."""
    best = rigs[0]
    if len(rigs) > 1:  # pick the rig whose name the file name mentions (e.g. rhino vs prince)
        stem = f.stem.lower()
        best = max(rigs, key=lambda r: sum(t in stem for t in re.split(r"[_\W]+", r.name.lower()) if len(t) > 3))
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    best.select_set(True); bpy.context.view_layer.objects.active = best
    acts = set(bpy.data.actions)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(f))
    for o in [o for o in bpy.data.objects if o not in before]:
        bpy.data.objects.remove(o)
    act = best.animation_data.action if best.animation_data else None
    if act is None or act in acts and act.name != f.stem:
        act = next((a for a in bpy.data.actions if a not in acts), None)
        if act is None:
            print("  no action in", f.stem); return None
        best.animation_data_create(); best.animation_data.action = act
        if act.slots:
            best.animation_data.action_slot = act.slots[0]
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                for fc in [fc for fc in cb.fcurves if any(f'"{b}"' in fc.data_path for b in HELPER_BONES)]:
                    cb.fcurves.remove(fc)
    for b in HELPER_BONES:
        if b in best.pose.bones:
            best.pose.bones[b].rotation_quaternion = (1, 0, 0, 0)
            best.pose.bones[b].location = (0, 0, 0)
    return act


def scene_setup():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = RES
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    w = bpy.data.worlds.new("w"); sc.world = w
    w.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 0.97, 0.92, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.1
    sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 2.4; sun.angle = math.radians(20)
    so = bpy.data.objects.new("sun", sun); sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(35), 0, math.radians(20))  # from above, slightly in front of the camera
    cam = bpy.data.cameras.new("cam"); cam.type = "ORTHO"; cam.ortho_scale = RES / PX_PER_UNIT
    co = bpy.data.objects.new("cam", cam); sc.collection.objects.link(co); sc.camera = co
    e = math.radians(ELEV)
    co.location = (0, -20 * math.cos(e), 20 * math.sin(e) + 0.7)
    co.rotation_euler = (mathutils.Vector((0, 0, 0.7)) - co.location).to_track_quat("-Z", "Y").to_euler()
    return sc


def main():
    setup_extension()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() and ONLY else []
    manifest = [m for m in manifest if not ONLY or m["class"] not in ONLY]
    for cls, spec in UNITS.items():
        if ONLY and cls not in ONLY:
            continue
        folder, models = spec[0], spec[1]
        pattern = spec[2] if len(spec) > 2 else None
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        clear()
        sc = scene_setup()
        try:
            root, rigs = load_unit(folder, models)
        except Exception as ex:
            print("FAIL load", cls, ex); continue
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        height = fit_camera(root, meshes)
        params = {o.name: [sc_params(s.material) if s.material else None for s in o.material_slots] for o in meshes}
        mats, state = {}, {}
        anims = animations(folder, pattern)
        print("UNIT", cls, "rigs", [r.name for r in rigs], "meshes", len(meshes), "anims", [a.stem for a in anims], flush=True)
        n = 0
        for team in (1, 0):
            for o in meshes:
                for i, s in enumerate(o.material_slots):
                    p = params[o.name][i]
                    if p and p.get("diffuseTex"):
                        key = (p["diffuseTex"], team)
                        if key not in mats:
                            mats[key] = build_material(p, team)
                        s.material = mats[key]
            for f in anims:
                act = apply_animation(f, rigs, root, state)
                if act is None:
                    print("  no rig for", f.stem); continue
                a0, a1 = act.frame_range
                kind = re.sub(r"^.*?(walk|run|attack|idle|deploy|drop|ability|charge|jump).*$", r"\1", f.stem.lower())
                k = NFRAMES if kind in ("walk", "run", "attack") else max(1, NFRAMES - 1)
                for fi in range(k):
                    fr = int(a0 + (a1 - a0) * fi / k)
                    sc.frame_set(fr)
                    for dk in range(1, 10):
                        root.rotation_euler = (math.radians(-90), 0, math.radians(180 - (dk - 1) * 22.5))  # d01 away ... d09 facing camera
                        out = OUT / cls / f"{cls}_{team}_r3d-{f.stem.replace('_', '')}-d{dk:02d}-f{fr:03d}.png"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        sc.render.filepath = str(out)
                        bpy.ops.render.render(write_still=True)
                        manifest.append({"file": str(out), "class": cls, "team": team, "anim": f.stem, "dir": dk, "frame": fr,
                                         "px_per_unit": RES / sc.camera.data.ortho_scale, "unit_height": height})
                        n += 1
        print("DONE", cls, n, flush=True)
        (OUT / "manifest.json").write_text(json.dumps(manifest))


main()
