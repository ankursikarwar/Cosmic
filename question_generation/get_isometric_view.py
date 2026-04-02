import bpy
import math, sys, argparse, time
import mathutils
from math import radians, atan2, pi
from mathutils import Matrix, Vector
from pathlib import Path

RED_RGBA   = (0.95, 0.26, 0.28, 0.24)   # ≈ #F24748 @ 24% opacity
BLUE_RGBA  = (0.48, 0.73, 0.91, 0.24)

# ---------------- CLI ----------------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", type=str, default="//top_view.png")
    p.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE"], default="CYCLES")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--margin", type=float, default=1.08, help="Padding around exterior bbox")
    p.add_argument("--hide_only", action="store_true", help="Hide instead of deleting ceiling/exterior")
    return p.parse_args(argv)
args = parse_args()

# --------------- utils ---------------
def objects_with_keyword(keyword: str):
    k = keyword.lower()
    return [o for o in bpy.data.objects if k in o.name.lower()]

def world_bbox(obj):
    M = obj.matrix_world
    corners = [M @ mathutils.Vector(c) for c in obj.bound_box]
    mn = mathutils.Vector((min(c.x for c in corners),
                           min(c.y for c in corners),
                           min(c.z for c in corners)))
    mx = mathutils.Vector((max(c.x for c in corners),
                           max(c.y for c in corners),
                           max(c.z for c in corners)))
    return mn, mx

def combined_bbox(objs):
    if not objs: return None
    inf, ninf = float("inf"), float("-inf")
    mn = mathutils.Vector((inf, inf, inf))
    mx = mathutils.Vector((ninf, ninf, ninf))
    for o in objs:
        if o.type not in {"MESH","CURVE","SURFACE","META","FONT"}:
            continue
        a, b = world_bbox(o)
        mn.x, mn.y, mn.z = min(mn.x,a.x), min(mn.y,a.y), min(mn.z,a.z)
        mx.x, mx.y, mx.z = max(mx.x,b.x), max(mx.y,b.y), max(mx.z,b.z)
    if mn.x == inf: return None
    return mn, mx

def scene_bbox():
    objs = [o for o in bpy.data.objects if o.type in {"MESH","CURVE","SURFACE","META","FONT"} and not o.hide_render]
    return combined_bbox(objs)

# ------- delete / hide ceiling+exterior -------
def remove_exterior_and_ceiling(hide_only=False):
    exts = objects_with_keyword("exterior")
    ceils = objects_with_keyword("ceiling")
    seen, targets = set(), []
    for o in (*exts, *ceils):
        if o.name_full not in seen:
            seen.add(o.name_full); targets.append(o)
    deleted, hidden = 0, 0
    if not targets:
        print("[INFO] No objects matched 'exterior' or 'ceiling'.")
        return deleted, hidden
    if hide_only:
        for o in targets:
            if o.users_scene: o.hide_set(True)
            o.hide_render = True; hidden += 1
        print(f"[INFO] Hidden (render): {hidden}")
        return deleted, hidden
    bpy.ops.object.select_all(action='DESELECT')
    for o in targets:
        try: o.select_set(True)
        except: pass
    try:
        res = bpy.ops.object.delete()
        if res == {'FINISHED'}:
            deleted = len(targets); targets = []
    except: pass
    for o in targets:
        try:
            if o.users_scene: o.hide_set(True)
            o.hide_render = True; hidden += 1
        except: pass
    print(f"[INFO] Deleted: {deleted}, Hidden: {hidden}")
    return deleted, hidden

# -------- camera placement (over EXTERIOR) --------
def ensure_camera(name="TopViewCam"):
    cam = next((o for o in bpy.data.objects if o.type == "CAMERA" and o.name == name), None)
    if cam is None:
        cam_data = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    return cam


def place_cam_over_exterior(cam, margin=1.08):
    # --- compute bounding box of visible geometry ---
    exts = [o for o in objects_with_keyword("exterior") if not o.hide_render]
    bbox = combined_bbox(exts) or scene_bbox()
    if bbox is None:
        print("[WARN] No visible geometry found.")
        return

    mn, mx = bbox
    cx = 0.5 * (mn.x + mx.x)
    cy = 0.5 * (mn.y + mx.y)
    cz = 0.5 * (mn.z + mx.z)
    width  = max(1e-6, (mx.x - mn.x) * margin)
    height = max(1e-6, (mx.y - mn.y) * margin)
    diag = max(width, height)

    # --- create orthographic camera in isometric orientation ---
    cam.data.type = 'ORTHO'
    iso_angle = math.radians(54.7356)   # standard isometric tilt
    cam.rotation_euler = (iso_angle, 0, math.radians(45))

    # --- position camera diagonally above the center ---
    cam.location = (
        cx - diag * 0.7,   # back-left from center
        cy - diag * 0.7,   # back-left from center
        cz + diag * 0.9    # slightly above the scene
    )

    # --- make camera look at the scene center ---
    direction = mathutils.Vector((cx, cy, cz)) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    # --- set orthographic scale and render settings ---
    cam.data.ortho_scale = diag * 1.2
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.view_layer.update()



# -------- render config --------
def configure_render():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = args.samples
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.film_transparent = True
    sc.render.filepath = bpy.path.abspath(args.output)

# -------- flat lighting / no shadows in Cycles --------
def disable_scene_lights_and_shadows():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"

    # remove all light objects
    for L in list(bpy.data.lights):
        bpy.data.lights.remove(L, do_unlink=True)

    # world: uniform white background lighting
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    W = sc.world
    W.use_nodes = True
    nt = W.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.2
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # cycles settings: minimize bounces and shadows
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 1
    sc.cycles.diffuse_bounces = 0
    sc.cycles.glossy_bounces = 0
    sc.cycles.transmission_bounces = 0
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.use_fast_gi = True
    if hasattr(sc.cycles, "use_caustics"):
        sc.cycles.use_caustics = False

    # kill object shadow casting
    for o in bpy.data.objects:
        if hasattr(o, "cycles_visibility"):
            o.cycles_visibility.shadow = False

    # kill material shadowing where applicable
    for m in bpy.data.materials:
        if not m or not m.use_nodes: continue
        try:
            if hasattr(m, "shadow_method"):
                m.shadow_method = 'NONE'
        except: pass


# ------------- glowing translucent FOV material -------------
def make_translucent_mat(name="FOV_Sector_Mat", rgba=(0.9, 0.2, 0.2, 0.35), emit_strength=4.0):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out  = nt.nodes.new("ShaderNodeOutputMaterial")
    mixT = nt.nodes.new("ShaderNodeMixShader")
    tr   = nt.nodes.new("ShaderNodeBsdfTransparent")
    mixE = nt.nodes.new("ShaderNodeMixShader")
    emis = nt.nodes.new("ShaderNodeEmission")
    princ= nt.nodes.new("ShaderNodeBsdfPrincipled")

    princ.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
    princ.inputs["Roughness"].default_value = 0.2
    princ.inputs["Metallic"].default_value = 0.0

    emis.inputs["Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
    emis.inputs["Strength"].default_value = emit_strength

    mixE.inputs["Fac"].default_value = 0.7
    nt.links.new(emis.outputs["Emission"], mixE.inputs[1])
    nt.links.new(princ.outputs["BSDF"],    mixE.inputs[2])

    mixT.inputs["Fac"].default_value = max(0.0, min(1.0, rgba[3]))
    nt.links.new(tr.outputs["BSDF"],      mixT.inputs[1])
    nt.links.new(mixE.outputs["Shader"],  mixT.inputs[2])
    nt.links.new(mixT.outputs["Shader"],  out.inputs["Surface"])

    if hasattr(mat, "blend_method"): mat.blend_method = 'BLEND'
    if hasattr(mat, "use_backface_culling"): mat.use_backface_culling = False
    try:
        if hasattr(mat, "shadow_method"): mat.shadow_method = 'NONE'
    except: pass
    return mat


def hide_objects_with_keyword(keyword: str):
    k = keyword.lower()
    hidden = 0
    for o in bpy.data.objects:
        if k in o.name.lower():
            o.hide_render = True
            o.hide_set(True)
            hidden += 1
    print(f"[INFO] Hidden {hidden} objects with keyword '{keyword}'.")


# ------------- compositor glow (bloom-like) -------------
def enable_compositor_glow(threshold=0.6, size=6, mix=0.0):
    sc = bpy.context.scene
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    R  = nt.nodes.new("CompositorNodeRLayers")
    GL = nt.nodes.new("CompositorNodeGlare")
    CM = nt.nodes.new("CompositorNodeMixRGB")
    OUT= nt.nodes.new("CompositorNodeComposite")

    GL.glare_type = 'FOG_GLOW'
    GL.quality = 'HIGH'
    GL.size = size
    GL.mix = mix
    GL.threshold = threshold

    CM.blend_type = 'ADD'
    CM.inputs[0].default_value = 1.0

    nt.links.new(R.outputs["Image"], GL.inputs["Image"])
    nt.links.new(R.outputs["Image"], CM.inputs[1])
    nt.links.new(GL.outputs["Image"], CM.inputs[2])
    nt.links.new(CM.outputs["Image"], OUT.inputs["Image"])

# ------------- agent sectors (FOV) -------------
def scene_floor_z():
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render]
    if not meshes: return 0.0
    return min((o.matrix_world @ Vector(b)).z for o in meshes for b in o.bound_box)

def ray_plane_intersect_z(origin: Vector, direction: Vector, z: float):
    if abs(direction.z) < 1e-8: return None
    t = (z - origin.z) / direction.z
    if t <= 0: return None
    return origin + direction * t

def camera_edge_dirs_world(cam_obj):
    cam = cam_obj.data
    frame = cam.view_frame(scene=bpy.context.scene)
    rt, rb, lb, lt = frame
    left_mid_local  = (lt + lb) * 0.5
    right_mid_local = (rt + rb) * 0.5
    M = cam_obj.matrix_world
    origin = M.translation
    left_world_pt  = M @ left_mid_local
    right_world_pt = M @ right_mid_local
    center_world_pt= M @ Vector((0, 0, -1))
    left_dir   = (left_world_pt  - origin).normalized()
    right_dir  = (right_world_pt - origin).normalized()
    center_dir = (center_world_pt - origin).normalized()
    if cam.type == 'ORTHO':
        center_dir = (M.to_3x3() @ Vector((0, 0, -1))).normalized()
        left_dir = right_dir = center_dir
    return origin, left_dir, right_dir, center_dir

def create_sector_mesh(name, center_xy: Vector, r: float, ang0: float, ang1: float, z: float, steps: int = 96):
    d = (ang1 - ang0) % (2*pi)
    if d > pi:
        d = 2*pi - d
        ang0, ang1 = ang1, ang0
    if d == 0: d = 2*pi
    n = max(3, int(steps * (d / (2*pi))))
    verts = [(center_xy.x, center_xy.y, z)]
    for i in range(n+1):
        t = ang0 + d * (i / n)
        verts.append((center_xy.x + r*math.cos(t), center_xy.y + r*math.sin(t), z))
    faces = [(0, i, i+1) for i in range(1, len(verts)-1)]
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def create_sector_fov_agent(name, cam_obj, color=(0.95, 0.30, 0.30, 1.0)):
    z_floor = scene_floor_z()
    origin, left_dir, right_dir, center_dir = camera_edge_dirs_world(cam_obj)
    P_left  = ray_plane_intersect_z(origin, left_dir,   z_floor) or (origin + left_dir * 5.0)
    P_right = ray_plane_intersect_z(origin, right_dir,  z_floor) or (origin + right_dir * 5.0)
    P_center= ray_plane_intersect_z(origin, center_dir, z_floor) or (origin + center_dir * 1.0)
    O2 = Vector((origin.x, origin.y))
    L2 = Vector((P_left.x, P_left.y))
    R2 = Vector((P_right.x, P_right.y))
    C2 = Vector((P_center.x, P_center.y))
    r = max(0.05, (C2 - O2).length) * 2.0
    ang0 = atan2((L2 - O2).y, (L2 - O2).x)
    ang1 = atan2((R2 - O2).y, (R2 - O2).x)
    sector = create_sector_mesh(name, O2, r, ang0, ang1, z_floor, steps=96)

    # disable shadows at object level
    try:
        if hasattr(sector, 'cycles_visibility'):
            sector.cycles_visibility.shadow = False
            sector.cycles_visibility.shadow_catcher = False
    except: pass

    rgba = (color[0], color[1], color[2], color[3])
    mat = make_translucent_mat(f"{name}_Mat", rgba, emit_strength=2.0)
    if sector.data.materials:
        sector.data.materials[0] = mat
    else:
        sector.data.materials.append(mat)
    sector.location.z += 2.0 #0.01
    return sector

# ------------- replace cameras with agents -------------
def replace_cameras_with_agents(top_cam_name="TopViewCam"):
    cams = [o for o in bpy.data.objects if o.type == 'CAMERA' and o.name != top_cam_name][:2]
    if not cams:
        print("[INFO] No extra cameras found; skipping agent placement.")
        return 0
    # create_sector_fov_agent("Agent_A_FOV", cams[0], color=(0.8, 0.15, 0.15, 1.0))
    create_sector_fov_agent("Agent_A_FOV", cams[0], color=RED_RGBA)
    if len(cams) > 1:
        # create_sector_fov_agent("Agent_B_FOV", cams[1], color=(0.15, 0.35, 0.8, 1.0))
        create_sector_fov_agent("Agent_B_FOV", cams[1], color=BLUE_RGBA)
    bpy.context.view_layer.update()
    return len(cams)

# ---------------- main ----------------
def main():
    bpy.ops.object.select_all(action='DESELECT')

    # camera
    cam = ensure_camera("TopViewCam")
    place_cam_over_exterior(cam, margin=args.margin)
    bpy.context.scene.camera = cam

    # agents
    # agents_count = replace_cameras_with_agents(top_cam_name="TopViewCam")
    # print(f"[INFO] Agents placed: {agents_count}")

    # remove ceiling + exterior
    remove_exterior_and_ceiling(hide_only=args.hide_only)

    # hide door blocking planes
    hide_objects_with_keyword("door_light_blocking_plane")


    # flat lighting, no shadows (Cycles)
    disable_scene_lights_and_shadows()

    # render settings
    configure_render()

    # compositor glow (bloom-like) for emissive sectors/badges
    enable_compositor_glow(threshold=0.6, size=6, mix=0.0)

    print(f"[INFO] Output: {bpy.context.scene.render.filepath}")
    bpy.ops.render.render(write_still=True)

    # save a copy of the blend next to source
    current_name = "with_agents"
    src = Path(bpy.data.filepath)
    if not src.exists():
        raise RuntimeError("Current .blend has no filepath; open via -b <file>.blend")
    out_dir = src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{current_name}_{stamp}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path), copy=True)
    print(f"[INFO] Saved copy: {out_path}")
    return str(out_path)

if __name__ == "__main__":
    main()

# Example:
# /home/mila/a/ankur.sikarwar/Work/MultiAgent/question_generation/blender-4.5.3-linux-x64/blender -b /home/mila/a/ankur.sikarwar/scratch/infinigen/infinigen_debang/infinigen/outputs/Bedroom_v1001_Final_Part1/40ef35db/coarse/scene.blend -P /home/mila/a/ankur.sikarwar/Work/MultiAgent/question_generation_v2/get_top_view_v1.py -- --output ./top_view.png --engine CYCLES --samples 32 --margin 1.2