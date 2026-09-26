"""組み立てたメッシュを Blender（bpy）に流し込み、.blend に保存する。"""
import bpy
import numpy as np

from .geom import Mesh

# 名前: (RGBA, 粗さ, 発光の強さ)
MATERIALS = {
    "Asphalt": ((0.06, 0.06, 0.065, 1), 0.9, 0),
    "Marking": ((0.9, 0.9, 0.88, 1), 0.6, 0),
    "Concrete": ((0.55, 0.55, 0.53, 1), 0.85, 0),
    "StructureConcrete": ((0.45, 0.45, 0.44, 1), 0.9, 0),
    "TunnelWall": ((0.75, 0.75, 0.72, 1), 0.5, 0),
    "Metal": ((0.5, 0.52, 0.55, 1), 0.4, 0),
    "LampLight": ((1.0, 0.85, 0.6, 1), 0.3, 8.0),
    "TunnelLight": ((1.0, 0.8, 0.5, 1), 0.3, 6.0),
    "SignGreen": ((0.0, 0.33, 0.18, 1), 0.5, 0),
    "SignText": ((0.95, 0.95, 0.95, 1), 0.5, 0),
    "SignRed": ((0.75, 0.05, 0.05, 1), 0.5, 0),
    "SignWhite": ((0.95, 0.95, 0.95, 1), 0.5, 0),
    "SpeedText": ((0.05, 0.15, 0.6, 1), 0.5, 0),
    "SignYellow": ((0.95, 0.75, 0.05, 1), 0.5, 0),
    "WarnText": ((0.02, 0.02, 0.02, 1), 0.5, 0),
    "Ground": ((0.25, 0.28, 0.2, 1), 1.0, 0),
}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def material(name):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    color, rough, emit = MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    if emit:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emit
    return mat


def collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def add_object(coll_name, obj_name, mesh, mat_name):
    if mesh.empty():
        return None
    verts, faces, uv = mesh.arrays()
    me = bpy.data.meshes.new(obj_name)
    me.from_pydata(verts.tolist(), [], faces)
    if np.any(uv):
        layer = me.uv_layers.new(name="UVMap")
        loop_v = np.zeros(len(me.loops), int)
        me.loops.foreach_get("vertex_index", loop_v)
        layer.data.foreach_set("uv", uv[loop_v].astype(np.float32).ravel())
    me.validate()
    me.update()
    me.materials.append(material(mat_name))
    obj = bpy.data.objects.new(obj_name, me)
    collection(coll_name).objects.link(obj)
    return obj


def text_meshes(texts, font_path):
    """文字を平らなメッシュにし、看板の面に貼る。材質ごとに Mesh を返す。"""
    out = {}
    if not texts:
        return out
    font = bpy.data.fonts.load(font_path)
    tmp = bpy.data.collections.new("_tmp_text")
    bpy.context.scene.collection.children.link(tmp)
    for t in texts:
        cu = bpy.data.curves.new("t", "FONT")
        cu.body = t["text"]
        cu.font = font
        cu.size = 1.0
        cu.align_x = "CENTER"
        cu.align_y = "CENTER"
        cu.resolution_u = 3
        ob = bpy.data.objects.new("t", cu)
        tmp.objects.link(ob)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        n = len(me.vertices)
        if n == 0:
            bpy.data.meshes.remove(me)
            continue
        local = np.zeros(n * 3)
        me.vertices.foreach_get("co", local)
        local = local.reshape(-1, 3)
        faces = [list(p.vertices) for p in me.polygons]
        bpy.data.meshes.remove(me)
        width = np.ptp(local[:, 0])
        scale = t["size"]
        if width * scale > t["max_width"]:
            scale = t["max_width"] / width
        f = np.asarray(t["facing"], float)
        f = f / np.linalg.norm(f)
        right = np.array([-f[1], f[0], 0.0])
        up = np.array([0.0, 0.0, 1.0])
        world = np.asarray(t["pos"]) + scale * (local[:, :1] * right + local[:, 1:2] * up)
        out.setdefault(t["material"], Mesh()).add(world, faces)
        bpy.data.objects.remove(ob)
        bpy.data.curves.remove(cu)
    bpy.data.collections.remove(tmp)
    return out


def save(path, metadata):
    scene = bpy.context.scene
    for k, v in metadata.items():
        scene[k] = v
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)

