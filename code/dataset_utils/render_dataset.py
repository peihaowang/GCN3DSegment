
import os, sys, math, time
import pyglet
import numpy as np
import cv2, trimesh, PIL
from transformations import *

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer,\
                     RenderFlags

pyglet.options['shadow_window'] = False
# To suppress the warning that the loading image exceeds the size limit
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# g_dataset_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Dynamics"
# g_dataset_dir = "F:/20190401_Data_Clothing/20190806_labeled_clothing/Dynamics"
g_dataset_dir = "dynaDome"
g_dataset_kind = "dynamic"
g_single_viewport_size = (640*2, 480*2)
g_log_filename = "render_dataset.log"

# Supported mesh format
g_mesh_exts = [".obj", ".off", ".ply"]

# Combinations of components to visualize
g_vis_list = {
    'origin': ['model']
    , 'clothes': ['top', 'bottom', 'shoes', ['top', 'bottom', 'shoes'], ['model', 'top', 'bottom', 'shoes']]
    , 'naked': ['naked', ['top', 'bottom', 'shoes', 'naked']]
}

g_view_angles = [0, 90, 180, 270]

def write_log(message, verbose=True):
    if g_log_filename:
        # Open log file
        with open(g_log_filename, mode='a', encoding="utf-8") as f:
            print(message, file=f)
    if verbose:
        print(message)

class DummyResolver(trimesh.visual.resolvers.FilePathResolver):
    def __init__(self, source):
        super(DummyResolver, self).__init__(source)

    def get(self, name):
        # Supported mesh format
        image_exts = [".png", ".jpg", ".jpeg", ".bmp"]

        _, ext_name = os.path.splitext(name)
        if ext_name.lower() in image_exts:
            fake_texture = np.zeros((5, 5), dtype=np.uint8)
            _, data = cv2.imencode('.png', fake_texture)
            return data.tobytes()
        else:
            return super(DummyResolver, self).get(name)

def extract_material(mesh):
    pyrender_mat = None
    if mesh.visual.kind == 'texture':
        trimesh_mat = mesh.visual.material
        if isinstance(trimesh_mat, trimesh.visual.texture.PBRMaterial):
            pyrender_mat = MetallicRoughnessMaterial(
                normalTexture=trimesh_mat.normalTexture,
                occlusionTexture=trimesh_mat.occlusionTexture,
                emissiveTexture=trimesh_mat.emissiveTexture,
                emissiveFactor=trimesh_mat.emissiveFactor,
                alphaMode='BLEND',
                baseColorFactor=trimesh_mat.baseColorFactor,
                baseColorTexture=trimesh_mat.baseColorTexture,
                metallicFactor=trimesh_mat.metallicFactor,
                metallicRoughnessTexture=trimesh_mat.metallicRoughnessTexture,
                doubleSided=trimesh_mat.doubleSided,
                alphaCutoff=trimesh_mat.alphaCutoff
            )
        elif isinstance(trimesh_mat, trimesh.visual.texture.SimpleMaterial):
            pyrender_mat = MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorTexture=trimesh_mat.image
            )
    return pyrender_mat

def read_mesh_from_dir(dir_path, load_texture=True):
    mesh = None
    if not os.path.isdir(dir_path): return None

    for filename in os.listdir(dir_path):
        base_name, ext_name = os.path.splitext(filename)
        if ext_name.lower() in g_mesh_exts:
            # Read mesh via trimesh
            mesh_path = os.path.join(dir_path, filename)

            try:
                resolver = None if load_texture else DummyResolver(dir_path)
                mesh = trimesh.load(mesh_path, resolver=resolver)

                if not isinstance(mesh, trimesh.Trimesh):
                    raise TypeError()
            except Exception as e:
                mesh = None
            finally:
                break

    return mesh

def read_mesh_by_name(dir_path, basename, suffix=None, load_texture=True):
    mesh = None
    if not os.path.isdir(dir_path): return None

    if suffix is None:
        exts = g_mesh_exts
    elif isinstance(suffix, list):
        exts = suffix
    elif isinstance(suffix, str):
        exts = [suffix]

    for ext in exts:
        # Read mesh via trimesh
        mesh_path = os.path.join(dir_path, basename + ext)
        if os.path.exists(mesh_path):
            try:
                resolver = None if load_texture else DummyResolver(dir_path)
                mesh = trimesh.load(mesh_path, resolver=resolver)

                if not isinstance(mesh, trimesh.Trimesh):
                    raise TypeError()
            except Exception as e:
                mesh = None
            finally:
                break

    return mesh

def traverse_static_dataset(root_dir):
    for dir_name in os.listdir(root_dir):
        parent_dir = os.path.join(root_dir, dir_name)

        if not os.path.isdir(parent_dir): continue

        write_log("Visualizing human model: %s" % dir_name)

        meshes = {}

        # Read origin mesh first
        write_log("Loading origin mesh ...")
        origin_mesh = read_mesh_from_dir(parent_dir, load_texture=True)
        if origin_mesh is None:
            write_log("Fatal: No origin mesh!")
            continue

        # Extract common material
        write_log("Extracting common material ...")
        common_material = extract_material(origin_mesh)
        if common_material is None:
            write_log("Fatal: Invalid material extracted")
            continue

        # Convert origin mesh to pyrender mesh
        meshes['model'] = Mesh.from_trimesh(origin_mesh, material=common_material)
        # Force to release memory
        del origin_mesh

        all_components = ['top', 'bottom', 'shoes', 'naked']
        for component in all_components:
            write_log("Loading component %s mesh ..." % component)
            component_dir = os.path.join(parent_dir, component)
            component_mesh = read_mesh_from_dir(component_dir, load_texture=False)
            if component_mesh is None:
                write_log("No component %s mesh" % component)
                continue

            # Convert to pyrender mesh
            material = common_material if component != 'naked' else None
            mesh = Mesh.from_trimesh(component_mesh, material=material)
            meshes[component] = mesh
            # Force to release memory
            del component_mesh

        if len(meshes) <= 1:
            write_log("Warning: Only origin mesh available")

        # Return vis image path
        vis_path = {}
        for key in g_vis_list.keys():
            vis_path[key] = os.path.join(parent_dir, key+'_vis.png')

        yield meshes, vis_path

def traverse_dynamic_dataset(root_dir):
    for dir_name in os.listdir(root_dir):
        parent_dir = os.path.join(root_dir, dir_name)

        if not os.path.isdir(parent_dir): continue

        write_log("Visualizing human model: %s" % dir_name)

        meshes = {}

        origin_dir = os.path.join(parent_dir, 'model')
        if not os.path.isdir(origin_dir):
            write_log("Fatal: No origin mesh!")
            continue

        for mesh_name in os.listdir(origin_dir):
            mesh_base, mesh_suffix = os.path.splitext(mesh_name)

            if mesh_suffix.lower() not in g_mesh_exts: continue

            write_log("Visualizing frame: %s/%s" % (dir_name, mesh_base))

            # Read origin mesh first
            write_log("Loading origin mesh ...")
            origin_mesh = read_mesh_by_name(origin_dir, mesh_base, suffix=mesh_suffix, load_texture=True)
            if origin_mesh is None:
                write_log("Fatal: No origin mesh!")
                continue

            # Extract common material
            write_log("Extracting common material ...")
            common_material = extract_material(origin_mesh)
            if common_material is None:
                write_log("Fatal: Invalid material extracted")
                continue

            # Convert origin mesh to pyrender mesh
            meshes['model'] = Mesh.from_trimesh(origin_mesh, material=common_material)
            # Force to release memory
            del origin_mesh

            all_components = ['top', 'bottom', 'shoes', 'naked']
            for component in all_components:
                write_log("Loading component %s mesh ..." % component)
                component_dir = os.path.join(parent_dir, component)
                component_mesh = read_mesh_by_name(component_dir, mesh_base, load_texture=False)
                if component_mesh is None:
                    write_log("No component %s mesh" % component)
                    continue

                # Convert to pyrender mesh
                material = common_material if component != 'naked' else None
                mesh = Mesh.from_trimesh(component_mesh, material=material)
                meshes[component] = mesh
                # Force to release memory
                del component_mesh

            if len(meshes) <= 1:
                write_log("Warning: Only origin mesh available")

            # Return vis image path
            vis_path = {}
            for key in g_vis_list.keys():
                vis_dir = os.path.join(parent_dir, key+'_vis')

                if not os.path.exists(vis_dir):
                    write_log("Making directory for visualization: %s" % vis_dir)
                    os.mkdir(vis_dir)

                vis_path[key] = os.path.join(vis_dir, mesh_base+'.png')

            yield meshes, vis_path

def traverse_dataset(root_dir, kind):
    if kind.lower() == 'static':
        yield from traverse_static_dataset(root_dir)
    elif kind.lower() == 'dynamic':
        yield from traverse_dynamic_dataset(root_dir)

if __name__ == "__main__":
    # Light creation
    direc_l = DirectionalLight(color=np.ones(3), intensity=5.0)
    # spot_l = SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
    point_l = PointLight(color=np.ones(3), intensity=10.0)

    # Camera creation
    cam = PerspectiveCamera(yfov=(np.pi / 3.0))

    # Scene creation
    scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

    # Adding objects to the scene
    cam_node = scene.add(cam)
    direc_l_node = scene.add(direc_l, parent_node=cam_node)
    # spot_l_node = scene.add(spot_l, parent_node=cam_node)
    point_l_node = scene.add(point_l, parent_node=cam_node)

    r = OffscreenRenderer(viewport_width=g_single_viewport_size[0], viewport_height=g_single_viewport_size[1])

    # Start logging flag
    write_log("Start logging at %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    write_log("Dataset path: %s" % os.path.abspath(g_dataset_dir))
    write_log("Logging path: %s" % os.path.abspath(g_log_filename))
    write_log("Dataset kind: %s" % g_dataset_kind)

    for meshes, save_paths in traverse_dataset(g_dataset_dir, g_dataset_kind):

        for filename, component_list in g_vis_list.items():

            write_log("Start rendering image: %s" % filename)

            image_mat = None
            for components in component_list:
                if isinstance(components, str): components = [components]

                write_log("Rendering row: %s" % ','.join(components))

                visible_meshes = [meshes[component] for component in components if component in meshes]
                if len(visible_meshes) == 0:
                    write_log("Warning: No components available while rendering the current row.")
                    continue

                # Add mesh into scenes
                mesh_nodes = [scene.add(mesh) for mesh in visible_meshes]

                # Calculate the AABB box for the combined meshes
                max_coords = min_coords = None
                sum_coords, sum_count = np.zeros(3, dtype=np.float32), 0
                for mesh in visible_meshes:
                    vertices = mesh.primitives[0].positions

                    if max_coords is None: max_coords = np.max(vertices, axis=0)[np.newaxis, :]
                    else: max_coords = np.vstack((max_coords, np.max(vertices, axis=0)))

                    if min_coords is None: min_coords = np.min(vertices, axis=0)[np.newaxis, :]
                    else: min_coords = np.vstack((min_coords, np.min(vertices, axis=0)))

                    sum_coords += np.sum(vertices, axis=0)
                    sum_count += vertices.shape[0]

                aabb_center = sum_coords / sum_count
                aabb_size = np.max(max_coords, axis=0) - np.min(min_coords, axis=0)

                # Calculate the cam position
                cam_pos = aabb_center.copy()

                max_offset = float('-inf')
                for angle in g_view_angles:
                    rad = math.radians(angle)

                    # 1. Distance
                    ratio = g_single_viewport_size[1] / g_single_viewport_size[0]
                    margin = 0.0
                    width = math.sqrt(
                        ((aabb_size[0]/2)*math.sin(rad+math.pi/2)) ** 2
                        + ((aabb_size[2]/2)*math.cos(rad+math.pi/2)) ** 2
                    ) * 2
                    height = max(aabb_size[1], width * ratio) + margin * 2
                    dist = (height / 2) / math.tan(cam.yfov / 2)

                    # 2. Offset
                    # Assume elliptical boundary
                    offset = math.sqrt(
                        ((aabb_size[0]/2)*math.sin(rad)) ** 2
                        + ((aabb_size[2]/2)*math.cos(rad)) ** 2
                    )

                    max_offset = max(max_offset, dist+offset)
                cam_pos[2] += max_offset

                # 3. Transform
                cam_mat = translation_matrix(cam_pos)
                cam_node.matrix = cam_mat

                # Concat each angle to form a row
                row_mat = None
                for angle in g_view_angles:
                    write_log("Rendering the angle of view: %d in degree" % angle)

                    rad = math.radians(angle)

                    # Calculate transform matrix
                    mesh_mat = translation_matrix(aabb_center)
                    mesh_mat = np.dot(rotation_matrix(rad, [0, 1, 0]), mesh_mat)
                    mesh_mat = np.dot(translation_matrix(-aabb_center), mesh_mat)

                    for node in mesh_nodes: node.matrix = mesh_mat

                    # Render scene to image
                    try:
                        color, _ = r.render(scene, flags=RenderFlags.OFFSCREEN | RenderFlags.ALL_SOLID)
                    except Exception as e:
                        write_log("Fatal: Render error!")
                        write_log(e, verbose=False)
                        # Drop current row
                        row_mat = None
                        break

                    assert color.shape[1] == g_single_viewport_size[0] and color.shape[0] == g_single_viewport_size[1], "Fatal: Inconsistent color map size"

                    if row_mat is None:
                        row_mat = color
                    else:
                        assert row_mat.shape[0] == color.shape[0], "Fatal: Inconsistent cell size"
                        row_mat = np.hstack((row_mat, color))

                for node in mesh_nodes: scene.remove_node(node)

                if row_mat is None:
                    write_log("Fatal: Failed to render the current row!")
                    continue

                if image_mat is None:
                    image_mat = row_mat
                else:
                    assert image_mat.shape[1] == row_mat.shape[1], "Fatal: Inconsistent row size"
                    image_mat = np.vstack((image_mat, row_mat))

            if image_mat is None:
                write_log("Fatal: Failed to render the current image!")
                continue

            # Save image
            image_path = save_paths[filename]
            image_mat = cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image_mat)

    # Recycle resources
    r.delete()
