"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
import os, sys, math
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
# To suppress the warning that the loaded image exceeds the size limit
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# g_dataset_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Static"
g_dataset_dir = "testDome"
g_single_viewport_size = (640*2, 480*2)

def apply_transform(origin, transfrom):
    return np.dot(transfrom, origin)

def read_mesh_from_dir(dir_path):
    # Supported mesh format
    mesh_exts = [".obj", ".off", ".ply"]

    mesh = None
    if not os.path.isdir(dir_path): return None

    for filename in os.listdir(dir_path):
        base_name, ext_name = os.path.splitext(filename)
        if ext_name.lower() in mesh_exts:
            # Read mesh via trimesh
            mesh_path = os.path.join(dir_path, filename)

            try:
                # sys.stderr = None
                mesh = trimesh.load(mesh_path)
                # sys.stderr = sys.__stderr__

                if not isinstance(mesh, trimesh.Trimesh):
                    raise TypeError()
            except Exception as e:
                mesh = None
            finally:
                break

    return mesh

def extract_material(mesh):
    trimesh_mat = mesh.visual.material
    pyrender_mat = None
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

if __name__ == "__main__":

    vis_list = {
        'origin': ['model']
        , 'clothes': ['top', 'bottom', 'shoes', ['top', 'bottom', 'shoes'], ['model', 'top', 'bottom', 'shoes']]
        , 'naked': ['naked', ['top', 'bottom', 'shoes', 'naked']]
    }

    view_angles = [0, 90, 180, 270]

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

    for dir_name in os.listdir(g_dataset_dir):
        parent_dir = os.path.join(g_dataset_dir, dir_name)

        if not os.path.isdir(parent_dir): continue

        print("Visualizing human model: %s" % dir_name)

        meshes = {}

        # Read origin mesh first
        print("Loading origin mesh ...")
        origin_mesh = read_mesh_from_dir(parent_dir)
        if origin_mesh is None:
            print("Fatal: No origin mesh!")
            continue

        # Extract common material
        print("Extracting common material ...")
        common_material = extract_material(origin_mesh)
        if common_material is None:
            print("Fatal: Invalid material extracted")
            continue

        # Convert origin mesh to pyrender mesh
        meshes['model'] = Mesh.from_trimesh(origin_mesh, material=common_material)
        # Force to release memory
        del origin_mesh

        all_components = ['top', 'bottom', 'shoes', 'naked']
        for component in all_components:
            print("Loading component %s mesh ..." % component)
            component_dir = os.path.join(parent_dir, component)
            component_mesh = read_mesh_from_dir(component_dir)
            if component_mesh is None:
                print("No component %s mesh" % component)
                continue

            # Convert to pyrender mesh
            material = common_material if component != 'naked' else None
            mesh = Mesh.from_trimesh(component_mesh, material=material)
            meshes[component] = mesh
            # Force to release memory
            del component_mesh

        if len(meshes) <= 1:
            print("Warning: Only origin mesh available")
            continue

        for filename, component_list in vis_list.items():

            print("Start rendering image: %s" % filename)

            image_mat = None
            for components in component_list:
                if isinstance(components, str): components = [components]

                print("Rendering row: %s" % ','.join(components))

                visible_meshes = [meshes[component] for component in components if component in meshes]
                if len(visible_meshes) == 0:
                    print("Warning: No components available while rendering the current row.")
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

                # 1. Distance
                ratio = g_single_viewport_size[1] / g_single_viewport_size[0]
                margin = 0.0
                height = max(aabb_size[1], aabb_size[0]*ratio) + margin*2
                dist = (height/2) / math.tan(cam.yfov/2)
                cam_pos[2] += dist

                # 2. Offset
                max_offset = float('-inf')
                for angle in view_angles:
                    rad = math.radians(angle)
                    # Assume elliptical boundary
                    offset = math.sqrt(
                        ((aabb_size[0]/2)*math.sin(rad)) ** 2
                        + ((aabb_size[2]/2)*math.cos(rad)) ** 2
                    )
                    max_offset = max(max_offset, offset)
                cam_pos[2] += offset

                # 3. Transform
                cam_mat = translation_matrix(cam_pos)
                cam_node.matrix = cam_mat

                # Concat each angle to form a row
                row_mat = None
                for angle in view_angles:
                    print("Rendering the angle of view: %d in degree" % angle)

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
                        print("Fatal: Render error.")
                        continue

                    assert color.shape[1] == g_single_viewport_size[0] and color.shape[0] == g_single_viewport_size[1], "Fatal: Inconsistent color map size"

                    if row_mat is None:
                        row_mat = color
                    else:
                        assert row_mat.shape[0] == color.shape[0], "Fatal: Inconsistent cell size"
                        row_mat = np.hstack((row_mat, color))

                for node in mesh_nodes: scene.remove_node(node)

                if image_mat is None:
                    image_mat = row_mat
                else:
                    assert image_mat.shape[1] == row_mat.shape[1], "Fatal: Inconsistent row size"
                    image_mat = np.vstack((image_mat, row_mat))

            # Save image
            image_path = os.path.join(parent_dir, filename + '_vis.png')
            image_mat = cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image_mat)

    r.delete()

