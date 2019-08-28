"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
pyglet.options['shadow_window'] = False
import os, math
import cv2
import numpy as np
import trimesh
from transformations import *

from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer

# g_dataset_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Static"
g_dataset_dir = "smallDome"
g_image_size = (640*2, 480*2)

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
            except:
                mesh = None
            finally:
                break

    return mesh

if __name__ == "__main__":

    vis_list = {
        'origin': ['model']
        , 'clothes': ['top', 'bottom', 'shoes', 'top;bottom;shoes', 'model;top;bottom;shoes']
        , 'naked': ['naked', 'top;bottom;shoes;naked']
    }

    view_angles = [0, 90, 180, 270]

    # Light creation
    direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = SpotLight(color=np.ones(3), intensity=10.0, innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
    point_l = PointLight(color=np.ones(3), intensity=10.0)

    # Camera creation
    cam = PerspectiveCamera(yfov=(np.pi / 3.0))

    # Scene creation
    scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

    # Adding objects to the scene
    cam_node = scene.add(cam)
    direc_l_node = scene.add(direc_l)
    spot_l_node = scene.add(spot_l)
    point_l_node = scene.add(point_l)

    track_with_cam = [cam_node, direc_l_node, spot_l_node, point_l_node]

    # Offscreen renderer
    r = OffscreenRenderer(viewport_width=g_image_size[0], viewport_height=g_image_size[1])

    for dir_name in os.listdir(g_dataset_dir):
        parent_dir = os.path.join(g_dataset_dir, dir_name)

        if not os.path.isdir(parent_dir): continue

        print("Computing labels for mesh %s" % dir_name)

        all_components = ['model', 'top', 'bottom', 'shoes', 'naked']
        meshes = {}
        for component in all_components:

            component_dir = os.path.join(parent_dir, component)
            if component == 'model': component_dir = parent_dir

            component_mesh = read_mesh_from_dir(component_dir)
            if component_mesh is None:
                print("No component %s mesh" % component)
                continue
            meshes[component] = component_mesh

        if len(meshes) == 0:
            continue

        # Rendering offscreen from that camera
        color, _ = r.render(scene)
        r.delete()

        for filename, component_list in vis_list.items():
            image_mat = None
            for components in component_list:
                components = components.split(';')
                visible_meshes = [meshes[component] for component in components if component in meshes]
                mesh_nodes = [scene.add(Mesh.from_trimesh(mesh)) for mesh in visible_meshes]

                # Calculate the AABB box for the combined meshes
                max_coords = min_coords = None
                for mesh in visible_meshes:
                    if max_coords is None: max_coords = np.max(mesh.vertices, axis=0)
                    else: max_coords = np.vstack((max_coords, np.max(mesh.vertices, axis=0)))

                    if min_coords is None: min_coords = np.min(mesh.vertices, axis=0)
                    else: min_coords = np.vstack((min_coords, np.min(mesh.vertices, axis=0)))

                if max_coords.ndim >= 2:
                    aabb_center = (np.max(max_coords, axis=0) + np.min(min_coords, axis=0)) / 2
                    aabb_size = np.max(max_coords, axis=0) - np.min(min_coords, axis=0)
                else:
                    aabb_center = (max_coords + min_coords) / 2
                    aabb_size = max_coords - min_coords

                # Concat each angle to form a row
                row_mat = None
                for angle in view_angles:
                    # Calculate the cam position
                    rad = np.radians(angle)
                    cam_pos = aabb_center
                    # Assume elliptical boundary
                    offset = np.square(
                        ((aabb_size[0]/2)*np.sin(rad)) ** 2
                        + ((aabb_size[2]/2)*np.cos(rad)) ** 2
                    )
                    # Calculate z coordinate
                    height = np.max([aabb_size[1], aabb_size[0]*(g_image_size[1]/g_image_size[0])])
                    margin = 0.1
                    height += margin*2
                    cam_pos[2] += (height/2) / np.tan(cam.yfov/2) + offset

                    # Calculate transform matrix
                    mesh_tr_mat = translation_matrix(aabb_center)
                    mesh_rot_mat = rotation_matrix(rad, [0, 1, 0])
                    mesh_trs_mat = np.dot(mesh_tr_mat, mesh_rot_mat)

                    cam_tr_mat = translation_matrix(cam_pos)

                    for node in track_with_cam: node.matrix = cam_tr_mat
                    for node in mesh_nodes: node.matrix = mesh_trs_mat

                    # Render scene to image
                    color, _ = r.render(scene)
                    if row_mat is None:
                        row_mat = color
                    else:
                        if row_mat.shape[0] != color.shape[0]: print("Fatal: Inconsistent cell size")
                        row_mat = np.hstack((row_mat, color))
                # image_path = os.path.join(parent_dir, filename+'_'+','.join(components)+'_vis.png')
                # cv2.imwrite(image_path, row_mat)

                for node in mesh_nodes: scene.remove_node(node)

                if image_mat is None:
                    image_mat = row_mat
                else:
                    if image_mat.shape[1] != row_mat.shape[1]: print("Fatal: Inconsistent row size")
                    image_mat = np.vstack((image_mat, row_mat))

            # Save image
            image_path = os.path.join(parent_dir, filename+'_vis.png')
            cv2.imwrite(image_path, image_mat)

    r.delete()
'''
#==============================================================================
# Mesh creation
#==============================================================================

#------------------------------------------------------------------------------
# Creating textured meshes from trimeshes
#------------------------------------------------------------------------------

# Human trimesh
human_trimesh = trimesh.load('./smallDome/f_c_10412256613/f_c_10412256613_model.obj')
human_mesh = Mesh.from_trimesh(human_trimesh)

aabb_center = (np.max(human_trimesh.vertices, axis=0) + np.min(human_trimesh.vertices, axis=0)) / 2
aabb_size = np.max(human_trimesh.vertices, axis=0) - np.min(human_trimesh.vertices, axis=0)

#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
point_l = PointLight(color=np.ones(3), intensity=10.0)

#==============================================================================
# Camera creation
#==============================================================================

cam = PerspectiveCamera(yfov=(np.pi / 3.0))
# cam_pose = transformations.rotation_matrix(math.pi, [0.0, 1.0, 0.0])
cam_pos = aabb_center
cam_pos[2] += (aabb_size[1]/2) / math.tan(cam.yfov/2) + (aabb_size[2]/2)
cam_pose = transformations.translation_matrix(cam_pos)
# cam_pose = np.array([
#     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
#     [1.0, 0.0,           0.0,           0.0],
#     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
#     [0.0,  0.0,           0.0,          1.0]
# ])

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

#==============================================================================
# Adding objects to the scene
#==============================================================================

#------------------------------------------------------------------------------
# By using the add() utility function
#------------------------------------------------------------------------------
human_node = scene.add(human_mesh)
direc_l_node = scene.add(direc_l, pose=cam_pose)
spot_l_node = scene.add(spot_l, pose=cam_pose)
point_l_node = scene.add(point_l, pose=cam_pose)

human_node.matrix = np.dot(transformations.rotation_matrix(math.pi, [0.0, 1.0, 0.0]), human_node.matrix)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================
cam_node = scene.add(cam, pose=cam_pose)
# v = Viewer(scene, central_node=drill_node)

#==============================================================================
# Rendering offscreen from that camera
#==============================================================================

r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
color, _ = r.render(scene)
r.delete()

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(color)
plt.show()
'''
