import sys, os, io, time
import trimesh
import PIL
import cv2
import scipy
import numpy as np

import mesh_loader
import coarsening

g_dataset_dir = "cosegCup/train"
g_dataset_kind = "coseg"
# g_dataset_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Static"
g_log_filename = "compute_metis.log"
g_metis_level = 4
g_override_mode = True

# To suppress the warning that the loading image exceeds the size limit
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Supported mesh format
g_mesh_exts = [".obj", ".off", ".ply"]

def write_log(message, verbose=True):
    if g_log_filename:
        # Open log file
        with open(g_log_filename, mode='a', encoding="utf-8") as f:
            print(message, file=f)
    if verbose:
        print(message)

def traverse_static_dataset(root_dir):
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path): continue

        mesh_name = None
        for filename in os.listdir(dir_path):
            # Skip hidden files
            if filename.startswith('.'): continue

            _, ext_name = os.path.splitext(filename)
            if ext_name.lower() in g_mesh_exts:
                mesh_name = filename
                break

        if mesh_name is None: continue
        metis_name = dir_name + '_metis.txt'

        mesh_path = os.path.join(dir_path, mesh_name)
        metis_path = os.path.join(dir_path, metis_name)

        # Skip if metis calculated when not in override mode
        if not g_override_mode and os.path.exists(metis_path):
            continue

        yield dir_name, mesh_path, metis_path

def traverse_coseg_dataset(root_dir):
    for filename in os.listdir(root_dir):
        # Skip hidden files
        if filename.startswith('.'): continue

        base_name, ext_name = os.path.splitext(filename)

        if ext_name.lower() not in g_mesh_exts:
            continue

        metis_name = base_name + '_metis.txt'

        mesh_path = os.path.join(root_dir, filename)
        metis_path = os.path.join(root_dir, metis_name)

        # Skip if metis calculated when not in override mode
        if not g_override_mode and os.path.exists(metis_path):
            continue

        yield base_name, mesh_path, metis_path

def traverse_dataset(root_dir, kind):
    if kind.lower() == 'static':
        yield from traverse_static_dataset(root_dir)
    elif kind.lower() == 'dynamic':
        yield from traverse_dynamic_dataset(root_dir)
    elif kind.lower() == 'coseg':
        yield from traverse_coseg_dataset(root_dir)

if __name__ == "__main__":
    # Start logging flag
    write_log("Start logging at %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    write_log("Dataset path: %s" % os.path.abspath(g_dataset_dir))
    write_log("Logging path: %s" % os.path.abspath(g_log_filename))
    write_log("Dataset kind: %s" % g_dataset_kind)
    write_log("Metis level: %s" % g_metis_level)

    for name, mesh_path, metis_path in traverse_dataset(g_dataset_dir, g_dataset_kind):
        write_log('Loading mesh %s' % mesh_path)

        try:
            # Disable error output
            sys.stderr = None
            mesh = mesh_loader.load(mesh_path, raw_mesh=True)
            sys.stderr = sys.__stderr__

            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError("TypeError: Loading failed or multiple geometry detected")
        except Exception as e:
            write_log(e, verbose=False)
            mesh = None

        if mesh is None:
            write_log("Error: Failed to load mesh.")
            continue

        adjacency = mesh.edges_sparse
        graphs, parents = coarsening.metis(adjacency, g_metis_level)

        # Print out metis information
        for l, graph in enumerate(graphs):
            A = graph.tocoo()
            write_log("Level %d - vertices %d, edges %d " % (l, A.shape[0], A.col.shape[0]))

        write_log('Write metis data to %s' % metis_path)
        with open(metis_path, 'w') as f:
            lines = [' '.join([str(p) for p in parent]) + '\n' for parent in parents]
            f.writelines(lines)

    