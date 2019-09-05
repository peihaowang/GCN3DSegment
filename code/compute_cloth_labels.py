import sys, os, io, time
import trimesh
import PIL
import cv2
import scipy
import numpy as np

from obj_loader import *

g_dataset_dir = "trainDome"
# g_dataset_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Static"
g_log_filename = "compute_cloth_labels.log"
g_override_mode = True

# To suppress the warning that the loading image exceeds the size limit
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def write_log(message, verbose=True):
    if g_log_filename:
        # Open log file
        with open(g_log_filename, mode='a', encoding="utf-8") as f:
            print(message, file=f)
    if verbose:
        print(message)

def read_mesh_from_dir(dir_path):

    # Supported mesh format
    mesh_exts = [".obj", ".off", ".ply"]

    mesh = None
    if not os.path.isdir(dir_path): return None

    for filename in os.listdir(dir_path):

        # Skip hidden files
        if filename.startswith('.'): continue

        base_name, ext_name = os.path.splitext(filename)
        if ext_name.lower() in mesh_exts:
            # Read mesh via trimesh
            mesh_path = os.path.join(dir_path, filename)

            try:
                # Disable error output
                sys.stderr = None
                mesh = load_obj_ex(mesh_path, raw_mesh=True)
                sys.stderr = sys.__stderr__

                if not isinstance(mesh, trimesh.Trimesh):
                    raise TypeError("TypeError: Loading failed or multiple geometry detected")
            except Exception as e:
                write_log(e, verbose=False)
                mesh = None
            finally:
                break

    return mesh

if __name__ == "__main__":

    # Name and class id of labels
    label_map = {
        'skin': 1
        , 'top': 2
        , 'bottom': 3
        , 'shoes': 4
    }

    write_log("Start logging at %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    for dir_name in os.listdir(g_dataset_dir):
        parent_dir = os.path.join(g_dataset_dir, dir_name)

        if not os.path.isdir(parent_dir): continue
        
        # Skip comupted meshes
        if not g_override_mode:
            skip = False
            for filename in os.listdir(parent_dir):
                if filename.endswith("_labelsV.txt"):
                    skip = True
                    break
            if skip:
                write_log("Warning: Avoid double computing labels, skip %s" % dir_name)
                continue

        write_log("Computing labels for mesh %s" % dir_name)

        origin_mesh = read_mesh_from_dir(parent_dir)
        if origin_mesh is None:
            write_log("Fatal: Cannot find the origin mesh %s" % dir_name)
            continue

        write_log("Load origin mesh successfully (vertices=%d, faces=%d)" % (origin_mesh.vertices.shape[0], origin_mesh.faces.shape[0]))

        # Maximum distance
        epsilon = 1e-8
        # Maximum outliers
        max_outliers_proportion = 0.25

        # Build up kdtree to search nearest neighbor
        tree = scipy.spatial.cKDTree(origin_mesh.vertices)
        # Calculate the sparse matrix to match the duplicated vertices
        coo = tree.sparse_distance_matrix(tree, epsilon, output_type="coo_matrix")

        labels = np.full(origin_mesh.vertices.shape[0], label_map['skin'], dtype=np.int32)
        components = ['top', 'bottom', 'shoes']
        for component in components:
            component_dir = os.path.join(parent_dir, component)
            component_mesh = read_mesh_from_dir(component_dir)
            if component_mesh is None:
                write_log("No component %s mesh" % component)
                continue

            write_log("Finding corresponding vertices: %s/%s" % (dir_name, component))

            # Calculate the distances between each vertex
            min_dists, idx = tree.query(component_mesh.vertices)

            # Give warning if the minimum gap is too large
            rought_idx = idx[min_dists > epsilon]
            rought_dists = idx[min_dists > epsilon]
            outliers_proportion = rought_idx.size / component_mesh.vertices.shape[0]
            if 0 < outliers_proportion and outliers_proportion <= max_outliers_proportion:
                write_log("Warning: Cannot find the accurate correspondence for %d(%0.2f%%) vertices, take the nearest vertex alternatively." % (rought_idx.size, outliers_proportion))
                write_log("Warning: The outliers are listed as follows:", verbose=False)
                write_log(" ".join(["(%d, %0.2f)" % (i, d) for i, d in zip(rought_idx, rought_dists)]), verbose=False)
            elif outliers_proportion > max_outliers_proportion:
                write_log("Fatal: Too many outliers(count: %d, percentage: %0.2f%%), cannot match two mesh: %s -> %s" % (rought_idx.size, outliers_proportion, component, dir_name))
                write_log("Fatal: The outliers are listed as follows:", verbose=False)
                write_log(" ".join(["(%d, %0.2f)" % (i, d) for i, d in zip(rought_idx, rought_dists)]), verbose=False)

            # Fill in the indices of duplicated vertices
            idx = np.concatenate([np.array(coo.col[coo.row == v]) for v in idx])
            idx = np.unique(idx)

            # Add labels
            labels[idx] = label_map[component]
    
            write_log("Successfully labeled %d vertices for %s/%s" % (idx.shape[0], dir_name, component))

        # Serialize to string
        label_name = dir_name + "_labelsV.txt"
        label_path = os.path.join(parent_dir, label_name)
        lines = [str(i)+'\n' for i in labels]
        with open(label_path, 'w') as f:
            f.writelines(lines)
        write_log("Successfully write labels to %s" % label_path)

    