import sys, os, io
import trimesh, PIL
import scipy
import numpy as np

g_dataset_dir = "smallDome"
# g_dataset_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Static"
g_log_filename = "compute_cloth_labels2.log"
g_override_mode = True

def write_log(lines, output=False):
    if isinstance(lines, str):
        lines = [lines+'\n']

    if output: print(''.join(lines), end='')

    with open(g_log_filename, 'a') as f:
        f.writelines(lines)

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
                # Disable error output
                sys.stderr = None
                mesh = trimesh.load(mesh_path)
                sys.stderr = sys.__stderr__

                if not isinstance(mesh, trimesh.Trimesh):
                    raise TypeError()
            except:
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
                print("Avoid double computing labels, skip %s" % dir_name)
                continue

        print("Computer labels for mesh %s" % dir_name)

        origin_mesh = read_mesh_from_dir(parent_dir)
        if origin_mesh is None:
            write_log("Cannot find the origin mesh %s" % dir_name, True)
            continue

        # Maximum distance
        epsilon = 0.01
        # Maximum outliers
        max_outliers = 50

        # Build up kdtree to search nearest neighbor
        tree = scipy.spatial.cKDTree(origin_mesh.vertices)

        labels = np.full(origin_mesh.vertices.shape[0], label_map['skin'], dtype=np.int32)
        components = ['top', 'bottom', 'shoes']
        for component in components:
            component_dir = os.path.join(parent_dir, component)
            component_mesh = read_mesh_from_dir(component_dir)
            if component_mesh is None:
                print("No component %s mesh" % component)
                continue

            print("Finding corresponding vertices: %s/%s" % (dir_name, component))

            # Calculate the distances between each vertex
            min_dists, idx = tree.query(component_mesh.vertices)

            # Give warning if the minimum distance is to large
            rought_idx = idx[min_dists > epsilon]
            rought_dists = min_dists[min_dists > epsilon]
            if 0 < rought_idx.size and rought_idx.size <= max_outliers:
                print("Warning: Cannot find the accurate correspondence for the following vertices, take the nearest vertex alternatively:")
                print(" ".join(["(%d, %0.2f)" % (i, d) for i, d in zip(rought_idx, rought_dists)]))
            elif rought_idx.size > max_outliers:
                write_log("Fatal: Too many outliers(%d), cannot match two mesh: %s -> %s" % (rought_idx.size, component, dir_name), True)

            labels[idx] = label_map[component]
    
            print("Successfully labeled %d vertices for %s/%s" % (idx.shape[0], dir_name, component))

        # Serialize to string
        label_name = dir_name + "_labelsV.txt"
        label_path = os.path.join(parent_dir, label_name)
        lines = [str(i)+'\n' for i in labels]
        with open(label_path, 'w') as f:
            f.writelines(lines)
        print("Successfully write labels to %s" % label_path)

    