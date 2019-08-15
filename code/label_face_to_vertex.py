import sys, os
import trimesh
import numpy as np

g_dataset_dir = "psbCup/test"

def parse_per_label_labels(label_path, face_num):
    label_id = 0
    label_arr = np.full(face_num, -1, dtype=np.int32)

    print(">>>>>>", label_path, face_num, label_arr.shape)
    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                # Note that label id should start from 1
                label_id += 1
            else:
                faces = [int(vert) - 1 for vert in line.split()]
                print(">>>>>>", i, label_arr.shape)
                label_arr[faces] = label_id
                print(">>>>>>", i, label_arr.shape)

    # Check labels, if there are vertices not initialized
    abnormals = label_arr[label_arr==-1]
    if abnormals.shape[0] != 0:
        print("Warning: The label file doesn't annotate all the vertices.")

    return label_arr


def parse_per_face_labels(label_path):
    # Read labels line by line
    with open(label_path, 'r') as f:
        label_arr = np.array([int(line.strip()) for line in f], dtype=np.int32)
    return label_arr

if __name__ == "__main__":

    # Supported mesh format
    mesh_exts = [".obj", ".off", ".ply"]

    # Enumerate files
    files = os.listdir(g_dataset_dir)

    # Aggregate each mesh and its label, then serialize to file
    for filename in files:
        base_name, ext_name = os.path.splitext(filename)

        if ext_name not in mesh_exts:
            continue

        mesh_path = os.path.join(g_dataset_dir, filename)

        if not os.path.exists(mesh_path):
            print("Failed to convert labels for mesh: %s" % base_name)
            continue

        # Read mesh via trimesh
        mesh = trimesh.load(mesh_path)

        # Read face labels
        old_label_name = base_name + '_labelsN.txt'
        new_label_name = base_name + '_labelsV.txt'

        old_label_path = os.path.join(g_dataset_dir, old_label_name)
        new_label_path = os.path.join(g_dataset_dir, new_label_name)

        if os.path.exists(old_label_path):
            labels = parse_per_face_labels(old_label_path)
        else:
            old_label_name = base_name + '_labels.txt'
            old_label_path = os.path.join(g_dataset_dir, old_label_name)
            if not os.path.exists(old_label_path):
                print("Failed to convert labels for mesh: %s" % base_name)
                continue

            labels = parse_per_label_labels(old_label_path, mesh.faces.shape[0])
            tmp_label_path = os.path.join(g_dataset_dir, base_name + '_labelsN.txt')
            vs = [str(i)+'\n' for i in labels]
            with open(tmp_label_path, 'w') as f:
                f.writelines(vs)

        # Map face correspondences to label correspondences
        vfunc = np.vectorize(lambda i: labels[i] if i >= 0 else i)
        m = vfunc(mesh.vertex_faces)

        # Find the mode to represent the final label of each vertex
        # 1. Filter out paddings
        w = (m>=0)
        m = [m[i][w[i]] for i in range(m.shape[0])]
        # 2. Find the mode
        [np.max(np.bincount(m[i])) for i in range(len(m))]
        m = [np.argmax(np.bincount(m[i])) for i in range(len(m))]

        # Serialize to string
        m = [str(i)+'\n' for i in m]
        with open(new_label_path, 'w') as f:
            f.writelines(m)

        print("Successfully convert labels for mesh: %s" % base_name)



