import sys, os
import trimesh
import numpy as np

g_dataset_dir = "cosegCup/test"
g_export_per_face_labels = False

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

def parse_per_label_labels(label_path):
    categories = []
    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 2 != 0:
                faces = np.array([int(vert) for vert in line.split()])
                categories.append(faces)

    face_num = max([np.max(cls) for cls in categories])
    label_arr = np.full(face_num, -1, dtype=np.int32)
    for label_id, faces in enumerate(categories):
        # Note that face number starts from 1
        # Note that label id starts from 1
        label_arr[faces - 1] = label_id + 1

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

        if ext_name.lower() not in mesh_exts:
            continue

        mesh_path = os.path.join(g_dataset_dir, filename)

        if not os.path.exists(mesh_path):
            print("Failed to convert labels for mesh: %s" % base_name)
            continue

        # Read mesh via trimesh
        # Note that disable texture to make loading more efficient
        mesh = trimesh.load(mesh_path, resolver=DummyResolver(g_dataset_dir))

        # Read face labels
        new_label_name = base_name + '_labelsV.txt'
        new_label_path = os.path.join(g_dataset_dir, new_label_name)

        label_loaders = {'_labelsN.txt': parse_per_face_labels
            , '_labels.txt': parse_per_label_labels
            , '.seg': parse_per_face_labels
        }
        labels = None
        for suffix, loader in label_loaders.items():
            old_label_name = base_name + suffix
            old_label_path = os.path.join(g_dataset_dir, old_label_name)
            if os.path.exists(old_label_path):
                labels = loader(old_label_path)
                if g_export_per_face_labels and suffix == '_labels.txt':
                    tmp_label_path = os.path.join(g_dataset_dir, base_name + '_labelsN.txt')
                    vs = [str(i)+'\n' for i in labels]
                    with open(tmp_label_path, 'w') as f:
                        f.writelines(vs)

        if labels is None:
            print("Failed to convert labels for mesh: %s" % base_name)
            continue

        # Map face correspondences to label correspondences
        vfunc = np.frompyfunc(lambda i: labels[i] if i >= 0 else i, 1, 1)
        m = vfunc(mesh.vertex_faces).astype(np.int64)

        # Find the mode to represent the final label of each vertex
        # 1. Filter out paddings
        w = (m>=0)
        m = [m[i][w[i]] for i in range(m.shape[0])]
        # 2. Find the mode
        m = [np.argmax(np.bincount(m[i])) for i in range(len(m))]

        # Serialize to string
        lines = [str(i)+'\n' for i in m]
        with open(new_label_path, 'w') as f:
            f.writelines(lines)

        print("Successfully convert labels for mesh: %s" % base_name)



