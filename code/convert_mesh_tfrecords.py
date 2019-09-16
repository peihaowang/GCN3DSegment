import sys, os
import trimesh
import numpy as np
import tensorflow as tf
import coarsening

import mesh_loader

g_dataset_dir = "cosegCup/train"
g_dataset_kind = "coseg"
# g_dataset_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Static"
g_output_path = "psbCup.train.tfrecords"

# Supported mesh format
g_mesh_exts = [".obj", ".off", ".ply"]

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if not isinstance(value, list): value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list): value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list): value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(mesh, labels, metis):
    vertices = tf.convert_to_tensor(mesh.vertices, dtype=tf.float32)
    triangles = tf.convert_to_tensor(mesh.faces, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    metis = [tf.convert_to_tensor(m, dtype=tf.int32) for m in metis]


    # Feature description, create a dictionary mapping the feature name to
    # the tf.Example-compatible data type
    feature = {
        'num_vertices': _int64_feature(mesh.vertices.shape[0])
        , 'num_triangles': _int64_feature(mesh.faces.shape[0])
        , 'vertices': _bytes_feature(tf.io.serialize_tensor(vertices).numpy())
        , 'triangles': _bytes_feature(tf.io.serialize_tensor(triangles).numpy())
        , 'labels': _bytes_feature(tf.io.serialize_tensor(labels).numpy())
        , 'metis': _bytes_feature([tf.io.serialize_tensor(m).numpy() for m in metis])
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_mesh(writer, paths):
    # Check if required path provided and valid
    keys_required = ('mesh', 'label')
    for key in keys_required:
        if key not in paths.keys():
            return False
        if not os.path.exists(paths[key]):
            return False

    # Read mesh via trimesh

    # Disable error output
    # sys.stderr = None
    mesh = mesh_loader.load(paths['mesh'], raw_mesh=True)
    # sys.stderr = sys.__stderr__

    if not isinstance(mesh, trimesh.Trimesh):
        return False

    # Read labels line by line
    with open(paths['label'], 'r') as f:
        # Note that the label id starts from 1 in txt files
        labels = np.array([int(line.strip())-1 for line in f], dtype=np.int32)

    # Read metis data line by line
    with open(paths['metis'], 'r') as f:
        metis = [np.array([int(n) for n in line.strip().split()], dtype=np.int32) for line in f]

    example = serialize_example(mesh, labels, metis)
    writer.write(example)

    return True

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

        label_name = dir_name + '_labelsV.txt'
        metis_name = dir_name + '_metis.txt'

        paths = {}
        paths['mesh'] = os.path.join(dir_path, mesh_name)
        paths['label'] = os.path.join(dir_path, label_name)
        paths['metis'] = os.path.join(dir_path, metis_name)

        yield dir_name, paths

def traverse_coseg_dataset(root_dir):
    for filename in os.listdir(root_dir):
        # Skip hidden files
        if filename.startswith('.'): continue

        base_name, ext_name = os.path.splitext(filename)

        if ext_name.lower() not in g_mesh_exts:
            continue

        label_name = base_name + '_labelsV.txt'
        metis_name = base_name + '_metis.txt'

        paths = {}
        paths['mesh'] = os.path.join(root_dir, filename)
        paths['label'] = os.path.join(root_dir, label_name)
        paths['metis'] = os.path.join(root_dir, metis_name)

        yield base_name, paths

def traverse_dataset(root_dir, kind):
    if kind.lower() == 'static':
        yield from traverse_static_dataset(root_dir)
    elif kind.lower() == 'dynamic':
        yield from traverse_dynamic_dataset(root_dir)
    elif kind.lower() == 'coseg':
        yield from traverse_coseg_dataset(root_dir)

if __name__ == "__main__":

    # Enable eager execution, operation should be executed immediately
    tf.enable_eager_execution()

    # Aggregate each mesh and its label, then serialize to file
    with tf.io.TFRecordWriter(g_output_path) as writer:
        for name, paths in traverse_dataset(g_dataset_dir, g_dataset_kind):
            # Write to tfrecord file
            if serialize_mesh(writer, paths):
                print("Successfully serialize mesh: %s" % name)
            else:
                print("Failed to serialize mesh: %s" % name)




