import sys, os
import trimesh
import numpy as np
import tensorflow as tf

g_dataset_dir = "psbCup/test"
g_output_path = "psbCup.test.tfrecords"

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(mesh, labels):
    vertices = tf.convert_to_tensor(mesh.vertices, dtype=tf.float32)
    triangles = tf.convert_to_tensor(mesh.faces, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Feature description, create a dictionary mapping the feature name to
    # the tf.Example-compatible data type
    feature = {
        'num_vertices': _int64_feature(mesh.vertices.shape[0])
        , 'num_triangles': _int64_feature(mesh.faces.shape[0])
        , 'vertices': _bytes_feature(tf.io.serialize_tensor(vertices).numpy())
        , 'triangles': _bytes_feature(tf.io.serialize_tensor(triangles).numpy())
        , 'labels': _bytes_feature(tf.io.serialize_tensor(labels).numpy())
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_mesh(writer, mesh_path, label_path):
    if not os.path.exists(mesh_path) or not os.path.exists(label_path):
        return False

    # Read mesh via trimesh
    mesh = trimesh.load(mesh_path)

    # Read labels line by line
    with open(label_path, 'r') as f:
        # Note that the label id starts from 1 in txt files
        labels = [int(line.strip())-1 for line in f]
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    example = serialize_example(mesh, labels)
    writer.write(example)

    return True

if __name__ == "__main__":

    # Enable eager execution, operation should be executed immediately
    tf.enable_eager_execution()

    # Supported mesh format
    mesh_exts = [".obj", ".off", ".ply"]

    # Enumerate files
    files = os.listdir(g_dataset_dir)

    # Aggregate each mesh and its label, then serialize to file
    with tf.io.TFRecordWriter(g_output_path) as writer:
        for filename in files:
            base_name, ext_name = os.path.splitext(filename)

            if ext_name not in mesh_exts:
                continue

            label_name = base_name + '_labelsV.txt'

            mesh_path = os.path.join(g_dataset_dir, filename)
            label_path = os.path.join(g_dataset_dir, label_name)

            # Write to tfrecord file
            if serialize_mesh(writer, mesh_path, label_path):
                print("Successfully serialize mesh: %s" % base_name)
            else:
                print("Failed to serialize mesh: %s" % base_name)




