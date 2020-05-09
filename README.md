# GCN Based learning on 3D Data for Segmentation

## Introduction

A research project on graph convolutional network, building a multi-scale model for learning method on 3D meshes, point cloud, etc., empowering classification, prediction tasks on 3D data.

In this work, we adopt GAT and FeaStNet based GCN architecture with task-specified multi-scale design to semantically segment high-resolution meshes of objects and human bodies. One of our main targets is to classify points from clothes and human skins and further tell the type of garments dressed on for each human part.

With this, we can fit a naked human model (via SMPL, etc.) and simulate the clothes segments separately on the human model. Once we obtain a rough human point cloud by 3D scanning, we can vividly simulate a realist human avatar.

In addition to the source code provided here, we also prepared a piece of [slides](https://peihaowang.github.io/archive/Wang_GCN_Segmentation_2019_slides.pdf) for a brief introduction.

## Source Code

We provide the code for our experiments under the `code` folder. Before running our code, please make sure these python or jupyter dependencies are well installed and configured.

```
numpy
scipy
trimesh
matplotlib
Pillow
OpenCV2
tensorflow>=2.0.0
tensorflow_graphics
tensorboard
threevis
```

Our code was adapted from "Geometry — 3D convolutions and pooling", a demo of `tensorflow_graphics`, and you can find the [official Colab notebook here](https://colab.research.google.com/github/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/mesh_segmentation_demo.ipynb).

However, compared with this exisiting model in Tensorflow for human body segementation, we notice that our mesh have more noises and the number of points are prohibitively larger. To this end, we employ an efficient downsampling algorithm and borrow the multi-layer architecture from FeaStNet. The code was implemented in Tensorflow 2.0.

To account for the usage of each file, we establish the following list.

* `gcn_mesh_learning.ipynb`: The main program including training and testing procedures. This file was adapted from an official demo of Tensorflow Graphics, as mentioned before.

* `custom_dataio.py`: Custom data IO for 3D mesh loading. In addition to 3D meshes, it will also load auxiliary precomputed data, e.g. subsampling correspondence. We adapted a part of source code from Tensorflow Graphics for our certain purpose.

* `coarsening.py`: The multi-level graph coarsening algorithm adapted from [Wang and Gan's code](https://github.com/yuangan/3D-Shape-Segmentation-via-Shape-Fully-Convolutional-Networks).

* `compute_metis.py`: Precompute correspondence on graphes of multiple levels for downsampling and upsampling. It will output a sparse representation dumped as additional files.

* `convert_mesh_tfrecords.py`: Convert `.obj` files into tensors, including point positions (vertices) and faces (edges), and export them to `.tfrecords` file for network IO.

* `label_face_to_vertex`: Interconvert between per-face labels and per-vertex labels.

* `mesh_loader.py`: Loading 3D mesh from `.obj` files, adapted from the source code of `Trimesh` library.

* `mesh_visualization.py`: Visualization utilities for 3D data powered by `threevis` library.

Files out of the list above are those you do not need to go through before testing this project. We preserve them only for a reference.

We have tested that our code can run on the public dataset [COSEG](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm). Another working dataset for human-clothes segmentation, however, has not been open yet. If you want to know more about the dataset, please drop me an email.

## Survey

Under the `survey` directory are the related papers we have referred to. Please note that there may exist copyright issues if you intend to download them. Please only refer to the name of these papers and download via other channels of yours.

## Reference

1. Verma, Nitika, Edmond Boyer, and Jakob Verbeek. "FeaStNet: Feature-steered graph convolutions for 3d shape analysis." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
2. Wang, Pengyu, et al. "3D shape segmentation via shape fully convolutional networks." Computers & Graphics 70 (2018): 128-139.
3. [Introducing TensorFlow Graphics](https://blog.tensorflow.org/2019/05/introducing-tensorflow-graphics_9.html).

