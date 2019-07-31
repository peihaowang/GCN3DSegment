
# Survey

## A Review on Mesh Segmentation Techniques (2008 Computer Graphics Forum)

基于 features 和 limitation 的 mesh 分割方法，提出将 mesh 以面为节点转化为图数据结构，将 mesh 的分割简化为图的 partition. 文章主要考虑 cardinality, geometric 和 topological 的限制条件，结合 planarity, geometry perspective, normal, surface, slipping, symmetry, convexity or concavity, medial axis, shape diameter function, motion characteristics 这九个 features，利用 Region Growing, Clustering, GraphCut 等算法优化，得到分割结果，属于非 learning 的 mesh 分割方法。

论文中有对测地距离(Geodesic distance), 曲率(Curvature), SDF 等特征值的讨论，在未来 feature engineering 中有较大参考价值。

## Learning 3D Mesh Segmentation and Labeling (2010 TOG)

3D Mesh 分割的经典论文！论文利用监督学习方法对模型进行分割，得到较好效果。论文使用 CRF 作为学习模型，JointBoost作为模型优化算法。模型同时优化 unary energy term 和 pairwise energy term, 前者代表某一个面片的 energy 而后者代表两个相邻面片的联合 energy. Feature 分为 unary, contextual label, pairwise 三类，学习过程分为两步，第一步将 unary, pairwise feature 送入 JointBoost 中进行学习/预测，再将分类结果并入 features 再次送入 JointBoost 中。第一步获得的学习结果作为第二步学习的先验信息，即结合相邻面片的分类先验和原本的 feature 预测面片分类。

## 3D Garment Segmentation Based on Semi-supervised Learning Method (2015 Journal of Fiber Bioengineering and Informatics)

该论文提出了半监督机器学习方法，用于服装模型的分割。论文指出，监督学习需要大量的标注数据，已有的模型分割方法多数基于基本几何优化或监督学习。本文应用 CRF 模型，利用 JointBoost 学习算法达到模型优化效果。论文结果部分显示，分类有较好结果。

文章中的服装模型类似人工建模，模型表面较为平整，真实性不高。更为重要的是，该文章中输入是衣服套装模型，没有人体四肢和躯干，分割难度较低。

## PointGrid: A Deep Network for 3D Shape Understanding (2018 CVPR)

论文主要讨论 3D point cloud 的分类分割方法，提出 PointGrid 概念，抛弃原来的 Occupancy grid, 使得模型输入的分辨率大幅降低。 其次论文提出 PointGrid 的分类网络和分割网络架构。文章还提到 3D 模型可以通过特殊的采样方法生成 point cloud, 因此任意三角面片组成的 mesh 可以转换为点云后输入到该论文的网络中进行分类分割。

## 3D Tooth Segmentation and Labeling using Deep Convolutional Neural Networks (2018 IEEE Transactions on Visualization and Computer Graphics)

基于 Learning 3D Mesh Segmentation and Labeling 的牙齿 mesh 分割论文，其 feature 的选取与之类似，但引入了较多前期和后期的 mesh 处理，如 mesh simplification 和 graph-cut optimization. 与前文不同的是，该文章使用 CNN 网络进行机器学习。

## Semantic Labeling and Instance Segmentation of 3D Point Clouds using Patch Context Analysis and Multiscale Processing (2019 IEEE Transactions on Visualization and Computer Graphics)

基于 clutsering, patch context 的无监督学习，对室内家具、物件分类分割的方法。