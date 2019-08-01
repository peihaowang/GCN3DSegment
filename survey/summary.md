
# Survey

## A Review on Mesh Segmentation Techniques (2008 Computer Graphics Forum)

基于 features 和 limitation 的 mesh 分割方法，提出将 mesh 以面为节点转化为图数据结构，将 mesh 的分割简化为图的 partition. 文章主要考虑 cardinality, geometric 和 topological 的限制条件，结合 planarity, geometry perspective, normal, surface, slipping, symmetry, convexity or concavity, medial axis, shape diameter function, motion characteristics 这九个 features，利用 Region Growing, Clustering, GraphCut 等算法优化，得到分割结果，属于非 learning 的 mesh 分割方法。

论文中有对测地距离(Geodesic distance), 曲率(Curvature), SDF 等特征值的讨论，在未来 feature engineering 中有较大参考价值。

## Learning 3D Mesh Segmentation and Labeling (2010 TOG)

3D Mesh 分割的经典论文！论文利用监督学习方法对模型进行分割，得到较好效果。论文使用 CRF 作为学习模型，JointBoost作为模型优化算法。模型同时优化 unary energy term 和 pairwise energy term, 前者代表某一个面片的 energy 而后者代表两个相邻面片的联合 energy. Feature 分为 unary, contextual label, pairwise 三类，学习过程分为两步，第一步将 unary, pairwise feature 送入 JointBoost 中进行学习/预测，再将分类结果并入 features 再次送入 JointBoost 中。第一步获得的学习结果作为第二步学习的先验信息，即结合相邻面片的分类先验和原本的 feature 预测面片分类。

## Unsupervised Co-Segmentation of a Set of Shapes via Descriptor-Space Spectral Clustering (2011 Siggraph Asia)

该论文提出利用非监督学习的方法对模型进行分类。论文中提到，针对相同一类物体模型，不同部位可能拥有较大的几何差异，如瓶子的把柄等，使用表述性质的 feature 更能得到好的分割结果。在无标注的数据集中，模型各部分的 shape 特征被提取出后，将这个集合进行聚类运算，得到分类结果。

## A complete system for garment segmentation and color classification (2013 MVAP)

基于机器学习和 GMM 分割的抠像算法，目标对于单张 RGB 图片，抠取其中衣物部分。基于颜色的衣服种类分类。

## 3D Garment Segmentation Based on Semi-supervised Learning Method (2015 Journal of Fiber Bioengineering and Informatics)

该论文提出了半监督机器学习方法，用于服装模型的分割。论文指出，监督学习需要大量的标注数据，已有的模型分割方法多数基于基本几何优化或监督学习。本文应用 CRF 模型，利用 JointBoost 学习算法达到模型优化效果。论文结果部分显示，分类有较好结果。

文章中的服装模型类似人工建模，模型表面较为平整，真实性不高。更为重要的是，该文章中输入是衣服套装模型，没有人体四肢和躯干，分割难度较低。

## 3D Shape Segmentation with Projective Convolutional Networks (2017 CVPR)

论文利用神经网络完成对模型分割。网络要求输入模型及其多个通过计算得到的 viewpoints, 通过这些 viewpoints 得到模型各个视角的渲染图(RGB)和深度图(Depth), 随后利用 FCN 网络得到每个视角分类的 confidence map, 反投影至 3D 模型上后得到呈三维分布的 confidence map, 最后通过 surface-based CRF 进行优化降噪，得到最终分类效果。

## 3D Graph Neural Networks for RGBD Semantic Segmentation (2017 ICCV)

基于 GNN 的 RGBD 三维场景分割算法。

## PointGrid: A Deep Network for 3D Shape Understanding (2018 CVPR)

论文主要讨论 3D point cloud 的分类分割方法，提出 PointGrid 概念，抛弃原来的 Occupancy grid, 使得模型输入的分辨率大幅降低。 其次论文提出 PointGrid 的分类网络和分割网络架构。文章还提到 3D 模型可以通过特殊的采样方法生成 point cloud, 因此任意三角面片组成的 mesh 可以转换为点云后输入到该论文的网络中进行分类分割。

## 3D Tooth Segmentation and Labeling using Deep Convolutional Neural Networks (2018 IEEE Transactions on Visualization and Computer Graphics)

基于 Learning 3D Mesh Segmentation and Labeling 的牙齿 mesh 分割论文，其 feature 的选取与之类似，但引入了较多前期和后期的 mesh 处理，如 mesh simplification 和 graph-cut optimization. 与前文不同的是，该文章使用 CNN 网络进行机器学习。

## Semantic Labeling and Instance Segmentation of 3D Point Clouds using Patch Context Analysis and Multiscale Processing (2019 IEEE Transactions on Visualization and Computer Graphics)

基于 clutsering, patch context 的无监督学习，对室内家具、物件分类分割的方法。