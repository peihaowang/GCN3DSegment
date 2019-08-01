
# Clothing Segmentation and Classification

## Introduction

作为虚拟现实技术的热门应用，虚拟试衣通常要求拍摄者借助虚拟影像设备，任意切换着装，并得到立体可视化预览。基于 3D 扫描的人体动态重建技术越发成熟，如[11][12]，高精度的采集设备也日益普及。[13][14]中介绍的的布料动态仿真技术，能够模拟服装在动态人体中的物理特性，使得这项应用的展示效果更加接近真实。同时，[15]中介绍的参数化人体体型还原算法，能预测衣服下方人体体型，从而辅助尺寸选择和增强衣物仿真效果。 但衣服模型的重建与分离却少有研究与进步。目前服装模型的生成主要依赖于手工制作，但这种方法效率较低，同时制作细节教差，难以匹配人体重建、布料动态仿真等技术的真实性。此外，在一些人体动态重建场景中，拍摄者被要求穿着较为紧身单薄的衣服进行拍摄，后期加入制作好的衣服模型，进行布料仿真，还原衣物的动态真实性。此类方法流程并不直接，其次后期处理难度较大。因此，在人体模型采集、生成的同时，分离出服装模型的技术尤为重要，它能简化拍摄工序，结合人体体型估计算法[15]和布料动态仿真技术[13][14]，一次拍摄后，即产生高真实度的动态人体模型。它也能应用于服装模型的采集与生成中，对穿好待采集服装的假人模特进行衣着面片的抠取，批量获取衣服模型。此外，[16]介绍的衣物 fitting 方法能够复用分割得到的服装模型，使得采集得到的大量服装模型能够得到更广泛的应用。

现有的算法鲜有直接讨论将三维人体与衣服进行分割。较为接近的有[6]中介绍的对衣服模型(不含人体)进行上下衣及其相关部位(如裙摆、长袖)的分割，但不足以解决上述提到的问题。此外，[7]介绍了用于 fashion 图片数据集的时装抠取算法，但未扩展至 3D 模型的分割。现存的大量 3D 分割算法，一般应用于 mesh 分割和点云分割。Mesh 分割方法，如[1][2][3][4]，涵盖面过于宽泛，并不专注于人体与服装的切割，对细节较多，衣着复杂、人体姿态变化较大的模型难以达到理想效果；点云切割算法，如[8][9]，能够较好的进行细节识别、全局优化，但多应用于界限模糊的室内点云图的分割与分类。

我们更多地参考了 mesh 分割方法，发现其对模型各部位的分割通常基于对几何、拓扑、基数等特征的识别，而衣物与肢体也恰好在几何、拓扑甚至是纹理等方面均存在大小差异。同时，我们参考[4]的方法，将模型进行多视角投影，将纹理、深度映射成为二维视图的同时，我们将每个面的几何、拓扑、骨骼特征也投影映射至二维平面。我们将每一个视角的多张特征图像送入 GraphNet/ShapeNet/CNN/FCN 中得到分类置信图(Confidence Map)，并将其反投影至原模型，映射重叠部分进行平均处理，得到最终人体模型分类置信图(Confidence Map)。最后在模型的初步分割结果上再执行图割(Graph Cut)和边缘优化。

## Related Work

### Mesh Segmentation

关于三维分割的方法较多，早期 Wu et al. 阐述的分割方法结合基数(Cardinality)、几何(Geometrical)、拓扑(Topological)等限制条件，和平坦度(Planarity)、曲率(Curvature)、测地距离(Geodesic distances)等特征值，通过区域增长(Region Growing)、聚类(Clustering)和图割(Graph Cut)等优化算法得到分割结果[2]。Kalogerakis 则利用条件随机场(CRF)和JointBoost，将学习/预测分为两步，第一步利用几何、拓扑等单元特征(Unary feature)获得初次分类结果，第二步则整合第一步的分类结果作为先验，计算与相邻面片的联合二元特征(Pairwise feature)，并与单元特征(Unary feature)结合，再次得到分类结果[1]。[5] 在 Kalogerakis 的方法基础上，将算法融入了模型简化(Mesh simplification) 和 分类优化(Labeling optimization)，并移植到牙齿数据集(Dental Dataset)上，获得较好的分类效果。O. Sidi et al. 提出合作分析(Co-analysis)的非监督学习方法，分割同一类物品模型。该算法将单个模型提取描述类参数(Descriptor)并做聚类处理，得到一个模型的分割，再将这些分割结果整合到一个集合，再次进行聚类，并从每个分类中计算统计指标，用获得分割的概率模型[3]。[4] 则是通过多视角投影(Multi-view projection)的方法，将模型各视角的渲染图像(Shaded map)和深度图像(Depty map)送入 FCN, 得到各视角的置信图(Confidence map)，反映射至三维模型，得到模型的分类结果。Le 和 Duan 提出基于 CNN 的 PointGrid 分类分割网络，将 mesh 模型采样成为点云，并引入点量化(Point quantization)的方法，能够进行精准分类分割[10]。

### Cloth Segmentation

已有衣服模型分割方法较少，Huang et al. 阐述的方法只针对服装模型(不带有人体)进行分割，得到衣服的各部位分类结果。其沿用了 Kalogerakis 所提出的特征量和条件随机场(CRF)方法，通过带条件熵(Conditional entropy)的目标函数(Objective function)进行半监督学习[6]。[7] 讨论了基于时装图像数据集(Fashion Dataset)的服装抠取算法，通过随机森林(Random Forest)的学习方法结合 GMM 分割进行抠像，而衣服的分类则主要基于颜色值。该算法也仅讨论对图片中上衣的抓取，

1. Kalogerakis, Evangelos, Aaron Hertzmann, and Karan Singh. "Learning 3D mesh segmentation and labeling." ACM Transactions on Graphics (TOG). Vol. 29. No. 4. ACM, 2010.

2. Wu, Ke & Rashad, Medhat & Khamiss, Mohamed. "A Review on Mesh Segmentation Techniques." International Journal of Engineering and Innovative Technology (IJEIT). Vol. 6. 2017. 

3. Sidi, Oana, et al. Unsupervised co-segmentation of a set of shapes via descriptor-space spectral clustering. Vol. 30. No. 6. ACM, 2011.

4. Kalogerakis, Evangelos, et al. "3D shape segmentation with projective convolutional networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

5. Xu, Xiaojie, Chang Liu, and Youyi Zheng. "3D tooth segmentation and labeling using deep convolutional neural networks." IEEE transactions on visualization and computer graphics 25.7 (2018): 2336-2348.

6. Huang, Mian, et al. "3D Garment Segmentation Based on Semi-supervised Learning Method." Journal of Fiber Bioengineering and Informatics 8.4: 657-665.

7. Manfredi, Marco, et al. "A complete system for garment segmentation and color classification." Machine Vision and Applications 25.4 (2014): 955-969.

8. Qi, Xiaojuan, et al. "3d graph neural networks for rgbd semantic segmentation." Proceedings of the IEEE International Conference on Computer Vision. 2017.

9. Hu, Shi-Min, Jun-Xiong Cai, and Yu-Kun Lai. "Semantic Labeling and Instance Segmentation of 3D Point Clouds using Patch Context Analysis and Multiscale Processing." IEEE transactions on visualization and computer graphics (2018).

10. Le, Truc, and Ye Duan. "Pointgrid: A deep network for 3d shape understanding." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

11. Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgib- bon. Kinectfusion: Real-time dense surface mapping and tracking. In Proceedings of the IEEE International Symposium on Mixed and Augmented Reality, pages 127–136. IEEE, 2011.

12. Zhen-zhong Lan, Lei Bao, Shoou-I Yu, Wei Liu, and Alexander G Hauptmann. Double fusion for multimedia event detection. In Proceedings of the International Conference on Multimedia Modeling, pages 173–185. Springer, 2012.

13. Yu, Tao, et al. "SimulCap: Single-View Human Performance Capture with Cloth Simulation." arXiv preprint arXiv:1903.06323 (2019).

14. Weidner, Nicholas J., et al. "Eulerian-on-lagrangian cloth simulation." ACM Transactions on Graphics (TOG) 37.4 (2018): 50.

15. Alexandru O Bălan and Michael J Black. The naked truth: Estimating body shape under clothing. In Proceedings of the European Conference on Computer Vision, pages 15–29. Springer, 2008.

16. Li, Jituo, et al. "Fitting 3D garment models onto individual human models." Computers & graphics 34.6 (2010): 742-755.

