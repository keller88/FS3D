# A Comprehensive Survey on Few-shot 3D Point Cloud Learning:

Classification, Segmentation and Object Detection

A curated list of papers and resources for **Few-Shot 3D Point Cloud Learning**, accompanying our survey paper:

> **A Comprehensive Survey on Few-shot 3D Point Cloud Learning:
> Classification, Segmentation and Object Detection**  
> **[Paper Link]()** 

This repository collects recent work on few-shot learning for 3D point clouds. We organize methods by three core tasks:

- Few-shot 3D Point Cloud Classification (FS3D-C)
- Few-shot 3D Point Cloud Segmentation (FS3D-S)
- Few-shot 3D Point Cloud Object Detection (FS3D-D)

If you find missing papers or mistakes, feel free to open an issue or pull request.  

---

## 💡 Methodology Abbreviation

| Abbreviation  | Full Name                       |
| :------------ | :------------------------------ |
| **Data Aug.** | Data Augmentation               |
| **Metric**    | Metric Learning                 |
| **Meta**      | Meta-Learning                   |
| **SSP**       | Self-Supervised Pre-training    |
| **PEFT**      | Parameter-Efficient Fine-Tuning |

## 📊 Benchmarks

A list of commonly used benchmark datasets for 3D point cloud understanding.

| Benchmark                                                    | Project                                                     | Publication |
| :----------------------------------------------------------- | :---------------------------------------------------------- | :---------- |
| [**ModelNet40 / ModelNet10**](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Wu_3D_ShapeNets_A_2015_CVPR_paper.html) | [Homepage](http://modelnet.cs.princeton.edu/)               | CVPR 2015   |
| [**ShapeNet**](https://arxiv.org/abs/1512.03012)             | [Homepage](https://shapenet.org/)                           | CVPRW 2015  |
| [**ScanNet**](https://www.cv-foundation.org/openaccess/content_cvpr_2017/html/Dai_ScanNet_Richly-Annotated_3D_2017_CVPR_paper.html) | [Homepage](http://www.scan-net.org/)                        | CVPR 2017   |
| [**S3DIS**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Armeni_3D_Semantic_Parsing_2016_CVPR_paper.html) | [Homepage](http://buildingparser.stanford.edu/dataset.html) | CVPR 2016   |
| [**ScanObjectNN**](https://openaccess.thecvf.com/content_ICCVW_2019/html/Object-CV/Uy_Revisiting_Point_Cloud_Classification_A_New_Benchmark_Dataset_and_ICCVW_2019_paper.html) | [Homepage](https://hkust-vgd.github.io/scanobjectnn/)       | ICCVW 2019  |
| [**PartNet**](https://openaccess.thecvf.com/content_CVPR_2019/html/Mo_PartNet_A_Large-Scale_Benchmark_for_Fine-Grained_and_Hierarchical_Part-Level_CVPR_2019_paper.html) | [Homepage](https://partnet.cs.stanford.edu/)                | CVPR 2019   |
| [**SemanticKITTI**](https://openaccess.thecvf.com/content_ICCV_2019/html/Behley_SemanticKITTI_A_Dataset_for_Semantic_Scene_Understanding_of_LiDAR_Sequences_ICCV_2019_paper.html) | [Homepage](http://www.semantic-kitti.org/)                  | ICCV 2019   |

------

##

---

## 📖 Few-Shot 3D Point Cloud Classification (FS3D-C)

| Year | Title / Paper                                                | Code                                                         | Methodology       |
| :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :---------------- |
|      | [**3D-FSCIL: Few-shot Class-incremental Learning for 3D Point Cloud Objects**](https://arxiv.org/pdf/2205.15225) | [Code](https://github.com/townim-faisal/FSCIL-3D)            | Model Adaptation  |
|      | [**3DFFL: privacy-preserving Federated Few-Shot Learning on 3D Point Clouds**](https://www.nature.com/articles/s41598-024-70363-5) | [Code](https://github.com/guoyukun2003/3DFFL)                | SSP               |
|      | [**A Closer Look at Few-Shot 3D Point Cloud Classification**](https://arxiv.org/pdf/2303.18210) | [Code](https://github.com/charlisye/FS-3D-PointCloud)        | Metric Learning   |
|      | [**Adapt PointFormer: 3D Point Cloud Analysis via Adapting 2D Visual Transformers**](https://arxiv.org/pdf/2407.13200) | [Code](https://vcc.tech/research/2024/PointFormer)           | PEFT-Additive     |
|      | [**BGS-Net: fine-grained classification networks based on handcrafted features and rotation-invariant MAE**](http://www.txxb.com.cn/EN/abstract/abstract243157.shtml) | `N/A`                                                        | SSP               |
|      | [**Bi-directional Few-Shot Data Augmentation for 3D Point Cloud Classification**](https://arxiv.org/pdf/2501.02230) | `N/A`                                                        | Data Augmentation |
|      | [**Cascade Graph Neural Networks for Robust Few-shot 3D Point Cloud Classification**](https://openaccess.thecvf.com/content/WACV2025/papers/Hu_Cascade_Graph_Neural_Networks_for_Robust_Few-shot_3D_Point_Cloud_WACV_2025_paper.pdf) | `N/A`                                                        | Metric Learning   |
|      | [**CLIP-based Point Cloud Classification via Point Cloud to Image Translation**](https://arxiv.org/pdf/2408.03545) | `N/A`                                                        | PEFT-Additive     |
|      | [**CLIP Goes 3D: Leveraging Prompt Tuning for Language-Grounded 3D Recognition**](https://openaccess.thecvf.com/content/ICCV2023/papers/Hegde_CLIP_Goes_3D_Leveraging_Prompt_Tuning_for_Language-Grounded_3D_Recognition_ICCV_2023_paper.pdf) | `N/A`                                                        | PEFT-Additive     |
|      | [**CLR-GAM: Contrastive Point Cloud Learning with Guided Augmentation and Feature Mapping**](https://arxiv.org/pdf/2302.14306) | [Code](https://github.com/srikanth-malla/CLR-GAM)            | Data Augmentation |
|      | [**Compositional prototype learning for few-shot 3D point cloud classification**](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cje2.12589) | `N/A`                                                        | Metric Learning   |
|      | [**CrossMoCo: Multi-modal Momentum Contrastive Learning for Point Cloud**](https://openaccess.thecvf.com/content/CVPR2023W/MULA/papers/Paul_CrossMoCo_Multi-Modal_Momentum_Contrastive_Learning_for_Point_Cloud_CVPRW_2023_paper.pdf) | [Code](https://github.com/anirudh257/CrossMoCo)              | SSP               |
|      | [**DAPT: Dynamic Adapter Meets Prompt Tuning: Parameter-Efficient Transfer Learning for Point Cloud Analysis**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Dynamic_Adapter_Meets_Prompt_Tuning_Parameter-Efficient_Transfer_Learning_for_Point_CVPR_2024_paper.pdf) | [Code](https://github.com/LMD0311/DAPT)                      | PEFT-Hybrid       |
|      | [**DMMPT: Dynamic Multimodal Prompt Tuning: Boost Few-Shot Learning with VLM-Guided Point Cloud Models**](https://arxiv.org/pdf/2407.07659) | [Code](https://github.com/eminentgu/DMMPT)                   | PEFT-Hybrid       |
|      | [**Elemental Composable Prototype Networks for Few-shot 3D Point Cloud Classification**](https://openaccess.thecvf.com/content/WACV2025/papers/De_Elemental_Composable_Prototype_Networks_for_Few-shot_3D_Point_Cloud_Classification_WACV_2025_paper.pdf) | `N/A`                                                        | Metric Learning   |
|      | [**Enriching 3D Object Detection in Autonomous Driving with Datasets from 2D via Foundation Models**](https://ieeexplore.ieee.org/abstract/document/10796674) | `N/A`                                                        | Model Adaptation  |
|      | [**Few Shot Learning for Point Cloud Data Using Model Agnostic Meta Learning**](https://ieeexplore.ieee.org/document/9239331) | [Code](https://github.com/RishiPuri/Few-Shot-Learning-on-Point-Cloud) | Meta-Learning     |
|      | [**Fine-grained Prototypical Networks for Few-shot 3D Point Cloud Classification**](https://cdn.techscience.press/files/cmc/2025/TSP_CMC-78-2/TSP_CMC_44088/TSP_CMC_44088.pdf) | `N/A`                                                        | Metric Learning   |
|      | [**GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model**](https://arxiv.org/pdf/2505.04119) | [Code](https://github.com/zhoujiahuan1991/ICML2025-GAPrompt) | PEFT-Additive     |
|      | [**Geometric feature embedding for 3D FSCIL**](https://arxiv.org/pdf/2504.16023) | [Code](https://github.com/songw-zju/PointLoRA)               | PEFT-Reparam.     |
|      | [**GPr-Net: Geometric Prototypical Network for Point Cloud Few-Shot Learning**](https://openaccess.thecvf.com/content/CVPR2023/papers/Anvekar_GPr-Net_Geometric_Prototypical_Network_for_Point_Cloud_Few-Shot_Learning_CVPR_2023_paper.pdf) | [Code](https://github.com/tejanvekar/Gpr-Net)                | Metric Learning   |
|      | [**HPR: Hyperbolic Prototype Rectification for Few-Shot 3D Point Cloud Classification**](https://arxiv.org/pdf/2303.00336) | [Code](https://github.com/Y-Z-F/HPR)                         | Metric Learning   |
|      | [**IDPT: Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models**](https://openaccess.thecvf.com/content/ICCV2023/papers/Zha_Instance-aware_Dynamic_Prompt_Tuning_for_Pre-trained_Point_Cloud_Models_ICCV_2023_paper.pdf) | [Code](https://github.com/zyh16143998882/ICCV23-IDPT)        | PEFT-Additive     |
|      | [**IVLM: LiDAR-to-image projection; VLM few-shot prompting**](https://arxiv.org/pdf/2502.04092) | `N/A`                                                        | Meta-Learning     |
|      | [**LBDA: Bi-directional FSDA augmentation; SAW weighting**](https://arxiv.org/pdf/2411.02218) | `N/A`                                                        | Data Augmentation |
|      | [**MCNet: GLSNM + MIRM multi-level consistency**](https://arxiv.org/pdf/2410.17852) | `N/A`                                                        | SSP               |
|      | [**Meta Episodic learning with Dynamic Task Sampling for CLIP-based Point Cloud Classification**](https://arxiv.org/pdf/2404.00857) | `N/A`                                                        | PEFT-Hybrid       |
|      | [**MetaSets: Meta-Learning on Point Sets for Generalizable Representations**](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_MetaSets_Meta-Learning_on_Point_Sets_for_Generalizable_Representations_CVPR_2021_paper.pdf) | [Code](https://github.com/med-air/MetaSets)                  | Meta-Learning     |
|      | [**MM-Point: Multi-modal self-supervised contrast; view fusion**](https://arxiv.org/pdf/2410.13589) | [Code](https://github.com/Anonymous/MM-Point)                | SSP               |
|      | [**PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter**](https://arxiv.org/pdf/2505.20941) | [Code](https://github.com/zyh16143998882/PMA)                | PEFT-Additive     |
|      | [**Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling**](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_CVPR_2022_paper.pdf) | [Code](https://github.com/lulutang0608/Point-BERT)           | SSP               |
|      | [**PointCLIP: Point Cloud Understanding by CLIP**](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.pdf) | [Code](https://github.com/g-sharma/prifit)                   | PEFT-Additive     |
|      | [**PointGST: Parameter-Efficient Fine-Tuning in Spectral Domain for Point Cloud Learning**](https://arxiv.org/pdf/2410.08114) | [Code](https://github.com/jerryfeng2003/PointGST)            | PEFT-Additive     |
|      | [**Point-LGMask: Local/global SSL with multi-ratio masking**](https://arxiv.org/pdf/2411.09642) | `N/A`                                                        | SSP               |
|      | [**PointLoRA: Low-Rank Adaptation with Token Selection for Point Cloud Learning**](https://arxiv.org/pdf/2504.16023) | [Code](https://github.com/songw-zju/PointLoRA)               | PEFT-Reparam.     |
|      | [**PointMixup: Augmentation for point clouds**](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_20) | [Code](https://github.com/yunlu-chen/PointMixup)             | Data Augmentation |
|      | [**PointOfView: Point–multi-view fusion; sqMA few-shot head**](https://arxiv.org/pdf/2412.02203) | `N/A`                                                        | Metric Learning   |
|      | [**Point-PEFT: Parameter-Efficient Fine-Tuning for 3D Pre-trained Models**](https://ojs.aaai.org/index.php/AAAI/article/view/28409) | [Code](https://github.com/Ivan-Tang-3D/Point-PEFT)           | PEFT-Hybrid       |
|      | [**PointSets: Training-free non-parametric pole classifier**](https://arxiv.org/pdf/2501.09709) | `N/A`                                                        | Metric Learning   |
|      | [**PPCITNet: CLIP-based Point Cloud Classification via Point Cloud to Image Translation**](https://arxiv.org/pdf/2408.03545) | `N/A`                                                        | PEFT-Additive     |
|      | [**PriFit: Learning to Fit Primitives Improves Few Shot Point Cloud Segmentation and Classification**](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14605) | [Code](https://github.com/g-sharma/prifit)                   | SSP               |
|      | [**PPT: Positional Prompt Tuning for Efficient 3D Representation Learning**](https://arxiv.org/pdf/2408.11567) | [Code](https://github.com/zsc000722/PPT)                     | PEFT-Additive     |
|      | [**Self-Supervised Few-Shot Learning on Point Clouds**](https://proceedings.neurips.cc/paper/2020/file/6172c5b96b274246ba7288688461fa5b-Paper.pdf) | [Code](https://github.com/charusharma1992/SSFSL-on-Point-Clouds) | SSP               |
|      | [**SimpliMix: Simplified manifold mixup; episodic feature interpolation**](https://arxiv.org/pdf/2410.11234) | [Code](https://github.com/xxx/SimpliMix)                     | Data Augmentation |
|      | [**TZSL-3D: Transductive Zero-Shot Learning for 3D Point Cloud Classification**](https://ieeexplore.ieee.org/document/9301475) | `N/A`                                                        | Metric Learning   |
|      | [**ViewNet: A Novel Projection-Based Backbone With View Pooling for Few-Shot Point Cloud Classification**](https://arxiv.org/pdf/2303.18210) | [Code](https://github.com/Flying-in-sky/ViewNet)             | Metric Learning   |
|      | [**What Makes for Effective Few-Shot Point Cloud Classification?**](https://openaccess.thecvf.com/content/WACV2022/papers/Ye_What_Makes_for_Effective_Few-Shot_Point_Cloud_Classification_WACV_2022_paper.pdf) | [Code](https://github.com/charlisye/FS-3D-PointCloud)        | Metric Learning   |



---

## 📖 Few-Shot 3D Point Cloud Segmentation (FS3D-S)

| Year | Title / Paper                                                | Code                                                 | Methodology       |
| :--- | :----------------------------------------------------------- | :--------------------------------------------------- | :---------------- |
|      | [**Compositional Prototype Network with Multi-View Correlation for Few-Shot 3D Point Cloud Segmentation**](https://arxiv.org/pdf/2012.14255) | `N/A`                                                | Metric Learning   |
|      | [**Few-shot 3D Point Cloud Semantic Segmentation**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Few-Shot_3D_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.pdf) | [Code](https://github.com/zhaon-ai/attMPTI)          | Meta-Learning     |
|      | [**Few-shot Meta-learning on Point Cloud for Semantic Segmentation**](https://arxiv.org/pdf/2104.02979) | `N/A`                                                | Meta-Learning     |
|      | [**Incorporating Depth Information into Few-shot Semantic Segmentation**](https://arxiv.org/pdf/2107.00749) | [Code](https://github.com/Zhang-Bowen/RDNet)         | Metric Learning   |
|      | [**WPS-Net: Warped Prototype Space for Few-Shot Point Cloud Segmentation**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/509_ECCV_2022_paper.php) | `N/A`                                                | Data Augmentation |
|      | [**Bidirectional feature globalization for few-shot semantic segmentation**](https://ieeexplore.ieee.org/document/9884784) | `N/A`                                                | Metric Learning   |
|      | [**Cam/cad point cloud part segmentation via few-shot learning**](https://ieeexplore.ieee.org/document/9850937) | `N/A`                                                | Metric Learning   |
|      | [**Geodesic-Former: A Geodesic-Guided Few-Shot 3D Point Cloud Instance Segmenter**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/557_ECCV_2022_paper.php) | `N/A`                                                | Metric Learning   |
|      | [**Boosting few-shot 3d point cloud segmentation via query-guided enhancement**](https://dl.acm.org/doi/10.1145/3581783.3611956) | [Code](https://github.com/ningzhenhua/QGE-Net)       | Metric Learning   |
|      | [**Generalized few-shot point cloud segmentation via geometric words**](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_Generalized_Few-Shot_Point_Cloud_Segmentation_via_Geometric_Words_CVPR_2023_paper.pdf) | [Code](https://github.com/Pixie888/GW-Net)           | Metric Learning   |
|      | [**Few-Shot Point Cloud Semantic Segmentation via Contrastive Self-Supervision and Multi-Resolution Attention**](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Few-Shot_Point_Cloud_Semantic_Segmentation_via_Contrastive_Self-Supervision_and_Multi-Resolution_CVPR_2023_paper.pdf) | [Code](https://github.com/jiahui-wang/CSS-MRA)       | SSP               |
|      | [**Rethinking Few-Shot 3D Point Cloud Semantic Segmentation**](https://openaccess.thecvf.com/content/CVPR2024/papers/An_Rethinking_Few-Shot_3D_Point_Cloud_Semantic_Segmentation_CVPR_2024_paper.pdf) | `N/A`                                                | Meta-Learning     |
|      | [**Multimodality helps few-shot 3d point cloud semantic segmentation**](https://arxiv.org/pdf/2401.12489) | `N/A`                                                | Metric Learning   |
|      | [**Crossmodal few-shot 3d point cloud semantic segmentation via view synthesis**](https://dl.acm.org/doi/10.1145/3651859.3690623) | [Code](https://github.com/canyu-zhang/ViewSynth)     | Data Augmentation |
|      | [**Point-SAM: Promptable 3d segmentation model for point clouds**](https://arxiv.org/pdf/2406.17741) | [Code](https://github.com/yuchen-zhou-umd/Point-SAM) | PEFT-Additive     |
|      | [**SAM2POINT: Segment Any 3D as Videos in Zero-shot and Promptable Manners**](https://arxiv.org/pdf/2408.16768) | [Code](https://github.com/guozix/sam2point)          | PEFT-Additive     |
|      | [**Segment any point cloud via large language model**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1516_ECCV_2024_paper.php) | [Code](https://github.com/HET-Cooper/SegPoint)       | PEFT-Additive     |
|      | [**Pseudo-embedding for generalized few-shot 3D segmentation**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1049_ECCV_2024_paper.php) | `N/A`                                                | Metric Learning   |
|      | [**Unified Few-shot Crack Segmentation and its Precise 3D Automatic Measurement**](https://arxiv.org/pdf/2501.09203) | `N/A`                                                | PEFT-Additive     |

---

## 📖 Few-Shot 3D Point Cloud Object Detection (FS3D-D)

| Year | Title / Paper                                                | Code                                                         | Methodology      |
| :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :--------------- |
|      | [**Few-Shot Object Detection and Viewpoint Estimation**](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_30) | [Code](https://github.com/yin-wei/FS-Net)                    | Meta-Learning    |
|      | [**Source-Free Unsupervised Domain Adaptation for 3D Object Detection**](https://proceedings.neurips.cc/paper/2020/file/a05c73b2246c262886f7c631007d4b24-Paper.pdf) | [Code](https://github.com/salt-die/SF-UDA-3D)                | Model Adaptation |
|      | [**A few-shot learning approach for 3D defect detection in lithium-ion batteries**](https://iopscience.iop.org/article/10.1088/1742-6596/1884/1/012024) | `N/A`                                                        | Metric Learning  |
|      | [**Meta-Det3D: Learn to Learn Few-Shot 3D Object Detection**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3082_ECCV_2022_paper.php) | [Code](https://github.com/L-DR/Meta-Det3D)                   | Meta-Learning    |
|      | [**FSD: Few-Shot 3D Detection**](https://openaccess.thecvf.com/content/CVPR2022/papers/Qi_FSD_Few-Shot_3D_Detection_CVPR_2022_paper.pdf) | [Code](https://github.com/fan-qi/FSD)                        | Metric Learning  |
|      | [**Few-Shot Object Detection Using Multimodal Sensor Systems of Unmanned Surface Vehicles**](https://www.mdpi.com/1424-8220/22/4/1511) | `N/A`                                                        | Metric Learning  |
|      | [**Prototypical VoteNet for Few-Shot 3D Point Cloud Object Detection**](https://arxiv.org/pdf/2307.12805) | [Code](https://github.com/Bo-Zhang-in-Southampton/CP-VoteNet) | Metric Learning  |
|      | [**Prototypical Variational Autoencoder for Few-shot 3D Object Detection**](https://proceedings.neurips.cc/paper_files/paper/2023/file/b23f01920a03d7350419349818833943-Paper-Conference.pdf) | `N/A`                                                        | Metric Learning  |
|      | [**Real3D-AD: A Dataset of Point Cloud Anomaly Detection**](https://openaccess.thecvf.com/content/WACV2023/papers/Tian_Real3D-AD_A_Dataset_of_Point_Cloud_Anomaly_Detection_WACV_2023_paper.pdf) | [Project](https://github.com/M-3LAB/Real3D-AD)               | Metric Learning  |
|      | [**Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection**](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_PiMAE_Point_Cloud_and_Image_Interactive_Masked_Autoencoders_for_3D_CVPR_2023_paper.pdf) | [Code](https://github.com/yatianchen/PiMAE)                  | SSP              |
|      | [**Few-shot Class-incremental Learning for 3D Point Cloud Objects**](https://openaccess.thecvf.com/content/CVPR2023/papers/Chowdhury_3D-FSCIL_Few-Shot_Class-Incremental_Learning_for_3D_Point_Cloud_Objects_CVPR_2023_paper.pdf) | [Code](https://github.com/sa-sabir/3D-FSCIL)                 | Model Adaptation |
|      | [**Generalized Few-Shot 3D Object Detection of Road Objects**](https://arxiv.org/pdf/2302.03914) | `N/A`                                                        | Model Adaptation |
|      | [**Open-Vocabulary Point-Cloud Object Detection Without 3D Annotation**](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Open-Vocabulary_Point-Cloud_Object_Detection_Without_3D_Annotation_ICCV_2023_paper.pdf) | [Code](https://github.com/yuhegno/OV-3DET)                   | PEFT             |
|      | [**Extending CLIP for 3D Few-Shot Anomaly Detection with Multi-View Images**](https://arxiv.org/pdf/2401.13963) | [Code](https://github.com/imics-lab/CLIP3D-AD)               | PEFT-Additive    |
|      | [**3D-OVS: 3D Open-Vocabulary Scene Understanding**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_3D-OVS_3D_Open-Vocabulary_Scene_Understanding_CVPR_2024_paper.pdf) | [Code](https://github.com/dj-zhan/3D-OVS)                    | PEFT             |
|      | [**OVC-3D: Open-Vocabulary Component-wise 3D Detection**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_OVC-3D_Open-Vocabulary_Component-wise_3D_Detection_CVPR_2024_paper.pdf) | [Code](https://github.com/yuchaozhang/OVC-3D)                | PEFT             |
|      | [**FILP-3D: Enhancing 3D Few-shot Class-incremental Learning**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_FILP-3D_Enhancing_3D_Few-shot_Class-incremental_Learning_CVPR_2024_paper.pdf) | [Code](https://github.com/Z-Z-W/FILP-3D)                     | Model Adaptation |
|      | [**VLM-3D: End-to-End Vision-Language Models for Open-World 3D Perception**](https://arxiv.org/pdf/2508.09061) | `N/A`                                                        | PEFT             |

