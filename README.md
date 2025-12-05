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
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------- |
| 2020 | [**Transductive Zero-Shot Learning for 3D Point Cloud Classification**](https://arxiv.org/abs/1912.07161) | [Code](https://github.com/ali-chr/Transductive_ZSL_3D_Point_Cloud) | Metric Learning   |
| 2020 | [**FEW SHOT LEARNING FOR POINT CLOUD DATA USING MODEL AGNOSTIC META LEARNING**](https://www-video.eecs.berkeley.edu/papers/rpuri/Icip_2019_Maml_for_pointclouds_Final.pdf) | `N/A`                                                        | Meta-Learning     |
| 2020 | [**Self-Supervised Few-Shot Learning on Point Clouds**](https://proceedings.neurips.cc/paper/2020/file/50c1f44e426560f3f2cdcb3e19e39903-Paper.pdf) | [Code](https://github.com/charusharma1991/SSL_PointClouds)   | SSP               |
| 2020 | [**PointMixup: Augmentation for point clouds**](https://arxiv.org/pdf/2008.06374) | [Code](https://github.com/yunlu-chen/PointMixup/)            | Data Augmentation |
| 2021 | [**MetaSets: Meta-Learning on Point Sets for Generalizable Representations**](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_MetaSets_Meta-Learning_on_Point_Sets_for_Generalizable_Representations_CVPR_2021_paper.pdf) | [Code](https://github.com/thuml/Metasets)                    | Meta-Learning     |
| 2022 | [**Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling**](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.pdf) | [Code](https://github.com/lulutang0608/Point-BERT)           | SSP               |
| 2022 | [**PointCLIP: Point Cloud Understanding by CLIP**](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.pdf) | [Code](https://github.com/ZrrSkywalker/PointCLIP)            | PEFT-Additive     |
| 2022 | [**What Makes for Effective Few-Shot Point Cloud Classification?**](https://openaccess.thecvf.com/content/WACV2022/papers/Ye_What_Makes_for_Effective_Few-Shot_Point_Cloud_Classification_WACV_2022_paper.pdf) | `N/A`                                                        | Metric Learning   |
| 2023 | [**GPr-Net: Geometric Prototypical Network for Point Cloud Few-Shot Learning**](https://openaccess.thecvf.com/content/CVPR2023W/DLGC/papers/Anvekar_GPr-Net_Geometric_Prototypical_Network_for_Point_Cloud_Few-Shot_Learning_CVPRW_2023_paper.pdf) | [Code](https://github.com/TejasAnvekar/GPr-Net)              | Metric Learning   |
| 2023 | [**A Closer Look at Few-Shot 3D Point Cloud Classification**](https://arxiv.org/pdf/2303.18210) | [Code](https://github.com/cgye96/A_Closer_Look_At_3DFSL)     | Metric Learning   |
| 2023 | [**ViewNet: A Novel Projection-Based Backbone With View Pooling for Few-Shot Point Cloud Classification**](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_ViewNet_A_Novel_Projection-Based_Backbone_With_View_Pooling_for_Few-Shot_CVPR_2023_paper.pdf) | `N/A`                                                        | Metric Learning   |
| 2023 | [**CrossMoCo: Multi-modal Momentum Contrastive Learning for Point Cloud**](https://www.researchgate.net/profile/Sneha-Paul-4/publication/373505772_CrossMoCo_Multi-Modal_Momentum_Contrastive_Learning_for_Point_Cloud/links/66c4bc734b25ef677f7207cf/CrossMoCo-Multi-Modal-Momentum-Contrastive-Learning-for-Point-Cloud.pdf) | [Code](https://github.com/snehaputul/CrossMoCo)              | SSP               |
| 2023 | [**Instance-aware Dynamic Prompt Tuning for Pre-trained Point Cloud Models**](https://openaccess.thecvf.com/content/ICCV2023/papers/Zha_Instance-aware_Dynamic_Prompt_Tuning_for_Pre-trained_Point_Cloud_Models_ICCV_2023_paper.pdf) | [Code](https://github.com/zyh16143998882/)                   | PEFT-Additive     |
| 2023 | [**CLR-GAM: Contrastive Point Cloud Learning with Guided Augmentation and Feature Mapping**](https://arxiv.org/pdf/2302.14306) | `N/A`                                                        | Data Augmentation |
| 2024 | [**Point-PEFT: Parameter-Efficient Fine-Tuning for 3D Pre-trained Models**](https://ojs.aaai.org/index.php/AAAI/article/view/28323/28635) | [Code](https://github.com/Ivan-Tang-3D/Point-PEFT)           | PEFT-Hybrid       |
| 2024 | [**CMNet: Component-Aware Matching Network for Few-Shot Point Cloud Classification**](https://ieeexplore.ieee.org/abstract/document/10496841) | [Code](https://github.com/JACKYLUO1991/CMNet)                | Meta-Learning     |
| 2024 | [**Dynamic Adapter Meets Prompt Tuning: Parameter-Efficient Transfer Learning for Point Cloud Analysis**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Dynamic_Adapter_Meets_Prompt_Tuning_Parameter-Efficient_Transfer_Learning_for_Point_CVPR_2024_paper.pdf) | [Code](https://github.com/LMD0311/DAPT)                      | PEFT-Hybrid       |
| 2024 | [**Meta Episodic learning with Dynamic Task Sampling for CLIP-based Point Cloud Classification**](https://arxiv.org/pdf/2404.00857) | `N/A`                                                        | PEFT-Hybrid       |
| 2024 | [**Dynamic Multimodal Prompt Tuning: Boost Few-Shot Learning with VLM-Guided Point Cloud Models**](https://digibuo.uniovi.es/dspace/bitstream/handle/10651/75852/2024%20ECAI.pdf?sequence=1) | [Code](https://github.com/eminentgu/DMMPT)                   | PEFT-Hybrid       |
| 2024 | [**CLIP-based Point Cloud Classification via Point Cloud to Image Translation**](https://arxiv.org/pdf/2408.03545) | `N/A`                                                        | PEFT-Additive     |
| 2024 | [**Positional Prompt Tuning for Efficient 3D Representation Learning**](https://arxiv.org/pdf/2408.11567) | [Code](https://github.com/zsc000722/PPT)                     | PEFT-Additive     |
| 2024 | [**Adapt PointFormer: 3D Point Cloud Analysis via Adapting 2D Visual Transformers**](https://arxiv.org/pdf/2407.13200) | [Code](https://vcc.tech/research/2024/PointFormer)           | PEFT-Additive     |
| 2024 | [**PointGST: Parameter-Efficient Fine-Tuning in Spectral Domain for Point Cloud Learning**](https://arxiv.org/pdf/2410.08114) | [Code](https://github.com/jerryfeng2003/PointGST)            | PEFT-Additive     |
| 2024 | [**Hyperbolic prototype rectification for few-shot 3D point cloud classification**](linkinghub.elsevier.com/retrieve/pii/S0031320324007933) | [Code](https://github.com/Jonathan-UCAS/HPR.)                | Metric Learning   |
| 2024 | [**PointLoRA: Low-Rank Adaptation with Token Selection for Point Cloud Learning**](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_PointLoRA_Low-Rank_Adaptation_with_Token_Selection_for_Point_Cloud_Learning_CVPR_2025_paper.pdf) | [Code](https://github.com/songw-zju/PointLoRA)               | PEFT-Reparam.     |
| 2024 | [**PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter**](https://openaccess.thecvf.com/content/CVPR2025/papers/Zha_PMA_Towards_Parameter-Efficient_Point_Cloud_Understanding_via_Point_Mamba_Adapter_CVPR_2025_paper.pdf) | [Code](https://github.com/zyh16143998882/PMA)                | PEFT-Additive     |
| 2024 | [**GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model**](https://arxiv.org/pdf/2505.04119) | [Code](https://github.com/zhoujiahuan1991/ICML2025-GAPrompt) | PEFT-Additive     |

---



## 📖 Few-Shot 3D Point Cloud Segmentation (FS3D-S)

| Year | Title / Paper                                                | Code                                                         | Methodology       |
| :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :---------------- |
| 2020 | [**Compositional Prototype Network with Multi-View Comparision for Few-Shot Point Cloud Semantic Segmentation**](https://arxiv.org/pdf/2012.14255) | `N/A`                                                        | Metric Learning   |
| 2020 | [**PointGLR: Unsupervised Structural Representation Learning of 3D Point Clouds**](https://drive.google.com/file/d/11Cl12q3rOkD8yi6nsNEhw8o38KeAbjcw/view) | [Code](https://github.com/raoyongming/PointGLR)              | SSP               |
| 2020 | [**Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions**](https://arxiv.org/pdf/2003.13834) | [Code](https://github.com/matheusgadelha/PointCloudLearningACD) | SSP               |
| 2021 | [**Few-shot 3D Point Cloud Semantic Segmentation**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Few-Shot_3D_Point_Cloud_Semantic_Segmentation_CVPR_2021_paper.pdf) | [Code](https://github.com/Na-Z/attMPTI)                      | Meta-Learning     |
| 2021 | [**Few-shot Meta-learning on Point Cloud for Semantic Segmentation**](https://arxiv.org/pdf/2104.02979) | `N/A`                                                        | Meta-Learning     |
| 2021 | [**Incorporating Depth Information into Few-shot Semantic Segmentation**](https://univ-evry.hal.science/hal-02887063/document) | `N/A`                                                        | Metric Learning   |
| 2022 | [**WPS-Net: Warped Prototype Space for Few-Shot Point Cloud Segmentation**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Few-Shot_Learning_of_Part-Specific_Probability_Space_for_3D_Shape_Segmentation_CVPR_2020_paper.pdf) | `N/A`                                                        | Data Augmentation |
| 2022 | [**PriFit: Learning to Fit Primitives Improves Few Shot Point Cloud Segmentation**](https://arxiv.org/pdf/2112.13942) | [Code](https://hippogriff.github.io/prifit/)                 | SSP               |
| 2022 | [**Bidirectional Feature Globalization for Few-shot Semantic Segmentation of 3D Point Cloud Scenes**](https://arxiv.org/pdf/2208.06671) | `N/A`                                                        | Metric Learning   |
| 2022 | [**Cam/cad point cloud part segmentation via few-shot learning**](https://arxiv.org/pdf/2207.01218) | `N/A`                                                        | Metric Learning   |
| 2022 | [**Geodesic-Former: A Geodesic-Guided Few-Shot 3D Point Cloud Instance Segmenter**](https://arxiv.org/pdf/2207.10859) | [Code](https://github.com/VinAIResearch/GeoFormer)           | Metric Learning   |
| 2022 | [**PointCLIP: Point Cloud Understanding by CLIP**](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.pdf) | [Code](https://github.com/ZrrSkywalker/PointCLIP)            | PEFT-Additive     |
| 2022 | [**Weakly Supervised 3D Point Cloud Segmentation via Multi-Prototype Learning**](https://arxiv.org/pdf/2205.03137) | `N/A`                                                        | Metric Learning   |
| 2022 | [**Starting From Non-Parametric Networks for 3D Point Cloud Analysis**](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Starting_From_Non-Parametric_Networks_for_3D_Point_Cloud_Analysis_CVPR_2023_paper.pdf) | [Code](https://github.com/ZrrSkywalker/Point-NN)             | Metric Learning   |
| 2022 | [**CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding**](https://openaccess.thecvf.com/content/CVPR2022/papers/Afham_CrossPoint_Self-Supervised_Cross-Modal_Contrastive_Learning_for_3D_Point_Cloud_Understanding_CVPR_2022_paper.pdf) | [Code](https://github.com/MohamedAfham/CrossPoint)           | SSP               |
| 2022 | [**Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling**](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.pdf) | [Code](https://github.com/lulutang0608/Point-BERT)           | SSP               |
| 2022 | [**Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training**](https://proceedings.neurips.cc/paper_files/paper/2022/file/ad1d7a4df30a9c0c46b387815a774a84-Paper-Conference.pdf) | [Code](https://github.com/ZrrSkywalker/Point-M2AE)           | SSP               |
| 2022 | [**Geodesic Self-Attention for 3D Point Clouds**](https://proceedings.neurips.cc/paper_files/paper/2022/file/28e4ee96c94e31b2d040b4521d2b299e-Paper-Conference.pdf) | `N/A`                                                        | SSP               |
| 2023 | [**Boosting Few-Shot 3D Point Cloud Segmentation via Query-Guided Enhancement**](https://arxiv.org/pdf/2308.03177) | [Code](https://github.com/AaronNZH/Boosting-Few-shot-3D-Point-Cloud-Segmentation-via-Query-Guided-Enhancement) | Metric Learning   |
| 2023 | [**Towards Robust Few-shot Point Cloud Semantic Segmentation**](https://arxiv.org/pdf/2309.11228) | [Code](https://github.com/Pixie8888/R3DFSSeg)                | Metric Learning   |
| 2023 | [**Prototype Adaption and Projection for Few- and Zero-Shot 3D Point Cloud Semantic Segmentation**](https://arxiv.org/pdf/2305.14335) | [Code](https://github.com/heshuting555/PAP-FZS3D)            | Metric Learning   |
| 2023 | [**Generalized Few-Shot Point Cloud Segmentation via Geometric Words**](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Generalized_Few-Shot_Point_Cloud_Segmentation_via_Geometric_Words_ICCV_2023_paper.pdf) | [Code](https://github.com/Pixie8888/GFS-3DSeg_GWs)           | Metric Learning   |
| 2023 | [**Few-shot 3D point cloud semantic segmentation via stratified class-specific attention based transformer network**](https://ojs.aaai.org/index.php/AAAI/article/view/25449/25221) | [Code](https://github.com/czzhang179/SCAT)                   | Model Adaptation  |
| 2023 | [**Few-shot Point Cloud Semantic Segmentation via Contrastive Self-supervision and Multi-resolution Attention**](https://arxiv.org/pdf/2302.10501) | `N/A`                                                        | SSP               |
| 2023 | [**Base and Meta: A New Perspective on Few-shot Segmentation**](https://drive.google.com/file/d/17nH4af7Xi-5OQ3DJfFIZXoOjCQuZkWzt/view) | [Code](https://github.com/chunbolang/BAM.)                   | Meta-Learning     |
| 2023 | [**Few-shot learning on point clouds for railroad segmentation**](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/ei/35/17/3DIA-100) | `N/A`                                                        | Metric Learning   |
| 2023 | [**Less is More: Towards Efficient Few-Shot 3D Semantic Segmentation via Training-Free Networks**](https://arxiv.org/pdf/2308.12961) | [Code](https://github.com/yangyangyang127/TFS3D)             | Metric Learning   |
| 2023 | [**360° from a Single Camera: A Few-Shot Approach for LiDAR Segmentation**](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/papers/Reichardt_360deg_from_a_Single_Camera_A_Few-Shot_Approach_for_LiDAR_ICCVW_2023_paper.pdf) | `N/A`                                                        | PEFT-Additive     |
| 2023 | [**Invariant Training 2D-3D Joint Hard Samples for Few-Shot Point Cloud Recognition**](https://openaccess.thecvf.com/content/ICCV2023/papers/Yi_Invariant_Training_2D-3D_Joint_Hard_Samples_for_Few-Shot_Point_Cloud_ICCV_2023_paper.pdf) | [Code](https://github.com/yxymessi/InvJoint)                 | PEFT-Additive     |
| 2023 | [**Geometry and Uncertainty-Aware 3D Point Cloud Class-Incremental Semantic Segmentation**](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Geometry_and_Uncertainty-Aware_3D_Point_Cloud_Class-Incremental_Semantic_Segmentation_CVPR_2023_paper.pdf) | [Code](https://github.com/leolyj/3DPC-CISS)                  | Model Adaptation  |
| 2023 | [**Analogy-Forming Transformers for Few-Shot 3D Parsing**](https://arxiv.org/pdf/2304.14382) | [Code](http://analogicalnets.github.io/)                     | Meta-Learning     |
| 2023 | [**Discriminative 3D Shape Modeling for Few-Shot Instance Segmentation**](https://merl.com/publications/docs/TR2023-010.pdf) | `N/A`                                                        | Metric Learning   |
| 2024 | [**Rethinking Few-Shot 3D Point Cloud Semantic Segmentation**](https://openaccess.thecvf.com/content/CVPR2024/papers/An_Rethinking_Few-shot_3D_Point_Cloud_Semantic_Segmentation_CVPR_2024_paper.pdf) | [Code](https://github.com/ZhaochongAn/COSeg)                 | Meta-Learning     |
| 2024 | [**Multimodality Helps Few-Shot 3D Point Cloud Semantic Segmentation**](https://arxiv.org/pdf/2410.22489) | [Code](https://github.com/ZhaochongAn/Multimodality-3D-Few-Shot) | Metric Learning   |
| 2024 | [**A Simple Framework of Few-shot Learning using Sparse Annotations for Semantic Segmentation of 3-D Point Clouds**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10423773) | `N/A`                                                        | Metric Learning   |
| 2024 | [**Localization and Expansion: A Decoupled Framework for Point Cloud Few-Shot Semantic Segmentation**](https://arxiv.org/pdf/2408.13752) | `N/A`                                                        | Metric Learning   |
| 2024 | [**Crossmodal Few-Shot 3D Point Cloud Semantic Segmentation via View Synthesis**](https://dl.acm.org/doi/pdf/10.1145/3664647.3681428) | `N/A`                                                        | Data Augmentation |
| 2024 | [**Point-SAM: Promptable 3D Segmentation Model for Point Clouds**](https://arxiv.org/pdf/2406.17741) | [Code](https://github.com/zyc00/Point-SAM)                   | PEFT-Additive     |
| 2024 | [**Aggregation and Purification: Dual Enhancement Network for Point Cloud Few-Shot Segmentation**](https://www.ijcai.org/proceedings/2024/0164.pdf) | `N/A`                                                        | Metric Learning   |
| 2024 | [**Dynamic Prototype Adaptation with Distillation for Few-shot Point Cloud Segmentation**](https://arxiv.org/pdf/2401.16051) | [Code](https://github.com/jliu4ai/DPA)                       | Metric Learning   |
| 2024 | [**Multiprototype Relational Network for Few-shot ALS Point Cloud Semantic Segmentation by Transferring Knowledge from Photogrammetric Point Clouds**](https://ieeexplore.ieee.org/abstract/document/10430180) | `N/A`                                                        | Meta-Learning     |
| 2024 | [**Pseudo-Embedding for Generalized Few-shot 3D Segmentation**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05346.pdf) | [Code](https://github.com/jimtsai23/PseudoEmbed)             | Metric Learning   |
| 2024 | [**SegPoint: Segment Any Point Cloud via Large Language Model**](https://arxiv.org/pdf/2407.13761) | [Code](https://heshuting555.github.io/SegPoint)              | PEFT-Additive     |
| 2024 | [**Part-Whole Relational Few-shot 3D Point Cloud Semantic Segmentation**](https://cdn.techscience.press/files/cmc/2024/TSP_CMC-78-3/TSP_CMC_45853/TSP_CMC_45853.pdf) | `N/A`                                                        | Metric Learning   |
| 2024 | [**SAM2Point: Segment Any 3D as Videos in Zero-shot and Promptable Manners**](https://arxiv.org/pdf/2408.16768) | [Code](https://github.com/ZiyuGuo99/SAM2Point)               | PEFT-Additive     |
| 2024 | [**Cross-Domain Few-Shot Incremental Learning for Point-Cloud Recognition**](https://openaccess.thecvf.com/content/WACV2024/papers/Tan_Cross-Domain_Few-Shot_Incremental_Learning_for_Point-Cloud_Recognition_WACV_2024_paper.pdf) | `N/A`                                                        | Model Adaptation  |
| 2024 | [**Generated and Pseudo Content Guided Prototype Refinement for Few-Shot Point Cloud Segmentation**](https://proceedings.neurips.cc/paper_files/paper/2024/file/377d0752059d3d4686aa021b664a25dd-Paper-Conference.pdf) | `N/A`                                                        | Metric Learning   |
| 2024 | [**Label-Efficient Semantic Segmentation of LiDAR Point Clouds in Adverse Weather Conditions**](https://arxiv.org/pdf/2406.09906) | `N/A`                                                        | Model Adaptation  |
| 2024 | [**TeFF: Tracking-enhanced Forgetting-free Few-shot 3D LiDAR Semantic Segmentation**](https://arxiv.org/pdf/2408.15657) | [Code](https://github.com/junbao-zhou/Track-no-forgetting)   | PEFT-Reparam.     |
| 2024 | [**UniRiT: Towards Few-Shot Non-Rigid Point Cloud Registration**](https://arxiv.org/pdf/2410.22909) | `N/A`                                                        | Model Adaptation  |
| 2025 | [**Unified Few-shot Crack Segmentation and its Precise 3D Automatic Measurement in Concrete Structures**](https://arxiv.org/pdf/2501.09203) | `N/A`                                                        | PEFT-Additive     |

---

## 📖 Few-Shot 3D Point Cloud Object Detection (FS3D-D)

| Year | Title / Paper                                                | Code                                                   | Methodology      |
| :--- | :----------------------------------------------------------- | :----------------------------------------------------- | :--------------- |
| 2020 | [**Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild**](https://arxiv.org/pdf/2007.12107) | [Code](http://imagine.enpc.fr/~xiaoy/FSDetView/)       | Meta-Learning    |
| 2020 | [**Source-free Unsupervised Domain Adaptation for 3D Object Detection in Adverse Weather**](https://ieeexplore.ieee.org/abstract/document/10161341) | [Code](https://github.com/saltoricristiano/SF-UDA-3DV) | Model Adaptation |
| 2021 | [**Few-shot learning approach for 3D defect detection in lithium-ion batteries**](https://iopscience.iop.org/article/10.1088/1742-6596/1884/1/012024/pdf) | `N/A`                                                  | Metric Learning  |
| 2022 | [**Meta-Det3D: Learn to Learn Few-Shot 3D Object Detection**](https://openaccess.thecvf.com/content/ACCV2022/papers/Yuan_Meta-Det3D_Learn_to_Learn_Few-Shot_3D_Object_Detection_ACCV_2022_paper.pdf) | `N/A`                                                  | Meta-Learning    |
| 2022 | [**FSD: Fully Sparse 3D Object Detection**](https://proceedings.neurips.cc/paper_files/paper/2022/file/0247fa3c511bbc415c8b768ee7b32f9e-Paper-Conference.pdf) | [Code](https://github.com/TuSimple/SST)                | Metric Learning  |
| 2022 | [**Prototypical VoteNet for Few-Shot 3D Point Cloud Object Detection**](https://proceedings.neurips.cc/paper_files/paper/2022/file/59e73ff865b56cba6ab7f6b2cce1425d-Paper-Conference.pdf) | [Code](https://shizhen-zhao.github.io/FS3D_page/)      | Metric Learning  |
| 2022 | [**Few-shot Object Detection Using Multimodal Sensor Systems of Unmanned Surface Vehicles**](https://pdfs.semanticscholar.org/1424/9780a1d9464c97b575fc45c26c0b104d8398.pdf) | `N/A`                                                  | Metric Learning  |
| 2023 | [**Prototypical Variational Autoencoder for Few-shot 3D Point Cloud Object Detection**](https://papers.neurips.cc/paper_files/paper/2023/file/076a93fd42aa85f5ccee921a01d77dd5-Paper-Conference.pdf) | `N/A`                                                  | Metric Learning  |
| 2023 | [**Real3D-AD: A Dataset of Point Cloud Anomaly Detection (with Reg3D-AD baseline)**](https://proceedings.neurips.cc/paper_files/paper/2023/file/611b896d447df43c898062358df4c114-Paper-Datasets_and_Benchmarks.pdf) | [Code](https://github.com/M-3LAB/Real3D-AD)            | Metric Learning  |
| 2023 | [**PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection**](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_PiMAE_Point_Cloud_and_Image_Interactive_Masked_Autoencoders_for_3D_CVPR_2023_paper.pdf) | [Code](https://github.com/BLVLab/PiMAE)                | SSP              |
| 2023 | [**3D-FSCIL: Few-shot Class-incremental Learning for 3D Point Cloud Objects**](https://arxiv.org/pdf/2205.15225) | [Code](https://github.com/townim-faisal/FSCIL-3D)      | Model Adaptation |
| 2023 | [**Generalized Few-Shot 3D Object Detection of LiDAR Point Cloud for Autonomous Driving**](https://arxiv.org/pdf/2302.03914) | `N/A`                                                  | Model Adaptation |
| 2023 | [**Open-Vocabulary Point-Cloud Object Detection Without 3D Annotation**](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Open-Vocabulary_Point-Cloud_Object_Detection_Without_3D_Annotation_CVPR_2023_paper.pdf) | [Code](https://github.com/lyhdet/OV-3DET)              | PEFT-Additive    |
| 2023 | [**CLIP Goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition**](https://openaccess.thecvf.com/content/ICCV2023W/OpenSUN3D/papers/Hegde_CLIP_Goes_3D_Leveraging_Prompt_Tuning_for_Language_Grounded_3D_ICCVW_2023_paper.pdf) | [Code](https://github.com/deeptibhegde/CLIPgoes-3D)    | PEFT-Additive    |
| 2023 | [**PartSLIP: Low-Shot Part Segmentation for 3D Point Clouds via Pretrained Image-Language Models**](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_PartSLIP_Low-Shot_Part_Segmentation_for_3D_Point_Clouds_via_Pretrained_CVPR_2023_paper.pdf) | `N/A`                                                  | PEFT-Additive    |
| 2024 | [**CP-VoteNet: Contrastive Prototypical VoteNet for Few-Shot Point Cloud Object Detection**](https://arxiv.org/pdf/2408.17036) | `N/A`                                                  | Metric Learning  |
| 2024 | [**CLIP3D-AD: Extending CLIP for 3D Few-Shot Anomaly Detection with Multi-View Images Generation**](https://arxiv.org/pdf/2406.18941) | `N/A`                                                  | PEFT-Additive    |
| 2024 | [**FS-3DSSN: an efficient few-shot learning for single-stage 3D object detection on point clouds**](https://link.springer.com/article/10.1007/s00371-023-03228-8) | `N/A`                                                  | Model Adaptation |
| 2024 | [**Elemental Composite Prototypical Network: Few-Shot Object Detection on Outdoor 3D Point Cloud Scenes**](https://openaccess.thecvf.com/content/WACV2025/papers/De_Elemental_Composite_Prototypical_Network_Few-Shot_Object_Detection_on_Outdoor_3D_WACV_2025_paper.pdf?trk=public_post_comment-text) | `N/A`                                                  | Metric Learning  |
| 2024 | [**A Category-Agnostic Hybrid Contrastive Learning Method for Few-Shot Point Cloud Object Detection**](https://cdn.techscience.press/files/cmc/2025/TSP_CMC-83-2/TSP_CMC_62161/TSP_CMC_62161.pdf) | [Code](https://github.com/offscuminSJTU/HC)            | Metric Learning  |
| 2024 | [**From Dataset to Real-world: General 3D Object Detection via Generalized Cross-domain Few-shot Learning**](https://arxiv.org/pdf/2503.06282) | `N/A`                                                  | Model Adaptation |
| 2023 | [**FILP-3D: Enhancing 3D Few-shot Class-incremental Learning**](https://arxiv.org/pdf/2312.17051) | [Code](https://github.com/HIT-leaderone/FILP-3D)       | Model Adaptation |
| 2025 | [**VLM-3D: End-to-End Vision-Language Models for Open-World 3D Perception**](https://arxiv.org/pdf/2508.09061) | `N/A`                                                  | PEFT-Reparam.    |
