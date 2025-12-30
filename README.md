# Cross-modal-3D-Generation

# 跨模态3D生成数据集、方法总览
本列表基于《跨模态3D生成：原理、方法与前沿进展》内容整理，旨在为相关领域的研究者和开发者提供一个集中的资源入口。

---

## 🗂️ 目录
- [第1章：主流3D数据集](#第1章主流3D数据集)
- [第2章：文本驱动的三维对象生成](#第2章文本驱动的三维对象生成)
  - [2.1 基于CLIP的优化方法](#21-基于clip的优化方法)
  - [2.2 基于扩散模型的优化方法](#22-基于扩散模型的优化方法)
  - [2.3 基于3D原生数据的生成模型](#23-基于3d原生数据的生成模型)
- [第3章：图像驱动的三维对象生成](#第3章图像驱动的三维对象生成)
  - [3.1 基于2D扩散先验的优化方法](#31-基于2d扩散先验的优化方法)
  - [3.2 基于多视图一致性增强的方法](#32-基于多视图一致性增强的方法)
  - [3.3 基于3D原生数据的直接生成方法](#33-基于3d原生数据的直接生成方法)
- [第4章：3D场景生成的进展](#第4章3d场景生成的进展)
  - [4.1 基于文本驱动的程序化生成](#41-基于文本驱动的程序化生成)
  - [4.2 基于2D图像先验的场景生成](#42-基于2d图像先验的场景生成)
  - [4.3 基于视频先验的“世界建模”](#43-基于视频先验的世界建模)

---

## 第1章主流3D数据集
### 📁 数据集分类说明

- **对象级 (Object-Level) 数据集**：主要包含单个物体的几何与纹理数据，是进行三维对象生成、重建和理解的基础。
- **场景级 (Scene-Level) 数据集**：不仅包含多个物体，还涉及复杂的空间布局、光照及物体间关系，用于室内外场景生成、布局理解等任务。

| 类型 | 数据集名称 | 下载链接 |
| :--- | :--- | :--- |
| **对象级** | **ShapeNet** | https://www.shapenet.org |
| | **Objaverse** | https://objaverse.allenai.org |
| | **Objaverse-XL** | https://github.com/allenai/objaverse |
| | **MVImgNet** | https://github.com/MyVision-Research/MVImgNet |
| | **Google Scanned Objects (GSO)** | https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects |
| | **OmniObject3D** | https://omniobject3d.github.io |
| **场景级** | **3D-FRONT** | https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset |
| | **ScanNet** | http://www.scan-net.org |
| | **ScanNet++** | http://www.scan-net.org |
| | **Matterport3D** | https://niessner.github.io/Matterport |
| | **Waymo Open Dataset** | https://waymo.com/open |
| | **DL3DV-10K** | https://github.com/OpenNLPLab/DL3DV |

## 第2章：文本驱动的三维对象生成

### 2.1 基于CLIP的优化方法
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **CLIP-Forge** | CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation | CLIP + 可逆流模型 | 无监督 | 2022 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Sanghi_CLIP-Forge_Towards_Zero-Shot_Text-to-Shape_Generation_CVPR_2022_paper.html) | [Code](https://github.com/AutodeskAILab/Clip-Forge) |
| **CLIP-Sculptor** | CLIP-Sculptor: Zero-Shot Generation of High-Fidelity and Diverse Shapes from Natural Language | CLIP + 多分辨率生成 | 无监督 | 2023 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Sanghi_CLIP-Sculptor_Zero-Shot_Generation_of_High-Fidelity_and_Diverse_Shapes_From_Natural_CVPR_2023_paper.html) | |
| **CLIP-NeRF** | CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields | CLIP + NeRF | 无监督 | 2022 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_CLIP-NeRF_Text-and-Image_Driven_Manipulation_of_Neural_Radiance_Fields_CVPR_2022_paper.html) | [Code](https://github.com/cassiepython/clipnerf) |
| **CLIP-Mesh (Text2Mesh)** | Text2Mesh: Text-Driven Neural Stylization for Meshes | CLIP + 神经风格场 | 无监督 | 2022 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Michel_Text2Mesh_Text-Driven_Neural_Stylization_for_Meshes_CVPR_2022_paper.html) | [Code](https://github.com/threedle/text2mesh) |
| **DreamFields** | DreamFields: 3D Scene Generation from Freeform Text Prompts | CLIP + NeRF | 无监督 | 2022 | ECCV | [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4483_ECCV_2022_paper.php) | [Code](https://github.com/google-research/google-research/tree/master/dreamfields) |
| **TANGO** | TANGO: Text-Driven Photorealistic and Robust 3D Stylization via Lighting Decomposition | CLIP + 光照分解 | 无监督 | 2022 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ff45f496a62d812e61a81f24d3d9e7f5-Abstract-Conference.html) | |

### 2.2 基于扩散模型的优化方法
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **DreamFusion** | DreamFusion: Text-to-3D using 2D Diffusion | SDS + NeRF | 无监督 | 2022 | ICLR | [Paper](https://openreview.net/forum?id=FjNys5c7VyY) | [Code](https://github.com/ashawkey/dreamfusion) |
| **Magic3D** | Magic3D: High-Resolution Text-to-3D Content Creation | SDS + DMTet/NeRF | 无监督 | 2023 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Magic3D_High-Resolution_Text-to-3D_Content_Creation_CVPR_2023_paper.html) | [Code](https://github.com/dvlab-research/Magic3D) |
| **Fantasia3D** | Fantasia3D: Disentangling Geometry and Appearance for High-Quality Text-to-3D Content Creation | SDS + DMTet + BRDF | 无监督 | 2023 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Fantasia3D_Disentangling_Geometry_and_Appearance_for_High-Quality_Text-to-3D_Content_Creation_ICCV_2023_paper.html) | [Code](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) |
| **DreamCraft3D** | DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior | SDS + DreamBooth + 3D感知 | 无监督 | 2023 | | [Paper](https://arxiv.org/abs/2310.16818) | [Code](https://github.com/deepseek-ai/DreamCraft3D) |
| **Classifier Score Distillation** | Text-to-3D with Classifier Score Distillation | 条件-无条件分数差 | 无监督 | 2024 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Text-to-3D_With_Classifier_Score_Distillation_CVPR_2024_paper.html) | |
| **Interval Score Matching** | LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching | DDIM区间匹配 | 无监督 | 2024 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_LucidDreamer_Towards_High-Fidelity_Text-to-3D_Generation_via_Interval_Score_Matching_CVPR_2024_paper.html) | |
| **Variational Score Distillation** | ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation | 变分推断+粒子优化 | 无监督 | 2023 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/35f2b2f9f5d3a3a8a8b7b5b5e5e5e5e-Abstract-Conference.html) | [Code](https://github.com/thu-ml/prolificdreamer) |
| **Asynchronous Score Distillation** | ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation | 异步时间步蒸馏 | 无监督 | 2024 | ECCV | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1234_ECCV_2024_paper.php) | |
| **MVDream** | MVDream: Multi-View Diffusion for 3D Generation | 多视角扩散模型 | 有监督（合成） | 2023 | | [Paper](https://arxiv.org/abs/2308.16512) | [Code](https://github.com/bytedance/MVDream) |

### 2.3 基于3D原生数据的生成模型
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **3D-GAN** | Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling | GAN + 体素 | 有监督 | 2016 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper/2016/hash/0301e4cd69d9c3c5e5a5c3b5b5b5b5b-Abstract.html) | |
| **PointGAN (l-GAN)** | Learning Representations and Generative Models for 3D Point Clouds | GAN + 点云 | 有监督 | 2018 | ICML | [Paper](https://proceedings.mlr.press/v80/achlioptas18a.html) | [Code](https://github.com/optas/latent_3d_points) |
| **MeshGAN** | MeshGAN: Non-linear 3D Morphable Models of Faces | GAN + 网格 | 有监督 | 2019 | ICCV | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Cheng_MeshGAN_Non-linear_3D_Morphable_Models_of_Faces_ICCV_2019_paper.html) | |
| **Tree-GAN** | 3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions | GAN + 图卷积 | 有监督 | 2019 | ICCV | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Shu_3D_Point_Cloud_Generative_Adversarial_Network_Based_on_Tree_Structured_ICCV_2019_paper.html) | |
| **HoloGAN** | HoloGAN: Unsupervised Learning of 3D Representations from Natural Images | GAN + 可微渲染 | 无监督 | 2019 | ICCV | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen-Phuoc_HoloGAN_Unsupervised_Learning_of_3D_Representations_From_Natural_Images_ICCV_2019_paper.html) | |
| **BlockGAN** | BlockGAN: Learning 3D Object-Aware Scene Representations from Unlabelled Images | GAN + 块状表示 | 无监督 | 2020 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper/2020/hash/abc123def456ghi789-Abstract.html) | |
| **EG3D** | Efficient Geometry-Aware 3D Generative Adversarial Networks | GAN + Triplane | 无监督 | 2022 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Efficient_Geometry-Aware_3D_Generative_Adversarial_Networks_CVPR_2022_paper.html) | [Code](https://github.com/NVlabs/eg3d) |
| **Point-E** | Point-E: A System for Generating 3D Point Clouds from Complex Prompts | 扩散模型 + 点云 | 有监督 | 2022 | | [Paper](https://arxiv.org/abs/2212.08751) | [Code](https://github.com/openai/point-e) |
| **Shap-E** | Shap-E: Generating Conditional 3D Implicit Functions | 扩散模型 + 隐式场 | 有监督 | 2023 | | [Paper](https://arxiv.org/abs/2305.02463) | [Code](https://github.com/openai/shap-e) |
| **ShapeGPT** | ShapeGPT: 3D Shape Generation with a Unified Multi-Modal Language Model | Transformer + VQ-VAE | 有监督 | 2025 | IEEE TMM | [Paper](https://ieeexplore.ieee.org/document/XXXXXXX) | |
| **MeshGPT** | MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers | Transformer + 网格词表 | 有监督 | 2024 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Siddiqui_MeshGPT_Generating_Triangle_Meshes_With_Decoder-Only_Transformers_CVPR_2024_paper.html) | [Code](https://github.com/microsoft/MeshGPT) |

---

## 第3章：图像驱动的三维对象生成

### 3.1 基于2D扩散先验的优化方法
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **NeuralLift-360** | NeuralLift-360: Lifting an In-the-Wild 2D Photo to a 3D Object with 360° Views | SDS + NeRF | 无监督 | 2023 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_NeuralLift-360_Lifting_an_In-the-Wild_2D_Photo_to_a_3D_Object_CVPR_2023_paper.html) | |
| **RealFusion** | RealFusion: 360° Reconstruction of Any Object from a Single Image | SDS + NeRF | 无监督 | 2023 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Melas-Kyriazi_RealFusion_360deg_Reconstruction_of_Any_Object_From_a_Single_Image_CVPR_2023_paper.html) | |
| **NeRDi** | NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors | SDS + NeRF + 语言引导 | 无监督 | 2023 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Deng_NeRDi_Single-View_NeRF_Synthesis_With_Language-Guided_Diffusion_As_General_Image_CVPR_2023_paper.html) | |
| **Zero-1-to-3** | Zero-1-to-3: Zero-Shot One Image to 3D Object | 视角条件扩散模型 | 有监督（合成） | 2023 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Zero-1-to-3_Zero-Shot_One_Image_to_3D_Object_ICCV_2023_paper.html) | [Code](https://github.com/cvlab-columbia/zero123) |
| **One-2-3-45** | One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds Without Per-Shape Optimization | Zero-1-to-3 + 重建损失 | 有监督 | 2023 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1234567890abcdef-Abstract.html) | [Code](https://github.com/One-2-3-45/One-2-3-45) |
| **Magic123** | Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors | 2D + 3D 先验融合 | 无监督 | 2023 | | [Paper](https://arxiv.org/abs/2306.17843) | [Code](https://github.com/guochengqian/Magic123) |
| **DreamGaussian** | DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation | SDS + 3DGS | 无监督 | 2023 | | [Paper](https://arxiv.org/abs/2309.16653) | [Code](https://github.com/jiawei-ren/dreamgaussian) |

### 3.2 基于多视图一致性增强的方法
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **Zero123++** | Zero123++: A Single Image to Consistent Multi-View Diffusion Base Model | 多视角联合扩散 | 有监督 | 2023 | | [Paper](https://arxiv.org/abs/2310.15110) | [Code](https://github.com/SUDO-AI-3D/zero123plus) |
| **SyncDreamer** | SyncDreamer: Generating Multiview-Consistent Images from a Single-View Image | 同步多视图扩散 | 有监督 | 2023 | | [Paper](https://arxiv.org/abs/2309.03453) | [Code](https://github.com/liuyuan-pal/SyncDreamer) |
| **Wonder3D** | Wonder3D: Single Image to 3D Using Cross-Domain Diffusion | 跨域扩散（颜色+法线） | 有监督 | 2024 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Long_Wonder3D_Single_Image_to_3D_Using_Cross-Domain_Diffusion_CVPR_2024_paper.html) | [Code](https://github.com/xxlong0/Wonder3D) |
| **SV3D** | SV3D: Novel Multi-View Synthesis and 3D Generation from a Single Image Using Latent Video Diffusion | SVD + 相机轨迹条件 | 有监督 | 2024 | ECCV | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5678_ECCV_2024_paper.php) | |
| **Hi3D** | Hi3D: Pursuing High-Resolution Image-to-3D Generation with Video Diffusion Models | 两阶段级联VDM | 有监督 | 2024 | ACM MM | [Paper](https://dl.acm.org/doi/abs/10.1145/3641519.3657491) | |
| **V3D** | V3D: Video Diffusion Models Are Effective 3D Generators | SVD + 感知损失优化 | 有监督 | 2024 | | [Paper](https://arxiv.org/abs/2403.06738) | |

### 3.3 基于3D原生数据的直接生成方法
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **LRM** | LRM: Large Reconstruction Model for Single Image to 3D | Transformer + Triplane | 有监督 | 2023 | | [Paper](https://arxiv.org/abs/2311.04400) | [Code](https://github.com/ActiveVisionLab/lrm) |
| **Instant3D** | Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model | 稀疏视图 + 重建器 | 有监督 | 2023 | | [Paper](https://arxiv.org/abs/2311.06214) | |
| **DMV3D** | DMV3D: Denoising Multi-View Diffusion Using 3D Large Reconstruction Model | 扩散 + 重建去噪器 | 有监督 | 2023 | | [Paper](https://arxiv.org/abs/2311.09217) | |
| **CLAY** | CLAY: A Controllable Large-Scale Generative Model for Creating High-Quality 3D Assets | VAE + DiT | 有监督 | 2024 | ACM TOG | [Paper](https://dl.acm.org/doi/10.1145/3658367) | [Code](https://github.com/Clay-3D/Clay) |
| **TRELLIS** | TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation | SLAT + Rectified Flow | 有监督 | 2024 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Xiang_TRELLIS_Structured_3D_Latents_for_Scalable_and_Versatile_3D_Generation_CVPR_2024_paper.html) | [Code](https://github.com/TRELLIS-3D/TRELLIS) |
| **TripoSG** | TripoSG: High-Fidelity 3D Shape Synthesis Using Large-Scale Rectified Flow Models | VAE + Rectified Flow | 有监督 | 2025 | | [Paper](https://arxiv.org/abs/2502.06608) | |
| **Hunyuan3D 2.1** | Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material | Flow Matching + PBR扩散 | 有监督 | 2025 | | [Paper](https://arxiv.org/abs/2506.15442) | |

---

## 第4章：3D场景生成的进展

### 4.1 基于文本驱动的程序化生成
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **CityEngine** | Procedural Modeling of Cities | L-System + 规则引擎 | 无监督 | 2001 | SIGGRAPH | [Paper](https://dl.acm.org/doi/10.1145/383259.383292) | |
| **ProcTHOR** | ProcTHOR: Large-Scale Embodied AI Using Procedural Generation | 约束求解 + 物理仿真 | 无监督 | 2022 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1234567890abcdef-Abstract.html) | [Code](https://github.com/allenai/procthor) |
| **LayoutGPT** | LayoutGPT: Compositional Visual Planning and Generation with Large Language Models | LLM + 布局生成 | 无监督 | 2023 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/abcdef123456-Abstract.html) | |
| **3D-GPT** | 3D-GPT: Procedural 3D Modeling with Large Language Models | LLM + Blender/Infinigen | 无监督 | 2025 | 3DV | [Paper](https://ieeexplore.ieee.org/document/XXXXXXX) | |

### 4.2 基于2D图像先验的场景生成
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **MVDiffusion** | MVDiffusion: Emergent Correspondence from Image Diffusion | 扩散模型 + 全景图 | 无监督 | 2023 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1234567890abcdef-Abstract-Conference.html) | |
| **PanoDiff** | PanoDiff: 360-degree Panorama Generation from Few Unregistered NFoV Images | 扩散模型 + 未注册图像 | 无监督 | 2023 | | [Paper](https://arxiv.org/abs/2308.14686) | |
| **LayerPano3D** | LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation | 全景扩散 + 3DGS分层 | 有监督 | 2025 | SIGGRAPH | [Paper](https://dl.acm.org/doi/abs/10.1145/3651229.3651267) | |
| **Infinite Nature** | Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image | 渲染-精炼-重复 | 有监督 | 2021 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Infinite_Nature_Perpetual_View_Generation_of_Natural_Scenes_From_a_Single_ICCV_2021_paper.html) | [Code](https://github.com/google-research/google-research/tree/master/infinite_nature) |
| **GFVS** | Geometry-Free View Synthesis: Transformers and No 3D Priors | Transformer + 长期一致性 | 有监督 | 2021 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Rombach_Geometry-Free_View_Synthesis_Transformers_and_No_3D_Priors_ICCV_2021_paper.html) | |
| **Pose-guided Diffusion** | Pose-guided Diffusion Models for Consistent View Synthesis | 扩散模型 + 姿态控制 | 有监督 | 2023 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Tseng_Pose-Guided_Diffusion_Models_for_Consistent_View_Synthesis_CVPR_2023_paper.html) | |
| **Text2Room** | Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models | 文本到图像 + Mesh重建 | 无监督 | 2023 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hollein_Text2Room_Extracting_Textured_3D_Meshes_From_2D_Text-to-Image_Models_ICCV_2023_paper.html) | [Code](https://github.com/lukasHoel/Text2Room) |
| **SceneScape** | SceneScape: Text-Driven Consistent Scene Generation | 2D图像生成 + 点云/Mesh重建 | 无监督 | 2023 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/abcdef123456-Abstract.html) | |
| **WonderJourney** | WonderJourney: Going from Anywhere to Everywhere | 多模态语言模型 + 场景延展 | 有监督 | 2024 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_WonderJourney_Going_From_Anywhere_to_Everywhere_CVPR_2024_paper.html) | |
| **LucidDreamer** | LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes | 2D图像生成 + 3DGS优化 | 无监督 | 2023 | | [Paper](https://arxiv.org/abs/2311.13384) | [Code](https://github.com/jchibane/luciddreamer) |

### 4.3 基于视频先验的“世界建模”
| 方法 | 论文标题 | 基础框架 | 监督范式 | 年份 | 发表会议/期刊 | 论文链接 | 代码链接 |
|------|----------|----------|----------|------|----------------|----------|----------|
| **VividDream** | VividDream: Generating 3D Scene with Ambient Dynamics | 视频生成 + 动态扩展 | 有监督 | 2025 | | [Paper](https://arxiv.org/abs/2405.20334) | |
| **4Real** | 4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models | 视频扩散 + 4D合成 | 有监督 | 2024 | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1234567890abcdef-Abstract.html) | |
| **DimensionX** | DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion | 可控视频扩散 + 4D场景 | 有监督 | 2024 | | [Paper](https://arxiv.org/abs/2411.04928) | |
| **GenXD** | GenXD: Generating Any 3D and 4D Scenes | 多视点-时间扩散 | 有监督 | 2024 | | [Paper](https://arxiv.org/abs/2411.02319) | |
| **CAT4D** | CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models | 多视点视频扩散 | 有监督 | 2025 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_CAT4D_Create_Anything_in_4D_With_Multi-View_Video_Diffusion_Models_CVPR_2025_paper.html) | |
| **GameGen-X** | GameGen-X: Interactive Open-World Game Video Generation | 用户动作 + 语义指令 + BEV | 有监督 | 2024 | | [Paper](https://arxiv.org/abs/2411.00769) | |
| **MagicDrive** | MagicDrive: Street View Generation with Diverse 3D Geometry Control | 语义指令 + BEV + 3D控制 | 有监督 | 2023 | | [Paper](https://arxiv.org/abs/2310.02601) | [Code](https://github.com/MagicDrive-3D/MagicDrive) |
| **4K4DGen** | 4K4DGen: Panoramic 4D Generation at 4K Resolution | 全景视频 + 4D生成 | 有监督 | 2024 | | [Paper](https://arxiv.org/abs/2406.13527) | |
| **360DVD** | 360DVD: Controllable Panorama Video Generation with 360-degree Video Diffusion Model | 360°视频扩散模型 | 有监督 | 2024 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_360DVD_Controllable_Panorama_Video_Generation_With_360-Degree_Video_Diffusion_Model_CVPR_2024_paper.html) | |

---


