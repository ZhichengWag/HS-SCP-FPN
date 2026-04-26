# HS-FPN with Semantic Context Prior (SCP)

**Disclaimer:** This is an unofficial extension of the AAAI2025 paper **[HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection](https://arxiv.org/abs/2412.10116)**. The original HS-FPN architecture was proposed by Zican Shi et al. 

My specific contribution to this repository is the design and integration of the **Semantic Context Prior (SCP) branch**, aimed at further enhancing tiny object detection by incorporating background semantic information.

<br>

## 🌟 My Contribution: Semantic Context Prior (SCP) Branch
While the original HS-FPN significantly improves tiny object detection using high-frequency and spatial perception, tiny objects still heavily depend on their surrounding environment due to their low visual quality. 

To address this, I have designed and integrated the **Semantic Context Prior (SCP)** branch into the baseline HS-FPN. This branch introduces:
- **LFE-LSH Module**: A lightweight module that extracts coarse background semantic information.
- **Semantic-HFP Cross-Attention**: Fuses background semantic priors with High-Frequency Perception (HFP) features to suppress background noise and enhance object representations.
- **Knowledge Distillation Strategy**: Utilizes pseudo-labels (e.g., generated via the Segment Anything Model) to provide semantic supervision during training using a Cosine Annealing weighted distillation loss.

<br>

## Original HS-FPN Introduction
*The following is the abstract and architecture from the original HS-FPN authors:*

> `Abstract:` The introduction of Feature Pyramid Network (FPN) has significantly improved object detection performance. However, substantial challenges remain in detecting tiny objects, as their features occupy only a very small proportion of the feature maps. Although FPN integrates multi-scale features, it does not directly enhance or enrich the features of tiny objects. Furthermore, FPN lacks spatial perception ability. To address these issues, we propose a novel High Frequency and Spatial Perception Feature Pyramid Network (HS-FPN) with two innovative modules. First, we designed a high frequency perception module (HFP) that generates high frequency responses through high pass filters. These high frequency responses are used as mask weights from both spatial and channel perspectives to enrich and highlight the features of tiny objects in the original feature maps. Second, we developed a spatial dependency perception module (SDP) to capture the spatial dependencies that FPN lacks. Our experiments demonstrate that detectors based on HS-FPN exhibit competitive advantages over state-of-the-art models on the AI-TOD dataset for tiny object detection.

- **HS-FPN Architecture**
<img src="photo/HS-FPN.jpg" alt="HS-FPN Architecture" width="600"/>

- **HFP & SDP Modules**  
<img src="photo\HFP.jpg" alt="HFP Architecture" width="400"/> <img src="photo\SDP.jpg" alt="SDP Architecture" width="400"/>

<br>

## Requirement
Required environments:
* Linux
* Python 3.7.16
* PyTorch 1.7.1
* CUDA 11.0
* torch_dct 0.1.6
* [MMdetection 2.24.1](https://github.com/open-mmlab/mmdetection/tree/v2.24.1)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)
* [`hsrequirement.txt`](hsfpn_requirements.txt)

> Note: The [`hsrequirement.txt`](hsfpn_requirements.txt) file contains all packages from the original development environment. This module may only require a subset of these packages to run properly.

<br>

## Installation

### Step 1: Install MMDetection
This project is built upon [MMDetection v2.24.1](https://github.com/open-mmlab/mmdetection/tree/v2.24.1). Make sure MMDetection is properly installed following its [official instructions](https://github.com/open-mmlab/mmdetection/blob/v2.24.1/docs/zh_cn/get_started.md).

---

### Step 2: Integrate HS-FPN & SCP Module
1. **Copy [`hs_fpn.py`](hs_fpn.py)** into the MMDetection `mmdet/models/necks` directory.
2. Edit `__init__.py` to register the modules:
```python
from .hs_fpn import HS_FPN, HS_FPN_SCP
```
Ensure the `__all__` list includes `"HS_FPN"` and `"HS_FPN_SCP"`.

<br>

## Training

### Train the SCP-Enhanced Version (My Contribution)
To train the SCP-enhanced version, ensure your pseudo-labels are correctly generated and placed in the dataset folder, then use the specific SCP config:
```bash
python tools/train.py config_hsfpn/cascade_rcnn_r50_aitod_scp.py
```

### Train Original HS-FPN Baseline
To train the original HS-FPN baseline with a single GPU:
```bash
python tools/train.py config_hsfpn/cascade_rcnn_r50_aitod.py
```

<br>

## Citations
Please credit the original authors of HS-FPN if you find the baseline architecture helpful:
```bibtex
@inproceedings{shi2025hs,
  title={HS-FPN: High frequency and spatial perception FPN for tiny object detection},
  author={Shi, Zican and Hu, Jing and Ren, Jie and Ye, Hengkang and Yuan, Xuyang and Ouyang, Yan and He, Jia and Ji, Bo and Guo, Junyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={6896--6904},
  year={2025}
}
```