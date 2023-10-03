# Towards Geospatial Foundation Models via Continual Pretraining [\[arxiv\]](https://arxiv.org/abs/2302.04476)

<!-- <p><img src="figures/gfm.png" width="1000" /></p> -->
<div align="center">
    <img src="figures/gfm.png" height="250px" />
</div>
<!-- <img src="gfm.png" width="300"> -->

**Abstract:** Geospatial technologies are becoming increasingly essential in our world for a wide range of applications, including agriculture, urban planning, and disaster response. To help improve the applicability and performance of deep learning models on these geospatial tasks, various works have begun investigating foundation models for this domain. Researchers have explored two prominent approaches for introducing such models in geospatial applications, but both have drawbacks in terms of limited performance benefit or prohibitive training cost. Therefore, in this work, we propose a novel paradigm for building highly effective geospatial foundation models with minimal resource cost and carbon impact. We first construct a compact yet diverse dataset from multiple sources to promote feature diversity, which we term GeoPile. Then, we investigate the potential of continual pretraining from large-scale ImageNet-22k models and propose a multi-objective continual pretraining paradigm, which leverages the strong representations of ImageNet while simultaneously providing the freedom to learn valuable in-domain features. Our approach outperforms previous state-of-the-art geospatial pretraining methods in an extensive evaluation on seven downstream datasets covering various tasks such as change detection, classification, multi-label classification, semantic segmentation, and super-resolution.
<br clear="left"/>

```
@inproceedings{mendieta2023towards,
  title={Towards Geospatial Foundation Models via Continual Pretraining},
  author={Mendieta, Mat{\'\i}as and Han, Boran and Shi, Xingjian and Zhu, Yi and Chen, Chen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16806--16816},
  year={2023}
}
```
## Setup
Coming soon.

## GeoPile
Coming soon.

## Acknowledgement
This code is based on [SimMIM](https://github.com/microsoft/SimMIM).