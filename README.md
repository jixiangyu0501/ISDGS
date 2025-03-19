Invariant semantic domain generalization shuffle network for cross-scene hyperspectral image classification
=====================

Articles can be found in the following link ： https://www.sciencedirect.com/science/article/abs/pii/S0957417425004403
---

Abstract
---
Cross-scene hyperspectral image classification is currently receiving widespread attention. However, domain
adaptation-based methods usually perform domain alignment by accessing specific target scenes during training
and require retraining for new scenes. In contrast, domain generalization only trains using the source domain
and then gradually generalizes to unseen domains. However, existing methods based on domain generalization
ignore the impact of domain invariant semantics on the invariant representation of the domain. To solve the
above problem, an invariant semantic domain generalization shuffle network for cross-scene hyperspectral
image classification is proposed, which follows a framework on the generative adversarial network. Feature
style covariance in style and content randomization generator with invariant semantic features is designed to
safely extend the style and content of features without changing the domain invariant semantics. We proposed
a spatial shuffling discriminator, which can reduce the impact of special spatial relationships within the domain
on class semantics. In addition, we proposed a dual sampling direct adversarial contrastive learning strategy.
It uses a dual sampling in two-stage training design to prevent the model from lazily entering the local
nash equilibrium point. And based on dual sampling, directly adversarial contrastive learning using clearer
contrastive samples is used to reduce the difficulty of network training. We conduct extensive experiments
on four datasets and demonstrate that the proposed method outperforms other current domain generalization
methods.


Environment：
-------------------

pyhton 3.8

torch 1.11.0

If you find our work helpful to you, please cite it.
-------------------

@article{ ISDGS,
Author = {Gao, Jingpeng and Ji, Xiangyu and Ye, Fang and Chen, Geng},
Title = {Invariant semantic domain generalization shuffle network for cross-scene
   hyperspectral image classification},
Journal = {EXPERT SYSTEMS WITH APPLICATIONS},
Year = {2025},
Volume = {273},
Month = {MAY 10},
DOI = {10.1016/j.eswa.2025.126818},
Article-Number = {126818},
ISSN = {0957-4174},
EISSN = {1873-6793},
}
