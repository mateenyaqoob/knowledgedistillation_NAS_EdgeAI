# Model Compression Labs: Knowledge Distillation, Efficient Net, Neural Architecture Search

# Model Compression II: Annotated Reading List

All items verified to exist with the stated venue and year during preparation.

## Knowledge Distillation (Lecture 1)
1. **Hinton, Vinyals, Dean.** *Distilling the Knowledge in a Neural Network.*
   arXiv 1503.02531, 2015. The foundational paper; read in full. Origin of
   "dark knowledge" and the temperature-scaled soft-target loss.
2. **Romero, Ballas, Kahou, Chassang, Gatta, Bengio.** *FitNets: Hints for Thin
   Deep Nets.* arXiv 1412.6550, 2015. Extends distillation to intermediate
   (feature-level) representations, not just final logits.
3. **Sanh, Debut, Chaumond, Wolf.** *DistilBERT, a distilled version of BERT:
   smaller, faster, cheaper and lighter.* arXiv 1910.01108, 2019. A clear,
   large-scale success story: 40% smaller, 60% faster, 97% of performance retained.
4. **Furlanello, Lipton, Tschannen, Itti, Anandkumar.** *Born-Again Neural
   Networks.* ICML 2018. Self-distillation, where a same-architecture student
   can exceed its teacher.
5. **Gou, Yu, Maybank, Tao.** *Knowledge Distillation: A Survey.* IJCV 2021.
   Comprehensive overview of variants and applications.

## Efficient Architectures (Lecture 2)
6. **Howard et al.** *MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications.* arXiv 1704.04861, 2017. Depthwise-separable
   convolutions; the width/resolution multiplier idea.
7. **Sandler, Howard, Zhu, Zhmoginov, Chen.** *MobileNetV2: Inverted Residuals
   and Linear Bottlenecks.* CVPR 2018.
8. **Iandola, Han, Moskewicz, Ashraf, Dally, Keutzer.** *SqueezeNet: AlexNet-level
   accuracy with 50x fewer parameters and <0.5MB model size.* arXiv 1602.07360,
   2016. Note: co-author Song Han also wrote Deep Compression (Model Compression
   I unit) -- the two techniques are designed to combine.
9. **Tan, Le.** *EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks.* ICML 2019. Compound scaling of depth, width and resolution together.

## Neural Architecture Search (Lecture 2)
10. **Zoph, Le.** *Neural Architecture Search with Reinforcement Learning.*
    ICLR 2017. The foundational NAS paper (not hardware-aware).
11. **Tan, Chen, Pang, Vasudevan, Sandler, Howard, Le.** *MnasNet: Platform-Aware
    Neural Architecture Search for Mobile.* CVPR 2019. Hardware-aware NAS with a
    weighted-product reward combining accuracy and REAL measured on-device latency.
12. **Liu, Simonyan, Yang.** *DARTS: Differentiable Architecture Search.*
    ICLR 2019. Gradient-based search, far cheaper than reinforcement-learning NAS.
13. **Cai, Gan, Wang, Zhang, Han.** *Once for All: Train One Network and
    Specialize it for Efficient Deployment.* ICLR 2020. Train one large
    "supernet" once, extract many deployment-specific sub-networks without
    retraining each from scratch.

## Suggested paper-reading exercise
Assign items 9 (EfficientNet) and 11 (MnasNet) as a pair: EfficientNet's base
architecture (B0) was itself found by a MnasNet-style search, then scaled by
hand via the compound scaling rule. Ask students to identify which parts of
the EfficientNet design are HAND-DESIGNED (the scaling rule, the exponents)
and which are SEARCHED (the base architecture's block choices) -- a good test
of whether they understand where NAS fits relative to hand-designed efficient
architectures.
