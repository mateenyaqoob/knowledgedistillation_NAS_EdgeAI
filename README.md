# Model Compression II Labs: Distillation, Efficient Architectures, NAS

Three labs, all NumPy/scikit-learn -- no deep-learning framework needed.
All three were executed end to end during preparation; numbers below are from that run.

| Lab | File | Lecture | What it teaches | Runtime |
|---|---|---|---|---|
| 8 | `lab8_distillation.py` | 1 (Distillation) | Soft targets, temperature, the two controls every distillation claim needs | ~15s |
| 9 | `lab9_efficient_architectures.py` | 2 (Architectures) | Analytic cost model for depthwise-separable, inverted-residual, Fire module; reproduces published MobileNetV2 parameter counts | ~1s |
| 10 | `lab10_nas_toy.py` | 2 (NAS) | A toy hardware-aware NAS: exhaustive ground truth + 3 search strategies | **~2.5 minutes** -- warn students before they run it |

## Setup
```
pip install numpy scikit-learn
```
Python 3.10+. No GPU needed.

## Key results (from the authoring run)
- **Lab 8 (distillation):** in a data-scarce regime, distilled students (90.5-91.3%) beat
  a hard-label student (88.2%) -- but a control using the teacher's HARD pseudo-labels on
  the same unlabelled pool reaches 91.9%, nearly matching distillation. In a data-RICH
  regime, the distilled student (95.4%) is slightly WORSE than the hard-label student
  (96.0%). A learning-rate sweep showed an apparent "T=4 collapse" (40.6% accuracy) was
  entirely a tuning artifact -- at the right learning rate, T=4 reaches 90.5%, in line
  with T=1 and T=2.
- **Lab 9 (architectures):** depthwise-separable gives 7.7-7.9x fewer params/MACs than a
  standard conv at matched channels. An inverted-residual block at expand=6, however, is
  actually MORE expensive (0.69-0.70x) than a standard conv at MATCHED channels -- the
  real MobileNetV2 saving is architecture-wide (keeping the residual path narrow), not a
  per-block property. The Fire module (SqueezeNet) IS cheaper per block: 11.9x fewer
  params. Our cost model reproduces the published MobileNetV2 parameter counts to within
  3% at every width multiplier (alpha = 1.0, 0.75, 0.5, 0.35).
- **Lab 10 (NAS):** the budget-aware reward optimum (890 params, 95.19% accuracy) is a
  genuinely different architecture from the accuracy-only optimum (10,730 params, 97.96%
  accuracy) -- 12x more parameters for under 3 accuracy points. All three search
  strategies (random, hill-climbing, evolutionary) reach within a few percent of the true
  optimum using only 16% of the full 125-candidate search space.

## Known limitations (share with students)
- Lab 8 uses a small MLP and the sklearn digits dataset. The qualitative lessons (soft
  targets carry information; controls matter; distillation's advantage tracks data
  scarcity) are the point -- not the exact percentages, which will not reproduce at
  ImageNet scale. The original paper's own large-scale results (a commercial speech
  acoustic model) show more clear-cut, larger benefits.
- Lab 9's cost model ignores BatchNorm parameters, bias-free convolutions in some
  variants, and exact stem/head details -- which is why the computed MobileNetV2 counts
  are close to, but not identical to, the published ones (within about 3%).
- Lab 10's search space (125 candidates, exhaustively enumerable) is dramatically smaller
  and smoother than any real NAS search space (often 10^10+ candidates). The finding that
  "random search does almost as well as hill-climbing or evolutionary search" is a
  property of this SMALL, SMOOTH space and should not be generalized to real NAS, where
  search strategy matters enormously more. The lab's own final discussion prompt asks
  students to reason about exactly this limitation.
- Lab 10's reward function uses parameter count as a latency/hardware-cost PROXY, unlike
  real hardware-aware NAS (e.g. MnasNet), which measures actual on-device latency. This
  is flagged explicitly in the lab and the lecture: FLOPS/parameter proxies are known to
  correlate poorly with real hardware latency.

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
