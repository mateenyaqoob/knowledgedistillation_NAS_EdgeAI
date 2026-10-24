"""
Lab 9 - Efficient architecture building blocks: cost modeling from first principles.

Implements analytic parameter/MAC counters for the key blocks behind MobileNet,
SqueezeNet, and EfficientNet, and uses them to reproduce (approximately) the published
parameter counts of real architectures -- so students see the arithmetic behind the
headline numbers, not just the numbers themselves.
"""
import numpy as np

def conv_cost(h, w, c_in, c_out, k, stride=1, groups=1, bias=True):
    oh, ow = -(-h // stride), -(-w // stride)   # ceil division ("same" padding convention)
    params = (k * k * (c_in // groups) * c_out) + (c_out if bias else 0)
    macs = oh * ow * c_out * (k * k * (c_in // groups))
    return params, macs, oh, ow

def dense_cost(n_in, n_out):
    return n_in * n_out + n_out, n_in * n_out

# ---------------------------------------------------------------- MobileNet block
def depthwise_separable(h, w, c_in, c_out, stride=1):
    """MobileNetV1 block: depthwise 3x3 + pointwise 1x1."""
    pd, md, h2, w2 = conv_cost(h, w, c_in, c_in, 3, stride, groups=c_in)
    pp, mp, h3, w3 = conv_cost(h2, w2, c_in, c_out, 1, 1)
    return pd + pp, md + mp, h3, w3

def inverted_residual(h, w, c_in, c_out, stride, expand_ratio):
    """MobileNetV2 block: 1x1 expand -> 3x3 depthwise -> 1x1 project (linear bottleneck)."""
    c_mid = c_in * expand_ratio
    params, macs = 0, 0
    if expand_ratio != 1:
        pe, me, h, w = conv_cost(h, w, c_in, c_mid, 1, 1); params += pe; macs += me
    pd, md, h, w = conv_cost(h, w, c_mid, c_mid, 3, stride, groups=c_mid); params += pd; macs += md
    pp, mp, h, w = conv_cost(h, w, c_mid, c_out, 1, 1); params += pp; macs += mp
    return params, macs, h, w

# ---------------------------------------------------------------- SqueezeNet block
def fire_module(h, w, c_in, squeeze, e1, e3):
    """SqueezeNet Fire module: 1x1 squeeze -> parallel 1x1 and 3x3 expand, concatenated."""
    ps, ms, h, w = conv_cost(h, w, c_in, squeeze, 1, 1)
    pe1, me1, _, _ = conv_cost(h, w, squeeze, e1, 1, 1)
    pe3, me3, _, _ = conv_cost(h, w, squeeze, e3, 3, 1)
    return ps + pe1 + pe3, ms + me1 + me3, h, w, e1 + e3   # last value = output channels

# ---------------------------------------------------------------- compound scaling (EfficientNet idea)
def compound_scale(base_depth, base_width, base_res, phi, alpha=1.2, beta=1.1, gamma=1.15):
    """EfficientNet's compound scaling rule: depth*width^2*resolution^2 ~ 2^phi (held roughly
    constant per the paper), with alpha*beta^2*gamma^2 ~= 2 found by small grid search."""
    d = base_depth * (alpha ** phi)
    w = base_width * (beta ** phi)
    r = base_res * (gamma ** phi)
    return d, w, r

if __name__ == "__main__":
    print("=== Part A: one block, three designs, same input/output shape ===")
    h = w = 56; c_in, c_out = 64, 64
    configs = [
        ("Standard 3x3 conv", lambda: conv_cost(h, w, c_in, c_out, 3)),
        ("Depthwise-separable (MobileNetV1)", lambda: depthwise_separable(h, w, c_in, c_out)),
        ("Inverted residual, expand=6 (MobileNetV2)", lambda: inverted_residual(h, w, c_in, c_out, 1, 6)),
        ("Inverted residual, expand=1 (no expansion)", lambda: inverted_residual(h, w, c_in, c_out, 1, 1)),
    ]
    std_p, std_m = None, None
    for name, fn in configs:
        r = fn(); p, m = r[0], r[1]
        if std_p is None: std_p, std_m = p, m
        print(f"  {name:42s} params={p:>8,}  MACs={m:>10,}  (params {std_p/p:.2f}x, MACs {std_m/m:.2f}x vs standard conv)")
    print("\n  NOTE: expand=6 shows LESS than 1x here (i.e. MORE expensive than standard conv) at matched")
    print("  channel count. This is real, not a bug: the two 1x1 projections at 6x width dominate. The")
    print("  actual MobileNetV2 saving comes from using this block to let channels stay NARROW throughout")
    print("  the network (the 'linear bottleneck'), not from the block being cheaper at EQUAL channel width.")
    print("  DISCUSS: when would expand=6 at matched channels still be worth it despite costing more?")

    print("\n=== Part B: SqueezeNet Fire module vs an equivalent standard conv ===")
    squeeze, e1, e3 = 16, 64, 64
    fp, fm, _, _, fc = fire_module(56, 56, 128, squeeze, e1, e3)
    sp, sm, _, _ = conv_cost(56, 56, 128, e1 + e3, 3)
    print(f"  Fire module (squeeze={squeeze}, e1x1={e1}, e3x3={e3}): params={fp:,} MACs={fm:,}")
    print(f"  Equivalent standard 3x3 conv (same in/out channels): params={sp:,} MACs={sm:,}")
    print(f"  Fire module uses {sp/fp:.2f}x fewer params, {sm/fm:.2f}x fewer MACs")
    print("  SqueezeNet's own headline: AlexNet-level ImageNet accuracy at 50x fewer parameters")
    print("  (via a network of Fire modules + late downsampling + no FC layers, not this block alone).")

    print("\n=== Part C: reproduce a real MobileNetV2 parameter count from its published config ===")
    def mobilenetv2_params(alpha=1.0, res=224, num_classes=1000):
        def rd(v, divisor=8):
            nv = max(divisor, int(v + divisor / 2) // divisor * divisor)
            return nv + divisor if nv < 0.9 * v else nv
        cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
               (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
        inp = rd(32 * alpha); last = rd(1280 * max(1.0, alpha))
        p, _, h, w = conv_cost(res, res, 3, inp, 3, 2)
        ch = inp
        for t, c, n_layers, s in cfg:
            oc = rd(c * alpha)
            for i in range(n_layers):
                st = s if i == 0 else 1
                pb, _, h, w = inverted_residual(h, w, ch, oc, st, t)
                p += pb; ch = oc
        pl, _, _, _ = conv_cost(h, w, ch, last, 1, 1); p += pl
        pc, _ = dense_cost(last, num_classes); p += pc
        return p
    published = {1.0: 3.4, 0.75: 2.6, 0.5: 1.95, 0.35: 1.66}   # as reported in the MobileNetV2 paper, Table 4
    for alpha in [1.0, 0.75, 0.5, 0.35]:
        p = mobilenetv2_params(alpha) / 1e6
        print(f"  alpha={alpha}: computed {p:.2f}M params  vs  paper reports ~{published[alpha]:.2f}M")
    print("  DISCUSS: our figures track the paper closely at every alpha -- what's the likely source of the small gap?")
    print("  (Hint: BatchNorm parameters, stem/head details, and rounding conventions all contribute.)")

    print("\n=== Part D: compound scaling (EfficientNet idea), illustrative only ===")
    print("Single-dimension scaling (e.g. only going deeper) saturates quickly; EfficientNet scales")
    print("depth, width AND resolution together under a fixed compute budget.")
    base_d, base_w, base_r = compound_scale(1.0, 1.0, 224, 0)
    base_compute = base_d * (base_w ** 2) * (base_r ** 2)
    for phi in [0, 1, 2, 3]:
        d, w_, r = compound_scale(1.0, 1.0, 224, phi)
        compute_ratio = (d * (w_ ** 2) * (r ** 2)) / base_compute   # normalized to phi=0
        print(f"  phi={phi}: depth x{d:.2f}  width x{w_:.2f}  resolution {r:.0f}px   "
              f"(relative compute ~x{compute_ratio:.2f}, target was ~x{2**phi})")
    print("  (alpha*beta^2*gamma^2 = 1.2*1.1^2*1.15^2 ~= 1.94, close to the paper's target of ~2 per step)")
