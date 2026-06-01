# -*- coding: utf-8 -*-
"""
nan_diagnosis.py  --  Empirical proof of WHY the loss goes nan and WHICH fix works.

RUN ON COLAB (GPU REQUIRED):
    %cd /content/Refer-Youtube-VOS/Baseline
    !python nan_diagnosis.py

Why GPU is mandatory:
    The bug is fp16 (half precision) OVERFLOW inside the attention soft-max.
    CUDA autocast uses float16 (max value ~= 65504) -> it can overflow.
    CPU autocast uses bfloat16 (same range as float32) -> it CANNOT reproduce
    the bug. So this must run on a CUDA device to be meaningful.

What it proves, in 3 experiments:
    EXP 1  The attention has NO 1/sqrt(d) scaling -> in fp16 the logits overflow
           to +inf and softmax(+inf) = nan.  (identifies the TRIGGER -> Fix B)
    EXP 2  Once weights cross the overflow threshold, every batch -- even fresh,
           clean data -- produces nan grads, GradScaler SKIPS every step, the
           weights are frozen at the bad point => permanent nan. (the PERMANENCE)
    EXP 3  Fix matrix: run the REAL Mask() model under AMP with each fix toggled
           and report which configuration actually stays finite.

Configs in EXP 3:
    control  : no inflation, no fix      -> expected FINITE (model is fine at init)
    baseline : inflated, no fix          -> expected NAN     (reproduces the bug)
    C only   : inflated, grad-clip       -> does clipping rescue a forward overflow?
    A only   : inflated, freeze-BN       -> does freezing BN rescue it?
    B only   : inflated, scaled-attn     -> does scaling the logits rescue it?
    A+B+C    : inflated, all three       -> the proposed full fix

"inflation" = multiply the attention Query/Key weights by a constant. This is a
fast, deterministic stand-in for "the weights grew over 2 epochs of training",
so we don't have to train for hours to reach the overflow regime.
"""

from __future__ import division
import os, sys, math, types, warnings
warnings.simplefilter("ignore")

# ---- make the repo's own imports resolve (run from Baseline/) -------------------
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
for p in [HERE, os.path.join(HERE, "utils"), os.path.join(HERE, "models"),
          os.path.join(HERE, "dataset")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.nn.functional as F

HALF_MAX = 65504.0  # largest finite value representable in float16

def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)

def require_cuda():
    if not torch.cuda.is_available():
        print("!! CUDA NOT AVAILABLE. This script must run on a GPU runtime.")
        print("!! On Colab: Runtime > Change runtime type > GPU.")
        sys.exit(1)
    print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))


# =================================================================================
# EXP 1 -- the structural trigger: unscaled fp16 dot-products overflow
# =================================================================================
def experiment1_overflow():
    hr("EXP 1  |  Does the attention overflow in fp16 because it is unscaled?")
    print("Setup: query/key shaped like CrossAtt's reshaped tensors (B*heads=16, "
          "d_k=64, L*H*W=1280).")
    print("We grow the feature magnitude 's' (simulating weight growth) and compare")
    print("the raw fp16 attention (as written in the code) vs a 1/sqrt(d_k)-scaled")
    print("fp32 softmax (Fix B).\n")
    print(f"{'feat_scale':>10} | {'max|logit|(fp16)':>17} | {'#inf logits':>11} | "
          f"{'#nan after softmax':>18} | {'#nan WITH Fix B':>15}")
    print("-" * 84)

    B, C, HW = 16, 64, 1280
    for s in [1.0, 2.0, 4.0, 8.0, 16.0]:
        torch.manual_seed(0)
        q = (torch.randn(B, C, HW) * s).cuda().half()
        k = (torch.randn(B, C, HW) * s).cuda().half()

        # --- exactly what the code does: raw bmm in fp16, then softmax ---
        att16 = torch.bmm(q.transpose(1, 2), k)            # fp16, no scaling
        n_inf = torch.isinf(att16).sum().item()
        finite_max = att16[torch.isfinite(att16)].abs().max().item() \
                     if torch.isfinite(att16).any() else float("inf")
        sm16 = torch.softmax(att16, dim=2)
        n_nan = torch.isnan(sm16).sum().item()

        # --- Fix B: divide by sqrt(d_k) and do softmax in fp32 ---
        att_fix = torch.bmm(q.transpose(1, 2).float(), k.float()) / math.sqrt(C)
        sm_fix = torch.softmax(att_fix, dim=2)
        n_nan_fix = torch.isnan(sm_fix).sum().item()

        flag = "  <-- OVERFLOW -> nan" if n_nan > 0 else ""
        print(f"{s:>10.1f} | {finite_max:>17.1f} | {n_inf:>11d} | "
              f"{n_nan:>18d} | {n_nan_fix:>15d}{flag}")

    print(f"\nfloat16 max finite value = {HALF_MAX:.0f}. Any logit above it becomes +inf,")
    print("and softmax(+inf) = inf/inf = nan. Fix B keeps the nan count at 0 throughout.")


# =================================================================================
# Helpers that build / patch the REAL model for EXP 2 and EXP 3
# =================================================================================
def build_model():
    from models.base_model import Mask
    return Mask().cuda()

def inflate_attention(model, factor):
    """Scale attention Query/Key weights -> stand-in for 'weights grew over epochs'."""
    if factor == 1.0:
        return
    with torch.no_grad():
        for m in model.modules():
            cn = type(m).__name__
            if cn in ("CrossAtt", "ConvSA", "LinearSA"):
                if hasattr(m, "Query"):
                    m.Query.weight.mul_(factor)
                if hasattr(m, "Key"):
                    m.Key.weight.mul_(factor)

def freeze_encoder_bn(model):
    """Fix A: put encoder BatchNorm in eval mode + stop its affine grads."""
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for prm in m.parameters():
                prm.requires_grad_(False)

# ---- Fix B: scaled / fp32-softmax replacements for the 3 attention forwards ----
def _crossatt_forward_scaled(self, vis, emb):
    B, C_v, H, W = vis.size()
    B, L, C_l = emb.size()
    vis_att = self.convSA(vis)
    lang_att = self.linearSA(emb).transpose(1, 2)
    vis_ = vis_att.unsqueeze(2).repeat(1, 1, L, 1, 1)
    emb_ = lang_att.unsqueeze(3).unsqueeze(3).repeat(1, 1, 1, H, W)
    multi = torch.cat((vis_, emb_), 1)
    multi = multi.transpose(1, 2).reshape(B * L, -1, H, W)
    query = self.Query(multi).view(B, L, -1, H, W)
    key = self.Key(multi).view(B, L, -1, H, W)
    value = self.Value(multi).view(B, L, -1, H, W)
    res_value = self.resValue(multi).view(B, L, -1, H, W)
    query = query.transpose(1, 2); key = key.transpose(1, 2)
    value = value.transpose(1, 2); res_value = res_value.transpose(1, 2)
    query_ = query.reshape(B * self.head_num, -1, L * H * W)
    key_ = key.reshape(B * self.head_num, -1, L * H * W)
    value_ = value.reshape(B * self.head_num, -1, L * H * W)
    d_k = query_.size(1)
    att = torch.bmm(query_.transpose(1, 2), key_) / math.sqrt(d_k)   # <-- scaling
    att = F.softmax(att.float(), dim=2).to(value_.dtype)             # <-- fp32 softmax
    v_att = torch.bmm(value_, att.transpose(1, 2))
    v_att = v_att.reshape(B, -1, L, H, W)
    return torch.mean(v_att + res_value, 2)

def _convsa_forward_scaled(self, feat):
    B, C, H, W = feat.size()
    query = self.Query(feat); key = self.Key(feat); value = self.Value(feat)
    query_ = query.reshape(B, -1, H * W)
    key_ = key.reshape(B, -1, H * W)
    value_ = value.reshape(B, -1, H * W)
    d_k = query_.size(1)
    att = torch.bmm(query_.transpose(1, 2), key_) / math.sqrt(d_k)
    att = F.softmax(att.float(), dim=2).to(value_.dtype)
    v_att = torch.bmm(value_, att.transpose(1, 2)).view(B, -1, H, W)
    return v_att + value

def _linearsa_forward_scaled(self, feat):
    B, L, C = feat.size()
    query = self.Query(feat); key = self.Key(feat); value = self.Value(feat)
    d_k = query.size(-1)
    att = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_k)
    att = F.softmax(att.float(), dim=2).to(value.dtype)
    v_att = torch.bmm(att, value)
    return v_att + value

def patch_attention_scaled(model):
    for m in model.modules():
        cn = type(m).__name__
        if cn == "CrossAtt":
            m.forward = types.MethodType(_crossatt_forward_scaled, m)
        elif cn == "ConvSA":
            m.forward = types.MethodType(_convsa_forward_scaled, m)
        elif cn == "LinearSA":
            m.forward = types.MethodType(_linearsa_forward_scaled, m)

# ---- synthetic CLEAN data (no 'poisoned sample' anywhere) ----------------------
DICT_SIZE, L = 12099, 20
def synth_batch(B, N, H, W, seed):
    g = torch.Generator().manual_seed(seed)
    frames = torch.rand(B, N, 3, H, W, generator=g)
    masks = torch.zeros(B, N, H, W)
    masks[:, :, H // 4:3 * H // 4, W // 4:3 * W // 4] = 1.0   # centered square
    words = torch.randint(0, DICT_SIZE, (B, L), generator=g)
    return frames.cuda(), masks.cuda(), words.cuda()

def forward_scheme(model, crit, frames, masks, words):
    """Faithful copy of Trainer.base_scheme(eval=False): 2-frame recurrent loss."""
    N = frames.size(1)
    loss = 0.0
    prev_frame, prev_mask = None, None
    for n in range(N):
        _, logit = model(prev_frame, prev_mask, frames[:, n], words, eval=False)
        loss = loss + crit(logit, masks[:, n].long()).mean()
        prev_frame, prev_mask = frames[:, n], masks[:, n]
    return loss

def grads_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


# =================================================================================
# EXP 2 -- the permanence: GradScaler freezes weights at the overflow threshold
# =================================================================================
def experiment2_permanence():
    hr("EXP 2  |  Why does nan PERSIST forever, even on clean batches?")
    print("Real Mask() model, AMP fp16, weights inflated past the overflow threshold.")
    print("Each step uses FRESH CLEAN synthetic data (no bad sample exists).")
    print("Watch: once loss is nan, grads are nan -> GradScaler SKIPS the step")
    print("(scale drops, params unchanged) -> next clean batch overflows again.\n")

    B, N, H, W = 2, 2, 256, 256
    model = build_model(); inflate_attention(model, 8.0)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=7e-4)
    scaler = torch.amp.GradScaler('cuda')
    crit = nn.CrossEntropyLoss(reduction='none').cuda()

    print(f"{'step':>4} | {'loss':>10} | {'loss finite':>11} | {'grads finite':>12} | "
          f"{'scaler.scale':>13} | {'step taken?':>11}")
    print("-" * 76)
    for step in range(8):
        model.train()
        frames, masks, words = synth_batch(B, N, H, W, seed=1000 + step)
        opt.zero_grad()
        scale_before = scaler.get_scale()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = forward_scheme(model, crit, frames, masks, words)
        lfin = bool(torch.isfinite(loss).item())
        scaler.scale(loss).backward()
        gfin = grads_finite(model)
        scaler.step(opt); scaler.update()
        # heuristic: if scale dropped, GradScaler detected non-finite grads & skipped
        took_step = scaler.get_scale() >= scale_before and gfin
        print(f"{step:>4} | {loss.item():>10.4f} | {str(lfin):>11} | {str(gfin):>12} | "
              f"{scale_before:>13.1f} | {str(bool(took_step)):>11}")

    print("\nReading: 'grads finite = False' -> GradScaler skips the optimizer step ->")
    print("weights stay frozen at the overflowing values -> every later (clean) batch is")
    print("nan too. This is a DEADLOCK, not a bad data sample. The data was clean each step.")


# =================================================================================
# EXP 3 -- the fix matrix
# =================================================================================
def run_config(name, fixA, fixB, fixC, inflate, steps=8, lr=7e-4):
    B, N, H, W = 2, 2, 256, 256
    torch.manual_seed(0)
    model = build_model()
    inflate_attention(model, inflate)
    if fixB:
        patch_attention_scaled(model)
    if fixA:
        freeze_encoder_bn(model)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    crit = nn.CrossEntropyLoss(reduction='none').cuda()

    first_nan = None
    n_nan = 0
    last_finite = True
    for step in range(steps):
        model.train()
        if fixA:
            freeze_encoder_bn(model)  # re-assert: model.train() flips BN back on
        frames, masks, words = synth_batch(B, N, H, W, seed=2000 + step)
        opt.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = forward_scheme(model, crit, frames, masks, words)
        finite = bool(torch.isfinite(loss).item())
        last_finite = finite
        if not finite:
            n_nan += 1
            if first_nan is None:
                first_nan = step
        scaler.scale(loss).backward()
        if fixC:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

    verdict = "FINITE (good)" if n_nan == 0 else \
              (f"NAN from step {first_nan}, {n_nan}/{steps} nan; "
               f"{'recovered' if last_finite else 'STILL nan at end'}")
    print(f"  {name:<34} -> {verdict}")
    return n_nan == 0

def experiment3_matrix():
    hr("EXP 3  |  Fix matrix on the REAL model (which configuration stays finite?)")
    print("All steps use fresh CLEAN data. 'inflated' = weights pushed into the")
    print("overflow regime (= after ~2 epochs of growth).\n")
    run_config("control  (no inflation, no fix)", False, False, False, inflate=1.0)
    run_config("baseline (inflated, no fix)",     False, False, False, inflate=8.0)
    run_config("C only   (inflated, grad-clip)",  False, False, True,  inflate=8.0)
    run_config("A only   (inflated, freeze-BN)",  True,  False, False, inflate=8.0)
    run_config("B only   (inflated, scaled-attn)",False, True,  False, inflate=8.0)
    run_config("A+B+C    (inflated, full fix)",   True,  True,  True,  inflate=8.0)
    print("\nInterpretation:")
    print("  - 'control' finite  => the model is healthy until the weights grow.")
    print("  - 'baseline' nan    => growth -> fp16 overflow -> nan (reproduced).")
    print("  - Compare C/A/B-only to see which one actually removes the nan.")
    print("  - Fix B (scaling) attacks the ROOT (the overflow itself);")
    print("    A and C only slow the drift toward the overflow regime.")


if __name__ == "__main__":
    require_cuda()
    experiment1_overflow()
    experiment2_permanence()
    experiment3_matrix()
    hr("DONE")
    print("Expected takeaway: Fix B is the decisive root-cause fix (no overflow, no nan).")
    print("Fix A removes a growth driver (trainable BN) and is recommended alongside B.")
    print("Fix C (clip) is cheap insurance but does NOT by itself stop a forward overflow.")
