# ============================================================
# SMOKE TEST — kiểm tra syntax/import/forward pass, KHÔNG cần dataset
# Chạy từ thư mục Baseline: %cd /content/Refer-Youtube-VOS/Baseline
# ============================================================
import sys, os

os.chdir("/content/Refer-Youtube-VOS/Baseline")
sys.path.extend(["utils/", "models/", "dataset/"])

print("1. Kiểm tra imports...")
import torch
import torch.nn as nn

print(
    f"   torch={torch.__version__}, cuda={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0)}"
)

import torchvision

print(f"   torchvision={torchvision.__version__}")

import numpy as np

print(f"   numpy={np.__version__}")

print("\n2. Kiểm tra model load (pretrained ResNet50)...")
import importlib

model = importlib.import_module("models.base_model").Mask()
model = nn.DataParallel(model).cuda()
print(f"   Model OK — params={sum(p.numel() for p in model.parameters()):,}")

print("\n3. Kiểm tra forward pass với dummy data...")
B, C, H, W = 2, 3, 320, 320
query_len = 20

in_frames = torch.randn(B, C, H, W).cuda()
words = torch.randint(0, 12099, (B, query_len)).cuda()

with torch.no_grad():
    with torch.amp.autocast("cuda"):
        mask, logit = model(None, None, in_frames, words, eval=True)

print(f"   mask.shape  = {mask.shape}")  # kỳ vọng: (B, 2, H, W)
print(f"   logit.shape = {logit.shape}")

print("\n4. Kiểm tra AMP backward (scaler)...")
model.train()
criterion = nn.CrossEntropyLoss().cuda()
scaler = torch.amp.GradScaler("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

with torch.amp.autocast("cuda"):
    mask, logit = model(None, None, in_frames, words, eval=False)
    gt = torch.zeros(B, H, W, dtype=torch.long).cuda()
    loss = criterion(logit, gt)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print(f"   loss = {loss.item():.4f} — AMP backward OK")

print("\n5. Kiểm tra batch GPU metrics (iou_per_frame_gpu + boundary_f_score_gpu)...")
from helpers import iou_per_frame_gpu, boundary_f_score_gpu

model.eval()
T_fake = 4  # giả lập 4 frame
with torch.no_grad():
    with torch.amp.autocast("cuda"):
        # chạy T_fake lần, stack thành (B, T, H, W)
        preds = torch.stack(
            [model(None, None, in_frames, words, eval=True)[0][:, 1] for _ in range(T_fake)],
            dim=1
        )  # (B, T, H, W)

gt_seq = torch.zeros(B, T_fake, H, W, dtype=torch.float32).cuda()

iou_bt = iou_per_frame_gpu(preds, gt_seq)   # (B, T)
f_bt   = boundary_f_score_gpu(preds, gt_seq) # (B, T)

assert iou_bt.shape == (B, T_fake), f"iou shape sai: {iou_bt.shape}"
assert f_bt.shape   == (B, T_fake), f"f shape sai: {f_bt.shape}"
assert iou_bt.device.type == "cuda", "iou_bt phải ở trên GPU"
assert f_bt.device.type   == "cuda", "f_bt phải ở trên GPU"
print(f"   iou_per_frame_gpu   shape={tuple(iou_bt.shape)}, mean={iou_bt.mean():.4f} — OK")
print(f"   boundary_f_score_gpu shape={tuple(f_bt.shape)},  mean={f_bt.mean():.4f} — OK")

print("\n6. Kiểm tra get_disk_kernel_gpu cache...")
from helpers import get_disk_kernel_gpu
import math
r = max(1, int(math.ceil(0.008 * math.sqrt(320**2 + 320**2))))
k1 = get_disk_kernel_gpu(r, torch.device("cuda"))
k2 = get_disk_kernel_gpu(r, torch.device("cuda"))
assert k1 is k2, "disk kernel phải được cache (cùng object)"
assert k1.device.type == "cuda", "disk kernel phải ở GPU"
print(f"   disk radius={r}, kernel shape={tuple(k1.shape)}, cached=True — OK")

print("\n✓ Tất cả OK — code sẵn sàng train!")
