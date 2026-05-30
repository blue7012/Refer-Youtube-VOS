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

print("\n✓ Tất cả OK — code sẵn sàng train!")
