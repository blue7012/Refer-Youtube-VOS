# -*- coding: utf-8 -*-
# ============================================================
# INFERENCE tren valid set -> ghi mask du doan theo dung format submission
# ------------------------------------------------------------
# Chay tu thu muc Baseline:
#     %cd /content/Refer-Youtube-VOS/Baseline
#     ! python infer_valid.py --arch base_model --dataset refer-yv-2019 --epoch 15
#
# Sinh ra:  <outdir>/Annotations/<vid>/<expr>/<frame>.png   (nhi phan {0,255})
#           <outdir>.zip                                      (de nop server / cham diem)
#
# Sau do cham diem:
#     ! python eval_benchmark.py --submission <outdir>
#
# TAI SU DUNG (khong tu viet lai):
#   - Trainer + load_model (giong pretrain.py) de dung model + trong so.
#   - trainer.scheme(..., eval=True) (base_scheme) -> lan truyen prev_mask y het
#     trainer.evaluate(); gt_masks chi dung cho loss nen truyen ZERO (bo qua loss).
#   - REFER_YV_2019(eval=True) chi de muon .corpus/.tokenize_sent/.resize/.size
#     (KHONG dung eval DataLoader vi no doc GT mask -> loi nested folder).
#
# Chay MOI expression rieng (theo meta_expressions.json) -> khop 6 thu muc GT.
# ============================================================
from __future__ import division
import warnings

warnings.simplefilter("ignore", UserWarning)

import os, sys, json, argparse, zipfile
from pathlib import Path

# trainer.py dung import tuong doi + ./data -> BAT BUOC cwd = Baseline.
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.append("utils/")
sys.path.append("models/")
sys.path.append("dataset/")

import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm

from trainer import Trainer
from utils.helpers import ToCuda
from dataset.refer_datasets import REFER_YV_2019

DATA_ROOT = Path("./data")


def main():
    def get_arguments():
        # Mirror pretrain.py de Trainer.__init__ co du args (init_lr, max_epoch, ...)
        parser = argparse.ArgumentParser(description="Refer-YouTube-VOS valid inference")
        parser.add_argument("--arch", type=str, default="base_model")
        parser.add_argument("--desc", type=str, default="")
        parser.add_argument("--eval", action="store_true")
        parser.add_argument("--eval_first", action="store_true")
        parser.add_argument("--init_lr", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--test_batch_size", type=int, default=0)
        parser.add_argument("--img_size", type=int, default=320, help="phai chia het 16")
        parser.add_argument("--max_epoch", type=int, default=150)
        parser.add_argument("--decay_epochs", type=int, default=[], nargs="+")
        parser.add_argument("--optimizer", type=str, default="adam")
        parser.add_argument("--lr_decay", type=float, default=0.1)
        parser.add_argument("--save_every", type=int, default=0)
        parser.add_argument("--max_N", type=int, default=0)
        parser.add_argument("--max_skip", type=int, default=2)
        parser.add_argument("--dataset", type=str, default="refer-yv-2019")
        parser.add_argument("--test_dataset", type=str, default=None)
        parser.add_argument("--splits", type=str, default=[], nargs="+")
        parser.add_argument("--checkpoint", type=str, default="")
        parser.add_argument("--epoch", type=int, default=15, help="checkpoint e{epoch}.pth")
        parser.add_argument("--no-eval", action="store_true")
        # rieng cho inference
        parser.add_argument("--split", type=str, default="valid", help="split de infer")
        parser.add_argument(
            "--outdir", type=str, default="", help="mac dinh: validation/<arch>_<split>_e<epoch>"
        )
        return parser.parse_args()

    args = get_arguments()
    assert args.img_size % 16 == 0, "img_size phai chia het 16 (model down-sample 16x)"

    out_dir = Path(args.outdir) if args.outdir else Path(
        "validation"
    ) / "{}_{}_e{}".format(args.arch, args.split, args.epoch)

    # --- Model: giong het pretrain.py (build -> cuda -> load_model -> eval) ---
    print("[INFER] >>> Building Trainer...", flush=True)
    trainer = Trainer(args)
    print("[INFER] >>> Moving model to CUDA...", flush=True)
    trainer.cuda()
    trainer.dataset = args.dataset  # set_dataset() bi bo qua -> gan thu cong cho load_model
    trainer.load_model(args.epoch)
    trainer.model.eval()
    print("[INFER] >>> Model ready (epoch loaded = {}).".format(trainer.epoch), flush=True)

    # --- Dataset chi de muon corpus/tokenize/resize/size (init nhanh, khong crash) ---
    testset = REFER_YV_2019(
        data_root=DATA_ROOT / "youtube-vos-2019",
        split=args.split,
        size=(args.img_size, args.img_size),
        eval=True,
    )

    meta_path = testset.data_root / testset.split / "meta_expressions.json"
    meta = json.load(open(meta_path))["videos"]
    print("[INFER] >>> {} video tu {}".format(len(meta), meta_path), flush=True)

    n_seq = 0
    for vid in tqdm(sorted(meta), dynamic_ncols=True):
        v = meta[vid]
        objs = v["objects"]

        for expr in sorted(v["expressions"], key=lambda x: int(x)):
            sent = v["expressions"][expr]["exp"]
            oid = v["expressions"][expr]["obj_id"]
            frame_ids = objs[oid]["frames"]  # = cac frame co GT cho object nay

            # nap + resize frame (y het load_pair/resize cua dataset), giu size goc
            frames, orig_size = [], None
            for fid in frame_ids:
                jpg = testset.data_root / testset.split / "JPEGImages" / vid / "{}.jpg".format(fid)
                pil = Image.open(jpg).convert("RGB")
                orig_size = pil.size  # (W, H) — dung de resize mask ve goc
                arr = np.float32(pil) / 255.0
                frm, _ = testset.resize(arr, np.zeros(arr.shape[:2], np.uint8), testset.size)
                frames.append(frm)

            Fs = torch.from_numpy(
                np.transpose(np.stack(frames, axis=0), (0, 3, 1, 2)).copy()
            ).float().unsqueeze(0)  # (1, T, 3, h, w)
            words = testset.tokenize_sent(sent).unsqueeze(0)  # (1, query_len)
            gt = torch.zeros(1, Fs.size(1), testset.size[0], testset.size[1])  # dummy cho loss

            Fs, gt, words = ToCuda([Fs, gt, words])
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=False):
                    est_masks, _, _, _ = trainer.scheme(Fs, gt, words, eval=True)  # (1, T, h, w)

            est = est_masks[0].detach().cpu().numpy()  # (T, h, w) — xac suat foreground
            save_dir = out_dir / "Annotations" / vid / expr
            save_dir.mkdir(parents=True, exist_ok=True)
            for t, fid in enumerate(frame_ids):
                pred = (est[t] > 0.5).astype(np.uint8)
                pred = cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)
                Image.fromarray((pred * 255).astype(np.uint8)).save(save_dir / "{}.png".format(fid))

            n_seq += 1

    print("[INFER] >>> Da ghi {} chuoi -> {}".format(n_seq, out_dir / "Annotations"), flush=True)

    # --- Nen thanh .zip de nop server (ZIP_STORED: PNG da nen san) ---
    zip_path = Path(str(out_dir) + ".zip")
    ann_root = out_dir / "Annotations"
    files = [p for p in ann_root.rglob("*") if p.is_file()]
    print("[INFER] >>> Nen {} file -> {} ...".format(len(files), zip_path), flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in files:
            zf.write(p, arcname=str(p.relative_to(out_dir).as_posix()))  # -> Annotations/<vid>/...
    print("[INFER] >>> XONG. Cham diem:  python eval_benchmark.py --submission {}".format(out_dir), flush=True)


if __name__ == "__main__":
    main()
