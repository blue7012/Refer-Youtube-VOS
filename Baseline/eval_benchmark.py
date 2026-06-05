# -*- coding: utf-8 -*-
# ============================================================
# BENCHMARK J & F cho Refer-YouTube-VOS valid (offline)
# ------------------------------------------------------------
# Chay tu thu muc Baseline:
#     %cd /content/Refer-Youtube-VOS/Baseline
#     ! python eval_benchmark.py --submission valid_sample_submission.zip
#
# Metric: Region Jaccard (J) + Boundary F (F). DUNG LAI ham co san trong
#   utils/eval_utils.py (db_eval_iou, db_eval_boundary, AverageMeter) de khop
#   y het voi trainer.evaluate(), KHONG tu viet metric moi.
#
# Tong hop (giong trainer.evaluate / script tham chieu cua ban):
#   - moi chuoi (video, expression): J,F = trung binh tren cac frame co GT
#   - chuoi 1 phieu (AverageMeter) -> J&F = (mean_seq J + mean_seq F) / 2
#
# Pham vi: cham diem TOAN BO chuoi GT (202 video). Chuoi khong xuat hien trong
#   submission -> coi nhu mask rong (J/F ~ 0). ID submission lech 1 ky tu duoi
#   (vd 0062f687f10 vs 0062f687f1) duoc tu dong khop theo prefix + canh bao.
#
# GT     = data/youtube-vos-2019/valid/Annotations/<vid>/<expr>/<frame>.png  ({0,255})
# SUB    = Annotations/<vid>/<expr>/<frame>.png  (zip hoac thu muc, ({0,255}))
# ============================================================
from __future__ import division
import warnings

warnings.simplefilter("ignore", UserWarning)

import os, io, sys, csv, zipfile, argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# trainer.py / 0_prepare_data.py deu BAT BUOC cwd = Baseline (import tuong doi +
# duong dan ./data). Chuyen cwd ve thu muc chua file nay roi gan path nhu trainer.
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.append("utils/")
sys.path.append("models/")
sys.path.append("dataset/")

from eval_utils import db_eval_iou, db_eval_boundary, AverageMeter


def log(msg):
    print("[EVAL] {}".format(msg), flush=True)


def is_pred_png(name):
    """Loc entry hop le: Annotations/<vid>/<expr>/<frame>.png. Bo __MACOSX, ._*,
    .ipynb_checkpoints va moi thanh phan an (bat dau bang '.')."""
    if not name.endswith(".png"):
        return False
    parts = name.split("/")
    if len(parts) != 4 or parts[0] != "Annotations":
        return False
    return not any(p.startswith(".") for p in parts)


def build_submission_index(sub_path):
    """Doc submission (zip hoac thu muc) -> (frames_map, reader, sub_vids).
    frames_map: {vid: {expr: {frame_png: handle}}}
    reader(handle) -> np.uint8 mask array."""
    frames_map = defaultdict(lambda: defaultdict(dict))

    if sub_path.endswith(".zip"):
        zf = zipfile.ZipFile(sub_path)
        for n in zf.namelist():
            if not is_pred_png(n):
                continue
            _, vid, expr, frame = n.split("/")
            frames_map[vid][expr][frame] = n

        def reader(handle):
            return np.uint8(Image.open(io.BytesIO(zf.read(handle))).convert("P"))

    else:
        root = Path(sub_path)
        base = root / "Annotations" if (root / "Annotations").is_dir() else root
        for p in base.rglob("*.png"):
            rel = p.relative_to(base).parts
            if len(rel) != 3 or any(s.startswith(".") for s in rel):
                continue
            vid, expr, frame = rel
            frames_map[vid][expr][frame] = str(p)

        def reader(handle):
            return np.uint8(Image.open(handle).convert("P"))

    sub_vids = sorted(frames_map.keys())
    return frames_map, reader, sub_vids


def reconcile_vid(gt_vid, sub_vids):
    """Khop GT vid (10 ky tu) voi vid trong submission. Tra ve (sub_vid|None, reconciled)."""
    if gt_vid in sub_vids:
        return gt_vid, False
    cands = [s for s in sub_vids if s.startswith(gt_vid)]  # vd 0062f687f1 -> 0062f687f10
    if len(cands) == 1:
        return cands[0], True
    return None, False


def main():
    def get_arguments():
        parser = argparse.ArgumentParser(description="Refer-YouTube-VOS J&F benchmark")
        parser.add_argument(
            "--gt",
            type=str,
            default="data/youtube-vos-2019/valid/Annotations",
            help="thu muc GT Annotations (<vid>/<expr>/<frame>.png)",
        )
        parser.add_argument(
            "--submission",
            type=str,
            default="valid_sample_submission.zip",
            help="file .zip hoac thu muc chua Annotations/<vid>/<expr>/<frame>.png",
        )
        parser.add_argument(
            "--bound_th", type=float, default=0.008, help="nguong bien cho F (db_eval_boundary)"
        )
        parser.add_argument(
            "--per_seq_csv", type=str, default="", help="(tuy chon) ghi vid,expr,J,F ra CSV"
        )
        return parser.parse_args()

    args = get_arguments()

    gt_root = Path(args.gt)
    if not gt_root.is_dir():
        raise SystemExit("[EVAL] Khong thay thu muc GT: {}".format(gt_root))

    log("Doc submission: {}".format(args.submission))
    frames_map, read_pred, sub_vids = build_submission_index(args.submission)
    log("Submission co {} video.".format(len(sub_vids)))

    gt_vids = sorted(
        d.name for d in gt_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    log("GT co {} video -> cham diem TOAN BO chuoi GT.".format(len(gt_vids)))

    J = AverageMeter("J", ":3.4f")
    F = AverageMeter("F", ":3.4f")

    rows = []  # cho per_seq_csv
    n_seq_total = 0
    n_seq_predicted = 0
    n_vid_missing = 0

    for gt_vid in tqdm(gt_vids, dynamic_ncols=True):
        sub_vid, reconciled = reconcile_vid(gt_vid, sub_vids)
        if reconciled:
            log("ID lech -> khop {} -> {}".format(gt_vid, sub_vid))
        if sub_vid is None:
            n_vid_missing += 1

        exprs = sorted(
            e.name
            for e in (gt_root / gt_vid).iterdir()
            if e.is_dir() and not e.name.startswith(".")
        )

        for expr in exprs:
            n_seq_total += 1
            pred_frames = frames_map.get(sub_vid, {}).get(expr, {}) if sub_vid else {}
            if len(pred_frames) > 0:
                n_seq_predicted += 1

            gt_frames = sorted(
                f.name
                for f in (gt_root / gt_vid / expr).glob("*.png")
                if not f.name.startswith(".")
            )

            seq_j, seq_f = [], []
            for frame in gt_frames:
                gt = np.uint8(Image.open(gt_root / gt_vid / expr / frame).convert("P"))
                gt_bin = (gt > 0).astype(np.float32)

                handle = pred_frames.get(frame)
                if handle is None:
                    pred_bin = np.zeros_like(gt_bin)  # thieu prediction -> mask rong
                else:
                    pr = read_pred(handle)
                    pred_bin = (pr > 0).astype(np.float32)
                    if pred_bin.shape != gt_bin.shape:  # an toan: ve dung size GT
                        pred_bin = cv2.resize(
                            pred_bin,
                            (gt_bin.shape[1], gt_bin.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                # J: numpy; F: tensor (1,H,W) dung db_eval_boundary co san
                j = db_eval_iou(gt_bin, pred_bin)
                f = db_eval_boundary(
                    torch.from_numpy(pred_bin[None]),
                    torch.from_numpy(gt_bin[None]),
                    bound_th=args.bound_th,
                )
                seq_j.append(float(j))
                seq_f.append(float(f))

            if len(seq_j) == 0:
                continue

            mean_j = float(np.mean(seq_j))
            mean_f = float(np.mean(seq_f))
            J.update(mean_j)  # moi chuoi (vid,expr) 1 phieu, giong trainer.evaluate
            F.update(mean_f)
            rows.append((gt_vid, expr, mean_j, mean_f))

    jf = (J.avg + F.avg) / 2

    log("==================== KET QUA ====================")
    log("J (region Jaccard) : {:.4f}".format(J.avg))
    log("F (boundary)       : {:.4f}".format(F.avg))
    log("J&F                : {:.4f}".format(jf))
    log("coverage           : du doan {}/{} chuoi ({} video GT thieu submission)".format(
        n_seq_predicted, n_seq_total, n_vid_missing
    ))

    if args.per_seq_csv:
        with open(args.per_seq_csv, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["vid", "expr", "J", "F"])
            writer.writerows(rows)
        log("Da ghi per-sequence CSV -> {}".format(args.per_seq_csv))


if __name__ == "__main__":
    main()
