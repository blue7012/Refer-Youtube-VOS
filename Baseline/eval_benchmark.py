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
from utils.helpers import boundary_f_score_gpu


def log(msg):
    print("[EVAL] {}".format(msg), flush=True)


def _score_seq_gpu(gt_np, pred_np, bound_th, device, chunk):
    """Cham 1 chuoi tren GPU. gt_np/pred_np: (T,H,W) uint8 {0,1}. Tra (J[T], F[T]).
    Khop so y het metric CPU goc (da kiem chung |sai khac| = 0)."""
    T = gt_np.shape[0]
    Js = np.empty(T, np.float32)
    Fs = np.empty(T, np.float32)
    for s in range(0, T, chunk):  # chunk theo frame de gioi han VRAM
        e = min(s + chunk, T)
        g = torch.from_numpy(gt_np[s:e]).to(device=device, dtype=torch.float32).unsqueeze(0)
        p = torch.from_numpy(pred_np[s:e]).to(device=device, dtype=torch.float32).unsqueeze(0)
        Fs[s:e] = boundary_f_score_gpu(p, g, bound_th=bound_th)[0].cpu().numpy()
        t = g.shape[1]
        gg = (g > 0.5).float().reshape(t, -1)
        pp = (p > 0.5).float().reshape(t, -1)
        inter = (pp * gg).sum(-1)
        union = ((pp + gg) - pp * gg).sum(-1)
        iou = inter / (union + 1e-5)
        both_empty = (gg.sum(-1) == 0) & (pp.sum(-1) == 0)  # khop db_eval_iou: ca 2 rong -> 1
        Js[s:e] = torch.where(both_empty, torch.ones_like(iou), iou).cpu().numpy()
    return Js, Fs


def _score_seq_cpu(gt_np, pred_np, bound_th):
    """Cham 1 chuoi tren CPU (ham skimage goc) — fallback khi khong co CUDA / --cpu."""
    T = gt_np.shape[0]
    Js = np.empty(T, np.float32)
    Fs = np.empty(T, np.float32)
    for t in range(T):
        gb = gt_np[t].astype(np.float32)
        pb = pred_np[t].astype(np.float32)
        Js[t] = float(db_eval_iou(gb, pb))
        Fs[t] = float(db_eval_boundary(
            torch.from_numpy(pb[None]), torch.from_numpy(gb[None]), bound_th=bound_th))
    return Js, Fs


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
        parser.add_argument(
            "--frame_chunk", type=int, default=16,
            help="so frame cham cung luc tren GPU (gioi han VRAM)",
        )
        parser.add_argument(
            "--cpu", action="store_true",
            help="ep cham bang CPU (cham hon nhieu, dung de doi chieu so)",
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

    # --- Thiet bi: mac dinh GPU (nhanh hon hang tram lan), fallback CPU ---
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    log("Metric chay tren: {} (frame_chunk={})".format(
        "GPU/cuda" if use_cuda else "CPU", args.frame_chunk))

    # Liet ke truoc TAT CA chuoi (vid, expr) -> tqdm co tong so + ETA chinh xac
    seqs = []  # (gt_vid, sub_vid, expr)
    n_reconciled = 0
    n_vid_missing = 0
    for gt_vid in gt_vids:
        sub_vid, reconciled = reconcile_vid(gt_vid, sub_vids)
        if reconciled:
            n_reconciled += 1
        if sub_vid is None:
            n_vid_missing += 1
        exprs = sorted(
            e.name
            for e in (gt_root / gt_vid).iterdir()
            if e.is_dir() and not e.name.startswith(".")
        )
        for expr in exprs:
            seqs.append((gt_vid, sub_vid, expr))
    log("{} chuoi GT / {} video ({} video thieu submission, {} ID khop theo prefix)".format(
        len(seqs), len(gt_vids), n_vid_missing, n_reconciled))

    J = AverageMeter("J", ":3.4f")
    F = AverageMeter("F", ":3.4f")
    rows = []  # cho per_seq_csv
    n_seq_total = 0
    n_seq_predicted = 0

    pbar = tqdm(seqs, dynamic_ncols=True, unit="seq")
    for gt_vid, sub_vid, expr in pbar:
        n_seq_total += 1
        pred_frames = frames_map.get(sub_vid, {}).get(expr, {}) if sub_vid else {}
        if len(pred_frames) > 0:
            n_seq_predicted += 1

        gt_frames = sorted(
            f.name
            for f in (gt_root / gt_vid / expr).glob("*.png")
            if not f.name.startswith(".")
        )
        if len(gt_frames) == 0:
            continue

        # Nap GT + pred cho ca chuoi -> (T, H, W) uint8 {0,1}
        gt_list, pred_list = [], []
        H = W = None
        for frame in gt_frames:
            gt = np.uint8(Image.open(gt_root / gt_vid / expr / frame).convert("P"))
            gt_bin = (gt > 0).astype(np.uint8)
            if H is None:
                H, W = gt_bin.shape
            handle = pred_frames.get(frame)
            if handle is None:
                pred_bin = np.zeros((H, W), np.uint8)  # thieu prediction -> mask rong
            else:
                pred_bin = (read_pred(handle) > 0).astype(np.uint8)
                if pred_bin.shape != (H, W):  # an toan: ve dung size GT
                    pred_bin = cv2.resize(pred_bin, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_list.append(gt_bin)
            pred_list.append(pred_bin)

        gt_np = np.stack(gt_list)      # (T, H, W)
        pred_np = np.stack(pred_list)  # (T, H, W)

        if use_cuda:
            seq_j, seq_f = _score_seq_gpu(gt_np, pred_np, args.bound_th, device, args.frame_chunk)
        else:
            seq_j, seq_f = _score_seq_cpu(gt_np, pred_np, args.bound_th)

        mean_j = float(seq_j.mean())
        mean_f = float(seq_f.mean())
        J.update(mean_j)  # moi chuoi (vid,expr) 1 phieu, giong trainer.evaluate
        F.update(mean_f)
        rows.append((gt_vid, expr, mean_j, mean_f))
        pbar.set_postfix(J="{:.3f}".format(J.avg), F="{:.3f}".format(F.avg))
    pbar.close()

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
