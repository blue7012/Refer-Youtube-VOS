# -*- coding: utf-8 -*-
# ============================================================
# CHUAN BI DATA + CACHE PKL QUA HUGGINGFACE
# ------------------------------------------------------------
# Chay tu Colab:
#     %cd /content/Refer-Youtube-VOS/Baseline
#     ! /usr/local/miniconda/envs/py38/bin/python /content/Refer-Youtube-VOS/Baseline/*.py
#
# Vi file nay ten "0_prepare_data.py" nen khi bash bung glob "*.py" no dung DAU
# danh sach -> Python chi chay file nay, cac file con lai chi roi vao sys.argv
# (vo hai vi script KHONG dung argparse).
#
# Logic:
#   - Repo HF CHUA ton tai  -> build pkl (mimic pretrain.py) roi upload TOAN BO
#                              data/ (dataset + pkl + vocabulary_Gref.txt + corpus.pth)
#   - Repo HF DA ton tai     -> chi tai ve pkl + corpus.pth + vocabulary_Gref.txt
#                              (fast path: bo qua doan doc moi mask PNG)
#
# Token doc tu file .env (bien HF_TOKEN), tim o Baseline/.env va ../.env
# ============================================================
from __future__ import print_function
import os
import sys
import subprocess
from pathlib import Path

# --- Cau hinh co dinh ---
HF_REPO_ID = "blue7012/refer-youtube-clean"
HF_REPO_TYPE = "dataset"

# Thu muc data goc (tinh tu Baseline). DATA_ROOT cua trainer.py = ./data
DATA_DIR = "data"

# Cac file cache se duoc tai ve o fast path (khop theo duong dan tuong doi trong repo)
FAST_PATH_PATTERNS = ["**/*.pkl", "**/corpus.pth", "**/vocabulary_Gref.txt"]


def log(msg):
    print("[PREPARE] {}".format(msg), flush=True)


def ensure_cwd_baseline():
    """trainer.py dung cac import tuong doi ('utils/', './data', ...) nen BAT BUOC
    cwd phai la thu muc Baseline (= thu muc chua file nay)."""
    here = Path(__file__).resolve().parent
    os.chdir(here)
    log("cwd = {}".format(here))


def ensure_deps():
    """Cai huggingface_hub neu thieu. python-dotenv la tuy chon (co fallback)."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log("Khong thay huggingface_hub -> dang cai...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
        )


def load_env_token():
    """Doc HF_TOKEN tu .env (Baseline/.env hoac ../.env) hoac tu bien moi truong."""
    # Uu tien python-dotenv neu co
    for env_path in [Path(".env"), Path("..") / ".env"]:
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                log("Da load .env tu {}".format(env_path.resolve()))
            except ImportError:
                # Fallback: tu parse KEY=VALUE
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                log("Da parse .env (thu cong) tu {}".format(env_path.resolve()))
            break

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "Khong tim thay HF_TOKEN. Hay tao file .env (Baseline/.env hoac repo-root/.env) "
            "voi noi dung:  HF_TOKEN=hf_xxx..."
        )
    return token


def build_pkl_via_trainer():
    """Mimic pretrain.py: dung Trainer + set_dataset de tac dong phu tao pkl.
    KHONG goi trainer.cuda() va KHONG goi train() — chi can build cache.
    KHONG sua bat ky file goc nao."""
    from types import SimpleNamespace
    from trainer import Trainer  # import o day de fast-path khong phai load torch model

    # args giong het get_arguments() trong pretrain.py (mac dinh) + chon arch/dataset/splits
    args = SimpleNamespace(
        arch="base_model",
        desc="",
        eval=False,
        eval_first=False,
        init_lr=1e-4,
        batch_size=16,
        test_batch_size=0,
        img_size=320,
        max_epoch=150,
        decay_epochs=[],
        optimizer="adam",
        lr_decay=0.1,
        save_every=0,
        max_N=0,
        max_skip=2,
        dataset="refer-yv-2019",
        test_dataset=None,
        splits=["valid"],
        checkpoint="",
        epoch=-1,
    )

    log("Build pkl: dung Trainer (mimic pretrain.py), BO QUA .cuda()...")
    trainer = Trainer(args)
    # Day la buoc trigger set_meta_file() -> build mymeta.pkl (train) + mymeta_eval.pkl (valid)
    trainer.set_dataset(args.dataset, args.splits, args.test_dataset)
    log("Build pkl DONE.")


def upload_full_dataset(api, token):
    """Upload toan bo thu muc data/ (dataset + pkl + vocab + corpus)."""
    if not Path(DATA_DIR).is_dir():
        raise SystemExit(
            "Khong thay thu muc '{}/'. Can co data tho truoc khi build/upload.".format(DATA_DIR)
        )
    log("Tao repo (neu chua co)...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token, exist_ok=True)

    log("Upload toan bo '{}/' len {} (co the lau, day la one-time)...".format(DATA_DIR, HF_REPO_ID))
    # upload_large_folder toi uu cho folder nang; fallback upload_folder neu ban hf_hub cu
    try:
        api.upload_large_folder(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            folder_path=DATA_DIR,
        )
    except AttributeError:
        log("hf_hub cu khong co upload_large_folder -> dung upload_folder...")
        api.upload_folder(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            folder_path=DATA_DIR,
            token=token,
            commit_message="Upload Refer-Youtube-VOS dataset + cache pkl",
        )
    log("Upload XONG: https://huggingface.co/datasets/{}".format(HF_REPO_ID))


def download_cache_only(token):
    """Fast path: chi keo pkl + corpus.pth + vocabulary_Gref.txt ve thu muc data/."""
    from huggingface_hub import snapshot_download

    log("Repo da ton tai -> chi tai cache ({}) ve '{}/'...".format(
        ", ".join(FAST_PATH_PATTERNS), DATA_DIR))
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        allow_patterns=FAST_PATH_PATTERNS,
        local_dir=DATA_DIR,
        token=token,
    )
    log("Tai cache XONG. Train sau nay se load pkl tuc thi (bo qua doc PNG).")


def main():
    ensure_cwd_baseline()
    ensure_deps()
    token = load_env_token()

    from huggingface_hub import HfApi
    api = HfApi()

    log("Kiem tra repo {} (type={})...".format(HF_REPO_ID, HF_REPO_TYPE))
    exists = api.repo_exists(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token)

    if exists:
        download_cache_only(token)
    else:
        log("Repo CHUA ton tai -> build pkl roi upload toan bo dataset.")
        build_pkl_via_trainer()
        upload_full_dataset(api, token)

    log("HOAN TAT.")


if __name__ == "__main__":
    main()
