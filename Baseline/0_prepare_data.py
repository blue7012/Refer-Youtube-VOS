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
#   - Repo HF DA ton tai     -> tai data.zip (FULL: anh + mask + meta_expressions.json
#                              + pkl + corpus + vocab) roi giai nen vao 'data/'.
#                              trainer.py / REFER_YV_2019 CAN meta_expressions.json va
#                              anh, nen fast-path cache-only khong du.
#                              Dat PREPARE_CACHE_ONLY=1 de chi keo cache pkl (khi anh +
#                              meta da co san tren may).
#
# Token doc tu file .env (bien HF_TOKEN), tim o Baseline/.env va ../.env
# ============================================================
from __future__ import print_function
import os
import sys
import logging
import subprocess
from pathlib import Path

# --- Tat log spam tu httpx (moi preupload POST in 1 dong INFO) ---
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for _noisy in ("httpx", "httpcore", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# --- Cau hinh co dinh ---
HF_REPO_ID = "blue7012/refer-youtube-clean"
HF_REPO_TYPE = "dataset"

# Thu muc data goc (tinh tu Baseline). DATA_ROOT cua trainer.py = ./data
DATA_DIR = "data"

# Ten file zip chua toan bo data/ tren repo (upload 1 file -> tranh rate limit 429)
DATA_ZIP_NAME = "data.zip"

# Cac file cache nho upload RIENG LE (ngoai zip) de fast-path keo nhanh,
# khong phai tai nguyen data.zip nang. Glob de quy tuong doi tinh tu DATA_DIR.
CACHE_GLOBS = ["**/*.pkl", "**/corpus.pth", "**/vocabulary_Gref.txt"]

# Fast-path chi keo cac file cache nho (dung y ban dau: "chi clone file pkl").
# Anh tho gia su da co san tren may (hoac giai nen tu data.zip neu can).
FAST_PATH_PATTERNS = CACHE_GLOBS


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

    # QUAN TRONG: splits o day PHAI khop y het --splits luc train. Voi moi split,
    # set_dataset build {split}/mymeta_eval.pkl; neu thieu, fast-path se khong co
    # data tho de build bu -> FileNotFoundError meta_expressions.json.
    # Override duoc qua bien moi truong PREPARE_SPLITS (vd: "train valid").
    # Mac dinh "train" vi lenh train that dung --splits train.
    splits = os.environ.get("PREPARE_SPLITS", "train").split()
    log("Build pkl cho splits = {} (set PREPARE_SPLITS de doi).".format(splits))

    # args giong het get_arguments() trong pretrain.py (mac dinh) + chon arch/dataset/splits
    # LUU Y: Trainer.__init__ doc args.no_eval (tu --no-eval) -> BAT BUOC phai co,
    # neu thieu se AttributeError. Build cache khong can eval nen de True.
    args = SimpleNamespace(
        arch="base_model",
        desc="",
        eval=False,
        eval_first=False,
        no_eval=True,
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
        splits=splits,
        checkpoint="",
        epoch=-1,
    )

    log("Build pkl: dung Trainer (mimic pretrain.py), BO QUA .cuda()...")
    trainer = Trainer(args)
    # Day la buoc trigger set_meta_file() -> build train/mymeta.pkl + {split}/mymeta_eval.pkl
    trainer.set_dataset(args.dataset, args.splits, args.test_dataset)
    log("Build pkl DONE.")


def zip_data_dir():
    """Nen toan bo data/ thanh 1 file data.zip (ZIP_STORED — JPEG/PNG da nen san,
    nen lai chi ton CPU ma khong giam dung luong). Tra ve Path cua zip."""
    import zipfile

    zip_path = Path(DATA_ZIP_NAME)
    if zip_path.exists():
        log("Da co {} san -> dung lai (xoa file nay neu muon nen lai).".format(zip_path))
        return zip_path

    data_root = Path(DATA_DIR)
    files = [p for p in data_root.rglob("*") if p.is_file()]
    log("Nen {} file tu '{}/' -> {} (ZIP_STORED)...".format(len(files), DATA_DIR, zip_path))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, p in enumerate(files):
            # arcname giu cau truc "data/..." de giai nen la ra dung cho cu
            zf.write(p, arcname=str(p))
            if (i + 1) % 2000 == 0:
                log("  ...da nen {}/{} file".format(i + 1, len(files)))
    log("Nen XONG: {} ({:.1f} MB)".format(zip_path, zip_path.stat().st_size / 1e6))
    return zip_path


def upload_full_dataset(api, token):
    """Upload data dang 1 file zip (tranh rate-limit 429 do upload tung anh) +
    cac file cache nho upload rieng le de fast-path keo nhanh."""
    if not Path(DATA_DIR).is_dir():
        raise SystemExit(
            "Khong thay thu muc '{}/'. Can co data tho truoc khi build/upload.".format(DATA_DIR)
        )
    log("Tao repo (neu chua co)...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token, exist_ok=True)

    # 1) Nen + upload data.zip (1 request lon, khong dung gioi han so request)
    zip_path = zip_data_dir()
    log("Upload {} len {} (one-time, co the lau)...".format(DATA_ZIP_NAME, HF_REPO_ID))
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=DATA_ZIP_NAME,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token=token,
        commit_message="Upload dataset as single zip",
    )

    # 2) Upload rieng cac file cache nho de fast-path khong phai tai nguyen zip
    cache_files = []
    for pat in CACHE_GLOBS:
        cache_files.extend(sorted(Path(DATA_DIR).glob(pat)))
    log("Upload {} file cache rieng le (pkl/corpus/vocab)...".format(len(cache_files)))
    for p in cache_files:
        # path_in_repo TUONG DOI voi DATA_DIR (khong co tien to "data/") de fast-path
        # snapshot_download(local_dir="data") tra ve dung "data/youtube-vos-2019/...".
        rel = p.relative_to(DATA_DIR).as_posix()
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=rel,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=token,
            commit_message="Upload cache {}".format(p.name),
        )
        log("  + {}".format(rel))
    log("Upload XONG: https://huggingface.co/datasets/{}".format(HF_REPO_ID))


def download_cache_only(token):
    """Fast path (chi cache): keo pkl + corpus.pth + vocabulary_Gref.txt ve 'data/'.
    LUU Y: KHONG du de train/eval — REFER_YV_2019 con can meta_expressions.json va
    anh JPEG/mask PNG. Chi dung khi anh+meta DA co san tren may. Mac dinh dung
    download_and_extract_data() de tai du. Bat lai bang PREPARE_CACHE_ONLY=1."""
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


def download_and_extract_data(token):
    """Tai data.zip (FULL dataset: anh + mask + meta_expressions.json + pkl + corpus
    + vocab) roi giai nen vao thu muc Baseline/ -> tao 'data/youtube-vos-2019/...'.

    Day la path mac dinh vi trainer.py / REFER_YV_2019 can meta_expressions.json (va
    anh) chu KHONG chi pkl; fast-path cache-only se bao thieu meta_expressions.json."""
    import zipfile
    from huggingface_hub import hf_hub_download

    data_root = Path(DATA_DIR)
    # Sentinel: neu meta_expressions.json (split train) da co thi coi nhu da giai nen.
    sentinel = data_root / "youtube-vos-2019" / "train" / "meta_expressions.json"
    if sentinel.exists():
        log("Data da co san ({}) -> bo qua tai/giai nen.".format(sentinel))
        return

    zip_path = Path(DATA_ZIP_NAME)
    if zip_path.exists():
        log("Da co {} san ({:.1f} GB) -> dung lai (xoa file neu muon tai lai).".format(
            zip_path, zip_path.stat().st_size / 1e9))
    else:
        log("Tai {} tu {} (one-time, ~9.7GB, co the lau)...".format(
            DATA_ZIP_NAME, HF_REPO_ID))
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=DATA_ZIP_NAME,
            local_dir=".",
            token=token,
        )
        zip_path = Path(local)

    # Entry trong zip co dang "data/youtube-vos-2019/..." (arcname=str(p) tinh tu
    # Baseline), nen giai nen tai cwd=Baseline la ra dung "data/...".
    log("Giai nen {} -> {}/ ...".format(zip_path, Path(".").resolve()))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(".")
    if not sentinel.exists():
        raise SystemExit(
            "Giai nen xong nhung khong thay {} — kiem tra lai cau truc data.zip.".format(
                sentinel))
    log("Giai nen XONG. Data day du san sang o '{}/'.".format(DATA_DIR))


def main():
    ensure_cwd_baseline()
    ensure_deps()
    token = load_env_token()

    from huggingface_hub import HfApi
    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_warning()
    except Exception:
        pass
    api = HfApi()

    log("Kiem tra repo {} (type={})...".format(HF_REPO_ID, HF_REPO_TYPE))
    exists = api.repo_exists(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token)

    if exists:
        # Mac dinh tai + giai nen FULL data.zip (co meta_expressions.json + anh) de
        # khop voi trainer.py. Dat PREPARE_CACHE_ONLY=1 neu chi muon keo cache pkl
        # (khi anh + meta da co san tren may).
        if os.environ.get("PREPARE_CACHE_ONLY") == "1":
            download_cache_only(token)
        else:
            download_and_extract_data(token)
    else:
        log("Repo CHUA ton tai -> build pkl roi upload toan bo dataset.")
        build_pkl_via_trainer()
        upload_full_dataset(api, token)

    log("HOAN TAT.")


if __name__ == "__main__":
    main()
