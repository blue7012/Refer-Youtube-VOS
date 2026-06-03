import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

# ============================================================
# CELL 1 — Set your zip file paths here
# ============================================================
TRAIN_ZIP = '/content/drive/MyDrive/train_data/train.zip'
VALID_ZIP = '/content/drive/MyDrive/valid_data/valid.zip'
META_ZIP  = '/content/drive/MyDrive/valid_data/meta_expressions.zip'
VOCAB_OUT = '/content/vocabulary_Gref.txt'   # where generated vocab will be saved


# ============================================================
# CELL 2 — General zip structure checker
# ============================================================

PASS = '✅'
FAIL = '❌'
WARN = '⚠️ '
INFO = 'ℹ️ '


def section(title):
    print(f'\n{"="*60}\n  {title}\n{"="*60}')


def check_zip_exists(path, label):
    if path and os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f'{PASS} Found {label}: {path}  ({size_mb:.1f} MB)')
        return True
    print(f'{FAIL} Not found: {label}  →  {path}')
    return False


def preview_structure(zf, max_show=15):
    names = zf.namelist()
    seen = set()
    for n in names:
        parts = Path(n).parts
        seen.add('/'.join(parts[:2]) if len(parts) >= 2 else parts[0])
    for i, k in enumerate(sorted(seen)):
        if i >= max_show:
            print(f'    ... ({len(seen)-max_show} more)')
            break
        print(f'    {k}/')


def check_meta_json_basic(zf, meta_path):
    try:
        with zf.open(meta_path) as f:
            data = json.load(f)
        if 'videos' not in data:
            print(f'  {FAIL} Missing top-level "videos" key')
            return
        videos = data['videos']
        print(f'  {PASS} "videos" key found — {len(videos)} videos')
        sample_vid = next(iter(videos))
        objs = videos[sample_vid].get('objects', {})
        if not objs:
            print(f'  {FAIL} No "objects" found under video "{sample_vid}"')
            return
        obj = next(iter(objs.values()))
        for key in ['expressions', 'frames', 'category']:
            status = PASS if key in obj else FAIL
            val = obj.get(key, 'MISSING')
            display = val[:2] if key == 'expressions' and isinstance(val, list) else val
            print(f'  {status} objects[id].{key}: {display}')
    except Exception as e:
        print(f'  {WARN} Could not parse {meta_path}: {e}')


def inspect_zip(zip_path, split_name):
    section(f'{split_name} zip: {zip_path}')
    if not check_zip_exists(zip_path, split_name):
        return

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        print(f'{INFO} Total entries: {len(names):,}')

        top_dirs = set(Path(n).parts[0] for n in names if n.strip('/'))
        prefix = (list(top_dirs)[0] + '/') if len(top_dirs) == 1 else ''
        if prefix:
            print(f'{INFO} Root prefix inside zip: "{prefix}"')

        jpeg_files = [n for n in names if (prefix + 'JPEGImages/') in n and n.endswith('.jpg')]
        jpeg_videos = set(Path(n).parts[-2] for n in jpeg_files if len(Path(n).parts) >= 2)
        if jpeg_files:
            print(f'{PASS} JPEGImages/  — {len(jpeg_files):,} .jpg files, {len(jpeg_videos)} videos')
            for s in sorted(jpeg_files)[:2]:
                print(f'    e.g. {s}')
        else:
            print(f'{FAIL} JPEGImages/ NOT found  (expected: {prefix}JPEGImages/<video_id>/<frame>.jpg)')

        ann_files = [n for n in names if (prefix + 'Annotations/') in n and n.endswith('.png')]
        ann_videos = set(Path(n).parts[-2] for n in ann_files if len(Path(n).parts) >= 2)
        if ann_files:
            print(f'{PASS} Annotations/ — {len(ann_files):,} .png files, {len(ann_videos)} videos')
            for s in sorted(ann_files)[:2]:
                print(f'    e.g. {s}')
        else:
            print(f'{WARN} Annotations/ NOT found — required for train/valid, OK for test-only')

        meta_hits = [n for n in names if 'meta_expressions' in n and n.endswith('.json')]
        if meta_hits:
            print(f'{PASS} meta_expressions.json: {meta_hits[0]}')
            check_meta_json_basic(zf, meta_hits[0])
        else:
            print(f'{WARN} meta_expressions.json not inside this zip (may be separate)')

        if jpeg_files:
            stem = Path(sorted(jpeg_files)[0]).stem
            fmt = PASS if stem.isdigit() else WARN
            label = 'numeric OK' if stem.isdigit() else 'unexpected, code expects numeric like 00000'
            print(f'{fmt} Frame ID format: "{stem}" — {label}')

        if jpeg_videos and ann_videos:
            miss_ann  = jpeg_videos - ann_videos
            miss_jpeg = ann_videos  - jpeg_videos
            if not miss_ann and not miss_jpeg:
                print(f'{PASS} Video IDs match between JPEGImages and Annotations')
            else:
                if miss_ann:
                    print(f'{WARN} {len(miss_ann)} videos in JPEGImages but missing Annotations')
                if miss_jpeg:
                    print(f'{WARN} {len(miss_jpeg)} videos in Annotations but missing JPEGImages')

        print(f'\n{INFO} Top structure preview:')
        preview_structure(zf)


if TRAIN_ZIP:
    inspect_zip(TRAIN_ZIP, 'TRAIN')
if VALID_ZIP:
    inspect_zip(VALID_ZIP, 'VALID')

section('EXPECTED FINAL LAYOUT after extraction')
print("""
  Baseline/data/youtube-vos-2019/
  ├── vocabulary_Gref.txt
  ├── train/
  │   ├── meta_expressions.json
  │   ├── JPEGImages/<video_id>/<frame>.jpg
  │   └── Annotations/<video_id>/<frame>.png
  └── valid/
      ├── meta_expressions.json
      ├── JPEGImages/<video_id>/<frame>.jpg
      └── Annotations/<video_id>/<frame>.png
""")


# ============================================================
# CELL 3 — Deep-check train & valid meta_expressions.json
#           from meta_expressions.zip
# ============================================================

def deep_check_meta(zf, json_path, split_name):
    section(f'Deep check: {split_name}  ({json_path})')
    with zf.open(json_path) as f:
        data = json.load(f)

    if 'videos' not in data:
        print(f'{FAIL} Missing top-level "videos" key — this file is unusable')
        return
    videos = data['videos']
    print(f'{PASS} "videos" found — {len(videos)} videos')

    sample_vid = next(iter(videos))
    vid_val = videos[sample_vid]
    print(f'{INFO} Sample video id: "{sample_vid}"')
    print(f'{INFO} Keys under video: {list(vid_val.keys())}')

    if 'objects' not in vid_val:
        print(f'{FAIL} No "objects" key — code at refer_datasets.py:83 requires it')
        print(f'  Structure found:  videos -> {list(vid_val.keys())}')
        print('  Structure needed: videos -> objects -> {expressions, frames, category}')
        print(f'\n{INFO} Actual content of video "{sample_vid}":')
        print(json.dumps(vid_val, indent=2)[:600])
        return

    print(f'{PASS} "objects" key present')
    objs = vid_val['objects']
    print(f'{INFO} Number of objects in sample video: {len(objs)}')

    sample_obj_id  = next(iter(objs))
    sample_obj_val = objs[sample_obj_id]
    print(f'{INFO} Sample object id: "{sample_obj_id}"  (code casts to int -> oid)')
    print(f'{INFO} Keys under object: {list(sample_obj_val.keys())}')

    for key, required in [('expressions', True), ('frames', True), ('category', False)]:
        if key in sample_obj_val:
            val = sample_obj_val[key]
            preview = val[:2] if isinstance(val, list) else val
            print(f'{PASS} objects[id].{key}: {preview}')
        else:
            status = FAIL if required else WARN
            suffix = '(REQUIRED)' if required else '(optional)'
            print(f'{status} objects[id].{key}: MISSING {suffix}')

    frames = sample_obj_val.get('frames', [])
    if frames:
        stem = frames[0]
        ok = stem.isdigit()
        label = 'numeric OK' if ok else 'unexpected (code expects numeric like 00000)'
        print(f'{PASS if ok else WARN} Frame ID format: "{stem}" — {label}')

    exprs = sample_obj_val.get('expressions', [])
    if exprs:
        words = re.split(r'(\W+)', exprs[0].strip())
        words = [w.lower() for w in words if w.strip()]
        print(f'{INFO} Sample expression: "{exprs[0]}"')
        print(f'{INFO} Tokenized to: {words}')

    total_objs = sum(len(v['objects']) for v in videos.values() if 'objects' in v)
    total_exprs = sum(
        len(obj.get('expressions', []))
        for v in videos.values() if 'objects' in v
        for obj in v['objects'].values()
    )
    print(f'\n{INFO} Total objects: {total_objs}')
    print(f'{INFO} Total expressions: {total_exprs}')
    if 'objects' in vid_val:
        print(f'{PASS} This meta_expressions.json is COMPATIBLE with the code')


section('Deep-checking meta_expressions.zip')
with zipfile.ZipFile(META_ZIP, 'r') as zf:
    all_names = zf.namelist()
    for split in ['train', 'valid']:
        candidates = [
            n for n in all_names
            if f'/{split}/' in n and n.endswith('.json') and 'MACOSX' not in n
        ]
        if candidates:
            deep_check_meta(zf, candidates[0], split.upper())
        else:
            print(f'{FAIL} No JSON found for split "{split}" inside {META_ZIP}')


# ============================================================
# CELL 4 — Convert meta_expressions.json to the format the
#           code expects, then generate vocabulary_Gref.txt
#
# Input  (what the zip has):
#   videos -> expressions[id]{exp, obj_id} + frames[]
#
# Output (what the code needs):
#   videos -> objects[obj_id]{expressions[], frames, category}
#
# Also handles valid: renames meta_expressions_challenge.json
# (already correct format) to meta_expressions.json.
# ============================================================

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

# Output paths — adjust if your extraction target is different
EXTRACT_ROOT  = '/content/data/youtube-vos-2019'
TRAIN_META_OUT = f'{EXTRACT_ROOT}/train/meta_expressions.json'
VALID_META_OUT = f'{EXTRACT_ROOT}/valid/meta_expressions.json'


def tokenize_expr(line):
    words = SENTENCE_SPLIT_REGEX.split(line.strip())
    words = [w.lower() for w in words if w.strip()]
    if words and words[-1] == '.':
        words = words[:-1]
    return words


def convert_meta_train(raw):
    """
    raw format:  videos[vid]{expressions{id: {exp, obj_id}}, frames[]}
    out format:  videos[vid]{objects{obj_id: {expressions[], frames, category}}}
    """
    out = {'videos': {}}
    for vid, vid_val in raw['videos'].items():
        frames = vid_val.get('frames', [])
        exprs_raw = vid_val.get('expressions', {})

        # group expressions by obj_id
        objects = {}
        for expr_entry in exprs_raw.values():
            obj_id = str(expr_entry.get('obj_id', '1'))
            text   = expr_entry.get('exp', '')
            if obj_id not in objects:
                objects[obj_id] = {'expressions': [], 'frames': frames, 'category': ''}
            objects[obj_id]['expressions'].append(text)

        out['videos'][vid] = {'objects': objects}
    return out


section('Converting meta_expressions.json files')

os.makedirs(f'{EXTRACT_ROOT}/train', exist_ok=True)
os.makedirs(f'{EXTRACT_ROOT}/valid', exist_ok=True)

# --- TRAIN: convert from zip ---
with zipfile.ZipFile(META_ZIP, 'r') as zf:
    train_jsons = [
        n for n in zf.namelist()
        if '/train/' in n and n.endswith('.json') and 'MACOSX' not in n
    ]
    valid_jsons = [
        n for n in zf.namelist()
        if '/valid/' in n and n.endswith('.json') and 'MACOSX' not in n
    ]

    if train_jsons:
        with zf.open(train_jsons[0]) as f:
            raw_train = json.load(f)
        converted_train = convert_meta_train(raw_train)
        with open(TRAIN_META_OUT, 'w', encoding='utf-8') as f:
            json.dump(converted_train, f)
        num_vids = len(converted_train['videos'])
        num_objs = sum(len(v['objects']) for v in converted_train['videos'].values())
        print(f'{PASS} Train meta_expressions.json written — {num_vids} videos, {num_objs} objects')
        print(f'  Saved to: {TRAIN_META_OUT}')
    else:
        print(f'{FAIL} Train meta_expressions.json not found in {META_ZIP}')

    # --- VALID: the challenge JSON inside valid.zip is already correct format ---
    # Extract meta_expressions_challenge.json from VALID_ZIP and rename it
    with zipfile.ZipFile(VALID_ZIP, 'r') as vzf:
        challenge_hits = [
            n for n in vzf.namelist()
            if 'meta_expressions_challenge' in n and n.endswith('.json')
        ]
        if challenge_hits:
            with vzf.open(challenge_hits[0]) as f:
                valid_meta = json.load(f)
            with open(VALID_META_OUT, 'w', encoding='utf-8') as f:
                json.dump(valid_meta, f)
            num_vids = len(valid_meta['videos'])
            print(f'{PASS} Valid meta_expressions.json written — {num_vids} videos')
            print(f'  Source: {challenge_hits[0]}')
            print(f'  Saved to: {VALID_META_OUT}')
        else:
            print(f'{WARN} meta_expressions_challenge.json not found in valid.zip')
            print('  Falling back to meta_expressions/valid/meta_expressions.json from META_ZIP')
            if valid_jsons:
                with zf.open(valid_jsons[0]) as f:
                    raw_valid = json.load(f)
                converted_valid = convert_meta_train(raw_valid)
                with open(VALID_META_OUT, 'w', encoding='utf-8') as f:
                    json.dump(converted_valid, f)
                print(f'{PASS} Valid meta_expressions.json converted and written')
            else:
                print(f'{FAIL} No valid meta_expressions.json found anywhere')


# --- Build vocabulary from converted train data ---
section('Generating vocabulary_Gref.txt')

vocab = set()
for vid_val in converted_train['videos'].values():
    for obj in vid_val.get('objects', {}).values():
        for expr in obj.get('expressions', []):
            vocab.update(tokenize_expr(expr))

if vocab:
    vocab_sorted = sorted(vocab)
    with open(VOCAB_OUT, 'w', encoding='utf-8') as f:
        for word in vocab_sorted:
            f.write(word + '\n')
    print(f'{PASS} vocabulary_Gref.txt generated — {len(vocab_sorted)} unique words')
    print(f'  Saved to: {VOCAB_OUT}')
    print(f'  Sample words: {vocab_sorted[:10]}')
else:
    print(f'{FAIL} No expressions found — vocabulary_Gref.txt NOT generated')


# ============================================================
# CELL 5 — Summary of what still needs manual attention
# ============================================================
section('REMAINING ISSUES TO KNOW')
print(f"""
{WARN} Valid Annotations path mismatch:
  Your valid.zip has: Annotations/<vid>/<obj_id>/<frame>.png  (only 10 videos)
  Code expects:       Annotations/<vid>/<frame>.png

  These are binary masks per-object, not palette PNGs.
  Options:
    A) For training only  → ignore valid Annotations, use train split only.
    B) For full eval      → a code patch is needed in refer_datasets.py load_pair().

{INFO} train/meta.json inside train.zip is NOT used — replaced by converted file above.

{INFO} Final files needed at {EXTRACT_ROOT}/:
    vocabulary_Gref.txt       <- generated above at {VOCAB_OUT}
    train/meta_expressions.json  <- converted above
    valid/meta_expressions.json  <- copied from challenge JSON above
    train/JPEGImages/...         <- extract train.zip
    train/Annotations/...        <- extract train.zip
    valid/JPEGImages/...         <- extract valid.zip
""")


# ============================================================
# CELL 6 — Clone repo and set up folder structure
# ============================================================

REPO_URL  = 'https://github.com/skynbe/Refer-Youtube-VOS.git'
REPO_ROOT = '/content/Refer-Youtube-VOS'
DATA_ROOT = f'{REPO_ROOT}/Baseline/data/youtube-vos-2019'

# Clone (skip if already cloned)
if not os.path.exists(REPO_ROOT):
    subprocess.run(['git', 'clone', REPO_URL, REPO_ROOT], check=True)
    print(f'{PASS} Repo cloned to {REPO_ROOT}')
else:
    print(f'{INFO} Repo already exists at {REPO_ROOT}, skipping clone')

# Create dataset directory skeleton
for split in ['train', 'valid']:
    os.makedirs(f'{DATA_ROOT}/{split}', exist_ok=True)
print(f'{PASS} Dataset folder ready: {DATA_ROOT}')


# ============================================================
# CELL 7 — Extract train.zip  (~8.5 GB, takes ~10-15 min)
#           Uses shell unzip for speed; strips root "train/" prefix
# ============================================================

# unzip directly into DATA_ROOT — the zip already has "train/" as root prefix
# so it will land at DATA_ROOT/train/JPEGImages/... and DATA_ROOT/train/Annotations/...
print('Extracting train.zip — this will take a while...')
ret = subprocess.run(
    ['unzip', '-q', '-o', TRAIN_ZIP, '-d', DATA_ROOT],
    capture_output=True, text=True
)
if ret.returncode == 0:
    jpg_count = sum(
        len([f for f in files if f.endswith('.jpg')])
        for _, _, files in os.walk(f'{DATA_ROOT}/train/JPEGImages')
    )
    png_count = sum(
        len([f for f in files if f.endswith('.png')])
        for _, _, files in os.walk(f'{DATA_ROOT}/train/Annotations')
    )
    print(f'{PASS} train.zip extracted')
    print(f'  JPEGImages: {jpg_count:,} .jpg files')
    print(f'  Annotations: {png_count:,} .png files')
else:
    print(f'{FAIL} unzip failed:\n{ret.stderr}')


# ============================================================
# CELL 8 — Extract valid.zip  (~500 MB)
#           Same approach — zip root prefix is "valid/"
# ============================================================

print('Extracting valid.zip...')
ret = subprocess.run(
    ['unzip', '-q', '-o', VALID_ZIP, '-d', DATA_ROOT],
    capture_output=True, text=True
)
if ret.returncode == 0:
    jpg_count = sum(
        len([f for f in files if f.endswith('.jpg')])
        for _, _, files in os.walk(f'{DATA_ROOT}/valid/JPEGImages')
    )
    print(f'{PASS} valid.zip extracted')
    print(f'  JPEGImages: {jpg_count:,} .jpg files')
else:
    print(f'{FAIL} unzip failed:\n{ret.stderr}')


# ============================================================
# CELL 9 — Copy meta JSONs + vocab, then verify final layout
# ============================================================

# Files already generated by Cell 4
files_to_copy = [
    (TRAIN_META_OUT, f'{DATA_ROOT}/train/meta_expressions.json'),
    (VALID_META_OUT, f'{DATA_ROOT}/valid/meta_expressions.json'),
    (VOCAB_OUT,      f'{DATA_ROOT}/vocabulary_Gref.txt'),
]

section('Copying generated files into dataset folder')
for src, dst in files_to_copy:
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f'{PASS} {os.path.basename(dst)}  ({size_kb:.1f} KB)  ->  {dst}')
    else:
        print(f'{FAIL} Source not found: {src}')
        print('  Make sure Cell 4 ran successfully first.')

# --- Verify final layout ---
section('Final dataset layout verification')

checks = [
    (f'{DATA_ROOT}/vocabulary_Gref.txt',               'vocabulary_Gref.txt'),
    (f'{DATA_ROOT}/train/meta_expressions.json',        'train/meta_expressions.json'),
    (f'{DATA_ROOT}/valid/meta_expressions.json',        'valid/meta_expressions.json'),
    (f'{DATA_ROOT}/train/JPEGImages',                   'train/JPEGImages/'),
    (f'{DATA_ROOT}/train/Annotations',                  'train/Annotations/'),
    (f'{DATA_ROOT}/valid/JPEGImages',                   'valid/JPEGImages/'),
]

all_ok = True
for path, label in checks:
    exists = os.path.exists(path)
    print(f'{PASS if exists else FAIL} {label}')
    if not exists:
        all_ok = False

if all_ok:
    print(f'\n{PASS} Dataset is ready. You can now run training or inference.')
    print(f'\n  cd {REPO_ROOT}/Baseline')
    print('  python pretrain.py \\')
    print('      --arch base_model \\')
    print('      --dataset refer-yv-2019 \\')
    print('      --splits valid \\')
    print('      --eval --epoch 0')
else:
    print(f'\n{FAIL} Some files are missing. Check the errors above.')


# ============================================================
# CELL 10 — Inspect all JSON files found in train/ and valid/
#            to decide which one to use and whether refer_datasets.py
#            needs patching at all
# ============================================================

DATA_ROOT = '/content/Refer-Youtube-VOS/Baseline/data/youtube-vos-2019'

# Code expects this exact structure:
#   data['videos'][vid]['objects'][obj_id]['expressions']  -> list of strings
#   data['videos'][vid]['objects'][obj_id]['frames']       -> list of frame IDs
#   data['videos'][vid]['objects'][obj_id]['category']     -> string  (unused but read)

def probe_json(path):
    """Returns a dict describing the structure of a meta JSON file."""
    result = {
        'path': path,
        'exists': os.path.isfile(path),
        'has_videos': False,
        'num_videos': 0,
        'top_video_keys': [],
        'has_objects': False,
        'has_expressions_list': False,
        'has_frames': False,
        'has_category': False,
        'expressions_format': None,   # 'list' | 'dict_with_exp' | 'unknown'
        'compatible': False,
        'sample_preview': '',
    }
    if not result['exists']:
        return result

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if 'videos' not in data:
        return result

    result['has_videos'] = True
    videos = data['videos']
    result['num_videos'] = len(videos)

    sample_vid = next(iter(videos))
    vid_val = videos[sample_vid]
    result['top_video_keys'] = list(vid_val.keys())

    if 'objects' in vid_val:
        result['has_objects'] = True
        objs = vid_val['objects']
        sample_obj = next(iter(objs.values()))
        result['has_frames'] = 'frames' in sample_obj
        result['has_category'] = 'category' in sample_obj

        exprs = sample_obj.get('expressions', None)
        if isinstance(exprs, list):
            result['has_expressions_list'] = True
            result['expressions_format'] = 'list'
        elif isinstance(exprs, dict):
            result['expressions_format'] = 'dict'
        else:
            result['expressions_format'] = 'missing'

        result['compatible'] = (
            result['has_objects'] and
            result['has_frames'] and
            result['expressions_format'] == 'list'
        )
    else:
        # No 'objects' — check alternate structure
        exprs = vid_val.get('expressions', None)
        if isinstance(exprs, dict):
            sample_expr = next(iter(exprs.values()))
            if 'exp' in sample_expr:
                result['expressions_format'] = 'dict_with_exp'   # needs conversion
            else:
                result['expressions_format'] = 'dict_unknown'
        result['compatible'] = False

    # compact preview
    import io
    buf = io.StringIO()
    buf.write(f'videos["{sample_vid}"] = ')
    buf.write(json.dumps(vid_val, indent=2)[:400])
    result['sample_preview'] = buf.getvalue()

    return result


def print_probe(r):
    print(f'\n  File : {r["path"]}')
    print(f'  Exists        : {PASS if r["exists"] else FAIL}')
    if not r['exists']:
        return
    print(f'  has "videos"  : {PASS if r["has_videos"] else FAIL}  ({r["num_videos"]} videos)')
    print(f'  has "objects" : {PASS if r["has_objects"] else FAIL}')
    print(f'  has "frames"  : {PASS if r["has_frames"] else FAIL}')
    print(f'  has "category": {PASS if r["has_category"] else FAIL}')
    print(f'  expressions   : {r["expressions_format"]}')
    print(f'  COMPATIBLE    : {PASS + " YES — use as-is" if r["compatible"] else FAIL + " NO  — needs conversion or patch"}')
    print('\n  Preview:')
    for line in r['sample_preview'].splitlines()[:12]:
        print(f'    {line}')


section('Probing all JSON files')

candidates = {
    'train/meta.json':                   f'{DATA_ROOT}/train/meta.json',
    'train/meta_expressions.json':       f'{DATA_ROOT}/train/meta_expressions.json',
    'valid/meta_expressions.json':       f'{DATA_ROOT}/valid/meta_expressions.json',
    'valid/meta_expressions_challenge.json': f'{DATA_ROOT}/valid/meta_expressions_challenge.json',
}

probes = {}
for label, path in candidates.items():
    probes[label] = probe_json(path)
    section(f'  {label}')
    print_probe(probes[label])

# --- Recommendation ---
section('RECOMMENDATION')

for split in ['train', 'valid']:
    split_files = {k: v for k, v in probes.items() if k.startswith(split)}
    compatible = [k for k, v in split_files.items() if v['compatible']]
    needs_conv = [k for k, v in split_files.items() if v['exists'] and not v['compatible']]

    if compatible:
        print(f'{PASS} {split.upper()}: use  "{compatible[0]}"  as  meta_expressions.json  (no changes needed)')
        if len(compatible) > 1:
            print(f'  (also compatible: {compatible[1:]})')
    elif needs_conv:
        print(f'{WARN} {split.upper()}: no compatible file found — "{needs_conv[0]}" needs conversion (run Cell 4)')
    else:
        print(f'{FAIL} {split.upper()}: no JSON files found at all')

# --- Check if refer_datasets.py patch is needed ---
section('Do we need to patch refer_datasets.py?')

train_ok  = any(v['compatible'] for v in probes.values() if 'train' in v['path'])
valid_ok  = any(v['compatible'] for v in probes.values() if 'valid' in v['path'])

ann_train = os.path.isdir(f'{DATA_ROOT}/train/Annotations')
ann_valid = os.path.isdir(f'{DATA_ROOT}/valid/Annotations')

# Check if valid annotations use the subfolder format
valid_sub_format = False
if ann_valid:
    for vid in os.listdir(f'{DATA_ROOT}/valid/Annotations'):
        vid_path = f'{DATA_ROOT}/valid/Annotations/{vid}'
        if os.path.isdir(vid_path):
            for item in os.listdir(vid_path):
                if os.path.isdir(f'{vid_path}/{item}'):
                    valid_sub_format = True
                    break
        if valid_sub_format:
            break

print(f'\n  Meta JSON structure   (train): {"compatible, no patch needed" if train_ok else "needs conversion"}')
print(f'  Meta JSON structure   (valid): {"compatible, no patch needed" if valid_ok else "needs conversion"}')
print(f'  Valid Annotations format     : {"<vid>/<obj_id>/<frame>.png  -> PATCH NEEDED" if valid_sub_format else "<vid>/<frame>.png  -> no patch needed"}')

if not valid_sub_format and train_ok and valid_ok:
    print(f'\n{PASS} refer_datasets.py does NOT need patching — original code works as-is')
else:
    print(f'\n{WARN} refer_datasets.py NEEDS patching for:')
    if valid_sub_format:
        print('   - Valid Annotations subfolder path format')
    if not train_ok:
        print('   - Train meta_expressions.json structure (run Cell 4 conversion first)')
    if not valid_ok:
        print('   - Valid meta_expressions.json structure (run Cell 4 conversion first)')


# ============================================================
# CELL 11 — Fix train/meta_expressions.json
#
# Root cause: Cell 4's conversion used obj_id from
# meta_expressions/train/meta_expressions.json as the
# annotation lookup key. That obj_id is a GLOBAL sequential
# ID, NOT the per-video palette pixel value in the PNG.
# Result: mask == oid is always zero → all ankers = 0 → crash.
#
# Fix: use meta.json (correct per-video local obj IDs matching
# PNG pixel values + per-object frames + category) as the base,
# and pull text expressions from meta_expressions/train/
# meta_expressions.json by matching obj_id.
# ============================================================

DATA_ROOT      = '/content/Refer-Youtube-VOS/Baseline/data/youtube-vos-2019'
META_JSON      = f'{DATA_ROOT}/train/meta.json'
META_EXPRS_ZIP = META_ZIP   # reuse from Cell 1
TRAIN_META_OUT = f'{DATA_ROOT}/train/meta_expressions.json'


def build_train_meta(meta_path, meta_expressions_zip, zip_train_json_name):
    """
    Combine:
      meta.json          → correct local obj IDs, per-object frames, category
      meta_expressions   → text expressions (obj_id maps to local ID)
    """
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    with zipfile.ZipFile(meta_expressions_zip, 'r') as zf:
        with zf.open(zip_train_json_name) as f:
            exprs_data = json.load(f)

    out = {'videos': {}}
    missing_expr = 0
    total_objs   = 0

    for vid, vid_val in meta['videos'].items():
        meta_objs = vid_val.get('objects', {})

        # build a map: local_obj_id -> [expression strings]
        exprs_map = {}
        if vid in exprs_data['videos']:
            for expr_entry in exprs_data['videos'][vid].get('expressions', {}).values():
                local_id = str(expr_entry.get('obj_id', ''))
                text     = expr_entry.get('exp', '').strip()
                if local_id and text:
                    exprs_map.setdefault(local_id, []).append(text)

        objects = {}
        for local_oid, obj_val in meta_objs.items():
            exprs = exprs_map.get(str(local_oid), [])
            if not exprs:
                missing_expr += 1
                continue   # skip objects with no expressions
            objects[local_oid] = {
                'expressions': exprs,
                'frames':      obj_val.get('frames', []),
                'category':    obj_val.get('category', ''),
            }
            total_objs += 1

        if objects:
            out['videos'][vid] = {'objects': objects}

    return out, total_objs, missing_expr


section('Rebuilding train/meta_expressions.json from meta.json + meta_expressions zip')

# Find train JSON inside the zip
with zipfile.ZipFile(META_EXPRS_ZIP, 'r') as zf:
    train_jsons = [
        n for n in zf.namelist()
        if '/train/' in n and n.endswith('.json') and 'MACOSX' not in n
    ]

if not train_jsons:
    print(f'{FAIL} Could not find train meta_expressions.json inside {META_EXPRS_ZIP}')
elif not os.path.isfile(META_JSON):
    print(f'{FAIL} meta.json not found at {META_JSON}')
    print('  It should have been extracted from train.zip')
else:
    result, total_objs, missing = build_train_meta(META_JSON, META_EXPRS_ZIP, train_jsons[0])

    with open(TRAIN_META_OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f)

    num_vids = len(result['videos'])
    print(f'{PASS} Written to {TRAIN_META_OUT}')
    print(f'  Videos with at least one expression: {num_vids}')
    print(f'  Total objects included:              {total_objs}')
    print(f'  Objects skipped (no expressions):    {missing}')

    # Quick sanity check: open one video and verify oid matches
    sample_vid = next(iter(result['videos']))
    sample_objs = result['videos'][sample_vid]['objects']
    sample_oid  = next(iter(sample_objs))
    sample_obj  = sample_objs[sample_oid]
    sample_frm  = sample_obj['frames'][0] if sample_obj['frames'] else None

    print(f'\n  Sanity check on video "{sample_vid}", obj "{sample_oid}":')
    print(f'    category   : {sample_obj["category"]}')
    print(f'    frames[0]  : {sample_frm}')
    print(f'    expressions: {sample_obj["expressions"][:1]}')

    if sample_frm:
        ann_path = f'{DATA_ROOT}/train/Annotations/{sample_vid}/{sample_frm}.png'
        if os.path.isfile(ann_path):
            mask = np.uint8(PILImage.open(ann_path).convert('P'))
            oid_int = int(sample_oid)
            match = int((mask == oid_int).sum())
            print(f'    annotation : {ann_path}  (exists ✅)')
            print(f'    unique vals: {sorted(set(mask.flatten().tolist()))[:10]}')
            print(f'    mask=={oid_int} pixels: {match}  {"✅ object found" if match > 0 else "❌ NOT FOUND — obj_id still wrong!"}')
        else:
            print(f'    annotation : {ann_path}  ❌ NOT FOUND')
