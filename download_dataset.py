#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download RAF-DB dataset from Google Drive.

Google Drive URL
----------------
https://drive.google.com/drive/folders/0B4E10azXECctRUgwVmFPblFUdUE
  resourcekey: 0-SCrrCMK2lc4IDmhDsDKhRw

Usage
-----
    python download_dataset.py                        # downloads to ./dataset/
    python download_dataset.py --output ./data        # custom output dir

After this script finishes it prints the RAF-DB root path and writes it to
.dataset_path so that run_training.sh picks it up automatically.

If gdown fails (quota exceeded, auth required, network issue) the script
prints step-by-step manual-download instructions and exits non-zero.
"""

import argparse
import os
import subprocess
import sys
import time

# ── Google Drive coordinates ──────────────────────────────────────────────────
FOLDER_ID    = "0B4E10azXECctRUgwVmFPblFUdUE"
RESOURCE_KEY = "0-SCrrCMK2lc4IDmhDsDKhRw"
GDRIVE_URL   = (
    f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    f"?resourcekey={RESOURCE_KEY}"
)
# path where the detected RAF-DB root is persisted for other scripts
DATASET_PATH_FILE = ".dataset_path"


def log(msg=""):
    print(msg, flush=True)


def ensure_gdown():
    """Install / upgrade gdown to >= 4.6.0 if not already present."""
    try:
        import gdown
        from packaging.version import Version
        if Version(gdown.__version__) >= Version("4.6.0"):
            return gdown
        raise ImportError("need upgrade")
    except (ImportError, Exception):
        log("Installing gdown >= 4.6.0 …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "gdown>=4.6.0"]
        )
        import gdown  # noqa: F811
        return gdown


def find_raf_root(search_dir):
    """Walk search_dir and return the first folder containing
    EmoLabel/, Annotation/, Image/ as direct children."""
    required = {"EmoLabel", "Annotation", "Image"}
    for root, dirs, _files in os.walk(search_dir):
        if required <= set(dirs):
            return root
    return None


def print_manual_instructions(output_dir):
    log()
    log("=" * 60)
    log("  MANUAL DOWNLOAD INSTRUCTIONS")
    log("=" * 60)
    log()
    log("Automated download failed. Download the dataset manually:")
    log()
    log("  Option A — via your browser:")
    log(f"    1. Open: {GDRIVE_URL}")
    log(f"    2. Click the folder → Download → Download as ZIP")
    log(f"    3. Upload the ZIP to your RunPod pod:")
    log(f"       scp ~/Downloads/<zip> root@<POD_IP>:{output_dir}/")
    log(f"    4. Unzip: cd {output_dir} && unzip <zip>")
    log()
    log("  Option B — gdown on RunPod (fresh IP has no rate-limit):")
    log(f"    pip install gdown")
    log(f"    gdown --folder \"{GDRIVE_URL}\" -O {output_dir}/")
    log()
    log("  Option C — rclone (if you have a Google account):")
    log(f"    rclone copy 'gdrive:RAF-DB' {output_dir}/ --progress")
    log()
    log("After placing the dataset, re-run:")
    log("    python download_dataset.py --output", output_dir, "(auto-detect only)")
    log()


def attempt_download(output_dir):
    """Try gdown folder download. Returns (success, gdown_error_msg)."""
    gdown = ensure_gdown()
    os.makedirs(output_dir, exist_ok=True)

    log(f"Downloading RAF-DB from Google Drive → {output_dir}")
    log(f"Size: ~1.8 GB  (this will take a few minutes)")
    log()

    # gdown.download_folder attempts to list & download all files in the folder.
    # use_cookies=False avoids stale auth from a previous session.
    try:
        result = gdown.download_folder(
            url=GDRIVE_URL,
            output=output_dir,
            quiet=False,
            use_cookies=False,
        )
        if result is None:
            return False, "gdown returned None (possible auth/quota error)"
        return True, None
    except Exception as exc:
        return False, str(exc)


def detect_or_download(output_dir):
    """
    1. If RAF root already exists under output_dir → return it (cached).
    2. Otherwise attempt gdown download, then re-detect.
    """
    # --- check if already downloaded ---
    existing = find_raf_root(output_dir)
    if existing:
        log(f"Dataset already present at: {existing}")
        return existing

    # --- also check beside the script (user may have uploaded manually) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    existing = find_raf_root(script_dir)
    if existing:
        log(f"Dataset found at: {existing}")
        return existing

    # --- download ---
    ok, err = attempt_download(output_dir)

    if not ok:
        quota_hint = (
            "quota" in (err or "").lower()
            or "rate" in (err or "").lower()
            or "too many" in (err or "").lower()
        )
        log()
        if quota_hint:
            log("Google Drive daily download quota exceeded.")
            log("Wait 24 hours or use a different Google account / IP.")
        else:
            log(f"gdown error: {err}")
        print_manual_instructions(output_dir)
        return None

    # --- find RAF root after download ---
    time.sleep(1)  # let filesystem sync
    raf_root = find_raf_root(output_dir)
    if raf_root is None:
        log()
        log("Download finished but RAF-DB folder structure not found.")
        log(f"Contents of {output_dir}:")
        for root, dirs, files in os.walk(output_dir):
            depth = root.replace(output_dir, "").count(os.sep)
            if depth > 3:
                dirs[:] = []
                continue
            indent = "  " * depth
            log(f"{indent}{os.path.basename(root)}/")
        print_manual_instructions(output_dir)
        return None

    return raf_root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", "--output-dir",
        default="./dataset",
        metavar="DIR",
        help="Directory to download the dataset into (default: ./dataset)",
    )
    ap.add_argument(
        "--detect-only",
        action="store_true",
        help="Skip download, just detect an already-present RAF-DB root and save path",
    )
    args = ap.parse_args()

    output_dir = os.path.abspath(args.output)

    if args.detect_only:
        raf_root = find_raf_root(output_dir)
        if raf_root is None:
            log(f"RAF-DB root not found under {output_dir}")
            sys.exit(1)
    else:
        raf_root = detect_or_download(output_dir)

    if raf_root is None:
        sys.exit(1)

    log()
    log("=" * 60)
    log(f"  RAF-DB root : {raf_root}")
    log("=" * 60)

    # Persist for run_training.sh
    with open(DATASET_PATH_FILE, "w") as f:
        f.write(raf_root.strip())
    log(f"Path saved to {DATASET_PATH_FILE}")

    log()
    log("Next step:")
    log(f"  bash run_training.sh")
    log("  (or: python convert_raf_to_yolo.py --dataset_root '{raf_root}')")
    return raf_root


if __name__ == "__main__":
    main()
