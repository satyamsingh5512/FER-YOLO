#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust installer for the CUDA extensions this project depends on:
    * causal-conv1d
    * mamba-ssm   (provides the `selective_scan_cuda` extension used by nets/yolo.py)

Why this script exists
----------------------
`pip install mamba-ssm` pulls an *sdist* from PyPI and compiles the CUDA kernels
from source. On Kaggle / RunPod that build takes 20-40 min and frequently fails
(OOM, missing nvcc, ABI mismatch). The maintainers instead publish *prebuilt*
wheels on GitHub Releases, one per (CUDA major, torch minor, C++ ABI, python,
arch) combination.

This script detects the running environment and installs the matching prebuilt
wheel directly, skipping compilation entirely. It only falls back to a source
build (with build isolation disabled) when no compatible wheel is published.

Usage
-----
    python install_mamba.py            # detect + install
    python install_mamba.py --force    # reinstall even if already importable

Environment overrides (rarely needed):
    MAMBA_VERSION   default 2.3.2.post1
    CAUSAL_VERSION  default 1.6.2.post1
"""

import argparse
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request

MAMBA_VERSION = os.environ.get("MAMBA_VERSION", "2.3.2.post1")
CAUSAL_VERSION = os.environ.get("CAUSAL_VERSION", "1.6.2.post1")

MAMBA = dict(
    pkg="mamba_ssm",
    repo="state-spaces/mamba",
    version=MAMBA_VERSION,
    import_name="selective_scan_cuda",
)
CAUSAL = dict(
    pkg="causal_conv1d",
    repo="Dao-AILab/causal-conv1d",
    version=CAUSAL_VERSION,
    import_name="causal_conv1d_cuda",
)

# Published wheel matrix (cuda_major -> {torch_minor -> available ABIs}).
# Mirrors the assets on the GitHub releases for the versions pinned above.
# Used to pick the closest compatible wheel when the exact torch minor is
# not published for the detected environment.
AVAILABLE = {
    "cu11": {"2.6": {"TRUE", "FALSE"}, "2.7": {"TRUE"}},
    "cu12": {
        "2.6": {"TRUE", "FALSE"},
        "2.7": {"TRUE"},
        "2.8": {"TRUE"},
        "2.9": {"TRUE"},
        "2.10": {"TRUE"},
    },
    "cu13": {"2.9": {"TRUE"}, "2.10": {"TRUE"}},
}


def log(msg=""):
    print(msg, flush=True)


def run(cmd):
    log("+ " + " ".join(cmd))
    subprocess.check_call(cmd)


def pip_install(*args):
    run([sys.executable, "-m", "pip", "install", *args])


def pip_install_soft(*args):
    """pip install that warns instead of aborting (for non-critical deps)."""
    try:
        pip_install(*args)
        return True
    except subprocess.CalledProcessError:
        log(f"  WARNING: 'pip install {' '.join(args)}' failed; continuing. "
            f"If imports later fail, install these manually.")
        return False


def is_importable(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def detect_env():
    """Return a dict describing the wheel-relevant parts of the environment."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is always present on Kaggle/RunPod
        raise SystemExit(
            "PyTorch is not installed in this environment. Install a CUDA build of "
            "torch first (Kaggle/RunPod GPU images already include one)."
        ) from exc

    cuda = torch.version.cuda
    if cuda is None:
        raise SystemExit(
            "This torch build has no CUDA support (torch.version.cuda is None).\n"
            "mamba-ssm / causal-conv1d are GPU-only. Use a CUDA-enabled GPU runtime "
            "(Kaggle GPU accelerator, or a RunPod CUDA template)."
        )

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_ver = torch.__version__.split("+")[0]
    tmaj, tmin = torch_ver.split(".")[:2]
    torch_minor = f"{tmaj}.{tmin}"
    cuda_major = "cu" + cuda.split(".")[0]
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"

    machine = platform.machine().lower()
    arch = "linux_aarch64" if machine in ("aarch64", "arm64") else "linux_x86_64"

    env = dict(
        py_tag=py_tag,
        torch_ver=torch_ver,
        torch_minor=torch_minor,
        cuda=cuda,
        cuda_major=cuda_major,
        abi=abi,
        arch=arch,
    )
    return env


def _minor_int(torch_minor):
    return int(torch_minor.split(".")[1])


def candidate_tags(env):
    """Ordered list of (torch_minor, abi) candidates, best match first."""
    avail = AVAILABLE.get(env["cuda_major"], {})
    if not avail:
        return []

    detected_minor = env["torch_minor"]
    detected_int = _minor_int(detected_minor)

    # Prefer the detected torch minor, then the numerically nearest published ones.
    minors = sorted(
        avail.keys(),
        key=lambda m: (m != detected_minor, abs(_minor_int(m) - detected_int)),
    )

    abi_order = [env["abi"], "TRUE" if env["abi"] == "FALSE" else "FALSE"]

    tags = []
    for m in minors:
        for a in abi_order:
            if a in avail[m] and (m, a) not in tags:
                tags.append((m, a))
    return tags


def wheel_url(spec, env, torch_minor, abi):
    fname = (
        f"{spec['pkg']}-{spec['version']}+{env['cuda_major']}torch{torch_minor}"
        f"cxx11abi{abi}-{env['py_tag']}-{env['py_tag']}-{env['arch']}.whl"
    )
    # '+' must be percent-encoded in the download URL.
    fname_url = fname.replace("+", "%2B")
    url = (
        f"https://github.com/{spec['repo']}/releases/download/"
        f"v{spec['version']}/{fname_url}"
    )
    return url, fname


def url_exists(url):
    """HEAD/GET probe so we only hand pip URLs that actually resolve."""
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return 200 <= getattr(resp, "status", 200) < 400
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            # 403/405 etc: try the other method before giving up.
            continue
        except Exception:
            return False
    return False


def install_prebuilt(spec, env):
    """Try each candidate wheel URL until one installs. Return True on success."""
    tags = candidate_tags(env)
    if not tags:
        log(f"  No published wheel matrix for {env['cuda_major']}.")
        return False

    for torch_minor, abi in tags:
        url, fname = wheel_url(spec, env, torch_minor, abi)
        exact = torch_minor == env["torch_minor"] and abi == env["abi"]
        label = "exact match" if exact else f"closest match (torch{torch_minor}/abi{abi})"
        if not url_exists(url):
            continue
        log(f"  Found {label}: {fname}")
        try:
            pip_install(url)
            return True
        except subprocess.CalledProcessError:
            log(f"  pip failed on {fname}, trying next candidate...")
            continue
    return False


def install_from_source(spec):
    log(f"  Falling back to source build for {spec['pkg']} (needs nvcc, slow)...")
    pip_install("ninja")
    pip_install(f"{spec['pkg'].replace('_', '-')}=={spec['version']}", "--no-build-isolation")


def ensure_package(spec, env, force):
    log(f"\n=== {spec['pkg']} ({spec['version']}) ===")
    if not force and is_importable(spec["import_name"]):
        log(f"  Already installed ({spec['import_name']} importable). Skipping.")
        return True

    if install_prebuilt(spec, env):
        return True

    log(f"  No compatible prebuilt wheel for torch {env['torch_ver']} / "
        f"{env['cuda_major']} / {env['py_tag']} / abi{env['abi']}.")
    try:
        install_from_source(spec)
        return True
    except subprocess.CalledProcessError:
        return False


def verify():
    log("\n=== Verifying ===")
    ok = True

    try:
        import torch
        log(f"  torch            : {torch.__version__} (cuda {torch.version.cuda})")
    except Exception as exc:
        log(f"  torch            : FAILED ({exc})")
        ok = False

    for mod in ("causal_conv1d_cuda", "selective_scan_cuda"):
        try:
            __import__(mod)
            log(f"  {mod:<17}: OK")
        except Exception as exc:
            log(f"  {mod:<17}: FAILED ({exc})")
            ok = False

    # timm moved layers between versions; the model handles both, just report it.
    try:
        import timm
        log(f"  timm             : {timm.__version__}")
    except Exception:
        log("  timm             : not installed (will be needed by the model)")

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reinstall even if importable")
    args = parser.parse_args()

    env = detect_env()
    log("=== Detected environment ===")
    log(f"  python : {env['py_tag']}")
    log(f"  torch  : {env['torch_ver']}  (minor {env['torch_minor']})")
    log(f"  cuda   : {env['cuda']}  ({env['cuda_major']})")
    log(f"  abi    : cxx11abi{env['abi']}")
    log(f"  arch   : {env['arch']}")

    # Light pure-python deps the model imports directly. Non-fatal: on Kaggle /
    # RunPod these are usually preinstalled, and the CUDA wheels are what matter.
    pip_install_soft("-q", "packaging", "einops", "timm")

    # causal-conv1d first: mamba-ssm can use it and it is a quick wheel.
    causal_ok = ensure_package(CAUSAL, env, args.force)
    mamba_ok = ensure_package(MAMBA, env, args.force)

    ok = verify()

    if not (causal_ok and mamba_ok and ok):
        log(
            "\nInstall did not fully succeed. If you are on an unusual torch/CUDA "
            "combo, the simplest fix is to use a runtime with torch 2.6-2.10 on "
            "CUDA 12.x (Kaggle GPU, or a RunPod CUDA 12.x template)."
        )
        sys.exit(1)

    log("\nAll CUDA extensions installed and importable. You can start training.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        log(f"\nA pip/command step failed (exit {exc.returncode}).")
        log("On a system-managed Python you may need a virtualenv, or pip's "
            "'--break-system-packages'. On Kaggle/RunPod GPU runtimes this step "
            "normally succeeds as-is.")
        sys.exit(1)
