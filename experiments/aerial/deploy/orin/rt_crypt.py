#!/usr/bin/env python3
"""AES-256-CBC (OpenSSL pbkdf2) at-rest encryption for product weights.

Ciphertext stays on disk; plaintext is materialised under /dev/shm/rtw.
Key file is NOT stored next to the weights.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AERIAL_ORIN_HOME", "/opt/aerial_orin"))
KEY = Path(os.environ.get("RT_KEY", str(Path.home() / ".config/rt/k")))
RAM = Path(os.environ.get("RT_RAM", "/dev/shm/rtw"))
ITER = "200000"


def _openssl(args: list[str]) -> None:
    r = subprocess.run(["openssl", *args], capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace"))
        raise SystemExit(r.returncode)


def ensure_key() -> Path:
    KEY.parent.mkdir(parents=True, exist_ok=True)
    if not KEY.is_file():
        KEY.write_bytes(os.urandom(32).hex().encode() + b"\n")
        KEY.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return KEY


def plain_files() -> list[Path]:
    out: list[Path] = []
    w = ROOT / "weights"
    for p in w.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix == ".enc" or p.name.endswith(".enc"):
            continue
        out.append(p)
    clip = Path.home() / ".cache/clip/ViT-B-32.pt"
    if clip.is_file() and clip.stat().st_size > 1000:
        out.append(clip)
    return out


def enc_path(src: Path) -> Path:
    if src.as_posix().endswith("/.cache/clip/ViT-B-32.pt") or src.name == "ViT-B-32.pt":
        return ROOT / "weights" / "m_t.bin.enc"
    rel = src.relative_to(ROOT / "weights") if ROOT / "weights" in src.parents or src.parent == ROOT / "weights" else None
    if rel is not None:
        return ROOT / "weights" / (str(rel) + ".enc")
    return ROOT / "weights" / (src.name + ".enc")


def encrypt() -> None:
    ensure_key()
    files = plain_files()
    if not files:
        print("encrypt: no plaintext weights")
        return
    for src in files:
        dst = enc_path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("enc", src.name, "->", dst.relative_to(ROOT))
        _openssl([
            "enc", "-aes-256-cbc", "-pbkdf2", "-iter", ITER, "-salt",
            "-in", str(src), "-out", str(dst), "-pass", f"file:{KEY}",
        ])
        src.unlink()
    # empty dirs
    clip = Path.home() / ".cache/clip"
    if clip.is_dir() and not any(clip.iterdir()):
        clip.rmdir()
    print("encrypt_ok", len(files))


def decrypt() -> Path:
    ensure_key()
    RAM.mkdir(parents=True, exist_ok=True)
    os.chmod(RAM, 0o700)
    wdst = RAM / "weights"
    wdst.mkdir(exist_ok=True)
    encs = sorted((ROOT / "weights").rglob("*.enc"))
    if not encs:
        raise SystemExit("no ciphertext under weights/")
    for enc in encs:
        if enc.name == "m_t.bin.enc":
            dest = RAM / "clip" / "ViT-B-32.pt"
        else:
            rel = enc.relative_to(ROOT / "weights")
            dest = wdst / Path(str(rel)[:-4])  # strip .enc
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        _openssl([
            "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", ITER,
            "-in", str(enc), "-out", str(dest), "-pass", f"file:{KEY}",
        ])
        os.chmod(dest, 0o600)
    cache = Path.home() / ".cache/clip"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_symlink() or cache.is_dir():
        if cache.is_symlink():
            cache.unlink()
        elif not any(cache.iterdir()):
            cache.rmdir()
    if not cache.exists():
        os.symlink(RAM / "clip", cache, target_is_directory=True)
    return RAM


def wipe() -> None:
    import shutil
    if RAM.exists():
        shutil.rmtree(RAM, ignore_errors=True)
    cache = Path.home() / ".cache/clip"
    if cache.is_symlink():
        cache.unlink()


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("encrypt", "decrypt", "wipe"):
        print("usage: rt_crypt.py encrypt|decrypt|wipe", file=sys.stderr)
        return 2
    cmd = sys.argv[1]
    if cmd == "encrypt":
        encrypt()
    elif cmd == "decrypt":
        p = decrypt()
        print(p)
    else:
        wipe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
