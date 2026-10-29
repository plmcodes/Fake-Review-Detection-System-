"""Download the PUBLIC datasets from their original sources (nothing is redistributed by this repo).

    python -m src.download_data            # Salminen + Ott (~16 MB)
    python -m src.download_data --glove    # also GloVe 6B 100d (downloads an 862 MB zip once)

YelpChi cannot be downloaded automatically - it is shared by its authors on request.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from .config import resolve_path
from .utils import get_logger

log = get_logger("download")

SALMINEN_URL = "https://osf.io/download/3vds7/"          # OSF project tyue9 (the authors' upload)
SALMINEN_SHA256 = "ea03422988824bc917b5f7217e3867e27b597c647d5261adfae73cd7a85e1e1a"
OTT_URL = "https://myleott.com/op_spam_v1.4.zip"           # Myle Ott's page
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"  # mirror of nlp.stanford.edu


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_salminen(raw: Path) -> None:
    dest = raw / "salminen" / "fake reviews dataset.csv"
    if not dest.exists():
        _download(SALMINEN_URL, dest)
    digest = _sha256(dest)
    if digest != SALMINEN_SHA256:
        log.warning("Salminen file checksum differs from the one used in this project (%s)", digest)
    log.info("Salminen OK: %s", dest)


def get_ott(raw: Path) -> None:
    folder = raw / "ott"
    if (folder / "op_spam_v1.4").exists():
        log.info("Ott OK: %s", folder / "op_spam_v1.4")
        return
    zpath = folder / "op_spam_v1.4.zip"
    if not zpath.exists():
        _download(OTT_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(folder)
    log.info("Ott extracted to %s", folder / "op_spam_v1.4")


def get_glove(raw: Path, dim: int = 100) -> None:
    target = raw / "glove" / f"glove.6B.{dim}d.txt"
    if target.exists():
        log.info("GloVe OK: %s", target)
        return
    zpath = raw / "glove" / "glove.6B.zip"
    if not zpath.exists():
        _download(GLOVE_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract(target.name, target.parent)
    zpath.unlink()   # keep only the dimension we use (saves ~800 MB)
    log.info("GloVe extracted: %s", target)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glove", action="store_true", help="also download GloVe 6B (862 MB zip)")
    args = ap.parse_args(argv)
    raw = resolve_path("data/raw")
    get_salminen(raw)
    get_ott(raw)
    if args.glove:
        get_glove(raw)
    log.info("YelpChi: optional, request it from the authors - see data/README.md")


if __name__ == "__main__":
    main()
