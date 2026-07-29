"""Download a pinned official Stockfish release for this platform.

The binary is never bundled with the repository (it is GPL software with
its own distribution channel, and it is large). This module fetches one
pinned release straight from the official GitHub releases page and checks
its SHA-256 before extracting, so every environment analyses positions
with the same engine.
"""

from __future__ import annotations

import hashlib
import io
import platform
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

RELEASE_TAG = "sf_18"
_BASE_URL = f"https://github.com/official-stockfish/Stockfish/releases/download/{RELEASE_TAG}/"


@dataclass(frozen=True)
class _Asset:
    file_name: str
    sha256: str


_ASSETS: dict[str, _Asset] = {
    "windows": _Asset(
        file_name="stockfish-windows-x86-64-avx2.zip",
        sha256="6f6c272ebd6ea594377715235c8a7326f75940ef4f4f856f45106028fe6ae900",
    ),
    "linux": _Asset(
        file_name="stockfish-ubuntu-x86-64-avx2.tar",
        sha256="536c0c2c0cf06450df0bfb5e876ef0d3119950703a8f143627f990c7b5417964",
    ),
}


def default_destination() -> Path:
    return Path.home() / ".opening-theory-engine" / "stockfish"


def fetch(dest_dir: Path | None = None, quiet: bool = False) -> Path:
    """Download and extract Stockfish for this platform. Returns the binary path."""
    key = _platform_key()
    asset = _ASSETS[key]
    dest_dir = dest_dir or default_destination()
    dest_dir.mkdir(parents=True, exist_ok=True)
    binary_path = dest_dir / ("stockfish.exe" if key == "windows" else "stockfish")
    if binary_path.is_file():
        return binary_path

    archive = _download(_BASE_URL + asset.file_name, asset.sha256, quiet=quiet)
    payload = _extract_binary(archive, key)
    binary_path.write_bytes(payload)
    if key != "windows":
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return binary_path


def _platform_key() -> str:
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise RuntimeError(
            f"No pinned Stockfish build for machine '{machine}'. Install Stockfish "
            "yourself and set STOCKFISH_PATH."
        )
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(
        "No pinned Stockfish build for this platform. On macOS use "
        "'brew install stockfish' and set STOCKFISH_PATH."
    )


def _download(url: str, expected_sha256: str, quiet: bool) -> bytes:
    digest = hashlib.sha256()
    chunks = []
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="stockfish",
            disable=quiet,
        )
        for chunk in response.iter_bytes():
            digest.update(chunk)
            chunks.append(chunk)
            progress.update(len(chunk))
        progress.close()
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}. "
            "Refusing to install."
        )
    return b"".join(chunks)


def _extract_binary(archive: bytes, key: str) -> bytes:
    if key == "windows":
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            names = [name for name in bundle.namelist() if name.endswith(".exe")]
            if not names:
                raise RuntimeError("no .exe found inside the Stockfish archive")
            return bundle.read(names[0])
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            base_name = member.name.rsplit("/", maxsplit=1)[-1]
            if member.isfile() and base_name.startswith("stockfish"):
                extracted = bundle.extractfile(member)
                if extracted is None:
                    continue
                return extracted.read()
    raise RuntimeError("no stockfish binary found inside the archive")
