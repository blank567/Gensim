"""公共 IO 与运行配置。所有输出、缓存和临时目录固定在 E 盘。"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts"
LOG = logging.getLogger("gensim_lab")


def setup() -> None:
    """在导入 jieba / matplotlib 前设置缓存位置。"""
    for path in [DATA / "raw", DATA / "processed", OUT, ROOT / ".cache" / "jieba"]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = "E:/codex/tmp"
    os.environ["TMP"] = "E:/codex/tmp"
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
    os.environ["JOBLIB_TEMP_FOLDER"] = str(ROOT / ".cache" / "joblib")
    # BLAS 并行与 Gensim 的 workers 是两套线程；小实验限制前者避免过度并行。
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if not LOG.handlers:
        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run.log", encoding="utf-8")]:
            handler.setFormatter(formatter)
            LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)


def load_config(path: str | Path | None = None) -> dict:
    """公共默认配置可叠加本机覆盖；显式指定的其他配置保持独立。"""
    default = ROOT / "configs" / "default.json"
    selected = Path(path) if path else default
    config = read_json(selected)
    local = ROOT / "configs" / "local.json"
    if selected.resolve() == default.resolve() and local.is_file():
        config.update(read_json(local))
    return config


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(word: str) -> int:
    """Word2Vec 初始化使用固定哈希，避免 Python 随机哈希影响重复实验。"""
    return zlib.crc32(word.encode("utf-8"))


def download(url: str, target: Path) -> Path:
    """只下载固定的公开语料/源码；已有文件直接复用，临时文件也在 E 盘。"""
    if target.exists():
        return target
    LOG.info("下载 %s", url)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "GensimCourseLab/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read()
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_bytes(content)
    partial.replace(target)
    return target
