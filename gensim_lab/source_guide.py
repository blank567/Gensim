"""把实际安装版本的源码复制到项目内，并生成可点击的准确行号索引。"""
from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path
import shutil

import gensim
from gensim.models import FastText, Word2Vec
from gensim.models.fasttext import FastTextKeyedVectors
from gensim.utils import SaveLoad

from .common import OUT, ROOT, LOG, checksum, write_json


def inspect_source() -> dict:
    version = importlib.metadata.version("gensim")
    package = Path(gensim.__file__).parent
    destination = OUT / "source" / f"gensim-{version}"
    files = ["models/fasttext.py", "models/fasttext_inner.pyx", "models/fasttext_corpusfile.pyx",
             "models/word2vec.py", "models/word2vec_inner.pyx", "models/keyedvectors.py", "utils.py"]
    copied = []
    for relative in files:
        source = package / relative
        if not source.exists():
            raise FileNotFoundError(f"本安装缺少源码文件 {source}；请获取对应版本的 Gensim sdist")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append({"relative": relative, "snapshot": str(target), "installed": str(source), "sha256": checksum(target),
                       "url": f"https://github.com/piskvorky/gensim/blob/{version}/gensim/{relative}"})
    targets = [
        (FastText, "__init__", "模型参数、继承与初始化"),
        (Word2Vec, "build_vocab", "扫描语料、计数、准备词表和矩阵"),
        (Word2Vec, "prepare_vocab", "min_count、sample、词频过滤"),
        (Word2Vec, "make_cum_table", "negative 与 ns_exponent 的负采样分布"),
        (Word2Vec, "train", "训练入口、轮次与学习率安排"),
        (Word2Vec, "_job_producer", "把可重迭代语料组织为批任务"),
        (Word2Vec, "_worker_loop", "线程队列、工作内存与进度汇总"),
        (FastText, "_do_train_job", "转交 Cython train_batch_any"),
        (FastTextKeyedVectors, "recalc_char_ngram_buckets", "字符 n-gram 与哈希桶索引缓存"),
        (FastTextKeyedVectors, "adjust_vectors", "将整词向量与子词向量组成查询向量"),
        (FastTextKeyedVectors, "get_vector", "词表内查询与 OOV 合成分支"),
        (SaveLoad, "save", "保存、分离大矩阵与生命周期元数据"),
        (SaveLoad, "load", "加载与兼容性处理"),
    ]
    symbols = []
    for cls, attribute, purpose in targets:
        function = getattr(cls, attribute)
        source_file = Path(inspect.getsourcefile(function))
        relative = source_file.relative_to(package).as_posix()
        line = inspect.getsourcelines(function)[1]
        symbols.append({"symbol": f"{cls.__name__}.{attribute}", "file": relative, "line": line, "purpose": purpose})
    # Cython 函数无法由 inspect 返回 Python 源码，因此在匹配版本的 .pyx 中定位定义。
    cython = destination / "models" / "fasttext_inner.pyx"
    for number, line in enumerate(cython.read_text(encoding="utf-8").splitlines(), 1):
        for token, purpose in [("def compute_ngrams_bytes(", "Unicode 字符边界与 UTF-8 字节子词"),
                               ("def ft_hash_bytes(", "FNV-1a 哈希"),
                               ("def train_batch_any(", "统一批训练入口，释放 GIL"),
                               ("cdef void fasttext_train_any(", "分派 SG/CBOW、负采样/HS"),
                               ("cdef void fasttext_fast_sentence_sg_neg(", "负采样梯度更新")]:
            if token in line:
                symbols.append({"symbol": token.split("(")[0].split()[-1], "file": "models/fasttext_inner.pyx", "line": number, "purpose": purpose})
    result = {"version": version, "files": copied, "symbols": symbols}
    write_json(OUT / "source" / "source_manifest.json", result)
    lines = ["# 实际安装版本源码索引", "", f"本次版本：Gensim **{version}**。以下快照直接复制自 llm 环境，SHA256 见 source_manifest.json。", "",
             "官方网页可能仍显示 4.3.3；本实验以实际安装的 4.4.0 和对应版本源码为准。", "",
             "| 入口 | 源码位置 | 阅读目的 |", "|---|---|---|"]
    for row in symbols:
        target = (destination / row["file"]).as_posix()
        lines.append(f"| `{row['symbol']}` | [{row['file']}:{row['line']}]({target}:{row['line']}) | {row['purpose']} |")
    reading = "建议先阅读 [论文与实现对照](FASTTEXT_PAPER_AND_CODE.md)" if (ROOT / "docs" / "FASTTEXT_PAPER_AND_CODE.md").exists() else "建议先阅读 [项目说明](../README.md) 中的 FastText 与源码阅读说明"
    lines += ["", reading + "，再按表从 Python 调用链走到 Cython 内核。", "",
              "快照保留上游许可证与作者信息，仅作为阅读材料；不要修改 site-packages 来做课程消融实验。"]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "SOURCE_MAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("已索引 %s 个源码入口，快照版本 %s", len(symbols), version)
    return result
