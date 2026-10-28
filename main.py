"""课程实验入口；使用 E:\\conda_envs\\llm\\python.exe 执行。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

from gensim_lab.common import OUT, ROOT, LOG, checksum, load_config, setup, write_json


def main() -> None:
    setup()
    parser = argparse.ArgumentParser(description="Gensim 中英文表示学习与人民日报分类实验")
    parser.add_argument("command", choices=["prepare", "explore", "classify", "source", "verify", "report", "all", "predict"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    parser.add_argument("--labels", type=Path, help="人工标签 CSV，列 id、manual_topic；同时需要配置 class_sets")
    parser.add_argument("--text", help="predict 的新闻文本")
    parser.add_argument("--classes", choices=["2", "4", "6"], default="4")
    parser.add_argument("--method", default="fasttext_tfidf", choices=["majority", "bow", "tfidf", "lsi", "lda", "word2vec_mean", "fasttext_mean", "fasttext_tfidf"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "predict":
        from gensim_lab.classification import predict
        if not args.text:
            parser.error("predict 需要 --text")
        print(json.dumps(predict(args.text, args.classes, args.method), ensure_ascii=False, indent=2))
        return
    environment = {"python_executable": sys.executable, "python_version": sys.version, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ["numpy", "scipy", "gensim", "scikit-learn", "jieba", "matplotlib", "joblib", "threadpoolctl"]}}
    # 只读报告/源码命令不覆盖分类训练时的环境与参数证据。
    write_json(OUT / f"environment_{args.command}.json", environment)
    if args.command in {"all", "classify"} or not (OUT / "environment.json").exists():
        write_json(OUT / "environment.json", environment)
        write_json(OUT / "effective_config.json", config)
    steps = ["prepare", "explore", "classify", "source", "verify", "report"] if args.command == "all" else [args.command]
    timings = {}
    for step in steps:
        LOG.info("开始步骤 %s", step)
        start = time.perf_counter()
        if step == "prepare":
            from gensim_lab.corpus import prepare
            prepare(config, args.labels)
        elif step == "explore":
            from gensim_lab.representations import explore
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                explore(config)
        elif step == "classify":
            from gensim_lab.classification import classify
            classify(config)
        elif step == "source":
            from gensim_lab.source_guide import inspect_source
            inspect_source()
        elif step == "verify":
            from verify_artifacts import main as verify_artifacts
            verify_artifacts()
        elif step == "report":
            from gensim_lab.reporting import generate_report
            generate_report()
        timings[step] = time.perf_counter() - start
        LOG.info("步骤 %s 完成，用时 %.1fs", step, timings[step])
    write_json(OUT / f"timings_{args.command}.json", timings)


if __name__ == "__main__":
    main()
