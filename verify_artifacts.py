"""训练后核验：输入指纹、源码指纹、训练边界和模型保存后的预测一致性。"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

from gensim_lab.common import DATA, OUT, checksum, read_json, setup, write_json
setup()
import joblib
import numpy as np
from gensim_lab.classification import check_split_integrity, subset
from gensim_lab.corpus import january_splits, tokenize_new_text
from gensim_lab.representations import FeatureSpace


def main():
    splits = january_splits()
    check_split_integrity(splits)
    manifest = read_json(OUT / "models" / "classification" / "training_manifest.json")
    assert set(manifest["fit_article_ids"]) == {a["id"] for a in splits["train"]}
    assert not (set(manifest["fit_article_ids"]) & {a["id"] for a in splits["validation"] + splits["test"]})
    audit = read_json(DATA / "processed" / "data_audit.json")
    for source in audit["sources"]:
        assert checksum(Path(source["path"])) == source["sha256"], "原始文件指纹不一致"
    source_manifest = read_json(OUT / "source" / "source_manifest.json")
    for source in source_manifest["files"]:
        assert checksum(Path(source["snapshot"])) == source["sha256"]
        assert checksum(Path(source["installed"])) == source["sha256"]
    space = FeatureSpace.load(OUT / "models" / "classification")
    results = read_json(OUT / "classification_results.json")
    with (OUT / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    checks = {}
    for method in ["tfidf", "fasttext_mean", "fasttext_tfidf"]:
        artifact = joblib.load(OUT / "models" / "classification" / f"classifier_4_{method}.joblib")
        indices, _ = subset(splits["test"], artifact["labels"])
        docs = [splits["test"][int(i)]["tokens"] for i in indices]
        features = space.transform(docs, method)
        actual = artifact["classifier"].predict(features)
        expected = [r["predicted_label"] for r in predictions if r["classes"] == "4" and r["method"] == method]
        assert list(actual) == expected, f"{method} 保存加载后的预测不一致"
        checks[method] = {"prediction_rows": len(actual), "identical_after_reload": True}
    new_tokens = tokenize_new_text("中国队在国际比赛中夺得冠军，运动员备战奥运会。")
    new_x = space.transform([new_tokens], "fasttext_tfidf")
    assert np.isfinite(new_x).all() and np.linalg.norm(new_x) > 0
    output = {"verified_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "split_integrity": True, "fit_scope_verified": True, "input_sha256_verified": True,
              "source_snapshots_match_installed": True, "saved_predictions": checks,
              "new_jieba_input_vector_shape": list(new_x.shape), "new_input_finite_nonzero": True}
    write_json(OUT / "verification.json", output)
    dependency_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, encoding="utf-8")
    write_json(OUT / "dependency_check.json", {"returncode": dependency_check.returncode,
               "stdout": dependency_check.stdout.strip(), "stderr": dependency_check.stderr.strip(),
               "note": "Full llm environment check; project artifact checks are independent. No unrelated package upgrades performed."})
    print(output)
    if dependency_check.returncode:
        print("llm 全环境依赖检查存在冲突，详见 artifacts/dependency_check.json；上述实验产物核验已通过。")


if __name__ == "__main__":
    main()
