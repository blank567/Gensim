"""冻结划分 -> 训练表示 -> 验证选参 -> 一次最终测试 -> 保存可预测模型。"""
from __future__ import annotations

from collections import Counter
import gc
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .common import OUT, LOG, write_csv, write_json
from .corpus import january_splits, sentences, tokenize_new_text
from .representations import FeatureSpace, average_vectors, coverage, train_embedding

METHODS = ["bow", "tfidf", "lsi", "lda", "word2vec_mean", "fasttext_mean", "fasttext_tfidf"]


def check_split_integrity(splits: dict) -> None:
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not ({a["id"] for a in splits[left]} & {a["id"] for a in splits[right]}), "文章 ID 泄漏"
        assert not ({a["content_hash"] for a in splits[left]} & {a["content_hash"] for a in splits[right]}), "重复文章泄漏"
    if not all(splits.values()):
        raise ValueError("某个数据划分为空，请检查日期或人工标签")


def subset(rows: list[dict], labels: list[str]):
    indices = np.array([i for i, row in enumerate(rows) if row["label"] in labels], dtype=int)
    y = np.array([rows[i]["label"] for i in indices])
    if set(y) != set(labels):
        raise ValueError(f"某个划分缺少类别：需要 {labels}，实际 {sorted(set(y))}")
    return indices, y


def select_c(xtrain, ytrain, xvalidation, yvalidation, config: dict):
    """只返回验证集最优模型；并列时保留更小的 C，防止用测试集选超参数。"""
    best_score, best_model, best_c = -1.0, None, None
    for c in config["svm_c_grid"]:
        classifier = LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=10000, random_state=config["seed"])
        classifier.fit(xtrain, ytrain)
        score = f1_score(yvalidation, classifier.predict(xvalidation), average="macro", zero_division=0)
        if score > best_score:
            best_score, best_model, best_c = float(score), classifier, c
    return best_model, best_c, best_score


def metrics(y, predictions, labels: list[str]) -> dict:
    return {"accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "classification_report": classification_report(y, predictions, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist()}


def classify(config: dict) -> dict:
    with threadpool_limits(limits=1):
        return _classify(config)


def _classify(config: dict) -> dict:
    start = time.perf_counter()
    splits = january_splits()
    check_split_integrity(splits)
    docs = {name: [a["tokens"] for a in rows] for name, rows in splits.items()}
    train_sentences = sentences(splits["train"])
    space = FeatureSpace(config)
    space.fit_topics(docs["train"])
    space.word2vec, word_stats = train_embedding(train_sentences, config, "word2vec")
    # 在四类任务上选 FastText 的子词范围与 SG/CBOW，过程只访问 train/validation。
    selection_task = "4" if "4" in config["class_sets"] else next(iter(config["class_sets"]))
    selection_labels = config["class_sets"][selection_task]
    train_idx, train_y = subset(splits["train"], selection_labels)
    val_idx, val_y = subset(splits["validation"], selection_labels)
    chosen_model, chosen_variant, chosen_stats, chosen_c, chosen_score = None, None, None, None, -1.0
    ablations = []
    weights = space.idf_by_word()
    for variant in config["fasttext_variants"]:
        model, stats = train_embedding(train_sentences, config, "fasttext", variant)
        train_x = average_vectors(docs["train"], model.wv, weights)[train_idx]
        val_x = average_vectors(docs["validation"], model.wv, weights)[val_idx]
        _, c, score = select_c(train_x, train_y, val_x, val_y, config)
        ablations.append({"variant": variant["name"], "seed": config["seed"], "validation_macro_f1": score,
                          "selected_c": c, "training_seconds": stats["training_seconds"], **variant})
        LOG.info("验证选参 %s：macro-F1=%.4f，C=%s", variant["name"], score, c)
        if score > chosen_score:
            chosen_model, chosen_variant, chosen_stats, chosen_c, chosen_score = model, variant, stats, c, score
        del model, train_x, val_x
        gc.collect()
    space.fasttext = chosen_model
    # 额外种子只测验证集稳定性；不使用这些种子的最终测试结果进行选择。
    stability = [{"seed": config["seed"], "validation_macro_f1": chosen_score}]
    for seed in config["stability_seeds"]:
        model, _ = train_embedding(train_sentences, config, "fasttext", chosen_variant, seed)
        train_x = average_vectors(docs["train"], model.wv, weights)[train_idx]
        val_x = average_vectors(docs["validation"], model.wv, weights)[val_idx]
        classifier = LinearSVC(C=chosen_c, class_weight="balanced", dual="auto", max_iter=10000, random_state=config["seed"])
        classifier.fit(train_x, train_y)
        score = float(f1_score(val_y, classifier.predict(val_x), average="macro", zero_division=0))
        stability.append({"seed": seed, "validation_macro_f1": score})
        LOG.info("验证稳定性 seed=%s：macro-F1=%.4f", seed, score)
        del model, train_x, val_x, classifier
        gc.collect()
    models_dir = OUT / "models" / "classification"
    space.save(models_dir)
    write_json(models_dir / "training_manifest.json", {
        "fit_article_ids": [a["id"] for a in splits["train"]], "fit_article_hashes": [a["content_hash"] for a in splits["train"]],
        "representation_fit_scope": "all January training articles across pages; no validation/test; no February/March; no English",
        "selected_fasttext_variant": chosen_variant,
    })
    write_csv(OUT / "hyperparameter_validation.csv", ablations)
    write_csv(OUT / "seed_validation.csv", stability)
    feature_matrices = {method: {name: space.transform(documents, method) for name, documents in docs.items()} for method in METHODS}
    # 先对全部方法/类别数完成验证与保存决策，再进入最终测试循环。
    frozen, fitted = [], {}
    for task, labels in config["class_sets"].items():
        tr_idx, ytr = subset(splits["train"], labels)
        va_idx, yva = subset(splits["validation"], labels)
        for method in ["majority"] + METHODS:
            if method == "majority":
                classifier = DummyClassifier(strategy="most_frequent")
                classifier.fit(np.zeros((len(ytr), 1)), ytr)
                score = float(f1_score(yva, classifier.predict(np.zeros((len(yva), 1))), average="macro", zero_division=0))
                c = None
            else:
                classifier, c, score = select_c(feature_matrices[method]["train"][tr_idx], ytr,
                                                 feature_matrices[method]["validation"][va_idx], yva, config)
            frozen.append({"task": task, "method": method, "labels": labels, "selected_c": c, "validation_macro_f1": score})
            fitted[(task, method)] = classifier
            joblib.dump({"classifier": classifier, "method": method, "labels": labels}, models_dir / f"classifier_{task}_{method}.joblib")
    write_json(OUT / "frozen_selection_before_test.json", frozen)
    summaries, detailed, prediction_rows = [], {}, []
    for row in frozen:
        task, method, labels = row["task"], row["method"], row["labels"]
        te_idx, ytest = subset(splits["test"], labels)
        x = np.zeros((len(te_idx), 1)) if method == "majority" else feature_matrices[method]["test"][te_idx]
        predictions = fitted[(task, method)].predict(x)
        result = metrics(ytest, predictions, labels)
        summary = {"classes": int(task), "method": method,
                   "train_n": sum(a["label"] in labels for a in splits["train"]),
                   "validation_n": sum(a["label"] in labels for a in splits["validation"]), "test_n": len(te_idx),
                   "selected_c": row["selected_c"], "validation_macro_f1": row["validation_macro_f1"],
                   **{key: result[key] for key in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]}}
        summaries.append(summary)
        detailed[f"{task}_{method}"] = {**summary, "labels": labels, **result}
        for idx, true, predicted in zip(te_idx, ytest, predictions):
            article = splits["test"][int(idx)]
            prediction_rows.append({"classes": task, "method": method, "id": article["id"], "true_label": true,
                                    "predicted_label": predicted, "correct": bool(true == predicted), "title": article["title"]})
        LOG.info("最终测试 %s 类 %-17s accuracy=%.4f macro-F1=%.4f", task, method, result["accuracy"], result["macro_f1"])
    token_coverage = {name: {"word2vec": coverage(documents, space.word2vec.wv), "fasttext": coverage(documents, space.fasttext.wv)}
                      for name, documents in docs.items()}
    results = {"protocol": "date split; fit representations on all January TRAIN pages only; tune on validation; frozen before final test",
               "class_sets": config["class_sets"], "selected_fasttext_variant": chosen_variant,
               "word2vec_training": word_stats, "fasttext_training": chosen_stats,
               "hyperparameter_validation": ablations, "seed_validation": stability,
               "validation_seed_mean": float(np.mean([s["validation_macro_f1"] for s in stability])),
               "validation_seed_std": float(np.std([s["validation_macro_f1"] for s in stability])),
               "coverage": token_coverage, "feature_shapes": {method: {name: list(x.shape) for name, x in matrix.items()} for method, matrix in feature_matrices.items()},
               "summary": summaries, "details": detailed, "total_seconds": time.perf_counter() - start}
    write_csv(OUT / "classification_summary.csv", summaries)
    write_csv(OUT / "test_predictions.csv", prediction_rows)
    write_json(OUT / "classification_results.json", results)
    LOG.info("分类完成，用时 %.1fs", results["total_seconds"])
    return results


def predict(text: str, task: str, method: str) -> dict:
    models_dir = OUT / "models" / "classification"
    artifact = joblib.load(models_dir / f"classifier_{task}_{method}.joblib")
    tokens = tokenize_new_text(text)
    if not tokens:
        raise ValueError("输入清洗后没有有效词，请输入一段完整新闻")
    if method == "majority":
        x = np.zeros((1, 1))
    else:
        space = FeatureSpace.load(models_dir)
        x = space.transform([tokens], method)
    prediction = artifact["classifier"].predict(x)[0]
    meaning = "预测版面归属；版面编号不能直接等同于人工主题标签" if all(label.startswith("page_") for label in artifact["labels"]) else "预测人工标注的主题类别"
    return {"tokens": tokens, "label": prediction, "task_classes": task, "method": method, "meaning": meaning}
