"""根据实际 JSON/CSV 生成中文报告与独立图片，不填入估计或虚构指标。"""
from __future__ import annotations

from collections import Counter
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import DATA, OUT, ROOT, LOG, read_json, write_json


def table(headers: list[str], rows: list[list]) -> str:
    text = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    text.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(text)


def generate_plots(results: dict) -> None:
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    methods = list(dict.fromkeys(row["method"] for row in results["summary"]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    colors = ["#a5acb5", "#899eae", "#267d8d", "#85b7bf", "#b5c6ba", "#cca76c", "#d27b58", "#a55145"]
    for ax, task in zip(axes, [2, 4, 6]):
        rows = [r for r in results["summary"] if r["classes"] == task]
        values = [next(r["macro_f1"] for r in rows if r["method"] == method) for method in methods]
        bars = ax.bar(np.arange(len(methods)), values, color=colors)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_title(f"{task} classes")
        ax.set_xticks(np.arange(len(methods)), [m.replace("_", "\n") for m in methods], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Held-out test macro-F1")
    fig.suptitle("January 1998 news: representation comparison (date split)")
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(figures / f"classification_comparison.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, ["2", "4", "6"]):
        row = results["details"][f"{task}_fasttext_tfidf"]
        matrix = np.array(row["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black", fontsize=8)
        labels = [label.replace("page_", "P") for label in row["labels"]]
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"FastText + IDF: {task} classes")
    fig.tight_layout()
    fig.savefig(figures / "fasttext_confusion.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def generate_report() -> None:
    audit = read_json(DATA / "processed" / "data_audit.json")
    result = read_json(OUT / "classification_results.json")
    exploration_path = OUT / "exploration.json"
    exploration = read_json(exploration_path) if exploration_path.exists() else None
    env = read_json(OUT / "environment.json")
    is_page_task = audit["label_type"] == "newspaper_page"
    target_name = "版面" if is_page_task else "主题"
    generate_plots(result)
    lines = [f"# Gensim 表示学习与人民日报{target_name}分类实验报告", "",
             "本文由实际运行产物自动生成；详细指标、逐文预测、训练清单和参数均可核查。", "",
             "## 1. 实验目标与结论边界", "",
             "实验覆盖词袋、TF-IDF、LSI、LDA、Word2Vec 和 FastText。分类器统一采用带类别权重的 LinearSVC，并增加多数类基线。", "",
             ("原始语料没有人工主题标签，本实验自选的目标是**根据正文预测文章版面编号**。这是一项可复核的版面分类任务；同一版面可能混合主题，不能把结果宣称为体育、财经等人工主题分类准确率。" if is_page_task else
              "本次根据用户提供的人工标注表进行主题分类；未标注文档被排除。标签质量和标注一致性仍需人工审核。"), "",
             "## 2. 数据处理与划分", "",
             table(["输入月份", "原始文章", "原始段落", "编码替代字符"],
                   [[a["path"].split("/")[-1].split("\\")[-1], a["raw_articles"], a["line_count"], a["replacement_characters"]] for a in audit["sources"]]), "",
             f"共 {audit['raw_articles']} 篇原始文章；清洗后少于 30 个词的短文移除 {audit['short_articles_removed']} 篇，全文精确/近重复移除 {audit['duplicates_removed']} 篇。1 月保留 {audit['january_articles']} 篇、{audit['january_tokens']:,} 个词；2/3 月保留 {audit['external_zh_articles']} 篇。", "",
             "按文章 ID 合并段落，沿用已有分词，删除拼音、标点、纯数字和固定停用词。编号和版号仅用于标签/分组，从不进入文本特征。GB18030 解码遇坏字用替代字符标记，位置记录在 data_audit.json，清洗时移除。", "",
             "重复检测使用清洗文本 SHA256 与词二元组 64 位 SimHash（汉明距离 ≤ 3），按日期保留最早文章。这只能控制词面高度相近的重复，不能保证排除所有同事件报道。", "",
             "时间划分：1 月 1—19 日训练，20—24 日验证，25—31 日测试。全部表示模型只使用 1 月训练文章；2/3 月和英文探索模型不会被分类代码加载。", "",
             table(["类别", "训练", "验证", "测试"], [[label] + [audit["split_label_counts"][s].get(label, 0) for s in ["train", "validation", "test"]]
                                                        for label in sorted(audit["split_label_counts"]["train"])]), "",
             "类别集合：" + "；".join(f"{task} 类={','.join(labels)}" for task, labels in result["class_sets"].items()) + "。样本数也会变化，不同类别数之间的 F1 差异不能完全归因于类别数量。", "",
             "## 3. 验证集选参与稳定性", "",
             "先在四类任务的验证集上比较 FastText 的字符子词范围与 SG/CBOW，再冻结选中的方案。各表示方法、各类别任务的 SVM C 从 0.1/1/10 中按验证 macro-F1 选择。没有在 train+validation 上重训，以保持所有方法统一的训练边界。", "",
             table(["方案", "种子", "验证 macro-F1", "C", "向量训练秒数"],
                   [[r["variant"], r["seed"], f"{r['validation_macro_f1']:.4f}", r["selected_c"], f"{r['training_seconds']:.1f}"] for r in result["hyperparameter_validation"]]), "",
             f"选中方案：`{result['selected_fasttext_variant']['name']}`。同一方案在种子 42/43/44 上的验证 macro-F1 为均值 {result['validation_seed_mean']:.4f}、标准差 {result['validation_seed_std']:.4f}。这只反映该固定划分下的训练随机性，不是跨数据划分的置信区间。", "",
             "## 4. 独立测试结果", "",
             table(["类别数", "表示", "测试文章", "验证 F1", "测试 Accuracy", "测试 macro-F1", "测试 weighted-F1"],
                   [[r["classes"], r["method"], r["test_n"], f"{r['validation_macro_f1']:.4f}", f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}"] for r in result["summary"]]), "",
             "![不同表示的测试 macro-F1](figures/classification_comparison.png)", "",
             "![FastText 混淆矩阵](figures/fasttext_confusion.png)", "",
             "图表和各方法分数用于本次预先约定的比较。部署方法应按验证集选择；看到测试结果后再次调参，需要重新保留独立测试集。", "",
             "## 5. OOV 与向量覆盖", "",
             table(["划分", "Word2Vec 词表内 token 占比", "FastText 词表内 token 占比", "FastText 可合成向量 token 占比"],
                   [[name, f"{values['word2vec']['in_vocabulary_rate']:.2%}", f"{values['fasttext']['in_vocabulary_rate']:.2%}",
                     f"{values['fasttext']['vector_constructible_rate']:.2%}"] for name, values in result["coverage"].items()]), "",
             "FastText 可合成 OOV 向量，不能据此推断语义正确。哈希碰撞、未充分训练的子词、短中文词和有限语料都可能降低表示质量。TF-IDF 在小规模新闻分类上具有较强的词面判别能力，实际表现应以本表为准。", ""]
    if exploration:
        en = exploration["english_corpus"]
        lines += ["## 6. 独立的中英文表示探索", "",
                  f"英文使用官方 FastText 教程同款 ABC/Lee 新闻，{en['documents']} 篇、{en['tokens']:,} 个词。下载地址和 SHA256 保存在 exploration.json。该集合规模小，用于理解 API、检索和子词行为，不能支撑普遍性能结论。", "",
                  "英文查询：`government election minister`。以下为各表示检索到的第一篇新闻：", "",
                  table(["表示", "文档序号", "相似度", "开头内容"],
                        [[method, rows[0]["document"], f"{rows[0]['cosine']:.4f}", rows[0]["snippet"]] for method, rows in exploration["retrieval"].items()]), "",
                  "2/3 月中文新闻另行训练 FastText，保存中文词近邻与 OOV 演示。它与 1 月分类训练分离，因此不会把后续月份作为分类的预训练资料。", "",
                  "LDA 主题词、词向量近邻与完整检索前三名见 [探索结果](exploration.json)。词向量近邻仅作为定性观察，没有人工词相似度金标准。", ""]
    lines += ["## 7. 方法选择、误差与限制", ""]
    decisions = {}
    with (OUT / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    for task in [2, 4, 6]:
        candidates = [r for r in result["summary"] if r["classes"] == task and r["method"] != "majority"]
        chosen = max(candidates, key=lambda r: r["validation_macro_f1"])
        decisions[str(task)] = {"method": chosen["method"], "validation_macro_f1": chosen["validation_macro_f1"]}
        lines += [f"### {task} 类任务", "", f"按验证集选择 `{chosen['method']}`：验证 macro-F1={chosen['validation_macro_f1']:.4f}；对应独立测试 macro-F1={chosen['macro_f1']:.4f}。", ""]
        errors = [r for r in prediction_rows if r["classes"] == str(task) and r["method"] == chosen["method"] and r["correct"] == "False"][:5]
        lines += [table(["文章 ID", "真实类别", "预测类别", "标题"], [[r["id"], r["true_label"], r["predicted_label"], r["title"]] for r in errors]), ""]
    label_limit = "版面标签具有混合主题和编辑排版因素；本次未开展人工主题分类；" if is_page_task else "人工标签数量与一致性影响结论；"
    lines += ["局限：" + label_limit + "同事件报道可能跨划分；单月样本量有限；三种子结果只在验证集比较；不同表示的模型容量和训练成本不同；原语料分词与 jieba 新输入分词存在差异。未开展跨年份测试和大规模性能基准。", "",
              "## 8. 复现证据", "",
              f"Python：`{env['python_executable']}`。包版本：" + "，".join(f"{name}={version}" for name, version in env["packages"].items()) + "。", "",
              "- effective_config.json：本次全部显式实验参数。",
              "- classification_summary.csv / classification_results.json：汇总与逐类指标。",
              "- test_predictions.csv：逐篇测试预测，可筛选错例。",
              "- frozen_selection_before_test.json：最终测试前冻结的选择。",
              "- models/classification/training_manifest.json：表示训练文章 ID 与内容哈希。",
              "- source/source_manifest.json：实际安装源码版本、行号、SHA256。",
              "- environment.json / run.log：环境与实际进度。", "",
              "项目说明、复现步骤和参考资料见 [README](../README.md)，已整理的实验结果见 [RESULTS](../RESULTS.md)。"]
    if (ROOT / "docs" / "FASTTEXT_PAPER_AND_CODE.md").exists():
        lines += ["", "本机补充材料：[论文对照说明](../docs/FASTTEXT_PAPER_AND_CODE.md)、[超参数手册](../docs/PARAMETERS.md)。"]
    dependency_path = OUT / "dependency_check.json"
    if dependency_path.exists():
        dependency = read_json(dependency_path)
        lines += ["", "## 9. 环境完整性检查", ""]
        if dependency["returncode"]:
            lines += ["本次 Gensim 的导入、训练、测试和模型加载验证均已通过，但 `pip check` 检测到 llm 共享环境中已有的其他依赖冲突：", "",
                      "```text", dependency["stdout"] or dependency["stderr"], "```", "",
                      "本次安装包列表不包含 LangChain/LangGraph，保留了已有配置。该冲突不能据此记为 pip check 通过，后续运行相关项目时应单独处理。"]
        else:
            lines += ["`pip check` 通过，未发现已安装包的声明依赖冲突。"]
    (OUT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "validation_selected_methods.json", decisions)
    LOG.info("已生成 artifacts/EXPERIMENT_REPORT.md 与 PNG/SVG 图表")
