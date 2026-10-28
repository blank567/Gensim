# Gensim 文本表示学习与新闻分类实验

使用 **Gensim、scikit-learn 和 jieba**，从新闻清洗、文本向量化到 FastText 训练与分类评估，完成一套可运行、可检查的表示学习实验。

项目比较 BoW、TF-IDF、LSI、LDA、Word2Vec 和 FastText，并基于人民日报 1998 年 1 月语料逐步开展 **2／4／6 类版面归属预测**。额外使用 ABC/Lee 英文新闻及 2—3 月中文新闻进行独立的向量探索。

> 原始文件没有人工主题标签，本项目使用文章编号中的版面号作为标签。同一版面可能混合多种主题，因此结果不能直接解释为“体育／财经／科技”等人工主题分类的准确率。

## 功能

- **数据处理**：GB18030 解码、合并文章段落、去拼音与停用词、记录坏字、去重和按日期划分。
- **表示学习**：BoW、TF-IDF、LSI、LDA、Word2Vec，FastText 均值与 IDF 加权文档向量。
- **分类比较**：多数类基线与 LinearSVC，验证选参、独立测试、逐类指标及错例。
- **参数实验**：Skip-gram／CBOW、字符子词范围、分类器 C 和随机种子稳定性。
- **向量探索**：词语近邻、OOV 合成、主题词、相似新闻检索。
- **源码与核验**：实际安装版本的 Python／Cython 入口索引，输入指纹与模型重载预测核验。

## 实验结果

以下来自 **2026-10-22 已完成的运行**，指标为独立测试 **macro-F1**。完整结果、参数与限制见 [RESULTS.md](RESULTS.md)。

| 类别数 | 测试文章 | TF-IDF + LinearSVC | FastText 均值 + LinearSVC | FastText + IDF + LinearSVC |
|---|---:|---:|---:|---:|
| 2 | 107 | **0.9605** | 0.8616 | 0.8790 |
| 4 | 158 | **0.7926** | 0.7577 | 0.7294 |
| 6 | 331 | **0.6215** | 0.5166 | 0.5402 |

本次 TF-IDF 在三个任务中均优于这两种 FastText 聚合方式。FastText 可为 OOV 构造向量，但不保证语义可靠或下游分类更准确；结果仅适用于本次语料、划分和参数。

![各表示方法的独立测试 macro-F1](results/figures/classification_comparison.png)

## 数据与评估协议

三份中文输入文件为 `1998-01-2003版-带音.txt`、`1998-02-2003版-带音.txt`、`1998-03-2003版-带音.txt`。每行包含段落编号和已分词正文，例如：

```text
19980101-01-001-001 迈向 充满 希望 的 新 世纪 ...
```

先按 `日期－版号－文章号` 聚合文章，再清洗与去重。三个月原始共 8,663 篇，移除过短文本 344 篇、精确或近重复 5 篇；1 月保留 2,991 篇，2—3 月保留 5,323 篇。

| 任务 | 标签集合 |
|---|---|
| 2 类 | `page_01`、`page_06` |
| 4 类 | 上述两类 + `page_08`、`page_05` |
| 6 类 | 上述四类 + `page_02`、`page_04` |

分类按 1 月日期划分：**1—19 日训练、20—24 日验证、25—31 日测试**。词典、IDF、主题模型和词向量只在训练文章上拟合；验证集选参数，全部选择冻结后再测试。文章编号和版面标签不进入文本特征。

英文探索使用官方教程的 300 篇 ABC/Lee 新闻，首次运行自动下载。英文模型和 2—3 月中文探索模型分开保存，不混入 1 月分类训练。

## 环境与运行

已验证：Windows、Python 3.10.19、Gensim 4.4.0、scikit-learn 1.7.2、jieba 0.42.1。训练使用 CPU／Cython，无需 PyTorch 或 CUDA。本机解释器为 `E:\conda_envs\llm\python.exe`。

```powershell
git clone https://github.com/blank567/Gensim.git E:\codex\course\Gensim
Set-Location E:\codex\course\Gensim
$env:TEMP = 'E:\codex\tmp'
$env:TMP = 'E:\codex\tmp'
$env:PIP_CACHE_DIR = 'E:\codex\course\Gensim\.cache\pip'
& E:\conda_envs\llm\python.exe -m pip install --cache-dir $env:PIP_CACHE_DIR -r requirements-lock.txt
```

将三份人民日报文本自行放入 `data/raw/`。语料和模型未随仓库分发，完整实验需要先准备输入。所有生成文件与缓存均保存到 E 盘。

如果文件位于其他位置，新建不会被 Git 跟踪的 `configs/local.json`：

```json
{
  "sources": [
    "E:/datasets/people_daily/1998-01-2003版-带音.txt",
    "E:/datasets/people_daily/1998-02-2003版-带音.txt",
    "E:/datasets/people_daily/1998-03-2003版-带音.txt"
  ]
}
```

默认配置会合并本地覆盖；显式 `--config` 指定的其他配置独立使用。

```powershell
# 完整流程：清洗 -> 独立探索 -> 分类 -> 源码索引 -> 核验 -> 报告
powershell -ExecutionPolicy Bypass -File .\run.ps1 all

# 加载已训练模型，预测新新闻的版面归属
powershell -ExecutionPolicy Bypass -File .\run.ps1 predict -Classes 4 -Method fasttext_tfidf -Text '中国队在比赛中夺得冠军，运动员备战奥运会。'
```

激活 `llm` 后也可使用 `python -X utf8 main.py all`。完整运行通常需数分钟，具体耗时以本地运行记录为准。

| 分步命令 | 功能 |
|---|---|
| `python main.py prepare` | 语料解析、清洗、去重与日期划分 |
| `python main.py explore` | 独立中英文表示学习、检索与 OOV 演示 |
| `python main.py classify` | 表示训练、参数选择和 2／4／6 类测试 |
| `python main.py source` | 生成本地源码快照和 `docs/SOURCE_MAP.md` |
| `python main.py verify` | 核对输入指纹、训练边界与模型重载预测 |
| `python main.py report` | 生成本地中文实验报告和图表 |

入门示例：`python examples/01_quickstart.py`；训练后的词向量查询：`python examples/02_query_trained_fasttext.py`。前者用玩具语料演示 API，不用于性能结论。

## FastText 与源码阅读

文章表示流程：`分词 -> 每词向量 [d] -> 均值或 IDF 加权平均 -> 文档矩阵 [N,d] -> L2 归一化 -> LinearSVC`，本实验 `d=64`。

FastText 额外学习字符子词参数，允许不同词共享片段。判断是否真正出现于训练词表应查看 `word in model.wv.key_to_index`；`word in model.wv` 还可能表示它可以通过子词构造向量。

Gensim 4.4.0 中主要调用链为：

```text
build_vocab -> prepare_vocab / recalc_char_ngram_buckets
train -> _job_producer / _worker_loop -> FastText._do_train_job
      -> fasttext_inner.train_batch_any -> fasttext_train_any
      -> SG/CBOW 与负采样/HS 内核

子词：compute_ngrams_bytes -> ft_hash_bytes -> 哈希桶
查询：FastTextKeyedVectors.get_vector / adjust_vectors
```

论文公式、训练内核与查询向量的组合缩放需分别核对。本实验使用 64 维、50,000 个桶、负采样幂次 0.75、单线程，未完整复现论文的多语言基准。

这里的分类是 **Gensim 词向量聚合 + scikit-learn 分类器**，分类损失不反向更新词向量，与 Facebook fastText 的端到端 `supervised` 分类流程不同。

## 验证与仓库范围

```powershell
& E:\conda_envs\llm\python.exe -m pytest -q --basetemp E:\codex\course\Gensim\.cache\pytest-tmp
```

发布前于 2026-10-28 在 `llm` 环境重新检查，**9 项测试通过**。测试覆盖文章聚合、拼音清洗、坏字审计、重复数据泄漏、训练词表／IDF 隔离、OOV 保存加载和本地配置覆盖。另核验了三种表示在 158 篇四类测试文章上的模型重载预测一致性，并成功处理新文本。

仓库保留源码、配置模板、测试、示例、项目说明和轻量实验汇总。`.gitignore` 排除 **`docs/`、`data/`、`artifacts/`、模型、缓存、虚拟环境、编辑器状态与本地配置**；这些内容保留在本地或通过运行生成。

```text
main.py / run.ps1        统一入口与 llm 环境脚本
configs/                公共参数与停用词
gensim_lab/             数据、表示、分类、报告与源码索引
examples/               入门和模型查询示例
tests/                  数据与实验协议测试
results/                已完成实验的轻量 CSV、图表和来源信息
RESULTS.md              详细结果、参数与结论边界
requirements*.txt       依赖与版本锁定
```

## 参考资料

- [Gensim 官网](https://radimrehurek.com/gensim/)与[官方 FastText 教程](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html)
- [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
- [scikit-learn 数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [jieba 官方仓库](https://github.com/fxsjy/jieba)
