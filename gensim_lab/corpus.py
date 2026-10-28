"""人民日报语料解析：段落 -> 文章 -> 去重 -> 按日期划分。

默认沿用原始分词；只有预测新输入时才调用 jieba，避免把已有词边界再次打乱。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import re
import unicodedata

from .common import DATA, ROOT, LOG, checksum, download, write_csv, write_json, write_jsonl, read_jsonl

LINE = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})\s+(.*)$")
PINYIN = re.compile(r"\{[^{}]*\}")
WORD = re.compile(r"[\u3400-\u9fffA-Za-z0-9]+")
STOPWORDS = set((ROOT / "configs" / "stopwords.txt").read_text(encoding="utf-8").split())
LEE_URL = "https://raw.githubusercontent.com/piskvorky/gensim/4.4.0/gensim/test/test_data/lee_background.cor"


def clean_tokens(text: str, remove_stopwords: bool = True) -> list[str]:
    """输入已分词段落 str；输出 list[str]，保留汉字词边界，去拼音和标点。"""
    text = unicodedata.normalize("NFKC", PINYIN.sub("", text))
    result = []
    for original in text.split():
        # 兼容常见词性标注版；本次原文件没有 /n、/v 这类后缀。
        original = re.sub(r"/[a-z]+$", "", original)
        for token in WORD.findall(original):
            token = token.lower()
            if token.isdigit():
                continue
            if remove_stopwords and token in STOPWORDS:
                continue
            if token.isascii() and len(token) < 2:
                continue
            result.append(token)
    return result


def tokenize_new_text(text: str) -> list[str]:
    """新中文文本先 jieba 分词，再与训练语料采用相同清洗规则。"""
    import jieba
    jieba.dt.tmp_dir = str(ROOT / ".cache" / "jieba")
    jieba.setLogLevel(30)
    return clean_tokens(" ".join(jieba.cut(text)))


def parse_source(path: Path) -> tuple[list[dict], dict]:
    raw = path.read_bytes()  # 用户 C 盘输入只读；生成文件从不写回原目录。
    text = raw.decode("gb18030", errors="replace")
    groups = defaultdict(list)
    repairs, invalid = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if "\ufffd" in line:
            repairs.append({"line": number, "replacement_count": line.count("\ufffd"), "text": line})
        match = LINE.fullmatch(line.strip())
        if not match:
            if line.strip():
                invalid.append({"line": number, "text": line[:120]})
            continue
        date, page, article, paragraph, content = match.groups()
        groups[(date, page, article)].append((int(paragraph), content))
    articles = []
    for (date, page, article), paragraphs in sorted(groups.items()):
        paragraphs.sort()
        cleaned = [clean_tokens(content) for _, content in paragraphs]
        tokens = [word for paragraph in cleaned for word in paragraph]
        # 去重签名在停用词过滤之前生成，降低不同短文误合并的可能性。
        canonical = "".join(word for _, content in paragraphs for word in clean_tokens(content, False))
        articles.append({
            "id": f"{date}-{page}-{article}", "date": date, "month": date[:6], "page": page,
            "label": f"page_{page}", "title": PINYIN.sub("", paragraphs[0][1]),
            "tokens": tokens, "paragraphs": [p for p in cleaned if p],
            "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        })
    audit = {"path": str(path), "bytes": len(raw), "sha256": checksum(path), "encoding": "gb18030",
             "line_count": len(text.splitlines()), "raw_articles": len(articles),
             "replacement_characters": text.count("\ufffd"), "decode_repairs": repairs,
             "invalid_lines": invalid, "page_counts": dict(sorted(Counter(a["page"] for a in articles).items()))}
    return articles, audit


def simhash(tokens: list[str]) -> int:
    """64 位词二元组 SimHash；汉明距离 <= 3 只表示词面高度接近。"""
    features = Counter(zip(tokens, tokens[1:]))
    accum = [0] * 64
    for (a, b), weight in features.items():
        value = int.from_bytes(hashlib.blake2b((a + "\0" + b).encode("utf-8"), digest_size=8).digest(), "little")
        for bit in range(64):
            accum[bit] += weight if value & (1 << bit) else -weight
    return sum(1 << bit for bit, count in enumerate(accum) if count > 0)


def deduplicate(articles: list[dict]) -> tuple[list[dict], list[dict]]:
    """日期升序保留第一篇；四个 16 位桶保证距离 <= 3 的签名至少共用一桶。"""
    exact, bands, signatures = {}, defaultdict(list), {}
    kept, removed = [], []
    for article in sorted(articles, key=lambda a: a["id"]):
        digest = article["content_hash"]
        other, reason = exact.get(digest), "exact"
        signature = simhash(article["tokens"]) if len(article["tokens"]) >= 50 else None
        if other is None and signature is not None:
            candidates = {idx for bit in range(4) for idx in bands[(bit, (signature >> (16 * bit)) & 65535)]}
            for idx in sorted(candidates):
                if (signature ^ signatures[idx]).bit_count() <= 3:
                    other, reason = kept[idx]["id"], "simhash_hamming_le_3"
                    break
        if other is not None:
            removed.append({"removed_id": article["id"], "kept_id": other, "reason": reason})
            continue
        idx = len(kept)
        exact[digest] = article["id"]
        if signature is not None:
            signatures[idx] = signature
            for bit in range(4):
                bands[(bit, (signature >> (16 * bit)) & 65535)].append(idx)
        kept.append(article)
    return kept, removed


def assign_split(date: str, config: dict) -> str:
    if date <= config["train_end"]:
        return "train"
    if date <= config["validation_end"]:
        return "validation"
    return "test"


def prepare(config: dict, manual_labels: Path | None = None) -> dict:
    all_articles, source_stats = [], []
    for value in config["sources"]:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"找不到用户语料：{path}；请放入 data/raw，或在 configs/local.json 中覆盖 sources")
        articles, stats = parse_source(path)
        LOG.info("%s：%s 篇文章，%s 处解码替代字符", path.name, len(articles), stats["replacement_characters"])
        all_articles.extend(articles)
        source_stats.append(stats)
    filtered = [a for a in all_articles if len(a["tokens"]) >= config["min_article_tokens"]]
    unique, duplicates = deduplicate(filtered)
    january = [a for a in unique if a["month"] == "199801"]
    external = [a for a in unique if a["month"] in {"199802", "199803"}]
    overrides = {}
    if manual_labels:
        with manual_labels.open(encoding="utf-8-sig", newline="") as handle:
            overrides = {r["id"]: r["manual_topic"].strip() for r in csv.DictReader(handle) if r["manual_topic"].strip()}
        if not overrides:
            raise ValueError("人工标签表的 manual_topic 列为空")
        january = [a for a in january if a["id"] in overrides]
    for article in january:
        article["split"] = assign_split(article["date"], config)
        if overrides:
            article["label"] = overrides[article["id"]]
    write_jsonl(DATA / "processed" / "january.jsonl", january)
    write_jsonl(DATA / "processed" / "external_zh.jsonl", external)
    write_csv(DATA / "processed" / "duplicates.csv", duplicates)
    write_csv(DATA / "processed" / "manual_labels_template.csv", [
        {"id": a["id"], "date": a["date"], "page": a["page"], "title": a["title"], "manual_topic": ""} for a in january
    ])
    split_counts = {split: dict(sorted(Counter(a["label"] for a in january if a["split"] == split).items()))
                    for split in ["train", "validation", "test"]}
    stats = {"sources": source_stats, "raw_articles": len(all_articles), "short_articles_removed": len(all_articles) - len(filtered),
             "duplicates_removed": len(duplicates), "january_articles": len(january), "external_zh_articles": len(external),
             "january_tokens": sum(len(a["tokens"]) for a in january),
             "external_zh_tokens": sum(len(a["tokens"]) for a in external), "split_label_counts": split_counts,
             "label_type": "manual_topic" if overrides else "newspaper_page",
             "deduplication": "exact SHA256 + 64-bit token-bigram SimHash (Hamming <= 3); no guarantee of semantic deduplication"}
    write_json(DATA / "processed" / "data_audit.json", stats)
    LOG.info("清洗完成：1 月 %s 篇，2/3 月 %s 篇；去重 %s 篇", len(january), len(external), len(duplicates))
    return stats


def january_splits() -> dict[str, list[dict]]:
    rows = read_jsonl(DATA / "processed" / "january.jsonl")
    return {name: [a for a in rows if a["split"] == name] for name in ["train", "validation", "test"]}


def sentences(articles: list[dict]) -> list[list[str]]:
    # 训练局部上下文以段落为界，不把整月或整篇新闻拼成一个超长句子。
    return [paragraph for a in articles for paragraph in a["paragraphs"] if len(paragraph) >= 2]


def english_news() -> tuple[list[list[str]], dict]:
    """独立的 ABC 新闻 Lee corpus，官方教程使用的 300 篇英文新闻。"""
    path = download(LEE_URL, DATA / "raw" / "lee_background.cor")
    rows = [[w.lower() for w in re.findall(r"[A-Za-z]+", line) if len(w) > 1]
            for line in path.read_text(encoding="iso-8859-1").splitlines() if line.strip()]
    return rows, {"url": LEE_URL, "sha256": checksum(path), "documents": len(rows),
                  "tokens": sum(map(len, rows)), "description": "ABC/Lee English news; exploratory use only"}
