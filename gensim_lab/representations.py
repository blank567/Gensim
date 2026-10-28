"""BoW、TF-IDF、LSI、LDA、Word2Vec 与 FastText 的统一表示接口。"""
from __future__ import annotations

from collections import Counter
import gc
from pathlib import Path
import time

from gensim import corpora, matutils, models
import numpy as np
from sklearn.preprocessing import normalize

from .common import DATA, OUT, LOG, read_json, read_jsonl, stable_hash, write_json
from .corpus import english_news, sentences


def train_embedding(training_sentences: list[list[str]], config: dict, kind: str, variant: dict | None = None,
                    seed: int | None = None):
    """输入 list[list[str]]；输出 V×d 的词向量模型，FastText 另有 B×d 子词矩阵。"""
    kwargs = {key: config[key] for key in ["vector_size", "window", "min_count", "workers", "epochs", "negative", "ns_exponent", "sample"]}
    kwargs.update(seed=config["seed"] if seed is None else seed, sg=1, hs=0, hashfxn=stable_hash,
                  alpha=0.025, min_alpha=0.0001, shrink_windows=True)
    if kind == "fasttext":
        kwargs.update(bucket=config["bucket"], min_n=2, max_n=4)
        kwargs.update({k: v for k, v in (variant or {}).items() if k != "name"})
        model = models.FastText(**kwargs)
    else:
        model = models.Word2Vec(**kwargs)
    start = time.perf_counter()
    model.build_vocab(corpus_iterable=training_sentences)
    if not model.wv.key_to_index:
        raise ValueError("词表为空：请增加语料或降低 min_count")
    model.train(corpus_iterable=training_sentences, total_examples=model.corpus_count, epochs=model.epochs)
    elapsed = time.perf_counter() - start
    LOG.info("%s %s：V=%s，d=%s，训练 %.1fs", kind, (variant or {}).get("name", ""), len(model.wv), model.vector_size, elapsed)
    return model, {"training_seconds": elapsed, "vocabulary": len(model.wv), "sentences": len(training_sentences),
                   "tokens": sum(map(len, training_sentences)), "vector_size": model.vector_size,
                   "seed": kwargs["seed"], "workers": kwargs["workers"], "epochs": kwargs["epochs"],
                   "sg": model.sg, "min_n": getattr(model.wv, "min_n", None), "max_n": getattr(model.wv, "max_n", None),
                   "bucket": getattr(model.wv, "bucket", None)}


def average_vectors(documents: list[list[str]], keyed_vectors, idf: dict[str, float] | None = None) -> np.ndarray:
    """词向量 [len(doc),d] -> 加权平均 [d] -> 文档矩阵 [N,d] -> L2 归一化。

    IDF 未见词使用训练集最大 IDF；这是一项固定规则，不在验证/测试集重算 IDF。
    """
    result = np.zeros((len(documents), keyed_vectors.vector_size), dtype=np.float32)
    unknown_weight = max(idf.values(), default=1.0) if idf is not None else 1.0
    for row, tokens in enumerate(documents):
        total = 0.0
        for word, count in Counter(tokens).items():
            # 对 Word2Vec 是词典查询，对 FastText 允许用字符子词合成 OOV。
            if word not in keyed_vectors:
                continue
            weight = count * (idf.get(word, unknown_weight) if idf is not None else 1.0)
            if weight > 0:
                result[row] += keyed_vectors[word] * weight
                total += weight
        if total:
            result[row] /= total
    return normalize(result, norm="l2", copy=False)


def coverage(documents: list[list[str]], keyed_vectors) -> dict:
    counts = Counter(word for doc in documents for word in doc)
    total = sum(counts.values())
    invocab = sum(count for word, count in counts.items() if word in keyed_vectors.key_to_index)
    constructible = sum(count for word, count in counts.items() if word in keyed_vectors)
    return {"token_count": total, "in_vocabulary_rate": invocab / max(total, 1),
            "vector_constructible_rate": constructible / max(total, 1),
            "oov_types": sum(word not in keyed_vectors.key_to_index for word in counts)}


class FeatureSpace:
    """所有 fit 只接收训练文档；transform 可接收任意新文档。"""

    def __init__(self, config: dict):
        self.config = config
        self.dictionary = self.tfidf = self.lsi = self.lda = None
        self.word2vec = self.fasttext = None

    def fit_topics(self, documents: list[list[str]]) -> None:
        start = time.perf_counter()
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.filter_extremes(no_below=self.config["min_df"], no_above=0.8, keep_n=self.config["max_vocab"])
        if len(self.dictionary) < 2:
            raise ValueError("主题模型至少需要两个有效词")
        bow = [self.dictionary.doc2bow(doc) for doc in documents]  # 每篇是 [(词ID,计数), ...]。
        self.tfidf = models.TfidfModel(bow, dictionary=self.dictionary, normalize=True)
        rank = min(self.config["lsi_topics"], len(self.dictionary) - 1, len(documents) - 1)
        self.lsi = models.LsiModel(self.tfidf[bow], id2word=self.dictionary, num_topics=rank,
                                   random_seed=self.config["seed"], power_iters=2)
        self.lda = models.LdaModel(bow, id2word=self.dictionary, num_topics=self.config["lda_topics"],
                                   passes=self.config["lda_passes"], iterations=50, chunksize=256,
                                   random_state=self.config["seed"], eval_every=None, minimum_probability=0.0)
        LOG.info("BoW/TF-IDF/LSI/LDA 已训练：词表=%s，用时 %.1fs", len(self.dictionary), time.perf_counter() - start)

    def idf_by_word(self) -> dict[str, float]:
        return {self.dictionary[idx]: float(weight) for idx, weight in self.tfidf.idfs.items()}

    def transform(self, documents: list[list[str]], method: str):
        if method in {"word2vec_mean", "fasttext_mean", "fasttext_tfidf"}:
            model = self.word2vec if method == "word2vec_mean" else self.fasttext
            weights = self.idf_by_word() if method == "fasttext_tfidf" else None
            return average_vectors(documents, model.wv, weights)
        bow = [self.dictionary.doc2bow(doc) for doc in documents]
        if method == "bow":
            matrix = matutils.corpus2csc(bow, num_terms=len(self.dictionary), dtype=np.float32).T.tocsr()
        elif method == "tfidf":
            matrix = matutils.corpus2csc(self.tfidf[bow], num_terms=len(self.dictionary), dtype=np.float32).T.tocsr()
        elif method == "lsi":
            matrix = matutils.corpus2dense(self.lsi[self.tfidf[bow]], num_terms=self.lsi.num_topics).T
        elif method == "lda":
            distributions = [self.lda.get_document_topics(doc, minimum_probability=0.0) for doc in bow]
            matrix = matutils.corpus2dense(distributions, num_terms=self.lda.num_topics).T
        else:
            raise ValueError(f"未知表示方法：{method}")
        return normalize(matrix, norm="l2", copy=False)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.dictionary.save(str(path / "dictionary.gensim"))
        for name in ["tfidf", "lsi", "lda", "word2vec", "fasttext"]:
            model = getattr(self, name)
            if model is not None:
                model.save(str(path / f"{name}.gensim"))
        write_json(path / "config.json", self.config)

    @classmethod
    def load(cls, path: Path):
        space = cls(read_json(path / "config.json"))
        space.dictionary = corpora.Dictionary.load(str(path / "dictionary.gensim"))
        for name, loader in [("tfidf", models.TfidfModel), ("lsi", models.LsiModel), ("lda", models.LdaModel),
                             ("word2vec", models.Word2Vec), ("fasttext", models.FastText)]:
            file = path / f"{name}.gensim"
            if file.exists():
                setattr(space, name, loader.load(str(file)))
        return space


def vector_probes(model, words: list[str]) -> dict:
    result = {}
    for word in words:
        known = word in model.wv.key_to_index
        obtainable = word in model.wv
        result[word] = {"in_vocabulary": known, "constructible": obtainable,
                        "vector_norm": float(np.linalg.norm(model.wv[word])) if obtainable else None,
                        "nearest": [[w, float(score)] for w, score in model.wv.most_similar(word, topn=5)] if obtainable else []}
    return result


def explore(config: dict) -> dict:
    """独立探索集可以全量训练，但这些模型绝不进入 1 月分类的特征管线。"""
    english, english_meta = english_news()
    quick_config = dict(config, epochs=5, min_count=2, lda_topics=8, lda_passes=3)
    space = FeatureSpace(quick_config)
    space.fit_topics(english)
    space.word2vec, w2v_stats = train_embedding(english, quick_config, "word2vec")
    space.fasttext, ft_stats = train_embedding(english, quick_config, "fasttext", {"min_n": 3, "max_n": 6, "sg": 1})
    query = ["government", "election", "minister"]
    retrieval = {}
    for method in ["bow", "tfidf", "lsi", "lda", "word2vec_mean", "fasttext_mean", "fasttext_tfidf"]:
        matrix = space.transform(english, method)
        q = space.transform([query], method)
        products = matrix @ q.T
        scores = products.toarray().ravel() if hasattr(products, "toarray") else np.asarray(products).ravel()
        retrieval[method] = [{"document": int(i), "cosine": float(scores[i]), "snippet": " ".join(english[i][:28])}
                             for i in np.argsort(-scores)[:3]]
    result = {"english_corpus": english_meta, "query": query, "retrieval": retrieval,
              "english_word2vec_training": w2v_stats, "english_fasttext_training": ft_stats,
              "english_word2vec": vector_probes(space.word2vec, ["government", "market", "governmentsuperpower"]),
              "english_fasttext": vector_probes(space.fasttext, ["government", "market", "governmentsuperpower"]),
              "english_lda_topics": [{"topic": i, "words": [w for w, _ in space.lda.show_topic(i, 8)]} for i in range(space.lda.num_topics)]}
    space.save(OUT / "models" / "exploration_en")
    del space
    gc.collect()
    external = read_jsonl(DATA / "processed" / "external_zh.jsonl")
    # 2/3 月是用户提供的额外中文新闻；不把后续月份混入 1 月分类训练。
    chinese, zh_stats = train_embedding(sentences(external), dict(config, epochs=5), "fasttext")
    result["chinese_external_training"] = zh_stats
    result["chinese_external_fasttext"] = vector_probes(chinese, ["经济", "体育", "金融", "人工智能技术平台"])
    chinese.save(str(OUT / "models" / "external_zh_fasttext.gensim"))
    write_json(OUT / "exploration.json", result)
    LOG.info("中英文表示探索完成，已保存近邻、OOV、主题和相似新闻结果")
    return result
