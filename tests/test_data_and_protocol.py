"""测试最容易让课程实验结论失真的边界：文章粒度、清洗、去重与数据泄漏。"""
from copy import deepcopy

import numpy as np
import pytest

from gensim_lab.common import load_config, setup
setup()
from gensim_lab.corpus import clean_tokens, deduplicate, parse_source
from gensim_lab.classification import check_split_integrity, select_c
from gensim_lab.representations import FeatureSpace, average_vectors, train_embedding


def test_paragraphs_are_grouped_as_articles_and_sorted(tmp_path):
    path = tmp_path / "news.txt"
    path.write_bytes(("19980101-01-001-002 中国 经济 增长\n"
                      "19980101-01-002-001 体育 比赛 新闻\n"
                      "19980101-01-001-001 新年 标题\n").encode("gb18030"))
    articles, audit = parse_source(path)
    assert len(articles) == 2
    assert articles[0]["title"] == "新年 标题"
    assert articles[0]["tokens"] == ["新年", "标题", "中国", "经济", "增长"]
    assert audit["raw_articles"] == 2 and audit["invalid_lines"] == []


def test_pinyin_and_punctuation_do_not_enter_features():
    assert clean_tokens("在/p 中国/ns 上{shang5} ＡＢＣ ， １９９８ 经济/n") == ["中国", "abc", "经济"]


def test_decode_error_is_visible_in_audit(tmp_path):
    path = tmp_path / "broken.txt"
    path.write_bytes(b"19980323-09-005-012 " + "北京 ".encode("gbk") + b"\xa3 \n")
    _, audit = parse_source(path)
    assert audit["replacement_characters"] == 1
    assert audit["decode_repairs"][0]["line"] == 1


def test_dedup_keeps_earliest_and_detects_near_copy():
    tokens = [f"词{i}" for i in range(80)]
    rows = [
        {"id": "19980102-01-001", "content_hash": "different", "tokens": tokens},
        {"id": "19980101-01-001", "content_hash": "first", "tokens": tokens},
        {"id": "19980103-01-001", "content_hash": "first", "tokens": tokens},
    ]
    kept, removed = deduplicate(rows)
    assert [r["id"] for r in kept] == ["19980101-01-001"]
    assert {r["reason"] for r in removed} == {"exact", "simhash_hamming_le_3"}


def test_split_integrity_rejects_duplicate_content():
    splits = {"train": [{"id": "a", "content_hash": "same"}],
              "validation": [{"id": "b", "content_hash": "different"}],
              "test": [{"id": "c", "content_hash": "same"}]}
    with pytest.raises(AssertionError, match="重复文章"):
        check_split_integrity(splits)


def test_heldout_words_cannot_change_dictionary_or_idf():
    config = dict(load_config(), min_df=1, max_vocab=100, lsi_topics=2, lda_topics=2, lda_passes=1)
    train = [["经济", "发展", "市场"], ["体育", "比赛", "胜利"], ["教育", "学校", "学习"], ["科技", "研究", "科学"]]
    space = FeatureSpace(config)
    space.fit_topics(train)
    vocabulary = dict(space.dictionary.token2id)
    idf = dict(space.tfidf.idfs)
    for method in ["bow", "tfidf", "lsi", "lda"]:
        x = space.transform([["从未见过的测试专用词"]], method)
        assert x.shape[0] == 1
        assert np.isfinite(x.toarray() if hasattr(x, "toarray") else x).all()
    assert space.dictionary.token2id == vocabulary
    assert space.tfidf.idfs == idf
    assert "从未见过的测试专用词" not in vocabulary


def test_fasttext_can_construct_oov_and_survives_save_load(tmp_path):
    from gensim.models import FastText
    config = dict(load_config(), vector_size=8, bucket=100, min_count=1, epochs=2)
    corpus = [["中国", "经济", "发展"], ["体育", "比赛", "冠军"]] * 4
    ft, _ = train_embedding(corpus, config, "fasttext")
    unknown = "中国经济新词"
    assert unknown not in ft.wv.key_to_index and unknown in ft.wv
    before = average_vectors([[unknown]], ft.wv)
    path = tmp_path / "ft.gensim"
    ft.save(str(path))
    restored = FastText.load(str(path))
    assert np.isfinite(before).all() and np.linalg.norm(before) > 0
    np.testing.assert_allclose(before, average_vectors([[unknown]], restored.wv), rtol=1e-6)


def test_classifier_selection_receives_only_training_and_validation():
    config = dict(load_config(), svm_c_grid=[0.1, 1.0])
    train_x = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    train_y = np.array(["a", "a", "b", "b"])
    model, c, score = select_c(train_x, train_y, train_x, train_y, config)
    assert c in config["svm_c_grid"] and score == 1.0
    assert list(model.predict([[1.0, 0], [0, 1.0]])) == ["a", "b"]


def test_local_config_preserves_private_paths_without_overriding_custom_config(tmp_path, monkeypatch):
    from gensim_lab import common
    configs = tmp_path / "configs"
    configs.mkdir()
    common.write_json(configs / "default.json", {"sources": ["data/raw/news.txt"], "seed": 42})
    common.write_json(configs / "local.json", {"sources": ["E:/private/news.txt"]})
    common.write_json(configs / "topics.json", {"sources": ["data/raw/topics.txt"], "seed": 7})
    monkeypatch.setattr(common, "ROOT", tmp_path)
    assert common.load_config() == {"sources": ["E:/private/news.txt"], "seed": 42}
    assert common.load_config(configs / "default.json")["sources"] == ["E:/private/news.txt"]
    assert common.load_config(configs / "topics.json") == {"sources": ["data/raw/topics.txt"], "seed": 7}
