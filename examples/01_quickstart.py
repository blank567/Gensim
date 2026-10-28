"""五分钟理解 Gensim API；玩具语料只演示用法，不用于性能结论。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim_lab.common import load_config, setup
setup()  # 先设置 E 盘缓存，再导入依赖。
from gensim import corpora, matutils, models
from gensim.models.fasttext_inner import compute_ngrams_bytes
from gensim_lab.corpus import tokenize_new_text
from gensim_lab.representations import average_vectors, train_embedding


def main():
    # 每个内部列表是一篇已分词文本，长度可不同。
    texts = [
        ["中国", "经济", "增长", "市场"], ["金融", "市场", "经济", "投资"],
        ["企业", "投资", "经济", "发展"], ["科技", "企业", "研究", "发展"],
        ["体育", "比赛", "冠军", "球队"], ["运动员", "比赛", "体育", "训练"],
        ["学校", "教育", "学生", "学习"], ["学生", "科技", "学习", "研究"],
    ]
    dictionary = corpora.Dictionary(texts)  # V 个词对应 V 个整数 ID。
    bow = [dictionary.doc2bow(text) for text in texts]  # 每篇只保存非零的 (ID,词频)。
    tfidf = models.TfidfModel(bow, dictionary=dictionary)
    matrix = matutils.corpus2csc(tfidf[bow], num_terms=len(dictionary)).T.tocsr()
    print("词典:", dictionary.token2id)
    print("第一篇 BoW:", bow[0])
    print("TF-IDF 文档矩阵 [N,V]:", matrix.shape)
    config = dict(load_config(), vector_size=16, min_count=1, epochs=10, bucket=1000)
    model, _ = train_embedding(texts * 5, config, "fasttext")
    print("词向量矩阵 [V,d]:", model.wv.vectors.shape)
    print("子词矩阵 [B,d]:", model.wv.vectors_ngrams.shape)
    print("文档平均向量矩阵 [N,d]:", average_vectors(texts, model.wv).shape)
    new_word = "经济发展新词"
    print("OOV 在词典内:", new_word in model.wv.key_to_index)
    print("OOV 可以合成:", new_word in model.wv)
    print("OOV 向量 [d]:", model.wv[new_word].shape)
    print("中国 2~4 字符片段:", [item.decode("utf-8") for item in compute_ngrams_bytes("中国", 2, 4)])
    print("jieba 新输入分词:", tokenize_new_text("中国经济持续发展，金融市场受到关注。"))
    print("玩具语料没有可靠的语义评测意义；正式结果请查看 artifacts/EXPERIMENT_REPORT.md。")


if __name__ == "__main__":
    main()
