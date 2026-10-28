"""加载正式训练的 FastText：查看 IV/OOV、近邻和文章表示。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim_lab.common import OUT, setup
setup()
from gensim.models import FastText
from gensim.models.fasttext_inner import compute_ngrams_bytes
from gensim_lab.corpus import tokenize_new_text
from gensim_lab.representations import average_vectors


def main():
    model = FastText.load(str(OUT / "models" / "classification" / "fasttext.gensim"))
    for word in ["经济", "体育", "人工智能技术平台"]:
        print("\n词:", word, "词表内:", word in model.wv.key_to_index, "可查询:", word in model.wv)
        print("向量 shape:", model.wv[word].shape)
        print("最近邻:", model.wv.most_similar(word, topn=5))
        print("前十个子词:", [piece.decode("utf-8") for piece in compute_ngrams_bytes(word, model.wv.min_n, model.wv.max_n)[:10]])
    tokens = tokenize_new_text("中国队在比赛中赢得冠军，运动员积极备战奥运会。")
    print("\n新新闻分词:", tokens)
    print("文章表示 [1,d]:", average_vectors([tokens], model.wv).shape)
    print("可以构造向量不代表语义正确，特别是未训练充分的 OOV；请结合下游结果判断。")


if __name__ == "__main__":
    main()
