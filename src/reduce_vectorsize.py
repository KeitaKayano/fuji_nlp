from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

import util


def main():
    # Word2Vecモデルのロード
    model: KeyedVectors = util.load_model("word2vec_skip-gram_text8_300")

    # ベクトルの取得
    words = list(model.key_to_index.keys())
    word_vectors = np.array([model.get_vector(word) for word in words])
    print(word_vectors.shape)

    # 次元削減 (例: 300次元から50次元へ)
    pca = PCA(n_components=0.9)
    reduced_vectors = pca.fit_transform(word_vectors)
    print(reduced_vectors.shape)

    # 新しいベクトルを保存
    reduced_model = {}
    for word, vec in zip(words, reduced_vectors):
        reduced_model[word] = vec

    # データの読み込みと類似度計算
    df = pd.read_csv('wordsim353/combined.csv')
    df['cosSim'] = df.apply(lambda row: util.culcCosSim(row, reduced_model), axis=1)
    missing_word_count = df['cosSim'].isna().sum()  # NaNの数をカウント

    # 欠損値を含む行を削除
    df = df.dropna()

    # スピアマン相関係数を計算して表示
    print(df[['Human (mean)', 'cosSim']].corr(method='spearman'))

    # 見つからなかった単語の数を表示
    print(f"見つからなかった単語の数: {missing_word_count}")

if __name__ == "__main__":
    main()
