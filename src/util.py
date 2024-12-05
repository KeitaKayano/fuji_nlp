import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors
import os

# コサイン類似度を計算する関数
def cosSim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 各行の単語ペアに対してコサイン類似度を計算する関数
def culcCosSim(row, model):
    try:
        # 単語ベクトルの取得
        w1v = model[row['Word 1']]
        w2v = model[row['Word 2']]
        # コサイン類似度の計算
        return cosSim(w1v, w2v)
    except KeyError as e:
        # 単語がモデルに存在しない場合にスキップし、単語を表示
        print(f"単語が見つかりません: {e}")
        return np.nan  # 存在しない単語の場合はNaNを返す

# モデルをロードする関数
def load_model(model_name):
    # Gensimで利用可能なモデルのリストを取得
    available_models = downloader.info()['models']

    # Gensimのモデルリストにある場合はロード
    if model_name in available_models:
        print(f"Gensimからモデル '{model_name}' をロードします。")
        return downloader.load(model_name)
    else:
        # custom_modelディレクトリにファイルが存在するかチェック
        model_path = f"/workspace/src/custom_model/{model_name}.bin"
        if os.path.exists(model_path):
            print(f"custom_modelディレクトリからモデル '{model_name}' をロードします。")
            return KeyedVectors.load_word2vec_format(model_path, binary=True)
        else:
            raise ValueError(f"指定されたモデル '{model_name}' がGensimにもcustom_modelディレクトリにも存在しません。")
