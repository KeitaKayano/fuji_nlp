import pandas as pd
import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import visualize_corr



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

# メイン処理
def main():
    # Gensimで利用可能なモデルのリストを取得
    available_models = list(downloader.info()['models'])
    model_list_str = "\n".join(available_models)

    # コマンドライン引数をパース
    parser = argparse.ArgumentParser(
        description="指定したモデルを使ってコサイン類似度を計算します\n"
                    "利用可能なモデル一覧:\n" + model_list_str
    )
    parser.add_argument("--model", type=str, required=True, help="使用する事前学習済みモデル名")
    args = parser.parse_args()

    # モデルをダウンロードしてロード
    model = load_model(args.model)

    missing_word_count = 0

    # データの読み込みと類似度計算
    tqdm.pandas()
    df = pd.read_csv('wordsim353/combined.csv')
    df['cosSim'] = df.apply(lambda row: culcCosSim(row, model), axis=1)
    missing_word_count = df['cosSim'].isna().sum()  # NaNの数をカウント

    print(df[1:6])

    # 欠損値を含む行を削除
    df = df.dropna()

    visualize_corr.plot_correlation(df['Human (mean)'], df['cosSim'], "Scatter Plot of Human (mean) vs cosSim", f"/workspace/figures/scatter_plot_{args.model}.png")

    # スピアマン相関係数を計算して表示
    print(df[['Human (mean)', 'cosSim']].corr(method='spearman'))

    # 見つからなかった単語の数を表示
    print(f"見つからなかった単語の数: {missing_word_count}")

if __name__ == "__main__":
    main()
