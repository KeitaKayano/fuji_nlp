import pandas as pd
import numpy as np
from gensim import downloader
from tqdm import tqdm
import argparse
import visualize_corr
import util


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
    model = util.load_model(args.model)

    missing_word_count = 0

    # データの読み込みと類似度計算
    tqdm.pandas()
    df = pd.read_csv('wordsim353/combined.csv')
    df['cosSim'] = df.apply(lambda row: util.culcCosSim(row, model), axis=1)
    missing_word_count = df['cosSim'].isna().sum()  # NaNの数をカウント

    print(df[1:6])

    # 欠損値を含む行を削除
    df = df.dropna()

    visualize_corr.plot_correlation(df['Human (mean)'], df['cosSim'], "Scatter Plot of Human (mean) vs cosSim", f"/workspace/figures/scatter_plot_{args.model}.png", separate_color=153, X_label='Human(mean)', Y_label='cosSim')

    # スピアマン相関係数を計算して表示
    print(df[['Human (mean)', 'cosSim']].corr(method='spearman'))

    # 見つからなかった単語の数を表示
    print(f"見つからなかった単語の数: {missing_word_count}")

if __name__ == "__main__":
    main()
