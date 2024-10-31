# コサイン類似度計算ツール

このリポジトリでは、事前学習済みの単語埋め込みモデルを利用して、単語ペア間のコサイン類似度を計算し、その結果をもとにスピアマン相関係数を求めるPythonスクリプトを提供します。

## 構成
- `src/main.py` : コサイン類似度計算とスピアマン相関係数を求めるメインスクリプト

## 必要なデータ
- **単語ペアデータ**  
  `wordsim353/combined.csv` というCSVファイルを使用します。このファイルには、比較する単語ペアとその「人間による類似度」が含まれています。
  
- **単語埋め込みモデル**  
  Gensimの`downloader`から指定したモデルを自動的にダウンロードし、計算に使用します。

## 事前準備
1. 必要なPythonライブラリをインストールしてください。  
   ```
   pip install -r requirements.txt
   ```
2. wordsim353/combined.csvをリポジトリのルートディレクトリに配置します。

## コマンドライン引数
--model : 使用する事前学習済みのモデル名を指定します。利用可能なモデル一覧は、`python src/main.py -h`で確認できます。

## 実行方法
```
python src/main.py --model word2vec-google-news-300
```
