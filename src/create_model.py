import gensim.downloader as api
from gensim.models import Word2Vec

dataset_name = "text8"

def main():
    # データセットのダウンロードと読み込み
    dataset = api.load(dataset_name)

    # Word2Vecのトレーニングのためにデータセットをリストに変換
    # このデータセットは文書単位で提供されているため、各文書をトークン化してリストに格納
    # sentences = [doc.split() for doc in dataset]

    # for vector_size in [10, 50, 100, 200, 300, 500, 700, 1000]:
    #     # CBOWモデルの作成とトレーニング
    #     model = Word2Vec(sentences=dataset, vector_size=vector_size, sg=0)

    #     # モデルの保存
    #     model.wv.save_word2vec_format(f"/workspace/src/custom_model/word2vec_CBOW_wiki_{vector_size}.bin", binary=True)

    for vector_size in [10, 50, 100, 200, 300, 500]:
        # skip-gramモデルの作成とトレーニング
        model = Word2Vec(sentences=dataset, vector_size=vector_size, sg=1)

        # モデルの保存
        model.wv.save_word2vec_format(f"/workspace/src/custom_model/word2vec_skip-gram_{dataset_name}_{vector_size}.bin", binary=True)

if __name__ == "__main__":
    main()
