import matplotlib.pyplot as plt

def plot_combined_spearman_vs_dimensions(dimensions, cbow_coefficients, skipgram_coefficients, output_file='combined_spearman_plot.png'):
    """
    単語ベクトルの次元数とスピアマン係数を、CBOWとSkip-gramの両方で1つのプロットに表示し、画像として保存する関数。

    Parameters:
    dimensions (list or array): 単語ベクトルの次元数のリスト。
    cbow_coefficients (list or array): CBOWのスピアマン係数のリスト。
    skipgram_coefficients (list or array): Skip-gramのスピアマン係数のリスト。
    output_file (str): 保存する画像ファイルの名前。デフォルトは 'combined_spearman_plot.png'。

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, cbow_coefficients, marker='o', linestyle='-', color='b', label='CBOW')
    plt.plot(dimensions, skipgram_coefficients, marker='s', linestyle='--', color='r', label='Skip-gram')

    plt.title('Spearman Correlation vs Word Vector Dimensions (CBOW & Skip-gram)')
    plt.xlabel('Word Vector Dimensions')
    plt.ylabel('Spearman Correlation')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Combined plot saved as {output_file}")


def main():
    # 単語ベクトルの次元数とスピアマン係数のリスト
    dimensions = [10, 50, 100, 200, 300, 500, 700, 1000]
    cbow_spearman_coefficients = [0.470771, 0.599524, 0.624657, 0.632473, 0.620522, 0.622779, 0.624177, 0.62555]
    skipgram_spearman_coefficients = [0.457659, 0.652793, 0.666129, 0.67224, 0.674722, 0.655788, 0.656167, 0.652508]

    # プロットを作成
    plot_combined_spearman_vs_dimensions(dimensions, cbow_spearman_coefficients, skipgram_spearman_coefficients, output_file='/workspace/figures/combined_spearman_plot.png')

if __name__ == "__main__":
    main()
