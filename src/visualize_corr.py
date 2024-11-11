import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr


# プロットを描画する関数
def plot_correlation(x, y, title, save_path, separate_color=0, X_label='X', Y_label='Y'):
    # スピアマン相関係数とピアソン相関係数を計算
    spearman_corr, _ = spearmanr(x, y)
    pearson_corr, _ = pearsonr(x, y)

    # 散布図の描画
    plt.figure(figsize=(10, 6))
    if separate_color > 0:
        sns.scatterplot(x=x[:separate_color], y=y[:separate_color], color='blue', alpha=0.6, edgecolor='w', s=60, label='set 1')
        sns.scatterplot(x=x[separate_color:], y=y[separate_color:], color='red', alpha=0.6, edgecolor='w', s=60, label='set 2')
    else:
        sns.scatterplot(x=x, y=y, color='blue', alpha=0.6, edgecolor='w', s=60)

    # 相関係数をプロット上に表示
    plt.title(title, fontsize=15)
    plt.xlabel(X_label, fontsize=12)
    plt.ylabel(Y_label, fontsize=12)
    plt.text(0.05, 0.95, f"Spearman: {spearman_corr:.2f}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=12, color='purple')
    plt.text(0.05, 0.90, f"Pearson: {pearson_corr:.2f}", ha='left', va='center', transform=plt.gca().transAxes, fontsize=12, color='green')

    # 画像の保存
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    # データセットを生成
    # 1. 一般的な楕円分布（分散を変える）
    np.random.seed(0)
    x1 = np.random.normal(0, 1, 100)
    y1 = 0.5 * x1 + np.random.normal(0, 0.5, 100)

    x2 = np.random.normal(0, 1, 100)
    y2 = 2 * x2 + np.random.normal(0, 2, 100)

    # 2. 単調増加する多変数関数
    x3 = np.linspace(0, 10, 100)
    y3 = np.log(x3 + 1)

    # 3. 外れ値を含む分布
    x4 = np.random.normal(0, 1, 100)
    y4 = x4 + np.random.normal(0, 0.5, 100)
    y4[-5:] = y4[-5:] + 10  # 外れ値を追加

    # プロットを作成
    plot_correlation(x1, y1, "Ellipse Distribution (Low Variance)", "/workspace/figures/ellipse_low_variance.png")
    plot_correlation(x2, y2, "Ellipse Distribution (High Variance)", "/workspace/figures/ellipse_high_variance.png")
    plot_correlation(x3, y3, "Monotonically Increasing Function", "/workspace/figures/monotonic_function.png")
    plot_correlation(x4, y4, "Distribution with Outliers", "/workspace/figures/distribution_with_outliers.png")


if __name__ == "__main__":
    main()
