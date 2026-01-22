# load javascript library from a server instead of isntalling it. Makes jupyter notebook more lightweight
# py.init_notebook_mode(connected=True)
import matplotlib
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD

matplotlib.use("Agg")
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import *
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.manifold import TSNE
from wordcloud import STOPWORDS, WordCloud

from enum import Enum
def display_cloud(df: pd.DataFrame, img_path: Path):
    plt.subplots(figsize=(10, 10))
    mask = np.zeros((600, 400, 4))
    wc = WordCloud(
        stopwords=STOPWORDS,
        mask=mask,
        background_color="white",
        contour_width=2,
        max_words=2000,
        max_font_size=256,
        random_state=42,
        width=mask.shape[1],
        height=mask.shape[0],
    )

    wc.generate(" ".join(df.text))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_path)


class DimRed(Enum):
    TRUNCATED_SVD = "truncated_svd"
    TSNE = "tsne"
    PCA = "pca"


def dimen_reduc_plot(test_data, test_label: list[str], option: DimRed):

    match option:
        
        case DimRed.TRUNCATED_SVD:
            print("svd")
            tsvd = TruncatedSVD(n_components=2, algorithm="randomized", random_state=42)
            tsvd_result = tsvd.fit_transform(test_data)
            plt.figure(figsize=(10, 8))
            colors = ["orange", "red"]

            sns.scatterplot(x=tsvd_result[:, 0], y=tsvd_result[:, 1], hue=test_label)

            plt.show()
            plt.figure(figsize=(10, 10))
            plt.scatter(
                tsvd_result[:, 0],
                tsvd_result[:, 1],
                c=test_label,
                cmap=matplotlib.colors.ListedColormap(colors),
            )
            color_red = mpatches.Patch(color="red", label="Negative Review")
            color_orange = mpatches.Patch(color="orange", label="Positive Review")
            plt.legend(handles=[color_orange, color_red])
            plt.title("TSVD")
            plt.savefig("")
        case DimRed.TSNE:
            print("tsne")
            tsne = TSNE(
                n_components=2, random_state=42
            )  # not recommended instead use PCA

            tsne_result = tsne.fit_transform(test_data)
            plt.figure(figsize=(10, 8))
            colors = ["orange", "red"]
            sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=test_label)

        case DimRed.PCA:
            print("pca")
            pca = SparsePCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(test_data.toarray())
            plt.figure(figsize=(10, 8))
            colors = ["orange", "red"]
            sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=test_label)
        
        case _:
            assert_never(option)
    plt.savefig(option.name)
    plt.close()
