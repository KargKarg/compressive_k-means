from sklearn.datasets import (
    make_blobs,
    load_digits,
    load_iris
)

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score
)

import matplotlib.pyplot as plt
import numpy as np


N = 1000
D = 2
K = 10
SEED = 12


datasets = {
    "blobs": {
        "function": make_blobs,
        "params": {
            "n_samples": N,
            "n_features": D,
            "centers": K,
            "random_state": SEED
        }
    },

    "digits": {
        "function": load_digits,
        "params": {
            "n_class": K,
            "return_X_y": True
        }
    },

    "iris": {
        "function": load_iris,
        "params": {
            "return_X_y": True
        }
    }
}


pred_X_metrics = {
    "Silhouette": {
        "function": silhouette_score
    },

    "Davies-Bouldin": {
        "function": davies_bouldin_score
    }
}


true_pred_metrics = {
    "Adjusted Rand Index": {
        "function": adjusted_rand_score
    },

    "Adjusted Mutual info": {
        "function": adjusted_mutual_info_score
    },

    "Homogeneity": {
        "function": homogeneity_score
    },

    "Completeness": {
        "function": completeness_score
    },

    "V-measure": {
        "function": v_measure_score
    },

    "Fowlkes-Mallows": {
        "function": fowlkes_mallows_score
    }
}

METRICS = list(true_pred_metrics.keys()) + list(pred_X_metrics.keys())

def plot(X, Y, dataset_name, names, models) -> None:
    """
    Docstring for plot
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[:, 0], X[:, 1], 
        c=Y,
        cmap="tab10",
        s=20,
        alpha=0.6,
    )

    colors = ["blue", "red", "green"]
    markers = ["X", "P", "o"]

    for name, model, color, marker in zip(names, models, colors, markers):

        plt.scatter(
            model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c=color,
            s=200,
            marker=marker,
            edgecolor="black",
            linewidth=1.5,
            label=name
        )

    plt.title(f"Centroids - dataset: {dataset_name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figs/centroids_{dataset_name}.png")
    plt.close()

    return None


def plot_single_metric_boxplot(metrics_dict: dict[str, dict[str, list[float]]], dataset_name: str, metric_name: str) -> None:
    """
    """
    _, ax = plt.subplots(figsize=(6, 6))

    positions = [0, 1, 2]

    data = [metrics_dict[model][metric_name] for model in metrics_dict]

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    bp["boxes"][0].set(facecolor="red", alpha=0.6)
    bp["boxes"][1].set(facecolor="blue", alpha=0.6)
    bp["boxes"][2].set(facecolor="grey", alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(metrics_dict.keys()))
    ax.set_title(f"{metric_name} — {dataset_name}")
    ax.set_ylabel(metric_name)

    plt.tight_layout()
    plt.savefig(f"figs/metrics_{dataset_name}_{metric_name}.png")
    plt.close()

    return None


class RandomClustering(object):
    """
    Docstring for RandomClustering
    """


    def __init__(self, K: int, seed: int) -> None:
        """
        Docstring for __init__
        """
        self.K: int = K
        self.seed: int = seed
        return None
    

    def fit(self, X: np.ndarray[float]) -> None:
        """
        Docstring for fit
        """
        self.cluster_centers_ = np.random.uniform(low=X.min(), high=X.max(), size=(self.K, X.shape[1]))
        return None
    

    def predict(self, X: np.ndarray[float]) -> np.ndarray[int]:
        """
        Docstring for predict
        """
        return np.argmin(a=np.linalg.norm(x=(X[:, None, :] - self.cluster_centers_), axis=2), axis=1)