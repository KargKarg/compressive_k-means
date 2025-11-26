import os, sys
import numpy as np
import presentation
from sklearn.cluster import KMeans
from typing import Callable
import json

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from clustering import *


SEED = 12
EXP = 5
np.random.seed(SEED)

METRICS: dict[str: dict[str: dict[str: list[float]]]] = {
        dataset: {
            model: {
            metric: [] for metric in presentation.METRICS
            } for model in ["KM", "CKM", "RANDOM"]
        }
        for dataset in presentation.datasets
}

for dataset in (data := presentation.datasets):

    output: tuple[np.ndarray[float], np.ndarray[int]] = data[dataset]["function"](**data[dataset]["params"])

    X: np.ndarray[float] = output[0]
    Y: np.ndarray[int] = output[1]

    N: int = X.shape[0]
    D: int = X.shape[1]
    K: int = int(max(np.unique(Y)) + 1)

    MODELS: dict[str: Callable] = {
        "KM": lambda s: KMeans(n_clusters=K, init="random", random_state=s, max_iter=3),
        "CKM": lambda s: CKM(K=K, m=N//2, lower=np.min(X), upper=np.max(X), iters=3, init_centroids=InitCentroids.SAMPLE, seed=s),
        "RANDOM": lambda s: presentation.RandomClustering(K=K, seed=s)
    }

    trained_models: list[KMeans, CKM, presentation.RandomClustering] = []

    for i in range(SEED, SEED+EXP):

        for name_model in MODELS:
            
            model = MODELS[name_model](i)

            model.fit(X)

            pred: np.ndarray[int] = match_labels(Y, model.predict(X))

            for metric in (metrics := presentation.true_pred_metrics):  METRICS[dataset][name_model][metric].append(metrics[metric]["function"](Y, pred))
            for metric in (metrics := presentation.pred_X_metrics): METRICS[dataset][name_model][metric].append(metrics[metric]["function"](X, pred))

            if i == SEED: trained_models.append(model)
    
    presentation.plot(X, Y, dataset, MODELS.keys(), trained_models)

    for metric_name in presentation.METRICS:    presentation.plot_single_metric_boxplot(METRICS[dataset], dataset, metric_name)

    for model in METRICS[dataset]:
        for metric in (values := METRICS[dataset][model]):   values[metric] = {"mean": np.mean(values[metric]), "std": np.std(values[metric])}

    with open("metrics.json", "w") as f:
        json.dump(METRICS, f, indent=4)