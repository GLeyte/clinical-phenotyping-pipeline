import sys
import os

DISCO_ROOT_PATH = "/export/share/pascalw777dm/DISCO"
sys.path.append(DISCO_ROOT_PATH)
os.environ["TZ"] = "Europe/Vienna"

from datasets.density_datasets import Datasets as DensityDatasets
from datasets.real_world_datasets import Datasets as RealWorldDatasets
from src.utils.metrics import create_and_rescale_df
from src.utils.experiments import cache, calc_eval_measures_for_multiple_datasets

import numpy as np
import pandas as pd

from src.Evaluation.dcdistances.dctree import DCTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score as ARI
from src.utils.metrics import SELECTED_METRICS


CACHE_PATH = "/export/share/pascalw777dm/DISCO/.cache/"


Datasets = [
    # DensityDatasets.aggregation,
    # DensityDatasets.chainlink,
    # DensityDatasets.complex8,
    # DensityDatasets.complex9,
    # DensityDatasets.diamond9,
    # DensityDatasets.compound,
    # DensityDatasets.dartboard1,
    # DensityDatasets.three_spiral,
    # DensityDatasets.smile1,
    RealWorldDatasets.COIL20,
    # RealWorldDatasets.Optdigits,
    # RealWorldDatasets.Pendigits,
]


def get_kcenter_and_dbscan_clusterings(X, ks):
    dctree = DCTree(X, min_points=5, min_points_mr=2)
    eps_list = [dctree.get_eps_for_k(k) for k in ks]
    kcenter_labels = [dctree.get_k_center(k) for k in ks]
    dbscan_labels = [DBSCAN(eps).fit(X).labels_ for eps in eps_list]
    return eps_list, kcenter_labels, dbscan_labels


def create_df_ari(ari_values, ks):
    df_ari = pd.DataFrame(ari_values, columns=["ARI"])
    df_ari["k"] = ks
    df_ari["measure"] = "ARI"
    return df_ari


def get_peak_positions(df, ks):
    cvi_scores = df.to_numpy()
    peak_positions = []
    all_values = []
    for metric in SELECTED_METRICS:
        values = cvi_scores[cvi_scores[:, 1] == metric][:, 3]
        all_values.append(values)
        # print(metric, values)
        max_idx = np.argmax(values)
        peak_positions.append([ks[max_idx], values[max_idx], max_idx])
    peak_positions = np.array(peak_positions)
    return peak_positions


def calc_ari_values(ground_truth_labels, labels_list, ks):
    return [(ARI(ground_truth_labels, labels_list[i])) for i, _ in enumerate(ks)]


def calculate_data(dataset, cached_name, ks):
    X, l = dataset.data_cached

    eps_list, kcenter_labels, dbscan_labels = get_kcenter_and_dbscan_clusterings(X, ks)
    kcenter_datasets = [[(X, kcenter_labels[i])] for i, k in enumerate(ks)]
    dbscan_datasets = [[(X, dbscan_labels[i])] for i, k in enumerate(ks)]

    if not os.path.exists(f"{CACHE_PATH}find_dbscan_eps##ARIs##{cached_name}.pkl"):
        print(f"Cached version '{CACHE_PATH}find_dbscan_eps##ARIs##{cached_name}.pkl' not found.")
    ari_values = cache(f"find_dbscan_eps##ARIs##{cached_name}", calc_ari_values, [l, dbscan_labels, ks], recalc=True)
    df_ari = create_df_ari(ari_values, ks)

    if not os.path.exists(f"{CACHE_PATH}find_dbscan_eps##CVIs##{cached_name}.pkl"):
        print(f"Cached version '{CACHE_PATH}find_dbscan_eps##CVIs##{cached_name}.pkl' not found.")
    eval_results = cache(
        f"find_dbscan_eps##CVIs##{cached_name}",
        calc_eval_measures_for_multiple_datasets,
        [dbscan_datasets, ks],
        {"n_jobs": 50},
        recalc=True,
    )

    df = create_and_rescale_df(eval_results)
    peak_positions = get_peak_positions(df, ks)

    return df_ari, df, ks, eps_list, peak_positions, ari_values, l, dbscan_labels


for dataset in Datasets:
    print(dataset.id, end=": ")
    ks = range(15,45+1)

    calculate_data(dataset, f"find_dbscan_eps##ARIs##{dataset.id}_15_45", ks)
