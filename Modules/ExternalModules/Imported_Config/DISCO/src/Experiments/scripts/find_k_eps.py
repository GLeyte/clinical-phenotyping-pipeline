import sys
import time

DISCO_ROOT_PATH = "/export/share/pascalw777dm/DISCO"
sys.path.append(DISCO_ROOT_PATH)

from datasets.real_world_datasets import Datasets as RealWorldDatasets
from datasets.density_datasets import Datasets as DensityDatasets
from src.utils.metrics import METRICS
from src.Experiments.scripts._calc_multiple_experiments import run_multiple_experiments


RESULTS_PATH = f"{DISCO_ROOT_PATH}/results/"
TASK_TIMEOUT = 12 * 60 * 60  # 12 hours


import numpy as np
from src.Evaluation.dcdistances.dctree import DCTree
from sklearn.cluster import DBSCAN


dataset = RealWorldDatasets.Synth_high
X, l = dataset.standardized_data_cached_no_noise

ks = range(2, 21)
dctree = DCTree(X, min_points=5, min_points_mr=2)
eps_list = [dctree.get_eps_for_k(k) for k in ks]
kcenter_labels = [dctree.get_k_center(k) for k in ks]
dbscan_labels = [DBSCAN(eps).fit(X).labels_ for eps in eps_list]

# kcenter_disco_results = [disco_score(X, l_) for l_ in kcenter_labels]
# dbscan_disco_results = [disco_score(X, l_) for l_ in dbscan_labels]

dbscan_dataset_names = [f"dbscan_{i}" for i in ks]
dbscan_dataset_ids = dict(zip(dbscan_dataset_names, dbscan_dataset_names))
dbscan_datasets = [lambda X=X, i=i: (X, dbscan_labels[i]) for i, k in enumerate(ks)]
dbscan_load_fn_dict = dict(zip(dbscan_dataset_names, dbscan_datasets))

kcenter_dataset_names = [f"kcenter_{i}" for i in ks]
kcenter_dataset_ids = dict(zip(kcenter_dataset_names, kcenter_dataset_names))
kcenter_datasets = [lambda X=X, i=i: (X, kcenter_labels[i]) for i, k in enumerate(ks)]
kcenter_load_fn_dict = dict(zip(kcenter_dataset_names, kcenter_datasets))

print("Start\n")
config_dbscan = {
    "save_folder": f"{RESULTS_PATH}find_dbscan_eps",
    "dataset_names": dbscan_dataset_names,
    "dataset_id_dict": dbscan_dataset_ids,
    "dataset_load_fn_dict": dbscan_load_fn_dict,
    "functions": METRICS,
    "n_jobs": 1,
    "runs": 1,
}

config_kcenter = {
    "save_folder": f"{RESULTS_PATH}find_kcenter_k",
    "dataset_names": kcenter_dataset_names,
    "dataset_id_dict": kcenter_dataset_ids,
    "dataset_load_fn_dict": kcenter_load_fn_dict,
    "functions": METRICS,
    "n_jobs": 1,
    "runs": 1,
}


if __name__ == "__main__":
    time.tzset()
    run_multiple_experiments(**config_dbscan)
    # run_multiple_experiments(**config_kcenter)
