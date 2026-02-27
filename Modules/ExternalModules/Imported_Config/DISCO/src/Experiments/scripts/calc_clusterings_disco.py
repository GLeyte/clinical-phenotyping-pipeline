import sys
import time
import pandas as pd
import numpy as np

DISCO_ROOT_PATH = "/export/share/pascalw777dm/DISCO"
sys.path.append(DISCO_ROOT_PATH)

from datasets.real_world_datasets import Datasets as RealWorldDatasets
from datasets.density_datasets import Datasets as DensityDatasets
from src.utils.cluster_algorithms import CLUSTER_ALGORITHMS
from src.Experiments.scripts._calc_multiple_experiments import run_multiple_experiments
from ast import literal_eval
from itertools import product
from src.Evaluation import disco_score


RESULTS_PATH = f"{DISCO_ROOT_PATH}/clustering_results/"
RESULTS_PATH2 = f"{DISCO_ROOT_PATH}/clustering_results2/"
TASK_TIMEOUT = 12 * 60 * 60  # 12 hours


def load_clustering(args):
    dataset, clusterer, run = args
    X, l = dataset.standardized_data_cached

    np.random.seed(0)
    seeds = np.random.choice(10_000, size=run + 1, replace=False)
    np.random.seed(seeds[-1])
    shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
    X = X[shuffle_data_index]

    df = pd.read_csv(f"{DISCO_ROOT_PATH}/clusterings/density_standardized/{dataset.id}/{clusterer}_{run}.csv")
    l_ = df["value"][0]
    l_ = np.array(literal_eval(",".join(l_.split()).replace("[,", "[")))

    return X, l_


# if sys.argv[1] == "real_world":
#     print("Use data without z-normalization\n")
#     config = {
#         "save_folder": f"{RESULTS_PATH}real_world",
#         "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
#         "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
#         "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached for dataset in RealWorldDatasets.get_experiments_list()},
#         "functions": CLUSTER_ALGORITHMS,
#         "n_jobs": 1,
#         "runs": 1,
#     }

# elif sys.argv[1] == "real_world_standardized":
#     print("Use data with z-normalization\n")
#     config = {
#         "save_folder": f"{RESULTS_PATH}real_world_standardized",
#         "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
#         "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
#         "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.standardized_data_cached for dataset in RealWorldDatasets.get_experiments_list()},
#         "functions": CLUSTER_ALGORITHMS,
#         "n_jobs": 1,
#         "runs": 1,
#     }


# elif sys.argv[1] == "density":
#     print("Use data without z-normalization\n")
#     config = {
#         "save_folder": f"{RESULTS_PATH}density",
#         "dataset_names": [dataset.name for dataset in DensityDatasets],
#         "dataset_id_dict": {dataset.name: dataset.id for dataset in DensityDatasets},
#         "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached for dataset in DensityDatasets},
#         "functions": CLUSTER_ALGORITHMS,
#         "n_jobs": 10,
#         "runs": 10,
#     }

if sys.argv[1] == "density_standardized":
    print("Use data with z-normalization\n")
    runs = 10
    datasets = list(product(DensityDatasets, CLUSTER_ALGORITHMS.keys(), range(runs)))
    dataset_names = list(map(lambda args: (args[0].name, args[1], args[2]), datasets))
    dataset_ids = list(map(lambda args: (args[0].id, args[1], args[2]), datasets))
    dataset_id_dict = dict(zip(dataset_names, dataset_ids))
    dataset_funcs = list(map(lambda args: lambda args=args: load_clustering(args), datasets))
    dataset_load_fn_dict = dict(zip(dataset_names, dataset_funcs))
    config = {
        "save_folder": f"{RESULTS_PATH}density_standardized",
        "dataset_names": dataset_names,
        "dataset_id_dict": dataset_id_dict,
        "dataset_load_fn_dict": dataset_load_fn_dict,
        "functions": {"DISCO": disco_score},
        "n_jobs": 64,
        "runs": 1,
    }
    paths = {
        "path": f"{RESULTS_PATH}density_standardized/",
        "new_path": f"{RESULTS_PATH2}density_standardized/",
    }

else:
    print("Need to select `standardized` or `normal`!\n")
    exit()


import os
import glob
from ast import literal_eval

def restructer_clustering_results(path, new_path):
    for file_path in glob.glob(f"{path}*/*"):
        folders = file_path.split("/")
        dataset, clusterer, run = literal_eval(folders[-2])
        new_file_path = f"{new_path}{dataset}/{clusterer}_{run}.csv"
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        df = pd.read_csv(file_path)
        df["function"] = df["dataset"].apply(lambda args: literal_eval(args)[1])
        df["dataset"] = df["dataset"].apply(lambda args: literal_eval(args)[0])
        df.to_csv(new_file_path, index=False, na_rep="nan")


if __name__ == "__main__":
    time.tzset()
    run_multiple_experiments(**config)
    restructer_clustering_results(**paths)
