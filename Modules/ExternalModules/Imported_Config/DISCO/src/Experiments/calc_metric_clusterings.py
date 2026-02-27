import sys
import os

import numpy as np
import pandas as pd

DISCO_ROOT_PATH = "/export/share/pascalw777dm/DISCO"
sys.path.append(DISCO_ROOT_PATH)
os.environ["TZ"] = "Europe/Vienna"

from ast import literal_eval

from src.utils.metrics import METRICS, METRICS2
from datasets.density_datasets import Datasets as DensityDatasets
from datasets.real_world_datasets import Datasets as RealWorldDatasets

from src.utils.cluster_algorithms import CLUSTER_ALGORITHMS

from mpire.pool import WorkerPool


n_jobs = 5
task_timeout = 12 * 60 * 60  # 12 hours

# DATASETs = DensityDatasets
# DATASET_PATH = "density_standardized"
# RUNS = 10

DATASETs = RealWorldDatasets.get_experiments_list()
DATASET_PATH = "real_world_standardized"
RUNS = 1

import psutil
import time


def add_time(text):
    return f"{time.strftime("%H:%M:%S")}: {text}"


def load_clustering(args):
    dataset, clusterer, run = args
    X, l = dataset.standardized_data_cached

    np.random.seed(0)
    seeds = np.random.choice(10_000, size=run + 1, replace=False)
    np.random.seed(seeds[-1])
    shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
    X = X[shuffle_data_index]

    df = pd.read_csv(f"{DISCO_ROOT_PATH}/clusterings/{DATASET_PATH}/{dataset.id}/{clusterer}_{run}.csv")
    l_ = df["value"][0]
    l_ = np.array(literal_eval(",".join(l_.split()).replace("[,", "[")))

    return X, l_


def calc_and_save_metric(dataset, clusterer, run, metric_name, metric_func, metric_save_path):
    X_, l_ = load_clustering([dataset, clusterer, run])
    print(add_time(f"Start - Dataset: {dataset.id}, Clusterer: {clusterer}, Run: {run}, Metric: {metric_name}"))
    try:
        metric_value = metric_func(X_, l_)
    except TimeoutError:
        return
    except:
        metric_value = np.nan
    print(f"{dataset.name=}, {clusterer=}, {run=}, {metric_name=} -- {metric_value}")
    os.makedirs(os.path.dirname(metric_save_path), exist_ok=True)
    np.savetxt(metric_save_path, [metric_value])


pool = WorkerPool(n_jobs=n_jobs, use_dill=True)
async_jobs = {}

for dataset in DATASETs:
    for clusterer in CLUSTER_ALGORITHMS.keys():
        for run in range(RUNS):
            for metric_name, metric_func in METRICS2.items():
                metric_save_path = f"{DISCO_ROOT_PATH}/clusterings_metrics/{DATASET_PATH}/{dataset.id}/{clusterer}_{run}##{metric_name}.csv"
                if os.path.exists(metric_save_path):
                    print(f"Skipping -- {dataset.id=}, {clusterer=}, {run=}, {metric_name=}")
                elif not os.path.exists(f"{DISCO_ROOT_PATH}/clusterings/{DATASET_PATH}/{dataset.id}/{clusterer}_{run}.csv"):
                    print(f"Clustering not found -- {dataset.id=}, {clusterer=}, {run=}, {metric_name=}")
                else:
                    job = pool.apply_async(calc_and_save_metric, args=(dataset, clusterer, run, metric_name, metric_func, metric_save_path), task_timeout=task_timeout)
                    async_jobs[(dataset.id, clusterer, run, metric_name)] = job
                    # calc_and_save_metric(dataset, clusterer, run, metric_name, metric_func, metric_save_path)


while async_jobs:
    used_ram = round(psutil.virtual_memory().percent, 2)
    free_mem = round(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, 2)
    print(add_time(f"RAM INFO - Used RAM: {used_ram}, Free RAM: {free_mem}"))
    print(add_time("-----"))
    time.sleep(10)
    current_tasks = 0
    for async_idx, async_job in list(async_jobs.items()):
        dataset_name, clusterer, run, metric_name = async_idx

        if not async_job.ready():
            current_tasks += 1
            if current_tasks <= n_jobs:
                print(add_time(f"Calculating - Dataset: {dataset_name}, Clusterer: {clusterer}, Run: {run}, Metric: {metric_name}"))
            continue

        if not async_job.successful():
            print(add_time(f"Failed - Dataset: {dataset_name}, Clusterer: {clusterer}, Run: {run}, Metric: {metric_name}"))
            del async_jobs[async_idx]
            continue

        async_job.get()
        print(add_time(f"Finished - Dataset: {dataset_name}, Clusterer: {clusterer}, Run: {run}, Metric: {metric_name}"))  # - `{value}`

        del async_jobs[async_idx]
