import numpy as np
import time
import os
import sys
import pickle
import psutil
import traceback


parent_folder = os.path.dirname(os.path.abspath("./"))
sys.path.append(parent_folder)

from collections import defaultdict
from src.utils.metrics import METRICS
from mpire.pool import WorkerPool


CACHE_FOLDER = "/export/share/pascalw777dm/DISCO/.cache/"


def cache(filename, func, args=[], kwargs={}, recalc=False):
    cache_folder = os.path.abspath(CACHE_FOLDER)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    cache_path = f"{cache_folder}/{filename}.pkl"
    if not recalc and os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    else:
        result = func(*args, **kwargs)
        with open(cache_path, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return result


def insert_dict(dict, key_value_dict):
    for key, value in key_value_dict.items():
        dict[key].append(value)


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        dict1[key] += value


def process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def exec_func(data, fn, args=[], kwargs={}):
    """Runs function for given dataset with data `X` and labels `l`."""
    start_time = time.time()
    start_process_time = time.process_time()
    mem_before = process_memory()
    try:
        value = fn(*data, *args, **kwargs)
    except Exception:
        traceback.print_exc()
        value = np.nan
    mem_after = process_memory()
    end_process_time = time.process_time()
    end_time = time.time()
    return value, end_time - start_time, end_process_time - start_process_time, mem_after - mem_before


def calc_eval_measures(X, l, name=None, metrics=METRICS, runs=10, n_jobs=1, task_timeout=None):
    """Calculate all evaluation measures for a given dataset with data `X` and labels `l`."""

    if n_jobs != 1:
        pool = WorkerPool(n_jobs=n_jobs, use_dill=True)
    async_results = {}

    np.random.seed(0)
    seeds = np.random.choice(10_000, size=runs, replace=False)

    for run, seed in enumerate(seeds):
        np.random.seed(seed)
        shuffle_data_index = np.random.choice(len(X), size=len(X), replace=False)
        X_ = X[shuffle_data_index]
        l_ = l[shuffle_data_index]

        for metric_name, metric_fn in metrics.items():
            async_idx = (run, metric_name)
            if n_jobs != 1:
                async_results[async_idx] = pool.apply_async(
                    exec_func, args=((X_, l_), metric_fn), task_timeout=task_timeout
                )
            else:
                async_results[async_idx] = exec_func((X_, l_), metric_fn)

    eval_results = defaultdict(list)
    for async_idx, async_result in async_results.items():
        (run, metric_name) = async_idx
        if n_jobs != 1:
            value, real_time, cpu_time, mem_usage = async_result.get()
        else:
            value, real_time, cpu_time, mem_usage = async_result

        insert_dict(
            eval_results,
            {
                "dataset": name,
                "measure": metric_name,
                "run": run,
                "value": value,
                "time": real_time,
                "process_time": cpu_time,
                "mem_usage": mem_usage,
            },
        )

    if n_jobs != 1:
        pool.stop_and_join()
        pool.terminate()
    return eval_results


def calc_eval_measures_for_multiple_datasets(
    data, param_values, metrics=METRICS, n_jobs=1, task_timeout=None
):
    """Calculates all evaluation measures for all datasets in data.

    Args:
        data: 2d matrix of type [datasets x runs]
    """

    if n_jobs != 1:
        pool = WorkerPool(n_jobs=n_jobs, use_dill=True)
    async_results = {}

    for param_value in range(len(param_values)):
        for run in range(len(data[param_value])):
            X, l = data[param_value][run]

            for metric_name, metric_fn in metrics.items():
                async_idx = (param_value, run, metric_name)
                if n_jobs != 1:
                    async_results[async_idx] = pool.apply_async(
                        exec_func, args=((X, l), metric_fn), task_timeout=task_timeout
                    )
                else:
                    async_results[async_idx] = exec_func((X, l), metric_fn)

    eval_results = defaultdict(list)
    for async_idx, async_result in async_results.items():
        (param_value, run, metric_name) = async_idx
        if n_jobs != 1:
            value, real_time, cpu_time, mem_usage = async_result.get()
        else:
            value, real_time, cpu_time, mem_usage = async_result

        insert_dict(
            eval_results,
            {
                "dataset": param_values[param_value],
                "measure": metric_name,
                "run": run,
                "value": value,
                "time": real_time,
                "process_time": cpu_time,
                "mem_usage": mem_usage,
            },
        )

    if n_jobs != 1:
        pool.stop_and_join()
        pool.terminate()
    return eval_results
