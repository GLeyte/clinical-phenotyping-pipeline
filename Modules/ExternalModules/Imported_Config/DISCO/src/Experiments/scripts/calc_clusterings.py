import sys
import time

DISCO_ROOT_PATH = "/export/share/pascalw777dm/DISCO"
sys.path.append(DISCO_ROOT_PATH)

from datasets.real_world_datasets import Datasets as RealWorldDatasets
from datasets.density_datasets import Datasets as DensityDatasets
from src.utils.cluster_algorithms import CLUSTER_ALGORITHMS
from src.Experiments.scripts._calc_multiple_experiments import run_multiple_experiments


RESULTS_PATH = f"{DISCO_ROOT_PATH}/clusterings/"
TASK_TIMEOUT = 12 * 60 * 60  # 12 hours


if sys.argv[1] == "real_world":
    print("Use data without z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}real_world",
        "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached for dataset in RealWorldDatasets.get_experiments_list()},
        "functions": CLUSTER_ALGORITHMS,
        "n_jobs": 50,
        "runs": 1,
    }

elif sys.argv[1] == "real_world_standardized":
    print("Use data with z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}real_world_standardized",
        "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.standardized_data_cached for dataset in RealWorldDatasets.get_experiments_list()},
        "functions": CLUSTER_ALGORITHMS,
        "n_jobs": 10,
        "runs": 10,
    }


elif sys.argv[1] == "density":
    print("Use data without z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}density",
        "dataset_names": [dataset.name for dataset in DensityDatasets],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in DensityDatasets},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached for dataset in DensityDatasets},
        "functions": CLUSTER_ALGORITHMS,
        "n_jobs": 10,
        "runs": 10,
    }

elif sys.argv[1] == "density_standardized":
    print("Use data with z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}density_standardized",
        "dataset_names": [dataset.name for dataset in DensityDatasets],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in DensityDatasets},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.standardized_data_cached for dataset in DensityDatasets},
        "functions": CLUSTER_ALGORITHMS,
        "n_jobs": 10,
        "runs": 10,
    }

else:
    print("Need to select `standardized` or `normal`!\n")
    exit()

if __name__ == "__main__":
    time.tzset()
    run_multiple_experiments(**config)
