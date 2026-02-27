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


# del METRICS["CDBW"]
# del METRICS["CVDD"]
# del METRICS["LCCV"]
# del METRICS["DBCV"]
# del METRICS["DBCV_eucl"]
# del METRICS["VIASCKDE"]


if sys.argv[1] == "real_world":
    # del METRICS["CDBW"]
    print("Use data without z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}real_world",
        "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached_no_noise for dataset in RealWorldDatasets.get_experiments_list()},
        "functions": METRICS,
        "n_jobs": 1,
        "runs": 1,
    }

elif sys.argv[1] == "real_world_standardized":
    # del METRICS["CDBW"]
    print("Use data with z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}real_world_standardized",
        "dataset_names": [dataset.name for dataset in RealWorldDatasets.get_experiments_list()],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in RealWorldDatasets.get_experiments_list()},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.standardized_data_cached_no_noise for dataset in RealWorldDatasets.get_experiments_list()},
        "functions": METRICS,
        "n_jobs": 64,
        "runs": 10,
    }


elif sys.argv[1] == "density":
    print("Use data without z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}density",
        "dataset_names": [dataset.name for dataset in DensityDatasets],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in DensityDatasets},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.data_cached_no_noise for dataset in DensityDatasets},
        "functions": METRICS,
        "n_jobs": 32,
        "runs": 10,
    }

elif sys.argv[1] == "density_standardized":
    print("Use data with z-normalization\n")
    config = {
        "save_folder": f"{RESULTS_PATH}density_standardized",
        "dataset_names": [dataset.name for dataset in DensityDatasets],
        "dataset_id_dict": {dataset.name: dataset.id for dataset in DensityDatasets},
        "dataset_load_fn_dict": {dataset.name: lambda dataset=dataset: dataset.standardized_data_cached_no_noise for dataset in DensityDatasets},
        "functions": METRICS,
        "n_jobs": 32,
        "runs": 10,
    }

else:
    print("Need to select `standardized` or `normal`!\n")
    exit()


if __name__ == "__main__":
    time.tzset()
    run_multiple_experiments(**config)
