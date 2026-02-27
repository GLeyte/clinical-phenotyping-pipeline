import numpy as np
import os

from .abstract_datasets import AbstractDatasets, standardize
from .DENSIRED import datagen


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}"


### Datasets ###


class Datasets(AbstractDatasets):
    # Synthetic data generated with DENSIRED
    Dataset1 = "Dataset1"
    Dataset2 = "Dataset2"
    DatasetDensiredExample = "DatasetDensiredExample"

    @property
    def skeleton(self):
        _X, _l, skeleton, _data = generate_dataset(self.config)
        return skeleton

    @property
    def config(self):
        match self:
            # Synthetic data generated with DENSIRED

            case self.Dataset1:
                return {
                    "dim": 2,
                    "n": 550,
                    "cluster_nums": [250, 250],
                    "core_nums": [50, 50],
                    "n_noise": 50,
                    "seed": 0,
                }

            case self.Dataset2:
                return {
                    "dim": 2,
                    "n": 1050,
                    "cluster_nums": [750, 250],
                    "core_nums": [50, 50],
                    "n_noise": 50,
                    "seed": 0,
                    "kwargs": {
                        "dens_factors": [1, 2],
                    },
                }

            case self.DatasetDensiredExample:
                return {
                    "n": 5000,
                    "kwargs": {
                        "dim": 2,
                        "ratio_noise": 0.1,
                        "max_retry": 5,
                        "dens_factors": [1, 1, 0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1],
                        "square": True,
                        "clunum": 10,
                        "seed": 6,
                        "core_num": 200,
                        "momentum": [0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6, 0.45, 0.7],
                        "branch": [0, 0.05, 0.1, 0, 0, 0.1, 0.02, 0, 0, 0.25],
                        "con_min_dist": 0.8,
                        "verbose": False,
                        "safety": True,
                        "domain_size": 20,
                        "random_start": False,
                    },
                }

            case _:
                raise AttributeError

    def load_dataset(self):
        X, l, _skeleton, _data = generate_dataset(self.config)
        return X, l

    def standardize_dataset(self, X, l):
        match self:
            case dataset if dataset in [
                # Synthetic data generated with DENSIRED
                self.Dataset1,
                self.Dataset2,
                self.DatasetDensiredExample,
            ]:
                return standardize(X, l, axis=0)
            case dataset if dataset in []:
                return standardize(X, l, axis=None)
            case _:
                raise AttributeError


def generate_dataset(config):
    dim = config.get("dim")
    n = config.get("n")
    cluster_nums = config.get("cluster_nums")
    core_nums = config.get("core_nums")
    n_noise = config.get("n_noise")
    seed = config.get("seed")

    N_CLUSTERS = len(cluster_nums) if cluster_nums else None

    kwargs = {
        "dim": dim,
        "clunum": N_CLUSTERS,
        "core_num": core_nums,
        "ratio_noise": n_noise / n if n_noise else None,
        "clu_ratios": cluster_nums,
        "seed": seed,
    }
    kwargs.update(config.get("kwargs", {}))

    if kwargs["clu_ratios"] and n_noise:
        assert (
            sum(kwargs["clu_ratios"]) + n_noise == n
        ), "`cluster_nums` + `n_noise` needs to sum up to `N`"

    skeleton = datagen.densityDataGen(**kwargs)

    data = skeleton.generate_data(n)
    X = data[:, 0:-1]
    l = data[:, -1]
    return X, l, skeleton, data
