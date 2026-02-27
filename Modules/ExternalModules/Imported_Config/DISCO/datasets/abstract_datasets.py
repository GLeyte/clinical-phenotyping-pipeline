import numpy as np
import os

from abc import abstractmethod
from enum import Enum


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}"


### Datasets ###


class AbstractDatasets(Enum):

    @property
    def id(self):
        return super(AbstractDatasets, self).name

    @property
    def name(self):
        return super(AbstractDatasets, self).value

    @property
    def data(self) -> tuple:
        """Returns (X, l)"""
        return self.load_dataset()

    @property
    def standardized_data(self) -> tuple:
        """Returns (X, l), with X standardized"""
        return self.standardize_dataset(*self.data)

    @property
    def data_cached(self) -> tuple:
        """Returns (X, l) and caches (X, l)"""
        return load_and_cache_dataset(self.id, lambda self=self: self.data)

    @property
    def standardized_data_cached(self) -> tuple:
        """Returns (X, l), with X standardized and caches (X, l)"""
        return load_and_cache_dataset(f"{self.id}_z", lambda self=self: self.standardized_data)

    @property
    def data_no_noise(self) -> tuple:
        """Returns (X, l)"""
        X, l = self.load_dataset()
        return X[l != -1], l[l != -1]

    @property
    def standardized_data_no_noise(self) -> tuple:
        """Returns (X, l), with X standardized"""
        X, l = self.standardize_dataset(*self.data)
        return X[l != -1], l[l != -1]

    @property
    def data_cached_no_noise(self) -> tuple:
        """Returns (X, l) and caches (X, l)"""
        X, l = load_and_cache_dataset(self.id, lambda self=self: self.data)
        return X[l != -1], l[l != -1]

    @property
    def standardized_data_cached_no_noise(self) -> tuple:
        """Returns (X, l), with X standardized and caches (X, l)"""
        X, l = load_and_cache_dataset(f"{self.id}_z", lambda self=self: self.standardized_data)
        return X[l != -1], l[l != -1]

    @abstractmethod
    def load_dataset(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def standardize_dataset(self, X, l) -> tuple:
        raise NotImplementedError


def standardize(X, l, axis=None):
    std = np.std(X, axis=axis)
    mean = np.mean(X, axis=axis)
    if axis is not None:
        std = np.expand_dims(std, axis)
        mean = np.expand_dims(mean, axis)
    X = (X - mean) / std
    return X, l


def load_np_dataset(path):
    X = np.load(path + "_data.npy", allow_pickle=True)
    l = np.load(path + "_labels.npy", allow_pickle=True)
    X = X.reshape((len(X), -1))
    return X, l


def load_and_cache_dataset(cache_name, func):
    if not os.path.exists(f"{DATASETS_FOLDER}/.cache/"):
        os.makedirs(f"{DATASETS_FOLDER}/.cache/")
    cache_path = f"{DATASETS_FOLDER}/.cache/{cache_name}"
    if os.path.exists(f"{cache_path}_data.npy") and os.path.exists(f"{cache_path}_labels.npy"):
        return load_np_dataset(cache_path)
    else:
        X, l = func()
        np.save(f"{cache_path}_data.npy", X)
        np.save(f"{cache_path}_labels.npy", l)
        return X, l
