import numpy as np
import os

from .abstract_datasets import AbstractDatasets, standardize
from clustpy.data import *


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}"


### Datasets ###


class Datasets(AbstractDatasets):
    # Tabular data
    Synth_low = "Synth_low"
    Synth_high = "Synth_high"
    HAR = "HAR"
    letterrec = "letterrec."
    htru2 = "htru2"
    Mice = "Mice"
    Pendigits = "Pendigits"
    # Video data
    Weizmann = "Weizmann"
    Keck = "Keck"
    # Image data
    COIL20 = "COIL20"
    COIL100 = "COIL100"
    cmu_faces = "cmu_faces"
    # MNIST data
    Optdigits = "Optdigits"
    USPS = "USPS"
    MNIST = "MNIST"
    FMNIST = "FMNIST"
    KMNIST = "KMNIST"

    @classmethod
    def get_experiments_list(cls):
        return [dataset for dataset in cls if dataset not in cls.__get_excluded()]

    @classmethod
    def __get_excluded(cls):
        return [
            cls.MNIST,
            cls.FMNIST,
            cls.KMNIST,
            cls.Keck,
            cls.Weizmann,
            cls.COIL100,
            # cls.COIL20,
            # cls.cmu_faces,
        ]

    def load_dataset(self):
        match self:
            # Tabular data
            case self.Synth_low:
                path = f"{DATASETS_FOLDER}/low_data_100.npy"
                X, l = np.hsplit(np.load(path), [-1])
                return X, l.reshape(-1)
            case self.Synth_high:
                path = f"{DATASETS_FOLDER}/high_data_100.npy"
                X, l = np.hsplit(np.load(path), [-1])
                return X, l.reshape(-1)
            case self.HAR:
                return load_har(return_X_y=True)
            case self.letterrec:
                return load_letterrecognition(return_X_y=True)
            case self.htru2:
                return load_htru2(return_X_y=True)
            case self.Mice:
                return load_mice_protein(return_X_y=True)
            case self.Pendigits:
                return load_pendigits(return_X_y=True)
            # Video data
            case self.Weizmann:
                X, l = load_video_weizmann(return_X_y=True)
                acts = l[:, 0]
                persons = l[:, 1]
                l = persons * len(np.unique(acts)) + acts
                return X, l
            case self.Keck:
                X, l = load_video_keck_gesture(return_X_y=True, image_size=(100, 100))
                acts = l[:, 0] - 1
                nr_of_acts = len(np.unique(acts)) - 1
                persons = l[:, 1]
                l_new = np.full(len(l), -1)
                l_new[acts != -1] = (persons * nr_of_acts)[acts != -1] + acts[acts != -1]
                return X, l_new
            # Image data
            case self.COIL20:
                return load_coil20(return_X_y=True)
            case self.COIL100:
                return load_coil100(return_X_y=True)
            case self.cmu_faces:
                X, l = load_cmu_faces(return_X_y=True)
                l = l[:, 0]
                return X, l
            # MNIST data
            case self.Optdigits:
                return load_optdigits(return_X_y=True)
            case self.USPS:
                return load_usps(return_X_y=True)
            case self.MNIST:
                return load_mnist(return_X_y=True)
            case self.FMNIST:
                return load_fmnist(return_X_y=True)
            case self.KMNIST:
                return load_kmnist(return_X_y=True)
            case _:
                raise AttributeError

    def standardize_dataset(self, X, l):
        match self:
            case dataset if dataset in [
                # Tabular data
                self.Synth_low,
                self.Synth_high,
                self.HAR,
                self.letterrec,
                self.htru2,
                self.Mice,
                self.Pendigits,
            ]:
                return standardize(X, l, axis=0)
            case dataset if dataset in [
                # Video data
                self.Weizmann,
                self.Keck,
                # Image data
                self.COIL20,
                self.COIL100,
                self.cmu_faces,
                # MNIST data
                self.Optdigits,
                self.USPS,
                self.MNIST,
                self.FMNIST,
                self.KMNIST,
            ]:
                return standardize(X, l, axis=None)
            case _:
                raise AttributeError
