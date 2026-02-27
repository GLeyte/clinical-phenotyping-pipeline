import numpy as np
import os
from .abstract_datasets import AbstractDatasets, standardize

from urllib.request import urlopen
from scipy.io import arff
from io import StringIO


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASETS_FOLDER = f"{CURRENT_DIRECTORY}"


### Datasets ###


class Datasets(AbstractDatasets):
    three_spiral = "three_spiral"
    aggregation = "aggregation"
    chainlink = "chainlink"
    cluto_t4_8k = "cluto-t4-8k"
    cluto_t5_8k = "cluto-t5-8k"
    cluto_t7_10k = "cluto-t7-10k"
    cluto_t8_8k = "cluto-t8-8k"
    complex8 = "complex8"
    complex9 = "complex9"
    compound = "compound"
    dartboard1 = "dartboard1"
    diamond9 = "diamond9"
    smile1 = "smile1"
    zelnik4 = "zelnik4"

    def load_dataset(self):
        match self:
            case self.three_spiral:
                return download_dataset("3-spiral")
            case self.aggregation:
                return download_dataset("aggregation")
            case self.chainlink:
                return download_dataset("chainlink")
            case self.cluto_t4_8k:
                return download_dataset("cluto-t4-8k")
            case self.cluto_t5_8k:
                return download_dataset("cluto-t5-8k")
            case self.cluto_t7_10k:
                return download_dataset("cluto-t7-10k")
            case self.cluto_t8_8k:
                return download_dataset("cluto-t8-8k")
            case self.complex8:
                return download_dataset("complex8")
            case self.complex9:
                return download_dataset("complex9")
            case self.compound:
                return download_dataset("compound")
            case self.dartboard1:
                return download_dataset("dartboard1")
            case self.diamond9:
                return download_dataset("diamond9")
            case self.smile1:
                return download_dataset("smile1")
            case self.zelnik4:
                return download_dataset("zelnik4")
            case _:
                raise AttributeError

    def standardize_dataset(self, X, l):
        return standardize(X, l, axis=0)


def download_dataset(dataset_name):
    github_url = f"https://raw.githubusercontent.com/deric/clustering-benchmark/master/src/main/resources/datasets/artificial/{dataset_name}.arff"
    arff_data = urlopen(github_url).read().decode("utf-8")
    arff_data = arff_data.replace("noise", "-1")
    arff_data_file_object = StringIO(arff_data)
    data, _meta = arff.loadarff(arff_data_file_object)
    np_data = np.array(data.tolist(), dtype=float)
    X, l = np.hsplit(np_data, [-1])
    return X, l.reshape(-1)
