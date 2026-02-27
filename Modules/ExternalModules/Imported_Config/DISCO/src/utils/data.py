import numpy as np
from sklearn.neighbors import NearestNeighbors


def convert_to_numpy(datasets):
    datasets_np = np.empty((len(datasets), len(datasets[0]), 2), dtype=object)
    datasets_np[:] = datasets
    return datasets_np


def sample_datasets(datasets, func):
    datasets = convert_to_numpy(datasets)

    def apply_to_sample(data):
        new_data = np.empty(2, dtype=object)
        data = func(data[0], data[1])
        new_data[:] = data
        return new_data

    return np.apply_along_axis(lambda data: apply_to_sample(data), 2, datasets)


def add_noise(X, l, n_noise, eps, noise_eps, border=0):
    """Add noise to data with at least eps distance to the data."""

    noise = np.empty((n_noise, X.shape[1]))
    noise_too_near = np.array(range(len(noise)))
    while len(noise_too_near) > 0:
        noise[noise_too_near] = np.random.uniform(
            np.min(X - border, axis=0), np.max(X + border, axis=0), size=(len(noise_too_near), X.shape[1])
        )
        nbrs_points = NearestNeighbors(n_neighbors=1).fit(X)
        dists_points = nbrs_points.kneighbors(noise)[0]
        noise_too_near_points = np.where(dists_points < eps)[0]
        nbrs_noise = NearestNeighbors(n_neighbors=2).fit(noise)
        dists_noise = nbrs_noise.kneighbors(noise)[0][:, 1]
        noise_too_near_noise = np.where(dists_noise < noise_eps)[0]
        noise_too_near = np.unique(np.hstack((noise_too_near_points, noise_too_near_noise)))

    X_ = np.vstack((X, noise))
    l_ = np.hstack((l, np.array([-1] * len(noise))))

    return X_, l_
