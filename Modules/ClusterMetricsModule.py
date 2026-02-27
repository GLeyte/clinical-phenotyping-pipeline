from ExternalModules.ExternalMetrics.CVI.registry import get_measures_dict
import ExternalModules.ExternalMetrics.DISCO.disco as disco
import dbcv
import warnings


class ClusterMetrics:
    """
    Base class for clustering metrics.
    This class provides a structure for implementing various clustering metrics.
    It should be subclassed to implement specific metrics.
    """

    def __init__(self):
        """
        Initialises the ClusterMetrics instance.
        This method may be overridden in subclasses to initialise specific metrics.
        """
        self._measures = get_measures_dict()

        self._silhouette_score = None
        self._s_dbw_index = None
        self._dbcv_index = None
        self._viaskde_index = None
        self._dsi_index = None
        self._hal_cvnn_index = None
        self._disco_index = None

    def silhouette_score(self, data, labels, consider_noise: bool = False) -> float:
        """
        Calculates the silhouette score for clustering.

        Args:
            data: numpy array of shape (n_samples, n_features) - the clustered data
            labels: array-like of cluster labels for each data point
            consider_noise: bool, optional
                Whether to consider noise points (default is False).

        Returns:
            float: The silhouette score (higher is better)
        """
        if consider_noise:
            labels = labels.copy()
            labels[labels == -1] = max(labels) + 1  # Treat noise as a separate cluster
            self._silhouette_score = self._measures["SWC"].score(data, labels)
            return self._silhouette_score

        mask = labels != -1  # Exclude noise points if any

        filtered_data = data[mask]
        filtered_labels = labels[mask]

        self._silhouette_score = self._measures["SWC"].score(
            filtered_data, filtered_labels
        )
        return self._silhouette_score

    def s_dbw_index(self, data, labels, consider_noise: bool = False) -> float:
        """
        Calculates the S_Dbw index for clustering.

        Args:
            data: numpy array of shape (n_samples, n_features) - the clustered data
            labels: array-like of cluster labels for each data point

        Returns:
            float: The S_Dbw index (lower is better)
        """
        if consider_noise:
            self._s_dbw_index = self._measures["S-Dbw"].score(data, labels)
            return self._s_dbw_index

        mask = labels != -1  # Exclude noise points if any

        filtered_data = data[mask]
        filtered_labels = labels[mask]
        self._s_dbw_index = self._measures["S-Dbw"].score(
            filtered_data, filtered_labels
        )
        return self._s_dbw_index

    def dbcv_index(
        self, data, labels, noise_id: int = -1, dist_function: str = "sqeuclidean"
    ) -> float:
        """
        Calculates the DBCV index for clustering.

        Args:
            data: numpy array of shape (n_samples, n_features) - the clustered data
            labels: array-like of cluster labels for each data point
            noise_id: int, optional
                The label used for noise points. Default is -1.
            dist_function: str or callable, optional
                The distance metric to use. If a string, the distance function can be
                'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
                'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
                'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
                'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                'sokalsneath', 'sqeuclidean', 'yule'.

        Returns:
            float: The DBCV index (higher is better)
        """

        self._dbcv_index = dbcv.dbcv(
            data, labels, noise_id=noise_id, metric=dist_function
        )
        return self._dbcv_index

    def viaskde_index(
        self,
        data,
        labels,
        kernel: str = "gaussian",
        bandwidth: float = 0.05,
        consider_noise: bool = False,
    ) -> float:
        """
        Calculates the VIASCKDE index for clustering.

        Args:
            data (numpy.ndarray): Input data points.
            labels (numpy.ndarray): Cluster labels for each data point.
            kernel (str, optional): {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}, default='gaussian'
                The kernel to use.
            bandwidth (float or string, optional): float or {"scott", "silverman"}, default=0.05
                The kernel bandwidth.
            consider_noise (bool, optional): Whether to consider noise points in calculation. Default is False.

        Returns:
            float: The VIASCKDE index (higher is better)
        """
        if consider_noise:
            self._viaskde_index = self._measures["VIASCKDE"].score(data, labels)
            return self._viaskde_index

        mask = labels != -1  # Exclude noise points if any

        filtered_data = data[mask]
        filtered_labels = labels[mask]

        self._viaskde_index = self._measures["VIASCKDE"].score(
            filtered_data, filtered_labels
        )
        return self._viaskde_index

    def dsi_index(self, data, labels, consider_noise: bool = False) -> float:
        """
        Calculates the DSI index for clustering.

        Args:
            data: numpy array of shape (n_samples, n_features) - the clustered data
            labels: array-like of cluster labels for each data point
            consider_noise: bool, optional
                Whether to consider noise points (default is False).
        Returns:
            float: The DSI index (higher is better)
        """
        if consider_noise:
            labels = labels.copy()
            labels[labels == -1] = max(labels) + 1  # Treat noise as a separate cluster
            self._dsi_index = self._measures["DSI"].score(data, labels)
            return self._dsi_index

        mask = labels != -1  # Exclude noise points if any

        filtered_data = data[mask]
        filtered_labels = labels[mask]

        self._dsi_index = self._measures["DSI"].score(filtered_data, filtered_labels)
        return self._dsi_index

    def hal_cvnn_index(self, data, labels, consider_noise: bool = False) -> float:
        """
        Calculates the CVNN index for clustering.

        Args:
            data: numpy array of shape (n_samples, n_features) - the clustered data
            labels: array-like of cluster labels for each data point
            consider_noise: bool, optional
                Whether to consider noise points (default is False).
        Returns:
            float: The CVNN index (higher is better)
        """
        if consider_noise:
            self._hal_cvnn_index = self._measures["hal_CVNN"].score(data, labels)
            return self._hal_cvnn_index

        mask = labels != -1  # Exclude noise points if any

        filtered_data = data[mask]
        filtered_labels = labels[mask]

        self._hal_cvnn_index = self._measures["hal_CVNN"].score(
            filtered_data, filtered_labels
        )
        return self._hal_cvnn_index

    def disco_index(self, data, labels, min_points: int = 5) -> float:
        """
        Calculates the DISCO index for clustering.

        Args:
            data: numpy array of shape (n_samples, n_features) - the clustered data
            labels: array-like of cluster labels for each data point
        Returns:
            float: The DISCO index (higher is better)
        """

        self._disco_index = disco.disco_score(data, labels, min_points=min_points)
        return self._disco_index

    def get_results(self) -> dict:
        """
        Retrieves the clustering metric results.

        Returns:
            dict: Dictionary containing the clustering metric results.
        """
        return {
            "silhouette_score": self._silhouette_score,
            "s_dbw_index": self._s_dbw_index,
            "dbcv_index": self._dbcv_index,
            "viaskde_index": self._viaskde_index,
            "dsi_index": self._dsi_index,
            "hal_cvnn_index": self._hal_cvnn_index,
            "disco_index": self._disco_index,
        }

    # Compatibility alias to avoid breaking existing code that uses getResults
    getResults = get_results
