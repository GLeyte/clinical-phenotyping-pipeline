"""
HDBSCAN Clustering Module

This module provides HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
functionality with:
- Configurable min_cluster_size, min_samples, and cluster_selection_method parameters
- Optuna-based hyperparameter search
- Metric evaluation over different parameter ranges
- Optuna dashboard visualisation with file persistence
- Proper handling of noise points (label -1)

Inherits from ClusterBaseHelper to share clustering utilities, visualisation
and metric calculation capabilities.

Date: 2026-02-19
"""

import contextlib
import logging
import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from hdbscan import HDBSCAN

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    logging.warning("hdbscan not available. HDBSCAN clustering will be disabled.")

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logging.warning("optuna not available. optuna_grid_search will be disabled.")

try:
    from optuna_dashboard import run_server

    HAS_OPTUNA_DASHBOARD = True
except ImportError:
    HAS_OPTUNA_DASHBOARD = False
    logging.warning(
        "optuna_dashboard not available. plot_optuna_dashboard will be disabled."
    )

try:
    from tqdm.notebook import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logging.warning("tqdm.notebook not available. Progress bars disabled.")

from ClusterBaseModule import ClusterBaseHelper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
_DEFAULT_PARAMETERS: Dict[str, Any] = {
    "min_cluster_size": {"min": 2, "max": 50},
    "min_samples": {"min": 1, "max": 20},
    "cluster_selection_method": "eom",
}

_PENALTY_CLUSTERING_FAILED = -1.0
_PENALTY_CLUSTER_COUNT = -2.0
_PENALTY_MAX_SIZE = -3.0

_ANNOTATION_BOXSTYLE = "round,pad=0.3"
_ANNOTATION_COLOR = "yellow"
_ANNOTATION_ALPHA = 0.7
_NOISE_LABEL = -1


# ============================================================================
# HDBSCAN Cluster Helper
# ============================================================================


class HDBSCANClusterHelper(ClusterBaseHelper):
    """
    HDBSCAN clustering helper with Optuna-based hyperparameter optimisation.

    Extends ClusterBaseHelper with HDBSCAN-specific functionality including:
    - HDBSCAN clustering with configurable min_cluster_size and min_samples
    - Support for different cluster selection methods (eom, leaf)
    - Proper handling of noise points (label -1)
    - Metric evaluation over different parameter ranges
    - Optuna-based hyperparameter search
    - Persistent SQLite storage of optimisation studies

    Parameters
    ----------
    data : pd.DataFrame
        Input data for clustering
    features : Optional[List[str]], default=None
        Feature columns to use. If None, uses all numerical columns.
    features_not_considered : Optional[List[str]], default=None
        Columns to exclude from clustering.
        Default: ['hadm_id', 'subject_id']
    scaler : str, default="standard"
        Normalisation method: "standard", "minmax", "robust", or "none"

    Attributes
    ----------
    VALID_METRICS : ClassVar[frozenset]
        Metrics supported for Optuna optimisation.
    VALID_SUFFIXES : ClassVar[frozenset]
        Valid suffixes for study naming.
    VALID_CLUSTER_SELECTION_METHODS : ClassVar[frozenset]
        Valid HDBSCAN cluster selection methods.

    Examples
    --------
    >>> helper = HDBSCANClusterHelper(data=df, scaler='standard')
    >>> helper.clustering(min_cluster_size=5, min_samples=5)
    >>> metrics = helper.get_metrics()
    """

    VALID_METRICS: ClassVar[frozenset] = frozenset(
        {"dbcv", "disco", "dsi", "silhouette"}
    )
    VALID_SUFFIXES: ClassVar[frozenset] = frozenset({"all", "death", "pca", "ae"})
    VALID_CLUSTER_SELECTION_METHODS: ClassVar[frozenset] = frozenset({"eom", "leaf"})

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        features_not_considered: Optional[List[str]] = None,
        scaler: str = "standard",
    ):
        """
        Initialises HDBSCANClusterHelper.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with data for clustering
        features : Optional[List[str]], default=None
            List of features to use. None = all numerical features.
        features_not_considered : Optional[List[str]], default=None
            Features to exclude from analysis.
            Default: ['hadm_id', 'subject_id']
        scaler : str, default='standard'
            Normalisation method ('standard', 'minmax', 'robust', 'none')

        Raises
        ------
        TypeError
            If data is not a DataFrame
        ValueError
            If DataFrame is empty or scaler is invalid
        """
        if features_not_considered is None:
            features_not_considered = ["hadm_id", "subject_id"]

        super().__init__(data, features, features_not_considered, scaler)
        logger.info(
            f"HDBSCANClusterHelper initialized: {len(data)} records, scaler='{scaler}'"
        )

    # ========== Clustering ==========

    def clustering(
        self,
        scale_categorical: bool = False,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        cluster_selection_method: str = "eom",
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Performs HDBSCAN clustering and assigns cluster labels.

        Parameters
        ----------
        scale_categorical : bool, default=False
            Whether to scale categorical features.
            Ignored when `data` is provided directly.
        min_cluster_size : int, default=5
            The minimum number of samples in a group for that group to be
            considered a cluster; groups smaller than this size
            will be left as noise.
        min_samples : Optional[int], default=None
            The number of samples in a neighbourhood for a point to be considered
            as a core point. If None, defaults to min_cluster_size.
        cluster_selection_method : str, default='eom'
            The method used to select clusters from the condensed tree.
            Options: 'eom' (Excess of Mass) or 'leaf'
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Configuration for PCA or autoencoder reduction.
            Ignored when `data` is provided directly.
        data : Optional[pd.DataFrame], default=None
            Pre-processed data. If None, calls _update_data() internally.

        Raises
        ------
        ImportError
            If hdbscan is not installed
        ValueError
            If cluster_selection_method is invalid

        Examples
        --------
        >>> helper.clustering(min_cluster_size=5, min_samples=5)
        """
        if not HAS_HDBSCAN:
            raise ImportError(
                "hdbscan is required for HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            )

        if cluster_selection_method not in self.VALID_CLUSTER_SELECTION_METHODS:
            raise ValueError(
                f"Invalid cluster_selection_method '{cluster_selection_method}'. "
                f"Choose from {self.VALID_CLUSTER_SELECTION_METHODS}"
            )

        if data is None:
            data = self._update_data(
                scale_categorical=scale_categorical,
                dimensionality_reduction=dimensionality_reduction,
            )

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )

        cluster_labels = model.fit_predict(data)

        clustered = self.full_data.copy()
        clustered["Cluster"] = cluster_labels
        self.clustered_data = clustered

        # Log statistics including noise points
        noise_count = np.sum(cluster_labels == _NOISE_LABEL)
        n_clusters = len(set(cluster_labels)) - (
            1 if _NOISE_LABEL in cluster_labels else 0
        )
        logger.info(
            f"HDBSCAN clustering complete: min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, clusters={n_clusters}, "
            f"noise_points={noise_count}"
        )

    # ========== Metric Visualization ==========

    def plot_metrics(
        self,
        metrics_info: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plots clustering metrics across different parameter configurations.

        Parameters
        ----------
        metrics_info : Dict[str, Any]
            Dictionary with keys: "config", "silhouette", "dbcv", "dsi", "disco"
            where "config" contains tuples (min_cluster_size, min_samples).
        figsize : Tuple[int, int], default=(12, 8)
            Figure size (width, height)

        Examples
        --------
        >>> helper.plot_metrics(metrics_info, figsize=(12, 8))
        """
        metrics = ["silhouette", "dbcv", "dsi", "disco"]
        metric_names = {
            "silhouette": "Silhouette Score",
            "dbcv": "DBCV Index",
            "dsi": "DSI Index",
            "disco": "DISCO Index",
        }
        metric_optimization = {
            "silhouette": "max",
            "dbcv": "max",
            "dsi": "max",
            "disco": "max",
        }

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()

        x_labels = [
            f"size={cfg[0]}\nsamples={cfg[1]}" for cfg in metrics_info["config"]
        ]
        x_positions = np.arange(len(x_labels))

        for idx, metric in enumerate(metrics):
            ax = axs[idx]

            ax.plot(
                x_positions,
                metrics_info[metric],
                "bo-",
                linewidth=2,
                markersize=8,
                label=metric_names[metric],
            )

            if metric_optimization[metric] == "max":
                best_idx = np.argmax(metrics_info[metric])
            else:
                best_idx = np.argmin(metrics_info[metric])

            best_config = metrics_info["config"][best_idx]
            best_score = metrics_info[metric][best_idx]

            ax.scatter([best_idx], [best_score], color="orange", s=100, zorder=5)

            config_text = (
                f"min_cluster_size={best_config[0]}\nmin_samples={best_config[1]}"
            )
            ax.annotate(
                f"{config_text}\n{best_score:.4f}",
                xy=(best_idx, best_score),
                xytext=(10, -10),
                textcoords="offset points",
                bbox=dict(
                    boxstyle=_ANNOTATION_BOXSTYLE,
                    fc=_ANNOTATION_COLOR,
                    alpha=_ANNOTATION_ALPHA,
                ),
                fontsize=8,
            )

            ax.set_title(metric_names[metric])
            ax.set_ylabel(metric_names[metric])
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{i}" for i in range(len(x_labels))], fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        plt.suptitle(
            "HDBSCAN Clustering Metrics vs. Configuration",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    # ========== Optuna Hyperparameter Search ==========

    def optuna_grid_search(
        self,
        suffix: str = "all",
        metric: str = "dbcv",
        parameters: Optional[Dict[str, Any]] = None,
        restrict_minsize_cluster: int = 5,
        restrict_maxsize_cluster: int = 80,
        restrict_min_cluster: int = 2,
        restrict_max_cluster: int = 10,
        n_trials: int = 100,
        info: bool = False,
        save_storage: bool = True,
        scale_categorical: bool = False,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
        """
        Optimises HDBSCAN hyperparameters using Optuna.

        Searches over min_cluster_size and min_samples with configurable cluster size
        constraints. Noise points (label -1) are automatically excluded from cluster count
        and size calculations.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier ('all', 'death', 'pca', 'ae')
        metric : str, default='dbcv'
            Metric to maximise ('dbcv', 'disco', 'dsi', 'silhouette')
        parameters : Optional[Dict[str, Any]], default=None
            Hyperparameter search space. Expected keys:
            - 'min_cluster_size': dict with 'min', 'max'
            - 'min_samples': dict with 'min', 'max'
            - 'cluster_selection_method': str (default 'eom')
            Uses _DEFAULT_PARAMETERS if None.
        restrict_minsize_cluster : int, default=5
            Minimum cluster size (%) to be considered valid
        restrict_maxsize_cluster : int, default=80
            Maximum cluster size (%) allowed for any cluster
        restrict_min_cluster : int, default=2
            Minimum number of valid clusters required
        restrict_max_cluster : int, default=10
            Maximum number of valid clusters allowed
        n_trials : int, default=100
            Number of optimisation trials
        info : bool, default=False
            If True, shows Optuna logs; if False, suppresses warnings
        save_storage : bool, default=True
            If True, persists study in SQLite in optuna-folder/
        scale_categorical : bool, default=False
            Whether to scale categorical features
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Dimensionality reduction configuration

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any], float]
            (trials_dataframe, best_parameters, best_metric_value)

        Raises
        ------
        ImportError
            If optuna is not available
        ValueError
            If suffix or metric are invalid

        Examples
        --------
        >>> df, params, score = helper.optuna_grid_search(
        ...     suffix='all', metric='silhouette', n_trials=50
        ... )
        """
        if not HAS_OPTUNA:
            raise ImportError(
                "optuna is required for optuna_grid_search. "
                "Install with: pip install optuna"
            )

        if suffix not in self.VALID_SUFFIXES:
            raise ValueError(
                f"Invalid suffix '{suffix}'. Choose from {self.VALID_SUFFIXES}"
            )
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. Choose from {self.VALID_METRICS}"
            )

        if parameters is None:
            parameters = _DEFAULT_PARAMETERS

        data = self._update_data(
            scale_categorical=scale_categorical,
            dimensionality_reduction=dimensionality_reduction,
        )

        def objective(trial: "optuna.Trial") -> float:
            """Optuna objective function for HDBSCAN hyperparameter search."""
            min_cluster_size = trial.suggest_int(
                "min_cluster_size",
                parameters["min_cluster_size"]["min"],
                parameters["min_cluster_size"]["max"],
            )
            min_samples = trial.suggest_int(
                "min_samples",
                parameters["min_samples"]["min"],
                parameters["min_samples"]["max"],
            )
            cluster_selection_method = parameters.get("cluster_selection_method", "eom")

            try:
                self.clustering(
                    scale_categorical=scale_categorical,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=cluster_selection_method,
                    dimensionality_reduction=dimensionality_reduction,
                    data=data,
                )
            except Exception as e:
                logger.warning(f"Clustering failed: {type(e).__name__}: {e}")
                return _PENALTY_CLUSTERING_FAILED

            # Validate clustering result
            if (
                self.clustered_data is None
                or "Cluster" not in self.clustered_data.columns
            ):
                return _PENALTY_CLUSTERING_FAILED

            # Apply cluster size constraints (excluding noise points)
            cluster_sizes = (
                self.clustered_data["Cluster"].value_counts(normalize=True) * 100
            )
            cluster_sizes = cluster_sizes[cluster_sizes.index != _NOISE_LABEL]
            cluster_sizes = cluster_sizes[cluster_sizes > restrict_minsize_cluster]

            if (
                len(cluster_sizes) < restrict_min_cluster
                or len(cluster_sizes) > restrict_max_cluster
            ):
                return _PENALTY_CLUSTER_COUNT

            if cluster_sizes.max() > restrict_maxsize_cluster:
                return _PENALTY_MAX_SIZE

            return self.single_metric(metric_name=metric)

        # Build storage and study arguments
        storage_name: Optional[str] = None
        study_name: Optional[str] = None

        if save_storage:
            optuna_dir = os.path.abspath("optuna-folder")
            os.makedirs(optuna_dir, exist_ok=True)

            counter = 0
            while True:
                db_filename = f"hdbscan-{suffix}-{metric}-{counter}.sqlite3"
                db_path = os.path.join(optuna_dir, db_filename)
                if not os.path.exists(db_path):
                    break
                counter += 1

            storage_name = f"sqlite:///{db_path}"
            study_name = (
                f"Optuna Study HDBSCAN {suffix.capitalize()} {metric.upper()} {counter}"
            )

        # Create and run study
        warn_ctx = warnings.catch_warnings() if not info else contextlib.nullcontext()

        with warn_ctx:
            if not info:
                warnings.simplefilter("ignore")

            if save_storage:
                study = optuna.create_study(
                    direction="maximize",
                    study_name=study_name,
                    storage=storage_name,
                )
            else:
                study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial

        # Restore instance state to best trial's clustering
        best_params = best_trial.params
        self.clustering(
            scale_categorical=scale_categorical,
            min_cluster_size=best_params["min_cluster_size"],
            min_samples=best_params["min_samples"],
            cluster_selection_method=parameters.get("cluster_selection_method", "eom"),
            dimensionality_reduction=dimensionality_reduction,
            data=data,
        )

        logger.info(f"Optuna search complete: best_params={best_params}")
        logger.info(f"Best {metric.upper()} score: {best_trial.value:.4f}")

        best_value = best_trial.value
        if best_value is None:
            raise RuntimeError("Best trial has no recorded value.")
        return study.trials_dataframe(), best_params, best_value

    # ========== Optuna Dashboard ==========

    def plot_optuna_dashboard(
        self,
        suffix: str = "all",
        metric: str = "dbcv",
        study_index: int = 0,
        port: int = 8080,
    ) -> None:
        """
        Launches the Optuna Dashboard to visualise optimisation results.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier used in the optimisation study
        metric : str, default='dbcv'
            Metric used in the study
        study_index : int, default=0
            Index of the SQLite database file to load
        port : int, default=8080
            Port for the dashboard server

        Raises
        ------
        ImportError
            If optuna_dashboard is not available
        ValueError
            If metric is invalid
        FileNotFoundError
            If the SQLite study file does not exist

        Examples
        --------
        >>> helper.plot_optuna_dashboard(suffix='all', metric='dbcv', port=8080)
        """
        if not HAS_OPTUNA_DASHBOARD:
            raise ImportError(
                "optuna_dashboard is required. "
                "Install with: pip install optuna-dashboard"
            )

        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. Choose from {self.VALID_METRICS}"
            )

        optuna_dir = os.path.abspath("optuna-folder")
        db_filename = f"hdbscan-{suffix}-{metric}-{study_index}.sqlite3"
        db_path = os.path.join(optuna_dir, db_filename)

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"SQLite study file not found: {db_path}. "
                "Run optuna_grid_search with save_storage=True first."
            )

        logger.info(f"Launching Optuna Dashboard from: {db_path} on port {port}")
        run_server(storage=f"sqlite:///{db_path}", port=port)


if __name__ == "__main__":
    logger.info("HDBSCANClusterHelper module loaded successfully")
