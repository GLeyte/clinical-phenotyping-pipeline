"""
Spectral Clustering Module

This module provides spectral clustering functionality with:
- Flexible affinity metrics (RBF, nearest neighbours)
- Configurable gamma and n_neighbors parameters
- Timeout handling for long-running optimisations
- Optuna-based hyperparameter search
- Optuna dashboard visualisation with file persistence

Inherits from ClusterBaseHelper to share clustering utilities, visualisation
and metric calculation capabilities.

Date: 2026-02-19
"""

import contextlib
import logging
import os
import signal
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

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

# Platform detection for signal.SIGALRM availability
_HAS_SIGALRM = hasattr(signal, "SIGALRM")

# Constants
_DEFAULT_TIMEOUT_SECONDS = 10
_PENALTY_TIMEOUT = -5.0
_PENALTY_CLUSTERING_FAILED = -4.0
_PENALTY_CLUSTER_COUNT = -3.0
_PENALTY_MAX_SIZE = -2.0

_DEFAULT_PARAMETERS: Dict[str, Any] = {
    "affinity": ["rbf", "nearest_neighbors"],
    "gamma": {"min": 1e-3, "max": 1e2, "log": True},
    "n_neighbors": {"min": 2, "max": 100},
    "n_clusters": {"min": 2, "max": 10},
}


# ============================================================================
# Timeout Exception
# ============================================================================


class TimeoutException(Exception):
    """Raised when spectral clustering exceeds the timeout limit."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException()


# ============================================================================
# Spectral Cluster Helper
# ============================================================================


class SpectralClusterHelper(ClusterBaseHelper):
    """
    Spectral clustering helper with Optuna-based hyperparameter optimisation.

    Extends ClusterBaseHelper with spectral-specific functionality including:
    - Spectral clustering with configurable affinity and parameters
    - Timeout handling for problematic affinity matrices
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
        Default: ['COVID', 'hadm_id', 'subject_id']
    scaler : str, default="standard"
        Normalisation method: "standard", "minmax", "robust", or "none"

    Attributes
    ----------
    VALID_AFFINITY_TYPES : ClassVar[frozenset]
        Affinity metrics supported for spectral clustering.
    VALID_METRICS : ClassVar[frozenset]
        Metrics supported for Optuna optimisation.
    VALID_SUFFIXES : ClassVar[frozenset]
        Valid suffixes for study naming.

    Examples
    --------
    >>> helper = SpectralClusterHelper(data=df, scaler='standard')
    >>> helper.clustering(n_clusters=3, affinity='rbf', gamma=1.0)
    >>> metrics = helper.get_metrics()
    """

    VALID_AFFINITY_TYPES: ClassVar[frozenset] = frozenset({"rbf", "nearest_neighbors"})
    VALID_METRICS: ClassVar[frozenset] = frozenset(
        {"dbcv", "disco", "dsi", "silhouette"}
    )
    VALID_SUFFIXES: ClassVar[frozenset] = frozenset({"all", "death", "pca", "ae"})

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        features_not_considered: Optional[List[str]] = None,
        scaler: str = "standard",
    ):
        """
        Initialize SpectralClusterHelper.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with data for clustering
        features : Optional[List[str]], default=None
            List of features to use. None = all numeric features.
        features_not_considered : Optional[List[str]], default=None
            Features to exclude from analysis.
            Default: ['hadm_id', 'subject_id']
        scaler : str, default='standard'
            Scaling method ('standard', 'minmax', 'robust', 'none')

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
            f"SpectralClusterHelper initialized: {len(data)} records, scaler='{scaler}'"
        )

    # ========== Clustering ==========

    def clustering(
        self,
        n_clusters: int,
        scale_categorical: bool = False,
        gamma: float = 1.0,
        affinity: str = "rbf",
        n_neighbors: int = 10,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Perform spectral clustering and assign cluster labels.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form
        scale_categorical : bool, default=False
            Whether to scale categorical features.
            Ignored when `data` is provided directly.
        gamma : float, default=1.0
            Gamma parameter for RBF affinity.
            Ignored if affinity='nearest_neighbors'
        affinity : str, default='rbf'
            Affinity metric: 'rbf' or 'nearest_neighbors'
        n_neighbors : int, default=10
            Number of neighbors for nearest_neighbors affinity.
            Ignored if affinity='rbf'
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Configuration for PCA or autoencoder reduction.
            Ignored when `data` is provided directly.
        random_state : int, default=42
            Seed for reproducibility
        data : Optional[pd.DataFrame], default=None
            Pre-processed data. If None, calls _update_data() internally.

        Raises
        ------
        ValueError
            If affinity is not in VALID_AFFINITY_TYPES

        Examples
        --------
        >>> helper.clustering(n_clusters=3, affinity='rbf', gamma=1.0)
        """
        if affinity not in self.VALID_AFFINITY_TYPES:
            raise ValueError(
                f"Invalid affinity '{affinity}'. "
                f"Choose from {self.VALID_AFFINITY_TYPES}"
            )

        if data is None:
            data = self._update_data(
                scale_categorical=scale_categorical,
                dimensionality_reduction=dimensionality_reduction,
            )

        model = SpectralClustering(
            n_clusters=n_clusters,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            n_jobs=-1,
            random_state=random_state,
        )

        cluster_labels = model.fit_predict(data)

        clustered = self.full_data.copy()
        clustered["Cluster"] = cluster_labels
        self.clustered_data = clustered

        logger.info(
            f"Spectral clustering complete: n_clusters={n_clusters}, "
            f"affinity='{affinity}', gamma={gamma}, n_neighbors={n_neighbors}"
        )

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
        Optimize spectral clustering hyperparameters using Optuna.

        Searches over affinity type, gamma, n_neighbors, and n_clusters with
        configurable cluster size constraints.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier ('all', 'death', 'pca', 'ae')
        metric : str, default='dbcv'
            Metric to maximize ('dbcv', 'disco', 'dsi', 'silhouette')
        parameters : Optional[Dict[str, Any]], default=None
            Hyperparameter search space. Expected keys:
            - 'affinity': list of affinity types
            - 'gamma': dict with 'min', 'max', optional 'log' (for rbf)
            - 'n_neighbors': dict with 'min', 'max' (for nearest_neighbors)
            - 'n_clusters': dict with 'min', 'max'
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
            Number of optimization trials
        info : bool, default=False
            If True, show Optuna logs; if False, suppress warnings
        save_storage : bool, default=True
            If True, persist study to SQLite in optuna-folder/
        scale_categorical : bool, default=False
            Whether to scale categorical features
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Dimensionality reduction configuration

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any], float]
            (trials_dataframe, best_params, best_metric_value)

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

        # Register signal handler for timeout (Unix only)
        if _HAS_SIGALRM:
            signal.signal(signal.SIGALRM, _timeout_handler)

        def objective(trial: "optuna.Trial") -> float:
            """Optuna objective function for spectral hyperparameter search."""
            affinity = trial.suggest_categorical("affinity", parameters["affinity"])

            if affinity == "rbf":
                gamma = trial.suggest_float(
                    "gamma",
                    parameters["gamma"]["min"],
                    parameters["gamma"]["max"],
                    log=parameters["gamma"].get("log", True),
                )
                n_clusters = trial.suggest_int(
                    "n_clusters",
                    parameters["n_clusters"]["min"],
                    parameters["n_clusters"]["max"],
                )
                n_neighbors = 10
            else:  # nearest_neighbors
                n_neighbors = trial.suggest_int(
                    "n_neighbors",
                    parameters["n_neighbors"]["min"],
                    parameters["n_neighbors"]["max"],
                )
                n_clusters = trial.suggest_int(
                    "n_clusters",
                    parameters["n_clusters"]["min"],
                    parameters["n_clusters"]["max"],
                )
                gamma = 1.0

            try:
                if _HAS_SIGALRM:
                    signal.alarm(_DEFAULT_TIMEOUT_SECONDS)

                self.clustering(
                    scale_categorical=scale_categorical,
                    n_clusters=n_clusters,
                    gamma=gamma,
                    affinity=affinity,
                    n_neighbors=n_neighbors,
                    data=data,
                )

                if _HAS_SIGALRM:
                    signal.alarm(0)

            except TimeoutException:
                logger.warning(
                    f"Timeout: affinity={affinity}, gamma={gamma}, "
                    f"n_neighbors={n_neighbors}"
                )
                return _PENALTY_TIMEOUT
            except Exception as e:
                logger.warning(f"Clustering failed: {type(e).__name__}: {e}")
                return _PENALTY_TIMEOUT

            # Validate clustering result
            if (
                self.clustered_data is None
                or "Cluster" not in self.clustered_data.columns
            ):
                return _PENALTY_CLUSTERING_FAILED

            # Apply cluster size constraints
            cluster_sizes = (
                self.clustered_data["Cluster"].value_counts(normalize=True) * 100
            )
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
                db_filename = f"spectral-{suffix}-{metric}-{counter}.sqlite3"
                db_path = os.path.join(optuna_dir, db_filename)
                if not os.path.exists(db_path):
                    break
                counter += 1

            storage_name = f"sqlite:///{db_path}"
            study_name = f"Optuna Study Spectral {suffix.capitalize()} {metric.upper()} {counter}"

        # Create and run study
        warning_ctx = (
            warnings.catch_warnings() if not info else contextlib.nullcontext()
        )
        with warning_ctx:
            if not info:
                warnings.simplefilter("ignore")

            study_kwargs: Dict[str, Any] = {"direction": "maximize"}
            if save_storage:
                study_kwargs["study_name"] = study_name
                study_kwargs["storage"] = storage_name

            study = optuna.create_study(**study_kwargs)
            study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial

        # Restore instance state to best trial's clustering
        best_params = best_trial.params
        self.clustering(
            n_clusters=best_params["n_clusters"],
            affinity=best_params["affinity"],
            gamma=best_params.get("gamma", 1.0),
            n_neighbors=best_params.get("n_neighbors", 10),
            scale_categorical=scale_categorical,
            dimensionality_reduction=dimensionality_reduction,
            data=data,
        )

        logger.info(f"Optuna search complete: best_params={best_params}")
        logger.info(f"Best {metric.upper()} score: {best_trial.value:.4f}")

        best_value = best_trial.value
        if best_value is None:
            raise RuntimeError("Best trial has no recorded value.")
        return study.trials_dataframe(), best_trial.params, best_value

    # ========== Optuna Dashboard ==========

    def plot_optuna_dashboard(
        self,
        suffix: str = "all",
        study_index: int = 0,
        metric: str = "dbcv",
        port: int = 8080,
    ) -> None:
        """
        Launch Optuna Dashboard to visualize optimization results.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier used in the optimization study
        study_index : int, default=0
            Index of the SQLite database file to load
        metric : str, default='dbcv'
            Metric used in the study
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
        db_filename = f"spectral-{suffix}-{metric}-{study_index}.sqlite3"
        db_path = os.path.join(optuna_dir, db_filename)

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"SQLite study file not found: {db_path}. "
                "Run optuna_grid_search with save_storage=True first."
            )

        logger.info(f"Launching Optuna Dashboard from: {db_path} on port {port}")
        run_server(storage=f"sqlite:///{db_path}", port=port)


if __name__ == "__main__":
    logger.info("SpectralClusterHelper module loaded successfully")
