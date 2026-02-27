"""
SHADE (Structure-preserving High-dimensional Analysis with Density-based Exploration) Clustering Module

This module provides SHADE clustering functionality with:
- Autoencoder-based dimensionality reduction and clustering
- Configurable batch size and training epochs
- Optuna-based hyperparameter search
- Optuna dashboard visualisation with file persistence
- 2D visualisation of clustered data with optional PCA fallback

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
from sklearn.decomposition import PCA

try:
    from ExternalModules.ExternalClustering.SHADE.shade import SHADE

    HAS_SHADE = True
except ImportError:
    HAS_SHADE = False
    logging.warning("SHADE not available. SHADE clustering will be disabled.")

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

from ClusterBaseModule import ClusterBaseHelper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
_DEFAULT_PARAMETERS: Dict[str, Any] = {
    "batch_size": [256, 512, 1024],
    "clustering_epochs": [50, 100, 150],
    "clustering_lr": {"min": 1e-4, "max": 1e-2, "log": True},
}

_PENALTY_CLUSTER_COUNT = -2.0
_PENALTY_MAX_SIZE = -3.0


# ============================================================================
# SHADE Cluster Helper
# ============================================================================


class SHADEClusterHelper(ClusterBaseHelper):
    """
    SHADE (Structure-preserving High-dimensional Analysis with Density-based Exploration) clustering helper.

    Extends ClusterBaseHelper with SHADE-specific functionality including:
    - Autoencoder-based clustering with configurable training parameters
    - Optuna-based hyperparameter search
    - Persistent SQLite storage of optimisation studies
    - Visualisation of encoded data with automatic PCA fallback for high-dimensional data

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
        Scaling method: "standard", "minmax", 'robust' or "none"

    Attributes
    ----------
    VALID_METRICS : ClassVar[frozenset]
        Metrics supported for Optuna optimisation.
    VALID_SUFFIXES : ClassVar[frozenset]
        Valid suffixes for study naming.

    Examples
    --------
    >>> helper = SHADEClusterHelper(data=df, scaler='standard')
    >>> helper.clustering(batch_size=512, clustering_epochs=100)
    >>> metrics = helper.get_metrics()
    """

    VALID_METRICS: ClassVar[frozenset] = frozenset(
        {"dbcv", "disco", "dsi", "silhouette"}
    )
    VALID_SUFFIXES: ClassVar[frozenset] = frozenset({"all", "death"})

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        features_not_considered: Optional[List[str]] = None,
        scaler: str = "standard",
    ):
        """
        Initialises SHADEClusterHelper.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with data for clustering
        features : Optional[List[str]], default=None
            List of features to use. None = all numerical features.
        features_not_considered : Optional[List[str]], default=None
            Features to exclude from analysis.
            Default: ['COVID', 'hadm_id', 'subject_id']
        scaler : str, default="standard"
            Scaling method ('standard', 'minmax', 'robust', 'none')

        Raises
        ------
        TypeError
            If data is not a DataFrame
        ValueError
            If DataFrame is empty or scaler is invalid
        """
        if features_not_considered is None:
            features_not_considered = ["COVID", "hadm_id", "subject_id"]

        super().__init__(data, features, features_not_considered, scaler)
        self._model: Optional[Any] = None

        logger.info(
            f"SHADEClusterHelper initialized: {len(data)} records, scaler='{scaler}'"
        )

    # ========== Clustering ==========

    def clustering(
        self,
        scale_categorical: bool = False,
        random_state: int = 42,
        batch_size: int = 500,
        clustering_epochs: int = 100,
        pretrain_epochs: int = 0,
        pretrain_optimizer_params: Optional[Dict[str, Any]] = None,
        clustering_optimizer_params: Optional[Dict[str, Any]] = None,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Performs SHADE clustering and assigns cluster labels.

        Uses a hybrid stacked autoencoder to learn cluster assignments
        from high-dimensional data.

        Parameters
        ----------
        scale_categorical : bool, default=False
            Whether to scale categorical features.
            Ignored when `data` is provided directly.
        random_state : int, default=42
            Seed for reproducibility
        batch_size : int, default=500
            Batch size for autoencoder training
        clustering_epochs : int, default=100
            Number of epochs for the clustering phase
        pretrain_epochs : int, default=0
            Number of epochs for the pre-training phase
        pretrain_optimizer_params : Optional[Dict[str, Any]], default=None
            Parameters for the pre-training optimiser (e.g. {"lr": 1e-3})
        clustering_optimizer_params : Optional[Dict[str, Any]], default=None
            Parameters for the clustering optimiser (e.g. {"lr": 1e-3})
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Configuration for PCA or autoencoder reduction.
            Ignored when `data` is provided directly.
        data : Optional[pd.DataFrame], default=None
            Pre-processed data. If None, calls _update_data() internally.

        Raises
        ------
        ImportError
            If SHADE is not available

        Examples
        --------
        >>> helper.clustering(batch_size=512, clustering_epochs=100)
        """
        if not HAS_SHADE:
            raise ImportError(
                "SHADE is required for SHADE clustering. "
                "Install from ExternalModules/ExternalClustering/SHADE/"
            )

        if pretrain_optimizer_params is None:
            pretrain_optimizer_params = {"lr": 1e-3}
        if clustering_optimizer_params is None:
            clustering_optimizer_params = {"lr": 1e-3}

        if data is None:
            data = self._update_data(
                scale_categorical=scale_categorical,
                dimensionality_reduction=dimensionality_reduction,
            )

        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            data_array = np.array(data)

        # Create and fit SHADE model
        model = SHADE(
            random_state=random_state,
            batch_size=batch_size,
            clustering_epochs=clustering_epochs,
            pretrain_epochs=pretrain_epochs,
            pretrain_optimizer_params=pretrain_optimizer_params,
            clustering_optimizer_params=clustering_optimizer_params,
        )

        cluster_labels = model.fit(data_array).labels_
        self._model = model

        clustered = self.full_data.copy()
        clustered["Cluster"] = cluster_labels
        self.clustered_data = clustered

        logger.info(
            f"SHADE clustering complete: batch_size={batch_size}, "
            f"clustering_epochs={clustering_epochs}"
        )

    # ========== Dimensionality Reduction Visualization ==========

    def dimensions_reduction_plot(
        self,
        scale_categorical: bool = False,
        dimensions: Optional[int] = None,
    ) -> None:
        """
        Plots a 2D scatter plot of clustered data after dimensionality reduction.

        Uses the SHADE encoder to project data into latent space and then applies
        PCA if latent dimension is > 2.

        Parameters
        ----------
        scale_categorical: bool, default=False
            Whether to scale categorical features
        dimensions: Optional[int], default=None
            Target dimensionality for reduction. If None, uses all features.

        Raises
        ------
        ValueError
            If clustering has not been performed yet.

        Examples
        --------
        >>> helper.dimensions_reduction_plot(scale_categorical=True)
        """
        if self._model is None:
            raise ValueError("Clustering has not been performed yet.")

        data = self._update_data(
            scale_categorical=scale_categorical,
            dimensionality_reduction=(
                {"method": "pca", "n_components": dimensions} if dimensions else None
            ),
        )

        # Encode data using SHADE encoder
        reduced_data = self._model.encode(
            data.values if isinstance(data, pd.DataFrame) else data
        )

        if self.clustered_data is None:
            raise ValueError("Clustering has not been performed yet.")

        cluster_labels = self.clustered_data["Cluster"].values

        logger.info(f"Reduced data shape: {reduced_data.shape}")

        # Apply PCA if needed to reduce to 2D
        if reduced_data.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(reduced_data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=cluster_labels,
            cmap="tab10",
            alpha=0.7,
        )
        plt.title("2D Scatter Plot of Clustered Data (SHADE Encoding)")
        plt.xlabel("Encoded Dimension 1")
        plt.ylabel("Encoded Dimension 2")
        plt.colorbar(scatter, label="Cluster Label")
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
        Optimises SHADE hyperparameters using Optuna.

        Performs a search over `batch_size`, `clustering_epochs` and `clustering_lr`
        with configurable cluster size constraints.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier ('all', 'death').
        metric : str, default='dbcv'
            Metric to maximise ('dbcv', 'disco', 'dsi', 'silhouette').
        parameters : Optional[Dict[str, Any]], default=None
            Hyperparameter search space. Expected keys:
            - 'batch_size': list of batch sizes
            - 'clustering_epochs': list of epoch counts
            - 'clustering_lr': dict with 'min', 'max' and optionally 'log'
            Uses _DEFAULT_PARAMETERS if None.
        restrict_minsize_cluster : int, default=5
            Minimum cluster size (%) to be considered valid.
        restrict_maxsize_cluster : int, default=80
            Maximum cluster size (%) allowed for any cluster.
        restrict_min_cluster : int, default=2
            Minimum number of valid clusters required.
        restrict_max_cluster : int, default=10
            Maximum number of valid clusters allowed.
        n_trials : int, default=100
            Number of optimisation trials.
        info : bool, default=False
            If True, displays Optuna logs; if False, suppresses warnings.
        save_storage : bool, default=True
            If True, persists the study in SQLite in optuna-folder/.
        scale_categorical : bool, default=False
            Whether to scale categorical features.
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Dimensionality reduction configuration.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any], float]
            (trials_dataframe, best_params, best_metric_value)

        Raises
        ------
        ImportError
            If optuna is not available.
        ValueError
            If suffix or metric are invalid.

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
            """Optuna objective function for SHADE hyperparameter search."""
            batch_size = trial.suggest_categorical(
                "batch_size", parameters["batch_size"]
            )
            clustering_epochs = trial.suggest_categorical(
                "clustering_epochs", parameters["clustering_epochs"]
            )
            clustering_lr = trial.suggest_float(
                "clustering_lr",
                parameters["clustering_lr"]["min"],
                parameters["clustering_lr"]["max"],
                log=parameters["clustering_lr"].get("log", True),
            )

            self.clustering(
                scale_categorical=scale_categorical,
                batch_size=batch_size,
                clustering_epochs=clustering_epochs,
                clustering_optimizer_params={"lr": clustering_lr},
                data=data,
            )

            # Validate clustering result
            if (
                self.clustered_data is None
                or "Cluster" not in self.clustered_data.columns
            ):
                return _PENALTY_CLUSTER_COUNT

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
                db_filename = f"shade-{suffix}-{metric}-{counter}.sqlite3"
                db_path = os.path.join(optuna_dir, db_filename)
                if not os.path.exists(db_path):
                    break
                counter += 1

            storage_name = f"sqlite:///{db_path}"
            study_name = (
                f"Optuna Study SHADE {suffix.capitalize()} {metric.upper()} {counter}"
            )

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
            batch_size=best_params["batch_size"],
            clustering_epochs=best_params["clustering_epochs"],
            clustering_optimizer_params={"lr": best_params["clustering_lr"]},
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
        Launches the Optuna Dashboard to visualise optimisation results.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier used in the optimisation study.
        study_index : int, default=0
            Index of the SQLite database file to load.
        metric : str, default='dbcv'
            Metric used in the study.
        port : int, default=8080
            Port for the dashboard server.

        Raises
        ------
        ImportError
            If optuna_dashboard is not available.
        ValueError
            If the metric is invalid.
        FileNotFoundError
            If the SQLite study file does not exist.

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
        db_filename = f"shade-{suffix}-{metric}-{study_index}.sqlite3"
        db_path = os.path.join(optuna_dir, db_filename)

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"SQLite study file not found: {db_path}. "
                "Run optuna_grid_search with save_storage=True first."
            )

        logger.info(f"Launching Optuna Dashboard from: {db_path} on port {port}")
        run_server(storage=f"sqlite:///{db_path}", port=port)


if __name__ == "__main__":
    logger.info("SHADEClusterHelper module loaded successfully")
