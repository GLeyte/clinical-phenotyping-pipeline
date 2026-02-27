"""
GMM Clustering Module

This module provides tools for clustering with Gaussian Mixture Models (GMM),
including hyperparameter search via Optuna and visualisation of results in
the Optuna Dashboard.

Inherits from ClusterBaseHelper for base cluster analysis functionality.

Date: 2026-02-19
"""

import contextlib
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.mixture import GaussianMixture

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

logger = logging.getLogger(__name__)


# ==================== CONSTANTS ====================

_DEFAULT_PARAMETERS: Dict[str, Any] = {
    "n_components": {"min": 2, "max": 10},
    "covariance_type": ["full", "tied", "diag", "spherical"],
}

# Penalty values returned inside the Optuna objective when hard constraints are violated.
# Must be below any realistic metric score so these trials are never selected as best.
_PENALTY_CLUSTER_COUNT = -2.0
_PENALTY_MAX_SIZE = -3.0


# ==================== MAIN CLASS ====================


class GmmClusterHelper(ClusterBaseHelper):
    """
    Specialised helper class for Gaussian Mixture Model (GMM) clustering.

    Extends ClusterBaseHelper with GMM-specific functionality:
    - GMM clustering with flexible covariance configuration
    - Hyperparameter search via Optuna (n_components, covariance_type)
    - Optuna study persistence in SQLite for the dashboard

    Attributes
    ----------
    VALID_COVARIANCE_TYPES : set
        Covariance types accepted by sklearn GMM.
    VALID_METRICS : set
        Metrics accepted for Optuna optimisation.
    VALID_SUFFIXES : set
        Valid suffixes for file naming.

    Examples
    --------
    >>> helper = GmmClusterHelper(data=df, scaler='standard')
    >>> helper.clustering(n_components=3, covariance_type='full')
    >>> stats = helper.get_stats_categorical()
    """

    VALID_COVARIANCE_TYPES = {"full", "tied", "diag", "spherical"}
    VALID_METRICS = {"dbcv", "disco", "dsi", "silhouette"}
    VALID_SUFFIXES = {"all", "death", "pca", "ae"}

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        features_not_considered: Optional[List[str]] = None,
        scaler: str = "standard",
    ):
        """
        Initialises GmmClusterHelper.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with data for clustering.
        features : Optional[List[str]], default=None
            List of features to use. None = all numerical features.
        features_not_considered : Optional[List[str]], default=None
            Features to exclude from analysis.
            Default: ['COVID', 'hadm_id', 'subject_id'].
        scaler : str, default='standard'
            Scaler type ('standard', 'minmax', 'robust', 'none').

        Raises
        ------
        TypeError
            If data is not a DataFrame.
        ValueError
            If DataFrame is empty or scaler is invalid.

        Examples
        --------
        >>> helper = GmmClusterHelper(df, scaler='standard')
        """
        if features_not_considered is None:
            features_not_considered = ["COVID", "hadm_id", "subject_id"]

        super().__init__(data, features, features_not_considered, scaler)
        logger.info(
            f"GmmClusterHelper initialized: {len(data)} records, scaler='{scaler}'"
        )

    # ==================== CLUSTERING ====================

    def clustering(
        self,
        n_components: int,
        scale_categorical: bool = False,
        covariance_type: str = "full",
        random_state: int = 42,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Fits a Gaussian Mixture Model and assigns component labels.

        Parameters
        ----------
        n_components : int
            Number of GMM components (clusters).
        scale_categorical : bool, default=False
            If True, also scales categorical features.
            Ignored when `data` is provided directly.
        covariance_type : str, default='full'
            GMM covariance type.
            Options: 'full', 'tied', 'diag', 'spherical'.
        random_state : int, default=42
            Seed for reproducibility.
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Optional dimensionality reduction configuration (PCA or AE).
            Ignored when `data` is provided directly.
        data : Optional[pd.DataFrame], default=None
            Pre-processed DataFrame. If None, calls _update_data() internally.

        Returns
        -------
        float
            BIC score of the fitted model (lower value indicates better fit).

        Raises
        ------
        ValueError
            If covariance_type is invalid.

        Examples
        --------
        >>> bic = helper.clustering(n_components=3, covariance_type='full')
        """
        if covariance_type not in self.VALID_COVARIANCE_TYPES:
            raise ValueError(
                f"Invalid covariance_type '{covariance_type}'. "
                f"Choose from {self.VALID_COVARIANCE_TYPES}"
            )

        if data is None:
            data = self._update_data(
                scale_categorical=scale_categorical,
                dimensionality_reduction=dimensionality_reduction,
            )

        model = GaussianMixture(
            n_components=n_components,
            init_params="k-means++",
            covariance_type=covariance_type,
            random_state=random_state,
        )

        # fit_predict fits and predicts in a single pass — no need for a separate fit()
        cluster_labels = model.fit_predict(data)

        clustered = self.full_data.copy()
        clustered["Cluster"] = cluster_labels
        self.clustered_data = clustered

        bic = model.bic(data)

        logger.info(
            f"GMM clustering complete: n_components={n_components}, "
            f"covariance_type='{covariance_type}', BIC={bic:.2f}"
        )
        return bic

    # ==================== OPTUNA SEARCH ====================

    def optuna_grid_search(
        self,
        suffix: str = "all",
        metric: str = "dbcv",
        parameters: Optional[Dict[str, Any]] = None,
        restrict_minsize_cluster: int = 2,
        restrict_maxsize_cluster: int = 80,
        restrict_min_cluster: int = 2,
        restrict_max_cluster: int = 15,
        n_trials: int = 100,
        info: bool = False,
        save_storage: bool = True,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        scale_categorical: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
        """
        GMM hyperparameter search via Optuna with cluster size constraints.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier ('all', 'death', 'pca', 'ae').
        metric : str, default='dbcv'
            Metric to maximise ('dbcv', 'disco', 'dsi', 'silhouette').
        parameters : Optional[Dict[str, Any]], default=None
            Search space. Expected keys:
            - 'n_components': dict with 'min' and 'max' (int)
            - 'covariance_type': list of strings
            Uses _DEFAULT_PARAMETERS if None.
        restrict_minsize_cluster : int, default=2
            Minimum cluster size (%) to be valid.
        restrict_maxsize_cluster : int, default=80
            Maximum size allowed for any cluster (%).
        restrict_min_cluster : int, default=2
            Minimum number of valid clusters required.
        restrict_max_cluster : int, default=15
            Maximum number of valid clusters allowed.
        n_trials : int, default=100
            Number of Optuna trials.
        info : bool, default=False
            If True, displays Optuna logs and warnings. If False, suppresses them.
        save_storage : bool, default=True
            If True, persists the study in SQLite in optuna-folder/.
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Optional dimensionality reduction configuration.
        scale_categorical : bool, default=False
            If True, scales categorical features.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any], float]
            Tuple (trials_dataframe, best_parameters, best_metric_value).

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
                "Install it with: pip install optuna"
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
            """Optuna objective function for GMM hyperparameter search."""
            n_components = trial.suggest_int(
                "n_components",
                parameters["n_components"]["min"],
                parameters["n_components"]["max"],
            )
            covariance_type = trial.suggest_categorical(
                "covariance_type", parameters["covariance_type"]
            )

            self.clustering(
                n_components=n_components,
                covariance_type=covariance_type,
                data=data,
            )

            if self.clustered_data is None:
                raise RuntimeError("clustering() did not set clustered_data")

            cluster_sizes = (
                self.clustered_data["Cluster"].value_counts(normalize=True) * 100
            )
            cluster_sizes = cluster_sizes[cluster_sizes > restrict_minsize_cluster]

            # Check cluster count constraints
            if (
                len(cluster_sizes) < restrict_min_cluster
                or len(cluster_sizes) > restrict_max_cluster
            ):
                return _PENALTY_CLUSTER_COUNT

            # Check max cluster size constraint
            if cluster_sizes.max() > restrict_maxsize_cluster:
                return _PENALTY_MAX_SIZE

            return self.single_metric(metric_name=metric)

        storage_name: Optional[str] = None
        study_name: Optional[str] = None

        if save_storage:
            optuna_folder = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "optuna-folder")
            )
            os.makedirs(optuna_folder, exist_ok=True)

            counter = 0
            while True:
                db_filename = f"gmm-{suffix}-{metric}-{counter}.sqlite3"
                db_path = os.path.join(optuna_folder, db_filename)
                if not os.path.exists(db_path):
                    break
                counter += 1

            storage_name = f"sqlite:///{db_path}"
            study_name = (
                f"Optuna Study GMM {suffix.capitalize()} {metric.upper()} {counter}"
            )

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

        # Restore instance state to the best trial's clustering result
        self.clustering(
            n_components=best_trial.params["n_components"],
            covariance_type=best_trial.params["covariance_type"],
            data=data,
        )

        logger.info(f"Optuna search complete: best_params={best_trial.params}")
        logger.info(f"Best {metric.upper()} score: {best_trial.value:.4f}")

        best_value = best_trial.value
        if best_value is None:
            raise RuntimeError("Best trial has no recorded value.")
        return study.trials_dataframe(), best_trial.params, best_value

    # ==================== DASHBOARD ====================

    def plot_optuna_dashboard(
        self,
        suffix: str = "all",
        study_index: int = 0,
        port: int = 8080,
        metric: str = "dbcv",
    ) -> None:
        """
        Launches the Optuna Dashboard server to visualise optimisation results.

        Parameters
        ----------
        suffix : str, default='all'
            Feature set identifier used in the search.
        study_index : int, default=0
            Numeric index of the SQLite file to load
            (generated by optuna_grid_search).
        port : int, default=8080
            Dashboard server port.
        metric : str, default='dbcv'
            Metric used in the corresponding Optuna study.

        Raises
        ------
        ImportError
            If optuna_dashboard is not available.
        FileNotFoundError
            If the SQLite file does not exist.

        Examples
        --------
        >>> helper.plot_optuna_dashboard(suffix='all', study_index=0, metric='dbcv')
        """
        if not HAS_OPTUNA_DASHBOARD:
            raise ImportError(
                "optuna_dashboard is required. "
                "Install it with: pip install optuna-dashboard"
            )

        optuna_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "optuna-folder")
        )
        db_path = os.path.join(
            optuna_folder, f"gmm-{suffix}-{metric}-{study_index}.sqlite3"
        )

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"SQLite study file not found: {db_path}. "
                "Run optuna_grid_search with save_storage=True first."
            )

        logger.info(f"Launching Optuna Dashboard from: {db_path} on port {port}")
        run_server(storage=f"sqlite:///{db_path}", port=port)


# ==================== END OF CLASS ====================

if __name__ == "__main__":
    logger.info("GmmClusterHelper module loaded successfully")
