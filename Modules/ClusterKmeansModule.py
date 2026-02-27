"""
K-Means Cluster Helper Module

This module provides specialised tools for K-Means clustering,
including metric evaluation across different k values and visualisations.

Inherits from ClusterBaseHelper for base cluster analysis functionality.

Date: 2026-01-19
"""

import logging
import os
import pickle
from typing import Optional, List, Dict, Tuple, Any
from warnings import filterwarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from tqdm.notebook import tqdm

# Import of base class
try:
    from ClusterBaseModule import ClusterBaseHelper

    HAS_CLUSTER_BASE = True
except ImportError:
    HAS_CLUSTER_BASE = False
    logging.error("ClusterBaseHelper not available")

# Configuration
filterwarnings("ignore")
logger = logging.getLogger(__name__)


class KmeansClusterHelper(ClusterBaseHelper):
    """
    Helper class especializada para clustering K-Means.

    This class extends ClusterBaseHelper with K-Means-specific
    functionality including:
    - Clustering K-Means com configuração flexível
    - Avaliação de múltiplos valores de k
    - Quality metric visualisation
    - Suporte para redução dimensional

    Attributes:
        DEFAULT_RANDOM_STATE: Random state padrão
        VALID_METRICS: Métricas válidas para avaliação

    Examples:
        >>> helper = KmeansClusterHelper(
        ...     data=df,
        ...     features=['age', 'weight', 'glucose'],
        ...     scaler='standard'
        ... )
        >>> helper.k_means(k=3)
        >>> helper.metrics_per_k(max_k=10)
    """

    # Class constants
    DEFAULT_RANDOM_STATE = 42
    VALID_METRICS = ["silhouette", "dbcv", "dsi", "disco"]
    METRIC_NAMES = {
        "silhouette": "Silhouette Score",
        "dbcv": "DBCV Index",
        "dsi": "DSI Index",
        "disco": "DISCO Index",
    }
    METRIC_BEST = {
        "silhouette": "max",
        "dbcv": "max",
        "dsi": "max",
        "disco": "max",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        features_not_considered: Optional[List[str]] = None,
        scaler: str = "standard",
    ):
        """
        Initialises KmeansClusterHelper.

        Args:
            data: DataFrame com dados para clustering
            features: Lista de features a usar (None = todas)
            features_not_considered: Features a excluir (padrão: ['COVID', 'hadm_id', 'subject_id'])
            scaler: Tipo de scaler ('standard', 'minmax', 'robust', 'none')

        Raises:
            TypeError: Se data não for DataFrame
            ValueError: Se DataFrame estiver vazio

        Examples:
            >>> helper = KmeansClusterHelper(
            ...     data=df,
            ...     features=['age', 'weight'],
            ...     scaler='standard'
            ... )
        """
        if features_not_considered is None:
            features_not_considered = ["COVID", "hadm_id", "subject_id"]

        # Initialise base class
        super().__init__(
            data=data,
            features=features,
            features_not_considered=features_not_considered,
            scaler=scaler,
        )

        # K-Means specific attributes
        self._current_k = None
        self._kmeans_model = None

        logger.info(
            f"KmeansClusterHelper initialized: {len(data)} records, "
            f"{len(self._features)} features"
        )

    @property
    def current_k(self) -> Optional[int]:
        """Returns the current number of clusters."""
        return self._current_k

    @property
    def kmeans_model(self) -> Optional[KMeans]:
        """Returns the current K-Means model."""
        return self._kmeans_model

    def get_clustered_data(self) -> pd.DataFrame:
        """
        Returns the clustered data with labels.

        Returns:
            DataFrame with 'Cluster' column

        Raises:
            ValueError: If clustering has not been run

        Examples:
            >>> helper.k_means(k=3)
            >>> clustered = helper.get_clustered_data()
        """
        if self._clustered_data is None:
            raise ValueError(
                "No clustering has been performed yet. " "Run k_means() first."
            )
        return self._clustered_data.copy()

    def k_means(
        self,
        k: int,
        params: Optional[Dict[str, Any]] = None,
        scale_categorical: bool = False,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> "KmeansClusterHelper":
        """
        Executa clustering K-Means com k clusters.

        Args:
            k: Número de clusters
            params: Parameters for KMeans (e.g. {'random_state': 42, 'max_iter': 300})
            scale_categorical: If True, scales categorical features
            dimensionality_reduction: Dict com configuração de redução dimensional
                Exemplo: {'method': 'PCA', 'dimensions': 10}
            data: DataFrame opcional (usa dados internos se None)

        Returns:
            Self para method chaining

        Raises:
            ValueError: Se k inválido

        Examples:
            >>> helper.k_means(k=3, params={'random_state': 42})
            >>> helper.k_means(
            ...     k=5,
            ...     dimensionality_reduction={'method': 'PCA', 'dimensions': 10}
            ... )
        """
        # Input validation
        if not isinstance(k, int) or k < 2:
            raise ValueError(f"k must be an integer >= 2, got {k}")

        # Prepare data
        if data is None:
            data = self._update_data(
                scale_categorical=scale_categorical,
                dimensionality_reduction=dimensionality_reduction,
            )

        # Configure parameters
        if params is None:
            params = {"random_state": self.DEFAULT_RANDOM_STATE}
        else:
            params = params.copy()

        params["n_clusters"] = k

        # Run K-Means
        try:
            kmeans = KMeans(**params)
            cluster_labels = kmeans.fit_predict(data)

            # Store results
            self._current_k = k
            self._kmeans_model = kmeans
            self._clustered_data = self.full_data.copy()
            self._clustered_data["Cluster"] = cluster_labels

            logger.info(
                f"K-Means clustering completed: k={k}, "
                f"inertia={kmeans.inertia_:.2f}"
            )

        except Exception as e:
            logger.error(f"K-Means clustering failed: {e}")
            raise

        return self

    def metrics_per_k(
        self,
        max_k: int = 10,
        min_k: int = 2,
        params: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (12, 10),
        scale_categorical: bool = False,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
        plot: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Calculates metrics for different k values.

        Evaluates clustering quality for k from min_k to max_k,
        calculating Silhouette, DBCV, DSI and DISCO for each value.

        Args:
            max_k: Maximum number of clusters
            min_k: Minimum number of clusters (default: 2)
            params: Parameters for KMeans
            figsize: Figure size
            scale_categorical: If True, scales categorical features
            dimensionality_reduction: Dimensionality reduction configuration
            show_progress: If True, shows progress bar

        Returns:
            Dictionary with metrics per k

        Raises:
            ValueError: If max_k < min_k
            ImportError: If metrics are not available

        Examples:
            >>> results = helper.metrics_per_k(
            ...     max_k=10,
            ...     min_k=2,
            ...     params={'random_state': 42}
            ... )
            >>> print(f"Best k: {results['k'][np.argmax(results['silhouette'])]}")
        """
        # Validação
        if max_k < min_k:
            raise ValueError(f"max_k ({max_k}) must be >= min_k ({min_k})")

        if not self.metrics_calculator:
            raise ImportError(
                "ClusterMetrics not available. " "Cannot calculate metrics."
            )

        # Initialise results
        results = {
            "k": [],
            "silhouette": [],
            "dbcv": [],
            "dsi": [],
            "disco": [],
        }

        # Prepare data once
        data = self._update_data(
            scale_categorical=scale_categorical,
            dimensionality_reduction=dimensionality_reduction,
        )

        logger.info(f"Calculating metrics for k={min_k} to k={max_k}")

        # Iterate over k values
        k_range = range(min_k, max_k + 1)
        if show_progress:
            k_range = tqdm(
                k_range,
                desc="Calculating metrics per k",
                position=1,
                leave=False,
            )

        for k in k_range:
            try:
                # Run clustering
                self.k_means(
                    k=k,
                    params=params,
                    scale_categorical=scale_categorical,
                    dimensionality_reduction=dimensionality_reduction,
                    data=data,
                )

                # Calculate metrics
                metrics = self.get_metrics()

                # Store results
                results["k"].append(k)
                results["silhouette"].append(metrics["silhouette"])
                results["dbcv"].append(metrics["dbcv"])
                results["dsi"].append(metrics["dsi"])
                results["disco"].append(metrics["disco"])

            except Exception as e:
                logger.warning(f"Failed to calculate metrics for k={k}: {e}")
                # Add NaN to maintain consistency
                results["k"].append(k)
                results["silhouette"].append(np.nan)
                results["dbcv"].append(np.nan)
                results["dsi"].append(np.nan)
                results["disco"].append(np.nan)

        # Visualise results (only if requested)
        if plot:
            self.plot_metrics(results, figsize=figsize)

        logger.info(
            f"Metrics calculation completed for {len(results['k'])} values of k"
        )

        return results

    def plot_metrics(
        self,
        results: Dict[str, List[float]],
        figsize: Tuple[int, int] = (12, 8),
        savepath: Optional[str] = None,
    ) -> None:
        """
        Visualiza métricas de clustering por valor de k.

        Cria grid 2x2 com plots de Silhouette, DBCV, DSI e DISCO,
        destacando o melhor valor de k para cada métrica.

        Args:
            results: Dicionário com métricas (retorno de metrics_per_k)
            figsize: Tamanho da figura
            savepath: Caminho para salvar figura (opcional)

        Examples:
            >>> results = helper.metrics_per_k(max_k=10)
            >>> helper.plot_metrics(results, savepath='metrics.png')
        """
        # Validação
        required_keys = ["k"] + self.VALID_METRICS
        missing = [k for k in required_keys if k not in results]
        if missing:
            raise ValueError(f"Missing required keys in results: {missing}")

        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("K-Means Clustering Metrics per k", fontsize=14, fontweight="bold")
        axs = axs.flatten()

        # Plot each metric
        for i, metric in enumerate(self.VALID_METRICS):
            ax = axs[i]

            # Filter valid values (non-NaN)
            valid_indices = ~np.isnan(results[metric])
            k_values = np.array(results["k"])[valid_indices]
            metric_values = np.array(results[metric])[valid_indices]

            if len(metric_values) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No valid data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Plot linha com marcadores
            ax.plot(
                k_values,
                metric_values,
                "bo-",
                linewidth=2,
                markersize=8,
                label=self.METRIC_NAMES[metric],
            )

            # Identify and highlight best value
            if self.METRIC_BEST[metric] == "max":
                best_idx = np.argmax(metric_values)
            else:
                best_idx = np.argmin(metric_values)

            best_k = k_values[best_idx]
            best_score = metric_values[best_idx]

            # Mark best point
            ax.scatter(
                [best_k],
                [best_score],
                color="orange",
                s=150,
                zorder=5,
                edgecolors="black",
                linewidths=2,
            )

            # Annotation
            ax.annotate(
                f"k={best_k}\n{best_score:.4f}",
                xy=(best_k, best_score),
                xytext=(10, -15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", alpha=0.8),
                fontsize=9,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

            # Formatação
            ax.set_xlabel("k (Number of Clusters)", fontsize=10)
            ax.set_ylabel(self.METRIC_NAMES[metric], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="best", fontsize=9)

            # Set x-axis limits
            if len(k_values) > 1:
                ax.set_xlim(k_values.min() - 0.5, k_values.max() + 0.5)

        plt.tight_layout()

        # Save figure if requested
        if savepath:
            try:
                abs_path = os.path.abspath(savepath)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                fig.savefig(abs_path, dpi=300, bbox_inches="tight")
                logger.info(f"Metrics plot saved to {abs_path}")
            except Exception as e:
                logger.error(f"Failed to save plot: {e}")

        plt.show()

        logger.info("Metrics plot displayed")

    def find_optimal_k(
        self,
        max_k: int = 10,
        min_k: int = 2,
        metric: str = "silhouette",
        params: Optional[Dict[str, Any]] = None,
        scale_categorical: bool = False,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, float]:
        """
        Find the optimal value of k based on a metric.

        Args:
            max_k: Maximum number of clusters
            min_k: Minimum number of clusters
            metric: Metric to optimize ('silhouette', 'dbcv', 'dsi', 'disco')
            params: Parameters for KMeans
            scale_categorical: If True, scales categorical features
            dimensionality_reduction: Dimensionality reduction configuration

        Returns:
            Tuple (best_k, best_score)

        Raises:
            ValueError: If metric is invalid

        Examples:
            >>> optimal_k, score = helper.find_optimal_k(
            ...     max_k=10,
            ...     metric='silhouette'
            ... )
            >>> print(f"Optimal k: {optimal_k} (score: {score:.3f})")
        """
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. " f"Choose from {self.VALID_METRICS}"
            )

        # Calculate metrics (without showing plot)
        logger.info(f"Finding optimal k based on {metric}")

        results = self.metrics_per_k(
            max_k=max_k,
            min_k=min_k,
            params=params,
            scale_categorical=scale_categorical,
            dimensionality_reduction=dimensionality_reduction,
            show_progress=False,
            plot=False,  # Do not generate plot; only calculate metrics
        )

        # Find best k
        metric_values = np.array(results[metric])
        valid_indices = ~np.isnan(metric_values)

        if not np.any(valid_indices):
            raise ValueError(f"No valid {metric} scores calculated")

        valid_values = metric_values[valid_indices]
        valid_k = np.array(results["k"])[valid_indices]

        if self.METRIC_BEST[metric] == "max":
            best_idx = np.argmax(valid_values)
        else:
            best_idx = np.argmin(valid_values)

        best_k = int(valid_k[best_idx])
        best_score = float(valid_values[best_idx])

        logger.info(f"Optimal k found: {best_k} " f"({metric}={best_score:.4f})")

        return best_k, best_score

    def save_model(
        self,
        filepath: str,
    ) -> None:
        """
        Salva o modelo K-Means atual em arquivo.

        Args:
            filepath: Caminho para salvar o modelo

        Raises:
            ValueError: Se modelo não foi treinado

        Examples:
            >>> helper.k_means(k=3)
            >>> helper.save_model('kmeans_model.pkl')
        """
        if self._kmeans_model is None:
            raise ValueError("No model to save. Run k_means() first.")

        try:
            import os

            abs_path = os.path.abspath(filepath)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            with open(abs_path, "wb") as f:
                pickle.dump(self._kmeans_model, f)

            logger.info(f"Model saved to {abs_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(
        self,
        filepath: str,
    ) -> None:
        """
        Loads a K-Means model from a file.

        Args:
            filepath: Path to the model file

        Raises:
            FileNotFoundError: If file does not exist

        Examples:
            >>> helper.load_model('kmeans_model.pkl')
            >>> predictions = helper.kmeans_model.predict(new_data)
        """
        try:
            import os

            abs_path = os.path.abspath(filepath)

            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Model file not found: {abs_path}")

            with open(abs_path, "rb") as f:
                self._kmeans_model = pickle.load(f)

            if hasattr(self._kmeans_model, "n_clusters"):
                self._current_k = self._kmeans_model.n_clusters

            logger.info(f"Model loaded from {abs_path} " f"(k={self._current_k})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# ==================== END OF CLASS ====================

if __name__ == "__main__":
    logger.info("KmeansClusterHelper module loaded successfully")
    logger.info("Version 2.0 - Refactored and optimized")
