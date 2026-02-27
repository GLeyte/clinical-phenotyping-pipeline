"""
Hierarchical Clustering Module

This module provides hierarchical agglomerative clustering functionality with:
- Dendrogram visualisation
- Support for multiple linkage methods
- Metric evaluation across different k values (clusters)
- Grid search over linkage parameters

Inherits from ClusterBaseHelper to share clustering utilities, visualisation
and metric calculation capabilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from ClusterBaseModule import ClusterBaseHelper

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_CHILDREN = 10
DEFAULT_LINKAGE_METHOD = "ward"
DEFAULT_FIGSIZE = (12, 8)
TQDM_POSITION = 0
METRIC_NAMES = {
    "silhouette": "Silhouette Score",
    "dbcv": "DBCV Index",
    "dsi": "DSI Index",
    "disco": "DISCO Index",
}
METRIC_OPTIMIZATION = {
    "silhouette": "max",
    "dbcv": "max",
    "dsi": "max",
    "disco": "max",
}
METRICS_TO_PLOT = ["silhouette", "dbcv", "dsi", "disco"]
ANNOTATION_BOXSTYLE = "round,pad=0.3"
ANNOTATION_COLOR = "yellow"
ANNOTATION_ALPHA = 0.7


class HierarchicalClusterHelper(ClusterBaseHelper):
    """
    Hierarchical agglomerative clustering helper.

    Extends ClusterBaseHelper with hierarchical-specific methods including
    dendrogram visualisation, linkage method evaluation and metric tracking
    across various cluster counts.

    Parameters
    ----------
    data : pd.DataFrame
        Input data for clustering
    features : List[str], optional
        Feature columns to use. If None, uses all numerical columns.
    features_not_considered : List[str], default=["COVID", "hadm_id", "subject_id"]
        Columns to exclude from clustering
    scaler : str, default="standard"
        Normalisation method: "standard", "minmax", "robust", or "none"
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        features_not_considered: Optional[List[str]] = None,
        scaler: str = "standard",
    ):
        if features_not_considered is None:
            features_not_considered = ["COVID", "hadm_id", "subject_id"]

        super().__init__(data, features, features_not_considered, scaler)

    # -------- Dendrogram Visualization --------

    def plot_dendrogram(
        self,
        scale_categorical: bool = False,
        max_children: int = DEFAULT_MAX_CHILDREN,
        linkage_method: str = DEFAULT_LINKAGE_METHOD,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        truncate_mode: str = "lastp",
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Plots dendrogram for hierarchical clustering.

        Parameters
        ----------
        scale_categorical : bool, default=False
            Whether to scale categorical features
        max_children : int, default=10
            Maximum number of leaf nodes to show in the dendrogram
        linkage_method : str, default="ward"
            Linkage criterion: "ward", "complete", "average", "single"
        figsize : Tuple[int, int], default=(12, 8)
            Figure size (width, height)
        truncate_mode : str, default="lastp"
            Dendrogram truncation mode
        dimensionality_reduction : Optional[Dict], default=None
            Dimensionality reduction configuration
        """
        data = self._update_data(
            scale_categorical, dimensionality_reduction=dimensionality_reduction
        )

        linkage_matrix = linkage(data, method=linkage_method)

        plt.figure(figsize=figsize)
        plt.title("Hierarchical Clustering Dendrogram (Pruned)")
        plt.xlabel("Number of points in node (or index of point if no parenthesis)")
        plt.ylabel("Distance")

        dendrogram(
            linkage_matrix,
            truncate_mode=truncate_mode,
            p=max_children,
            show_contracted=True,
        )

        plt.tight_layout()
        plt.show()

    # -------- Clustering --------

    def clustering(
        self,
        n_clusters: int,
        scale_categorical: bool = False,
        linkage_method: str = DEFAULT_LINKAGE_METHOD,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Performs hierarchical agglomerative clustering.

        Assigns cluster labels to the full dataset and stores the result
        in the clustered_data property.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form
        scale_categorical : bool, default=False
            Whether to scale categorical features
        linkage_method : str, default="ward"
            Linkage criterion for hierarchical clustering
        dimensionality_reduction : Optional[Dict], default=None
            Dimensionality reduction configuration
        data : Optional[pd.DataFrame], default=None
            Pre-processed data. If None, uses _update_data()
        """
        if data is None:
            data = self._update_data(
                scale_categorical, dimensionality_reduction=dimensionality_reduction
            )

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        cluster_labels = model.fit_predict(data)

        clustered_df = self.full_data.copy()
        clustered_df["Cluster"] = cluster_labels
        self.clustered_data = clustered_df

    # -------- Metric Evaluation --------

    def evaluate_k_range(
        self,
        max_k: int = 10,
        linkage_method: str = DEFAULT_LINKAGE_METHOD,
        figsize: Tuple[int, int] = (12, 10),
        scale_categorical: bool = False,
        dimensionality_reduction: Optional[Dict] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Evaluates clustering metrics for different k values (cluster counts).

        Calculates Silhouette, DBCV, DSI and DISCO metrics for k from 2 to max_k
        and visualises the results.

        Parameters
        ----------
        max_k : int, default=10
            Maximum number of clusters to test
        linkage_method : str, default="ward"
            Linkage criterion
        figsize : Tuple[int, int], default=(12, 10)
            Figure size for metric plots
        scale_categorical : bool, default=False
            Whether to scale categorical features
        dimensionality_reduction : Optional[Dict[str, Any]], default=None
            Dimensionality reduction configuration
        data : Optional[pd.DataFrame], default=None
            Pre-processed data
        """
        metrics_info = {
            "k": [],
            "silhouette": [],
            "dbcv": [],
            "dsi": [],
            "disco": [],
        }

        if data is None:
            data = self._update_data(
                scale_categorical, dimensionality_reduction=dimensionality_reduction
            )

        for k in tqdm(
            range(2, max_k + 1),
            desc="Calculating metrics per k",
            position=2,
            leave=False,
        ):
            self.clustering(
                n_clusters=k,
                scale_categorical=scale_categorical,
                linkage_method=linkage_method,
                data=data,
            )
            metrics = self.get_metrics()
            metrics_info["k"].append(k)
            metrics_info["silhouette"].append(metrics["silhouette"])
            metrics_info["dbcv"].append(metrics["dbcv"])
            metrics_info["dsi"].append(metrics["dsi"])
            metrics_info["disco"].append(metrics["disco"])

        self._plot_metrics(metrics_info, figsize=figsize, linkage_method=linkage_method)

    def grid_search_linkage_methods(
        self,
        linkage_methods: List[str],
        scale_categorical: bool = False,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """
        Grid search over linkage methods.

        Evaluates multiple linkage methods and their performance across
        different k values.

        Parameters
        ----------
        linkage_methods : List[str]
            List of linkage methods: "ward", "complete", "average", "single"
        scale_categorical : bool, default=False
            Whether to scale categorical features
        dimensionality_reduction : Optional[Dict], default=None
            Dimensionality reduction configuration
        figsize : Tuple[int, int], default=(12, 10)
            Figure size for plots

        Returns
        -------
        pd.DataFrame
            Empty dataframe (grid search results are visualised, not returned)

        Note
        ----
        Results are visualised using evaluate_k_range() for each linkage method.
        Failed linkage methods are skipped with a warning.
        """
        data = self._update_data(
            scale_categorical, dimensionality_reduction=dimensionality_reduction
        )

        for linkage_method in tqdm(
            linkage_methods, desc="Linkage Methods", position=TQDM_POSITION
        ):
            try:
                self.evaluate_k_range(
                    linkage_method=linkage_method,
                    scale_categorical=scale_categorical,
                    data=data,
                    figsize=figsize,
                )
                plt.suptitle(f"Linkage: {linkage_method}", fontweight="bold")
            except Exception as e:
                logger.warning(
                    f"Linkage method '{linkage_method}' failed: {type(e).__name__}: {e}"
                )
                continue

        return

    # -------- Visualization --------

    @staticmethod
    def _annotate_best_point(
        ax,
        best_config: int,
        best_score: float,
    ) -> None:
        """
        Annotate best score point on a plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to annotate
        best_config : int
            K value of best score
        best_score : float
            Best metric score
        """
        config_text = f"k={best_config}"
        annotation_text = f"{config_text}\n{best_score:.4f}"

        ax.scatter([best_config], [best_score], color="orange", s=100, zorder=5)
        ax.annotate(
            annotation_text,
            xy=(best_config, best_score),
            xytext=(10, -10),
            textcoords="offset points",
            bbox=dict(
                boxstyle=ANNOTATION_BOXSTYLE,
                fc=ANNOTATION_COLOR,
                alpha=ANNOTATION_ALPHA,
            ),
            fontsize=8,
        )

    def _plot_metrics(
        self,
        metrics_info: Dict,
        figsize: Tuple[int, int] = (12, 8),
        linkage_method: Optional[str] = None,
    ) -> None:
        """
        Plot clustering metrics across k values.

        Parameters
        ----------
        metrics_info : Dict
            Dictionary with keys: "k", "silhouette", "dbcv", "dsi", "disco"
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        linkage_method : Optional[str], default=None
            Linkage method name for title
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()

        for idx, metric in enumerate(METRICS_TO_PLOT):
            ax = axs[idx]

            ax.plot(
                metrics_info["k"],
                metrics_info[metric],
                "bo-",
                linewidth=2,
                markersize=8,
                label=METRIC_NAMES[metric],
            )

            if METRIC_OPTIMIZATION[metric] == "max":
                best_idx = np.argmax(metrics_info[metric])
            else:
                best_idx = np.argmin(metrics_info[metric])

            best_config = metrics_info["k"][best_idx]
            best_score = metrics_info[metric][best_idx]

            is_maximized = METRIC_OPTIMIZATION[metric] == "max"
            self._annotate_best_point(ax, best_config, best_score)

            ax.set_title(METRIC_NAMES[metric])
            ax.set_ylabel(METRIC_NAMES[metric])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        plt.suptitle(
            f"Clustering Metrics vs. Number of Clusters\nLinkage: {linkage_method}",
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    logger.info("HierarchicalClusterHelper loaded successfully")
