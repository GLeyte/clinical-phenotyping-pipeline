"""
Cluster Base Helper Module - Cluster Analysis and Visualisation

This module provides tools for statistical analysis, comparison and visualisation
of clusters in medical data, including dimensionality reduction (PCA, UMAP, Autoencoder)
and clustering quality metrics.

Date: 2026-01-19
"""

import os
import re
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats as scipy_stats

# Conditional imports (may not be available)
try:
    import ClusterMetricsModule as cm

    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    logging.warning("ClusterMetricsModule not available. Metrics will be disabled.")

try:
    from ExternalModules.ExternalClustering.SHADE.shade import SHADE

    HAS_SHADE = True
except ImportError:
    HAS_SHADE = False
    logging.warning("SHADE not available. Autoencoder clustering will be disabled.")

try:
    from config import (
        DATAPATH,
        COVID_TRAIN_FILE,
        COVID_TEST_FILE,
        NORMAL_VALUES,
        CATEGORICAL,
        NUMERICAL,
    )

    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    DATAPATH: str = ""
    COVID_TRAIN_FILE: str = ""
    COVID_TEST_FILE: str = ""
    NORMAL_VALUES: dict = {}
    CATEGORICAL: dict = {}
    NUMERICAL: dict = {}
    logging.warning("config module not available. Some features may be limited.")

# Logging configuration
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION AND DATACLASSES ====================


@dataclass
class VisualizationConfig:
    """Configuration for cluster visualisations."""

    figsize: Tuple[int, int] = (10, 10)
    alpha: float = 0.7
    palette: str = "tab10"
    dpi: int = 300
    show_outliers: bool = False


@dataclass
class ComparisonConfig:
    """Configuration for cluster comparison."""

    scaled: str = "standard"
    by_variance: bool = True
    max_features: int = -1
    min_cluster_size: int = 5
    with_reference_value: bool = True


# ==================== MAIN CLASS ====================


class ClusterBaseHelper:
    """
    Helper class for cluster analysis and visualisation.

    This class provides methods for:
    - Statistical analysis of clusters
    - Comparison of numerical and categorical features
    - Visualisation with PCA, UMAP and Autoencoder
    - Calculation of quality metrics
    - Generation of heatmaps and comparative plots

    Attributes:
        DEFAULT_SCALER: Default scaler
        VALID_SCALERS: Valid scalers
        VALID_SCALING_METHODS: Valid scaling methods

    Examples:
        >>> helper = ClusterBaseHelper(
        ...     data=df,
        ...     features=['age', 'weight', 'glucose'],
        ...     scaler='standard'
        ... )
        >>> stats = helper.get_stats_categorical()
        >>> helper.show_cluster_compare_numerical()
    """

    # Class constants
    DEFAULT_SCALER = "standard"
    VALID_SCALERS = {"standard", "minmax", "robust", "none"}
    VALID_SCALING_METHODS = {"standard", "minmax", "robust", "proportional", "none"}
    DEFAULT_MIN_CLUSTER_SIZE = 5
    DEFAULT_FIGURE_SIZE = (10, 10)
    DEFAULT_ALPHA = 0.7
    # Summary column names (used in get_stats_* and get_ranking_by_variance)
    _COL_SAMPLE_PCT = "Sample Percentage"
    _COL_SAMPLE_PCT_CAT = "Sample Percentage (%)"
    _COL_OCCURRENCES = "Occurrences"
    _COL_PERCENTAGE = "Percentage"

    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        features_not_considered: Optional[List[str]] = None,
        scaler: str = "standard",
    ):
        """
        Initialises ClusterBaseHelper.

        Args:
            data: DataFrame with data for analysis
            features: List of features to consider (None = all)
            features_not_considered: Features to exclude (default: ['hadm_id', 'subject_id'])
            scaler: Scaler type ('standard', 'minmax', 'robust', 'none')

        Raises:
            TypeError: If data is not a DataFrame
            ValueError: If DataFrame is empty or scaler is invalid

        Examples:
            >>> helper = ClusterBaseHelper(df, scaler='standard')
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(data)}")

        if data.empty:
            raise ValueError("DataFrame cannot be empty")

        if features_not_considered is None:
            features_not_considered = ["hadm_id", "subject_id"]

        # Store original data
        self._full_data = data.copy()
        self._all_features = data.columns.tolist()

        # Select features
        if features is None:
            features_used = data.columns.tolist()
        else:
            # Validate that features exist
            missing = [f for f in features if f not in data.columns]
            if missing:
                raise ValueError(f"Features not found in data: {missing}")
            features_used = features

        # Filter data
        data_filtered = data[features_used].drop(
            columns=features_not_considered, errors="ignore"
        )

        self._data = data_filtered
        self._features_not_considered = features_not_considered
        self._features = data_filtered.columns.tolist()

        # Configure scaler
        self._scaled = self._set_scaler(scaler)

        # Identify categorical and numerical features
        self._categorical_features = self._identify_categorical_features(data)
        self._numerical_features = list(
            set(data.columns) - set(self._categorical_features)
        )

        # Initialise cluster data and metrics
        self._clustered_data = None
        self._clustered_data_autoencoder = None

        if HAS_METRICS:
            self._metrics_calculator = cm.ClusterMetrics()
        else:
            self._metrics_calculator = None

        logger.info(
            f"ClusterBaseHelper initialized: {len(data)} records, "
            f"{len(self._features)} features ({len(self._categorical_features)} categorical, "
            f"{len(self._numerical_features)} numerical)"
        )

    def _identify_categorical_features(self, data: pd.DataFrame) -> List[str]:
        """
        Identifies categorical features (binary 0/1).

        Args:
            data: DataFrame for analysis

        Returns:
            List of categorical feature names
        """
        categorical = []
        for col in data.columns:
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                categorical.append(col)
        return categorical

    def _set_scaler(self, scaler: str) -> bool:
        """
        Configures the scaler for normalisation.

        Args:
            scaler: Scaler type

        Returns:
            True if scaler configured, False if 'none'

        Raises:
            ValueError: If scaler is invalid
        """
        if scaler not in self.VALID_SCALERS:
            raise ValueError(
                f"Invalid scaler '{scaler}'. Choose from {self.VALID_SCALERS}"
            )

        if scaler == "standard":
            self._scaler = StandardScaler()
            return True
        elif scaler == "minmax":
            self._scaler = MinMaxScaler()
            return True
        elif scaler == "robust":
            self._scaler = RobustScaler()
            return True
        else:  # none
            self._scaler = None
            return False

    # ==================== PROPERTIES ====================

    @property
    def metrics_calculator(self):
        """Returns the metrics calculator."""
        return self._metrics_calculator

    @property
    def data(self) -> pd.DataFrame:
        """Returns a copy of the filtered data."""
        return self._data.copy()

    @property
    def full_data(self) -> pd.DataFrame:
        """Returns a copy of the full data."""
        return self._full_data.copy()

    @property
    def clustered_data(self) -> Optional[pd.DataFrame]:
        """Returns clustered data."""
        return self._clustered_data

    @clustered_data.setter
    def clustered_data(self, value: pd.DataFrame):
        """
        Sets clustered data.

        Args:
            value: DataFrame with a 'Cluster' column

        Raises:
            TypeError: If not a DataFrame
            ValueError: If it does not contain a 'Cluster' column
        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError("clustered_data must be a pandas DataFrame")

        if "Cluster" not in value.columns:
            raise ValueError("clustered_data must contain a 'Cluster' column")

        self._clustered_data = value
        logger.info(
            f"Clustered data set: {len(value)} records, "
            f"{value['Cluster'].nunique()} clusters"
        )

    @property
    def features_not_considered(self) -> List[str]:
        """Returns the list of features not considered."""
        return self._features_not_considered.copy()

    def get_feature_info(self) -> Dict[str, List[str]]:
        """
        Returns information about features.

        Returns:
            Dictionary with lists of features by type

        Examples:
            >>> info = helper.get_feature_info()
            >>> print(f"Categorical: {info['categorical']}")
        """
        return {
            "categorical": self._categorical_features.copy(),
            "numerical": self._numerical_features.copy(),
            "all": self._features.copy(),
        }

    # ==================== SAFE SAVING METHODS ====================

    def _save_figure_safely(
        self,
        fig: Optional[plt.Figure] = None,
        savepath: Optional[str] = None,
    ) -> None:
        """
        Saves a figure safely without using os.chdir().

        Args:
            fig: Matplotlib figure (if None, uses plt.gcf())
            savepath: Path to save to (None = do not save)

        Examples:
            >>> fig, ax = plt.subplots()
            >>> # ... create plot ...
            >>> helper._save_figure_safely(fig, 'output/plot.png')
        """
        if savepath is None:
            return

        if fig is None:
            fig = plt.gcf()

        # Use absolute path - SAFE
        abs_path = os.path.abspath(savepath)

        # Ensure directory exists
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        try:
            fig.savefig(abs_path)
            logger.info(f"Figure saved to {abs_path}")
        except Exception as e:
            logger.error(f"Failed to save figure to {abs_path}: {e}")
            raise

    # ==================== STATISTICS ====================

    def get_stats_all(
        self,
        show_std: bool = True,
    ) -> pd.DataFrame:
        """
        Calculates statistics for all features by cluster.

        Args:
            show_std: If True, returns "mean ± std", otherwise only mean

        Returns:
            DataFrame with statistics by cluster

        Raises:
            ValueError: If clustered data is not available

        Examples:
            >>> stats = helper.get_stats_all(show_std=True)
        """
        if self._clustered_data is None:
            raise ValueError(
                "No clustered data available. " "Set clustered_data property first."
            )

        clustered_data = self._clustered_data

        def calculate_stats(group):
            stats = {}
            for col in group.columns:
                if col == "Cluster":
                    continue

                if col in self._categorical_features:
                    # Binary features: percentage
                    total = clustered_data[col].sum()
                    if total > 0:
                        pct = group[col].sum() / total * 100
                        stats[col] = f"{pct:.2f}%"
                    else:
                        stats[col] = "0.00%"
                else:
                    # Numerical features
                    if show_std:
                        stats[col] = f"{group[col].mean():.2f} ± {group[col].std():.2f}"
                    else:
                        stats[col] = group[col].mean()

            return pd.Series(stats)

        # Calculate statistics by cluster
        cluster_stats = clustered_data.groupby("Cluster").apply(calculate_stats).T

        # Add percentage of each cluster
        cluster_counts = (
            self._clustered_data["Cluster"].value_counts(normalize=True) * 100
        )
        cluster_stats.loc[self._COL_SAMPLE_PCT] = cluster_counts.map(
            lambda x: f"{x:.2f}%"
        )

        logger.debug(f"Calculated stats for {len(cluster_stats.columns)} clusters")
        return cluster_stats

    def get_stats_categorical(
        self,
        show_covid: bool = False,
        show_total_number_of_data: bool = False,
        selected_clusters: Optional[List[int]] = None,
        intracluster: bool = True,
    ) -> pd.DataFrame:
        """
        Calculates statistics for categorical features by cluster.

        Args:
            show_covid: If True, includes COVID statistics
            show_total_number_of_data: If True, shows counts instead of %
            selected_clusters: List of specific clusters
            intracluster: If True, calculates % within the cluster; otherwise, global %

        Returns:
            DataFrame with categorical statistics

        Examples:
            >>> stats = helper.get_stats_categorical(intracluster=True)
        """
        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        clustered_data = self._clustered_data

        def calculate_stats(group):
            stats = {}
            for col in group.columns:
                if col == "Cluster":
                    continue

                if col in self._categorical_features:
                    total_ones = clustered_data[col].sum()
                    if total_ones > 0:
                        if intracluster:
                            # Percentage within the cluster
                            pct = (group[col].sum() / len(group)) * 100
                        else:
                            # Global percentage
                            pct = (group[col].sum() / total_ones) * 100
                        stats[f"{col} (%)"] = round(pct, 2)
            return pd.Series(stats)

        # Calculate stats
        cluster_stats = clustered_data.groupby("Cluster").apply(calculate_stats).T

        # Remove COVID if not requested
        if not show_covid and "COVID (%)" in cluster_stats.index:
            cluster_stats = cluster_stats.drop(index=["COVID (%)"])

        # Filter clusters if specified
        if selected_clusters is not None:
            valid_clusters = [
                c for c in selected_clusters if c in cluster_stats.columns
            ]
            cluster_stats = cluster_stats[valid_clusters]

        # Add total occurrences
        occurrences = {}
        percentages = {}
        for index in cluster_stats.index:
            if "Cluster" in index:
                continue
            col_name = index.replace(" (%)", "")
            if col_name in clustered_data.columns:
                occurrences[index] = int(clustered_data[col_name].sum())
                percentages[index] = round(
                    occurrences[index] / len(clustered_data) * 100, 2
                )

        cluster_stats["Occurrences"] = pd.Series(occurrences).astype(int)
        cluster_stats["Percentage"] = pd.Series(percentages).astype(float)

        # Add sample percentage row
        cluster_counts = clustered_data["Cluster"].value_counts()
        total_records = len(clustered_data)

        cluster_percentage = {}
        for cluster in cluster_stats.columns:
            if cluster not in ["Occurrences", "Percentage"]:
                count = cluster_counts.get(cluster, 0)
                if show_total_number_of_data:
                    cluster_percentage[cluster] = f"{int(count)}"
                else:
                    pct = round((count / total_records) * 100, 2)
                    cluster_percentage[cluster] = pct

        # Sort indices
        desired_order = [
            "gender_M (%)",
            "died (%)",
            "died_in_stay (%)",
            "died_after (%)",
        ]
        remaining = sorted([c for c in cluster_stats.index if c not in desired_order])
        sorted_index = remaining + desired_order
        cluster_stats = cluster_stats.reindex(
            [i for i in sorted_index if i in cluster_stats.index]
        )

        # Add percentage row
        row_name = (
            "Number of Patients"
            if show_total_number_of_data
            else self._COL_SAMPLE_PCT_CAT
        )
        cluster_stats.loc[row_name] = pd.Series(cluster_percentage)
        cluster_stats.loc[row_name, "Occurrences"] = int(total_records)
        cluster_stats.loc[row_name, "Percentage"] = 100.0

        cluster_stats["Occurrences"] = (
            cluster_stats["Occurrences"].fillna(0).astype(int)
        )

        # Remove COVID if present
        if "COVID (%)" in cluster_stats.index:
            cluster_stats = cluster_stats.drop(index=["COVID (%)"])

        logger.info(
            f"Calculated categorical stats for {len(cluster_stats.columns)-2} clusters"
        )
        return cluster_stats

    def get_stats_numerical(
        self,
        scaled: str = "standard",
        by_variance: bool = True,
        max_features: int = -1,
        show_std: bool = True,
        show_total_number_of_data: bool = False,
        selected_clusters: Optional[List[int]] = None,
        selected_features_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns statistics for numerical features by cluster.

        Args:
            scaled: Scaling method ('standard', 'minmax', 'robust', 'proportional', 'none')
            by_variance: If True, sorts by variance
            max_features: Maximum number of features (-1 = all)
            show_std: If True, shows "mean ± std"; otherwise only mean
            show_total_number_of_data: If True, shows counts
            selected_clusters: List of specific clusters
            selected_features_list: List of specific features

        Returns:
            DataFrame with features as rows and clusters as columns

        Examples:
            >>> stats = helper.get_stats_numerical(scaled='standard', max_features=10)
        """
        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        def calculate_stats(group):
            stats = {}
            for col in group.columns:
                if col == "Cluster":
                    continue
                if show_std:
                    stats[col] = f"{group[col].mean():.2f} ± {group[col].std():.2f}"
                else:
                    stats[col] = group[col].mean()
            return pd.Series(stats)

        # Prepare data
        (
            data,
            selected_features,
            feature_display_names,
            _,
            _,
            _,
            _,
        ) = self._prepare_cluster_data(
            selected_clusters=selected_clusters,
            scaled=scaled,
            by_variance=by_variance,
            max_features=max_features,
            selected_features_list=selected_features_list,
        )

        if selected_clusters is not None:
            data = data[data["Cluster"].isin(selected_clusters)]

        # Rename columns
        renamed_columns = {
            col: feature_display_names[col]
            for col in selected_features
            if col in feature_display_names
        }
        data = data.rename(columns=renamed_columns)

        # Calculate statistics
        cluster_stats = data.groupby("Cluster").apply(calculate_stats).T
        cluster_stats = cluster_stats.reindex(sorted(cluster_stats.index))

        # Add percentage/count row
        if show_total_number_of_data:
            cluster_counts = self._clustered_data["Cluster"].value_counts()
            cluster_stats.loc["Cluster Total"] = cluster_counts
        else:
            cluster_counts = (
                self._clustered_data["Cluster"].value_counts(normalize=True) * 100
            )
            cluster_stats.loc[self._COL_SAMPLE_PCT] = cluster_counts.map(
                lambda x: f"{x:.2f}%"
            )

        logger.info(f"Calculated numerical stats for {len(selected_features)} features")
        return cluster_stats

    def get_ranking_by_variance(
        self,
        max_features: int = 20,
        display: str = "all",
    ) -> pd.DataFrame:
        """
        Returns a ranking of features by variance between clusters.

        Args:
            max_features: Maximum number of features in the ranking
            display: 'all', 'categorical', or 'numerical'

        Returns:
            DataFrame with feature ranking by variance

        Raises:
            ValueError: If display is invalid

        Examples:
            >>> ranking = helper.get_ranking_by_variance(max_features=10, display='all')
        """
        valid_displays = {"all", "categorical", "numerical"}
        if display not in valid_displays:
            raise ValueError(
                f"Invalid display '{display}'. Choose from {valid_displays}"
            )

        # Get data
        numerical_data = self.get_stats_numerical(show_std=False).T
        numerical_data = numerical_data.drop(
            columns=self._COL_SAMPLE_PCT, errors="ignore"
        )

        categorical_data = self.get_stats_categorical(show_covid=False).T
        categorical_data = categorical_data.drop(
            columns=self._COL_SAMPLE_PCT_CAT, errors="ignore"
        )
        categorical_data = categorical_data.drop(
            columns=[self._COL_OCCURRENCES, self._COL_PERCENTAGE], errors="ignore"
        )

        # Calculate variances
        if display in ["all", "numerical"]:
            numerical_variances = numerical_data.var(axis=0)
            numerical_df = pd.DataFrame(numerical_variances, columns=["Variance"])
            numerical_df["Type"] = "Numerical"
        else:
            numerical_df = pd.DataFrame(columns=["Variance", "Type"])

        if display in ["all", "categorical"]:
            categorical_data = categorical_data / 100  # Normalise
            categorical_variances = categorical_data.var(axis=0)
            categorical_df = pd.DataFrame(categorical_variances, columns=["Variance"])
            categorical_df["Type"] = "Categorical"
        else:
            categorical_df = pd.DataFrame(columns=["Variance", "Type"])

        # Combine and sort
        combined_df = pd.concat([numerical_df, categorical_df])
        sorted_df = combined_df.sort_values(by="Variance", ascending=False)
        sorted_df = sorted_df.reset_index().rename(columns={"index": "Feature"})

        result = sorted_df.head(max_features)

        logger.info(f"Generated variance ranking with {len(result)} features")
        return result

    # ==================== DATA PREPARATION ====================

    def _prepare_cluster_data(
        self,
        selected_clusters: Optional[List[int]] = None,
        scaled: str = "standard",
        selected_features_list: Optional[List[str]] = None,
        by_variance: bool = True,
        max_features: int = -1,
    ) -> Tuple[
        pd.DataFrame,
        List[str],
        Dict[str, str],
        Dict[str, str],
        Dict[str, str],
        Dict[str, Tuple[float, float]],
        List[Tuple[str, float]],
    ]:
        """
        Prepares data for cluster comparison.

        Args:
            selected_clusters: Clusters to include
            scaled: Scaling method
            selected_features_list: Specific features
            by_variance: If True, sorts by variance
            max_features: Maximum number of features

        Returns:
            Tuple of (data, selected_features, display_names, colors, categories, normal_values, pvalues)
        """
        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        if scaled not in self.VALID_SCALING_METHODS:
            raise ValueError(
                f"Invalid scaling method '{scaled}'. "
                f"Choose from {self.VALID_SCALING_METHODS}"
            )

        # 1. Select numerical features
        if selected_features_list is None:
            numeric_cols = [col for col in self._numerical_features if col != "Cluster"]
        else:
            numeric_cols = selected_features_list

        # 2. Calculate means (for display)
        means = {col: self._clustered_data[col].mean() for col in numeric_cols}

        # 3. Scale data
        data, scaled_normal_values = self._scale_cluster_data(numeric_cols, scaled)

        # 4. Remove unwanted columns
        data, numeric_cols = self._remove_unwanted_columns(data, numeric_cols)

        # 5. Add cluster labels
        data["Cluster"] = self._clustered_data["Cluster"].values

        # 6. Filter clusters if specified
        if selected_clusters is not None:
            data = data[data["Cluster"].isin(selected_clusters)]

        # 7. Select features by variance
        pvalues = []
        if by_variance and len(data["Cluster"].unique()) == 2:
            selected_features = self._select_features_by_variance(
                data, numeric_cols, max_features
            )
            pvalues = self._calculate_pvalues(data, numeric_cols)
        else:
            selected_features = numeric_cols

        # 8. Create display names and metadata
        (feature_display_names, feature_colors, feature_categories) = (
            self._create_feature_metadata(selected_features, means)
        )

        # 9. Sort by category
        selected_features.sort(
            key=lambda x: (feature_categories[x], feature_display_names[x])
        )

        logger.debug(
            f"Prepared cluster data: {len(data)} records, "
            f"{len(selected_features)} features"
        )

        return (
            data,
            selected_features,
            feature_display_names,
            feature_colors,
            feature_categories,
            scaled_normal_values,
            pvalues,
        )

    def _scale_cluster_data(
        self,
        numeric_cols: List[str],
        scaled: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """
        Scales numerical data using the specified method.

        Args:
            numeric_cols: Numerical columns to scale
            scaled: Scaling method

        Returns:
            Tuple (scaled data, scaled normal values)
        """
        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        scaled_normal_values = {}
        scaler: MinMaxScaler | StandardScaler | RobustScaler | None = None

        clustered_data = self._clustered_data

        if scaled == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_values = scaler.fit_transform(clustered_data[numeric_cols])
            data = pd.DataFrame(data_values, columns=numeric_cols)
        elif scaled == "standard":
            scaler = StandardScaler()
            data_values = scaler.fit_transform(clustered_data[numeric_cols])
            data = pd.DataFrame(data_values, columns=numeric_cols)
        elif scaled == "robust":
            scaler = RobustScaler()
            data_values = scaler.fit_transform(clustered_data[numeric_cols])
            data = pd.DataFrame(data_values, columns=numeric_cols)
        elif scaled == "proportional":
            data = clustered_data[numeric_cols].copy()
            for col in numeric_cols:
                col_mean = data[col].mean()
                if col_mean != 0:
                    data[col] = data[col] / col_mean
                    if col in NORMAL_VALUES:
                        scaled_normal_values[col] = NORMAL_VALUES[col] / col_mean
        else:  # none
            data = clustered_data[numeric_cols].copy()
            scaler = None

        # Scale normal values if a scaler is present
        if scaled in ["minmax", "standard", "robust"] and HAS_CONFIG:
            for feature in NORMAL_VALUES:
                if feature in clustered_data.columns and scaler is not None:
                    scaler.fit(clustered_data[[feature]])
                    min_val, max_val = NORMAL_VALUES[feature]
                    scaled_min = scaler.transform([[min_val]])[0, 0]
                    scaled_max = scaler.transform([[max_val]])[0, 0]
                    scaled_normal_values[feature] = (scaled_min, scaled_max)

        return data, scaled_normal_values

    def _remove_unwanted_columns(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Removes unwanted columns from the data."""
        numeric_cols = numeric_cols.copy()  # nunca muta o argumento do chamador
        unwanted = ["COVID", "hadm_id", "subject_id"]
        for col in unwanted:
            if col in data.columns:
                data = data.drop(columns=[col])
                if col in numeric_cols:
                    numeric_cols.remove(col)
        return data, numeric_cols

    def _select_features_by_variance(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
        max_features: int,
    ) -> List[str]:
        """Selects features with the highest variance between clusters."""
        cluster_col = data["Cluster"]
        clusters = cluster_col.unique()

        if len(clusters) != 2:
            return numeric_cols

        mask_0 = cluster_col == clusters[0]
        mask_1 = cluster_col == clusters[1]

        # Calculate p-values (KS test)
        pvalues = [
            (
                col,
                scipy_stats.ks_2samp(
                    data.loc[mask_0, col].values, data.loc[mask_1, col].values
                )[1],
            )
            for col in numeric_cols
        ]

        pvalues.sort(key=lambda x: x[1])

        n_select = max_features if max_features > 0 else len(pvalues)
        return [col for col, _ in pvalues[:n_select]]

    def _calculate_pvalues(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
    ) -> List[Tuple[str, float]]:
        """Calculates p-values for numerical features."""
        cluster_col = data["Cluster"]
        clusters = cluster_col.unique()

        if len(clusters) != 2:
            return []

        mask_0 = cluster_col == clusters[0]
        mask_1 = cluster_col == clusters[1]

        pvalues = [
            (
                col,
                scipy_stats.ks_2samp(
                    data.loc[mask_0, col].values, data.loc[mask_1, col].values
                )[1],
            )
            for col in numeric_cols
        ]

        pvalues.sort(key=lambda x: x[1])
        return pvalues

    def _create_feature_metadata(
        self,
        features: List[str],
        means: Dict[str, float],
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """
        Creates metadata for features (display names, colours, categories).

        Args:
            features: List of features
            means: Feature means

        Returns:
            Tuple (display_names, colors, categories)
        """
        display_names = {}
        colors = {}
        categories = {}

        for feature in features:
            # Defaults
            display_name = feature
            color = "black"
            category = "other"

            # Parse structured name
            match = re.match(r"([^_]+)_(Hematology|Chemistry)_([^_]+)_([^_]+)", feature)
            if match:
                display_name = f"{match.group(1)} ({match.group(4)})"
                category = match.group(2).lower()
                color = "red" if category == "hematology" else "blue"

            # Use NUMERICAL dict if available
            if HAS_CONFIG:
                display_name = NUMERICAL.get(display_name, display_name)

            # Add mean
            if feature in means:
                display_name += f" - {means[feature]:.2f}"

            display_names[feature] = display_name
            colors[display_name] = color
            categories[feature] = category

        return display_names, colors, categories

    # ==================== VISUALISATION METHODS ====================

    def show_cluster_compare_categorical(
        self,
        scaled: bool = True,
        figsize: Tuple[int, int] = (10, 10),
        by_variance: bool = True,
        max_features: int = -1,
        min_cluster_size: int = 10,
        top_features: int = -1,
        selected_clusters: Optional[List[int]] = None,
        intracluster: bool = False,
    ) -> Tuple[List[str], Optional[int]]:
        """
        Visualises the comparison of categorical features between clusters.

        Args:
            scaled: If True, scales data
            figsize: Figure size
            by_variance: If True, sorts by variance
            max_features: Maximum number of features
            min_cluster_size: Minimum cluster size (%)
            top_features: Number of top features to return
            selected_clusters: Specific clusters
            intracluster: If True, % within the cluster; otherwise global %

        Returns:
            Tuple (list of top features, cluster with most deaths)

        Examples:
            >>> features, death_cluster = helper.show_cluster_compare_categorical(
            ...     max_features=10
            ... )
        """
        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        clustered_data = self._clustered_data

        def calculate_stats(group):
            stats = {}
            for col in group.columns:
                if col == "Cluster":
                    continue
                if col in self._categorical_features:
                    if intracluster:
                        stats[col + " (%)"] = group[col].sum() / len(group) * 100
                    else:
                        total = clustered_data[col].sum()
                        if total > 0:
                            stats[col + " (%)"] = group[col].sum() / total * 100
            return pd.Series(stats)

        # Prepare data
        data = self._clustered_data.drop(columns=["COVID"], errors="ignore")

        if selected_clusters is not None:
            min_cluster_size = 0
            data = data[data["Cluster"].isin(selected_clusters)]

        cluster_stats = data.groupby("Cluster").apply(calculate_stats)

        # Filter by minimum size
        cluster_counts = data["Cluster"].value_counts(normalize=True) * 100
        cluster_stats[self._COL_SAMPLE_PCT] = cluster_counts
        cluster_stats = cluster_stats[
            cluster_stats[self._COL_SAMPLE_PCT] >= min_cluster_size
        ]
        cluster_stats = cluster_stats.drop(columns=[self._COL_SAMPLE_PCT])

        # Sort by variance
        if by_variance:
            variances = cluster_stats.var()
            sorted_features = variances.sort_values(ascending=True)

            if max_features != -1 and len(sorted_features) > max_features:
                cluster_stats = cluster_stats[
                    sorted_features[(len(sorted_features) - max_features) :].index
                ]
            else:
                cluster_stats = cluster_stats[sorted_features.index]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        cluster_stats.T.plot(kind="barh", ax=ax, width=0.8, legend=True)

        ax.set_title("Comparação de Clusters - Features Categóricas")
        ax.set_xlabel("Valores")
        ax.set_ylabel("Features")
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True)

        plt.tight_layout()
        plt.show()

        # Retorna top features
        if top_features == -1:
            top_features = len(cluster_stats.columns)

        variances = cluster_stats.var()
        top_feature_list: List[str] = list(
            variances.head(top_features).sort_values(ascending=False).index.tolist()
        )
        top_feature_list = [f.replace(" (%)", "") for f in top_feature_list]

        # Identify cluster with most deaths
        if "died (%)" in cluster_stats.T.index:
            death_column = cluster_stats.T.loc["died (%)"]
            max_death_cluster = death_column.idxmax()
        else:
            max_death_cluster = None

        logger.info(
            f"Displayed categorical comparison: {len(cluster_stats.columns)} features, "
            f"{len(cluster_stats)} clusters"
        )

        return top_feature_list, max_death_cluster

    def show_cluster_compare_numerical(
        self,
        config: Optional[ComparisonConfig] = None,
        selected_clusters: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (10, 15),
        show_outliers: bool = False,
        top_features: int = -1,
        savepath: Optional[str] = None,
        selected_features_list: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Visualises the comparison of numerical features between clusters using a boxplot.

        Args:
            config: Comparison configuration
            figsize: Figure size
            show_outliers: If True, shows outliers
            top_features: Number of top features
            savepath: Path to save the figure
            selected_features_list: Specific features

        Returns:
            List of (feature, p-value) ordered by significance

        Examples:
            >>> config = ComparisonConfig(scaled='standard', max_features=10)
            >>> pvalues = helper.show_cluster_compare_numerical(config=config)
        """
        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        # Use default config if not provided
        if config is None:
            config = ComparisonConfig()

        # Prepare data
        (
            data,
            selected_features,
            feature_display_names,
            feature_colors,
            feature_categories,
            scaled_normal_values,
            pvalues,
        ) = self._prepare_cluster_data(
            selected_clusters=selected_clusters,
            scaled=config.scaled,
            by_variance=config.by_variance,
            max_features=config.max_features,
            selected_features_list=selected_features_list,
        )

        # Filter clusters
        if selected_clusters is not None:
            data = data[data["Cluster"].isin(selected_clusters)]
            data["Cluster"] = pd.Categorical(
                data["Cluster"], categories=selected_clusters, ordered=True
            )
            data = data.sort_values("Cluster")

        # Remove small clusters
        cluster_counts = (
            data["Cluster"].value_counts().apply(lambda x: x / len(data) * 100)
        )
        valid_clusters = cluster_counts[cluster_counts >= config.min_cluster_size].index
        data = data[data["Cluster"].isin(valid_clusters)]

        # Organise features by category
        sorted_features = self._organize_features_by_category(
            selected_features, feature_categories, top_features
        )

        # Prepare data for plot
        data_melted = pd.melt(
            data,
            id_vars="Cluster",
            value_vars=sorted_features,
            var_name="Feature",
            value_name="Value",
        )
        data_melted["Feature"] = data_melted["Feature"].map(feature_display_names)

        # Create plot
        fig, ax = self._create_numerical_comparison_plot(
            data_melted,
            feature_display_names,
            feature_colors,
            scaled_normal_values,
            config,
            figsize,
            show_outliers,
        )

        # Save and display (safe method)
        self._save_figure_safely(fig, savepath)
        plt.show()

        logger.info(
            f"Displayed numerical comparison: {len(sorted_features)} features, "
            f"{len(valid_clusters)} clusters"
        )

        return pvalues

    def _organize_features_by_category(
        self,
        features: List[str],
        categories: Dict[str, str],
        top_n: int,
    ) -> List[str]:
        """Organises features by category."""
        categorized = {
            "other": [],
            "hematology": [],
            "chemistry": [],
        }

        for feature in features:
            cat = categories.get(feature, "other")
            categorized[cat].append(feature)

        if top_n != -1:
            sorted_features = (
                categorized["other"][:top_n]
                + categorized["hematology"][:top_n]
                + categorized["chemistry"][:top_n]
            )
        else:
            sorted_features = (
                categorized["other"]
                + categorized["hematology"]
                + categorized["chemistry"]
            )

        return sorted_features

    def _create_numerical_comparison_plot(
        self,
        data_melted: pd.DataFrame,
        feature_display_names: Dict[str, str],
        feature_colors: Dict[str, str],
        scaled_normal_values: Dict[str, Tuple[float, float]],
        config: ComparisonConfig,
        figsize: Tuple[int, int],
        show_outliers: bool,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Creates the numerical comparison plot."""
        fig, ax = plt.subplots(figsize=figsize)

        # Reference line for proportional scaling
        if config.scaled == "proportional":
            ax.axvline(x=1.0, color="black", alpha=0.4)

        # Boxplot
        sns.boxplot(
            data=data_melted,
            x="Value",
            y="Feature",
            hue="Cluster",
            palette="tab10",
            showfliers=show_outliers,
            ax=ax,
        )

        # Add reference value ranges
        if config.with_reference_value and HAS_CONFIG:
            self._add_reference_ranges(
                ax,
                data_melted,
                feature_display_names,
                scaled_normal_values,
            )

        # Logarithmic scale for proportional
        if config.scaled == "proportional":
            ax.set_xscale("log")
            ax.set_xlim(0.1, 10)
        else:
            ax.set_xlim(right=5)

        # Label colours
        for tick in ax.get_yticklabels():
            tick.set_color(feature_colors.get(tick.get_text(), "black"))

        # Labels and title
        ax.set_title("Comparação entre Clusters - Features Numéricas")
        if config.scaled == "proportional":
            ax.set_xlabel("Valor Proporcional (Relativo à Média)")
        else:
            ax.set_xlabel("Valor Normalizado")
        ax.set_ylabel("Feature (Unidade) - Média")

        # Legends
        cluster_legend = ax.legend(
            title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left"
        )

        category_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=8,
                label="Geral",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Hematologia",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Bioquímica",
            ),
        ]

        ax.add_artist(cluster_legend)
        ax.legend(
            handles=category_elements,
            title="Categoria",
            bbox_to_anchor=(1.05, 0.8),
            loc="upper left",
        )

        # Grid
        if config.scaled == "proportional":
            ax.grid(
                True, which="both", axis="x", alpha=0.5, linestyle="-", linewidth=0.5
            )
        ax.grid(True, which="major", axis="y", alpha=0.5, linestyle="-", linewidth=0.5)

        plt.tight_layout()

        return fig, ax

    def _add_reference_ranges(
        self,
        ax: plt.Axes,
        data_melted: pd.DataFrame,
        feature_display_names: Dict[str, str],
        scaled_normal_values: Dict[str, Tuple[float, float]],
    ) -> None:
        """Adds reference value range bands to the plot."""
        display_to_original = {v: k for k, v in feature_display_names.items()}
        features = data_melted["Feature"].unique()

        for idx, display_feature in enumerate(features):
            original_feature = display_to_original.get(display_feature)
            if original_feature and original_feature in scaled_normal_values:
                min_val, max_val = scaled_normal_values[original_feature]

                ax.axvspan(
                    min_val,
                    max_val,
                    ymin=1 - (idx) / len(features),
                    ymax=1 - (idx + 1) / len(features),
                    color="lightgreen",
                    alpha=0.4,
                    zorder=0,
                )

    def heatmap_clusters_categorical(
        self,
        figsize: Tuple[int, int] = (10, 8),
        intracluster: bool = True,
        relative_total: bool = False,
        min_cluster_size: int | float = 5,
        selected_clusters: Optional[List[int]] = None,
        savepath: Optional[str] = None,
    ) -> None:
        """
        Creates a heatmap of categorical features by cluster.

        Args:
            figsize: Figure size
            intracluster: If True, % within cluster; otherwise global %
            relative_total: If True, values relative to total sample
            min_cluster_size: Minimum cluster size (%)
            selected_clusters: Specific clusters
            savepath: Path to save

        Examples:
            >>> helper.heatmap_clusters_categorical(
            ...     relative_total=True,
            ...     savepath='output/heatmap.png'
            ... )
        """
        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        # Get statistics
        df = self.get_stats_categorical(intracluster=intracluster)
        occurrences = df["Occurrences"]
        percentages = df["Percentage"]
        df = df.drop(columns=["Occurrences", "Percentage"])

        # Normalise by total percentage if requested
        if relative_total:
            percentages_series = pd.Series(percentages, index=df.index)
            min_cluster_size = min_cluster_size / 100
            for column in df.columns:
                df[column] = df[column] / percentages_series

        # Filter clusters
        if selected_clusters is not None:
            min_cluster_size = 0
            df = df[selected_clusters]

        # Filter by minimum size
        if min_cluster_size > 0:
            last_row_name = df.index[-1]
            valid_clusters = [
                cluster
                for cluster in df.columns
                if df.loc[last_row_name, cluster] >= min_cluster_size
            ]
            df = df[valid_clusters]

        # Rename indices
        if HAS_CONFIG:
            if relative_total:
                df.index = [
                    CATEGORICAL.get(idx.replace(" (%)", ""), idx) for idx in df.index
                ]
            else:
                df.index = [
                    f"{CATEGORICAL.get(idx.replace(' (%)', ''), idx)} (%)"
                    for idx in df.index
                ]

        # Create heatmap
        fig = plt.figure(figsize=figsize)

        if relative_total:
            # Create custom labels
            annot_labels = df.copy()
            for idx in df.index:
                if "Porcentagem" not in idx and "Número" not in idx:
                    annot_labels.loc[idx] = df.loc[idx].map(lambda x: f"{x:.2f} x")
                else:
                    annot_labels.loc[idx] = df.loc[idx].map(lambda x: f"{100*x:.2f}")

            ax = sns.heatmap(
                df,
                annot=annot_labels,
                fmt="",
                cmap="YlOrRd",
                cbar=False,
                vmin=0,
                vmax=None,
            )
            title = "Prevalência de Condições Clínicas por Cluster em Relação à Amostra Global"
        else:
            ax = sns.heatmap(
                df,
                annot=df,
                fmt=".2f",
                cmap="YlOrRd",
                cbar=False,
                vmin=0,
                vmax=100,
            )
            title = "Comparação de Features Categóricas"

        # Add occurrences on the right side
        for i, occurrence in enumerate(occurrences):
            ax.text(
                df.shape[1] + 0.1,
                i + 0.5,
                f"n = {occurrence:<4} ({percentages[i]:.2f}%)",
                verticalalignment="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.text(
            df.shape[1] + 0.1,
            -0.5,
            "Ocorrências",
            verticalalignment="center",
            fontsize=12,
            fontweight="bold",
        )

        plt.title(title)
        plt.xlabel("Clusters")
        plt.ylabel("Features")
        plt.tight_layout()

        # Save and display (safe method)
        self._save_figure_safely(fig, savepath)
        plt.show()

        logger.info(
            f"Created categorical heatmap: {len(df)} features, {len(df.columns)} clusters"
        )

    # ==================== DIMENSIONALITY REDUCTION ====================

    def pca_reduction_information(
        self,
        scale_categorical: bool = False,
        max_dimensions: int = 50,
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """
        Visualises the explained variance information for PCA components.

        Args:
            scale_categorical: If True, scales categorical features
            max_dimensions: Maximum number of components
            figsize: Figure size

        Examples:
            >>> helper.pca_reduction_information(max_dimensions=20)
        """
        data = self._update_data(
            scale_categorical=scale_categorical,
            dimensionality_reduction=None,
        )

        n_components = min(max_dimensions, data.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(data)

        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        xticks = [
            i
            for i in range(1, len(explained_variance) + 1)
            if i % 5 == 0 or i == 1 or i == len(explained_variance)
        ]

        # Cumulative variance plot
        ax.plot(
            np.arange(1, len(cumulative_variance) + 1),
            cumulative_variance,
            label="Variância Explicada Cumulativa",
            color="red",
        )
        ax.axhline(
            y=90, color="gray", linestyle="--", label="90% da Variância Explicada"
        )
        ax.set_xticks(xticks)
        ax.set_xlabel("Componente Principal")
        ax.set_ylabel("Razão Cumulativa da Variância Explicada (%)")
        ax.set_title("Variância Explicada Cumulativa")
        ax.legend(loc="best")
        ax.grid(True)

        plt.tight_layout()
        plt.show()

        logger.info(
            f"PCA analysis: {n_components} components, "
            f"{cumulative_variance[-1]:.2f}% variance explained"
        )

    def _update_data(
        self,
        scale_categorical: bool,
        dimensionality_reduction: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Updates data with scaling and dimensionality reduction.

        Args:
            scale_categorical: If True, also scales categorical features
            dimensionality_reduction: Dict with reduction configuration

        Returns:
            Processed DataFrame
        """
        data = self._data.copy()

        # Scaling
        if self._scaled:
            if self._scaler is None:
                raise ValueError("Scaler not initialized")

            if scale_categorical:
                features_to_scale = data.columns
            else:
                features_to_scale = [
                    col for col in data.columns if col not in self._categorical_features
                ]

            scaled_values = self._scaler.fit_transform(data[features_to_scale])
            data[features_to_scale] = scaled_values

        # Dimensionality reduction
        if dimensionality_reduction is not None:
            data = self._apply_dimensionality_reduction(data, dimensionality_reduction)

        return data

    def _apply_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Applies dimensionality reduction to the data."""
        method = config.get("method", "")
        valid_methods = {"PCA", "AE"}

        if method not in valid_methods:
            raise ValueError(
                f"Invalid dimensionality reduction method '{method}'. "
                f"Choose from {valid_methods}"
            )

        dimensions = config.get("dimensions")
        if dimensions is None:
            raise ValueError("'dimensions' must be specified")

        if method == "PCA":
            pca = PCA(n_components=dimensions)
            principal_components = pca.fit_transform(data)
            pc_columns = [f"PC{i+1}" for i in range(dimensions)]
            data = pd.DataFrame(principal_components, columns=pc_columns)
            logger.info(f"Applied PCA reduction to {dimensions} dimensions")

        elif method == "AE":
            if not HAS_SHADE:
                raise ImportError("SHADE not available for autoencoder reduction")

            data_values = data.values
            autoencoder = SHADE(
                random_state=42,
                batch_size=128,
                pretrain_epochs=0,
                clustering_epochs=100,
                clustering_optimizer_params={"lr": 0.00789809362496747},
                embedding_size=dimensions,
            )
            autoencoder.fit(data_values)
            embeddings = autoencoder.encode(data_values)
            ae_columns = [f"AE{i+1}" for i in range(dimensions)]
            data = pd.DataFrame(embeddings, columns=ae_columns)
            logger.info(f"Applied Autoencoder reduction to {dimensions} dimensions")

        return data

    def show_clustered_data_pca(
        self,
        with_cluster: bool = True,
        selected_clusters: Optional[List[int]] = None,
        scaled: bool = True,
        scale_categorical: bool = False,
        reduction_from_absolute_data: bool = False,
    ) -> None:
        """
        Visualises clustered data with PCA reduction.

        Args:
            with_cluster: If True, colours by cluster
            selected_clusters: Specific clusters to highlight
            scaled: If True, scales data
            scale_categorical: If True, scales categorical features
            reduction_from_absolute_data: If True, uses the full dataset

        Examples:
            >>> helper.show_clustered_data_pca(with_cluster=True)
        """
        data = self._get_reduction_data(
            reduction_from_absolute_data, scaled, scale_categorical
        )

        # Apply PCA
        reducer = PCA(n_components=2, random_state=42)
        data_pca = reducer.fit_transform(data)

        # Prepare data for plot
        clustered_data = pd.DataFrame(data_pca, columns=["PCA1", "PCA2"])

        if with_cluster and self._clustered_data is not None:
            clustered_data["Cluster"] = self._clustered_data["Cluster"].values
            if selected_clusters is not None:
                clustered_data["Cluster"] = clustered_data["Cluster"].apply(
                    lambda x: x if x in selected_clusters else -1
                )
        else:
            clustered_data["Cluster"] = 1

        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=clustered_data["PCA1"],
            y=clustered_data["PCA2"],
            hue=clustered_data["Cluster"],
            palette="tab10",
            alpha=0.7,
        )
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("Clustering with PCA")
        plt.legend(title="Cluster")
        plt.show()

        logger.info("Displayed PCA visualization")

    def show_clustered_data_umap(
        self,
        with_cluster: bool = True,
        selected_clusters: Optional[List[int]] = None,
        scaled: bool = True,
        scale_categorical: bool = False,
        reduction_from_absolute_data: bool = False,
    ) -> None:
        """
        Visualises clustered data with UMAP reduction.

        Args:
            with_cluster: If True, colours by cluster
            selected_clusters: Specific clusters to highlight
            scaled: If True, scales data
            scale_categorical: If True, scales categorical features
            reduction_from_absolute_data: If True, uses the full dataset

        Examples:
            >>> helper.show_clustered_data_umap(with_cluster=True)
        """
        data = self._get_reduction_data(
            reduction_from_absolute_data, scaled, scale_categorical
        )

        # Apply UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        data_umap = reducer.fit_transform(data)

        # Prepare data for plot
        clustered_data = pd.DataFrame(data_umap, columns=["UMAP1", "UMAP2"])

        if with_cluster and self._clustered_data is not None:
            clustered_data["Cluster"] = self._clustered_data["Cluster"].values
            if selected_clusters is not None:
                clustered_data["Cluster"] = clustered_data["Cluster"].apply(
                    lambda x: x if x in selected_clusters else -1
                )
        else:
            clustered_data["Cluster"] = 1

        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=clustered_data["UMAP1"],
            y=clustered_data["UMAP2"],
            hue=clustered_data["Cluster"],
            palette="tab10",
            alpha=0.7,
        )
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.title("Clustering with UMAP")
        plt.legend(title="Cluster")
        plt.show()

        logger.info("Displayed UMAP visualization")

    def _get_reduction_data(
        self,
        reduction_from_absolute_data: bool,
        scaled: bool,
        scale_categorical: bool,
    ) -> pd.DataFrame:
        """Gets data for dimensionality reduction."""
        if reduction_from_absolute_data and HAS_CONFIG:
            # Load full data
            train = pd.read_csv(DATAPATH + COVID_TRAIN_FILE)
            test = pd.read_csv(DATAPATH + COVID_TEST_FILE)
            train["died_after"] = (
                (train["died"] == 1) & (train["died_in_stay"] == 0)
            ).astype(int)
            test["died_after"] = (
                (test["died"] == 1) & (test["died_in_stay"] == 0)
            ).astype(int)
            full_data = pd.concat([train, test], axis=0)
            data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            data = self._full_data.copy()

        # Remove unwanted columns
        data = data.drop(columns=["hadm_id", "subject_id", "COVID"], errors="ignore")

        # Scaling
        if scaled:
            if self._scaler is None:
                raise ValueError("Scaler not initialized")

            if scale_categorical:
                features_to_scale = data.columns
            else:
                features_to_scale = [
                    col for col in data.columns if col not in self._categorical_features
                ]

            scaled_values = self._scaler.fit_transform(data[features_to_scale])
            data[features_to_scale] = scaled_values

        return data

    # ==================== AUTOENCODER ====================

    def set_clustered_autoencoder(
        self,
        scaled: bool = True,
        scale_categorical: bool = False,
        epochs: int = 0,
        clustering_epochs: int = 100,
        embedding_size: int = 2,
        reduction_from_absolute_data: bool = False,
        model: str = "SHADE",
    ) -> None:
        """
        Creates clustering using an autoencoder.

        Args:
            scaled: If True, scales data
            scale_categorical: If True, scales categorical features
            epochs: Pre-training epochs
            clustering_epochs: Clustering epochs
            embedding_size: Embedding size
            reduction_from_absolute_data: If True, uses the full dataset
            model: Model type ('SHADE' or other)

        Raises:
            ImportError: If SHADE is not available

        Examples:
            >>> helper.set_clustered_autoencoder(embedding_size=2)
        """
        if not HAS_SHADE:
            raise ImportError("SHADE not available for autoencoder clustering")

        data = self._get_reduction_data(
            reduction_from_absolute_data, scaled, scale_categorical
        )

        # Identify patients from the original dataset
        if reduction_from_absolute_data and HAS_CONFIG:
            patients = data.index.isin(self._full_data.index)
        else:
            patients = pd.Series([True] * len(data))

        data_values = data.values

        # Train autoencoder
        if model == "SHADE":
            ae_model = SHADE(
                random_state=42,
                batch_size=500,
                pretrain_epochs=epochs,
                clustering_epochs=clustering_epochs,
                embedding_size=embedding_size,
            )
            ae_model.fit(data_values)
            reduced_data = ae_model.encode(data_values)
        else:
            ae_model = SHADE(
                random_state=42,
                batch_size=500,
                pretrain_epochs=epochs,
                clustering_epochs=clustering_epochs,
                embedding_size=10,
            )
            ae_model.fit(data_values)
            reduced_data = ae_model.encode(data_values)

            # Apply additional UMAP
            reducer = umap.UMAP()
            reduced_data = reducer.fit_transform(reduced_data)

        # Reduce to 2D if necessary
        if reduced_data.shape[1] > 2:
            logger.info(f"Reducing from {reduced_data.shape[1]}D to 2D with PCA")
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(reduced_data)

        # Store data
        clustered_data = pd.DataFrame(reduced_data, columns=["X1", "X2"])
        clustered_data["Cluster"] = 1
        clustered_data = clustered_data[patients].reset_index(drop=True)

        self._clustered_data_autoencoder = clustered_data

        logger.info(
            f"Autoencoder clustering completed: {len(clustered_data)} records, "
            f"embedding_size={embedding_size}"
        )

    def show_autoencoder_data(
        self,
        label: str = "died",
        unlabeled: bool = False,
        savepath: Optional[str] = None,
    ) -> None:
        """
        Visualises autoencoder data coloured by a binary label.

        Args:
            label: Binary column to colour by (must be 0/1)
            unlabeled: If True, do not colour by label
            savepath: Path to save

        Raises:
            ValueError: If label is invalid

        Examples:
            >>> helper.show_autoencoder_data(
            ...     label='died',
            ...     savepath='output/autoencoder.png'
            ... )
        """
        if self._clustered_data_autoencoder is None:
            raise ValueError(
                "No autoencoder data available. "
                "Run set_clustered_autoencoder() first."
            )

        if label not in self._full_data.columns:
            raise ValueError(
                f"Label '{label}' not found in data. "
                f"Available: {list(self._full_data.columns)}"
            )

        clustered_data = self._clustered_data_autoencoder.copy()
        if unlabeled:
            clustered_data["Cluster"] = 0
            clustered_data["Cluster"] = clustered_data["Cluster"].astype(int)
        else:
            clustered_data["Cluster"] = self._full_data[label].values
            clustered_data["Cluster"] = clustered_data["Cluster"].astype(int)
            if set(clustered_data["Cluster"].unique()) != {0, 1}:
                raise ValueError(f"Label '{label}' must be binary (0 and 1)")

        clustered_data["Cluster"] = clustered_data["Cluster"].apply(lambda x: bool(x))

        # Calculate limits
        x_min, x_max = clustered_data["X1"].min(), clustered_data["X1"].max()
        y_min, y_max = clustered_data["X2"].min(), clustered_data["X2"].max()
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)

        # Fixed colours
        unique_clusters = sorted(clustered_data["Cluster"].unique())
        colors = sns.color_palette("tab10", n_colors=len(unique_clusters))
        cluster_color_map = {
            cluster: colors[i] for i, cluster in enumerate(unique_clusters)
        }

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()

        for idx in [0, 1]:
            subset = clustered_data[clustered_data["Cluster"] == bool(idx)]
            sns.scatterplot(
                data=subset,
                x="X1",
                y="X2",
                hue="Cluster",
                palette=cluster_color_map,
                alpha=0.7,
                s=50,
                ax=axes[idx],
            )
            axes[idx].set_xlabel("Component 1", fontsize=12)
            axes[idx].set_ylabel("Component 2", fontsize=12)
            axes[idx].set_xlim(x_limits)
            axes[idx].set_ylim(y_limits)
            axes[idx].grid()

        fig.suptitle(
            f"Redução da dimensionalidade do Autoencoder - Tem/Não tem {label}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save and display (safe method)
        self._save_figure_safely(fig, savepath)
        plt.show()

        logger.info(f"Displayed autoencoder data colored by '{label}'")

    def show_clustered_autoencoder(
        self,
        selected_clusters: Optional[List[int]] = None,
        seperate_clusters: bool = True,
        savepath: Optional[str] = None,
    ) -> None:
        """
        Visualises clustering in the autoencoder latent space.

        Args:
            selected_clusters: Specific clusters to visualise
            seperate_clusters: If True, shows each cluster in a separate subplot
            savepath: Path to save

        Examples:
            >>> helper.show_clustered_autoencoder(
            ...     selected_clusters=[0, 1, 2],
            ...     savepath='output/ae_clusters.png'
            ... )
        """
        if self._clustered_data_autoencoder is None:
            raise ValueError(
                "No autoencoder data available. "
                "Run set_clustered_autoencoder() first."
            )

        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        clustered_data = self._clustered_data_autoencoder.copy()
        clustered_data["Cluster"] = self._clustered_data["Cluster"].values

        if selected_clusters is not None:
            clustered_data["Cluster"] = clustered_data["Cluster"].apply(
                lambda x: x if x in selected_clusters else 100
            )
            clusters_to_show = selected_clusters
        else:
            clusters_to_show = sorted(clustered_data["Cluster"].unique())

        clustered_data = clustered_data.sort_values(by="Cluster")

        # Calculate limits
        x_min, x_max = clustered_data["X1"].min(), clustered_data["X1"].max()
        y_min, y_max = clustered_data["X2"].min(), clustered_data["X2"].max()
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)

        # Fixed colours
        colors = sns.color_palette("tab10", n_colors=len(clusters_to_show))
        cluster_color_map = {
            cluster: colors[i] for i, cluster in enumerate(clusters_to_show)
        }

        # Create subplots
        if seperate_clusters:
            rows = (len(clusters_to_show) + 1) // 2
            cols = 2
            fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
            axes = axes.flatten()

            for idx, cluster_id in enumerate(clusters_to_show):
                cluster_data = clustered_data[clustered_data["Cluster"] == cluster_id]

                sns.scatterplot(
                    data=cluster_data,
                    x="X1",
                    y="X2",
                    hue="Cluster",
                    palette=cluster_color_map,
                    alpha=0.7,
                    s=50,
                    ax=axes[idx],
                    legend=True,
                )
                axes[idx].set_xlabel("Component 1", fontsize=12)
                axes[idx].set_ylabel("Component 2", fontsize=12)
                axes[idx].set_title(
                    f"Cluster {cluster_id} (n={len(cluster_data)})",
                    fontsize=13,
                    fontweight="bold",
                )
                axes[idx].set_xlim(x_limits)
                axes[idx].set_ylim(y_limits)
                axes[idx].grid(alpha=0.3)

            # Hide unused subplots
            for idx in range(len(clusters_to_show), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(
                f"Visualização dos Clusters no Espaço Latente do Autoencoder\n"
                f"Clusters: {clusters_to_show}",
                fontsize=16,
                fontweight="bold",
                y=0.995,
            )

        else:

            # Remove cluster = 100
            clustered_data = clustered_data[clustered_data["Cluster"] != 100]

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                data=clustered_data,
                x="X1",
                y="X2",
                hue="Cluster",
                palette=cluster_color_map,
                alpha=0.7,
                s=50,
                ax=ax,
            )
            ax.set_xlabel("Component 1", fontsize=12)
            ax.set_ylabel("Component 2", fontsize=12)
            ax.set_title(
                "Visualização dos Clusters no Espaço Latente do Autoencoder",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.grid(alpha=0.3)
            ax.legend(title="Cluster")

        plt.tight_layout()

        # Save and display (safe method)
        self._save_figure_safely(fig, savepath)
        plt.show()

        logger.info(f"Displayed autoencoder clusters: {len(clusters_to_show)} clusters")

    # ==================== METRICS ====================

    def get_metrics(
        self,
        consider_noise: bool = True,
    ) -> Dict[str, float]:
        """
        Calculates clustering quality metrics.

        Args:
            consider_noise: If True, considers noise points

        Returns:
            Dictionary with metrics (silhouette, dbcv, dsi, disco)

        Raises:
            ImportError: If ClusterMetrics is not available
            ValueError: If clustered data is not available

        Examples:
            >>> metrics = helper.get_metrics()
            >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
        """
        if not HAS_METRICS:
            raise ImportError(
                "ClusterMetricsModule not available. " "Cannot calculate metrics."
            )

        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        if self._metrics_calculator is None:
            raise ValueError("ClusterMetricsModule not available")

        data = self._clustered_data.copy()
        X = data.drop(
            columns=["Cluster", "subject_id", "hadm_id"], errors="ignore"
        ).values
        labels = data["Cluster"].values

        result = {
            "silhouette": self._metrics_calculator.silhouette_score(
                X, labels, consider_noise
            ),
            "dbcv": self._metrics_calculator.dbcv_index(X, labels),
            "dsi": self._metrics_calculator.dsi_index(
                X, labels, consider_noise=consider_noise
            ),
            "disco": self._metrics_calculator.disco_index(X, labels),
        }

        logger.info(
            f"Calculated metrics: silhouette={result['silhouette']:.3f}, "
            f"dbcv={result['dbcv']:.3f}, dsi={result['dsi']:.3f}, "
            f"disco={result['disco']:.3f}"
        )

        return result

    def single_metric(
        self,
        metric_name: str,
        consider_noise: bool = True,
    ) -> float:
        """
        Calculates a specific metric.

        Args:
            metric_name: Metric name ('silhouette', 'dbcv', 'dsi', 'disco')
            consider_noise: If True, considers noise points

        Returns:
            Metric value

        Raises:
            ValueError: If metric is invalid

        Examples:
            >>> silhouette = helper.single_metric('silhouette')
        """
        valid_metrics = {"silhouette", "dbcv", "dsi", "disco"}
        if metric_name not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric_name}'. " f"Choose from {valid_metrics}"
            )

        if not HAS_METRICS:
            raise ImportError("ClusterMetricsModule not available")

        if self._clustered_data is None:
            raise ValueError("No clustered data available")

        if self._metrics_calculator is None:
            raise ValueError("ClusterMetricsModule not available")

        data = self._clustered_data.copy()
        X = data.drop(
            columns=["Cluster", "subject_id", "hadm_id"], errors="ignore"
        ).values
        labels = data["Cluster"].values

        if metric_name == "silhouette":
            value = self._metrics_calculator.silhouette_score(X, labels, consider_noise)
        elif metric_name == "dbcv":
            value = self._metrics_calculator.dbcv_index(X, labels)
        elif metric_name == "dsi":
            value = self._metrics_calculator.dsi_index(
                X, labels, consider_noise=consider_noise
            )
        else:  # disco
            value = self._metrics_calculator.disco_index(X, labels)

        logger.debug(f"Calculated {metric_name}: {value:.3f}")
        return value


# ==================== END OF CLASS ====================

if __name__ == "__main__":
    logger.info("ClusterBaseHelper module loaded successfully")
    logger.info("Version 2.0 - Refactored and optimized")
