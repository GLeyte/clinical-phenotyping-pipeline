"""
Association Helper Module - Association Rule Mining

This module provides tools for data preparation and application of
association rule mining algorithms (Apriori, FP-Growth) to medical
and transactional data.

Date: 2026-01-19
"""

import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AssociationHelper:
    """
    Helper class for association rule mining and market basket analysis.

    This class facilitates the conversion of numerical data to categorical,
    transformation to transactional format, and application of frequent
    pattern mining algorithms (Apriori, FP-Growth).

    Attributes:
        DEFAULT_BINS: Default number of bins for discretisation
        DEFAULT_MIN_SUPPORT: Default minimum support for itemsets
        DEFAULT_MIN_CONFIDENCE: Default minimum confidence for rules
        FULL_CATEGORICAL_FEATURES: Full list of medical categorical features

    Examples:
        >>> helper = AssociationHelper(data=df)
        >>> itemsets, rules = helper.run_apriori(min_support=0.05)
        >>> helper.display_top_rules(rules, top_n=10)
    """

    # Class constants
    DEFAULT_BINS = 3
    DEFAULT_MIN_SUPPORT = 0.01
    DEFAULT_MIN_CONFIDENCE = 0.5
    DEFAULT_METRIC = "confidence"
    DEFAULT_MIN_THRESHOLD = 0.1

    # Full categorical features (medical)
    FULL_CATEGORICAL_FEATURES = [
        "myocardial_infarct",
        "congestive_heart_failure",
        "peripheral_vascular_disease",
        "cerebrovascular_disease",
        "dementia",
        "chronic_pulmonary_disease",
        "rheumatic_disease",
        "peptic_ulcer_disease",
        "mild_liver_disease",
        "diabetes_without_cc",
        "diabetes_with_cc",
        "paraplegia",
        "renal_disease",
        "malignant_cancer",
        "severe_liver_disease",
        "metastatic_solid_tumor",
        "aids",
        "gender_M",
        "died_in_stay",
        "died",
        "COVID",
    ]

    def __init__(
        self,
        data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ):
        """
        Initialises the AssociationHelper.

        Args:
            data: DataFrame with the data to be analysed
            categorical_features: List of categorical features. If None, uses FULL_CATEGORICAL_FEATURES

        Raises:
            TypeError: If data is not a DataFrame
            ValueError: If the DataFrame is empty

        Examples:
            >>> helper = AssociationHelper(df, categorical_features=['gender_M', 'died'])
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(data)}")

        if data.empty:
            raise ValueError("DataFrame cannot be empty")

        # Use default list if not provided
        if categorical_features is None:
            categorical_features = self.FULL_CATEGORICAL_FEATURES

        # Filter only features that exist in the DataFrame
        categorical_features = [
            col for col in data.columns if col in categorical_features
        ]

        numerical_features = [
            col for col in data.columns if col not in categorical_features
        ]

        self._data = data.copy()  # Copy to avoid external modifications
        self._numerical_features = numerical_features
        self._categorical_features = categorical_features
        self._transactional_data = None

        logger.info(
            f"AssociationHelper initialized with {len(data)} records, "
            f"{len(categorical_features)} categorical and {len(numerical_features)} numerical features"
        )

    def update_data_cluster(
        self,
        cluster_of_interest: int,
        used_features: List[str],
        removed_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Filters data by specific cluster and selected features.

        IMPORTANT: This method modifies the object's internal state (_data).
        If you need to keep the original data, create a new instance.

        Args:
            cluster_of_interest: Cluster number to filter
            used_features: List of features to keep
            removed_features: List of features to remove (applied after used_features)

        Returns:
            Filtered and updated DataFrame

        Raises:
            ValueError: If the 'Cluster' column does not exist
            KeyError: If specified features do not exist

        Examples:
            >>> helper.update_data_cluster(
            ...     cluster_of_interest=1,
            ...     used_features=['age', 'gender_M', 'diabetes'],
            ...     removed_features=['subject_id']
            ... )
        """
        # Validation
        if "Cluster" not in self._data.columns:
            raise ValueError("DataFrame must contain a 'Cluster' column")

        if cluster_of_interest not in self._data["Cluster"].unique():
            logger.warning(
                f"Cluster {cluster_of_interest} not found in data. "
                f"Available clusters: {sorted(self._data['Cluster'].unique())}"
            )

        # Filter by cluster
        original_len = len(self._data)
        self._data = self._data[
            self._data["Cluster"] == cluster_of_interest
        ].reset_index(drop=True)

        logger.info(
            f"Filtered to cluster {cluster_of_interest}: "
            f"{len(self._data)} records (from {original_len})"
        )

        # Filter features
        missing_features = [f for f in used_features if f not in self._data.columns]
        if missing_features:
            raise KeyError(f"Features not found in DataFrame: {missing_features}")

        self._data = self._data[used_features]

        # Remove specified features
        if removed_features:
            self._data.drop(columns=removed_features, inplace=True, errors="ignore")

        # Update feature lists
        self._numerical_features = [
            col for col in self._data.columns if col in self._numerical_features
        ]
        self._categorical_features = [
            col for col in self._data.columns if col in self._categorical_features
        ]

        logger.info(
            f"Updated data: {len(self._data.columns)} columns, "
            f"{len(self._numerical_features)} numerical, "
            f"{len(self._categorical_features)} categorical"
        )

        return self._data.copy()

    def convert_numerical_to_categorical(
        self,
        bins: Optional[Union[int, List, Dict]] = None,
        labels: Union[bool, List, Dict] = False,
        strategy: Union[str, Dict] = "uniform",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts numerical columns to categorical using binning.

        PERFORMANCE: Uses vectorised operations for efficiency on large datasets.

        Args:
            bins: Number of bins or custom edges
                - int: Number of bins for all features
                - list: Custom bin edges
                - dict: Feature-specific bins {'feature': 5}
            labels: Labels for the created bins
                - False: Uses integer indicators
                - True: Generates default labels
                - list/dict: Custom labels
            strategy: Binning strategy
                - 'uniform': Equal-width bins
                - 'quantile': Bins with approximately equal counts
                - dict: Strategy per feature {'feature': 'quantile'}

        Returns:
            Tuple of (converted DataFrame, DataFrame with bin boundaries)

        Examples:
            >>> data, limits = helper.convert_numerical_to_categorical(
            ...     bins={'age': 5, 'weight': 3},
            ...     strategy='quantile'
            ... )
        """
        if bins is None:
            bins = self.DEFAULT_BINS

        data = self._data.copy()
        numerical_cols = self._numerical_features

        if not numerical_cols:
            logger.warning("No numerical columns found for conversion")
            return data, pd.DataFrame()

        limits = {}

        for col in numerical_cols:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            # Determine parameters for this column
            current_bins = (
                bins if not isinstance(bins, dict) else bins.get(col, self.DEFAULT_BINS)
            )
            current_labels = (
                labels if not isinstance(labels, dict) else labels.get(col, False)
            )
            current_strategy = (
                strategy
                if not isinstance(strategy, dict)
                else strategy.get(col, "uniform")
            )

            try:
                # Store original boundaries
                col_min, col_max = data[col].min(), data[col].max()
                limits[col] = [col_min, col_max]

                # Apply binning
                if current_strategy == "uniform":
                    data[col] = pd.cut(
                        data[col].astype(float),
                        bins=current_bins,
                        labels=current_labels,
                        include_lowest=True,
                        retbins=False,
                        duplicates="drop",
                    )
                elif current_strategy == "quantile":
                    data[col] = pd.qcut(
                        data[col].astype(float),
                        q=current_bins,
                        labels=current_labels,
                        duplicates="drop",
                    )
                else:
                    # Assume current_bins is a list of edges
                    data[col] = pd.cut(
                        data[col].astype(float),
                        bins=current_bins,
                        labels=current_labels,
                        include_lowest=True,
                        retbins=False,
                        duplicates="drop",
                    )

                # VECTORISED: Create descriptive labels efficiently
                data[col] = self._create_descriptive_labels_vectorized(
                    data[col], col, col_min, col_max
                )

                logger.debug(
                    f"Converted column '{col}' to {data[col].nunique()} categories"
                )

            except Exception as e:
                logger.error(f"Failed to convert column '{col}': {e}")
                raise

        # Convert limits to DataFrame
        limits_df = pd.DataFrame.from_dict(
            limits, orient="index", columns=["min", "max"]
        )
        limits_df.index.name = "feature"
        limits_df.reset_index(inplace=True)

        logger.info(f"Converted {len(numerical_cols)} numerical columns to categorical")
        return data, limits_df

    def _create_descriptive_labels_vectorized(
        self, series: pd.Series, col_name: str, col_min: float, col_max: float
    ) -> pd.Series:
        """
        Creates descriptive labels for bins in a vectorised manner.

        Args:
            series: Series with categorical values (bins)
            col_name: Column name
            col_min: Original minimum value
            col_max: Original maximum value

        Returns:
            Series with descriptive labels in the format "colname[min-max]"
        """
        # Check if the Series is of Categorical type
        if not isinstance(series.dtype, pd.CategoricalDtype):
            logger.warning(
                f"Series '{col_name}' is not categorical. "
                "Converting to string directly."
            )
            return series.astype(str)

        # Convert categories to numeric
        n_categories = series.nunique()

        if n_categories == 0:
            return series.astype(str)

        # Map categories to numeric indices
        # Convert to codes (int) instead of mapping manually
        numeric_series = pd.Series(series.cat.codes, index=series.index)

        # Handle NaN values (code = -1 in pandas Categorical)
        valid_mask = numeric_series >= 0

        # Calculate lower and upper boundaries in a vectorised manner
        range_size = (col_max - col_min) / n_categories
        inferior_margin = (numeric_series * range_size) + col_min
        superior_margin = ((numeric_series + 1) * range_size) + col_min

        # Create vectorised descriptive labels only for valid values
        labels = pd.Series(index=series.index, dtype=str)
        labels[valid_mask] = (
            col_name
            + "["
            + inferior_margin[valid_mask].round(1).astype(str)
            + "-"
            + superior_margin[valid_mask].round(1).astype(str)
            + "]"
        )
        labels[~valid_mask] = np.nan  # Keep NaN where applicable

        return labels

    def convert_boolean_to_categorical(
        self,
        df: Optional[pd.DataFrame] = None,
        use_nan: bool = False,
    ) -> pd.DataFrame:
        """
        Converts boolean columns to categorical format.

        Args:
            df: Input DataFrame. If None, uses internal data
            use_nan: If True, converts False to NaN; otherwise to "colname_False"

        Returns:
            DataFrame with boolean columns converted

        Examples:
            >>> # True becomes "diabetes", False becomes NaN
            >>> data = helper.convert_boolean_to_categorical(use_nan=True)

            >>> # True becomes "diabetes_True", False becomes "diabetes_False"
            >>> data = helper.convert_boolean_to_categorical(use_nan=False)
        """
        if df is None:
            df = self._data.copy()
        else:
            df = df.copy()

        for col in self._categorical_features:
            if col not in df.columns:
                continue

            if use_nan:
                # True → "colname", False → NaN
                df[col] = df[col].apply(lambda x: col if x == 1 else np.nan)
            else:
                # True → "colname_True", False → "colname_False"
                df[col] = df[col].apply(
                    lambda x: f"{col}_True" if x == 1 else f"{col}_False"
                )

        logger.debug(f"Converted {len(self._categorical_features)} boolean columns")
        return df

    def convert_to_transactional(
        self,
        df: Optional[pd.DataFrame] = None,
        transaction_id_col: Optional[str] = None,
    ) -> Union[pd.DataFrame, List[List[str]]]:
        """
        Converts DataFrame to transactional format for mining.

        Args:
            df: Input DataFrame. If None, uses internal data
            transaction_id_col: Column for transaction ID. If None, each row is a transaction

        Returns:
            If transaction_id_col provided: DataFrame with columns ['transaction_id', 'items']
            If None: List of lists, where each inner list contains the items of a transaction

        Raises:
            ValueError: If transaction_id_col does not exist in the DataFrame

        Examples:
            >>> # Each row is a transaction
            >>> transactions = helper.convert_to_transactional()
            >>> # [[item1, item2], [item3, item4], ...]

            >>> # With transaction IDs
            >>> trans_df = helper.convert_to_transactional(transaction_id_col='hadm_id')
            >>> # DataFrame with columns: hadm_id, items
        """
        if df is None:
            df = self._data.copy()
        else:
            df = df.copy()

        if transaction_id_col:
            if transaction_id_col not in df.columns:
                raise ValueError(
                    f"Transaction ID column '{transaction_id_col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

            # Value columns (excluding ID)
            value_vars = [col for col in df.columns if col != transaction_id_col]
            if not value_vars:
                raise ValueError(
                    "No value columns found to create items from, "
                    "aside from the transaction ID column."
                )

            # Melt to long format
            melted_df = df.melt(
                id_vars=[transaction_id_col],
                value_vars=value_vars,
                var_name="item_category",
                value_name="item_value",
            )

            # Create descriptive items
            melted_df["item"] = (
                melted_df["item_category"].astype(str)
                + "_"
                + melted_df["item_value"].astype(str)
            )

            # Remove NaNs
            melted_df.dropna(subset=["item_value"], inplace=True)

            # Group by transaction_id
            transactional_df = (
                melted_df.groupby(transaction_id_col)["item"]
                .apply(list)
                .reset_index(name="items")  # type: ignore[no-matching-overload]
            )

            self._transactional_data = transactional_df
            logger.info(
                f"Created transactional data: {len(transactional_df)} transactions"
            )
            return transactional_df
        else:
            # Each row is a transaction
            transactions = []
            for _, row in df.iterrows():
                transaction_items = [
                    str(value)
                    for value in row.dropna().astype(str)
                    if str(value).lower() != "nan"
                ]
                if transaction_items:
                    transactions.append(transaction_items)

            self._transactional_data = transactions
            logger.info(f"Created transactional data: {len(transactions)} transactions")
            return transactions

    def _prepare_data_for_mining(
        self, data: Union[pd.DataFrame, List[List[str]]]
    ) -> pd.DataFrame:
        """
        Prepares transactional data for mining by encoding in one-hot format.

        Args:
            data: Transactional data (list of lists or DataFrame with 'items' column)

        Returns:
            One-hot encoded DataFrame suitable for mlxtend algorithms

        Raises:
            ValueError: If data format is invalid
        """
        # Determine data format
        if isinstance(data, pd.DataFrame):
            if "items" not in data.columns:
                raise ValueError(
                    "DataFrame must have an 'items' column containing lists of items. "
                    f"Available columns: {list(data.columns)}"
                )
            transactions = data["items"].tolist()
        elif isinstance(data, list):
            transactions = data
        else:
            raise ValueError(
                f"Data must be a list of lists or a DataFrame with 'items' column. "
                f"Got {type(data)}"
            )

        # Encode using TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

        logger.debug(
            f"Encoded {len(transactions)} transactions with {len(te.columns_)} unique items"
        )
        return encoded_df

    def _run_frequent_pattern_mining(
        self,
        algorithm: str,
        data: Optional[Union[pd.DataFrame, List[List[str]]]] = None,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        return_rules: bool = True,
        metric: Optional[str] = None,
        min_threshold: Optional[float] = None,
        verbose: int = 0,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Unified internal method to run pattern mining algorithms.

        Args:
            algorithm: 'apriori' or 'fpgrowth'
            data: Transactional data
            min_support: Minimum support for itemsets
            min_confidence: Minimum confidence for rules
            return_rules: If True, returns rules in addition to itemsets
            metric: Metric for rule evaluation
            min_threshold: Minimum threshold for metric
            verbose: Verbosity level

        Returns:
            (frequent_itemsets, rules) if return_rules=True, otherwise just frequent_itemsets
        """
        # Default values
        if min_support is None:
            min_support = self.DEFAULT_MIN_SUPPORT
        if min_confidence is None:
            min_confidence = self.DEFAULT_MIN_CONFIDENCE
        if metric is None:
            metric = self.DEFAULT_METRIC
        if min_threshold is None:
            min_threshold = self.DEFAULT_MIN_THRESHOLD

        # Use internal data if not provided
        if data is None:
            if self._transactional_data is None:
                raise ValueError(
                    "No transactional data available. "
                    "Please run convert_to_transactional() first."
                )
            data = self._transactional_data

        try:
            # Prepare data
            encoded_df = self._prepare_data_for_mining(data)

            # Select algorithm
            if algorithm.lower() == "apriori":
                logger.info(
                    f"Finding frequent itemsets with Apriori (min_support={min_support})"
                )
                frequent_itemsets = apriori(
                    encoded_df,
                    min_support=min_support,
                    use_colnames=True,
                    verbose=verbose,
                )
            elif algorithm.lower() == "fpgrowth":
                logger.info(
                    f"Finding frequent itemsets with FP-Growth (min_support={min_support})"
                )
                frequent_itemsets = fpgrowth(
                    encoded_df,
                    min_support=min_support,
                    use_colnames=True,
                    verbose=verbose,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Check result
            if frequent_itemsets.empty:
                logger.warning(
                    "No frequent itemsets found. Try lowering the min_support threshold."
                )
                return (
                    (frequent_itemsets, pd.DataFrame())
                    if return_rules
                    else frequent_itemsets
                )

            logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")

            # Generate rules if requested
            if return_rules:
                logger.info(
                    f"Generating association rules with {metric}>={min_threshold}"
                )

                try:
                    rules = association_rules(
                        frequent_itemsets, metric=metric, min_threshold=min_threshold
                    )
                    logger.info(f"Found {len(rules)} association rules")
                    return frequent_itemsets, rules
                except ValueError as e:
                    logger.warning(f"Could not generate rules: {e}")
                    return frequent_itemsets, pd.DataFrame()

            return frequent_itemsets

        except Exception as e:
            logger.error(f"Error in {algorithm} algorithm: {e}")
            raise

    def apriori(
        self,
        data: Optional[Union[pd.DataFrame, List[List[str]]]] = None,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        return_rules: bool = True,
        metric: Optional[str] = None,
        min_threshold: Optional[float] = None,
        verbose: int = 0,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Applies the Apriori algorithm to find frequent itemsets and association rules.

        Args:
            data: Transactional data. If None, uses internal data
            min_support: Minimum support for itemsets (default: 0.01)
            min_confidence: Minimum confidence for rules (default: 0.5)
            return_rules: If True, returns rules in addition to itemsets
            metric: Metric for evaluation ('confidence', 'lift', 'support')
            min_threshold: Minimum threshold for metric
            verbose: Verbosity level (0=silent, 1=verbose)

        Returns:
            (frequent_itemsets, rules) if return_rules=True, otherwise just frequent_itemsets

        Examples:
            >>> itemsets, rules = helper.apriori(min_support=0.05, min_confidence=0.7)
            >>> print(f"Found {len(rules)} rules")
        """
        return self._run_frequent_pattern_mining(
            algorithm="apriori",
            data=data,
            min_support=min_support,
            min_confidence=min_confidence,
            return_rules=return_rules,
            metric=metric,
            min_threshold=min_threshold,
            verbose=verbose,
        )

    def fp_growth(
        self,
        data: Optional[Union[pd.DataFrame, List[List[str]]]] = None,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        return_rules: bool = True,
        metric: Optional[str] = None,
        min_threshold: Optional[float] = None,
        verbose: int = 0,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Applies the FP-Growth algorithm to find frequent itemsets and association rules.

        FP-Growth is generally faster than Apriori for large datasets.

        Args:
            data: Transactional data. If None, uses internal data
            min_support: Minimum support for itemsets (default: 0.01)
            min_confidence: Minimum confidence for rules (default: 0.5)
            return_rules: If True, returns rules in addition to itemsets
            metric: Metric for evaluation ('confidence', 'lift', 'support')
            min_threshold: Minimum threshold for metric
            verbose: Verbosity level (0=silent, 1=verbose)

        Returns:
            (frequent_itemsets, rules) if return_rules=True, otherwise just frequent_itemsets

        Examples:
            >>> itemsets, rules = helper.fp_growth(min_support=0.05)
            >>> top_rules = rules.nlargest(10, 'lift')
        """
        return self._run_frequent_pattern_mining(
            algorithm="fpgrowth",
            data=data,
            min_support=min_support,
            min_confidence=min_confidence,
            return_rules=return_rules,
            metric=metric,
            min_threshold=min_threshold,
            verbose=verbose,
        )

    def run_apriori(
        self,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        return_rules: bool = True,
        metric: Optional[str] = None,
        min_threshold: Optional[float] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Convenience method: runs full pipeline with Apriori.

        Pipeline: numerical→categorical → boolean→categorical → transactional → Apriori

        Args:
            min_support: Minimum support (default: 0.01)
            min_confidence: Minimum confidence (default: 0.5)
            return_rules: If True, returns rules
            metric: Metric for evaluation
            min_threshold: Minimum threshold for metric

        Returns:
            (frequent_itemsets, rules) if return_rules=True, otherwise just frequent_itemsets

        Examples:
            >>> helper = AssociationHelper(df)
            >>> itemsets, rules = helper.run_apriori(min_support=0.05)
        """
        logger.info("Running full Apriori pipeline")

        categorized_data, _ = self.convert_numerical_to_categorical()
        categorized_data = self.convert_boolean_to_categorical(
            categorized_data, use_nan=True
        )
        transactional_data = self.convert_to_transactional(categorized_data)

        return self.apriori(
            data=transactional_data,
            min_support=min_support,
            min_confidence=min_confidence,
            return_rules=return_rules,
            metric=metric,
            min_threshold=min_threshold,
        )

    def run_fp_growth(
        self,
        min_support: Optional[float] = None,
        min_confidence: Optional[float] = None,
        return_rules: bool = True,
        metric: Optional[str] = None,
        min_threshold: Optional[float] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Convenience method: runs full pipeline with FP-Growth.

        Pipeline: numerical→categorical → boolean→categorical → transactional → FP-Growth

        Args:
            min_support: Minimum support (default: 0.01)
            min_confidence: Minimum confidence (default: 0.5)
            return_rules: If True, returns rules
            metric: Metric for evaluation
            min_threshold: Minimum threshold for metric

        Returns:
            (frequent_itemsets, rules) if return_rules=True, otherwise just frequent_itemsets

        Examples:
            >>> helper = AssociationHelper(df)
            >>> itemsets, rules = helper.run_fp_growth(min_support=0.05)
        """
        logger.info("Running full FP-Growth pipeline")

        categorized_data, _ = self.convert_numerical_to_categorical()
        categorized_data = self.convert_boolean_to_categorical(
            categorized_data, use_nan=True
        )
        transactional_data = self.convert_to_transactional(categorized_data)

        return self.fp_growth(
            data=transactional_data,
            min_support=min_support,
            min_confidence=min_confidence,
            return_rules=return_rules,
            metric=metric,
            min_threshold=min_threshold,
        )

    def filter_rules_by_items(
        self,
        antecedent: List[str],
        consequent: List[str],
        rules: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filters rules that contain specific antecedents and consequents.

        Args:
            antecedent: Items that must be in the antecedents
            consequent: Items that must be in the consequents
            rules: DataFrame with association rules

        Returns:
            Filtered DataFrame with matching rules

        Examples:
            >>> filtered = helper.filter_rules_by_items(
            ...     antecedent=['diabetes'],
            ...     consequent=['heart_failure'],
            ...     rules=rules
            ... )
        """
        if rules.empty:
            logger.warning("No rules provided for filtering")
            return rules

        filtered_rules = rules[
            (rules["antecedents"].apply(lambda x: set(antecedent).issubset(x)))
            & (rules["consequents"].apply(lambda x: set(consequent).issubset(x)))
        ]

        logger.info(
            f"Filtered {len(rules)} rules to {len(filtered_rules)} "
            f"matching criteria"
        )
        return filtered_rules

    def display_top_rules(
        self,
        rules: pd.DataFrame,
        top_n: int = 10,
        sort_by: str = "lift",
    ) -> None:
        """
        Displays the top N association rules sorted by metric.

        Args:
            rules: DataFrame with association rules
            top_n: Number of rules to display
            sort_by: Metric for sorting ('lift', 'confidence', 'support')

        Examples:
            >>> helper.display_top_rules(rules, top_n=5, sort_by='lift')
        """
        if rules.empty:
            logger.warning("No rules to display")
            print("No rules to display.")
            return

        if sort_by not in rules.columns:
            logger.error(
                f"Metric '{sort_by}' not found in rules. "
                f"Available: {list(rules.columns)}"
            )
            print(
                f"Metric '{sort_by}' not found in rules. "
                f"Available metrics: {list(rules.columns)}"
            )
            return

        top_rules = rules.nlargest(top_n, sort_by)

        print(f"\n{'='*80}")
        print(f"Top {top_n} Association Rules (sorted by {sort_by})")
        print(f"{'='*80}")

        for idx, rule in top_rules.iterrows():
            antecedents = ", ".join(list(rule["antecedents"]))
            consequents = ", ".join(list(rule["consequents"]))

            print(f"\nRule {idx + 1}:")                 # type: ignore[index]
            print(f"  {antecedents} → {consequents}")
            print(f"  Support:    {rule['support']:.4f}")
            print(f"  Confidence: {rule['confidence']:.4f}")
            print(f"  Lift:       {rule['lift']:.4f}")

            if "conviction" in rule:
                print(f"  Conviction: {rule['conviction']:.4f}")

        print(f"{'='*80}\n")

    def get_transactional_data(self) -> Union[pd.DataFrame, List[List[str]], None]:
        """
        Returns the stored transactional data.

        Returns:
            Transactional data or None if not yet created
        """
        return self._transactional_data

    def get_data(self) -> pd.DataFrame:
        """
        Returns a copy of the internal data.

        Returns:
            Copy of the internal DataFrame
        """
        return self._data.copy()

    def get_feature_info(self) -> Dict[str, List[str]]:
        """
        Returns information about categorical and numerical features.

        Returns:
            Dictionary with lists of features by type

        Examples:
            >>> info = helper.get_feature_info()
            >>> print(f"Categorical: {info['categorical']}")
            >>> print(f"Numerical: {info['numerical']}")
        """
        return {
            "categorical": self._categorical_features.copy(),
            "numerical": self._numerical_features.copy(),
            "all": list(self._data.columns),
        }


# Example usage
if __name__ == "__main__":
    logger.info("AssociationHelper module loaded successfully")
