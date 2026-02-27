"""
Multiple Imputation by Chained Equations (MICE) Module

This module provides two implementations for handling missing data:
- MICE_Imputer: Sklearn-based IterativeImputer (BayesianRidge)
- MiceForestImputer: miceforest-based LightGBM with memory optimisation

Both classes inherit from BaseMICEImputer and follow the
sklearn BaseEstimator/TransformerMixin interface for pipeline compatibility.
"""

import logging
import pandas as pd
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from typing import List, Tuple, Union, Dict, Optional, cast
import re
import gc
import warnings

try:
    import miceforest as mf

    HAS_MICEFOREST = True
except ImportError:
    HAS_MICEFOREST = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Base MICE Imputer Class
# ============================================================================


class BaseMICEImputer(BaseEstimator, TransformerMixin):
    """
    Base class for Multiple Imputation by Chained Equations (MICE).

    Handles shared logic for aggregation, train/test splitting and result grouping.
    """

    def __init__(self, max_iter=10, m_imputations=20, random_state=42, verbose=False):
        self.max_iter = max_iter
        self.m_imputations = m_imputations
        self.random_state = random_state
        self.verbose = verbose
        self.is_fitted_ = False
        self.column_names_ = None

    def fit(self, X, y=None):
        """Abstract fit method"""
        raise NotImplementedError("Subclasses must implement fit()")

    def transform(self, X, return_multiple=True):
        """Abstract transform method"""
        raise NotImplementedError("Subclasses must implement transform()")

    def fit_transform(self, X, y=None, **fit_params):
        """Fits and transforms in a single step"""
        return_multiple = fit_params.pop("return_multiple", True)
        return self.fit(X, y).transform(X, return_multiple=return_multiple)


    def aggregate_datasets(self, datasets_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregates multiple imputed datasets by computing the mean of numerical columns
        and taking the mode for categorical columns.

        Parameters
        ----------
        datasets_list : List[pd.DataFrame]
            List of imputed DataFrames to aggregate

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame

        Raises
        ------
        ValueError
            If datasets_list is empty
        """
        if not datasets_list:
            raise ValueError("No datasets provided for aggregation")

        if len(datasets_list) == 1:
            return datasets_list[0].copy()

        df_template = datasets_list[0]
        numeric_cols = df_template.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df_template.select_dtypes(exclude=[np.number]).columns

        if self.verbose:
            logger.info(f"Aggregating {len(datasets_list)} datasets...")

        result = df_template.copy()

        if len(numeric_cols) > 0:
            numeric_sum = pd.DataFrame(
                0.0, index=df_template.index, columns=numeric_cols
            )
            for df in datasets_list:
                numeric_sum += df[numeric_cols]
            result[numeric_cols] = numeric_sum / len(datasets_list)

        if len(non_numeric_cols) > 0:
            for col in non_numeric_cols:
                series_list = [df[col] for df in datasets_list]
                temp_df = pd.concat(series_list, axis=1)
                result[col] = temp_df.mode(axis=1).iloc[:, 0]

        if self.verbose:
            logger.info("Dataset aggregation completed!")

        return result

    def _restore_ignored_columns(
        self,
        imputed_list: List[pd.DataFrame],
        original_df: pd.DataFrame,
        used_cols: List[str],
    ) -> List[pd.DataFrame]:
        """
        Restores ignored columns and reorders to match the original DataFrame.

        Parameters
        ----------
        imputed_list : List[pd.DataFrame]
            List of imputed DataFrames (column subset)
        original_df : pd.DataFrame
            Original DataFrame with all columns
        used_cols : List[str]
            List of columns that were imputed

        Returns
        -------
        List[pd.DataFrame]
            List of restored DataFrames with all original columns
        """
        restored_list = []
        ignored_cols = [col for col in original_df.columns if col not in used_cols]

        for imp_df in imputed_list:
            if not isinstance(imp_df, pd.DataFrame):
                imp_df = pd.DataFrame(
                    imp_df, columns=used_cols, index=original_df.index
                )

            for col in ignored_cols:
                if col in original_df.columns:
                    imp_df[col] = original_df[col].values

            restored_list.append(imp_df[original_df.columns])

        return restored_list

    def train_test_impute(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        not_consider: List[str],
        return_multiple: bool = True,
    ) -> Union[
        Tuple[List[pd.DataFrame], List[pd.DataFrame]], Tuple[pd.DataFrame, pd.DataFrame]
    ]:
        """
        Standardised workflow for fitting on training data and imputing both training and test sets,
        preserving ignored columns.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data
        df_test : pd.DataFrame
            Test data
        not_consider : List[str]
            Columns to exclude from imputation
        return_multiple : bool, default=True
            If True, returns list of imputed datasets; if False, returns aggregated dataset

        Returns
        -------
        Tuple
            (imputed_train, imputed_test) where each can be a list or a single DataFrame
        """
        used_cols = [col for col in df_train.columns if col not in not_consider]
        df_train_subset = df_train[used_cols].copy()
        df_test_subset = df_test[used_cols].copy()

        if self.verbose:
            logger.info("Fitting imputer on training data...")
        self.fit(df_train_subset)

        if self.verbose:
            logger.info("Imputing training data...")
        df_train_imputed_list = self.transform(df_train_subset, return_multiple=True)

        if self.verbose:
            logger.info("Imputing test data...")
        df_test_imputed_list = self.transform(df_test_subset, return_multiple=True)

        df_train_imputed_list = self._restore_ignored_columns(
            df_train_imputed_list, df_train, used_cols
        )
        df_test_imputed_list = self._restore_ignored_columns(
            df_test_imputed_list, df_test, used_cols
        )

        if return_multiple:
            return df_train_imputed_list, df_test_imputed_list
        else:
            return self.aggregate_datasets(
                df_train_imputed_list
            ), self.aggregate_datasets(df_test_imputed_list)

    def train_test_impute_with_aggregation(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, not_consider: List[str]
    ) -> Tuple[
        Optional[List[pd.DataFrame]],
        Optional[List[pd.DataFrame]],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Returns individual imputed datasets AND the consensus aggregate dataset.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data
        df_test : pd.DataFrame
            Test data
        not_consider : List[str]
            Columns to exclude from imputation

        Returns
        -------
        Tuple[Optional[List[pd.DataFrame]], Optional[List[pd.DataFrame]], pd.DataFrame, pd.DataFrame]
            (conjuntos_treino, conjuntos_teste, treino_agregado, teste_agregado)
            Conjuntos individuais são None se m_imputations == 1
        """
        train_result, test_result = self.train_test_impute(
            df_train, df_test, not_consider, return_multiple=True
        )
        train_datasets = cast(List[pd.DataFrame], train_result)
        test_datasets = cast(List[pd.DataFrame], test_result)

        if self.m_imputations == 1:
            return None, None, train_datasets[0], test_datasets[0]

        if self.verbose:
            logger.info("Aggregating multiple imputed datasets...")

        train_aggregated = self.aggregate_datasets(train_datasets)
        test_aggregated = self.aggregate_datasets(test_datasets)

        return train_datasets, test_datasets, train_aggregated, test_aggregated


# ============================================================================
# Sklearn-based MICE Imputer
# ============================================================================


class MICE_Imputer(BaseMICEImputer):
    """
    Implementação padrão baseada em Sklearn usando IterativeImputer (BayesianRidge).

    Cria múltiplas imputações ajustando IterativeImputer com diferentes sementes aleatórias,
    permitindo sample_posterior=True para variabilidade entre imputações.
    """

    def __init__(self, max_iter=10, m_imputations=20, random_state=42, verbose=False):
        super().__init__(max_iter, m_imputations, random_state, verbose)
        self.col_imputers_ = []

    def fit(self, X, y=None):
        """Ajusta múltiplas instâncias de IterativeImputer"""
        if isinstance(X, pd.DataFrame):
            self.column_names_ = X.columns.tolist()
            X_train_array = X.values.astype(float)
        else:
            self.column_names_ = None
            X_train_array = np.array(X, dtype=float)

        self._fit_internal(X_train_array)
        if self.verbose:
            logger.info("Fitting complete.")
        self.is_fitted_ = True
        return self

    def _fit_internal(self, X_train):
        """Ajusta m_imputations instâncias de IterativeImputer"""
        self.col_imputers_ = []
        iterator = range(self.m_imputations)

        if self.verbose and HAS_TQDM:
            iterator = tqdm(iterator, desc="Fitting MICE Imputers")

        for imp_idx in iterator:
            imputation_seed = (
                self.random_state + imp_idx if self.random_state is not None else None
            )
            col_imputer = IterativeImputer(
                estimator=BayesianRidge(),
                sample_posterior=True,
                max_iter=self.max_iter,
                tol=0.001,
                initial_strategy="mean",
                imputation_order="ascending",
                random_state=imputation_seed,
                verbose=0,
            )
            col_imputer.fit(X_train)
            self.col_imputers_.append(col_imputer)

    def transform(
        self, X, return_multiple=True
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        Transforma dados usando imputadores ajustados.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Dados a imputar
        return_multiple : bool, default=True
            Se True, retorna lista de conjuntos de dados imputados; se False, retorna primeiro conjunto

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            Conjunto(s) de dados imputado(s)
        """
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform")

        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            X_array = X.values.astype(float)
            index = X.index
        else:
            X_array = np.array(X, dtype=float)
            index = None

        imputed_datasets = []
        for col_imputer in self.col_imputers_:
            imputed_datasets.append(col_imputer.transform(X_array))

        if is_dataframe and self.column_names_ is not None:
            imputed_datasets = [
                pd.DataFrame(dataset, columns=self.column_names_, index=index)
                for dataset in imputed_datasets
            ]

        if return_multiple:
            return imputed_datasets
        else:
            if not imputed_datasets:
                raise ValueError("No imputed datasets available. Ensure the imputer is fitted correctly.")
            return imputed_datasets[0]


# ============================================================================
# LightGBM-based MICE Imputer with Memory Optimization
# ============================================================================


@dataclass
class MiceForestConfig:
    """Configuration parameters for MiceForestImputer"""

    mean_match_candidates: int = 5
    defrag_every: int = 4
    suppress_warnings: bool = False


class MiceForestImputer(BaseMICEImputer):
    """
    Implementação MICE baseada em LightGBM usando a biblioteca 'miceforest'.

    Apresenta gerenciamento otimizado de memória para evitar fragmentação de DataFrame através
    desfragmentação periódica durante iterações MICE.

    Parameters
    ----------
    max_iter : int, padrão=10
        Número máximo de iterações MICE
    m_imputations : int, padrão=20
        Número de conjuntos de dados imputados a gerar
    random_state : int, padrão=42
        Semente aleatória para reprodutibilidade
    config : MiceForestConfig, opcional
        Objeto de configuração para parâmetros avançados. Se None, usa padrões.
    verbose : bool, padrão=False
        Se deve imprimir informações de progresso
    """

    def __init__(
        self,
        max_iter=10,
        m_imputations=20,
        random_state=42,
        config: Optional[MiceForestConfig] = None,
        verbose=False,
    ):
        if not HAS_MICEFOREST:
            raise ImportError(
                "miceforest is not installed. Install it with: pip install miceforest"
            )

        super().__init__(max_iter, m_imputations, random_state, verbose)

        self.config = config or MiceForestConfig()
        self.mean_match_candidates = self.config.mean_match_candidates
        self.defrag_every = self.config.defrag_every
        self.suppress_warnings = self.config.suppress_warnings

        self.kernel_: Optional["mf.ImputationKernel"] = None
        self.original_column_names_: Optional[List[str]] = None
        self.safe_column_names_: Optional[List[str]] = None
        self.column_mapping_ = None
        self.training_data_hash_ = None

    # -------- Column Name Sanitization --------

    _UNSAFE_CHAR_PATTERN = re.compile(r'[\[\]{}<>:"\,\\/|?\*\s]')
    _MULTIPLE_UNDERSCORES = re.compile(r"_+")

    @classmethod
    def _sanitize_column_name(cls, col_name: str) -> str:
        """
        Remove ou substitui caracteres especiais que LightGBM não suporta.

        Parameters
        ----------
        col_name : str
            Nome da coluna a sanitizar

        Returns
        -------
        str
            Nome da coluna sanitizado
        """
        col_name = str(col_name)

        replacements = {
            "[": "_",
            "]": "_",
            "{": "_",
            "}": "_",
            ":": "_",
            '"': "",
            ",": "_",
            "<": "_lt_",
            ">": "_gt_",
            "\\": "_",
            "/": "_",
            "|": "_",
            "?": "_",
            "*": "_",
            " ": "_",
        }

        safe_name = col_name
        for char, replacement in replacements.items():
            safe_name = safe_name.replace(char, replacement)

        safe_name = cls._MULTIPLE_UNDERSCORES.sub("_", safe_name)
        safe_name = safe_name.strip("_")

        return safe_name

    def _create_column_mapping(self, columns) -> None:
        """
        Cria mapeamento bidirecional entre nomes de coluna originais e seguros.

        Cuida de nomes seguros duplicados adicionando contadores.
        """
        original_names = columns.tolist()
        safe_names = [self._sanitize_column_name(col) for col in original_names]

        seen = {}
        unique_safe_names = []

        for orig, safe in zip(original_names, safe_names):
            if safe in seen:
                counter = seen[safe]
                safe_with_counter = f"{safe}_{counter}"
                seen[safe] = counter + 1
            else:
                safe_with_counter = safe
                seen[safe] = 1
            unique_safe_names.append(safe_with_counter)

        self.original_column_names_ = original_names
        self.safe_column_names_ = unique_safe_names
        self.column_mapping_ = dict(zip(unique_safe_names, original_names))

    # -------- Data Hashing --------

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """
        Computa um hash para identificar se são os mesmos dados em que treinamos.

        Usa forma e amostra de valores de canto para comparação leve.
        """
        shape_str = f"{df.shape[0]}x{df.shape[1]}"

        sample_positions = [
            (0, 0),
            (min(10, df.shape[0] - 1), 0),
            (0, min(5, df.shape[1] - 1)),
        ]

        sample_values = []
        for i, j in sample_positions:
            if i < df.shape[0] and j < df.shape[1]:
                val = df.iloc[i, j]
                sample_values.append(str(val) if pd.notna(val) else "NaN")

        return shape_str + "_" + "_".join(sample_values)

    # -------- Memory Management --------

    def _defragment_kernel_data(self) -> None:
        """
        Desfragmenta DataFrames de kernel miceforest internos para otimizar memória.

        Cria cópias novas de DataFrames fragmentados e força coleta de lixo.
        """
        if not hasattr(self.kernel_, "imputation_values"):
            return

        if self.verbose:
            logger.debug("Defragmenting internal DataFrames...")

        for variable in self.kernel_.imputation_values.keys():
            self.kernel_.imputation_values[variable] = self.kernel_.imputation_values[
                variable
            ].copy()

        if hasattr(self.kernel_, "candidate_preds") and self.kernel_.candidate_preds:
            for variable in self.kernel_.candidate_preds.keys():
                self.kernel_.candidate_preds[variable] = self.kernel_.candidate_preds[
                    variable
                ].copy()

        gc.collect()

    @contextmanager
    def _maybe_suppress_warnings(self):
        """Gerenciador de contexto para opcionalmente suprimir avisos de desempenho do pandas"""
        if self.suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
                yield
        else:
            yield

    # -------- Fitting --------

    def _run_mice_in_steps(self, iterations_per_step: int) -> None:
        """
        Executa MICE em etapas com desfragmentação periódica.

        Parameters
        ----------
        iterations_per_step : int
            Número de iterações a executar antes de desfragmentar
        """
        total_iterations = self.max_iter
        current_iteration = 0

        while current_iteration < total_iterations:
            remaining = total_iterations - current_iteration
            step_iters = min(iterations_per_step, remaining)

            if self.verbose:
                logger.info(
                    f"Running iterations {current_iteration + 1} to {current_iteration + step_iters}"
                )

            with self._maybe_suppress_warnings():
                assert self.kernel_ is not None
                self.kernel_.mice(iterations=step_iters, verbose=self.verbose)

            current_iteration += step_iters

            if current_iteration < total_iterations and self.defrag_every > 0:
                self._defragment_kernel_data()

    def fit(self, X, y=None) -> "MiceForestImputer":
        """
        Ajusta o imputador nos dados de treino com gerenciamento otimizado de memória.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training data

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
            X_df.columns = [f"col_{i}" for i in range(X_df.shape[1])]

        self.training_data_hash_ = self._compute_data_hash(X_df)
        self._create_column_mapping(X_df.columns)

        if self.verbose:
            assert self.original_column_names_ is not None
            assert self.safe_column_names_ is not None
            logger.info(f"Original columns: {len(self.original_column_names_)}")
            problematic = [
                orig
                for orig, safe in zip(
                    self.original_column_names_, self.safe_column_names_
                )
                if orig != safe
            ]
            if problematic:
                logger.info(
                    f"Sanitized {len(problematic)} column names with special characters"
                )
                if len(problematic) <= 5:
                    for orig in problematic:
                        safe = self.safe_column_names_[
                            self.original_column_names_.index(orig)
                        ]
                        logger.info(f"  '{orig}' -> '{safe}'")

        assert self.safe_column_names_ is not None
        X_df.columns = self.safe_column_names_

        if self.verbose:
            logger.info(
                f"Initializing ImputationKernel with {self.m_imputations} datasets..."
            )
            logger.info(
                f"Running {self.max_iter} iterations with defragmentation every {self.defrag_every} iterations"
            )

        with self._maybe_suppress_warnings():
            self.kernel_ = mf.ImputationKernel(
                X_df,
                num_datasets=self.m_imputations,
                save_all_iterations_data=True,
                random_state=self.random_state,
                mean_match_candidates=self.mean_match_candidates,
            )

        if self.defrag_every > 0 and self.max_iter > self.defrag_every:
            self._run_mice_in_steps(iterations_per_step=self.defrag_every)
        else:
            with self._maybe_suppress_warnings():
                self.kernel_.mice(iterations=self.max_iter, verbose=self.verbose)

        if self.defrag_every > 0:
            self._defragment_kernel_data()

        self.is_fitted_ = True
        return self

    # -------- Transform --------

    def transform(
        self, X, return_multiple=True, is_training: Optional[bool] = None
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        Transforma dados com especificação opcional explícita de dados de treino.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Dados a imputar
        return_multiple : bool, default=True
            Se True, retorna lista de conjuntos de dados imputados; se False, retorna primeiro conjunto
        is_training : Optional[bool], padrão=None
            Especifica explicitamente se X são dados de treino. Se None, auto-detecta via hash.
            Use True para forçar uso de imputações em cache, False para usar impute_new_data().

        Returns
        -------
        Union[List[pd.DataFrame], pd.DataFrame]
            Conjunto(s) de dados imputado(s)
        """
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before transform")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.original_column_names_)

        if is_training is None:
            current_hash = self._compute_data_hash(X_df)
            is_training_data = current_hash == self.training_data_hash_
        else:
            is_training_data = is_training

        if self.verbose:
            if is_training_data:
                logger.info("Using cached imputations for training data")
            else:
                logger.info("Performing imputation for new data")

        assert self.safe_column_names_ is not None
        X_df.columns = self.safe_column_names_

        with self._maybe_suppress_warnings():
            imputed_datasets = self._transform_internal(X_df, is_training_data)

        if return_multiple:
            return imputed_datasets
        else:
            if not imputed_datasets:
                raise ValueError("No imputed datasets available. Ensure the imputer is fitted correctly.")
            return imputed_datasets[0]

    def _transform_internal(
        self, X_df: pd.DataFrame, is_training_data: bool
    ) -> List[pd.DataFrame]:
        """
        Lógica de transformação interna.

        Usa imputações em cache para dados de treino ou impute_new_data para dados de teste.
        """
        imputed_datasets = []
        assert self.kernel_ is not None

        if is_training_data:
            for imp_idx in range(self.m_imputations):
                df_imputed = self.kernel_.complete_data(dataset=imp_idx)
                df_imputed = df_imputed.rename(columns=self.column_mapping_)
                imputed_datasets.append(df_imputed)
        else:
            new_data_imputed = self.kernel_.impute_new_data(
                new_data=X_df, verbose=self.verbose
            )
            for imp_idx in range(self.m_imputations):
                df_imputed = new_data_imputed.complete_data(dataset=imp_idx)
                df_imputed = df_imputed.rename(columns=self.column_mapping_)
                imputed_datasets.append(df_imputed)

        return imputed_datasets

    # -------- Utilities --------

    def get_memory_info(self) -> Dict[str, Union[Dict, float]]:
        """
        Obtém informações sobre o uso de memória de DataFrames internos.

        Útil para monitorar fragmentação e otimizar o parâmetro defrag_every.

        Returns
        -------
        Dict[str, Union[Dict, float]]
            Dicionário com imputation_values, candidate_preds e total_mb

        Raises
        ------
        ValueError
            Se o imputador não foi ajustado
        """
        if not self.is_fitted_:
            raise ValueError("Imputer must be fitted before calling get_memory_info()")

        info: Dict[str, Union[Dict, float]] = {
            "imputation_values": {},
            "candidate_preds": {},
            "total_mb": 0.0,
        }

        assert self.kernel_ is not None
        for var, df in self.kernel_.imputation_values.items():
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            info["imputation_values"][var] = {
                "shape": df.shape,
                "memory_mb": round(memory_mb, 2),
            }
            info["total_mb"] += memory_mb

        if hasattr(self.kernel_, "candidate_preds"):
            for var, df in self.kernel_.candidate_preds.items():
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                info["candidate_preds"][var] = {
                    "shape": df.shape,
                    "memory_mb": round(memory_mb, 2),
                }
                info["total_mb"] += memory_mb

        info["total_mb"] = round(float(info["total_mb"]), 2)

        return info
