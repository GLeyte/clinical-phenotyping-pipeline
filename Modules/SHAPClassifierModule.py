"""
SHAP Explainer for Binary Classification Models

This module provides a comprehensive interface for training binary classifiers
(Logistic Regression, Random Forest, XGBoost) with hyperparameter optimisation
and SHAP-based interpretability.

Key Features:
- Flexible model selection with Optuna-based hyperparameter optimisation
- SHAP value calculation for model explainability
- Feature importance ranking and visualisation
- Confusion matrix and metrics report
"""

import logging
from typing import Dict, List, Optional, Tuple, Literal, Union
from contextlib import nullcontext
from dataclasses import dataclass
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from xgboost import XGBClassifier as XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import optuna
    from optuna_dashboard import run_server

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
VALID_MODELS = {"xgboost", "logreg", "randomForest"}
VALID_SCALERS = {"standard", "minmax", "robust", "none"}
SHAP_BACKGROUND_SAMPLES = 100
RANDOM_STATE = 42
VERBOSE_LEVELS = Literal[0, 1, 2]


# ============================================================================
# Configuration Dataclass
# ============================================================================


@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter optimisation"""

    n_trials: int = 150
    show_progress: bool = False
    use_dashboard: bool = False


# ============================================================================
# SHAP Classifier Helper
# ============================================================================


class ShapHelperClassifier:
    """
    Binary classification helper with SHAP-based explainability.

    Supports Logistic Regression, Random Forest and XGBoost models with
    integrated hyperparameter optimisation via Optuna and SHAP value analysis.

    Parameters
    ----------
    train : pd.DataFrame
        Training data (features + target column)
    test : pd.DataFrame
        Test data (features + target column)
    target : str
        Name of the target column
    scaler : str, default="standard"
        Scaling method: "standard", "minmax", "robust" or "none"

    Raises
    ------
    ValueError
        If scaler is not in VALID_SCALERS
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        scaler: str = "standard",
    ):
        if scaler not in VALID_SCALERS:
            raise ValueError(f"Invalid scaler. Choose from {VALID_SCALERS}")

        X_train = train.drop(columns=[target]).copy()
        X_test = test.drop(columns=[target]).copy()
        y_train = train[target].copy()
        y_test = test[target].copy()

        logger.info(f"Train size: {len(train)}, Test size: {len(test)}")

        self._X_train_original = X_train
        self._X_test_original = X_test
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._features = X_train.columns

        self._scaler = self._initialize_scaler(scaler)
        self._is_scaled = scaler != "none"
        self._features_to_scale: Optional[List[str]] = None

        self._categorical_features = self._detect_categorical_features(X_train)
        self._numeric_features = list(
            set(X_train.columns) - set(self._categorical_features)
        )

        self._model: Optional[Union[LogisticRegression, RandomForestClassifier]] = None
        self._model_name: Optional[str] = None
        self._best_trial_params: Optional[Dict] = None
        self._best_trial_value: Optional[float] = None

        self._conf_matrix: Optional[np.ndarray] = None
        self._accuracy: Optional[float] = None
        self._precision: Optional[float] = None
        self._recall: Optional[float] = None
        self._f1: Optional[float] = None
        self._shap_values: Optional[shap.Explanation] = None

    # -------- Initialization Helpers --------

    def _initialize_scaler(
        self, scaler: str
    ) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        """Initialises the appropriate scaler."""
        if scaler == "standard":
            return StandardScaler()
        elif scaler == "minmax":
            return MinMaxScaler()
        elif scaler == "robust":
            return RobustScaler()
        return None

    @staticmethod
    def _detect_categorical_features(X: pd.DataFrame) -> List[str]:
        """
        Detects categorical features as binary columns (0/1).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame

        Returns
        -------
        List[str]
            List of categorical feature names
        """
        return [
            col
            for col in X.columns
            if X[col].dropna().isin([0, 1]).all() and len(X[col].dropna().unique()) <= 2
        ]

    # -------- Feature Management --------

    def remove_features(self, features_to_remove: List[str]) -> None:
        """
        Removes specified features from the training and test datasets.

        Parameters
        ----------
        features_to_remove : List[str]
            List of feature names to remove
        """
        self._X_train = self._X_train_original.drop(columns=features_to_remove)
        self._X_test = self._X_test_original.drop(columns=features_to_remove)
        self._categorical_features = self._detect_categorical_features(self._X_train)
        self._numeric_features = list(
            set(self._X_train.columns) - set(self._categorical_features)
        )
        self._features = self._X_train.columns

    # -------- Data Scaling --------

    def _update_data(
        self, scale_categorical: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scales features and returns scaled train/test data.

        Parameters
        ----------
        scale_categorical : bool, default=False
            Whether to scale categorical features

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Dataframes de treino e teste dimensionados

        Raises
        ------
        ValueError
            Se o dimensionamento está ativado mas o escalador não foi inicializado
        """
        X_train, X_test = self._X_train.copy(), self._X_test.copy()

        if self._is_scaled:
            if self._scaler is None:
                raise ValueError("Scaler not initialized. Check your configuration.")

            if scale_categorical:
                features_to_scale = X_train.columns.tolist()
            else:
                features_to_scale = [
                    col
                    for col in X_train.columns
                    if col not in self._categorical_features
                ]

            self._features_to_scale = features_to_scale
            scaled_values = self._scaler.fit_transform(X_train[features_to_scale])
            X_train[features_to_scale] = scaled_values
            scaled_values_test = self._scaler.transform(X_test[features_to_scale])
            X_test[features_to_scale] = scaled_values_test

        return X_train, X_test

    # -------- Model Training --------

    def train_single_model(
        self,
        params: Dict,
        model_name: str = "logreg",
        scale_categorical: bool = False,
    ) -> None:
        """
        Treina um modelo único com parâmetros fornecidos.

        Parameters
        ----------
        params : Dict
            Model parameters
        model_name : str, padrão="logreg"
            Tipo de modelo: "logreg", "randomForest" ou "xgboost"
        scale_categorical : bool, default=False
            Whether to scale categorical features

        Raises
        ------
        ValueError
            Se model_name é inválido
        """
        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model. Choose from {VALID_MODELS}")

        if model_name == "xgboost" and not HAS_XGBOOST:
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost"
            )

        X_train, X_test = self._update_data(scale_categorical=scale_categorical)
        y_train, y_test = self._y_train.copy(), self._y_test.copy()

        if model_name == "logreg":
            model = LogisticRegression(**params)
        elif model_name == "randomForest":
            model = RandomForestClassifier(**params)
        else:
            model = XGBClassifier(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self._model = model
        self._model_name = model_name
        self._conf_matrix = confusion_matrix(y_test, y_pred)
        self._accuracy = accuracy_score(y_test, y_pred)
        self._precision = precision_score(y_test, y_pred, average="binary")
        self._recall = recall_score(y_test, y_pred, average="binary")
        self._f1 = f1_score(y_test, y_pred, average="binary")

    # -------- Hyperparameter Optimization --------

    def _get_optuna_params(self, trial: optuna.Trial, model_name: str) -> Dict:
        """
        Gera sugestões de hiperparâmetros Optuna para um modelo.

        Parameters
        ----------
        trial : optuna.Trial
            Objeto trial do Optuna
        model_name : str
            Tipo de modelo

        Returns
        -------
        Dict
            Suggested parameters
        """
        if model_name == "logreg":
            solver_penalty = trial.suggest_categorical(
                "solver_penalty",
                ["lbfgs_l2", "lbfgs_none", "liblinear_l1", "liblinear_l2"],
            )
            solver, penalty = solver_penalty.split("_")
            penalty = None if penalty == "none" else penalty

            return {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
                "solver": solver,
                "penalty": penalty,
                "max_iter": trial.suggest_int("max_iter", 50, 1000),
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
                "warm_start": trial.suggest_categorical("warm_start", [True, False]),
                "random_state": RANDOM_STATE,
            }

        elif model_name == "xgboost":
            return {
                "eval_metric": "logloss",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "gamma": trial.suggest_float("gamma", 0, 0.3),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "scale_pos_weight": trial.suggest_int(
                    "scale_pos_weight", 1, 100, log=True
                ),
                "random_state": RANDOM_STATE,
            }

        else:  # randomForest
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
                "random_state": RANDOM_STATE,
            }

    def optimize_hyperparameters(
        self,
        model_name: str = "logreg",
        config: Optional[OptunaConfig] = None,
        scale_categorical: bool = False,
    ) -> Tuple[Optional[Union[str, "optuna.storages.BaseStorage"]], Dict, float]:
        """
        Otimiza hiperparâmetros do modelo usando Optuna.

        Parameters
        ----------
        model_name : str, padrão="logreg"
            Tipo de modelo: "logreg", "randomForest" ou "xgboost"
        config : OptunaConfig, opcional
            Configuração do Optuna. Usa padrões se None.
        scale_categorical : bool, default=False
            Whether to scale categorical features

        Returns
        -------
        Tuple[Optional[object], Dict, float]
            (armazenamento_ou_None, melhores_params, melhor_f1_score)

        Raises
        ------
        ImportError
            Se Optuna não estiver instalado
        ImportError
            Se o modelo requer biblioteca ausente
        ValueError
            Se model_name é inválido
        """
        if not HAS_OPTUNA:
            raise ImportError(
                "optuna is not installed. Install with: pip install optuna"
            )

        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model. Choose from {VALID_MODELS}")

        if config is None:
            config = OptunaConfig()

        def objective(trial: optuna.Trial) -> float:
            """Função objetivo do Optuna"""
            params = self._get_optuna_params(trial, model_name)
            self.train_single_model(
                params, model_name=model_name, scale_categorical=scale_categorical
            )
            if self._f1 is None:
                raise ValueError("F1 score was not set after training.")
            return self._f1

        # Create study with optional storage
        if config.use_dashboard:
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(
                direction="maximize",
                study_name="Optuna Study",
                storage=storage,
            )
        else:
            storage = None
            study = optuna.create_study(direction="maximize")

        # Suppress warnings if not showing progress
        warning_context = (
            warnings.catch_warnings() if not config.show_progress else nullcontext()
        )

        with warning_context:
            if not config.show_progress:
                warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=config.n_trials)

        best_trial = study.best_trial
        self._best_trial_params = best_trial.params
        self._best_trial_value = best_trial.value

        logger.info(f"Best parameters: {best_trial.params}")
        logger.info(f"Best F1 score: {best_trial.value}")

        best_value = best_trial.value
        if best_value is None:
            raise RuntimeError("Best trial has no recorded value.")

        return storage, best_trial.params, best_value

    def show_optuna_dashboard(self, storage: Union[str, "optuna.storages.BaseStorage"]) -> None:
        """
        Lança dashboard Optuna para visualização de estudo.

        Parameters
        ----------
        storage : object
            Objeto de armazenamento Optuna de optimize_hyperparameters
        """
        if not HAS_OPTUNA:
            raise ImportError("optuna is not installed")
        run_server(storage)

    # -------- Results & Metrics --------

    def get_results(self) -> Dict:
        """
        Obtém resultados de treino e métricas.

        Returns
        -------
        Dict
            Dicionário com modelo, métricas e parâmetros
        """
        return {
            "model": self._model,
            "model_name": self._model_name,
            "conf_matrix": self._conf_matrix,
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "best_trial_params": self._best_trial_params,
            "best_trial_value": self._best_trial_value,
        }

    def print_metrics(self) -> None:
        """Imprime métricas do modelo em saída formatada."""
        if self._model is None:
            logger.warning("No model trained. Call train_single_model() first.")
            return

        print(f"Accuracy:  {self._accuracy:.4f}")
        print(f"Precision: {self._precision:.4f}")
        print(f"Recall:    {self._recall:.4f}")
        print(f"F1 Score:  {self._f1:.4f}")

    def show_confusion_matrix(self) -> None:
        """Plota matriz de confusão como mapa de calor."""
        if self._conf_matrix is None:
            logger.warning("No confusion matrix. Call train_single_model() first.")
            return

        plt.figure(figsize=(4, 3))

        conf_matrix_percent = self._conf_matrix / self._conf_matrix.sum() * 100
        annot_labels = [
            [
                f"{conf_matrix_percent[i, j]:.1f}%"
                for j in range(self._conf_matrix.shape[1])
            ]
            for i in range(self._conf_matrix.shape[0])
        ]

        sns.heatmap(
            conf_matrix_percent,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            xticklabels=["Predição Não", "Predição Sim"],
            yticklabels=["Real Não", "Real Sim"],
        )
        plt.xlabel("Predição")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.tight_layout()
        plt.show()

    # -------- Feature Name Formatting --------

    @staticmethod
    def _format_feature_names(
        features: List[str], verbosity: VERBOSE_LEVELS = 2
    ) -> List[str]:
        """
        Formata nomes de features para melhor visualização.

        Extrai partes significativas de padrões de nomenclatura como "WBC_Hematology_Complete_Delta".

        Parameters
        ----------
        features : List[str]
            Nomes de features a formatar
        verbosity : {0, 1, 2}, padrão=2
            0: Apenas primeira parte ("WBC")
            1: Primeira + quarta parte ("WBC (Delta)")
            2: Detalhe completo ("WBC (Hematology-Complete) Delta")

        Returns
        -------
        List[str]
            Nomes de features formatados
        """
        formatted = []
        pattern = r"([^_]+)_(Hematology|Chemistry)_([^_]+)_([^_]+)"

        for item in features:
            match = re.match(pattern, item)
            if match:
                if verbosity == 0:
                    formatted.append(match.group(1))
                elif verbosity == 1:
                    formatted.append(f"{match.group(1)} ({match.group(4)})")
                else:
                    formatted.append(
                        f"{match.group(1)} ({match.group(2)}-{match.group(3)}) {match.group(4)}"
                    )
            else:
                formatted.append(item)

        return formatted

    # -------- SHAP Configuration --------

    def compute_shap_values(self, scale_categorical: bool = False) -> None:
        """
        Computa valores SHAP para o modelo treinado.

        Seleciona explicador SHAP apropriado com base no tipo de modelo e dimensiona dados
        se necessário para interpretabilidade.

        Parameters
        ----------
        scale_categorical : bool, default=False
            Whether to scale categorical features

        Raises
        ------
        ValueError
            Se modelo não foi treinado ou problemas de escalador
        ImportError
            Se SHAP não estiver instalado
        """
        if not HAS_SHAP:
            raise ImportError("shap is not installed. Install with: pip install shap")

        if self._model is None:
            raise ValueError("Model not trained. Call train_single_model() first.")

        X_train_scaled, X_test_scaled = self._update_data(
            scale_categorical=scale_categorical
        )

        if self._model_name == "randomForest":
            background = shap.sample(X_train_scaled, SHAP_BACKGROUND_SAMPLES)
            explainer = shap.KernelExplainer(self._model.predict, background)
            shap_values = explainer.shap_values(X_test_scaled)
        else:
            explainer = shap.Explainer(self._model, X_train_scaled)
            shap_values = explainer(X_test_scaled)

        if self._is_scaled and self._features_to_scale:
            if self._scaler is None:
                raise ValueError("Scaler not available for inverse transform")

            if scale_categorical:
                unscaled_values = self._scaler.inverse_transform(shap_values.data)
                shap_values.data = unscaled_values
            else:
                shap_df = pd.DataFrame(shap_values.data, columns=self._X_train.columns)
                unscaled_cols = [
                    x for x in self._X_train.columns if x not in self._features_to_scale
                ]
                shap_df_unscaled = shap_df[unscaled_cols]
                shap_df_scaled = self._scaler.inverse_transform(
                    shap_df[self._features_to_scale]
                )
                shap_df = pd.concat(
                    [
                        pd.DataFrame(shap_df_scaled, columns=self._features_to_scale),
                        shap_df_unscaled,
                    ],
                    axis=1,
                )[self._X_train.columns]
                shap_values.data = shap_df.values

        self._shap_values = shap_values

    def get_shap_values(self) -> Optional[shap.Explanation]:
        """
        Obtém valores SHAP computados.

        Returns
        -------
        Optional[shap.Explanation]
            Objeto de explicação SHAP ou None se não foi computado
        """
        return self._shap_values

    # -------- Feature Importance --------

    def _compute_feature_importance(self) -> pd.DataFrame:
        """
        Computa valores SHAP absolutos médios por feature.

        Returns
        -------
        pd.DataFrame
            Dataframe de importância de feature com colunas Feature e Mean SHAP Value
        """
        if self._shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )

        return pd.DataFrame(
            {
                "Feature": self._X_test.columns,
                "Mean SHAP Value": np.abs(self._shap_values.values).mean(axis=0),
            }
        ).sort_values(by="Mean SHAP Value", ascending=False)

    def get_top_features(self, n: int) -> List[str]:
        """
        Obtém as n features mais importantes por valores SHAP.

        Parameters
        ----------
        n : int
            Número de features principais a retornar

        Returns
        -------
        List[str]
            Nomes de features principais em ordem de importância
        """
        importance_df = self._compute_feature_importance()
        return importance_df.head(n)["Feature"].tolist()

    # -------- Visualization --------

    def plot_shap_summary(
        self,
        max_value: float = 10.0,
        rank: Optional[int] = None,
        verbosity: VERBOSE_LEVELS = 0,
    ) -> None:
        """
        Plota visualização de resumo SHAP.

        Parameters
        ----------
        max_value : float, padrão=10.0
            Corta valores SHAP para [-max_value, max_value]
        rank : Optional[int], padrão=None
            Se especificado, plota apenas esta feature classificada (1 = mais importante)
        verbosity : {0, 1, 2}, padrão=0
            Nível de formatação do nome da feature

        Raises
        ------
        ValueError
            Se valores SHAP não foram computados
        """
        if self._shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )

        if not HAS_SHAP:
            raise ImportError("shap is not installed")

        values = self._shap_values.values.copy()
        if np.max(np.abs(values)) > max_value:
            logger.info(f"Clipping SHAP values to range [-{max_value}, {max_value}]")
            values = np.clip(values, -max_value, max_value)

        plt.figure(figsize=(8, 5))

        if rank is None:
            aux = shap.Explanation(
                values,
                self._shap_values.base_values,
                self._shap_values.data,
                self._shap_values.feature_names,
            )
            shap.summary_plot(
                aux,
                self._X_test,
                feature_names=self._format_feature_names(
                    list(self._features), verbosity=verbosity
                ),
            )
        else:
            feature_to_plot = self.get_top_features(rank)[-1]
            mask = np.zeros_like(values)
            feature_index = list(self._features).index(feature_to_plot)
            mask[:, feature_index] = values[:, feature_index]

            aux = shap.Explanation(
                mask,
                self._shap_values.base_values,
                self._shap_values.data,
                self._shap_values.feature_names,
            )
            shap.summary_plot(
                aux,
                self._X_test,
                feature_names=self._format_feature_names(
                    list(self._features), verbosity=verbosity
                ),
                max_display=1,
            )

        plt.tight_layout()
        plt.show()

    def plot_shap_feature(
        self,
        rank: int = 1,
        color: bool = False,
        verbosity: VERBOSE_LEVELS = 0,
    ) -> None:
        """
        Plota valores SHAP para uma feature específica.

        Parameters
        ----------
        rank : int, padrão=1
            Rank da feature (1 = mais importante)
        color : bool, padrão=False
            Se deve colorir pontos pelo valor da feature
        verbosity : {0, 1, 2}, padrão=0
            Nível de formatação do nome da feature

        Raises
        ------
        ValueError
            Se valores SHAP não foram computados
        """
        if self._shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )

        if not HAS_SHAP:
            raise ImportError("shap is not installed")

        importance_df = self._compute_feature_importance()
        feature = importance_df.iloc[rank - 1]["Feature"]

        plt.figure(figsize=(10, 7))

        if color:
            shap.plots.scatter(
                self._shap_values[:, feature],
                color=self._shap_values[:, feature],
            )
        else:
            shap.plots.scatter(self._shap_values[:, feature])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("SHAPClassifierModule loaded successfully")
