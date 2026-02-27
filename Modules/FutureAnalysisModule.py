"""
Future Analysis Module - Patient Cluster Evolution Analysis

This module provides tools for longitudinal analysis of clinical data,
comparing past and future hospital admissions to identify
patterns of comorbidity acquisition and readmissions.
"""

import os
import re
import logging
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import CATEGORICAL

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FutureAnalysisHelper:
    """
    Helper class for patient cluster evolution analysis.

    This class compares past and future hospital admission data,
    identifying patterns of comorbidity acquisition, readmission rates
    and mortality by cluster.

    Attributes:
        DAYS_BIN_SHORT: Threshold for short readmission category (days)
        DAYS_BIN_MEDIUM: Threshold for medium readmission category (days)
        DAYS_BIN_LABELS: Labels for time categories
        TIME_CATEGORY_ORDER: Preferred order for time categories

    Examples:
        >>> helper = FutureAnalysisHelper(
        ...     past_data=df_past,
        ...     future_data=df_future,
        ...     control_data=df_control,
        ...     control_readmission_data=df_control_readm
        ... )
        >>> delta = helper.get_delta_clusters()
        >>> helper.show_delta_heatmap()
    """

    # Class constants
    DAYS_BIN_SHORT = 30
    DAYS_BIN_MEDIUM = 180
    DAYS_BIN_LABELS = {
        "short": "< 30 days",
        "medium": "30-180 days",
        "long": "> 180 days",
    }
    TIME_CATEGORY_ORDER = ["< 30 days", "30-180 days", "> 180 days"]

    def __init__(
        self,
        past_data: pd.DataFrame,
        future_data: pd.DataFrame,
        control_data: pd.DataFrame,
        control_readmission_data: pd.DataFrame,
    ):
        """
        Initialises the future analysis helper.

        Args:
            past_data: DataFrame with past admission data (must contain 'Cluster')
            future_data: DataFrame with future admission data (must not contain 'Cluster')
            control_data: DataFrame with control group data
            control_readmission_data: DataFrame with control group readmissions

        Raises:
            ValueError: If column validations fail
            TypeError: If DataFrames are not of the correct type
        """
        # Type validation
        self._validate_dataframe_types(
            past_data, future_data, control_data, control_readmission_data
        )

        # Required column validation
        self._validate_required_columns(past_data, future_data)

        # Filter subjects that exist in both datasets
        future_data = self._filter_common_subjects(past_data, future_data)
        control_readmission_data = self._filter_common_subjects(
            control_data, control_readmission_data
        )

        # Store data
        self._past_data = past_data
        self._future_data = future_data
        self._control_data = control_data
        self._control_readmission_data = control_readmission_data

        # Calculate control group baseline
        self._baseline = self._get_control_values()
        self._delta_data = None  # Will be populated when getDeltaClusters is called

        logger.info(
            f"FutureAnalysisHelper initialized with {len(past_data)} past records "
            f"and {len(future_data)} future records"
        )

    def _validate_dataframe_types(self, *dataframes: pd.DataFrame) -> None:
        """Validates that all arguments are pandas DataFrames."""

        for i, df in enumerate(dataframes):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Argument {i+1} must be a pandas DataFrame, got {type(df)}"
                )

    def _validate_required_columns(
        self, past_data: pd.DataFrame, future_data: pd.DataFrame
    ) -> None:
        """
        Validates required columns in the DataFrames.

        Raises:
            ValueError: If any required column is missing
        """
        if "Cluster" in future_data.columns:
            raise ValueError(
                "future_data already contains a 'Cluster' column. "
                "This column should not exist in the input data."
            )

        if "Cluster" not in past_data.columns:
            raise ValueError("past_data must contain a 'Cluster' column")

        if "days_gap" not in future_data.columns:
            raise ValueError("future_data must contain a 'days_gap' column")

        for df_name, df in [("past_data", past_data), ("future_data", future_data)]:
            if "subject_id" not in df.columns:
                raise ValueError(f"{df_name} must contain a 'subject_id' column")

    def _filter_common_subjects(
        self, reference_data: pd.DataFrame, target_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filters target_data to include only subjects present in reference_data.

        Args:
            reference_data: Reference DataFrame with valid subject_ids
            target_data: DataFrame to be filtered

        Returns:
            Filtered DataFrame containing only common subjects
        """
        missing_subjects = set(target_data["subject_id"]) - set(
            reference_data["subject_id"]
        )

        if missing_subjects:
            logger.warning(
                f"{len(missing_subjects)} subject_ids in target data are missing "
                f"from reference data. These will be filtered out."
            )

        return target_data[
            target_data["subject_id"].isin(reference_data["subject_id"])
        ].copy()

    def _get_control_values(self) -> Dict[str, float]:
        """
        Calculates baseline values for the control group.

        Returns:
            Dictionary with baseline percentages for each feature

        Examples:
            {'feature1 (%)': 12.5, 'feature2 (%)': 8.3, ...}
        """
        # Create copy of control data and assign all to cluster 0
        control_past = self._control_data.copy()
        control_past["Cluster"] = 0
        control_future = self._control_readmission_data.copy()

        # Calculate delta for control group
        delta_control = self.get_delta_clusters(
            past_data=control_past,
            future_data=control_future,
            percentage=True,
            use_baseline=False,
        )

        # Remove Total column
        delta_control = delta_control.drop(columns=["Total"], errors="ignore")

        # Extract values from control cluster (cluster 0)
        control_dict = delta_control.to_dict()
        if 0 in control_dict:
            return control_dict[0]
        else:
            logger.error("Control cluster (0) not found in delta calculations")
            raise ValueError("Failed to calculate control baseline values")

    def _format_days_gap_vectorized(self, arr: np.ndarray) -> np.ndarray:
        """
        Categoriza gaps de dias em bins predefinidos de forma vetorizada.

        Args:
            arr: Array numpy com valores de days_gap

        Returns:
            Array numpy com categorias textuais
        """
        return np.where(
            arr < self.DAYS_BIN_SHORT,
            self.DAYS_BIN_LABELS["short"],
            np.where(
                arr < self.DAYS_BIN_MEDIUM,
                self.DAYS_BIN_LABELS["medium"],
                self.DAYS_BIN_LABELS["long"],
            ),
        )

    def insert_clusters_in_future_data(
        self,
        past_data: Optional[pd.DataFrame] = None,
        future_data: Optional[pd.DataFrame] = None,
        bin_days: bool = False,
        only_first_admission: bool = True,
    ) -> pd.DataFrame:
        """
        Insere rótulos de cluster nos dados futuros.

        Args:
            past_data: DataFrame com clusters (usa dados internos se None)
            future_data: DataFrame para inserir clusters (usa dados internos se None)
            bin_days: Se True, cria nome composto com bins de dias
            only_first_admission: Se True, mantém apenas primeira readmissão por sujeito

        Returns:
            DataFrame com coluna 'Cluster' adicionada

        Raises:
            ValueError: Se subject_ids não puderem ser mapeados para clusters

        Examples:
            >>> df_clustered = helper.insert_clusters_in_future_data(bin_days=True)
            >>> df_clustered['Cluster'].head()
            0    Cluster 1 - time: < 30 days
            1    Cluster 2 - time: 30-180 days
        """
        # Usa dados internos se não fornecidos
        if future_data is None:
            future_data = self._future_data.copy()
        if past_data is None:
            past_data = self._past_data.copy()

        if bin_days:
            # Formata gaps de dias em categorias
            future_data["days_gap_binned"] = self._format_days_gap_vectorized(
                future_data["days_gap"].values
            )

            # Merge com informação de cluster
            cluster_info = past_data[["subject_id", "Cluster"]].copy()
            future_data = future_data.merge(cluster_info, on="subject_id", how="left")

            # Cria nomes compostos de cluster
            future_data["Cluster"] = (
                "Cluster "
                + future_data["Cluster"].astype(str)
                + " - time: "
                + future_data["days_gap_binned"].astype(str)
            )

            # Remove coluna temporária
            future_data.drop(columns=["days_gap_binned"], inplace=True)
        else:
            # Mapeia subject_id para Cluster
            cluster_mapping = past_data.set_index("subject_id")["Cluster"].to_dict()
            future_data["Cluster"] = future_data["subject_id"].map(cluster_mapping)

            # Verifica mapeamentos falhados
            if future_data["Cluster"].isna().any():
                n_failed = future_data["Cluster"].isna().sum()
                raise ValueError(
                    f"{n_failed} subject_ids could not be mapped to clusters. "
                    "Ensure all subject_ids in future_data exist in past_data."
                )

        # Mantém apenas primeira admissão se solicitado
        if only_first_admission:
            future_data = future_data.sort_values(
                by=["subject_id", "age"]
            ).drop_duplicates(subset=["subject_id"], keep="first")

        logger.info(f"Clusters inserted for {len(future_data)} records")
        return future_data

    def _calculate_percentage_change(
        self,
        delta_value: float,
        total_value: float,
        cluster: Any,
        feature: str,
        baseline: Dict[str, float],
        relative_total: bool,
    ) -> float:
        """
        Calcula mudança percentual tratando casos extremos.

        Args:
            delta_value: Valor delta absoluto
            total_value: Valor total para normalização
            cluster: Identificador do cluster
            feature: Nome da feature
            baseline: Valores baseline do controle (pode ser vazio)
            relative_total: Se True, calcula relativo ao baseline

        Returns:
            Mudança percentual calculada ou NaN se inválido
        """
        try:
            if total_value > 0:
                percentage_value = (delta_value / total_value) * 100

                if relative_total and baseline:
                    baseline_key = f"{feature} (%)"
                    if baseline_key in baseline:
                        baseline_value = baseline[baseline_key]
                        if baseline_value > 0:
                            return percentage_value / baseline_value
                        else:
                            logger.warning(
                                f"Baseline value is zero for {feature}. "
                                "Cannot calculate relative change."
                            )
                            return np.nan
                    else:
                        logger.warning(f"Feature {feature} not found in baseline")
                        return np.nan
                elif relative_total and not baseline:
                    logger.warning(
                        "relative_total=True but baseline is empty. "
                        "Returning absolute percentage."
                    )
                    return round(percentage_value, 2)
                else:
                    return round(percentage_value, 2)
            else:
                logger.debug(
                    f"Total value is zero for cluster {cluster}, feature {feature}"
                )
                return np.nan

        except ZeroDivisionError:
            logger.error(f"Division by zero for cluster {cluster}, feature {feature}")
            return np.nan

    def get_delta_clusters(
        self,
        past_data: Optional[pd.DataFrame] = None,
        future_data: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        percentage: bool = True,
        bin_days: bool = False,
        only_first_admission: bool = True,
        relative_total: bool = False,
        use_baseline: bool = True,
    ) -> pd.DataFrame:
        """
        Compara distribuições de clusters entre datasets passado e futuro.

        Este método identifica casos de "onset" (aquisição) de comorbidades,
        definidos como features que eram 0 no passado e se tornaram 1 no futuro.

        Args:
            past_data: Dataset passado com coluna 'Cluster'
            future_data: Dataset futuro para comparação
            features: Lista de features a analisar (auto-detecta se None)
            percentage: Se True, converte contagens em percentuais
            bin_days: Se True, agrupa por bins de dias
            only_first_admission: Se True, considera apenas primeira readmissão
            relative_total: Se True, calcula mudanças relativas ao grupo controle
            use_baseline: Se True, usa baseline do controle (evita recursão quando False)

        Returns:
            DataFrame transposto com mudanças por cluster e feature

        Raises:
            ValueError: Se validações falharem

        Examples:
            >>> delta = helper.get_delta_clusters(percentage=True)
            >>> delta.loc['diabetes (%)']
            Cluster 0    12.5
            Cluster 1    18.3
        """
        # Usa dados internos se não fornecidos
        if future_data is None:
            future_data = self._future_data.copy()
        if past_data is None:
            past_data = self._past_data.copy()

        # Obtém baseline apenas se solicitado e se disponível
        # Isso evita recursão infinita durante cálculo inicial do baseline
        baseline = {}
        if use_baseline and hasattr(self, "_baseline"):
            baseline = self._baseline
        elif use_baseline and not hasattr(self, "_baseline"):
            logger.warning(
                "Baseline not yet calculated. Proceeding without baseline comparison."
            )

        # Insere clusters nos dados futuros
        future_data_clustered = self.insert_clusters_in_future_data(
            past_data=past_data,
            future_data=future_data,
            bin_days=bin_days,
            only_first_admission=only_first_admission,
        )

        # Valida presença de coluna Cluster
        if (
            "Cluster" not in past_data.columns
            or "Cluster" not in future_data_clustered.columns
        ):
            raise ValueError("Both DataFrames must contain a 'Cluster' column")

        # Auto-detecta features se não fornecidas
        if features is None:
            features = [
                col
                for col in past_data.columns
                if col not in ["Cluster", "gender_M"] and past_data[col].nunique() == 2
            ]
            logger.info(f"Auto-detected {len(features)} binary features for analysis")

        # Prepara datasets
        df_past = past_data[["subject_id", "Cluster"] + features].copy()
        df_future = future_data_clustered[
            ["subject_id", "Cluster", "days_gap"] + features
        ].copy()

        days_gap_mean = df_future["days_gap"].mean()

        # Merge past e future por subject_id
        df_merged = df_future.merge(
            df_past, on="subject_id", suffixes=("_future", "_past")
        )

        # Identifica casos de onset: past=0 AND future=1
        final_features = ["subject_id", "Cluster_future", "days_gap"] + features
        for feature in features:
            past_col = f"{feature}_past"
            future_col = f"{feature}_future"
            df_merged[feature] = (
                (df_merged[past_col] == 0) & (df_merged[future_col] == 1)
            ).astype(int)

        # Prepara DataFrame delta
        df_delta = df_merged[final_features].rename(
            columns={"Cluster_future": "Cluster"}
        )

        # Agrupa por cluster
        df_delta = df_delta.groupby("Cluster").agg(
            {**{feature: "sum" for feature in features}, "days_gap": "mean"}
        )

        # Adiciona linha Total
        df_delta.loc["Total"] = [
            (df_delta[feature].sum() if feature != "days_gap" else days_gap_mean)
            for feature in features + ["days_gap"]
        ]

        # Converte para percentagens se solicitado
        rename_dict = {}
        if percentage:
            for cluster in df_delta.index:
                for feature in features:
                    if cluster != "Total":
                        delta_value = df_delta.loc[cluster, feature]

                        # Calcula total excluindo mortes
                        total_value = len(
                            df_past[df_past["Cluster"] == cluster]
                        ) - self.get_values_by(
                            df_past, cluster, feature, "died_in_stay"
                        )

                        # Calcula percentual
                        percentage_value = self._calculate_percentage_change(
                            delta_value=delta_value,
                            total_value=total_value,
                            cluster=cluster,
                            feature=feature,
                            baseline=baseline,
                            relative_total=relative_total,
                        )

                        df_delta.loc[cluster, feature] = percentage_value

                        # Prepara renomeação de colunas
                        if not relative_total and feature not in rename_dict:
                            rename_dict[feature] = f"{feature} (%)"

        # Ordena colunas
        desired_order = ["died_in_stay", "days_gap"]
        remaining_features = sorted(
            [
                col
                for col in features
                if col not in desired_order + ["died", "died_after"]
            ]
        )
        sorted_columns = remaining_features + desired_order
        df_delta = df_delta[sorted_columns]

        # Renomeia colunas
        df_delta.rename(columns=rename_dict, inplace=True)

        # Arredonda days_gap
        df_delta["days_gap"] = df_delta["days_gap"].round(1)

        # Armazena internamente
        self._delta_data = df_delta

        logger.info(f"Delta clusters calculated for {len(df_delta)-1} clusters")
        return df_delta.T

    def get_values_by(
        self,
        data: pd.DataFrame,
        cluster: int,
        feature: str,
        condition: Optional[str] = None,
    ) -> int:
        """
        Obtém valores agregados para um cluster e feature específicos.

        Args:
            data: DataFrame contendo dados de clusters
            cluster: Número do cluster a consultar
            feature: Nome da feature a agregar
            condition: Feature condicional opcional para filtro adicional (OR lógico)

        Returns:
            Soma dos valores da feature para o cluster especificado

        Raises:
            KeyError: Se cluster ou feature não existirem no DataFrame

        Examples:
            >>> helper.get_values_by(df, cluster=1, feature='died_in_stay')
            42
            >>> helper.get_values_by(df, 1, 'diabetes', condition='hypertension')
            58
        """
        try:
            if condition:
                # Filtro com condição OR
                mask = (data["Cluster"] == cluster) & (
                    (data[condition] == 1) | (data[feature] == 1)
                )
                return len(data[mask])
            else:
                # Apenas soma do feature
                return data[data["Cluster"] == cluster][feature].sum()

        except KeyError as e:
            logger.error(f"Column not found: {e}")
            raise

    def restructure_dynamic_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reestrutura DataFrame com N clusters para colunas hierárquicas.

        Organiza colunas do tipo "Cluster X - time: Y dias" em estrutura
        multi-index com clusters como header principal e categorias de tempo
        como sub-headers.

        Args:
            df: DataFrame com colunas no formato "Cluster N - time: categoria"

        Returns:
            DataFrame com MultiIndex columns organizados hierarquicamente

        Examples:
            Colunas de entrada: ["Cluster 0 - time: < 30 days", "Cluster 1 - time: < 30 days"]
            Saída: MultiIndex com ("Cluster 0", "< 30 days"), ("Cluster 1", "< 30 days")
        """
        # Define ordem preferencial de tempo
        preferred_time_order = self.TIME_CATEGORY_ORDER

        # Pattern para extrair informação de cluster
        cluster_pattern = r"Cluster\s*(\d+)\s*-\s*time:\s*(.+)"

        cluster_cols = defaultdict(list)
        other_cols = []
        time_categories = set()

        # Analisa colunas existentes
        for col in df.columns:
            match = re.match(cluster_pattern, col)
            if match:
                cluster_num = int(match.group(1))
                time_category = match.group(2).strip()
                cluster_cols[cluster_num].append((col, time_category))
                time_categories.add(time_category)
            else:
                other_cols.append(col)

        # Ordena categorias de tempo
        ordered_time_categories = [
            time_cat for time_cat in preferred_time_order if time_cat in time_categories
        ]

        # Adiciona categorias não previstas (ordenadas)
        ordered_time_categories.extend(
            sorted(time_categories - set(ordered_time_categories))
        )

        # Cria nova estrutura de colunas
        new_columns = []

        # Adiciona colunas de clusters em ordem numérica
        for cluster_num in sorted(cluster_cols.keys()):
            for time_cat in ordered_time_categories:
                new_columns.append((f"Cluster {cluster_num}", time_cat))

        # Adiciona outras colunas ao Summary
        for col in other_cols:
            new_columns.append(("Summary", col))

        # Cria novo DataFrame
        df_new = pd.DataFrame(index=df.index)

        # Preenche dados
        for main_header, sub_header in new_columns:
            if main_header.startswith("Cluster"):
                # Encontra coluna original
                cluster_num = int(main_header.split()[1])
                original_col = None

                for orig_col, time_cat in cluster_cols[cluster_num]:
                    if time_cat == sub_header:
                        original_col = orig_col
                        break

                if original_col and original_col in df.columns:
                    df_new[(main_header, sub_header)] = df[original_col]
                else:
                    df_new[(main_header, sub_header)] = None
            else:
                # Colunas Summary
                if sub_header in df.columns:
                    df_new[(main_header, sub_header)] = df[sub_header]

        # Converte para MultiIndex
        df_new.columns = pd.MultiIndex.from_tuples(df_new.columns)

        logger.info(f"Restructured DataFrame with {len(cluster_cols)} clusters")
        return df_new

    def show_delta_heatmap(
        self,
        color_limit: int = 100,
        figsize: Tuple[int, int] = (12, 8),
        selected_clusters: Optional[List[str]] = None,
        savepath: Optional[str] = None,
        relative_total: bool = False,
        title: Optional[str] = None,
    ) -> None:
        """
        Exibe mapa de calor dos dados delta.

        Args:
            color_limit: Limite superior para escala de cores (valores acima são cortados)
            figsize: Tupla (largura, altura) para tamanho da figura
            selected_clusters: Lista de clusters específicos a plotar
            savepath: Caminho para salvar figura (None = não salva)
            relative_total: Se True, usa título indicando comparação com controle
            title: Título customizado (sobrescreve automático)

        Raises:
            ValueError: Se delta data não foi computado

        Examples:
            >>> helper.show_delta_heatmap(color_limit=150, savepath='heatmap.png')
        """
        if not hasattr(self, "_delta_data") or self._delta_data is None:
            raise ValueError(
                "Delta data not computed. Please run get_delta_clusters() first."
            )

        # Prepara dados
        data = self._delta_data.copy()
        data = data.drop(columns=["days_gap"], errors="ignore")
        data = data.drop(index=["Total"], errors="ignore")

        # Converte para numérico
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.fillna(0).T

        # Filter clusters if specified
        if selected_clusters is not None:
            data = data[selected_clusters]

        # Renomear índices para melhor legibilidade
        data.index = [
            CATEGORICAL[idx] if idx in CATEGORICAL else idx for idx in data.index
        ]

        # Cria versão clipped para cores
        color_data = data.clip(0, color_limit)

        # Cria uma matriz de strings formatadas (ex: "2.51 x") para a anotação
        annot_labels = data.apply(lambda col: col.map("{:.2f} x".format))

        # Cria figura
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            color_data,
            annot=annot_labels,
            fmt="",
            cmap="YlOrRd",
            cbar_kws={"label": "Razão de Incidência"},
        )

        # Define título
        if title is None:
            if relative_total:
                title = (
                    "Comparação da Incidência de Comorbidades (Razão COVID/Controle)"
                )
            else:
                title = "Incidência de aquisição de comorbidades dos clusters"
        ax.set_title(title)

        plt.xlabel("Clusters")
        plt.ylabel("Features")
        plt.tight_layout()

        # Salva figura se solicitado
        if savepath is not None:
            abs_path = os.path.abspath(savepath)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            try:
                plt.savefig(abs_path, dpi=300, bbox_inches="tight")
                logger.info(f"Figure saved to {abs_path}")
            except Exception as e:
                logger.error(f"Failed to save figure to {abs_path}: {e}")
                raise

        plt.show()

    def get_mean_readmission(self) -> Dict[str, float]:
        """
        Calcula taxa média de readmissão para cada cluster.

        Returns:
            Dicionário com médias de readmissão por cluster e geral

        Examples:
            >>> readmissions = helper.get_mean_readmission()
            >>> readmissions['Mean readmission Cluster 1']
            1.42
            >>> readmissions['Overall Mean readmission']
            1.28
        """
        # Obtém dados com todas as admissões
        data_full = self.insert_clusters_in_future_data(only_first_admission=False)

        # Aggregate by cluster
        data = (
            data_full[["subject_id", "hadm_id", "Cluster"]]
            .groupby("Cluster")
            .agg({"subject_id": "nunique", "hadm_id": "nunique"})
        )
        data.reset_index(inplace=True)

        # Calcula média
        data["mean_readmission"] = data["hadm_id"] / data["subject_id"]

        # Cria dicionário de resultados
        readmissions_mean = {}
        for _, row in data.iterrows():
            cluster = int(row["Cluster"])
            mean_readm = round(row["mean_readmission"], 2)
            readmissions_mean[f"Mean readmission Cluster {cluster}"] = mean_readm

        # Adiciona média geral
        overall_mean = round(
            data_full["hadm_id"].nunique() / data_full["subject_id"].nunique(), 2
        )
        readmissions_mean["Overall Mean readmission"] = overall_mean

        logger.info(f"Calculated mean readmission for {len(data)} clusters")
        return readmissions_mean

    def get_mean_days_gap(self) -> Dict[str, float]:
        """
        Calcula gap médio de dias entre admissões para cada cluster.

        Returns:
            Dicionário com gaps médios por cluster e geral

        Examples:
            >>> gaps = helper.get_mean_days_gap()
            >>> gaps['Mean days gap Cluster 1']
            45.2
            >>> gaps['Overall Mean days gap']
            62.8
        """
        # Obtém dados (primeira admissão apenas)
        data_full = self.insert_clusters_in_future_data(only_first_admission=True)

        # Aggregate by cluster
        data = (
            data_full[["Cluster", "days_gap"]]
            .groupby("Cluster")
            .agg({"days_gap": "mean"})
        )
        data.reset_index(inplace=True)

        # Cria dicionário de resultados
        days_gap_mean = {}
        for _, row in data.iterrows():
            cluster = int(row["Cluster"])
            mean_gap = round(row["days_gap"], 2)
            days_gap_mean[f"Mean days gap Cluster {cluster}"] = mean_gap

        # Adiciona média geral
        overall_mean = float(round(data_full["days_gap"].mean(), 2))
        days_gap_mean["Overall Mean days gap"] = overall_mean

        logger.info(f"Calculated mean days gap for {len(data)} clusters")
        return days_gap_mean

    def get_mortality_rates(
        self, only_first_admission: bool = False
    ) -> Dict[str, float]:
        """
        Calcula taxas de mortalidade para cada cluster.

        A taxa é calculada como: (óbitos em readmissões futuras) / (óbitos em admissões passadas)

        Args:
            only_first_admission: Se True, considera apenas primeira readmissão

        Returns:
            Dicionário com taxas de mortalidade por cluster e geral

        Examples:
            >>> mortality = helper.get_mortality_rates()
            >>> mortality['Mortality rate Cluster 1']
            0.85
            >>> mortality['Overall Mortality rate']
            0.92
        """
        # Obtém dados futuros com clusters
        data = self.insert_clusters_in_future_data(
            only_first_admission=only_first_admission
        )

        # Aggregate deaths by cluster
        cluster_deaths = (
            data[["Cluster", "died_in_stay"]]
            .groupby("Cluster")
            .agg({"died_in_stay": "sum"})
        )
        cluster_deaths.reset_index(inplace=True)

        # Calcula taxas
        mortality_rates = {}
        for _, row in cluster_deaths.iterrows():
            cluster = int(row["Cluster"])
            deaths_future = row["died_in_stay"]

            # Obtém mortes no passado para normalização
            deaths_past = self.get_values_by(
                self._past_data, cluster, "died_in_stay", None
            )

            if deaths_past > 0:
                rate = float(round(deaths_future / deaths_past, 2))
            else:
                logger.warning(
                    f"No past deaths for cluster {cluster}. "
                    "Setting mortality rate to 0.0"
                )
                rate = 0.0

            mortality_rates[f"Mortality rate Cluster {cluster}"] = rate

        # Calcula taxa geral
        total_deaths_future = cluster_deaths["died_in_stay"].sum()
        total_deaths_past = self._past_data["died_in_stay"].sum()

        if total_deaths_past > 0:
            overall_rate = float(round(total_deaths_future / total_deaths_past, 2))
        else:
            logger.warning(
                "No past deaths found. Setting overall mortality rate to 0.0"
            )
            overall_rate = 0.0

        mortality_rates["Overall Mortality rate"] = overall_rate

        logger.info(f"Calculated mortality rates for {len(cluster_deaths)} clusters")
        return mortality_rates


# Exemplo de uso
if __name__ == "__main__":
    # Este bloco só executa se o arquivo for executado diretamente
    logger.info("FutureAnalysisModule loaded successfully")

    # Exemplo de inicialização (com dados fictícios)
    # helper = FutureAnalysisHelper(
    #     past_data=pd.DataFrame(...),
    #     future_data=pd.DataFrame(...),
    #     control_data=pd.DataFrame(...),
    #     control_readmission_data=pd.DataFrame(...)
    # )
