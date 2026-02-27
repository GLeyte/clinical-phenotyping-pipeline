import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Optional, Union


# ====================  HELPERS  ====================

def _as_display_int(value: Union[int, float]) -> Union[int, float]:
    """Returns value as integer when it is a whole-number float, otherwise unchanged."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# ====================  ANALYSIS CLASS  ====================

class Analysis:
    """
    Comprehensive missing data analysis for a DataFrame.

    Usage
    ---
    analyst = Analysis(df)
    analyst.analyze_missing_data(threshold=0.0, show_info=True)
    analyst.plot_missing_data("Missing Data by Columns", top_missing=15)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Populated by analyze_missing_data(); guards plot_missing_data() calls.
        self.figures_info: Optional[Dict] = None
        self.analysis_results: Optional[Dict] = None

    # ==================== FORMAT HELPER ====================

    def format_name(self, name: str, verbose: int) -> str:
        """
        Reformats a raw column name based on verbosity level.

        Parameters
        ----------
        name : str
            Raw column name, expected pattern:
            ``<Marcador>_(Hematology|Chemistry)_<Painel>_<Unidade>``
        verbose : int
            0 — marker name only (``Hemoglobin``).
            1 — marker + unit in parentheses (``Hemoglobin (g/dL)``).
            2 — marker + panel + unit (``Hemoglobin (Hematology-CBCDIFF) g/dL``).
            Any other value — original name unchanged.

        Returns
        -------
        str
            Reformatted display name.
        """
        if verbose == 0:
            match = re.match(r"([^_]+)_(Hematology|Chemistry).*", name)
            if match:
                return match.group(1)
        elif verbose == 1:
            match = re.match(r"([^_]+)_(Hematology|Chemistry)_([^_]+)_([^_]+)", name)
            if match:
                return f"{match.group(1)} ({match.group(4)})"
        elif verbose == 2:
            match = re.match(r"([^_]+)_(Hematology|Chemistry)_([^_]+)_([^_]+)", name)
            if match:
                return f"{match.group(1)} ({match.group(2)}-{match.group(3)}) {match.group(4)}"
        return name

    # ==================== ANALYSIS ====================

    def analyze_missing_data(
        self,
        threshold: float = 0.0,
        good_threshold: float = 5.0,
        fair_threshold: float = 20.0,
        return_figures: bool = True,
        verbose: int = 0,
        show_info: bool = False,
    ) -> Dict:
        """
        Performs a comprehensive missing data analysis.

        Parameters
        ----------
        threshold : float, default 0.0
            Columns with a missing percentage <= this value are classified as
            "Excellent". Also acts as the minimum threshold for showing a
            column in the "significant missing" filter.
        good_threshold : float, default 5.0
            Upper limit (%) for the "Good" quality category.
        fair_threshold : float, default 20.0
            Upper limit (%) for the "Fair" quality category.
            Columns above this value are classified as "Poor".
        return_figures : bool, default True
            Whether to store figure data in the instance for use with
            ``plot_missing_data()``.
        verbose : int, default 0
            Controls column name formatting (see ``format_name``).
        show_info : bool, default False
            Prints a text summary to stdout.

        Returns
        -------
        dict
            Dictionary with keys: ``overview``, ``column_summary``,
            ``missing_patterns``, ``missing_correlation``,
            ``quality_categories``, ``quality_thresholds``,
            and optionally ``figures``.
        """
        # Work on a local copy so that column renaming never mutates self.df.
        df = self.df.copy()

        excellent_threshold = threshold

        df.columns = df.columns.map(lambda x: self.format_name(x, verbose=verbose))

        # ---- basic statistics ----
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        missing_percentage = (total_missing / total_cells) * 100

        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        significant_missing = missing_percentages[missing_percentages > threshold]

        rows_with_missing = df.isnull().any(axis=1).sum()
        rows_missing_percentage = (rows_with_missing / len(df)) * 100

        missing_patterns = df.isnull().value_counts()

        # ---- summary table ----
        # Index on "Column" so that quality-category slices yield column names
        # when callers call .index.tolist().
        summary_df = (
            pd.DataFrame(
                {
                    "Column": df.columns,
                    "Missing_Count": missing_counts.values,
                    "Missing_Percentage": missing_percentages.values,
                    "Data_Type": df.dtypes.values,
                    "Non_Missing_Count": (len(df) - missing_counts).values,
                }
            )
            .set_index("Column")
            .sort_values("Missing_Percentage", ascending=False)
        )

        missing_corr = df.isnull().corr()

        # ---- quality categories ----
        excellent = summary_df[summary_df["Missing_Percentage"] <= excellent_threshold]
        good = summary_df[
            (summary_df["Missing_Percentage"] > excellent_threshold)
            & (summary_df["Missing_Percentage"] <= good_threshold)
        ]
        fair = summary_df[
            (summary_df["Missing_Percentage"] > good_threshold)
            & (summary_df["Missing_Percentage"] <= fair_threshold)
        ]
        poor = summary_df[summary_df["Missing_Percentage"] > fair_threshold]

        # Normalize display versions of thresholds (drop trailing .0 for readability).
        exc_disp = _as_display_int(excellent_threshold)
        good_disp = _as_display_int(good_threshold)
        fair_disp = _as_display_int(fair_threshold)

        # ---- optional stdout report ----
        if show_info:
            print("=" * 60)
            print("MISSING DATA ANALYSIS REPORT")
            print("=" * 60)

            print(f"\nOVERVIEW:")
            print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
            print(f"Total Cells: {total_cells:,}")
            print(f"Total Missing Values: {total_missing:,}")
            print(f"Overall Missing Percentage: {missing_percentage:.2f}%")
            print(
                f"Rows with Missing Data: {rows_with_missing:,} ({rows_missing_percentage:.2f}%)"
            )
            print(
                f"Complete Rows: {len(df) - rows_with_missing:,} ({100 - rows_missing_percentage:.2f}%)"
            )

            print(f"\nCOLUMN-WISE SUMMARY:")
            print(
                f"Columns with Missing Data: {len(significant_missing)}/{len(df.columns)}"
            )

            if len(significant_missing) > 0:
                print(f"\nTop 10 Columns with Most Missing Data:")
                print("-" * 50)
                for col, row in summary_df.head(10).iterrows():
                    if row["Missing_Percentage"] > 0:
                        print(
                            f"{col:<30} {row['Missing_Count']:>8,} ({row['Missing_Percentage']:>6.2f}%)"
                        )

            print(f"\nDATA QUALITY CATEGORIES:")
            print("-" * 40)
            print(f"Excellent (<{exc_disp}% missing): {len(excellent)} columns")
            print(
                f"Good ({exc_disp}%-{good_disp}% missing): {len(good)} columns"
            )
            print(
                f"Fair ({good_disp}%-{fair_disp}% missing): {len(fair)} columns"
            )
            print(f"Poor (>{fair_disp}% missing): {len(poor)} columns")

            if len(poor) > 0:
                print(f"\nPOOR QUALITY COLUMNS (>{fair_disp}% missing):")
                for col, row in poor.iterrows():
                    print(f"  - {col}: {row['Missing_Percentage']:.1f}% missing")

            print(f"\nMISSING PATTERNS:")
            print("-" * 30)
            if len(missing_patterns) > 1:
                print(f"Unique missing patterns found: {len(missing_patterns)}")
                print(
                    f"Most common pattern frequency: {missing_patterns.iloc[0]:,} rows"
                )
                print(f"\nTop 5 Missing Patterns:")
                for i, (pattern, count) in enumerate(missing_patterns.head().items()):
                    pattern_desc = (
                        "Complete row"
                        if not any(pattern)
                        else f"{sum(pattern)} missing values"
                    )
                    print(
                        f"  {i+1}. {pattern_desc}: {count:,} rows ({count/len(df)*100:.1f}%)"
                    )

        # ---- figure data cache ----
        figures_dict = {
            "Missing Data by Columns": (
                significant_missing,
                fair_threshold,
                good_threshold,
            ),
            "Missing Data Heatmap": (df, significant_missing),
            "Completeness Distribution": missing_percentages,
            "Missing Data Correlation": (significant_missing, missing_corr),
            "Missing Values by Row": df,
            "Feature Data Quality Distribution": (
                excellent,
                good,
                fair,
                poor,
                excellent_threshold,
                good_threshold,
                fair_threshold,
            ),
        }

        # ---- result dictionary ----
        analysis_results = {
            "overview": {
                "total_rows": df.shape[0],
                "total_columns": df.shape[1],
                "total_cells": total_cells,
                "total_missing": total_missing,
                "missing_percentage": missing_percentage,
                "rows_with_missing": rows_with_missing,
                "complete_rows": len(df) - rows_with_missing,
            },
            "column_summary": summary_df,
            "missing_patterns": missing_patterns,
            "missing_correlation": missing_corr,
            "quality_categories": {
                "excellent": excellent.index.tolist(),
                "good": good.index.tolist(),
                "fair": fair.index.tolist(),
                "poor": poor.index.tolist(),
            },
            "quality_thresholds": {
                "excellent_threshold": excellent_threshold,
                "good_threshold": good_threshold,
                "fair_threshold": fair_threshold,
            },
        }

        if return_figures:
            analysis_results["figures"] = figures_dict

        # Always cache figures so plot_missing_data() works regardless of
        # what the caller does with the return value.
        self.figures_info = figures_dict
        self.analysis_results = analysis_results

        return analysis_results

    # ==================== PLOT HELPERS ====================

    def get_plot_types(self) -> List[str]:
        """Returns the list of valid plot_type values for plot_missing_data()."""
        return [
            "Missing Data by Columns",
            "Missing Data Heatmap",
            "Completeness Distribution",
            "Missing Data Correlation",
            "Missing Values by Row",
            "Feature Data Quality Distribution",
        ]

    def plot_missing_data(
        self,
        plot_type: str,
        figsize: tuple = (10, 6),
        top_missing: int = 10,
        sort: bool = True,
        show_heatmap_numbers: bool = False,
        add_title: Optional[str] = None,
    ) -> None:
        """
        Renderiza uma visualização única de dados faltantes.

        Deve ser chamado após ``analyze_missing_data()``.

        Parameters
        ----------
        plot_type : str
            Uma das strings retornadas por ``get_plot_types()``.
        figsize : tuple, padrão (10, 6)
            Tamanho da figura Matplotlib.
        top_missing : int, padrão 10
            Número máximo de colunas a exibir (onde aplicável).
            Passe -1 para incluir todas as colunas.
        sort : bool, padrão True
            Classifica as colunas por percentual de falta (descendente) antes de fatiar
            para ``top_missing``.
        show_heatmap_numbers : bool, padrão False
            Anota o mapa de calor "Correlação de Dados Faltantes" com valores numéricos.
        add_title : str, opcional
            Texto extra anexado ao título do subplot.
        """
        if self.figures_info is None:
            raise RuntimeError(
                "No figure data found. Call analyze_missing_data() first."
            )

        valid_plot_types = self.get_plot_types()
        if plot_type not in valid_plot_types:
            raise ValueError(
                f"Invalid plot_type. Choose from: {', '.join(valid_plot_types)}"
            )

        info = self.figures_info[plot_type]
        title_suffix = f" {add_title}" if add_title else ""

        # ---- helper: apply sort + top_missing slice ----
        def _slice(series: pd.Series) -> pd.Series:
            s = series.sort_values(ascending=False) if sort else series
            return s if top_missing == -1 else s.head(top_missing)

        if plot_type == "Missing Data by Columns":
            significant_missing, fair_threshold, good_threshold = info
            top_cols = _slice(significant_missing)

            fig, ax = plt.subplots(figsize=figsize)
            if len(top_cols) > 0:
                bars = ax.barh(range(len(top_cols)), top_cols.values)
                ax.set_yticks(range(len(top_cols)))
                ax.set_yticklabels(top_cols.index)
                ax.set_xlabel("Porcentagem de Dados Faltantes (%)")
                ax.set_title(f"Dados faltantes por coluna{title_suffix}")
                ax.invert_yaxis()
                ax.grid(axis="x", linestyle="--", alpha=0.5)

                for i, bar in enumerate(bars):
                    if top_cols.iloc[i] > fair_threshold:
                        bar.set_color("red")
                    elif top_cols.iloc[i] > good_threshold:
                        bar.set_color("orange")
                    else:
                        bar.set_color("green")

            plt.show()

        elif plot_type == "Missing Data Heatmap":
            df, significant_missing = info
            top_cols = _slice(significant_missing)

            fig, ax = plt.subplots(figsize=figsize)
            if len(top_cols) > 0:
                sns.heatmap(
                    df[top_cols.index].isnull().T, cmap="viridis", cbar=True, ax=ax
                )
                ax.set_title(f"Mapa de Calor dos Dados Faltantes{title_suffix}")
                plt.show()

        elif plot_type == "Completeness Distribution":
            missing_percentages = info
            completeness = 100 - missing_percentages

            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(completeness, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
            ax.set_xlabel("Porcentagem de Completude (%)")
            ax.set_ylabel("Number of Features")
            ax.set_title(f"Distribuicao da Completude dos Dados{title_suffix}")
            ax.axvline(
                x=completeness.mean(),
                color="red",
                linestyle="--",
                label=f"Media: {completeness.mean():.1f}%",
            )
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif plot_type == "Missing Data Correlation":
            significant_missing, missing_corr = info

            fig, ax = plt.subplots(figsize=figsize)
            if len(significant_missing) > 1:
                top_cols = _slice(significant_missing)
                corr_data = missing_corr.loc[top_cols.index, top_cols.index]
                sns.heatmap(
                    corr_data,
                    annot=show_heatmap_numbers,
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    fmt=".2f",
                    ax=ax,
                )
                ax.set_title(f"Correlacao dos Dados Faltantes{title_suffix}")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient missing data\nfor correlation analysis",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Correlacao dos Dados Faltantes{title_suffix}")
            plt.tight_layout()
            plt.show()

        elif plot_type == "Missing Values by Row":
            df = info

            fig, ax = plt.subplots(figsize=figsize)
            missing_per_row = df.isnull().sum(axis=1)
            ax.hist(
                missing_per_row,
                bins=min(30, df.shape[1]),
                alpha=0.7,
                color="coral",
                edgecolor="black",
            )
            ax.set_xlabel("Numero de Valores Faltantes por Registro")
            ax.set_ylabel("Numero de Registros")
            ax.set_title(
                f"Distribuicao dos Valores Faltantes por Registro{title_suffix}"
            )
            plt.tight_layout()
            plt.show()

        elif plot_type == "Feature Data Quality Distribution":
            excellent, good, fair, poor = info[0], info[1], info[2], info[3]
            exc_t = _as_display_int(info[4])
            good_t = _as_display_int(info[5])
            fair_t = _as_display_int(info[6])

            quality_counts = [len(excellent), len(good), len(fair), len(poor)]
            quality_labels = [
                f"Excelente\n(<{exc_t}%)",
                f"Bom\n({exc_t}%-{good_t}%)",
                f"Regular\n({good_t}%-{fair_t}%)",
                f"Ruim\n(>{fair_t}%)",
            ]
            colors = ["green", "lightgreen", "orange", "red"]

            non_zero = [
                (count, label, color)
                for count, label, color in zip(quality_counts, quality_labels, colors)
                if count > 0
            ]

            fig, ax = plt.subplots(figsize=figsize)
            if non_zero:
                counts, labels, pie_colors = zip(*non_zero)
                ax.pie(
                    counts,
                    labels=labels,
                    colors=pie_colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.set_title(
                    f"Distribuicao da Qualidade dos Dados das Features{title_suffix}"
                )

            plt.tight_layout()
            plt.show()
