import sys

DISCO_ROOT_PATH = "/export/share/pascalw777dm/DISCO"
sys.path.append(DISCO_ROOT_PATH)

from src.utils.metrics import METRIC_ABBREV_TABLES

import glob
import itertools
import pandas as pd
import numpy as np
import subprocess

from collections import defaultdict
from collections.abc import Mapping


### pandas refactoring
def deep_update(dict, dict_update):
    for k, v in dict_update.items():
        if isinstance(v, Mapping):
            dict[k] = deep_update(dict.get(k, {}), v)
        else:
            dict[k] = v
    return dict


def group_by_column_and_aggregate_values(df, column, aggregation_func):
    try:
        column_values = df.groupby(["dataset", "measure"])[column]
    except:
        column_values = df.groupby(["dataset", "function"])[column]
    aggregated_values = getattr(column_values, aggregation_func)()

    data_dict = defaultdict(dict)
    for (dataset, eval_method), aggregated_value in aggregated_values.to_dict().items():
        data_dict[(dataset, aggregation_func)][(eval_method, column)] = aggregated_value

    return data_dict


def calc_pairwise_column_aggregation_func_dict(df, columns, aggregation_funcs):
    pairwise_column_aggregation_func = itertools.product(columns, aggregation_funcs)
    pairwise_column_aggregation_func_data_dict = {}
    for column, aggregations_func in pairwise_column_aggregation_func:
        column_aggregation_func_data_dict = group_by_column_and_aggregate_values(
            df, column, aggregations_func
        )
        deep_update(pairwise_column_aggregation_func_data_dict, column_aggregation_func_data_dict)
    return pairwise_column_aggregation_func_data_dict


def gather_and_aggregate_data(paths, columns, aggregation_funcs):
    dataframes = [pd.read_csv(file_path) for path in paths for file_path in glob.glob(f"{path}*/*")]
    df = pd.concat(dataframes)

    data_dict = calc_pairwise_column_aggregation_func_dict(df, columns, aggregation_funcs)
    return pd.DataFrame.from_dict(data_dict, orient="index")


### Reindex stuff
def reindex_df(df, row_index, column_index, precision=3):
    df = df.reindex(row_index)
    df = df.reindex(columns=df.columns.reindex(column_index)[0])
    df = df.round(precision)
    return df


def calc_indices_and_reindex(df, dataset_names, aggregation_funcs, metrics, selection, precision=3):
    if metrics is None:
        metrics = [column[0] for column in df.columns]
    row_index = list(itertools.product(dataset_names, aggregation_funcs))
    column_index = list(itertools.product(metrics, selection))
    return reindex_df(df, row_index=row_index, column_index=column_index, precision=precision)


### Regex stuff
def run_regex(expr_list, path):
    for expr in expr_list:
        expr = expr.replace("'", r"\x27")
        expr_ = '\'' + expr + '\''
        path_ = '"' + path + '"'
        subprocess.run(f"perl -pwi -e {expr_} {path_}", shell=True, executable="/bin/bash")


def regex_file(path, caption, categories=[], metric_abbrev={}):
    generell_stuff_regex = [
        r's/table/table\*/g',
        r's/_/\\_/g',
        r's/\\\\\n/ \\\\\n/g',
        r'1 while s/\$nan( \\pm nan)?\$/-/g',
        # Two digits
        r's/ \\pm 0.00?\$/\$/g',
        r's/ \\pm nan?\$/\$/g',
        r's/(\d\.\d)( |\$)/${1}0$2/g',
        # arraystretch
        r's/(\\begin\{table\*\})/\\renewcommand\{\\arraystretch\}\{1.2\}\n\n\n$1/g',
        r's/(\\end\{table\*\}.*$)/$1\n\n\\renewcommand\{\\arraystretch\}\{1\}\n/g',
    ]

    remove_second_row_index_regex = [
        # Remove second level of row index
        r's/ & \\textbf\{Metric\}//g',
        # r's/^(.*?& ).*?& /$1/g',
        r's/(\{tabular\}\{l\|)l\|/$1/g',
        # Remove all midrules
        r's/\\midrule\n//g',
    ]

    caption_regex = [
        # Caption
        r's/\\caption\{TODO\}/\\caption\{%s\}/g'%(caption),
    ]

    categories_regex = [
        # Generell categories stuff
        r's/(\{tabular\}\{)/$1r/g',
        r's/^(.*?& )/& $1/g',
    ] + [
        # Categories
        r"s/(^& %s)/\\midrule\n\\parbox[t]\{2mm\}\{\\multirow\{%s\}\{*\}\{\\rotatebox[origin=c]\{90\}\{%s\}\}\}\n$1/g"
        % (first_dataset_in_category, nr_of_datasets_in_category, category_name)
        for first_dataset_in_category, nr_of_datasets_in_category, category_name in categories
    ]

    run_regex(remove_second_row_index_regex, path)
    run_regex(caption_regex, path)
    if len(categories) > 0:
        run_regex(categories_regex, path)
    else:
        # Insert midrule after headline
        run_regex([r's/(^\\textbf\{Dataset\}.*?$)/$1\n\\midrule/g'], path)

    if metric_abbrev:
        for metric, abbrev in metric_abbrev.items():
            abbrev = abbrev.replace("$", "\\$")
            run_regex([r's/& %s( |\\)/& %s$1/g'%(metric, abbrev)], path)

    run_regex(generell_stuff_regex, path)


### Coloring stuff
def latex_coloring(
    path,
    skiprows=9,
    axis=0,
    metric_selection=None,
    higher_is_better=None,
    lower_is_better=[],
    inverse_color=[],
    bold_underline=True,
    min_value = None,
    max_value = None,
):
    df = pd.read_csv(path, sep="&", header=0, index_col=0, skiprows=skiprows, skipfooter=3, engine="python")
    df = df.drop(df.columns[0], axis=1)
    if None in df.index:
        df = df.drop(index=[None], axis=0)
    if "\\midrule" in df.index:
        df = df.drop(index=["\\midrule"], axis=0)
    df.columns = df.columns.str.replace("\\", "")
    df.columns = df.columns.str.strip()
    df_std = df.copy()
    df_std = df_std.replace(r"\$(.*?) ?(\\pm.*?)?\$(.*\\\\)?", value=r"\2", regex=True)
    df_std = df_std.replace(r" $", value="", regex=True)
    df = df.replace(r"\$(.*?)( \\pm.*?)?\$(.*\\\\)?", value=r"\1", regex=True)
    df = df.astype(float)

    if metric_selection is None:
        metric_selection = df.columns
    df_selected = df[metric_selection].abs()

    df_min = df.copy()
    df_max = df.copy()
    if axis is None:
        df_min.loc[:, metric_selection] = df_selected.min(axis=axis, skipna=True)
        df_max.loc[:, metric_selection] = df_selected.max(axis=axis, skipna=True)
    else:
        df_min.loc[:, metric_selection] = np.expand_dims(df_selected.min(axis=axis, skipna=True).values, axis=axis)  # type: ignore
        df_max.loc[:, metric_selection] = np.expand_dims(df_selected.max(axis=axis, skipna=True).values, axis=axis)  # type: ignore

    if min_value:
        df_min.loc[:, metric_selection] = min_value
    if max_value:
        df_max.loc[:, metric_selection] = max_value

    df2 = df.copy()
    df2[df_selected < 0] = df_min[df_selected < 0] + df[df_selected < 0]

    df_color_saturation = df.copy()
    df_color_saturation.loc[:, :] = 0
    if higher_is_better is None:
        higher_is_better = df.columns
    # df_color_saturation.loc[:, higher_is_better] = (df2.loc[:, higher_is_better] - df_min.loc[:, higher_is_better]) / (
    #     df_max.loc[:, higher_is_better] - df_min.loc[:, higher_is_better]
    # )
    # lower_is_better = [metric for metric in lower_is_better if metric in df.columns]
    # df_color_saturation.loc[:, lower_is_better] = (df_max.loc[:, lower_is_better] - df.loc[:, lower_is_better]) / (
    #     df_max.loc[:, lower_is_better] - df_min.loc[:, lower_is_better]
    # )
    df_color_saturation.loc[:, higher_is_better] = (df2.loc[:, higher_is_better] - 0) / (
        df_max.loc[:, higher_is_better] - 0
    )
    lower_is_better = [metric for metric in lower_is_better if metric in df.columns]
    df_color_saturation.loc[:, lower_is_better] = (0 - df.loc[:, lower_is_better]) / (
        0 - df_min.loc[:, lower_is_better]
    )
    df_color_saturation = df_color_saturation.abs()
    df_color_saturation = df_color_saturation * 65 + 5
    df_color_saturation.replace(np.nan, 0, inplace=True)
    df_color_saturation = df_color_saturation.astype(int)

    df_latex = df.astype(str).combine(df_std.astype(str), lambda val, std: val + std)

    if bold_underline:
        df_largest = df.copy().astype(float)
        df_largest[:] = 0
        df_largest[df.T.apply(lambda x: np.sort(x[~(np.isnan(x))].unique())[-1] == x).T] = 1
        
        def safe_check(x):
            unique_vals = np.sort(x[~np.isnan(x)].unique())
            if len(unique_vals) < 2:
                # No second-to-last value, return False mask (no changes)
                return pd.Series([False]*len(x), index=x.index)
            else:
                return x == unique_vals[-2]
        df_largest[df.T.apply(safe_check).T] = 2

        df_latex = df_latex.astype(str).combine(
            df_largest,
            lambda value, large: r"$"
            + large.apply(lambda large_val: r"\bm{" if large_val == 1 else r"\underline{" if large_val == 2 else "")
            + value
            + large.apply(lambda large_val: r"}" if large_val != 0 else "")
            + r"$",
        )

    inverse_color = [metric for metric in inverse_color if metric in df.columns]

    df_coloring = df.astype(str).combine(
        df_color_saturation.astype(str),
        lambda value, color_saturation: "\\cellcolor{"
        + value.apply(lambda x, col=value.name: "Green" if (float(x) >= 0 if col not in inverse_color else float(x) <= 0) else "Red")
        + "!"
        + color_saturation
        + r"}",
    )
    df_latex = df_coloring + df_latex

    df_latex.insert(0, "dataset", df_latex.index.str.strip())
    df_joined_columns = df_latex[df_latex.columns[:]].apply(lambda x: " & ".join(x), axis=1)
    df_joined_columns = df_joined_columns.replace("\\\\", "\\\\\\\\", regex=True)
    df_joined_columns.index = df_joined_columns.index.str.replace("\\", "\\\\")

    for dataset, row in df_joined_columns.items():
        row = row.replace("$", "\\$")
        run_regex([r"s/%s.*\\\\/%s \\\\/g" % (dataset, row)], path)


### Generate_latex_file function
from clustpy.utils import evaluation_df_to_latex_table
from src.utils.metrics import METRIC_ABBREV_TABLES

def generate_latex_file(
    paths,
    latex_path,
    dataset_names,
    aggregation_funcs=["mean", "std"],
    metrics=None,
    metric_abbrev=METRIC_ABBREV_TABLES,
    selection=["value", "time", "process_time"],
    caption="TODO",
    categories=[],
    precision=3,
    latex_coloring_axis="no coloring",
    latex_coloring_selection=None,
    higher_is_better=None,
    lower_is_better=[],
):
    df_matrix = gather_and_aggregate_data(
        paths, columns=selection, aggregation_funcs=aggregation_funcs
    )
    df_reindexed = calc_indices_and_reindex(
        df_matrix, dataset_names, aggregation_funcs, metrics, selection, precision=precision
    )
    evaluation_df_to_latex_table(
        df_reindexed,
        latex_path,
        best_in_bold=False,
        second_best_underlined=False,
        in_percent=False,
        decimal_places=2,
    )

    if latex_coloring_axis != "no coloring":
        latex_coloring(latex_path, skiprows=6, axis=latex_coloring_axis, metric_selection=latex_coloring_selection, higher_is_better=higher_is_better, lower_is_better=lower_is_better)

    regex_file(latex_path, caption=caption, categories=categories, metric_abbrev=metric_abbrev)
    print(f"Generated: `{latex_path}`")
