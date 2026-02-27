import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from src.utils.metrics import SELECTED_METRICS, RESCALED_METRICS, METRIC_ABBREV

GRAY = (0.6392156862745098, 0.6392156862745098, 0.6392156862745098)


def plot_datasets(
    data,
    param_values,
    n_rows=None,
    n_cols=None,
    figsize=2.0,
    dpi=200,
    cmap=None,
):
    """Plots all datasets in data with corresponding param_value as title.
    `fig_x` columns and `fig_y` rows.

    Args:
        data: 2d matrix of type [datasets x runs]
        param_values: 1d matrix with parameter values per dataset. Used for title.
    """

    if n_rows is None and n_cols is None:
        n_rows = n_cols = int(math.sqrt(len(data) - 1) + 1)
    if n_rows is None:
        n_rows = (len(data) - 1) // n_cols + 1
    if n_cols is None:
        n_cols = (len(data) - 1) // n_rows + 1

    fig = plt.figure(
        figsize=(figsize * n_cols, (figsize + 0.2) * n_rows),
        dpi=dpi,
        layout="tight",
    )
    G = gridspec.GridSpec(n_rows, n_cols)

    length = min(len(data), len(param_values), n_cols * n_rows)
    data = data[:length]
    param_values = param_values[:length]

    for param_value in range(0, len(data)):
        ax = plt.subplot(G[param_value // n_cols, param_value % n_cols])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{param_values[param_value]}")
        X, l = data[param_value][0]
        ax.scatter(X[:, 0], X[:, 1], s=1, c=l, cmap=cmap)

    return fig


def plot_lineplot(
    df,
    x_axis,
    y_axis,
    grouping=None,
    order=SELECTED_METRICS,
    x_range=(None, None),
    y_range=(None, None),
    x_label=None,
    y_label=None,
    # figsize=(15, 5),
    # figsize=(9, 4),
    figsize=(10, 6),
    dpi=200,
    # errorbar="se",
    errorbar=("ci", 75),
    highlight=1,
    highlight_size=2,
    red_legend_lables=[],
    metric_abbrev=METRIC_ABBREV,
    font_size=16,
    ncol=4,
    row_wise=True,
    palette=None,
    fig=None,
    ax=None,
    markersize=10,
    sizes=None,
    dashes=None,
    markers=None,
):
    """Plot a line plot for a dataframe."""
    plt.rcParams.update({"font.size": font_size})

    if fig is None:
        fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
            layout="tight",
        )
    highlight -= 1

    if x_label is not None:
        df = df.rename(columns={x_axis: x_label})
        x_axis = x_label

    if y_label is not None:
        df = df.rename(columns={y_axis: y_label})
        y_axis = y_label

    if grouping is not None:
        if order is None:
            order = list(df[grouping].unique())
        for metric in order.copy():
            if metric not in df[grouping].unique():
                order.remove(metric)

    highlight_index = (
        [highlight] + list(range(0, highlight)) + list(range(highlight + 1, len(order)))
    )
    order = list(np.array(order)[highlight_index])

    def repeat(array):
        return array * ((len(order) - 1 + len(array)) // len(array))

    if markers is None:
        markers = ["o"] + repeat(["v", "^", "<", ">", "p", "P", "X", "d", "D", "H"])
    if palette is None:
        palette = ["black"] + repeat(sns.color_palette("bright"))
    if sizes is None:
        sizes = [highlight_size] + repeat([1])
    # sizes = np.array(sizes) * 2
    # sizes = np.array(sizes) / 2
    if dashes is None:
        dashes = [(1, 0)] + repeat([(1, 2), (4, 2), (3, 2, 1, 2)])

    ax = sns.lineplot(
        data=df,
        x=x_axis,
        y=y_axis,
        markers=dict(zip(order, markers)),
        markersize=markersize,
        hue=grouping,
        palette=dict(zip(order, palette)),
        hue_order=order[::-1],
        style=grouping,
        dashes=dict(zip(order, dashes)),
        size=grouping,
        sizes=dict(zip(order, sizes)),
        errorbar=errorbar,
        ax=ax,
    )

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    ### Coloring of the plot
    ax.set_facecolor("white")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.grid(color="lightgray")

    ### Legend
    handles, _ = ax.get_legend_handles_labels()
    inverse_index = np.empty(len(order), dtype=int)
    inverse_index[highlight_index] = np.arange(0, len(order))
    if row_wise:
        row_wise_index = sum((list(range(len(order))[i::ncol]) for i in range(ncol)), [])
    else:
        row_wise_index = list(range(len(order)))
    leg = ax.legend(
        handles=list(np.array(handles[::-1])[inverse_index][row_wise_index]),
        labels=list(np.array(order)[inverse_index][row_wise_index]),
        # loc="center left",
        # bbox_to_anchor=(1, 0.5),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        # loc="upper center",
        # bbox_to_anchor=(0.5, 0),
        # fontsize=19,
        ncol=ncol,
    )
    for text in leg.get_texts():
        if text.get_text() in red_legend_lables:
            text.set_color("red")
        if text.get_text() in metric_abbrev:
            text.set_text(metric_abbrev[text.get_text()])

    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")

    return fig


def plot_barplot(
    df,
    x_axis,
    y_axis,
    grouping=None,
    order=None,
    x_range=(None, None),
    y_range=(None, None),
    figsize=(15, 5),
    errorbar="se",
):
    """Plot a barplot for a dataframe."""

    fig = plt.figure(
        figsize=figsize,
        layout="tight",
    )
    sns.set_theme(style="whitegrid", palette="bright")

    ax = sns.barplot(
        df,
        x=x_axis,
        y=y_axis,
        hue=grouping,
        hue_order=order,
        errorbar=errorbar,
    )

    for container in ax.containers:
        tmp_hue = df.loc[df[grouping] == container.get_label()]
        ax.bar_label(container, labels=tmp_hue[y_axis])

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    ### Coloring of the plot
    # ax.spines["bottom"].set_color("black")
    # ax.spines["left"].set_color("black")
    # ax.spines["right"].set_color("white")
    # ax.spines["top"].set_color("white")

    sns.reset_orig()
    return fig
