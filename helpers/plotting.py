"""Plotting utilities for visualizing Gram-Schmidt orthogonalization results."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    "CGS": "#1f77b4", "CGS2": "#ff7f0e", "MGS": "#e5a000",
    "MGS2": "#7f3c8d", "RGS": "#4c9f38", "RGS-L2C": "#39a9c6",
    "RGS-L2M": "#c72535",
}
SKETCH_COLORS = {
    "srht": "#1f77b4",
    "gaussian": "#ff7f0e",
    "sparse": "#2ca02c",
}
MARKERS = {
    "CGS": "*", "CGS2": "s", "MGS": "o", "MGS2": "d",
    "RGS": None, "RGS-L2C": "<", "RGS-L2M": "+",
}

def base_method_name(label: str) -> str:
    """Extract the base method name from an experiment label."""
    name = label.split(" k=", maxsplit=1)[0]
    return name.split(" sketch=", maxsplit=1)[0]


def sketch_size_from_label(label: str) -> int | None:
    """Extract k from labels with optional sketch information."""
    if " k=" not in label:
        return None

    value = label.rsplit(" k=", maxsplit=1)[1]
    value = value.split(maxsplit=1)[0]
    return int(value)

def sketch_type_from_label(label: str) -> str | None:
    """Extract the sketch type from an experiment label."""
    if " sketch=" not in label:
        return None

    return label.split(" sketch=", maxsplit=1)[1].split(
        maxsplit=1
    )[0]


K_LINESTYLES = ["-", "--", ":", "-."]


def _plot_metric(ax, checkpoints, results, key, title, ylabel=None):
    rgs_k_values = sorted({
        k
        for name in results
        if (k := sketch_size_from_label(name)) is not None
    })
    linestyle_by_k = {
        k: K_LINESTYLES[index % len(K_LINESTYLES)]
        for index, k in enumerate(rgs_k_values)
    }

    for name, metrics in results.items():
        style_name = base_method_name(name)
        k = sketch_size_from_label(name)
        sketch_type = sketch_type_from_label(name)
        values = np.maximum(metrics[key], np.finfo(np.float64).tiny)
        marker = MARKERS[style_name]
        color = (
            SKETCH_COLORS[sketch_type]
            if sketch_type is not None
            else COLORS[style_name]
        )
        kwargs = {}
        if marker in ("o", "s", "d", "<"):
            kwargs["markerfacecolor"] = "none"
        ax.semilogy(
            checkpoints,
            values,
            label=name,
            color=color,
            marker=marker,
            linestyle=linestyle_by_k[k] if k is not None else "--",
            linewidth=1.4,
            markersize=5,
            **kwargs,
        )
    ax.set_title(title)
    ax.set_xlabel(r"$i$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)


def create_plots(
    checkpoints, cond_a, results, output_dir: Path, show=False, metric_name="M"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig1, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].semilogy(checkpoints, cond_a, linewidth=1.5)
    axes[0].set_title(r"Condition numbers of $A_i$")
    axes[0].set_xlabel(r"$i$")
    axes[0].set_ylabel(r"$\kappa(A_i)$")
    axes[0].grid(True, which="both", alpha=0.25)
    _plot_metric(axes[1], checkpoints, results, "cond_q",
                 r"Condition numbers of $Q_i$", r"$\kappa(Q_i)$")
    fig1.savefig(output_dir / "figure1_condition_numbers.png", dpi=220)

    fig2, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    _plot_metric(axes[0], checkpoints, results, "residual",
                 r"Error $\|A_i-Q_iR_i\|_F/\|A_i\|_F$")
    _plot_metric(axes[1], checkpoints, results, "orthogonality2",
                 rf"Loss of orthogonality $\|I_i-Q_i^T{metric_name}Q_i\|_2$")
    _plot_metric(axes[2], checkpoints, results, "orthogonalityF",
                 rf"Loss of orthogonality $\|I_i-Q_i^T{metric_name}Q_i\|_F$")
    fig2.savefig(output_dir / "figure2_errors.png", dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)


def create_sketch_plot(
    checkpoints,
    sketch_results,
    output_dir: Path,
    show=False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )

    _plot_metric(
        axes[0],
        checkpoints,
        sketch_results,
        "orthogonality2",
        r"Sketched orthogonality error $\|I_i-S_i^TS_i\|_2$",
        r"$\|I_i-S_i^TS_i\|_2$",
    )

    _plot_metric(
        axes[1],
        checkpoints,
        sketch_results,
        "orthogonalityF",
        r"Sketched orthogonality error $\|I_i-S_i^TS_i\|_F$",
        r"$\|I_i-S_i^TS_i\|_F$",
    )

    fig.savefig(
        output_dir / "figure3_sketch_orthogonality.png",
        dpi=220,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)