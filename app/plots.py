"""
Plotly chart builders for the four experiments.

Each function takes a filtered DataFrame (already subset to desired strategies
and parameter range) and returns a Plotly Figure. All functions share the same
color/dash scheme so the visual identity is consistent across all four pages.

Color scheme:
  Static  — warm reds/oranges (S1=red, S2=orange, S3=gold)
  Dynamic — cool blues/greens (D1=steelblue, D2=dodgerblue, D3=mediumseagreen)
  Novel   — purple (D4=mediumpurple)
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

STRATEGY_COLORS: dict[str, str] = {
    "S1": "#e63946",        # red
    "S2": "#f4a261",        # orange
    "S3": "#e9c46a",        # gold
    "D1": "#4895ef",        # steel blue
    "D2": "#0077b6",        # deep blue
    "D3": "#2a9d8f",        # teal
    "D4": "#9b5de5",        # purple (novel contribution)
}

STRATEGY_DASH: dict[str, str] = {
    "S1": "dot",
    "S2": "dash",
    "S3": "dashdot",
    "D1": "solid",
    "D2": "solid",
    "D3": "solid",
    "D4": "solid",
}

STRATEGY_LABELS: dict[str, str] = {
    "S1": "S1 — Fixed 15W",
    "S2": "S2 — Fixed 30W",
    "S3": "S3 — Fixed 60W",
    "D1": "D1 — Long-term",
    "D2": "D2 — Adaptive",
    "D3": "D3 — Dynamic PM",
    "D4": "D4 — Energy-proportional (novel)",
}

_LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13),
    legend=dict(
        orientation="v",
        x=1.02, y=1,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ddd",
        borderwidth=1,
    ),
    margin=dict(l=60, r=180, t=50, b=60),
    hovermode="x unified",
)


def _add_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    strategies: list[str],
    err_col: str | None = None,
    mode: str = "lines+markers",
) -> None:
    """Add one trace per strategy with shared style."""
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values(x_col)
        if sub.empty:
            continue
        error_y = None
        if err_col and err_col in sub.columns:
            error_y = dict(type="data", array=sub[err_col].tolist(), visible=True)
        fig.add_trace(go.Scatter(
            x=sub[x_col],
            y=sub[y_col],
            name=STRATEGY_LABELS.get(strat, strat),
            mode=mode,
            line=dict(color=STRATEGY_COLORS[strat], dash=STRATEGY_DASH[strat], width=2.5),
            marker=dict(size=7),
            error_y=error_y,
        ))


# ---------------------------------------------------------------------------
# Experiment 1 — Energy sweep plots
# ---------------------------------------------------------------------------

def exp1_downtime(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Inactive fraction vs mean energy arrival rate."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_inactive_fraction", strategies,
                err_col="std_inactive_fraction")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 1 — Device Downtime vs Energy Arrival Rate",
        xaxis_title="Mean energy arrival [kJ/slot]  (×1000 = J/slot)",
        yaxis_title="Inactive fraction (downtime)",
        yaxis=dict(range=[0, None], tickformat=".0%"),
    )
    return fig


def exp1_throughput(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Normalised throughput vs mean energy arrival rate."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_throughput", strategies,
                err_col="std_throughput")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 1 — Normalised Throughput vs Energy Arrival Rate",
        xaxis_title="Mean energy arrival [kJ/slot]  (×1000 = J/slot)",
        yaxis_title="Normalised throughput (completed / arrived)",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def exp1_battery(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Average battery level vs mean energy arrival rate."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_battery", strategies,
                err_col="std_battery")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 1 — Average Battery Level vs Energy Arrival Rate",
        xaxis_title="Mean energy arrival [kJ/slot]",
        yaxis_title="Average battery level [kJ]",
    )
    return fig


# ---------------------------------------------------------------------------
# Experiment 2 — Arrival probability sweep plots
# ---------------------------------------------------------------------------

def exp2_dropped(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Jobs dropped vs job arrival probability."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_jobs_dropped", strategies,
                err_col="std_jobs_dropped")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 2 — Jobs Dropped vs Job Arrival Probability",
        xaxis_title="Job arrival probability p",
        yaxis_title="Mean jobs dropped per run",
    )
    return fig


def exp2_throughput(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Normalised throughput vs job arrival probability."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_throughput", strategies,
                err_col="std_throughput")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 2 — Normalised Throughput vs Job Arrival Probability",
        xaxis_title="Job arrival probability p",
        yaxis_title="Normalised throughput (completed / arrived)",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def exp2_downtime(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Inactive fraction vs job arrival probability."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_inactive_fraction", strategies,
                err_col="std_inactive_fraction")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 2 — Device Downtime vs Job Arrival Probability",
        xaxis_title="Job arrival probability p",
        yaxis_title="Inactive fraction (downtime)",
        yaxis=dict(range=[0, None], tickformat=".0%"),
    )
    return fig


# ---------------------------------------------------------------------------
# Experiment 3 — Diurnal energy model plots
# ---------------------------------------------------------------------------

def exp3_downtime(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Downtime vs diurnal peak energy."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_inactive_fraction", strategies,
                err_col="std_inactive_fraction")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 3 — Device Downtime under Diurnal Energy (novel)",
        xaxis_title="Diurnal peak energy [kJ/slot]",
        yaxis_title="Inactive fraction (downtime)",
        yaxis=dict(range=[0, None], tickformat=".0%"),
    )
    return fig


def exp3_throughput(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Throughput vs diurnal peak energy."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_throughput", strategies,
                err_col="std_throughput")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 3 — Throughput under Diurnal Energy (novel)",
        xaxis_title="Diurnal peak energy [kJ/slot]",
        yaxis_title="Normalised throughput",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def exp3_comparison_bar(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """
    Bar chart comparing mean downtime across strategies at a single peak level.
    Useful for highlighting D4 vs D2 performance gap.
    """
    # Use the median peak value for the bar chart
    peaks = sorted(df["param_value"].unique())
    mid_peak = peaks[len(peaks) // 2]
    sub = df[df["param_value"].round(3) == round(mid_peak, 3)]
    sub = sub[sub["strategy"].isin(strategies)].sort_values("mean_inactive_fraction")

    fig = go.Figure(go.Bar(
        x=sub["strategy"],
        y=sub["mean_inactive_fraction"],
        error_y=dict(type="data", array=sub["std_inactive_fraction"].tolist()),
        marker_color=[STRATEGY_COLORS.get(s, "#888") for s in sub["strategy"]],
        text=[f"{v:.1%}" for v in sub["mean_inactive_fraction"]],
        textposition="outside",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"Exp 3 — Downtime Comparison at peak={mid_peak:.2f} kJ/slot",
        xaxis_title="Strategy",
        yaxis_title="Mean inactive fraction",
        yaxis=dict(tickformat=".0%"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Experiment 4 — Heterogeneous devices plots
# ---------------------------------------------------------------------------

def exp4_downtime(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Downtime vs heterogeneity scale."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_inactive_fraction", strategies,
                err_col="std_inactive_fraction")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 4 — Device Downtime vs Heterogeneity Scale (novel)",
        xaxis_title="Heterogeneity scale (0 = identical, 1 = wide spread)",
        yaxis_title="Inactive fraction (downtime)",
        yaxis=dict(range=[0, None], tickformat=".0%"),
    )
    return fig


def exp4_throughput(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Throughput vs heterogeneity scale."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_throughput", strategies,
                err_col="std_throughput")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 4 — Throughput vs Heterogeneity Scale (novel)",
        xaxis_title="Heterogeneity scale (0 = identical, 1 = wide spread)",
        yaxis_title="Normalised throughput",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def exp4_battery(df: pd.DataFrame, strategies: list[str]) -> go.Figure:
    """Battery vs heterogeneity scale."""
    fig = go.Figure()
    _add_traces(fig, df, "param_value", "mean_battery", strategies,
                err_col="std_battery")
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Exp 4 — Average Battery vs Heterogeneity Scale (novel)",
        xaxis_title="Heterogeneity scale",
        yaxis_title="Average battery level [kJ]",
    )
    return fig


# ---------------------------------------------------------------------------
# Summary / overview plot
# ---------------------------------------------------------------------------

def summary_heatmap(dfs: dict[str, pd.DataFrame], metric: str = "mean_throughput") -> go.Figure:
    """
    Heatmap of a chosen metric across all strategies and experiments.

    dfs : dict mapping experiment name ('exp1', 'exp2', 'exp3', 'exp4') to DataFrame
    metric : column name to aggregate (mean across all param values)
    """
    import numpy as np
    strategies = ["S1", "S2", "S3", "D1", "D2", "D3", "D4"]
    exp_labels = {
        "exp1": "Exp 1 (energy sweep)",
        "exp2": "Exp 2 (arrival sweep)",
        "exp3": "Exp 3 (diurnal)",
        "exp4": "Exp 4 (heterogeneous)",
    }
    matrix = []
    valid_exps = []
    for exp_key, label in exp_labels.items():
        if exp_key not in dfs or dfs[exp_key].empty:
            continue
        df = dfs[exp_key]
        if metric not in df.columns:
            continue
        row = []
        for s in strategies:
            sub = df[df["strategy"] == s]
            row.append(float(sub[metric].mean()) if not sub.empty else float("nan"))
        matrix.append(row)
        valid_exps.append(label)

    if not matrix:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=18))
        return fig

    z = [row for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=strategies,
        y=valid_exps,
        colorscale="RdYlGn",
        text=[[f"{v:.3f}" if not (v != v) else "N/A" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title=metric),
    ))
    metric_label = metric.replace("mean_", "").replace("_", " ").title()
    layout = {**_LAYOUT_BASE, "margin": dict(l=180, r=60, t=50, b=60)}
    fig.update_layout(
        **layout,
        title=f"Strategy Comparison Overview — {metric_label}",
        xaxis_title="Strategy",
        yaxis_title="Experiment",
    )
    return fig
