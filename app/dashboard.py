"""
Streamlit dashboard for the decentralized LLM energy strategy comparison.

Run from the project root:
  uv run streamlit run app/dashboard.py
  # or
  streamlit run app/dashboard.py

Pages
-----
  Overview      — heatmap comparing all strategies across all experiments
  Exp 1         — energy sweep (reproduce paper Fig 3a/4a)
  Exp 2         — arrival rate sweep (reproduce paper Fig 3b/4b)
  Exp 3         — diurnal energy model (novel)
  Exp 4         — heterogeneous devices (novel)
  Raw Data      — browse and download any CSV
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from plots import (
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    exp1_battery,
    exp1_downtime,
    exp1_throughput,
    exp2_downtime,
    exp2_dropped,
    exp2_throughput,
    exp3_comparison_bar,
    exp3_downtime,
    exp3_throughput,
    exp4_battery,
    exp4_downtime,
    exp4_throughput,
    summary_heatmap,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Edge LLM Energy Strategies",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
ALL_STRATEGIES = ["S1", "S2", "S3", "D1", "D2", "D3", "D4"]

EXPERIMENT_FILES = {
    "exp1": RESULTS_DIR / "exp1_energy_sweep.csv",
    "exp2": RESULTS_DIR / "exp2_arrival_sweep.csv",
    "exp3": RESULTS_DIR / "exp3_diurnal.csv",
    "exp4": RESULTS_DIR / "exp4_heterogeneous.csv",
}

EXPERIMENT_TITLES = {
    "exp1": "Exp 1 — Energy Arrival Rate Sweep",
    "exp2": "Exp 2 — Job Arrival Probability Sweep",
    "exp3": "Exp 3 — Diurnal Energy Model (novel)",
    "exp4": "Exp 4 — Heterogeneous Devices (novel)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_all() -> dict[str, pd.DataFrame | None]:
    return {key: load_csv(path) for key, path in EXPERIMENT_FILES.items()}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar(dfs: dict) -> tuple[str, list[str]]:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Golden_Gate_Bridge_20100906_03.jpg/1px-x.gif",
        width=1,
    )
    st.sidebar.title("⚡ Edge LLM Energy")
    st.sidebar.caption(
        "Comparative Analysis of Static & Dynamic Energy Management "
        "Strategies for Decentralized LLM Inference on Energy-Harvesting "
        "Edge Networks"
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Exp 1 — Energy Sweep", "Exp 2 — Arrival Rate",
         "Exp 3 — Diurnal (novel)", "Exp 4 — Heterogeneous (novel)", "Raw Data"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter strategies")

    # Color-coded checkboxes
    selected = []
    for s in ALL_STRATEGIES:
        color = STRATEGY_COLORS[s]
        label_html = (
            f'<span style="color:{color};font-weight:600">{s}</span> '
            f'— {STRATEGY_LABELS[s].split("—")[1].strip()}'
        )
        # Streamlit doesn't support HTML in checkboxes, use plain label
        checked = st.sidebar.checkbox(
            f"{s}  {STRATEGY_LABELS[s].split('—')[1].strip()}",
            value=True,
            key=f"chk_{s}",
        )
        if checked:
            selected.append(s)

    if not selected:
        st.sidebar.warning("Select at least one strategy.")
        selected = ALL_STRATEGIES[:]

    st.sidebar.markdown("---")

    # Data status
    st.sidebar.subheader("Data status")
    for key, df in dfs.items():
        icon = "✅" if df is not None else "❌"
        rows = f"{len(df)} rows" if df is not None else "missing"
        st.sidebar.caption(f"{icon} {EXPERIMENT_TITLES[key]} — {rows}")

    return page, selected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def no_data_warning(exp_key: str) -> None:
    path = EXPERIMENT_FILES[exp_key]
    st.warning(
        f"No data found for **{EXPERIMENT_TITLES[exp_key]}**.\n\n"
        f"Expected file: `{path}`\n\n"
        "Run the experiment first:\n"
        "```bash\n"
        f"uv run python -m experiments.run_experiments --exp {exp_key[-1]}\n"
        "```"
    )


def metric_cards(df: pd.DataFrame, strategies: list[str]) -> None:
    """Show key summary statistics as metric cards."""
    cols = st.columns(len(strategies))
    for i, s in enumerate(strategies):
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        best_throughput = sub["mean_throughput"].max()
        best_downtime = sub["mean_inactive_fraction"].min()
        with cols[i]:
            st.metric(
                label=STRATEGY_LABELS.get(s, s),
                value=f"{best_throughput:.1%}",
                delta=f"downtime {best_downtime:.1%}",
                delta_color="inverse",
                help=f"Best normalised throughput and lowest downtime across all sweep points",
            )


def data_table(df: pd.DataFrame, strategies: list[str]) -> None:
    """Show filtered data table with download button."""
    filtered = df[df["strategy"].isin(strategies)]
    with st.expander("View data table"):
        st.dataframe(
            filtered.style.format({
                c: "{:.4f}" for c in filtered.select_dtypes("float").columns
            }),
            use_container_width=True,
        )
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False),
            file_name="filtered_results.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def page_overview(dfs: dict, strategies: list[str]) -> None:
    st.title("Overview — Strategy Comparison Across All Experiments")
    st.markdown(
        """
        The heatmap below shows the **mean performance** of each strategy
        across all four experiments. Darker green = better performance.

        **Research question:** How do static vs dynamic energy strategies
        compare in throughput, downtime, and device availability for
        decentralized LLM inference on energy-harvesting edge networks?
        """
    )

    metric = st.selectbox(
        "Metric to display",
        ["mean_throughput", "mean_inactive_fraction", "mean_jobs_dropped", "mean_battery"],
        format_func=lambda x: x.replace("mean_", "").replace("_", " ").title(),
    )

    loaded = {k: v for k, v in dfs.items() if v is not None}
    if not loaded:
        st.error("No experiment data found. Run experiments first.")
        return

    filtered = {k: df[df["strategy"].isin(strategies)] for k, df in loaded.items()}
    fig = summary_heatmap(filtered, metric)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key findings")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**Claim 1 — Static insufficient**\n\n"
            "S1/S2/S3 diverge sharply as energy becomes scarce. "
            "No single static wattage is universally optimal."
        )
    with col2:
        st.success(
            "**Claim 2 — Dynamism pays off**\n\n"
            "D1 > S1–S3. D2 > D1. D3+D2 > D2 alone. "
            "Each layer of adaptation adds measurable gain."
        )
    with col3:
        st.warning(
            "**Claim 3 — D4 leads under correlated energy**\n\n"
            "Continuous proportional weighting (D4) outperforms "
            "threshold-based D2 under diurnal and heterogeneous conditions."
        )


def page_exp1(df: pd.DataFrame | None, strategies: list[str]) -> None:
    st.title("Experiment 1 — Energy Arrival Rate Sweep")
    st.markdown(
        """
        **What this tests:** How all 7 strategies perform as mean energy
        arrival varies from scarce (50 kJ/slot) to abundant
        (600 kJ/slot), with fixed job load p=0.3.

        **Reproduces:** Khoshsirat et al. (GLOBECOM 2024) Fig 3a / Fig 4a,
        extended to all 7 strategies.

        **Expected ordering:** Downtime: D2+D3 < D1 < S2 < S1 < S3 at low energy.
        Throughput: D2+D3 > D1 > S2 > S3 > S1 consistently.
        """
    )

    if df is None:
        no_data_warning("exp1")
        return

    metric_cards(df, strategies)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Downtime", "Throughput", "Battery"])
    with tab1:
        st.plotly_chart(exp1_downtime(df, strategies), use_container_width=True)
    with tab2:
        st.plotly_chart(exp1_throughput(df, strategies), use_container_width=True)
    with tab3:
        st.plotly_chart(exp1_battery(df, strategies), use_container_width=True)

    data_table(df, strategies)


def page_exp2(df: pd.DataFrame | None, strategies: list[str]) -> None:
    st.title("Experiment 2 — Job Arrival Probability Sweep")
    st.markdown(
        """
        **What this tests:** How strategies respond as job load increases
        from p=0.1 (light) to p=1.0 (every slot), with energy fixed at
        550 kJ/slot (baseline).

        **Reproduces:** Khoshsirat et al. (GLOBECOM 2024) Fig 3b / Fig 4b,
        extended to all 7 strategies.

        **Expected ordering:** Jobs dropped increases for all, but static
        strategies drop far more than dynamic ones at p > 0.6.
        """
    )

    if df is None:
        no_data_warning("exp2")
        return

    metric_cards(df, strategies)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Jobs Dropped", "Throughput", "Downtime"])
    with tab1:
        st.plotly_chart(exp2_dropped(df, strategies), use_container_width=True)
    with tab2:
        st.plotly_chart(exp2_throughput(df, strategies), use_container_width=True)
    with tab3:
        st.plotly_chart(exp2_downtime(df, strategies), use_container_width=True)

    data_table(df, strategies)


def page_exp3(df: pd.DataFrame | None, strategies: list[str]) -> None:
    st.title("Experiment 3 — Diurnal Energy Model (Novel)")
    st.markdown(
        """
        **What this tests:** How strategies respond when energy arrives in a
        sinusoidal daily pattern (solar harvesting) rather than i.i.d. uniform.
        The temporal correlation creates extended low-energy periods that stress
        threshold-based approaches.

        **Novel experiment** — no paper figures to match.

        **Expected findings:**
        - **D4 outperforms D2** because it reacts continuously rather than
          only when devices hit the 15W threshold.
        - **D1 degrades most** — its weights were computed assuming i.i.d.
          arrivals, so correlated low-energy stretches invalidate the model.
        """
    )

    if df is None:
        no_data_warning("exp3")
        return

    tab1, tab2, tab3 = st.tabs(["Downtime", "Throughput", "Bar at Mid-Peak"])
    with tab1:
        st.plotly_chart(exp3_downtime(df, strategies), use_container_width=True)
    with tab2:
        st.plotly_chart(exp3_throughput(df, strategies), use_container_width=True)
    with tab3:
        st.plotly_chart(exp3_comparison_bar(df, strategies), use_container_width=True)
        st.caption(
            "Bar chart shows downtime at the median diurnal peak value. "
            "D4 vs D2 gap should be visible here."
        )

    data_table(df, strategies)


def page_exp4(df: pd.DataFrame | None, strategies: list[str]) -> None:
    st.title("Experiment 4 — Heterogeneous Devices (Novel)")
    st.markdown(
        """
        **What this tests:** How strategies perform when devices have different
        battery capacities (E_max) and solar panel strengths, ranging from
        fully homogeneous (scale=0) to wide heterogeneity (scale=1).

        **Novel experiment** — no paper figures to match.

        **Expected findings:**
        - **D4 becomes relatively more valuable** under heterogeneity —
          proportional weighting adapts naturally to different E_max values.
        - **D2 may over-route to large-battery devices** because its fixed
          absolute threshold (10 kJ) triggers proportionally earlier on
          small-battery devices.
        - Dynamic strategies should maintain their advantage over static ones.
        """
    )

    if df is None:
        no_data_warning("exp4")
        return

    metric_cards(df, strategies)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Downtime", "Throughput", "Battery"])
    with tab1:
        st.plotly_chart(exp4_downtime(df, strategies), use_container_width=True)
    with tab2:
        st.plotly_chart(exp4_throughput(df, strategies), use_container_width=True)
    with tab3:
        st.plotly_chart(exp4_battery(df, strategies), use_container_width=True)

    data_table(df, strategies)


def page_raw_data(dfs: dict) -> None:
    st.title("Raw Data Browser")
    st.markdown("Browse, filter, and download any experiment CSV.")

    exp_choice = st.selectbox(
        "Select experiment",
        list(EXPERIMENT_TITLES.keys()),
        format_func=lambda k: EXPERIMENT_TITLES[k],
    )
    df = dfs.get(exp_choice)

    if df is None:
        no_data_warning(exp_choice)
        return

    st.markdown(f"**{len(df)} rows** | Columns: {', '.join(df.columns)}")

    # Strategy filter
    available_strategies = sorted(df["strategy"].unique())
    selected = st.multiselect(
        "Filter strategies", available_strategies, default=available_strategies
    )
    filtered = df[df["strategy"].isin(selected)]

    st.dataframe(
        filtered.style.format({
            c: "{:.4f}" for c in filtered.select_dtypes("float").columns
        }),
        use_container_width=True,
        height=450,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        st.download_button(
            "Download CSV",
            data=filtered.to_csv(index=False),
            file_name=f"{exp_choice}_filtered.csv",
            mime="text/csv",
        )
    with col2:
        st.caption(f"Source: `{EXPERIMENT_FILES[exp_choice]}`")

    st.markdown("---")
    st.subheader("Summary statistics")
    st.dataframe(
        filtered.groupby("strategy")[
            ["mean_throughput", "mean_inactive_fraction",
             "mean_jobs_dropped", "mean_battery"]
        ].agg(["mean", "std"]).round(4),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    dfs = load_all()
    page, strategies = sidebar(dfs)

    if page == "Overview":
        page_overview(dfs, strategies)
    elif page == "Exp 1 — Energy Sweep":
        page_exp1(dfs.get("exp1"), strategies)
    elif page == "Exp 2 — Arrival Rate":
        page_exp2(dfs.get("exp2"), strategies)
    elif page == "Exp 3 — Diurnal (novel)":
        page_exp3(dfs.get("exp3"), strategies)
    elif page == "Exp 4 — Heterogeneous (novel)":
        page_exp4(dfs.get("exp4"), strategies)
    elif page == "Raw Data":
        page_raw_data(dfs)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "CECS 574 — Distributed Systems\n\n"
        "Comparative Analysis of Static & Dynamic Energy Management\n\n"
        "Khoshsirat et al. (GLOBECOM 2024) extension"
    )


if __name__ == "__main__":
    main()
