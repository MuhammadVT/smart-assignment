"""Streamlit dashboard — explore committed TW1 slots by customer and route."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repo root is importable when Streamlit runs this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dlvr_window.charts import (
    build_customer_timeline,
    build_route_timeline,
    build_tier_distribution,
    committed_summary_metrics,
)
from analysis.dlvr_window.data import (
    filter_customer,
    filter_route,
    list_customers,
    list_routes,
    load_committed_tw1_slots_df,
)

DISPLAY_COLUMNS = [
    "route_id",
    "co_cust_nbr",
    "cust_tier",
    "tw1opendate",
    "tw1closedate",
    "tw1opentime",
    "tw1closetime",
    "window_minutes",
    "latitude",
    "longitude",
]


def _parse_cli_source() -> str:
    if "--sample" in sys.argv:
        return "sample"
    if "--sql" in sys.argv:
        return "sql"
    if "--cache" in sys.argv:
        return "cache"
    return "auto"


@st.cache_data(show_spinner="Loading committed TW1 slots…")
def _cached_load(source: str) -> pd.DataFrame:
    return load_committed_tw1_slots_df(source=source)  # type: ignore[arg-type]


def _render_customer_view(committed_df: pd.DataFrame) -> None:
    customers = list_customers(committed_df)
    if customers.empty:
        st.warning("No customers found in committed TW1 data.")
        return

    all_tiers = sorted(committed_df["cust_tier"].fillna("Other").astype(str).unique().tolist())

    st.markdown("#### Customer controls")
    control_col1, control_col2 = st.columns([1.6, 1])
    with control_col1:
        search = st.text_input("Search co_cust_nbr or name", "", key="cust_search")
        filtered = customers
        if search.strip():
            mask = customers["label"].str.contains(search.strip(), case=False, na=False)
            filtered = customers.loc[mask]

        if filtered.empty:
            st.warning("No customers match your search.")
            return

        selected_label = st.selectbox(
            "Select customer",
            options=filtered["label"].tolist(),
            index=0,
        )
        co_cust_nbr = filtered.loc[filtered["label"] == selected_label, "co_cust_nbr"].iloc[0]
    with control_col2:
        selected_tiers = st.multiselect("Customer tiers", options=all_tiers, default=all_tiers)

    customer_df = filter_customer(committed_df, co_cust_nbr, cust_tiers=selected_tiers or None)
    metrics = committed_summary_metrics(customer_df)

    if customer_df.empty:
        st.warning("No rows match your customer and tier filters.")
        return

    metric_cols = st.columns(6)
    metric_cols[0].metric("Rows", metrics.get("Rows", 0))
    metric_cols[1].metric("Routes", metrics.get("Routes", 0))
    metric_cols[2].metric("Customers", metrics.get("Customers", 0))
    metric_cols[3].metric("Avg TW1 minutes", metrics.get("Avg TW1 minutes", 0))
    metric_cols[4].metric("Earliest open", metrics.get("Earliest open", "n/a"))
    metric_cols[5].metric("Latest close", metrics.get("Latest close", "n/a"))

    st.subheader(f"Customer {co_cust_nbr}")
    st.caption(
        f"{len(customer_df)} committed slot row(s) across {customer_df['route_id'].astype(str).nunique()} route(s)"
    )

    timeline_col, dist_col = st.columns([1.8, 1])
    with timeline_col:
        st.plotly_chart(
            build_customer_timeline(customer_df, customer_label=str(co_cust_nbr)),
            use_container_width=True,
        )
    with dist_col:
        st.plotly_chart(
            build_tier_distribution(customer_df, title="Tier mix in customer slots"),
            use_container_width=True,
        )

    st.subheader("Committed TW1 rows")
    table_cols = [col for col in DISPLAY_COLUMNS if col in customer_df.columns]
    st.dataframe(customer_df[table_cols], use_container_width=True, hide_index=True)

    with st.expander("All columns"):
        st.dataframe(customer_df, use_container_width=True, hide_index=True)


def _render_route_view(committed_df: pd.DataFrame) -> None:
    routes = list_routes(committed_df)
    if routes.empty:
        st.warning("No routes found in committed TW1 data.")
        return

    all_tiers = sorted(committed_df["cust_tier"].fillna("Other").astype(str).unique().tolist())

    st.markdown("#### Route controls")
    route_col1, route_col2, route_col3 = st.columns([1.6, 1, 1.2])
    with route_col1:
        route_label = st.selectbox("Select route", options=routes["label"].tolist(), index=0)
        route_row = routes.loc[routes["label"] == route_label].iloc[0]
    with route_col2:
        enable_tier_filter = st.checkbox(
            "Filter by customer tier",
            value=False,
            help="Off = show delivery windows for all customers on the selected route",
        )
    with route_col3:
        selected_tiers = all_tiers
        if enable_tier_filter:
            selected_tiers = st.multiselect(
                "Customer tiers",
                options=all_tiers,
                default=all_tiers,
                help="Filter which customer slots appear in the route timeline",
            )

    active_tiers = (selected_tiers or None) if enable_tier_filter else None
    all_route_stops = filter_route(committed_df, route_id=str(route_row["route_id"]), cust_tiers=None)
    route_stops = filter_route(committed_df, route_id=str(route_row["route_id"]), cust_tiers=active_tiers)

    if route_stops.empty:
        st.warning("No rows match your route and tier filters.")
        return

    metrics = committed_summary_metrics(route_stops)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Rows", metrics.get("Rows", 0))
    metric_cols[1].metric("Routes", metrics.get("Routes", 0))
    metric_cols[2].metric("Customers", metrics.get("Customers", 0))
    metric_cols[3].metric("Avg TW1 minutes", metrics.get("Avg TW1 minutes", 0))
    metric_cols[4].metric("Earliest open", metrics.get("Earliest open", "n/a"))
    metric_cols[5].metric("Latest close", metrics.get("Latest close", "n/a"))

    st.subheader(route_label)
    if enable_tier_filter:
        tier_text = ", ".join(selected_tiers) if selected_tiers else "none"
        st.caption(
            f"{len(route_stops)} of {len(all_route_stops)} committed slot row(s) shown · tier filter: {tier_text}"
        )
    else:
        st.caption(f"{len(route_stops)} committed slot row(s) shown · all customer tiers")

    timeline_col, dist_col = st.columns([1.8, 1])
    with timeline_col:
        st.plotly_chart(
            build_route_timeline(route_stops, route_label=str(route_row["route_id"])),
            use_container_width=True,
        )
    with dist_col:
        st.plotly_chart(
            build_tier_distribution(route_stops, title="Tier mix on selected route"),
            use_container_width=True,
        )

    table_cols = [col for col in DISPLAY_COLUMNS if col in route_stops.columns]
    st.dataframe(route_stops[table_cols], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Committed TW1 Explorer",
        page_icon="🕒",
        layout="wide",
    )

    st.title("Committed TW1 slot explorer")
    st.caption("Visualize committed_tw1_slots_df from customer and route perspectives.")

    with st.sidebar:
        st.header("Data")
        source = st.selectbox(
            "Source",
            options=["auto", "cache", "sql", "sample"],
            index=["auto", "cache", "sql", "sample"].index(_parse_cli_source()),
            help="auto = cache then SQL; sample = offline demo rows",
        )
        if st.button("Reload data", use_container_width=True):
            _cached_load.clear()
            st.rerun()

    customer_tab, route_tab = st.tabs(["Customer view", "Route view"])

    try:
        committed_df = _cached_load(source)
    except Exception as exc:
        st.error(f"Could not load committed TW1 data: {exc}")
        st.info(
            "Run smart_assignment/data_prep/prep_delivery_data.py to populate cache, "
            "or restart with --sample for demo data."
        )
        return

    with customer_tab:
        _render_customer_view(committed_df)

    with route_tab:
        _render_route_view(committed_df)


if __name__ == "__main__":
    main()
