"""Plotly charts for committed TW1 slot exploration."""

from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import plotly.graph_objects as go

KNOWN_TIER_COLORS = {
    "Perks": "#8b5cf6",
    "Non-Perks": "#64748b",
    "4": "#f97316",
    "5": "#06b6d4",
    "Other": "#a8a29e",
}

TIER_FALLBACK_PALETTE = ["#e11d48", "#0d9488", "#ca8a04", "#4f46e5", "#be185d", "#15803d"]


def _tier_color_map(tiers: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    fallback_idx = 0
    for tier in sorted({str(t) for t in tiers}):
        if tier in KNOWN_TIER_COLORS:
            colors[tier] = KNOWN_TIER_COLORS[tier]
        else:
            colors[tier] = TIER_FALLBACK_PALETTE[fallback_idx % len(TIER_FALLBACK_PALETTE)]
            fallback_idx += 1
    return colors


def _to_minutes(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.hour * 60 + value.minute + value.second / 60.0
    if isinstance(value, time):
        return value.hour * 60 + value.minute + value.second / 60.0
    text = str(value).strip()
    if not text or text.lower() in {"none", "nat", "nan"}:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.hour * 60 + parsed.minute + parsed.second / 60.0
        except ValueError:
            continue
    return None


def _minutes_label(minutes: float) -> str:
    whole = int(minutes // 60)
    mins = int(minutes % 60)
    suffix = "AM" if whole < 12 else "PM"
    hour = whole % 12 or 12
    return f"{hour}:{mins:02d} {suffix}"


def _minutes_label_24h(minutes: float) -> str:
    whole = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{whole:02d}:{mins:02d}"


def _slot_segments(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    work = df.copy()
    work["start_min"] = work["tw1opentime"].apply(_to_minutes)
    work["end_min"] = work["tw1closetime"].apply(_to_minutes)
    seg = work[(work["start_min"].notna()) & (work["end_min"].notna())].copy()
    seg = seg[seg["end_min"] > seg["start_min"]].copy()
    seg["duration"] = seg["end_min"] - seg["start_min"]
    seg["label"] = seg[label_col].astype(str)
    return seg


def _timeline_layout(
    fig: go.Figure,
    title: str,
    height: int,
    max_minute: float,
    *,
    use_24h: bool = False,
) -> go.Figure:
    ticks = list(range(0, int(max(max_minute, 18 * 60)) + 60, 60))
    label_fn = _minutes_label_24h if use_24h else _minutes_label
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis={
            "title": "Time of day",
            "tickmode": "array",
            "tickvals": ticks,
            "ticktext": [label_fn(t) for t in ticks],
            "range": [0, max(ticks) + 30],
        },
        height=height,
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
    )
    return fig


def build_customer_timeline(customer_df: pd.DataFrame, customer_label: str) -> go.Figure:
    """Committed TW1 slots on each route for a selected customer."""
    if customer_df.empty:
        return go.Figure().update_layout(title="No committed TW1 slots for selected customer")

    seg = _slot_segments(customer_df, label_col="route_id")
    if seg.empty:
        return go.Figure().update_layout(title="No valid TW1 times for selected customer")

    seg = seg.sort_values("route_id")
    y = seg["label"].astype(str)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="TW1",
            y=y,
            x=seg["duration"],
            base=seg["start_min"],
            orientation="h",
            marker={"color": "#2563eb", "opacity": 0.9},
            customdata=[
                [_minutes_label(start), _minutes_label(end), tier]
                for start, end, tier in zip(seg["start_min"], seg["end_min"], seg["cust_tier"], strict=True)
            ],
            hovertemplate=(
                "Route: %{y}<br>TW1: %{customdata[0]} - %{customdata[1]}<br>Tier: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(yaxis={"title": "Route"})
    max_minute = float(seg["end_min"].max())
    return _timeline_layout(fig, f"Committed TW1 by route - {customer_label}", max(340, 70 * len(seg)), max_minute)


def build_route_timeline(route_df: pd.DataFrame, route_label: str) -> go.Figure:
    """Committed TW1 slots by customer for a selected route."""
    if route_df.empty:
        return go.Figure().update_layout(title="No committed TW1 slots for selected route")

    seg = _slot_segments(route_df, label_col="co_cust_nbr")
    if seg.empty:
        return go.Figure().update_layout(title="No valid TW1 times for selected route")

    seg = seg.sort_values(["start_min", "label"], ascending=[True, True])
    customer_order = list(dict.fromkeys(seg["label"].astype(str).tolist()))

    tier_colors = _tier_color_map(seg["cust_tier"].astype(str).tolist())
    fig = go.Figure()
    for tier, color in tier_colors.items():
        part = seg[seg["cust_tier"].astype(str) == tier]
        if part.empty:
            continue
        fig.add_trace(
            go.Bar(
                name=tier,
                y=part["label"].astype(str),
                x=part["duration"],
                base=part["start_min"],
                orientation="h",
                marker={"color": color, "opacity": 0.92},
                customdata=[
                    [_minutes_label_24h(start), _minutes_label_24h(end)]
                    for start, end in zip(part["start_min"], part["end_min"], strict=True)
                ],
                hovertemplate=(
                    "Customer: %{y}<br>TW1: %{customdata[0]} - %{customdata[1]}<br>Tier: "
                    + tier
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        yaxis={
            "title": "Customer",
            "categoryorder": "array",
            "categoryarray": customer_order[::-1],
        },
        legend={"title": "Customer tier"},
    )
    max_minute = float(seg["end_min"].max())
    return _timeline_layout(
        fig,
        f"Committed TW1 by customer - route {route_label}",
        max(360, 70 * len(seg)),
        max_minute,
        use_24h=True,
    )


def build_tier_distribution(df: pd.DataFrame, title: str) -> go.Figure:
    """Simple count bar by customer tier."""
    if df.empty:
        return go.Figure().update_layout(title="No tier distribution data")

    counts = df["cust_tier"].fillna("Other").astype(str).value_counts().sort_index()
    colors = _tier_color_map(counts.index.tolist())
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker={"color": [colors[str(idx)] for idx in counts.index.tolist()]},
                hovertemplate="Tier: %{x}<br>Rows: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=title, xaxis={"title": "Customer tier"}, yaxis={"title": "Count"}, height=300)
    return fig


def committed_summary_metrics(df: pd.DataFrame) -> dict[str, str | int]:
    if df.empty:
        return {}

    starts = df["tw1opentime"].apply(_to_minutes).dropna()
    ends = df["tw1closetime"].apply(_to_minutes).dropna()
    duration = df.get("window_minutes", pd.Series(dtype=float)).fillna(0)

    earliest_open = _minutes_label(float(starts.min())) if not starts.empty else "n/a"
    latest_close = _minutes_label(float(ends.max())) if not ends.empty else "n/a"

    return {
        "Rows": int(len(df)),
        "Routes": int(df["route_id"].astype(str).nunique()),
        "Customers": int(df["co_cust_nbr"].astype(str).nunique()),
        "Avg TW1 minutes": int(round(float(duration.mean()))) if len(duration) else 0,
        "Earliest open": earliest_open,
        "Latest close": latest_close,
    }
