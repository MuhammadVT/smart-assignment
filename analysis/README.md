# Analysis

Exploratory dashboards and notebooks for EAT Smart Assignment data prep outputs.

## Committed TW1 slot explorer

Interactive dashboard for committed TW1 slots derived from delivery-window facts
(`committed_tw1_slots_df`) with customer and route perspectives.

### Setup

```powershell
uv sync --extra analysis
```

### Run

**From cache or live SQL** (requires `data/dev/dlvr_window.parquet` or database creds):

```powershell
uv run streamlit run analysis/dlvr_window/app.py
```

**With bundled sample data** (no database needed):

```powershell
uv run streamlit run analysis/dlvr_window/app.py -- --sample
```

The app has two tabs:
- **Customer view** — committed TW1 slots for one `co_cust_nbr` across routes
- **Route view** — committed TW1 slots for one `route_id` across customers; filter by `cust_tier`

### Reuse in a notebook

```python
from analysis.dlvr_window.data import load_committed_tw1_slots_df, filter_customer
from analysis.dlvr_window.charts import build_customer_timeline

df = load_committed_tw1_slots_df(source="cache")  # or "sql", "auto", "sample"
cust = filter_customer(df, "067-123456")
fig = build_customer_timeline(cust, customer_label="067-123456")
fig.show()
```

### Layout

```
analysis/
  README.md
  run_dlvr_window_dashboard.py   # convenience launcher
  dlvr_window/
    data.py      # committed TW1 load / filter helpers
    charts.py    # plotly figures for customer/route views
    app.py       # Streamlit UI
```
