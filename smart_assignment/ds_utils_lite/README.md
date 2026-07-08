# ds-utils-lite

Minimal SQL access layer extracted from `ds_utils`. Enough to run
`smart_assignment/data_prep/prep_dlvry_tw_data.py` without the full ds_utils
dependency tree.

## Install

```bash
pip install -e smart_assignment/ds_utils_lite
```

Optional extras:

- `pip install -e "smart_assignment/ds_utils_lite[aws]"` — AWS Parameter Store credentials
- `pip install -e "smart_assignment/ds_utils_lite[mssql]"` — SQL Server via pyodbc

## Use with data prep scripts

This package installs as `ds_utils`, so existing scripts keep working:

```python
from ds_utils.deploy.mode import Mode
from ds_utils.sql import SQLAccess

mode = Mode(Mode.LOCAL)
sql = SQLAccess(mode)
df = pull_route_data(sql)
```

Set `DATABASE_CREDENTIALS_LOCATION` to your credentials JSON (see
`database_creds_template.json.example`).

If both the full `smart_assignment/ds_utils` folder and this package are on
`PYTHONPATH`, prefer the installed lite package:

```bash
pip install -e smart_assignment/ds_utils_lite
```
