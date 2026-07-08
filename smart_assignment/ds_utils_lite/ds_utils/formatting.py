"""Helps with consistent formatting for DataFrames."""

import pandas as pd


class ColumnFormatter:
    """Formats known columns in a pd.DataFrame."""

    @classmethod
    def format(cls, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            return df
        df = cls.format_standard_columns(df)
        df = cls.format_custom_columns(df)
        return df

    @classmethod
    def format_custom_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @classmethod
    def format_standard_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        if "co_nbr" in df.columns:
            df = cls.format_co_nbr_column(df)
        if "itm_nbr" in df.columns:
            df = cls.format_itm_nbr_column(df)
        if "cust_nbr" in df.columns:
            df = cls.format_cust_nbr_column(df)
        if "taxon_id" in df.columns:
            df = cls.format_taxon_id_column(df)
        if "oblig_dt" in df.columns:
            df = cls.format_dt_column(df, "oblig_dt")
        if "day_dt" in df.columns:
            df = cls.format_dt_column(df, "day_dt")
        return df

    @staticmethod
    def format_taxon_id_column(df, col="taxon_id"):
        indexer = df[col].notna()
        df.loc[indexer, col] = df.loc[indexer, col].astype(str).str.replace("-", "")
        df.loc[indexer, col] = df.loc[indexer, col].str.zfill(8)
        return df

    @staticmethod
    def format_zfill_str_column(df: [pd.DataFrame, pd.Series], col: str, fill_to: int):
        if isinstance(df, pd.DataFrame):
            df[col] = df[col].astype(str).str.zfill(fill_to)
            assert (df[col].str.len() == fill_to).all(), f"Some entries in {col} are longer than {fill_to} digits."
            return df
        if isinstance(df, pd.Series):
            df = df.astype(str).str.zfill(fill_to)
            assert (df.str.len() == fill_to).all(), f"Some entries in {col} are longer than {fill_to} digits."
            return df
        raise ValueError("You must pass a pd.DataFrame to this function.")

    @classmethod
    def format_co_nbr_column(cls, df: pd.DataFrame):
        return cls.format_zfill_str_column(df, "co_nbr", 3)

    @classmethod
    def format_itm_nbr_column(cls, df: pd.DataFrame):
        return cls.format_zfill_str_column(df, "itm_nbr", 7)

    @classmethod
    def format_cust_nbr_column(cls, df: pd.DataFrame):
        return cls.format_zfill_str_column(df, "cust_nbr", 6)

    @staticmethod
    def format_dt_column(df: pd.DataFrame, col="oblig_dt") -> pd.DataFrame:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
        return df
