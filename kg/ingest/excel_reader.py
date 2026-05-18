"""Read an Excel file and return raw column metadata dicts.

This is the ingestion equivalent of excel_schema.py — same logic,
no JSON files written or expected.
"""
from __future__ import annotations

import pandas as pd


def _detect_header_row(df_raw: pd.DataFrame, max_scan: int = 30) -> int:
    max_scan = min(max_scan, len(df_raw) - 1)
    for i in range(max_scan):
        next_row = df_raw.iloc[i + 1].dropna()
        if len(next_row) == 0:
            continue
        numeric_ratio = sum(
            1 for v in next_row if isinstance(v, (int, float)) and not isinstance(v, bool)
        ) / len(next_row)
        if numeric_ratio >= 0.5:
            return i
    best_row, best_score = 0, -1
    for i in range(max_scan + 1):
        row = df_raw.iloc[i].dropna()
        if len(row) == 0:
            continue
        str_ratio = sum(1 for v in row if isinstance(v, str)) / len(row)
        if str_ratio > best_score:
            best_score = str_ratio
            best_row = i
    return best_row


def read_excel_schema(filepath: str, sheet_name: str | None = None) -> dict[str, dict]:
    """
    Returns:
        {
          sheet_name: {
            "row_count": int,
            "col_count": int,
            "detected_header_row": int,
            "columns": { col_name: { dtype, null_count, null_rate, n_unique,
                                     is_unique, min?, max?, mean?, std?,
                                     min_date?, max_date?, min_length?,
                                     max_length?, likely_datetime?, categories? } }
          }
        }
    """
    xls = pd.ExcelFile(filepath)
    sheets = xls.sheet_names if sheet_name is None else [sheet_name]
    schema: dict[str, dict] = {}

    for sheet in sheets:
        df_raw = pd.read_excel(filepath, sheet_name=sheet, header=None, nrows=40)
        header_row = _detect_header_row(df_raw)
        df = pd.read_excel(filepath, sheet_name=sheet, header=header_row)
        df = df.dropna(axis=1, how="all")

        n_rows = len(df)
        columns: dict[str, dict] = {}

        for col in df.columns:
            series = df[col].dropna()
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            null_rate = round(null_count / n_rows, 4) if n_rows > 0 else 0.0
            unique_vals = series.unique()
            n_unique = int(len(unique_vals))

            col_info: dict = {
                "dtype": dtype,
                "null_count": null_count,
                "null_rate": null_rate,
                "n_unique": n_unique,
                "is_unique": bool(n_unique == len(series) and len(series) > 0),
            }

            if pd.api.types.is_numeric_dtype(series):
                col_info["min"] = float(series.min())
                col_info["max"] = float(series.max())
                col_info["mean"] = round(float(series.mean()), 4)
                col_info["std"] = round(float(series.std()), 4)
            elif pd.api.types.is_datetime64_any_dtype(series):
                col_info["min_date"] = str(series.min())
                col_info["max_date"] = str(series.max())
            elif dtype == "object":
                str_series = series.astype(str)
                col_info["min_length"] = int(str_series.str.len().min())
                col_info["max_length"] = int(str_series.str.len().max())
                try:
                    parsed = pd.to_datetime(series, infer_datetime_format=True)
                    col_info["likely_datetime"] = True
                    col_info["min_date"] = str(parsed.min())
                    col_info["max_date"] = str(parsed.max())
                except Exception:
                    col_info["likely_datetime"] = False
                if n_unique <= 5:
                    col_info["categories"] = sorted([str(v) for v in unique_vals.tolist()])

            columns[str(col)] = col_info

        schema[sheet] = {
            "row_count": n_rows,
            "col_count": len(df.columns),
            "detected_header_row": header_row,
            "columns": columns,
        }

    return schema
