from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
            "constant": ["1", "1", "1", "1"]
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 5
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_has_constant_columns():
    df = _sample_df()
    const_df = compute_quality_flags(summarize_dataset(df), missing_table(df))["has_constant_columns"]
    assert const_df == 1

def test_has_suspicious_id_duplicates():
    df = _sample_df()
    duplicated_df = compute_quality_flags(summarize_dataset(df), missing_table(df))["has_suspicious_id_duplicates"]
    assert duplicated_df == 0

def test_flags():
    df = compute_quality_flags(summarize_dataset(_sample_df()), missing_table(_sample_df()))

    assert df["too_few_rows"] == 1
    assert df["too_many_columns"] == 0
    assert df["max_missing_share"] == 0.25
    assert df["too_many_missing"] == 0
    assert df["has_constant_columns"] == 1
    assert df["has_suspicious_id_duplicates"] == 0

    assert df["quality_score"] >= 0.5 and df["quality_score"] <= 0.8

