"""Load company-event data and score historical records with the saved model."""

from pathlib import Path

import joblib
import pandas as pd


IDENTIFIER_COLUMNS = [
    "company_id",
    "metadata_ticker",
    "metadata_name",
    "prediction_date",
    "bankruptcy_date",
    "target",
    "sector",
    "industry",
]

METRIC_GROUPS = {
    "Financial statements": {
        "Total assets": "original_total_assets",
        "Total liabilities": "original_total_liabilities",
        "Total debt": "original_total_debt",
        "Cash and short-term investments": "original_cash_short_term_investments",
        "Revenue": "original_revenue",
        "EBITDA": "original_ebitda",
        "Net income": "original_net_income",
        "Shareholders' equity": "original_shareholders_equity",
    },
    "Financial ratios": {
        "Debt / assets": "derived_debt_to_assets",
        "Current ratio": "derived_current_ratio",
        "Cash / assets": "derived_cash_to_assets",
        "Interest coverage": "derived_interest_coverage",
    },
    "Market and macro": {
        "Market capitalization": "original_market_cap",
        "12-month stock return": "market_stock_return_12m",
        "12-month volatility": "market_volatility_12m",
        "12-month maximum drawdown": "market_maximum_drawdown_12m",
        "Inflation": "macro_inflation_level_at_prediction_date",
        "Unemployment rate": "macro_unemployment_rate_level_at_prediction_date",
        "Federal funds rate": "macro_federal_funds_rate_level_at_prediction_date",
        "Credit spread": "macro_credit_spread_level_at_prediction_date",
    },
}

KEY_TREND_GROUPS = {
    "Asset growth": "trend_total_assets_growth",
    "Debt growth": "trend_total_debt_growth",
    "Revenue growth": "trend_revenue_growth",
    "Net income growth": "trend_net_income_growth",
    "Cash / assets change": "trend_cash_to_assets_change",
    "Debt / assets change": "trend_debt_to_assets_change",
    "EBITDA margin change": "trend_ebitda_margin_change",
    "Interest coverage change": "trend_interest_coverage_change",
    "Operating cash flow / assets change": "trend_operating_cash_flow_to_assets_change",
}


def load_model(path: str | Path):
    """Load the trusted fitted Elastic Net classifier."""
    model = joblib.load(path)
    if not hasattr(model, "predict_proba") or not hasattr(model, "feature_names_in_"):
        raise TypeError("The saved artifact is not a compatible fitted classifier.")
    return model


def load_company_events(path: str | Path, feature_columns):
    """Read only company descriptors, displayed metrics, and model inputs."""
    feature_columns = list(feature_columns)
    available_columns = pd.read_csv(path, nrows=0).columns
    trend_columns = [
        column for column in available_columns if column.startswith("trend_")
    ]
    metric_columns = [
        column for group in METRIC_GROUPS.values() for column in group.values()
    ]
    columns = list(
        dict.fromkeys(
            IDENTIFIER_COLUMNS + metric_columns + feature_columns + trend_columns
        )
    )
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"], errors="coerce")
    frame["bankruptcy_date"] = pd.to_datetime(frame["bankruptcy_date"], errors="coerce")
    frame = frame.dropna(subset=["company_id", "prediction_date"])
    frame["company_id"] = frame["company_id"].astype(str)
    return frame.sort_values(["metadata_name", "prediction_date"]).reset_index(
        drop=True
    )


def score_event(model, event):
    """Return the relative bankruptcy risk score for one historical event row."""
    features = list(model.feature_names_in_)
    row = pd.DataFrame([event[features].to_dict()])
    class_index = list(model.classes_).index(1)
    return float(model.predict_proba(row)[0, class_index])


def test_metrics(root: str | Path):
    """Read stored final-test metrics when the report outputs are available."""
    reports = Path(root) / "reports"
    elastic_path = reports / "05_elastic_net_reporting" / "elastic_net_metrics.csv"
    forest_path = reports / "06_random_forest_reporting" / "random_forest_metrics.csv"
    if not elastic_path.is_file() or not forest_path.is_file():
        return None

    elastic = pd.read_csv(elastic_path)
    forest = pd.read_csv(forest_path)
    elastic = elastic.loc[elastic["split"].eq("final_test")]
    forest = forest.loc[forest["model"].eq("Random Forest final test")]
    if elastic.empty or forest.empty:
        return None
    return elastic.iloc[0], forest.iloc[0]
