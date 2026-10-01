"""Regression checks for notebook feature calculations; no source-file changes."""

import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "notebooks/03_clean_raw_data.ipynb").read_text(encoding="utf-8"))


def namespace():
    scope = {}
    for cell in NOTEBOOK["cells"]:
        if set(cell.get("metadata", {}).get("tags", [])) & {"config", "definitions", "helpers"}:
            exec("".join(cell["source"]), scope)
    return scope


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.scope = namespace()

    def series(self, values, dates, name):
        return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name=name, dtype=float)

    def market_inputs(self, dates, values):
        return {name: self.series(values, dates, name) for name in self.scope["MARKET_FIELDS"]}

    def test_bloomberg_errors_excel_dates_and_conflicting_duplicates(self):
        report = []
        rows = [("x_Date", "x_Value"), (43831, 2), (43832, "#N/A N/A"),
                ("#N/A Field Not Applicable", None), (43833, "=BDH(...)")]
        result = self.scope["parse_field"](rows, "x", {}, report)
        self.assertEqual(result.index[0], pd.Timestamp("2020-01-01"))
        self.assertEqual(result.iloc[0], 2)
        self.assertTrue(result.iloc[1:].isna().all())
        self.assertEqual(report[0]["invalid_date_rows"], 1)
        with self.assertRaises(ValueError):
            self.scope["parse_field"]([("x_Date", "x_Value"), (43831, 2), (43831, 3)], "x", {}, [])

    def test_annual_change_signed_values_zero_base_and_missing_year(self):
        dates = ["2018-12-31", "2019-12-31", "2021-12-31", "2022-12-31"]
        fields = {field: self.series([100, 120, 200, 0], dates, field)
                  for field in self.scope["ANNUAL_TREND_METHOD"]}
        fields["ebit"] = self.series([-10, -5, 10, 12], dates, "ebit")
        annual = self.scope["annual_history"](fields)
        self.assertAlmostEqual(annual.loc["2019-12-31", "total_assets_pct_change"], .2)
        self.assertTrue(pd.isna(annual.loc["2021-12-31", "total_assets_pct_change"]))
        self.assertEqual(annual.loc["2019-12-31", "ebit_absolute_change"], 5)
        self.assertEqual(annual.loc["2022-12-31", "total_assets_pct_change"], -1)
        self.assertTrue(np.isnan(self.scope["safe_divide"](2, 0)))
        self.assertEqual(self.scope["safe_divide"](2, -4), -.5)

    def test_monthly_grid_does_not_skip_missing_months(self):
        fields = self.market_inputs(["2020-01-31", "2020-03-31", "2020-04-30"], [100, 121, 133.1])
        frame = self.scope["monthly_history"](fields, "2020-01-01", "2020-04-30")
        self.assertEqual(len(frame), 4)
        self.assertTrue(pd.isna(frame.loc["2020-02", "adjusted_price"]))
        self.assertTrue(pd.isna(frame.loc["2020-03", "monthly_return"]))
        self.assertAlmostEqual(frame.loc["2020-04", "monthly_return"], .1)
        self.assertTrue(frame["volatility_3m"].isna().all())

    def test_market_trends_returns_rolling_statistics_and_drawdown(self):
        dates = pd.date_range("2018-01-31", periods=40, freq="ME")
        values = 100 * np.power(1.01, np.arange(40))
        frame = self.scope["monthly_history"](self.market_inputs(dates, values), dates[0], dates[-1])
        self.assertAlmostEqual(frame.iloc[-1]["adjusted_price_pct_change_3y"], 1.01**36 - 1)
        self.assertAlmostEqual(frame.iloc[-1]["return_12m"], 1.01**12 - 1)
        self.assertAlmostEqual(frame.iloc[-1]["volatility_12m"], 0)
        self.assertAlmostEqual(frame.iloc[-1]["maximum_drawdown_12m"], 0)
        self.assertTrue(frame["maximum_drawdown_12m"].iloc[:22].isna().all())
        self.assertAlmostEqual(frame.iloc[-1]["average_volume_12m"], values[-12:].mean())

    def test_requested_financial_formulas(self):
        values = {field: 2.0 for field in self.scope["ANNUAL_TREND_METHOD"]}
        values.update(total_assets=100, total_debt=30, working_capital=10,
                      market_cap=200, cash_short_term_investments=20, ebitda=7)
        formulas = self.scope["DERIVED_FINANCIAL_FORMULAS"]
        calculate = self.scope["financial_formula"]
        self.assertEqual(calculate(formulas["working_capital_to_assets"], values), .1)
        self.assertEqual(calculate(formulas["debt_to_assets"], values), .3)
        self.assertEqual(calculate(formulas["enterprise_value_to_assets"], values), 2.1)
        self.assertEqual(calculate(formulas["enterprise_value_to_ebitda"], values), 30)
        with self.assertRaises(ValueError):
            calculate("__import__('os')", values)

    def test_macro_alignment_no_future_values_and_missing_inversion(self):
        fields = {field: self.series([], [], field) for field in self.scope["MACRO_FIELDS"]}
        fields["real_gdp_growth"] = self.series([2, 99], ["2020-03-31", "2020-06-30"], "real_gdp_growth")
        fields["vix"] = self.series([10, 12], ["2020-03-31", "2020-04-30"], "vix")
        frame, provenance = self.scope["macro_monthly_history"](fields, "2020-03-01", "2020-05-31")
        self.assertEqual(frame.loc["2020-05", "real_gdp_growth"], 2)
        self.assertEqual(frame.loc["2020-04", "vix_change"], 2)
        self.assertTrue(pd.isna(frame.loc["2020-05", "vix"]))
        self.assertTrue(frame["yield_curve_inverted"].isna().all())
        self.assertEqual(provenance.loc["2020-05", "real_gdp_growth"], pd.Timestamp("2020-03-31"))

    def test_snapshots_respect_cutoffs_publication_delay_and_company_boundaries(self):
        scope = self.scope
        rows = []
        annual, market = {}, {}
        for company_id, label, multiplier in [("SA_1", 1, 1), ("NB_1", 0, 10)]:
            rows.append({"company_id": company_id, "company_type": "bankrupt" if label else "nonbankrupt",
                         "prediction_date": pd.Timestamp("2023-01-15"), "bankruptcy_label": label})
            fields = {field: self.series([10 * multiplier, 20 * multiplier, 30 * multiplier],
                      ["2020-12-31", "2021-12-31", "2022-12-31"], field)
                      for field in scope["ANNUAL_TREND_METHOD"]}
            annual[company_id] = scope["annual_history"](fields)
            dates = pd.date_range("2017-01-31", "2023-01-31", freq="ME")
            market[company_id] = scope["monthly_history"](
                self.market_inputs(dates, np.full(len(dates), multiplier)), dates[0], dates[-1])
        metadata = pd.DataFrame(rows).set_index("company_id", drop=False)
        macro_fields = {field: self.series([], [], field) for field in scope["MACRO_FIELDS"]}
        macro, _ = scope["macro_monthly_history"](macro_fields, "2017-01-01", "2023-01-31")
        scope.update(metadata=metadata, annual_by_company=annual, market_by_company=market,
                     macro_monthly=macro, display=lambda _: None, ANNUAL_PUBLICATION_DELAY_DAYS=90)
        source = next("".join(c["source"]) for c in NOTEBOOK["cells"]
                      if "snapshots" in c.get("metadata", {}).get("tags", []))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, scope)
        result = scope["final_company_dataset"].set_index("company_id")
        self.assertEqual(result.at["SA_1", "total_assets_lag_1y"], 20)
        self.assertEqual(result.at["NB_1", "total_assets_lag_1y"], 200)
        self.assertEqual(result.at["SA_1", "total_assets_lag_2y"], 10)
        self.assertEqual(result.at["SA_1", "market_month_lag_1y"], pd.Timestamp("2022-12-31"))
        self.assertEqual(result.at["SA_1", "bankruptcy_label"], 1)
        self.assertEqual(result.at["NB_1", "bankruptcy_label"], 0)


if __name__ == "__main__":
    unittest.main()
