"""Central project configuration for Bloomberg data and field selections."""

ANNUAL_FIELDS = {'total_assets': 'BS_TOT_ASSET',
 'total_liabilities': 'BS_TOT_LIAB2',
 'total_debt': 'SHORT_AND_LONG_TERM_DEBT',
 'short_term_debt': 'BS_ST_BORROW',
 'long_term_debt': 'BS_LT_BORROW',
 'cash_short_term_investments': 'BS_CASH_NEAR_CASH_ITEM',
 'current_assets': 'BS_CUR_ASSET_REPORT',
 'current_liabilities': 'BS_CUR_LIAB',
 'shareholders_equity': 'BS_SHAREHOLDER_EQY_ENDING_PERIOD',
 'accounts_receivable': 'BS_ACCT_NOTE_RCV',
 'inventory': 'BS_INVENTORIES',
 'working_capital': 'WORKING_CAPITAL',
 'revenue': 'SALES_REV_TURN',
 'ebit': 'EBIT',
 'ebitda': 'EBITDA',
 'net_income': 'NET_INCOME',
 'interest_expense': 'IS_INT_EXPENSE',
 'depreciation_amortization': 'IS_DEPR_EXP',
 'operating_cash_flow': 'CF_CASH_FROM_OPER',
 'capital_expenditures': 'CAPITAL_EXPEND',
 'free_cash_flow': 'CF_FREE_CASH_FLOW'}

MARKET_FIELDS = {'adjusted_price': 'PX_LAST',
 'market_cap': 'CUR_MKT_CAP',
 'shares_outstanding': 'EQY_SH_OUT',
 'volume': 'PX_VOLUME',
 'turnover': 'TURNOVER'}

REFERENCE_FIELDS = {'bloomberg_sector': 'BICS_LEVEL_1_SECTOR_NAME', 'bloomberg_industry': 'BICS_LEVEL_3_INDUSTRY_NAME'}

MACRO_FIELDS = {'inflation_cpi_yoy': {'security': 'CPI YOY Index', 'field': 'PX_LAST', 'periodicity': 'M'},
 'real_gdp_growth': {'security': 'GDP CQOQ Index', 'field': 'PX_LAST', 'periodicity': 'Q'},
 'unemployment_rate': {'security': 'USURTOT Index', 'field': 'PX_LAST', 'periodicity': 'M'},
 'effective_federal_funds_rate': {'security': 'FEDL01 Index',
                                  'field': 'PX_LAST',
                                  'periodicity': 'D'},
 'treasury_3m_rate': {'security': 'USGG3M Index', 'field': 'PX_LAST', 'periodicity': 'D'},
 'treasury_10y_rate': {'security': 'USGG10YR Index', 'field': 'PX_LAST', 'periodicity': 'D'},
 'credit_spread': {'security': 'LUACTRUU Index', 'field': 'PX_LAST', 'periodicity': 'M'},
 'vix': {'security': 'VIX Index', 'field': 'PX_LAST', 'periodicity': 'D'}}

FINANCIAL_FIELDS = set(ANNUAL_FIELDS)
MARKET_FIELD_NAMES = set(MARKET_FIELDS)

NONNEGATIVE_FIELDS = {'accounts_receivable',
 'cash_short_term_investments',
 'current_assets',
 'current_liabilities',
 'inventory',
 'long_term_debt',
 'market_cap',
 'shares_outstanding',
 'short_term_debt',
 'total_assets',
 'total_debt',
 'total_liabilities',
 'turnover',
 'vix',
 'volume'}

ERROR_PATTERN = '^(?:#|=|REQUESTING\\b|RETRIEVING\\b|PENDING\\b)'
MISSING_TOKENS = {'', 'NA', 'N/A', 'NONE', 'NAN', 'NAT', 'NULL', '-', '--'}

MAJOR_FINANCIAL_FIELDS = ['total_assets',
 'total_liabilities',
 'total_debt',
 'cash_short_term_investments',
 'current_assets',
 'current_liabilities',
 'revenue',
 'ebitda',
 'net_income',
 'shareholders_equity',
 'operating_cash_flow']

DISTRIBUTION_FIELDS = ['total_assets',
 'total_liabilities',
 'total_debt',
 'revenue',
 'ebitda',
 'net_income',
 'cash_short_term_investments',
 'shareholders_equity',
 'operating_cash_flow',
 'market_cap']

MARKET_FEATURE_SOURCE_FIELDS = ['adjusted_price', 'market_cap', 'volume', 'turnover', 'bid_ask_spread']


# Modeling feature schemas shared across feature-generation and merge notebooks.
FINANCIAL_COLUMNS = list(ANNUAL_FIELDS)
MARKET_COLUMNS = list(MARKET_FIELDS)
ORIGINAL_COLUMNS = FINANCIAL_COLUMNS + MARKET_COLUMNS

RATIO_FORMULAS = {
    "debt_to_assets": ("total_debt", "total_assets"),
    "liabilities_to_assets": ("total_liabilities", "total_assets"),
    "current_ratio": ("current_assets", "current_liabilities"),
    "cash_to_assets": ("cash_short_term_investments", "total_assets"),
    "debt_to_equity": ("total_debt", "shareholders_equity"),
    "liabilities_to_equity": ("total_liabilities", "shareholders_equity"),
    "ebit_margin": ("ebit", "revenue"),
    "ebitda_margin": ("ebitda", "revenue"),
    "net_income_margin": ("net_income", "revenue"),
    "operating_cash_flow_margin": ("operating_cash_flow", "revenue"),
    "interest_coverage": ("ebit", "interest_expense"),
    "operating_cash_flow_to_debt": ("operating_cash_flow", "total_debt"),
    "free_cash_flow_to_debt": ("free_cash_flow", "total_debt"),
    "working_capital_to_assets": ("working_capital", "total_assets"),
    "capex_to_revenue": ("capital_expenditures", "revenue"),
    "market_to_book": ("market_cap", "shareholders_equity"),
}
RATIO_FORMULAS.update({
    "ebit_to_assets": ("ebit", "total_assets"),
    "ebitda_to_assets": ("ebitda", "total_assets"),
    "operating_cash_flow_to_assets": ("operating_cash_flow", "total_assets"),
})
RATIO_COLUMNS = list(RATIO_FORMULAS)

TREND_YEARS = (1, 2, 3)
TREND_CHANGE_TYPES = ("change", "growth")
TREND_SOURCE_COLUMNS = ORIGINAL_COLUMNS + RATIO_COLUMNS
TREND_FEATURE_COLUMNS = [
    f"{field}_{kind}_{year}y"
    for field in TREND_SOURCE_COLUMNS
    for year in TREND_YEARS
    for kind in TREND_CHANGE_TYPES
]
TREND_FEATURE_COLUMNS += [f"{field}_slope_3y" for field in TREND_SOURCE_COLUMNS]
TREND_FEATURE_COLUMNS += ["asset_growth_1y", "debt_growth_1y"]

MARKET_HORIZONS_MONTHS = (12, 24, 36)
MARKET_FEATURE_COLUMNS = [
    f"{kind}_{months}m"
    for kind in ("return", "volatility", "maximum_drawdown")
    for months in MARKET_HORIZONS_MONTHS
]
MARKET_FEATURE_COLUMNS += [
    "market_cap_change_12m",
    "volume_change_12m",
    "turnover_change_12m",
    "bid_ask_spread_average_12m",
    "stock_return_12m",
]

SIGNED_LOG_FIELDS = [
    "working_capital", "ebit", "ebitda", "net_income", "interest_expense",
    "operating_cash_flow", "capital_expenditures", "free_cash_flow",
]

# Final prefixed column names in modeling_dataset_unprocessed.csv.
COMPACT_FEATURE_COLUMNS = [
    "original_log_total_assets", "original_log_market_cap",
    "derived_debt_to_assets", "derived_liabilities_to_assets", "derived_cash_to_assets",
    "derived_current_ratio", "derived_ebit_to_assets", "derived_ebitda_to_assets",
    "derived_operating_cash_flow_to_assets", "derived_interest_coverage",
    "derived_free_cash_flow_to_debt", "derived_market_to_book",
    "trend_revenue_growth_1y", "trend_asset_growth_1y", "trend_debt_growth_1y",
    "market_stock_return_12m", "market_volatility_12m", "market_maximum_drawdown_12m",
    "macro_inflation", "macro_unemployment", "macro_interest_rate", "macro_credit_spread",
]

# Split membership columns shared by split export and preprocessing.
ID_COLUMNS = [
    "company_id",
    "prediction_date",
    "metadata_event_id",
    "target",
    "prediction_year",
    "split",
    "company_group_id",
    "metadata_outcome_end_date",
]
