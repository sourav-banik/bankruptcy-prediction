"""Validate the modeling sample, matching design, data coverage, and leakage."""

import ast
import base64
import hashlib
import html
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils import canonical_security, project_paths, safe_divide  # noqa: E402
from project_config import (  # noqa: E402
    COMPACT_FEATURE_COLUMNS,
    FINANCIAL_COLUMNS as financial_columns,
    MARKET_COLUMNS as market_columns,
    MARKET_FEATURE_COLUMNS as market_feature_columns,
    NONNEGATIVE_FIELDS,
    ORIGINAL_COLUMNS as original_columns,
    RATIO_FORMULAS as ratio_formulas,
    TREND_FEATURE_COLUMNS as trend_columns,
)

PROCESSED = project_paths(ROOT).processed
REPORTS = project_paths(ROOT).reports / "02_modeling_data_validation"
EXPECTED_MATCH_RATIO = 2
MAX_CONTROL_REUSE = 3
KEYS = ["company_id", "prediction_date"]
CORE = [
    "company_id",
    "prediction_date",
    "target",
    "bankruptcy_date",
    "event_type",
    "sector",
    "industry",
    "prediction_year",
]
checks = []
evidence = {}
input_hashes = {}
notebook_source_transformer = TransformerManager()


def audited_csv(path, **kwargs):
    input_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, **kwargs)


def literal(filename, variable):
    path = ROOT / "notebooks" / filename
    notebook = json.loads(path.read_text(encoding="utf-8"))
    input_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = notebook_source_transformer.transform_cell("".join(cell["source"]))
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == variable for t in node.targets
            ):
                return ast.literal_eval(node.value)
    raise ValueError(f"Missing {variable} in {filename}")


# A failed check is recorded, rather than stopping before reports are generated.
def check(section, name, failures, checked, detail, sample=None):
    failures = int(failures)
    checks.append(
        {
            "section": section,
            "check": name,
            "status": "FAIL" if failures else "PASS",
            "checked": int(checked),
            "violations": failures,
            "detail": detail,
        }
    )
    if sample is not None and len(sample):
        evidence[name] = sample.head(20).copy()


def unverified(section, name, detail):
    checks.append(
        {
            "section": section,
            "check": name,
            "status": "UNVERIFIED",
            "checked": 0,
            "violations": np.nan,
            "detail": detail,
        }
    )


def parse_dates(frame, columns, section="structure"):
    converted = {}
    for column in columns:
        raw = frame[column]
        dates = pd.to_datetime(raw, errors="coerce").astype("datetime64[ns]")
        invalid = raw.notna() & dates.isna()
        check(
            section,
            f"Valid dates: {column}",
            invalid.sum(),
            len(frame),
            "Nonblank dates must parse.",
            frame.loc[invalid, [c for c in KEYS + [column] if c in frame]].loc[
                :, lambda x: ~x.columns.duplicated()
            ],
        )
        converted[column] = dates
    return pd.DataFrame(
        {c: converted.get(c, frame[c]) for c in frame}, index=frame.index
    )


model = audited_csv(
    PROCESSED / "modeling_dataset_unprocessed.csv", dtype={"company_id": "string"}
)
required = CORE + [
    "metadata_event_id",
    "metadata_outcome_end_date",
    "metadata_company_type",
    "metadata_financial_reference_date",
    "metadata_market_reference_date",
    "metadata_macro_reference_date",
    "metadata_matching_records_json",
    "metadata_matching_ratio_requested",
    "metadata_matching_pair_count_original",
    "metadata_matching_pair_count_retained",
    "metadata_ticker",
    "metadata_bloomberg_identifier",
    "original_market_cap",
    "original_total_assets",
]
absent = sorted(set(required) - set(model.columns))
if absent:
    raise ValueError(f"Cannot audit a dataset missing required columns: {absent}")
date_columns = ["prediction_date", "bankruptcy_date"] + [
    c for c in model if c.startswith("metadata_") and "_date" in c
]
model = parse_dates(model, date_columns)
availability = audited_csv(
    PROCESSED / "company_availability.csv", dtype={"company_id": "string"}
)
availability = parse_dates(
    availability, ["bankruptcy_date", "end_date", "last_market_date"], "source"
)
LEAD_DAYS = 90
OUTCOME_MONTHS = 12
VERIFIED_LABEL_END_DATE = pd.Timestamp("2022-12-31")
confirmed_screening_end = None
collection_end = availability.loc[
    availability.company_type.eq("nonbankrupt"), "end_date"
].min()
screening_end = VERIFIED_LABEL_END_DATE
positive = model.target.eq(1)
negative = model.target.eq(0)
feature_columns = [
    c
    for c in model
    if c.startswith(("original_", "derived_", "trend_", "market_", "macro_"))
]

# Check event identity, labels, filing dates, and membership in upstream event tables.
missing_identity = model[KEYS + ["metadata_event_id"]].isna().any(axis=1)
check(
    "structure",
    "Complete event identifiers",
    missing_identity.sum(),
    len(model),
    "Company/date/event ID are mandatory.",
)
duplicates = model.duplicated(KEYS, keep=False)
check(
    "structure",
    "One row per company-event",
    duplicates.sum(),
    len(model),
    "Duplicate company/prediction-date rows are forbidden.",
    model.loc[duplicates, CORE],
)
check(
    "structure",
    "Unique event IDs",
    model.metadata_event_id.duplicated(keep=False).sum(),
    len(model),
    "Event IDs must also be unique.",
)
check(
    "structure",
    "Binary target",
    (~model.target.isin([0, 1])).sum(),
    len(model),
    "Only 0 and 1 are allowed.",
)
check(
    "structure",
    "Prediction year agrees with date",
    (~model.prediction_year.eq(model.prediction_date.dt.year)).sum(),
    len(model),
    "Prediction year comes from each event date.",
)
expected_event_ids = (
    model.company_id + "@" + model.prediction_date.dt.strftime("%Y-%m-%d")
)
check(
    "structure",
    "Event ID matches company/date",
    (~model.metadata_event_id.eq(expected_event_ids)).sum(),
    len(model),
    "The event ID encodes the same company/date key.",
)
check(
    "structure",
    "Unique company metadata",
    availability.company_id.duplicated(keep=False).sum(),
    len(availability),
    "Company availability is one row per company.",
)
if availability.company_id.duplicated().any():
    raise ValueError("Ambiguous company metadata prevents reliable filing validation")
company_meta = (
    availability.set_index("company_id")
    .reindex(model.company_id)
    .reset_index(drop=True)
)
check(
    "structure",
    "Company metadata membership",
    (~model.company_id.isin(availability.company_id)).sum(),
    len(model),
    "Every company must have upstream metadata.",
)
expected_filing = company_meta.bankruptcy_date
filing_agrees = model.bankruptcy_date.eq(expected_filing) | (
    model.bankruptcy_date.isna() & expected_filing.isna()
)
check(
    "structure",
    "Bankruptcy dates agree with company registry",
    (~filing_agrees).sum(),
    len(model),
    "Missing filings remain missing; pseudo-events are not filing dates.",
)
check(
    "structure",
    "Positive filing dates and 90-day prediction lead",
    (
        ~(
            model.bankruptcy_date.notna()
            & model.prediction_date.eq(
                model.bankruptcy_date - pd.Timedelta(days=LEAD_DAYS)
            )
        )
    )
    .loc[positive]
    .sum(),
    positive.sum(),
    f"Positive prediction date must equal filing minus {LEAD_DAYS} days.",
)
role_agrees = (
    positive
    & model.metadata_company_type.eq("bankrupt")
    & model.event_type.eq("bankruptcy")
) | (
    negative
    & model.metadata_company_type.eq("nonbankrupt")
    & model.event_type.eq("nonbankruptcy_candidate")
)
check(
    "structure",
    "Target and event/company types agree",
    (~role_agrees).sum(),
    len(model),
    "Positive and negative event definitions remain distinct.",
)
for category in ["sector", "industry"]:
    equal = model[category].eq(company_meta[f"bloomberg_{category}"])
    check(
        "structure",
        f"Bloomberg {category} assigned",
        (~equal | model[category].isna()).sum(),
        len(model),
        "Use the Bloomberg company classification.",
    )

# Validate controls against the entire candidate-date table, not one common date.
event_tables = []
for name in ["bankruptcy_events", "nonbankruptcy_candidate_events"]:
    frame = audited_csv(PROCESSED / f"{name}.csv", dtype={"company_id": "string"})
    frame = parse_dates(frame, ["prediction_date", "outcome_end_date"], "source")
    event_tables.append(frame)
registry = pd.concat(event_tables, ignore_index=True)
check(
    "structure",
    "Unique source events",
    registry.duplicated(KEYS, keep=False).sum(),
    len(registry),
    "Event registry keys must be unique.",
)
if registry.duplicated(KEYS).any():
    raise ValueError("Ambiguous source event keys")
model_index = pd.MultiIndex.from_frame(model[KEYS])
registered = registry.set_index(KEYS).reindex(model_index).reset_index(drop=True)
for column in ["target", "event_type", "prediction_year"]:
    check(
        "structure",
        f"Source event agreement: {column}",
        (~model[column].eq(registered[column])).sum(),
        len(model),
        "Every event agrees with notebook 05 output.",
    )
check(
    "structure",
    "Source event outcome date",
    (~model.metadata_outcome_end_date.eq(registered.outcome_end_date)).sum(),
    len(model),
    "Outcome dates agree with the event registry.",
)
check(
    "outcome",
    "No non-bankrupt observation has an incomplete outcome window",
    (
        model.metadata_outcome_end_date.isna()
        | model.metadata_outcome_end_date.gt(VERIFIED_LABEL_END_DATE)
    )
    .loc[negative]
    .sum(),
    negative.sum(),
    f"Every zero label must be followed through {VERIFIED_LABEL_END_DATE:%Y-%m-%d}.",
)
registry_negative = registered.target.eq(0)
if "outcome_complete" in registered:
    registry_outcome_incomplete = ~registered.outcome_complete.fillna(False).astype(
        bool
    )
else:
    registry_outcome_incomplete = (
        registered.outcome_end_date.isna()
        | registered.outcome_end_date.gt(VERIFIED_LABEL_END_DATE)
    )
check(
    "outcome",
    "No non-bankrupt source event has an incomplete outcome window",
    registry_outcome_incomplete.loc[registry_negative].sum(),
    registry_negative.sum(),
    "The saved notebook 05 event registry must carry only completed negative outcomes.",
)
check(
    "outcome",
    "No bankrupt prediction date is after filing",
    (
        ~model.loc[positive, "prediction_date"].le(
            model.loc[positive, "bankruptcy_date"]
        )
    ).sum(),
    positive.sum(),
    "Positive prediction dates must precede or equal their filing dates.",
)
check(
    "structure",
    "Control pseudo-date is an eligible historical prediction date",
    (~model.prediction_date.isin(event_tables[0].prediction_date)).loc[negative].sum(),
    negative.sum(),
    "Every negative date comes from bankrupt-company prediction dates and is eligible for that company.",
)

# Missingness indicators and finite original/engineered values remain mechanical checks.
for column in original_columns:
    expected = model[f"original_{column}"].isna().astype(int)
    actual = model[f"missing_{column}"]
    check(
        "quality",
        f"Missingness indicator: {column}",
        (~actual.eq(expected)).sum(),
        len(model),
        "1 means missing; 0 means observed.",
    )
numeric_types = [
    c for c in feature_columns if not pd.api.types.is_numeric_dtype(model[c])
]
check(
    "quality",
    "Numeric predictor types",
    len(numeric_types),
    len(feature_columns),
    str(numeric_types),
)
numeric_predictors = [
    c for c in feature_columns if pd.api.types.is_numeric_dtype(model[c])
]
values = (
    model[numeric_predictors].to_numpy(dtype=float)
    if numeric_predictors
    else np.empty((len(model), 0))
)
check(
    "feature",
    "No infinite predictor values",
    np.isinf(values).sum(),
    values.size,
    "Missing is allowed; infinite numeric predictor values are forbidden.",
)
required_predictors = (
    [f"original_{column}" for column in original_columns]
    + [f"derived_{column}" for column in ratio_formulas]
    + [f"trend_{column}" for column in trend_columns]
    + [f"market_{column}" for column in market_feature_columns]
    + list(COMPACT_FEATURE_COLUMNS)
)
missing_required_predictors = sorted(set(required_predictors) - set(model.columns))
check(
    "feature",
    "Required predictor fields are present",
    len(missing_required_predictors),
    len(set(required_predictors)),
    str(missing_required_predictors),
)
zero_denominator_rows = []
for ratio, (_, denominator) in ratio_formulas.items():
    column = f"original_{denominator}"
    zero_count = int(model[column].eq(0).sum()) if column in model else 0
    nonmissing_zero_results = (
        int(model.loc[model[column].eq(0), f"derived_{ratio}"].notna().sum())
        if column in model and f"derived_{ratio}" in model
        else 0
    )
    zero_denominator_rows.append(
        {
            "ratio": ratio,
            "denominator": column,
            "zero_denominator_observations": zero_count,
            "nonmissing_ratio_when_denominator_zero": nonmissing_zero_results,
            "missing_denominator_observations": int(model[column].isna().sum())
            if column in model
            else np.nan,
        }
    )
    check(
        "feature",
        f"Zero denominator yields missing ratio: {ratio}",
        nonmissing_zero_results,
        zero_count,
        "Ratios with a zero denominator must remain missing, not infinite or finite.",
    )
zero_denominator_summary = pd.DataFrame(zero_denominator_rows)
negative_value_rows = []
for field in sorted(NONNEGATIVE_FIELDS):
    column = f"original_{field}" if f"original_{field}" in model else None
    if (
        column is None
        and field == "vix"
        and "macro_vix_level_at_prediction_date" in model
    ):
        column = "macro_vix_level_at_prediction_date"
    if column is not None:
        count = int(pd.to_numeric(model[column], errors="coerce").lt(0).sum())
        negative_value_rows.append({"field": column, "negative_observations": count})
        check(
            "feature",
            f"Nonnegative field has no negative values: {column}",
            count,
            len(model),
            "Negative raw values require upstream data-quality review.",
            model.loc[model[column].lt(0), KEYS + [column]],
        )
equity = pd.to_numeric(
    model.get("original_shareholders_equity", pd.Series(np.nan, index=model.index)),
    errors="coerce",
)
market_to_book = pd.to_numeric(
    model.get("derived_market_to_book", pd.Series(np.nan, index=model.index)),
    errors="coerce",
)
negative_equity_count = int(equity.lt(0).sum())
zero_equity_count = int(equity.eq(0).sum())
zero_equity_ratio_errors = int(market_to_book.loc[equity.eq(0)].notna().sum())
check(
    "feature",
    "Zero equity never produces market-to-book infinity/value",
    zero_equity_ratio_errors,
    zero_equity_count,
    "safe_divide must leave market-to-book missing when equity is zero.",
)
feature_diagnostics = pd.DataFrame(
    [
        {
            "diagnostic": "negative_shareholders_equity",
            "observations": negative_equity_count,
            "interpretation": "Reported for review; negative equity is economically valid.",
        },
        {
            "diagnostic": "zero_shareholders_equity",
            "observations": zero_equity_count,
            "interpretation": "Reported; zero denominators should yield missing ratios.",
        },
    ]
    + [
        {
            "diagnostic": f"negative_nonnegative_field:{row['field']}",
            "observations": row["negative_observations"],
            "interpretation": "Requires review against Bloomberg source values.",
        }
        for row in negative_value_rows
    ]
)
check(
    "quality",
    "Recognizable column groups",
    sum(
        not (
            c in CORE
            or c.startswith(
                (
                    "metadata_",
                    "original_",
                    "derived_",
                    "trend_",
                    "market_",
                    "macro_",
                    "missing_",
                )
            )
        )
        for c in model
    ),
    len(model.columns),
    "All noncore columns have one requested prefix.",
)

# Recover every matching pair, validate repeated copies, and recompute actual reuse.
pair_copies = {}
json_errors = []
record_fields = [
    "bankrupt_company_id",
    "control_company_id",
    "bankrupt_prediction_date",
    "control_pseudo_event_date",
    "prediction_year",
    "bankrupt_sector",
    "control_sector",
    "bankrupt_industry",
    "control_industry",
    "bankrupt_size",
    "control_size",
    "size_distance",
    "match_stage",
    "size_basis",
    "bankrupt_size_date",
    "control_size_date",
    "control_prediction_year",
    "match_rank",
    "matching_set_id",
    "matching_pair_id",
    "pair_retained",
    "bankrupt_event_retained",
    "control_event_retained",
]
for row_number, row in model.iterrows():
    try:
        records = json.loads(row.metadata_matching_records_json)
        if not isinstance(records, list) or not records:
            raise ValueError("Matching records must be a nonempty list")
        if len(records) != row.metadata_matching_pair_count_original:
            raise ValueError("Stored original pair count disagrees")
        own_key = (row.company_id, row.prediction_date)
        seen = set()
        for record in records:
            if not isinstance(record, dict) or not set(record_fields).issubset(record):
                raise ValueError("Incomplete matching record")
            bank_key = (
                record["bankrupt_company_id"],
                pd.Timestamp(record["bankrupt_prediction_date"]),
            )
            control_key = (
                record["control_company_id"],
                pd.Timestamp(record["control_pseudo_event_date"]),
            )
            if own_key != (bank_key if row.target == 1 else control_key):
                raise ValueError("Matching record belongs to another event/role")
            if record["matching_pair_id"] in seen:
                raise ValueError("Duplicate matching pair within an event")
            seen.add(record["matching_pair_id"])
            pair_copies.setdefault(record["matching_pair_id"], []).append(record)
    except (ValueError, TypeError, KeyError) as exc:
        json_errors.append(
            {"row": row_number, "company_id": row.company_id, "error": str(exc)}
        )
check(
    "matching",
    "Valid matching JSON and event membership",
    len(json_errors),
    len(model),
    "All pair records must be assigned to the correct side/event without duplicate links.",
    pd.DataFrame(json_errors),
)
if not pair_copies:
    raise ValueError("No usable matching records; cannot assess matching design")
conflicting_pairs = [
    pair_id
    for pair_id, copies in pair_copies.items()
    if any(copy != copies[0] for copy in copies[1:])
]
check(
    "matching",
    "Matching copies agree",
    len(conflicting_pairs),
    len(pair_copies),
    "A pair shared by two events must have identical metadata.",
)
pairs = pd.DataFrame([copies[0] for copies in pair_copies.values()])
pairs = parse_dates(
    pairs,
    [
        "bankrupt_prediction_date",
        "control_pseudo_event_date",
        "bankrupt_size_date",
        "control_size_date",
    ],
    "matching",
)
event_keys = set(model_index.tolist())
actual_bank_retained = pd.Series(
    [
        (r.bankrupt_company_id, r.bankrupt_prediction_date) in event_keys
        for r in pairs.itertuples()
    ],
    index=pairs.index,
)
actual_control_retained = pd.Series(
    [
        (r.control_company_id, r.control_pseudo_event_date) in event_keys
        for r in pairs.itertuples()
    ],
    index=pairs.index,
)
actual_pair_retained = actual_bank_retained & actual_control_retained
for column, actual in [
    ("bankrupt_event_retained", actual_bank_retained),
    ("control_event_retained", actual_control_retained),
    ("pair_retained", actual_pair_retained),
]:
    check(
        "matching",
        f"Retained membership: {column}",
        (~pairs[column].eq(actual)).sum(),
        len(pairs),
        "Recompute membership from actual modeling keys.",
    )
copy_counts = pd.Series(
    {pair_id: len(copies) for pair_id, copies in pair_copies.items()}
)
expected_copy_counts = actual_bank_retained.astype(
    int
) + actual_control_retained.astype(int)
check(
    "matching",
    "No matching links lost or multiplied",
    (~pairs.matching_pair_id.map(copy_counts).eq(expected_copy_counts)).sum(),
    len(pairs),
    "Each retained side stores exactly one copy of each original link.",
)
retained_pairs = pairs.loc[actual_pair_retained].copy()
positive_keys = pd.MultiIndex.from_frame(model.loc[positive, KEYS])
positive_keys.names = ["bankrupt_company_id", "bankrupt_prediction_date"]


def counts_per_bank(table):
    return (
        table.groupby(["bankrupt_company_id", "bankrupt_prediction_date"])
        .control_company_id.nunique()
        .reindex(positive_keys, fill_value=0)
    )


raw_counts = counts_per_bank(pairs)
retained_counts = counts_per_bank(retained_pairs)
matching_set_sizes = pd.DataFrame(
    {"original_controls": raw_counts, "retained_controls": retained_counts}
).reset_index()
check(
    "matching",
    "Requested 1:2 design declared",
    (~model.metadata_matching_ratio_requested.eq(EXPECTED_MATCH_RATIO)).sum(),
    len(model),
    "The dataset's declared design must match the requested 1:2 validation design.",
)
check(
    "matching",
    "Exactly two original controls per bankrupt event",
    raw_counts.ne(EXPECTED_MATCH_RATIO).sum(),
    len(raw_counts),
    "Count distinct control companies in each matched event.",
    matching_set_sizes.loc[
        matching_set_sizes.original_controls.ne(EXPECTED_MATCH_RATIO)
    ],
)
check(
    "matching",
    "Exactly two retained controls per bankrupt event",
    retained_counts.ne(EXPECTED_MATCH_RATIO).sum(),
    len(retained_counts),
    "Assess the sample after alignment exclusions, not the aggregate class ratio.",
    matching_set_sizes.loc[
        matching_set_sizes.retained_controls.ne(EXPECTED_MATCH_RATIO)
    ],
)
raw_usage = pairs.groupby("control_company_id").size()
kept_usage = retained_pairs.groupby("control_company_id").size()
control_usage = (
    pd.DataFrame({"original_pair_uses": raw_usage, "retained_pair_uses": kept_usage})
    .fillna(0)
    .astype(int)
)
control_usage.index.name = "control_company_id"
check(
    "matching",
    "Control reuse stays within limit",
    control_usage.original_pair_uses.gt(MAX_CONTROL_REUSE).sum(),
    len(control_usage),
    f"Count all dates/links per control company; limit = {MAX_CONTROL_REUSE}.",
    control_usage.loc[
        control_usage.original_pair_uses.gt(MAX_CONTROL_REUSE)
    ].reset_index(),
)
check(
    "matching",
    "No repeated control company in a matched set",
    pairs.duplicated(
        ["bankrupt_company_id", "bankrupt_prediction_date", "control_company_id"],
        keep=False,
    ).sum(),
    len(pairs),
    "Multiple pseudo-dates of one firm are not multiple independent controls within a set.",
)

# Compare encoded pair records to the actual source design where it is identifiable.
designs = model.metadata_matching_ratio_requested.dropna().unique()
if len(designs) == 1 and designs[0] in [1, 2, 3]:
    actual_design = int(designs[0])
    source_matches = audited_csv(
        PROCESSED / f"matched_event_sample_1to{actual_design}.csv"
    )
    source_matches = parse_dates(
        source_matches,
        [
            "bankrupt_prediction_date",
            "control_pseudo_event_date",
            "bankrupt_size_date",
            "control_size_date",
        ],
        "source",
    )
    applicable = source_matches.apply(
        lambda r: (r.bankrupt_company_id, r.bankrupt_prediction_date) in event_keys
        or (r.control_company_id, r.control_pseudo_event_date) in event_keys,
        axis=1,
    )
    join_keys = [
        "bankrupt_company_id",
        "bankrupt_prediction_date",
        "control_company_id",
        "control_pseudo_event_date",
    ]
    source_value_columns = [
        column for column in source_matches.columns if column not in join_keys
    ]
    available_value_columns = [
        column for column in source_value_columns if column in pairs.columns
    ]
    missing_encoded_columns = sorted(
        set(source_value_columns) - set(available_value_columns)
    )
    check(
        "matching",
        "Source matching metadata present in combined dataset",
        len(missing_encoded_columns),
        len(source_value_columns),
        "Rerun notebook 13 after notebook 06 when the match CSV has newer audit columns.",
        pd.DataFrame({"missing_matching_metadata_column": missing_encoded_columns}),
    )
    encoded = pairs.set_index(join_keys)
    expected_matches = source_matches.loc[applicable].set_index(join_keys)
    shared_pair_keys = encoded.index.intersection(expected_matches.index)
    difference_count = len(encoded.index.difference(expected_matches.index)) + len(
        expected_matches.index.difference(encoded.index)
    )
    check(
        "matching",
        "Encoded links agree with matching source",
        difference_count,
        len(expected_matches),
        "Only pairs with at least one retained member belong in event-level JSON.",
    )
    pair_value_failures = 0
    compared_values = 0
    for column in available_value_columns:
        left = encoded.reindex(shared_pair_keys)[column]
        right = expected_matches.reindex(shared_pair_keys)[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            equal = (
                left.astype("string").fillna("<NA>").to_numpy()
                == right.astype("string").fillna("<NA>").to_numpy()
            )
        pair_value_failures += int((~equal).sum())
        compared_values += len(equal)
    check(
        "matching",
        "Matching variable values preserved",
        pair_value_failures,
        compared_values,
        "Compare the matching fields shared by the source CSV and embedded pair records.",
    )
else:
    unverified(
        "matching",
        "Matching source design identifiable",
        "Declared designs are missing or inconsistent.",
    )

# The 1:2 file is a separate sensitivity sample; its existence does not make the
# current dataset a 1:2 dataset. Show what remains if its members are available here.
sensitivity = audited_csv(PROCESSED / "matched_event_sample_1to2.csv")
sensitivity = parse_dates(
    sensitivity, ["bankrupt_prediction_date", "control_pseudo_event_date"], "source"
)
sensitivity_retained = sensitivity.apply(
    lambda r: (r.bankrupt_company_id, r.bankrupt_prediction_date) in event_keys
    and (r.control_company_id, r.control_pseudo_event_date) in event_keys,
    axis=1,
)
sensitivity_counts = pd.DataFrame(
    {
        "original_1to2_controls": counts_per_bank(sensitivity),
        "retained_1to2_controls": counts_per_bank(
            sensitivity.loc[sensitivity_retained]
        ),
    }
).reset_index()

# Source dates must precede the individual prediction date and positive filing date.
for column in [
    "metadata_financial_reference_date",
    "metadata_market_reference_date",
    "metadata_macro_reference_date",
]:
    valid = model[column].notna() & model[column].le(model.prediction_date)
    check(
        "leakage",
        f"Prediction cutoff: {column}",
        (~valid).sum(),
        len(model),
        "Required reference dates must be known and on/before prediction.",
    )
source_date_columns = [
    c
    for c in date_columns
    if c.startswith("metadata_")
    and c != "metadata_outcome_end_date"
    and not c.startswith("metadata_matching_")
]
future_date_summary = []
for column in source_date_columns:
    observed = model[column].notna()
    after_prediction = observed & model[column].gt(model.prediction_date)
    after_filing = observed & positive & model[column].gt(model.bankruptcy_date)
    future_date_summary.append(
        {
            "source_column": column,
            "observed_dates": int(observed.sum()),
            "after_prediction": int(after_prediction.sum()),
            "after_positive_filing": int(after_filing.sum()),
        }
    )
future_date_summary = pd.DataFrame(future_date_summary)
check(
    "leakage",
    "All source/audit dates on or before prediction",
    future_date_summary.after_prediction.sum(),
    future_date_summary.observed_dates.sum(),
    "Includes financial, monthly prices, market endpoints, macro endpoints/windows, and trend sources.",
    future_date_summary.loc[future_date_summary.after_prediction.gt(0)],
)
check(
    "outcome",
    "No post-filing source dates in bankrupt predictors",
    future_date_summary.after_positive_filing.sum(),
    future_date_summary.observed_dates.sum(),
    "Predictor source dates and audit endpoints must not extend beyond the bankrupt firm's filing date.",
    future_date_summary.loc[future_date_summary.after_positive_filing.gt(0)],
)
lag_correct = model.metadata_financial_reference_date.le(
    model.prediction_date - pd.Timedelta(days=LEAD_DAYS)
)
check(
    "leakage",
    "Financial statements respect 90-day reporting lag",
    (~lag_correct).sum(),
    len(model),
    "Use fiscal statement dates no later than prediction minus 90 days.",
)
for lag in [1, 2, 3]:
    prior_prediction = model.prediction_date - pd.DateOffset(years=lag)
    ref = f"metadata_trend_reference_prediction_date_{lag}y"
    fin_cutoff = f"metadata_trend_financial_cutoff_date_{lag}y"
    fin_ref = f"metadata_trend_financial_reference_date_{lag}y"
    valid_cutoffs = model[ref].eq(prior_prediction) & model[fin_cutoff].eq(
        prior_prediction - pd.Timedelta(days=LEAD_DAYS)
    )
    check(
        "leakage",
        f"Trend lag-{lag} cutoffs calculated correctly",
        (~valid_cutoffs).sum(),
        len(model),
        "Apply each prior prediction date's own financial reporting lag.",
    )
    observed = model[fin_ref].notna()
    valid_fin = model[fin_ref].le(
        prior_prediction - pd.Timedelta(days=LEAD_DAYS)
    ) & model[fin_ref].lt(model.metadata_financial_reference_date)
    check(
        "leakage",
        f"Trend lag-{lag} financial source precedes its cutoff",
        (~valid_fin & observed).sum(),
        observed.sum(),
        "A lagged statement cannot come from a later annual slot.",
    )
    for field in market_columns:
        column = f"metadata_trend_market_{field}_reference_date_{lag}y"
        observed = model[column].notna()
        check(
            "leakage",
            f"Trend lag-{lag} market cutoff: {field}",
            (observed & model[column].gt(prior_prediction)).sum(),
            observed.sum(),
            "Lagged market values must precede their own earlier prediction date.",
        )

# Outcome coverage uses historical collection endpoints, never today's date alone.
expected_outcome = model.prediction_date + pd.DateOffset(months=OUTCOME_MONTHS)
check(
    "leakage",
    "12-calendar-month outcome dates",
    (~model.metadata_outcome_end_date.eq(expected_outcome)).sum(),
    len(model),
    "Use DateOffset(months=12), including leap-year calendar behavior.",
)
complete = model.metadata_outcome_end_date.notna() & model.metadata_outcome_end_date.le(
    VERIFIED_LABEL_END_DATE
)
check(
    "outcome",
    "Negative outcome window ends by verified label cutoff",
    (~complete).loc[negative].sum(),
    negative.sum(),
    f"Verified label end = {VERIFIED_LABEL_END_DATE:%Y-%m-%d}; Bloomberg extraction dates do not establish negative labels.",
)
followup = model.metadata_outcome_end_date.le(
    company_meta.end_date
) & model.metadata_outcome_end_date.le(company_meta.last_market_date)
check(
    "leakage",
    "Company-specific negative follow-up extends through outcome",
    (~followup).loc[negative].sum(),
    negative.sum(),
    "Both the extraction endpoint and observed market history must extend past the outcome end; future prices are follow-up only.",
)
filing_in_window = model.bankruptcy_date.gt(
    model.prediction_date
) & model.bankruptcy_date.le(model.metadata_outcome_end_date)
check(
    "leakage",
    "Positive filing lies in outcome window",
    (~filing_in_window).loc[positive].sum(),
    positive.sum(),
    "A positive outcome must occur after prediction and within the 12-month window.",
)

# Screen known bankruptcy filings using security and ticker identifiers, as in 05.
filings = audited_csv(ROOT / "data/raw/bankrupt/data.csv")
filings["bankruptcy_date"] = pd.to_datetime(
    filings.bankruptcy_date, format="mixed", errors="raise"
)
filing_frames = [filings, availability.loc[availability.company_type.eq("bankrupt")]]
security_filings = pd.concat(
    [
        pd.DataFrame(
            {
                "key": canonical_security(f.bloomberg_identifier),
                "date": f.bankruptcy_date,
            }
        )
        for f in filing_frames
    ]
)
ticker_filings = pd.concat(
    [
        pd.DataFrame({"key": canonical_security(f.ticker), "date": f.bankruptcy_date})
        for f in filing_frames
    ]
)
security_first = security_filings.dropna(subset=["key"]).groupby("key").date.min()
ticker_first = ticker_filings.dropna(subset=["key"]).groupby("key").date.min()
known_first = pd.concat(
    [
        canonical_security(model.metadata_bloomberg_identifier).map(security_first),
        canonical_security(model.metadata_ticker).map(ticker_first),
    ],
    axis=1,
).min(axis=1)
known_violation = (
    negative & known_first.notna() & known_first.le(model.metadata_outcome_end_date)
)
check(
    "leakage",
    "No known negative filing through outcome end",
    known_violation.sum(),
    negative.sum(),
    "Known earlier filings also invalidate an unscreened non-bankrupt identity.",
    model.loc[known_violation, CORE],
)
if confirmed_screening_end is None:
    unverified(
        "leakage",
        "Comprehensive bankruptcy surveillance for zero labels",
        "Notebook 05 has no confirmed bankruptcy-screening end. Calendar/market follow-up passes do not prove absence of bankruptcy; negative labels remain provisional.",
    )
else:
    unverified(
        "leakage",
        "Independent bankruptcy-surveillance evidence",
        "A configured cutoff is present, but comprehensive independent registry/source evidence must still be documented.",
    )
unverified(
    "leakage",
    "Financial publication dates and macro real-time vintages",
    "Fiscal/observation dates and a reporting-lag proxy do not verify release dates or whether revised macro values were available historically.",
)
unverified(
    "leakage",
    "Historical sector/industry and entity continuity",
    "Current Bloomberg classifications and security/ticker identity do not verify historical classification or legal-entity continuity.",
)

# Reconstruct notebook 05's candidate universe and exclusion reasons from cleaned history.
COVERAGE_MIN_FINANCIAL_YEARS = 3
COVERAGE_MIN_MARKET_MONTHS = 12
COVERAGE_MAX_FINANCIAL_AGE_MONTHS = 18
COVERAGE_MAX_MARKET_AGE_DAYS = 45
coverage_history_path = PROCESSED / "clean_company_long.csv"
input_hashes[coverage_history_path] = hashlib.sha256(
    coverage_history_path.read_bytes()
).hexdigest()
coverage_chunks = []
for chunk in pd.read_csv(
    coverage_history_path,
    usecols=["company_id", "date", "field", "value"],
    dtype={"company_id": "string"},
    chunksize=250_000,
):
    selected = chunk.loc[chunk["field"].isin(["total_assets", "adjusted_price"])].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected.loc[selected["value"].notna() & np.isfinite(selected["value"])]
    if len(selected):
        coverage_chunks.append(selected)
coverage_history = pd.concat(coverage_chunks, ignore_index=True)
coverage_history["company_id"] = coverage_history["company_id"].str.strip()
coverage_history["date"] = pd.to_datetime(
    coverage_history["date"], errors="raise"
).dt.normalize()
coverage_assets = coverage_history.loc[
    coverage_history.field.eq("total_assets") & coverage_history.value.ge(0)
]
coverage_prices = coverage_history.loc[
    coverage_history.field.eq("adjusted_price") & coverage_history.value.gt(0)
]
coverage_financial_dates = {
    key: group.date.sort_values().drop_duplicates().to_numpy(dtype="datetime64[ns]")
    for key, group in coverage_assets.groupby("company_id", sort=False)
}
coverage_market_dates = {
    key: group.date.sort_values().drop_duplicates().to_numpy(dtype="datetime64[ns]")
    for key, group in coverage_prices.groupby("company_id", sort=False)
}


def candidate_history_flags(company_id, prediction_dates):
    predictions = pd.DatetimeIndex(prediction_dates)
    query = predictions.to_numpy(dtype="datetime64[ns]")
    financial_dates = coverage_financial_dates.get(
        company_id, np.array([], dtype="datetime64[ns]")
    )
    market_dates = coverage_market_dates.get(
        company_id, np.array([], dtype="datetime64[ns]")
    )
    year_starts = (
        financial_dates[
            np.r_[
                True,
                financial_dates.astype("datetime64[Y]")[1:]
                != financial_dates.astype("datetime64[Y]")[:-1],
            ]
        ]
        if len(financial_dates)
        else financial_dates
    )
    financial_years = np.searchsorted(year_starts, query, side="left")
    financial_positions = np.searchsorted(financial_dates, query, side="left") - 1
    market_positions = np.searchsorted(market_dates, query, side="left") - 1
    last_financial = np.full(
        len(query), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    last_market = last_financial.copy()
    has_financial, has_market = financial_positions >= 0, market_positions >= 0
    last_financial[has_financial] = financial_dates[financial_positions[has_financial]]
    last_market[has_market] = market_dates[market_positions[has_market]]
    required_months = (
        query.astype("datetime64[M]").astype(np.int64)[:, None]
        - np.arange(1, COVERAGE_MIN_MARKET_MONTHS + 1)[None, :]
    )
    observed_months = np.unique(market_dates.astype("datetime64[M]").astype(np.int64))
    recent_month_counts = np.isin(required_months, observed_months).sum(axis=1)
    financial_cutoffs = (
        predictions - pd.DateOffset(months=COVERAGE_MAX_FINANCIAL_AGE_MONTHS)
    ).to_numpy()
    market_cutoffs = query - np.timedelta64(COVERAGE_MAX_MARKET_AGE_DAYS, "D")
    flags = pd.DataFrame(
        {
            "company_id": company_id,
            "prediction_date": predictions,
            "enough_financial_history": financial_years >= COVERAGE_MIN_FINANCIAL_YEARS,
            "fresh_financial_history": pd.Series(last_financial).to_numpy()
            >= financial_cutoffs,
            "enough_market_history": recent_month_counts == COVERAGE_MIN_MARKET_MONTHS,
            "recent_market_observation": last_market >= market_cutoffs,
        }
    )
    flags["history_eligible"] = flags[
        [
            "enough_financial_history",
            "fresh_financial_history",
            "enough_market_history",
            "recent_market_observation",
        ]
    ].all(axis=1)
    return flags


# Positive candidates are all recorded bankrupt firms; negative dates follow 05's eligible filings.
positive_candidates = availability.loc[
    availability.company_type.eq("bankrupt"),
    ["company_id", "bankruptcy_date", "bloomberg_identifier", "ticker"],
].copy()
positive_candidates = positive_candidates.dropna(subset=["bankruptcy_date"])
positive_candidates["prediction_date"] = (
    positive_candidates.bankruptcy_date - pd.Timedelta(days=LEAD_DAYS)
)
positive_candidates["outcome_end_date"] = (
    positive_candidates.prediction_date + pd.DateOffset(months=OUTCOME_MONTHS)
)
positive_flags = pd.concat(
    [
        candidate_history_flags(row.company_id, [row.prediction_date])
        for row in positive_candidates.itertuples(index=False)
    ],
    ignore_index=True,
)
positive_candidates = positive_candidates.reset_index(drop=True).join(
    positive_flags.drop(columns=["company_id", "prediction_date"])
)
positive_candidates["filing_in_outcome"] = positive_candidates.bankruptcy_date.gt(
    positive_candidates.prediction_date
) & positive_candidates.bankruptcy_date.le(positive_candidates.outcome_end_date)
positive_candidates["outcome_complete"] = positive_candidates.outcome_end_date.le(
    VERIFIED_LABEL_END_DATE
)
positive_candidates["eligible"] = (
    positive_candidates.history_eligible & positive_candidates.filing_in_outcome
)
positive_candidates["event_group"] = "bankrupt"

candidate_dates = pd.DatetimeIndex(
    event_tables[0].prediction_date.drop_duplicates().sort_values()
)
nonbankrupt_metadata = availability.loc[
    availability.company_type.eq("nonbankrupt"),
    ["company_id", "bloomberg_identifier", "ticker"],
].copy()
known_security = canonical_security(nonbankrupt_metadata.bloomberg_identifier).map(
    security_first
)
known_ticker = canonical_security(nonbankrupt_metadata.ticker).map(ticker_first)
nonbankrupt_metadata["known_first_filing"] = pd.concat(
    [known_security, known_ticker], axis=1
).min(axis=1)
negative_parts = []
for row in nonbankrupt_metadata.itertuples(index=False):
    part = candidate_history_flags(row.company_id, candidate_dates)
    part["outcome_end_date"] = part.prediction_date + pd.DateOffset(
        months=OUTCOME_MONTHS
    )
    part["outcome_complete"] = part.outcome_end_date.le(VERIFIED_LABEL_END_DATE)
    part["known_first_filing"] = row.known_first_filing
    part["no_known_filing_through_outcome"] = (
        part.known_first_filing.isna()
        | part.outcome_end_date.lt(part.known_first_filing)
    )
    part["eligible"] = (
        part.history_eligible
        & part.outcome_complete
        & part.no_known_filing_through_outcome
    )
    part["event_group"] = "nonbankrupt"
    negative_parts.append(part)
negative_candidates = pd.concat(negative_parts, ignore_index=True)
candidate_coverage = pd.concat(
    [positive_candidates, negative_candidates], ignore_index=True, sort=False
)
candidate_coverage["event_key"] = (
    candidate_coverage.company_id.astype(str)
    + "|"
    + candidate_coverage.prediction_date.dt.strftime("%Y-%m-%d")
)
eligible_keys = set(candidate_coverage.loc[candidate_coverage.eligible, "event_key"])
registry_keys = set(
    registry.company_id.astype(str)
    + "|"
    + registry.prediction_date.dt.strftime("%Y-%m-%d")
)
check(
    "coverage",
    "Reconstructed eligible candidates agree with notebook 05",
    len(eligible_keys.symmetric_difference(registry_keys)),
    len(registry_keys),
    "Candidate history and outcome rules must reproduce the saved eligible event tables.",
)
coverage_candidate_count = len(candidate_coverage)
coverage_eligible_count = int(candidate_coverage.eligible.sum())
coverage_excluded_count = coverage_candidate_count - coverage_eligible_count
coverage_summary = pd.DataFrame(
    [
        {
            "stage": "Candidate event universe",
            "observations": coverage_candidate_count,
            "detail": "All bankrupt firms with filing dates plus every non-bankrupt firm at each eligible bankrupt prediction date.",
        },
        {
            "stage": "Eligible after notebook 05 rules",
            "observations": coverage_eligible_count,
            "detail": "History requirements, positive filing window, complete negative outcome, and known-filing screen.",
        },
        {
            "stage": "Excluded by notebook 05 rules",
            "observations": coverage_excluded_count,
            "detail": "Candidate observations failing one or more eligibility criteria; reasons below can overlap.",
        },
    ]
)
matched_source = audited_csv(
    PROCESSED / "matched_event_sample_1to2.csv",
    dtype={"bankrupt_company_id": "string", "control_company_id": "string"},
)
matched_event_keys = set(
    matched_source.bankrupt_company_id.astype(str)
    + "|"
    + pd.to_datetime(matched_source.bankrupt_prediction_date).dt.strftime("%Y-%m-%d")
)
matched_event_keys |= set(
    matched_source.control_company_id.astype(str)
    + "|"
    + pd.to_datetime(matched_source.control_pseudo_event_date).dt.strftime("%Y-%m-%d")
)
model_event_keys = set(
    model.company_id.astype(str) + "|" + model.prediction_date.dt.strftime("%Y-%m-%d")
)
coverage_summary = pd.concat(
    [
        coverage_summary,
        pd.DataFrame(
            [
                {
                    "stage": "Eligible observations included in primary matches",
                    "observations": len(eligible_keys & matched_event_keys),
                    "detail": "Unique company-event members appearing in the 1:2 match file.",
                },
                {
                    "stage": "Eligible observations excluded by matching",
                    "observations": len(eligible_keys - matched_event_keys),
                    "detail": "Eligible candidates with no selected link in the primary 1:2 sample.",
                },
                {
                    "stage": "Retained in modeling dataset",
                    "observations": len(model_event_keys),
                    "detail": "Matched events remaining after event-time alignment and feature generation.",
                },
                {
                    "stage": "Matched observations excluded downstream",
                    "observations": len(matched_event_keys - model_event_keys),
                    "detail": "Primary matched events absent from the final modeling dataset.",
                },
            ]
        ),
    ],
    ignore_index=True,
)
reason_columns = {
    "insufficient_financial_history": ~candidate_coverage.enough_financial_history,
    "stale_financial_history": ~candidate_coverage.fresh_financial_history,
    "insufficient_market_history": ~candidate_coverage.enough_market_history,
    "stale_market_observation": ~candidate_coverage.recent_market_observation,
    "filing_outside_positive_outcome_window": candidate_coverage.event_group.eq(
        "bankrupt"
    )
    & ~candidate_coverage.filing_in_outcome.fillna(False),
    "incomplete_negative_outcome_window": candidate_coverage.event_group.eq(
        "nonbankrupt"
    )
    & ~candidate_coverage.outcome_complete.fillna(False),
    "known_filing_through_negative_outcome": candidate_coverage.event_group.eq(
        "nonbankrupt"
    )
    & ~candidate_coverage.no_known_filing_through_outcome.fillna(False),
}
coverage_exclusion_reasons = pd.DataFrame(
    [
        {
            "exclusion_reason": reason,
            "observations": int(mask.sum()),
            "overlap": "Reasons are non-exclusive.",
        }
        for reason, mask in reason_columns.items()
    ]
)
coverage_exclusion_reasons = pd.concat(
    [
        coverage_exclusion_reasons,
        pd.DataFrame(
            [
                {
                    "exclusion_reason": "not_selected_in_primary_match",
                    "observations": len(eligible_keys - matched_event_keys),
                    "overlap": "Downstream exclusions are separate from notebook 05 eligibility.",
                },
                {
                    "exclusion_reason": "alignment_or_feature_stage_exclusion",
                    "observations": len(matched_event_keys - model_event_keys),
                    "overlap": "Includes alignment/history eligibility filters after matching.",
                },
            ]
        ),
    ],
    ignore_index=True,
)

# Summarize the actual primary matching file and the corresponding retained links.
primary_case_events = matched_source[
    ["bankrupt_company_id", "bankrupt_prediction_date"]
].drop_duplicates()
primary_control_uses = matched_source.groupby("control_company_id").size()
primary_case_count = len(primary_case_events)
eligible_bankrupt_count = len(event_tables[0])
primary_controls_per_case = matched_source.groupby(
    ["bankrupt_company_id", "bankrupt_prediction_date"]
).control_company_id.nunique()
check(
    "matching",
    "Source 1:2 matched set size",
    primary_controls_per_case.ne(EXPECTED_MATCH_RATIO).sum(),
    len(primary_controls_per_case),
    "The primary matching file must contain two unique controls per case event.",
)
check(
    "matching",
    "Source control reuse stays within limit",
    primary_control_uses.gt(MAX_CONTROL_REUSE).sum(),
    len(primary_control_uses),
    f"Reuse in the complete 1:2 source file must not exceed {MAX_CONTROL_REUSE}.",
)
matching_summary = pd.DataFrame(
    [
        {
            "metric": "bankrupt firms",
            "value": primary_case_events.bankrupt_company_id.nunique(),
        },
        {
            "metric": "eligible bankrupt firms",
            "value": event_tables[0].company_id.nunique(),
        },
        {"metric": "bankrupt case events matched", "value": primary_case_count},
        {
            "metric": "unique controls",
            "value": matched_source.control_company_id.nunique(),
        },
        {
            "metric": "average controls per matched case",
            "value": len(matched_source) / max(primary_case_count, 1),
        },
        {
            "metric": "matching success rate (%)",
            "value": 100 * primary_case_count / max(eligible_bankrupt_count, 1),
        },
        {
            "metric": "eligible bankrupt candidate events",
            "value": eligible_bankrupt_count,
        },
        {
            "metric": "retained matched pairs after alignment",
            "value": len(retained_pairs),
        },
        {
            "metric": "average retained controls per retained case",
            "value": len(retained_pairs)
            / max(
                retained_pairs[["bankrupt_company_id", "bankrupt_prediction_date"]]
                .drop_duplicates()
                .shape[0],
                1,
            ),
        },
    ]
)
control_reuse_distribution = (
    primary_control_uses.value_counts()
    .sort_index()
    .rename_axis("times_used")
    .rename("control_companies")
    .reset_index()
)
control_reuse_distribution["reuse_limit"] = MAX_CONTROL_REUSE

# Independently reconstruct raw snapshots, ratios, and all 294 trend features.
# Use audited source dates and explicitly mask any source later than its own cutoff.
# This checks feature values, not merely apparently safe reference-date labels.
company_path = PROCESSED / "clean_company_long.csv"
input_hashes[company_path] = hashlib.sha256(company_path.read_bytes()).hexdigest()
company_ids = set(model.company_id.dropna())
raw_chunks = []
for chunk in pd.read_csv(
    company_path,
    usecols=["company_id", "date", "field", "value"],
    chunksize=250_000,
    dtype={"company_id": "string"},
):
    selected = chunk.loc[
        chunk.company_id.isin(company_ids) & chunk.field.isin(original_columns)
    ]
    if len(selected):
        raw_chunks.append(selected)
raw = pd.concat(raw_chunks, ignore_index=True)
raw["date"] = pd.to_datetime(raw.date, errors="raise").astype("datetime64[ns]")
check(
    "source",
    "Unique cleaned company/date/field observations",
    raw.duplicated(["company_id", "date", "field"], keep=False).sum(),
    len(raw),
    "Raw provenance lookups require unique observations.",
)
if raw.duplicated(["company_id", "date", "field"]).any():
    raise ValueError(
        "Duplicate cleaned source records prevent independent reconstruction"
    )
raw_lookup = raw.set_index(["company_id", "date", "field"])["value"]


def observed_values(field, dates, cutoff):
    index = pd.MultiIndex.from_arrays(
        [model.company_id, dates, np.repeat(field, len(model))],
        names=["company_id", "date", "field"],
    )
    values = raw_lookup.reindex(index).to_numpy(dtype=float)
    valid = dates.notna() & dates.le(cutoff)
    return pd.Series(np.where(valid, values, np.nan), index=model.index)


value_comparisons = []


def compare_values(output_column, expected, section, detail):
    actual = pd.to_numeric(model[output_column], errors="coerce").to_numpy(dtype=float)
    expected = np.asarray(expected, dtype=float)
    agrees = np.isclose(actual, expected, equal_nan=True, rtol=1e-9, atol=1e-10)
    failures = ~agrees
    value_comparisons.append(
        {
            "column": output_column,
            "checked": len(model),
            "mismatches": int(failures.sum()),
        }
    )
    sample = model.loc[failures, KEYS].copy()
    sample["actual"] = actual[failures]
    sample["expected_from_sources"] = expected[failures]
    check(
        section,
        f"Source value: {output_column}",
        failures.sum(),
        len(model),
        detail,
        sample,
    )


annual = {}
for lag in range(4):
    prediction_cutoff = (
        model.prediction_date
        if lag == 0
        else model.prediction_date - pd.DateOffset(years=lag)
    )
    financial_cutoff = prediction_cutoff - pd.Timedelta(days=LEAD_DAYS)
    fin_date = (
        model.metadata_financial_reference_date
        if lag == 0
        else model[f"metadata_trend_financial_reference_date_{lag}y"]
    )
    fields = {
        field: observed_values(field, fin_date, financial_cutoff)
        for field in financial_columns
    }
    for field in market_columns:
        date_column = (
            f"metadata_market_{field}_reference_date"
            if lag == 0
            else f"metadata_trend_market_{field}_reference_date_{lag}y"
        )
        fields[field] = observed_values(field, model[date_column], prediction_cutoff)
    for name, (numerator, denominator) in ratio_formulas.items():
        fields[name] = safe_divide(fields[numerator], fields[denominator])
    annual[lag] = fields
for field in original_columns:
    compare_values(
        f"original_{field}",
        annual[0][field],
        "preprocessing",
        "Original observations must equal cleaned source values at their audited pre-event dates; no scaling or imputation.",
    )
for slot in range(13):
    column = f"original_market_monthly_price_{slot:02d}"
    dates = model[f"metadata_market_monthly_price_date_{slot:02d}"]
    compare_values(
        column,
        observed_values("adjusted_price", dates, model.prediction_date),
        "preprocessing",
        "Observed monthly price-history values remain unchanged; unavailable observations stay missing.",
    )
for field in ratio_formulas:
    compare_values(
        f"derived_{field}",
        annual[0][field],
        "preprocessing",
        "Ratios use audited original observations and safe division, retaining valid negative values.",
    )

# Independent OLS slope formula based on centered available annual coordinates.
coordinates = np.array([0.0, -1.0, -2.0, -3.0])
for field in original_columns + list(ratio_formulas):
    for lag in [1, 2, 3]:
        change = (annual[0][field] - annual[lag][field]).replace(
            [np.inf, -np.inf], np.nan
        )
        growth = (safe_divide(annual[0][field], annual[lag][field]) - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        compare_values(
            f"trend_{field}_change_{lag}y",
            change,
            "trend",
            "Recomputed using only the original current and audited earlier observations.",
        )
        compare_values(
            f"trend_{field}_growth_{lag}y",
            growth,
            "trend",
            "No future values or pre-IPO zero filling; zero/missing lag denominator produces missing growth.",
        )
    matrix = np.column_stack([annual[lag][field] for lag in range(4)])
    slopes = np.full(len(model), np.nan)
    counts = np.isfinite(matrix).sum(axis=1)
    for row_number, row in enumerate(matrix):
        observed = np.isfinite(row)
        if observed.sum() >= 2:
            x = coordinates[observed]
            y = row[observed]
            centered_x = x - x.mean()
            slope = np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x)
            if np.isfinite(slope):
                slopes[row_number] = slope
    compare_values(
        f"trend_{field}_slope_3y",
        slopes,
        "trend",
        "OLS across only available audited annual slots; no interpolation/extrapolation.",
    )
    count_column = f"metadata_{field}_slope_3y_n_obs"
    check(
        "trend",
        f"Observed slope count: {field}",
        (~model[count_column].eq(counts)).sum(),
        len(model),
        "Audit count matches available finite annual observations.",
    )
value_comparisons = pd.DataFrame(value_comparisons)

# Check unprocessed feature provenance and inspect upstream preprocessing calls.
# These checks support the local notebook pipeline; they do not certify external
# Bloomberg edits, publication timing, or source vintages not present in the project.
upstream_exports = {
    name: audited_csv(PROCESSED / f"{name}.csv", dtype={"company_id": "string"})
    for name in [
        "original_features",
        "derived_features",
        "trend_features",
        "market_features",
        "macro_features",
    ]
}
previous_columns = set()
provenance_summary = []
for source, frame in upstream_exports.items():
    frame["prediction_date"] = pd.to_datetime(
        frame.prediction_date, errors="raise"
    ).astype("datetime64[ns]")
    is_unique = not frame.duplicated(KEYS).any()
    check(
        "preprocessing",
        f"Unique feature-source events: {source}",
        int(not is_unique),
        len(frame),
        "Exports must be uniquely keyed before comparison.",
    )
    if not is_unique:
        continue
    source_index = pd.MultiIndex.from_frame(frame[KEYS])
    cohort_difference = len(model_index.difference(source_index)) + len(
        source_index.difference(model_index)
    )
    check(
        "preprocessing",
        f"Feature-source cohort: {source}",
        cohort_difference,
        len(model),
        "No extra/deleted events during combination.",
    )
    indexed = frame.set_index(KEYS).reindex(model_index)
    additions = [c for c in frame if c not in previous_columns]
    count = 0
    mismatches = 0
    for column in additions:
        if source == "original_features" and (
            column in original_columns
            or column.startswith("market_monthly_price_")
            and not column.startswith("market_monthly_price_date_")
        ):
            output_column = f"original_{column}"
        elif source == "derived_features" and column in ratio_formulas:
            output_column = f"derived_{column}"
        elif source == "trend_features" and (
            "_change_" in column or "_growth_" in column or column.endswith("_slope_3y")
        ):
            output_column = f"trend_{column}"
        elif source == "market_features" and column in [
            "return_12m",
            "return_24m",
            "return_36m",
            "volatility_12m",
            "volatility_24m",
            "volatility_36m",
            "maximum_drawdown_12m",
            "maximum_drawdown_24m",
            "maximum_drawdown_36m",
            "market_cap_change_12m",
            "volume_change_12m",
            "turnover_change_12m",
        ]:
            output_column = f"market_{column}"
        elif (
            source == "macro_features"
            and "_macro_" not in column
            and column != "macro_feature_reference_date"
        ):
            output_column = f"macro_{column}"
        else:
            continue
        count += 1
        if output_column not in model:
            mismatches += len(model)
        else:
            agrees = np.isclose(
                model[output_column].to_numpy(dtype=float),
                indexed[column].to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-9,
                atol=1e-10,
            )
            mismatches += int((~agrees).sum())
    previous_columns.update(frame.columns)
    provenance_summary.append(
        {"source": source, "predictor_columns": count, "mismatched_values": mismatches}
    )
    check(
        "preprocessing",
        f"Unaltered feature export: {source}",
        mismatches,
        count * len(model),
        "Group prefixes rename values only; compare every contributed numeric predictor.",
    )
provenance_summary = pd.DataFrame(provenance_summary)

preprocessing_calls = []
for number in range(7, 14):
    path = next((ROOT / "notebooks").glob(f"{number:02d}_*.ipynb"))
    input_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell_number, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        # Convert IPython magics/shell commands to Python without executing them.
        source = notebook_source_transformer.transform_cell("".join(cell["source"]))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if name not in {
                "fit",
                "fit_transform",
                "transform",
                "fillna",
                "ffill",
                "bfill",
                "interpolate",
                "clip",
                "quantile",
                "StandardScaler",
                "MinMaxScaler",
                "RobustScaler",
                "QuantileTransformer",
                "PowerTransformer",
                "SimpleImputer",
                "KNNImputer",
                "IterativeImputer",
                "scale",
                "minmax_scale",
                "robust_scale",
            }:
                continue
            # Notebook 13 fills only control_usage_report's absent matching counts.
            owner = next(
                (
                    statement
                    for statement in tree.body
                    if isinstance(statement, ast.Assign)
                    and any(child is node for child in ast.walk(statement))
                ),
                None,
            )
            assigned_names = (
                [t.id for t in owner.targets if isinstance(t, ast.Name)]
                if owner
                else []
            )
            benign = (
                number == 13
                and name == "fillna"
                and assigned_names == ["control_usage_report"]
            )
            preprocessing_calls.append(
                {
                    "notebook": path.name,
                    "cell": cell_number,
                    "call": name,
                    "assessment": "matching count report only"
                    if benign
                    else "requires review",
                    "expression": ast.get_source_segment(source, node),
                }
            )
preprocessing_calls = pd.DataFrame(
    preprocessing_calls,
    columns=["notebook", "cell", "call", "assessment", "expression"],
)
suspect_count = preprocessing_calls.assessment.eq("requires review").sum()
check(
    "preprocessing",
    "No full-sample predictor preprocessing in upstream code",
    suspect_count,
    7,
    "AST scan of notebooks 07-13 checks fitting, scaling, imputation, interpolation, clipping, and quantiles. The sole count-report fillna is not predictor imputation.",
    preprocessing_calls.loc[preprocessing_calls.assessment.eq("requires review")],
)

# Compare groups using unique company-event rows, then separately describe pair links.
# Reused companies at different prediction dates remain distinct events; distributions
# here are descriptive and do not imply independent observations or hypothesis tests.
LABELS = {0: "Non-bankrupt", 1: "Bankrupt"}
observation_summary = model.groupby("target").agg(
    events=("company_id", "size"),
    unique_companies=("company_id", "nunique"),
    first_prediction=("prediction_date", "min"),
    last_prediction=("prediction_date", "max"),
)
observation_summary.index = observation_summary.index.map(LABELS)
company_event_counts = (
    model.groupby(["target", "company_id"])
    .size()
    .rename("events_per_company")
    .reset_index()
)
company_event_counts["group"] = company_event_counts.target.map(LABELS)
repeat_observation_summary = company_event_counts.groupby(
    "group"
).events_per_company.describe()


def standardized_difference(a, b):
    # Positive values mean the bankrupt mean/proportion exceeds the control value.
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (
        (a.mean() - b.mean()) / denominator
        if denominator > 0
        else (0.0 if a.mean() == b.mean() else np.nan)
    )


categorical_tables = {}
category_distance = []
for column in ["industry", "sector", "prediction_year"]:
    category = model[column].astype("string").fillna("Missing")
    counts = pd.crosstab(category, model.target).reindex(columns=[0, 1], fill_value=0)
    shares = counts.divide(counts.sum(axis=0).replace(0, np.nan), axis=1)
    table = pd.DataFrame(
        {
            "nonbankrupt_events": counts[0],
            "bankrupt_events": counts[1],
            "nonbankrupt_pct": shares[0] * 100,
            "bankrupt_pct": shares[1] * 100,
        }
    )
    table["percentage_point_difference"] = table.bankrupt_pct - table.nonbankrupt_pct
    table["standardized_difference"] = [
        standardized_difference(
            category.loc[positive].eq(value).astype(int),
            category.loc[negative].eq(value).astype(int),
        )
        for value in table.index
    ]
    categorical_tables[column] = (
        table.sort_index()
        if column == "prediction_year"
        else table.sort_values("bankrupt_events", ascending=False)
    )
    category_distance.append(
        {
            "variable": column,
            "categories": len(table),
            "total_variation_distance": 0.5 * (shares[1] - shares[0]).abs().sum(),
        }
    )
category_distance = pd.DataFrame(category_distance)

original_missing_fraction = (
    model[[f"original_{c}" for c in original_columns]].isna().mean(axis=1)
)
continuous_variables = {
    "market_cap (source units)": model.original_market_cap,
    "total_assets (source units)": model.original_total_assets,
    "log market_cap (positive values only)": np.log(
        model.original_market_cap.where(model.original_market_cap.gt(0))
    ),
    "log total_assets (positive values only)": np.log(
        model.original_total_assets.where(model.original_total_assets.gt(0))
    ),
    "original-field missing fraction": original_missing_fraction,
}
continuous_rows = []
continuous_balance = []
for name, values in continuous_variables.items():
    for target, mask in [(0, negative), (1, positive)]:
        group = values.loc[mask]
        valid = group.dropna()
        continuous_rows.append(
            {
                "variable": name,
                "group": LABELS[target],
                "events": len(group),
                "observed": len(valid),
                "missing_pct": group.isna().mean() * 100,
                "mean": valid.mean(),
                "std": valid.std(ddof=1),
                "min": valid.min(),
                "p25": valid.quantile(0.25),
                "median": valid.median(),
                "p75": valid.quantile(0.75),
                "max": valid.max(),
            }
        )
    continuous_balance.append(
        {
            "variable": name,
            "standardized_difference": standardized_difference(
                values.loc[positive], values.loc[negative]
            ),
        }
    )
continuous_summary = pd.DataFrame(continuous_rows)
continuous_balance = pd.DataFrame(continuous_balance)

# Show every field's missingness, and also group averages to expose unavailable series.
missingness = pd.DataFrame(
    {
        "nonbankrupt_missing_pct": model.loc[negative, feature_columns].isna().mean()
        * 100,
        "bankrupt_missing_pct": model.loc[positive, feature_columns].isna().mean()
        * 100,
        "overall_missing_pct": model[feature_columns].isna().mean() * 100,
    }
)
missingness["percentage_point_difference"] = (
    missingness.bankrupt_missing_pct - missingness.nonbankrupt_missing_pct
)
missingness.index.name = "feature"
missingness = missingness.sort_values("overall_missing_pct", ascending=False)
missing_group_rows = []
for group in ["original", "derived", "trend", "market", "macro"]:
    columns = [c for c in feature_columns if c.startswith(group + "_")]
    for target, mask in [(0, negative), (1, positive)]:
        missing_group_rows.append(
            {
                "feature_group": group,
                "group": LABELS[target],
                "columns": len(columns),
                "missing_cells_pct": model.loc[mask, columns].isna().to_numpy().mean()
                * 100,
            }
        )
missing_group_summary = pd.DataFrame(missing_group_rows)
all_missing_fields = pd.DataFrame(
    {"feature": [c for c in feature_columns if model[c].isna().all()]}
)
check(
    "quality",
    "No entirely missing original snapshot field",
    sum(model[f"original_{c}"].isna().all() for c in original_columns),
    len(original_columns),
    "Entirely missing engineered fields are retained and listed separately; they cannot inform a model.",
)

# Pair balance is reported separately: controls here may legitimately be repeated.
pair_balance = pd.DataFrame(
    {
        "comparison": [
            "same prediction year",
            "same Bloomberg industry",
            "same Bloomberg sector",
            "same calendar prediction date",
        ],
        "matching_pairs": len(retained_pairs),
        "agreeing_pairs": [
            retained_pairs.prediction_year.eq(
                retained_pairs.control_prediction_year
            ).sum(),
            retained_pairs.bankrupt_industry.eq(retained_pairs.control_industry).sum(),
            retained_pairs.bankrupt_sector.eq(retained_pairs.control_sector).sum(),
            retained_pairs.bankrupt_prediction_date.eq(
                retained_pairs.control_pseudo_event_date
            ).sum(),
        ],
    }
)
pair_balance["agreeing_pct"] = (
    pair_balance.agreeing_pairs / max(len(retained_pairs), 1) * 100
)
stage_summary = retained_pairs.groupby(["match_stage", "size_basis"]).agg(
    pairs=("matching_pair_id", "size"),
    mean_size_distance=("size_distance", "mean"),
    median_size_distance=("size_distance", "median"),
    max_size_distance=("size_distance", "max"),
)
size_basis_balance = []
for basis, table in retained_pairs.groupby("size_basis"):
    size_basis_balance.append(
        {
            "size_basis": basis,
            "pairs": len(table),
            "mean_bankrupt_log_size": table.bankrupt_size.mean(),
            "mean_control_log_size": table.control_size.mean(),
            "standardized_difference": standardized_difference(
                table.bankrupt_size, table.control_size
            ),
        }
    )
size_basis_balance = pd.DataFrame(size_basis_balance)


# Embed readable charts directly in the three HTML reports; no PNG files are added.
def table_html(frame, index=True):
    return frame.to_html(
        index=index,
        escape=True,
        border=0,
        na_rep="Unavailable",
        float_format=lambda value: f"{value:,.5g}",
        classes="data-table",
    )


def figure_html(figure, caption):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


fig, ax = plt.subplots(figsize=(12, 4))
year_counts = pd.crosstab(model.prediction_year, model.target).reindex(
    columns=[0, 1], fill_value=0
)
year_counts.rename(columns=LABELS).plot.bar(
    ax=ax, color=["#3979a8", "#b35a46"], width=0.85
)
ax.set(
    title="Company-events by prediction year",
    xlabel="Prediction year",
    ylabel="Unique company-events",
)
fig.tight_layout()
year_chart = figure_html(
    fig,
    "Counts refer to company-events. A company with several prediction dates contributes several events.",
)
category_charts = {}
for column in ["sector", "industry"]:
    table = categorical_tables[column].nlargest(15, "bankrupt_events")
    fig, ax = plt.subplots(figsize=(11, max(4, len(table) * 0.35)))
    table[["nonbankrupt_pct", "bankrupt_pct"]].rename(
        columns={"nonbankrupt_pct": "Non-bankrupt", "bankrupt_pct": "Bankrupt"}
    ).iloc[::-1].plot.barh(ax=ax, color=["#3979a8", "#b35a46"])
    ax.set(
        title=f"Company-event distribution by Bloomberg {column}",
        xlabel="Percentage within target group",
        ylabel="",
    )
    category_charts[column] = figure_html(
        fig,
        f"Up to 15 {column} categories with the most bankrupt events; full table appears below.",
    )
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, column, label in zip(
    axes,
    ["original_total_assets", "original_market_cap"],
    ["Total assets", "Market capitalization"],
):
    data = [
        np.log10(model.loc[mask, column].where(model.loc[mask, column].gt(0)))
        .dropna()
        .to_numpy()
        for mask in [negative, positive]
    ]
    available = [values for values in data if len(values)]
    bins = (
        np.histogram_bin_edges(np.concatenate(available), bins=30) if available else 30
    )
    for values, group, color in zip(
        data, ["Non-bankrupt", "Bankrupt"], ["#3979a8", "#b35a46"]
    ):
        if len(values):
            ax.hist(
                values, bins=bins, density=True, alpha=0.5, label=group, color=color
            )
    ax.set(title=label, xlabel="log10(value in source units)", ylabel="Density")
    ax.legend()
size_chart = figure_html(
    fig,
    "Log charts include positive observed values only. Raw/nonpositive/missing values remain unchanged and are reflected in the tables.",
)
fig, ax = plt.subplots(figsize=(11, 7))
missingness.head(20)[["nonbankrupt_missing_pct", "bankrupt_missing_pct"]].rename(
    columns={
        "nonbankrupt_missing_pct": "Non-bankrupt",
        "bankrupt_missing_pct": "Bankrupt",
    }
).iloc[::-1].plot.barh(ax=ax, color=["#3979a8", "#b35a46"])
ax.set(
    title="Fields with the highest missingness", xlabel="Missing percentage", ylabel=""
)
missing_chart = figure_html(
    fig,
    "Unavailable true credit spread and recession features are retained without imputation.",
)

check_results = pd.DataFrame(checks)
failed_checks = check_results.loc[check_results.status.eq("FAIL")]
unverified_checks = check_results.loc[check_results.status.eq("UNVERIFIED")]
observed_checks_passed = failed_checks.empty
validation_passed = observed_checks_passed and unverified_checks.empty
summary = (
    check_results.groupby(["section", "status"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["PASS", "FAIL", "UNVERIFIED"], fill_value=0)
)

CSS = """body{font:15px/1.6 system-ui,sans-serif;color:#202b39;background:#f4f6f9;margin:0}
main{max-width:1200px;margin:24px auto;padding:30px;background:white;border-radius:12px}
h1{font-size:28px}h2{margin-top:32px;color:#193f5c}.notice{padding:14px 18px;background:#fff2d8;border-left:5px solid #b47914}
.status{padding:12px;background:#fbe8e6;border-left:5px solid #b5473c}.ok{background:#e8f4ec;border-color:#2b794b}
.table-wrap{overflow-x:auto;margin:14px 0}.data-table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid #e0e5eb;text-align:left;vertical-align:top}th{background:#edf2f7;position:sticky;top:0}
tr:nth-child(even){background:#f8fafc}img{max-width:100%;height:auto}figure{margin:22px 0}figcaption{font-size:13px;color:#536273}
code{background:#edf2f7;padding:2px 5px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}small{color:#596779}"""


def section(title, body):
    return f"<h2>{html.escape(title)}</h2>{body}"


def wrap_table(frame, index=True):
    return '<div class="table-wrap">' + table_html(frame, index) + "</div>"


def check_sections(names):
    selected = check_results.loc[check_results.section.isin(names)]
    result = wrap_table(selected, index=False)
    for name in selected.loc[selected.status.eq("FAIL"), "check"]:
        if name in evidence:
            result += (
                "<details><summary>"
                + html.escape(name)
                + ": first 20 violations</summary>"
                + wrap_table(evidence[name], index=False)
                + "</details>"
            )
    return result


def report(title, content):
    status = (
        "PASS"
        if validation_passed
        else "FAIL / evidence incomplete"
        if not observed_checks_passed
        else "Evidence incomplete"
    )
    overview = f'<p class="status{" ok" if validation_passed else ""}"><strong>Overall validation: {status}.</strong> '
    overview += f"{len(failed_checks)} failed checks; {len(unverified_checks)} unverified checks. "
    overview += "Do not interpret completed date checks as full certification of outcome surveillance or historical information availability.</p>"
    overview += (
        "<p>Requested matching design: <strong>1:2</strong>. Current declared design(s): <strong>"
        + html.escape(str(sorted(designs.tolist())))
        + "</strong>.</p>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>"
        f"<p><small>Input: data/processed/modeling_dataset_unprocessed.csv ? {len(model):,} company-events ? {len(model.columns):,} columns</small></p>"
        + overview
        + content
        + "<p><small>Generated by notebook 14. Dataset values and rows were not modified. PASS/FAIL/UNVERIFIED describe only the stated check and available local evidence.</small></p></main></body></html>"
    )


quality_content = section("Validation summary", wrap_table(summary))
quality_content += section(
    "Blocking failures and evidence gaps",
    wrap_table(check_results.loc[~check_results.status.eq("PASS")], index=False),
)
quality_content += section(
    "Observation coverage",
    wrap_table(observation_summary)
    + year_chart
    + wrap_table(repeat_observation_summary),
)
quality_content += section(
    "Candidate, eligible, and excluded observations",
    wrap_table(coverage_summary, index=False)
    + wrap_table(coverage_exclusion_reasons, index=False),
)
quality_content += section(
    "Feature validity diagnostics",
    wrap_table(feature_diagnostics, index=False)
    + wrap_table(zero_denominator_summary, index=False),
)
quality_content += section(
    "Structural and mechanical checks",
    check_sections(["structure", "quality", "matching", "source"]),
)
quality_content += section(
    "Unprocessed numeric source comparisons",
    wrap_table(provenance_summary, index=False)
    + wrap_table(value_comparisons, index=False),
)
quality_content += section(
    "Entirely unavailable fields",
    "<p>Missing fields are retained, never filled with zero or imputed. These columns require a later predictor-selection decision.</p>"
    + wrap_table(all_missing_fields, index=False),
)
quality_content += section(
    "Missingness by feature group",
    wrap_table(missing_group_summary, index=False) + missing_chart,
)
quality_content += section("Missingness for every feature", wrap_table(missingness))

balance_content = (
    '<p class="notice">Declared matching design(s): '
    + html.escape(str(sorted(designs.tolist())))
    + " controls per bankrupt event. Retained set sizes (controls: bankrupt events): "
    + html.escape(str(retained_counts.value_counts().sort_index().to_dict()))
    + ". Notebook 14 evaluates the requested 1:2 design without changing the dataset.</p>"
)
balance_content += section(
    "Matching summary",
    wrap_table(matching_summary, index=False)
    + wrap_table(control_reuse_distribution, index=False),
)
balance_content += section("Matching sanity checks", check_sections(["matching"]))
balance_content += section(
    "Original and retained controls per bankrupt event",
    wrap_table(
        matching_set_sizes.groupby(["original_controls", "retained_controls"])
        .size()
        .rename("bankrupt_events")
        .reset_index(),
        index=False,
    )
    + wrap_table(matching_set_sizes, index=False),
)
balance_content += section(
    "Separate 1:2 sensitivity sample",
    "<p>This existing sample is checked only for available membership. Its existence does not convert the current dataset to 1:2. Counts below also expose any members excluded by alignment.</p>"
    + wrap_table(
        sensitivity_counts.groupby(["original_1to2_controls", "retained_1to2_controls"])
        .size()
        .rename("bankrupt_events")
        .reset_index(),
        index=False,
    ),
)
balance_content += section(
    "Control-company reuse",
    "<p>Reuse counts are matching links over all pseudo-dates, not just repeated company-event rows. Maximum permitted original reuse is three.</p>"
    + wrap_table(
        control_usage.value_counts().rename("companies").reset_index(), index=False
    )
    + "<details><summary>Every control company</summary>"
    + wrap_table(control_usage)
    + "</details>",
)
balance_content += section(
    "Company-event counts", wrap_table(observation_summary) + year_chart
)
balance_content += section(
    "Group balance interpretation",
    "<p>Group distributions use unique company-event rows. Pair summaries separately include reused matching links. Standardized differences use bankrupt minus non-bankrupt and pooled sample standard deviations; undefined cases remain unavailable. They are descriptive, without independence assumptions. Source units inherit Bloomberg collection settings; logarithms use positive values only.</p>"
    + wrap_table(category_distance, index=False)
    + wrap_table(continuous_balance, index=False),
)
for column in ["industry", "sector", "prediction_year"]:
    balance_content += section(
        f"Bloomberg {column}" if column != "prediction_year" else "Prediction year",
        category_charts.get(column, "") + wrap_table(categorical_tables[column]),
    )
balance_content += section(
    "Market capitalization and assets",
    size_chart + wrap_table(continuous_summary, index=False),
)
balance_content += section(
    "Missingness balance",
    wrap_table(missing_group_summary, index=False)
    + missing_chart
    + wrap_table(missingness),
)
balance_content += section(
    "Retained pair agreement and size matching",
    wrap_table(pair_balance, index=False)
    + wrap_table(stage_summary)
    + wrap_table(size_basis_balance, index=False),
)

leakage_content = section(
    "What these checks establish",
    "<p>Source/reference dates must be on or before each individual prediction date. Financial statements must also respect the 90-day reporting lag. Historical trend endpoints are checked against their own earlier prediction cutoffs, and all trend values are independently reconstructed from cleaned observations.</p><p>Filing dates and outcome-end dates are future labels/audits and are excluded from predictor source-date checks. Control follow-up observations after prediction establish outcome coverage only.</p>",
)
leakage_content += section(
    "Unverified evidence", wrap_table(unverified_checks, index=False)
)
leakage_content += section(
    "Date and outcome-window checks", check_sections(["leakage", "outcome"])
)
leakage_content += section(
    "Candidate coverage and eligibility exclusions",
    wrap_table(coverage_summary, index=False)
    + wrap_table(coverage_exclusion_reasons, index=False),
)
leakage_content += section(
    "Source date audit", wrap_table(future_date_summary, index=False)
)
leakage_content += section(
    "Independent trend reconstruction", check_sections(["trend"])
)
leakage_content += section(
    "No full-sample preprocessing",
    check_sections(["preprocessing"]) + wrap_table(preprocessing_calls, index=False),
)
leakage_content += section(
    "Predictor-selection boundary",
    "<p>Use the original_, derived_, trend_, market_, macro_, and missing_ groups for later predictor selection. Exclude company IDs, target, bankruptcy dates, event types, outcome dates, and all metadata_ columns, including matching stages, sizes, partners, and reuse counts. No train/test split or model was created at this stage.</p>",
)

# Validate read-only inputs before writing all three requested reports.
for path, digest in input_hashes.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Read-only input changed during validation: {path}")
REPORTS.mkdir(parents=True, exist_ok=True)
outputs = {
    "data_quality_report.html": report(
        "Modeling dataset: data quality report", quality_content
    ),
    "matching_balance_report.html": report(
        "Modeling dataset: matching balance report", balance_content
    ),
    "leakage_check_report.html": report(
        "Modeling dataset: leakage check report", leakage_content
    ),
}
for filename, content in outputs.items():
    path = REPORTS / filename
    path.write_text(content, encoding="utf-8")
