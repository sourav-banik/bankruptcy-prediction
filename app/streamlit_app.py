"""Project overview and company-level demo using the processed event dataset."""

import base64
import json
import mimetypes
import re
import sys
from urllib.parse import unquote
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_demo import (  # noqa: E402
    KEY_TREND_GROUPS,
    METRIC_GROUPS,
    load_company_events,
    load_model,
    score_event,
    test_metrics,
)

REPORTS_PATH = ROOT / "reports"
FIGURES_PATH = ROOT / "figures"
MODEL_OPTIONS = {
    "Elastic Net logistic": ROOT / "models" / "elastic_net_logistic.joblib",
    "Random Forest": ROOT / "models" / "random_forest.joblib",
    "Gradient Boosting (sensitivity)": ROOT
    / "models"
    / "gradient_boosting_sensitivity.joblib",
}
DATA_PATH = ROOT / "data" / "processed" / "modeling_dataset_unprocessed.csv"
PHASE_LABELS = {
    "01_data_profile": "01 - Data profile",
    "02_modeling_data_validation": "02 - Modeling data validation",
    "03_event_aligned_visualization": "03 - Event-aligned analysis",
    "04_feature_diagnostics": "04 - Feature diagnostics",
    "05_elastic_net_reporting": "05 - Elastic Net",
    "06_random_forest_reporting": "06 - Random Forest and Gradient Boosting",
    "07_robustness_visualization": "07 - Robustness analysis",
    "08_probability_calibration": "08 - Probability calibration",
}


@st.cache_resource
def load_demo_model(model_path):
    """Cache each fitted classifier across Streamlit reruns."""
    return load_model(model_path)


@st.cache_data
def load_demo_events(feature_columns):
    """Cache event rows used by both the company selector and model scoring."""
    return load_company_events(DATA_PATH, feature_columns)


@st.cache_data(max_entries=8)
def load_html_report(relative_path):
    """Embed local figure links so static HTML reports render inside the app."""
    report_path = ROOT / relative_path
    html = report_path.read_text(encoding="utf-8", errors="replace")

    def embed_image(match):
        source = unquote(match.group("source"))
        if source.startswith(("data:", "http://", "https://", "#")):
            return match.group(0)

        image_path = (report_path.parent / source).resolve()
        try:
            image_path.relative_to(ROOT)
        except ValueError:
            return match.group(0)
        if not image_path.is_file():
            return match.group(0)

        mime_type = (
            mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        )
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"{match.group('prefix')}data:{mime_type};base64,{image_data}{match.group('suffix')}"

    image_source = re.compile(
        r'(?P<prefix>\bsrc=["\'])(?P<source>[^"\']+)(?P<suffix>["\'])',
        re.IGNORECASE,
    )
    return image_source.sub(embed_image, html)


def display_value(value):
    """Format one source value without assigning an undocumented unit."""
    if pd.isna(value):
        return "Not available"
    return f"{value:,.4g}"


st.set_page_config(page_title="Bankruptcy Early-Warning Lab", layout="wide")
st.title("Bankruptcy Early-Warning Lab")
st.caption(
    "Explore the project and inspect model scores for historical company events."
)

project_tab, company_tab = st.tabs(["Project", "Explore companies"])

with project_tab:
    st.subheader("What the project does")
    st.write(
        "This research workflow aligns company financial statements, market history, "
        "and macroeconomic conditions to prediction dates before bankruptcy filings. "
        "Non-bankrupt firms receive historical pseudo-events matched to bankrupt firms."
    )

    metrics = test_metrics(ROOT)
    if metrics is not None:
        elastic, forest = metrics
        metric_columns = st.columns(3)
        for column, (name, row) in zip(
            metric_columns, [("Elastic Net", elastic), ("Random Forest", forest)]
        ):
            accuracy = (row["tn"] + row["tp"]) / row["n"]
            column.metric(f"{name} test accuracy", f"{accuracy:.1%}", border=True)
            column.metric(f"{name} ROC-AUC", f"{row['roc_auc']:.3f}", border=True)
        st.caption(
            "PR-AUC - Elastic Net: "
            f"{elastic['pr_auc_average_precision']:.3f}; Random Forest: "
            f"{forest['pr_auc_average_precision']:.3f}."
        )
        st.caption(
            f"Held-out test results cover {int(elastic['bankrupt_n'])} bankrupt events; "
            "the matched sample is not population-representative."
        )

    st.markdown("**Workflow**  ")
    st.write(
        "Bloomberg data -> event alignment -> financial, market, and macro features -> "
        "time-based validation -> model comparison"
    )

    audit_path = (
        ROOT / "reports" / "02_modeling_data_validation" / "data_quality_report.html"
    )
    if audit_path.is_file():
        audit_text = audit_path.read_text(encoding="utf-8", errors="replace")
        if "Overall validation: FAIL / evidence incomplete" in audit_text:
            st.warning(
                "The latest data audit reports failed or unverified checks. Review the "
                "data-quality and leakage reports when interpreting these results."
            )

    st.info(
        "The matched sample has artificial bankruptcy prevalence. Model estimates are "
        "not calibrated real-world probabilities or financial advice."
    )

    st.divider()
    st.subheader("Analysis outputs")
    phase_names = sorted(
        {
            folder.name
            for root in (REPORTS_PATH, FIGURES_PATH)
            if root.is_dir()
            for folder in root.iterdir()
            if folder.is_dir()
        }
    )

    if not phase_names:
        st.info("No report or figure folders are available yet.")
    else:
        selected_phase = st.selectbox(
            "Analysis phase",
            phase_names,
            format_func=lambda phase: PHASE_LABELS.get(
                phase, phase.replace("_", " ").title()
            ),
        )
        report_files = sorted(
            file_path
            for file_path in (REPORTS_PATH / selected_phase).glob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in {".html", ".csv", ".json"}
        )
        figure_files = sorted(
            file_path
            for file_path in (FIGURES_PATH / selected_phase).glob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

        artifact_groups = {}
        if report_files:
            artifact_groups["Reports"] = report_files
        if figure_files:
            artifact_groups["Figures"] = figure_files

        if not artifact_groups:
            st.info("This phase does not have report or figure files yet.")
        else:
            selected_kind = st.selectbox("Output type", list(artifact_groups))
            files_by_name = {
                file_path.name: file_path
                for file_path in artifact_groups[selected_kind]
            }
            selected_name = st.selectbox("File", list(files_by_name))
            selected_path = files_by_name[selected_name]
            mime_type = mimetypes.guess_type(selected_path.name)[0]

            with st.container(border=True):
                if selected_kind == "Figures":
                    st.image(
                        selected_path,
                        caption=selected_path.name,
                        alt=f"Figure from {PHASE_LABELS.get(selected_phase, selected_phase)}",
                        width="stretch",
                    )
                    st.download_button(
                        "Download figure",
                        data=selected_path.read_bytes(),
                        file_name=selected_path.name,
                        mime=mime_type or "application/octet-stream",
                    )
                elif selected_path.suffix.lower() == ".html":
                    st.html(
                        load_html_report(selected_path.relative_to(ROOT).as_posix()),
                        width="stretch",
                    )
                    st.download_button(
                        "Download report",
                        data=selected_path.read_bytes(),
                        file_name=selected_path.name,
                        mime="text/html",
                    )
                elif selected_path.suffix.lower() == ".csv":
                    report_preview = pd.read_csv(selected_path, nrows=1000)
                    st.caption(
                        "Previewing up to the first 1,000 rows; download for the full file."
                    )
                    st.dataframe(
                        report_preview,
                        hide_index=True,
                        alt=f"Preview of {selected_path.name}",
                    )
                    st.download_button(
                        "Download report data",
                        data=selected_path.read_bytes(),
                        file_name=selected_path.name,
                        mime="text/csv",
                    )
                else:
                    st.json(json.loads(selected_path.read_text(encoding="utf-8")))
                    st.download_button(
                        "Download report data",
                        data=selected_path.read_bytes(),
                        file_name=selected_path.name,
                        mime="application/json",
                    )

with company_tab:
    st.subheader("Historical company event")

    if not DATA_PATH.is_file():
        st.error(
            "The processed modeling dataset is missing. Run notebooks 13 and 17-19 "
            "to create the event data and fitted model."
        )
    else:
        available_models = {
            name: path for name, path in MODEL_OPTIONS.items() if path.is_file()
        }
        if not available_models:
            st.error("No supported fitted model is available in the models folder.")
            st.stop()

        selected_model_name = st.selectbox(
            "Model", options=list(available_models), index=0
        )
        selected_model_path = available_models[selected_model_name]
        try:
            model = load_demo_model(str(selected_model_path))
            events = load_demo_events(tuple(model.feature_names_in_))
        except Exception as error:
            st.error(f"Could not load the company data or saved model: {error}")
        else:
            if events.empty:
                st.warning(
                    "The processed dataset contains no selectable company events."
                )
            else:
                companies = (
                    events[["company_id", "metadata_name", "metadata_ticker"]]
                    .drop_duplicates("company_id")
                    .sort_values("metadata_name", na_position="last")
                )
                company_labels = {}
                for row in companies.itertuples(index=False):
                    name = (
                        row.metadata_name
                        if pd.notna(row.metadata_name)
                        else row.company_id
                    )
                    ticker = (
                        row.metadata_ticker
                        if pd.notna(row.metadata_ticker)
                        else "no ticker"
                    )
                    company_labels[row.company_id] = f"{name} ({ticker})"

                selected_company = st.selectbox(
                    "Company",
                    options=companies["company_id"].tolist(),
                    format_func=lambda company_id: company_labels[company_id],
                )
                company_events = events.loc[events["company_id"].eq(selected_company)]

                def event_label(index):
                    row = events.loc[index]
                    kind = (
                        "Bankruptcy event" if row["target"] == 1 else "Matched control"
                    )
                    return f"{row['prediction_date']:%Y-%m-%d} | {kind}"

                selected_event_index = st.selectbox(
                    "Prediction date",
                    options=company_events.index.tolist(),
                    format_func=event_label,
                )
                event = events.loc[selected_event_index]

                company_name = event["metadata_name"]
                ticker = event["metadata_ticker"]
                name = company_name if pd.notna(company_name) else selected_company
                ticker_label = ticker if pd.notna(ticker) else "Ticker unavailable"
                sector = (
                    event["sector"]
                    if pd.notna(event["sector"])
                    else "Sector unavailable"
                )
                industry = (
                    event["industry"]
                    if pd.notna(event["industry"])
                    else "Industry unavailable"
                )
                st.markdown(f"**{name}** | {ticker_label} | {sector} / {industry}")

                try:
                    estimate = score_event(model, event)
                except Exception as error:
                    st.error(f"Could not score this company event: {error}")
                else:
                    result_columns = st.columns(2)
                    result_columns[0].metric(
                        "Model-estimated probability", f"{estimate:.1%}"
                    )
                    label = (
                        "Bankrupt" if event["target"] == 1 else "Non-bankrupt control"
                    )
                    result_columns[1].metric("Historical sample label", label)
                    st.caption(
                        "This is the model estimate for a row in the matched research sample. "
                        "It is not a calibrated real-world probability."
                    )

                    if pd.notna(event["bankruptcy_date"]):
                        st.caption(
                            f"Recorded filing date: {event['bankruptcy_date']:%Y-%m-%d}"
                        )

                    trend_rows = []
                    for label, prefix in KEY_TREND_GROUPS.items():
                        fields = [f"{prefix}_{horizon}y" for horizon in (1, 2, 3)]
                        if any(field in event.index for field in fields):
                            trend_rows.append(
                                {
                                    "Trend": label,
                                    **{
                                        f"{horizon}-year": (
                                            f"{event[field]:.1%}"
                                            if pd.notna(event.get(field))
                                            and "_growth_" in field
                                            else display_value(event.get(field))
                                        )
                                        for horizon, field in zip((1, 2, 3), fields)
                                    },
                                }
                            )

                    st.markdown("**Trend data**")
                    st.caption(
                        "Event-time changes and growth features compare this event with "
                        "prior annual observations."
                    )
                    st.dataframe(
                        pd.DataFrame(trend_rows),
                        hide_index=True,
                        alt="Selected financial trends over one, two, and three years",
                    )

                    all_trend_columns = [
                        column
                        for column in events.columns
                        if column.startswith("trend_")
                    ]
                    with st.expander("All trend features"):
                        all_trends = pd.DataFrame(
                            [
                                {
                                    "Trend feature": column.removeprefix(
                                        "trend_"
                                    ).replace("_", " "),
                                    "Value": display_value(event[column]),
                                }
                                for column in all_trend_columns
                            ]
                        )
                        st.dataframe(
                            all_trends,
                            hide_index=True,
                            alt="All trend features for the selected historical event",
                        )

                    for group_name, fields in METRIC_GROUPS.items():
                        display_rows = [
                            {"Metric": label, "Value": display_value(event[column])}
                            for label, column in fields.items()
                        ]
                        with st.expander(
                            group_name, expanded=group_name == "Financial statements"
                        ):
                            st.dataframe(
                                pd.DataFrame(display_rows),
                                hide_index=True,
                                alt=f"{group_name} for the selected historical event",
                            )

                    st.caption(
                        "Metrics are taken from the selected event date. Source units are "
                        "shown as stored in the processed Bloomberg dataset."
                    )
