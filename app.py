from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import TARGET_COLUMNS, load_ticket_data, regularize_intervals
from src.models.artifacts import train_or_load
from src.models.forecasting import HORIZONS, forecast_all_horizons
from src.prediction.demand_monitor import classify_forecast

st.set_page_config(
    page_title="Ferry Intelligence",
    page_icon="⛴️",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_DIR / "data" / "Toronto Island Ferry Ticket Counts(1).csv"
SEASONS = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


@st.cache_data
def read_csv_bytes(file_bytes):
    return pd.read_csv(BytesIO(file_bytes))


def detect_column(columns, kind):
    patterns = {
        "timestamp": [
            r"timestamp", r"datetime", r"date.?time", r"recorded.?at",
            r"^date$", r"date$", r"^time$", r"time$",
        ],
        "sales": [
            r"sales?", r"tickets?.?(sold|issued|purchased)",
            r"(sold|issued|purchased).?tickets?",
        ],
        "redemptions": [
            r"redemptions?", r"tickets?.?(redeemed|scanned|used)",
            r"(redeemed|scanned|used).?tickets?",
        ],
    }
    normalized = {
        column: re.sub(r"[^a-z0-9]", "", str(column).strip().lower())
        for column in columns
    }
    for pattern in patterns[kind]:
        for column, name in normalized.items():
            if re.search(pattern, name):
                return column
    return None


def prepare_data(raw_data, mapping):
    data = raw_data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    timestamp = pd.to_datetime(data[mapping["timestamp"]], errors="coerce")
    numeric_values = {}
    for kind in ["sales", "redemptions"]:
        cleaned = data[mapping[kind]].astype("string").str.strip()
        cleaned = cleaned.str.replace(",", "", regex=False)
        cleaned = cleaned.mask(cleaned.eq(""))
        numeric_values[kind] = pd.to_numeric(cleaned, errors="coerce")

    invalid_timestamp = timestamp.isna()
    invalid_sales = numeric_values["sales"].isna()
    invalid_redemptions = numeric_values["redemptions"].isna()
    negative_sales = numeric_values["sales"] < 0
    negative_redemptions = numeric_values["redemptions"] < 0
    valid = ~(
        invalid_timestamp
        | invalid_sales
        | invalid_redemptions
        | negative_sales
        | negative_redemptions
    )

    cleaned_data = data.loc[valid].copy()
    for kind, canonical_name, values in [
        ("timestamp", "Timestamp", timestamp),
        ("sales", "Sales Count", numeric_values["sales"]),
        ("redemptions", "Redemption Count", numeric_values["redemptions"]),
    ]:
        if mapping[kind] != canonical_name and canonical_name in cleaned_data:
            source_name = f"Source {canonical_name}"
            while source_name in cleaned_data:
                source_name = f"Source {source_name}"
            cleaned_data = cleaned_data.rename(
                columns={canonical_name: source_name}
            )
        cleaned_data[canonical_name] = values.loc[valid]
    cleaned_data["Total Activity Load"] = (
        cleaned_data["Sales Count"] + cleaned_data["Redemption Count"]
    )
    validation = {
        "invalid_timestamps": int(invalid_timestamp.sum()),
        "invalid_sales": int(invalid_sales.sum()),
        "invalid_redemptions": int(invalid_redemptions.sum()),
        "negative_sales": int(negative_sales.sum()),
        "negative_redemptions": int(negative_redemptions.sum()),
        "excluded_rows": int((~valid).sum()),
    }
    return cleaned_data, validation


def render_overview(filtered, full_data, start_date, end_date, granularity):
    st.header("Overview dashboard")

    total_sales = filtered["Sales Count"].sum()
    total_redemptions = filtered["Redemption Count"].sum()
    average_activity = filtered["Total Activity Load"].mean()
    sales_redemption_ratio = (
        total_sales / total_redemptions if total_redemptions else None
    )
    highest_row = filtered.loc[filtered["Total Activity Load"].idxmax()]
    latest = full_data.loc[full_data["Timestamp"].idxmax()]

    metrics = st.columns(4)
    metrics[0].metric(
        "Historical ticket sales",
        f"{total_sales:,.0f}",
        help="Sum of recorded ticket sales in the selected date range.",
    )
    metrics[1].metric(
        "Historical redemptions",
        f"{total_redemptions:,.0f}",
        help="Sum of recorded ticket redemptions in the selected date range.",
    )
    metrics[2].metric(
        "Latest available timestamp",
        latest["Timestamp"].strftime("%Y-%m-%d %H:%M"),
    )
    metrics[3].metric(
        "Latest interval activity",
        f"{latest['Total Activity Load']:,.0f}",
        help="Sales plus redemptions at the latest recorded timestamp.",
    )

    metrics = st.columns(3)
    metrics[0].metric(
        "Average Activity per Interval",
        f"{average_activity:,.2f}",
        help="Average sales plus redemptions per valid recorded observation.",
    )
    metrics[1].metric(
        "Total Recorded Intervals",
        f"{len(filtered):,}",
        help="Number of valid recorded observations in the selected date range.",
    )
    metrics[2].metric(
        "Sales-to-Redemption Ratio",
        f"{sales_redemption_ratio:,.2f}"
        if sales_redemption_ratio is not None
        else "N/A",
        help="Total ticket sales divided by total ticket redemptions.",
    )
    st.caption(
        f"Highest recorded activity: {highest_row['Total Activity Load']:,.0f} "
        f"at {highest_row['Timestamp']}."
    )

    st.caption(
        "Activity is sales plus redemptions, not a passenger count or "
        "verified vessel-capacity utilization."
    )

    st.subheader("Activity at a glance")
    if granularity == "Hourly":
        recent = filtered.loc[
            filtered["Timestamp"] >= filtered["Timestamp"].max() - pd.Timedelta(days=90)
        ]
        period = recent["Timestamp"].dt.floor("h")
    elif granularity == "Daily":
        recent = filtered.loc[
            filtered["Timestamp"] >= filtered["Timestamp"].max() - pd.Timedelta(days=365)
        ]
        period = recent["Timestamp"].dt.floor("D")
    else:
        recent = filtered.loc[
            filtered["Timestamp"] >= filtered["Timestamp"].max() - pd.Timedelta(days=14)
        ]
        period = recent["Timestamp"]
    timeline = (
        recent.assign(Period=period)
        .groupby("Period", as_index=False)
        .agg(
            Sales=("Sales Count", "sum"),
            Redemptions=("Redemption Count", "sum"),
            Activity=("Total Activity Load", "sum"),
        )
    )
    st.caption(
        "Trend charts show the trailing 14 days at 15-minute granularity, "
        "90 days hourly, or 365 days daily; KPI totals use the full selected range."
    )
    st.plotly_chart(
        px.line(
            timeline,
            x="Period",
            y="Activity",
            title=f"Ticket activity — {granularity}",
            labels={"Period": "Time", "Activity": "Sales + redemptions"},
        ),
        width="stretch",
    )
    activity_comparison = timeline.melt(
        id_vars=["Period"],
        value_vars=["Sales", "Redemptions"],
        var_name="Ticket activity",
        value_name="Count",
    )
    st.plotly_chart(
        px.line(
            activity_comparison,
            x="Period",
            y="Count",
            color="Ticket activity",
            title="Sales and redemptions over time",
            labels={"Period": "Time", "Count": "Recorded tickets"},
        ),
        width="stretch",
    )

    st.subheader("Business insights")
    insight_columns = st.columns(3)
    month_activity = filtered.groupby(
        filtered["Timestamp"].dt.month_name()
    )["Total Activity Load"].mean()
    weekday_activity = filtered.groupby(
        filtered["Timestamp"].dt.day_name()
    )["Total Activity Load"].mean()
    hour_activity = filtered.groupby(
        filtered["Timestamp"].dt.hour
    )["Total Activity Load"].mean()

    insight_columns[0].markdown(
        f"**Highest-activity month: {month_activity.idxmax()}**  \n"
        f"Average activity was {month_activity.max():,.2f} per interval.  \n"
        "_Suggestion:_ review seasonal staffing and service plans for this month."
    )
    insight_columns[1].markdown(
        f"**Highest-activity weekday: {weekday_activity.idxmax()}**  \n"
        f"Average activity was {weekday_activity.max():,.2f} per interval.  \n"
        "_Suggestion:_ compare planned service levels for this weekday."
    )
    insight_columns[2].markdown(
        f"**Busiest hour: {int(hour_activity.idxmax()):02d}:00**  \n"
        f"Average activity was {hour_activity.max():,.2f} per interval.  \n"
        "_Suggestion:_ review ticketing and queue support around this hour."
    )

    days_in_range = (end_date - start_date).days + 1
    previous_start = start_date - timedelta(days=days_in_range)
    previous_end = start_date - timedelta(days=1)
    previous = full_data.loc[
        (full_data["Timestamp"].dt.date >= previous_start)
        & (full_data["Timestamp"].dt.date <= previous_end)
    ]
    st.markdown("**Sales compared with the previous equal-length period**")
    if previous.empty:
        st.info(
            "There are no recorded observations in the preceding "
            "equal-length period for comparison."
        )
    else:
        previous_sales = previous["Sales Count"].sum()
        change = total_sales - previous_sales
        if previous_sales == 0:
            st.info(
                "Previous-period sales were zero, so a percentage change "
                "cannot be calculated."
            )
        else:
            change_percent = change / previous_sales * 100
            direction = "increased" if change >= 0 else "decreased"
            st.write(
                f"Sales {direction} by {abs(change_percent):,.1f}% "
                f"({change:+,.0f} tickets) versus "
                f"{previous_start} to {previous_end}."
            )
            st.caption(
                "Suggestion: use this comparison as a planning signal; "
                "it does not explain the cause of the change."
            )

    p95 = filtered["Total Activity Load"].quantile(0.95)
    unusual = filtered[filtered["Total Activity Load"] >= p95].nlargest(
        3, "Total Activity Load"
    )
    st.markdown("**Unusually high recorded activity timestamps (P95+)**")
    st.dataframe(
        unusual[
            ["Timestamp", "Sales Count", "Redemption Count", "Total Activity Load"]
        ].rename(columns={"Total Activity Load": "Activity"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Suggestion: check these periods against operating records before "
        "changing service plans. High ticket counts are not confirmed incidents."
    )
    st.download_button(
        "Download high-activity records (P95+)",
        data=unusual.to_csv(index=False).encode("utf-8"),
        file_name="high_activity_records.csv",
        mime="text/csv",
    )
    summary = pd.DataFrame(
        [
            ("Source", st.session_state.get("data_source_name", "Uploaded dataset")),
            ("Report generated", datetime.now().astimezone().isoformat(timespec="seconds")),
            ("Selected date range", f"{start_date} to {end_date}"),
            ("Total ticket sales", total_sales),
            ("Total ticket redemptions", total_redemptions),
            ("Average activity per interval", average_activity),
            ("Recorded intervals", len(filtered)),
            ("Sales-to-redemption ratio", sales_redemption_ratio),
            ("Highest activity recorded", highest_row["Total Activity Load"]),
            ("Highest activity timestamp", highest_row["Timestamp"]),
            ("Metric definition", "Activity = ticket sales + ticket redemptions"),
            ("Metric limitation", "Activity is not a passenger count or capacity utilization"),
        ],
        columns=["Metric", "Value"],
    )
    st.download_button(
        "Download analysis summary",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="ferry_analysis_summary.csv",
        mime="text/csv",
    )


def render_trends(filtered):
    st.header("Trends and seasonality")

    trend_data = filtered.assign(
        Year=filtered["Timestamp"].dt.year,
        Month=filtered["Timestamp"].dt.month,
        MonthName=filtered["Timestamp"].dt.month_name(),
        Hour=filtered["Timestamp"].dt.hour,
        DayType=filtered["Timestamp"].dt.day_of_week.map(
            lambda day: "Weekend" if day >= 5 else "Weekday"
        ),
        Season=filtered["Timestamp"].dt.month.map(SEASONS),
    )
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    yearly = (
        trend_data.groupby("Year", as_index=False)
        .agg(
            Average_Activity=("Total Activity Load", "mean"),
        )
        .sort_values("Year")
    )
    yearly["YoY Activity Change (%)"] = (
        yearly["Average_Activity"].pct_change() * 100
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                yearly,
                x="Year",
                y="Average_Activity",
                title="Average activity by year",
                labels={
                    "Year": "Year",
                    "Average_Activity": "Average activity per interval",
                },
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.bar(
                yearly,
                x="Year",
                y="YoY Activity Change (%)",
                title="Year-over-year activity change",
                labels={"YoY Activity Change (%)": "Change (%)"},
                hover_data=["Average_Activity"],
            ),
            width="stretch",
        )
    st.caption(
        "YoY compares average activity per valid recorded interval with the "
        "prior calendar year; it is not adjusted for different coverage."
    )

    monthly = (
        trend_data.groupby(["Year", "Month", "MonthName"], as_index=False)
        .agg(Average_Activity=("Total Activity Load", "mean"))
        .sort_values(["Year", "Month"])
    )
    monthly["MonthName"] = pd.Categorical(
        monthly["MonthName"], categories=month_order, ordered=True
    )
    st.plotly_chart(
        px.line(
            monthly.sort_values(["Year", "Month"]),
            x="MonthName",
            y="Average_Activity",
            color="Year",
            markers=True,
            title="Same-month activity comparison across years",
            labels={
                "MonthName": "Month",
                "Average_Activity": "Average activity per interval",
            },
        ),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        weekday = (
            trend_data.groupby(["DayType"], as_index=False)
            .agg(Average_Activity=("Total Activity Load", "mean"))
        )
        st.plotly_chart(
            px.bar(
                weekday,
                x="DayType",
                y="Average_Activity",
                title="Weekday vs weekend activity",
                labels={"Average_Activity": "Average activity per interval"},
                color="DayType",
            ),
            width="stretch",
        )
    with right:
        hourly = (
            trend_data.groupby("Hour", as_index=False)
            .agg(Average_Activity=("Total Activity Load", "mean"))
        )
        st.plotly_chart(
            px.line(
                hourly,
                x="Hour",
                y="Average_Activity",
                markers=True,
                title="Activity profile by hour",
                labels={
                    "Hour": "Hour of day",
                    "Average_Activity": "Average activity per interval",
                },
            ),
            width="stretch",
        )

    seasonal = (
        trend_data.groupby("Season", as_index=False)
        .agg(Average_Activity=("Total Activity Load", "mean"))
    )
    seasonal["Season"] = pd.Categorical(
        seasonal["Season"],
        categories=["Winter", "Spring", "Summer", "Autumn"],
        ordered=True,
    )
    st.plotly_chart(
        px.bar(
            seasonal.sort_values("Season"),
            x="Season",
            y="Average_Activity",
            title="Seasonal activity comparison",
            labels={"Average_Activity": "Average activity per interval"},
            color="Season",
        ),
        width="stretch",
    )
    st.caption(
        "Winter groups December, January, and February. These comparisons "
        "describe recorded ticket activity and do not establish passenger loads."
    )


def render_anomalies(filtered):
    st.header("Unusual activity")
    st.info(
        "Threshold-based outliers are unusual recorded ticket counts, not "
        "confirmed operational incidents. Bulk recording or data-quality "
        "issues can also produce spikes."
    )

    activity = filtered["Total Activity Load"]
    p90 = activity.quantile(0.90)
    p95 = activity.quantile(0.95)
    p99 = activity.quantile(0.99)
    p10 = activity.quantile(0.10)
    q1 = activity.quantile(0.25)
    q3 = activity.quantile(0.75)
    upper_fence = q3 + 1.5 * (q3 - q1)
    anomalies = filtered.loc[
        (activity >= p95) | (activity > upper_fence)
    ].copy()
    anomalies["Severity"] = anomalies["Total Activity Load"].map(
        lambda value: (
            "High (P99+)" if value >= p99
            else "Elevated (P95+ or IQR)"
        )
    )

    legacy_columns = st.columns(2)
    legacy_columns[0].metric("High activity threshold (P90)", f"{p90:,.2f}")
    legacy_columns[1].metric("Low activity threshold (P10)", f"{p10:,.2f}")

    metric_columns = st.columns(3)
    metric_columns[0].metric("P95 threshold", f"{p95:,.2f}")
    metric_columns[1].metric("P99 threshold", f"{p99:,.2f}")
    metric_columns[2].metric("IQR upper fence", f"{upper_fence:,.2f}")
    st.caption(
        "IQR upper fence = Q3 + 1.5 × (Q3 − Q1). P95/P99 and IQR flags "
        "are calculated from individual observations in the selected range."
    )
    high_periods = filtered.loc[activity >= p90].nlargest(
        10, "Total Activity Load"
    )
    low_periods = filtered.loc[activity <= p10].nsmallest(
        10, "Total Activity Load"
    )
    high_low_columns = st.columns(2)
    high_low_columns[0].markdown("**Highest P90+ observations**")
    high_low_columns[0].dataframe(
        high_periods[
            ["Timestamp", "Total Activity Load"]
        ].rename(columns={"Total Activity Load": "Activity"}),
        width="stretch",
        hide_index=True,
    )
    high_low_columns[1].markdown("**Lowest P10− observations**")
    high_low_columns[1].dataframe(
        low_periods[
            ["Timestamp", "Total Activity Load"]
        ].rename(columns={"Total Activity Load": "Activity"}),
        width="stretch",
        hide_index=True,
    )

    if anomalies.empty:
        st.info("No observations exceeded the selected-range anomaly thresholds.")
        return

    normal = filtered.loc[~filtered.index.isin(anomalies.index)]
    normal_step = max(1, (len(normal) + 4999) // 5000)
    chart_data = pd.concat(
        [
            normal.iloc[::normal_step].assign(Severity="Typical (sampled)"),
            anomalies.assign(Severity=anomalies["Severity"]),
        ]
    ).sort_values("Timestamp")
    st.plotly_chart(
        px.scatter(
            chart_data,
            x="Timestamp",
            y="Total Activity Load",
            color="Severity",
            title="Typical activity and unusual spikes",
            labels={
                "Timestamp": "Recorded time",
                "Total Activity Load": "Sales + redemptions",
            },
            hover_data=["Sales Count", "Redemption Count"],
        ),
        width="stretch",
    )
    st.caption(
        "Typical observations are sampled for chart performance; every "
        "threshold-flagged observation is included."
    )
    st.subheader(f"Flagged observations ({len(anomalies):,})")
    st.dataframe(
        anomalies.sort_values("Total Activity Load", ascending=False)
        .head(100)[
            [
                "Timestamp",
                "Sales Count",
                "Redemption Count",
                "Total Activity Load",
                "Severity",
            ]
        ]
        .rename(columns={"Total Activity Load": "Activity"}),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download anomaly report",
        data=anomalies.sort_values(
            "Total Activity Load", ascending=False
        ).to_csv(index=False).encode("utf-8"),
        file_name="ferry_anomaly_report.csv",
        mime="text/csv",
    )
    if len(anomalies) > 100:
        st.caption("Showing the 100 highest-activity flagged observations.")


def render_data_quality(raw, mapping, validation, source_name):
    st.header("Data quality")
    st.caption(f"Data quality report for {source_name}.")
    timestamps = pd.to_datetime(raw[mapping["timestamp"]], errors="coerce")
    valid_timestamps = timestamps.dropna().sort_values().drop_duplicates()
    gaps = valid_timestamps.diff().dropna()
    long_gaps = gaps[gaps > pd.Timedelta(minutes=15)]
    estimated_missing_slots = 0
    irregular_gaps = 0
    for gap in long_gaps:
        number_of_intervals = gap / pd.Timedelta(minutes=15)
        if number_of_intervals.is_integer():
            estimated_missing_slots += int(number_of_intervals) - 1
        else:
            irregular_gaps += 1

    blank_counts = raw.apply(
        lambda column: column.astype("string").str.strip().eq("").sum()
    )
    missing_counts = (
        raw.isna().sum() + blank_counts
    ).rename("Missing or blank values").to_frame()

    first = valid_timestamps.min() if not valid_timestamps.empty else None
    last = valid_timestamps.max() if not valid_timestamps.empty else None
    st.markdown(
        f"**Data coverage:** {first if first is not None else 'N/A'} "
        f"to {last if last is not None else 'N/A'}"
    )

    quality_metrics = st.columns(3)
    quality_metrics[0].metric(
        "Missing or blank values",
        f"{int(missing_counts.iloc[:, 0].sum()):,}",
    )
    quality_metrics[1].metric("Duplicate records", f"{int(raw.duplicated().sum()):,}")
    quality_metrics[2].metric("Invalid timestamps", f"{int(timestamps.isna().sum()):,}")
    quality_metrics = st.columns(3)
    quality_metrics[0].metric(
        f"Invalid {mapping['sales']} values", f"{validation['invalid_sales']:,}"
    )
    quality_metrics[1].metric(
        f"Invalid {mapping['redemptions']} values",
        f"{validation['invalid_redemptions']:,}",
    )
    quality_metrics[2].metric(
        "Rows excluded from analysis",
        f"{validation['excluded_rows']:,}",
        help="Rows with invalid dates, invalid counts, or negative counts.",
    )
    quality_metrics = st.columns(3)
    quality_metrics[0].metric(
        f"Negative {mapping['sales']} values",
        f"{validation['negative_sales']:,}",
    )
    quality_metrics[1].metric(
        f"Negative {mapping['redemptions']} values",
        f"{validation['negative_redemptions']:,}",
    )
    quality_metrics[2].metric(
        "Estimated missing 15-min slots",
        f"{estimated_missing_slots:,}",
        help="Estimate from gaps between distinct valid timestamps.",
    )

    st.subheader("Missing values by column")
    st.dataframe(missing_counts, width="stretch")
    st.markdown(
        f"- Gaps longer than 15 minutes: **{len(long_gaps):,}**\n"
        f"- Irregular gaps not divisible by 15 minutes: **{irregular_gaps:,}**"
    )
    st.warning(
        "Time gaps are descriptive estimates only. Operating hours, "
        "service schedules, and seasonal closures are not available here, "
        "so gaps are not automatically treated as errors."
    )


@st.cache_resource(show_spinner="Loading the cached forecasting model...")
def load_forecast_state(dataset_path):
    data, quality = load_ticket_data(dataset_path)
    grid = regularize_intervals(data)
    artifact_path = PROJECT_DIR / "data" / "model_artifacts" / "forecaster.joblib"
    artifact, trained_now = train_or_load(
        dataset_path, artifact_path, rolling_folds=3
    )
    return grid, artifact, quality, trained_now


@st.cache_data(show_spinner=False)
def cached_forecast_horizons(source_digest, target):
    grid, artifact, _, _ = load_forecast_state(str(SOURCE_FILE))
    if artifact["source_digest"] != source_digest:
        raise ValueError("The forecast artifact does not match the source dataset.")
    return forecast_all_horizons(artifact, grid, target)


def horizon_label(steps):
    return f"{HORIZONS[steps]} ({steps * 15} min)"


def render_demand_forecasting(grid, artifact):
    st.header("Demand forecasting")
    st.caption(
        "Independent direct forecasts for ticket sales and redemptions. "
        "Forecasts start from the latest observed interval for the selected target."
    )
    target = st.selectbox("Forecast target", TARGET_COLUMNS, key="forecast_target")
    forecast = cached_forecast_horizons(artifact["source_digest"], target)
    forecast = classify_forecast(forecast, grid[target].dropna())

    chart = go.Figure()
    history = grid[target].dropna().tail(96)
    chart.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Observed (last 24 hours)",
            line={"color": "#64748b", "width": 2},
        )
    )
    chart.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["prediction"],
            mode="lines+markers",
            name="Forecast",
            line={"color": "#087e8b", "width": 3},
            marker={"size": 9},
        )
    )
    chart.update_layout(
        title=f"{target}: observed demand and short-term forecasts",
        xaxis_title="Timestamp",
        yaxis_title="Tickets per 15-minute interval",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(chart, width="stretch")

    horizon_columns = st.columns(4)
    for column, (_, row) in zip(horizon_columns, forecast.iterrows()):
        column.metric(
            horizon_label(row["horizon_steps"]),
            f"{row['prediction']:,.1f}",
            help=f"Forecast timestamp: {row['timestamp']}",
        )
    st.dataframe(
        forecast[
            [
                "horizon",
                "timestamp",
                "prediction",
                "model",
                "Demand Level",
                "Spike Warning",
            ]
        ].rename(
            columns={
                "horizon": "Horizon",
                "timestamp": "Forecast timestamp",
                "prediction": "Predicted tickets",
                "model": "Selected model",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Demand levels are relative to historical percentiles for this target. "
        "They are not passenger counts, capacity estimates, or safety limits."
    )

    st.subheader("Held-out actual vs predicted")
    test_horizon = st.selectbox(
        "Test horizon",
        list(HORIZONS),
        format_func=horizon_label,
        key="backtest_horizon",
    )
    test = artifact["targets"][target][test_horizon]["test_predictions"]
    if test.empty:
        st.info(
            "No held-out test predictions are available for this target and horizon."
        )
    else:
        chart_rows = test.sort_values("target_time").tail(1500)
        actual_vs_predicted = go.Figure()
        actual_vs_predicted.add_trace(
            go.Scatter(
                x=chart_rows["target_time"],
                y=chart_rows["actual"],
                mode="lines",
                name="Actual",
                line={"color": "#64748b"},
            )
        )
        actual_vs_predicted.add_trace(
            go.Scatter(
                x=chart_rows["target_time"],
                y=chart_rows["prediction"],
                mode="lines",
                name="Predicted",
                line={"color": "#e07a5f"},
            )
        )
        actual_vs_predicted.update_layout(
            title=f"Chronological held-out test — {target}, {HORIZONS[test_horizon]}",
            xaxis_title="Target timestamp",
            yaxis_title="Tickets per interval",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(actual_vs_predicted, width="stretch")
        st.caption(
            "The chart shows the latest 1,500 held-out test observations "
            "for readability; "
            "model metrics are calculated over the full test partition."
        )


def render_prediction_intervals(grid, artifact):
    st.header("Prediction intervals")
    st.caption(
        "Empirical, horizon-specific validation-residual intervals. "
        "These bounds are not confidence scores or guaranteed coverage."
    )
    target_column, horizon_column = st.columns(2)
    target = target_column.selectbox(
        "Forecast target", TARGET_COLUMNS, key="interval_target"
    )
    horizon = horizon_column.selectbox(
        "Forecast horizon",
        list(HORIZONS),
        format_func=horizon_label,
        key="interval_horizon",
    )
    result = cached_forecast_horizons(
        artifact["source_digest"], target
    ).loc[lambda rows: rows["horizon_steps"].eq(horizon)].iloc[0].to_dict()

    metric_columns = st.columns(5)
    metric_columns[0].metric("Point forecast", f"{result['prediction']:,.1f}")
    metric_columns[1].metric("80% lower bound", f"{result['lower_80']:,.1f}")
    metric_columns[2].metric("80% upper bound", f"{result['upper_80']:,.1f}")
    metric_columns[3].metric("95% lower bound", f"{result['lower_95']:,.1f}")
    metric_columns[4].metric("95% upper bound", f"{result['upper_95']:,.1f}")
    st.write(
        f"**Forecast time:** {result['timestamp']}  ·  "
        f"**Model:** `{result['model']}`  ·  "
        f"**Origin:** {result['origin']}"
    )

    history = grid[target].dropna().tail(96)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Observed (last 24 hours)",
            line={"color": "#64748b"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[
                result["origin"],
                result["timestamp"],
                result["timestamp"],
                result["origin"],
            ],
            y=[
                result["lower_95"],
                result["lower_95"],
                result["upper_95"],
                result["upper_95"],
            ],
            fill="toself",
            fillcolor="rgba(8, 126, 139, 0.12)",
            line={"color": "rgba(8, 126, 139, 0)"},
            name="95% prediction interval",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[
                result["origin"],
                result["timestamp"],
                result["timestamp"],
                result["origin"],
            ],
            y=[
                result["lower_80"],
                result["lower_80"],
                result["upper_80"],
                result["upper_80"],
            ],
            fill="toself",
            fillcolor="rgba(8, 126, 139, 0.24)",
            line={"color": "rgba(8, 126, 139, 0)"},
            name="80% prediction interval",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[result["timestamp"]],
            y=[result["prediction"]],
            mode="markers",
            name="Point forecast",
            marker={"color": "#087e8b", "size": 11},
        )
    )
    figure.update_layout(
        title=f"{target}: {HORIZONS[horizon]} forecast range",
        xaxis_title="Timestamp",
        yaxis_title="Tickets per interval",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")
    bounds = pd.DataFrame(
        [
            {
                "Interval": "80%",
                "Lower bound": result["lower_80"],
                "Forecast": result["prediction"],
                "Upper bound": result["upper_80"],
            },
            {
                "Interval": "95%",
                "Lower bound": result["lower_95"],
                "Forecast": result["prediction"],
                "Upper bound": result["upper_95"],
            },
        ]
    )
    st.dataframe(bounds, width="stretch", hide_index=True)
    st.caption(
        "Bounds are computed by adding validation-residual quantiles to the point "
        "forecast and clipping at zero, using the existing pipeline methodology."
    )


def render_model_performance(artifact):
    st.header("Model performance")
    target_column, horizon_column = st.columns(2)
    target = target_column.selectbox(
        "Evaluation target", TARGET_COLUMNS, key="performance_target"
    )
    horizon = horizon_column.selectbox(
        "Evaluation horizon",
        list(HORIZONS),
        format_func=horizon_label,
        key="performance_horizon",
    )
    item = artifact["targets"][target][horizon]
    metrics = item["test_metrics"]
    score_columns = st.columns(3)
    score_columns[0].metric("Held-out MAE", f"{metrics['MAE']:,.3f} tickets")
    score_columns[1].metric("Held-out RMSE", f"{metrics['RMSE']:,.3f} tickets")
    score_columns[2].metric(
        "Held-out MAPE",
        f"{metrics['MAPE']:,.2f}%" if pd.notna(metrics["MAPE"]) else "N/A",
    )
    st.caption(
        f"Selected model: **{item['model_name']}** · Validation begins "
        f"{item['validation_start']} · Held-out test begins {item['test_start']}."
    )
    st.info(
        "Supervised observations are split chronologically 70% train, 15% "
        "validation, and 15% test. Validation MAE selects a model; test metrics "
        "are reported only for the later held-out period."
    )

    validation = pd.DataFrame(item["candidate_metrics"])
    validation["Selected"] = validation["model"].eq(item["model_name"])
    st.subheader("Validation model comparison")
    st.dataframe(
        validation.rename(
            columns={
                "model": "Model",
                "MAE": "MAE (tickets)",
                "RMSE": "RMSE (tickets)",
                "MAPE": "MAPE (%)",
                "WAPE": "WAPE (%)",
                "sMAPE": "sMAPE (%)",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "MAPE excludes zero actual values. MAE and RMSE are in tickets per "
        "interval; "
        "no accuracy percentage is inferred from these error metrics."
    )

    rolling = pd.DataFrame(item["rolling_origin_metrics"])
    st.subheader("Expanding rolling-origin validation")
    if rolling.empty:
        st.info("No rolling-origin validation folds are stored in this artifact.")
    else:
        st.dataframe(
            rolling.rename(
                columns={
                    "fold": "Fold",
                    "model": "Model",
                    "validation_start": "Validation start",
                    "validation_end": "Validation end",
                    "MAE": "MAE (tickets)",
                    "RMSE": "RMSE (tickets)",
                    "MAPE": "MAPE (%)",
                    "WAPE": "WAPE (%)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    evaluation_csv = artifact["evaluation"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download complete model evaluation",
        data=evaluation_csv,
        file_name="ferry_model_evaluation.csv",
        mime="text/csv",
    )


def render_operational_insights(grid, artifact):
    st.header("Operational insights")
    st.caption(
        "Use these ticket-demand signals to support monitoring and readiness "
        "discussions; they do not prescribe capacity, departures, or staffing."
    )
    target = st.selectbox(
        "Demand series", TARGET_COLUMNS, key="insights_target"
    )
    observed = grid[target].dropna()
    forecast = classify_forecast(
        cached_forecast_horizons(artifact["source_digest"], target), observed
    )
    st.subheader("Short-term demand watch")
    st.dataframe(
        forecast[
            [
                "horizon",
                "timestamp",
                "prediction",
                "Demand Level",
                "Spike Warning",
                "Operational Note",
            ]
        ].rename(
            columns={
                "horizon": "Horizon",
                "timestamp": "Forecast timestamp",
                "prediction": "Predicted tickets",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "A spike warning means the forecast is at or above the historical P90 "
        "for this target; it is a relative threshold, not an incident alert."
    )

    by_hour = (
        grid[[target]]
        .dropna()
        .assign(Hour=lambda frame: frame.index.hour)
        .groupby("Hour", as_index=False)[target]
        .mean()
    )
    by_weekday = (
        grid[[target]]
        .dropna()
        .assign(Weekday=lambda frame: frame.index.day_name())
        .groupby("Weekday", as_index=False)[target]
        .mean()
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(
                by_hour,
                x="Hour",
                y=target,
                markers=True,
                title="Historical average by hour",
                labels={
                    target: "Tickets per observed interval",
                    "Hour": "Hour of day",
                },
            ),
            width="stretch",
        )
    with right:
        weekday_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        by_weekday["Weekday"] = pd.Categorical(
            by_weekday["Weekday"], categories=weekday_order, ordered=True
        )
        st.plotly_chart(
            px.bar(
                by_weekday.sort_values("Weekday"),
                x="Weekday",
                y=target,
                title="Historical average by weekday",
                labels={target: "Tickets per observed interval"},
            ),
            width="stretch",
        )

    threshold = observed.quantile(0.95)
    spikes = (
        grid.loc[grid[target] >= threshold, [target]]
        .nlargest(10, target)
        .rename(columns={target: "Tickets"})
        .reset_index()
        .rename(columns={"Timestamp": "Timestamp"})
    )
    st.subheader("Historical high-demand observations (P95+)")
    st.dataframe(spikes, width="stretch", hide_index=True)
    st.caption(
        "Consider checking high-demand timestamps against operating records. "
        "Ticket counts alone do not establish passenger loads or explain why "
        "demand changed; missing intervals are not treated as zero."
    )


st.title("Ferry Intelligence")
st.caption("Short-term ferry ticket demand forecasting and predictive decision support")
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f9fc; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf0;
        padding: 14px 16px;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] { background-color: #f1f5f9; }
    h1, h2, h3 { color: #17324d; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Ferry Intelligence")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Demand Forecasting",
            "Prediction Intervals",
            "Model Performance",
            "Operational Insights",
            "Trends",
            "Anomalies",
            "Data Quality",
        ],
        label_visibility="collapsed",
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    source_name = uploaded_file.name
    forecast_source = False
    try:
        raw_data = read_csv_bytes(uploaded_file.getvalue())
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        st.error(f"Could not read `{source_name}` as a CSV file: {error}")
        st.stop()
else:
    if not SOURCE_FILE.exists():
        st.error(
            f"The required source dataset was not found at `{SOURCE_FILE}`. "
            "Restore the repository data file or upload a CSV to use the "
            "historical analytics pages."
        )
        st.stop()
    source_name = SOURCE_FILE.name
    forecast_source = True
    try:
        raw_data = read_csv_bytes(SOURCE_FILE.read_bytes())
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        st.error(f"Could not read the source dataset `{source_name}`: {error}")
        st.stop()

raw_data.columns = [str(column).strip() for column in raw_data.columns]
if raw_data.empty:
    st.error(f"`{source_name}` has no data rows.")
    st.stop()
if not raw_data.columns.is_unique:
    st.error(
        "Column names are not unique after trimming whitespace. "
        "Rename duplicate columns in the CSV and upload it again."
    )

detected = {
    kind: detect_column(raw_data.columns, kind)
    for kind in ["timestamp", "sales", "redemptions"]
}

with st.sidebar:
    st.subheader("Column mapping")
    column_options = ["-- Select a column --"] + raw_data.columns.tolist()
    mapping = {}
    for kind, label in [
        ("timestamp", "Timestamp / date"),
        ("sales", "Ticket sales"),
        ("redemptions", "Ticket redemptions"),
    ]:
        default_column = detected[kind]
        default_index = (
            column_options.index(default_column)
            if default_column in column_options
            else 0
        )
        mapping[kind] = st.selectbox(
            label,
            column_options,
            index=default_index,
            key=f"mapping_{kind}",
        )

missing_mappings = [
    label
    for kind, label in [
        ("timestamp", "timestamp/date"),
        ("sales", "ticket sales"),
        ("redemptions", "ticket redemptions"),
    ]
    if mapping[kind] == "-- Select a column --"
]
selected_columns = [
    mapping[kind]
    for kind in ["timestamp", "sales", "redemptions"]
    if mapping[kind] != "-- Select a column --"
]
if missing_mappings:
    st.error(
        "Could not detect columns for "
        f"{', '.join(missing_mappings)}. Select the matching columns in "
        "the sidebar to continue."
    )
    st.stop()
if len(set(selected_columns)) != len(selected_columns):
    st.error(
        "Timestamp, sales, and redemption mappings must use three different "
        "columns. Update the column mapping in the sidebar."
    )
    st.stop()

df, validation = prepare_data(raw_data, mapping)
if df.empty:
    st.error(
        "No valid rows remain after validation. Check the selected columns "
        "and correct invalid dates/counts or negative values in the CSV."
    )
    st.write(
        {
            "Invalid timestamps": validation["invalid_timestamps"],
            "Invalid sales values": validation["invalid_sales"],
            "Invalid redemption values": validation["invalid_redemptions"],
            "Negative sales values": validation["negative_sales"],
            "Negative redemption values": validation["negative_redemptions"],
        }
    )
    st.stop()

st.caption(
    f"Dataset: {source_name} | "
    f"{len(df):,} valid rows"
)
if validation["excluded_rows"]:
    st.warning(
        f"{validation['excluded_rows']:,} row(s) were excluded because their "
        "timestamp or ticket counts were blank/invalid, or their counts were "
        "negative. See Data Quality for details."
    )

st.session_state["data_source_name"] = source_name

with st.sidebar:
    st.subheader("Date filters")
    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    granularity = st.selectbox(
        "Overview chart interval",
        ["15-minute", "Hourly", "Daily"],
    )
    st.caption("Date filters apply to Overview, Trends, and Anomalies.")

if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.info("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates
if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

filtered = df.loc[
    (df["Timestamp"].dt.date >= start_date)
    & (df["Timestamp"].dt.date <= end_date)
].copy()
if filtered.empty and page in {"Overview", "Trends", "Anomalies"}:
    st.warning("No recorded observations were found for the selected dates.")
    st.stop()

if page == "Overview":
    render_overview(filtered, df, start_date, end_date, granularity)
elif page in {
    "Demand Forecasting",
    "Prediction Intervals",
    "Model Performance",
    "Operational Insights",
}:
    if not forecast_source:
        st.info(
            "Forecasting is available for the committed Toronto Island Ferry "
            "dataset. Uploaded CSV files remain available for historical "
            "analysis and are not silently substituted into the trained models."
        )
    else:
        try:
            grid, artifact, quality, trained_now = (
                load_forecast_state(str(SOURCE_FILE))
            )
        except (FileNotFoundError, ImportError, OSError, ValueError) as error:
            st.error(
                f"Could not load forecasting data or model artifacts: {error}"
            )
            st.stop()
        st.caption(
            f"Latest forecast source observation: {artifact['latest_observation']} · "
            f"{quality['observed_rows']:,} valid observed intervals"
        )
        if trained_now:
            st.warning(
                "A matching pre-trained model artifact was not available, so the "
                "existing chronological training pipeline generated one for this dataset."
            )
        if page == "Demand Forecasting":
            render_demand_forecasting(grid, artifact)
        elif page == "Prediction Intervals":
            render_prediction_intervals(grid, artifact)
        elif page == "Model Performance":
            render_model_performance(artifact)
        else:
            render_operational_insights(grid, artifact)
elif page == "Trends":
    render_trends(filtered)
elif page == "Anomalies":
    render_anomalies(filtered)
elif page == "Data Quality":
    render_data_quality(raw_data, mapping, validation, source_name)

if not filtered.empty:
    st.subheader("Export filtered data")
    st.download_button(
        label="Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="ferry_filtered_analysis.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    f"Source: {source_name}. Sales and "
    "redemptions are separate ticket events; this dataset does not provide "
    "sailing-level passenger counts, vessel capacity, schedules, or operating "
    "hours. Activity is not passenger count or capacity utilization."
)
