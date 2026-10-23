<<<<<<< HEAD
# Ferry Operational Analytics

A Streamlit dashboard for exploring ticket sales and redemptions from ferry
services. It can use the included Toronto Island Ferry dataset or a CSV you
upload.

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Alternatively, install the packages in `requirements.txt` in your Python
environment before running the command.

## Use another ferry dataset

1. Open the dashboard and choose a CSV with one observation per row.
2. Select the file using **Upload CSV** in the sidebar.
3. The dashboard attempts to detect the timestamp, ticket sales, and ticket
   redemption columns. Review or change these selections in **Column mapping**.
4. Rows with invalid or blank dates/counts, or negative counts, are excluded
   from analysis and reported on the **Data Quality** page.
5. Select a date range and navigate between Overview, Trends, Anomalies, and
   Data Quality. The filtered data, P95+ activity records, anomaly report, and
   analysis summary can be downloaded from the dashboard.

If no file is uploaded, the Toronto Island Ferry CSV is used as the demo
dataset. The original uploaded columns are retained in the filtered CSV
download, alongside the normalized timestamp, sales, redemption, and activity
columns.

## Metric notes

- **Activity** is ticket sales plus ticket redemptions in a recorded row.
- Average activity is calculated per valid observation, not per scheduled
  interval.
- Ticket sales and redemptions may happen at different times. They do not
  directly represent unique passengers or vessel occupancy.
- Missing time intervals are estimates; operating hours and seasonal closures
  are not known, so gaps are not automatically classified as errors.
- Anomaly thresholds flag unusual recorded counts, not confirmed operational
  incidents.
=======
# Ferry-Operational-Analytics
Data analytics project for exploring ferry operations, passenger demand, schedules, and operational performance.
>>>>>>> 2f2694d018301ba26f6a24179971be907e239289
