# Data dictionary

## Source fields

| Field | Type | Meaning |
|---|---|---|
| `_id` | Integer | Source row identifier; not used as a model feature |
| `Timestamp` | Datetime | End time of a 15-minute ticket-count interval |
| `Sales Count` | Non-negative count | Tickets sold in the interval |
| `Redemption Count` | Non-negative count | Tickets redeemed in the interval |

## Derived fields

| Field | Meaning |
|---|---|
| `Total Activity Load` | Sales plus redemptions; an activity proxy, not a passenger or occupancy count |
| Lag features | Counts available at the forecast origin at the specified 15-minute offset |
| Rolling statistics | Statistics of recorded historical counts through the forecast origin |
| Calendar features | Known time-of-day, weekday, month, day, and weekend descriptors |
| Forecast | Direct point prediction for one target and one forecast horizon |
| Prediction interval | Empirical range based on validation residual quantiles; not a confidence score |
| Demand level | Relative class based on historical P50/P75/P90 ticket-count thresholds |

Sales and redemptions are separate events and should not be interpreted as
unique passengers. The dataset does not include vessel occupancy or capacity.
