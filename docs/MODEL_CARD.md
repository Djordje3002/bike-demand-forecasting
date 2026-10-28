# Model card — Bike Demand Forecasting

Version: 1.0.0 · Experiment seed: 42 · Selected model: `scaled_ridge`

## Intended use

Forecast the next day's total bike rentals using historical demand and known calendar information. This models an operational planning question: how much demand should a bike-sharing operator prepare for tomorrow after today's demand is observed?

The intended audience is engineers reviewing a reproducible ML portfolio and learners inspecting model selection, evaluation and inference. The project has no claimed deployment history or customer results.

## Training and evaluation

The first 28 days supply lag history, leaving 703 feature rows. The final 120 days are reserved for testing. Four expanding-window folds, each with a 60-day validation window, select among scaled ridge, histogram gradient boosting and a seven-day seasonal-naive baseline. Lags use shifts of 1, 7, 14 and 28 days; rolling averages use `shift(1)` before rolling. The final fitted model remains fixed during testing. Predictions for each next day may use all counts observed by the preceding day, including earlier test days.

| Model | mae | rmse | r2 |
| --- | ---: | ---: | ---: |
| `seasonal_naive` | 1342.2667 | 1911.1827 | 0.0640 |
| `scaled_ridge` | 931.6062 | 1298.9894 | 0.5676 |
| `hist_gradient_boosting` | 1382.2933 | 1878.0498 | 0.0961 |


## Interpretation

Ridge wins the backtests and reduces test MAE from 1,342 to 932 rentals/day versus the seven-day baseline. The boosted trees generalize worse in this setup. This is a sequence of one-day-ahead predictions with actual past counts, not a 120-day forecast made at one origin. The difficult late-2012 period remains visible in the chart.

## Limitations

Only two years from one city are available. Growth, unusual weather, closures and holidays create distribution changes. No weather-forecast archive is used. Linear trend extrapolation can become implausible far outside the training period; the sample 2013 date demonstrates the interface, not present-day demand. A multi-day recursive forecast requires a different evaluation design.

## Artifacts and maintenance

Run the training command to rebuild local artifacts and reports. Load only your own trusted local artifacts. For any data or model change, rerun training, compare with the baseline under the same split protocol, and inspect failure cases. A changed dataset hash or split constitutes a different experiment and must be documented as such. Fresh labeled data is required to estimate real deployment performance.
