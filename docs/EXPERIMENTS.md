# Experiment protocol

## Reproduce

1. Install the project in a fresh Python 3.11 environment with `python -m pip install -e ".[dev]"`.
2. Run `python -m bike_forecast train` from the repository root.
3. Inspect `reports/metrics.json` and the generated figures. The report records seed, Python version, installed library versions and dataset source information.
4. Run `python -m bike_forecast predict --input examples/sample.json` to exercise the trained model.
5. Run `make check test` to verify the contracts.

## Selection rule

The first 28 days supply lag history, leaving 703 feature rows. The final 120 days are reserved for testing. Four expanding-window folds, each with a 60-day validation window, select among scaled ridge, histogram gradient boosting and a seven-day seasonal-naive baseline. Lags use shifts of 1, 7, 14 and 28 days; rolling averages use `shift(1)` before rolling. The final fitted model remains fixed during testing. Predictions for each next day may use all counts observed by the preceding day, including earlier test days.

## Reading the result

Ridge wins the backtests and reduces test MAE from 1,342 to 932 rentals/day versus the seven-day baseline. The boosted trees generalize worse in this setup. This is a sequence of one-day-ahead predictions with actual past counts, not a 120-day forecast made at one origin. The difficult late-2012 period remains visible in the chart.

The committed results are one seeded benchmark run. They do not establish confidence intervals, statistical significance, robustness across random seeds or production behavior. CV standard deviations summarize folds, not uncertainty on future production metrics. Do not tune settings against the final test set; use new validation experiments and reserve a fresh holdout if making repeated test-driven changes.

## Notebook

`notebooks/01_review_results.ipynb` reads committed reports and displays the evaluation figure. It is executed and can be viewed directly on GitHub without training. To rerun it locally, install `python -m pip install -e ".[notebook]"` and select that environment in your notebook editor. Training logic remains in the package so notebook execution cannot silently diverge from the command-line experiment.

## Resource boundaries

All examples run on CPU without cloud accounts or paid APIs. The default dataset archives are small, and raw data is cached locally. The CNN uses two CPU threads; the wine forest uses two workers. Reports and small images are committed, while model binaries, raw archives and environment folders are ignored. CI has a 15-minute execution limit.
