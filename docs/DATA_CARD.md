# Data card

UCI Bike Sharing daily file: 731 observations from Washington, D.C. in 2011–2012. Target: daily count `cnt`. Same-day `casual` and `registered` sum to the target and are deliberately excluded. Actual target-day weather is excluded because it is unavailable at forecast time; holiday status is assumed known.

## Source and citation

[Original dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894

## License

UCI lists the dataset under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the creators and source when reusing data or derived figures. The repository's MIT license applies to its code, not to the dataset.

## Integrity and storage

```text
{
  "url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
  "sha256": "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
}
```

For downloaded datasets, every run verifies archive bytes against this SHA-256 before parsing. The downloader has a 60-second timeout and 20 MB response cap. It reads named members without extracting filesystem paths. Raw files stay in the ignored `data/raw/` cache. To recover from a corrupted cache, remove `data/raw/source.zip`, verify the upstream source, then rerun training. Do not silently replace the expected hash after a source change.

## Processing and sampling

The first 28 days supply lag history, leaving 703 feature rows. The final 120 days are reserved for testing. Four expanding-window folds, each with a 60-day validation window, select among scaled ridge, histogram gradient boosting and a seven-day seasonal-naive baseline. Lags use shifts of 1, 7, 14 and 28 days; rolling averages use `shift(1)` before rolling. The final fitted model remains fixed during testing. Predictions for each next day may use all counts observed by the preceding day, including earlier test days.

## Representation and privacy

Only two years from one city are available. Growth, unusual weather, closures and holidays create distribution changes. No weather-forecast archive is used. Linear trend extrapolation can become implausible far outside the training period; the sample 2013 date demonstrates the interface, not present-day demand. A multi-day recursive forecast requires a different evaluation design.

No private account data, API credentials, or scraped personal profiles are used. Synthetic fixtures exist only in tests and manually constructed examples; published metrics come from the cited dataset.
