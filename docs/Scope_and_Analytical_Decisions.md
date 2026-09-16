# Project Scope and Analytical Decisions

## Bankruptcy Risk Prediction Project — agreed team specification

### Prediction date

The 10-K filing date from EDGAR, matched on CIK; where unavailable, fiscal year end plus 90 days. Fiscal year end is not used, because financials are published two to three months later and anchoring there would give the model information no analyst could have had. Days late relative to the statutory deadline is kept as a feature, since late filing is observable at the prediction date and is itself a distress signal.

### Outcome window

Label = 1 if a qualifying petition date falls within 365 days of the prediction date. A firm-year is kept only if its full window closes by 31 December 2022, making 31 December 2021 the last usable prediction date; the BRD stops there, so later windows are unobservable and are dropped rather than coded zero.

### Event definition

Positives are BRD cases by DateFiled, pooling Chapter 11 and Chapter 7 and including involuntary petitions. Out-of-court restructurings, distressed exchanges, take-unders, and delistings without a filing are coded zero; this biases measured performance downward, since some false positives will be genuinely distressed firms that resolved outside court, and we will say so in the write-up. Firms are censored after their first filing, and firm-years on or after a filing date are dropped. A zero means no filing was observed; an exclusion means none could have been.

### Eligible universe

Screens are applied identically to positives and negatives, which is what licenses treating absence from the BRD as a valid zero. Included: U.S.-incorporated 10-K or Form 10 filers, excluding SIC 6000–6999 and 20-F filers, with total assets of at least $100 million in 1980 dollars inflated by CPI-U (computed per year, roughly $355 million by 2022), at least three prior annual reports as of the prediction date, and non-missing assets, liabilities, and revenue. Sample begins with fiscal year 1993. Utilities are retained with an industry indicator rather than excluded.

### Train, validation, and test periods

Splits are by prediction-date year, never random. Development covers 1994–2016 using expanding-window walk-forward cross-validation, preferred over a single validation block because 2012–2016 holds too few events to tune on. The 2017–2021 holdout is touched once, at the end, and includes the 2020 default wave as an out-of-regime stress test. Standard errors are clustered by firm, and all preprocessing is fit on training folds only.

### To confirm in office hours

- Whether event scarcity in 2012–2016 justifies walk-forward CV over a three-block split.
- Whether late 10-K filing is a feature or contamination.
- Whether pooling Chapter 7 with Chapter 11 is acceptable given it assumes one shared hazard process.
