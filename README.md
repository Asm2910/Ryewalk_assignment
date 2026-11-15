# SBA 7(a) Data Science Case Study

Two analyses of the SBA FOIA 7(a) loan file (status as of 31 March 2026):

| | Question | Data window | Deliverables |
|---|---|---|---|
| **Assignment 1** | Small Business Lending Companies (SBLCs): who they are, who is growing, who is lending well, which to watch | Approvals FY2010-FY2019 | Notebooks, report, deck |
| **Assignment 2** | SBA bad-loan prediction: which loans are likely to default, what drives default, and how to use the model | Approvals FY2014-FY2019 | Notebooks, report, deck |

All notebooks are saved with their outputs. The written report and the presentation for each assignment are in [`docs/`](docs/).

---

## Repository layout

```
.
├── common/
│   └── data.py                  # shared loader and cleaner for the SBA FOIA 7(a) file
├── assignment1/
│   ├── notebooks/               # notebooks 0-4 (run in order)
│   ├── src/                     # script versions of the SBLC identification and growth/credit steps
│   └── outputs/                 # CSV results used in the report and deck
├── assignment2/
│   ├── notebooks/               # 01, 02, 03, 03b, 04 (run in order)
│   └── outputs/                 # metrics, feature groups, tiers, model file and settings
├── data/
│   ├── raw/                     # SBA FOIA file goes here (git-ignored)
│   ├── external/                # optional outside data for notebook 03b (git-ignored)
│   └── processed/               # cleaned and model-ready tables (git-ignored, rebuilt by the notebooks)
├── docs/                        # reports (.docx) and presentations (.pptx)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Asm2910/Ryewalk_assignment.git
cd Ryewalk_assignment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Data.** Download the SBA FOIA 7(a) file for FY2010-FY2019 (as of 31 March 2026) from the SBA open-data site and save it as:

```
data/raw/foia-7a-fy2010-fy2019-asof-260331.csv
```

The raw and processed data are not committed because of their size.

---

## Assignment 1: SBLC competitive landscape

**Questions.** Who are the SBLCs? Who is growing? Who is lending well? Which two or three are worth watching? What else stands out?

| Notebook | What it does |
|---|---|
| 0 | Loads and cleans the file (`common/data.py`), exploratory statistics, saves `data/processed/loans_clean.parquet` |
| 1 | Identifies SBLCs: hard gate (no FDIC or NCUA number, not a government entity), point score, match to published SBLC rosters, tiers |
| 2 | Growth: share of 7(a) dollars by year, three-year windows, concentration (HHI), loan-size shift |
| 3 | Credit quality: charge-off rates, observed vs expected (matched on approval year and loan size), confidence intervals, trend |
| 4 | Watchlist (size, growth and credit flags, tested across 27 threshold settings) and other insights |

**Key decisions**

- Cleaning: removed exact duplicates and loans that were cancelled or never funded (`CANCLD`, `COMMIT`); 545,751 rows became 478,650 loans.
- Bad loan: any loan with a gross charge-off amount above zero. The status label alone misses about 21% of losses.
- Lender identity: `bankname`.
- Fair credit comparison: each lender's charge-off count is compared with the count expected for loans of the same approval year and size band (observed ÷ expected), because losses arrive late and SBLC loan sizes differ widely.

**Headline results**

- Twelve SBLCs: 11,465 loans, about $6.0bn; all twelve appear on at least one published SBLC roster.
- SBLC share of 7(a) dollars rose from 2.7% to 4.6%; SBLC lending grew 2.87 times against 1.68 times for the market.
- SBLC charge-off rate 15.5% against 7.6% for other lenders, but one lender (Velocity) accounts for 75% of SBLC charge-offs; without it the rate is 6.9%.
- Lenders to watch: Newtek, Harvest and Readycap.

---

## Assignment 2: SBA bad-loan prediction

**Goal.** Flag loans likely to default using only information a lender has at approval, to support approval decisions, pricing, monitoring and collections.

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Part A: data quality, distributions, default patterns and trends, time to loss |
| `02_features.ipynb` | Part B: target, time-based split, leakage audit, 97 features, missing values, outliers, encoding, imbalance; writes the model tables |
| `03_models.ipynb` | Part C: four models compared, validation, tuning, ablation studies, feature selection, final test, precision / recall / confusion matrix, risk deciles |
| `03b_experiments.ipynb` | Experiments: undersampling, other target definitions, BLS state unemployment, an older SBA snapshot (proof of the term-field leak, borrower and lender history, more training years) |
| `04_explain_and_decide.ipynb` | SHAP risk drivers, recalibration, risk tiers, cost-based review rule, pricing view |

Run in order: 01, 02, 03, 03b, 04.

**Key decisions**

| Decision | Choice | Why |
|---|---|---|
| Target | `bad = 1` if gross charge-off amount > 0 | The status label misses 2,775 loans that lost money ($166m); 1,985 of them are marked "paid in full" |
| Check on the target | Charged off within 60 months of approval | Gives every year the same window; the model scores AUC 0.764 against it |
| Split | Train FY2014-17, tune FY2018, test once on FY2019 | The model will always score future loans |
| Leakage | Only fields known at approval | The loan-term field is rewritten after a loan goes bad (61% of loans that went bad between the 2022 and 2026 snapshots had their term changed); with it the model scores AUC 0.97, without it 0.75 |
| Business age | Regrouped (Startup / Under 2y / Existing 2y+ / Unanswered) | SBA changed its categories in FY2018 |
| Encoding | Smoothed, out-of-fold history for industry and lender | Too many levels to one-hot; avoids each loan seeing its own outcome |
| Imbalance (7.6% bad) | No resampling | Tested: undersampling does not improve ranking and distorts probabilities |
| Model | LightGBM, 30 features | Best ranking and calibration on FY2018 and FY2019 |

**Results on FY2019 (never used for training or tuning)**

| Model | AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic regression | 0.740 | 0.186 | 0.217 |
| Random forest | 0.750 | 0.187 | 0.204 |
| LightGBM, 97 features | 0.754 | 0.198 | 0.058 |
| **LightGBM, 30 features (chosen)** | **0.752** | **0.198** | **0.058** |

- The riskiest 20% of loans contain 51% of all bad loans (precision 16.8% against 6.6% at random).
- Risk tiers: tier A (42% of loans) 2.0% bad; tier E (5% of loans) 25.5% bad.
- Strongest drivers: interest rate relative to that year's market, the lender's track record, the industry's track record; small, unsecured, variable-rate loans are riskier.
- Cost-based review rule (illustrative $500 review cost, 20% of loss avoided): net benefit $8.8m on FY2019, against $4.2m when ranking by probability alone.

**Limitations.** The public file has no borrower financials (credit score, cash flow, debt) and contains only approved loans, which caps AUC near 0.75. The model is meant to prioritise review, pricing and monitoring, not to approve or decline loans on its own.

**Data for notebook 03b (optional).** Place the three SBA FOIA files dated 31 December 2022 in `data/external/`:

```
foia-7afy1991-fy1999-asof-221231.csv
foia-7afy2000-fy2009-asof-221231.csv
foia-7afy2010-fy2019-asof-221231.csv
```

The BLS state unemployment series is downloaded automatically on first run (from the `jjchern/laus` GitHub data package) into the same folder.

---

## Reports and presentations

| File | Contents |
|---|---|
| `docs/SBLC_Assignment1_Findings.docx` | Assignment 1 report: method, results, sources, limitations |
| `docs/SBLC_Assignment1_Deck.pptx` | Assignment 1 presentation |
| `docs/Bad_Loan_Assignment2_Report.docx` | Assignment 2 report: exploration, features, models, drivers, business use, limitations |
| `docs/Bad_Loan_Assignment2_Deck.pptx` | Assignment 2 presentation (15 slides) |
