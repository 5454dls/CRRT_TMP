# Cumulative operating time and early TMP surge during CRRT — analysis code

Code for:

> *Cumulative operating time and early transmembrane pressure surge in relation to mortality during continuous renal replacement therapy*
> Dong-Seop Kim, Inyong Jeong, Nam-Jun Cho, Jin-Hyun Park, Yeongmin Kim, MyeongGyun Jang, Hwamin Lee, Hyo-Wook Gil.
> *Scientific Reports* (under revision as of this release).

This repository contains the statistical analysis pipeline. It does not contain patient data.
See [Data access](#data-access) below for how to obtain the source databases.

## What the study does

The paper asks whether two properties of a CRRT circuit carry information about patient mortality,
beyond their role as circuit-maintenance signals:

1. **Cumulative operating time** — how long the circuit has been running, analyzed as a
   time-varying exposure. Confirmatory analysis, MIMIC-IV with a multicenter check in eICU.
2. **Early transmembrane pressure (TMP) surge** — a rise of at least 100 mmHg within 12 hours of
   a patient's baseline TMP. Exploratory analysis, MIMIC-IV only (TMP is not available in eICU).

Cumulative operating time was not associated with mortality in either database. An early TMP surge
was associated with 28-day mortality in an adjusted model, but the association did not survive
false discovery rate correction and rests on a small exposed group (46 of 862 patients).

## Repository contents

```
README.md            this file
__init__.py          package marker
common.py            shared definitions: imports, constants, and the ~40 analysis functions
                     (SOFA scoring, MICE imputation, Cox model wrappers, effect-size and
                     multiplicity-correction helpers, table/figure builders)
analysis.ipynb       the full analysis, organized into 8 labeled sections (data loading through
                     revision-response sensitivity analyses). Each code cell is tagged with the
                     index it held in the original submission notebook, so a reviewer can trace
                     any number in the paper back to the exact cell that produced it
```

This is a reassembly of the notebook used to produce the submitted manuscript.

## What each part of the notebook produces

| Notebook section | Produces |
|---|---|
| 1. Data loading | Raw cohort extraction from MIMIC-IV / eICU |
| 2. Cohort construction | Analytic cohorts, sepsis flag, admission-year grouping, Figure 1 (selection flow) |
| 3. Primary analysis — Tier 1 | Table 2 rows for cumulative operating time; Figure 2 (forest plot); Supplementary Table S2 |
| 4. Primary analysis — Tier 2 | The primary early-surge hazard ratio; Figure 3; Bayes factor; E-value |
| 5. Sensitivity & robustness | Supplementary Tables S3–S6 (alternative surge definitions, landmark analyses, subgroup and complete-case checks); Supplementary Figures S1–S4 |
| 6. Figures | Figure 1–3 and Supplementary Figures S1–S4 rendering |
| 7. Table reproduction & QC | Table 1, internal consistency checks (these cells do not compute new results — they re-display or cross-check values computed earlier) |
| 8. Revision-response analyses | Cluster-robust bootstrap, restricted cubic spline, measurement-intensity check, inverse-probability-of-observation weighting, and the m=20-imputation confirmation of Table 2 |

## Data access

The source databases, MIMIC-IV (v2.2) and eICU Collaborative Research Database, are **not included
in this repository**. Both are PhysioNet credentialed-access datasets and redistribution is
prohibited by their data use agreements.

1. Create a PhysioNet account and complete the required CITI training
   (https://physionet.org/).
2. Sign the data use agreement for MIMIC-IV and for eICU-CRD and request access.
3. Once approved, place the downloaded data anywhere on your machine and point the code at it
   with two environment variables:

```bash
export MIMIC4_PATH="/path/to/mimic-iv-2.2"
export EICU_PATH="/path/to/eicu-crd"
```

The original notebook, before this release, referenced a fixed internal file-server path instead
of these environment variables. That is the only substantive text change made during reassembly;
it does not affect any computation, only where the code looks for input files.

## Environment

- **Python 3.11.15** — the version reported by the kernel that produced the submitted results.
- Packages (no pinned versions are shipped; the list below is what the analysis imports):

  | Package | Used for |
  |---|---|
  | `numpy`, `pandas` | data handling |
  | `matplotlib` | figures |
  | `scipy` | statistical distributions and tests |
  | **`lifelines` (>= 0.30)** | Cox and time-varying Cox models |
  | `scikit-learn` | `IterativeImputer` (experimental API), `BayesianRidge`, `LogisticRegression`, `roc_auc_score` |
  | `statsmodels` | `multipletests` (Benjamini-Hochberg FDR) |

## What is and is not in this repository

| Included | Not included |
|---|---|
| Analysis code (`src/`, `notebooks/analysis.ipynb`) | MIMIC-IV and eICU source data (PhysioNet credentialed access, redistribution prohibited) |
| This README | Patient-level or cohort-intermediate files (`cache_final/`, `figs/`, any `.parquet`) |
| — | Prior notebook execution outputs — all cell outputs were cleared before release |
| — | Internal file-server paths or hostnames |

A file-level scan for internal paths, IP addresses, credentials, and non-empty notebook outputs
was run before release. The scanning script is a development-time check and is not part of the
analysis, so it is not included here.

## License

**MIT License. Copyright (c) 2026 the authors of the associated manuscript.**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## How to cite

Please cite the associated manuscript once it is published. Until then, cite this repository by
its URL. On acceptance the repository will be archived on Zenodo and a citable DOI will be added
here.

## Code availability statement (for the manuscript)

> The analysis code is available at https://gitfront.io/r/inyong/u4gFr7V3yUcQ/CRRT-TMP/ and, on
> acceptance, at https://github.com/5454dls/CRRT_TMP with a Zenodo DOI added on deposit. The MIMIC-IV and eICU
> databases are publicly available through PhysioNet (https://physionet.org/) after completion of
> required training and a data use agreement; they are not redistributed with this repository.
