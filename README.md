# Cumulative operating time and early transmembrane pressure surge in relation to mortality during continuous renal replacement therapy

> Dong-Seop Kim, Inyong Jeong, Nam-Jun Cho, Jin-Hyun Park, Yeongmin Kim, MyeongGyun Jang, Hwamin Lee, Hyo-Wook Gil.
> 
> *Scientific Reports* (under revision as of this release).

## What the study does

The paper asks whether two properties of a CRRT circuit carry information about patient mortality,
beyond their role as circuit-maintenance signals:

1. **Cumulative operating time**: how long the circuit has been running, analyzed as a
   time-varying exposure. Confirmatory analysis, MIMIC-IV with a multicenter check in eICU.
2. **Early transmembrane pressure (TMP) surge**: a rise of at least 100 mmHg within 12 hours of
   a patient's baseline TMP. Exploratory analysis, MIMIC-IV only (TMP is not available in eICU).

Cumulative operating time was not associated with mortality in either database. An early TMP surge
was associated with 28-day mortality in an adjusted model, but the association did not survive
false discovery rate correction and rests on a small exposed group (46 of 862 patients).

## Contents

```
README.md            this file
common.py            imports, constants, and the analysis functions (SOFA scoring, MICE
                     imputation, Cox model wrappers, effect-size and multiplicity-correction
                     helpers, table and figure builders)
analysis.ipynb       the analysis: primary models, sensitivity and robustness analyses, figures,
                     and the analyses added in response to review
```

## Data

MIMIC-IV (v2.2) and the eICU Collaborative Research Database are PhysioNet credentialed-access
datasets and are not redistributed here. They are available at https://physionet.org/ after the
required training and a data use agreement.


### Inputs

| File (`common.py` constant) | One row per | Required columns | Used in |
|---|---|---|---|
| `cohort_final_v2.parquet` (`COHORT_PARQUET`) | ICU stay | `stay_id`, `t0`, plus the outcome/covariate columns named in `CONT_M`/`BIN_M`/`DEMO_*`/`SEV_*`/`LAB_C`/`TX_*` and the endpoint columns referenced throughout the notebook (e.g. `end28_h`, `event28`, `end_inhosp_h`, `death_inhosp`) | Sections 1-5 (`c`, `cmeas`, `tmp`, `XC_MEAS`, `XC_C`, `XC_BASE` are derived from this table upstream; its full column list is the cohort-construction step's contract, not part of this release) |
| `crrt_procedure_intervals.parquet` (`CRRT_PROC_PARQUET`) | one CRRT operating interval, procedure-chart definition | `stay_id` (int), `starttime` (datetime), `endtime` (datetime, `> starttime`) | Section 1 (Tier 1 MIMIC-IV grid, `source="procedure"`) |
| `crrt_input_intervals.parquet` (`CRRT_INPUT_PARQUET`) | one CRRT operating interval, anticoagulant/replacement-fluid infusion definition | same schema as above | Section 1 (Tier 1 MIMIC-IV grid, `source="input_union"`) |
| `eicu_cohort.parquet` (`EICU_COHORT_PARQUET`) | eICU CRRT patient | `patientunitstayid`, `t0_off`, `term_off`, `n_rec`, `med_gap_min` (minutes, may be null), `disch_h`, `event` (0/1, in-hospital death), `end_h` (hours), plus the covariates in `CONT_E` (`age_num`, `weight_kg`, `aps`, `map_value`, `platelet`, `hemoglobin`, `lactate`, `inr`, `aptt`, `bilirubin`) and `BIN_E` (`male`, `vaso_use`, `mech_vent`, `v3_systemic_hep`) | Section 1 (Tier 1 eICU grid) |

see the manuscript's Methods for how each column is derived from the raw PhysioNet tables.

## Requirements

Python 3.11 with `numpy`, `pandas`, `matplotlib`, `scipy`, `lifelines` (>= 0.30), `scikit-learn`,
and `statsmodels`.

`lifelines >= 0.30` matters: on that version `CoxTimeVaryingFitter.fit(robust=True)` raises
`NotImplementedError`, and the cluster-robust step branches on that behavior.

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

Please cite the associated manuscript once it is published. Until then, cite this repository by its
URL. On acceptance the repository will be archived on Zenodo and a citable DOI will be added here.
