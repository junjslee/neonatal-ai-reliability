# Revision-round analyses

Scripts that produce the additional statistics added during the round-1
revision at *npj Digital Medicine*. Each script is self-contained, reads its
input from a CSV (path supplied via environment variable), and writes its
output next to itself.

| Script | Reviewer comment | What it produces |
|---|---|---|
| `r2_6_reading_time_lmm.py` | R2-6 (reading-time LMM full output) | Cell descriptives, Satterthwaite-df fixed-effects table, random-effect variances, Nakagawa-Schielzeth marginal/conditional R^2, Supplementary Table 5 (markdown) |
| `r2_7_between_group_differential.py` | R2-7 (is the disagree-vs-agree slowdown different across reader groups?) | Type-III ANOVA omnibus, group-specific simple slopes (emtrends), Tukey-adjusted pairwise contrasts |
| `r2_8_error_type_stratification.py` | R2-8 (does reader vulnerability depend on error direction in the Error-Injected arm?) | Cell rates of agreement with the AI stratified by FP vs FN with Wilson 95% CIs, mixed-effects logistic regression with random reader/case intercepts, Firth penalized logistic regression as sensitivity analysis under separation |

## Statistical specifications

All three scripts re-fit models on the aided set (Set B, the AI-assisted
arm), with crossed random intercepts for reader and case where applicable.
Categorical references match the primary manuscript pipeline:

- `reliability` reference = `Reliable`
- `group` reference = `Pediatric Radiologist`
- `error_type` reference = `FP` (R2-8 only)

The R2-6 specification is identical to the manuscript's mechanism LMM
(`reader_study_full_analysis.py`, "Aided-only mechanism analysis"). The
revision adds the full-output extraction; it does not change the fitted
model.

## Inputs

Each script reads a CSV of reader-case rows. The expected columns are
documented at the top of every script. In all cases the schema is a subset
of the columns produced by the primary pipeline's `dfB` and
`dfB_time_mech` DataFrames:

| Column | Type | Description |
|---|---|---|
| `reader` | string | Reader identifier |
| `group` | string | One of `Pediatric Radiologist`, `Neonatologist`, `Radiology Resident` |
| `reliability` | string | `Reliable` or `Unreliable` (aka Error-Injected) |
| `filename` | string | Case identifier (image / reading) |
| `time_sec` | float | Reading time in seconds (R2-6 only) |
| `log_time` | float | `log(time_sec)` (R2-6 only) |
| `disagree` | int (0/1) | 1 if reader's prediction differs from displayed AI prediction (R2-6, R2-7) |
| `agree_with_ai` | int (0/1) | 1 if reader's prediction equals displayed AI prediction (R2-8) |
| `error_type` | string | `FP` or `FN`; defined on the AI-wrong subset of the Error-Injected arm (R2-8 only) |
| `pgy_within_5` | int | PGY band covariate (R2-6, R2-7); 0 outside band, 1 inside |

The CSV path is supplied via the `RS_DATA_CSV` environment variable.

## Environment

Each script uses `pymer4` (which wraps R's `lme4` + `lmerTest`) for the
mixed-effects fits. R needs to be discoverable: set `R_HOME` to the R
installation that has `lme4`, `lmerTest`, and `emmeans` installed (the
manuscript pipeline uses the R that ships with conda's `base`
environment). Example:

```bash
export R_HOME="$(R RHOME)"
export RPY2_CFFI_MODE=ABI
export RS_DATA_CSV=/path/to/your/aided_set_long_format.csv
python r2_6_reading_time_lmm.py
python r2_7_between_group_differential.py
python r2_8_error_type_stratification.py
```

R2-8 additionally uses `firthlogist` (Python package) for the penalized-
likelihood sensitivity model. Install with `pip install firthlogist`.

## Public-data note

These scripts are released for methodological transparency. The reader-
study CSV is not redistributed in this repository — see the manuscript's
Data Availability statement. De-identified derived data are available from
the corresponding authors on reasonable request.
