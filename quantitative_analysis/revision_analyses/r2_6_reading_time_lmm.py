#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2-6: Reading-time linear mixed-effects model — full output.

Re-fits the mechanism LMM from the primary pipeline and emits the
revision-required diagnostics: cell-level descriptives (raw seconds),
Satterthwaite-df fixed-effects table with 95% CIs, random-effect variance
components, and Nakagawa-Schielzeth marginal / conditional R^2.

Model
-----
    log(reading_time_sec) ~ disagree * reliability * group + pgy_within_5
                            + (1 | reader) + (1 | filename)

Fit by REML with lme4/lmerTest (via pymer4), bobyqa optimizer, 200,000
function evaluations. Satterthwaite degrees of freedom from lmerTest.

Inputs
------
Expects a CSV at $RS_DATA_CSV with columns: reader, group, reliability,
filename, time_sec, log_time, disagree, pgy_within_5. See the folder
README for the full schema.

Outputs (written next to this script)
-------------------------------------
- r2_6_cell_descriptives_raw_seconds.csv
- r2_6_reading_time_lmm_fixed_effects.csv
- r2_6_reading_time_lmm_random_variances.csv
- r2_6_reading_time_lmm_r2.csv
- r2_6_supplementary_table5.md  (publication-ready Supp Table 5)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

GROUP_ORDER = ["Pediatric Radiologist", "Neonatologist", "Radiology Resident"]
RELIABILITY_LEVELS = ["Reliable", "Unreliable"]


def _load_data() -> pd.DataFrame:
    csv_path = os.environ.get("RS_DATA_CSV")
    if not csv_path:
        sys.exit(
            "Set RS_DATA_CSV to the path of the aided-set reader-case CSV. "
            "See revision_analyses/README.md for the expected schema."
        )
    df = pd.read_csv(csv_path)
    needed = {
        "reader", "group", "reliability", "filename",
        "time_sec", "log_time", "disagree", "pgy_within_5",
    }
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing required columns: {sorted(missing)}")

    df["reader"] = df["reader"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df["reliability"] = pd.Categorical(
        df["reliability"], categories=RELIABILITY_LEVELS, ordered=True
    )
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    df = df.dropna(subset=["log_time"]).copy()
    return df


def _iqr(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float("nan") if s.empty else float(np.percentile(s, 75) - np.percentile(s, 25))


def _cell_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["group", "reliability", "disagree"], observed=True)
        .agg(
            n_obs=("time_sec", "size"),
            n_readers=("reader", pd.Series.nunique),
            n_cases=("filename", pd.Series.nunique),
            time_mean=("time_sec", "mean"),
            time_sd=("time_sec", "std"),
            time_median=("time_sec", "median"),
            time_iqr=("time_sec", _iqr),
        )
        .reset_index()
    )
    out["disagree_label"] = out["disagree"].map(
        {0: "Concordance (agree with AI)", 1: "Discordance (disagree with AI)"}
    )
    return out


def _fmt_term(t: str) -> str:
    return (
        t.replace("reliability1", "reliability[Unreliable]")
        .replace("group1", "group[Neonatologist]")
        .replace("group2", "group[Radiology Resident]")
        .replace(":", " x ")
    )


def main() -> None:
    df = _load_data()
    print(
        f"[R2-6] rows={len(df)} readers={df['reader'].nunique()} "
        f"cases={df['filename'].nunique()}"
    )

    cell = _cell_descriptives(df)
    cell.to_csv(HERE / "r2_6_cell_descriptives_raw_seconds.csv", index=False)
    print("[R2-6] wrote r2_6_cell_descriptives_raw_seconds.csv")

    from pymer4.models import Lmer

    formula = (
        "log_time ~ disagree * reliability * group + pgy_within_5 "
        "+ (1|reader) + (1|filename)"
    )
    print(f"[R2-6] fitting LMM: {formula}")

    m = Lmer(formula, data=df, family="gaussian")
    m.fit(
        factors={"reliability": RELIABILITY_LEVELS, "group": GROUP_ORDER},
        summarize=False,
        control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
    )

    coefs = m.coefs.reset_index().rename(columns={"index": "term"})
    coefs_out = pd.DataFrame(
        {
            "term": coefs["term"],
            "beta": coefs["Estimate"].round(4),
            "SE": coefs["SE"].round(4),
            "t": coefs["T-stat"].round(3),
            "df_satterthwaite": coefs["DF"].round(2),
            "p_value": coefs["P-val"].apply(
                lambda p: f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"
            ),
            "p_raw": coefs["P-val"].astype(float),
            "ci_lo_95": coefs["2.5_ci"].round(4),
            "ci_hi_95": coefs["97.5_ci"].round(4),
        }
    )
    coefs_out.to_csv(HERE / "r2_6_reading_time_lmm_fixed_effects.csv", index=False)
    print("[R2-6] wrote r2_6_reading_time_lmm_fixed_effects.csv")

    ranef_var = m.ranef_var.reset_index().rename(columns={"index": "random_effect"})
    ranef_var.to_csv(HERE / "r2_6_reading_time_lmm_random_variances.csv", index=False)
    print("[R2-6] wrote r2_6_reading_time_lmm_random_variances.csv")

    # Nakagawa-Schielzeth R^2.
    X = np.asarray(m.design_matrix)
    betas = m.coefs["Estimate"].values
    var_fixed = float(np.var(X @ betas, ddof=0))
    var_reader = float(ranef_var.loc[ranef_var["random_effect"] == "reader", "Var"].iloc[0])
    var_case = float(ranef_var.loc[ranef_var["random_effect"] == "filename", "Var"].iloc[0])
    var_resid = float(ranef_var.loc[ranef_var["random_effect"] == "Residual", "Var"].iloc[0])
    denom = var_fixed + var_reader + var_case + var_resid
    r2_marg = var_fixed / denom
    r2_cond = (var_fixed + var_reader + var_case) / denom

    r2_df = pd.DataFrame(
        [
            {"quantity": "sigma^2 (reader)", "value": var_reader},
            {"quantity": "sigma^2 (case)", "value": var_case},
            {"quantity": "sigma^2 (residual)", "value": var_resid},
            {"quantity": "var(fixed-effect linear predictor)", "value": var_fixed},
            {"quantity": "R^2 marginal (Nakagawa-Schielzeth)", "value": r2_marg},
            {"quantity": "R^2 conditional (Nakagawa-Schielzeth)", "value": r2_cond},
        ]
    )
    r2_df["value"] = r2_df["value"].astype(float).round(4)
    r2_df.to_csv(HERE / "r2_6_reading_time_lmm_r2.csv", index=False)
    print("[R2-6] wrote r2_6_reading_time_lmm_r2.csv")

    # Supplementary Table 5 markdown.
    lines: list[str] = [
        "# Supplementary Table 5. Reading-time linear mixed model full output",
        "",
        "Model: `log(reading_time_sec) ~ disagree * reliability * group + pgy_within_5 "
        "+ (1|reader) + (1|case)` fit by REML with lme4/lmerTest (bobyqa optimizer, "
        "200,000 evaluations). Satterthwaite degrees of freedom.",
        "",
        "## (a) Cell descriptives on raw seconds",
        "",
        "| Group | Reliability | Concordance | N obs | N readers | N cases | "
        "Mean +- SD (s) | Median [IQR] (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ordered = cell.sort_values(["group", "reliability", "disagree"]).reset_index(drop=True)
    for _, r in ordered.iterrows():
        lines.append(
            f"| {r['group']} | {r['reliability']} | {r['disagree_label']} | "
            f"{int(r['n_obs'])} | {int(r['n_readers'])} | {int(r['n_cases'])} | "
            f"{r['time_mean']:.2f} +- {r['time_sd']:.2f} | "
            f"{r['time_median']:.2f} [IQR {r['time_iqr']:.2f}] |"
        )
    lines += [
        "",
        "## (b) Fixed-effect estimates",
        "",
        "| Term | beta | SE | t | df (Satterthwaite) | p | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in coefs_out.iterrows():
        lines.append(
            f"| {_fmt_term(r['term'])} | {r['beta']:.3f} | {r['SE']:.3f} | "
            f"{r['t']:.2f} | {r['df_satterthwaite']:.1f} | {r['p_value']} | "
            f"{r['ci_lo_95']:.3f} to {r['ci_hi_95']:.3f} |"
        )
    lines += [
        "",
        "## (c) Random-effect variance components",
        "",
        "| Source | sigma^2 |",
        "|---|---|",
        f"| Reader intercept | {var_reader:.4f} |",
        f"| Case intercept | {var_case:.4f} |",
        f"| Residual | {var_resid:.4f} |",
        "",
        "## (d) Model fit (Nakagawa-Schielzeth R^2)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Marginal R^2 (fixed effects only) | {r2_marg:.3f} |",
        f"| Conditional R^2 (fixed + random) | {r2_cond:.3f} |",
        "",
    ]
    (HERE / "r2_6_supplementary_table5.md").write_text("\n".join(lines), encoding="utf-8")
    print("[R2-6] wrote r2_6_supplementary_table5.md")


if __name__ == "__main__":
    main()
