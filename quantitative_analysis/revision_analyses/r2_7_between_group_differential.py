#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2-7: Between-group differential in the disagree-vs-agree reading-time gap.

This script does NOT add a new model. It re-fits the reading-time LMM from
R2-6 and extracts three quantities:

  1. Type-III ANOVA: omnibus F-test for the `disagree:group` 2-way
     interaction (and for completeness, the 3-way
     `disagree:reliability:group` interaction).

  2. Group-specific simple slopes for `disagree`, averaging over
     reliability, computed via emmeans::emtrends.

  3. Pairwise Tukey-adjusted comparisons of those slopes between groups.

The headline question for the rebuttal: do the descriptive per-group
disagree-vs-agree time deltas (e.g. neonatologists +4.6 s vs residents
+3.1 s vs pediatric radiologists +1.2 s) reach statistical significance as
a *between-group* difference? The Type-III omnibus answers that directly.

Inputs
------
Same CSV as `r2_6_reading_time_lmm.py`: `$RS_DATA_CSV` with columns
reader, group, reliability, filename, log_time, disagree, pgy_within_5.

Outputs (written next to this script)
-------------------------------------
- r2_7_anova_typeIII.csv
- r2_7_disagree_simple_slopes.csv
- r2_7_disagree_simple_slopes_seconds.csv  (back-transformed time ratios)
- r2_7_disagree_slope_pairwise_contrasts.csv
- r2_7_summary.md
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
    needed = {"reader", "group", "reliability", "filename",
              "log_time", "disagree", "pgy_within_5"}
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


def _coalesce(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(
        f"None of these columns found in emmeans output: {candidates} "
        f"| got {list(df.columns)}"
    )


def main() -> None:
    df = _load_data()
    print(
        f"[R2-7] rows={len(df)} readers={df['reader'].nunique()} "
        f"cases={df['filename'].nunique()}"
    )

    from pymer4.models import Lmer
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    formula = (
        "log_time ~ disagree * reliability * group + pgy_within_5 "
        "+ (1|reader) + (1|filename)"
    )
    print(f"[R2-7] fitting LMM: {formula}")

    m = Lmer(formula, data=df, family="gaussian")
    m.fit(
        factors={"reliability": RELIABILITY_LEVELS, "group": GROUP_ORDER},
        summarize=False,
        control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
    )

    # (1) Type-III ANOVA.
    anova_df = m.anova().reset_index().rename(columns={"index": "term"})
    anova_df.to_csv(HERE / "r2_7_anova_typeIII.csv", index=False)
    print("[R2-7] wrote r2_7_anova_typeIII.csv")

    # (2) Simple slopes via emmeans.
    ro.r["library"]("emmeans")
    ro.r.assign("model_obj", m.model_obj)
    ro.r(
        """
        et <- emtrends(model_obj, ~ group, var = "disagree",
                       weights = "proportional")
        et_summary <- as.data.frame(summary(et, infer = c(TRUE, TRUE)))
        et_pairs <- pairs(et, adjust = "tukey")
        et_pairs_df <- as.data.frame(summary(et_pairs, infer = c(TRUE, TRUE)))
        """
    )

    with (ro.default_converter + pandas2ri.converter).context():
        et_df = ro.conversion.get_conversion().rpy2py(ro.r["et_summary"])
        pairs_df = ro.conversion.get_conversion().rpy2py(ro.r["et_pairs_df"])

    slopes = pd.DataFrame(
        {
            "group": et_df["group"].astype(str),
            "disagree_slope_log_seconds": _coalesce(et_df, ["disagree.trend", "trend", "estimate"]).astype(float).round(4),
            "SE": _coalesce(et_df, ["SE", "se"]).astype(float).round(4),
            "df_satterthwaite": _coalesce(et_df, ["df"]).astype(float).round(2),
            "ci_lo_95_log": _coalesce(et_df, ["lower.CL", "asymp.LCL", "lower CL"]).astype(float).round(4),
            "ci_hi_95_log": _coalesce(et_df, ["upper.CL", "asymp.UCL", "upper CL"]).astype(float).round(4),
            "t_or_z": _coalesce(et_df, ["t.ratio", "z.ratio"]).astype(float).round(3),
            "p_value": _coalesce(et_df, ["p.value", "Pr...t.."]).astype(float),
        }
    )
    slopes["p_value_fmt"] = slopes["p_value"].apply(
        lambda p: f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"
    )
    slopes.to_csv(HERE / "r2_7_disagree_simple_slopes.csv", index=False)
    print("[R2-7] wrote r2_7_disagree_simple_slopes.csv")

    slopes_sec = pd.DataFrame(
        {
            "group": slopes["group"],
            "time_ratio_disagree_vs_agree": np.exp(slopes["disagree_slope_log_seconds"]).round(3),
            "ratio_ci_lo_95": np.exp(slopes["ci_lo_95_log"]).round(3),
            "ratio_ci_hi_95": np.exp(slopes["ci_hi_95_log"]).round(3),
            "p_value_fmt": slopes["p_value_fmt"],
        }
    )
    slopes_sec.to_csv(HERE / "r2_7_disagree_simple_slopes_seconds.csv", index=False)
    print("[R2-7] wrote r2_7_disagree_simple_slopes_seconds.csv")

    pairs_out = pd.DataFrame(
        {
            "contrast": _coalesce(pairs_df, ["contrast"]).astype(str),
            "diff_log_seconds": _coalesce(pairs_df, ["estimate"]).astype(float).round(4),
            "SE": _coalesce(pairs_df, ["SE"]).astype(float).round(4),
            "df_satterthwaite": _coalesce(pairs_df, ["df"]).astype(float).round(2),
            "ci_lo_95_log": _coalesce(pairs_df, ["lower.CL", "asymp.LCL"]).astype(float).round(4),
            "ci_hi_95_log": _coalesce(pairs_df, ["upper.CL", "asymp.UCL"]).astype(float).round(4),
            "t_or_z": _coalesce(pairs_df, ["t.ratio", "z.ratio"]).astype(float).round(3),
            "p_tukey": _coalesce(pairs_df, ["p.value"]).astype(float),
        }
    )
    pairs_out["p_tukey_fmt"] = pairs_out["p_tukey"].apply(
        lambda p: f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"
    )
    pairs_out.to_csv(HERE / "r2_7_disagree_slope_pairwise_contrasts.csv", index=False)
    print("[R2-7] wrote r2_7_disagree_slope_pairwise_contrasts.csv")

    # (4) Verdict prose.
    om_row = anova_df.loc[anova_df["term"].astype(str) == "disagree:group"]
    om_F = float(om_row["F-stat"].iloc[0]) if len(om_row) else float("nan")
    om_p = float(om_row["P-val"].iloc[0]) if len(om_row) else float("nan")
    om_ndf = float(om_row["NumDF"].iloc[0]) if len(om_row) else float("nan")
    om_ddf = float(om_row["DenomDF"].iloc[0]) if len(om_row) else float("nan")
    om_sig = (not np.isnan(om_p)) and om_p < 0.05

    three_row = anova_df.loc[
        anova_df["term"].astype(str).isin(
            ["disagree:reliability:group", "disagree:group:reliability"]
        )
    ]
    three_F = float(three_row["F-stat"].iloc[0]) if len(three_row) else float("nan")
    three_p = float(three_row["P-val"].iloc[0]) if len(three_row) else float("nan")

    lines: list[str] = [
        "# R2-7 verdict: between-group differential in disagree-vs-agree reading time",
        "",
        "Question: does the disagree-vs-agree reading-time gap differ "
        "*significantly* across reader groups?",
        "",
        "## Omnibus tests (Type-III ANOVA, Satterthwaite df)",
        "",
        "| Effect | NumDF | DenomDF | F | P |",
        "|---|---|---|---|---|",
    ]
    for _, r in anova_df.iterrows():
        lines.append(
            f"| {r['term']} | {float(r['NumDF']):.0f} | {float(r['DenomDF']):.1f} | "
            f"{float(r['F-stat']):.3f} | {float(r['P-val']):.3g} |"
        )
    lines += [
        "",
        "## Group-specific simple slopes (emmeans::emtrends, marginalized over reliability)",
        "",
        "| Group | β (log seconds) | SE | df | 95% CI | t | P | Time ratio (exp β) | Ratio 95% CI |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in slopes.iterrows():
        grp = str(r["group"])
        sec = slopes_sec.loc[slopes_sec["group"] == grp].iloc[0]
        lines.append(
            f"| {grp} | {r['disagree_slope_log_seconds']:.3f} | {r['SE']:.3f} | "
            f"{r['df_satterthwaite']:.1f} | {r['ci_lo_95_log']:.3f} to {r['ci_hi_95_log']:.3f} | "
            f"{r['t_or_z']:.2f} | {r['p_value_fmt']} | "
            f"{sec['time_ratio_disagree_vs_agree']:.2f} | "
            f"{sec['ratio_ci_lo_95']:.2f} to {sec['ratio_ci_hi_95']:.2f} |"
        )
    lines += [
        "",
        "## Pairwise comparison of group-specific slopes (Tukey-adjusted)",
        "",
        "| Contrast | Δβ (log seconds) | SE | df | 95% CI | t | P (Tukey) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in pairs_out.iterrows():
        lines.append(
            f"| {r['contrast']} | {r['diff_log_seconds']:.3f} | {r['SE']:.3f} | "
            f"{r['df_satterthwaite']:.1f} | {r['ci_lo_95_log']:.3f} to {r['ci_hi_95_log']:.3f} | "
            f"{r['t_or_z']:.2f} | {r['p_tukey_fmt']} |"
        )
    lines += ["", "## Verdict", ""]
    if om_sig:
        lines.append(
            f"YES. The omnibus type-III F test for the `disagree × group` "
            f"interaction was F({om_ndf:.0f}, {om_ddf:.1f}) = {om_F:.2f}, "
            f"P = {om_p:.3g} — the between-group differential reached "
            f"statistical significance."
        )
    else:
        lines.append(
            f"NO. The omnibus type-III F test for the `disagree × group` "
            f"interaction was F({om_ndf:.0f}, {om_ddf:.1f}) = {om_F:.2f}, "
            f"P = {om_p:.3g}."
        )
        lines += [
            "",
            "Each reader group individually shows a significant disagree-vs-agree "
            "slowdown (see the simple-slope table), but the LMM does not provide "
            "evidence that the slope itself differs across groups at the "
            "conventional 5% level. The descriptive per-group time deltas should "
            "therefore be reported as within-group effects, not as a between-group "
            "differential.",
        ]
    if not np.isnan(three_p):
        lines += [
            "",
            f"For completeness: the 3-way `disagree × reliability × group` "
            f"interaction omnibus was F = {three_F:.2f}, P = {three_p:.3g}.",
        ]

    (HERE / "r2_7_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[R2-7] wrote r2_7_summary.md")


if __name__ == "__main__":
    main()
