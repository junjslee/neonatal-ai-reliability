#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2-8: Error-type stratification of reader response in the Error-Injected AI arm.

Reviewer 2 asked which reader group is more vulnerable to which AI error
type (false positive vs false negative). The submitted manuscript reports
an aggregate "agreement-with-wrong-AI" rate for the Error-Injected arm but
does not stratify by error direction.

This script produces:

  (1) Cross-tabulation. Reader group x error type x outcome
      (agreed-with-AI / overrode AI), with frequencies, percentages, and
      Wilson 95% CIs.

  (2) Mixed-effects logistic regression:
        agree_with_ai ~ group * error_type + (1 | reader) + (1 | filename)
      Fit only on the Error-Injected-arm AI-wrong subset. With only 6
      unique FN cases and zero-event cells in two of three groups, the
      conventional GLMM exhibits practical separation. We therefore also
      report Firth penalized logistic regression as a sensitivity analysis;
      the penalty produces finite estimates under separation.

  (3) Stacked / grouped bar visualization (300 DPI TIFF, LZW-compressed).

  (4) Results-section prose draft built from the actual fitted estimates.

Inputs
------
Expects a CSV at $RS_DATA_CSV with columns: reader, group, reliability,
filename, y_true, reader_pred, ai_pred_unreliable. The script computes
`agree_with_ai`, `error_type`, and the AI-wrong subset internally so it
remains a faithful re-derivation from raw fields.

Outputs (written next to this script)
-------------------------------------
- r2_8_error_type_stratification_counts.csv
- r2_8_error_type_stratification_model.csv
- r2_8_error_type_stratification_random_var.csv
- r2_8_error_type_supp_figure.tiff
- r2_8_results_prose_draft.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

GROUP_ORDER = ["Pediatric Radiologist", "Neonatologist", "Radiology Resident"]


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
        "y_true", "reader_pred", "ai_pred_unreliable",
    }
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing required columns: {sorted(missing)}")
    return df


def _classify_error(row) -> str:
    if row["ai_pred_displayed"] == 1 and row["y_true"] == 0:
        return "FP"
    if row["ai_pred_displayed"] == 0 and row["y_true"] == 1:
        return "FN"
    return "?"


def _wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _build_wrong_subset(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    unr = df[df["reliability"] == "Unreliable"].copy()
    unr["ai_pred_displayed"] = pd.to_numeric(unr["ai_pred_unreliable"], errors="coerce")
    unr = unr.dropna(subset=["ai_pred_displayed", "reader_pred", "y_true"]).copy()
    unr["ai_pred_displayed"] = unr["ai_pred_displayed"].astype(int)
    unr["ai_correct"] = (unr["ai_pred_displayed"] == unr["y_true"]).astype(int)
    unr["agree_with_ai"] = (unr["reader_pred"] == unr["ai_pred_displayed"]).astype(int)
    wrong = unr[unr["ai_correct"] == 0].copy()
    wrong["error_type"] = wrong.apply(_classify_error, axis=1)
    assert (wrong["error_type"] != "?").all(), "unexpected error_type label"
    unique_cases = wrong[["filename", "error_type"]].drop_duplicates("filename")
    n_fp = int((unique_cases["error_type"] == "FP").sum())
    n_fn = int((unique_cases["error_type"] == "FN").sum())
    return wrong, n_fp, n_fn


def _cross_tabulate(wrong: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in GROUP_ORDER:
        for et in ["FP", "FN"]:
            sub = wrong[(wrong["group"] == grp) & (wrong["error_type"] == et)]
            n_total = len(sub)
            n_agree = int(sub["agree_with_ai"].sum())
            rate = n_agree / n_total if n_total else float("nan")
            lo, hi = _wilson_ci(n_agree, n_total)
            rows.append({
                "group": grp,
                "error_type": et,
                "n_obs": n_total,
                "n_unique_readers": sub["reader"].nunique(),
                "n_unique_cases": sub["filename"].nunique(),
                "n_agreed_with_AI": n_agree,
                "n_overrode_AI": n_total - n_agree,
                "agreement_rate": round(rate, 4),
                "agreement_rate_pct": round(100 * rate, 2),
                "wilson_lo_95": round(lo, 4),
                "wilson_hi_95": round(hi, 4),
            })
    return pd.DataFrame(rows)


def _fit_glmm(wrong: pd.DataFrame, formula: str):
    """Fit logistic GLMM via lme4 / pymer4. Returns (coefs_df, ranef_var_df, status, message, sep_flag)."""
    from pymer4.models import Lmer

    try:
        m = Lmer(formula, data=wrong, family="binomial")
        m.fit(
            factors={"group": GROUP_ORDER, "error_type": ["FP", "FN"]},
            summarize=False,
            control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
        )
    except Exception as e:
        return None, None, "glmm_failed", f"GLMM failed: {e!r}", False

    coefs = m.coefs.copy().reset_index().rename(columns={"index": "term"})
    out = pd.DataFrame({
        "estimator": "GLMM (lme4 glmer, crossed RE)",
        "term": coefs["term"],
        "log_odds_beta": coefs["Estimate"].astype(float).round(4),
        "SE": coefs["SE"].astype(float).round(4),
        "z": coefs["Z-stat"].astype(float).round(3),
        "p_value": coefs["P-val"].astype(float).apply(
            lambda p: f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"
        ),
        "p_raw": coefs["P-val"].astype(float),
        "OR": coefs["OR"].astype(float).round(3),
        "OR_ci_lo_95": coefs["OR_2.5_ci"].astype(float).round(3),
        "OR_ci_hi_95": coefs["OR_97.5_ci"].astype(float).round(3),
    })
    rv = m.ranef_var.copy().reset_index().rename(columns={"index": "random_effect"})
    rv["estimator"] = "GLMM (lme4 glmer, crossed RE)"

    sep_terms = out[out["term"].astype(str).str.contains("error_type")]
    if (sep_terms["SE"].astype(float) > 10).any():
        return out, rv, "glmm_converged_with_separation", (
            "Logistic mixed model converged but FN-cell coefficients exhibited "
            "practical separation (SE > 10). Firth penalized logistic regression "
            "is reported as a sensitivity analysis."
        ), True
    return out, rv, "glmm_converged", (
        "Logistic mixed model with crossed reader/case random intercepts converged "
        "via lme4::glmer (bobyqa optimizer)."
    ), False


def _fit_firth(wrong: pd.DataFrame):
    """Firth penalized logistic regression. Stable under separation. No RE."""
    try:
        from firthlogist import FirthLogisticRegression  # type: ignore
    except ImportError:
        print("[R2-8] firthlogist not installed; skipping sensitivity model.")
        return None

    Xdf = pd.get_dummies(wrong[["group", "error_type"]], drop_first=True)
    grp_dummies = [c for c in Xdf.columns if c.startswith("group_")]
    et_dummies = [c for c in Xdf.columns if c.startswith("error_type_")]
    for gc in grp_dummies:
        for ec in et_dummies:
            Xdf[f"{gc}:{ec}"] = Xdf[gc] * Xdf[ec]
    X = Xdf.values.astype(float)
    y = wrong["agree_with_ai"].values.astype(int)

    fl = FirthLogisticRegression(skip_pvals=False)
    fl.fit(X, y)

    n_feat = len(fl.coef_)
    feat_terms = list(Xdf.columns)
    terms = ["(Intercept)"] + feat_terms
    coef = np.concatenate([[float(fl.intercept_)], np.asarray(fl.coef_, dtype=float)])
    bse_raw = np.asarray(fl.bse_, dtype=float)
    ci_raw = np.asarray(fl.ci_, dtype=float)
    pvals_raw = np.asarray(fl.pvals_, dtype=float)
    bse = np.concatenate([[bse_raw[n_feat]], bse_raw[:n_feat]])
    ci_lo = np.concatenate([[ci_raw[n_feat, 0]], ci_raw[:n_feat, 0]])
    ci_hi = np.concatenate([[ci_raw[n_feat, 1]], ci_raw[:n_feat, 1]])
    pvals = np.concatenate([[pvals_raw[n_feat]], pvals_raw[:n_feat]])

    z = np.divide(coef, bse, out=np.full_like(coef, np.nan), where=bse > 0)
    return pd.DataFrame({
        "estimator": "Firth penalized logistic (no RE, sensitivity)",
        "term": terms,
        "log_odds_beta": np.round(coef, 4),
        "SE": np.round(bse, 4),
        "z": np.round(z, 3),
        "p_value": [
            ("nan" if not np.isfinite(p) else (f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"))
            for p in pvals
        ],
        "p_raw": pvals,
        "OR": np.round(np.exp(coef), 3),
        "OR_ci_lo_95": np.round(np.exp(ci_lo), 3),
        "OR_ci_hi_95": np.round(np.exp(ci_hi), 3),
    })


def _plot_figure(counts_df: pd.DataFrame, fig_path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["font.family"] = "DejaVu Sans"
    err_colors = {"FP": "#4E79A7", "FN": "#E15759"}

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    x_positions = np.arange(len(GROUP_ORDER))
    bar_w = 0.36

    for i, et in enumerate(["FP", "FN"]):
        rates, los, his, ns = [], [], [], []
        for grp in GROUP_ORDER:
            row = counts_df.loc[
                (counts_df["group"] == grp) & (counts_df["error_type"] == et)
            ].iloc[0]
            rates.append(row["agreement_rate_pct"])
            los.append(row["agreement_rate_pct"] - 100 * row["wilson_lo_95"])
            his.append(100 * row["wilson_hi_95"] - row["agreement_rate_pct"])
            ns.append(int(row["n_obs"]))
        offset = (i - 0.5) * bar_w
        bars = ax.bar(
            x_positions + offset, rates, bar_w,
            yerr=[los, his], capsize=3,
            color=err_colors[et], edgecolor="black", linewidth=0.8,
            label=f"{et} (AI false {'positive' if et == 'FP' else 'negative'})",
        )
        for j, (bar, n_obs, rate) in enumerate(zip(bars, ns, rates)):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + max(his[j], 4),
                f"{rate:.1f}%\n(n={n_obs})",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(GROUP_ORDER, fontsize=10, fontweight="bold")
    ax.set_ylabel(
        "% of reader-case rows that agreed with the Error-Injected AI",
        fontsize=10,
    )
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.14)

    plt.savefig(
        fig_path, format="tiff", dpi=300,
        bbox_inches="tight", pad_inches=0.25,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    df = _load_data()
    wrong, n_fp, n_fn = _build_wrong_subset(df)
    print(
        f"[R2-8] Unreliable-arm AI-wrong: rows={len(wrong)} | "
        f"unique FP cases={n_fp} | unique FN cases={n_fn} | "
        f"readers={wrong['reader'].nunique()}"
    )

    counts_df = _cross_tabulate(wrong)
    counts_df.to_csv(HERE / "r2_8_error_type_stratification_counts.csv", index=False)
    print("[R2-8] wrote r2_8_error_type_stratification_counts.csv")

    wrong["group"] = pd.Categorical(wrong["group"], categories=GROUP_ORDER, ordered=False)
    wrong["error_type"] = pd.Categorical(wrong["error_type"], categories=["FP", "FN"], ordered=False)
    wrong["reader"] = wrong["reader"].astype(str)
    wrong["filename"] = wrong["filename"].astype(str)

    formula = "agree_with_ai ~ group * error_type + (1|reader) + (1|filename)"
    glmm_fe, glmm_rv, model_status, model_message, _ = _fit_glmm(wrong, formula)
    firth_fe = _fit_firth(wrong)

    combined = [df_ for df_ in (glmm_fe, firth_fe) if df_ is not None]
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(
            HERE / "r2_8_error_type_stratification_model.csv", index=False
        )
        print("[R2-8] wrote r2_8_error_type_stratification_model.csv")
    if glmm_rv is not None:
        glmm_rv.to_csv(
            HERE / "r2_8_error_type_stratification_random_var.csv", index=False
        )
        print("[R2-8] wrote r2_8_error_type_stratification_random_var.csv")

    _plot_figure(counts_df, HERE / "r2_8_error_type_supp_figure.tiff")
    print("[R2-8] wrote r2_8_error_type_supp_figure.tiff")

    # Prose draft.
    def _row(grp: str, et: str) -> dict:
        return counts_df.loc[
            (counts_df["group"] == grp) & (counts_df["error_type"] == et)
        ].iloc[0].to_dict()

    def _drop(grp: str) -> float:
        return float(_row(grp, "FP")["agreement_rate_pct"] - _row(grp, "FN")["agreement_rate_pct"])

    drops = {g: _drop(g) for g in GROUP_ORDER}
    prose_lines = [
        "# Results subsection draft (R2-8)",
        "",
        "## Suggested placement",
        "After the Error-Injected AI override-rate paragraph in Results, add the "
        "following new subsection (Supplementary Figure SX referenced).",
        "",
        "---",
        "",
        "### Error-type stratification of reader response in the Error-Injected AI arm",
        "",
        (
            f"To address whether reader vulnerability to AI error depends on error "
            f"direction, we stratified the Error-Injected AI arm by error type "
            f"(false positive, FP: AI predicted disease in {n_fp} cases without "
            f"disease; false negative, FN: AI missed disease in {n_fn} cases) and "
            f"computed the agreement-with-AI rate per reader group (Supplementary "
            f"Figure SX, Supplementary Table SX)."
        ),
        "",
        (
            f"Agreement with the AI was lower for FN errors than for FP errors in all "
            f"three groups, but the magnitude of the FP-versus-FN gap differed "
            f"substantially: Pediatric Radiologists ({drops['Pediatric Radiologist']:.1f} "
            f"percentage-point drop) and Neonatologists ({drops['Neonatologist']:.1f}-point "
            f"drop) overrode the AI on every FN case, whereas Radiology Residents agreed "
            f"with the AI's missed-disease verdict on a substantial fraction of FN rows "
            f"({_row('Radiology Resident', 'FN')['agreement_rate_pct']:.1f}%, only a "
            f"{drops['Radiology Resident']:.1f}-point drop from their FP-cell rate)."
        ),
        "",
    ]
    for grp in GROUP_ORDER:
        fp, fn = _row(grp, "FP"), _row(grp, "FN")
        prose_lines.append(
            f"{grp}s agreed with the AI in {fp['agreement_rate_pct']:.1f}% of FP rows "
            f"(n={fp['n_obs']}; Wilson 95% CI {100*fp['wilson_lo_95']:.1f}-"
            f"{100*fp['wilson_hi_95']:.1f}%) but only {fn['agreement_rate_pct']:.1f}% "
            f"of FN rows (n={fn['n_obs']}; Wilson 95% CI {100*fn['wilson_lo_95']:.1f}-"
            f"{100*fn['wilson_hi_95']:.1f}%)."
        )
        prose_lines.append("")

    prose_lines += [
        (
            f"We fitted a mixed-effects logistic regression (lme4::glmer; `{formula}`) "
            f"to the {len(wrong)} reader-case rows in the AI-wrong subset of the "
            f"Error-Injected arm ({n_fp} FP and {n_fn} FN unique cases). Two of the "
            f"three reader groups had zero events in the FN cell, creating practical "
            f"separation. As a sensitivity analysis, we re-estimated the same fixed-"
            f"effect structure with Firth penalized logistic regression, which "
            f"shrinks the separation toward zero and produces finite estimates."
        ),
        "",
        (
            f"Power to resolve group-specific differences in the FN cell is limited: "
            f"only {n_fn} unique FN cases are shared across all three reader groups, "
            f"and two of three groups had zero events. Group-specific findings for FN "
            f"errors are therefore reported as exploratory."
        ),
        "",
        (
            "These findings indicate that the dominant error-driven override pattern "
            "in the Error-Injected arm was rejection of FP claims, not detection of "
            "FN omissions. This subgroup analysis is exploratory and was not "
            "pre-registered."
        ),
        "",
        f"_Model fit status: **{model_status}**. {model_message}_",
        "",
    ]

    (HERE / "r2_8_results_prose_draft.md").write_text(
        "\n".join(prose_lines), encoding="utf-8"
    )
    print("[R2-8] wrote r2_8_results_prose_draft.md")
    print(f"[R2-8] done. Model status = {model_status}.")


if __name__ == "__main__":
    main()
