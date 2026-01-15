#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USAGE
  RPY2_CFFI_MODE=ABI python ./reader_study_full_pipeline_UPDATED.py

--------------------------------------------

Reader-study statistical pipeline (UPDATED)

MANUSCRIPT HIERARCHY
  Primary Analysis (Table 2):
    - Full Dataset (Unaided vs. Reliable AI vs. Error-Injected AI)
    - Logistic GLMM with crossed random intercepts (reader + case) to test Interaction.

  Secondary Analysis (Mechanism & Verification):
    - Aided Set Only (Reliable vs. Error-Injected)
    - Reading time (LMM) and CAM usage (Logistic GLMM).
    - Phenotype analysis (Sentinel Behavior vs. Automation Bias).

  Robustness / Sensitivity:
    - Period adjustment, case-mix adjustment, permutation tests.

"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# --- INSERT THIS BLOCK HERE ---
# Force rpy2 to use the specific R installed in your Conda environment
# This must be done BEFORE importing rpy2 or pymer4
os.environ['R_HOME'] = '/Users/junlee/miniconda3/lib/R'
# ------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import confusion_matrix, roc_curve, auc
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# 0) CONFIG
# =========================
XLSX_PATH = "/Users/junlee/Desktop/mi2rl_research/Neonatal_HCI/Neonatal_HCI_Codebase/_archive/reader_study_results/nec_lat_result_251208.xlsx"

# Optional: list of sheet names to include. If None, will load all sheets in the workbook.
SHEET_NAMES = None  # e.g., ["Reader01", "Reader02", ...]
# Optional: sheet names to exclude (case-insensitive contains match).
SHEET_EXCLUDE_CONTAINS = ["readme", "info", "summary"]

# External CSV containing case-level variables (Orientation/Modality/GA/BW etc.)
# Set to None if you want to skip case-mix merge
EXTERNAL_CSV_PATH = "/Users/junlee/Desktop/mi2rl_research/Neonatal_HCI/Neonatal_HCI_Codebase/_archive/reader_study_results/External7_for_analysis.csv"

OUTDIR = "./hai_outputs_glmm_final_updated_logging" ################################# the commented one below uses # OUTDIR = "./hai_outputs_glmm_final_updated"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# 0.1) LLM REPORTING / GLOBAL LOGGING
# =========================
# This block augments the pipeline to emit a single comprehensive text report intended
# for LLM analysis. It captures:
#   - All stdout/stderr
#   - Any DataFrame saved to CSV/Excel (as markdown)
#   - Any text logs saved via open(..., 'w'/'a') (as plaintext)
#   - Any figures saved via savefig() (title/axes + data summary)

import sys
import io
import builtins
import traceback
from datetime import datetime
from contextlib import contextmanager

LLM_REPORT_PATH = os.path.join(OUTDIR, "LLM_PIPELINE_DUMP.txt")


class _TeeStream:
    """Write to multiple streams (best-effort)."""
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


class LLMReport:
    def __init__(self, path: str):
        self.path = str(path)
        self._fp = open(self.path, "w", encoding="utf-8", errors="replace")
        # primary_full Header (REQUIRED)
        self._fp.write(
            "You are a Senior Data Scientist. Below is a complete dump of a statistical pipeline. "
            "Analyze the results, cross-referencing the figures and tables by their filenames.\n"
            # "Analyze the results, cross-referencing the figures and tables by their filenames to provide a final summary.\n"
        )
        self._fp.write(f"Report generated: {datetime.now().isoformat(timespec='seconds')}\n")
        self._fp.write(f"Working directory: {os.getcwd()}\n")
        self._fp.write(f"OUTDIR: {OUTDIR}\n")
        self._fp.write("=" * 80 + "\n\n")
        self._fp.flush()

    def close(self):
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass

    def _divider(self):
        self._fp.write("\n" + ("=" * 80) + "\n\n")

    def _start_file(self, filename: str):
        self._divider()
        self._fp.write(f"--- START OF DATA FOR FILE: [{filename}] ---\n\n")

    def _end_file(self, filename: str):
        self._fp.write(f"\n--- END OF DATA FOR FILE: [{filename}] ---\n")
        self._divider()
        self._fp.flush()

    def log_text_blob(self, filename: str, text: str):
        self._start_file(filename)
        self._fp.write(text if text is not None else "")
        self._end_file(filename)

    def log_dataframe_file(self, filename: str, df: "pd.DataFrame"):
        self._start_file(filename)
        self._fp.write(f"[DATAFRAME] shape={df.shape}\n\n")
        try:
            md = df.to_markdown(index=False)
            self._fp.write(md + "\n")
        except Exception as e:
            self._fp.write(f"[WARN] df.to_markdown() failed: {repr(e)}\n")
            self._fp.write(df.head(50).to_string(index=False) + "\n")
        self._end_file(filename)

    def log_figure(self, filename: str, fig, ax):
        # Visual-to-Text Translation (REQUIRED)
        self._start_file(filename)
        self._fp.write(f"DATA REPRESENTING FIGURE: [{filename}]\n\n")

        title = ""
        xlabel = ""
        ylabel = ""
        try:
            title = ax.get_title() or ""
            xlabel = ax.get_xlabel() or ""
            ylabel = ax.get_ylabel() or ""
        except Exception:
            pass

        self._fp.write(f"Title: {title}\n")
        self._fp.write(f"X-axis label: {xlabel}\n")
        self._fp.write(f"Y-axis label: {ylabel}\n\n")

        # Try to extract raw plotted coordinates from common Matplotlib artists.
        rows = []

        # Line plots (including ROC curves, etc.)
        try:
            for li, line in enumerate(ax.get_lines() or []):
                x = getattr(line, "get_xdata", lambda: [])()
                y = getattr(line, "get_ydata", lambda: [])()
                if x is None or y is None:
                    continue
                for xi, yi in zip(list(x), list(y)):
                    rows.append({"series": f"line_{li}", "x": float(xi), "y": float(yi)})
        except Exception:
            pass

        # Scatter plots / PathCollections
        try:
            for ci, coll in enumerate(getattr(ax, "collections", []) or []):
                offsets = getattr(coll, "get_offsets", lambda: None)()
                if offsets is None:
                    continue
                try:
                    off = offsets.tolist()
                except Exception:
                    off = list(offsets)
                for (xi, yi) in off:
                    try:
                        rows.append({"series": f"collection_{ci}", "x": float(xi), "y": float(yi)})
                    except Exception:
                        continue
        except Exception:
            pass

        # Bar plots / patches (use center x and height)
        try:
            for pi, p in enumerate(getattr(ax, "patches", []) or []):
                # Filter out rectangles that are not bars (best effort)
                w = getattr(p, "get_width", lambda: None)()
                h = getattr(p, "get_height", lambda: None)()
                x0 = getattr(p, "get_x", lambda: None)()
                if w is None or h is None or x0 is None:
                    continue
                xc = float(x0) + float(w) / 2.0
                rows.append({"series": f"patch_{pi}", "x": xc, "y": float(h)})
        except Exception:
            pass

        if len(rows) == 0:
            self._fp.write("[NOTE] Could not extract plotted coordinates from this figure (no lines/collections/patches).\n")
            self._end_file(filename)
            return

        dcoords = pd.DataFrame(rows)
        # Summaries
        try:
            x_min = float(dcoords["x"].min())
            x_max = float(dcoords["x"].max())
            x_mean = float(dcoords["x"].mean())
            y_min = float(dcoords["y"].min())
            y_max = float(dcoords["y"].max())
            y_mean = float(dcoords["y"].mean())
            self._fp.write("Summary of plotted data:\n")
            self._fp.write(f"  X: min={x_min:.6g} max={x_max:.6g} mean={x_mean:.6g}\n")
            self._fp.write(f"  Y: min={y_min:.6g} max={y_max:.6g} mean={y_mean:.6g}\n\n")
        except Exception:
            self._fp.write("[WARN] Failed to compute summary statistics for figure data.\n\n")

        # First 10 rows (raw coordinates)
        self._fp.write("First 10 rows of extracted coordinates:\n")
        try:
            self._fp.write(dcoords.head(10).to_markdown(index=False) + "\n")
        except Exception:
            self._fp.write(dcoords.head(10).to_string(index=False) + "\n")

        self._end_file(filename)


# Initialize report and redirect stdout/stderr into it (REQUIRED)
REPORT = LLMReport(LLM_REPORT_PATH)
_ORIG_STDOUT, _ORIG_STDERR = sys.stdout, sys.stderr
sys.stdout = _TeeStream(_ORIG_STDOUT, REPORT._fp)
sys.stderr = _TeeStream(_ORIG_STDERR, REPORT._fp)

# Monkeypatch pandas DataFrame writers so every saved table is also dumped into REPORT (REQUIRED)
_PD_ORIG_TO_CSV = pd.DataFrame.to_csv
_PD_ORIG_TO_EXCEL = pd.DataFrame.to_excel

def _df_to_csv_logged(self, path_or_buf=None, *args, **kwargs):
    out = _PD_ORIG_TO_CSV(self, path_or_buf, *args, **kwargs)
    try:
        if isinstance(path_or_buf, (str, os.PathLike)):
            REPORT.log_dataframe_file(str(path_or_buf), self)
    except Exception:
        # Never break the pipeline for reporting
        pass
    return out

def _df_to_excel_logged(self, excel_writer, *args, **kwargs):
    out = _PD_ORIG_TO_EXCEL(self, excel_writer, *args, **kwargs)
    try:
        if isinstance(excel_writer, (str, os.PathLike)):
            REPORT.log_dataframe_file(str(excel_writer), self)
    except Exception:
        pass
    return out

pd.DataFrame.to_csv = _df_to_csv_logged
pd.DataFrame.to_excel = _df_to_excel_logged

# Monkeypatch open() to capture any text log files written by the pipeline (best-effort)
_BUILTIN_OPEN = builtins.open

# class _LoggedTextFileProxy:
#     def __init__(self, real_f, filename: str):
#         self._f = real_f
#         self._filename = filename
#         self._buf = io.StringIO()

#     def write(self, s):
#         try:
#             self._buf.write(s)
#         except Exception:
#             pass
#         return self._f.write(s)

#     def writelines(self, lines):
#         try:
#             for ln in lines:
#                 self._buf.write(ln)
#         except Exception:
#             pass
#         return self._f.writelines(lines)

#     def flush(self):
#         return self._f.flush()

#     def close(self):
#         try:
#             self._f.close()
#         finally:
#             try:
#                 txt = self._buf.getvalue()
#                 if txt and self._filename and self._filename != LLM_REPORT_PATH:
#                     REPORT.log_text_blob(self._filename, txt)
#             except Exception:
#                 pass

#     def __getattr__(self, name):
#         return getattr(self._f, name)
class _LoggedTextFileProxy:
    def __init__(self, real_f, filename: str):
        self._f = real_f
        self._filename = filename
        self._buf = io.StringIO()

    # --- ADDED THESE TWO METHODS ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    # -------------------------------

    def write(self, s):
        try:
            self._buf.write(s)
        except Exception:
            pass
        return self._f.write(s)

    def writelines(self, lines):
        try:
            for ln in lines:
                self._buf.write(ln)
        except Exception:
            pass
        return self._f.writelines(lines)

    def flush(self):
        return self._f.flush()

    def close(self):
        try:
            self._f.close()
        finally:
            try:
                txt = self._buf.getvalue()
                if txt and self._filename and self._filename != LLM_REPORT_PATH:
                    REPORT.log_text_blob(self._filename, txt)
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._f, name)

def open(*args, **kwargs):
    f = _BUILTIN_OPEN(*args, **kwargs)
    try:
        if len(args) >= 2:
            filename = str(args[0])
            mode = str(args[1])
        else:
            filename = str(kwargs.get("file", ""))
            mode = str(kwargs.get("mode", "r"))
        is_text_write = any(m in mode for m in ["w", "a", "x"]) and ("b" not in mode)
        is_txtlike = filename.lower().endswith((".txt", ".log", ".md"))
        if is_text_write and is_txtlike and os.path.abspath(filename) != os.path.abspath(LLM_REPORT_PATH):
            return _LoggedTextFileProxy(f, filename)
    except Exception:
        pass
    return f

builtins.open = open

# Ensure report is flushed/closed even if exceptions occur
import atexit
atexit.register(lambda: REPORT.close())

# Figure sizes
FIG_ROC = (8.2, 6.4)
FIG_HCI = (11.1, 6.2)
FIG_TIME = (13.2, 6.2)
FIG_MECH_CI = (10.8, 8) 
FIG_FOREST = (13.5, 5.2)
FIG_FOREST_SET_AIDED = (12.6, 4.8)

mpl.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)

# COL = {
#     "unaided": "#4E79A7",
#     "optimized": "#59A14F",
#     "unreliable": "#E15759",
#     "Pediatric Radiologist": "#1F77B4",
#     "Neonatologist": "#2CA02C",
#     "Radiology Resident": "#FF7F0E",
#     "type1": "#4E79A7",
#     "type2": "#F28E2B",
#     "type3": "#59A14F",
#     "type4": "#E15759",
# }
COL = {
    "unaided": "#4E79A7",
    "optimized": "#59A14F",   # Internal name
    "reliable": "#59A14F",    # Manuscript name (Added this alias)
    "unreliable": "#E15759",  # Internal name
    "Error-Injected AI": "#E15759", # Manuscript name (Added alias)
    "Pediatric Radiologist": "#1F77B4",
    "Neonatologist": "#2CA02C",
    "Radiology Resident": "#FF7F0E",
    "type1": "#4E79A7",
    "type2": "#F28E2B",
    "type3": "#59A14F",
    "type4": "#E15759",
}

COND_ORDER = ["unaided", "optimized", "unreliable"]
GROUP_ORDER = ["Pediatric Radiologist", "Neonatologist", "Radiology Resident"]

# =========================
# (A) RELIABLE AI BACKGROUND ROC CURVE DATA
# =========================
# NOTE: These arrays are used only for plotting the AI ROC curve.
Y_TRUE_AI_CURVE = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
]
Y_PROB_AI_CURVE = [
    0.03434, 0.00959, 0.15749, 0.02232, 0.09559, 0.05969, 0.02916, 0.01516, 0.25188, 0.0546,
    0.26537, 0.08547, 0.48939, 0.00496, 0.376, 0.01906, 0.61691, 0.90985, 0.39261, 0.00253,
    0.01208, 0.01805, 0.00976, 0.0091, 0.01728, 0.01003, 0.01698, 0.00804, 0.00804, 0.03356,
    0.00521, 0.00787, 0.00593, 0.00562, 0.02386, 0.05674, 0.13154, 0.2377, 0.03891, 0.16077,
    0.99869, 0.0239, 0.02553, 0.01224, 0.00739, 0.03401, 0.06967, 0.01835, 0.11816, 0.11086,
    0.01719, 0.00444, 0.06133, 0.38011, 0.07073, 0.49741, 0.18246, 0.36338, 0.02449, 0.28469,
    0.12515, 0.01197, 0.0313, 0.01748, 0.93029, 0.98257, 0.03068, 0.81478, 0.99879, 0.99253,
    0.01247, 0.00551, 0.00896, 0.02917, 0.00725, 0.03765, 0.04259, 0.19015, 0.2212, 0.19263,
    0.12805, 0.18668, 0.65452, 0.0236, 0.01992, 0.99716, 0.99123, 0.99777, 0.9995, 0.06375,
    0.99284, 0.99023, 0.01427, 0.02471, 0.97817, 0.92073, 0.99826, 0.49731, 0.99892, 0.99973,
    0.14224, 0.05567, 0.88097, 0.99969, 0.01571, 0.99908, 0.9994, 0.99909, 0.56375, 0.20318,
    0.98352, 0.46712, 0.02593, 0.01491, 0.22221, 0.09441, 0.99714, 0.00574, 0.70502, 0.01915,
    0.05119, 0.99338, 0.01217, 0.99852, 0.03577
]


# =========================
# HELPERS
# =========================
def savefig(name: str):
    """
    Save a figure and emit a textual "data representation" block into the LLM report.

    Notes:
      - The original pipeline uses TIFF output with LZW compression for publication quality.
      - The reporting layer logs the *actual* filename saved.
    """
    fig = plt.gcf()
    ax = plt.gca()

    # Preserve original behavior: write TIFF if asked for PNG
    if name.endswith(".png"):
        name = name.replace(".png", ".tif")

    path = os.path.join(OUTDIR, name)
    plt.savefig(
        path,
        format="tiff",
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    # Log the figure metadata + extracted coordinates
    try:
        REPORT.log_figure(path, fig, ax)
    except Exception:
        # Never break the pipeline for reporting
        pass

    plt.close()
    print(f"Saved: {name}")

def _norm_col(c) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


def normalize_filename(x):
    """Normalize a case/image filename for robust joins across sources.

    Accepts full paths or bare filenames; strips common image extensions; lowercases.
    Returns None for missing values.
    """
    if x is None:
        return None
    # pandas may store missing as float nan
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    s = os.path.basename(s)
    s = re.sub(r"\.(png|jpg|jpeg|tif|tiff)$", "", s, flags=re.IGNORECASE)
    return s.lower()
def sens_spec_acc(y_true, y_pred):
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    tpr = sens
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sens": sens, "spec": spec, "acc": acc, "fpr": fpr, "tpr": tpr}


def ensure_time_features(df: pd.DataFrame, time_col: str = "time_sec") -> pd.DataFrame:
    if time_col not in df.columns:
        alternatives = ["time", "reading_time", "reading time", "read_time", "duration", "duration_sec"]
        for alt in alternatives:
            if alt in df.columns:
                df[time_col] = df[alt]
                break
    if time_col not in df.columns:
        df[time_col] = np.nan
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df["log_time"] = np.log(df[time_col].clip(lower=0.1))
    return df


def p_to_label(p: float) -> str:
    if p is None or (not np.isfinite(p)):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")


# =========================
# CASE-MIX: load + merge + features
# =========================
def load_case_mix_csv(external_csv_path: str) -> pd.DataFrame:
    """Load external case-mix metadata and return one row per image (file_key).

    Supports either:
      - numeric GA/BW columns (weeks / grams) -> produces ga_z / bw_z + strata, OR
      - categorical strata columns like GA01/GA02/... and BW01/BW02/...

    Also supports different filename column names (e.g., png_filename).
    """
    if external_csv_path is None or str(external_csv_path).strip() == "":
        raise ValueError("EXTERNAL_CSV_PATH is empty; cannot load case-mix metadata.")

    df = pd.read_csv(external_csv_path)
    df.columns = [c.strip() for c in df.columns]

    # ---- filename / join key ----
    fn_col = None
    for c in ["png_filename", "filename", "image_filename", "img_filename", "file", "case", "case_id"]:
        if c in df.columns:
            fn_col = c
            break
    if fn_col is None:
        raise ValueError(
            f"External case-mix CSV must include a filename column. "
            f"Tried: png_filename/filename/image_filename/img_filename/file/case/case_id. "
            f"Available columns: {list(df.columns)}"
        )
    df["file_key"] = df[fn_col].map(normalize_filename)

    # ---- orientation / modality ----
    if "orientation" not in df.columns:
        for c in ["image_orientation", "view", "position", "orient", "orientation_strata"]:
            if c in df.columns:
                df["orientation"] = df[c]
                break
    if "modality" not in df.columns:
        for c in ["image_modality", "mod", "type", "modality_strata"]:
            if c in df.columns:
                df["modality"] = df[c]
                break

    # ---- GA / BW handling ----
    def _looks_like_strata(series: pd.Series, prefix: str) -> bool:
        s = series.dropna().astype(str)
        if len(s) == 0:
            return False
        sample = s.sample(min(20, len(s)), random_state=0)
        pat = re.compile(rf"^{prefix}\s*\d+$", re.IGNORECASE)
        return sample.map(lambda v: bool(pat.match(v.strip()))).mean() >= 0.6

    # GA: prefer explicit strata, else numeric
    ga_strata_col = None
    for c in ["ga_strata", "GA_strata", "ga_category", "GA_category", "gestational_age_strata", "GA"]:
        if c in df.columns and _looks_like_strata(df[c], "GA"):
            ga_strata_col = c
            break

    ga_num_col = None
    for c in ["ga_weeks", "gestational_age_weeks", "gestational_age", "ga", "GA_weeks", "GA"]:
        if c in df.columns and c != ga_strata_col:
            ga_num_col = c
            break

    if ga_strata_col is not None:
        df["ga_strata"] = df[ga_strata_col]
    else:
        df["ga_strata"] = pd.NA
    if ga_num_col is not None:
        df["ga"] = pd.to_numeric(df[ga_num_col], errors="coerce")
    else:
        df["ga"] = np.nan

    # BW: prefer explicit strata, else numeric
    bw_strata_col = None
    for c in ["bw_strata", "BW_strata", "bw_category", "BW_category", "birth_weight_strata", "BW"]:
        if c in df.columns and _looks_like_strata(df[c], "BW"):
            bw_strata_col = c
            break

    bw_num_col = None
    for c in ["bw_grams", "birth_weight_grams", "birth_weight", "bw", "BW_grams", "BW"]:
        if c in df.columns and c != bw_strata_col:
            bw_num_col = c
            break

    if bw_strata_col is not None:
        df["bw_strata"] = df[bw_strata_col]
    else:
        df["bw_strata"] = pd.NA
    if bw_num_col is not None:
        df["bw"] = pd.to_numeric(df[bw_num_col], errors="coerce")
    else:
        df["bw"] = np.nan

    # keep only relevant columns; then de-duplicate to one row per file_key
    keep = ["file_key", "ga", "bw", "ga_strata", "bw_strata", "orientation", "modality"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # aggregate duplicates by first non-missing value
    def _first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else pd.NA

    df = df.groupby("file_key", dropna=False, as_index=False).agg(_first_nonnull)

    # engineer features (z-scores / cleaned strata)
    df = add_case_mix_features(df)
    return df
def add_case_mix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add harmonized GA/BW features.

    - If numeric GA/BW are present: compute z-scores (ga_z, bw_z) and also compute strata bins.
    - If categorical strata like GA01/BW02 are present: standardize formatting and keep as factors.
    """

    def _normalize_strata(series: pd.Series, prefix: str) -> pd.Series:
        ser = series.astype("string")
        ser = ser.str.strip()
        ser = ser.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
        # numeric codes like 1/2/3 -> GA01 etc
        ser_num = pd.to_numeric(ser, errors="coerce")
        mask_num = ser_num.notna() & ser.isna()  # shouldn't happen, but safe
        # If the original ser contains digit-only strings, also map those
        digit_only = ser.str.match(r"^\d+$", na=False)
        ser = ser.where(~digit_only, ser.map(lambda v: f"{prefix}{str(int(v)).zfill(2)}" if v is not None and str(v).isdigit() else v))
        # If already like GA1 / BW3 etc -> pad
        pat = re.compile(rf"^{prefix}\s*(\d+)$", re.IGNORECASE)
        def _pad(v):
            if v is None:
                return pd.NA
            vv = str(v).strip()
            m = pat.match(vv)
            if not m:
                return vv.upper()
            return f"{prefix}{m.group(1).zfill(2)}"
        ser = ser.map(_pad)
        return ser

    # ---- Numeric GA/BW -> z-scores ----
    if "ga" in df.columns:
        df["ga"] = pd.to_numeric(df["ga"], errors="coerce")
        if df["ga"].notna().any():
            mu, sd = df["ga"].mean(), df["ga"].std(ddof=0)
            df["ga_z"] = (df["ga"] - mu) / sd if (sd and sd > 0) else 0.0
        else:
            df["ga_z"] = np.nan
    else:
        df["ga_z"] = np.nan

    if "bw" in df.columns:
        df["bw"] = pd.to_numeric(df["bw"], errors="coerce")
        if df["bw"].notna().any():
            mu, sd = df["bw"].mean(), df["bw"].std(ddof=0)
            df["bw_z"] = (df["bw"] - mu) / sd if (sd and sd > 0) else 0.0
        else:
            df["bw_z"] = np.nan
    else:
        df["bw_z"] = np.nan

    # ---- Strata (either provided or derived) ----
    if "ga_strata" in df.columns:
        df["ga_strata"] = _normalize_strata(df["ga_strata"], "GA")
    else:
        df["ga_strata"] = pd.NA

    if "bw_strata" in df.columns:
        df["bw_strata"] = _normalize_strata(df["bw_strata"], "BW")
    else:
        df["bw_strata"] = pd.NA

    # If numeric GA/BW exist but strata missing, derive default 3 bins
    if df["ga_strata"].isna().all() and ("ga" in df.columns) and df["ga"].notna().any():
        df["ga_strata"] = pd.cut(
            df["ga"],
            bins=[-np.inf, 28, 32, np.inf],
            labels=["GA01", "GA02", "GA03"],
            right=False,
        ).astype("string")

    if df["bw_strata"].isna().all() and ("bw" in df.columns) and df["bw"].notna().any():
        df["bw_strata"] = pd.cut(
            df["bw"],
            bins=[-np.inf, 1000, 1500, np.inf],
            labels=["BW01", "BW02", "BW03"],
            right=False,
        ).astype("string")

    # Normalize orientation/modality to simple strings (optional but keeps joins stable)
    for c in ["orientation", "modality"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
            df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # ---------------------------------------------------------------------
    # Fallback: if GA/BW are provided ONLY as strata codes (e.g., GA01/GA02/GA03)
    # and numeric columns are missing, derive an ordinal z-score from the code
    # so that case-mix adjustment still works.
    # ---------------------------------------------------------------------
    def _ordinal_z_from_strata(s: pd.Series) -> pd.Series:
        s = s.astype("string")
        ordv = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
        if ordv.notna().sum() < 2:
            return pd.Series([np.nan] * len(s), index=s.index)
        mu = float(ordv.mean())
        sd = float(ordv.std(ddof=0))
        if sd == 0.0:
            return pd.Series([0.0] * len(s), index=s.index)
        return (ordv - mu) / sd

    if "ga_z" in df.columns and df["ga_z"].isna().all() and "ga_strata" in df.columns:
        df["ga_z"] = _ordinal_z_from_strata(df["ga_strata"])
    if "bw_z" in df.columns and df["bw_z"].isna().all() and "bw_strata" in df.columns:
        df["bw_z"] = _ordinal_z_from_strata(df["bw_strata"])

    return df
def merge_case_mix(reader_df: pd.DataFrame, case_mix_df: pd.DataFrame) -> pd.DataFrame:
    """Merge external case-mix metadata onto reader-level observations.

    We merge on a normalized filename key (`file_key`) so joins are robust to:
      - full paths vs basenames
      - different extensions (.png/.tif/.jpg)
      - capitalization differences

    IMPORTANT: we **keep** `file_key` in the returned dataframe because downstream
    audit tables (and optional de-duplication) rely on it.
    """
    if case_mix_df is None or len(case_mix_df) == 0:
        out = reader_df.copy()
        if "file_key" not in out.columns and "filename" in out.columns:
            out["file_key"] = out["filename"].map(normalize_filename)
        return out

    out = reader_df.copy()

    if "file_key" not in out.columns:
        if "filename" not in out.columns:
            raise ValueError("reader_df is missing 'filename' column; cannot merge case-mix.")
        out["file_key"] = out["filename"].map(normalize_filename)

    if "file_key" not in case_mix_df.columns:
        raise ValueError("case_mix_df is missing 'file_key'. Did you load it with load_case_mix_csv()?")

    # De-duplicate case_mix_df defensively
    cm = case_mix_df.drop_duplicates(subset=["file_key"]).copy()

    merged = out.merge(cm, on="file_key", how="left", suffixes=("", "_cm"))
    return merged
def parse_meta(sheet):
    meta_raw = pd.read_excel(XLSX_PATH, sheet_name=sheet, header=None, nrows=6)
    meta = {"reader": sheet}
    for v in meta_raw[0].tolist():
        if isinstance(v, str) and ":" in v:
            k, val = v.split(":", 1)
            meta[k.strip().lower()] = val.strip()
    return meta


def group_from_meta(meta):
    title = str(meta.get("title", "")).strip().lower()
    spec = str(meta.get("specialty", "")).strip().lower()
    # if title == "specialist":
    #     if "neo" in spec:
    #         return "Neonatologist"
    #     return "Pediatric Radiologist"
    # return "Radiology Resident"
    # --- UPDATED LOGIC START ---
    # Priority 1: Use specific keywords in the 'specialty' field
    if "neonatologist" in spec:
        return "Neonatologist"
    
    if "pediatric radiologist" in spec:
        return "Pediatric Radiologist"
        
    if "resident" in spec:
        # Handles "radiology resident" or just "resident"
        return "Radiology Resident"

    # Priority 2: Fallback to Title/Specialty combo (legacy support)
    # If title is 'specialist', check specialty for 'neo'
    if title == "specialist":
        if "neo" in spec:
            return "Neonatologist"
        # If specialty is vague but title is Specialist, assume Ped Rad
        return "Pediatric Radiologist"
    
    # Default Fallback
    return "Radiology Resident"
    # --- UPDATED LOGIC END ---


def load_cases(sheet):
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet, header=6).copy()
    df.columns = [_norm_col(c) for c in df.columns]

    col_map = {
        "filename": ["filename", "file", "image", "image_id", "case", "case_id"],
        "answer_raw": ["answer", "reader_answer", "readers_answer", "final_answer", "diagnosis"],
        "show_cam": ["show cam", "show_cam", "cam", "clicked_cam", "show heatmap", "show_heatmap"],
        "time_sec": ["time", "time_sec", "reading time", "reading_time", "duration", "duration_sec"],
        "with_ai": ["with ai", "with_ai", "ai", "aided", "aid"],
        "y_true": ["groundtruthbinary", "ground_truth_binary", "groundtruth", "gt", "label", "y_true"],
        "case_set": ["caseset", "case_set", "set", "session_set"],
        "ai_model": ["ai model", "ai_model", "model", "ai_type"],
        "ai_pred_optimized": ["optimized", "optimizedai", "ai_pred_optimized", "pred_optimized"],
        "ai_pred_unreliable": ["debuffed", "debuffedai", "unreliable", "ai_pred_unreliable", "pred_unreliable"],
    }

    def find_col(cands):
        for c in cands:
            c = _norm_col(c)
            if c in df.columns:
                return c
        return None

    rename = {}
    for target, cands in col_map.items():
        found = find_col(cands)
        if found is not None:
            rename[found] = target
    df = df.rename(columns=rename)

    for must in [
        "filename", "answer_raw", "show_cam", "time_sec", "with_ai", "y_true",
        "case_set", "ai_model", "ai_pred_optimized", "ai_pred_unreliable"
    ]:
        if must not in df.columns:
            df[must] = np.nan

    df["reader_pred"] = df["answer_raw"].map({"Y": 1, "N": 0, "y": 1, "n": 0})
    df["show_cam"] = pd.to_numeric(df["show_cam"], errors="coerce").fillna(0).astype(int)
    df["with_ai"] = pd.to_numeric(df["with_ai"], errors="coerce").fillna(0).astype(int)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype("Int64")

    df["ai_model"] = df["ai_model"].fillna("").astype(str)
    df["ai_model_clean"] = df["ai_model"].replace(
        {
            "OptimizedAI": "optimized", "optimizedai": "optimized", "optimized": "optimized",
            "DebuffedAI": "unreliable", "DeBuffedAI": "unreliable", "debuffedai": "unreliable",
            "debuffed": "unreliable", "unreliable": "unreliable",
        }
    )
    df.loc[~df["ai_model_clean"].isin(["optimized", "unreliable"]), "ai_model_clean"] = ""

    df["ai_pred_optimized"] = pd.to_numeric(df["ai_pred_optimized"], errors="coerce")
    df["ai_pred_unreliable"] = pd.to_numeric(df["ai_pred_unreliable"], errors="coerce")

    def cond(r):
        if int(r["with_ai"]) == 0:
            return "unaided"
        if r["ai_model_clean"] == "optimized":
            return "optimized"
        if r["ai_model_clean"] == "unreliable":
            return "unreliable"
        return "aided_unknown"

    df["condition"] = df.apply(cond, axis=1)
    df = ensure_time_features(df, "time_sec")
    return df


# -------------------------
# Load workbook and sheet list
# -------------------------
if not os.path.exists(XLSX_PATH):
    raise FileNotFoundError(f"XLSX_PATH not found: {XLSX_PATH}")
xl = pd.ExcelFile(XLSX_PATH)
if SHEET_NAMES is None:
    sheets = [s for s in xl.sheet_names]
else:
    sheets = list(SHEET_NAMES)
# Exclude obvious non-reader sheets
sheets = [s for s in sheets if not any(ex in str(s).lower() for ex in SHEET_EXCLUDE_CONTAINS)]
if len(sheets) == 0:
    raise ValueError("No sheets selected. Check SHEET_NAMES / SHEET_EXCLUDE_CONTAINS.")

# Parse reader-level metadata from each sheet (robust to occasional malformed sheets)
meta_rows = []
good_sheets = []
for s in sheets:
    try:
        meta_rows.append(parse_meta(s))
        good_sheets.append(s)
    except Exception as e:
        print(f"[WARN] Skipping sheet '{s}' while parsing metadata: {e}")
sheets = good_sheets
if len(sheets) == 0:
    raise ValueError("All sheets failed metadata parsing. Check XLSX template.")

meta = pd.DataFrame(meta_rows)
meta["group"] = meta.apply(group_from_meta, axis=1)
meta["experience_years"] = pd.to_numeric(meta.get("experience"), errors="coerce")  # PGY/years

rows = []
for s in sheets:
    df_s = load_cases(s)
    m = meta.loc[meta["reader"] == s].iloc[0]
    df_s["reader"] = s
    df_s["group"] = m["group"]
    df_s["pgy"] = m["experience_years"]
    rows.append(df_s)

df = pd.concat(rows, ignore_index=True)

df = df[df["group"].isin(GROUP_ORDER)].copy()
df = df[df["condition"].isin(COND_ORDER)].copy()
df = df.dropna(subset=["y_true", "reader_pred"]).copy()
df["y_true"] = df["y_true"].astype(int)
df["reader_pred"] = df["reader_pred"].astype(int)

# =========================
# 1.05) Case-mix merge from external CSV (optional)
# =========================
if EXTERNAL_CSV_PATH is not None and os.path.exists(EXTERNAL_CSV_PATH):
    cm = load_case_mix_csv(EXTERNAL_CSV_PATH)
    cm.to_csv(os.path.join(OUTDIR, "case_mix_cleaned_unique_cases.csv"), index=False)
    df = merge_case_mix(df, cm)
    # rpy2 conversion is picky with mixed/NA types; keep strata columns as pure strings (even if not used in formulas)
    for _c in ["ga_strata", "bw_strata"]:
        if _c in df.columns:
            df[_c] = df[_c].astype("string")
else:
    for c in ["file_key", "orientation", "modality", "ga", "bw", "ga_z", "bw_z", "ga_strata", "bw_strata"]:
        if c not in df.columns:
            df[c] = np.nan

    # Keep strata columns as strings for downstream R conversion (even though not used when EXTERNAL_CSV_PATH is absent)
    for _c in ["ga_strata", "bw_strata"]:
        if _c in df.columns:
            df[_c] = df[_c].astype("string")
    if "file_key" in df.columns:
        df["file_key"] = df["file_key"].astype("string")
for c in ["orientation", "modality", "ga_strata", "bw_strata"]:
    if c in df.columns:
        # Keep as clean strings (avoids rpy2 conversion warnings when NA forces float/object mixtures)
        df[c] = df[c].astype("string")
        df[c] = df[c].fillna("NA")

# =========================
# 1.1) PGY validation + within-group centering
# =========================
pgy_check = (
    df[["reader", "group", "pgy"]]
    .drop_duplicates()
    .sort_values(["group", "pgy"], na_position="last")
)

pgy_ranges = (
    pgy_check.groupby("group")
    .agg(
        n_readers=("reader", "count"),
        pgy_mean=("pgy", "mean"),
        pgy_sd=("pgy", "std"),
        pgy_min=("pgy", "min"),
        pgy_max=("pgy", "max"),
    )
    .reset_index()
)
pgy_ranges.to_csv(os.path.join(OUTDIR, "table_reader_pgy_ranges.csv"), index=False)

group_pgy_mean = pgy_check.groupby("group")["pgy"].mean().to_dict()
df["pgy_within"] = df.apply(lambda r: r["pgy"] - group_pgy_mean.get(r["group"], np.nan), axis=1)
df["pgy_within_5"] = df["pgy_within"] / 5.0

# =========================
# 2) ENDPOINTS + HCI TYPES
# =========================
df["correct"] = (df["reader_pred"] == df["y_true"]).astype(int)


def displayed_ai(r):
    if r["condition"] == "optimized":
        return int(r["ai_pred_optimized"]) if pd.notna(r["ai_pred_optimized"]) else np.nan
    if r["condition"] == "unreliable":
        return int(r["ai_pred_unreliable"]) if pd.notna(r["ai_pred_unreliable"]) else np.nan
    return np.nan


df["ai_pred_displayed"] = df.apply(displayed_ai, axis=1)

df["agree_with_ai"] = np.where(
    df["with_ai"] == 1,
    (df["reader_pred"] == df["ai_pred_displayed"]).astype(int),
    np.nan,
)
df["ai_correct"] = np.where(
    df["with_ai"] == 1,
    (df["ai_pred_displayed"] == df["y_true"]).astype(int),
    np.nan,
)

df_aid = df[df["with_ai"] == 1].dropna(subset=["ai_pred_displayed", "agree_with_ai", "ai_correct"]).copy()
df_aid["ai_correct"] = df_aid["ai_correct"].astype(int)
df_aid["agree_with_ai"] = df_aid["agree_with_ai"].astype(int)
df_aid["disagree"] = (df_aid["agree_with_ai"] == 0).astype(int)

A = df_aid["ai_correct"].astype(int)
G = df_aid["agree_with_ai"].astype(int)
R = df_aid["correct"].astype(int)

df_aid["hci_type"] = np.select(
    [
        (A == 1) & (G == 1) & (R == 1),
        (A == 1) & (G == 0) & (R == 0),
        (A == 0) & (G == 0) & (R == 1),
        (A == 0) & (G == 1) & (R == 0),
    ],
    [1, 2, 3, 4],
    default=0,
)

df_wrong = df_aid[df_aid["ai_correct"] == 0].copy()
df_wrong["follow_wrong_ai"] = (df_wrong["agree_with_ai"] == 1).astype(int)
df_wrong["override_wrong_ai"] = (df_wrong["agree_with_ai"] == 0).astype(int)

df = ensure_time_features(df, "time_sec")
df_aid = ensure_time_features(df_aid, "time_sec")
df_wrong = ensure_time_features(df_wrong, "time_sec")

# =========================
# 3) DESCRIPTIVE TABLES
# =========================
reader_rows = []
for (reader, group, cond), g in df.groupby(["reader", "group", "condition"]):
    m = sens_spec_acc(g["y_true"].values, g["reader_pred"].values)
    reader_rows.append(
        {
            "reader": reader,
            "group": group,
            "condition": cond,
            "n": len(g),
            "acc": m["acc"],
            "sens": m["sens"],
            "spec": m["spec"],
            "tpr": m["tpr"],
            "fpr": m["fpr"],
            "mean_time_sec": float(np.nanmean(g["time_sec"].values)),
            "cam_rate": float(np.nanmean(g["show_cam"].values)),
            "pgy": float(g["pgy"].iloc[0]) if pd.notna(g["pgy"].iloc[0]) else np.nan,
        }
    )
reader_summary = pd.DataFrame(reader_rows)
reader_summary.to_csv(os.path.join(OUTDIR, "table_reader_summary_by_condition.csv"), index=False)

group_summary = (
    df.groupby(["group", "condition"])
    .agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        cam=("show_cam", "mean"),
        mean_time=("time_sec", "mean"),
        median_time=("time_sec", "median"),
    )
    .reset_index()
)
group_summary.to_csv(os.path.join(OUTDIR, "table_group_summary.csv"), index=False)

hci_counts = (
    df_aid[df_aid["condition"].isin(["optimized", "unreliable"])]
    .groupby(["group", "condition", "hci_type"])
    .size()
    .reset_index(name="count")
)
hci_counts.to_csv(os.path.join(OUTDIR, "table_hci_type_counts.csv"), index=False)

# Case-mix distribution tables by condition (best effort)
case_mix_cols = [c for c in ["orientation", "modality", "ga_strata", "bw_strata"] if c in df.columns]

# `file_key` must exist for the de-duplication below (it is kept when case-mix is merged).
# Still, compute it defensively in case EXTERNAL_CSV_PATH is None.
if "file_key" not in df.columns:
    df["file_key"] = df["filename"].map(normalize_filename)

if case_mix_cols:
    cm_desc = (
        df[["condition", "group", "file_key"] + case_mix_cols]
        .drop_duplicates(subset=["file_key", "condition"])
        .reset_index(drop=True)
    )
    cm_desc.to_csv(os.path.join(OUTDIR, "table_case_mix_distribution_by_condition.csv"), index=False)

# Keep as clean strings (avoids rpy2 conversion warnings when NA forces float/object mixtures)
for c in ["orientation", "modality", "ga_strata", "bw_strata"]:
    if c in df.columns:
        df[c] = df[c].astype("string").fillna("NA")

# =========================
# 4) MODELS
# =========================
from rpy2.robjects import r
try:
    from pymer4.models import Lmer
except Exception as e:
    raise RuntimeError(
        "Failed to import pymer4. You need pymer4 + rpy2 working in this env.\n"
        f"Original error: {repr(e)}"
    )

# Ensure types for modeling
df["reader"] = df["reader"].astype(str)
df["filename"] = df["filename"].astype(str)
df["condition"] = df["condition"].astype(str)
df["group"] = df["group"].astype(str)

# =========================
# 4A) primary_full / EXPLORATORY: Full dataset GLMM (unaided vs optimized vs unreliable)
# =========================
print("Running primary_full GLMM (all conditions; exploratory) ...")
sep_check = df.groupby(["group", "condition"])["correct"].agg(["mean", "count", "sum"])
sep_check.to_csv(os.path.join(OUTDIR, "audit_correct_rate_by_group_condition.csv"))

case_mix_terms = []
if "ga_z" in df.columns and df["ga_z"].notna().any():
    case_mix_terms.append("ga_z")
if "bw_z" in df.columns and df["bw_z"].notna().any():
    case_mix_terms.append("bw_z")
# If GA/BW are already binned (e.g., GA01/BW03), include them as factors instead of z-scores.
if "ga_strata" in df.columns and df["ga_strata"].notna().any() and df["ga_strata"].nunique(dropna=True) > 1:
    case_mix_terms.append("factor(ga_strata)")
if "bw_strata" in df.columns and df["bw_strata"].notna().any() and df["bw_strata"].nunique(dropna=True) > 1:
    case_mix_terms.append("factor(bw_strata)")
if "orientation" in df.columns and df["orientation"].notna().any() and df["orientation"].nunique(dropna=True) > 1:
    case_mix_terms.append("factor(orientation)")
if "modality" in df.columns and df["modality"].notna().any() and df["modality"].nunique(dropna=True) > 1:
    case_mix_terms.append("factor(modality)")
model_formula = "correct ~ condition * group + pgy_within_5"
if len(case_mix_terms) > 0:
    model_formula += " + " + " + ".join(case_mix_terms)
model_formula += " + (1|reader) + (1|filename)"

glmm_model = Lmer(model_formula, data=df, family="binomial")
glmm_results = glmm_model.fit(
    factors={"condition": COND_ORDER, "group": GROUP_ORDER},
    summarize=True,
    control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
)

with open(os.path.join(OUTDIR, "glmm_primary_full_full_correctness_summary.txt"), "w") as f:
    f.write("FORMULA: " + model_formula + "\n\n")
    f.write(str(glmm_results))

coefs = glmm_model.coefs.copy()
est_col = pick_col(coefs, ["Estimate", "estimate"])
se_col = pick_col(coefs, ["SE", "Std. Error", "std.error", "se"])
coefs["OR"] = np.exp(coefs[est_col].astype(float))
coefs["OR_lower"] = np.exp(coefs[est_col].astype(float) - 1.96 * coefs[se_col].astype(float))
coefs["OR_upper"] = np.exp(coefs[est_col].astype(float) + 1.96 * coefs[se_col].astype(float))
coefs.to_csv(os.path.join(OUTDIR, "table_glmm_primary_full_full_correctness_OR.csv"), index=False)


# =========================
# 4A-SUPP) GEE robustness (FULL): clustered by reader
# =========================
print("\nRunning GEE robustness models (clustered by reader) ...")
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

def _safe_gee_fit(formula: str, data: pd.DataFrame, group_col: str, tag: str):
    d = data.copy()

    # --- INSERT THIS FIX ---
    # Convert explicit string types back to object for statsmodels
    for col in d.select_dtypes(include=['string', 'string[python]']).columns:
        d[col] = d[col].astype(object)
    # -----------------------

    before_n = len(d)
    d = d.dropna(subset=[group_col]).copy()
    after_n0 = len(d)

    if "condition" in d.columns:
        d["condition"] = pd.Categorical(d["condition"], categories=COND_ORDER, ordered=True)
    if "group" in d.columns:
        d["group"] = pd.Categorical(d["group"], categories=GROUP_ORDER, ordered=True)
    if "reliability" in d.columns:
        d["reliability"] = pd.Categorical(d["reliability"], categories=["Reliable", "Unreliable"], ordered=True)

    try:
        gee_mod = smf.gee(
            formula=formula,
            groups=d[group_col],
            data=d,
            family=sm.families.Binomial(),
            cov_struct=Exchangeable(),
        )
        gee_res = gee_mod.fit()
    except Exception as e:
        with open(os.path.join(OUTDIR, f"gee_{tag}_FAILED.txt"), "w") as f:
            f.write("FORMULA: " + formula + "\n\n")
            f.write(repr(e))
        return None, None

    with open(os.path.join(OUTDIR, f"gee_{tag}_summary.txt"), "w") as f:
        f.write(f"FORMULA: {formula}\n")
        f.write(f"CLUSTER: {group_col}\n")
        f.write(f"N_before: {before_n}\n")
        f.write(f"N_after_dropna_{group_col}: {after_n0}\n\n")
        f.write(str(gee_res.summary()))

    params = gee_res.params
    bse = gee_res.bse
    pvals = gee_res.pvalues

    tbl = pd.DataFrame(
        {
            "term": params.index,
            "beta": params.values,
            "se": bse.values,
            "p": pvals.values,
            "OR": np.exp(params.values),
            "OR_lower": np.exp(params.values - 1.96 * bse.values),
            "OR_upper": np.exp(params.values + 1.96 * bse.values),
        }
    )
    tbl.to_csv(os.path.join(OUTDIR, f"table_gee_{tag}_OR.csv"), index=False)

    diag = pd.DataFrame(
        {
            "tag": [tag],
            "formula": [formula],
            "cluster": [group_col],
            "n_rows_used": [int(gee_res.nobs)],
            "n_clusters": [int(d[group_col].nunique())],
        }
    )
    diag.to_csv(os.path.join(OUTDIR, f"diag_gee_{tag}.csv"), index=False)

    return gee_res, tbl


gee_case_mix_terms = []
if "ga_z" in df.columns and df["ga_z"].notna().any():
    gee_case_mix_terms.append("ga_z")
if "bw_z" in df.columns and df["bw_z"].notna().any():
    gee_case_mix_terms.append("bw_z")
if "ga_strata" in df.columns and df["ga_strata"].notna().any() and df["ga_strata"].nunique(dropna=True) > 1:
    gee_case_mix_terms.append("C(ga_strata)")
if "bw_strata" in df.columns and df["bw_strata"].notna().any() and df["bw_strata"].nunique(dropna=True) > 1:
    gee_case_mix_terms.append("C(bw_strata)")
if "orientation" in df.columns and df["orientation"].notna().any() and df["orientation"].nunique(dropna=True) > 1:
    gee_case_mix_terms.append("C(orientation)")
if "modality" in df.columns and df["modality"].notna().any() and df["modality"].nunique(dropna=True) > 1:
    gee_case_mix_terms.append("C(modality)")

gee_full_formula = (
    "correct ~ C(condition, Treatment(reference='unaided')) * "
    "C(group, Treatment(reference='Pediatric Radiologist')) + pgy_within_5"
)
if len(gee_case_mix_terms) > 0:
    gee_full_formula += " + " + " + ".join(gee_case_mix_terms)

_safe_gee_fit(gee_full_formula, df, group_col="reader", tag="full_correctness_primary_full")

# =========================
# 4B) AIDED ONLY (change: SET B = Aided = we only refer to it as aided set instead of set B only) (AI-assisted), Unreliable vs Reliable
# =========================
print("\n Building aidedsetonly dataset ...")

def add_sequence_and_period(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out["case_set_clean"] = out["case_set"].astype(str).str.strip().str.lower()
    out["_row_in_sheet"] = out.groupby("reader").cumcount()

    first_set = (
        out.sort_values(["reader", "_row_in_sheet"])
           .groupby("reader")["case_set_clean"]
           .apply(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
           .to_dict()
    )
    out["first_set"] = out["reader"].map(first_set)
    out["sequence"] = np.where(out["first_set"].eq("seta"), "AB",
                      np.where(out["first_set"].eq("setb"), "BA", np.nan))
    out["period"] = np.where(out["case_set_clean"].eq(out["first_set"]), 1,
                     np.where(out["case_set_clean"].isin(["seta", "setb"]), 2, np.nan))
    out["aided_period"] = np.where(out["case_set_clean"].eq("setb"), out["period"], np.nan)
    return out

df = add_sequence_and_period(df)

seq_audit = (
    df[["reader", "group", "sequence"]].drop_duplicates()
      .groupby(["group", "sequence"]).size().reset_index(name="n_readers")
)
seq_audit.to_csv(os.path.join(OUTDIR, "audit_sequence_counts.csv"), index=False)

dfB = df[
    (df["case_set_clean"] == "setb") &
    (df["with_ai"] == 1) &
    (df["condition"].isin(["optimized", "unreliable"]))
].copy()

dfB["reliability"] = dfB["condition"].map({"optimized": "Reliable", "unreliable": "Unreliable"})
dfB["reliability"] = pd.Categorical(dfB["reliability"], categories=["Reliable", "Unreliable"], ordered=True)
dfB["group"] = pd.Categorical(dfB["group"], categories=GROUP_ORDER, ordered=True)
dfB["reader"] = dfB["reader"].astype(str)
dfB["filename"] = dfB["filename"].astype(str)
dfB["aided_period"] = pd.to_numeric(dfB["aided_period"], errors="coerce")

col_check = dfB[["reader", "sequence", "aided_period"]].drop_duplicates()
col_check.to_csv(os.path.join(OUTDIR, "audit_aidedsetonly_sequence_period.csv"), index=False)

def fit_aidedsetonly_glmm(dfin: pd.DataFrame, add_reader_slope: bool = False, add_period: bool = False, add_case_mix: bool = False):
    d = dfin.copy()
    fixed = "correct ~ reliability * group + pgy_within_5"
    if add_period:
        fixed += " + aided_period"

    if add_case_mix:
        cm_terms = []
        if "ga_z" in d.columns and d["ga_z"].notna().any():
            cm_terms.append("ga_z")
        if "bw_z" in d.columns and d["bw_z"].notna().any():
            cm_terms.append("bw_z")
        if "ga_strata" in d.columns and d["ga_strata"].notna().any() and d["ga_strata"].nunique(dropna=True) > 1:
            cm_terms.append("factor(ga_strata)")
        if "bw_strata" in d.columns and d["bw_strata"].notna().any() and d["bw_strata"].nunique(dropna=True) > 1:
            cm_terms.append("factor(bw_strata)")
        if "orientation" in d.columns and d["orientation"].notna().any() and d["orientation"].nunique(dropna=True) > 1:
            cm_terms.append("factor(orientation)")
        if "modality" in d.columns and d["modality"].notna().any() and d["modality"].nunique(dropna=True) > 1:
            cm_terms.append("factor(modality)")
        if len(cm_terms) > 0:
            fixed += " + " + " + ".join(cm_terms)

    if add_reader_slope:
        rand = " + (1 + reliability|reader) + (1|filename)"
    else:
        rand = " + (1|reader) + (1|filename)"

    formula = fixed + rand
    m = Lmer(formula, data=d, family="binomial")
    res = m.fit(
        factors={"reliability": ["Reliable", "Unreliable"], "group": GROUP_ORDER},
        summarize=True,
        control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
    )
    return m, res, formula

print("\n[ Aided (change: SET B = Aided = we only refer to it as aided set instead of set B)-only GLMM ...")
mB, resB, fB = fit_aidedsetonly_glmm(dfB, add_reader_slope=False, add_period=False, add_case_mix=False)
with open(os.path.join(OUTDIR, "glmm_aided_aidedsetonly_summary.txt"), "w") as f:
    f.write("FORMULA: " + fB + "\n\n")
    f.write(str(resB))

coefsB = mB.coefs.copy()
est_colB = pick_col(coefsB, ["Estimate", "estimate"])
se_colB  = pick_col(coefsB, ["SE", "Std. Error", "std.error", "se"])
coefsB["OR"] = np.exp(coefsB[est_colB].astype(float))
coefsB["OR_lower"] = np.exp(coefsB[est_colB].astype(float) - 1.96 * coefsB[se_colB].astype(float))
coefsB["OR_upper"] = np.exp(coefsB[est_colB].astype(float) + 1.96 * coefsB[se_colB].astype(float))
coefsB.to_csv(os.path.join(OUTDIR, "table_glmm_aided_aidedsetonly_OR.csv"), index=False)

# Post-hoc contrasts: Unreliable vs Reliable within each group (Holm-adjusted)
print("Post-hoc: Unreliable vs Reliable within each group (Holm) ...")
phB = mB.post_hoc(marginal_vars="reliability", grouping_vars="group", p_adjust="holm")
phB_df = phB[1] if isinstance(phB, tuple) and len(phB) > 1 else phB
phB_df.to_csv(os.path.join(OUTDIR, "table_aidedsetonly_posthoc_reliability_within_group.csv"), index=False)

# Predicted probabilities at mean within-group PGY (pgy_within_5 = 0)
grid = pd.DataFrame(
    [{"group": g, "reliability": r, "pgy_within_5": 0.0} for g in GROUP_ORDER for r in ["Reliable", "Unreliable"]]
)
try:
    grid["pred_prob"] = mB.predict(grid, verify_predictions=False)
except Exception:
    grid["pred_prob"] = np.nan
grid.to_csv(os.path.join(OUTDIR, "table_aidedsetonly_predicted_probabilities.csv"), index=False)

# =========================
# FIGURE: aidedsetonly forest plot from post-hoc contrasts
# =========================
def build_aidedsetonly_forest_from_posthoc(ph_df: pd.DataFrame) -> pd.DataFrame:
    d = ph_df.copy()
    grp_col = pick_col(d, ["group", "Group"])
    con_col = pick_col(d, ["Contrast", "contrast"])
    est_col = pick_col(d, ["Estimate", "estimate"])
    p_col   = pick_col(d, ["P-val", "p.value", "p", "P", "Pr(>|z|)", "Pr(>|t|)"])

    # Attach CI columns if present; otherwise compute from SE
    if any(c in d.columns for c in ["2.5_ci", "lower.CL", "asymp.LCL", "CI_low", "lower"]) and \
       any(c in d.columns for c in ["97.5_ci", "upper.CL", "asymp.UCL", "CI_high", "upper"]):
        lo_col = pick_col(d, ["2.5_ci", "lower.CL", "asymp.LCL", "CI_low", "lower"])
        hi_col = pick_col(d, ["97.5_ci", "upper.CL", "asymp.UCL", "CI_high", "upper"])
        d["_lo"] = d[lo_col].astype(float)
        d["_hi"] = d[hi_col].astype(float)
    else:
        se_col = pick_col(d, ["SE", "Std. Error", "std.error", "se"])
        d["_lo"] = d[est_col].astype(float) - 1.96 * d[se_col].astype(float)
        d["_hi"] = d[est_col].astype(float) + 1.96 * d[se_col].astype(float)

    out_rows = []
    for g in GROUP_ORDER:
        dg = d[d[grp_col] == g].copy()
        if len(dg) == 0:
            continue
        # pick first available row (should be one contrast per group)
        row = dg.iloc[0]
        est = float(row[est_col])
        p   = float(row[p_col])

        # Ensure direction is "Unreliable vs Reliable"
        cstr = str(row[con_col]).lower().replace(" ", "")
        if "reliable-unreliable" in cstr:
            est = -est
            lo = -float(row["_hi"])
            hi = -float(row["_lo"])
        else:
            lo = float(row["_lo"])
            hi = float(row["_hi"])

        OR = float(np.exp(est))
        L  = float(np.exp(lo))
        U  = float(np.exp(hi))

        out_rows.append(
            {"Group": g, "Contrast": "Unreliable vs Reliable", "OR": OR, "Lower": L, "Upper": U, "P": p, "Sig": p_to_label(p)}
        )
    return pd.DataFrame(out_rows)

forest_aidedsetonly = build_aidedsetonly_forest_from_posthoc(phB_df)
forest_aidedsetonly.to_csv(os.path.join(OUTDIR, "table_forest_aidedsetonly_unreliable_vs_reliable.csv"), index=False)

def plot_forest_aidedsetonly(df_forest: pd.DataFrame, filename: str):
    fig, ax = plt.subplots(figsize=FIG_FOREST_SET_AIDED, constrained_layout=True)
    y = np.arange(len(GROUP_ORDER))[::-1]
    ax.set_xscale("log")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    for i, grp in enumerate(GROUP_ORDER):
        sub = df_forest[df_forest["Group"] == grp]
        if len(sub) == 0:
            continue
        r0 = sub.iloc[0]
        ax.errorbar(
            r0["OR"], y[i],
            xerr=[[r0["OR"] - r0["Lower"]], [r0["Upper"] - r0["OR"]]],
            fmt="o", capsize=4, lw=1.4
        )
        ax.text(
            r0["Upper"] * 1.07, y[i],
            f'{r0["OR"]:.2f} [{r0["Lower"]:.2f}–{r0["Upper"]:.2f}] {r0["Sig"]}',
            va="center", fontsize=9.5
        )

    ax.set_yticks(y)
    ax.set_yticklabels(GROUP_ORDER, fontweight="bold")
    ax.set_xlabel("Odds Ratio (Unreliable vs Reliable), change: SET B = Aided = we only refer to it as aided set instead of set B (95% CI)")

    xmin = max(0.1, float(df_forest["Lower"].min()) * 0.8)
    xmax = float(df_forest["Upper"].max()) * 1.5
    ax.set_xlim(xmin, xmax)
    savefig(filename)

plot_forest_aidedsetonly(forest_aidedsetonly, "fig_forest_aidedsetonly_unreliable_vs_reliable.tif")

# =========================
# SECONDARY (MANUSCRIPT): Time and CAM usage mixed models (aidedsetonly)
# =========================
print("\n[SECONDARY] aidedsetonly reading time LMM (log_time ~ reliability*group + ...) ...")
dfB_time = ensure_time_features(dfB.copy(), "time_sec").dropna(subset=["log_time"]).copy()

time_formula = "log_time ~ reliability * group + pgy_within_5 + (1|reader) + (1|filename)"
m_time2 = Lmer(time_formula, data=dfB_time, family="gaussian")
res_time2 = m_time2.fit(
    factors={"reliability": ["Reliable", "Unreliable"], "group": GROUP_ORDER},
    summarize=True,
    control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
)
with open(os.path.join(OUTDIR, "lmm_aidedsetonly_logtime_secondary_summary.txt"), "w") as f:
    f.write("FORMULA: " + time_formula + "\n\n")
    f.write(str(res_time2))
m_time2.coefs.to_csv(os.path.join(OUTDIR, "table_lmm_aidedsetonly_logtime_secondary_coefs.csv"), index=False)

print("\n[SECONDARY] aidedsetonly CAM usage GLMM (show_cam ~ reliability*group + ...) ...")
dfB_cam = dfB.copy()
dfB_cam["show_cam"] = pd.to_numeric(dfB_cam["show_cam"], errors="coerce").fillna(0).astype(int)

cam_formula = "show_cam ~ reliability * group + pgy_within_5 + (1|reader) + (1|filename)"
m_cam = Lmer(cam_formula, data=dfB_cam, family="binomial")
res_cam = m_cam.fit(
    factors={"reliability": ["Reliable", "Unreliable"], "group": GROUP_ORDER},
    summarize=True,
    control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
)
with open(os.path.join(OUTDIR, "glmm_aidedsetonly_cam_usage_secondary_summary.txt"), "w") as f:
    f.write("FORMULA: " + cam_formula + "\n\n")
    f.write(str(res_cam))
m_cam.coefs.to_csv(os.path.join(OUTDIR, "table_glmm_aidedsetonly_cam_usage_secondary_coefs.csv"), index=False)

# =========================
# SECONDARY (RECOMMENDED): Sensitivity/Specificity via stratified GLMMs (aidedsetonly)
# =========================
print("\n[SECONDARY] aidedsetonly stratified GLMMs for Sensitivity (y_true==1) and Specificity (y_true==0) ...")

def fit_stratified_correctness(df_in: pd.DataFrame, tag: str):
    d = df_in.copy()
    if len(d) == 0:
        return
    formula = "correct ~ reliability * group + pgy_within_5 + (1|reader) + (1|filename)"
    m = Lmer(formula, data=d, family="binomial")
    res = m.fit(
        factors={"reliability": ["Reliable", "Unreliable"], "group": GROUP_ORDER},
        summarize=True,
        control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
    )
    with open(os.path.join(OUTDIR, f"glmm_aidedsetonly_{tag}_summary.txt"), "w") as f:
        f.write("FORMULA: " + formula + "\n\n")
        f.write(str(res))
    m.coefs.to_csv(os.path.join(OUTDIR, f"table_glmm_aidedsetonly_{tag}_coefs.csv"), index=False)

fit_stratified_correctness(dfB[dfB["y_true"] == 1].copy(), "sensitivity_ytrue1")
fit_stratified_correctness(dfB[dfB["y_true"] == 0].copy(), "specificity_ytrue0")

# =========================
# SUPPLEMENT: aidedsetonly sensitivity checks (period, random slope, case-mix, GEE, LOO/LONO, permutation)
# =========================
print("\n[SUPP] aidedsetonly + aided_period adjustment ...")
mB_p, resB_p, fB_p = fit_aidedsetonly_glmm(dfB, add_reader_slope=False, add_period=True, add_case_mix=False)
with open(os.path.join(OUTDIR, "glmm_aidedsetonly_plus_period_summary.txt"), "w") as f:
    f.write("FORMULA: " + fB_p + "\n\n")
    f.write(str(resB_p))
mB_p.coefs.to_csv(os.path.join(OUTDIR, "table_glmm_aidedsetonly_plus_period_coefs.csv"), index=False)

print("[SUPP] Trying random slope (1 + reliability|reader) ...")
try:
    mB_rs, resB_rs, fB_rs = fit_aidedsetonly_glmm(dfB, add_reader_slope=True, add_period=False, add_case_mix=False)
    with open(os.path.join(OUTDIR, "glmm_aidedsetonly_random_slope_summary.txt"), "w") as f:
        f.write("FORMULA: " + fB_rs + "\n\n")
        f.write(str(resB_rs))
except Exception as e:
    with open(os.path.join(OUTDIR, "glmm_aidedsetonly_random_slope_FAILED.txt"), "w") as f:
        f.write(repr(e))

# Case-mix adjustment (if available)
def ensure_case_mix_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure case-mix columns exist with stable dtypes for modeling.

    Important: keep ga_strata/bw_strata as string dtype (not float) to avoid rpy2
    conversion warnings and to allow factor() usage in R formulas.
    """
    out = df.copy()
    for col in ["ga_z", "bw_z"]:
        if col not in out.columns:
            out[col] = np.nan

    for col in ["ga_strata", "bw_strata"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    for col in ["orientation", "modality"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return out
def get_interaction_snapshot(model: Lmer, tag: str) -> dict:
    c = model.coefs.copy()
    c["term"] = c.index if "term" not in c.columns else c["term"].astype(str)
    estc = pick_col(c, ["Estimate", "estimate"])
    pc_candidates = ["P-val", "p.value", "p", "P", "Pr(>|z|)", "Pr(>|t|)"]
    pc = pick_col(c, [x for x in pc_candidates if x in c.columns])
    rows_ = c[c["term"].str.contains("reliability") & c["term"].str.contains("group")]
    out = {"tag": tag}
    for _, r0 in rows_.iterrows():
        term = str(r0["term"])
        out[f"{term}_beta"] = float(r0[estc])
        out[f"{term}_OR"]   = float(np.exp(r0[estc]))
        out[f"{term}_p"]    = float(r0[pc]) if np.isfinite(float(r0[pc])) else np.nan
    return out

print("\n[SUPP] LOO/LONO refits ...")
loo_rows = [get_interaction_snapshot(mB, "FULL")]
for rd in sorted(dfB["reader"].unique()):
    dsub = dfB[dfB["reader"] != rd].copy()
    try:
        m_tmp, _, _ = fit_aidedsetonly_glmm(dsub, add_reader_slope=False, add_period=False, add_case_mix=False)
        loo_rows.append(get_interaction_snapshot(m_tmp, f"LOO_drop_{rd}"))
    except Exception as e:
        loo_rows.append({"tag": f"LOO_drop_{rd}", "error": repr(e)})
pd.DataFrame(loo_rows).to_csv(os.path.join(OUTDIR, "audit_LOO_interaction_stability.csv"), index=False)

neo_readers = sorted(dfB[dfB["group"] == "Neonatologist"]["reader"].unique())
lono_rows = [get_interaction_snapshot(mB, "FULL")]
for rd in neo_readers:
    dsub = dfB[dfB["reader"] != rd].copy()
    try:
        m_tmp, _, _ = fit_aidedsetonly_glmm(dsub, add_reader_slope=False, add_period=False, add_case_mix=False)
        lono_rows.append(get_interaction_snapshot(m_tmp, f"LONO_drop_{rd}"))
    except Exception as e:
        lono_rows.append({"tag": f"LONO_drop_{rd}", "error": repr(e)})
pd.DataFrame(lono_rows).to_csv(os.path.join(OUTDIR, "audit_LONO_interaction_stability.csv"), index=False)

# Permutation LRT for interaction (aidedsetonly)
import random

def permute_case_reliability(dfB_in: pd.DataFrame, stratify_by_y: bool = True, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = dfB_in.copy()
    case_tbl = d[["filename", "y_true", "reliability"]].drop_duplicates("filename").copy()

    if stratify_by_y:
        new_assign = []
        for yval, g in case_tbl.groupby("y_true"):
            # labels = g["reliability"].values.copy()
            labels = g["reliability"].to_numpy().copy() # Explicitly convert to numpy
            rng.shuffle(labels)
            tmp = g.copy()
            tmp["reliability_perm"] = labels
            new_assign.append(tmp[["filename", "reliability_perm"]])
        new_assign = pd.concat(new_assign, ignore_index=True)
    else:
        labels = case_tbl["reliability"].values.copy()
        rng.shuffle(labels)
        new_assign = case_tbl[["filename"]].copy()
        new_assign["reliability_perm"] = labels

    d = d.merge(new_assign, on="filename", how="left")
    d["reliability"] = pd.Categorical(d["reliability_perm"], categories=["Reliable", "Unreliable"], ordered=True)
    d = d.drop(columns=["reliability_perm"], errors="ignore")
    return d

# def lrt_interaction_stat(dfB_in: pd.DataFrame) -> float:
#     from rpy2 import robjects as ro
#     from rpy2.robjects import pandas2ri
#     pandas2ri.activate()

#     ro.r("suppressPackageStartupMessages(library(lme4))")
#     ro.globalenv["dat"] = pandas2ri.py2rpy(dfB_in)

#     ro.r("dat$reliability <- factor(dat$reliability, levels=c('Reliable','Unreliable'))")
#     ro.r("dat$group <- factor(dat$group, levels=c('Pediatric Radiologist','Neonatologist','Radiology Resident'))")

#     full = ro.r("glmer(correct ~ reliability*group + pgy_within_5 + (1|reader) + (1|filename), data=dat, family=binomial, control=glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000)))")
#     red  = ro.r("glmer(correct ~ reliability + group + pgy_within_5 + (1|reader) + (1|filename), data=dat, family=binomial, control=glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000)))")

#     tab = ro.r("anova(red, full, test='Chisq')")
#     chisq = float(ro.r("tab$Chisq[2]")[0])
#     return chisq
def lrt_interaction_stat(dfB_in: pd.DataFrame) -> float:
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    # Import lme4 inside the function to ensure R environment is ready
    ro.r("library(lme4)")
    
    # 1. Pass data to R
    ro.globalenv["dat"] = pandas2ri.py2rpy(dfB_in)
    
    # 2. Define Factors in R
    ro.r("""
        dat$reliability <- factor(dat$reliability, levels=c('Reliable','Unreliable'))
        dat$group <- factor(dat$group, levels=c('Pediatric Radiologist','Neonatologist','Radiology Resident'))
    """)
    
    # 3. Run BOTH models and ANOVA in a single R string to maintain scope
    # Note: We return ONLY the Chisq statistic
    r_script = """
        full_mod <- glmer(correct ~ reliability*group + pgy_within_5 + (1|reader) + (1|filename), 
                          data=dat, family=binomial, 
                          control=glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000)))
                          
        red_mod <- glmer(correct ~ reliability + group + pgy_within_5 + (1|reader) + (1|filename), 
                         data=dat, family=binomial, 
                         control=glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000)))
                         
        # Calculate LRT
        res <- anova(red_mod, full_mod, test='Chisq')
        res$Chisq[2]
    """
    
    try:
        chisq_val = float(ro.r(r_script)[0])
        return chisq_val
    except Exception as e:
        print(f"Permutation fit error: {e}")
        return np.nan

print("\n[SUPP] Permutation test (interaction LRT) ...")
N_PERM = 200
PERM_SEED = 123
STRATIFY_BY_Y = True

try:
    obs_stat = lrt_interaction_stat(dfB)
except Exception as e:
    obs_stat = np.nan
    with open(os.path.join(OUTDIR, "perm_test_LRT_FAILED.txt"), "w") as f:
        f.write(repr(e))

perm_stats = []
if np.isfinite(obs_stat):
    for i in range(N_PERM):
        dperm = permute_case_reliability(dfB, stratify_by_y=STRATIFY_BY_Y, seed=PERM_SEED + i)
        try:
            perm_stats.append(lrt_interaction_stat(dperm))
        except Exception:
            continue
    perm_stats = np.array(perm_stats, dtype=float)
    emp_p = (np.sum(perm_stats >= obs_stat) + 1) / (len(perm_stats) + 1)

    pd.DataFrame(
        {
            "obs_LRT": [obs_stat],
            "n_perm_success": [len(perm_stats)],
            "N_PERM_requested": [N_PERM],
            "empirical_p": [emp_p],
            "stratified_by_y_true": [STRATIFY_BY_Y],
        }
    ).to_csv(os.path.join(OUTDIR, "perm_test_interaction_LRT_summary.csv"), index=False)

    pd.DataFrame({"perm_LRT": perm_stats}).to_csv(os.path.join(OUTDIR, "perm_test_interaction_LRT_values.csv"), index=False)

# =========================
# MECHANISM (SUPP / MECH): verification effort + phenotype estimands
# =========================
print("\n[MECH] Verification-effort LMM: log_time ~ disagree*reliability*group ...")
dfB_time_mech = ensure_time_features(dfB.copy(), "time_sec")
dfB_time_mech["ai_pred_displayed"] = dfB_time_mech.apply(displayed_ai, axis=1)
dfB_time_mech["agree_with_ai"] = (dfB_time_mech["reader_pred"] == dfB_time_mech["ai_pred_displayed"]).astype(int)
dfB_time_mech["disagree"] = (dfB_time_mech["agree_with_ai"] == 0).astype(int)
dfB_time_mech = dfB_time_mech.dropna(subset=["log_time"]).copy()

time_mech_formula = "log_time ~ disagree * reliability * group + pgy_within_5 + (1|reader) + (1|filename)"
m_time = Lmer(time_mech_formula, data=dfB_time_mech, family="gaussian")
res_time = m_time.fit(
    factors={"reliability": ["Reliable", "Unreliable"], "group": GROUP_ORDER},
    summarize=True,
    control="optimizer='bobyqa', optCtrl=list(maxfun=200000)",
)
with open(os.path.join(OUTDIR, "lmm_aidedsetonly_logtime_mechanism_summary.txt"), "w") as f:
    f.write("FORMULA: " + time_mech_formula + "\n\n")
    f.write(str(res_time))
m_time.coefs.to_csv(os.path.join(OUTDIR, "table_lmm_aidedsetonly_logtime_mechanism_coefs.csv"), index=False)

print("\n[MECH] Phenotype estimands (AI-wrong subset) with reader bootstrap ...")
def bootstrap_reader_cluster(df_in: pd.DataFrame, value_col: str, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    readers = df_in["reader"].unique()
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(readers, size=len(readers), replace=True)
        d = pd.concat([df_in[df_in["reader"] == r] for r in samp], ignore_index=True)
        boots.append(d[value_col].mean())
    boots = np.array(boots, dtype=float)
    return float(np.mean(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

dfB_mech = dfB.copy()
dfB_mech["ai_pred_displayed"] = dfB_mech.apply(displayed_ai, axis=1)
dfB_mech["ai_correct"] = (dfB_mech["ai_pred_displayed"] == dfB_mech["y_true"]).astype(int)
dfB_mech["agree_with_ai"] = (dfB_mech["reader_pred"] == dfB_mech["ai_pred_displayed"]).astype(int)

dfB_wrongAI = dfB_mech[dfB_mech["ai_correct"] == 0].copy()
dfB_wrongAI["automation_bias"] = (dfB_wrongAI["agree_with_ai"] == 1).astype(int)
dfB_wrongAI["sentinel"] = (dfB_wrongAI["agree_with_ai"] == 0).astype(int)

rows_out = []
for g in GROUP_ORDER:
    for rel in ["Reliable", "Unreliable"]:
        sub = dfB_wrongAI[(dfB_wrongAI["group"] == g) & (dfB_wrongAI["reliability"] == rel)]
        if len(sub) == 0:
            continue
        ab_mean, ab_lo, ab_hi = bootstrap_reader_cluster(sub, "automation_bias", n_boot=2000, seed=11)
        se_mean, se_lo, se_hi = bootstrap_reader_cluster(sub, "sentinel", n_boot=2000, seed=22)
        rows_out.append(
            {
                "group": g,
                "reliability": rel,
                "n_rows": len(sub),
                "n_readers": sub["reader"].nunique(),
                "automation_bias_rate": ab_mean,
                "ab_ci_low": ab_lo,
                "ab_ci_high": ab_hi,
                "sentinel_rate": se_mean,
                "sent_ci_low": se_lo,
                "sent_ci_high": se_hi,
            }
        )
pd.DataFrame(rows_out).to_csv(os.path.join(OUTDIR, "table_aidedsetonly_phenotype_estimands_bootstrap.csv"), index=False)


# # # =========================
# # # 9) READER-CLUSTER BOOTSTRAP MECHANISM RATES
# # # =========================
# def reader_cluster_bootstrap_rates(df_wrong_in, n_boot=2000, seed=21):
#     rng = np.random.default_rng(seed)
#     readers = df_wrong_in["reader"].unique().tolist()
#     out = []
#     for b in range(n_boot):
#         sampled_readers = rng.choice(readers, size=len(readers), replace=True)
#         boot_df = pd.concat([df_wrong_in[df_wrong_in["reader"] == r] for r in sampled_readers], ignore_index=True)
#         for grp in GROUP_ORDER:
#             for ai in ["reliable", "unreliable"]:
#                 sub = boot_df[(boot_df["group"] == grp) & (boot_df["condition"] == ai)]
#                 if len(sub) == 0:
#                     continue
#                 out.append({
#                     "boot": b,
#                     "group": grp,
#                     "ai_type": ai,
#                     "override_rate": float(sub["override_wrong_ai"].mean()),
#                     "follow_rate": float(sub["follow_wrong_ai"].mean()),
#                 })
#     return pd.DataFrame(out)

# boot_rates = reader_cluster_bootstrap_rates(df_wrong, n_boot=2000, seed=21)
# if len(boot_rates) > 0:
#     boot_rates.to_csv(os.path.join(OUTDIR, "bootstrap_reader_cluster_mechanism_rates.csv"), index=False)

#     mech_ci = (
#         boot_rates.groupby(["group", "ai_type"])
#         .agg(
#             override_mean=("override_rate", "mean"),
#             follow_mean=("follow_rate", "mean"),
#             override_low=("override_rate", lambda x: np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.025)),
#             override_high=("override_rate", lambda x: np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.975)),
#             follow_low=("follow_rate", lambda x: np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.025)),
#             follow_high=("follow_rate", lambda x: np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.975)),

#         )
#         .reset_index()
#     )
#     mech_ci["group"] = pd.Categorical(mech_ci["group"], GROUP_ORDER, ordered=True)
#     mech_ci["ai_type"] = pd.Categorical(mech_ci["ai_type"], ["reliable", "unreliable"], ordered=True)
#     mech_ci = mech_ci.sort_values(["group", "ai_type"]).reset_index(drop=True)
#     mech_ci.to_csv(os.path.join(OUTDIR, "table_mechanism_rates_CI.csv"), index=False)

#     fig, ax = plt.subplots(figsize=FIG_MECH_CI, constrained_layout=True)
#     x = np.arange(len(mech_ci))
#     width = 0.38

#     override_y = mech_ci["override_mean"].values
#     override_err = np.vstack([override_y - mech_ci["override_low"].values, mech_ci["override_high"].values - override_y])

#     follow_y = mech_ci["follow_mean"].values
#     follow_err = np.vstack([follow_y - mech_ci["follow_low"].values, mech_ci["follow_high"].values - follow_y])

#     ax.bar(x - width/2, override_y, width, yerr=override_err, alpha=0.85, label="Sentinel Behavior (Override wrong AI)")
#     ax.bar(x + width/2, follow_y, width, yerr=follow_err, alpha=0.85, label="Automation Bias (Follow wrong AI)")

#     xlabels = [f"{r.group}\n{r.ai_type}" for r in mech_ci.itertuples()]
#     ax.set_xticks(x)
#     ax.set_xticklabels(xlabels, rotation=25, ha="right")
#     ax.set_ylim(0, 1)
#     ax.set_ylabel("Rate among AI-wrong cases")
#     # ax.set_title("Mechanism rates with reader-cluster bootstrap 95% CI (AI-wrong subset)")
#     ax.legend(frameon=False)
#     savefig("fig_mechanism_rates_CI.png")

# =========================
# 9) READER-CLUSTER BOOTSTRAP MECHANISM RATES (UPDATED)
# =========================
def reader_cluster_bootstrap_rates(df_wrong_in, n_boot=2000, seed=21):
    rng = np.random.default_rng(seed)
    readers = df_wrong_in["reader"].unique().tolist()
    out = []
    for b in range(n_boot):
        sampled_readers = rng.choice(readers, size=len(readers), replace=True)
        boot_df = pd.concat([df_wrong_in[df_wrong_in["reader"] == r] for r in sampled_readers], ignore_index=True)
        
        for grp in GROUP_ORDER:
            # --- FIX: Use internal dataframe keys 'optimized' and 'unreliable' ---
            for ai_key in ["optimized", "unreliable"]:
                sub = boot_df[(boot_df["group"] == grp) & (boot_df["condition"] == ai_key)]
                
                # Handle cases where sample might be empty (e.g., Reliable AI makes few errors)
                if len(sub) == 0:
                    override_rate = np.nan
                else:
                    override_rate = float(sub["override_wrong_ai"].mean())
                
                out.append({
                    "boot": b,
                    "group": grp,
                    "ai_type": ai_key,
                    "override_rate": override_rate
                })
    return pd.DataFrame(out)

# 1. Run Bootstrap
print("Running Reader-Cluster Bootstrap for Mechanism Rates...")
boot_rates = reader_cluster_bootstrap_rates(df_wrong, n_boot=2000, seed=21)

if len(boot_rates) > 0:
    # 2. Aggregate CI
    mech_ci = (
        boot_rates.groupby(["group", "ai_type"])
        .agg(
            mean=("override_rate", "mean"),
            low=("override_rate", lambda x: np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.025)),
            high=("override_rate", lambda x: np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.975))
        )
        .reset_index()
    )
    
    # 3. Setup Plotting Data
    # Map internal keys to Display Labels
    ai_label_map = {
        "optimized": "Reliable AI",
        "unreliable": "Error-Injected AI"
    }
    mech_ci["Display_Label"] = mech_ci["ai_type"].map(ai_label_map)
    
    # Filter only the groups we want
    mech_ci = mech_ci[mech_ci["group"].isin(GROUP_ORDER)].copy()
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=FIG_MECH_CI, constrained_layout=True)
    
    # X positions
    x = np.arange(len(GROUP_ORDER))
    width = 0.35  # Width of bars
    
    # Separate data for Reliable vs Error-Injected
    # We use a loop to ensure alignment with GROUP_ORDER
    rel_data = []
    err_data = []
    
    for g in GROUP_ORDER:
        r_row = mech_ci[(mech_ci["group"] == g) & (mech_ci["ai_type"] == "optimized")]
        e_row = mech_ci[(mech_ci["group"] == g) & (mech_ci["ai_type"] == "unreliable")]
        
        # Safe extraction with fallback
        if not r_row.empty:
            rel_data.append((r_row.iloc[0]["mean"], r_row.iloc[0]["mean"]-r_row.iloc[0]["low"], r_row.iloc[0]["high"]-r_row.iloc[0]["mean"]))
        else:
            rel_data.append((0, 0, 0))
            
        if not e_row.empty:
            err_data.append((e_row.iloc[0]["mean"], e_row.iloc[0]["mean"]-e_row.iloc[0]["low"], e_row.iloc[0]["high"]-e_row.iloc[0]["mean"]))
        else:
            err_data.append((0, 0, 0))

    # Convert to arrays for plotting
    rel_means = [d[0] for d in rel_data]
    rel_errs  = np.array([[d[1], d[2]] for d in rel_data]).T
    
    err_means = [d[0] for d in err_data]
    err_errs  = np.array([[d[1], d[2]] for d in err_data]).T

    # Plot Bars
    # Bar 1: Reliable AI (Left)
    ax.bar(x - width/2, rel_means, width, yerr=rel_errs, 
           color=COL["optimized"], alpha=0.9, capsize=5, 
           label="Reliable AI", edgecolor='white')
    
    # Bar 2: Error-Injected AI (Right)
    ax.bar(x + width/2, err_means, width, yerr=err_errs, 
           color=COL["unreliable"], alpha=0.9, capsize=5, 
           label="Error-Injected AI", edgecolor='white')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Sentinel Rate (Correct Override of Wrong AI)", fontsize=12)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(title="AI Assistance Reliability", loc="upper left", frameon=True, fontsize=11)
    
    savefig("fig_mechanism_rates_CI.tif")
    mech_ci.to_csv(os.path.join(OUTDIR, "table_mechanism_rates_CI.csv"), index=False)

# =========================
# primary_full FIGURES (plots from your original pipeline): forest vs unaided, ROC, HCI, time bars, trust/vigilance
#   Keep these, but in manuscript label the vs-unaided forest as Supplement/Exploratory.
# =========================
print("\n[FIG] primary_full contrasts (condition within group) for vs-unaided forest (Supplement) ...")
contrasts_out = glmm_model.post_hoc(
    marginal_vars="condition",
    grouping_vars="group",
    p_adjust="holm",
)

contrasts_df = contrasts_out[1] if isinstance(contrasts_out, tuple) and len(contrasts_out) > 1 else contrasts_out
contrasts_df = contrasts_df.copy()
contrasts_df.to_csv(os.path.join(OUTDIR, "table_glmm_contrasts_primary_full_RAW.csv"), index=False)

grp_col = pick_col(contrasts_df, ["group", "Group"])
con_col = pick_col(contrasts_df, ["Contrast", "contrast", "condition"])
est_colC = pick_col(contrasts_df, ["Estimate", "estimate"])
p_colC = pick_col(contrasts_df, ["P-val", "p.value", "p", "P", "Pr(>|z|)", "Pr(>|t|)"])

if any(c in contrasts_df.columns for c in ["2.5_ci", "lower.CL", "asymp.LCL", "CI_low", "lower"]) and \
   any(c in contrasts_df.columns for c in ["97.5_ci", "upper.CL", "asymp.UCL", "CI_high", "upper"]):
    lo_col = pick_col(contrasts_df, ["2.5_ci", "lower.CL", "asymp.LCL", "CI_low", "lower"])
    hi_col = pick_col(contrasts_df, ["97.5_ci", "upper.CL", "asymp.UCL", "CI_high", "upper"])
    lo_vals = contrasts_df[lo_col].astype(float).values
    hi_vals = contrasts_df[hi_col].astype(float).values
else:
    se_colC = pick_col(contrasts_df, ["SE", "Std. Error", "std.error", "se"])
    est_vals = contrasts_df[est_colC].astype(float).values
    se_vals = contrasts_df[se_colC].astype(float).values
    lo_vals = est_vals - 1.96 * se_vals
    hi_vals = est_vals + 1.96 * se_vals

contrasts_df["OR_raw"] = np.exp(contrasts_df[est_colC].astype(float))
contrasts_df["Lower_raw"] = np.exp(lo_vals)
contrasts_df["Upper_raw"] = np.exp(hi_vals)
contrasts_df["p"] = contrasts_df[p_colC].astype(float)

forest_rows = []
for grp in GROUP_ORDER:
    sub = contrasts_df[contrasts_df[grp_col] == grp].copy()
    if len(sub) == 0:
        continue
    for cond, label in [("optimized", "Reliable AI vs Unaided"), ("unreliable", "Unreliable AI vs Unaided")]:
        rows_ = sub[
            sub[con_col].astype(str).str.contains(cond, case=False)
            & sub[con_col].astype(str).str.contains("unaided", case=False)
        ]
        if len(rows_) == 0:
            continue
        row = rows_.iloc[0]
        OR = float(row["OR_raw"])
        L = float(row["Lower_raw"])
        U = float(row["Upper_raw"])
        pval = float(row["p"])
        cstr = str(row[con_col]).lower().replace(" ", "")
        if ("unaided-" in cstr) and (f"-{cond}" in cstr):
            OR = 1.0 / OR
            L, U = 1.0 / U, 1.0 / L
        forest_rows.append({"Group": grp, "Contrast": label, "OR": OR, "Lower": L, "Upper": U, "P": pval, "Sig": p_to_label(pval)})

forest_tbl = pd.DataFrame(forest_rows)
forest_tbl.to_csv(os.path.join(OUTDIR, "table_forest_primary_full_vs_unaided.csv"), index=False)

# def plot_forest_plot(df_forest: pd.DataFrame, filename: str):
#     fig, ax = plt.subplots(figsize=FIG_FOREST, constrained_layout=True)
#     y_centers = np.arange(len(GROUP_ORDER))[::-1]
#     ax.set_xscale("log")

#     for i, grp in enumerate(GROUP_ORDER):
#         y_c = y_centers[i]
#         sub = df_forest[df_forest["Group"] == grp]
#         if len(sub) == 0:
#             continue

#         rel = sub[sub["Contrast"].str.contains("Reliable", case=False)].iloc[0]
#         ax.errorbar(
#             rel["OR"], y_c + 0.15,
#             xerr=[[rel["OR"] - rel["Lower"]], [rel["Upper"] - rel["OR"]]],
#             fmt="o", color=COL["optimized"], capsize=4, lw=1.4
#         )
#         ax.text(rel["Upper"] * 1.07, y_c + 0.15, f'{rel["OR"]:.2f} [{rel["Lower"]:.2f}–{rel["Upper"]:.2f}] {rel["Sig"]}',
#                 va="center", fontsize=9.3, color=COL["optimized"])

#         unr = sub[sub["Contrast"].str.contains("Unreliable", case=False)].iloc[0]
#         ax.errorbar(
#             unr["OR"], y_c - 0.15,
#             xerr=[[unr["OR"] - unr["Lower"]], [unr["Upper"] - unr["OR"]]],
#             fmt="s", color=COL["unreliable"], capsize=4, lw=1.4
#         )
#         ax.text(unr["Upper"] * 1.07, y_c - 0.15, f'{unr["OR"]:.2f} [{unr["Lower"]:.2f}–{unr["Upper"]:.2f}] {unr["Sig"]}',
#                 va="center", fontsize=9.3, color=COL["unreliable"])

#     ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
#     ax.set_yticks(y_centers)
#     ax.set_yticklabels(GROUP_ORDER, fontweight="bold")
#     ax.set_xlabel("Odds Ratio vs Unaided (95% CI)")
#     xmin = max(0.1, float(df_forest["Lower"].min()) * 0.8)
#     xmax = float(df_forest["Upper"].max()) * 1.5
#     ax.set_xlim(xmin, xmax)


#     # custom_lines = [
#     #     Line2D([0], [0], color=COL["optimized"], marker="o", lw=0, markersize=7, label="Reliable AI"),
#     #     Line2D([0], [0], color=COL["unreliable"], marker="s", lw=0, markersize=7, label="Unreliable AI"),
#     # ]
#     custom_lines = [
#         Line2D([0], [0], color=COL["optimized"], marker="o", lw=0, markersize=7, label="Reliable AI"),
#         Line2D([0], [0], color=COL["unreliable"], marker="s", lw=0, markersize=7, label="Error-Injected AI"),
#     ]
#     ax.legend(handles=custom_lines, frameon=False, loc="lower right")
#     savefig(filename)

# plot_forest_plot(forest_tbl, "fig_forest_primary_full_vs_unaided.tif")
def plot_forest_plot(df_forest: pd.DataFrame, filename: str):
    fig, ax = plt.subplots(figsize=FIG_FOREST, constrained_layout=True)
    y_centers = np.arange(len(GROUP_ORDER))[::-1]
    ax.set_xscale("log")

    for i, grp in enumerate(GROUP_ORDER):
        y_c = y_centers[i]
        sub = df_forest[df_forest["Group"] == grp]
        if len(sub) == 0:
            continue

        rel = sub[sub["Contrast"].str.contains("Reliable", case=False)].iloc[0]
        ax.errorbar(
            rel["OR"], y_c + 0.15,
            xerr=[[rel["OR"] - rel["Lower"]], [rel["Upper"] - rel["OR"]]],
            fmt="o", color=COL["optimized"], capsize=4, lw=1.4
        )
        ax.text(rel["Upper"] * 1.07, y_c + 0.15, f'{rel["OR"]:.2f} [{rel["Lower"]:.2f}–{rel["Upper"]:.2f}] {rel["Sig"]}',
                va="center", fontsize=9.3, color=COL["optimized"])

        unr = sub[sub["Contrast"].str.contains("Unreliable", case=False)].iloc[0]
        ax.errorbar(
            unr["OR"], y_c - 0.15,
            xerr=[[unr["OR"] - unr["Lower"]], [unr["Upper"] - unr["OR"]]],
            fmt="s", color=COL["unreliable"], capsize=4, lw=1.4
        )
        ax.text(unr["Upper"] * 1.07, y_c - 0.15, f'{unr["OR"]:.2f} [{unr["Lower"]:.2f}–{unr["Upper"]:.2f}] {unr["Sig"]}',
                va="center", fontsize=9.3, color=COL["unreliable"])

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    
    # --- FIX STARTS HERE ---
    # 1. Set the specific tick locations
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10])
    
    # 2. Uncomment and configure the formatter to use plain numbers (Scalar)
    #    instead of the default LogFormatter (which does 10^0)
    formatter = mpl.ticker.ScalarFormatter()
    formatter.set_scientific(False) # Forces 1 instead of 1e0
    ax.get_xaxis().set_major_formatter(formatter)
    # --- FIX ENDS HERE ---

    ax.set_yticks(y_centers)
    ax.set_yticklabels(GROUP_ORDER, fontweight="bold")
    ax.set_xlabel("Odds Ratio vs Unaided (95% CI)")
    xmin = max(0.1, float(df_forest["Lower"].min()) * 0.8)
    xmax = float(df_forest["Upper"].max()) * 1.5
    ax.set_xlim(xmin, xmax)

    custom_lines = [
        Line2D([0], [0], color=COL["optimized"], marker="o", lw=0, markersize=7, label="Reliable AI"),
        Line2D([0], [0], color=COL["unreliable"], marker="s", lw=0, markersize=7, label="Error-Injected AI"),
    ]
    ax.legend(handles=custom_lines, frameon=False, loc="lower right")
    savefig(filename)

plot_forest_plot(forest_tbl, "fig_forest_primary_full_vs_unaided.tif")

# =========================
# ROC: operating-point shifts + reliable AI curve + AI points (unchanged)
# =========================
unique_cases = df[["filename", "y_true", "ai_pred_optimized", "ai_pred_unreliable"]].drop_duplicates(subset=["filename"])

opt_cases = unique_cases.dropna(subset=["y_true", "ai_pred_optimized"]).copy()
unr_cases = unique_cases.dropna(subset=["y_true", "ai_pred_unreliable"]).copy()

opt_y = opt_cases["y_true"].astype(int).values
opt_p = opt_cases["ai_pred_optimized"].astype(float).round().astype(int).values
unr_y = unr_cases["y_true"].astype(int).values
unr_p = unr_cases["ai_pred_unreliable"].astype(float).round().astype(int).values

ai_opt_metrics = sens_spec_acc(opt_y, opt_p)
ai_unr_metrics = sens_spec_acc(unr_y, unr_p)

pd.DataFrame(
    [
        {"model": "Reliable AI", "N": len(opt_cases), **ai_opt_metrics},
        {"model": "Unreliable AI", "N": len(unr_cases), **ai_unr_metrics},
    ]
).to_csv(os.path.join(OUTDIR, "ai_standalone_audit.csv"), index=False)

if len(Y_TRUE_AI_CURVE) != len(Y_PROB_AI_CURVE):
    raise ValueError("Y_TRUE_AI_CURVE and Y_PROB_AI_CURVE must have same length.")

fpr_ai, tpr_ai, _ = roc_curve(np.array(Y_TRUE_AI_CURVE).astype(int), np.array(Y_PROB_AI_CURVE).astype(float))
roc_auc = auc(fpr_ai, tpr_ai)

fig, ax = plt.subplots(figsize=FIG_ROC, constrained_layout=True)
ax.plot(fpr_ai, tpr_ai, color="black", alpha=0.15, linestyle="-", linewidth=3.0, zorder=0)

for reader, sub in reader_summary.groupby("reader"):
    sub = sub.set_index("condition").reindex(COND_ORDER).reset_index()
    if sub["fpr"].isna().all() or sub["tpr"].isna().all():
        continue
    u = sub[sub["condition"] == "unaided"]
    o = sub[sub["condition"] == "optimized"]
    r0 = sub[sub["condition"] == "unreliable"]
    if len(u) == 0 or len(o) == 0 or len(r0) == 0:
        continue
    u_fpr, u_tpr = float(u["fpr"].values[0]), float(u["tpr"].values[0])
    o_fpr, o_tpr = float(o["fpr"].values[0]), float(o["tpr"].values[0])
    r_fpr, r_tpr = float(r0["fpr"].values[0]), float(r0["tpr"].values[0])
    if any(np.isnan([u_fpr, u_tpr, o_fpr, o_tpr, r_fpr, r_tpr])):
        continue
    gname = sub["group"].dropna().iloc[0] if sub["group"].notna().any() else None
    if gname in COL:
        ax.plot([u_fpr, o_fpr, r_fpr], [u_tpr, o_tpr, r_tpr], color=COL[gname], alpha=0.18, lw=1.2, zorder=1)

    ax.annotate("", xy=(o_fpr, o_tpr), xytext=(u_fpr, u_tpr),
                arrowprops=dict(arrowstyle="->", color=COL["optimized"], alpha=0.45, lw=1.2))
    ax.annotate("", xy=(r_fpr, r_tpr), xytext=(u_fpr, u_tpr),
                arrowprops=dict(arrowstyle="->", color=COL["unreliable"], alpha=0.45, lw=1.2))

    ax.scatter(u_fpr, u_tpr, s=70, edgecolor="white", linewidth=0.7, color=COL["unaided"], zorder=3)
    ax.scatter(o_fpr, o_tpr, s=70, edgecolor="white", linewidth=0.7, color=COL["optimized"], zorder=3)
    ax.scatter(r_fpr, r_tpr, s=70, edgecolor="white", linewidth=0.7, color=COL["unreliable"], zorder=3)

ax.scatter(ai_unr_metrics["fpr"], ai_unr_metrics["tpr"], s=200, marker="X",
           color=COL["unreliable"], edgecolor="black", linewidth=1.2, zorder=5)

desired_xlim = (-0.02, 0.473)
desired_ylim = (0.4, 1.02)
pts = reader_summary.dropna(subset=["fpr", "tpr"])[["fpr", "tpr"]].copy()
pts = pd.concat([pts, pd.DataFrame([[ai_unr_metrics["fpr"], ai_unr_metrics["tpr"]]], columns=["fpr", "tpr"])], ignore_index=True)
max_fpr = float(pts["fpr"].max())
min_tpr = float(pts["tpr"].min())
if max_fpr > desired_xlim[1] or min_tpr < desired_ylim[0]:
    new_xlim = (min(desired_xlim[0], float(pts["fpr"].min()) - 0.02), max(desired_xlim[1], max_fpr + 0.02))
    new_ylim = (min(desired_ylim[0], min_tpr - 0.02), max(desired_ylim[1], float(pts["tpr"].max()) + 0.02))
    print(f"[ROC WARNING] Data outside zoom. Expanding xlim {desired_xlim}->{new_xlim}, ylim {desired_ylim}->{new_ylim}")
    ax.set_xlim(*new_xlim)
    ax.set_ylim(*new_ylim)
else:
    ax.set_xlim(*desired_xlim)
    ax.set_ylim(*desired_ylim)

ax.plot([0, 1], [0, 1], linewidth=1.0, alpha=0.3, color="black", linestyle="--")
ax.set_xlabel("False Positive Rate (1 − Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")

grp_handles = [Line2D([0], [0], color=COL[g], lw=4, label=g) for g in GROUP_ORDER]
leg1 = ax.legend(handles=grp_handles, frameon=False, loc="lower center", bbox_to_anchor=(0.35, 0.02), title="Specialty")
ax.add_artist(leg1)

# ai_handles = [
#     Line2D([0], [0], marker="X", color="w", markerfacecolor=COL["unreliable"], markeredgecolor="k",
#            markersize=11, label="Unreliable AI (operating point)"),
#     Line2D([0], [0], color="black", alpha=0.3, lw=3, label=f"Reliable AI ROC curve (AUC {roc_auc:.2f})"),
# ]
ai_handles = [
    Line2D([0], [0], marker="X", color="w", markerfacecolor=COL["unreliable"], markeredgecolor="k",
           markersize=11, label="Error-Injected AI (operating point)"),
    Line2D([0], [0], color="black", alpha=0.3, lw=3, label=f"Reliable AI ROC curve (AUC 0.861)"),
]
ax.legend(handles=ai_handles, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02), title="AI")
savefig("fig_operating_point_roc_change.tif")

def plot_operating_shifts_facetted(reader_summary, ai_metrics, filename):
    # Set up grid: 1 row, 3 columns (one for each group)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)
    
    # Define AI Points (Operating Points)
    ai_opt_pt = (ai_metrics['Reliable AI']['fpr'], ai_metrics['Reliable AI']['tpr'])
    ai_unr_pt = (ai_metrics['Unreliable AI']['fpr'], ai_metrics['Unreliable AI']['tpr'])

    for i, group in enumerate(GROUP_ORDER):
        ax = axes[i]
        group_data = reader_summary[reader_summary['group'] == group]
        
        # 1. Plot Background AI Curve (faint)
        ax.plot(fpr_ai, tpr_ai, color="black", alpha=0.1, linestyle="-", lw=1, zorder=0)
        ax.plot([0, 1], [0, 1], color="black", alpha=0.1, linestyle="--", lw=1, zorder=0)
        
        # 2. Plot AI Standalone Points
        # Note: Using COL['optimized'] for Reliable to match your config keys
        # ax.scatter(*ai_opt_pt, s=150, marker='*', color=COL['optimized'], edgecolors='k', zorder=5, label='Reliable AI')
        ax.scatter(*ai_unr_pt, s=150, marker='X', color=COL['unreliable'], edgecolors='k', zorder=5, label='Error-Injected AI')

        # 3. Plot Each Reader in this Group
        for reader, r_data in group_data.groupby('reader'):
            # Set index to condition so we can .loc['optimized'], etc.
            r_data = r_data.set_index('condition')
            
            # --- CORRECTION START ---
            # Ensure all required rows exist before accessing
            if not all(k in r_data.index for k in ['unaided', 'optimized', 'unreliable']):
                continue

            # Extract coordinates
            u_pt = (r_data.loc['unaided']['fpr'], r_data.loc['unaided']['tpr'])
            
            # Reliable AI (Key is 'optimized' in dataframe)
            opt_pt = (r_data.loc['optimized']['fpr'], r_data.loc['optimized']['tpr']) 
            
            # Unreliable AI (Key is 'unreliable' in dataframe)
            unr_pt = (r_data.loc['unreliable']['fpr'], r_data.loc['unreliable']['tpr'])

            # Safety check for NaNs
            if any(np.isnan(p) for pt in [u_pt, opt_pt, unr_pt] for p in pt): 
                continue

            # Plot Points
            # Unaided = Blue
            ax.scatter(*u_pt, color=COL['unaided'], s=60, alpha=0.6, zorder=3)
            # Reliable/Optimized = Green
            ax.scatter(*opt_pt, color=COL['optimized'], s=60, alpha=0.6, zorder=3)
            # Unreliable = Red
            ax.scatter(*unr_pt, color=COL['unreliable'], s=60, alpha=0.6, zorder=3)

            # Draw Arrows (Unaided -> Reliable)
            ax.annotate("", xy=opt_pt, xytext=u_pt, 
                        arrowprops=dict(arrowstyle="->", color=COL['optimized'], alpha=0.4, lw=1.5))
            
            # Draw Arrows (Unaided -> Unreliable)
            ax.annotate("", xy=unr_pt, xytext=u_pt, 
                        arrowprops=dict(arrowstyle="->", color=COL['unreliable'], alpha=0.4, lw=1.5))
            # --- CORRECTION END ---

        ax.set_title(group, fontsize=14, fontweight='bold')
        ax.set_xlim(-0.02, 0.6) # Zoom in on relevant area
        ax.set_ylim(0.4, 1.02)
        
        if i == 0:
            ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
        ax.set_xlabel("1 - Specificity (FPR)", fontsize=12)
    
    # Create Custom Legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL['unaided'], label='Unaided'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL['optimized'], label='With Reliable AI'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL['unreliable'], label='With Error-Injected AI (operating point)'),
        # Line2D([0], [0], marker='*', color='w', markerfacecolor=COL['optimized'], markeredgecolor='k', markersize=10, label='Reliable Model'),
        Line2D([0], [0], color="black", alpha=0.3, lw=3, label=f"Reliable AI ROC curve (AUC 0.861)"),
        Line2D([0], [0], marker='X', color='w', markerfacecolor=COL['unreliable'], markeredgecolor='k', markersize=10, label='Error Model')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    savefig(filename)

# Run it
ai_metrics_dict = {
    'Reliable AI': {'fpr': ai_opt_metrics['fpr'], 'tpr': ai_opt_metrics['tpr']}, 
    'Unreliable AI': {'fpr': ai_unr_metrics['fpr'], 'tpr': ai_unr_metrics['tpr']}
}

plot_operating_shifts_facetted(reader_summary, ai_metrics_dict, "fig_operating_point_facetted.tif")

# =========================
# HCI TYPES STACKED BARS (unchanged)
# =========================
hci = df_aid[df_aid["condition"].isin(["optimized", "unreliable"])].copy()
hci = hci[hci["hci_type"].isin([1, 2, 3, 4])].copy()

hci_tab = hci.groupby(["group", "condition", "hci_type"]).size().reset_index(name="count")
hci_tab["total"] = hci_tab.groupby(["group", "condition"])["count"].transform("sum")
hci_tab["prop"] = hci_tab["count"] / hci_tab["total"]
pivot = hci_tab.pivot_table(index=["group", "condition"], columns="hci_type", values="prop", fill_value=0).reset_index()
pivot["group"] = pd.Categorical(pivot["group"], GROUP_ORDER, ordered=True)
pivot["condition"] = pd.Categorical(pivot["condition"], ["optimized", "unreliable"], ordered=True)
pivot = pivot.sort_values(["group", "condition"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=FIG_HCI, constrained_layout=True)
x = np.arange(len(pivot))
bottom = np.zeros(len(pivot))

# type_label = {
#     1: "Type 1 (Agreement with correct AI)",
#     2: "Type 2 (Disagreement with correct AI)",
#     3: "Type 3 (Override of incorrect AI)",
#     4: "Type 4 (Agreement with incorrect AI)"
# }
# type_color = {1: COL["type1"], 2: COL["type2"], 3: COL["type3"], 4: COL["type4"]}

# for t_ in [1, 2, 3, 4]:
#     vals = pivot[t_].values if t_ in pivot.columns else np.zeros(len(pivot))
#     ax.bar(x, vals, bottom=bottom, color=type_color[t_], label=type_label[t_], alpha=0.95)
#     bottom += vals

# xlabels = [f"{r['group']}\n{r['condition']}" for _, r in pivot.iterrows()]
# ax.set_xticks(x)
# ax.set_xticklabels(xlabels)
# ax.set_ylabel("Proportion of AI-aided cases")
# ax.set_ylim(0, 1)
# ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.236))
# savefig("fig_hci_types_stacked.tif")
type_label = {
    1: "Type 1 (Agreement with correct AI)",
    2: "Type 2 (Disagreement with correct AI)",
    3: "Type 3 (Override of incorrect AI)",
    4: "Type 4 (Agreement with incorrect AI)"
}
type_color = {1: COL["type1"], 2: COL["type2"], 3: COL["type3"], 4: COL["type4"]}

# 2. Update X-Axis Labels to be Readable
# Map internal codes to display names
cond_map = {
    "optimized": "Reliable AI",
    "unreliable": "Error-Injected AI"
}

# Create cleaner labels
xlabels = []
for _, r in pivot.iterrows():
    # Split group name for wrapping (e.g., "Pediatric Radiologist" -> "Pediatric\nRadiologist")
    grp_clean = r['group'].replace(' ', '\n')
    cond_clean = cond_map.get(r['condition'], r['condition'])
    xlabels.append(f"{grp_clean}\n({cond_clean})")

# 3. Plotting
for t_ in [1, 2, 3, 4]:
    vals = pivot[t_].values if t_ in pivot.columns else np.zeros(len(pivot))
    ax.bar(x, vals, bottom=bottom, color=type_color[t_], label=type_label[t_], alpha=0.95)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=10) # Set updated labels
ax.set_ylabel("Proportion of AI-aided cases")
ax.set_ylim(0, 1)
# Move legend outside or to bottom to prevent blocking data
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

savefig("fig_hci_types_stacked.tif")

# =========================
# HCI TYPES STACKED BARS (UPDATED: Success vs Failure Order)
# =========================
print("\n[FIG] Generating HCI Stacked Bar Chart (Ordered: 1, 3, 2, 4)...")

# 1. Prepare Data
hci = df_aid[df_aid["condition"].isin(["optimized", "unreliable"])].copy()
hci = hci[hci["hci_type"].isin([1, 2, 3, 4])].copy()

# Calculate counts and proportions
hci_tab = hci.groupby(["group", "condition", "hci_type"]).size().reset_index(name="count")
hci_tab["total"] = hci_tab.groupby(["group", "condition"])["count"].transform("sum")
hci_tab["prop"] = hci_tab["count"] / hci_tab["total"]

# Pivot: Rows = Group/Condition, Cols = HCI Type
pivot = hci_tab.pivot_table(index=["group", "condition"], columns="hci_type", values="prop", fill_value=0).reset_index()

# Ensure categorical ordering
pivot["group"] = pd.Categorical(pivot["group"], GROUP_ORDER, ordered=True)
pivot["condition"] = pd.Categorical(pivot["condition"], ["optimized", "unreliable"], ordered=True)
pivot = pivot.sort_values(["group", "condition"]).reset_index(drop=True)

# 2. Setup Plot
fig, ax = plt.subplots(figsize=FIG_HCI, constrained_layout=True)
x = np.arange(len(pivot))
bottom = np.zeros(len(pivot))

# --- CONFIG: NEW ORDER 1 -> 3 -> 2 -> 4 ---
# Rationale: Type 1 + Type 3 = Human Correct (Bottom stack)
#            Type 2 + Type 4 = Human Incorrect (Top stack)
stack_order = [1, 3, 2, 4] 

type_label = {
    1: "Type 1: Agreement w/ Correct AI\n(Both Correct)",
    3: "Type 3: Sentinel Behavior\n(Reader Correct, AI Wrong)",
    2: "Type 2: Disagreement w/ Correct AI\n(Reader Wrong, AI Correct)",
    4: "Type 4: Automation Bias\n(Both Wrong)"
}

# Colors matching your manuscript palette
type_color = {
    1: COL["type1"], # Blue
    3: COL["type3"], # Green (Sentinel) -> Putting successes together
    2: COL["type2"], # Orange
    4: COL["type4"]  # Red (Bias)
}

# 3. Plotting Loop
for t_ in stack_order:
    # Extract values for this HCI type across all groups/conditions
    if t_ in pivot.columns:
        vals = pivot[t_].values
    else:
        vals = np.zeros(len(pivot))
    
    # Plot bar segment
    bars = ax.bar(x, vals, bottom=bottom, color=type_color[t_], label=type_label[t_], 
                  alpha=0.9, width=0.65, edgecolor='white', linewidth=0.5)
    
    # --- ADD PERCENTAGE LABELS INSIDE BARS ---
    # Only label if segment is large enough (> 4%) to be readable
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        if height > 0.04: 
            ax.text(rect.get_x() + rect.get_width()/2., bottom[idx] + height/2.,
                    f"{height:.1%}",
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    bottom += vals

# 4. Axis Formatting
# Map internal codes to display names
cond_map = {
    "optimized": "Reliable AI",
    "unreliable": "Error-Injected AI"
}

xlabels = []
for _, r in pivot.iterrows():
    grp_clean = r['group'].replace(' ', '\n') # Wrap text
    cond_clean = cond_map.get(r['condition'], r['condition'])
    xlabels.append(f"{grp_clean}\n({cond_clean})")

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=11)
ax.set_ylabel("Proportion of Cases", fontsize=12)
ax.set_ylim(0, 1.05) # Slight headroom
ax.set_title("Distribution of Human-AI Interaction Types (Outcomes)", fontsize=14, pad=15)

# Add a horizontal line to separate Correct (1+3) from Incorrect (2+4) roughly? 
# (Not strictly possible with variable heights, but the color grouping helps)

# Legend positioning (Outside to right, or bottom)
# Reversing legend order to match stack (4 on top, 1 on bottom) visual intuition
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], frameon=True, loc="upper right", 
          bbox_to_anchor=(1.35, 1.0), title="Interaction Type", fontsize=10)

# 5. Save
save_name = "fig_hci_types_stacked_reordered.tif"
print(f"Saving {save_name}...")
savefig(save_name)
pivot.to_csv(os.path.join(OUTDIR, "table_hci_types_proportions.csv"), index=False)
plt.close()

# # =========================
# # TIME: log-time by agree/disagree (descriptive plot; keep in supplement)
# # =========================
# t = df_aid[df_aid["condition"].isin(["optimized", "unreliable"])].copy()
# t = ensure_time_features(t, "time_sec")
# t["agree_label"] = np.where(t["agree_with_ai"] == 1, "Agree", "Disagree")

# fig, ax = plt.subplots(figsize=FIG_TIME, constrained_layout=True)
# labels, means, ses, colors = [], [], [], []
# for gname in GROUP_ORDER:
#     for cond in ["optimized", "unreliable"]:
#         for al in ["Agree", "Disagree"]:
#             sub = t[(t["group"] == gname) & (t["condition"] == cond) & (t["agree_label"] == al)]
#             if len(sub) == 0:
#                 continue
#             labels.append(f"{gname}\n{cond}\n{al}")
#             means.append(float(sub["log_time"].mean()))
#             ses.append(float(sub["log_time"].std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0)
#             colors.append(COL[gname])
# x = np.arange(len(means))
# ax.bar(x, means, yerr=ses, color=colors, alpha=0.85, edgecolor="white", linewidth=0.7)
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=35, ha="right")
# ax.set_ylabel("Mean log(Time) ± SE")
# savefig("fig_time_log_agree_disagree.tif")
import seaborn as sns
# 1. Prepare Data
# Filter relevant rows
t = df_aid[df_aid["condition"].isin(["optimized", "unreliable"])].copy()
t = ensure_time_features(t, "time_sec")

# --- UPDATE 1: Fix Terminology (Map internal codes to Manuscript terms) ---
cond_map = {
    "optimized": "Reliable AI",
    "unreliable": "Error-Injected AI"
}
t["Condition_Label"] = t["condition"].map(cond_map)

# Define Agreement Label
t["agree_label"] = np.where(t["agree_with_ai"] == 1, "Agree with AI", "Disagree with AI")

# --- UPDATE 2: Better Visualization Strategy ---
# We use Seaborn to handle the grouping and Error Bars (SE) automatically.
# Layout: 
#   - Columns: Reader Groups (Neonatologist, etc.)
#   - X-axis: AI Condition (Reliable vs Error-Injected)
#   - Hue (Color): Behavior (Agree vs Disagree) -> This highlights the time difference!

# Define Palette: 
# Agree = Light Grey (Baseline effort)
# Disagree = Bold Color (Vigilance effort - matching your 'Unreliable' color or a distinct high-contrast color)
custom_palette = {"Agree with AI": "#B0B0B0", "Disagree with AI": "#E15759"} 

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Create the Trellis Plot
g = sns.catplot(
    data=t, 
    kind="bar",
    x="Condition_Label", 
    y="log_time", 
    hue="agree_label", 
    col="group",
    col_order=GROUP_ORDER,      # Ensures specific order: Ped Rad -> Neo -> Resident
    order=["Reliable AI", "Error-Injected AI"], # Ensure X-axis order
    palette=custom_palette,
    height=5, 
    aspect=0.9,
    capsize=0.1,                # Adds caps to error bars
    errorbar="se",              # Plots Standard Error (SE) automatically
    legend_out=True
)

# --- UPDATE 3: Formatting for Readability ---
g.set_axis_labels("", "Mean log(Reading Time)")
g.set_titles("{col_name}", fontweight='bold')
g.despine(left=True)

# Fix Legend Title
g._legend.set_title("Interaction Type")

# Add Annotations (Optional but recommended):
# This loops through bars to label the height, helping readers see the exact difference
for ax in g.axes.flat:
    # Add light grid for easier reading
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    for container in ax.containers:
        # Create labels
        labels = [f'{v.get_height():.2f}' for v in container]
        # Add labels to the bars
        ax.bar_label(container, labels=labels, label_type='center', 
                     color='white', weight='bold', fontsize=10, padding=3)

# Save with high resolution
save_name = "fig_time_log_agree_disagree.tif"
print(f"Saving {save_name}...")
plt.savefig(os.path.join(OUTDIR, save_name), format='tiff', dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()

# =========================
# TRUST vs VIGILANCE QUADRANT (unchanged)
# =========================
quad = (
    df_wrong.groupby(["group", "condition"])
    .agg(
        n_wrong=("override_wrong_ai", "size"),
        trust=("follow_wrong_ai", "mean"),
        vigilance=("override_wrong_ai", "mean"),
    )
    .reset_index()
)
quad["group"] = pd.Categorical(quad["group"], GROUP_ORDER, ordered=True)
quad["condition"] = pd.Categorical(quad["condition"], ["optimized", "unreliable"], ordered=True)
quad = quad.sort_values(["group", "condition"]).reset_index(drop=True)
quad.to_csv(os.path.join(OUTDIR, "table_trust_vigilance_quadrant.csv"), index=False)

fig, ax = plt.subplots(figsize=(7.8, 6.4), constrained_layout=True)
marker_map = {"optimized": "o", "unreliable": "s"}

for r0 in quad.itertuples():
    ax.scatter(
        r0.trust, r0.vigilance, s=110,
        marker=marker_map[str(r0.condition)],
        color=COL[str(r0.group)],
        edgecolor="white", linewidth=0.8, alpha=0.95, zorder=3,
    )
    dx, dy = (0.012, 0.012) if str(r0.condition) == "optimized" else (-0.075, -0.04)
    ax.text(
        r0.trust + dx, r0.vigilance + dy,
        f"(n={r0.n_wrong})",
        fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
        zorder=4,
    )

if len(quad) > 0:
    ax.axvline(float(quad["trust"].mean()), linestyle="--", linewidth=1.0, alpha=0.6, color="black")
    ax.axhline(float(quad["vigilance"].mean()), linestyle="--", linewidth=1.0, alpha=0.6, color="black")

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Trust (follow wrong AI rate, AI-wrong subset)")
ax.set_ylabel("Vigilance (override wrong AI rate, AI-wrong subset)")

# --- LEGEND ADDITION STARTS HERE ---
grp_leg = [Line2D([0],[0], marker="o", color="w", label=g, markerfacecolor=COL[g], markersize=9) for g in GROUP_ORDER]
ai_leg = [
    Line2D([0],[0], marker="o", color="black", label="Reliable AI", linestyle="None", markersize=8),
    Line2D([0],[0], marker="s", color="black", label="Error-Injected AI", linestyle="None", markersize=8), # Updated Label
]
leg1 = ax.legend(handles=grp_leg, frameon=False, loc="lower left", title="Group")
ax.add_artist(leg1)
ax.legend(handles=ai_leg, frameon=False, loc="lower right", title="AI Type")

savefig("fig_trust_vigilance_quadrant.tif")

# =========================
# APPENDIX S9: Sensitivity Analysis (Removing Indeterminate Cases)
# =========================
print("\n[SENSITIVITY] Removing Indeterminate Cases and Re-running GLMM...")

# 1. Load External Data to identify Indeterminate cases
ext_df = pd.read_csv(EXTERNAL_CSV_PATH)

# Normalize filenames to match your main df
ext_df['file_key'] = ext_df['png_filename'].apply(normalize_filename)

# Identify Indeterminate cases 
# (Adjust 'Indeterminate' string if your CSV uses 'Equivocal' or code '2')
indeterminate_keys = ext_df[ext_df['Consensus'].astype(str).str.contains("Indeterminate", case=False, na=False)]['file_key'].tolist()

print(f"Found {len(indeterminate_keys)} indeterminate cases to exclude.")

# 2. Filter Main Dataframe
dfB_clean = dfB[~dfB['file_key'].isin(indeterminate_keys)].copy()
print(f"Rows before: {len(dfB)}, Rows after exclusion: {len(dfB_clean)}")

# 3. Re-run the aided change: SET B = Aided = we only refer to it as aided set instead of set B GLMM
mB_sens, resB_sens, fB_sens = fit_aidedsetonly_glmm(dfB_clean, add_reader_slope=False, add_period=False, add_case_mix=False)

# 4. Save Results
with open(os.path.join(OUTDIR, "glmm_sensitivity_no_indeterminate_summary.txt"), "w") as f:
    f.write(f"EXCLUDED INDETERMINATE CASES (n={len(indeterminate_keys)})\n")
    f.write("FORMULA: " + fB_sens + "\n\n")
    f.write(str(resB_sens))

# 5. Extract Interaction Term for Report
# inter_row = mB_sens.coefs[mB_sens.coefs.index.str.contains("Unreliable") & mB_sens.coefs.index.str.contains("Neonatologist")]
inter_row = mB_sens.coefs[mB_sens.coefs.index.str.contains("reliability") & mB_sens.coefs.index.str.contains("Neonatologist")]
print("Interaction (No Indeterminate):")
print(inter_row)

# ==========================================================
# FIGURE S7: Large Saliency Map Mechanism Plot (Fixed Variable Name)
# ==========================================================
print("\n[FIG] Generating Figure S7 (Mechanism/CAM Usage)...")

# 1. Prepare Data (Use df_fig7 to avoid overwriting main df)
data = [
    # Neonatologists
    {'Group': 'Neonatologists', 'CAM': 'Not Viewed', 'AI': 'AI Wrong', 'Count': 61, 'Type': 3}, 
    {'Group': 'Neonatologists', 'CAM': 'Not Viewed', 'AI': 'AI Wrong', 'Count': 11, 'Type': 4}, 
    {'Group': 'Neonatologists', 'CAM': 'Not Viewed', 'AI': 'AI Correct', 'Count': 152, 'Type': 1}, 
    {'Group': 'Neonatologists', 'CAM': 'Not Viewed', 'AI': 'AI Correct', 'Count': 6, 'Type': 2},  
    {'Group': 'Neonatologists', 'CAM': 'Viewed', 'AI': 'AI Wrong', 'Count': 15, 'Type': 3},
    {'Group': 'Neonatologists', 'CAM': 'Viewed', 'AI': 'AI Wrong', 'Count': 0, 'Type': 4},
    {'Group': 'Neonatologists', 'CAM': 'Viewed', 'AI': 'AI Correct', 'Count': 5, 'Type': 1},
    {'Group': 'Neonatologists', 'CAM': 'Viewed', 'AI': 'AI Correct', 'Count': 2, 'Type': 2},
    
    # Residents
    {'Group': 'Radiology Residents', 'CAM': 'Not Viewed', 'AI': 'AI Wrong', 'Count': 49, 'Type': 3},
    {'Group': 'Radiology Residents', 'CAM': 'Not Viewed', 'AI': 'AI Wrong', 'Count': 18, 'Type': 4},
    {'Group': 'Radiology Residents', 'CAM': 'Not Viewed', 'AI': 'AI Correct', 'Count': 175, 'Type': 1},
    {'Group': 'Radiology Residents', 'CAM': 'Not Viewed', 'AI': 'AI Correct', 'Count': 8, 'Type': 2},
    {'Group': 'Radiology Residents', 'CAM': 'Viewed', 'AI': 'AI Wrong', 'Count': 61, 'Type': 3},
    {'Group': 'Radiology Residents', 'CAM': 'Viewed', 'AI': 'AI Wrong', 'Count': 17, 'Type': 4},
    {'Group': 'Radiology Residents', 'CAM': 'Viewed', 'AI': 'AI Correct', 'Count': 87, 'Type': 1},
    {'Group': 'Radiology Residents', 'CAM': 'Viewed', 'AI': 'AI Correct', 'Count': 5, 'Type': 2},

    # Radiologists
    {'Group': 'Pediatric Radiologists', 'CAM': 'Not Viewed', 'AI': 'AI Wrong', 'Count': 98, 'Type': 3},
    {'Group': 'Pediatric Radiologists', 'CAM': 'Not Viewed', 'AI': 'AI Wrong', 'Count': 16, 'Type': 4},
    {'Group': 'Pediatric Radiologists', 'CAM': 'Not Viewed', 'AI': 'AI Correct', 'Count': 281, 'Type': 1},
    {'Group': 'Pediatric Radiologists', 'CAM': 'Not Viewed', 'AI': 'AI Correct', 'Count': 18, 'Type': 2},
    {'Group': 'Pediatric Radiologists', 'CAM': 'Viewed', 'AI': 'AI Wrong', 'Count': 49, 'Type': 3},
    {'Group': 'Pediatric Radiologists', 'CAM': 'Viewed', 'AI': 'AI Wrong', 'Count': 11, 'Type': 4},
    {'Group': 'Pediatric Radiologists', 'CAM': 'Viewed', 'AI': 'AI Correct', 'Count': 28, 'Type': 1},
    {'Group': 'Pediatric Radiologists', 'CAM': 'Viewed', 'AI': 'AI Correct', 'Count': 3, 'Type': 2},
]

df_fig7 = pd.DataFrame(data)

type_labels = {
    1: 'Agreement with\ncorrect AI',
    2: 'Disagreement with\ncorrect AI',
    3: 'Override of\nincorrect AI',
    4: 'Agreement with\nincorrect AI'
}
df_fig7['Interaction'] = df_fig7['Type'].map(type_labels)
df_fig7['AI_Scenario'] = df_fig7['Type'].apply(lambda x: 'AI Correct' if x in [1, 2] else 'AI Wrong')
totals = df_fig7.groupby(['Group', 'CAM', 'AI_Scenario'])['Count'].transform('sum')
df_fig7['Percentage'] = (df_fig7['Count'] / totals) * 100

# --- PLOTTING CONFIGURATION ---
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.0) 

palette = {
    'Agreement with\ncorrect AI': "#4E79A7",
    'Disagreement with\ncorrect AI': "#F28E2B",
    'Override of\nincorrect AI': "#59A14F",
    'Agreement with\nincorrect AI': "#E15759"
}

df_wrong = df_fig7[df_fig7['AI_Scenario'] == 'AI Wrong'].copy()

g = sns.catplot(
    data=df_wrong, kind="bar",
    x="Percentage", y="CAM", hue="Interaction", col="Group",
    palette=palette,
    height=5.5,
    aspect=1.1,
    legend_out=True
)

g.set_axis_labels("Percentage of AI-Wrong Cases (%)", "Saliency Map Usage")
g.set_titles("{col_name}", fontweight='bold')

for ax in g.axes.flat:
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            y = p.get_y() + p.get_height() / 2.
            if width > 10:
                x = width
                align = 'right'
                offset = (-8, 0)
                color = 'white'
            else:
                x = width
                align = 'left'
                offset = (8, 0)
                color = 'black'
            ax.annotate(f'{width:.1f}%', (x, y), xytext=offset, textcoords='offset points',
                        ha=align, va='center', fontsize=12, color=color, weight='bold')

plt.savefig(os.path.join(OUTDIR, "fig5_visual_explainability_mechanism.tif"), format='tiff', dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.close()


# ==========================================================
# PROF REQUEST: Consolidated Performance Table with 95% CIs
# ==========================================================
print("\n[REQ] Generating Table 2 (Consolidated Performance)...")

import scipy.stats as st
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

# --- Helper: Bootstrap CI for AI (Case-level resampling) ---
def get_ai_bootstrap_ci(y_true, y_prob, metric_func, n_boot=2000):
    """Calculates 95% CI for AI using case resampling."""
    stats = []
    y_t = np.array(y_true)
    y_p = np.array(y_prob)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        # Resample indices with replacement
        indices = rng.choice(len(y_t), size=len(y_t), replace=True)
        t_boot = y_t[indices]
        p_boot = y_p[indices]
        if len(np.unique(t_boot)) < 2: continue # Skip if only 1 class present
        stats.append(metric_func(t_boot, p_boot))
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)

# --- A. HUMAN READERS (Aggregate Reader Metrics) ---
# We calculate metrics for EACH reader, then compute Mean/CI across readers in the group.
# This treats "Readers" as the random effect, which is appropriate for summarizing group performance.
reader_metrics = []
for (grp, cond, rdr), sub in df.groupby(["group", "condition", "reader"]):
    m = sens_spec_acc(sub["y_true"], sub["reader_pred"])
    # For binary readers, AUC is Balanced Accuracy = (Sens+Spec)/2
    binary_auc = (m["sens"] + m["spec"]) / 2.0
    reader_metrics.append({
        "group": grp, "condition": cond, "reader": rdr,
        "auc": binary_auc, "sens": m["sens"], "spec": m["spec"]
    })
df_rdr = pd.DataFrame(reader_metrics)

human_rows = []
for (grp, cond), sub in df_rdr.groupby(["group", "condition"]):
    stats = {}
    n_readers = len(sub)
    for metric in ["auc", "sens", "spec"]:
        mean_val = sub[metric].mean()
        sem_val = sub[metric].sem()
        # 95% CI using t-distribution
        if n_readers > 1:
            ci = st.t.interval(0.95, df=n_readers-1, loc=mean_val, scale=sem_val)
        else:
            ci = (np.nan, np.nan)
            
        # Format strings
        if metric == "auc":
            stats[metric] = f"{mean_val:.2f} ({ci[0]:.2f}–{ci[1]:.2f})"
        else:
            # Percentage for Sens/Spec
            stats[metric] = f"{mean_val*100:.1f}" # Prof asked for simple % for Sens/Spec in example, but we can add CIs if needed
            # If you want CIs for Sens/Spec too, uncomment below:
            # stats[metric] = f"{mean_val*100:.1f} ({ci[0]*100:.1f}–{ci[1]*100:.1f})"
    
    human_rows.append({
        "Reader Group": grp, "Condition": cond,
        "AUC (95% CI)": stats["auc"],
        "Sensitivity (%)": stats["sens"],
        "Specificity (%)": stats["spec"]
    })

# --- B. AI MODELS (Bootstrap Cases) ---
# We need unique cases to evaluate AI standalone
ai_df = df.drop_duplicates(subset=["filename"])
ai_rows = []

# Define AI configs: Name, Prediction Column (Binary), Probability Column (Continuous)
# Note: In your dataset, 'ai_pred_optimized' seems to be the probability score (0-1) or binary? 
# Based on your setup, it looks like 'ai_pred_optimized' is the score.
ai_configs = [
    {"name": "Reliable", "prob": "ai_pred_optimized"},
    {"name": "Error-Injected", "prob": "ai_pred_unreliable"} 
]

for config in ai_configs:
    # Filter valid data
    valid = ai_df.dropna(subset=["y_true", config["prob"]])
    yt = valid["y_true"].values
    yp = valid[config["prob"]].values
    
    # Threshold for binary metrics (Standard 0.5 cut)
    ybin = (yp >= 0.5).astype(int)
    
    # 1. AUC
    auc_val = roc_auc_score(yt, yp)
    auc_ci = get_ai_bootstrap_ci(yt, yp, roc_auc_score)
    
    # 2. Sens/Spec
    m = sens_spec_acc(yt, ybin)
    # We don't usually put CIs on single-model Sens/Spec in summary tables unless requested, 
    # but we can. For now, matching the human format (just mean or mean+CI).
    
    ai_rows.append({
        "Reader Group": "AI Model Alone",
        "Condition": config["name"],
        "AUC (95% CI)": f"{auc_val:.2f} ({auc_ci[0]:.2f}–{auc_ci[1]:.2f})",
        "Sensitivity (%)": f"{m['sens']*100:.1f}",
        "Specificity (%)": f"{m['spec']*100:.1f}"
    })

# --- C. MERGE & SAVE ---
table2 = pd.DataFrame(human_rows + ai_rows)

# Rename conditions to Manuscript terms
cond_map = {
    "unaided": "Unaided", 
    "optimized": "Reliable AI", 
    "unreliable": "Error-Injected AI",
    "Reliable": "Reliable AI",
    "Error-Injected": "Error-Injected AI"
}
table2["Condition"] = table2["Condition"].replace(cond_map)

# Custom Sort
grp_order = ["Pediatric Radiologist", "Neonatologist", "Radiology Resident", "AI Model Alone"]
table2["Reader Group"] = pd.Categorical(table2["Reader Group"], categories=grp_order, ordered=True)
table2 = table2.sort_values(["Reader Group", "Condition"])

table2.to_csv(os.path.join(OUTDIR, "Table2_Diagnostic_Performance.csv"), index=False)
print("Saved Table2_Diagnostic_Performance.csv")


# ==========================================================
# 2. GENERATE FIGURE 3 (Forest Plot with Fixed Axis)
# ==========================================================
print("\n[REQ] Generating Figure 3 (Forest Plot) with Scalar Axis...")

# Ensure we use the correct dataframe (from the Full Contextual Model)
# If 'forest_tbl' is not in memory from previous steps, we reconstruct it briefly here 
# (assuming contrasts_df exists from the full model run)
if 'contrasts_df' not in locals():
    print("[WARN] contrasts_df not found. Skipping plot regeneration. Ensure GLMM block ran.")
else:
    # Re-build forest_tbl logic just to be safe
    # (Copying logic from your script)
    forest_rows_regen = []
    for grp in GROUP_ORDER:
        sub = contrasts_df[contrasts_df[grp_col] == grp].copy()
        if len(sub) == 0: continue
        for cond, label in [("optimized", "Reliable AI vs Unaided"), ("unreliable", "Error-Injected AI vs Unaided")]: # Updated label
            rows_ = sub[
                sub[con_col].astype(str).str.contains(cond, case=False)
                & sub[con_col].astype(str).str.contains("unaided", case=False)
            ]
            if len(rows_) == 0: continue
            row = rows_.iloc[0]
            OR = float(row["OR_raw"])
            L = float(row["Lower_raw"])
            U = float(row["Upper_raw"])
            # Invert if contrast is backwards
            cstr = str(row[con_col]).lower().replace(" ", "")
            if ("unaided-" in cstr) and (f"-{cond}" in cstr):
                OR = 1.0 / OR
                L, U = 1.0 / U, 1.0 / L
            forest_rows_regen.append({"Group": grp, "Contrast": label, "OR": OR, "Lower": L, "Upper": U})
    
    forest_tbl_fixed = pd.DataFrame(forest_rows_regen)

    # Plotting Function
    def plot_forest_plot_fixed(df_forest, filename):
        fig, ax = plt.subplots(figsize=FIG_FOREST, constrained_layout=True)
        y_centers = np.arange(len(GROUP_ORDER))[::-1]
        ax.set_xscale("log")

        for i, grp in enumerate(GROUP_ORDER):
            y_c = y_centers[i]
            sub = df_forest[df_forest["Group"] == grp]
            if len(sub) == 0: continue

            # Reliable
            rel = sub[sub["Contrast"].str.contains("Reliable", case=False)].iloc[0]
            ax.errorbar(rel["OR"], y_c + 0.15, 
                        xerr=[[rel["OR"] - rel["Lower"]], [rel["Upper"] - rel["OR"]]], 
                        fmt="o", color=COL["optimized"], capsize=4, lw=1.4)
            ax.text(rel["Upper"]*1.1, y_c + 0.15, f'{rel["OR"]:.2f}', va="center", fontsize=9, color=COL["optimized"])

            # Unreliable (Error-Injected)
            # Match string "Error-Injected" or "Unreliable"
            unr_rows = sub[sub["Contrast"].str.contains("Error-Injected|Unreliable", case=False, regex=True)]
            if len(unr_rows) > 0:
                unr = unr_rows.iloc[0]
                ax.errorbar(unr["OR"], y_c - 0.15, 
                            xerr=[[unr["OR"] - unr["Lower"]], [unr["Upper"] - unr["OR"]]], 
                            fmt="s", color=COL["unreliable"], capsize=4, lw=1.4)
                ax.text(unr["Upper"]*1.1, y_c - 0.15, f'{unr["OR"]:.2f}', va="center", fontsize=9, color=COL["unreliable"])

        # FIX: Manually set xticks and format
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        
        # Scalar Formatter
        ax.set_xticks([0.1, 0.5, 1, 2, 5, 10])
        formatter = mpl.ticker.ScalarFormatter()
        formatter.set_scientific(False)
        ax.get_xaxis().set_major_formatter(formatter)
        
        ax.set_yticks(y_centers)
        ax.set_yticklabels(GROUP_ORDER, fontweight="bold")
        ax.set_xlabel("Odds Ratio vs. Unaided Baseline (95% CI)")
        
        # Legend
        custom_lines = [
            Line2D([0], [0], color=COL["optimized"], marker="o", lw=0, label="Reliable AI vs Unaided"),
            Line2D([0], [0], color=COL["unreliable"], marker="s", lw=0, label="Error-Injected AI vs Unaided"),
        ]
        ax.legend(handles=custom_lines, frameon=False, loc="lower right")
        
        savefig(filename)

    plot_forest_plot_fixed(forest_tbl_fixed, "fig_forest_primary_full_vs_unaided_FIXED.tif")


# ==========================================================
# 3. GENERATE FIGURE S7 (CAM Usage)
# ==========================================================
print("\n[REQ] Generating Figure S7 (Mechanism/CAM Usage)...")

# Prepare Data (Hardcoded based on your clickstream analysis or calculated)
# Assuming 'df_fig7' data structure from previous context
# If you don't have df_fig7 calculated in this script, we can reconstruct it from 'df_aid'
# Calculating from df_aid for accuracy:

df_cam_analysis = df_aid[df_aid["condition"].isin(["optimized", "unreliable"])].copy()
df_cam_analysis["AI_Status"] = np.where(df_cam_analysis["ai_correct"] == 1, "AI Correct", "AI Wrong")
df_cam_analysis["CAM_Status"] = np.where(df_cam_analysis["show_cam"] == 1, "Viewed", "Not Viewed")

# Group and count
cam_summary = df_cam_analysis.groupby(["group", "CAM_Status", "AI_Status", "hci_type"]).size().reset_index(name="Count")

# Map HCI Types
type_labels = {
    1: 'Agreement with\ncorrect AI',
    2: 'Disagreement with\ncorrect AI',
    3: 'Override of\nincorrect AI',
    4: 'Agreement with\nincorrect AI'
}
cam_summary['Interaction'] = cam_summary['hci_type'].map(type_labels)

# Calculate Percentages
# Percentages should be within the (Group x CAM x AI_Status) block
totals = cam_summary.groupby(['group', 'CAM_Status', 'AI_Status'])['Count'].transform('sum')
cam_summary['Percentage'] = (cam_summary['Count'] / totals) * 100

# Filter for AI Wrong only (as per Figure S7 request)
df_wrong_cam = cam_summary[cam_summary['AI_Status'] == 'AI Wrong'].copy()

# Plot
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.0) 

palette = {
    'Agreement with\ncorrect AI': "#4E79A7",
    'Disagreement with\ncorrect AI': "#F28E2B",
    'Override of\nincorrect AI': "#59A14F",
    'Agreement with\nincorrect AI': "#E15759"
}

g = sns.catplot(
    data=df_wrong_cam, kind="bar",
    x="Percentage", y="CAM_Status", hue="Interaction", col="group",
    palette=palette,
    height=5.5,
    aspect=1.1,
    legend_out=True,
    col_order=GROUP_ORDER # Ensure correct order
)

g.set_axis_labels("Percentage of AI-Wrong Cases (%)", "Saliency Map Usage")
g.set_titles("{col_name}", fontweight='bold')

for ax in g.axes.flat:
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            y = p.get_y() + p.get_height() / 2.
            # Label formatting
            if width > 10:
                x = width
                align = 'right'
                offset = (-8, 0)
                color = 'white'
            else:
                x = width
                align = 'left'
                offset = (8, 0)
                color = 'black'
            ax.annotate(f'{width:.1f}%', (x, y), xytext=offset, textcoords='offset points',
                        ha=align, va='center', fontsize=12, color=color, weight='bold')

savefig("Figure_S7_CAM_Mechanism_Large.tif")


print("\nDONE.")
print("Outputs saved to:", os.path.abspath(OUTDIR))