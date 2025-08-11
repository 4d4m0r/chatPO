import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# -------- Config --------
INPUT_CSV = "./csv/perguntas_respostas_llm_avaliado.csv"            # path to your JSON (paste the big array into this file)
ERRORS_CSV = "rag_errors.csv"             # where to save misclassified cases
BIN_REPORT_CSV = "rag_bin_report.csv"     # accuracy by similarity bins
THRESH_SEARCH_STEPS = 500                 # granularity for threshold search

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    required = [
        "question","answer","resposta_llm",
        "sim_cosseno","rougeL_f","len_pred","acerto_bin"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    # Optional columns guard
    if "kw_coverage2+" not in df.columns:
        df["kw_coverage2+"] = np.nan
    # Types
    df["acerto_bin"] = df["acerto_bin"].astype(bool)
    for c in ["sim_cosseno","rougeL_f"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["len_pred"] = pd.to_numeric(df["len_pred"], errors="coerce")
    return df

def summarize(df: pd.DataFrame) -> dict:
    y = df["acerto_bin"].astype(int).values
    sim = df["sim_cosseno"].values
    rouge = df["rougeL_f"].values

    acc = float(df["acerto_bin"].mean())
    mean_sim = float(np.nanmean(sim))
    mean_rouge = float(np.nanmean(rouge))
    mean_len = float(np.nanmean(df["len_pred"]))
    mean_kwcov = float(np.nanmean(df["kw_coverage2+"]))

    # Correlations (point-biserial via Pearson with binary labels)
    corr_sim = float(pd.Series(sim).corr(pd.Series(y))) if not np.isnan(sim).all() else np.nan
    corr_rouge = float(pd.Series(rouge).corr(pd.Series(y))) if not np.isnan(rouge).all() else np.nan

    # ROC-AUC (if both classes present)
    auc_sim = np.nan
    auc_rouge = np.nan
    if len(np.unique(y)) > 1:
        try:
            auc_sim = float(roc_auc_score(y, sim))
            auc_rouge = float(roc_auc_score(y, rouge))
        except Exception:
            pass

    return {
        "n": int(len(df)),
        "accuracy_acerto_bin": acc,
        "mean_cosine": mean_sim,
        "mean_rougeL_f": mean_rouge,
        "mean_len_pred": mean_len,
        "mean_kw_coverage2+": mean_kwcov,
        "corr(cosine,correct)": corr_sim,
        "corr(rougeL,correct)": corr_rouge,
        "roc_auc(cosine)": auc_sim,
        "roc_auc(rougeL)": auc_rouge,
    }

def best_threshold(y_true: np.ndarray, scores: np.ndarray, metric="f1"):
    """Brute-force threshold search to optimize F1 by default."""
    scores = np.array(scores, dtype=float)
    y = np.array(y_true, dtype=int)
    mask = ~np.isnan(scores)
    scores = scores[mask]; y = y[mask]

    if len(scores) == 0:
        return np.nan, {}

    thr_min, thr_max = np.nanmin(scores), np.nanmax(scores)
    if not math.isfinite(thr_min) or not math.isfinite(thr_max):
        return np.nan, {}

    best = {"thr": np.nan, "f1": -1, "prec": np.nan, "rec": np.nan, "acc": np.nan}
    for thr in np.linspace(thr_min, thr_max, THRESH_SEARCH_STEPS):
        y_pred = (scores >= thr).astype(int)
        acc = accuracy_score(y, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1), "prec": float(prec), "rec": float(rec), "acc": float(acc)}

    # Confusion matrix at best threshold
    y_hat = (scores >= best["thr"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    best["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return best["thr"], best

def export_errors(df: pd.DataFrame, path: str | Path):
    errs = df.loc[~df["acerto_bin"], [
        "question","answer","resposta_llm","sim_cosseno","rougeL_f","len_pred","kw_coverage2+"
    ]].copy()
    errs.to_csv(path, index=False, encoding="utf-8")
    return len(errs)

def accuracy_by_bins(df: pd.DataFrame, score_col: str, bins=(0,0.3,0.5,0.7,0.85,1.01)):
    cats = pd.cut(df[score_col], bins=bins, include_lowest=True, right=False)
    grp = df.groupby(cats)["acerto_bin"].mean().reset_index()
    grp.columns = [f"{score_col}_bin", "accuracy"]
    return grp

def main():
    df = load_csv(INPUT_CSV)

    # Overall summary
    summary = summarize(df)
    print("\n=== Overall Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    y = df["acerto_bin"].astype(int).values

    # Best threshold (cosine)
    thr_sim, stats_sim = best_threshold(y, df["sim_cosseno"].values, metric="f1")
    print("\n=== Best Threshold (Cosine) ===")
    print(f"thr: {thr_sim:.4f}")
    for k, v in stats_sim.items():
        if k != "confusion":
            print(f"{k}: {v}")
    print(f"confusion: {stats_sim.get('confusion')}")

    # Best threshold (ROUGE-L F1)
    thr_rouge, stats_rouge = best_threshold(y, df["rougeL_f"].values, metric="f1")
    print("\n=== Best Threshold (ROUGE-L F1) ===")
    print(f"thr: {thr_rouge:.4f}")
    for k, v in stats_rouge.items():
        if k != "confusion":
            print(f"{k}: {v}")
    print(f"confusion: {stats_rouge.get('confusion')}")

    # Export error cases
    n_errors = export_errors(df, ERRORS_CSV)
    print(f"\nSaved {n_errors} error rows to: {ERRORS_CSV}")

    # Bin analysis for cosine & rouge
    bin_cos = accuracy_by_bins(df, "sim_cosseno")
    bin_rouge = accuracy_by_bins(df, "rougeL_f")
    bin_report = bin_cos.merge(bin_rouge, left_index=False, right_index=False, how="outer", suffixes=("_cos", "_rouge"))
    # Clean up for CSV: separate into two tables stacked
    bin_cos["metric"] = "cosine"
    bin_rouge["metric"] = "rougeL"
    stacked = pd.concat([
        bin_cos.rename(columns={"sim_cosseno_bin": "score_bin"}),
        bin_rouge.rename(columns={"rougeL_f_bin": "score_bin"})
    ], ignore_index=True)
    stacked.to_csv(BIN_REPORT_CSV, index=False, encoding="utf-8")
    print(f"Saved bin accuracy report to: {BIN_REPORT_CSV}")

    # Quick top examples
    print("\n=== Top Correct by Lowest Cosine (borderline true positives) ===")
    print(df[df["acerto_bin"]].sort_values("sim_cosseno").head(5)[["question","sim_cosseno"]].to_string(index=False))

    print("\n=== Top Incorrect by Highest Cosine (problematic false positives) ===")
    print(df[~df["acerto_bin"]].sort_values("sim_cosseno", ascending=False).head(5)[["question","sim_cosseno"]].to_string(index=False))

if __name__ == "__main__":
    main()
