#!/usr/bin/env python3
"""
Summarise all finished runs (no torch needed).

For every run: external AUROC (percentile bootstrap 95% CI), AUPRC, Brier.
For every comparison pair in experiments.json and every shared seed:
paired DeltaAUROC with paired bootstrap 95% CI and DeLong P.
Across seeds: mean, SD, min-max; number of seeds with Delta > 0.

Outputs go to results/ (CSV + summary.md).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
HORIZONS = {"3y": ("label_5y", "prob_5y"), "8y": ("label_10y", "prob_10y")}
N_BOOT, BOOT_SEED = 2000, 42


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, p1, p2):
    """Paired DeLong test (Sun & Xu 2014). Returns (auc1, auc2, delta, se, P)."""
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([p1, p2])
    v01, v10, aucs = [], [], []
    for p in preds:
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    s = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    delta = aucs[0] - aucs[1]
    se = float(np.sqrt(max(var, 1e-12)))
    return aucs[0], aucs[1], delta, se, float(2 * norm.sf(abs(delta) / se))


def boot_idx(n):
    rng = np.random.RandomState(BOOT_SEED)
    return [rng.choice(n, n, replace=True) for _ in range(N_BOOT)]


def ci_auc(y, p, idx):
    vals = [roc_auc_score(y[i], p[i]) for i in idx if len(np.unique(y[i])) == 2]
    return np.percentile(vals, [2.5, 97.5])


def ci_delta(y, p1, p2, idx):
    vals = [roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]) for i in idx if len(np.unique(y[i])) == 2]
    return np.percentile(vals, [2.5, 97.5])


def to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load_runs(root):
    runs = {}
    for f in sorted(Path(root).glob("*/seed*/val_predictions.csv")):
        if (f.parent / "DONE").exists():
            runs[(f.parent.parent.name, int(f.parent.name[4:]))] = pd.read_csv(f, dtype={"patient_id": str}).sort_values("patient_id").reset_index(drop=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    plan = json.load(open(HERE / "experiments.json"))
    runs = load_runs(args.runs)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ids = {tuple(df.patient_id) for df in runs.values()}
    assert len(ids) == 1, "external-validation patient sets differ between runs"
    idx = boot_idx(len(next(iter(runs.values()))))

    rows = []
    for (exp, seed), df in runs.items():
        r = {"exp": exp, "seed": seed}
        for h, (lc, pc) in HORIZONS.items():
            y, p = df[lc].values, df[pc].values
            lo, hi = ci_auc(y, p, idx)
            r.update({f"auroc_{h}": roc_auc_score(y, p), f"auroc_{h}_lo": lo, f"auroc_{h}_hi": hi,
                      f"auprc_{h}": average_precision_score(y, p), f"brier_{h}": brier_score_loss(y, p),
                      f"events_{h}": int(y.sum())})
        rows.append(r)
    met = pd.DataFrame(rows).sort_values(["exp", "seed"])
    met.to_csv(out / "run_metrics.csv", index=False)

    comp = []
    for a, b in plan["comparisons"]:
        for seed in sorted({s for e, s in runs if e == a} & {s for e, s in runs if e == b}):
            da, db = runs[(a, seed)], runs[(b, seed)]
            assert (da.patient_id == db.patient_id).all()
            r = {"A": a, "B": b, "seed": seed}
            for h, (lc, pc) in HORIZONS.items():
                y = da[lc].values
                _, _, d, se, pval = delong(y, da[pc].values, db[pc].values)
                lo, hi = ci_delta(y, da[pc].values, db[pc].values, idx)
                r.update({f"delta_{h}": d, f"delta_{h}_lo": lo, f"delta_{h}_hi": hi, f"delong_p_{h}": pval})
            comp.append(r)
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "comparisons.csv", index=False)

    def agg(x):
        return f"{x.mean():.3f} ± {x.std(ddof=1):.3f} ({x.min():.3f}–{x.max():.3f})" if len(x) > 1 else f"{x.iloc[0]:.3f}"
    se = met.groupby("exp").agg(n_seeds=("seed", "count"), auroc_3y=("auroc_3y", agg), auroc_8y=("auroc_8y", agg)).reset_index()
    se.to_csv(out / "summary_experiments.csv", index=False)
    if len(comp):
        sc = comp.groupby(["A", "B"], sort=False).agg(
            n_seeds=("seed", "count"),
            delta_3y=("delta_3y", agg), seeds_pos_3y=("delta_3y", lambda x: int((x > 0).sum())),
            seeds_p05_3y=("delong_p_3y", lambda x: int((x < 0.05).sum())),
            delta_8y=("delta_8y", agg), seeds_pos_8y=("delta_8y", lambda x: int((x > 0).sum())),
            seeds_p05_8y=("delong_p_8y", lambda x: int((x < 0.05).sum()))).reset_index()
        sc.to_csv(out / "summary_comparisons.csv", index=False)
    else:
        sc = pd.DataFrame()
    with open(out / "summary.md", "w") as f:
        f.write("# R1 retraining summary\n\nAUROC: mean ± SD (min–max) across seeds, external validation (n = 94).\n\n")
        f.write(to_md(se) + "\n\n")
        if len(sc):
            f.write("DeltaAUROC = A − B, paired on the same seed; seeds_pos = seeds with Delta > 0; "
                    "seeds_p05 = seeds with DeLong P < 0.05.\n\n" + to_md(sc) + "\n")
    print(open(out / "summary.md").read())


if __name__ == "__main__":
    main()
