"""Calibrated TTFT intervals at routing time, and their behaviour under distribution shift.

Train regime: agentic trace, N=4, 70% load, cap_frac 1.0 (the paper's main configuration).
  seeds 0,1 -> fit, seed 2 -> conformal calibration. Test: seeds 3,4 of the training regime and of five
  shifted regimes (90% load, 5 s tools with 0.25x cache, synthetic sessions, N=8 replicas, chat traffic).

Methods, all giving an 80% interval [lo, hi] for TTFT (and a median):
  A    analytic   : the router's own estimate est_ttft x [r10, r90], ratios from the calibration seed
                    (= split conformal on the multiplicative residual of the paper's estimator)
  QR   quantile regression (gradient-boosted, pinball loss) on log TTFT, q = 0.1 / 0.5 / 0.9
  CQR  QR + split-conformal correction (Romano et al. 2019) from the calibration seed
  A+on / CQR+on: the same, with the conformal offset updated online from observed TTFTs
                    (quantile tracking, Gibbs & Candes 2021 / Angelopoulos et al. 2023). The router sees each
                    request's TTFT when its first token comes back, so this uses only observable information;
                    predictions are made at arrival with the offset as of that moment.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATS = ["turn", "prompt", "match", "new", "match_frac", "pend", "dec", "ctx", "outst", "est", "age_ratio",
         "shadow_fill", "gap", "pend_all", "dec_all", "outst_all", "rate5"]
ALPHA = 0.2           # 80% intervals
ETA = 0.02            # online step (log-TTFT units)
REGIMES = ["base", "util0.9", "slowtool", "synth", "N8", "chat"]
LABEL = {"base": "in-distribution", "util0.9": "90% load", "slowtool": "5 s tools, 0.25x cache",
         "synth": "synthetic sessions", "N8": "8 replicas", "chat": "chat traffic"}


def load(regime, seed):
    return pd.read_csv(f"logs/{regime}__seed{seed}.csv")


def fit_qr(train, residual=False):
    X, y = train[FEATS].values, np.log(train.ttft.values)
    if residual:
        y = y - np.log(train.est.values)
    ms = {}
    for q in (ALPHA / 2, 0.5, 1 - ALPHA / 2):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.08,
                                          max_leaf_nodes=31, min_samples_leaf=40, random_state=0)
        ms[q] = m.fit(X, y)
    return ms


def qr_predict(ms, df, residual=False):
    X = df[FEATS].values
    lo, md, hi = (ms[q].predict(X) for q in (ALPHA / 2, 0.5, 1 - ALPHA / 2))
    if residual:
        b = np.log(df.est.values)
        lo, md, hi = lo + b, md + b, hi + b
    return lo, md, np.maximum(hi, lo)


def conformal_offset(scores, alpha=ALPHA):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


def online(df, lo, hi, q0, eta=ETA, alpha=ALPHA):
    """Interval [lo - q, hi + q] in log space; q is updated when each TTFT is observed (at t_first)."""
    ev = [(a, 0, i) for i, a in enumerate(df.arrival.values)] + [(t, 1, i) for i, t in enumerate(df.t_first.values)]
    ev.sort()
    y = np.log(df.ttft.values)
    q, used = q0, np.empty(len(df))
    for _, kind, i in ev:
        if kind == 0:
            used[i] = q
        else:
            err = float(y[i] < lo[i] - used[i] or y[i] > hi[i] + used[i])
            q = q + eta * (err - alpha)
    return lo - used, hi + used


def metrics(df, lo, hi, md=None):
    """lo/hi/md in log-seconds. Width and interval score in ms."""
    y = df.ttft.values
    L, H = np.exp(lo), np.exp(hi)
    cov = (y >= L) & (y <= H)
    w = df.warm.values
    miss = (df.new.values > 1024) & w
    isc = (H - L) + (2 / ALPHA) * ((L - y) * (y < L) + (y - H) * (y > H))     # Winkler interval score
    out = {"coverage": cov[w].mean(), "width_ms": 1e3 * np.median((H - L)[w]),
           "interval_score_ms": 1e3 * isc[w].mean(),
           "cov_cache_miss": cov[miss].mean() if miss.any() else np.nan,
           "n_miss": int(miss.sum())}
    if md is not None:
        out["mape_median"] = float(np.median(np.abs(np.exp(md[w]) - y[w]) / y[w]))
    return out


def main():
    train = pd.concat([load("base", 0), load("base", 1)])
    train = train[train.warm]
    cal = load("base", 2)
    cal = cal[cal.warm]
    ms = fit_qr(train)
    mr = fit_qr(train, residual=True)

    # analytic: multiplicative residual quantiles of the router's estimator on the calibration seed
    lr = np.log(cal.ttft / cal.est)
    a_lo, a_hi = np.quantile(lr, ALPHA / 2), np.quantile(lr, 1 - ALPHA / 2)
    a_med = np.median(lr)
    a_q0 = 0.0
    # CQR offset
    clo, _, chi = qr_predict(ms, cal)
    yc = np.log(cal.ttft.values)
    cqr_q = conformal_offset(np.maximum(clo - yc, yc - chi))
    rlo, _, rhi = qr_predict(mr, cal, residual=True)
    hyb_q = conformal_offset(np.maximum(rlo - yc, yc - rhi))
    # online offsets start from the calibration-seed values
    rows = []
    for reg in REGIMES:
        for seed in (3, 4):
            df = load(reg, seed)
            le = np.log(df.est.values)
            preds = {
                "A": (le + a_lo, le + a_hi, le + a_med),
                "QR": qr_predict(ms, df),
            }
            lo, md, hi = preds["QR"]
            preds["CQR"] = (lo - cqr_q, hi + cqr_q, md)
            olo, ohi = online(df, le + a_lo, le + a_hi, a_q0)
            preds["A+online"] = (olo, ohi, le + a_med)
            olo, ohi = online(df, lo, hi, cqr_q)
            preds["CQR+online"] = (olo, ohi, md)
            lo2, md2, hi2 = qr_predict(mr, df, residual=True)
            preds["Hybrid"] = (lo2 - hyb_q, hi2 + hyb_q, md2)
            olo, ohi = online(df, lo2, hi2, hyb_q)
            preds["Hybrid+online"] = (olo, ohi, md2)
            for name, (l, h, m) in preds.items():
                r = metrics(df, l, h, m)
                r.update(regime=reg, seed=seed, method=name)
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv("results_intervals.csv", index=False)
    agg = res.groupby(["regime", "method"], sort=False).mean(numeric_only=True).drop(columns="seed")
    pd.set_option("display.width", 200)
    print(agg.round(3).to_string())
    json.dump({"cqr_offset": cqr_q, "analytic_ratio_q10_q50_q90": [float(np.exp(a_lo)), float(np.exp(a_med)),
               float(np.exp(a_hi))], "hybrid_offset": hyb_q, "n_train": len(train), "n_cal": len(cal)}, open("fit_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
