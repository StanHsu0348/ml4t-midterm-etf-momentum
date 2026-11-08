"""Signal and performance statistics: IC with Newey-West CIs, Sharpe inference, DSR."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

NW_LAG = 21


def daily_ic(panel: pd.DataFrame, score: str, label: str, min_names: int = 10) -> pd.Series:
    d = panel[[score, label]].dropna()
    g = d.groupby(level="date")
    n = g.size()
    ic = g.apply(lambda x: stats.spearmanr(x[score], x[label]).statistic if len(x) >= min_names else np.nan)
    return ic[n >= min_names].dropna()


def nw_mean(x: pd.Series | np.ndarray, lag: int = NW_LAG) -> dict:
    """Mean with Newey-West (Bartlett) standard error, 95% CI and two-sided p-value."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    T = len(x)
    if T < 5:
        return {"mean": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan, "t": np.nan, "p": np.nan, "T": T}
    m = x.mean()
    e = x - m
    L = min(lag, T - 1)
    lrv = e @ e / T
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        lrv += 2 * w * (e[k:] @ e[:-k]) / T
    se = np.sqrt(max(lrv, 1e-18) / T)
    t = m / se
    return {"mean": m, "se": se, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "t": t,
            "p": 2 * (1 - stats.norm.cdf(abs(t))), "T": T}


def ic_summary(ic: pd.Series, lag: int = NW_LAG) -> dict:
    s = nw_mean(ic, lag)
    yearly = ic.groupby(ic.index.year).mean()
    s.update({
        "icir_daily": ic.mean() / ic.std(),
        "icir_annual": yearly.mean() / yearly.std() if len(yearly) > 1 else np.nan,
        "share_pos_folds": float((yearly > 0).mean()),
        "n_folds": int(len(yearly)),
        "n_eff": effective_n(ic),
    })
    return s


def effective_n(x: pd.Series, max_lag: int = 63) -> float:
    """Kish/Bartlett effective sample size of an autocorrelated series."""
    x = pd.Series(x).dropna()
    T = len(x)
    if T < 10:
        return float(T)
    rho = [x.autocorr(k) for k in range(1, min(max_lag, T // 4))]
    s = 0.0
    for r in rho:
        if not np.isfinite(r) or r <= 0:
            break
        s += r
    return float(T / (1 + 2 * s))


def benjamini_hochberg(p: pd.Series, alpha: float = 0.05) -> pd.Series:
    p = p.dropna().sort_values()
    m = len(p)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p.values <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = pd.Series(False, index=p.index)
    out.iloc[:k] = True
    return out


# ----------------------------------------------------------------------------- performance
def sharpe(r: pd.Series, ppy: int = 252) -> float:
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else np.nan


def sharpe_ci(r: pd.Series, ppy: int = 252) -> dict:
    """SE(SR) ~ sqrt((1 + SR^2/2)/T) on the per-period Sharpe, annualised (Lo 2002)."""
    r = pd.Series(r).dropna()
    T = len(r)
    sr_p = r.mean() / r.std()
    se_p = np.sqrt((1 + 0.5 * sr_p ** 2) / T)
    k = np.sqrt(ppy)
    return {"sr": sr_p * k, "lo": (sr_p - 1.96 * se_p) * k, "hi": (sr_p + 1.96 * se_p) * k, "se": se_p * k, "T": T}


def psr(r: pd.Series, sr_star_ann: float = 0.0, ppy: int = 252) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012), per-period moments."""
    r = pd.Series(r).dropna()
    T = len(r)
    sr = r.mean() / r.std()
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    sr_star = sr_star_ann / np.sqrt(ppy)
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def expected_max_sr(n_trials: int, var_sr_ann: float) -> float:
    """E[max SR] of n zero-skill strategies (annualised), Bailey et al. (2014)."""
    if n_trials <= 1:
        return 0.0
    g = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr_ann) * ((1 - g) * z1 + g * z2))


def dsr(r: pd.Series, n_trials: int, var_sr_ann: float | None = None, ppy: int = 252) -> dict:
    r = pd.Series(r).dropna()
    if var_sr_ann is None:  # variance of SR estimator under H0 for this sample length, annualised
        var_sr_ann = ppy / len(r)
    sr0 = expected_max_sr(n_trials, var_sr_ann)
    return {"dsr": psr(r, sr0, ppy), "sr0_ann": sr0, "n_trials": n_trials,
            "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False))}


def min_trl_years(sr_ann: float, alpha: float = 0.05, power: float = 0.8) -> float:
    if not np.isfinite(sr_ann) or sr_ann <= 0:
        return np.inf
    return float(((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) / sr_ann) ** 2)


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1


def dd_stats(r: pd.Series) -> dict:
    eq = (1 + r.fillna(0)).cumprod()
    dd = drawdown(eq)
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]]
    recovery = rec.index[0] if len(rec) else None
    return {"max_dd": float(dd.min()), "peak": peak, "trough": trough, "recovery": recovery,
            "duration_sessions": int(((eq.index >= peak) & (eq.index <= trough)).sum()),
            "recovery_sessions": int(((eq.index > trough) & (eq.index <= recovery)).sum()) if recovery is not None else None}


def perf_summary(r: pd.Series, ppy: int = 252) -> dict:
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / ppy
    ci = sharpe_ci(r, ppy)
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(r.std() * np.sqrt(ppy)),
            "sharpe": ci["sr"], "sr_lo": ci["lo"], "sr_hi": ci["hi"], "max_dd": dd_stats(r)["max_dd"]}
