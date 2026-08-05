"""CRRT/TMP analysis: shared definitions (imports, constants, functions).

This module is `analysis.ipynb`'s first cell, extracted so the notebook can
`from common import *`. All heavy inputs (the MIMIC-IV cohort table and the
CRRT operating-time / eICU cohort tables) are read as prepared, relative
parquet files -- see README.md "Prepared inputs" for their column schema.
This module itself never reads MIMIC-IV or eICU raw tables.
"""


import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import t as tdist, norm, mannwhitneyu, chi2_contingency

from lifelines import CoxTimeVaryingFitter, CoxPHFitter, NelsonAalenFitter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings("ignore")
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

SEED   = 42          # global seed
M_IMP  = 5           # number of imputations
STEP_H = 6           # TVC grid step, hours
FU_H   = 28 * 24     # primary endpoint: 28 days, in hours
FU7    = 7 * 24      # secondary endpoint: 7 days, in hours
LM_H   = 12          # landmark time (early window), hours
PENALIZER = 0.1      # Cox L2 penalty; varied in sensitivity analyses

MAIN      = ("median", 100, 12)            # primary surge definition: baseline mode, rise (mmHg), window (h)
BASELINES = ["median", "first"]
RISES     = [50, 75, 100, 125, 150]
WINDOWS   = [6, 12, 18, 24, 48]

CACHE      = Path("cache_final"); CACHE.mkdir(exist_ok=True)
FIGDIR     = Path("figs");        FIGDIR.mkdir(exist_ok=True)

# Prepared analysis inputs. Relative filenames, read from the working directory
# (same convention as COHORT_PARQUET below). Produced by the cohort-construction
# step, which is not part of this repository. See README.md "Prepared inputs"
# for the column schema each file must have.
COHORT_PARQUET      = "cohort_final_v2.parquet"           # MIMIC-IV cohort (source of c / cmeas / tmp)
CRRT_PROC_PARQUET   = "crrt_procedure_intervals.parquet"  # MIMIC-IV CRRT procedure-chart operating intervals
CRRT_INPUT_PARQUET  = "crrt_input_intervals.parquet"      # MIMIC-IV CRRT anticoagulant/replacement-fluid infusion intervals
EICU_COHORT_PARQUET = "eicu_cohort.parquet"                # eICU CRRT cohort with covariates already attached
REBUILD = False      # cohort is loaded from COHORT_PARQUET, not rebuilt here

CONT_M = ["anchor_age", "weight_kg", "sofa_total", "map_value", "platelet",
          "hemoglobin", "lactate", "inr", "aptt", "bilirubin", "blood_flow"]
BIN_M  = ["male", "vaso_use", "mech_vent", "v3_systemic_hep", "v3_prophylaxis"]

DEMO_C = ["anchor_age", "weight_kg"]; DEMO_B = ["male"]
SEV_C  = ["sofa_total", "map_value"]; SEV_B  = ["mech_vent", "vaso_use"]
LAB_C  = ["platelet", "hemoglobin", "lactate", "inr", "aptt", "bilirubin"]
TX_C   = ["blood_flow"];              TX_B   = ["v3_systemic_hep", "v3_prophylaxis"]

CONT_E = ["age_num", "weight_kg", "aps", "map_value", "platelet",
          "hemoglobin", "lactate", "inr", "aptt", "bilirubin"]
BIN_E  = ["male", "vaso_use", "mech_vent", "v3_systemic_hep"]

STAT_VARS = ["tmp_min", "tmp_mean", "tmp_median", "tmp_max",
             "tmp_range", "tmp_time2max", "tmp_slope"]

TMP_CLIP = 1000
BASE_WIN_H = 3; PEAK_START_H = 3
PEAK_WINDOWS = [6, 12, 18, 24, 48]; DROP_FOLLOW_H = 6
MIN_CRRT_RANGE_H = 3

BF_LABELS = [(1, 3, "weak/inconclusive"), (3, 20, "positive"),
             (20, 150, "strong"), (150, np.inf, "very strong")]

def cached(name, loader):
    """Parquet cache: load if present, else run loader() and save."""
    p = CACHE / name
    if p.exists():
        df = pd.read_parquet(p); print(f"  {name} cache: {len(df):,}"); return df
    print(f"  loading {name}..."); t = time.time(); df = loader()
    df.to_parquet(p); print(f"  {name}: {len(df):,} ({time.time()-t:.0f}s)"); return df

def sofa_resp(pf, v):
    if pd.isna(pf): return np.nan
    if pf < 100 and v: return 4
    if pf < 200 and v: return 3
    if pf < 300: return 2
    if pf < 400: return 1
    return 0

def sofa_coag(p):
    if pd.isna(p): return np.nan
    return 0 if p >= 150 else 1 if p >= 100 else 2 if p >= 50 else 3 if p >= 20 else 4

def sofa_liver(b):
    if pd.isna(b): return np.nan
    return 0 if b < 1.2 else 1 if b < 2.0 else 2 if b < 6.0 else 3 if b < 12.0 else 4

def sofa_cv(ne, ep_, dp, db, m):
    if dp > 15 or ep_ > 0.1 or ne > 0.1: return 4
    if (5 < dp <= 15) or (0 < ep_ <= 0.1) or (0 < ne <= 0.1): return 3
    if (0 < dp <= 5) or db > 0: return 2
    if pd.isna(m): return np.nan
    return 1 if m < 70 else 0

def sofa_cns(g):
    if pd.isna(g): return np.nan
    return 0 if g == 15 else 1 if g >= 13 else 2 if g >= 10 else 3 if g >= 6 else 4

def pool_hr(est, var, scale=1.0):
    """Rubin's rules pooling across imputations. Returns (HR, lo, hi, p, Q, se)."""
    mn = len(est); Q = np.mean(est); U = np.mean(var); B = np.var(est, ddof=1)
    T = U + (1 + 1/mn) * B; se = np.sqrt(T)
    dof = (mn-1) * (1 + U/((1+1/mn)*B))**2 if B > 0 else np.inf
    tc = tdist.ppf(0.975, dof) if dof < 1000 else norm.ppf(0.975)
    p = 2*(1 - tdist.cdf(abs(Q/se), dof)) if dof < 1000 else 2*(1 - norm.cdf(abs(Q/se)))
    return (np.exp(Q*scale), np.exp((Q-tc*se)*scale), np.exp((Q+tc*se)*scale), p, Q, se)

def _cox_partial_ll(cp, covars, event_col="event"):
    """Unpenalized partial-likelihood fit for the BIC-approximation Bayes factor.
       Returns (logPL, k=n_covariates, d=n_events)."""
    cols = ["id", "start", "stop", event_col] + covars
    m = CoxTimeVaryingFitter(penalizer=0.0)
    m.fit(cp[cols], id_col="id", start_col="start", stop_col="stop",
          event_col=event_col, show_progress=False)
    return m.log_likelihood_, len(covars), int(cp[event_col].sum())

def bayes_factor_bic(cp_list, exposure, adjust_covars, event_col="event"):
    """BIC-approximation Bayes factor (Volinsky & Raftery 2000).
       full = exposure + adjust_covars, reduced = adjust_covars.
       BIC = -2*logPL + k*ln(d); BF01 = exp((BIC_full - BIC_reduced)/2).
       Averaged over MICE sets (Kass-Wasserman approximation)."""
    LLf, LLr, ds = [], [], []
    for cp in cp_list:
        full_cov = [exposure] + [c for c in adjust_covars if cp[c].nunique() > 1]
        red_cov  = [c for c in adjust_covars if cp[c].nunique() > 1]
        llf, kf, d = _cox_partial_ll(cp, full_cov, event_col)
        llr, kr, _ = _cox_partial_ll(cp, red_cov,  event_col)
        LLf.append(llf); LLr.append(llr); ds.append(d)
    logPL_full = np.mean(LLf); logPL_red = np.mean(LLr); d = np.mean(ds)
    dBIC = (-2*logPL_full + 1*np.log(d)) - (-2*logPL_red + 0*np.log(d))
    BF01 = np.exp(dBIC / 2)
    BF10 = 1.0 / BF01
    label = next(l for lo, hi, l in BF_LABELS
                 if (lo <= (BF01 if BF01 >= 1 else BF10) < hi))
    favors = "H0 (no effect)" if BF01 > 1 else "H1 (effect)"
    return {"BF01": BF01, "BF10": BF10, "dBIC": dBIC, "d": d,
            "logPL_full": logPL_full, "logPL_reduced": logPL_red,
            "favors": favors, "strength": label}

def evalue(hr):
    """E-value (VanderWeele & Ding 2017); inverted if HR<1."""
    hr = hr if hr >= 1 else 1/hr
    return hr + np.sqrt(hr*(hr-1))

def bh_fdr(pv):
    """Benjamini-Hochberg FDR q-values."""
    pv = np.asarray(pv, float); m = len(pv)
    order = np.argsort(pv); ranks = np.empty(m, int); ranks[order] = np.arange(1, m+1)
    q_raw = pv * m / ranks
    q_sorted = q_raw[order]; q_mono = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(m); q[order] = np.clip(q_mono, 0, 1); return q

def hr_per_window(coef_per_h, hours):
    """Convert an hourly log-coefficient to the HR over a given window."""
    return np.exp(coef_per_h * hours)

def risk_diff(df, group_col, event_col, group1=1, group0=0):
    """Observed (unadjusted) absolute risk difference, %. Returns (rd, r1, r0)."""
    r1 = df[df[group_col] == group1][event_col].mean()
    r0 = df[df[group_col] == group0][event_col].mean()
    return (r1 - r0) * 100, r1 * 100, r0 * 100

def mice_impute(df, cont_cols, event_col, time_col, m=M_IMP, seed=SEED):
    """MICE imputation with outcome in the predictor matrix (White & Royston 2009):
       [continuous covariates, event indicator, Nelson-Aalen cumulative hazard H(t)].
       Returns a list of m DataFrames restricted to cont_cols."""
    naf = NelsonAalenFitter()
    naf.fit(df[time_col].values, event_observed=df[event_col].values)
    H = naf.cumulative_hazard_at_times(df[time_col].values).values
    aux = pd.DataFrame({"_event": df[event_col].values, "_naH": H},
                       index=df.index)
    mat = pd.concat([df[cont_cols], aux], axis=1)
    out = []
    for mi in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=20,
                               random_state=seed + mi, sample_posterior=True,
                               initial_strategy="median")
        filled = pd.DataFrame(imp.fit_transform(mat), columns=mat.columns,
                              index=df.index)
        out.append(filled[cont_cols])
    return out

def build(d, Xc, onset_map, cont, binc, end_col="end28_h", ev_col="event28"):
    """Absorbing-state 2-row time-varying-covariate table for surge exposure
       (0 = pre-surge, 1 = post-surge). onset_map = {stay_id: surge time, h}."""
    rows = []
    for i in range(len(d)):
        sid = int(d.iloc[i]["stay_id"]); endh = d.iloc[i][end_col]
        if pd.isna(endh) or endh <= 0: continue
        ev = int(d.iloc[i][ev_col]); sh = onset_map.get(sid, np.nan)
        cov = {cc: float(Xc.iloc[i][cc]) for cc in cont}
        cov.update({cc: int(d.iloc[i][cc]) for cc in binc})
        if pd.isna(sh) or sh >= endh:
            rows.append({"id": sid, "start": 0., "stop": endh, "surge": 0,
                         "event": ev, **cov})
        else:
            sh = max(sh, 0.001)
            rows.append({"id": sid, "start": 0., "stop": sh, "surge": 0,
                         "event": 0, **cov})
            rows.append({"id": sid, "start": sh, "stop": endh, "surge": 1,
                         "event": ev, **cov})
    cp = pd.DataFrame(rows); return cp[cp.stop > cp.start]

def build_ph(d, Xc, onset_map, cont, binc, end_col="end28_h", ev_col="event28"):
    """For the PH test: splits the post-surge interval into STEP_H bins with surge*log(t)."""
    rows = []
    for i in range(len(d)):
        sid = int(d.iloc[i]["stay_id"]); endh = d.iloc[i][end_col]
        if pd.isna(endh) or endh <= 0: continue
        ev = int(d.iloc[i][ev_col]); sh = onset_map.get(sid, np.nan)
        cov = {cc: float(Xc.iloc[i][cc]) for cc in cont}
        cov.update({cc: int(d.iloc[i][cc]) for cc in binc})
        if pd.isna(sh) or sh >= endh:
            rows.append({"id": sid, "start": 0., "stop": endh, "surge": 0,
                         "surge_logt": 0., "event": ev, **cov})
        else:
            sh = max(sh, 0.001)
            rows.append({"id": sid, "start": 0., "stop": sh, "surge": 0,
                         "surge_logt": 0., "event": 0, **cov})
            cuts = sorted(set([sh] + list(np.arange(np.ceil(sh/STEP_H)*STEP_H,
                                                    endh, STEP_H)) + [endh]))
            cuts = [t for t in cuts if sh <= t <= endh]
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b <= a: continue
                mid = (a+b)/2; last = np.isclose(b, endh)
                rows.append({"id": sid, "start": a, "stop": b, "surge": 1,
                             "surge_logt": np.log(mid),
                             "event": ev if last else 0, **cov})
    cp = pd.DataFrame(rows); return cp[cp.stop > cp.start]

def build_grid_mimic(d, Xcont, segs, cont, binc, end_col, ev_col):
    """Tier 1 (MIMIC): splits follow-up into a STEP_H grid with cumulative operating time."""
    rows = []
    for i in range(len(d)):
        sid = int(d.iloc[i]["stay_id"]); end = d.iloc[i][end_col]
        if pd.isna(end) or end <= 0: continue
        ev = int(d.iloc[i][ev_col]); seg = segs.get(sid, [])
        cov = {cc: float(Xcont.iloc[i][cc]) for cc in cont}
        cov.update({cc: int(d.iloc[i][cc]) for cc in binc})
        cuts = sorted(set([0.0] + list(np.arange(STEP_H, end, STEP_H)) + [end]))
        cuts = [t for t in cuts if 0 <= t <= end]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a: continue
            mid = (a+b)/2; last = np.isclose(b, end)
            rows.append({"id": sid, "start": a, "stop": b,
                         "cum_dur_h": cum_on_at(seg, mid),
                         "event": ev if last else 0, **cov})
    cp = pd.DataFrame(rows); return cp[cp.stop > cp.start]

def build_grid_eicu(d, Xcont, cont, binc):
    """Tier 1 (eICU): grid TVC table from the I/O-proxy operating time (term_h)."""
    rows = []
    for i in range(len(d)):
        end = d.iloc[i]["end_h"]; ev = int(d.iloc[i]["event"])
        term = min(d.iloc[i]["term_h"], end)
        if pd.isna(end) or end <= 0: continue
        if pd.isna(term) or term < 0: term = 0.0
        cov = {cc: float(Xcont.iloc[i][cc]) for cc in cont}
        cov.update({cc: int(d.iloc[i][cc]) for cc in binc})
        cuts = sorted(set([0.0] + list(np.arange(STEP_H, end, STEP_H)) + [term, end]))
        cuts = [t for t in cuts if 0 <= t <= end]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a: continue
            mid = (a+b)/2; last = np.isclose(b, end)
            rows.append({"id": i, "start": a, "stop": b,
                         "cum_dur_h": min(mid, term),
                         "event": ev if last else 0, **cov})
    cp = pd.DataFrame(rows); return cp[cp.stop > cp.start]

def surge_onset(tmp, base_mode, rise, win):
    """Time of the early TMP surge. tmp=(stay_id, valuenum, h); baseline within t0+3h,
       first threshold crossing evaluated over (3h, win]. Returns {stay_id: onset time, h}."""
    g = tmp[tmp["h"] <= win]; res = {}
    for sid, grp in g.groupby("stay_id"):
        v = grp["valuenum"].values; h = grp["h"].values
        if len(v) < 2: continue
        if base_mode == "first":
            b = grp[grp["h"] <= 3]["valuenum"]; base = b.iloc[0] if len(b) else v[0]
        elif base_mode == "median":
            b = grp[grp["h"] <= 3]["valuenum"]; base = b.median() if len(b) else np.median(v)
        else:
            base = v.min()
        post = (h > 3) & (h <= win)
        if not post.any(): continue
        vp = v[post]; hp = h[post]; ov = np.where(vp - base >= rise)[0]
        if len(ov): res[sid] = hp[ov[0]]
    return res

def build_segments(raw, t0map, merge_gap_h):
    """Tier 1 operating-time segments: per-patient union with merge_gap.
       raw=(stay_id, starttime, endtime); t0map=stay_id -> t0."""
    segs = {}
    for sid, g in raw.groupby("stay_id"):
        t0 = t0map.get(sid)
        if pd.isna(t0): continue
        sh = (g["starttime"] - t0).dt.total_seconds().values / 3600
        eh = (g["endtime"]   - t0).dt.total_seconds().values / 3600
        ints = sorted(zip(sh.tolist(), eh.tolist()))
        merged = []
        for s, e in ints:
            if merged and (s - merged[-1][1]) <= merge_gap_h:
                if e > merged[-1][1]: merged[-1][1] = e
            else:
                merged.append([s, e])
        out = [(max(s, 0.0), e) for s, e in merged if e > max(s, 0.0)]
        if out: segs[sid] = out
    return segs

def cum_on_at(segs_list, t):
    """Cumulative operating time up to time t."""
    cum = 0.0
    for s, e in segs_list:
        if e <= t: cum += (e - s)
        elif s < t: cum += (t - s)
    return cum

def stats_12h(g):
    """7 landmark summary statistics over [0, LM_H]. g = one patient's measured TMP
       (h, valuenum). slope/time2max use the first measurement as baseline, which
       differs from the surge MAIN (median) baseline; this is a descriptive summary,
       independent of the surge definition."""
    w = g[(g["h"] >= 0) & (g["h"] <= LM_H)]
    if len(w) == 0: return None
    v = w["valuenum"].values; h = w["h"].values
    base = v[0]
    vmin, vmax = v.min(), v.max()
    imax = int(np.argmax(v)); tmax = h[imax]
    slope = (vmax - base) / tmax if tmax > 0 else np.nan
    return {"tmp_min": vmin, "tmp_mean": v.mean(), "tmp_median": np.median(v),
            "tmp_max": vmax, "tmp_range": vmax - vmin,
            "tmp_time2max": tmax, "tmp_slope": slope, "n_w": len(w)}

def fit_surge_cox(d, XC_list, onset_map, cont, binc, end_col="end28_h",
                  ev_col="event28", penalizer=PENALIZER, full_return=False):
    """Tier 2 surge Cox model with Rubin pooling. XC_list = list of imputed sets
       aligned with d. If full_return, also returns the pooled covariate table and
       cp_list (reused by bayes_factor_bic)."""
    est, var = [], []; all_est, all_var = {}, {}; cp_list = []
    for mi in range(len(XC_list)):
        Xc = XC_list[mi][cont] if cont else pd.DataFrame(index=d.index)
        cp = build(d, Xc, onset_map, cont, binc, end_col, ev_col)
        if cp["surge"].sum() == 0 or cp[cp.surge == 1]["event"].sum() == 0:
            return (None, None, None) if full_return else None
        cp_list.append(cp)
        fit = ["surge"] + [cc for cc in (cont + binc) if cp[cc].nunique() > 1]
        m = CoxTimeVaryingFitter(penalizer=penalizer)
        m.fit(cp[["id", "start", "stop", "event"] + fit], id_col="id",
              start_col="start", stop_col="stop", event_col="event",
              show_progress=False)
        est.append(m.summary.loc["surge", "coef"])
        var.append(m.summary.loc["surge", "se(coef)"]**2)
        if full_return:
            for v_ in m.summary.index:
                all_est.setdefault(v_, []).append(m.summary.loc[v_, "coef"])
                all_var.setdefault(v_, []).append(m.summary.loc[v_, "se(coef)"]**2)
    surge_res = pool_hr(est, var)
    if not full_return:
        return surge_res
    pooltab = []
    for v_ in all_est:
        if len(all_est[v_]) < len(XC_list): continue
        hr, lo, hi, p, _, _ = pool_hr(all_est[v_], all_var[v_])
        pooltab.append((v_, hr, lo, hi, p))
    return surge_res, pd.DataFrame(pooltab, columns=["var", "HR", "lo", "hi", "p"]), cp_list

def fit_grid_cox(d, XC_list, segs, cont, binc, end_col, ev_col,
                 builder="mimic", penalizer=PENALIZER, scale=24,
                 mask=None, return_cp=False):
    """Tier 1 cumulative-operating-time Cox model (grid TVC) with Rubin pooling.
       builder='mimic' uses build_grid_mimic(segs); builder='eicu' uses
       build_grid_eicu(d.term_h). mask selects the matching rows when XC_list is a
       superset of d. Returns (HR, lo, hi, p, N, events), plus cp_list if return_cp."""
    est, var = [], []; cp_list = []
    for mi in range(len(XC_list)):
        Xc = (XC_list[mi].loc[mask].reset_index(drop=True)
              if mask is not None else XC_list[mi].reset_index(drop=True))
        Xc = Xc[cont] if cont else pd.DataFrame(index=d.index)
        if builder == "mimic":
            cp = build_grid_mimic(d, Xc, segs, cont, binc, end_col, ev_col)
        else:
            cp = build_grid_eicu(d, Xc, cont, binc)
        if len(cp) == 0 or cp["event"].sum() == 0:
            return ((np.nan,)*4 + (len(d), 0) + (([],) if return_cp else ()))
        cp_list.append(cp)
        fit = ["cum_dur_h"] + [cc for cc in (cont + binc) if cp[cc].nunique() > 1]
        m = CoxTimeVaryingFitter(penalizer=penalizer)
        m.fit(cp[["id", "start", "stop", "event"] + fit], id_col="id",
              start_col="start", stop_col="stop", event_col="event",
              show_progress=False)
        est.append(m.summary.loc["cum_dur_h", "coef"])
        var.append(m.summary.loc["cum_dur_h", "se(coef)"]**2)
    hr, lo, hi, p, _, _ = pool_hr(est, var, scale=scale)
    ev = int(d[ev_col].sum()) if ev_col in d else int(d["event"].sum())
    base = (hr, lo, hi, p, len(d), ev)
    return base + (cp_list,) if return_cp else base

def line(label, r, extra=""):
    """Print one HR result line."""
    if r is None: print(f"  {label:<34} NA (non-identifiable)"); return
    hr, lo, hi, p = r[0], r[1], r[2], r[3]
    print(f"  {label:<34} HR={hr:.3f} [{lo:.3f},{hi:.3f}] "
          f"p={p:.3f}{'*' if p < 0.05 else ''} {extra}")

def line_bf(label, bf):
    """Print one Bayes-factor line."""
    print(f"  {label:<28} BF01={bf['BF01']:.2f} (BF10={bf['BF10']:.2f}) "
          f"ΔBIC={bf['dBIC']:.2f} d={bf['d']:.0f} → {bf['favors']}, {bf['strength']}")

def tab1(df, group_col, cont_vars, bin_vars, cat_vars=None):
    """Table 1: continuous = median + Mann-Whitney U, binary = % + chi-square, categorical = distribution + chi-square."""
    out = []
    for v in cont_vars:
        if v not in df: continue
        g1 = df[df[group_col] == 1][v].dropna()
        g0 = df[df[group_col] == 0][v].dropna(); al = df[v].dropna()
        try: p = mannwhitneyu(g1, g0).pvalue
        except: p = np.nan
        out.append((v, f"{al.median():.1f}", f"{g0.median():.1f}",
                    f"{g1.median():.1f}", p))
    for v in bin_vars:
        if v not in df: continue
        al = df[v]; g0 = df[df[group_col] == 0][v]; g1 = df[df[group_col] == 1][v]
        try: p = chi2_contingency(pd.crosstab(df[group_col], df[v]))[1]
        except: p = np.nan
        out.append((v, f"{al.mean()*100:.0f}%", f"{g0.mean()*100:.0f}%",
                    f"{g1.mean()*100:.0f}%", p))
    if cat_vars:
        for v in cat_vars:
            if v not in df: continue
            try: p = chi2_contingency(pd.crosstab(df[group_col], df[v]))[1]
            except: p = np.nan
            for k, cat in enumerate(sorted(df[v].dropna().unique())):
                al = (df[v] == cat).mean()*100
                a0 = (df[df[group_col] == 0][v] == cat).mean()*100
                a1 = (df[df[group_col] == 1][v] == cat).mean()*100
                out.append((f"{v}={cat}", f"{al:.0f}%", f"{a0:.0f}%",
                            f"{a1:.0f}%", p if k == 0 else np.nan))
    return pd.DataFrame(out, columns=["var", "all", "group=0", "group=1", "p"])

def forest_plot(labels, hrs, los, his, title, fname, ref=1.0, figsize=None):
    """Shared HR forest-plot helper."""
    n = len(labels); figsize = figsize or (7, 0.5*n + 1.2)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)[::-1]
    ax.errorbar(hrs, y, xerr=[np.array(hrs)-np.array(los), np.array(his)-np.array(hrs)],
                fmt="o", color="#2c3e50", ecolor="#7f8c8d", capsize=3, lw=1, ms=5)
    ax.axvline(ref, ls="--", color="#c0392b", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hazard ratio (95% CI)"); ax.set_title(title, fontsize=11, loc="left")
    plt.tight_layout(); plt.savefig(FIGDIR/fname, dpi=150, bbox_inches="tight"); plt.show()

print("common.py loaded")
print(f"  constants: SEED={SEED}, M_IMP={M_IMP}, STEP_H={STEP_H}, FU_H={FU_H}, "
      f"LM_H={LM_H}, PENALIZER={PENALIZER}")
print(f"  MAIN={MAIN}, grid {len(BASELINES)}x{len(RISES)}x{len(WINDOWS)}="
      f"{len(BASELINES)*len(RISES)*len(WINDOWS)} combinations")
print(f"  covariates: MIMIC CONT_M {len(CONT_M)} / BIN_M {len(BIN_M)}, "
      f"eICU CONT_E {len(CONT_E)} / BIN_E {len(BIN_E)}")
print(f"  stats: pool_hr, bayes_factor_bic, evalue, bh_fdr, "
      f"hr_per_window, risk_diff")
print(f"  imputation: mice_impute (includes event + Nelson-Aalen cumulative hazard)")
print(f"  TVC: build, build_ph, build_grid_mimic, build_grid_eicu")
print(f"  fitting: fit_surge_cox, fit_grid_cox (penalizer as argument)")
print(f"  surge/operating time: surge_onset, build_segments, cum_on_at, stats_12h")
print(f"  output: line, line_bf, tab1, forest_plot")
print(f"  build (unused by this notebook, kept for methods transparency): cached, sofa_* / REBUILD={REBUILD}")
print(f"  prepared inputs: COHORT={COHORT_PARQUET}, CRRT_PROC={CRRT_PROC_PARQUET}, "
      f"CRRT_INPUT={CRRT_INPUT_PARQUET}, EICU_COHORT={EICU_COHORT_PARQUET}")
print(f"  paths: CACHE={CACHE}, FIG={FIGDIR.resolve()}")
