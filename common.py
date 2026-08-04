# FROZEN-OK: 원본 projects/CRRT/source/[최종] 코드 통합_260624.ipynb 셀 0 의 verbatim 추출.
#   유일한 변경: MIMIC/EICU 경로 2줄(환경변수화) + import os 1줄 추가. 그 외 로직 0글자 변경.
#   변경 근거·diff: ../변경-로그.md · ../재조립-설계.md §3-A
"""
common.py — CRRT/TMP 투고본 분석 공용 정의 (import·상수·함수).

이 파일은 노트북 셀이 아니라 모듈이다. 원본 노트북의 셀 0을 그대로 옮겼을 뿐이며
계산 로직은 바뀌지 않았다. `notebooks/analysis.ipynb` 가 이 모듈을 import 해서 쓴다.

데이터 접근: MIMIC4_PATH · EICU_PATH 환경변수로 원자료 위치를 지정한다(README.md 참조).
헌법 §8: 이 파일에는 실데이터 값도, 사내 인프라 경로도 없다.
"""

# =====================================================================
# [셀 0] 헤더: 모든 import · 상수 · 공용 함수 (1회 정의, 실행 로직 없음)
#   원칙: 이후 모든 셀은 여기 정의만 참조. 재import·재정의·재호출 금지.
#   구성:
#     (1) import
#     (2) 전역 상수 — 분석 공용 + eICU + 빌더 전용 itemid 매핑
#     (3) 빌드 헬퍼 — chunked_load, cached
#     (4) SOFA 구성요소 스코어 함수
#     (5) 통계 공용 — pool_hr, bayes_factor_bic, evalue, bh_fdr
#     (6) 효과크기 — risk_diff, hr_per_window
#     (7) 결측 대치 — mice_impute (결과변수 + Nelson-Aalen 누적위험 포함)
#     (8) TVC 빌더 — build, build_ph, build_grid_mimic, build_grid_eicu
#     (9) surge / 가동구간 / landmark — surge_onset, build_segments,
#         cum_on_at, stats_12h
#    (10) 적합 래퍼 — fit_surge_cox (MICE 풀링), fit_grid_cox
#    (11) 출력/표/그림 — line, line_bf, tab1, forest_plot
# =====================================================================

# ---------------------------------------------------------------------
# (1) import
# ---------------------------------------------------------------------
import os
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import (t as tdist, norm, mannwhitneyu, chi2_contingency,
                         fisher_exact, linregress)

from lifelines import CoxTimeVaryingFitter, CoxPHFitter, NelsonAalenFitter
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings("ignore")
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------
# (2) 전역 상수
# ---------------------------------------------------------------------
# ---- 분석 공용 ----
SEED   = 42          # 전역 시드 (MICE·재현성)
M_IMP  = 5           # 다중대치 세트 수
STEP_H = 6           # TVC 격자 분할 간격(시간)
FU_H   = 28 * 24     # 1차 종점: 28일(시간)
FU7    = 7 * 24      # 보조 종점: 7일(시간)
LM_H   = 12          # landmark 시점(시간) = 초기 창 길이
PENALIZER = 0.1      # Cox L2 penalty (sparse event 수렴 안정). 민감도에서 변동.

# surge 정의 격자
MAIN      = ("median", 100, 12)            # 주 정의: baseline / rise(mmHg) / window(h)
BASELINES = ["median", "first"]
RISES     = [50, 75, 100, 125, 150]
WINDOWS   = [6, 12, 18, 24, 48]

# 경로 (로컬 환경 기준)
MIMIC      = Path(os.environ.get("MIMIC4_PATH", "./data/mimic4"))  # FROZEN-OK 재조립: 사내 UNC 경로를 환경변수로 치환(원본 값은 재조립-설계.md §3-A 참조)
MIMIC_ICU  = MIMIC / "icu"
MIMIC_HOSP = MIMIC / "hosp"
EICU       = Path(os.environ.get("EICU_PATH", "./data/eicu"))  # FROZEN-OK 재조립: 사내 UNC 경로를 환경변수로 치환(원본 값은 재조립-설계.md §3-A 참조)
CACHE      = Path("cache_final"); CACHE.mkdir(exist_ok=True)
FIGDIR     = Path("figs");        FIGDIR.mkdir(exist_ok=True)

# 산출물 parquet
COHORT_PARQUET = "cohort_final_v2.parquet"
REBUILD = False      # True일 때만 빌드부(셀 1·2) 실행. 평소 parquet 로드.

# ---- 공변량 네임스페이스 ----
# MIMIC 분석 공변량 (cohort_final_v2 컬럼명)
CONT_M = ["anchor_age", "weight_kg", "sofa_total", "map_value", "platelet",
          "hemoglobin", "lactate", "inr", "aptt", "bilirubin", "blood_flow"]
BIN_M  = ["male", "vaso_use", "mech_vent", "v3_systemic_hep", "v3_prophylaxis"]

# M5 위계 보정 블록 (셀 8 surge 위계 + 셀 10 PH 검정 공용)
DEMO_C = ["anchor_age", "weight_kg"]; DEMO_B = ["male"]
SEV_C  = ["sofa_total", "map_value"]; SEV_B  = ["mech_vent", "vaso_use"]
LAB_C  = ["platelet", "hemoglobin", "lactate", "inr", "aptt", "bilirubin"]
TX_C   = ["blood_flow"];              TX_B   = ["v3_systemic_hep", "v3_prophylaxis"]

# eICU 분석 공변량 (대리 변수)
CONT_E = ["age_num", "weight_kg", "aps", "map_value", "platelet",
          "hemoglobin", "lactate", "inr", "aptt", "bilirubin"]
BIN_E  = ["male", "vaso_use", "mech_vent", "v3_systemic_hep"]

# landmark 7통계량
STAT_VARS = ["tmp_min", "tmp_mean", "tmp_median", "tmp_max",
             "tmp_range", "tmp_time2max", "tmp_slope"]

# ---- 빌더 전용 상수 (셀 1 코호트 빌더에서만 사용) ----
TMP_ID = 229247                                   # measured TMP itemid
TMP_CLIP = 1000
BASE_WIN_H = 3; PEAK_START_H = 3
PEAK_WINDOWS = [6, 12, 18, 24, 48]; DROP_FOLLOW_H = 6
COV_PRE_H, COV_POST_H = 24, 3                      # 공변량 창: t0-24h ~ t0+3h
MIN_CRRT_RANGE_H = 3                               # 전체 CRRT 범위 최소 기준
CRRT_PROC = [225802, 225803, 225809, 225955]       # CRRT, CVVHD, CVVHDF, SCUF
CRRT_IN   = [227525, 227536, 227526, 227528, 227529]  # Ca/KCl/Citrate/ACD-A (1층위 가동구간 보완)

PRESS = {224150: "filter_p", 224152: "return_p", 224151: "effluent_p"}
PRESS_BOUNDS = {"filter_p": (-50, 500), "return_p": (-50, 500),
                "effluent_p": (-300, 300)}

WEIGHT_KG = [226512, 224639]; WEIGHT_LB = [226531]
WEIGHT_PRIORITY = {226512: 1, 224639: 2, 226531: 3}
LAB_SPEC = {
    "platelet":   {"lab": [51265],        "chart": [227457], "lo": 1,   "hi": 2000},
    "hemoglobin": {"lab": [51222, 50811], "chart": [220228], "lo": 2,   "hi": 25},
    "lactate":    {"lab": [50813, 53154], "chart": [225668], "lo": 0.1, "hi": 30},
    "inr":        {"lab": [51237, 51675], "chart": [227467], "lo": 0.5, "hi": 20},
    "aptt":       {"lab": [51275],        "chart": [227466], "lo": 5,   "hi": 200},
    "bilirubin":  {"lab": [50885],        "chart": [225690], "lo": 0.0, "hi": 60},
    "wbc":        {"lab": [51301, 51300], "chart": [],       "lo": 0.1, "hi": 500},
    "creatinine": {"lab": [50912, 52546], "chart": [220615], "lo": 0.1, "hi": 30},
    "bun":        {"lab": [51006],        "chart": [],       "lo": 1,   "hi": 300},
    "bicarbonate":{"lab": [50882],        "chart": [],       "lo": 1,   "hi": 60},
    "pao2":       {"lab": [50821],        "chart": [220224], "lo": 20,  "hi": 700}}
MAP_ITEMS = [220052, 220181, 225312]
GCS_SPEC  = {"gcs_eye": (220739, 1, 4), "gcs_verbal": (223900, 1, 5),
             "gcs_motor": (223901, 1, 6)}
FIO2_ITEM = [223835]; BF_ITEM = [224144]; MV_ITEM = [225792]
VASO_SPEC = {"norepi": {"ids": [221906], "max": 5.0},
             "epi":    {"ids": [221289], "max": 5.0},
             "dopa":   {"ids": [221662], "max": 50.0},
             "dobu":   {"ids": [221653], "max": 50.0},
             "phenyl": {"ids": [221749, 229630, 229632], "max": 10.0},
             "vaso":   {"ids": [222315], "max": 6.0}}
VASO_IDS = [i for s in VASO_SPEC.values() for i in s["ids"]]
HEPARIN_SYSTEMIC = [225152, 229597]; HEPARIN_PROPH = [225975]
HEPARIN_CRRT = [230044]
OTHER_ANTICOAG = [225147, 225148, 229781, 221892, 225906, 225908]
CITRATE_INPUT = [225164, 227526, 227528, 227529]; CITRATE_CHART = [228004]
CALCIUM_CRRT = [227525]; HEPARIN_CHART_224145 = [224145]
ALL_CHART_COV = sorted(set(
    WEIGHT_KG + WEIGHT_LB + [i for s in LAB_SPEC.values() for i in s["chart"]] +
    MAP_ITEMS + [v[0] for v in GCS_SPEC.values()] + FIO2_ITEM + BF_ITEM + [220224]))
ALL_LAB_IDS = sorted(set(i for s in LAB_SPEC.values() for i in s["lab"]))
ALL_INPUT_ANTICOAG = sorted(set(
    HEPARIN_SYSTEMIC + HEPARIN_PROPH + HEPARIN_CRRT + OTHER_ANTICOAG +
    CITRATE_INPUT + CALCIUM_CRRT))
ALL_CHART_ANTICOAG = sorted(set(CITRATE_CHART + HEPARIN_CHART_224145))

# 감염 의심(Sepsis-3 근사)용 항생제 키워드 (셀 4 sepsis3)
ABX_KW = ["amikacin", "amoxicillin", "clavulanate", "ampicillin", "sulbactam",
    "unasyn", "augmentin", "azithromycin", "aztreonam", "bactrim", "cefazolin",
    "cefepime", "cefotetan", "cefotaxime", "ceftazidime", "ceftriaxone",
    "cefuroxime", "cephalexin", "ciprofloxacin", "clarithromycin", "clindamycin",
    "daptomycin", "doxycycline", "erythromycin", "gentamicin", "levofloxacin",
    "linezolid", "meropenem", "metronidazole", "minocycline", "moxifloxacin",
    "nafcillin", "nitrofurantoin", "norfloxacin", "ofloxacin", "penicillin",
    "piperacillin", "tazobactam", "zosyn", "imipenem", "rifampin", "tetracycline",
    "tobramycin", "trimethoprim", "sulfamethoxazole", "vancomycin", "ceftaroline",
    "colistin", "polymyxin", "tigecycline", "ertapenem", "cefoxitin", "cefdinir",
    "cefpodoxime", "cubicin", "invanz", "primaxin", "merrem", "zyvox", "cipro",
    "levaquin", "flagyl", "rocephin", "maxipime", "fortaz", "claforan", "septra",
    "macrobid"]
# eICU CRRT 식별 키워드 (셀 6)
EICU_CRRT_KW = ["c v v h", "cvvh", "c a v h", "cavh", "sled",
                "ultrafiltration", "hemofiltration"]
EICU_VASO_KW = ["norepinephrine", "epinephrine", "dopamine", "dobutamine",
                "phenylephrine", "vasopressin", "levophed", "neo-synephrine"]
EICU_LAB_MAP = {"platelet": ["platelets"], "lactate": ["lactate"],
                "inr": ["pt - inr", "inr"], "aptt": ["ptt"],
                "hemoglobin": ["hgb", "hemoglobin"]}

# 베이즈 인자 해석 구간 (Kass & Raftery 1995, BF01 기준)
BF_LABELS = [(1, 3, "weak/inconclusive"), (3, 20, "positive"),
             (20, 150, "strong"), (150, np.inf, "very strong")]

# ---------------------------------------------------------------------
# (3) 빌드 헬퍼 (셀 1 코호트 빌더 전용)
# ---------------------------------------------------------------------
def chunked_load(path, usecols, itemid_filter, subject_filter=None,
                 id_col="stay_id", chunksize=2_000_000):
    """대용량 csv.gz를 itemid·id로 청크 필터링해 로드."""
    items = set(int(x) for x in itemid_filter)
    filt = set(int(x) for x in subject_filter) if subject_filter is not None else None
    use = list(usecols)
    if "itemid" not in use: use.append("itemid")
    if filt is not None and id_col not in use: use.append(id_col)
    chunks = []
    for ch in pd.read_csv(path, chunksize=chunksize, low_memory=False,
                          usecols=use, compression="gzip"):
        ch = ch[ch["itemid"].isin(items)]
        if filt is not None and len(ch): ch = ch[ch[id_col].isin(filt)]
        if len(ch): chunks.append(ch)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=use)

def cached(name, loader):
    """parquet 캐시. 있으면 로드, 없으면 loader() 실행 후 저장."""
    p = CACHE / name
    if p.exists():
        df = pd.read_parquet(p); print(f"  {name} cache: {len(df):,}"); return df
    print(f"  loading {name}..."); t = time.time(); df = loader()
    df.to_parquet(p); print(f"  {name}: {len(df):,} ({time.time()-t:.0f}s)"); return df

# ---------------------------------------------------------------------
# (4) SOFA 구성요소 스코어 (빌더 전용, 비신장 5개 도메인)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# (5) 통계 공용
# ---------------------------------------------------------------------
def pool_hr(est, var, scale=1.0):
    """Rubin's rules 풀링. est=계수 리스트, var=분산 리스트.
       scale=계수 배율(가동시간 per-24h이면 24). 반환 (HR,lo,hi,p,Q,se)."""
    mn = len(est); Q = np.mean(est); U = np.mean(var); B = np.var(est, ddof=1)
    T = U + (1 + 1/mn) * B; se = np.sqrt(T)
    dof = (mn-1) * (1 + U/((1+1/mn)*B))**2 if B > 0 else np.inf
    tc = tdist.ppf(0.975, dof) if dof < 1000 else norm.ppf(0.975)
    p = 2*(1 - tdist.cdf(abs(Q/se), dof)) if dof < 1000 else 2*(1 - norm.cdf(abs(Q/se)))
    return (np.exp(Q*scale), np.exp((Q-tc*se)*scale), np.exp((Q+tc*se)*scale), p, Q, se)

def _cox_partial_ll(cp, covars, event_col="event"):
    """penalty 없는 부분우도 적합 → (logPL, k=계수수, d=event수).
       BF용. penalizer=0으로 적합해 BIC 근사의 우도항을 얻는다."""
    cols = ["id", "start", "stop", event_col] + covars
    m = CoxTimeVaryingFitter(penalizer=0.0)
    m.fit(cp[cols], id_col="id", start_col="start", stop_col="stop",
          event_col=event_col, show_progress=False)
    return m.log_likelihood_, len(covars), int(cp[event_col].sum())

def bayes_factor_bic(cp_list, exposure, adjust_covars, event_col="event"):
    """BIC 근사 베이즈 인자 (Volinsky & Raftery 2000: event 수 페널티).
       full = exposure + adjust_covars, reduced = adjust_covars.
       BIC = -2*logPL + k*ln(d), d=uncensored event 수.
       BF01 = exp((BIC_full - BIC_reduced)/2)  → H0(노출효과 없음) 지지 배수.
       MICE 다중 세트(cp_list)는 logPL·d를 평균해 통합(Kass-Wasserman 근사).
       반환 dict: BF01, BF10, dBIC, logPL_full/reduced, d, label."""
    LLf, LLr, ds = [], [], []
    for cp in cp_list:
        # 적합 가능한 변수만 (분산 0 제외)
        full_cov = [exposure] + [c for c in adjust_covars if cp[c].nunique() > 1]
        red_cov  = [c for c in adjust_covars if cp[c].nunique() > 1]
        llf, kf, d = _cox_partial_ll(cp, full_cov, event_col)
        llr, kr, _ = _cox_partial_ll(cp, red_cov,  event_col)
        LLf.append(llf); LLr.append(llr); ds.append(d)
    logPL_full = np.mean(LLf); logPL_red = np.mean(LLr); d = np.mean(ds)
    # exposure 1개 추가에 대한 ΔBIC (k_full - k_red = 1)
    dBIC = (-2*logPL_full + 1*np.log(d)) - (-2*logPL_red + 0*np.log(d))
    BF01 = np.exp(dBIC / 2)            # H0/H1 (노출 무효 지지)
    BF10 = 1.0 / BF01
    label = next(l for lo, hi, l in BF_LABELS
                 if (lo <= (BF01 if BF01 >= 1 else BF10) < hi))
    favors = "H0 (no effect)" if BF01 > 1 else "H1 (effect)"
    return {"BF01": BF01, "BF10": BF10, "dBIC": dBIC, "d": d,
            "logPL_full": logPL_full, "logPL_reduced": logPL_red,
            "favors": favors, "strength": label}

def evalue(hr):
    """E-value (VanderWeele & Ding 2017). HR<1이면 역수 취함."""
    hr = hr if hr >= 1 else 1/hr
    return hr + np.sqrt(hr*(hr-1))

def bh_fdr(pv):
    """Benjamini-Hochberg FDR q값."""
    pv = np.asarray(pv, float); m = len(pv)
    order = np.argsort(pv); ranks = np.empty(m, int); ranks[order] = np.arange(1, m+1)
    q_raw = pv * m / ranks
    q_sorted = q_raw[order]; q_mono = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(m); q[order] = np.clip(q_mono, 0, 1); return q

# ---------------------------------------------------------------------
# (6) 효과크기
# ---------------------------------------------------------------------
def hr_per_window(coef_per_h, hours):
    """시간당 로그계수 → 임의 구간(hours) HR. 1층위 per-24h/per-7d 환산."""
    return np.exp(coef_per_h * hours)

def risk_diff(df, group_col, event_col, group1=1, group0=0):
    """관측 절대 위험차(%). 보정 위험차 아님(기술용). 반환 (rd, r1, r0)."""
    r1 = df[df[group_col] == group1][event_col].mean()
    r0 = df[df[group_col] == group0][event_col].mean()
    return (r1 - r0) * 100, r1 * 100, r0 * 100

# ---------------------------------------------------------------------
# (7) 결측 대치 (MICE + 결과변수·Nelson-Aalen 누적위험 포함)
# ---------------------------------------------------------------------
def mice_impute(df, cont_cols, event_col, time_col, m=M_IMP, seed=SEED):
    """White & Royston(2009) 표준: 대치 모형에 결과 정보 포함.
       예측행렬 = [연속공변량, event indicator, Nelson-Aalen 누적위험 H(t)].
       반환: 길이 m의 리스트, 각 원소는 cont_cols만 채운 DataFrame
             (event/H는 대치 보조에만 쓰고 결과 프레임에는 남기지 않음)."""
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
        out.append(filled[cont_cols])   # 보조열 제거, 공변량만 반환
    return out

# ---------------------------------------------------------------------
# (8) TVC 빌더
# ---------------------------------------------------------------------
def build(d, Xc, onset_map, cont, binc, end_col="end28_h", ev_col="event28"):
    """흡수형 2-row TVC (2층위 surge). 각 환자를 surge 전(0)·후(1) 분할.
       Xc=대치된 연속공변량 DF(인덱스 d와 정렬). onset_map={stay_id: surge 시각h}."""
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
    """PH 검정용: surge 후 구간을 STEP_H로 쪼개 surge×log(t) 부여."""
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
    """1층위 MIMIC: 추적을 STEP_H 격자로 분할, 각 구간에 누적 가동시간 부여."""
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
    """1층위 eICU: I/O proxy 가동시간(term_h) 기반 격자 TVC."""
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

# ---------------------------------------------------------------------
# (9) surge / 가동구간 / landmark
# ---------------------------------------------------------------------
def surge_onset(tmp, base_mode, rise, win):
    """초기 TMP 급상승 발생 시각. tmp=(stay_id,valuenum,h).
       baseline=base_mode(median/first, t0+3h 이내), 판정=3h이후~win 첫 초과.
       반환 {stay_id: 첫 초과 시각(h)}."""
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
    """1층위 가동구간: 환자별 union(merge_gap) → {stay_id:[(s_h,e_h)..]}.
       raw=(stay_id,starttime,endtime). t0map=stay_id→t0."""
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
    """t 시점까지 누적 가동시간."""
    cum = 0.0
    for s, e in segs_list:
        if e <= t: cum += (e - s)
        elif s < t: cum += (t - s)
    return cum

def stats_12h(g):
    """landmark 7통계량 (0~LM_H 창). g=한 환자 measured TMP(h,valuenum).
       주의: slope/time2max의 base는 첫 측정값(v[0]) — surge MAIN(median)과
       정의가 다르며, 이 통계량은 surge와 독립적인 기술 요약이다."""
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

# ---------------------------------------------------------------------
# (10) 적합 래퍼
# ---------------------------------------------------------------------
def fit_surge_cox(d, XC_list, onset_map, cont, binc, end_col="end28_h",
                  ev_col="event28", penalizer=PENALIZER, full_return=False):
    """2층위 surge Cox + Rubin 풀링. XC_list=대치세트 리스트(d와 정렬).
       full_return=True면 (surge_res, 전계수 풀링표, cp_list) 반환.
       cp_list는 BF 계산(bayes_factor_bic)에 재사용."""
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
    """1층위 누적 가동시간 Cox(격자 TVC) + Rubin 풀링.
       builder='mimic' → build_grid_mimic(segs 사용),
       builder='eicu'  → build_grid_eicu(segs 무시, d.term_h 사용).
       mask=XC_list가 상위집합일 때 d에 해당하는 행 인덱스(부분집합 정렬).
       반환 (HR,lo,hi,p,N,ev) 또는 return_cp 시 cp_list 추가."""
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

# ---------------------------------------------------------------------
# (11) 출력 / 표 / 그림
# ---------------------------------------------------------------------
def line(label, r, extra=""):
    """HR 결과 한 줄 출력."""
    if r is None: print(f"  {label:<34} na(분리불가)"); return
    hr, lo, hi, p = r[0], r[1], r[2], r[3]
    print(f"  {label:<34} HR={hr:.3f} [{lo:.3f},{hi:.3f}] "
          f"p={p:.3f}{'*' if p < 0.05 else ''} {extra}")

def line_bf(label, bf):
    """베이즈 인자 한 줄 출력."""
    print(f"  {label:<28} BF01={bf['BF01']:.2f} (BF10={bf['BF10']:.2f}) "
          f"ΔBIC={bf['dBIC']:.2f} d={bf['d']:.0f} → {bf['favors']}, {bf['strength']}")

def tab1(df, group_col, cont_vars, bin_vars, cat_vars=None):
    """Table 1. 연속=median+MWU, 이진=%+chi2, 범주=분포+chi2."""
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
    """HR forest plot 공용."""
    n = len(labels); figsize = figsize or (7, 0.5*n + 1.2)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)[::-1]
    ax.errorbar(hrs, y, xerr=[np.array(hrs)-np.array(los), np.array(his)-np.array(hrs)],
                fmt="o", color="#2c3e50", ecolor="#7f8c8d", capsize=3, lw=1, ms=5)
    ax.axvline(ref, ls="--", color="#c0392b", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hazard ratio (95% CI)"); ax.set_title(title, fontsize=11, loc="left")
    plt.tight_layout(); plt.savefig(FIGDIR/fname, dpi=150, bbox_inches="tight"); plt.show()

# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------
print("[셀 0] 헤더 로드 완료")
print(f"  상수: SEED={SEED}, M_IMP={M_IMP}, STEP_H={STEP_H}, FU_H={FU_H}, "
      f"LM_H={LM_H}, PENALIZER={PENALIZER}, TMP_ID={TMP_ID}")
print(f"  MAIN={MAIN}, 격자 {len(BASELINES)}×{len(RISES)}×{len(WINDOWS)}="
      f"{len(BASELINES)*len(RISES)*len(WINDOWS)}조합")
print(f"  공변량: MIMIC CONT_M {len(CONT_M)} / BIN_M {len(BIN_M)}, "
      f"eICU CONT_E {len(CONT_E)} / BIN_E {len(BIN_E)}")
print(f"  통계함수: pool_hr, bayes_factor_bic, evalue, bh_fdr, "
      f"hr_per_window, risk_diff")
print(f"  대치: mice_impute (event + Nelson-Aalen 누적위험 포함)")
print(f"  TVC: build, build_ph, build_grid_mimic, build_grid_eicu")
print(f"  적합: fit_surge_cox, fit_grid_cox (penalizer 인자화)")
print(f"  surge/가동: surge_onset, build_segments, cum_on_at, stats_12h")
print(f"  출력/표/그림: line, line_bf, tab1, forest_plot")
print(f"  빌드: chunked_load, cached, sofa_* / REBUILD={REBUILD}")
print(f"  경로: COHORT={COHORT_PARQUET}, CACHE={CACHE}, FIG={FIGDIR.resolve()}")