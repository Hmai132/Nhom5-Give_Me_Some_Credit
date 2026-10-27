# -*- coding: utf-8 -*-
"""
Dashboard hỗ trợ ra quyết định tín dụng - Nhóm 5
Môn Phân tích dữ liệu nâng cao | GV: TS. Nguyễn Thôn Dã

Cách chạy:
    pip install streamlit plotly scikit-learn imbalanced-learn xgboost shap joblib
    streamlit run app.py
(Đặt file credit_model.joblib cùng thư mục với app.py;
 file .streamlit/config.toml đi kèm giúp đồng bộ theme. Yêu cầu Streamlit >= 1.40)
"""
import sys
import html
import inspect
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer

set_config(transform_output="pandas")

# =====================================================================
# 1. KHAI BÁO LẠI CÁC TRANSFORMER (bắt buộc để joblib.load không lỗi)
#    Các class phải trùng TÊN với lúc lưu trong notebook.
# =====================================================================
LATE_COLS = ["NumberOfTime30-59DaysPastDueNotWorse",
             "NumberOfTime60-89DaysPastDueNotWorse",
             "NumberOfTimes90DaysLate"]
MISSING_COLS = ["MonthlyIncome", "NumberOfDependents"]
SKEW_COLS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
SPECIAL_CODES = [96, 98]
RANDOM_STATE = 42


class SpecialCodeCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, late_cols=None, codes=None):
        self.late_cols = late_cols
        self.codes = codes

    def fit(self, X, y=None):
        self.late_cols_ = self.late_cols if self.late_cols is not None else LATE_COLS
        self.codes_ = self.codes if self.codes is not None else SPECIAL_CODES
        return self

    def transform(self, X):
        if hasattr(X, "columns") and not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = X.copy()
        for c in self.late_cols_:
            X.loc[X[c].isin(self.codes_), c] = np.nan
        X.loc[X["age"] <= 0, "age"] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray(self.late_cols_ if hasattr(self, "late_cols_") else LATE_COLS, dtype=object)


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.cols_ = self.cols if self.cols is not None else MISSING_COLS
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[f"{c}_missing"] = X[c].isna().astype(int)
        return X


class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalPastDue"] = X[LATE_COLS].sum(axis=1)
        X["HasPastDue"] = (X["TotalPastDue"] > 0).astype(int)
        X["IncomePerPerson"] = X["MonthlyIncome"] / (X["NumberOfDependents"] + 1)
        if "MonthlyIncome_missing" in X.columns:
            X["MonthlyDebt"] = np.where(
                X["MonthlyIncome_missing"] == 1,
                X["DebtRatio"],
                X["DebtRatio"] * X["MonthlyIncome"],
            )
        else:
            X["MonthlyDebt"] = X["DebtRatio"] * X["MonthlyIncome"]
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, lower=0.01, upper=0.99):
        self.cols = cols
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.cols_ = [c for c in (self.cols if self.cols is not None else SKEW_COLS)
                      if c in X.columns]
        self.low_ = X[self.cols_].quantile(self.lower)
        self.high_ = X[self.cols_].quantile(self.upper)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_] = X[self.cols_].clip(lower=self.low_, upper=self.high_, axis=1)
        return X


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.cols_ = [c for c in (self.cols if self.cols is not None else SKEW_COLS)
                      if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_] = np.log1p(X[self.cols_].clip(lower=0))
        return X


class ScaledKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, max_ref=None, random_state=RANDOM_STATE):
        self.n_neighbors = n_neighbors
        self.max_ref = max_ref
        self.random_state = random_state

    def fit(self, X, y=None):
        self.center_ = X.median()
        iqr = X.quantile(0.75) - X.quantile(0.25)
        self.scale_ = iqr.replace(0, 1).fillna(1)
        Z = (X - self.center_) / self.scale_
        if self.max_ref is not None and len(Z) > self.max_ref:
            Z = Z.sample(n=self.max_ref, random_state=self.random_state)
        self.knn_ = KNNImputer(n_neighbors=self.n_neighbors).fit(Z)
        return self

    def transform(self, X):
        Z = (X - self.center_) / self.scale_
        Z_imp = pd.DataFrame(np.asarray(self.knn_.transform(Z)),
                             columns=X.columns, index=X.index)
        return Z_imp * self.scale_ + self.center_


# Đăng ký class vào __main__ để joblib luôn tìm được (an toàn với mọi môi trường)
_main = sys.modules["__main__"]
for _cls in (SpecialCodeCleaner, MissingIndicatorAdder, FeatureCreator,
             Winsorizer, Log1pTransformer, ScaledKNNImputer):
    setattr(_main, _cls.__name__, _cls)

# =====================================================================
# TỰ ĐỘNG TƯƠNG THÍCH MODEL LINUX (COLAB) VÀ WINDOWS CHO XGBOOST
# Tránh lỗi "input stream corrupted" khi tải file joblib từ Colab về Windows
# =====================================================================
import os
import xgboost as _xgb

_orig_booster_setstate = _xgb.Booster.__setstate__

def _safe_booster_setstate(self, state):
    h = state.get("handle") if isinstance(state, dict) else None
    if isinstance(h, (bytes, bytearray)):
        pos_s = h.find(b"rng_state")
        if pos_s != -1:
            pos_next = h.find(b"L\x00\x00\x00\x00\x00\x00\x00", pos_s + 10)
            if pos_next != -1:
                if os.name == "nt":
                    if not hasattr(_xgb.Booster, "_msvc_rng_field"):
                        _dummy = _xgb.XGBClassifier().fit(np.zeros((2, 2)), np.array([0, 1]))
                        _dh = bytes(_dummy.get_booster().__getstate__()["handle"])
                        _dp = _dh.find(b"rng_state")
                        _dn = _dh.find(b"L\x00\x00\x00\x00\x00\x00\x00", _dp + 10)
                        _xgb.Booster._msvc_rng_field = _dh[_dp:_dn]
                    state["handle"] = bytearray(h[:pos_s] + _xgb.Booster._msvc_rng_field + h[pos_next:])
    _orig_booster_setstate(self, state)

_xgb.Booster.__setstate__ = _safe_booster_setstate


# =====================================================================
# 2. NẠP MÔ HÌNH VÀ CÁC HÀM DỰ BÁO
# =====================================================================
MODEL_PATHS = ["credit_model.joblib", "credit_model (1).joblib"]

TIERS = {
    "Thấp":       {"color": "#16a34a", "bg": "#dcfce7", "action": "Đề xuất phê duyệt"},
    "Trung bình": {"color": "#d97706", "bg": "#fef3c7", "action": "Chuyển thẩm định thủ công"},
    "Cao":        {"color": "#dc2626", "bg": "#fee2e2",
                   "action": "Đề xuất từ chối / yêu cầu tài sản bảo đảm"},
}

FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Tỷ lệ sử dụng hạn mức tín chấp",
    "age": "Tuổi",
    "NumberOfTime30-59DaysPastDueNotWorse": "Số lần trễ hạn 30–59 ngày",
    "DebtRatio": "Tỷ lệ nợ / thu nhập",
    "MonthlyIncome": "Thu nhập hằng tháng",
    "NumberOfOpenCreditLinesAndLoans": "Số khoản vay / hạn mức đang mở",
    "NumberOfTimes90DaysLate": "Số lần trễ hạn ≥ 90 ngày",
    "NumberRealEstateLoansOrLines": "Số khoản vay bất động sản",
    "NumberOfTime60-89DaysPastDueNotWorse": "Số lần trễ hạn 60–89 ngày",
    "NumberOfDependents": "Số người phụ thuộc",
}


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_artifact():
    for p in MODEL_PATHS:
        try:
            art = joblib.load(p)
            if isinstance(art, dict) and "pipeline" in art and "feature_cols" in art:
                cols = np.asarray(art["feature_cols"], dtype=object)
                for _, step in getattr(art["pipeline"], "steps", []):
                    if not hasattr(step, "feature_names_in_"):
                        try:
                            step.feature_names_in_ = cols
                        except Exception:
                            pass
            return art, p
        except FileNotFoundError:
            continue
    return None, None


def to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(art, p_raw):
    if art["cal_method"] == "platt":
        return art["calibrator"].predict_proba(to_logit(p_raw))[:, 1]
    return np.clip(art["calibrator"].predict(np.asarray(p_raw).reshape(-1, 1)), 0, 1)


def assign_tier(art, p):
    if p < art["low_thr"]:
        return "Thấp"
    if p < art["high_thr"]:
        return "Trung bình"
    return "Cao"


def split_pipeline(pipe):
    """Tách phần tiền xử lý (bỏ SMOTE) và mô hình cuối để tính SHAP."""
    steps = [s for n, s in pipe.steps[:-1] if n != "smote"]

    def transform(X):
        for s in steps:
            X = s.transform(X)
        return X
    return transform, pipe.steps[-1][1]


def predict_frame(art, df_raw):
    """Trả về DataFrame gồm xác suất gốc, xác suất hiệu chỉnh, tầng và hành động."""
    X = df_raw[art["feature_cols"]].astype(float)
    p_raw = art["pipeline"].predict_proba(X)[:, 1]
    p_cal = calibrate(art, p_raw)
    out = df_raw.copy()
    out["Xác suất vỡ nợ"] = p_cal
    out["Tầng rủi ro"] = [assign_tier(art, p) for p in p_cal]
    out["Hành động đề xuất"] = out["Tầng rủi ro"].map(lambda t: TIERS[t]["action"])
    return out, p_raw

# =====================================================================
# 3. GIAO DIỆN — phong cách "Ocean Private Banking"
#    Navy sâu + xanh biển, điểm nhấn vàng champagne, tiêu đề serif.
# =====================================================================
st.set_page_config(page_title="Hệ thống hỗ trợ quyết định tín dụng",
                   page_icon=":material/account_balance:", layout="wide",
                   initial_sidebar_state="expanded")

PAL = dict(navy="#06263D", ocean="#0B5C8E", teal="#1B9AAA", gold="#C9A24B",
           mist="#F3F7FA", line="#D9E4EC", ink="#0B1F33", muted="#5B7083")
FONT_STACK = "Segoe UI, Arial, sans-serif"


def _svg(inner, size=18):
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{inner}</svg>')


# Mỗi tầng rủi ro có một biểu tượng riêng để không chỉ phân biệt bằng màu.
ICON_LOW = _svg('<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.6 2.6L16 9.5"/>')
ICON_MID = _svg('<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17.2v.1"/>')
ICON_HIGH = _svg('<path d="M8.5 3h7L21 8.5v7L15.5 21h-7L3 15.5v-7z"/><path d="M9 9l6 6M15 9l-6 6"/>')

TIERS["Thấp"].update(color="#1F8A70", bg="#E3F3EE", text="#176B57", icon=ICON_LOW)
TIERS["Trung bình"].update(color="#C98B2B", bg="#FBF1DC", text="#7A4E08", icon=ICON_MID)
TIERS["Cao"].update(color="#A8323E", bg="#F8E4E6", text="#8C2531", icon=ICON_HIGH)
TIER_ORDER = ["Thấp", "Trung bình", "Cao"]

FEATURE_LABELS.update({
    "TotalPastDue": "Tổng số lần trễ hạn",
    "HasPastDue": "Từng có trễ hạn",
    "IncomePerPerson": "Thu nhập bình quân đầu người",
    "MonthlyDebt": "Nợ phải trả hằng tháng",
    "MonthlyIncome_missing": "Thiếu thông tin thu nhập",
    "NumberOfDependents_missing": "Thiếu thông tin người phụ thuộc",
})

LOGO = ('<svg viewBox="0 0 48 48" width="48" height="48" fill="none" stroke-linecap="round" '
        'aria-hidden="true"><circle cx="24" cy="24" r="22.5" stroke="#C9A24B" stroke-width="1.5"/>'
        '<path d="M9 18c4.5-6 9-6 13.5 0s9 6 13.5 0" stroke="#7FB8D6" stroke-width="2.2"/>'
        '<path d="M9 25c4.5-6 9-6 13.5 0s9 6 13.5 0" stroke="#FFFFFF" stroke-width="2.2"/>'
        '<path d="M9 32c4.5-6 9-6 13.5 0s9 6 13.5 0" stroke="#C9A24B" stroke-width="2.2"/></svg>')
WAVES = ('<svg class="hero-waves" viewBox="0 0 600 160" preserveAspectRatio="none" aria-hidden="true">'
         '<path d="M0 70 C100 30 200 110 300 70 S500 30 600 70"/>'
         '<path d="M0 98 C100 58 200 138 300 98 S500 58 600 98"/>'
         '<path d="M0 126 C100 86 200 166 300 126 S500 86 600 126"/></svg>')

CSS = """
:root { --navy:#06263D; --ocean:#0B5C8E; --teal:#1B9AAA; --gold:#C9A24B; --mist:#F3F7FA;
        --line:#D9E4EC; --ink:#0B1F33; --muted:#5B7083; --serif:'Segoe UI',Arial,sans-serif; }
.stApp { background: var(--mist); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp [data-testid='stMarkdownContainer'] { font-family: 'Segoe UI', Arial, sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Segoe UI', Arial, sans-serif !important; color: var(--navy); font-weight: 700; letter-spacing: 0; }
.stApp h3 { font-size: 1.85rem; margin-top: .4rem; }
.stApp h4 { font-size: 1.4rem; }
.lead { color: var(--muted); font-size: 1rem; margin: -.2rem 0 1rem; max-width: 70ch; }
.hero { position: relative; display: flex; align-items: center; gap: 1.1rem; overflow: hidden;
        padding: 1.5rem 1.9rem; margin-bottom: 1.4rem; border-radius: 10px; color: #fff;
        background: linear-gradient(120deg, #06263D 0%, #0A4468 55%, #0B5C8E 100%);
        border-bottom: 3px solid var(--gold); }
.hero > * { position: relative; z-index: 1; }
.hero-title { font-family: 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: clamp(1.45rem, 2.5vw, 2rem); line-height: 1.2; color: #fff; }
.hero p { margin: .45rem 0 0; color: #C3D9E8; font-size: 1.02rem; }
.hero-date { margin-left: auto; text-align: right; color: #A9C7DB; font-size: .85rem; white-space: nowrap; }
.hero-date b { display: block; font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.15rem; color: #fff; }
.hero .hero-waves { position: absolute; z-index: 0; right: 0; bottom: 0; width: 62%; height: 100%; fill: none;
        stroke: rgba(255,255,255,.10); stroke-width: 1.4; }
section[data-testid="stSidebar"] { background: var(--navy); border-right: 1px solid #0D3A58; }
section[data-testid="stSidebar"] * { color: #D5E4EF; }
.sb-brand { display: flex; align-items: center; gap: .7rem; padding: .3rem 0 1rem; border-bottom: 1px solid rgba(201,162,75,.55); }
.sb-brand b { font-family: var(--serif); font-size: 1.35rem; color: #fff; line-height: 1.1; }
.sb-h { margin: 1.3rem 0 .2rem; font-family: var(--serif); font-size: 1.15rem; font-weight: 700; color: #fff; }
.sb-row { display: flex; justify-content: space-between; gap: 1rem; padding: .55rem 0; font-size: .92rem;
          border-bottom: 1px solid rgba(255,255,255,.09); }
.sb-row span { color: #8FB0C8; }
.sb-row b { color: #fff; font-weight: 600; text-align: right; }
.sb-tier { display: flex; align-items: center; gap: .6rem; padding: .5rem 0; font-size: .92rem; border-bottom: 1px solid rgba(255,255,255,.09); }
.sb-dot { width: 4px; height: 1.5rem; border-radius: 2px; flex: none; }
.sb-tier b { color: #fff; font-weight: 600; min-width: 5.2rem; }
.sb-note { margin-top: 1.4rem; font-size: .82rem; line-height: 1.5; color: #8FB0C8; }
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 3rem; padding: 0 1.1rem; color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--ocean); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--gold); height: 3px; }
[class*="st-key-grp_"] { background: #fff; border: 1px solid var(--line); border-radius: 10px;
        padding: 1.1rem 1.3rem .6rem; box-shadow: 0 1px 2px rgba(6,38,61,.05); }
.grp-title { font-family: var(--serif); font-size: 1.3rem; font-weight: 700; color: var(--navy); line-height: 1.2; }
.grp-sub { font-size: .85rem; color: var(--muted); margin: .15rem 0 .9rem; padding-bottom: .7rem; border-bottom: 1px solid var(--line); }
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .92rem; color: var(--ink); }
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 6px; }
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] { background: var(--ocean); border: 1px solid var(--ocean);
        color: #fff; border-radius: 6px; padding: .6rem 1.7rem; font-weight: 600; box-shadow: 0 2px 8px rgba(11,92,142,.25); }
button[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover { background: var(--navy); border-color: var(--navy); color: #fff; }
button[data-testid="stBaseButton-secondary"] { border: 1px solid var(--ocean); color: var(--ocean); border-radius: 6px; font-weight: 600; background: #fff; }
button[data-testid="stBaseButton-secondary"]:hover { background: var(--ocean); color: #fff; border-color: var(--ocean); }
.stApp button:focus-visible { outline: 2px solid var(--gold) !important; outline-offset: 2px; }
[data-testid="stExpander"] { background: #fff; border: 1px solid var(--line); border-radius: 8px; }
[data-testid="stAlert"] { border-radius: 8px; }
[data-testid="stFileUploaderDropzone"] { background: #fff; border: 1.5px dashed #9DB9CC; border-radius: 10px; }
.report { background: #fff; border: 1px solid var(--line); border-radius: 10px; overflow: hidden;
          box-shadow: 0 8px 28px rgba(6,38,61,.08); animation: surface .45s ease-out; }
@keyframes surface { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.report-head { display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; padding: .7rem 1.4rem;
               background: var(--navy); color: #BFD6E6; font-size: .85rem; border-bottom: 2px solid var(--gold); }
.report-head b { color: #fff; font-weight: 600; }
.report-body { padding: 1.3rem 1.5rem 1.2rem; }
.rp-label { font-size: .92rem; color: var(--muted); }
.rp-prob { font-family: 'Segoe UI', Arial, sans-serif; font-variant-numeric: tabular-nums; font-size: 3.8rem; font-weight: 700; line-height: 1; margin: .2rem 0 .9rem; }
.tier-badge { display: inline-flex; align-items: center; gap: .5rem; padding: .42rem 1rem .42rem .8rem; border-radius: 999px; font-weight: 700; font-size: .98rem; }
.rp-action { margin-top: 1.1rem; padding: .8rem 1rem; border-left: 3px solid; border-radius: 0 6px 6px 0; background: var(--mist); font-weight: 600; color: var(--ink); }
.rp-foot { margin-top: .9rem; font-size: .85rem; color: var(--muted); }
.fx-title { margin-top: 1.1rem; font-size: .92rem; font-weight: 600; color: var(--ink); }
.fx { display: flex; align-items: baseline; gap: .6rem; padding: .45rem 0; font-size: .93rem; border-bottom: 1px solid var(--line); }
.fx:last-child { border-bottom: 0; }
.fx i { font-style: normal; font-size: .7rem; flex: none; }
.fx em { margin-left: auto; font-style: normal; font-size: .82rem; font-weight: 600; white-space: nowrap; }
.panel-title { font-weight: 600; color: var(--ink); margin-bottom: .2rem; }
.kpis { display: grid; grid-template-columns: 1.15fr 1fr 1fr 1fr; gap: .9rem; margin: .4rem 0 1.3rem; }
.kpi { background: #fff; border: 1px solid var(--line); border-top: 3px solid var(--c); border-radius: 8px; padding: .95rem 1.15rem; }
.kpi .k-label { display: flex; align-items: center; gap: .45rem; font-size: .88rem; font-weight: 600; color: var(--c); }
.kpi .k-num { font-family: 'Segoe UI', Arial, sans-serif; font-variant-numeric: tabular-nums; font-size: 2.35rem; font-weight: 700; line-height: 1.05; color: var(--navy); margin-top: .2rem; }
.kpi .k-sub { font-size: .85rem; color: var(--muted); }
.kpi.total { background: var(--navy); border-color: var(--navy); border-top-color: var(--gold); }
.kpi.total .k-label, .kpi.total .k-sub { color: #A9C7DB; }
.kpi.total .k-num { color: #fff; }
.empty { background: #fff; border: 1px dashed #9DB9CC; border-radius: 10px; padding: 1.5rem 1.7rem; color: var(--muted); }
.empty b { color: var(--navy); }
.about { font-size: 1.02rem; line-height: 1.65; max-width: 78ch; }
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: .6rem 0 1.4rem; }
.step { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.1rem; }
.step .n { font-family: 'Segoe UI', Arial, sans-serif; font-size: 2rem; font-weight: 700; color: var(--gold); line-height: 1; }
.step b { display: block; margin: .35rem 0 .25rem; color: var(--navy); }
.step span { font-size: .9rem; color: var(--muted); line-height: 1.5; }
.policy { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: .6rem 0 1.4rem; }
.pol { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--c); border-radius: 8px; padding: 1rem 1.2rem; }
.pol .p-name { display: flex; align-items: center; gap: .5rem; font-weight: 700; color: var(--c); }
.pol .p-thr { font-family: 'Segoe UI', Arial, sans-serif; font-variant-numeric: tabular-nums; font-size: 1.7rem; font-weight: 700; color: var(--navy); margin: .3rem 0; }
.pol .p-act { font-size: .92rem; color: var(--muted); }
.notice { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.3rem; font-size: .95rem; line-height: 1.6; color: var(--ink); }
.footer { margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: .83rem; color: var(--muted); line-height: 1.6; }
@media (max-width: 900px) { .kpis, .steps { grid-template-columns: 1fr 1fr; } .policy { grid-template-columns: 1fr; }
        .hero-date { display: none; } .hero-title { font-size: 1.55rem; } .rp-prob { font-size: 3.2rem; } }
@media (max-width: 560px) { .kpis, .steps { grid-template-columns: 1fr; } .hero { padding: 1rem; gap: .7rem; } .hero-title { font-size: 1.35rem; } .report-body { padding: 1rem; } .fx { align-items: flex-start; } .fx em { white-space: normal; text-align: right; } }
@media (prefers-reduced-motion: reduce) { .report { animation: none; } }
"""


def _flat(s):
    """Gộp HTML/CSS thành một dòng để Markdown không hiểu nhầm thụt lề là khối code."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def html_block(s, target=st):
    target.markdown(_flat(s), unsafe_allow_html=True)


# Streamlit mới dùng width="stretch"; bản cũ dùng use_container_width.
_NEW_API = "width" in inspect.signature(st.plotly_chart).parameters
STRETCH = {"width": "stretch"} if _NEW_API else {"use_container_width": True}


def show_plot(fig):
    st.plotly_chart(fig, config={"displayModeBar": False}, **STRETCH)


def style_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=13, color=PAL["ink"]),
        hoverlabel=dict(font_family=FONT_STACK, bgcolor=PAL["navy"], font_color="#fff"),
        showlegend=False,
    )
    return fig


def fmt_val(v):
    if pd.isna(v):
        return "Không có thông tin"
    v = float(v)
    return f"{int(v):,}" if v.is_integer() else f"{v:,.2f}"



def risk_gauge(p, low, high):
    """Thanh đo rủi ro với nhãn ngưỡng nằm dưới trục, hạn chế chồng lấn."""
    p = float(np.clip(p, 0, 1))
    low, high = float(low), float(high)

    xmax = float(min(1.0, max(high * 2.0, p * 1.15, 0.1)))
    zones = [
        (0, low, "Thấp"),
        (low, high, "Trung bình"),
        (high, 1, "Cao")
    ]

    fig = go.Figure()

    for x0, x1, name in zones:
        x0v, x1v = min(x0, xmax), min(x1, xmax)

        if x1v > x0v:
            fig.add_shape(
                type="rect",
                x0=x0v, x1=x1v,
                y0=0, y1=1,
                line=dict(width=0),
                fillcolor=TIERS[name]["color"],
                layer="below"
            )

            if (x1v - x0v) / xmax >= 0.16:
                fig.add_annotation(
                    x=(x0v + x1v) / 2,
                    y=0.5,
                    text=f"<b>{name}</b>",
                    showarrow=False,
                    font=dict(color="#fff", size=12)
                )

    # Vạch chỉ vị trí xác suất của hồ sơ
    fig.add_shape(
        type="line",
        x0=p, x1=p,
        y0=-0.12, y1=1.05,
        line=dict(color=PAL["navy"], width=2.5)
    )

    fig.add_trace(
        go.Scatter(
            x=[p],
            y=[1.30],
            mode="markers+text",
            text=[f"<b>{p:.1%}</b>"],
            textposition="top center",
            textfont=dict(size=14, color=PAL["navy"]),
            marker=dict(
                symbol="triangle-down",
                size=13,
                color=PAL["navy"]
            ),
            hoverinfo="skip",
            cliponaxis=False
        )
    )

    # Chỉ hiển thị hai mốc ở hai đầu trục
    fig.update_xaxes(
        range=[0, xmax],
        tickvals=[0, xmax],
        ticktext=["0%", f"{xmax:.0%}"],
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        tickfont=dict(color=PAL["muted"], size=10)
    )

    # Hiển thị hai ngưỡng thành một dòng chú thích riêng
    fig.add_annotation(
        x=0.5,
        y=-0.32,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Ngưỡng thấp:</b> {low:.1%}"
            f"&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
            f"<b>Ngưỡng cao:</b> {high:.1%}"
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=11, color=PAL["muted"]),
        align="center"
    )

    fig.update_yaxes(
        range=[-0.1, 1.75],
        visible=False,
        fixedrange=True
    )

    # Áp dụng style trước, sau đó thiết lập lề riêng cho thanh đo
    fig = style_fig(fig, 290)
    fig.update_layout(
        height=290,
        margin=dict(l=18, r=18, t=35, b=105),
        autosize=True
    )

    return fig

def factor_chart(contrib):
    labels = [FEATURE_LABELS.get(i, i) for i in contrib.index]
    colors = ["#A8323E" if v > 0 else PAL["ocean"] for v in contrib.values]
    fig = go.Figure(go.Bar(
        x=contrib.values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.2f}" for v in contrib.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Đóng góp: %{x:+.3f}<extra></extra>"))
    fig.update_xaxes(zeroline=True, zerolinecolor=PAL["navy"], zerolinewidth=1.5, gridcolor=PAL["line"],
                     title_text="Đóng góp vào dự báo (log-odds): dương làm tăng rủi ro, âm làm giảm rủi ro",
                     title_font=dict(size=12, color=PAL["muted"]), fixedrange=True)
    fig.update_yaxes(automargin=True, fixedrange=True, tickfont=dict(size=11))
    fig.update_xaxes(automargin=True)
    fig.update_layout(margin=dict(l=12, r=72, t=16, b=58), bargap=0.28)
    return style_fig(fig, max(300, 54 * len(contrib)))


def explain(art, df_in):
    """Đóng góp của từng biến vào dự báo (mô hình tuyến tính hoặc dạng cây qua SHAP)."""
    pre, model = split_pipeline(art["pipeline"])
    Xt = pre(df_in[art["feature_cols"]].astype(float))
    if hasattr(model, "coef_"):
        vals = np.ravel(model.coef_) * Xt.iloc[0].values
    else:
        import shap
        sv = shap.TreeExplainer(model)(Xt)
        if sv.values.ndim == 3:
            sv = sv[:, :, 1]
        vals = sv.values[0]
    s = pd.Series(np.asarray(vals, dtype=float), index=Xt.columns)
    return s.sort_values(key=np.abs).tail(10)


def build_report_html(case, avg):
    """Phiếu đánh giá dạng HTML độc lập, mở bằng trình duyệt và in ra PDF được."""
    t = TIERS[case["tier"]]
    inputs = "".join(f"<tr><td>{html.escape(FEATURE_LABELS.get(k, k))}</td><td>{html.escape(fmt_val(v))}</td></tr>"
                     for k, v in case["raw"].items())
    contrib = case.get("contrib")
    factors = ""
    if contrib is not None:
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
        factors = "".join(
            f"<tr><td>{html.escape(FEATURE_LABELS.get(k, k))}</td>"
            f"<td>{'Tăng rủi ro' if v > 0 else 'Giảm rủi ro'}</td><td>{v:+.3f}</td></tr>"
            for k, v in top.items())
        factors = ("<h2>Yếu tố tác động mạnh nhất</h2><table><tr><th>Yếu tố</th><th>Chiều tác động</th>"
                   f"<th>Đóng góp (log-odds)</th></tr>{factors}</table>")
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>Phiếu đánh giá {html.escape(case['id'])}</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:'Segoe UI',Arial,sans-serif;color:#0B1F33;max-width:820px;margin:2rem auto;padding:0 1.2rem;line-height:1.5}}
h1,h2{{font-family:'Segoe UI',Arial,sans-serif;color:#06263D;letter-spacing:0}}
h1{{margin:0 0 .35rem;font-size:1.8rem}}
h2{{margin-top:1.5rem;font-size:1.2rem}}
.meta{{color:#5B7083;border-bottom:3px solid #C9A24B;padding-bottom:.8rem}}
.box{{margin-top:1.2rem;padding:1rem 1.3rem;border-left:5px solid {t['color']};background:{t['bg']}}}
.big{{font-size:2.6rem;font-weight:700;color:{t['text']}}}
table{{width:100%;border-collapse:collapse;margin-top:.5rem;table-layout:fixed;font-size:.9rem}}
td,th{{padding:.5rem .6rem;border-bottom:1px solid #D9E4EC;text-align:left;vertical-align:top;overflow-wrap:anywhere}}
th{{background:#F3F7FA}}
tr{{break-inside:avoid;page-break-inside:avoid}}
.note{{margin-top:1.5rem;font-size:.85rem;color:#5B7083}}
@page{{size:A4;margin:16mm}}
@media print{{body{{max-width:none;margin:0;padding:0;font-size:10pt}}h1{{font-size:18pt}}h2{{font-size:12pt;break-after:avoid}}.box{{break-inside:avoid}}table{{font-size:9pt}}}}
</style></head><body>
<h1>Phiếu đánh giá rủi ro tín dụng</h1>
<div class="meta">Mã hồ sơ: {html.escape(case['id'])} &nbsp;|&nbsp; Thời điểm: {html.escape(case['time'])}
 &nbsp;|&nbsp; Mô hình: {html.escape(str(art['model_name']))}</div>
<div class="box"><div>Xác suất vỡ nợ (đã hiệu chỉnh)</div><div class="big">{case['p']:.1%}</div>
<div><b>Rủi ro {html.escape(case['tier'])}</b>. Đề xuất: {html.escape(t['action'])}.</div>
<div>Bình quân danh mục: {avg:.1%}. Xác suất gốc của mô hình: {case['p_raw']:.3f}.</div></div>
{factors}
<h2>Dữ liệu đầu vào</h2><table><tr><th>Chỉ tiêu</th><th>Giá trị</th></tr>{inputs}</table>
<p class="note">Kết quả chỉ mang tính tham khảo. Quyết định cuối cùng thuộc về cán bộ thẩm định tín dụng.</p>
</body></html>"""


def tier_cell_style(v):
    t = TIERS.get(v)
    return f"background-color:{t['bg']};color:{t['text']};font-weight:600" if t else ""


def score_batch(upload):
    df = pd.read_csv(upload)
    df = df.drop(columns=[c for c in ["Unnamed: 0", "SeriousDlqin2yrs"] if c in df.columns])
    missing = [c for c in art["feature_cols"] if c not in df.columns]
    if missing:
        raise ValueError("Thiếu các cột: " + ", ".join(missing))
    res, _ = predict_frame(art, df)
    return res


# ---------- Khung trang ----------
html_block(f"<style>{CSS}</style>")

art, used_path = load_artifact()
if art is None:
    st.error("Không tìm thấy credit_model.joblib. Hãy đặt file cùng thư mục với app.py.")
    st.stop()

MODEL_NAME = html.escape(str(art["model_name"]))
AVG_RATE = float(art.get("base_rate", 0.067))  # tỷ lệ vỡ nợ bình quân của tập huấn luyện

html_block(f"""
<div class="hero">{WAVES}{LOGO}
  <div>
    <div class="hero-title">Hệ thống hỗ trợ ra quyết định tín dụng</div>
    <p>Chấm điểm rủi ro vỡ nợ cho khách hàng cá nhân bằng mô hình {MODEL_NAME}.</p>
  </div>
  <div class="hero-date">Ngày làm việc<b>{datetime.now():%d/%m/%Y}</b></div>
</div>""")

with st.sidebar:
    html_block(f"""
    <div class="sb-brand">{LOGO}<b>Hệ thống<br>thẩm định tín dụng</b></div>
    <div class="sb-h">Thông tin mô hình</div>
    <div class="sb-row"><span>Mô hình chốt</span><b>{MODEL_NAME}</b></div>
    <div class="sb-row"><span>Hiệu chỉnh xác suất</span><b>{html.escape(str(art['cal_method']))}</b></div>
    <div class="sb-row"><span>Chi phí FN:FP</span><b>{html.escape(str(art.get('cost_ratio', '10:1')))}</b></div>
    <div class="sb-h">Ngưỡng phân tầng</div>
    <div class="sb-tier"><i class="sb-dot" style="background:{TIERS['Thấp']['color']}"></i><b>Thấp</b>dưới {art['low_thr']:.1%}</div>
    <div class="sb-tier"><i class="sb-dot" style="background:{TIERS['Trung bình']['color']}"></i><b>Trung bình</b>{art['low_thr']:.1%} đến dưới {art['high_thr']:.1%}</div>
    <div class="sb-tier"><i class="sb-dot" style="background:{TIERS['Cao']['color']}"></i><b>Cao</b>từ {art['high_thr']:.1%}</div>
    <div class="sb-note">Kết quả chỉ mang tính tham khảo. Quyết định cuối cùng thuộc về cán bộ thẩm định tín dụng.</div>
    """)

tab1, tab2, tab3 = st.tabs([":material/fact_check: Chấm điểm hồ sơ",
                            ":material/upload_file: Chấm điểm hàng loạt (CSV)",
                            ":material/info: Giới thiệu mô hình"])


def group_title(title, sub):
    html_block(f'<div class="grp-title">{title}</div><div class="grp-sub">{sub}</div>')


# ---------- TAB 1: CHẤM ĐIỂM TỪNG HỒ SƠ ----------
with tab1:
    st.subheader("Nhập thông tin hồ sơ khách hàng")
    html_block('<p class="lead">Điền ba nhóm thông tin bên dưới. Nếu khách hàng chưa cung cấp thu nhập hoặc '
               'số người phụ thuộc, hãy tắt công tắc tương ứng để hệ thống xử lý như dữ liệu thiếu.</p>')
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(key="grp_personal"):
            group_title("Nhân thân và thu nhập", "Thông tin cá nhân của người vay")
            age = st.number_input("Tuổi", 18, 110, 45)
            inc_known = st.toggle("Có thông tin thu nhập", value=True)
            income = st.number_input("Thu nhập hằng tháng (USD)", 0.0, 1e7, 5000.0,
                                     step=100.0, disabled=not inc_known)
            dep_known = st.toggle("Có thông tin số người phụ thuộc", value=True)
            dependents = st.number_input("Số người phụ thuộc", 0, 20, 1, disabled=not dep_known)
    with c2:
        with st.container(key="grp_debt"):
            group_title("Dư nợ và hạn mức", "Mức độ sử dụng tín dụng hiện tại")
            revol = st.number_input("Tỷ lệ sử dụng hạn mức tín chấp", 0.0, 50.0, 0.3, step=0.01,
                                    help="Dư nợ / hạn mức. Lớn hơn 1 nghĩa là dùng vượt hạn mức.")
            debt = st.number_input("Tỷ lệ nợ / thu nhập (DebtRatio)", 0.0, 1e5, 0.35, step=0.01)
            open_lines = st.number_input("Số khoản vay / hạn mức đang mở", 0, 100, 8)
            real_estate = st.number_input("Số khoản vay bất động sản", 0, 60, 1)
    with c3:
        with st.container(key="grp_history"):
            group_title("Lịch sử trả nợ", "Số lần trễ hạn trong 2 năm gần nhất")
            late30 = st.number_input("Số lần trễ hạn 30–59 ngày", 0, 98, 0)
            late60 = st.number_input("Số lần trễ hạn 60–89 ngày", 0, 98, 0)
            late90 = st.number_input("Số lần trễ hạn ≥ 90 ngày", 0, 98, 0)

    raw = {
        "RevolvingUtilizationOfUnsecuredLines": revol, "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late30, "DebtRatio": debt,
        "MonthlyIncome": income if inc_known else np.nan,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late90, "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late60,
        "NumberOfDependents": dependents if dep_known else np.nan,
    }
    sig = repr(sorted(raw.items()))
    st.write("")
    go_btn = st.button("Đánh giá hồ sơ", type="primary", icon=":material/search:")

    if go_btn:
        with st.spinner("Đang chấm điểm hồ sơ..."):
            df_in = pd.DataFrame([raw])
            res, p_raw = predict_frame(art, df_in)
            contrib, err = None, None
            try:
                contrib = explain(art, df_in)
            except Exception as e:  # giải trình lỗi không được làm hỏng kết quả chính
                err = str(e)
        now = datetime.now()
        st.session_state["case"] = dict(
            sig=sig, raw=raw, p=float(res["Xác suất vỡ nợ"].iloc[0]), p_raw=float(p_raw[0]),
            tier=res["Tầng rủi ro"].iloc[0], contrib=contrib, err=err,
            id=f"HS-{now:%y%m%d-%H%M%S}", time=f"{now:%d/%m/%Y %H:%M}")

    case = st.session_state.get("case")
    if case is None:
        html_block('<div class="empty"><b>Chưa có kết quả.</b> Nhập thông tin hồ sơ ở trên rồi bấm '
                   '<b>Đánh giá hồ sơ</b> để xem xác suất vỡ nợ, tầng rủi ro và lý do dẫn đến kết quả.</div>')
    else:
        p, tier = case["p"], case["tier"]
        t = TIERS[tier]
        st.divider()
        st.subheader("Kết quả đánh giá")
        if case["sig"] != sig:
            st.info("Thông tin nhập đã thay đổi so với lần đánh giá gần nhất. "
                    "Bấm **Đánh giá hồ sơ** để cập nhật kết quả.")

        contrib = case["contrib"]
        top3 = ""
        if contrib is not None:
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(3)
            rows = []
            for k, v in top.items():
                up_ = v > 0
                col = "#A8323E" if up_ else PAL["ocean"]
                rows.append(f'<div class="fx"><i style="color:{col}">{"▲" if up_ else "▼"}</i>'
                            f'<span>{html.escape(FEATURE_LABELS.get(k, k))}</span>'
                            f'<em style="color:{col}">{"Tăng rủi ro" if up_ else "Giảm rủi ro"}</em></div>')
            top3 = '<div class="fx-title">Yếu tố tác động mạnh nhất</div>' + "".join(rows)

        r1, r2 = st.columns([1, 1.25], gap="large")
        with r1:
            html_block(f"""
            <div class="report">
              <div class="report-head"><span>Mã hồ sơ <b>{case['id']}</b></span><span>{case['time']}</span></div>
              <div class="report-body">
                <div class="rp-label">Xác suất vỡ nợ (đã hiệu chỉnh)</div>
                <div class="rp-prob" style="color:{t['color']}">{p:.1%}</div>
                <span class="tier-badge" style="color:{t['text']};background:{t['bg']}">{t['icon']}Rủi ro {tier}</span>
                <div class="rp-action" style="border-left-color:{t['color']}">Đề xuất: {t['action']}</div>
                {top3}
                <div class="rp-foot">Xác suất gốc của mô hình: {case['p_raw']:.3f}</div>
              </div>
            </div>""")
        with r2:
            html_block('<div class="panel-title">Vị trí hồ sơ trên thang rủi ro</div>')
            show_plot(risk_gauge(p, art["low_thr"], art["high_thr"]))
            rel = abs(p - AVG_RATE) / AVG_RATE if AVG_RATE else 0
            st.caption(f"Tỷ lệ vỡ nợ bình quân của danh mục khoảng {AVG_RATE:.1%}. "
                       f"Hồ sơ này {'cao hơn' if p > AVG_RATE else 'thấp hơn'} mức bình quân {rel:.0%}.")
            st.download_button("Tải phiếu đánh giá",
                               build_report_html(case, AVG_RATE).encode("utf-8"),
                               file_name=f"{case['id']}.html", mime="text/html",
                               icon=":material/download:")

        st.subheader("Giải trình quyết định")
        if contrib is not None:
            st.markdown("**Cách đọc biểu đồ:** Cột đỏ làm tăng dự báo rủi ro; cột xanh làm giảm dự báo rủi ro. "
                        "Độ dài cột thể hiện mức đóng góp vào dự báo, không chứng minh quan hệ nhân quả.")
            show_plot(factor_chart(contrib))
        else:
            st.warning(f"Không tạo được biểu đồ giải trình: {case['err']}")

        with st.expander("Xem dữ liệu đầu vào"):
            show = pd.DataFrame({"Giá trị": [fmt_val(v) for v in case["raw"].values()]},
                                index=[FEATURE_LABELS.get(i, i) for i in case["raw"]])
            st.dataframe(show, **STRETCH)


# ---------- TAB 2: CHẤM ĐIỂM HÀNG LOẠT ----------
with tab2:
    st.subheader("Chấm điểm danh mục hồ sơ từ file CSV")
    html_block('<p class="lead">File CSV cần có đủ 10 cột đầu vào. Có thể tải trực tiếp cs-training.csv của bộ dữ liệu; '
               'cột SeriousDlqin2yrs nếu có sẽ được bỏ qua khi dự báo.</p>')
    up = st.file_uploader("Chọn file CSV", type=["csv"])
    res = None
    if up is None:
        html_block('<div class="empty"><b>Chưa có file.</b> Tải lên một file CSV để chấm điểm toàn bộ danh mục '
                   'và xem phân bố hồ sơ theo ba tầng rủi ro.</div>')
    else:
        fkey = (up.name, up.size)
        if st.session_state.get("batch_key") != fkey:
            try:
                with st.spinner("Đang chấm điểm danh mục..."):
                    st.session_state["batch_res"] = score_batch(up)
                st.session_state["batch_key"] = fkey
            except ValueError as e:
                st.session_state.pop("batch_key", None)
                st.error(str(e))
            except Exception as e:
                st.session_state.pop("batch_key", None)
                st.error(f"Lỗi đọc/chấm điểm file: {e}")
        if st.session_state.get("batch_key") == fkey:
            res = st.session_state["batch_res"]

    if res is not None:
        total = len(res)
        counts = res["Tầng rủi ro"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int)

        cards = [f'<div class="kpi total"><div class="k-label">Tổng hồ sơ</div>'
                 f'<div class="k-num">{total:,}</div><div class="k-sub">đã được chấm điểm</div></div>']
        for name in TIER_ORDER:
            tt = TIERS[name]
            cards.append(f'<div class="kpi" style="--c:{tt["color"]}"><div class="k-label">{tt["icon"]}Rủi ro {name}</div>'
                         f'<div class="k-num">{counts[name]:,}</div>'
                         f'<div class="k-sub">{counts[name] / total:.1%} danh mục</div></div>')
        html_block('<div class="kpis">' + "".join(cards) + "</div>")

        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown("#### Phân bố tầng rủi ro")
            fig = go.Figure(go.Bar(
                x=TIER_ORDER, y=counts.values, marker_color=[TIERS[n]["color"] for n in TIER_ORDER],
                text=[f"{c:,}  ({c / total:.1%})" for c in counts.values], textposition="outside",
                cliponaxis=False, hovertemplate="%{x}: %{y:,} hồ sơ<extra></extra>"))
            fig.update_yaxes(title_text="Số hồ sơ", gridcolor=PAL["line"], fixedrange=True,
                             title_font=dict(size=12, color=PAL["muted"]))
            fig.update_xaxes(fixedrange=True)
            fig.update_layout(bargap=0.45)
            show_plot(style_fig(fig, 320))
        with g2:
            st.markdown("#### Phân bố xác suất vỡ nợ")
            h, edges = np.histogram(res["Xác suất vỡ nợ"].clip(0, 1), bins=40, range=(0, 1))
            mids = (edges[:-1] + edges[1:]) / 2
            cols = [TIERS[assign_tier(art, m)]["color"] for m in mids]
            fig = go.Figure(go.Bar(x=mids, y=h, marker_color=cols, width=(edges[1] - edges[0]) * 0.92,
                                   hovertemplate="Xác suất khoảng %{x:.1%}: %{y:,} hồ sơ<extra></extra>"))
            for thr in (art["low_thr"], art["high_thr"]):
                fig.add_vline(x=thr, line_dash="dash", line_color=PAL["navy"], line_width=1.2)
            fig.update_xaxes(tickformat=".0%", fixedrange=True, range=[0, 1])
            fig.update_yaxes(title_text="Số hồ sơ", gridcolor=PAL["line"], fixedrange=True,
                             title_font=dict(size=12, color=PAL["muted"]))
            show_plot(style_fig(fig, 320))
            st.markdown(
                    "**Ghi chú:**\n"
                    "- Đường nét đứt: Hai ngưỡng phân tầng\n"
                    "- Màu cột theo tầng rủi ro"
                )

        st.markdown("#### Chi tiết hồ sơ")
        pick = st.multiselect("Lọc theo tầng rủi ro", TIER_ORDER, default=TIER_ORDER)
        view = res[res["Tầng rủi ro"].isin(pick)] if pick else res.iloc[0:0]
        extra = ["Xác suất vỡ nợ", "Tầng rủi ro", "Hành động đề xuất"]
        disp = view.head(200)[extra + [c for c in view.columns if c not in extra]].copy()
        disp["Xác suất vỡ nợ"] = disp["Xác suất vỡ nợ"] * 100
        disp = disp.rename(columns=FEATURE_LABELS)
        # Tên cột ngắn giúp bảng dễ đọc; dữ liệu xuất CSV vẫn giữ nguyên tên biến gốc.
        short_labels = {
            "Tỷ lệ sử dụng hạn mức tín chấp": "Sử dụng hạn mức",
            "Số lần trễ hạn 30–59 ngày": "Trễ hạn 30–59 ngày",
            "Số lần trễ hạn 60–89 ngày": "Trễ hạn 60–89 ngày",
            "Số lần trễ hạn ≥ 90 ngày": "Trễ hạn ≥ 90 ngày",
            "Số khoản vay / hạn mức đang mở": "Khoản vay đang mở",
            "Số khoản vay bất động sản": "Vay bất động sản",
            "Thu nhập hằng tháng": "Thu nhập tháng",
            "Số người phụ thuộc": "Người phụ thuộc",
            "Tỷ lệ nợ / thu nhập": "Nợ / thu nhập",
        }
        disp = disp.rename(columns=short_labels)
        styler = disp.style
        styler = (styler.map if hasattr(styler, "map") else styler.applymap)(tier_cell_style, subset=["Tầng rủi ro"])
        col_config = {
            "Xác suất vỡ nợ": st.column_config.ProgressColumn(
                "Xác suất vỡ nợ", min_value=0, max_value=100, format="%.1f%%", width="medium"),
            "Tầng rủi ro": st.column_config.TextColumn("Tầng rủi ro", width="small"),
            "Hành động đề xuất": st.column_config.TextColumn("Hành động đề xuất", width="large"),
        }
        st.dataframe(styler, height=390, hide_index=True, **STRETCH, column_config=col_config)
        st.caption(f"Hiển thị tối đa 200 dòng đầu trong {len(view):,} hồ sơ phù hợp bộ lọc.")
        st.download_button("Tải toàn bộ kết quả (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                           "ket_qua_cham_diem.csv", "text/csv", icon=":material/download:")


# ---------- TAB 3: GIỚI THIỆU MÔ HÌNH ----------
with tab3:
    st.subheader("Về hệ thống")
    cost = html.escape(str(art.get("cost_ratio", "10:1")))
    html_block(f"""
    <p class="about">Hệ thống dự báo xác suất vỡ nợ (trễ hạn từ 90 ngày trở lên trong 2 năm) của khách hàng cá nhân,
    dựa trên bộ dữ liệu <b>Give Me Some Credit</b> (Kaggle).</p>""")

    st.markdown("#### Quy trình mô hình")
    html_block(f"""
    <div class="steps">
      <div class="step"><div class="n">1</div><b>Tiền xử lý</b><span>Làm sạch mã đặc biệt 96/98, gắn cờ dữ liệu thiếu,
        điền thiếu bằng KNNImputer, tạo đặc trưng mới như tổng số lần trễ hạn.</span></div>
      <div class="step"><div class="n">2</div><b>Chọn mô hình</b><span>Mô hình chốt {MODEL_NAME}, chọn theo PR-AUC
        trên kiểm định chéo 5-fold.</span></div>
      <div class="step"><div class="n">3</div><b>Hiệu chỉnh xác suất</b><span>Dùng phương pháp {html.escape(str(art['cal_method']))}
        để xác suất phản ánh đúng tần suất thực tế.</span></div>
      <div class="step"><div class="n">4</div><b>Phân tầng rủi ro</b><span>Ngưỡng tối ưu theo chi phí (FN:FP = {cost}),
        chia khách hàng thành ba tầng.</span></div>
    </div>""")

    st.markdown("#### Chính sách ba tầng")
    pol = []
    thr = {"Thấp": f"dưới {art['low_thr']:.1%}",
           "Trung bình": f"{art['low_thr']:.1%} đến dưới {art['high_thr']:.1%}",
           "Cao": f"từ {art['high_thr']:.1%}"}
    for name in TIER_ORDER:
        tt = TIERS[name]
        pol.append(f'<div class="pol" style="--c:{tt["color"]}"><div class="p-name">{tt["icon"]}Rủi ro {name}</div>'
                   f'<div class="p-thr">{thr[name]}</div><div class="p-act">{tt["action"]}</div></div>')
    html_block('<div class="policy">' + "".join(pol) + "</div>")

    st.markdown("#### Lưu ý sử dụng")
    html_block("""
    <div class="notice">Mô hình đóng vai trò <b>hỗ trợ</b>, không thay thế quyết định của cán bộ thẩm định.<br>
    Định nghĩa "vỡ nợ" là đại diện thực nghiệm (biến SeriousDlqin2yrs), không phải định nghĩa pháp lý theo
    Basel hoặc IFRS 9. Ma trận chi phí FN:FP chỉ mang tính mô phỏng.</div>""")

html_block(f"""
<div class="footer">Nhóm 5 - Phân tích dữ liệu nâng cao.<br> Giảng viên hướng dẫn: TS. Nguyễn Thôn Dã.<br>
Nguồn mô hình: {html.escape(str(used_path))}</div>""")
