#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run three forecasting experiments on state construction employment with:
  • XGBoost (multivariate, excludes any column whose name contains 'dist')
  • Prophet (univariate)

Experiments:
  - L24_H6  : use last 24 months to forecast next 6 months
  - L36_H12 : use last 36 months to forecast next 12 months
  - L48_H24 : use last 48 months to forecast next 24 months

Outputs:
  - statewise_xgb_MAE_multivar_noDist.xlsx      (metrics, forecasts, summary)
  - statewise_prophet_MAE_windows.xlsx          (metrics, forecasts, summary)
  - compare_summary.xlsx                        (method-by-experiment Avg MAE table)

Usage:
  1) Place this script in the same folder as From_1990.xlsx
  2) pip install -U xgboost scikit-learn prophet openpyxl pandas numpy
  3) python run_windows_models.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, List

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise SystemExit("xgboost is required. Install it with: pip install xgboost")

try:
    from prophet import Prophet  # current name
except Exception:
    try:
        from fbprophet import Prophet  # legacy name
    except Exception:
        raise SystemExit("prophet is required. Install it with: pip install prophet")

SEED = 42
INPUT = 'From_1990.xlsx'
DATECOL = 'Date'
STATECOL = 'State'
TARGET = 'Construction_Employment_State'

EXPERIMENTS = [
    {"L":24, "H":6,  "label":"L24_H6"},
    {"L":36, "H":12, "label":"L36_H12"},
    {"L":48, "H":24, "label":"L48_H24"},
]

# ---------------------------
# Helpers
# ---------------------------
def make_lag_supervised(series: pd.Series, L: int, H: int):
    """Return (X_lag, Y_h) for one series indexed by Date.
    X has columns lag_1..lag_L (oldest->newest), Y has h+1..h+H.
    Index = anchor date (end of context window).
    """
    y = series.values.astype(float)
    idx = series.index
    Xs, Ys, anchors = [], [], []
    for t in range(L-1, len(y) - H):
        Xs.append(y[t-L+1:t+1])
        Ys.append(y[t+1:t+1+H])
        anchors.append(idx[t])
    if not Xs:
        return pd.DataFrame(), pd.DataFrame(), pd.DatetimeIndex([])
    X = pd.DataFrame(np.asarray(Xs), index=pd.DatetimeIndex(anchors),
                     columns=[f"lag_{i}" for i in range(1, L+1)])
    Y = pd.DataFrame(np.asarray(Ys), index=pd.DatetimeIndex(anchors),
                     columns=[f"h+{i}" for i in range(1, H+1)])
    return X, Y, pd.DatetimeIndex(anchors)

# ---------------------------
# Load data
# ---------------------------
path = Path(INPUT)
if not path.exists():
    raise SystemExit(f"Input file not found: {INPUT}")
df = pd.read_excel(INPUT)
df.columns = [re.sub(r"\s+", "_", c).strip() for c in df.columns]
df[DATECOL] = pd.to_datetime(df[DATECOL])
# Exclude any column that contains 'dist' (case-insensitive)
df = df.loc[:, ~df.columns.str.lower().str.contains('dist')].copy()

df = df.sort_values([STATECOL, DATECOL]).reset_index(drop=True)
states = sorted(df[STATECOL].dropna().unique().tolist())

# ---------------------------
# XGBoost (multivariate) per state
# ---------------------------
xgb_metrics_rows: List[dict] = []
xgb_fcst_rows: List[dict] = []

for exp in EXPERIMENTS:
    L, H, label = exp['L'], exp['H'], exp['label']
    for st in states:
        sdf = df.loc[df[STATECOL]==st].copy().sort_values(DATECOL)
        feat_cols = [c for c in sdf.columns if c not in [TARGET, DATECOL, STATECOL]]
        sdf = sdf[[DATECOL, TARGET] + feat_cols].dropna(subset=[TARGET])
        if sdf.empty:
            xgb_metrics_rows.append({'State':st,'experiment':label,'MAE':np.nan,'note':'empty'})
            continue
        sdf = sdf.set_index(DATECOL)
        X_lag, Y, anchors = make_lag_supervised(sdf[TARGET], L, H)
        if X_lag.empty or len(X_lag) < 2:
            xgb_metrics_rows.append({'State':st,'experiment':label,'MAE':np.nan,'note':'insufficient_windows'})
            continue
        exog_at_anchor = sdf[feat_cols].reindex(anchors)
        X = pd.concat([X_lag.reset_index(drop=True), exog_at_anchor.reset_index(drop=True)], axis=1)
        X.index = anchors
        pre = ColumnTransformer([('num', SimpleImputer(strategy='median'), X.columns.tolist())], remainder='drop')
        X_train, Y_train = X.iloc[:-1], Y.iloc[:-1]
        X_test,  Y_test  = X.iloc[-1:], Y.iloc[-1:]
        base = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective='reg:squarederror', random_state=SEED, n_jobs=4
        )
        model = Pipeline([
            ('pre', pre),
            ('reg', MultiOutputRegressor(base))
        ])
        model.fit(X_train, Y_train)
        Y_hat = pd.DataFrame(model.predict(X_test), columns=Y.columns, index=Y_test.index)
        mae = float(mean_absolute_error(Y_test.values.reshape(-1), Y_hat.values.reshape(-1)))
        xgb_metrics_rows.append({'State':st,'experiment':label,'MAE':mae})
        anchor = X_test.index[0]
        fdates = pd.date_range(anchor + pd.offsets.MonthBegin(1), periods=H, freq='MS')
        for i, fd in enumerate(fdates, start=1):
            xgb_fcst_rows.append({'State':st,'experiment':label,'forecast_date':fd,'h':i,'y_pred':float(Y_hat.iloc[0, i-1]),'y_actual':float(Y_test.iloc[0, i-1])})

xgb_metrics = pd.DataFrame(xgb_metrics_rows)
xgb_fcst = pd.DataFrame(xgb_fcst_rows)
xgb_summary = (
    xgb_metrics.groupby(['experiment'], dropna=False)['MAE']
    .mean().reset_index().rename(columns={'MAE':'Avg_MAE'})
)

with pd.ExcelWriter('statewise_xgb_MAE_multivar_noDist.xlsx', engine='openpyxl') as w:
    xgb_metrics.to_excel(w, sheet_name='metrics', index=False)
    xgb_fcst.to_excel(w, sheet_name='test_forecasts', index=False)
    xgb_summary.to_excel(w, sheet_name='summary', index=False)

# ---------------------------
# Prophet (univariate) per state
# ---------------------------
pp_metrics_rows: List[dict] = []
pp_fcst_rows: List[dict] = []

for exp in EXPERIMENTS:
    L, H, label = exp['L'], exp['H'], exp['label']
    for st in states:
        sdf = df.loc[df[STATECOL]==st, [DATECOL, TARGET]].dropna().sort_values(DATECOL)
        if len(sdf) < L + H:
            pp_metrics_rows.append({'State':st,'experiment':label,'MAE':np.nan,'note':'insufficient_history'})
            continue
        train = sdf.iloc[-(L+H):-H, :].copy()
        test  = sdf.iloc[-H:, :].copy()
        m = Prophet(growth='linear', yearly_seasonality=True,
                    weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode='additive')
        m.fit(train.rename(columns={DATECOL:'ds', TARGET:'y'}))
        future = m.make_future_dataframe(periods=H, freq='MS').iloc[-H:]
        fc = m.predict(future)[['ds','yhat']]
        merged = test.rename(columns={DATECOL:'ds', TARGET:'y'}).merge(fc, on='ds', how='left')
        mae = float(mean_absolute_error(merged['y'], merged['yhat']))
        pp_metrics_rows.append({'State':st,'experiment':label,'MAE':mae})
        for i, r in enumerate(merged.itertuples(index=False), start=1):
            pp_fcst_rows.append({'State':st,'experiment':label,'forecast_date':r.ds,'h':i,'y_pred':float(r.yhat),'y_actual':float(r.y)})

pp_metrics = pd.DataFrame(pp_metrics_rows)
pp_fcst = pd.DataFrame(pp_fcst_rows)
pp_summary = (
    pp_metrics.groupby(['experiment'], dropna=False)['MAE']
    .mean().reset_index().rename(columns={'MAE':'Avg_MAE'})
)

with pd.ExcelWriter('statewise_prophet_MAE_windows.xlsx', engine='openpyxl') as w:
    pp_metrics.to_excel(w, sheet_name='metrics', index=False)
    pp_fcst.to_excel(w, sheet_name='test_forecasts', index=False)
    pp_summary.to_excel(w, sheet_name='summary', index=False)

# ---------------------------
# Combined summary (Avg MAE across states by experiment & method)
# ---------------------------
compare = (pd.concat([
    xgb_summary.assign(Method='XGBoost'),
    pp_summary.assign(Method='Prophet')
])
    .pivot(index='experiment', columns='Method', values='Avg_MAE')
    .reset_index()
)

with pd.ExcelWriter('compare_summary.xlsx', engine='openpyxl') as w:
    compare.to_excel(w, sheet_name='avg_mae_by_method', index=False)

print('Saved files:')
print(' - statewise_xgb_MAE_multivar_noDist.xlsx')
print(' - statewise_prophet_MAE_windows.xlsx')
print(' - compare_summary.xlsx')
