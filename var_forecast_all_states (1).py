# VAR forecasting of Construction_Employment_State with rolling MAE at h = 6, 12, 24
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.api import VAR

INPUT_FILE = 'From_1990.xlsx'
TARGET_COL = 'Construction_Employment_State'
STATE_COL = 'State'
DATE_COL = 'Date'

CANDIDATE_STATE_COLS = [
    'Construction_Employment_State',
    'Coincident_Economic_Activity_State',
    'Labor_Force_Participation_State',
    'Permits_Housing_State',
    'Unemployment_Rate_State',
    'Property_Damage_State',
]

HORIZONS = [6, 12, 24]
MAX_LAGS = 12
MIN_TRAIN_YEARS = 8
REFIT_EVERY = 12
VAR_TREND = 'ct'


def _to_monthly_index(s: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(s, errors='coerce')
    return pd.DatetimeIndex(dt).to_period('M').to_timestamp(how='start')


def prepare_state_frame(df: pd.DataFrame, state: str) -> pd.DataFrame:
    sub = df[df[STATE_COL] == state].copy()
    if sub.empty:
        return sub
    sub[DATE_COL] = _to_monthly_index(sub[DATE_COL])
    sub = sub.sort_values(DATE_COL)
    keep_cols = [c for c in CANDIDATE_STATE_COLS if c in sub.columns]
    if TARGET_COL not in keep_cols:
        raise ValueError(f"Target {TARGET_COL} not found for state {state}")
    sub = sub[[DATE_COL] + keep_cols].dropna(subset=[TARGET_COL])
    sub = sub.set_index(DATE_COL).asfreq('MS').ffill()
    return sub


def select_var_order(y: pd.DataFrame, maxlags: int = MAX_LAGS, trend: str = VAR_TREND):
    model = VAR(y)
    res = model.fit(maxlags=maxlags, ic='aic', trend=trend)
    return res, res.k_ar


def rolling_var_mae_fast(y: pd.DataFrame,
                         target_col: str = TARGET_COL,
                         horizons = HORIZONS,
                         maxlags: int = MAX_LAGS,
                         min_train_years: int = MIN_TRAIN_YEARS,
                         trend: str = VAR_TREND,
                         refit_every: int = REFIT_EVERY):
    y = y.copy()
    n_min = int(min_train_years * 12)
    max_h = max(horizons)
    if y.shape[0] < n_min + max_h + 1:
        return {h: np.nan for h in horizons}, [], 0

    start = n_min
    errors = {h: [] for h in horizons}
    used_orders = []

    res = None
    last_fit_end = None
    fixed_k_ar = None

    for t in range(start, y.shape[0] - max_h):
        need_refit = (res is None) or (last_fit_end is None) or ((t - (last_fit_end + 1)) % refit_every == 0)
        train = y.iloc[:t]
        if need_refit:
            if fixed_k_ar is None:
                res, fixed_k_ar = select_var_order(train, maxlags=maxlags, trend=trend)
            else:
                res = VAR(train).fit(fixed_k_ar, trend=trend)
            used_orders.append(fixed_k_ar)
            last_fit_end = t - 1

        k_ar = res.k_ar
        y_last = y.iloc[t - k_ar: t]
        fcst = res.forecast(y_last.values, steps=max_h)
        fcst = pd.DataFrame(fcst, index=y.index[t: t + max_h], columns=y.columns)

        for h in horizons:
            y_true = y[target_col].iloc[t + h - 1]
            y_hat  = fcst[target_col].iloc[h - 1]
            errors[h].append(abs(y_true - y_hat))

    mae_by_h = {h: float(np.mean(errors[h])) if errors[h] else np.nan for h in horizons}
    n_origins = y.shape[0] - max_h - start
    return mae_by_h, used_orders, n_origins


def fit_full_and_forecast(y: pd.DataFrame, horizons=HORIZONS, maxlags: int = MAX_LAGS, trend: str = VAR_TREND):
    res, k_ar = select_var_order(y, maxlags=maxlags, trend=trend)
    max_h = max(horizons)
    fc = res.forecast(y.values[-k_ar:], steps=max_h)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=max_h, freq='MS')
    fc = pd.DataFrame(fc, index=idx, columns=y.columns)
    picks = {h: float(fc[TARGET_COL].iloc[h - 1]) for h in horizons}
    return fc, picks, k_ar


def main(output_path: str = 'var_forecast_results.xlsx'):
    df = pd.read_excel(INPUT_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    states = sorted([s for s in df[STATE_COL].dropna().unique()])

    results_rows, fc_rows, orders_rows = [], [], []

    for state in states:
        try:
            y = prepare_state_frame(df, state)
            if y.empty or TARGET_COL not in y.columns:
                continue

            mae_by_h, used_orders, n_origins = rolling_var_mae_fast(y)
            fc_full, picks, order_full = fit_full_and_forecast(y)

            results_rows.append({
                'State': state,
                'n_obs': y.shape[0],
                'n_origins': n_origins,
                **{f'MAE_h{h}': mae_by_h[h] for h in HORIZONS},
            })
            fc_rows.append({
                'State': state,
                **{f'Forecast_h{h}': picks[h] for h in HORIZONS},
                'SelectedLag_full': order_full,
            })
            if used_orders:
                orders_rows.append({'State': state,
                                    'order_fixed': int(used_orders[0]),
                                    'refits': len(used_orders)})
        except Exception as ex:
            results_rows.append({'State': state, 'error': str(ex)})

    mae_df = pd.DataFrame(results_rows).sort_values('State')
    fc_df  = pd.DataFrame(fc_rows).sort_values('State')
    orders_df = pd.DataFrame(orders_rows).sort_values('State') if len(orders_rows) else pd.DataFrame()

    with pd.ExcelWriter(output_path, engine='openpyxl') as xlw:
        mae_df.to_excel(xlw, sheet_name='MAE_by_state', index=False)
        fc_df.to_excel(xlw, sheet_name='PointForecasts_h6_h12_h24', index=False)
        if not orders_df.empty:
            orders_df.to_excel(xlw, sheet_name='LagOrder_diagnostics', index=False)

    print(f'Wrote results to: {Path(output_path).resolve().as_posix()}')

if __name__ == '__main__':
    main()
