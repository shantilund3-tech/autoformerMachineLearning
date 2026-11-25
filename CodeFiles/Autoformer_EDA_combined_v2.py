# Combined EDA for the Autoformer project: ETTm2, ECL (Electricity), Commodities, Exchange-Rate.
# Written for Google Colab. Each "# %%" block is one cell. Run top to bottom.
# Every dataset keeps its own variable name (ett, ecl, cmd, exh), so sections cannot overwrite each other.

# %% [markdown]
# # EDA for the Autoformer project
# Steps per dataset: load, audit, plots, outliers, trend and seasonality, ACF, FFT, stationarity, relationships, split and scaling, naive baselines.
# Univariate steps study one series at a time. Multivariate steps study how series relate and are forecast together.

# %% Imports and helper functions
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams["figure.figsize"] = (12, 4)
warnings.filterwarnings("ignore")      # after the imports: statsmodels re-enables its own warnings when imported

def audit(d, name):
    print(f"== {name} ==")
    print("Shape:", d.shape)
    print("Range:", d.index.min(), "to", d.index.max())
    miss = d.isna().sum()
    print("Missing values:", miss[miss > 0].to_dict() if miss.any() else "none")
    print("Duplicate index values:", d.index.duplicated().sum(), "| duplicate rows:", d.duplicated().sum())
    print("Most common gaps between rows:\n", d.index.to_series().diff().value_counts().head(5))
    display(d.describe().T)

def iqr_outliers(d):
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    i = q3 - q1
    return ((d < q1 - 1.5 * i) | (d > q3 + 1.5 * i)).sum()

def stationarity(x):
    return {"ADF p": round(adfuller(x)[1], 4), "KPSS p": round(kpss(x, regression="c", nlags="auto")[1], 4)}
# ADF p small = stationary. KPSS p small = NOT stationary. If they disagree, the series is borderline.

def fft_peaks(x, min_p, max_p):
    y = (x - x.mean()).values
    spec = np.abs(np.fft.rfft(y))
    per = 1 / np.maximum(np.fft.rfftfreq(len(y), d=1.0), 1e-12)      # period in steps
    m = (per >= min_p) & (per <= max_p)
    return per[m], spec[m]

def cycle_table(d, lags, min_p, max_p, step_hours):
    rows = []
    for c in d.columns:
        ac = acf(d[c], nlags=max(lags), fft=True)
        per, sp = fft_peaks(d[c], min_p, max_p)
        row = {"series": c}
        row.update({f"ACF lag {l}": round(ac[l], 2) for l in lags})
        row["strongest period (h)"] = round(per[sp.argmax()] * step_hours, 1)
        row.update(stationarity(d[c]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")

def seasonal_strength(x, period):
    r = seasonal_decompose(x, period=period, model="additive", extrapolate_trend="freq")
    return max(0.0, 1 - r.resid.var() / (r.seasonal + r.resid).var())

def split_info(d, ett=False):
    n = len(d)
    if ett:                                  # ETT protocol: 12 / 4 / 4 months (15-minute data)
        return 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
    n_train, n_test = int(n * 0.7), int(n * 0.2)
    return n_train, n - n_train - n_test, n_test

def split_stats(d, n_train, n_val, n_test):
    tr = d.iloc[:n_train]
    va = d.iloc[n_train:n_train + n_val]
    te = d.iloc[n_train + n_val:n_train + n_val + n_test]
    print("train", tr.shape, "|", tr.index.min(), "to", tr.index.max())
    print("val  ", va.shape, "|", va.index.min(), "to", va.index.max())
    print("test ", te.shape, "|", te.index.min(), "to", te.index.max())
    display(pd.DataFrame({"train mean": tr.mean(), "val mean": va.mean(), "test mean": te.mean(),
                          "train std": tr.std(), "test std": te.std()}).round(3))
    # A test mean far from the train mean means distribution shift.

def scale_train_only(d, n_train):
    mu, sd = d.iloc[:n_train].mean(), d.iloc[:n_train].std()      # scaler fitted on the training part only
    return ((d - mu) / sd).values

def naive_baselines(scaled, n_train, n_val, n_test, season=None, cols=None, seq=96, horizons=(96, 192, 336, 720)):
    # A baseline is a simple rule, not a trained model. It is the error a trained model must beat.
    seg = scaled[n_train + n_val - seq:n_train + n_val + n_test]
    rows = []
    for H in horizons:
        win = sliding_window_view(seg, seq + H, axis=0)            # (windows, series, seq + H)
        x, yt = win[:, :, :seq], win[:, :, seq:]
        preds = {"last value": np.repeat(x[:, :, -1:], H, axis=2)}
        if season:
            preds[f"repeat last {season} steps"] = np.tile(x[:, :, -season:], (1, 1, int(np.ceil(H / season))))[:, :, :H]
        else:
            preds["input mean"] = np.repeat(x.mean(axis=2, keepdims=True), H, axis=2)
        for name, p in preds.items():
            err = p - yt
            row = {"horizon": H, "baseline": name, "MSE": round(float(np.mean(err ** 2)), 3),
                   "MAE": round(float(np.mean(np.abs(err))), 3)}
            if cols is not None:
                row["MSE selected"] = round(float(np.mean(err[:, cols] ** 2)), 3)
                row["MAE selected"] = round(float(np.mean(np.abs(err[:, cols]))), 3)
            rows.append(row)
    return pd.DataFrame(rows)

# %% Upload the four files (select all four at once)
try:
    from google.colab import files
    files.upload()      # ETTm2.csv, electricity_subset10.csv, commodities_full.csv, exchange_rate.csv
except ImportError:
    pass

# %% [markdown]
# # 1. ETTm2 (15-minute data, 7 variables, target OT = oil temperature)

# %% ETTm2: load and audit
ett = pd.read_csv("ETTm2.csv", parse_dates=["date"], index_col="date")
audit(ett, "ETTm2")
# HUFL/HULL = high useful/useless load, MUFL/MULL = middle, LUFL/LULL = low, OT = oil temperature.

# %% ETTm2: raw plots (univariate view of every series)
ett.plot(subplots=True, figsize=(12, 14), lw=0.5, title="ETTm2 - all variables"); plt.tight_layout(); plt.show()
ett.iloc[:96 * 14].plot(subplots=True, figsize=(12, 14), lw=0.8, title="First 14 days"); plt.tight_layout(); plt.show()

# %% ETTm2: distributions and outliers
ett.hist(bins=60, figsize=(12, 8)); plt.tight_layout(); plt.show()
ett.plot(kind="box", subplots=True, layout=(2, 4), figsize=(12, 6), sharex=False, sharey=False); plt.tight_layout(); plt.show()
print("IQR outlier counts:\n", iqr_outliers(ett))
# Outliers are kept: deleting rows breaks the regular time spacing the model needs, and the IQR rule over-flags skewed data.

# %% ETTm2: trend and seasonality of OT
ot = ett["OT"]
seasonal_decompose(ot.iloc[:96 * 60], model="additive", period=96).plot()          # 96 steps = 1 day
plt.suptitle("OT: first 60 days, daily period"); plt.show()
ot.resample("D").mean().plot(title="OT daily mean (long-term trend)"); plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ot.groupby(ot.index.hour).mean().plot(ax=ax[0], title="OT by hour")
ot.groupby(ot.index.dayofweek).mean().plot(ax=ax[1], title="OT by weekday (0 = Mon)")
ot.groupby(ot.index.month).mean().plot(ax=ax[2], title="OT by month")
plt.tight_layout(); plt.show()

# %% ETTm2: autocorrelation and FFT of OT
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(ot, lags=96 * 7, ax=ax[0], title="ACF of OT, 7 days of lags")
plot_pacf(ot, lags=200, ax=ax[1], title="PACF of OT")
plt.tight_layout(); plt.show()
per, sp = fft_peaks(ot, 4, 24 * 60 * 4)
plt.semilogx(per * 0.25, sp); plt.xlabel("period (hours)"); plt.title("OT spectrum"); plt.show()
print("Top periods (hours):", np.round(per[np.argsort(sp)[::-1][:5]] * 0.25, 1))

# %% ETTm2: stationarity of OT and cycle/stationarity table for all variables
print("OT:", stationarity(ot), "| first difference ADF p =", round(adfuller(ot.diff().dropna())[1], 4))
display(cycle_table(ett, [96, 672], 4, 24 * 60 * 4, 0.25))

# %% ETTm2: relationships between variables (multivariate view)
plt.figure(figsize=(7, 6)); sns.heatmap(ett.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0); plt.title("Correlation"); plt.show()
lags = [0, 4, 24, 96, 192]
cc = pd.DataFrame({c: [ot.corr(ett[c].shift(l)) for l in lags] for c in ett.columns if c != "OT"}, index=lags)
cc.index.name = "lag (steps)"
print("Correlation of OT with each variable lagged by k steps:"); display(cc.round(2))

# %% ETTm2: split, scaling and naive baselines
n_train, n_val, n_test = split_info(ett, ett=True)
split_stats(ett, n_train, n_val, n_test)
ett_scaled = scale_train_only(ett, n_train)
display(naive_baselines(ett_scaled, n_train, n_val, n_test, season=96))        # season 96 = repeat the last day

# %% [markdown]
# # 2. ECL (hourly electricity consumption, 10-series subset)

# %% ECL: load and audit
ecl = pd.read_csv("electricity_subset10.csv", parse_dates=["date"], index_col="date")
audit(ecl, "ECL subset")
print("Share of zero values per series:\n", (ecl == 0).mean().round(3))
# The original UCI data covers 2011-2014, so the dates in this file may be nominal.

# %% ECL: raw plots
ecl.plot(subplots=True, figsize=(12, 18), lw=0.4, title="ECL subset - all series"); plt.tight_layout(); plt.show()
ecl.iloc[:24 * 14].plot(subplots=True, figsize=(12, 18), lw=0.8, title="First 14 days"); plt.tight_layout(); plt.show()

# %% ECL: distributions and outliers
ecl.hist(bins=60, figsize=(12, 8)); plt.tight_layout(); plt.show()
ecl.plot(kind="box", subplots=True, layout=(2, 5), figsize=(14, 6), sharex=False, sharey=False); plt.tight_layout(); plt.show()
print("IQR outlier counts:\n", iqr_outliers(ecl))
# Kept: heavy upper tails are real consumption peaks; zeros are flagged as data-quality notes.

# %% ECL: trend and seasonality of OT
s = ecl["OT"]
seasonal_decompose(s.iloc[:24 * 60], model="additive", period=24).plot(); plt.suptitle("OT: daily period (24 h)"); plt.show()
seasonal_decompose(s.iloc[:24 * 120], model="additive", period=168).plot(); plt.suptitle("OT: weekly period (168 h)"); plt.show()
s.resample("D").mean().plot(title="OT daily mean"); plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
s.groupby(s.index.hour).mean().plot(ax=ax[0], title="By hour")
s.groupby(s.index.dayofweek).mean().plot(ax=ax[1], title="By weekday (0 = Mon)")
s.groupby(s.index.month).mean().plot(ax=ax[2], title="By month")
plt.tight_layout(); plt.show()
plt.figure(figsize=(12, 3.5)); sns.heatmap(s.groupby([s.index.dayofweek, s.index.hour]).mean().unstack(), cmap="viridis")
plt.xlabel("hour"); plt.ylabel("day of week"); plt.title("OT: average by weekday and hour"); plt.show()

# %% ECL: autocorrelation and FFT of OT
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(s, lags=24 * 14, ax=ax[0], title="ACF of OT, 14 days of lags")
plot_acf(s, lags=48, ax=ax[1], title="ACF, first 48 lags (peak at 24 = daily cycle)")
plt.tight_layout(); plt.show()
per, sp = fft_peaks(s, 2, 24 * 60)
plt.semilogx(per, sp); plt.xlabel("period (hours)"); plt.title("OT spectrum"); plt.show()
print("Top periods (hours):", np.round(per[np.argsort(sp)[::-1][:6]], 1))

# %% ECL: cycle/stationarity table for all series, relationships, split and baselines
display(cycle_table(ecl, [24, 168], 2, 24 * 60, 1.0))
plt.figure(figsize=(8, 6)); sns.heatmap(ecl.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0); plt.title("Correlation"); plt.show()
lags = [0, 1, 6, 12, 24, 168]
cc = pd.DataFrame({c: [s.corr(ecl[c].shift(l)) for l in lags] for c in ecl.columns if c != "OT"}, index=lags)
cc.index.name = "lag (hours)"
print("Correlation of OT with each series lagged by k hours:"); display(cc.round(2))
n_train, n_val, n_test = split_info(ecl)
split_stats(ecl, n_train, n_val, n_test)
ecl_scaled = scale_train_only(ecl, n_train)
display(naive_baselines(ecl_scaled, n_train, n_val, n_test, season=24))        # season 24 = repeat the last day

# %% [markdown]
# # 3. Commodities (prices, weather, Dollar Index)

# %% Commodities: load and audit
cmd = pd.read_csv("commodities_full.csv", parse_dates=["date"], index_col="date")
PRICES = ["NaturalGas", "WTI", "Wheat", "Corn", "Soybeans", "Sugar", "Coffee", "Copper"]
WEATHER = [c for c in cmd.columns if c.endswith("_temp") or c.endswith("_precip")]
FX = [c for c in ["DollarIndex", "USD_PKR"] if c in cmd.columns]
# Data-quality fix: the file has 22 rows in January 2003, then jumps 333 days to 2003-12-01.
# A return across that gap is not a daily return (it caused the largest "returns" for 4 commodities), so start after the last long gap.
long_gaps = cmd.index.to_series().diff().dt.days
print("Gaps longer than 7 days:\n", long_gaps[long_gaps > 7])
if (long_gaps > 30).any():
    cmd = cmd.loc[long_gaps[long_gaps > 30].index[-1]:]
print("Rows per year (about 252 expected; fewer means trading days were lost when weather/FX were merged in):\n",
      cmd.groupby(cmd.index.year).size().to_dict())
audit(cmd[PRICES], "Commodity prices")
print("Days between rows (3 = weekend):\n", cmd.index.to_series().diff().dt.days.value_counts().head(8))
print("Non-positive prices (WTI went negative in April 2020):\n", (cmd[PRICES] <= 0).sum())
print("Share of days with unchanged price:\n", (cmd[PRICES].diff() == 0).mean().round(3))

# %% Commodities: raw plots
cmd[PRICES].plot(subplots=True, figsize=(12, 16), lw=0.6, title="Commodity prices (units differ)"); plt.tight_layout(); plt.show()
pos_prices = cmd[PRICES].where(cmd[PRICES] > 0)
(pos_prices / pos_prices.bfill().iloc[0]).plot(figsize=(12, 5), lw=0.8, logy=True, title="Rebased to 1 (log scale)"); plt.show()
cmd[FX].plot(subplots=True, figsize=(12, 3 * len(FX)), lw=0.7, title="Dollar Index and USD/PKR"); plt.tight_layout(); plt.show()

# %% Commodities: distributions and outliers
cmd[PRICES].hist(bins=50, figsize=(12, 8)); plt.tight_layout(); plt.show()
cmd[PRICES].plot(kind="box", subplots=True, layout=(2, 4), figsize=(12, 6), sharex=False, sharey=False); plt.tight_layout(); plt.show()
display(cmd[WEATHER].describe().T.round(2))

# %% Commodities: returns and volatility
cmd_ret = np.log(cmd[PRICES].where(cmd[PRICES] > 0)).diff()                  # undefined where price <= 0
cmd_ret.plot(subplots=True, figsize=(12, 16), lw=0.4, title="Daily log returns"); plt.tight_layout(); plt.show()
display(pd.DataFrame({"skew": cmd_ret.skew(), "excess kurtosis": cmd_ret.kurt()}).round(2))
cmd_ret.rolling(21).std().plot(figsize=(12, 5), lw=0.8, title="21-day rolling volatility"); plt.show()
display(pd.DataFrame({"largest |return|": cmd_ret.abs().max(), "date": cmd_ret.abs().idxmax()}).round(3))

# %% Commodities: seasonality
print("Seasonal strength (0 = none, 1 = very strong); periods in trading days: 5 week, 21 month, 252 year")
display(pd.DataFrame({f"period {p}": {c: round(seasonal_strength(cmd[c], p), 3) for c in PRICES} for p in (5, 21, 252)}))
seasonal_decompose(cmd["NaturalGas"], model="additive", period=252, extrapolate_trend="freq").plot()
plt.suptitle("Natural gas: yearly period (252 trading days)"); plt.show()
ratio = cmd[PRICES] / cmd[PRICES].rolling(252, center=True, min_periods=126).mean()
plt.figure(figsize=(9, 5)); sns.heatmap(ratio.groupby(ratio.index.month).mean().T, annot=True, fmt=".2f", cmap="coolwarm", center=1)
plt.title("Average price relative to its 1-year average, by month"); plt.xlabel("month"); plt.show()

# %% Commodities: autocorrelation (prices, returns, volatility)
rows = []
for c in PRICES:
    a = acf(cmd_ret[c].dropna(), nlags=21, fft=True)
    ad = acf((cmd[c] - cmd[c].rolling(252).mean()).dropna(), nlags=252, fft=True)
    av = acf(cmd_ret[c].abs().dropna(), nlags=63, fft=True)
    rows.append({"commodity": c, "return lag 1": a[1], "return lag 5": a[5], "return lag 21": a[21],
                 "detrended price lag 126": ad[126], "detrended price lag 252": ad[252],
                 "|ret| lag 1": av[1], "|ret| lag 21": av[21], "|ret| lag 63": av[63]})
display(pd.DataFrame(rows).set_index("commodity").round(2))
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(cmd["Wheat"], lags=400, ax=ax[0], title="ACF of Wheat PRICE")
plot_acf(cmd_ret["Wheat"].dropna(), lags=100, ax=ax[1], title="ACF of Wheat RETURNS")
plt.tight_layout(); plt.show()

# %% Commodities: stationarity
rows = []
for c in PRICES:
    r = cmd_ret[c].dropna()
    rows.append({"commodity": c, **{"price " + k: v for k, v in stationarity(cmd[c]).items()},
                 **{"return " + k: v for k, v in stationarity(r).items()}})
display(pd.DataFrame(rows).set_index("commodity"))

# %% Commodities: relationships with oil, the dollar and each other (multivariate view)
fx_ret = np.log(cmd[FX]).diff()
allret = pd.concat([cmd_ret, fx_ret], axis=1).dropna()
plt.figure(figsize=(9, 7)); sns.heatmap(allret.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0); plt.title("Correlation of daily returns"); plt.show()
dxy = allret["DollarIndex"]
lead = pd.DataFrame({c: [allret[c].corr(dxy.shift(k)) for k in range(0, 6)] for c in PRICES}, index=range(0, 6))
lead.index.name = "dollar leads by k days"
display(lead.round(2))
allret[PRICES].rolling(252).corr(dxy).plot(figsize=(12, 5), lw=0.8, title="Rolling 1-year correlation with the Dollar Index")
plt.axhline(0, color="k", lw=0.5); plt.show()

# %% Commodities: weather link (monthly weather anomaly vs price change)
agg = {c: "last" for c in PRICES}
agg.update({c: ("sum" if c.endswith("_precip") else "mean") for c in WEATHER})
try:
    m = cmd[PRICES + WEATHER].resample("ME").agg(agg)
except ValueError:
    m = cmd[PRICES + WEATHER].resample("M").agg(agg)
anom = m[WEATHER] - m[WEATHER].groupby(m.index.month).transform("mean")
lp = np.log(m[PRICES].where(m[PRICES] > 0)).diff()
pairs = [("Wheat", "Kansas_precip"), ("Wheat", "Kansas_temp"), ("Corn", "Iowa_precip"), ("Corn", "Iowa_temp"),
         ("Soybeans", "Iowa_precip"), ("Coffee", "MinasGerais_precip"), ("Coffee", "MinasGerais_temp"),
         ("Sugar", "SaoPaulo_precip"), ("NaturalGas", "Chicago_temp")]
rows, rows2 = [], []
for c, w in pairs:
    rows.append({"pair": f"{c} ~ {w}", **{f"price change k={k}": round(anom[w].corr(lp[c].shift(-k)), 2) for k in (0, 1, 2, 3, 6)}})
    fwd3 = lp[c].rolling(3).sum().shift(-3)
    hi, lo = anom[w].quantile(0.9), anom[w].quantile(0.1)
    rows2.append({"pair": f"{c} ~ {w}", "after extreme high": round(fwd3[anom[w] >= hi].mean(), 3),
                  "after extreme low": round(fwd3[anom[w] <= lo].mean(), 3),
                  "all other months": round(fwd3[(anom[w] > lo) & (anom[w] < hi)].mean(), 3)})
print("Weather anomaly in month t vs price change in month t+k:"); display(pd.DataFrame(rows).set_index("pair"))
print("Average price change over the next 3 months after unusual weather:"); display(pd.DataFrame(rows2).set_index("pair"))

# %% Commodities: split, scaling and naive baselines
n_train, n_val, n_test = split_info(cmd)
split_stats(cmd[PRICES], n_train, n_val, n_test)
cmd_scaled = scale_train_only(cmd, n_train)
price_idx = [cmd.columns.get_loc(c) for c in PRICES]
display(naive_baselines(cmd_scaled, n_train, n_val, n_test, cols=price_idx))      # "selected" columns = the 8 prices

# %% [markdown]
# # 4. Exchange-Rate (8 currencies against the US dollar)

# %% Exchange-Rate: load, rename and audit
exh = pd.read_csv("exchange_rate.csv")
if "date" in exh.columns:
    exh["date"] = pd.to_datetime(exh["date"])
    exh = exh.set_index("date")
else:
    exh.index = pd.date_range("1990-01-01", periods=len(exh), freq="D", name="date")      # assumed dates
if list(exh.columns)[:3] == ["0", "1", "2"]:
    exh.columns = ["Australia", "British", "Canada", "Switzerland", "China", "Japan", "New_Zealand", "Singapore"]
audit(exh, "Exchange-Rate")
print("First row:\n", exh.iloc[0])
# The file's dates look nominal (the shape of the series suggests data up to about 2016). Do not quote them as real years.

# %% Exchange-Rate: repeated days and unchanged days
same_as_prev = (exh == exh.shift()).all(axis=1)
pos = np.where(same_as_prev)[0]                                   # row numbers of repeated days
print(same_as_prev.sum(), "rows identical to the previous row")
print("Gaps between repeated days:\n", pd.Series(np.diff(pos)).value_counts().head())
print("Share of unchanged days per currency:\n", (exh.diff() == 0).mean().round(3))

# %% Exchange-Rate: raw plots, distributions and outliers
exh.plot(subplots=True, figsize=(12, 14), lw=0.6, title="Exchange rates vs USD"); plt.tight_layout(); plt.show()
(exh / exh.iloc[0]).plot(figsize=(12, 5), lw=0.8, title="Rebased to 1 at the start"); plt.show()
exh.hist(bins=50, figsize=(12, 8)); plt.tight_layout(); plt.show()
exh.plot(kind="box", subplots=True, layout=(2, 4), figsize=(12, 6), sharex=False, sharey=False); plt.tight_layout(); plt.show()

# %% Exchange-Rate: returns, volatility, seasonality
exh_ret = np.log(exh).diff().dropna()
exh_ret.plot(subplots=True, figsize=(12, 14), lw=0.4, title="Daily log returns"); plt.tight_layout(); plt.show()
display(pd.DataFrame({"skew": exh_ret.skew(), "excess kurtosis": exh_ret.kurt()}).round(2))
exh_ret.rolling(30).std().plot(figsize=(12, 5), lw=0.8, title="30-day rolling volatility"); plt.show()
print("Seasonal strength (0 = none, 1 = very strong); periods are in rows, not real days:")
display(pd.DataFrame({"weekly (7)": {c: round(seasonal_strength(exh[c], 7), 3) for c in exh.columns},
                      "yearly (365)": {c: round(seasonal_strength(exh[c], 365), 3) for c in exh.columns}}))

# %% Exchange-Rate: autocorrelation, FFT and stationarity
col = "Australia"
seasonal_decompose(exh[col], model="additive", period=365, extrapolate_trend="freq").plot(); plt.suptitle(f"{col}: yearly period"); plt.show()
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(exh[col], lags=400, ax=ax[0], title=f"ACF of {col} PRICE (slow decay = trending)")
plot_acf(exh_ret[col], lags=100, ax=ax[1], title=f"ACF of {col} RETURNS (near zero = random-walk-like)")
plt.tight_layout(); plt.show()
per, sp = fft_peaks(exh_ret[col], 2, 730)
plt.semilogx(per, sp); plt.xlabel("period (rows)"); plt.title(f"{col} returns spectrum"); plt.show()
print("Top periods (rows):", np.round(per[np.argsort(sp)[::-1][:5]], 1))
rows = []
for c in exh.columns:
    rows.append({"series": c, **{"price " + k: v for k, v in stationarity(exh[c]).items()},
                 **{"return " + k: v for k, v in stationarity(exh_ret[c]).items()}})
display(pd.DataFrame(rows).set_index("series"))

# %% Exchange-Rate: relationships, split, scaling and naive baselines
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(exh.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title("Correlation of PRICES")
sns.heatmap(exh_ret.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1]); ax[1].set_title("Correlation of RETURNS")
plt.tight_layout(); plt.show()
n_train, n_val, n_test = split_info(exh)
split_stats(exh, n_train, n_val, n_test)
exh_scaled = scale_train_only(exh, n_train)
display(naive_baselines(exh_scaled, n_train, n_val, n_test))

# %% [markdown]
# # 5. Findings (from the run of 19 Nov 2026; numbers are on the train-fitted standard-scaled data)
#
# Autoformer paper numbers (Wu et al., NeurIPS 2021, Table 1 = multivariate, input 96; Table 2 is univariate):
# ETTm2 96/192/336/720 MSE|MAE = 0.255|0.339, 0.281|0.340, 0.339|0.372, 0.422|0.419
# Electricity (all 321 series) = 0.201|0.317, 0.222|0.334, 0.231|0.338, 0.254|0.361
# Exchange = 0.197|0.323, 0.300|0.369, 0.509|0.524, 1.447|0.941
#
# ## ETTm2
# - 69,680 rows of 15-minute data, no missing values, no timestamp gaps. 1,734 rows exactly repeat another row (flat readings); kept.
# - Strong daily cycle: OT autocorrelation 0.93 at lag 96 (1 day) and 0.82 at lag 672 (1 week); the FFT peak is at 24 h, with 12 h next.
#   HUFL, MUFL and OT peak at 24 h; LUFL/LULL are dominated by slow cycles (~56 days).
# - ADF says stationary, KPSS says not, for every variable: the series have a slow trend on top of the cycle,
#   which is exactly the trend + seasonal split that Autoformer's decomposition block assumes.
# - OT is only weakly linked to the loads (MULL 0.50, HULL 0.34, others below 0.2), and lagging the loads by up to 2 days does not
#   raise the correlation, so the loads carry little lead information about OT.
# - Distribution shift: OT mean is 26.9 in train, 36.1 in validation (summer) and 14.4 in test (winter). HULL and MULL means fall from 12.3 / 10.5 to 0.6 / 1.8.
# - Baselines: "repeat the last day" gives MSE|MAE 0.263|0.301 at 96 and 0.471|0.422 at 720. Its MSE is close to Autoformer's 0.255 at 96,
#   and its MAE is lower than Autoformer's at 96, 192 and 336. Any reproduced Autoformer must be compared with this baseline.
#
# ## ECL (10-series subset)
# - 26,304 hourly rows, no missing values, almost no zeros (series 2: 0.6%).
# - Daily and weekly cycles: strongest periods 24 h and 168 h. For OT the weekly autocorrelation (0.84 at lag 168) is above the daily one (0.73 at lag 24).
# - "Repeat the last day" more than halves the error of "last value" (MSE 0.377 vs 0.970 at 96; 0.579 vs 1.145 at 720).
# - Series 2 has large spikes only in training (train std 58.4, test std 1.4), which distorts its scaling.
# - Correlations of 6 to 12 h lags turn negative for series 3-5: the clients peak at different times of day.
# - These results use 10 of 321 clients, so they cannot be compared directly with the paper's Electricity numbers.
#
# ## Commodities
# - Data quality: (1) 22 rows in Jan 2003, then a 333-day gap; the script now starts on 2003-12-01.
#   (2) About 220 rows per year instead of about 252: roughly 30 trading days a year, almost all Fridays, are missing.
#   The prices-only file (commodities/commodities.csv) has every trading day, so the days were lost when weather and FX were merged in.
#   Fix in data preparation: keep the price calendar (left join) and forward-fill weather/FX.
#   (3) WTI is negative on one day (April 2020), so its log return is undefined there. (4) NaturalGas drops from 7.46 to 3.92 on 2026-01-29,
#   which looks like a front-month contract roll rather than a market move; check the source.
# - Prices are non-stationary and returns are stationary. Returns have almost no autocorrelation (lag 1 between -0.09 and 0.00),
#   but absolute returns do (0.06 to 0.27 at lag 1): volatility clusters. Tails are heavy (excess kurtosis 2 to 20).
# - No usable seasonality: seasonal strength is 0.08 or less at weekly, monthly and yearly periods for every commodity.
# - Dollar Index: same-day return correlation from -0.07 (NaturalGas) to -0.32 (Copper); at lags of 1 to 5 days it is about 0, so the dollar does not lead.
# - Weather: all monthly correlations are small (largest |r| = 0.19, Wheat vs Kansas rainfall in the same month). The "after extreme weather" averages
#   rest on about 28 months each and are noisy.
# - Strong shift: test means are far above train means for Coffee (259 vs 136) and Copper (4.5 vs 2.8).
# - Baselines on the 8 prices: "last value" MSE|MAE 0.345|0.391 at 96; "input mean" has lower MSE only at 720.
#
# ## Exchange-Rate
# - 7,588 daily rows with nominal dates. 182 rows exactly repeat the previous row, mostly 7 rows apart (filled weekend/holiday rows).
# - China: the rate is unchanged on 32.6% of days and its return kurtosis is about 1,840, which fits a peg with one large devaluation jump.
# - Prices are non-stationary (ADF p 0.05 to 0.60), returns are stationary; no seasonality (strength 0.08 or less); the return spectrum is flat.
# - "Last value" gives MSE|MAE 0.081|0.196, 0.167|0.289, 0.306|0.398, 0.810|0.676 at 96/192/336/720:
#   better than Autoformer's published numbers at all four horizons.
#   This is the random-walk result also reported by Zeng et al. (2023, "Are Transformers Effective for Time Series Forecasting?").
#
# ## What this means for the project
# - Autoformer's decomposition and Auto-Correlation are built for periodic series: they suit ETTm2 and ECL, not Exchange-Rate or Commodities.
# - A ready-made "own experiment" (assignment option C, baseline comparison): Autoformer vs "repeat last day" / "last value" on ETTm2 and Exchange.
#   Commodities works as a dataset-variation experiment (option D): a series with no cycle to find.
