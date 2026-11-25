"""
Autoformer forecasting demo (Streamlit).

Pipeline: Input (dataset, model, horizon, cutoff) -> Preprocessing (96-step window, train-only scaling, date features)
          -> Trained model (checkpoint from Colab) -> Prediction (H steps) -> Output (charts, metrics, table).

Run:  streamlit run app.py
"""
import glob
import os
import re
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from utils.timefeatures import time_features                     # noqa: E402  (official code)
from models import Autoformer, Autoformer_FullAttn, Autoformer_NoDecomp, Transformer  # noqa: E402

st.set_page_config(page_title="Autoformer forecasting demo", page_icon="📈", layout="wide")
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

# ----------------------------------------------------------------------------------------------------------------
# Configuration (must match training: same split, same scaling, same date features as the official data loader)
# ----------------------------------------------------------------------------------------------------------------
SEQ_LEN, LABEL_LEN = 96, 48
DATASETS = {
    "ETTm2": dict(file="ETTm2.csv", freq="t", step=pd.Timedelta(minutes=15), step_name="15 minutes", season=96,
                  split="ett", desc="Electricity transformer data: 7 variables every 15 minutes. OT = oil temperature (°C)."),
    "Exchange": dict(file="exchange_rate.csv", freq="h", step=pd.Timedelta(days=1), step_name="1 day", season=None,
                     split="ratio", desc="Daily exchange rates of 8 currencies against the US dollar. "
                                         "The dates in this file are placeholders, not real calendar dates."),
}
MODEL_CODE = {"Autoformer": Autoformer, "Autoformer_NoDecomp": Autoformer_NoDecomp,
              "Autoformer_FullAttn": Autoformer_FullAttn, "Transformer": Transformer}
MODEL_LABEL = {"Autoformer": "Autoformer (paper model)", "Autoformer_NoDecomp": "Ablation: no decomposition",
               "Autoformer_FullAttn": "Ablation: standard attention", "Transformer": "Plain Transformer"}
PAPER = {"ETTm2": {96: (0.255, 0.339), 192: (0.281, 0.340), 336: (0.339, 0.372), 720: (0.422, 0.419)},
         "Exchange": {96: (0.197, 0.323), 192: (0.300, 0.369), 336: (0.509, 0.524), 720: (1.447, 0.941)}}
PARENT = os.path.dirname(APP_DIR)
CKPT_DIRS = [os.path.join(APP_DIR, "checkpoints"), os.path.join(PARENT, "runs", "checkpoints"),
             os.path.join(PARENT, "runs_phase3", "checkpoints"), os.path.join(PARENT, "trained_models")]
RESULT_FILES = {"Reproduction (Phase 2)": ["results/comparison.csv", "../runs/comparison.csv"],
                "Experiment MSE (Phase 3)": ["results/table_mse.csv", "../runs_phase3/table_mse.csv"],
                "Experiment MAE (Phase 3)": ["results/table_mae.csv", "../runs_phase3/table_mae.csv"]}
SETTING_RE = re.compile(
    r"^(?P<ds>ETTm2|Exchange)_96_(?P<h>\d+)_(?:.*_)?(?P<model>Autoformer_NoDecomp|Autoformer_FullAttn|Autoformer|Transformer)"
    r"_(?:ETTm2|custom)_ftM_sl(?P<sl>\d+)_ll(?P<ll>\d+)_pl(?P<pl>\d+)_dm(?P<dm>\d+)_nh(?P<nh>\d+)_el(?P<el>\d+)"
    r"_dl(?P<dl>\d+)_df(?P<df>\d+)_fc(?P<fc>\d+)_eb(?P<eb>[a-zA-Z]+)_dt(?:True|False)_(?P<tag>.+)$")


# ----------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def find_checkpoints():
    """Every checkpoint the app can use, read from the run name the training code gave it."""
    found = []
    for d in CKPT_DIRS:
        files = glob.glob(os.path.join(d, "*.pth")) + glob.glob(os.path.join(d, "*", "checkpoint.pth"))
        for f in files:
            name = os.path.basename(f)[:-4] if not f.endswith("checkpoint.pth") else os.path.basename(os.path.dirname(f))
            m = SETTING_RE.match(name)
            if m:
                info = m.groupdict()
                info.update(path=f, setting=name, h=int(info["h"]), quick="quick" in info["tag"])
                found.append(info)
    df = pd.DataFrame(found)
    if len(df):   # one checkpoint per dataset/model/horizon: prefer full runs over quick tests, then run 0
        df = df.sort_values(["quick", "tag"]).drop_duplicates(["ds", "model", "h"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_data(name):
    cfg = DATASETS[name]
    df = pd.read_csv(os.path.join(APP_DIR, "data", cfg["file"]))
    df["date"] = pd.to_datetime(df["date"])
    cols = [c for c in df.columns if c != "date"]
    if "OT" in cols:                                   # the official loader puts the target column last
        cols = [c for c in cols if c != "OT"] + ["OT"]
    n = len(df)
    if cfg["split"] == "ett":                          # 12 / 4 / 4 months of 30 days
        n_train, n_val, n_test = 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
    else:                                              # 70 / 10 / 20 %
        n_train, n_test = int(n * 0.7), int(n * 0.2)
        n_val = n - n_train - n_test
    values = df[cols].values.astype("float64")
    mu, sd = values[:n_train].mean(0), values[:n_train].std(0)   # StandardScaler fitted on the training part only
    return df[["date"] + cols], cols, mu, sd, (n_train, n_val, n_test)


@st.cache_resource(show_spinner=False, max_entries=3)   # at most 3 models in memory (Streamlit Cloud has ~1 GB)
def load_model(path, model, ds, dm, nh, el, dl, d_ff, fc, pl, n_vars):
    cfg = DATASETS[ds]
    args = Namespace(seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=pl, enc_in=n_vars, dec_in=n_vars, c_out=n_vars,
                     d_model=dm, n_heads=nh, e_layers=el, d_layers=dl, d_ff=d_ff, factor=fc, moving_avg=25,
                     dropout=0.05, embed="timeF", freq=cfg["freq"], activation="gelu", output_attention=False, distil=True)
    net = MODEL_CODE[model].Model(args).float()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net, sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------------------------------------------------------
# Inference (same tensors the official exp_main._predict builds)
# ----------------------------------------------------------------------------------------------------------------
def forecast(net, df, cols, mu, sd, i, H, cfg):
    """i = row of the first forecast step. The model sees rows i-96 .. i-1."""
    scaled = (df[cols].values - mu) / sd
    n_have = max(0, min(H, len(df) - i))
    fut_dates = list(df["date"].iloc[i:i + n_have])
    last = df["date"].iloc[-1]
    fut_dates += [last + cfg["step"] * (k + 1) for k in range(H - n_have)]   # beyond the end of the file
    in_dates = pd.DatetimeIndex(df["date"].iloc[i - SEQ_LEN:i])
    dec_dates = pd.DatetimeIndex(list(df["date"].iloc[i - LABEL_LEN:i]) + fut_dates)
    x = scaled[i - SEQ_LEN:i]
    dec = np.concatenate([scaled[i - LABEL_LEN:i], np.zeros((H, len(cols)))])  # 48 known steps + H placeholders
    x_mark = time_features(in_dates, freq=cfg["freq"]).T
    y_mark = time_features(dec_dates, freq=cfg["freq"]).T
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = net(t(x), t(x_mark), t(dec), t(y_mark))
    pred_scaled = out[0].numpy().astype("float64")
    return pd.DatetimeIndex(fut_dates), pred_scaled, pred_scaled * sd + mu, n_have, scaled


def split_name(i, splits):
    n_train, n_val, n_test = splits
    if i < n_train:
        return "training"
    if i < n_train + n_val:
        return "validation"
    if i < n_train + n_val + n_test:
        return "test"
    return "after the evaluated range"


# ----------------------------------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------------------------------
st.title("Autoformer: long-term time series forecasting")
st.caption("Reproduction of Wu et al., *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term "
           "Series Forecasting*, NeurIPS 2021. Models trained in Google Colab and run here on CPU.")

ckpts = find_checkpoints()
if ckpts.empty:
    st.error("No trained models found.")
    st.markdown(
        "Copy your `.pth` files from Colab into **one** of these folders, then refresh the page:\n\n"
        + "\n".join(f"- `{d}`" for d in CKPT_DIRS)
        + "\n\nThe file names must stay as Colab saved them, for example "
          "`ETTm2_96_96_Autoformer_ETTm2_ftM_sl96_ll48_pl96_dm512_..._Exp_run0_0.pth`: the app reads the model settings from them.")
    st.stop()

# ---- Sidebar: the user's input ---------------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Input")
    ds = st.selectbox("Dataset", [d for d in DATASETS if d in set(ckpts.ds)])
    cfg = DATASETS[ds]
    st.caption(cfg["desc"])
    models_here = [m for m in MODEL_CODE if m in set(ckpts[ckpts.ds == ds].model)]
    model = st.selectbox("Model", models_here, format_func=lambda m: MODEL_LABEL[m])
    hs = sorted(ckpts[(ckpts.ds == ds) & (ckpts.model == model)].h)
    H = st.selectbox("Forecast horizon", hs, format_func=lambda h: f"{h} steps ({h} × {cfg['step_name']})")

    df, cols, mu, sd, splits = load_data(ds)
    n_train, n_val, n_test = splits
    test_start = n_train + n_val
    n_windows = n_test - H + 1

    st.subheader("Forecast start (cutoff)")
    how = st.radio("Choose by", ["Test window", "Date"], horizontal=True,
                   help="Test windows are the period the model never saw during training, the same windows used for the scores.")
    if how == "Test window":
        key = f"win_{ds}_{H}"
        if key not in st.session_state:
            st.session_state[key] = min(2000, n_windows - 1)
        if st.button("🎲 Random test window", width="stretch"):
            st.session_state[key] = int(np.random.randint(0, n_windows))
        k = st.slider("Test window number", 0, n_windows - 1, key=key)
        i = test_start + k
    else:
        first_ok, last_ok = df["date"].iloc[SEQ_LEN], df["date"].iloc[-1] + cfg["step"]
        default = df["date"].iloc[test_start]
        d = st.date_input("Date", value=default.date(), min_value=first_ok.date(), max_value=last_ok.date())
        tm = st.time_input("Time", value=default.time(), step=900) if cfg["freq"] == "t" else default.time()
        when = pd.Timestamp.combine(d, tm)
        i = int(np.clip(df["date"].searchsorted(when), SEQ_LEN, len(df)))
    var = st.selectbox("Variable to plot", cols, index=len(cols) - 1)
    show_base = st.checkbox("Show naive baselines", value=True)

row = ckpts[(ckpts.ds == ds) & (ckpts.model == model) & (ckpts.h == H)].iloc[0]
net, n_params = load_model(row.path, model, ds, int(row.dm), int(row.nh), int(row.el), int(row.dl), int(row["df"]),
                           int(row.fc), int(row.pl), len(cols))
fut_dates, pred_s, pred, n_have, scaled = forecast(net, df, cols, mu, sd, i, H, cfg)
v = cols.index(var)
hist = df.iloc[i - SEQ_LEN:i]
actual = df[cols].values[i:i + n_have]
actual_s = scaled[i:i + n_have]
last_val = df[cols].values[i - 1]
base = {"last value": np.tile(last_val, (H, 1))}
if cfg["season"]:
    s = cfg["season"]
    base["repeat last day"] = np.tile(df[cols].values[i - s:i], (int(np.ceil(H / s)), 1))[:H]

# ---- Pipeline strip ----------------------------------------------------------------------------------------------
where = split_name(i, splits)
steps = [("1. Input", f"{ds}, cutoff {df['date'].iloc[i - 1] + cfg['step']:%Y-%m-%d %H:%M}"),
         ("2. Preprocessing", f"last {SEQ_LEN} steps, scaled with training mean/std, date features"),
         ("3. Trained model", f"{MODEL_LABEL[model]}, {n_params:,} parameters"),
         ("4. Prediction", f"{H} steps × {len(cols)} variables"),
         ("5. Output", "charts, error scores, table")]
for c, (title, text) in zip(st.columns(5), steps):
    with c.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)

if where == "training":
    st.warning("This cutoff is inside the **training** period: the model learned from these values, so the forecast "
               "will look better than on unseen data. Choose a test window for an honest view.")
elif where == "validation":
    st.info("This cutoff is in the **validation** period (used for early stopping, not for learning the weights).")
elif where == "test":
    st.success("This cutoff is in the **test** period: data the model never saw during training.")
if n_have < H:
    st.info(f"Only {n_have} of the {H} forecast steps have actual values in the file: the rest is a true forecast "
            "beyond the end of the data, so no error can be computed for it.")

tab_f, tab_cmp, tab_all, tab_tbl, tab_model, tab_how = st.tabs(
    ["📈 Forecast", "⚖️ Compare models", "🗂️ All variables", "📋 Forecast table", "🧠 Model & results", "⚙️ How it works"])

# ---- Forecast tab -------------------------------------------------------------------------------------------------
with tab_f:
    if n_have == H:
        err_s = pred_s - actual_s
        mse, mae = float(np.mean(err_s ** 2)), float(np.mean(np.abs(err_s)))
        lv_s = (base["last value"] - mu) / sd - actual_s
        lv_mse = float(np.mean(lv_s ** 2))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE, this window (scaled)", f"{mse:.3f}", f"{mse - lv_mse:+.3f} vs last value",
                  delta_color="inverse", help="Same metric as the paper: all variables, standardised.")
        c2.metric("MAE, this window (scaled)", f"{mae:.3f}")
        c3.metric(f"MAE for {var} (original units)", f"{np.mean(np.abs(pred[:, v] - actual[:, v])):.3f}")
        tst = PAPER.get(ds, {}).get(H)
        c4.metric("Paper MSE (whole test set)", f"{tst[0]:.3f}" if tst else "n/a",
                  help="One window is noisy; the paper's score averages thousands of windows.")
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist[var], name=f"Input ({SEQ_LEN} steps the model saw)", line=dict(color="gray"))
    if n_have:
        fig.add_scatter(x=fut_dates[:n_have], y=actual[:, v], name="Actual", line=dict(color="#1f77b4", width=2))
    fig.add_scatter(x=fut_dates, y=pred[:, v], name=MODEL_LABEL[model], line=dict(color="#ff7f0e", width=2))
    if show_base:
        for (bname, b), dash in zip(base.items(), ["dot", "dash"]):
            fig.add_scatter(x=fut_dates, y=b[:, v], name=bname, line=dict(dash=dash, width=1))
    fig.add_vline(x=fut_dates[0], line=dict(color="black", dash="dash", width=1))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08),
                      title=f"{ds} · {var} · {H}-step forecast", yaxis_title=var, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    if n_have == H:
        e = pd.DataFrame({"steps ahead": np.arange(1, H + 1), MODEL_LABEL[model]: np.abs(pred[:, v] - actual[:, v])})
        for bname, b in base.items():
            e[bname] = np.abs(b[:, v] - actual[:, v])
        fig2 = go.Figure([go.Scatter(x=e["steps ahead"], y=e[c], name=c) for c in e.columns[1:]])
        fig2.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10), title=f"Absolute error by step ahead ({var})",
                           xaxis_title="steps ahead", yaxis_title="|error|")
        st.plotly_chart(fig2, width="stretch")

# ---- Compare models tab (Phase 3: ablation + baselines on the same window) ------------------------------------------
with tab_cmp:
    same = ckpts[(ckpts.ds == ds) & (ckpts.h == H)]
    order = [m for m in MODEL_CODE if m in set(same.model)]
    if len(order) < 2:
        st.info("Only one trained model is available for this dataset and horizon. After Phase 3, put the "
                "`runs_phase3` folder from Google Drive next to `runs` in the Project folder to compare "
                "Autoformer with the ablation models and the plain Transformer here.")
    st.markdown("**Question of our experiment:** which of Autoformer's two ideas reduces the error: "
                "*series decomposition* or *Auto-Correlation*? Every model below sees the same 96 input steps.")
    colors = {"Autoformer": "#ff7f0e", "Autoformer_NoDecomp": "#d62728", "Autoformer_FullAttn": "#9467bd",
              "Transformer": "#8c564b"}
    st.markdown(f"**{ds} · {var} · all models on the same window**")
    figc = go.Figure()
    figc.add_scatter(x=hist["date"], y=hist[var], name="Input", line=dict(color="gray"))
    if n_have:
        figc.add_scatter(x=fut_dates[:n_have], y=actual[:, v], name="Actual", line=dict(color="#1f77b4", width=3))
    rows_cmp = []
    for m in order:
        r = same[same.model == m].iloc[0]
        net_m, np_m = load_model(r.path, m, ds, int(r.dm), int(r.nh), int(r.el), int(r.dl), int(r["df"]),
                                 int(r.fc), int(r.pl), len(cols))
        _, ps_m, p_m, _, _ = forecast(net_m, df, cols, mu, sd, i, H, cfg)
        figc.add_scatter(x=fut_dates, y=p_m[:, v], name=MODEL_LABEL[m], line=dict(color=colors[m], width=2))
        if n_have == H:
            rows_cmp.append({"model": MODEL_LABEL[m], "decomposition": "no" if m in ("Autoformer_NoDecomp", "Transformer") else "yes",
                             "Auto-Correlation": "yes" if m in ("Autoformer", "Autoformer_NoDecomp") else "no",
                             "parameters": f"{np_m:,}", "MSE (scaled)": np.mean((ps_m - actual_s) ** 2),
                             "MAE (scaled)": np.mean(np.abs(ps_m - actual_s)),
                             f"MAE {var} (original)": np.mean(np.abs(p_m[:, v] - actual[:, v]))})
    if show_base:
        for (bname, b), dash in zip(base.items(), ["dot", "dash"]):
            figc.add_scatter(x=fut_dates, y=b[:, v], name=bname, line=dict(dash=dash, width=1))
            if n_have == H:
                bs = (b - mu) / sd
                rows_cmp.append({"model": f"Baseline: {bname}", "decomposition": "-", "Auto-Correlation": "-",
                                 "parameters": "0", "MSE (scaled)": np.mean((bs - actual_s) ** 2),
                                 "MAE (scaled)": np.mean(np.abs(bs - actual_s)),
                                 f"MAE {var} (original)": np.mean(np.abs(b[:, v] - actual[:, v]))})
    figc.add_vline(x=fut_dates[0], line=dict(color="black", dash="dash", width=1))
    figc.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified")
    st.plotly_chart(figc, width="stretch")
    if rows_cmp:
        tbl = pd.DataFrame(rows_cmp).sort_values("MSE (scaled)").reset_index(drop=True)
        st.markdown("**This window** (lower is better; sorted by MSE)")
        st.dataframe(tbl.style.format({"MSE (scaled)": "{:.3f}", "MAE (scaled)": "{:.3f}", f"MAE {var} (original)": "{:.3f}"})
                     .highlight_min(subset=["MSE (scaled)", "MAE (scaled)"], color="#d4edda"),
                     width="stretch", hide_index=True)
        st.caption("One window can disagree with the overall result. The whole-test-set comparison is below "
                   "(and in the Model & results tab).")
    for p in ["results/table_mse.csv", "../runs_phase3/table_mse.csv"]:
        full = os.path.normpath(os.path.join(APP_DIR, p))
        if os.path.exists(full):
            st.markdown("**Whole test set, MSE by horizon (Phase 3)**")
            st.dataframe(pd.read_csv(full).round(3), width="stretch", hide_index=True)
            break
    for p in ["results/mse_by_horizon.png", "../runs_phase3/mse_by_horizon.png"]:
        full = os.path.normpath(os.path.join(APP_DIR, p))
        if os.path.exists(full):
            st.image(full, caption="Test MSE by horizon for every model (Phase 3, cell 11)")
            break

# ---- All variables tab ----------------------------------------------------------------------------------------------
with tab_all:
    ncol = 2
    nrow = int(np.ceil(len(cols) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=cols, vertical_spacing=0.08)
    for j, c in enumerate(cols):
        r, cc = j // ncol + 1, j % ncol + 1
        fig.add_scatter(x=hist["date"], y=hist[c], line=dict(color="gray", width=1), showlegend=j == 0, name="Input", row=r, col=cc)
        if n_have:
            fig.add_scatter(x=fut_dates[:n_have], y=actual[:, j], line=dict(color="#1f77b4", width=1.5),
                            showlegend=j == 0, name="Actual", row=r, col=cc)
        fig.add_scatter(x=fut_dates, y=pred[:, j], line=dict(color="#ff7f0e", width=1.5), showlegend=j == 0,
                        name="Forecast", row=r, col=cc)
    fig.update_layout(height=260 * nrow, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.03))
    st.plotly_chart(fig, width="stretch")
    if n_have == H:
        per_var = pd.DataFrame({"MAE (original units)": np.abs(pred - actual).mean(0),
                                "MAE (scaled)": np.abs(pred_s - actual_s).mean(0),
                                "last value MAE (original units)": np.abs(base["last value"] - actual).mean(0)}, index=cols)
        st.dataframe(per_var.round(4), width="stretch")

# ---- Table tab ------------------------------------------------------------------------------------------------------
with tab_tbl:
    out = pd.DataFrame(pred, columns=[f"{c} forecast" for c in cols])
    out.insert(0, "date", fut_dates)
    out.insert(1, "steps ahead", np.arange(1, H + 1))
    if n_have:
        act = pd.DataFrame(np.vstack([actual, np.full((H - n_have, len(cols)), np.nan)]), columns=[f"{c} actual" for c in cols])
        out = pd.concat([out, act], axis=1)
    st.dataframe(out, width="stretch", height=420)
    st.download_button("⬇️ Download forecast (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"forecast_{ds}_{model}_{H}_{fut_dates[0]:%Y%m%d_%H%M}.csv", mime="text/csv")

# ---- Model & results tab ----------------------------------------------------------------------------------------------
with tab_model:
    a, b = st.columns([1, 1])
    with a:
        st.subheader("This model")
        st.table(pd.DataFrame({"value": [MODEL_LABEL[model], f"{n_params:,}", row.dm, row.nh, row.el, row.dl, row["df"],
                                         row.fc, f"{SEQ_LEN} in → {H} out", row.tag, os.path.basename(row.path)]},
                              index=["model", "parameters", "d_model", "attention heads", "encoder layers",
                                     "decoder layers", "feed-forward size", "factor (top-k = factor·ln L)",
                                     "input → output", "run", "checkpoint file"]).astype(str))
        if row.quick:
            st.warning("This checkpoint is from a QUICK test run (tiny model, 1 epoch), not a real trained model.")
        st.subheader("Models available")
        st.dataframe(ckpts[["ds", "model", "h", "dm", "tag"]].rename(columns={"ds": "dataset", "h": "horizon"}),
                     width="stretch", hide_index=True)
    with b:
        st.subheader("Paper results (Table 1, multivariate)")
        st.dataframe(pd.DataFrame([{"dataset": d, "horizon": h, "MSE": m, "MAE": e}
                                   for d, r in PAPER.items() for h, (m, e) in r.items()]),
                     width="stretch", hide_index=True)
        for title, options in RESULT_FILES.items():
            for p in options:
                full = os.path.normpath(os.path.join(APP_DIR, p))
                if os.path.exists(full):
                    st.subheader(f"Our results: {title}")
                    st.dataframe(pd.read_csv(full), width="stretch", hide_index=True)
                    break
        st.caption("Put `comparison.csv` (Phase 2) and `table_mse.csv` / `table_mae.csv` (Phase 3) in the app's `results` "
                   "folder to show your own result tables here.")

# ---- How it works tab -----------------------------------------------------------------------------------------------------
with tab_how:
    st.markdown(f"""
**1. Input.** You chose **{ds}**, **{MODEL_LABEL[model]}**, horizon **{H}** steps and a cutoff. The model gets the
{SEQ_LEN} rows before the cutoff ({hist['date'].iloc[0]:%Y-%m-%d %H:%M} to {hist['date'].iloc[-1]:%Y-%m-%d %H:%M}), all {len(cols)} variables.

**2. Preprocessing**, exactly as in training:
- *Scaling:* each variable becomes (value − mean) / std, with the mean and std of the **training period only**
  ({n_train:,} rows). For `{var}`: mean = {mu[v]:.3f}, std = {sd[v]:.3f}.
- *Date features:* every time step also gets its {'minute, hour, ' if cfg['freq'] == 't' else 'hour, '}weekday, day of month and
  day of year, each rescaled to −0.5…0.5, so the model knows *when* each value happened.
- *Decoder input:* the last {LABEL_LEN} known steps followed by {H} zeros (placeholders for the future).

**3. Trained model.** The checkpoint `{os.path.basename(row.path)}` is loaded on CPU. Inside Autoformer:
- *Series decomposition* splits the input into a **trend** (moving average over 25 steps) and a **seasonal** part (input − trend).
  The decoder starts its trend from the input's level and its seasonal part from zeros, and refines both layer by layer.
- *Auto-Correlation* finds the strongest repeating lags with an FFT (top {int(row.fc)}·ln L of them) and combines whole shifted
  copies of the series at those lags, instead of attending to single time points.

**4. Prediction.** The model outputs a {H} × {len(cols)} block of scaled values.

**5. Output.** The values are converted back to original units (× std + mean) and compared with the actual data when it exists.
Errors are reported on the scaled values (the paper's metric) and in original units. The naive baselines
(*last value* and, for ETTm2, *repeat last day*) show what a trivial rule achieves on the same window.
""")
    st.caption("Model and data code: official thuml/Autoformer repository (with our compatibility fixes). "
               "Ablation models, experiment code and this app: our team.")
