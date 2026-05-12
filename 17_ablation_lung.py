import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Dropout
from tensorflow.keras import Model

import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

# Configuration — best hyperparameters from 15_model_lung.py
BEST_LSTM_UNITS    = 64
BEST_LEARNING_RATE = 1e-4
BEST_DROPOUT_RATE  = 0.1

LAG_YEARS   = 10
N_FOLDS     = 5
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SLM_R2   = 0.474
SLM_RMSE = 17.88

DNN_FEATURES = [
    "smoking_pct",
    "obesity_pct",
    "unemployment_pct",
    "child_poverty_pct",
    "median_income",
    "uninsured_adults_pct",
    "rural_pct",
    "pct_black",
    "pct_hispanic",
]

LSTM_FEATURES = ["pm25_satellite", "no2_satellite_ppb"]

# Step 1: Load data
df_main = pd.read_csv("data/processed/complete_data_original.csv")
df_acag = pd.read_csv("data/processed/acag_county_pm25.csv")
df_no2  = pd.read_csv("data/processed/no2_satellite_county.csv")

df_poll = df_acag.merge(df_no2, on=["county_fips", "year"], how="outer")
df_poll.columns = ["county_fips", "year", "pm25_satellite", "no2_satellite_ppb"]
df_poll = df_poll.set_index(["county_fips", "year"])

df_lung = df_main[df_main["mortality_lung_per100k"].notna()].copy()
print(f"  Lung cancer rows: {len(df_lung)}")

# Step 2: Build LSTM sequences (same logic as scripts 15 and 16)
lstm_sequences = []
valid_indices  = []

for idx, row in df_lung.iterrows():
    fips        = int(row["county_fips"])
    cancer_year = int(row["year"])
    poll_years  = list(range(cancer_year - LAG_YEARS, cancer_year))

    seq = []
    for py in poll_years:
        try:
            row_data = df_poll.loc[(fips, py)]
            pm25     = row_data["pm25_satellite"]
            no2      = row_data["no2_satellite_ppb"]
        except KeyError:
            pm25, no2 = np.nan, np.nan
        seq.append([pm25, no2])

    lstm_sequences.append(seq)
    valid_indices.append(idx)

lstm_array = np.array(lstm_sequences, dtype="float32")

for feature_idx in range(lstm_array.shape[2]):
    col_mean = np.nanmean(lstm_array[:, :, feature_idx])
    mask     = np.isnan(lstm_array[:, :, feature_idx])
    lstm_array[:, :, feature_idx][mask] = col_mean

df_lung_valid = df_lung.loc[valid_indices].reset_index(drop=True)
X_dnn_raw     = df_lung_valid[DNN_FEATURES].values.astype("float32")
y             = df_lung_valid["mortality_lung_per100k"].values.astype("float32")

imputer   = SimpleImputer(strategy="median")
X_dnn_imp = imputer.fit_transform(X_dnn_raw)

print(f"  LSTM array shape:    {lstm_array.shape}")
print(f"  DNN feature shape:   {X_dnn_imp.shape}")
print(f"  Target mean: {y.mean():.2f}  std: {y.std():.2f}")

# Step 3: Define three model variants
# Each uses the same best hyperparameters; only the input branches differ

def build_dnn_only(dnn_shape):
    # DNN-only: socioeconomic features alone, no pollution time series
    dnn_input = Input(shape=(dnn_shape,), name="socioeconomic")
    dnn_out   = Dense(BEST_LSTM_UNITS * 2, activation="relu")(dnn_input)
    dnn_out   = Dense(BEST_LSTM_UNITS,     activation="relu")(dnn_out)
    dnn_out   = Dense(BEST_LSTM_UNITS // 2, activation="relu")(dnn_out)
    dnn_out   = Dropout(BEST_DROPOUT_RATE)(dnn_out)
    output    = Dense(1, activation="linear", name="lung_mortality")(dnn_out)

    model = Model(inputs=dnn_input, outputs=output)
    model.compile(
        loss      = "mse",
        optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_LEARNING_RATE),
        metrics   = ["mae"]
    )
    return model


def build_lstm_only(lstm_shape):
    # LSTM-only: pollution time series alone, no socioeconomic features
    lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
    lstm_out   = LSTM(BEST_LSTM_UNITS, return_sequences=False)(lstm_input)
    lstm_out   = Dense(BEST_LSTM_UNITS // 2, activation="relu")(lstm_out)
    lstm_out   = Dropout(BEST_DROPOUT_RATE)(lstm_out)
    output     = Dense(1, activation="linear", name="lung_mortality")(lstm_out)

    model = Model(inputs=lstm_input, outputs=output)
    model.compile(
        loss      = "mse",
        optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_LEARNING_RATE),
        metrics   = ["mae"]
    )
    return model


def build_full_model(lstm_shape, dnn_shape):
    # Full model: both branches merged (identical to best model from script 15)
    lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
    lstm_out   = LSTM(BEST_LSTM_UNITS, return_sequences=False)(lstm_input)
    lstm_out   = Dense(BEST_LSTM_UNITS // 2, activation="relu")(lstm_out)

    dnn_input  = Input(shape=(dnn_shape,), name="socioeconomic")
    dnn_out    = Dense(BEST_LSTM_UNITS * 2, activation="relu")(dnn_input)
    dnn_out    = Dense(BEST_LSTM_UNITS,     activation="relu")(dnn_out)

    merged     = Concatenate()([lstm_out, dnn_out])
    merged     = Dense(BEST_LSTM_UNITS, activation="relu")(merged)
    merged     = Dropout(BEST_DROPOUT_RATE)(merged)
    output     = Dense(1, activation="linear", name="lung_mortality")(merged)

    model = Model(inputs=[lstm_input, dnn_input], outputs=output)
    model.compile(
        loss      = "mse",
        optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_LEARNING_RATE),
        metrics   = ["mae"]
    )
    return model


# Step 4: 5-fold CV for all three variants
# Each fold uses the same splits so results are directly comparable

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

results = {
    "dnn_only":   {"r2": [], "rmse": []},
    "lstm_only":  {"r2": [], "rmse": []},
    "full_model": {"r2": [], "rmse": []},
}

for fold, (train_idx, test_idx) in enumerate(kf.split(y)):

    print(f"\nFold {fold + 1}/{N_FOLDS}")

    X_lstm_train = lstm_array[train_idx]
    X_lstm_test  = lstm_array[test_idx]
    X_dnn_train  = X_dnn_imp[train_idx]
    X_dnn_test   = X_dnn_imp[test_idx]
    y_train      = y[train_idx]
    y_test       = y[test_idx]

    # Scale DNN features
    scaler_dnn  = StandardScaler()
    X_dnn_train = scaler_dnn.fit_transform(X_dnn_train)
    X_dnn_test  = scaler_dnn.transform(X_dnn_test)

    # Scale LSTM sequences
    scaler_lstm  = StandardScaler()
    n_samples, n_steps, n_feats = X_lstm_train.shape
    X_lstm_train = scaler_lstm.fit_transform(
        X_lstm_train.reshape(-1, n_feats)
    ).reshape(n_samples, n_steps, n_feats)
    X_lstm_test  = scaler_lstm.transform(
        X_lstm_test.reshape(-1, n_feats)
    ).reshape(-1, n_steps, n_feats)

    # Scale target
    scaler_y  = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor              = "val_loss",
        patience             = 15,
        restore_best_weights = True
    )

    # Variant A: DNN-only
    model_dnn = build_dnn_only(dnn_shape=X_dnn_imp.shape[1])
    model_dnn.fit(
        X_dnn_train, y_train_s,
        epochs=200, batch_size=64, validation_split=0.1,
        shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )]
    )
    y_pred_s = model_dnn.predict(X_dnn_test, verbose=0).flatten()
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    results["dnn_only"]["r2"].append(r2_score(y_test, y_pred))
    results["dnn_only"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"  DNN-only:   R²={results['dnn_only']['r2'][-1]:.4f}  RMSE={results['dnn_only']['rmse'][-1]:.2f}")

    # Variant B: LSTM-only
    model_lstm = build_lstm_only(lstm_shape=(LAG_YEARS, len(LSTM_FEATURES)))
    model_lstm.fit(
        X_lstm_train, y_train_s,
        epochs=200, batch_size=64, validation_split=0.1,
        shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )]
    )
    y_pred_s = model_lstm.predict(X_lstm_test, verbose=0).flatten()
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    results["lstm_only"]["r2"].append(r2_score(y_test, y_pred))
    results["lstm_only"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"  LSTM-only:  R²={results['lstm_only']['r2'][-1]:.4f}  RMSE={results['lstm_only']['rmse'][-1]:.2f}")

    # Variant C: Full LSTM+DNN model
    model_full = build_full_model(
        lstm_shape=(LAG_YEARS, len(LSTM_FEATURES)),
        dnn_shape=X_dnn_imp.shape[1]
    )
    model_full.fit(
        [X_lstm_train, X_dnn_train], y_train_s,
        epochs=200, batch_size=64, validation_split=0.1,
        shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )]
    )
    y_pred_s = model_full.predict([X_lstm_test, X_dnn_test], verbose=0).flatten()
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    results["full_model"]["r2"].append(r2_score(y_test, y_pred))
    results["full_model"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"  Full model: R²={results['full_model']['r2'][-1]:.4f}  RMSE={results['full_model']['rmse'][-1]:.2f}")

# Step 5: Summarise results
summary = []
for variant, label in [
    ("dnn_only",   "DNN-only (socioeconomic)"),
    ("lstm_only",  "LSTM-only (pollution)"),
    ("full_model", "LSTM+DNN (full)"),
]:
    mean_r2   = np.mean(results[variant]["r2"])
    mean_rmse = np.mean(results[variant]["rmse"])
    summary.append({
        "variant":    variant,
        "label":      label,
        "mean_r2":    round(mean_r2,   4),
        "mean_rmse":  round(mean_rmse, 2),
        "vs_SLM_r2":  round(mean_r2 - SLM_R2, 4),
    })

df_summary = pd.DataFrame(summary)
print(f"\nAblation results (lung cancer):")
print(df_summary[["label", "mean_r2", "mean_rmse", "vs_SLM_r2"]].to_string(index=False))
print(f"\nSLM baseline: R²={SLM_R2}  RMSE={SLM_RMSE}")

df_summary.to_csv("data/processed/ablation_results.csv", index=False)
print(f"Saved: data/processed/ablation_results.csv")

# Step 6: Ablation bar chart — R² across all variants plus SLM baseline
labels    = [r["label"] for r in summary] + ["SLM baseline"]
r2_vals   = [r["mean_r2"] for r in summary] + [SLM_R2]
rmse_vals = [r["mean_rmse"] for r in summary] + [SLM_RMSE]
colors    = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Ablation Study — Lung Cancer (what drives model performance?)", fontsize=13)

# Plot 1: R²
bars = axes[0].bar(labels, r2_vals, color=colors)
axes[0].set_ylabel("Mean R² (5-fold CV)")
axes[0].set_title("R² — higher is better")
axes[0].set_ylim(0, max(r2_vals) * 1.15)
axes[0].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
axes[0].tick_params(axis="x", labelsize=9)

# Plot 2: RMSE
bars = axes[1].bar(labels, rmse_vals, color=colors)
axes[1].set_ylabel("Mean RMSE (per 100k, 5-fold CV)")
axes[1].set_title("RMSE — lower is better")
axes[1].set_ylim(0, max(rmse_vals) * 1.15)
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, rmse_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10)
axes[1].tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/13_ablation_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/13_ablation_results.png")

# Step 7: Incremental gain chart
# Shows R² added by each component above the DNN-only baseline
dnn_r2  = df_summary.loc[df_summary["variant"] == "dnn_only",   "mean_r2"].values[0]
full_r2 = df_summary.loc[df_summary["variant"] == "full_model", "mean_r2"].values[0]
lstm_r2 = df_summary.loc[df_summary["variant"] == "lstm_only",  "mean_r2"].values[0]

# Incremental gain of adding LSTM to DNN vs using DNN alone
lstm_gain = full_r2 - dnn_r2

gain_labels = ["Socioeconomic\n(DNN base)", "LSTM gain\n(full − DNN)", "SLM baseline"]
gain_vals   = [dnn_r2, lstm_gain, SLM_R2]
gain_colors = ["#4C72B0", "#55A868", "#DD8452"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(gain_labels, gain_vals, color=gain_colors)
ax.set_ylabel("R²")
ax.set_title("Incremental R² contribution of LSTM pollution branch\n(lung cancer, 5-fold CV)")
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, gain_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:+.3f}" if bar == bars[1] else f"{val:.3f}",
            ha="center", va="bottom", fontsize=11)

plt.tight_layout()
plt.savefig("outputs/figures/13b_lstm_incremental_gain.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/13b_lstm_incremental_gain.png")