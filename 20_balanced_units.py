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

# Fixed hyperparameters (best from script 15)
BEST_LEARNING_RATE = 1e-4
BEST_DROPOUT_RATE  = 0.1
LAG_YEARS          = 10
N_FOLDS            = 5
RANDOM_SEED        = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SLM_R2   = 0.474
SLM_RMSE = 17.88

# Original asymmetric model baseline (script 15 best result)
ASYMMETRIC_R2   = 0.514
ASYMMETRIC_RMSE = 17.31

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

# Unit configurations to test:
#   "asymmetric" (original script 15): LSTM=64, DNN first layer=128
#   "balanced_32":  LSTM=32,  DNN first layer=32
#   "balanced_64":  LSTM=64,  DNN first layer=64
#   "balanced_128": LSTM=128, DNN first layer=128
#
# In all balanced configs the DNN second layer = units,
# LSTM post-dense = units // 2, merged Dense = units.
# This keeps the post-branch output sizes proportional
# so the Concatenate merge is always fair.
CONFIGS = [
    {"label": "asymmetric (original)", "lstm_units": 64,  "dnn_first": 128},
    {"label": "balanced_32",           "lstm_units": 32,  "dnn_first": 32 },
    {"label": "balanced_64",           "lstm_units": 64,  "dnn_first": 64 },
    {"label": "balanced_128",          "lstm_units": 128, "dnn_first": 128},
]

# Step 1: Load data
df_main = pd.read_csv("data/processed/complete_data_original.csv")
df_acag = pd.read_csv("data/processed/acag_county_pm25.csv")
df_no2  = pd.read_csv("data/processed/no2_satellite_county.csv")

df_poll = df_acag.merge(df_no2, on=["county_fips", "year"], how="outer")
df_poll.columns = ["county_fips", "year", "pm25_satellite", "no2_satellite_ppb"]
df_poll = df_poll.set_index(["county_fips", "year"])

df_lung = df_main[df_main["mortality_lung_per100k"].notna()].copy()
print(f"  Main dataset:   {df_main.shape}")
print(f"  Lung cancer rows: {len(df_lung)}")

# Step 2: Build LSTM sequences (identical logic to scripts 15-19)
lstm_sequences = []
valid_indices  = []

for idx, row in df_lung.iterrows():
    fips        = row["county_fips"]
    target_year = int(row["year"])
    seq         = []
    for lag in range(LAG_YEARS, 0, -1):
        yr = target_year - lag
        try:
            vals = df_poll.loc[(fips, yr), LSTM_FEATURES]
            pm25 = float(vals["pm25_satellite"])
            no2  = float(vals["no2_satellite_ppb"])
        except KeyError:
            pm25, no2 = np.nan, np.nan
        seq.append([pm25, no2])
    lstm_sequences.append(seq)
    valid_indices.append(idx)

lstm_array = np.array(lstm_sequences, dtype="float32")

# Mean-impute missing values within each feature channel
for feat_idx in range(lstm_array.shape[2]):
    col_mean = np.nanmean(lstm_array[:, :, feat_idx])
    mask     = np.isnan(lstm_array[:, :, feat_idx])
    lstm_array[:, :, feat_idx][mask] = col_mean

df_lung_valid = df_lung.loc[valid_indices].reset_index(drop=True)
X_dnn_raw     = df_lung_valid[DNN_FEATURES].values.astype("float32")
y             = df_lung_valid["mortality_lung_per100k"].values.astype("float32")

imputer   = SimpleImputer(strategy="median")
X_dnn_imp = imputer.fit_transform(X_dnn_raw)

print(f"  LSTM array shape:  {lstm_array.shape}")
print(f"  DNN feature shape: {X_dnn_imp.shape}")
print(f"  Target mean: {y.mean():.2f}  std: {y.std():.2f}")

# Step 3: Model builder — parameterised by lstm_units and dnn_first
# Architecture for each config:
#   LSTM branch: Input -> LSTM(lstm_units) -> Dense(lstm_units // 2, relu)
#   DNN branch:  Input -> Dense(dnn_first, relu) -> Dense(dnn_first // 2, relu)
#   Merge:       Concatenate -> Dense(lstm_units, relu) -> Dropout -> Dense(1)
#
# In the asymmetric original: lstm_units=64, dnn_first=128
#   LSTM output size: 32  (lstm_units // 2)
#   DNN output size:  64  (dnn_first // 2)
#   Concat size:      96
#
# In balanced configs: lstm_units == dnn_first
#   Both output sizes equal, so neither branch has a size advantage at merge.
def build_model(lstm_shape, dnn_shape, lstm_units, dnn_first):
    lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
    lstm_out   = LSTM(lstm_units, return_sequences=False)(lstm_input)
    lstm_out   = Dense(lstm_units // 2, activation="relu")(lstm_out)

    dnn_input  = Input(shape=(dnn_shape,), name="socioeconomic")
    dnn_out    = Dense(dnn_first,     activation="relu")(dnn_input)
    dnn_out    = Dense(dnn_first // 2, activation="relu")(dnn_out)

    merged     = Concatenate()([lstm_out, dnn_out])
    merged     = Dense(lstm_units, activation="relu")(merged)
    merged     = Dropout(BEST_DROPOUT_RATE)(merged)
    output     = Dense(1, activation="linear", name="lung_mortality")(merged)

    model = Model(inputs=[lstm_input, dnn_input], outputs=output)
    model.compile(
        loss      = "mse",
        optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_LEARNING_RATE),
        metrics   = ["mae"]
    )
    return model

# Step 4: 5-fold CV for each configuration
# The same KFold object ensures all configs are tested on identical splits,
# so R² differences reflect architecture alone, not data partitioning luck.
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
splits = list(kf.split(y))

all_results = []

for cfg in CONFIGS:
    label      = cfg["label"]
    lstm_units = cfg["lstm_units"]
    dnn_first  = cfg["dnn_first"]

    print(f"\nTesting: {label}  (lstm={lstm_units}, dnn_first={dnn_first})")

    fold_r2, fold_rmse = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_lstm_train = lstm_array[train_idx]
        X_lstm_test  = lstm_array[test_idx]
        X_dnn_train  = X_dnn_imp[train_idx]
        X_dnn_test   = X_dnn_imp[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]

        # Scale DNN features per fold (fit on train only)
        scaler_dnn   = StandardScaler()
        X_dnn_train  = scaler_dnn.fit_transform(X_dnn_train)
        X_dnn_test   = scaler_dnn.transform(X_dnn_test)

        # Scale LSTM sequences per fold
        n_samples, n_steps, n_feats = X_lstm_train.shape
        scaler_lstm  = StandardScaler()
        X_lstm_train = scaler_lstm.fit_transform(
            X_lstm_train.reshape(-1, n_feats)
        ).reshape(n_samples, n_steps, n_feats)
        X_lstm_test  = scaler_lstm.transform(
            X_lstm_test.reshape(-1, n_feats)
        ).reshape(-1, n_steps, n_feats)

        # Scale target per fold
        scaler_y  = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        model = build_model(
            lstm_shape = (LAG_YEARS, len(LSTM_FEATURES)),
            dnn_shape  = X_dnn_imp.shape[1],
            lstm_units = lstm_units,
            dnn_first  = dnn_first,
        )

        model.fit(
            [X_lstm_train, X_dnn_train], y_train_s,
            epochs           = 150,
            batch_size       = 64,
            validation_split = 0.1,
            shuffle          = True,
            verbose          = 0,
            callbacks        = [tf.keras.callbacks.EarlyStopping(
                monitor              = "val_loss",
                patience             = 15,
                restore_best_weights = True,
            )],
        )

        y_pred_s = model.predict([X_lstm_test, X_dnn_test], verbose=0).flatten()
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_r2.append(r2)
        fold_rmse.append(rmse)
        print(f"  Fold {fold_idx + 1}: R²={r2:.4f}  RMSE={rmse:.2f}")

    mean_r2   = float(np.mean(fold_r2))
    mean_rmse = float(np.mean(fold_rmse))
    print(f"  Mean R²={mean_r2:.4f}  RMSE={mean_rmse:.2f}")

    all_results.append({
        "config":       label,
        "lstm_units":   lstm_units,
        "dnn_first":    dnn_first,
        "mean_r2":      round(mean_r2,   4),
        "mean_rmse":    round(mean_rmse, 2),
        "vs_SLM_r2":    round(mean_r2 - SLM_R2, 4),
        "vs_asymmetric_r2": round(mean_r2 - ASYMMETRIC_R2, 4),
    })

# Step 5: Summarise results
df_results = pd.DataFrame(all_results).sort_values("mean_r2", ascending=False)
print(f"\nAll configurations sorted by R²:")
print(df_results[["config", "lstm_units", "dnn_first",
                   "mean_r2", "mean_rmse",
                   "vs_SLM_r2", "vs_asymmetric_r2"]].to_string(index=False))

df_results.to_csv("data/processed/balanced_units_results.csv", index=False)
print(f"\nSaved: data/processed/balanced_units_results.csv")

# Step 6: Bar chart comparing all configurations
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Balanced vs Asymmetric Unit Configurations — Lung Cancer", fontsize=13)

labels    = df_results["config"].tolist()
r2_vals   = df_results["mean_r2"].tolist()
rmse_vals = df_results["mean_rmse"].tolist()
colors    = ["#C44E52" if "asymmetric" in l else "#4C72B0" for l in labels]

ax = axes[0]
bars = ax.bar(range(len(labels)), r2_vals, color=colors, width=0.55, edgecolor="white")
ax.axhline(SLM_R2,         color="#D85A30", linewidth=1.5, linestyle="--", label=f"SLM baseline ({SLM_R2})")
ax.axhline(ASYMMETRIC_R2,  color="#C44E52", linewidth=1.5, linestyle=":",  label=f"Original asymmetric ({ASYMMETRIC_R2})")
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Mean R² (5-fold CV)")
ax.set_title("R² — higher is better")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(r2_vals) + 0.08)

ax = axes[1]
bars = ax.bar(range(len(labels)), rmse_vals, color=colors, width=0.55, edgecolor="white")
ax.axhline(SLM_RMSE,         color="#D85A30", linewidth=1.5, linestyle="--", label=f"SLM baseline ({SLM_RMSE})")
ax.axhline(ASYMMETRIC_RMSE,  color="#C44E52", linewidth=1.5, linestyle=":",  label=f"Original asymmetric ({ASYMMETRIC_RMSE})")
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Mean RMSE (per 100k, 5-fold CV)")
ax.set_title("RMSE — lower is better")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/figures/16_balanced_units.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/16_balanced_units.png")