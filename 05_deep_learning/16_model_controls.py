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

# Configuration
# Best hyperparameters from 15_model_lung.py tuning
BEST_LSTM_UNITS   = 64
BEST_LEARNING_RATE = 1e-4
BEST_DROPOUT_RATE  = 0.1

LAG_YEARS   = 10
N_FOLDS     = 5
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# SLM baselines for each cancer type (from 14_spatial_models.py)
# Lung is included here for side-by-side comparison in the final figure
SLM_BASELINES = {
    "lung":        {"r2": 0.474, "rmse": 17.88},
    "breast":      {"r2": 0.057, "rmse": 4.38},
    "colorectal":  {"r2": 0.116, "rmse": 8.92},
}

# Socioeconomic features for DNN branch (same as lung model)
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

# Pollution features for LSTM branch (same as lung model)
LSTM_FEATURES = ["pm25_satellite", "no2_satellite_ppb"]

# Cancer targets to evaluate as negative controls
# Maps cancer name -> column name in complete_data_original.csv
CONTROL_CANCERS = {
    "breast":     "mortality_breast_per100k",
    "colorectal": "mortality_colorectal_per100k",
}

# Step 1: Load data
df_main = pd.read_csv("data/processed/complete_data_original.csv")
df_acag = pd.read_csv("data/processed/acag_county_pm25.csv")
df_no2  = pd.read_csv("data/processed/no2_satellite_county.csv")

# Merge ACAG and NO2 into one pollution table by county and year
df_poll = df_acag.merge(df_no2, on=["county_fips", "year"], how="outer")
df_poll.columns = ["county_fips", "year", "pm25_satellite", "no2_satellite_ppb"]

# Pre-index by county and year for fast sequence lookup
df_poll = df_poll.set_index(["county_fips", "year"])

print(f"  Main dataset:   {df_main.shape}")
print(f"  Pollution data: {df_poll.shape}")

# Step 2: Define the model builder with best fixed hyperparameters
# Architecture is identical to the best lung model from tuning
def build_best_model(lstm_shape, dnn_shape, output_name):

    # LSTM branch — processes temporal pollution sequences
    lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
    lstm_out   = LSTM(BEST_LSTM_UNITS, return_sequences=False)(lstm_input)
    lstm_out   = Dense(BEST_LSTM_UNITS // 2, activation="relu")(lstm_out)

    # DNN branch — processes static socioeconomic features
    dnn_input  = Input(shape=(dnn_shape,), name="socioeconomic")
    dnn_out    = Dense(BEST_LSTM_UNITS * 2, activation="relu")(dnn_input)
    dnn_out    = Dense(BEST_LSTM_UNITS, activation="relu")(dnn_out)

    # Merge both branches
    merged     = Concatenate()([lstm_out, dnn_out])
    merged     = Dense(BEST_LSTM_UNITS, activation="relu")(merged)
    merged     = Dropout(BEST_DROPOUT_RATE)(merged)

    # Output — single mortality prediction
    output     = Dense(1, activation="linear", name=output_name)(merged)

    model = Model(inputs=[lstm_input, dnn_input], outputs=output)
    model.compile(
        loss      = "mse",
        optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_LEARNING_RATE),
        metrics   = ["mae"]
    )
    return model

# Step 3: Run 5-fold CV for each control cancer
# We reuse the same LSTM sequence builder as in 15_model_lung.py
control_results = []

for cancer_name, target_col in CONTROL_CANCERS.items():

    print(f"\nRunning control: {cancer_name.upper()}")
    slm_r2   = SLM_BASELINES[cancer_name]["r2"]
    slm_rmse = SLM_BASELINES[cancer_name]["rmse"]

    # Filter to rows with observed mortality for this cancer type
    df_cancer = df_main[df_main[target_col].notna()].copy()
    print(f"  Observed rows: {len(df_cancer)}")

    # Build LSTM sequences (same 10-year lag window as lung model)
    lstm_sequences = []
    valid_indices  = []

    for idx, row in df_cancer.iterrows():
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
    print(f"  LSTM array shape: {lstm_array.shape}")

    # Fill missing pollution values with column mean
    for feature_idx in range(lstm_array.shape[2]):
        col_mean = np.nanmean(lstm_array[:, :, feature_idx])
        mask     = np.isnan(lstm_array[:, :, feature_idx])
        lstm_array[:, :, feature_idx][mask] = col_mean

    # Build DNN feature matrix and target
    df_valid  = df_cancer.loc[valid_indices].reset_index(drop=True)
    X_dnn_raw = df_valid[DNN_FEATURES].values.astype("float32")
    y         = df_valid[target_col].values.astype("float32")

    imputer   = SimpleImputer(strategy="median")
    X_dnn_imp = imputer.fit_transform(X_dnn_raw)

    print(f"  Target mean: {y.mean():.2f}  std: {y.std():.2f}")

    # Step 4: 5-fold CV evaluation
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_r2    = []
    fold_rmse  = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(y)):

        X_lstm_train = lstm_array[train_idx]
        X_lstm_test  = lstm_array[test_idx]
        X_dnn_train  = X_dnn_imp[train_idx]
        X_dnn_test   = X_dnn_imp[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]

        # Scale DNN features (fit on train, apply to test)
        scaler_dnn  = StandardScaler()
        X_dnn_train = scaler_dnn.fit_transform(X_dnn_train)
        X_dnn_test  = scaler_dnn.transform(X_dnn_test)

        # Scale LSTM sequences (reshape to 2D, scale, reshape back)
        scaler_lstm  = StandardScaler()
        n_samples, n_steps, n_feats = X_lstm_train.shape
        X_lstm_train = scaler_lstm.fit_transform(
            X_lstm_train.reshape(-1, n_feats)
        ).reshape(n_samples, n_steps, n_feats)
        X_lstm_test  = scaler_lstm.transform(
            X_lstm_test.reshape(-1, n_feats)
        ).reshape(-1, n_steps, n_feats)

        # Scale target (predict in standardized space, then invert)
        scaler_y  = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        model = build_best_model(
            lstm_shape  = (LAG_YEARS, len(LSTM_FEATURES)),
            dnn_shape   = X_dnn_imp.shape[1],
            output_name = f"{cancer_name}_mortality",
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 15,
            restore_best_weights = True
        )

        model.fit(
            [X_lstm_train, X_dnn_train],
            y_train_s,
            epochs           = 200,
            batch_size       = 64,
            validation_split = 0.1,
            shuffle          = True,
            verbose          = 0,
            callbacks        = [early_stop]
        )

        y_pred_s = model.predict([X_lstm_test, X_dnn_test], verbose=0).flatten()
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

        fold_r2.append(r2_score(y_test, y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"  Fold {fold+1}: R²={fold_r2[-1]:.4f}  RMSE={fold_rmse[-1]:.2f}")

    mean_r2   = np.mean(fold_r2)
    mean_rmse = np.mean(fold_rmse)

    print(f"  Mean R²={mean_r2:.4f}  RMSE={mean_rmse:.2f}")
    print(f"  SLM baseline: R²={slm_r2}  RMSE={slm_rmse}")
    print(f"  DeltaR²={mean_r2 - slm_r2:+.4f}")

    control_results.append({
        "cancer":     cancer_name,
        "dl_r2":      round(mean_r2,   4),
        "dl_rmse":    round(mean_rmse, 2),
        "slm_r2":     slm_r2,
        "slm_rmse":   slm_rmse,
        "delta_r2":   round(mean_r2 - slm_r2,   4),
        "delta_rmse": round(mean_rmse - slm_rmse, 2),
    })

# Step 5: Save results CSV
df_controls = pd.DataFrame(control_results)
print(f"\nControl results summary:")
print(df_controls.to_string(index=False))

df_controls.to_csv("data/processed/control_model_results.csv", index=False)
print(f"\nSaved: data/processed/control_model_results.csv")

# Step 6: Three-cancer comparison bar chart
# Include lung (best DL result from tuning) alongside controls for full picture
LUNG_DL_R2   = 0.514
LUNG_DL_RMSE = 17.31

all_cancers = ["lung", "breast", "colorectal"]
dl_r2_vals  = [LUNG_DL_R2] + df_controls["dl_r2"].tolist()
slm_r2_vals = [SLM_BASELINES[c]["r2"] for c in all_cancers]

dl_rmse_vals  = [LUNG_DL_RMSE] + df_controls["dl_rmse"].tolist()
slm_rmse_vals = [SLM_BASELINES[c]["rmse"] for c in all_cancers]

x     = np.arange(len(all_cancers))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "DL Model vs. SLM Baseline — Lung (target) vs. Breast & Colorectal (controls)",
    fontsize=13
)

# Plot 1: R²
bars_dl  = axes[0].bar(x - width / 2, dl_r2_vals,  width, label="DL model",     color="#4C72B0")
bars_slm = axes[0].bar(x + width / 2, slm_r2_vals, width, label="SLM baseline", color="#DD8452")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Lung\n(target)", "Breast\n(control)", "Colorectal\n(control)"])
axes[0].set_ylabel("R²")
axes[0].set_title("R² — higher is better")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)
for bar in bars_dl:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars_slm:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

# Plot 2: RMSE
bars_dl  = axes[1].bar(x - width / 2, dl_rmse_vals,  width, label="DL model",     color="#4C72B0")
bars_slm = axes[1].bar(x + width / 2, slm_rmse_vals, width, label="SLM baseline", color="#DD8452")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Lung\n(target)", "Breast\n(control)", "Colorectal\n(control)"])
axes[1].set_ylabel("RMSE (per 100k)")
axes[1].set_title("RMSE — lower is better")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)
for bar in bars_dl:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
for bar in bars_slm:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/12_control_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/12_control_comparison.png")

# Step 7: Delta-R² summary plot
# Shows how much the DL model improves (or does not improve) over SLM per cancer type
# Lung should show positive delta; breast and colorectal should show near-zero or negative
delta_r2_vals = [LUNG_DL_R2 - SLM_BASELINES["lung"]["r2"]] + df_controls["delta_r2"].tolist()
colors        = ["#2ca02c" if d > 0 else "#d62728" for d in delta_r2_vals]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(all_cancers, delta_r2_vals, color=colors)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("ΔR² (DL − SLM)")
ax.set_title("DL improvement over SLM baseline\nGreen = DL better, Red = DL worse")
ax.set_xticklabels(["Lung\n(target)", "Breast\n(control)", "Colorectal\n(control)"])
ax.grid(axis="y", alpha=0.3)
for i, (val, cancer) in enumerate(zip(delta_r2_vals, all_cancers)):
    ax.text(i, val + (0.002 if val >= 0 else -0.005),
            f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/figures/12b_delta_r2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/12b_delta_r2.png")