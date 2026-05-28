import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Concatenate, Dropout, GaussianNoise
)
from tensorflow.keras import regularizers
from tensorflow.keras import Model

import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

# Configuration — best hyperparameters fixed from 15_model_lung.py
BEST_LSTM_UNITS    = 64
BEST_LEARNING_RATE = 1e-4
BEST_DROPOUT_RATE  = 0.1

LAG_YEARS   = 10
N_FOLDS     = 5
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SLM_R2           = 0.474
SLM_RMSE         = 17.88
BASELINE_DL_R2   = 0.513
BASELINE_DL_RMSE = 18.88

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

# Search grid: 3 noise levels x 2 L2 strengths = 6 combinations
PARAM_GRID = {
    "noise_stddev": [0.01, 0.05, 0.1],
    "l2_lambda":    [1e-4, 1e-3],
}

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

# Step 2: Build LSTM sequences (same logic as scripts 15-18)
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

print(f"  LSTM array shape:  {lstm_array.shape}")
print(f"  DNN feature shape: {X_dnn_imp.shape}")
print(f"  Target mean: {y.mean():.2f}  std: {y.std():.2f}")

# Step 3: Define regularized model
# New architecture vs scripts 15-17:
#   LSTM branch: GaussianNoise -> LSTM(64) -> Dense(64,relu,L2)
#                -> Dense(32,relu,L2) -> Dropout -> Dense(16,relu)
#   DNN branch:  GaussianNoise -> Dense(64,relu,L2) -> Dropout
#                -> Dense(32,relu,L2) -> Dense(16,relu)
#   Merge:       Concatenate(32) -> Dense(32,relu,L2) -> Dropout -> Dense(1)
#
# GaussianNoise is active only during training and automatically
# disabled during prediction, so evaluation uses clean inputs.
# L2 penalises large weights during training, preventing over-reliance
# on any single feature — especially useful given high collinearity
# between the socioeconomic variables.
def build_regularized_model(lstm_shape, dnn_shape, noise_stddev, l2_lambda):

    l2 = regularizers.l2(l2_lambda)

    # LSTM branch
    lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
    lstm_out   = GaussianNoise(noise_stddev)(lstm_input)
    lstm_out   = LSTM(BEST_LSTM_UNITS, return_sequences=False)(lstm_out)
    lstm_out   = Dense(64, activation="relu", kernel_regularizer=l2)(lstm_out)
    lstm_out   = Dense(32, activation="relu", kernel_regularizer=l2)(lstm_out)
    lstm_out   = Dropout(BEST_DROPOUT_RATE)(lstm_out)
    lstm_out   = Dense(16, activation="relu")(lstm_out)

    # DNN branch
    dnn_input  = Input(shape=(dnn_shape,), name="socioeconomic")
    dnn_out    = GaussianNoise(noise_stddev)(dnn_input)
    dnn_out    = Dense(64, activation="relu", kernel_regularizer=l2)(dnn_out)
    dnn_out    = Dropout(BEST_DROPOUT_RATE)(dnn_out)
    dnn_out    = Dense(32, activation="relu", kernel_regularizer=l2)(dnn_out)
    dnn_out    = Dense(16, activation="relu")(dnn_out)

    # Merge
    merged = Concatenate()([lstm_out, dnn_out])
    merged = Dense(32, activation="relu", kernel_regularizer=l2)(merged)
    merged = Dropout(BEST_DROPOUT_RATE)(merged)
    output = Dense(1, activation="linear", name="lung_mortality")(merged)

    model = Model(inputs=[lstm_input, dnn_input], outputs=output)
    model.compile(
        loss      = "mse",
        optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_LEARNING_RATE),
        metrics   = ["mae"]
    )
    return model

# Print architecture once before the search
sample_model = build_regularized_model(
    lstm_shape   = (LAG_YEARS, len(LSTM_FEATURES)),
    dnn_shape    = X_dnn_imp.shape[1],
    noise_stddev = 0.01,
    l2_lambda    = 1e-4
)
sample_model.summary()

# Step 4: Grid search with 5-fold CV
keys   = list(PARAM_GRID.keys())
combos = list(product(*PARAM_GRID.values()))
print(f"\nTotal combinations: {len(combos)}  (each with {N_FOLDS}-fold CV)\n")

kf             = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
tuning_results = []

for combo in combos:
    params = dict(zip(keys, combo))
    print(f"Testing: noise_stddev={params['noise_stddev']}  l2_lambda={params['l2_lambda']}")

    fold_r2   = []
    fold_rmse = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(y)):

        X_lstm_train = lstm_array[train_idx]
        X_lstm_test  = lstm_array[test_idx]
        X_dnn_train  = X_dnn_imp[train_idx]
        X_dnn_test   = X_dnn_imp[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]

        scaler_dnn  = StandardScaler()
        X_dnn_train = scaler_dnn.fit_transform(X_dnn_train)
        X_dnn_test  = scaler_dnn.transform(X_dnn_test)

        scaler_lstm  = StandardScaler()
        n_samples, n_steps, n_feats = X_lstm_train.shape
        X_lstm_train = scaler_lstm.fit_transform(
            X_lstm_train.reshape(-1, n_feats)
        ).reshape(n_samples, n_steps, n_feats)
        X_lstm_test  = scaler_lstm.transform(
            X_lstm_test.reshape(-1, n_feats)
        ).reshape(-1, n_steps, n_feats)

        scaler_y  = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        model = build_regularized_model(
            lstm_shape   = (LAG_YEARS, len(LSTM_FEATURES)),
            dnn_shape    = X_dnn_imp.shape[1],
            noise_stddev = params["noise_stddev"],
            l2_lambda    = params["l2_lambda"]
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

    tuning_results.append({
        **params,
        "mean_r2":          round(mean_r2,   4),
        "mean_rmse":        round(mean_rmse, 2),
        "vs_baseline_r2":   round(mean_r2   - BASELINE_DL_R2,   4),
        "vs_baseline_rmse": round(mean_rmse - BASELINE_DL_RMSE, 2),
        "vs_SLM_r2":        round(mean_r2   - SLM_R2,           4),
    })

# Step 5: Report results
df_results = pd.DataFrame(tuning_results).sort_values("mean_r2", ascending=False)
print(f"\nAll results sorted by R²:")
print(df_results.to_string(index=False))

best = df_results.iloc[0]
print(f"\nBest configuration:")
print(f"  noise_stddev={best.noise_stddev}  l2_lambda={best.l2_lambda}")
print(f"  R²={best.mean_r2:.4f}  RMSE={best.mean_rmse:.2f}")
print(f"  vs unweighted DL baseline: dR2={best.vs_baseline_r2:+.4f}")
print(f"  vs SLM baseline:           dR2={best.vs_SLM_r2:+.4f}")

df_results.to_csv("data/processed/regularization_results.csv", index=False)
print(f"\nSaved: data/processed/regularization_results.csv")

# Step 6: Heatmap — R² and RMSE across noise x L2 combinations
pivot_r2   = df_results.pivot(index="noise_stddev", columns="l2_lambda", values="mean_r2")
pivot_rmse = df_results.pivot(index="noise_stddev", columns="l2_lambda", values="mean_rmse")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Regularization Search — Lung Cancer Model\n(Gaussian Noise x L2 strength)",
    fontsize=13
)

for ax, pivot, title, cmap in [
    (axes[0], pivot_r2,   "Mean R² (higher = better)",           "YlGn"),
    (axes[1], pivot_rmse, "Mean RMSE per 100k (lower = better)", "YlOrRd_r"),
]:
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"L2={v:.0e}" for v in pivot.columns], fontsize=9)
    ax.set_yticklabels([f"noise={v}" for v in pivot.index], fontsize=9)
    ax.set_xlabel("L2 lambda")
    ax.set_ylabel("Gaussian noise stddev")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=10, color="black")

plt.tight_layout()
plt.savefig("outputs/figures/15_regularization_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/15_regularization_heatmap.png")

# Step 7: Bar chart comparing all regularization variants vs baselines
labels    = [f"noise={r.noise_stddev}\nL2={r.l2_lambda:.0e}"
             for _, r in df_results.sort_values("noise_stddev").iterrows()]
r2_vals   = df_results.sort_values("noise_stddev")["mean_r2"].tolist()
rmse_vals = df_results.sort_values("noise_stddev")["mean_rmse"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Regularized Model Variants vs Baselines — Lung Cancer", fontsize=13)

x = range(len(labels))

for ax, vals, ylabel, title, baseline_dl, baseline_slm in [
    (axes[0], r2_vals,   "Mean R² (5-fold CV)",          "R² — higher is better",
     BASELINE_DL_R2,   SLM_R2),
    (axes[1], rmse_vals, "Mean RMSE (per 100k, 5-fold)", "RMSE — lower is better",
     BASELINE_DL_RMSE, SLM_RMSE),
]:
    bars = ax.bar(x, vals, color="#4C72B0")
    ax.axhline(baseline_dl,  color="#55A868", linewidth=1.5, linestyle="--",
               label=f"Unweighted DL ({baseline_dl})")
    ax.axhline(baseline_slm, color="#DD8452", linewidth=1.5, linestyle="--",
               label=f"SLM baseline ({baseline_slm})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/figures/15b_regularization_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/15b_regularization_comparison.png")