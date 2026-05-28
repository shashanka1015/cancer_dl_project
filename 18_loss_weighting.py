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

# Configuration — best hyperparameters fixed from 15_model_lung.py
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

# Baseline from ablation study (script 17): full model without loss weighting
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

# Loss weights to search over
# Main output weight is fixed at 1.0; we vary the auxiliary LSTM output weight
# Higher values force the LSTM branch to develop stronger standalone representations
LSTM_WEIGHT_GRID = [1.0, 2.0, 3.0, 5.0]

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

# Step 2: Build LSTM sequences (same logic as scripts 15–17)
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

# Step 3: Define the dual-output model
# The key architectural change vs scripts 15–17:
#   - lstm_out feeds into TWO places:
#       (a) a standalone Dense(1) called lstm_aux_output  <- auxiliary output
#       (b) the Concatenate merge with dnn_out            <- as before
#   - model.compile() receives loss_weights as a dict keyed by output name
#   - During training both outputs are penalised simultaneously
#   - During evaluation only the main output (lung_mortality) is used
def build_weighted_model(lstm_shape, dnn_shape, lstm_weight):

    # LSTM branch
    lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
    lstm_out   = LSTM(BEST_LSTM_UNITS, return_sequences=False)(lstm_input)
    lstm_out   = Dense(BEST_LSTM_UNITS // 2, activation="relu")(lstm_out)

    # Auxiliary output: LSTM branch alone predicts mortality
    # This output is only used during training to compute the weighted auxiliary loss
    lstm_aux_output = Dense(1, activation="linear", name="lstm_aux_output")(lstm_out)

    # DNN branch
    dnn_input  = Input(shape=(dnn_shape,), name="socioeconomic")
    dnn_out    = Dense(BEST_LSTM_UNITS * 2, activation="relu")(dnn_input)
    dnn_out    = Dense(BEST_LSTM_UNITS,     activation="relu")(dnn_out)

    # Merge both branches — main prediction path
    merged  = Concatenate()([lstm_out, dnn_out])
    merged  = Dense(BEST_LSTM_UNITS, activation="relu")(merged)
    merged  = Dropout(BEST_DROPOUT_RATE)(merged)
    main_output = Dense(1, activation="linear", name="lung_mortality")(merged)

    model = Model(
        inputs  = [lstm_input, dnn_input],
        outputs = [main_output, lstm_aux_output]
    )

    # loss_weights controls how hard each output is penalised during backprop
    # lstm_weight > 1.0 means gradients from the auxiliary LSTM loss are amplified,
    # forcing the LSTM branch to build stronger standalone representations
    model.compile(
        loss = {
            "lung_mortality":   "mse",
            "lstm_aux_output":  "mse",
        },
        loss_weights = {
            "lung_mortality":   1.0,
            "lstm_aux_output":  lstm_weight,
        },
        optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_LEARNING_RATE),
        metrics   = {"lung_mortality": "mae", "lstm_aux_output": "mae"}
    )
    return model

# Print summary once with lstm_weight=1.0 (symmetric, no extra pressure)
sample_model = build_weighted_model(
    lstm_shape  = (LAG_YEARS, len(LSTM_FEATURES)),
    dnn_shape   = X_dnn_imp.shape[1],
    lstm_weight = 1.0
)
sample_model.summary()

# Step 4: Search over LSTM weight values with 5-fold CV
# We fix the same KFold splits for all weights so results are directly comparable
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

weight_results = []

for lstm_weight in LSTM_WEIGHT_GRID:

    print(f"\nLSTM weight = {lstm_weight}")
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

        model = build_weighted_model(
            lstm_shape  = (LAG_YEARS, len(LSTM_FEATURES)),
            dnn_shape   = X_dnn_imp.shape[1],
            lstm_weight = lstm_weight
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 15,
            restore_best_weights = True
        )

        # Both outputs receive the same target y_train_s during training
        # The model optimises a weighted sum of both MSE losses simultaneously
        model.fit(
            [X_lstm_train, X_dnn_train],
            {"lung_mortality": y_train_s, "lstm_aux_output": y_train_s},
            epochs           = 200,
            batch_size       = 64,
            validation_split = 0.1,
            shuffle          = True,
            verbose          = 0,
            callbacks        = [early_stop]
        )

        # Only the main output is used for evaluation — aux output is discarded
        main_pred_s, _ = model.predict([X_lstm_test, X_dnn_test], verbose=0)
        y_pred = scaler_y.inverse_transform(
            main_pred_s.flatten().reshape(-1, 1)
        ).flatten()

        fold_r2.append(r2_score(y_test, y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        print(f"  Fold {fold+1}: R²={fold_r2[-1]:.4f}  RMSE={fold_rmse[-1]:.2f}")

    mean_r2   = np.mean(fold_r2)
    mean_rmse = np.mean(fold_rmse)
    print(f"  Mean R²={mean_r2:.4f}  RMSE={mean_rmse:.2f}")

    weight_results.append({
        "lstm_weight": lstm_weight,
        "mean_r2":     round(mean_r2,   4),
        "mean_rmse":   round(mean_rmse, 2),
        "vs_baseline_r2":   round(mean_r2   - BASELINE_DL_R2,   4),
        "vs_baseline_rmse": round(mean_rmse - BASELINE_DL_RMSE, 2),
        "vs_SLM_r2":        round(mean_r2   - SLM_R2,           4),
    })

# Step 5: Report results
df_weights = pd.DataFrame(weight_results).sort_values("mean_r2", ascending=False)
print(f"\nLoss weight search results (sorted by R²):")
print(df_weights.to_string(index=False))

best = df_weights.iloc[0]
print(f"\nBest LSTM weight: {best.lstm_weight}")
print(f"  R²={best.mean_r2:.4f}  RMSE={best.mean_rmse:.2f}")
print(f"  vs ablation baseline (no weighting): ΔR²={best.vs_baseline_r2:+.4f}")
print(f"  vs SLM baseline:                     ΔR²={best.vs_SLM_r2:+.4f}")

df_weights.to_csv("data/processed/loss_weight_results.csv", index=False)
print(f"\nSaved: data/processed/loss_weight_results.csv")

# Step 6: R² and RMSE across weight values — compared to unweighted DL and SLM baselines
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Effect of Auxiliary LSTM Loss Weight on Lung Cancer Model Performance",
    fontsize=13
)

weights = df_weights.sort_values("lstm_weight")["lstm_weight"].tolist()
r2_vals = df_weights.sort_values("lstm_weight")["mean_r2"].tolist()
rmse_vals = df_weights.sort_values("lstm_weight")["mean_rmse"].tolist()

# Plot 1: R²
axes[0].plot(weights, r2_vals, marker="o", color="#4C72B0",
             linewidth=2, markersize=8, label="Weighted DL model")
axes[0].axhline(BASELINE_DL_R2, color="#55A868", linewidth=1.5,
                linestyle="--", label=f"Unweighted DL (R²={BASELINE_DL_R2})")
axes[0].axhline(SLM_R2, color="#DD8452", linewidth=1.5,
                linestyle="--", label=f"SLM baseline (R²={SLM_R2})")
for w, r in zip(weights, r2_vals):
    axes[0].text(w, r + 0.002, f"{r:.3f}", ha="center", va="bottom", fontsize=9)
axes[0].set_xlabel("LSTM auxiliary loss weight")
axes[0].set_ylabel("Mean R² (5-fold CV)")
axes[0].set_title("R² — higher is better")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Plot 2: RMSE
axes[1].plot(weights, rmse_vals, marker="o", color="#C44E52",
             linewidth=2, markersize=8, label="Weighted DL model")
axes[1].axhline(BASELINE_DL_RMSE, color="#55A868", linewidth=1.5,
                linestyle="--", label=f"Unweighted DL (RMSE={BASELINE_DL_RMSE})")
axes[1].axhline(SLM_RMSE, color="#DD8452", linewidth=1.5,
                linestyle="--", label=f"SLM baseline (RMSE={SLM_RMSE})")
for w, r in zip(weights, rmse_vals):
    axes[1].text(w, r + 0.05, f"{r:.2f}", ha="center", va="bottom", fontsize=9)
axes[1].set_xlabel("LSTM auxiliary loss weight")
axes[1].set_ylabel("Mean RMSE (per 100k, 5-fold CV)")
axes[1].set_title("RMSE — lower is better")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/figures/14_loss_weight_search.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/14_loss_weight_search.png")