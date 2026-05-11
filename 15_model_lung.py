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
LAG_YEARS   = 10
N_FOLDS     = 5
EPOCHS      = 150
BATCH_SIZE  = 64
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Socioeconomic features for DNN branch
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

# Pollution features for LSTM branch
# Ground PM2.5 excluded: 70.9% missing, mean imputation adds noise not signal
LSTM_FEATURES = ["pm25_satellite", "no2_satellite_ppb"]

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

# Filter to lung cancer observations only
df_lung = df_main[df_main["mortality_lung_per100k"].notna()].copy()
print(f"  Lung cancer rows (observed): {len(df_lung)}")

# Step 2: Build LSTM sequences
# For each county-cancer year T, extract pollution values from T-10 to T-1
# This gives the LSTM a 10-year window to learn temporal exposure patterns
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
print(f"  LSTM array shape: {lstm_array.shape}")
print(f"  Missing in PM2.5 sequences: {np.isnan(lstm_array[:,:,0]).sum()}")
print(f"  Missing in NO2 sequences:   {np.isnan(lstm_array[:,:,1]).sum()}")

# Fill missing values with column mean
for feature_idx in range(lstm_array.shape[2]):
    col_mean = np.nanmean(lstm_array[:, :, feature_idx])
    mask     = np.isnan(lstm_array[:, :, feature_idx])
    lstm_array[:, :, feature_idx][mask] = col_mean

print(f"  Missing after fill: {np.isnan(lstm_array).sum()}")

# Step 3: Build DNN feature matrix
df_lung_valid = df_lung.loc[valid_indices].reset_index(drop=True)
X_dnn_raw     = df_lung_valid[DNN_FEATURES].values.astype("float32")
y             = df_lung_valid["mortality_lung_per100k"].values.astype("float32")

# Impute missing socioeconomic values with median
imputer   = SimpleImputer(strategy="median")
X_dnn_imp = imputer.fit_transform(X_dnn_raw)

print(f"  DNN feature matrix shape: {X_dnn_imp.shape}")
print(f"  Target shape:             {y.shape}")
print(f"  Target mean: {y.mean():.2f}  std: {y.std():.2f}")

# Step 4: Define LSTM + DNN Functional API model
def build_model(lstm_shape, dnn_shape):

    # LSTM branch — processes temporal pollution sequences
    lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
    lstm_out   = LSTM(32, return_sequences=False)(lstm_input)
    lstm_out   = Dense(16, activation="relu")(lstm_out)

    # DNN branch — processes static socioeconomic features
    dnn_input  = Input(shape=(dnn_shape,), name="socioeconomic")
    dnn_out    = Dense(32, activation="relu")(dnn_input)
    dnn_out    = Dense(16, activation="relu")(dnn_out)

    # Merge both branches
    merged     = Concatenate()([lstm_out, dnn_out])
    merged     = Dense(32, activation="relu")(merged)
    merged     = Dropout(0.2)(merged)

    # Output — single value for lung cancer mortality rate
    output     = Dense(1, activation="linear", name="lung_mortality")(merged)

    model = Model(inputs=[lstm_input, dnn_input], outputs=output)
    model.compile(
        loss      = "mse",
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4),
        metrics   = ["mae"]
    )
    return model

# Print model summary once before training
sample_model = build_model(
    lstm_shape = (LAG_YEARS, len(LSTM_FEATURES)),
    dnn_shape  = X_dnn_imp.shape[1]
)
sample_model.summary()

# Step 5: Hyperparameter tuning with K-Fold cross-validation
# We search over key architecture and training parameters
# EarlyStopping handles epoch count automatically so we fix EPOCHS=200

from itertools import product

PARAM_GRID = {
    "lstm_units":    [32, 64],
    "learning_rate": [1e-3, 3e-4, 1e-4],
    "dropout_rate":  [0.1, 0.2, 0.3],
}

# Build all combinations
keys   = list(PARAM_GRID.keys())
values = list(PARAM_GRID.values())
combos = list(product(*values))

print(f"Total hyperparameter combinations: {len(combos)}")
print(f"Each trained with {N_FOLDS}-fold CV\n")

tuning_results = []

for combo in combos:
    params = dict(zip(keys, combo))
    print(f"Testing: {params}")

    def build_model_tuned(lstm_shape, dnn_shape, lstm_units, learning_rate, dropout_rate):

        lstm_input = Input(shape=lstm_shape, name="pollution_timeseries")
        lstm_out   = LSTM(lstm_units, return_sequences=False)(lstm_input)
        lstm_out   = Dense(lstm_units // 2, activation="relu")(lstm_out)

        dnn_input  = Input(shape=(dnn_shape,), name="socioeconomic")
        dnn_out    = Dense(lstm_units * 2, activation="relu")(dnn_input)
        dnn_out    = Dense(lstm_units, activation="relu")(dnn_out)

        merged     = Concatenate()([lstm_out, dnn_out])
        merged     = Dense(lstm_units, activation="relu")(merged)
        merged     = Dropout(dropout_rate)(merged)

        output     = Dense(1, activation="linear", name="lung_mortality")(merged)

        model = Model(inputs=[lstm_input, dnn_input], outputs=output)
        model.compile(
            loss      = "mse",
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics   = ["mae"]
        )
        return model

    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    combo_r2   = []
    combo_rmse = []

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

        scaler_lstm = StandardScaler()
        n_samples, n_steps, n_feats = X_lstm_train.shape
        X_lstm_train = scaler_lstm.fit_transform(
            X_lstm_train.reshape(-1, n_feats)
        ).reshape(n_samples, n_steps, n_feats)
        X_lstm_test  = scaler_lstm.transform(
            X_lstm_test.reshape(-1, n_feats)
        ).reshape(-1, n_steps, n_feats)

        scaler_y  = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        model = build_model_tuned(
            lstm_shape    = (LAG_YEARS, len(LSTM_FEATURES)),
            dnn_shape     = X_dnn_imp.shape[1],
            lstm_units    = params["lstm_units"],
            learning_rate = params["learning_rate"],
            dropout_rate  = params["dropout_rate"],
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
            batch_size       = BATCH_SIZE,
            validation_split = 0.1,
            shuffle          = True,
            verbose          = 0,
            callbacks        = [early_stop]
        )

        y_pred_s      = model.predict([X_lstm_test, X_dnn_test], verbose=0).flatten()
        y_pred        = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        combo_r2.append(r2_score(y_test, y_pred))
        combo_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    mean_r2   = np.mean(combo_r2)
    mean_rmse = np.mean(combo_rmse)
    print(f"  Mean R²={mean_r2:.4f}  RMSE={mean_rmse:.2f}")

    tuning_results.append({
        **params,
        "mean_r2":   round(mean_r2,   4),
        "mean_rmse": round(mean_rmse, 2),
    })

# Step 6: Report best configuration
df_tuning = pd.DataFrame(tuning_results).sort_values("mean_r2", ascending=False)
print(f"\nAll results sorted by R²:")
print(df_tuning.to_string(index=False))

best = df_tuning.iloc[0]
print(f"\nBest configuration:")
print(f"  lstm_units={int(best.lstm_units)}  lr={best.learning_rate}  dropout={best.dropout_rate}")
print(f"  R²={best.mean_r2:.4f}  RMSE={best.mean_rmse:.2f}")
print(f"  SLM baseline: R²=0.474  RMSE=17.88")
print(f"  DeltaR²={best.mean_r2 - 0.474:+.4f}")

df_tuning.to_csv("data/processed/lung_tuning_results.csv", index=False)
print(f"\nSaved: data/processed/lung_tuning_results.csv")

# Step 7: Visualise tuning results

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Hyperparameter Tuning Results — Lung Cancer Model", fontsize=13)

# Plot 1: R² by LSTM units
for lr in df_tuning["learning_rate"].unique():
    subset = df_tuning[df_tuning["learning_rate"] == lr]
    grouped = subset.groupby("lstm_units")["mean_r2"].mean()
    axes[0].plot(grouped.index, grouped.values,
                 marker="o", label=f"lr={lr}")
axes[0].axhline(0.474, color="#D85A30", linewidth=1.5,
                linestyle="--", label="SLM baseline")
axes[0].set_title("R² by LSTM units")
axes[0].set_xlabel("LSTM units")
axes[0].set_ylabel("Mean R²")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Plot 2: R² by learning rate
for units in df_tuning["lstm_units"].unique():
    subset = df_tuning[df_tuning["lstm_units"] == units]
    grouped = subset.groupby("learning_rate")["mean_r2"].mean()
    axes[1].plot(grouped.index, grouped.values,
                 marker="o", label=f"units={int(units)}")
axes[1].axhline(0.474, color="#D85A30", linewidth=1.5,
                linestyle="--", label="SLM baseline")
axes[1].set_title("R² by learning rate")
axes[1].set_xlabel("Learning rate")
axes[1].set_ylabel("Mean R²")
axes[1].set_xscale("log")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Plot 3: R² by dropout rate
for units in df_tuning["lstm_units"].unique():
    subset = df_tuning[df_tuning["lstm_units"] == units]
    grouped = subset.groupby("dropout_rate")["mean_r2"].mean()
    axes[2].plot(grouped.index, grouped.values,
                 marker="o", label=f"units={int(units)}")
axes[2].axhline(0.474, color="#D85A30", linewidth=1.5,
                linestyle="--", label="SLM baseline")
axes[2].set_title("R² by dropout rate")
axes[2].set_xlabel("Dropout rate")
axes[2].set_ylabel("Mean R²")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/figures/11_lung_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/11_lung_tuning_curves.png")

# Step 8: Heatmap of all 18 combinations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Tuning Heatmap — R² and RMSE across all configurations", fontsize=13)

for ax, metric, title, cmap in [
    (axes[0], "mean_r2",   "Mean R²",   "YlGn"),
    (axes[1], "mean_rmse", "Mean RMSE", "YlOrRd_r"),
]:
    # Pivot: rows = lstm_units, columns = learning_rate, averaged over dropout
    pivot = df_tuning.groupby(["lstm_units", "learning_rate"])[metric].mean().unstack()
    im    = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in pivot.columns], fontsize=9)
    ax.set_yticklabels([f"{int(u)}" for u in pivot.index], fontsize=9)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LSTM units")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if pivot.values[i,j] < pivot.values.max() * 0.95 else "white")

plt.tight_layout()
plt.savefig("outputs/figures/11b_lung_tuning_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: outputs/figures/11b_lung_tuning_heatmap.png")

# Step 9: Full results table saved as CSV for presentation
df_tuning_display = df_tuning.copy()
df_tuning_display["vs_SLM_R2"]   = (df_tuning_display["mean_r2"] - 0.474).round(4)
df_tuning_display["vs_SLM_RMSE"] = (df_tuning_display["mean_rmse"] - 17.88).round(2)
df_tuning_display["rank"]         = range(1, len(df_tuning_display) + 1)

print(f"\nFull tuning results table:")
print(df_tuning_display.to_string(index=False))

df_tuning_display.to_csv("data/processed/lung_tuning_results.csv", index=False)
print(f"\nSaved: data/processed/lung_tuning_results.csv")