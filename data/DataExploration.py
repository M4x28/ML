import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# ==========================================================
# CONFIGURAZIONE
# ==========================================================
DATASET_PATH = "ML-CUP25-TR.csv"   # <-- modifica qui
OUTPUT_DIR = "eda_plots"

# Creazione cartella output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# CARICAMENTO
# ==========================================================
df = pd.read_csv(DATASET_PATH, comment='#', header=None)

# Identifica e rimuove la prima colonna (ID)
ID_COL = df.columns[0]
print(f"Rilevata colonna ID: {ID_COL} — verrà esclusa dalle analisi.")
df = df.drop(columns=[ID_COL])


# Rinomina le colonne rimanenti in modo leggibile
num_cols = df.shape[1]
feature_count = num_cols - 4

new_names = (
    [f"F_{i}" for i in range(feature_count)] +
    [f"TARGET_{i+1}" for i in range(4)]
)

df.columns = new_names

print("Nuovi nomi assegnati:")
print(df.columns.tolist())

print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIZIONE ===")
print(df.describe().T)

# ==========================================================
# 1. ISTOGRAMMI DI TUTTE LE FEATURE
# ==========================================================
print("Plotting histograms...")
for col in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram – {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hist_{col}.png")
    plt.close()

# ==========================================================
# 2. BOXPLOT PER OUTLIER DETECTION
# ==========================================================
print("Plotting boxplots...")
for col in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot – {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/box_{col}.png")
    plt.close()

# ==========================================================
# 3. CORRELAZIONI
# ==========================================================
print("Plotting correlation heatmap...")

plt.figure(figsize=(12,10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
plt.close()

# ==========================================================
# 4. CORRELAZIONI CON I TARGET (se ultimi 4 sono i target)
# ==========================================================
TARGET_COLS = df.columns[-4:]   # <-- modifica se necessario
FEATURE_COLS = df.columns[:-4]

corr_targets = df.corr()[TARGET_COLS].loc[FEATURE_COLS]

plt.figure(figsize=(10,8))
sns.heatmap(corr_targets, cmap="viridis", annot=True, fmt=".2f")
plt.title("Feature–Target Correlations")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_target_correlation.png")
plt.close()

# ==========================================================
# 5. PAIRPLOT DI UN SOTTOINSIEME (per non esplodere la memoria)
# ==========================================================
print("Plotting pairplot (subset)...")
subset_cols = list(FEATURE_COLS[:5]) + list(TARGET_COLS)
sns.pairplot(df[subset_cols], diag_kind="kde")
plt.savefig(f"{OUTPUT_DIR}/pairplot_subset.png")
plt.close()

# ==========================================================
# 6. SKEWNESS E KURTOSIS
# ==========================================================
print("\n=== SKEWNESS / KURTOSIS ===")
stats_df = pd.DataFrame({
    "skewness": df.skew(),
    "kurtosis": df.kurt()
})
print(stats_df)

stats_df.to_csv(f"{OUTPUT_DIR}/skewness_kurtosis.csv")

# ==========================================================
# 7. VIF (Multicollinearità)
# ==========================================================
print("Computing VIF...")
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[FEATURE_COLS].dropna()

vif_df = pd.DataFrame()
vif_df["feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_df)
vif_df.to_csv(f"{OUTPUT_DIR}/vif.csv", index=False)

print("\n=== EDA COMPLETATA ===")
print(f"Tutti i grafici sono stati salvati in: {OUTPUT_DIR}/")


print("Saving JSON summary...")

eda_summary = {
    "n_rows": len(df),
    "n_columns": df.shape[1],
    "features": FEATURE_COLS.tolist(),
    "targets": TARGET_COLS.tolist(),
    "missing_values": df.isnull().sum().to_dict(),
    "describe": df.describe().to_dict(),
    "skewness": stats_df["skewness"].to_dict(),
    "kurtosis": stats_df["kurtosis"].to_dict(),
    "vif": dict(zip(vif_df["feature"], vif_df["VIF"])),
    "correlation_feature_target": corr_targets.to_dict()
}

with open(f"{OUTPUT_DIR}/eda_summary.json", "w") as f:
    json.dump(eda_summary, f, indent=4)

print("JSON summary saved.")