import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ----------------------------
# 1. Caricamento dati
# ----------------------------

def load_data(path="ML-CUP25-TR.csv"):
    # Il file ha commenti iniziali con '#'
    df = pd.read_csv(path, comment="#", header=None)
    df.columns = ["id"] + [f"x{i}" for i in range(1, 13)] + [f"y{j}" for j in range(1, 5)]
    return df


# ----------------------------
# 2. Info generali + missing
# ----------------------------

def basic_info(df: pd.DataFrame):
    print("=== INFO GENERALI ===")
    print(f"Shape (righe, colonne): {df.shape}")
    print("Colonne:", df.columns.tolist())
    print("\nTipi di dato:")
    print(df.dtypes)
    print("\nPrime righe:")
    print(df.head())


def check_missing(df: pd.DataFrame):
    print("\n=== MISSING VALUES ===")
    print("Per colonna:")
    print(df.isna().sum())
    missing_rows = df.isna().sum(axis=1)
    print(f"\nNumero di righe con almeno un NaN: {(missing_rows > 0).sum()}")


# ----------------------------
# 3. Statistiche descrittive
# ----------------------------

def describe_stats(df: pd.DataFrame):
    X_cols = [c for c in df.columns if c.startswith("x")]
    Y_cols = [c for c in df.columns if c.startswith("y")]

    print("\n=== STATISTICHE DESCRITTIVE: INPUT ===")
    print(df[X_cols].describe().T)

    print("\n=== STATISTICHE DESCRITTIVE: TARGET ===")
    print(df[Y_cols].describe().T)


# ----------------------------
# 4. Boxplot / outlier (IQR)
# ----------------------------

def outlier_analysis(df: pd.DataFrame):
    print("\n=== OUTLIER (regola 1.5 * IQR) ===")
    numeric_cols = [c for c in df.columns if c != "id"]
    stats_rows = []
    for col in numeric_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        stats_rows.append(
            {
                "colonna": col,
                "mean": s.mean(),
                "variance": s.var(),
                "std": s.std(),
                "outlier_count": int(mask.sum()),
            }
        )
        print(
            f"{col:>4}: outlier = {mask.sum():3d} "
            f"({mask.mean()*100:5.2f}%), "
            f"range = [{s.min():.2f}, {s.max():.2f}], "
            f"bounds = [{lower:.2f}, {upper:.2f}]"
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df["has_outlier"] = stats_df["outlier_count"] > 0

    # Boxplot unico e dettagliato per tutte le feature (x + y)
    feature_cols = [c for c in df.columns if c != "id"]

    def sort_key(col):
        prefix = col[0]
        suffix = col[1:]
        return (prefix, int(suffix) if suffix.isdigit() else suffix)

    ordered_cols = sorted(feature_cols, key=sort_key)
    long_df = df[ordered_cols].melt(var_name="feature", value_name="valore")
    long_df["feature"] = pd.Categorical(long_df["feature"], categories=ordered_cols, ordered=True)

    palette = ["#4575b4" if col.startswith("x") else "#d73027" for col in ordered_cols]

    plt.figure(figsize=(max(12, len(ordered_cols) * 0.6), 6))
    ax = sns.boxplot(data=long_df, x="feature", y="valore", order=ordered_cols, palette=palette, showfliers=True)

    means = stats_df.set_index("colonna").loc[ordered_cols, "mean"]
    ax.scatter(
        np.arange(len(ordered_cols)),
        means,
        marker="D",
        s=40,
        color="#313131",
        label="Media",
        zorder=3,
    )

    for idx, col in enumerate(ordered_cols):
        out_count = int(stats_df.loc[stats_df["colonna"] == col, "outlier_count"].values[0])
        ax.text(
            idx,
            ax.get_ylim()[1] * 0.98,
            f"out={out_count}",
            ha="center",
            va="top",
            fontsize=8,
            rotation=90,
            color="#333333",
        )

    legend_handles = [
        Patch(facecolor="#4575b4", label="Feature input (x)"),
        Patch(facecolor="#d73027", label="Target (y)"),
        Line2D([0], [0], marker="D", color="w", label="Media", markerfacecolor="#313131", markersize=6),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title("Boxplot completo feature di input (x) e target (y)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Valore")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 6. Correlazioni
# ----------------------------

def correlation_analysis(df: pd.DataFrame):
    print("\n=== MATRICE DI CORRELAZIONE COMPLETA ===")
    corr = df.drop(columns=["id"]).corr()
    print(corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Heatmap correlazione feature (x) e target (y)")
    plt.tight_layout()
    plt.show()



# ----------------------------
# 7. Test di normalita' + momenti
# ----------------------------


def chi_square_normality(data: np.ndarray, bins: int = 10):
    """Test chi-quadrato per verificare l'aderenza a una gaussiana stimata dai dati."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    n = data.size
    if n == 0:
        return np.nan, np.nan

    mu = data.mean()
    sigma = data.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan, np.nan

    hist, bin_edges = np.histogram(data, bins=bins)
    cdf = stats.norm.cdf(bin_edges, loc=mu, scale=sigma)
    expected = n * np.diff(cdf)

    mask = expected > 0
    if mask.sum() <= 1:
        return np.nan, np.nan

    chi_stat = np.sum(((hist[mask] - expected[mask]) ** 2) / expected[mask])
    dof = mask.sum() - 1 - 2  # meno i due parametri stimati (mu e sigma)
    if dof <= 0:
        return np.nan, np.nan
    p_value = stats.chi2.sf(chi_stat, dof)
    return chi_stat, p_value


def distribution_tests(df: pd.DataFrame, mu0: float = 0.0, chi_bins: int = 10):
    print("\n=== TEST DI NORMALITA' (Gauss/Chi^2) + TEST T-STUDENT ===")
    numeric_cols = [c for c in df.columns if c != "id"]

    rows = []
    for col in numeric_cols:
        data = df[col].values
        W, p_shapiro = stats.shapiro(data)
        kurt = stats.kurtosis(data)  # 0 se gaussiana (kurtosi di Fisher)

        if len(data) >= 8:
            gauss_stat, gauss_p = stats.normaltest(data)
        else:
            gauss_stat, gauss_p = np.nan, np.nan

        chi_stat, chi_p = chi_square_normality(data, bins=chi_bins)
        t_stat, t_p = stats.ttest_1samp(data, popmean=mu0, nan_policy="omit")

        rows.append(
            {
                "colonna": col,
                "Shapiro_W": W,
                "Shapiro_p": p_shapiro,
                "kurtosis": kurt,
                "Gauss_stat": gauss_stat,
                "Gauss_p": gauss_p,
                "Chi2_stat": chi_stat,
                "Chi2_p": chi_p,
                "t_stat": t_stat,
                "t_p": t_p,
                "gaussiana? (alpha=0.05)": "NO" if p_shapiro < 0.05 else "SI",
            }
        )

    res = pd.DataFrame(rows)
    print(res.sort_values("colonna").to_string(index=False))


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    df = load_data()

    basic_info(df)
    check_missing(df)
    describe_stats(df)
    outlier_analysis(df)
    correlation_analysis(df)
    distribution_tests(df)
