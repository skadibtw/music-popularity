import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка визуализации
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def run_eda():
    os.makedirs("plots", exist_ok=True)
    print("Загрузка данных...")
    df = pd.read_csv("data/processed/extended_features.csv")

    print("Очистка данных...")
    if df["tempo"].dtype == "object":
        df["tempo"] = df["tempo"].apply(
            lambda x: float(str(x).strip("[]")) if pd.notna(x) else np.nan
        )

    df = df.dropna(subset=["tempo", "key", "popular"])

    print("Генерация графиков...")

    # 1. Корреляционная матрица (числовые фичи)
    plt.figure(figsize=(8, 6))
    corr_cols = [
        "popular",
        "popularity_weight",
        "peak_rank",
        "weeks-on-board",
        "tempo",
        "duration_seconds",
        "rms_mean",
        "spectral_centroid_mean",
    ]
    corr_cols = [col for col in corr_cols if col in df.columns]
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.close()

    # 2. Tempo by source.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="label_source", y="tempo", data=df)
    plt.title("Tempo by label source")
    plt.ylim(50, 200)
    plt.tight_layout()
    plt.savefig("plots/tempo_by_source.png")
    plt.close()

    # 3. Common keys by popular target.
    plt.figure(figsize=(12, 6))
    top_keys = df["key"].value_counts().head(10).index
    sns.countplot(
        data=df[df["key"].isin(top_keys)], x="key", hue="popular", order=top_keys
    )
    plt.title("Common keys: charted vs low-download reference")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/top_keys_popular.png")
    plt.close()

    # 4. Duration by source to expose source-quality artifacts.
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="label_source", y="duration_seconds")
    plt.title("Duration by label source")
    plt.tight_layout()
    plt.savefig("plots/duration_by_source.png")
    plt.close()

    print("EDA завершена. Графики сохранены в папке 'plots'.")


if __name__ == "__main__":
    run_eda()
