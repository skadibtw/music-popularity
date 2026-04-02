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
    df = pd.read_csv("data/processed/tracks_with_features.csv")

    print("Очистка данных...")
    # Парсим темп
    if df["tempo"].dtype == "object":
        df["tempo"] = df["tempo"].apply(
            lambda x: float(str(x).strip("[]")) if pd.notna(x) else np.nan
        )

    df = df.dropna(subset=["tempo", "key"])
    df["success"] = (df["peak_rank"] <= 20).astype(int)

    # Десятилетия
    df["decade"] = (df["year"] // 10) * 10

    print("Генерация графиков...")

    # 1. Корреляционная матрица (числовые фичи)
    plt.figure(figsize=(8, 6))
    corr_cols = ["peak_rank", "weeks-on-board", "tempo", "year", "rank"]
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Корреляционная матрица базовых фичей")
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png")
    plt.close()

    # 2. Темп по десятилетиям
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="decade", y="tempo", data=df)
    plt.title("Изменение темпа (BPM) по десятилетиям")
    plt.ylim(50, 200)
    plt.tight_layout()
    plt.savefig("plots/tempo_by_decade.png")
    plt.close()

    # 3. Топовые тональности для хитов vs не-хитов
    plt.figure(figsize=(12, 6))
    top_keys = df["key"].value_counts().head(10).index
    sns.countplot(
        data=df[df["key"].isin(top_keys)], x="key", hue="success", order=top_keys
    )
    plt.title("Популярные тональности: Топ-20 Хиты vs Остальные")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/top_keys_success.png")
    plt.close()

    # 4. Распределение недель в чарте по тональностям
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df[df["key"].isin(top_keys)], x="key", y="weeks-on-board", order=top_keys
    )
    plt.title("Сколько недель держатся в чарте разные тональности")
    plt.ylim(0, 50)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/weeks_on_board_by_key.png")
    plt.close()

    print("EDA завершена. Графики сохранены в папке 'plots'.")


if __name__ == "__main__":
    run_eda()
