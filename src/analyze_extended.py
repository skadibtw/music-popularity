import sys
import os
import joblib
import pandas as pd
import numpy as np
import shap
from music_success_predictor import AudioFeatureExtractor
import warnings

warnings.filterwarnings("ignore")


def analyze_extended_song(audio_path):
    print("\nЗагрузка моделей XGBoost...")
    try:
        model = joblib.load("models/xgboost_music_model.pkl")
        scaler = joblib.load("models/xgboost_scaler.pkl")
        le = joblib.load("models/xgboost_key_encoder.pkl")
        feature_cols = joblib.load("models/xgboost_features.pkl")
    except Exception as e:
        print(
            f"Ошибка загрузки моделей: {e}. Сначала запустите 'python train_extended_model.py'"
        )
        return

    print("\n" + "=" * 35)
    print("   РАСШИРЕННЫЙ АНАЛИЗ МУЗЫКАЛЬНОГО ТРЕКА (XGBoost + SHAP)")
    print("=" * 35 + "\n")
    print(f"Файл: {audio_path}")
    print("Извлечение фич (может занять около 1 минуты)...")

    features = AudioFeatureExtractor.extract_features(audio_path)
    if features is None:
        print("Ошибка извлечения фич")
        return

    # Подготовка вектора
    row = pd.DataFrame([features])

    # Кодируем key
    try:
        row["key_encoded"] = le.transform([row["key"].iloc[0]])[0]
    except Exception:
        # Неизвестный ключ - берем самый частый
        row["key_encoded"] = le.transform(["C major"])[0]

    # Собираем в нужном порядке
    X = row[feature_cols].values
    X_scaled = scaler.transform(X)

    # Предсказание
    prob = model.predict_proba(X_scaled)[0, 1] * 100
    pred = model.predict(X_scaled)[0]

    print("\n" + "-" * 70)
    bar_length = 50
    filled = int(bar_length * prob / 100)
    bar = "#" * filled + "-" * (bar_length - filled)

    print(f"\nВЕРОЯТНОСТЬ УСПЕХА: {prob:.2f}%")
    print(f"   [{bar}]")
    print(f"\nПредсказание: {'Успешный трек' if pred == 1 else 'Менее успешный трек'}")

    # Интерпретация с SHAP
    print("\n" + "-" * 70)
    print("\nАНАЛИЗ ВЛИЯНИЯ ХАРАКТЕРИСТИК (SHAP):")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Сортируем фичи по абсолютному влиянию
    contribs = []
    for i, col in enumerate(feature_cols):
        val = shap_values[0, i] if len(shap_values.shape) > 1 else shap_values[i]
        contribs.append((col, val))

    contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n   Топ факторов, повлиявших на прогноз:")
    for col, val in contribs[:7]:  # Топ 7 факторов
        direction = "УВЕЛИЧИЛ" if val > 0 else "УМЕНЬШИЛ"
        sign = "+" if val > 0 else ""
        raw_val = row[col].iloc[0]
        if col == "key_encoded":
            col_name = "Тональность"
            raw_val = row["key"].iloc[0]
        else:
            col_name = col
            if isinstance(raw_val, float):
                raw_val = f"{raw_val:.2f}"

        print(f"   {sign}{val:.2f} | {col_name} = {raw_val} -> {direction} вероятность")

    print("\n" + "=" * 35 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python analyze_extended.py <путь_к_mp3_файлу>")
        sys.exit(1)

    audio_path = sys.argv[1]
    analyze_extended_song(audio_path)
