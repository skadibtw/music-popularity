import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import plotly.express as px
from src.music_success_predictor import AudioFeatureExtractor
import warnings

warnings.filterwarnings("ignore")

# --- Настройки страницы ---
st.set_page_config(
    page_title="Music Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Кастомный CSS для красоты ---
st.markdown(
    """
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #ffffff; }
    .stProgress .st-bo { background-color: #ff4b4b; }
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-value { font-size:36px; font-weight: 800; color: #ff4b4b; }
</style>
""",
    unsafe_allow_html=True,
)


# --- Загрузка моделей ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("models/xgboost_music_model.pkl")
        scaler = joblib.load("models/xgboost_scaler.pkl")
        le = joblib.load("models/xgboost_key_encoder.pkl")
        feature_cols = joblib.load("models/xgboost_features.pkl")
        return model, scaler, le, feature_cols
    except Exception as e:
        return None, None, None, None


model, scaler, le, feature_cols = load_models()

# --- Боковая панель ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=100)
    st.title("Hit Predictor AI 🤖")
    st.write(
        "Загрузи свой MP3-файл, и наша модель (XGBoost) предскажет, какова вероятность того, что трек попадет в чарты!"
    )
    st.write("---")
    st.info("Обучено на данных Billboard Hot 100 и Official UK Charts.")

# --- Основной контент ---
st.title("🎵 Анализатор Музыкального Потенциала")
st.write(
    "Узнай, насколько твой трек близок к хитам мировых чартов по аудио-характеристикам."
)

if model is None:
    st.error(
        "❌ Модель не найдена! Сначала обучи ее локально скриптом `train_extended_model.py`."
    )
    st.stop()

uploaded_file = st.file_uploader("Загрузите ваш трек (MP3, WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🚀 Проанализировать трек", use_container_width=True):
        with st.spinner("Извлекаем фичи и анализируем... (Это займет около 30 секунд)"):
            # Сохраняем временно файл
            temp_path = "temp_audio.mp3"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Извлекаем фичи
            features = AudioFeatureExtractor.extract_features(temp_path)

            if features is None:
                st.error("Не удалось обработать аудио. Проверьте формат файла.")
            else:
                # Подготовка к предсказанию
                row = pd.DataFrame([features])
                try:
                    row["key_encoded"] = le.transform([row["key"].iloc[0]])[0]
                except:
                    row["key_encoded"] = le.transform(["C major"])[0]  # Фолбэк

                X = row[feature_cols].values
                X_scaled = scaler.transform(X)

                # Предсказание
                prob = model.predict_proba(X_scaled)[0, 1] * 100
                pred = model.predict(X_scaled)[0]

                # --- РЕЗУЛЬТАТЫ ---
                st.markdown("---")
                st.markdown(
                    '<p class="big-font">🎯 Результат анализа</p>',
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(
                        f'<div class="metric-value">{prob:.1f}%</div>',
                        unsafe_allow_html=True,
                    )
                    st.write("**Вероятность успеха**")
                    st.progress(int(prob))

                    if prob > 70:
                        st.success("⭐⭐⭐ Потенциальный ХИТ!")
                    elif prob > 40:
                        st.warning("⭐⭐ Хороший трек. Есть шансы.")
                    else:
                        st.info("⭐ Андерграунд. Не типичный формат для топ-чартов.")

                with col2:
                    st.write("**Основные характеристики:**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("BPM (Темп)", f"{features['tempo']:.0f}")
                    c2.metric("Тональность", features["key"])
                    c3.metric("Энергия (RMS)", f"{features['rms_mean']:.3f}")

                # --- SHAP Анализ ---
                st.markdown("---")
                st.markdown(
                    '<p class="big-font">💡 Что повлияло на оценку?</p>',
                    unsafe_allow_html=True,
                )

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)

                # Собираем данные для графика
                shap_df = pd.DataFrame(
                    {
                        "Feature": feature_cols,
                        "Impact": shap_values[0]
                        if len(shap_values.shape) > 1
                        else shap_values,
                    }
                )
                # Берем топ-10 по модулю
                shap_df["Abs_Impact"] = shap_df["Impact"].abs()
                shap_df = shap_df.sort_values("Abs_Impact", ascending=False).head(10)
                shap_df["Direction"] = shap_df["Impact"].apply(
                    lambda x: "Увеличило вероятность"
                    if x > 0
                    else "Уменьшило вероятность"
                )

                fig = px.bar(
                    shap_df,
                    x="Impact",
                    y="Feature",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={
                        "Увеличило вероятность": "#2ca02c",
                        "Уменьшило вероятность": "#ff4b4b",
                    },
                    title="Топ факторов в твоем треке (SHAP Values)",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)
