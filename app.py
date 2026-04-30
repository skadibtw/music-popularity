import streamlit as st
import pandas as pd
import joblib
import shap
import os
import tempfile
import plotly.express as px
from src.music_success_predictor import (
    AudioFeatureExtractor,
    DEFAULT_PREVIEW_SECONDS,
    add_key_features,
    charted_similarity_percentile,
    robust_feature_distance,
    score_percentile,
)
import warnings

warnings.filterwarnings("ignore")

# --- Настройки страницы ---
st.set_page_config(
    page_title="Music Hit-Likeness Analyzer",
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
        feature_cols = joblib.load("models/xgboost_features.pkl")
        metadata = joblib.load("models/xgboost_score_metadata.pkl")
        return model, feature_cols, metadata
    except Exception as e:
        return None, None, None


model, feature_cols, metadata = load_models()

# --- Боковая панель ---
with st.sidebar:
    st.title("Hit-Likeness Analyzer")
    st.write(
        "Upload an MP3 or WAV file. The app compares its local audio features with the current chart-reference dataset."
    )
    st.write("---")
    st.info("Score = audio similarity percentile against the weighted charted reference distribution, not commercial success probability.")

# --- Основной контент ---
st.title("Music Hit-Likeness Analyzer")
st.write(
    "Scores how close an uploaded track is to the weighted charted audio reference profile, relative to the current training dataset."
)

if model is None:
    st.error(
        "Model artifacts are missing. Train locally with `src/train_extended_model.py` first."
    )
    st.stop()

with st.expander("How to read the score"):
    st.write(
        "The main number is a percentile of closeness to the weighted charted audio-feature distribution. "
        "For example, 80% means the track is closer to the charted reference profile than roughly 80% of the current reference tracks."
    )
    st.write(
        "The XGBoost percentile is shown separately as a secondary signal because the current audio-only classifier remains source-confounded."
    )
    if metadata:
        source_counts = metadata.get("label_source_counts", {})
        st.write(
            f"Reference set: {metadata.get('n_tracks', 'N/A')} training tracks, "
            f"charted base rate: {metadata.get('base_rate', 0) * 100:.1f}%, "
            f"OOF ROC-AUC: {metadata.get('oof_auc', 0):.3f}."
        )
        if source_counts:
            st.write(f"Label sources: {source_counts}")

st.warning(
    "The model does not know marketing, artist reputation, release timing, or platform exposure. "
    "Use the result as a comparative audio score, not a chart-success forecast."
)

uploaded_file = st.file_uploader("Upload a track (MP3, WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Analyze track", use_container_width=True):
        with st.spinner(f"Extracting features from the first {DEFAULT_PREVIEW_SECONDS} seconds..."):
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = temp_file.name
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()

            features = AudioFeatureExtractor.extract_features(
                temp_path,
                preview_seconds=DEFAULT_PREVIEW_SECONDS,
            )

            if features is None:
                st.error("Audio processing failed. Check the file format and try again.")
            else:
                row = pd.DataFrame([features])
                row = add_key_features(row)
                X = row[feature_cols]

                raw_score = float(model.predict_proba(X)[0, 1])
                model_percentile = score_percentile(raw_score, metadata)
                score, charted_distance = charted_similarity_percentile(X, metadata)
                distance, threshold, in_distribution = robust_feature_distance(X, metadata)
                pred = model.predict(X)[0]

                # --- РЕЗУЛЬТАТЫ ---
                st.markdown("---")
                st.markdown(
                    '<p class="big-font">Analysis Result</p>',
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(
                        f'<div class="metric-value">{score:.1f}%</div>',
                        unsafe_allow_html=True,
                    )
                    st.write("**Hit-likeness score**")
                    st.progress(int(score))
                    st.caption(
                        f"Charted-reference distance: {charted_distance:.2f}; "
                        f"XGBoost percentile: {model_percentile:.1f}%; raw model score: {raw_score * 100:.1f}%"
                    )

                    if score > 70:
                        st.success("Close to the weighted charted audio reference profile.")
                    elif score > 40:
                        st.warning("Middle range: some features resemble charted reference tracks.")
                    else:
                        st.info("Far from the charted audio reference profile with the current features.")

                    if in_distribution:
                        st.caption(f"Reliability: within training feature range ({distance:.2f}/{threshold:.2f}).")
                    else:
                        st.warning(
                            f"Low reliability: this track is far from the training feature range ({distance:.2f}/{threshold:.2f})."
                        )

                with col2:
                    st.write("**Audio Features:**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("BPM", f"{features['tempo']:.0f}")
                    c2.metric("Key", features["key"])
                    c3.metric("RMS Energy", f"{features['rms_mean']:.3f}")

                # --- SHAP Анализ ---
                st.markdown("---")
                st.markdown(
                    '<p class="big-font">Feature Contributions</p>',
                    unsafe_allow_html=True,
                )

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

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
                    lambda x: "Increased score"
                    if x > 0
                    else "Decreased score"
                )

                fig = px.bar(
                    shap_df,
                    x="Impact",
                    y="Feature",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={
                        "Increased score": "#2ca02c",
                        "Decreased score": "#ff4b4b",
                    },
                    title="Top XGBoost feature contributions (SHAP)",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)
