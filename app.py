import streamlit as st
import pandas as pd
import joblib
import shap
import os
import tempfile
import plotly.express as px
from src.research_insights import (
    DEFAULT_MODEL_REPORT_PATH,
    DEFAULT_QC_REPORT_PATH,
    build_dataset_snapshot,
    build_research_summary,
    compare_track_to_reference,
    load_json_report,
)
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


@st.cache_data
def load_reports():
    return {
        "qc": load_json_report(DEFAULT_QC_REPORT_PATH),
        "model": load_json_report(DEFAULT_MODEL_REPORT_PATH),
    }


@st.cache_data
def load_research_summary():
    return build_research_summary()


model, feature_cols, metadata = load_models()


def format_optional_number(value, digits=3):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_optional_percent(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def score_band(score):
    if score >= 70:
        return "High similarity to the charted audio reference profile", "success"
    if score >= 40:
        return "Middle range: some traits resemble the charted reference", "warning"
    return "Low similarity to the charted audio reference profile", "info"


def render_feature_comparison(row, metadata, research_summary):
    feature_frame = pd.DataFrame(research_summary.get("feature_profiles", []))
    if feature_frame.empty:
        feature_names = [name for name in ["tempo", "rms_mean", "chroma_mean", "tonnetz_mean", "zcr_std"] if name in row.columns]
    else:
        feature_names = feature_frame["feature"].dropna().astype(str).head(10).tolist()

    comparisons = compare_track_to_reference(row, metadata, feature_names)
    if not comparisons:
        st.info("Feature comparison is unavailable for this track and metadata combination.")
        return

    comparison_df = pd.DataFrame(comparisons)
    st.dataframe(
        comparison_df[["label", "value", "reference_median", "reference_low", "reference_high", "status"]],
        use_container_width=True,
        hide_index=True,
    )

# --- Боковая панель ---
with st.sidebar:
    st.title("Hit-Likeness Analyzer")
    st.write(
        "Upload an MP3 or WAV file. The app compares its local audio features with the current chart-reference dataset."
    )
    st.write("---")
    st.info("Score = audio similarity percentile against the weighted charted reference distribution, not commercial success probability.")

if model is None:
    st.error(
        "Model artifacts are missing. Train locally with `src/train_extended_model.py` first."
    )
    st.stop()

reports = load_reports()
research_summary = load_research_summary()
dataset_snapshot = build_dataset_snapshot(reports["qc"], reports["model"])

analyze_tab, research_tab, method_tab = st.tabs(
    ["Analyze", "Research insights", "Method & limits"]
)

with analyze_tab:
    st.title("Music Hit-Likeness Analyzer")
    st.write(
        "Scores how close an uploaded track is to the weighted charted audio reference profile, relative to the current training dataset."
    )

    snapshot_cols = st.columns(4)
    snapshot_cols[0].metric(
        "Feature rows",
        dataset_snapshot.get("feature_rows") or "N/A",
    )
    snapshot_cols[1].metric(
        "Charted rows",
        dataset_snapshot.get("charted_rows") or "N/A",
    )
    snapshot_cols[2].metric(
        "Low-download rows",
        dataset_snapshot.get("low_download_rows") or "N/A",
    )
    snapshot_cols[3].metric(
        "Extraction success",
        format_optional_percent(dataset_snapshot.get("feature_extraction_success_rate")),
    )

    st.info(
        "Score = audio similarity percentile against the weighted charted reference distribution, not commercial success probability."
    )
    source_auc = dataset_snapshot.get("source_auc")
    if source_auc is not None and source_auc >= 0.8:
        st.warning(
            f"Source/domain separability is high (audio source AUC {source_auc:.3f}), so results can reflect collection-path artifacts as well as audio traits."
        )
    else:
        st.info(
            f"Audio source/domain AUC: {format_optional_number(source_auc)}."
        )
    st.warning(
        "The model does not know marketing, artist reputation, release timing, or platform exposure. "
        "Use the result as a comparative audio score, not a chart-success forecast."
    )

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

                    st.markdown("---")
                    st.subheader("Guided analysis report")

                    band_text, band_type = score_band(score)
                    score_cols = st.columns(3)
                    score_cols[0].metric("Hit-likeness percentile", f"{score:.1f}%")
                    score_cols[1].metric("XGBoost diagnostic percentile", f"{model_percentile:.1f}%")
                    score_cols[2].metric("Charted-reference distance", f"{charted_distance:.2f}")

                    st.progress(int(max(0, min(score, 100))))
                    if band_type == "success":
                        st.success(band_text)
                    elif band_type == "warning":
                        st.warning(band_text)
                    else:
                        st.info(band_text)

                    if in_distribution:
                        st.success(f"Reliability: within training feature range ({distance:.2f}/{threshold:.2f}).")
                    else:
                        st.warning(
                            f"Low reliability: this track is far from the training feature range ({distance:.2f}/{threshold:.2f})."
                        )

                    st.caption(
                        f"Raw XGBoost score: {raw_score * 100:.1f}%. "
                        "The XGBoost signal is shown as a diagnostic model output, not as the headline score."
                    )

                    st.write("**Extracted audio profile**")
                    audio_cols = st.columns(6)
                    audio_cols[0].metric("BPM", f"{features['tempo']:.0f}")
                    audio_cols[1].metric("Key", features["key"])
                    audio_cols[2].metric("RMS mean", f"{features['rms_mean']:.3f}")
                    audio_cols[3].metric("Chroma mean", f"{features.get('chroma_mean', 0):.3f}")
                    audio_cols[4].metric("Tonnetz mean", f"{features.get('tonnetz_mean', 0):.3f}")
                    audio_cols[5].metric("ZCR std", f"{features.get('zcr_std', 0):.3f}")

                    st.write("**Uploaded track vs charted reference band**")
                    render_feature_comparison(row, metadata, research_summary)

                    # --- SHAP Анализ ---
                    st.markdown("---")
                    st.subheader("Diagnostic model contributions")
                    st.caption(
                        "SHAP values explain the secondary XGBoost diagnostic model in its raw feature space. "
                        "They are not causal production advice."
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
                        title="Top diagnostic XGBoost feature contributions (SHAP)",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(
                        "Research context: the feature comparison is anchored to the current charted-reference dataset. "
                        "Source/domain bias remains high, so treat this as exploratory audio similarity rather than a popularity forecast."
                    )

                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.remove(temp_path)

with research_tab:
    st.subheader("Research Insights")
    st.write(
        "Dataset-level audio feature summaries for charted reference tracks and low-download archive tracks."
    )

    feature_profiles = research_summary.get("feature_profiles", [])
    if feature_profiles:
        profiles_df = pd.DataFrame(feature_profiles)
        display_cols = [
            "label",
            "charted_median",
            "charted_q25",
            "charted_q75",
            "top20_median",
            "chart_21_50_median",
            "chart_51_100_median",
            "low_stream_median",
        ]
        profiles_df = profiles_df[[col for col in display_cols if col in profiles_df.columns]]
        profiles_df = profiles_df.rename(
            columns={
                "label": "Feature",
                "charted_median": "Charted median",
                "charted_q25": "Charted q25",
                "charted_q75": "Charted q75",
                "top20_median": "Top 20 median",
                "chart_21_50_median": "21-50 median",
                "chart_51_100_median": "51-100 median",
                "low_stream_median": "Low-stream median",
            }
        )
        st.dataframe(profiles_df, use_container_width=True, hide_index=True)
    else:
        st.info("No feature profile summary is available.")

    key_distribution = research_summary.get("key_distribution", [])
    if key_distribution:
        key_df = pd.DataFrame(key_distribution)
        key_fig = px.bar(
            key_df,
            x="key",
            y=["all_count", "charted_count"],
            barmode="group",
            labels={
                "key": "Key",
                "value": "Tracks",
                "variable": "Group",
            },
            title="Most Common Keys in the Dataset",
        )
        key_fig.update_layout(legend_title_text="Group")
        st.plotly_chart(key_fig, use_container_width=True)
    else:
        st.info("No key distribution summary is available.")

    st.caption("Full research notes: [reports/popular_feature_research.md](reports/popular_feature_research.md)")

with method_tab:
    st.subheader("Method & Limits")
    st.write(
        "The primary score is a distance-based percentile against a weighted charted audio-feature reference profile. "
        "The XGBoost output remains visible as a secondary diagnostic signal, but it is not presented as a probability of commercial success."
    )
    st.write(
        "The current dataset compares charted references with low-download Internet Archive tracks. "
        "Validation metrics therefore mix useful audio-pattern signal with source and collection-path effects."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Distance score AUC",
        format_optional_number(dataset_snapshot.get("distance_score_auc")),
    )
    metric_cols[1].metric(
        "XGBoost OOF AUC",
        format_optional_number(dataset_snapshot.get("xgboost_oof_auc")),
    )
    metric_cols[2].metric(
        "Audio source AUC",
        format_optional_number(dataset_snapshot.get("source_auc")),
    )
    metric_cols[3].metric(
        "Format source AUC",
        format_optional_number(dataset_snapshot.get("quality_source_auc")),
    )

    st.warning(
        "Non-modeled commercial factors such as promotion, audience size, playlist placement, release timing, social media activity, and artist reputation are outside this model."
    )
    st.info(
        "Next data milestone: refresh the reference dataset with better-balanced charted and non-charted sources, then track source/domain diagnostics beside every model metric."
    )
