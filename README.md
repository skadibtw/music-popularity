# Music Success Predictor

This repository contains a full Machine Learning pipeline to analyze MP3 features and predict whether a song will achieve chart success using an XGBoost model. It employs local audio feature extraction using `librosa` and provides a web application frontend built with `Streamlit`.

## Features
- Complete dataset pipeline extracting 40+ musical features per track (MFCCs, spectral roll-off, zero-crossing rate).
- AMD ROCm PyTorch integration for rapid, accelerated feature parsing.
- Trained XGBoost classifier.
- Real-time drag-and-drop song evaluation Streamlit Web UI.
- Explainable AI using SHAP.

## Directory Structure
- `src/`: Python scripts for feature extraction, model training, and predictions.
- `data/`: Processed dataset files.
- `models/`: Exported model binaries and scalers.
- `app.py`: Streamlit frontend application.

## Quickstart

Run the web app:
```bash
streamlit.exe run app.py
```
