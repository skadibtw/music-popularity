import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


def train_xgboost():
    print("Загрузка расширенных фич...")
    df = pd.read_csv("data/processed/extended_features.csv")

    # Удаляем строки с нанами если есть
    df = df.dropna()

    # Фичи и таргет
    exclude_cols = ["file_path", "success", "peak_rank", "weeks-on-board"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and c != "key"]

    # Кодируем key
    le = LabelEncoder()
    df["key_encoded"] = le.fit_transform(df["key"])
    feature_cols.append("key_encoded")

    X = df[feature_cols].values
    y = df["success"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nОбучение XGBoost модели...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\nРЕЗУЛЬТАТЫ:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    except ValueError:
        pass
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feat_imp)
    plt.title("Топ-15 самых важных аудио-фичей (XGBoost)")
    plt.tight_layout()
    plt.savefig("plots/feature_importance_xgb.png")
    plt.close()

    # Сохраняем модель
    joblib.dump(model, "models/xgboost_music_model.pkl")
    joblib.dump(scaler, "models/xgboost_scaler.pkl")
    joblib.dump(le, "models/xgboost_key_encoder.pkl")
    joblib.dump(feature_cols, "models/xgboost_features.pkl")
    print("\nМодель и scaler сохранены (xgboost_music_model.pkl и т.д.)")
    print("График feature importance сохранен в plots/feature_importance_xgb.png")


if __name__ == "__main__":
    train_xgboost()
