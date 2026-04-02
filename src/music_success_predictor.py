"""
Модель для предсказания успешности музыкальных треков
Использует данные из Billboard и UK Charts с аудио-характеристиками
"""

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import os
from pathlib import Path

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Извлечение расширенного набора аудио-фич из MP3 файлов"""
    
    @staticmethod
    def extract_features(audio_path):
        """
        Извлекает аудио-характеристики из файла
        
        Returns:
            dict: словарь с фичами
        """
        try:
            # Загружаем аудио
            y, sr = librosa.load(audio_path, mono=True, duration=30)
            
            features = {}
            
            # 1. Темп (BPM)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # 2. Тональность
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = AudioFeatureExtractor._estimate_key(chroma)
            features['key'] = key
            
            # 3. Спектральные характеристики
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # 4. Zero Crossing Rate (мера перкуссивности)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 5. MFCC (Mel-frequency cepstral coefficients) - тембральные характеристики
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            # 6. Chroma features (гармонические характеристики)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma_stft))
            features['chroma_std'] = float(np.std(chroma_stft))
            
            # 7. RMS Energy (громкость/энергия)
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # 8. Tonnetz (тональное пространство)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features['tonnetz_mean'] = float(np.mean(tonnetz))
            features['tonnetz_std'] = float(np.std(tonnetz))
            
            return features
            
        except Exception as e:
            print(f"Ошибка при обработке {audio_path}: {e}")
            return None
    
    @staticmethod
    def _estimate_key(chroma):
        """Определяет тональность по хромаграмме"""
        # Профили Крамхансла
        MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Нормализация
        chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
        pitch_profile = chroma.mean(axis=1)
        
        def best_match(profile):
            scores = []
            p = profile / profile.sum()
            for i in range(12):
                scores.append(np.corrcoef(p, np.roll(pitch_profile, i))[0, 1])
            key_index = int(np.argmax(scores))
            return key_index, float(np.max(scores))
        
        maj_root, maj_score = best_match(MAJOR_PROFILE)
        min_root, min_score = best_match(MINOR_PROFILE)
        
        if maj_score >= min_score:
            return f"{NOTES[maj_root]} major"
        else:
            return f"{NOTES[min_root]} minor"


class MusicSuccessPredictor:
    """Модель для предсказания успешности музыкальных треков"""
    
    def __init__(self, model_path='models/music_success_model.pkl', scaler_path='models/music_scaler.pkl'):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        self.scaler_path = scaler_path
        
    def prepare_data(self, df_path='data/processed/tracks_with_features.csv'):
        """
        Подготовка данных для обучения
        Создает целевую переменную: успех = топ-20, неуспех = 21+
        """
        print("Загрузка данных...")
        df = pd.read_csv(df_path)
        
        print(f"Всего записей: {len(df)}")
        print(f"Колонки: {df.columns.tolist()}")
        
        # Создаем целевую переменную: 1 = успех (топ-20), 0 = менее успешный
        df['success'] = (df['rank'] <= 20).astype(int)
        
        print(f"\nРаспределение успешных треков:")
        print(df['success'].value_counts())
        
        # Удаляем строки с пропущенными значениями в ключевых колонках
        df = df.dropna(subset=['tempo', 'key'])
        
        # Парсим темп (может быть в формате '[129.19921875]')
        if df['tempo'].dtype == 'object':
            df['tempo'] = df['tempo'].apply(lambda x: float(str(x).strip('[]')) if pd.notna(x) else np.nan)
        
        # Кодируем тональность
        df['key_encoded'] = self.label_encoder.fit_transform(df['key'])
        
        print(f"\nПосле очистки: {len(df)} записей")
        
        return df
    
    def train_model(self, df_path='data/processed/tracks_with_features.csv'):
        """Обучение модели на существующих данных"""
        
        df = self.prepare_data(df_path)
        
        # Базовые фичи (темп и тональность)
        feature_cols = ['tempo', 'key_encoded']
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['success'].values
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Обучение модели
        print("\nОбучение модели...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Оценка
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        print("="*50)
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Неуспешный', 'Успешный']))
        
        # Важность признаков
        print("\nВажность признаков:")
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            print(f"  {name}: {importance:.4f}")
        
        # Сохранение модели
        self.save_model()
        
        return self.model, self.scaler
    
    def predict_from_file(self, audio_path):
        """
        Предсказание успешности трека из MP3 файла
        
        Args:
            audio_path: путь к MP3 файлу
            
        Returns:
            dict: процент вероятности успеха и детали
        """
        if self.model is None:
            self.load_model()
        
        print(f"\nАнализ файла: {audio_path}")
        print("-" * 50)
        
        # Извлекаем фичи
        features = AudioFeatureExtractor.extract_features(audio_path)
        if features is None:
            return {"error": "Не удалось извлечь характеристики из файла"}
        
        # Кодируем тональность
        try:
            key_encoded = self.label_encoder.transform([features['key']])[0]
        except:
            # Если тональность не встречалась при обучении, используем среднее
            key_encoded = len(self.label_encoder.classes_) // 2
        
        # Подготавливаем фичи для предсказания
        X = np.array([[features['tempo'], key_encoded]])
        X_scaled = self.scaler.transform(X)
        
        # Предсказание
        success_probability = self.model.predict_proba(X_scaled)[0, 1]
        prediction = self.model.predict(X_scaled)[0]
        
        result = {
            'success_probability': float(success_probability * 100),
            'prediction': 'Успешный трек' if prediction == 1 else 'Менее успешный трек',
            'features': {
                'tempo': features['tempo'],
                'key': features['key'],
                'spectral_centroid_mean': features.get('spectral_centroid_mean', 'N/A'),
                'zcr_mean': features.get('zcr_mean', 'N/A'),
                'rms_mean': features.get('rms_mean', 'N/A')
            }
        }
        
        return result
    
    def save_model(self):
        """Сохранение модели и scaler"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoder, 'models/key_encoder.pkl')
        print(f"\n✓ Модель сохранена: {self.model_path}")
        print(f"✓ Scaler сохранен: {self.scaler_path}")
        print(f"✓ Encoder сохранен: key_encoder.pkl")
    
    def load_model(self):
        """Загрузка сохраненной модели"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder = joblib.load('models/key_encoder.pkl')
            self.feature_names = ['tempo', 'key_encoded']
            print("✓ Модель загружена")
        else:
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")


def main():
    """Основная функция для обучения и демонстрации"""
    
    predictor = MusicSuccessPredictor()
    
    # Обучение модели
    if not os.path.exists('models/music_success_model.pkl'):
        print("=" * 70)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 70)
        predictor.train_model('data/processed/tracks_with_features.csv')
    else:
        predictor.load_model()
        print("Модель уже обучена и загружена")
    
    # Демонстрация предсказания на примерах из папки music
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ")
    print("=" * 70)
    
    music_dir = Path('music')
    if music_dir.exists():
        mp3_files = list(music_dir.glob('*.mp3'))[:5]  # Берем первые 5 файлов для демо
        
        for audio_file in mp3_files:
            result = predictor.predict_from_file(str(audio_file))
            
            if 'error' not in result:
                print(f"\nФайл: {audio_file.name}")
                print(f"Вероятность успеха: {result['success_probability']:.2f}%")
                print(f"Предсказание: {result['prediction']}")
                print(f"Темп: {result['features']['tempo']:.1f} BPM")
                print(f"Тональность: {result['features']['key']}")
    else:
        print("Папка 'music' не найдена")
    
    print("\n" + "=" * 70)
    print("Готово! Используйте predictor.predict_from_file('путь_к_файлу.mp3')")
    print("для анализа новых треков")
    print("=" * 70)


if __name__ == "__main__":
    main()
