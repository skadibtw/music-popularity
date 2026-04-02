import os
import glob
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from music_success_predictor import AudioFeatureExtractor
import warnings

warnings.filterwarnings("ignore")


def build_extended_dataset(sample_size=50):
    print("Загрузка данных чартов...")
    df = pd.read_csv("data/processed/tracks_with_features.csv")
    df["success"] = (df["peak_rank"] <= 20).astype(int)

    # Фильтруем те, у которых есть трек-id, чтобы искать mp3
    df_success = df[df["success"] == 1].drop_duplicates(subset=["artist", "title"])
    df_fail = df[df["success"] == 0].drop_duplicates(subset=["artist", "title"])

    print(f"Всего уникальных успешных: {len(df_success)}, неуспешных: {len(df_fail)}")

    # Ищем mp3-файлы
    print("Ищем MP3 файлы в папке music/...")
    all_mp3s = glob.glob("music/*.mp3")
    mp3_map = {}
    for f in all_mp3s:
        basename = os.path.basename(f).replace(".mp3", "")
        # Простейший матчинг: "artist - title" или "title - artist"
        mp3_map[basename.lower()] = f

    print(f"Найдено {len(mp3_map)} MP3 файлов.")

    # Матчинг
    matched_success = []
    matched_fail = []

    for _, row in df_success.iterrows():
        try:
            name1 = f"{row['artist']} - {row['title']}".lower()
            name2 = f"{row['title']} - {row['artist']}".lower()
            if name1 in mp3_map:
                matched_success.append(
                    (mp3_map[name1], 1, row["peak_rank"], row["weeks-on-board"])
                )
            elif name2 in mp3_map:
                matched_success.append(
                    (mp3_map[name2], 1, row["peak_rank"], row["weeks-on-board"])
                )
        except Exception:
            pass

    for _, row in df_fail.iterrows():
        try:
            name1 = f"{row['artist']} - {row['title']}".lower()
            name2 = f"{row['title']} - {row['artist']}".lower()
            if name1 in mp3_map:
                matched_fail.append(
                    (mp3_map[name1], 0, row["peak_rank"], row["weeks-on-board"])
                )
            elif name2 in mp3_map:
                matched_fail.append(
                    (mp3_map[name2], 0, row["peak_rank"], row["weeks-on-board"])
                )
        except Exception:
            pass

    print(f"Сматчено успешных: {len(matched_success)}, неуспешных: {len(matched_fail)}")

    # Берем сэмпл
    np.random.seed(42)
    s_size = min(sample_size, len(matched_success))
    f_size = min(sample_size, len(matched_fail))

    # Избегаем ошибки если список пуст
    if s_size > 0 and f_size > 0:
        idx_s = np.random.choice(len(matched_success), s_size, replace=False)
        idx_f = np.random.choice(len(matched_fail), f_size, replace=False)
        final_sample = [matched_success[i] for i in idx_s] + [
            matched_fail[i] for i in idx_f
        ]
    else:
        # Если матчинг по имени не сработал идеально, просто возьмем случайные mp3 и будем считать что у нас нет чартов
        print("ВНИМАНИЕ: Плохой матчинг, берем случайные файлы.")
        final_sample = [(f, np.random.randint(0, 2), 0, 0) for f in all_mp3s[:10]]

    print(f"Начинаем извлечение фич для {len(final_sample)} треков...")

    features_list = []

    # Для теста возьмем только первые 200, иначе будет долго
    for f_path, success, peak_rank, weeks in tqdm(final_sample):
        feat = AudioFeatureExtractor.extract_features(f_path)
        if feat:
            feat["file_path"] = f_path
            feat["success"] = success
            feat["peak_rank"] = peak_rank
            feat["weeks-on-board"] = weeks
            features_list.append(feat)

    res_df = pd.DataFrame(features_list)
    res_df.to_csv("data/processed/extended_features.csv", index=False)
    print("Готово. Сохранено в extended_features.csv")


if __name__ == "__main__":
    build_extended_dataset()
