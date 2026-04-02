"""
Простой скрипт для анализа загруженного MP3 файла
Использование: python analyze_song.py path/to/song.mp3
"""

import sys
from music_success_predictor import MusicSuccessPredictor


def analyze_song(audio_path):
    """Анализирует один MP3 файл и выводит результат"""
    
    predictor = MusicSuccessPredictor()
    
    try:
        predictor.load_model()
    except FileNotFoundError:
        print("❌ Модель не найдена. Сначала запустите обучение:")
        print("   python music_success_predictor.py")
        return
    
    print("\n" + "🎵" * 35)
    print("   АНАЛИЗ МУЗЫКАЛЬНОГО ТРЕКА")
    print("🎵" * 35 + "\n")
    
    result = predictor.predict_from_file(audio_path)
    
    if 'error' in result:
        print(f"❌ Ошибка: {result['error']}")
        return
    
    # Красивый вывод результата
    success_prob = result['success_probability']
    
    print(f"📁 Файл: {audio_path}")
    print("\n" + "─" * 70)
    
    # Визуальный индикатор вероятности
    bar_length = 50
    filled = int(bar_length * success_prob / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\n🎯 ВЕРОЯТНОСТЬ УСПЕХА: {success_prob:.2f}%")
    print(f"   [{bar}]")
    print(f"\n📊 Предсказание: {result['prediction']}")
    
    print("\n" + "─" * 70)
    print("\n🎼 МУЗЫКАЛЬНЫЕ ХАРАКТЕРИСТИКИ:\n")
    print(f"   ⏱️  Темп (BPM):        {result['features']['tempo']:.1f}")
    print(f"   🎹 Тональность:       {result['features']['key']}")
    
    if result['features'].get('spectral_centroid_mean') != 'N/A':
        print(f"   🌈 Спектр. центроид:  {result['features']['spectral_centroid_mean']:.1f}")
    if result['features'].get('zcr_mean') != 'N/A':
        print(f"   📈 Zero Cross Rate:   {result['features']['zcr_mean']:.4f}")
    if result['features'].get('rms_mean') != 'N/A':
        print(f"   🔊 RMS энергия:       {result['features']['rms_mean']:.4f}")
    
    print("\n" + "─" * 70)
    
    # Интерпретация результата
    print("\n💡 ИНТЕРПРЕТАЦИЯ:\n")
    if success_prob >= 70:
        print("   ⭐⭐⭐ Отличный потенциал! У этого трека высокие шансы")
        print("   попасть в топ-20 чартов.")
    elif success_prob >= 50:
        print("   ⭐⭐ Хороший потенциал. Трек имеет средние шансы успеха.")
    elif success_prob >= 30:
        print("   ⭐ Умеренный потенциал. Требуется дополнительная работа")
        print("   над аранжировкой или продвижением.")
    else:
        print("   💭 Низкий потенциал по аудио-характеристикам. Возможно,")
        print("   стоит пересмотреть темп, тональность или структуру трека.")
    
    print("\n" + "🎵" * 35 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python analyze_song.py <путь_к_mp3_файлу>")
        print("\nПример:")
        print("  python analyze_song.py music/my_song.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    analyze_song(audio_path)
