"""
qte_bot_auto.py — Автоматичний бот для QTE міні-ігор.

Розпізнає БУДЬ-ЯКУ літеру (A–Z) через Tesseract OCR (режим одиночного символу)
та натискає відповідну клавішу через DirectInput.

Залежності:  pip install mss numpy opencv-python pytesseract pydirectinput keyboard
Додатково:   Tesseract OCR (див. README.md для інструкції зі встановлення)
"""

import mss
import numpy as np
import cv2
import pytesseract
import pydirectinput
import keyboard
import time
import string

# ======================== НАЛАШТУВАННЯ ========================

# Шлях до Tesseract OCR (Windows).
# Якщо ви встановили Tesseract за іншим шляхом — змініть тут.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Координати зони на екрані, де з'являється літера (кружечок QTE).
# ⚠️  ОБОВ'ЯЗКОВО підлаштуйте під свою роздільну здатність!
# Дивіться README.md для покрокової інструкції.
MONITOR = {"top": 400, "left": 800, "width": 150, "height": 150}

# Поріг бінаризації (яскравість). Якщо бот не бачить літеру — зменшіть (100, 80).
THRESHOLD_VALUE = 140

# Множина допустимих символів
VALID_CHARS = set(string.ascii_lowercase)  # {'a', 'b', ..., 'z'}

# =============================================================


def preprocess(img_bgra: np.ndarray) -> np.ndarray:
    """
    Підготовлює скріншот для Tesseract:
      1. Переводить у відтінки сірого.
      2. Бінаризує (біла літера → біла, темний фон → чорний).
      3. Інвертує (Tesseract найкраще працює з ТЕМНИМ текстом на БІЛОМУ фоні).
      4. Додає білу рамку навколо (padding) — Tesseract потребує відступи.
    """
    gray = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    padded = cv2.copyMakeBorder(inverted, 20, 20, 20, 20,
                                cv2.BORDER_CONSTANT, value=255)
    return padded


def recognize_char(image: np.ndarray) -> str | None:
    """
    Розпізнає один символ на зображенні.

    Використовує Tesseract у режимі PSM 10 (одиночний символ) із білим списком
    A–Z + цифри 0/1 (Tesseract часто плутає I↔1 та O↔0).
    """
    config = (
        "--psm 10 "
        "-c tesseract_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01"
    )
    text = pytesseract.image_to_string(image, config=config).strip().lower()

    # Фолбек для відомих помилок Tesseract
    CONFUSION_MAP = {"0": "o", "1": "i"}
    text = CONFUSION_MAP.get(text, text)

    if len(text) == 1 and text in VALID_CHARS:
        return text
    return None


def process_frame(sct):
    """Захоплює область екрана, розпізнає літеру та натискає клавішу."""
    screenshot = sct.grab(MONITOR)
    img_np = np.array(screenshot)

    processed = preprocess(img_np)
    char = recognize_char(processed)

    if char:
        print(f"[{time.strftime('%H:%M:%S')}] Розпізнано: {char.upper()} → Натискаю!")
        pydirectinput.press(char)
        time.sleep(0.3)  # Затримка проти подвійного натискання


def main():
    print("=" * 55)
    print("  QTE Bot — Автоматичне проходження міні-ігор")
    print("  OCR: Tesseract (PSM 10 — режим одиночного символу)")
    print("=" * 55)

    # Перевірка доступності Tesseract
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"  Tesseract знайдено: v{ver}")
    except pytesseract.TesseractNotFoundError:
        print("\n❌ ПОМИЛКА: Tesseract OCR не знайдено!")
        print("   Встановіть його за інструкцією з README.md")
        print(f"   Очікуваний шлях: {pytesseract.pytesseract.tesseract_cmd}")
        return

    print("-" * 55)
    print("  Гарячі клавіші:")
    print("    F8  — ЗАПУСК / ПАУЗА сканування")
    print("    F9  — ЗАКРИТИ програму")
    print("-" * 55)

    running = False

    with mss.MSS() as sct:
        while True:
            if keyboard.is_pressed('f8'):
                running = not running
                state = "🟢 СКАНУВАННЯ УВІМКНЕНО" if running else "🔴 СКАНУВАННЯ ПРИЗУПИНЕНО"
                print(f"\n{state}")
                time.sleep(0.3)

            if keyboard.is_pressed('f9'):
                print("\nРоботу бота завершено. Гарної гри!")
                break

            if running:
                process_frame(sct)

            time.sleep(0.01)


if __name__ == "__main__":
    main()
