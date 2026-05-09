"""
test_bot.py — Тестовий скрипт для перевірки розпізнавання QTE-бота.

Як користуватися:
  1. Покладіть скріншоти з гри (.png/.jpg) у папку test_images/.
  2. Запустіть:  python test_bot.py
  3. Скрипт виведе, яку літеру розпізнав OCR на кожному зображенні.

Якщо папка test_images/ порожня — скрипт автоматично згенерує
тестові зображення для ВСІХ 26 літер (A–Z).
"""

import os
import sys
import cv2
import numpy as np
import pytesseract
import string

# ---------------------------------------------------------------------------
# Шлях до Tesseract (має збігатися з qte_bot_auto.py)
# ---------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------------------------------------------------------------------------
# Параметри обробки — ДЗЕРКАЛЮЮТЬ qte_bot_auto.py
# ---------------------------------------------------------------------------
THRESHOLD_VALUE = 140
VALID_CHARS = set(string.ascii_lowercase)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Та сама обробка, що й у боті, але для BGR (cv2.imread),
    а не BGRA (mss.grab).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    padded = cv2.copyMakeBorder(inverted, 20, 20, 20, 20,
                                cv2.BORDER_CONSTANT, value=255)
    return padded


def recognize_char(image: np.ndarray) -> str | None:
    """Розпізнає один символ (PSM 10 + whitelist A-Z + 01 fallback)."""
    config = (
        "--psm 10 "
        "-c tesseract_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01"
    )
    text = pytesseract.image_to_string(image, config=config).strip().lower()

    # Фолбек для відомих помилок Tesseract (I↔1, O↔0)
    CONFUSION_MAP = {"0": "o", "1": "i"}
    text = CONFUSION_MAP.get(text, text)

    if len(text) == 1 and text in VALID_CHARS:
        return text
    return None


# ---------------------------------------------------------------------------
# Генерація тестових зображень
# ---------------------------------------------------------------------------
def generate_test_images(out_dir: str):
    """
    Генерує 26 тестових зображень (A–Z), що імітують QTE з гри:
    темний фон, напівпрозоре коло, біла літера.
    """
    os.makedirs(out_dir, exist_ok=True)
    for char in string.ascii_uppercase:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (40, 45, 40)
        cv2.circle(img, (100, 100), 60, (120, 120, 120), 2)
        cv2.circle(img, (100, 100), 59, (80, 80, 80), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        ts = cv2.getTextSize(char, font, 2.2, 4)[0]
        tx = 100 - ts[0] // 2
        ty = 100 + ts[1] // 2
        cv2.putText(img, char, (tx, ty), font, 2.2, (255, 255, 255), 4)
        path = os.path.join(out_dir, f"qte_{char.lower()}.png")
        cv2.imwrite(path, img)
    print(f"   Згенеровано 26 тестових зображень (A–Z) у {out_dir}")


# ---------------------------------------------------------------------------
# Основна логіка тестування
# ---------------------------------------------------------------------------
def main():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")

    print("=" * 60)
    print("🧪  Тестовий скрипт для QTE-бота (Tesseract OCR)")
    print("=" * 60)

    # Перевірка Tesseract
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"Tesseract знайдено: v{ver}")
    except pytesseract.TesseractNotFoundError:
        print("\n❌ ПОМИЛКА: Tesseract OCR не знайдено!")
        print(f"   Очікуваний шлях: {pytesseract.pytesseract.tesseract_cmd}")
        print("   Встановіть його за інструкцією з README.md")
        sys.exit(1)

    print(f"Папка з тестовими зображеннями: {test_dir}")
    print(f"Поріг бінаризації: {THRESHOLD_VALUE}")

    # Перевірка наявності зображень
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    need_generate = False

    if not os.path.isdir(test_dir):
        need_generate = True
    else:
        files_check = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_ext)]
        if not files_check:
            need_generate = True

    if need_generate:
        print(f"\n📸  Тестових зображень не знайдено. Генерую автоматично...")
        generate_test_images(test_dir)

    # Збираємо файли
    files = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(valid_ext)
    ])
    print(f"Знайдено зображень: {len(files)}")

    # Аналізуємо кожне
    total = len(files)
    success = 0
    failures = []

    print(f"\n{'Файл':<25} {'Результат':<15} {'Статус'}")
    print("-" * 55)

    for filepath in files:
        fname = os.path.basename(filepath)
        img = cv2.imread(filepath)
        if img is None:
            print(f"{fname:<25} {'Помилка читання':<15} ❌")
            failures.append(fname)
            continue

        processed = preprocess(img)
        char = recognize_char(processed)

        if char:
            # Спробуємо визначити очікувану літеру з імені файлу (qte_a.png → a)
            expected = None
            base = os.path.splitext(fname)[0].lower()
            if base.startswith("qte_") and len(base) == 5:
                expected = base[-1]

            if expected and char == expected:
                status = "✅"
            elif expected and char != expected:
                status = f"⚠️  (очікувалось {expected.upper()})"
            else:
                status = "✅"

            print(f"{fname:<25} {char.upper():<15} {status}")
            success += 1
        else:
            print(f"{fname:<25} {'—':<15} ❌ Не розпізнано")
            failures.append(fname)

    # Підсумок
    print(f"\n{'=' * 55}")
    print(f"📊  ПІДСУМОК: Розпізнано {success}/{total} зображень")

    if success == total:
        print("🎉  Усі літери розпізнані! Бот готовий до роботи.")
    elif failures:
        print(f"❌  Не вдалось розпізнати: {', '.join(failures)}")
        print("    Поради:")
        print(f"    1. Зменшіть THRESHOLD_VALUE (зараз {THRESHOLD_VALUE}) до 100 або 80.")
        print("    2. Переконайтесь, що на скріншоті є чітка біла літера.")
    print("=" * 55)


if __name__ == "__main__":
    main()
