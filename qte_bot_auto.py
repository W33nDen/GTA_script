import mss
import numpy as np
import pydirectinput
import keyboard
import time
import cv2

# ================= НАЛАШТУВАННЯ =================
# Координати області, де з'являється літера (кружечок із QTE).
# ⚠️ Обов'язково підлаштуйте під свою роздільну здатність!
# Дивіться README.md для інструкції, як знайти правильні координати.
MONITOR = {"top": 400, "left": 800, "width": 150, "height": 150}

# Поріг бінаризації (яскравість пікселів). Якщо бот не бачить літеру,
# зменшіть це значення (наприклад, зі 140 до 100).
THRESHOLD_VALUE = 140

# Поріг збігу форми. Чим менше — тим суворіше.
# Не рекомендується ставити вище 0.5.
MATCH_SCORE_LIMIT = 0.25

# Мін/макс розмір контуру (у пікселях). Фільтрує шум та велике коло.
MIN_CONTOUR_SIDE = 15
MAX_CONTOUR_SIDE = 80
# ================================================


def create_internal_templates():
    """
    Створює еталонні геометричні контури для літер W, A, S, D у пам'яті.

    Використовує КІЛЬКА варіантів шрифтів та товщин ліній, щоб покрити
    максимальну кількість можливих ігрових шрифтів. Для порівняння
    використовується метод Hu Moments (cv2.CONTOURS_MATCH_I3), який
    інваріантний до масштабу, повороту та дзеркального відображення.
    """
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ]
    thicknesses = [1, 2, 3, 4, 5, 6]
    scales = [1.5, 2, 2.5, 3]

    templates = {}
    for char in ['w', 'a', 's', 'd']:
        char_contours = []
        for font in fonts:
            for thickness in thicknesses:
                for scale in scales:
                    img = np.zeros((120, 120), dtype=np.uint8)
                    cv2.putText(img, char.upper(), (10, 100), font, scale, 255, thickness)
                    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        char_contours.append(max(contours, key=cv2.contourArea))
        templates[char] = char_contours
    return templates


def process_frame(sct, templates):
    """Захоплює область екрана, шукає літеру QTE та натискає відповідну клавішу."""
    # 1. Захоплення екрана
    screenshot = sct.grab(MONITOR)
    img_np = np.array(screenshot)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)

    # 2. Бінаризація: все яскравіше за THRESHOLD_VALUE стає білим
    _, thresh = cv2.threshold(img_gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # 3. Пошук контурів. RETR_LIST бачить ВСЕ (і коло, і літеру всередині).
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # ФІЛЬТР 1: Розмір — відсіює велике зовнішнє коло та дрібний шум
        if w < MIN_CONTOUR_SIDE or h < MIN_CONTOUR_SIDE:
            continue
        if w > MAX_CONTOUR_SIDE or h > MAX_CONTOUR_SIDE:
            continue

        # ФІЛЬТР 2: Пропорції — літери приблизно квадратні
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.3 or aspect_ratio > 2.5:
            continue

        # 4. Порівняння з усіма варіантами шаблонів (мульти-шрифт)
        best_char = None
        best_score = float('inf')

        for char, char_contours in templates.items():
            for temp_cnt in char_contours:
                # CONTOURS_MATCH_I3 — найстабільніший метод для різних шрифтів
                score = cv2.matchShapes(temp_cnt, cnt, cv2.CONTOURS_MATCH_I3, 0)
                if score < best_score:
                    best_score = score
                    best_char = char

        # 5. Якщо форма достатньо схожа на одну з літер — натискаємо
        if best_score < MATCH_SCORE_LIMIT:
            print(f"[{time.strftime('%H:%M:%S')}] Знайдено: {best_char.upper()} "
                  f"(точність: {best_score:.3f}) → Натискаю!")
            pydirectinput.press(best_char)
            time.sleep(0.5)  # Затримка проти подвійного натискання
            return


def main():
    print("=" * 55)
    print("  Автоматичний бот для QTE міні-ігор")
    print("  Метод: мульти-шрифтовий аналіз контурів (OpenCV)")
    print("=" * 55)
    print("Ініціалізація шаблонів...")

    templates = create_internal_templates()
    total = sum(len(v) for v in templates.values())
    print(f"Згенеровано {total} варіантів шаблонів ({total // 4} на літеру)")

    print("-" * 55)
    print(" Гарячі клавіші:")
    print("   F8  — ЗАПУСК / ПАУЗА сканування")
    print("   F9  — ЗАКРИТИ програму")
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
                process_frame(sct, templates)

            time.sleep(0.01)


if __name__ == "__main__":
    main()
