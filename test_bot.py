"""
test_bot.py — Тестовий скрипт для перевірки логіки розпізнавання QTE-бота.

Як користуватися:
  1. Створіть папку  test_images/  поруч із цим файлом.
  2. Покладіть туди .png або .jpg скріншоти з гри (той самий фрагмент,
     який потрапляє у MONITOR бота — кружечок з літерою).
  3. Запустіть:  python test_bot.py
  4. У консолі побачите детальний звіт по кожному зображенню.

Якщо в папці test_images/ немає зображень, скрипт автоматично
згенерує 4 тестові зображення, що імітують QTE з гри.
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Генерація шаблонів — ДЗЕРКАЛЮЄ логіку з qte_bot_auto.py
# (мульти-шрифтовий підхід: 5 шрифтів × 6 товщин × 4 масштаби)
# ---------------------------------------------------------------------------
THRESHOLD_VALUE = 140        # Поріг бінаризації  (такий самий, як у боті)
MATCH_SCORE_LIMIT = 0.25     # Поріг збігу        (такий самий, як у боті)
MIN_SIDE = 15                # Мін. розмір контуру
MAX_SIDE = 80                # Макс. розмір контуру


def create_internal_templates():
    """
    Створює еталонні геометричні контури для літер W, A, S, D у пам'яті.
    Повністю повторює логіку з qte_bot_auto.py.
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


# ---------------------------------------------------------------------------
# Генерація тестових зображень, що імітують QTE з гри
# ---------------------------------------------------------------------------
def generate_test_images(out_dir):
    """Генерує 4 тестові зображення (W, A, S, D) з темним фоном і колом."""
    os.makedirs(out_dir, exist_ok=True)
    for char in ['W', 'A', 'S', 'D']:
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        img[:] = (40, 45, 40)   # Темно-зелений фон (як у GTA)
        cv2.circle(img, (75, 75), 45, (120, 120, 120), 2)
        cv2.circle(img, (75, 75), 44, (80, 80, 80), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        ts = cv2.getTextSize(char, font, 1.8, 2)[0]
        tx = 75 - ts[0] // 2
        ty = 75 + ts[1] // 2
        cv2.putText(img, char, (tx, ty), font, 1.8, (255, 255, 255), 2)
        path = os.path.join(out_dir, f"qte_{char.lower()}.png")
        cv2.imwrite(path, img)
    print(f"   Згенеровано 4 тестові зображення у {out_dir}")


# ---------------------------------------------------------------------------
# Аналіз одного зображення — ДЗЕРКАЛЮЄ process_frame
# ---------------------------------------------------------------------------
def analyze_image(filepath: str, templates: dict) -> dict:
    """
    Прогоняє одне зображення через ту саму логіку, що й бот:
      threshold → findContours → matchShapes (I3, multi-font).
    """
    img = cv2.imread(filepath)
    if img is None:
        return {"file": filepath, "error": "Не вдалося прочитати файл"}

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results = {
        "file": os.path.basename(filepath),
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "total_contours": len(contours),
        "candidates": [],
        "detected": None,
    }

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < MIN_SIDE or h < MIN_SIDE or w > MAX_SIDE or h > MAX_SIDE:
            continue

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.3 or aspect_ratio > 2.5:
            continue

        # --- Мульти-шрифтове порівняння (метод I3) ---
        scores_best = {}
        best_char = None
        best_score = float('inf')

        for char, char_contours in templates.items():
            min_score_for_char = float('inf')
            for temp_cnt in char_contours:
                score = cv2.matchShapes(temp_cnt, cnt, cv2.CONTOURS_MATCH_I3, 0)
                min_score_for_char = min(min_score_for_char, score)
            scores_best[char] = round(min_score_for_char, 4)
            if min_score_for_char < best_score:
                best_score = min_score_for_char
                best_char = char

        candidate = {
            "bbox": f"x={x} y={y} w={w} h={h}",
            "aspect_ratio": round(aspect_ratio, 2),
            "best_char": best_char.upper(),
            "best_score": round(best_score, 4),
            "all_scores": scores_best,
            "accepted": best_score < MATCH_SCORE_LIMIT,
        }
        results["candidates"].append(candidate)

        if candidate["accepted"] and results["detected"] is None:
            results["detected"] = best_char.upper()

    return results


# ---------------------------------------------------------------------------
# Форматований вивід
# ---------------------------------------------------------------------------
def print_report(res: dict):
    """Красиво друкує результат аналізу одного зображення."""

    print(f"\n{'=' * 60}")
    print(f"📄 Файл: {res['file']}")

    if "error" in res:
        print(f"   ❌ ПОМИЛКА: {res['error']}")
        return

    print(f"   Роздільна здатність: {res['resolution']}")
    print(f"   Знайдено контурів (усього): {res['total_contours']}")
    print(f"   Кандидатів після фільтрів:  {len(res['candidates'])}")

    if not res["candidates"]:
        print("   ⚠️  Жоден контур не пройшов фільтри розміру/пропорцій.")
        print("       → Спробуйте зменшити THRESHOLD_VALUE (наприклад, зі 140 до 100)")
        print("       → або перевірте, чи зображення містить літеру у правильній області.")
        return

    for i, c in enumerate(res["candidates"], 1):
        status = "✅ ПРИЙНЯТО" if c["accepted"] else "❌ Відхилено (score > 0.25)"
        print(f"\n   --- Кандидат #{i} ---")
        print(f"   Рамка:       {c['bbox']}")
        print(f"   Пропорції:   {c['aspect_ratio']}")
        print(f"   Найкращий:   {c['best_char']}  (score = {c['best_score']})")
        print(f"   Статус:      {status}")
        print(f"   Усі score:   W={c['all_scores']['w']}  A={c['all_scores']['a']}  "
              f"S={c['all_scores']['s']}  D={c['all_scores']['d']}")

        if c["accepted"] and c["best_score"] > 0.18:
            print(f"   ⚠️  Score близький до ліміту (0.25). "
                  f"Можливі неточності у реальній грі.")

    print(f"\n   🎯 ФІНАЛЬНИЙ РЕЗУЛЬТАТ: ", end="")
    if res["detected"]:
        print(f"Бот натисне  [ {res['detected']} ]")
    else:
        print("Бот НЕ натисне нічого (жоден кандидат не пройшов поріг 0.25)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")

    print("=" * 60)
    print("🧪  Тестовий скрипт для QTE-бота (мульти-шрифтовий метод)")
    print("=" * 60)
    print(f"Папка з тестовими зображеннями: {test_dir}")
    print(f"Поріг бінаризації (THRESHOLD):   {THRESHOLD_VALUE}")
    print(f"Поріг збігу (MATCH_SCORE_LIMIT): {MATCH_SCORE_LIMIT}")
    print(f"Метод порівняння:                CONTOURS_MATCH_I3")

    # Перевіряємо наявність папки та зображень
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

    print(f"\nЗнайдено зображень: {len(files)}")

    # Генеруємо шаблони
    print("Генерація мульти-шрифтових шаблонів...")
    templates = create_internal_templates()
    total_variants = sum(len(v) for v in templates.values())
    print(f"Згенеровано {total_variants} варіантів ({total_variants // 4} на літеру)")

    # Аналізуємо
    total = len(files)
    detected_count = 0

    for filepath in files:
        res = analyze_image(filepath, templates)
        print_report(res)
        if res.get("detected"):
            detected_count += 1

    # Підсумок
    print(f"\n{'=' * 60}")
    print(f"📊  ПІДСУМОК: Розпізнано {detected_count}/{total} зображень")
    if detected_count == total:
        print("🎉  Усі зображення розпізнані успішно! Бот готовий до роботи.")
    elif detected_count == 0:
        print("😞  Жодне зображення не розпізнано.")
        print("    Поради:")
        print("    1. Переконайтесь, що скріншоти містять кружечок з літерою.")
        print("    2. Спробуйте знизити THRESHOLD_VALUE (рядок 27) зі 140 до 100.")
        print("    3. Спробуйте підвищити MATCH_SCORE_LIMIT (рядок 28) з 0.25 до 0.35.")
    else:
        print(f"    Деякі зображення не розпізнані — перевірте звіт вище.")
    print("=" * 60)


if __name__ == "__main__":
    main()
