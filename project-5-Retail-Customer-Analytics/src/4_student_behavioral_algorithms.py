#1 nfsk
import pandas as pd
import math
import os

# 1. Умная загрузка данных
# Проверяем, где мы находимся, и ищем файл
file_name = 'retail_customer_loyalty_realistic.csv'

if not os.path.exists(file_name):
    # Если файла нет рядом со скриптом, пробуем путь через папку src
    file_name = os.path.join('src', 'retail_customer_loyalty_realistic.csv')
    if not os.path.exists(file_name):
        # Если и так не нашли, пробуем выйти на уровень выше (на случай запуска из src)
        file_name = os.path.join('..', 'src', 'retail_customer_loyalty_realistic.csv')


try:
    df = pd.read_csv(file_name)
    print(f"✅ Файл успешно загружен из: {file_name}")
except FileNotFoundError:
    print("❌ Ошибка: Не удалось найти CSV-файл. Проверьте, что он лежит в папке src!")
    exit()

# --- АЛГОРИТМИЧЕСКИЙ ПОИСК АНОМАЛИЙ (Задание 4 студента) ---

# Шаг 1: Расчет среднего арифметического через цикл
total_spent_sum = 0
for val in df['total_spent']:
    total_spent_sum += val
mean_spent = total_spent_sum / len(df)

# Шаг 2: Расчет стандартного отклонения вручную
sum_sq_diff = 0
for val in df['total_spent']:
    # Квадрат разности (отклонение от среднего)
    sum_sq_diff += (val - mean_spent) ** 2

variance = sum_sq_diff / len(df)
std_dev = math.sqrt(variance)

# Шаг 3: Поиск аномалий по правилу 2-х сигм (2 * std_dev)
lower_bound = mean_spent - 2 * std_dev
upper_bound = mean_spent + 2 * std_dev

# Фильтруем клиентов, чьи траты выходят за эти границы
anomalies = df[(df['total_spent'] < lower_bound) | (df['total_spent'] > upper_bound)]

# --- ВЫВОД РЕЗУЛЬТАТОВ ---
print("-" * 50)
print(f"Средний чек по всей базе: {mean_spent:.2f}")
print(f"Стандартное отклонение (разброс): {std_dev:.2f}")
print(f"Границы нормы: от {lower_bound:.2f} до {upper_bound:.2f}")
print("-" * 50)
print(f"Найдено аномальных записей: {len(anomalies)}")
print("Топ-10 аномальных клиентов:")
print(anomalies[['customer_id', 'total_spent', 'loyalty_score']].head(10))