#1 task
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


#task 2
import pandas as pd

# 1. Сортировка DataFrame по стажу членства
df_sorted = df.sort_values(by='membership_years').reset_index(drop=True)

# 2. Параметры скользящего окна
window_size = 10
rolling_averages = []

# 3. Реализация скользящего окна через цикл
# Проходим от 0 до (общее кол-во строк - размер окна + 1)
for i in range(len(df_sorted) - window_size + 1):
    # Выделяем текущее "окно" данных
    window = df_sorted.iloc[i: i + window_size]

    # Вычисляем среднюю частоту покупок в этом окне
    avg_freq = window['purchase_frequency'].mean()

    # 4. Сохраняем результат в список
    rolling_averages.append(avg_freq)

# Вывод первых 10 результатов скользящего среднего
print(f"Первые 10 значений скользящего среднего (окно {window_size}):")
print(rolling_averages[:10])


#task 3
import pandas as pd
import numpy as np

# 1. Выбираем признаки для расчета
features = ['total_spent', 'loyalty_score', 'purchase_frequency']
target_client = df.iloc[0]  # Первый клиент
distances = []

# 2. Проходим через цикл для вычисления расстояний до всех остальных
for index, row in df.iterrows():
    if index == 0:
        continue  # Пропускаем сравнение клиента с самим собой

    # Вычисляем Евклидово расстояние вручную [cite: 200]
    dist = np.sqrt(
        (target_client['total_spent'] - row['total_spent']) ** 2 +
        (target_client['loyalty_score'] - row['loyalty_score']) ** 2 +
        (target_client['purchase_frequency'] - row['purchase_frequency']) ** 2
    )

    distances.append((row['customer_id'], dist))

# 3. Сортируем список по расстоянию (от меньшего к большему)
distances.sort(key=lambda x: x[1])

# 4. Находим 5 ближайших клиентов
top_5_neighbors = distances[:5]

print(f"Поиск похожих клиентов для ID: {target_client['customer_id']}")
print("-" * 30)
for cid, d in top_5_neighbors:
    print(f"Customer ID: {cid} | Расстояние: {round(d, 4)}")  #




#task4
import pandas as pd


# 1. Определяем функцию рекомендаций на основе бизнес-логики
def get_recommendation(row):
    # Условие для премиальной электроники
    if row['preferred_category'] == 'Electronics' and row['total_spent'] > 5000:
        return 'Premium Electronics'

    # Условие для активных покупателей (например, частота выше 10 покупок)
    # Порог частоты можно адаптировать под медиану вашего датасета
    if row['purchase_frequency'] > 10:
        return 'Discount Campaign'

    # Рекомендация по умолчанию
    return 'Standard Offer'


# 2. Создаем новую колонку, применяя функцию к каждой строке
df['recommendation'] = df.apply(get_recommendation, axis=1)

# 3. Вывод результатов для проверки
print("Примеры рекомендаций для клиентов:")
print(df[['customer_id', 'preferred_category', 'total_spent', 'purchase_frequency', 'recommendation']].head(10))

# Статистика по рекомендациям
print("\nРаспределение рекомендаций:")
print(df['recommendation'].value_counts())