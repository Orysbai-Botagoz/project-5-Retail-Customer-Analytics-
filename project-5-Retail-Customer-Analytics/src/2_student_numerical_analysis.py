import numpy as np
import pandas as pd


def run_numpy_analysis():
    try:
        # 1. Загружаем данные (предположим, Студент 1 сохранит их в этот файл)
        # Если файл называется иначе, просто поменяй название тут
        df = pd.read_csv('retail_customer_loyalty_realistic.csv')
        print("NumPy анализі үшін деректер сәтті жүктелді.")

        # ЗАДАЧА 1: Преобразование в числовые массивы [cite: 64]
        # Выбираем только нужные колонки
        columns_to_use = [
            'age', 'total_spent', 'avg_purchase_value', 'loyalty_score',
            'purchase_frequency', 'returns_count',
            'store_visits_per_month', 'website_visits_per_month'
        ]

        # Создаем массив numeric_data
        numeric_data = df[columns_to_use].to_numpy()

        print(f"Массивтің формасы: {numeric_data.shape}")[cite: 64]
        print(f"Деректер түрі: {numeric_data.dtype}")[cite: 64]
        print("Массивтің алғашқы 5 жолы:")
        print(numeric_data[:5])[cite: 64]

        # ЗАДАЧА 2: Расчет базовой статистики
        # axis=0 значит, что считаем вертикально по каждой колонке
        means = np.mean(numeric_data, axis=0)
        medians = np.median(numeric_data, axis=0)
        stds = np.std(numeric_data, axis=0)

        print("\n--- Колонкалар бойынша статистика ---")
        for i, col in enumerate(columns_to_use):
            print(f"{col}: Орташа = {means[i]:.2f}, Медиана = {medians[i]:.2f}, Ауытқу = {stds[i]:.2f}")[cite: 69]

        # ЗАДАЧА 3: Продвинутая фильтрация (булевы маски) [cite: 74]
        # Находим 75-й перцентиль для total_spent (это колонка под индексом 1)
        spent_75_percentile = np.percentile(numeric_data[:, 1], 75)

        mask_spent = numeric_data[:, 1] > spent_75_percentile
        mask_loyalty = numeric_data[:, 3] > 70  # loyalty_score это индекс 3

        combined_mask = mask_spent & mask_loyalty
        high_value_indices = np.where(combined_mask)[0]

        print(f"\nНайдено клиентов с высокой ценностью: {len(high_value_indices)}")
        print(f"Индексы первых 10: {high_value_indices[:10]}")[cite: 74]

    except FileNotFoundError:
        print("")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    run_numpy_analysis()