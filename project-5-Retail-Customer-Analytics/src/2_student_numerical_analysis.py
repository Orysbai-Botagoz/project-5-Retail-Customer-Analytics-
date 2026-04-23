import pandas as pd
import numpy as np

try:
   
    df = pd.read_csv('../data/retail_customer_loyalty_realistic.csv')
    print("✅ Файл сәтті жүктелді!")

    #1 тапсырма
    numeric_cols = ['age', 'total_spent', 'avg_purchase_value', 'loyalty_score']
    numeric_data = df[numeric_cols].to_numpy()

    #2 тапсырма
    means = np.mean(numeric_data, axis=0)
    medians = np.median(numeric_data, axis=0)

    print("\n--- Результаты анализа  ---")
    for i, col in enumerate(numeric_cols):
        print(f"Показатель {col}: Среднее = {means[i]:.2f}, Медиана = {medians[i]:.2f}")

except Exception as e:
    print(f"❌ Произошла ошибка: {e}")
