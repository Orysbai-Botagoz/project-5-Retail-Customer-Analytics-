import numpy as np
import pandas as pd
from analytics import Client

#task 1#
#ЗАГРУЗКА ДАННЫХ
df = pd.read_csv('../data/retail_customer_loyalty_realistic.csv')
subset = df.head(20)
clients_list = []

for _, row in subset.iterrows():
    c = Client(
        customer_id=row['customer_id'],
        total_spent=row['total_spent'],
        loyalty_score=row['loyalty_score'],
        purchase_frequency=row['purchase_frequency'],
        returns_count=row['returns_count']
    )
    clients_list.append(c)
print("\n--- Анализ Студента №2 (NumPy) ---")

spent_array = np.array([c.total_spent for c in clients_list])
loyalty_array = np.array([c.loyalty_score for c in clients_list])

mean_spent = np.mean(spent_array)
median_spent = np.median(spent_array)
std_spent = np.std(spent_array)

print(f"Средний чек: {mean_spent:.2f}")
print(f"Медианный чек: {median_spent:.2f}")
print(f"Разброс трат (станд. отклонение): {std_spent:.2f}")

correlation = np.corrcoef(spent_array, loyalty_array)[0, 1]
print(f"Корреляция между тратами и лояльностью: {correlation:.4f}")

if correlation > 0.7:
    print("Вывод: Высокая лояльность сильно влияет на траты.")
else:
    print("Вывод: Связь между лояльностью и тратами слабая.")