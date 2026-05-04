import pandas as pd
#1 task
# 1. Excel немесе CSV файлын жүктеу (сенің файлың CSV форматында екен)
file_path = '../data/retail_customer_loyalty_realistic.csv'

with open(file_path, 'r', encoding='utf-8') as f:
    df = pd.read_csv(f)

# 2. Алғашқы 10 жолды шығару
print("--- 1. Деректердің алғашқы 10 жолы ---")
print(df.head(10))

# 3. Жолдар мен бағандар санын анықтау
rows, cols = df.shape
print(f"\n--- 2. Кесте өлшемі ---\nЖолдар саны: {rows}\nБағандар саны: {cols}")

# 4. Деректер типтерін шығару
print("\n--- 3. Бағандардың деректер типтері ---")
print(df.dtypes)

# 5. customer_id бойынша дубликаттарды тексеру
duplicates = df.duplicated(subset=['customer_id']).sum()
print(f"\n--- 4. Қайталанатын ID-лер саны: {duplicates}")

# 6. Пропускілерді (NaN) есептеу
print("\n--- 5. Бос орындар (Missing values) саны ---")
print(df.isnull().sum())

# 7. Теріс мәндерді іздеу (total_spent және avg_purchase_value)
# Саудада шығын мен орташа чек теріс сан болмауы керек
problematic_data = df[(df['total_spent'] < 0) | (df['avg_purchase_value'] < 0)]
print(f"\n--- 6. Теріс мәні бар жолдар саны: {len(problematic_data)}")

# 8. Проблемалы жолдардың пайыздық үлесін шығару
percent_bad = (len(problematic_data) / rows) * 100
print(f"\n--- 7. Қате деректердің үлесі: {percent_bad:.2f}%")