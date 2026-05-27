import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


def run_feature_engineering():
    try:
        # Load data
        # Note: Ensure the file path is correct relative to your script
        df = pd.read_csv('../data/retail_customer_loyalty_realistic.csv', low_memory=False)
        print("Данные успешно загружены для Feature Engineering.\n")

        numeric_cols = [
            'purchase_frequency', 'membership_years', 'app_sessions_per_month',
            'website_visits_per_month', 'total_spent', 'loyalty_score',
            'last_purchase_days_ago', 'avg_purchase_value'
        ]
        numeric_cols = [
            'purchase_frequency', 'membership_years', 'app_sessions_per_month',
            'website_visits_per_month', 'total_spent', 'loyalty_score',
            'last_purchase_days_ago', 'avg_purchase_value'
        ]

        # --- КРИТИЧЕСКИЙ ШАГ: Приведение типов к числовым ---
        for col in numeric_cols:
            if col in df.columns:
                # errors='coerce' превратит любой текст, который не разделился в число, в NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Заполняем NaN нулями (или средним), чтобы не сломать формулы
                df[col] = df[col].fillna(0)

        # --- TASK 1: Behavioral Features ---
        print("--- Задача 1: Поведенческие признаки ---")
        df['purchase_intensity'] = df['purchase_frequency'] / (df['membership_years'] + 1)
        df['digital_engagement'] = (df['app_sessions_per_month'] + df['website_visits_per_month']) / 2
        df['value_per_year'] = df['total_spent'] / (df['membership_years'] + 1)

        print("Топ-10 клиентов по purchase_intensity:")
        print(df[['customer_id', 'purchase_intensity']].sort_values(by='purchase_intensity', ascending=False).head(10))
        print("\n")

        # --- TASK 2: Logical Segmentation (Fixed to use apply and lambda as requested) ---
        print("--- Задача 2: Сегментация клиентов ---")

        # Функция для применения через lambda
        def segment_customer(row):
            if row['total_spent'] > 7000 and row['loyalty_score'] > 70:
                return "High Value"
            elif row['total_spent'] > 3000:
                return "Medium Value"
            else:
                return "Low Value"

        df['customer_class'] = df.apply(lambda row: segment_customer(row), axis=1)

        print("Количество клиентов в каждой категории:")
        print(df['customer_class'].value_counts())
        print("\n")

        # --- TASK 3: Engagement Index ---
        print("--- Задача 3: Индекс вовлеченности ---")
        df['engagement_index'] = (df['app_sessions_per_month'] * 0.4 +
                                  df['website_visits_per_month'] * 0.3 +
                                  df['purchase_frequency'] * 0.3)

        # Min-Max Normalization
        min_eng = df['engagement_index'].min()
        max_eng = df['engagement_index'].max()
        if max_eng != min_eng:
            df['engagement_index'] = (df['engagement_index'] - min_eng) / (max_eng - min_eng)
        else:
            df['engagement_index'] = 0

        print("Топ-10 клиентов по engagement_index:")
        print(df[['customer_id', 'engagement_index']].sort_values(by='engagement_index', ascending=False).head(10))
        print("\n")

        # --- TASK 4: Churn Proxy ---
        print("--- Задача 4: Proxy-показатель оттока ---")
        df['churn_flag'] = np.where(
            (df['last_purchase_days_ago'] > 180) & (df['engagement_index'] < 0.3), 1, 0
        )
        churn_pct = (df['churn_flag'].mean()) * 100
        print(f"Процент клиентов с churn_flag = 1: {churn_pct:.2f}%")
        print("\n")

        # --- TASK 5: Interaction Features ---
        print("--- Задача 5: Взаимодействующие признаки ---")
        df['loyalty_spend'] = df['loyalty_score'] * df['total_spent']
        df['activity_value'] = df['purchase_frequency'] * df['avg_purchase_value']
        df['engagement_value'] = df['engagement_index'] * df['total_spent']

        print("Топ-10 клиентов по engagement_value:")
        print(df[['customer_id', 'engagement_value']].sort_values(by='engagement_value', ascending=False).head(10))
        print("\n")

        # --- TASK 6: Polynomial Features ---
        print("--- Задача 6: Полиномиальные признаки ---")
        df['total_spent_squared'] = df['total_spent'] ** 2
        df['loyalty_score_squared'] = df['loyalty_score'] ** 2
        df['interaction_term'] = df['total_spent'] * df['loyalty_score']

        poly_features = ['total_spent_squared', 'loyalty_score_squared', 'interaction_term']
        print("Корреляция новых признаков с churn_flag:")
        print(df[poly_features + ['churn_flag']].corr()['churn_flag'].drop('churn_flag'))
        print("\n")

        # --- TASK 7: Correlation Analysis ---
        print("--- Задача 7: Анализ корреляций ---")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        top_5_corr = corr_matrix['churn_flag'].abs().sort_values(ascending=False).head(6).index[1:]
        print("Топ-5 признаков, наиболее коррелирующих с churn_flag:")
        print(top_5_corr.tolist())
        print("\n")

        # --- TASK 8: Pivot Table ---
        print("--- Задача 8: Сводная таблица (Pivot Table) ---")
        pivot = pd.pivot_table(
            df,
            index='city',
            columns='customer_class',
            values='engagement_index',
            aggfunc='mean'
        )
        print("Средний engagement_index по городам и классам клиентов:")
        print(pivot.fillna(0))
        print("\n")

        # --- TASK 9: Weighted Metrics (Fixed to show TOP-10) ---
        print("--- Задача 9: Взвешенные показатели ---")
        total_spent_sum = df['total_spent'].sum()
        df['weighted_loyalty'] = df['loyalty_score'] * (df['total_spent'] / total_spent_sum)

        print("Топ-10 клиентов по weighted_loyalty:")
        print(df[['customer_id', 'weighted_loyalty']].sort_values(by='weighted_loyalty', ascending=False).head(10))
        print("\n")

        # --- TASK 10: Final Dataset Preparation (Added) ---
        print("--- Задача 10: Подготовка финального датасета ---")

        # Список колонок, которые не пригодятся для ML-моделей (например, сырые ID или текстовые города, если не кодировать)
        # Примечание: 'customer_id' можно удалить или оставить как индекс. Удалим лишние нечисловые (или оставим только то, что нужно)
        # Для примера удалим только явные текстовые/идентификационные поля, если это необходимо, либо оставим все новые сгенерированные фичи.

        # Сохраняем в CSV
        output_path = 'dataset_ready.csv'
        df.to_csv(output_path, index=False)
        print(f"Итоговый датасет сохранен в файл: {output_path}")
        print("\nFeature Engineering завершен успешно.")

        return df

    except FileNotFoundError:
        print("Ошибка: Файл '../data/retail_customer_loyalty_realistic.csv' не найден.")
    except Exception as e:
        print(f"Произошла ошибка в процессе выполнения: {e}")
        return None


# Вызов функции
if __name__ == "__main__":
    final_df = run_feature_engineering()

    # Настраиваем красивый стиль графиков
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.titlesize": 16})


    def generate_plots_for_presentation(df):
        print("--- Генерация графиков для презентации ---")

        # Создаем папку для графиков, если её нет
        os.makedirs("plots", exist_ok=True)

        # ==========================================
        # ГРАФИК 1: Тепловая карта корреляций (Heatmap)
        # ==========================================
        plt.figure(figsize=(10, 8))

        # Выберем ключевые признаки для демонстрации, включая новые и churn_flag
        features_to_plot = [
            "churn_flag",
            "total_spent",
            "total_spent_squared",
            "loyalty_score",
            "interaction_term",
            "engagement_index",
            "purchase_intensity",
        ]

        # Считаем корреляцию только для существующих в df колонок
        available_features = [col for col in features_to_plot if col in df.columns]
        corr_matrix = df[available_features].corr()

        # Рисуем хитмап
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Матрица корреляций: Взаимосвязь признаков с Churn Flag")
        plt.tight_layout()
        plt.savefig("plots/1_correlation_matrix.png", dpi=300)
        plt.close()
        print("1. График корреляций сохранен в 'plots/1_correlation_matrix.png'")

        # ==========================================
        # ГРАФИК 2: Распределение сегментов клиентов
        # ==========================================
        if "customer_class" in df.columns:
            plt.figure(figsize=(8, 5))

            # Считаем и сортируем для красивого отображения
            order = ["High Value", "Medium Value", "Low Value"]
            # Фильтруем order, оставляя только те классы, которые реально есть в данных
            order = [cat for cat in order if cat in df["customer_class"].unique()]

            sns.countplot(
                data=df,
                x="customer_class",
                order=order,
                palette="Dark2",
                hue="customer_class",
                legend=False,
            )

            plt.title("Распределение клиентов по созданным бизнес-сегментам")
            plt.xlabel("Сегмент (Customer Class)")
            plt.ylabel("Количество клиентов")
            plt.tight_layout()
            plt.savefig("plots/2_customer_segmentation.png", dpi=300)
            plt.close()
            print(
                "2. График сегментации сохранен в 'plots/2_customer_segmentation.png'"
            )

        # ==========================================
        # ГРАФИК 3: Распределение Индекса Вовлеченности
        # ==========================================
        if "engagement_index" in df.columns:
            plt.figure(figsize=(8, 5))

            # Строим гистограмму с линией плотности (KDE)
            sns.histplot(
                data=df,
                x="engagement_index",
                kde=True,
                color="purple",
                bins=30,
                stat="density",
            )

            plt.title("Распределение нормализованного Индекса Вовлеченности")
            plt.xlabel("Engagement Index (от 0 до 1)")
            plt.ylabel("Плотность распределения")
            plt.tight_layout()
            plt.savefig("plots/3_engagement_index_distribution.png", dpi=300)
            plt.close()
            print(
                "3. График индекса вовлеченности сохранен в 'plots/3_engagement_index_distribution.png'"
            )

        print("\nВсе графики успешно сгенерированы и готовы для вставки в слайды!")

    if final_df is not None:
        generate_plots_for_presentation(final_df)
    else:
        print("Ошибка: Не удалось сгенерировать графики, так как датасет пуст.")