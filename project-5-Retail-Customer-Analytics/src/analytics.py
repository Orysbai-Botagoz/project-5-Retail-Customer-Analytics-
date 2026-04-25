#1 task
import pandas as pd

class Client:
    def __init__(self, customer_id, total_spent, loyalty_score, purchase_frequency,  returns_count):
        self.customer_id = customer_id
        self.total_spent = total_spent
        self.loyalty_score = loyalty_score
        self.purchase_frequency = purchase_frequency
        self.returns_count = returns_count
    def value(self):
        """Возвращает оценку ценности клиента (Total Spent * Loyalty Score)."""
        return self.total_spent * self.loyalty_score
    def risk(self):
        """Возвращает оценку риска возвратов."""
        return self.returns_count/(self.purchase_frequency + 1)


df = pd.read_csv('retail_customer_loyalty_realistic.csv')
subset = df.head(20)
clients_list = []
for _, row in subset.iterrows():
    clients = Client(
        customer_id=row['customer_id'],
        total_spent=row['total_spent'],
        loyalty_score=row['loyalty_score'],
        purchase_frequency=row['purchase_frequency'],
        returns_count=row['returns_count']
    )
    clients_list.append(clients)
print(f"{'Customer ID':<15} | {'Value (Total*Score)':<20} | {'Risk':<10}")
print("-" * 50)
for c in clients_list:
    print(f"{c.customer_id:<15} | {c.value():<20.2f} | {c.risk():<10.4f}")