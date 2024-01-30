import pandas as pd
import random

# Function to generate a more diverse sample dataset
def generate_sample_dataset(num_records=100, num_products=50):
    # Ensure that the sample size is not greater than the population size
    num_records = min(num_records, num_products)

    data = {
        "Export Time": pd.date_range(start="2022-01-01", periods=num_records),
        "Export Context": [f"Context{i}" for i in range(1, num_records + 1)],
        "Context ID": list(range(1, num_records + 1)),
        "Workspace ID": list(range(101, 101 + num_records)),
        "Use Context Locale": [random.choice([True, False]) for _ in range(num_records)],
        "List of Value Group ID": list(range(1, num_records + 1)),
        "List of Value Group Name": [f"Group{i}" for i in range(1, num_records + 1)],
        "Attribute ID": list(range(1, num_records + 1)),
        "Show in Workbench": [random.choice([True, False]) for _ in range(num_records)],
        "Manually Sorted": [random.choice([True, False]) for _ in range(num_records)],
        # Introduce more variety in product purchases
        "Product ID": [f"Product{i}" for i in random.sample(range(1, num_products + 1), num_records)],
        "User Type ID": list(range(1, num_records + 1)),
        "Parent ID": [f"Parent{i}" for i in range(1, num_records + 1)],
        "Product Name": [f"Product {i}" for i in range(1, num_records + 1)],
    }

    df = pd.DataFrame(data)
    return df

# Generate a more diverse sample dataset with 100 records and 50 products
sample_data = generate_sample_dataset(num_records=100, num_products=50)

# Save the updated sample dataset to a CSV file
sample_data.to_csv("sample_dataset.csv", index=False)

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the updated sample dataset
df = pd.read_csv("sample_dataset.csv")

# Assuming 'Product ID' is the column representing the purchased products
basket = df.groupby(['Export Context', 'Product ID'])['Product ID'].count().unstack().fillna(0)

# Convert the counts to binary values (1 if the product was purchased, 0 otherwise)
basket_sets = basket.apply(lambda x: x > 0).astype(int)

# Apply the Apriori algorithm with adjusted min_support
frequent_itemsets = apriori(basket_sets, min_support=0.02, use_colnames=True)

# Generate association rules with adjusted min_threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Filter rules for upselling and cross-selling
upsell_rules = rules[rules['lift'] > 1]

# Display the upsell recommendations
print("Upsell Recommendations:")
print(upsell_rules[['antecedents', 'consequents', 'lift']])
