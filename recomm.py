import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the sample dataset
df = pd.read_csv(sample_dataset.csv)

# Assuming 'Product ID' is the column representing the purchased products
basket = df.groupby(['Export Context', 'Product ID'])['Product ID'].count().unstack().fillna(0)

# Convert the counts to binary values (1 if the product was purchased, 0 otherwise)
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# Apply the Apriori algorithm
frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filter rules for upselling and cross-selling
upsell_rules = rules[rules['lift'] > 1]

# Display the upsell recommendations
print("Upsell Recommendations:")
print(upsell_rules[['antecedents', 'consequents', 'lift']])
