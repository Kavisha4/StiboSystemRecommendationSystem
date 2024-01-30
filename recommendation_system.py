import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load_sample_dataset():
    # Load your sample dataset (GroceryStoreDataSet.csv or any other)
    # Assuming it's in the format you provided and has no header
    df = pd.read_csv("apriopri/GroceryStoreDataSet.csv", header=None)
    return df

def get_recommendations(product, df):
    # Split the items in each transaction
    transactions = [transaction.split(',') for transaction in df[0]]

    # Convert the transactions to a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)
    basket_df = pd.DataFrame(te_data, columns=te.columns_).astype(int)

    # Assuming 'product' is in the first column
    basket = basket_df.groupby(['Product Name'])['Product Name'].count().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Apply the Apriori algorithm
    frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # Filter rules for upselling and cross-selling
    upsell_rules = rules[rules['lift'] > 1]

    # Get associated products
    associated_products = list(upsell_rules[upsell_rules['antecedents'] == frozenset({product})]['consequents'])

    return [item for sublist in associated_products for item in sublist]
