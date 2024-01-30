from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

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

@app.route('/')
def home():
    products = df["Product ID"].unique().tolist()
    return render_template('index.html', products=products)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_product = request.form.get('product')
    if selected_product:
        recommendations = get_recommendations(selected_product)
        return render_template('index.html', products=df["Product ID"].unique().tolist(), selected_product=selected_product, recommendations=recommendations)
    else:
        return render_template('index.html', products=df["Product ID"].unique().tolist())

print("Association Rules:")
print(upsell_rules)

def get_recommendations(product):
    associated_products = list(upsell_rules[upsell_rules['antecedents'] == frozenset({product})]['consequents'])
    print("Associated Products:", associated_products)  # Add this line for debugging
    return [item for sublist in associated_products for item in sublist]


if __name__ == '__main__':
    app.run(debug=True)
