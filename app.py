from flask import Flask, render_template, request, jsonify
import pandas as pd  # Assuming you'll use pandas for data manipulation and analysis
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import xml.etree.ElementTree as ET

app = Flask(__name__)



def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    products = []

    # Iterate through each Product element
    for product_elem in root.findall('.//Product'):
        product = {}

        # Extract product information (replace with actual XML structure)
        product['Name'] = product_elem.find('Name').text
        product['AttributeLink'] = product_elem.find('AttributeLink').text if product_elem.find('AttributeLink') is not None else None
        # Add more attributes as needed

        products.append(product)

    return products

xml_file_path = 'C:/Users/kvsha/Kavisha Mathur/StiboSystems/DataSet3.xml'
product_data = parse_xml(xml_file_path)

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    products = []

    # Iterate through each Product element
    for product_elem in root.findall('.//Products/Product'):
        product = {}

        # Extract product information
        product['ID'] = product_elem.get('ID')
        product['usertypeID'] = product_elem.get('usertypeID')
        product['parentID'] = product_elem.get('parentID')
        product['Name'] = product_elem.get('Name')

        products.append(product)

    return products

# Example: Get product data
xml_file_path = 'C:/Users/kvsha/Kavisha Mathur/StiboSystems/DataSet3.xml'
product_data = parse_xml(xml_file_path)

# Print the extracted product data for demonstration
for product in product_data:
    print(product)


# Your cross-sell and upsell recommendation function
def recommend_products(product_data):
    # Implement your recommendation logic here using association rule mining
    # For demonstration purposes, let's assume a basic Apriori algorithm for finding associations
    # Replace this with your actual recommendation logic
    df = pd.DataFrame(product_data)
    df['product_id'] = df['product_id'].astype('str')  # Convert product_id to string for Apriori
    basket_sets = df.groupby(['order_id', 'product_id'])['quantity'].sum().unstack().fillna(0)
    frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_products', methods=['POST'])
def recommend_products_route():
    try:
        data = request.get_json()
        product_data = data.get('product_data')
        recommended_rules = recommend_products(product_data)
        return jsonify({'recommended_rules': recommended_rules.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/parse_xml_data', methods=['POST'])
def parse_xml_data():
    try:
        data = request.get_json()
        xml_data = data.get('xml_data')

        # Your XML parsing logic here (modify according to your XML structure)
        parsed_data = parse_xml(xml_data)

        return jsonify({'parsed_data': parsed_data})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
