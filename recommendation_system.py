# recommendation_system.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import random

def generate_sample_dataset():
    data = {
        "Export Time": ["2022-01-01", "2022-02-01", "2022-03-01"],
        "Export Context": ["Context1", "Context2", "Context3"],
        "Context ID": [1, 2, 3],
        "Workspace ID": [101, 102, 103],
        "Use Context Locale": [True, False, True],
        "List of Value Group ID": [1, 2, 3],
        "List of Value Group Name": ["Group1", "Group2", "Group3"],
        "ID": ["IceCat_Category_4776", "IceCat_Category_2554", "IceCat_Category_2555"],
        "Name": ["Building & Construction", "Electrical Equipment & Supplies", "Power Conditioning"]
        # Add more columns based on your structure
    }

    df = pd.DataFrame(data)
    df.to_csv("sample_dataset.csv", index=False)

def load_sample_dataset():
    return pd.read_csv("sample_dataset.csv")

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=['int64', 'float64'])))
    df_scaled.columns = df.select_dtypes(include=['int64', 'float64']).columns
    return df_scaled

def calculate_similarity_matrix(df_scaled):
    return cosine_similarity(df_scaled, df_scaled)

def get_recommendations(product_id, df, similarity_matrix):
    product_index = df[df["ID"] == product_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_products = [i[0] for i in similarity_scores[1:4]]  # Get top 3 similar products
    return df["Name"].iloc[similar_products].tolist()
