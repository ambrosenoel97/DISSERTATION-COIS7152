from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)



data = pd.read_csv("dataset/online_retail_II.csv")
user_item_matrix = data.pivot_table(index='Customer ID', columns='StockCode', values='Quantity', fill_value=0)

user_similarity = cosine_similarity(user_item_matrix)

def recommend_products(user_id, num_recommendations=5):
    print(type(user_id))
    if user_id not in user_item_matrix.index:
        print("User ID not found in the dataset.")
        return []

    if user_id in user_item_matrix.index:
        user_idx = user_item_matrix.index.get_loc(user_id)
        similar_users = np.argsort(user_similarity[user_idx])[::-1][1:] 
    else:
        print("Similarity data not available for User ID", user_id)
        return []

    recommended_items = set()
    for similar_user_idx in similar_users:
        if similar_user_idx < len(user_item_matrix.index):
            purchased_items = set(user_item_matrix.columns[user_item_matrix.iloc[similar_user_idx] > 0])
            target_user_items = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
            recommended_items.update(purchased_items - target_user_items)
            if len(recommended_items) >= num_recommendations:
                break
        else:
            print("Similar user index", similar_user_idx, "out of bounds.")
    
    return list(recommended_items)[:num_recommendations]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        num_recommendations = int(request.form['num_recommendations'])
        recommendations = recommend_products(user_id, num_recommendations)
        product=[]
        if recommendations:
            for idx, stock_code in enumerate(recommendations, 1):
                description = data[data['StockCode'] == stock_code]['Description'].iloc[0]
                product.append(description)
        return render_template('index.html', recommendations=product)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)