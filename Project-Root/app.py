import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

# 1. Initialize Flask App
app = Flask(__name__)

# 2. Set Directory and Load Models
basedir = os.path.abspath(os.path.dirname(__file__))

# Ensure your .pkl files are inside a folder named 'models' in your project root
# We load these globally so they are available to all routes
model = pickle.load(open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb'))
recommendation_final = pickle.load(open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb'))
df_clean = pd.read_pickle(os.path.join(basedir, 'models/df_cleaned.pkl'))

# Helper function to get user list for the dropdown
def get_user_list():
    return list(recommendation_final.index.sort_values())

@app.route('/')
def home():
    # Fetch user list for the dropdown
    users = get_user_list()
    return render_template('index.html', user_list=users)

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get and clean user input
    user_input = request.form['username'].strip().lower()
    users = get_user_list()
    
    # 2. Case-insensitive user check
    # We create a mapping of lowercase_name -> original_name
    user_map = {str(name).lower(): name for name in recommendation_final.index}
    
    if user_input in user_map:
        actual_user = user_map[user_input]
        
        # 3. Get Top 20 Recommendations (Collaborative Filtering)
        # Assuming recommendation_final is your predicted ratings matrix
        top_20_products = recommendation_final.loc[actual_user].sort_values(ascending=False)[0:20].index
        
        # 4. Filter by Sentiment (using your logic from the notebook)
        # Filter reviews for these 20 products
        temp_df = df_clean[df_clean.name.isin(top_20_products)]
        
        # Transform reviews using saved TF-IDF
        X = vectorizer.transform(temp_df["reviews_text"].values.astype(str))
        
        # Predict sentiment (1 for positive, 0 for negative)
        temp_df["predicted_sentiment"] = model.predict(X)
        
        # Calculate percentage of positive reviews per product
        product_sentiment = temp_df.groupby('name')['predicted_sentiment'].agg(['count', 'sum'])
        product_sentiment['pos_percent'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
        
        # Get Top 5 products based on positive sentiment percentage
        top_5_final = product_sentiment.sort_values(by='pos_percent', ascending=False).head(5).index.tolist()
        
        return render_template('index.html', user_list=users, recommendations=top_5_final, selected_user=actual_user)
    
    else:
        # If user is not found
        return render_template('index.html', user_list=users, message="User not found. Please select a valid user.")

if __name__ == '__main__':
    # PORT is required for public deployment (Render/Railway/Heroku)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
