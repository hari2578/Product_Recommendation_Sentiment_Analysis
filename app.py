import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request

# 1. Initialize Flask App
app = Flask(__name__)

# 2. Setup NLTK (Crucial for Render/Public Deployment)
# These must be downloaded so the vectorizer/model can process text
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# 3. Load Models and Data using Relative Paths
basedir = os.path.abspath(os.path.dirname(__file__))

# Update these filenames if they differ in your GitHub repo
model = pickle.load(open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb'))
recommendation_final = pickle.load(open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb'))
df_clean = pd.read_pickle(os.path.join(basedir, 'models/df_cleaned.pkl'))

# Helper function to provide user list for the UI dropdown
def get_user_list():
    return list(recommendation_final.index.sort_values())

@app.route('/')
def home():
    users = get_user_list()
    return render_template('index.html', user_list=users)

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get user input and list
    user_input = request.form['username'].strip().lower()
    users = get_user_list()
    
    # 2. Case-insensitive mapping
    user_map = {str(name).lower(): name for name in recommendation_final.index}
    
    if user_input in user_map:
        actual_user = user_map[user_input]
        
        # 3. Collaborative Filtering: Get Top 20 Recommendations
        # We get the predicted ratings for this specific user
        top_20_products = recommendation_final.loc[actual_user].sort_values(ascending=False)[0:20].index
        
        # 4. Sentiment Filtering: Rank the Top 20 down to Top 5
        # Filter the master dataframe for reviews of these 20 products
        temp_df = df_clean[df_clean.name.isin(top_20_products)]
        
        # Transform the text using the loaded TF-IDF Vectorizer
        # We use the cleaned text column created in your notebook
        X = vectorizer.transform(temp_df["reviews_text"].values.astype(str))
        
        # Predict sentiments using the ML Model (e.g., Logistic Regression or XGBoost)
        temp_df["predicted_sentiment"] = model.predict(X)
        
        # Group by product and calculate the ratio of positive reviews
        product_sentiment = temp_df.groupby('name')['predicted_sentiment'].agg(['count', 'sum'])
        product_sentiment['pos_percent'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
        
        # Sort by percentage and pick the Top 5
        top_5_final = product_sentiment.sort_values(by='pos_percent', ascending=False).head(5).index.tolist()
        
        return render_template('index.html', 
                               user_list=users, 
                               recommendations=top_5_final, 
                               selected_user=actual_user)
    
    else:
        return render_template('index.html', 
                               user_list=users, 
                               message="User not found. Please select a user from the list.")

if __name__ == '__main__':
    # PORT is assigned dynamically by Render; default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
