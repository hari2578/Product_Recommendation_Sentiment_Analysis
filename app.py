import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request

# Initialize Flask
app = Flask(__name__)

# Setup NLTK (Necessary for cloud environment)
# Using a list for cleaner downloading
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'])

# Global loading
basedir = os.path.abspath(os.path.dirname(__file__))

def load_artifacts():
    # 1. Load Sentiment Model
    with open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # 2. Load TF-IDF Vectorizer
    with open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    # 3. Load Optimized Recommendation Dictionary
    with open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb') as f:
        recom_dict = pickle.load(f)
    
    # 4. Load Cleaned DataFrame
    # FIX: Using 'rb' mode and the standard pickle engine to avoid the 'pyarrow' dependency error
    with open(os.path.join(basedir, 'models/df_cleaned.pkl'), 'rb') as f:
        df_c = pd.read_pickle(f)
        
    return model, vectorizer, recom_dict, df_c

# Initialize artifacts once at startup
sentiment_model, tfidf_vectorizer, recommendation_dict, df_clean = load_artifacts()

def get_user_list():
    # Return sorted list of users for the dropdown
    return sorted(list(recommendation_dict.keys()))

@app.route('/')
def home():
    return render_template('index.html', user_list=get_user_list())

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture and Lowercase User Input
    user_input = request.form['username'].strip().lower()
    users = get_user_list()
    
    if user_input in recommendation_dict:
        # 2. Collaborative Filtering (Top 20 from Dictionary)
        top_20_products = recommendation_dict[user_input]
        
        # 3. Sentiment-Based Filtering
        # Create a copy of the subset to avoid SettingWithCopy warnings
        temp_df = df_clean[df_clean.name.isin(top_20_products)].copy()
        
        # Transform reviews using the fitted vectorizer
        X = tfidf_vectorizer.transform(temp_df["reviews_text"].values.astype(str))
        
        # Predict sentiments (1=Positive, 0=Negative)
        temp_df["predicted_sentiment"] = sentiment_model.predict(X)
        
        # Group and find % of positive reviews
        product_sentiment = temp_df.groupby('name')['predicted_sentiment'].agg(['count', 'sum'])
        product_sentiment['pos_percent'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
        
        # Rank by positive sentiment and pick Top 5
        top_5_final = product_sentiment.sort_values(by='pos_percent', ascending=False).head(5).index.tolist()
        
        return render_template('index.html', 
                               user_list=users, 
                               recommendations=top_5_final, 
                               selected_user=user_input)
    else:
        return render_template('index.html', 
                               user_list=users, 
                               message="User not found. Please select a name from the list.")

if __name__ == '__main__':
    # Bind to PORT provided by Render environment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
