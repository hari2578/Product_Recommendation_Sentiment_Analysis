import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request

# Initialize Flask
app = Flask(__name__)

# Setup NLTK (Necessary for cloud environment)
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'])

# Global loading
basedir = os.path.abspath(os.path.dirname(__file__))

def load_artifacts():
    # Loading optimized dictionary and models
    model = pickle.load(open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb'))
    vectorizer = pickle.load(open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb'))
    recom_dict = pickle.load(open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb'))
    df_c = pd.read_pickle(os.path.join(basedir, 'models/df_cleaned.pkl'))
    return model, vectorizer, recom_dict, df_c

sentiment_model, tfidf_vectorizer, recommendation_dict, df_clean = load_artifacts()

def get_user_list():
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
        temp_df = df_clean[df_clean.name.isin(top_20_products)].copy()
        
        # Transform reviews
        X = tfidf_vectorizer.transform(temp_df["reviews_text"].values.astype(str))
        
        # Predict sentiments (1=Positive, 0=Negative)
        temp_df["predicted_sentiment"] = sentiment_model.predict(X)
        
        # Group and find % of positive reviews
        product_sentiment = temp_df.groupby('name')['predicted_sentiment'].agg(['count', 'sum'])
        product_sentiment['pos_percent'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
        
        # Rank and Pick Top 5
        top_5_final = product_sentiment.sort_values(by='pos_percent', ascending=False).head(5).index.tolist()
        
        return render_template('index.html', 
                               user_list=users, 
                               recommendations=top_5_final, 
                               selected_user=user_input)
    else:
        return render_template('index.html', 
                               user_list=users, 
                               message="User not found. Please select from the dropdown.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
