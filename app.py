import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request

# 1. Initialize Flask App
app = Flask(__name__)

# 2. Setup NLTK (Necessary for Render/Public Deployment)
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'])

# 3. Global loading of Models and Data
basedir = os.path.abspath(os.path.dirname(__file__))

def load_all_artifacts():
    """Load all pickled models and dataframes from the models folder."""
    with open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb') as f:
        recom_dict = pickle.load(f)
    
    with open(os.path.join(basedir, 'models/df_cleaned.pkl'), 'rb') as f:
        df_c = pd.read_pickle(f)
        
    return model, vectorizer, recom_dict, df_c

# Initialize artifacts once at startup
sentiment_model, tfidf_vectorizer, recommendation_dict, df_clean = load_all_artifacts()

def get_user_list():
    """Helper to provide sorted user list for the UI dropdown."""
    return sorted(list(recommendation_dict.keys()))

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html', user_list=get_user_list())

@app.route('/predict', methods=['POST'])
def predict():
    """Process recommendation request and perform sentiment-based ranking."""
    user_input = request.form['username'].strip().lower()
    users = get_user_list()
    
    if user_input in recommendation_dict:
        top_20_products = recommendation_dict[user_input]
        
        # Normalize for filtering
        temp_df = df_clean[df_clean['name'].astype(str).str.lower().str.strip().isin(
            [p.lower().strip() for p in top_20_products]
        )].copy()
        
        # Safety Check: Fallback if no review data exists
        if temp_df.empty:
            top_5_fallback = top_20_products[:5]
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_5_fallback, 
                                   username=user_input,  # FIXED: Matches {{ username }} in HTML
                                   message="Note: Showing direct recommendations (sentiment data unavailable).")

        try:
            # Transform and Predict
            X = tfidf_vectorizer.transform(temp_df["reviews_text"].values.astype(str))
            temp_df["predicted_sentiment"] = sentiment_model.predict(X)
            
            # Group by product
            product_sentiment = temp_df.groupby('name')['predicted_sentiment'].agg(['count', 'sum'])
            product_sentiment['pos_percent'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
            
            # Select Final Top 5
            top_5_final = product_sentiment.sort_values(by='pos_percent', ascending=False).head(5).index.tolist()
            
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_5_final, 
                                   username=user_input) # FIXED: Matches {{ username }} in HTML
        
        except Exception as e:
            print(f"Prediction Error: {str(e)}")
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_20_products[:5], 
                                   username=user_input, # FIXED: Matches {{ username }} in HTML
                                   message="Sentiment processing failed. Showing default recommendations.")
    
    else:
        return render_template('index.html', 
                               user_list=users, 
                               message="User not found. Please select a user from the list.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
