import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request

# 1. Initialize Flask App
app = Flask(__name__)

# 2. Setup NLTK (Necessary for Render/Public Deployment)
# Standardizing common downloads to ensure the vectorizer works as expected
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'])

# 3. Global loading of Models and Data
basedir = os.path.abspath(os.path.dirname(__file__))

def load_all_artifacts():
    """Load all pickled models and dataframes from the models folder."""
    # Using 'rb' and standard pickle to bypass PyArrow dependency issues
    with open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb') as f:
        recom_dict = pickle.load(f)
    
    # Using pd.read_pickle with an open file handle for stability
    with open(os.path.join(basedir, 'models/df_cleaned.pkl'), 'rb') as f:
        df_c = pd.read_pickle(f)
        
    return model, vectorizer, recom_dict, df_c

# Initialize artifacts once at startup to save memory and time
sentiment_model, tfidf_vectorizer, recommendation_dict, df_clean = load_all_artifacts()

def get_user_list():
    """Helper to provide sorted user list for the UI dropdown."""
    return sorted(list(recommendation_dict.keys()))

@app.route('/')
def home():
    """Render the homepage with the user list."""
    return render_template('index.html', user_list=get_user_list())

@app.route('/predict', methods=['POST'])
def predict():
    """Process recommendation request and perform sentiment-based ranking."""
    user_input = request.form['username'].strip().lower()
    users = get_user_list()
    
    # 1. Check if user exists in the optimized dictionary
    if user_input in recommendation_dict:
        # Get the top 20 recommendations from collaborative filtering
        top_20_products = recommendation_dict[user_input]
        
        # 2. Robust Filtering: Normalize names to ensure a match in df_clean
        # This prevents the 'ValueError: 0 samples' crash
        temp_df = df_clean[df_clean['name'].str.lower().str.strip().isin(
            [p.lower().strip() for p in top_20_products]
        )].copy()
        
        # 3. Safety Check: If no review data found, use fallback
        if temp_df.empty:
            top_5_fallback = top_20_products[:5]
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_5_fallback, 
                                   selected_user=user_input,
                                   message="Note: Showing direct recommendations (sentiment data unavailable).")

        # 4. Perform Sentiment Prediction
        try:
            # Transform text using the fitted TF-IDF Vectorizer
            X = tfidf_vectorizer.transform(temp_df["reviews_text"].values.astype(str))
            
            # Predict sentiments using the loaded model (1=Pos, 0=Neg)
            temp_df["predicted_sentiment"] = sentiment_model.predict(X)
            
            # Group by product and calculate the ratio of positive reviews
            product_sentiment = temp_df.groupby('name')['predicted_sentiment'].agg(['count', 'sum'])
            product_sentiment['pos_percent'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
            
            # Sort by percentage and select the final Top 5
            top_5_final = product_sentiment.sort_values(by='pos_percent', ascending=False).head(5).index.tolist()
            
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_5_final, 
                                   selected_user=user_input)
        
        except Exception as e:
            # Graceful error handling for transformation issues
            print(f"Prediction Error: {str(e)}")
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_20_products[:5], 
                                   message="Sentiment processing failed. Showing default recommendations.")
    
    else:
        # Handle cases where the user input doesn't exist
        return render_template('index.html', 
                               user_list=users, 
                               message="User not found. Please select a user from the list.")

if __name__ == '__main__':
    # PORT is assigned dynamically by Render; defaults to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
