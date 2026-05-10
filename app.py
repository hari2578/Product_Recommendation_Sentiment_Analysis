import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request

# 1. Initialize Flask App
app = Flask(__name__)

# 2. Setup NLTK (Download during startup for Render stability)
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'])

# 3. Global loading of Models and Data
basedir = os.path.abspath(os.path.dirname(__file__))

def load_all_artifacts():
    """Load all pickled models and dataframes from the models folder."""
    # Load recommendation dictionary (optimized for memory)
    with open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb') as f:
        recom_dict = pickle.load(f)
    
    # Load the "Skinny" DataFrame for lookup
    with open(os.path.join(basedir, 'models/df_cleaned.pkl'), 'rb') as f:
        df_c = pd.read_pickle(f)
    
    # Load sentiment model
    with open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load TF-IDF vectorizer
    with open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer, recom_dict, df_c

# Initialize artifacts once at startup
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
    
    # 1. Check if user exists in the dictionary
    if user_input in recommendation_dict:
        # Get the top 20 recommendations from collaborative filtering
        top_20_products = recommendation_dict[user_input]
        
        # 2. Robust Filtering Logic
        # Explicitly normalizing target product names for comparison
        target_products = [p.lower().strip() for p in top_20_products]
        
        # Using lambda-apply to avoid the 'DataFrame has no attribute str' error
        # This ensures we only work with the 'name' column Series
        temp_df = df_clean[df_clean['name'].astype(str).apply(lambda x: x.lower().strip()).isin(target_products)].copy()
        
        # 3. Safety Check: If no review data found, use fallback
        if temp_df.empty:
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_20_products[:5], 
                                   username=user_input,
                                   message="Note: Showing general recommendations (sentiment data unavailable).")

        # 4. Perform Sentiment Prediction
        try:
            # Transform text using the fitted TF-IDF Vectorizer
            # Assuming column 'reviews_text' exists based on the notebook export logic
            X = tfidf_vectorizer.transform(temp_df["reviews_text"].values.astype(str))
            
            # Predict sentiments (1=Pos, 0=Neg)
            temp_df["predicted_sentiment"] = sentiment_model.predict(X)
            
            # Group by product and calculate the ratio of positive reviews
            product_sentiment = temp_df.groupby('name')['predicted_sentiment'].agg(['count', 'sum'])
            product_sentiment['pos_percent'] = (product_sentiment['sum'] / product_sentiment['count']) * 100
            
            # Sort by percentage and select final Top 5
            top_5_final = product_sentiment.sort_values(by='pos_percent', ascending=False).head(5).index.tolist()
            
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_5_final, 
                                   username=user_input)
        
        except Exception as e:
            print(f"Prediction Error: {str(e)}")
            return render_template('index.html', 
                                   user_list=users, 
                                   recommendations=top_20_products[:5], 
                                   username=user_input,
                                   message="Sentiment processing failed. Showing top recommendations.")
    
    else:
        return render_template('index.html', 
                               user_list=users, 
                               message="User not found in database.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
