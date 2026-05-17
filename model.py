import pickle
import pandas as pd
import os
import numpy as np

# Resolve base directories safely across any host ecosystem environment
basedir = os.path.abspath(os.path.dirname(__file__))

def load_models():
    """
    Initializes and returns the final ML model and recommendation artifacts.
    This fulfills the requirement to initialize the final models inside model.py.
    """
    with open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb') as f:
        sentiment_model = pickle.load(f)
        
    with open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        
    with open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb') as f:
        recom_dict = pickle.load(f)
        
    df_clean = pd.read_pickle(os.path.join(basedir, 'models/df_cleaned.pkl'))
    
    return sentiment_model, tfidf_vectorizer, recom_dict, df_clean

def get_sentiment_recommendations(user_input, sentiment_model, tfidf_vectorizer, recom_dict, df_clean):
    """
    1. Fetches Top 20 products for a user from the recommendation system.
    2. Uses the Sentiment Model to rank those 20 products based on review sentiment.
    3. Returns the Top 5 products with the highest positive sentiment.
    """
    user_input = str(user_input).strip().lower()
    
    # SYSTEM SAFETY FALLBACK (Cold Start Guard): 
    # If user does not exist in the interaction grid, return the top 5 highest-sentiment products globally.
    if user_input not in recom_dict:
        temp_df = df_clean.copy()
        temp_df['name_lower'] = temp_df['name'].str.lower().str.strip()
        
        # Calculate global sentiment metrics
        X_global = tfidf_vectorizer.transform(temp_df['reviews_text'].values.astype(str))
        temp_df['predicted_sentiment'] = sentiment_model.predict(X_global)
        
        global_scores = temp_df.groupby('name')['predicted_sentiment'].mean().reset_index()
        top_5_global = global_scores.sort_values(by='predicted_sentiment', ascending=False).head(5)['name'].tolist()
        return top_5_global
    
    # Step 1: Extract the top 20 pre-masked candidate keys
    top_20_products = recom_dict[user_input]
    
    # Step 2: Build a vectorized reference frame for target evaluations
    temp_df = df_clean.copy()
    temp_df['name_lower'] = temp_df['name'].str.lower().str.strip()
    
    # Filter using normalized lower-case keys
    filtered_df = temp_df[temp_df['name_lower'].isin(top_20_products)].copy()
    
    if filtered_df.empty:
        # Emergency backup text mapping fallback
        return [p.title() for p in top_20_products[:5]]
    
    # Step 3: Run textual parsing arrays through the TF-IDF Vectorizer
    X = tfidf_vectorizer.transform(filtered_df['reviews_text'].values.astype(str))
    
    # Step 4: Run sentiment model class predictions
    filtered_df['predicted_sentiment'] = sentiment_model.predict(X)
    
    # Step 5 & 6: Aggregate metrics by item name using capitalization preservation structures
    sentiment_scores = filtered_df.groupby('name')['predicted_sentiment'].mean().reset_index()
    sorted_recommendations = sentiment_scores.sort_values(by='predicted_sentiment', ascending=False)
    
    # Step 7: Final Selection Extraction
    top_5_products = sorted_recommendations['name'].head(5).tolist()
    
    # Backfill protection to ensure exactly 5 valid items are always returned
    if len(top_5_products) < 5:
        # Create unique mappings of display titles from the dataset
        name_map = dict(zip(temp_df['name_lower'], temp_df['name']))
        for prod_lower in top_20_products:
            display_name = name_map.get(prod_lower, prod_lower.title())
            if display_name not in top_5_products:
                top_5_products.append(display_name)
                if len(top_5_products) == 5:
                    break
                    
    return top_5_products