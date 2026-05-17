import pickle
import pandas as pd
import os

# Get the directory where model.py is located to handle paths correctly on any server
basedir = os.path.abspath(os.path.dirname(__file__))

def load_models():
    """
    Initializes and returns the final ML model and recommendation artifacts.
    This fulfills the requirement to initialize the final models inside model.py.
    """
    # Loading the Sentiment Model (Logistic Regression)
    with open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb') as f:
        sentiment_model = pickle.load(f)
        
    # Loading the TF-IDF Vectorizer
    with open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        
    # Loading the User-User Collaborative Filtering Dictionary (Top 20 per user)
    with open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb') as f:
        recom_dict = pickle.load(f)
        
    # Loading the "Skinny" dataframe for product-review lookups
    df_clean = pd.read_pickle(os.path.join(basedir, 'models/df_cleaned.pkl'))
    
    return sentiment_model, tfidf_vectorizer, recom_dict, df_clean

def get_sentiment_recommendations(user_input, sentiment_model, tfidf_vectorizer, recom_dict, df_clean):
    """
    1. Fetches Top 20 products for a user from the recommendation system.
    2. Uses the Sentiment Model to rank those 20 products based on review sentiment.
    3. Returns the Top 5 products with the highest positive sentiment.
    """
    user_input = user_input.strip().lower()
    
    # Check if user exists in our recommendation dictionary
    if user_input not in recom_dict:
        return None
    
    # Step 1: Get the Top 20 products from Collaborative Filtering
    top_20_products = recom_dict[user_input]
    
    # Step 2: Filter the cleaned dataframe safely using lowercase normalized matching
    # This prevents errors caused by case sensitivity or accidental string whitespace
    top_20_lower = [str(p).lower().strip() for p in top_20_products]
    temp_df = df_clean[df_clean['name'].str.lower().str.strip().isin(top_20_lower)].copy()
    
    # SAFETY GUARD: Check if the product review dataframe is empty
    if temp_df.empty:
        # Fallback: If no matched text records exist in df_clean, return the top 5 directly 
        # from the collaborative filter list rather than letting the web server crash
        return top_20_products[:5]
    
    # Step 3: Vectorize the reviews of these products
    X = tfidf_vectorizer.transform(temp_df['reviews_text'].values.astype(str))
    
    # Step 4: Predict sentiment (1 for Positive, 0 for Negative)
    temp_df['predicted_sentiment'] = sentiment_model.predict(X)
    
    # Step 5: Group by product and calculate the mean positive sentiment score
    # Higher mean = higher percentage of positive reviews
    sentiment_scores = temp_df.groupby('name')['predicted_sentiment'].mean().reset_index()
    
    # Step 6: Sort products by positive sentiment score in descending order
    sorted_recommendations = sentiment_scores.sort_values(by='predicted_sentiment', ascending=False)
    
    # Step 7: Extract the top 5 product names
    top_5_products = sorted_recommendations['name'].head(5).tolist()
    
    # FALLBACK GUARD 2: If the group-by yielded fewer than 5 matching text items,
    # fill remaining recommendation slots directly from the base collaborative filtering list
    if len(top_5_products) < 5:
        for prod in top_20_products:
            if prod not in top_5_products:
                top_5_products.append(prod)
            if len(top_5_products) == 5:
                break
                
    return top_5_products
