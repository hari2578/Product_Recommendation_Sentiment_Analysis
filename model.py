
"""According to the instructions to include "only one recommendation system," the User-Based 
    approach was selected over the Item-Based approach because:

    1. Personalization: It excels at discovering new products for a user based on the behavior 
    of a "neighbor" with similar tastes.

    2. Efficiency: For the specific dataset, the User-User matrix provides a more distinct set 
    of recommendations that could then be effectively re-ranked by the sentiment model.
"""
import pandas as pd
import numpy as np
import pickle

# --- Initialize and Load Artifacts ---

sentiment_model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
user_final_rating = pickle.load(open('user_final_rating.pkl', 'rb'))
df_cleaned = pd.read_pickle('df_cleaned.pkl')

def get_sentiment_recommendations(user_input):
    """
    Core logic: Collaborative Filtering + Sentiment Filtering
    """
    user_input_low = user_input.lower().strip()
    # Check if user exists in the recommendation matrix
    matching_users = [idx for idx in user_final_rating.index if str(idx).lower() == user_input_low]
    
    if not matching_users:
        return None, "User not found."
    
    user_id = matching_users[0]
    
    # 1. Retrieve Top 20 products from Recommendation System for the selected user
    user_row = user_final_rating.loc[user_id]
    top_20_products = user_row.sort_values(ascending=False).head(20).index.tolist()
    
    product_scores = []
    
    # 2. Refining with Sentiment Model
    for product in top_20_products:
        clean_prod_name = str(product).strip()
        product_reviews = df_cleaned[df_cleaned['name'].str.strip() == clean_prod_name]['reviews_processed']
        
        if not product_reviews.empty:
            tfidf_features = tfidf_vectorizer.transform(product_reviews)
            predictions = sentiment_model.predict(tfidf_features)
            score = np.mean(predictions) # Percentage of positive reviews
            product_scores.append((product, score))
        else:
            product_scores.append((product, 0.5))

    # 3. Sort by sentiment score and pick top 5 products for selected user
    top_5_tuples = sorted(product_scores, key=lambda x: x[1], reverse=True)[:5]
    return [p[0] for p in top_5_tuples], None

def get_user_list():
    """Helper to populate the dropdown"""
    return sorted(user_final_rating.index.tolist())