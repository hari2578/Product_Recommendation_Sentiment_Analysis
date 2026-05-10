import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request

app = Flask(__name__)

# Pre-download NLTK
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'])

basedir = os.path.abspath(os.path.dirname(__file__))

def load_artifacts():
    with open(os.path.join(basedir, 'models/user_final_rating.pkl'), 'rb') as f:
        recom_dict = pickle.load(f)
    with open(os.path.join(basedir, 'models/sentiment_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(basedir, 'models/tfidf_vectorizer.pkl'), 'rb') as f:
        vec = pickle.load(f)
    # Load DF and ensure it's a clean DataFrame
    df = pd.read_pickle(os.path.join(basedir, 'models/df_cleaned.pkl'))
    return model, vec, recom_dict, df

# Load once
model, tfidf, recom_dict, df_clean = load_artifacts()

@app.route('/')
def home():
    users = sorted(list(recom_dict.keys()))
    return render_template('index.html', user_list=users)

@app.route('/predict', methods=['POST'])
def predict():
    target_user = request.form['username'].strip().lower()
    users = sorted(list(recom_dict.keys()))
    
    if target_user not in recom_dict:
        return render_template('index.html', user_list=users, message="User not found.")

    # 1. Get Top 20
    top_20 = recom_dict[target_user]
    
    # 2. Filter data (Using the pre-cleaned 'name' column from our new Notebook logic)
    # This is much faster and avoids the .str.lower() crash
    temp_df = df_clean[df_clean['name'].isin(top_20)].copy()
    
    if temp_df.empty:
        return render_template('index.html', user_list=users, username=target_user, recommendations=top_20[:5])

    # 3. Sentiment Ranking
    try:
        X = tfidf.transform(temp_df['reviews_text'].values.astype(str))
        temp_df['preds'] = model.predict(X)
        
        # Calculate percentage of positive reviews
        scores = temp_df.groupby('name')['preds'].mean().sort_values(ascending=False)
        top_5 = scores.head(5).index.tolist()
        
        return render_template('index.html', user_list=users, username=target_user, recommendations=top_5)
    except:
        return render_template('index.html', user_list=users, username=target_user, recommendations=top_20[:5])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
