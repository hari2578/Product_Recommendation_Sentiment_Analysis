import os
import pickle
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request
from flask import Flask, render_template_string, request
import model  # This imports your model.py file

app = Flask(__name__)

# Initialize models and data structures from model.py
# This ensures that your ML model and Recommendation system are loaded into memory once
sentiment_model, vectorizer, recom_dict, df_clean = model.load_models()

# HTML code for the user interface embedded directly within app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ebuss - Product Recommendation System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .container { max-width: 800px; margin-top: 50px; }
        .header-section { background-color: #004a99; color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .main-card { background: white; padding: 30px; border-radius: 0 0 10px 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .btn-primary { background-color: #004a99; border: none; padding: 10px 20px; }
        .btn-primary:hover { background-color: #003366; }
        .recommendation-list { margin-top: 30px; }
        .list-group-item-success { border-left: 5px solid #198754; font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-section text-center">
            <h1>Ebuss</h1>
            <p class="mb-0">Sentiment-Based Product Recommendation Engine</p>
        </div>
        
        <div class="main-card">
            <form action="/predict" method="post">
                <div class="mb-3">
                    <label for="username" class="form-label"><strong>Enter Username</strong> (e.g., jason, tina, mike):</label>
                    <input class="form-control form-control-lg" list="user_options" name="username" id="username" placeholder="Search for a user..." required>
                    <datalist id="user_options">
                        {% for user in user_list %}
                        <option value="{{ user }}">
                        {% endfor %}
                    </datalist>
                    <div id="emailHelp" class="form-text">Type a registered username to get personalized suggestions.</div>
                </div>
                <button type="submit" class="btn btn-primary w-100 btn-lg">View Recommendations</button>
            </form>

            {% if message %}
            <div class="alert alert-danger mt-4" role="alert">
                {{ message }}
            </div>
            {% endif %}

            {% if recommendations %}
            <div class="recommendation-list animate-in">
                <h4 class="mb-3">Top 5 Products for <span class="text-primary">{{ username }}</span>:</h4>
                <div class="list-group">
                    {% for prod in recommendations %}
                    <li class="list-group-item list-group-item-action list-group-item-success mb-2">
                        {{ loop.index }}. {{ prod|title }}
                    </li>
                    {% endfor %}
                </div>
                <p class="text-muted mt-3 small">*These products are filtered using real-time sentiment analysis of all existing reviews.</p>
            </div>
            {% endif %}
        </div>
        
        <div class="text-center mt-4 text-muted small">
            Built for IIITB Capstone Project | Machine Learning Engineer: Hari Vittal Mahendrakar
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Renders the landing page with the list of valid usernames."""
    users = sorted(list(recom_dict.keys()))
    return render_template_string(HTML_TEMPLATE, user_list=users)

@app.route('/predict', methods=['POST'])
def predict():
    """Connects the frontend input to the backend model and returns the Top 5 products."""
    user_input = request.form['username'].strip().lower()
    users = sorted(list(recom_dict.keys()))
    
    # Use the logic defined in model.py to get recommendations
    top_5 = model.get_sentiment_recommendations(user_input, sentiment_model, vectorizer, recom_dict, df_clean)
    
    if top_5 is None:
        # Return error if user is not in the collaborative filtering dictionary
        return render_template_string(HTML_TEMPLATE, user_list=users, message="User not found. Please try a different username.")
        
    # Return successful recommendations
    return render_template_string(HTML_TEMPLATE, 
                                  user_list=users, 
                                  recommendations=top_5, 
                                  username=user_input)

if __name__ == '__main__':
    # Running locally - Gunicorn should be used for production (Render)
    app.run()
