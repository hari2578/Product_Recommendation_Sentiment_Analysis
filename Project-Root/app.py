# app.py
from flask import Flask, render_template, request
from model import get_sentiment_recommendations, get_user_list

app = Flask(__name__)

@app.route('/')
def home():
    # Fetch user list for the dropdown from model.py
    users = get_user_list()
    return render_template('index.html', user_list=users)

@app.route('/predict', methods=['POST'])
def predict():
    user_name = request.form['username']
    
    # Call the recommendation system defined in model.py
    recommendations, error_message = get_sentiment_recommendations(user_name)
    
    users = get_user_list()
    
    return render_template('index.html', 
                           recommendations=recommendations, 
                           user=user_name,
                           user_list=users,
                           message=error_message)

if __name__ == '__main__':
    app.run(debug=True)