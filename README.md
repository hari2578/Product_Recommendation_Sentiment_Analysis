# Ebuss: Sentiment-Based Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Sentiment%20Analysis-orange.svg)
![Deployment](https://img.shields.io/badge/Deploy-Render-brightgreen.svg)

## 🎯 Project Overview
**Ebuss** is an e-commerce giant aiming to improve customer experience through a personalized recommendation engine. This project implements a **Hybrid Sentiment-Aware Recommendation System** that ensures users are not only recommended products they are likely to buy but also products that have a verified high positive reputation.

The system utilizes **Collaborative Filtering** to identify the top 20 potential products and uses a **Logistic Regression** model to rank them based on the positive sentiment percentage of existing reviews, finally presenting the Top 5.

---

## 🚀 Live Demo
You can access the live application here:  
**[Ebuss Recommendation Engine on Render](https://product-recommendation-sentiment-analysis.onrender.com/)**

---
Category,Tools & Libraries,Purpose

### 🛠 Tech Stack

| Category | Tools & Libraries | Purpose |
| :--- | :--- | :--- |
| **Language & Env** | **Python 3.10+**, Jupyter Notebook | Core programming and interactive prototyping. |
| **Data Analysis** | **Pandas**, **NumPy** | Data cleaning, pivot tables, and matrix operations. |
| **NLP** | **NLTK**, Regex | Text cleaning, tokenization, and lemmatization. |
| **Feature Engineering**| **TF-IDF (Scikit-Learn)** | Converting text into weighted numerical vectors. |
| **Machine Learning** | **Scikit-Learn**, **XGBoost** | Model training (Logistic Regression, RF) and evaluation. |
| **Imbalance Handling** | **SMOTE** (Imbalanced-learn) | Handling skewed sentiment class distributions. |
| **Web Framework** | **Flask**, Jinja2 | Building the web interface and API routing. |
| **Production Server** | **Gunicorn** | WSGI HTTP Server for stable cloud deployment. |
| **Deployment** | **Render** | Cloud hosting for the live web application. |
| **Frontend** | HTML5, CSS3, **Bootstrap** | Designing a responsive user interface. |

---
## 🛠️ Technical Workflow

### 1. Data Cleaning and Pre-Processing
* **Quality Checks:** Addressed missing values in `reviews_username` and `user_sentiment`.
* **Feature Engineering:** Merged `reviews_title` and `reviews_text` into a single `reviews_combined` feature to maximize context for the NLP model.
* **Datatypes:** Standardized ratings and labels for mathematical modeling.

### 2. Text Processing (NLP)
A modular pipeline was built using **NLTK** to clean raw user feedback:
* **Tokenization & Lowercasing**: Standardizing the text format.
* **Stop-word Removal**: Custom logic to **retain negation words** (e.g., "not", "no") as they are critical for sentiment accuracy.
* **Lemmatization**: Reducing words to their root forms using `WordNetLemmatizer`.

### 3. Feature Extraction
* **Vectorization**: Used **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* **N-Grams**: Utilized (1,2) N-Grams to capture context (e.g., "not good").

### 4. Model Building (Sentiment Analysis)
We evaluated three classifiers to find the best balance between accuracy and deployment latency:
* **Logistic Regression (Selected)**: F1-Score of ~92%.
* **Random Forest**: High accuracy but larger memory footprint.
* **XGBoost**: Strong performance but higher inference time.

### 5. Recommendation Engine
Built and compared two Collaborative Filtering approaches:
* **User-User Filtering**: Based on user similarity (Selected due to lower **RMSE**).
* **Item-Item Filtering**: Based on product similarity.

---

## 💻 Deployment & Optimization
To handle the memory constraints of cloud hosting (512MB RAM):
* **Memory Optimization**: Converted the 150MB+ similarity matrix into a serialized **Python Dictionary** (~2MB).
* **Production Artifacts**: Cleaned and "skinny" DataFrames were pickled to ensure sub-second response times.

---

## 📁 Repository Structure
```text
├── models/
│   ├── sentiment_model.pkl       # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF Vectorizer
│   ├── user_final_rating.pkl     # Optimized Recommendation Dictionary
│   └── df_cleaned.pkl            # Skinny lookup DataFrame
├── templates/
│   └── index.html                # Frontend UI
├── app.py                        # Flask Application logic
├── requirements.txt              # Production dependencies
└── README.md                     # Project documentation
