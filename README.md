# Ebuss: Sentiment-Based Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Sentiment%20Analysis-orange.svg)
![Deployment](https://img.shields.io/badge/Deploy-Render-brightgreen.svg)

## 🎯 Project Overview
Ebuss is an e-commerce giant aiming to improve customer experience through a personalized recommendation engine. This project implements a **Hybrid Sentiment-Aware Recommendation System** that ensures users are not only recommended products they are likely to buy but also products that have a verified high positive reputation.

The system utilizes **Collaborative Filtering** to identify the top 20 potential products and uses a **Logistic Regression** model to rank them based on the positive sentiment percentage of existing reviews, finally presenting the Top 5.

---

## 🚀 Live Demo
You can access the live application here:  
**[Ebuss Recommendation Engine on Render](https://product-recommendation-sentiment-analysis.onrender.com/)**

---

## 🛠 Tech Stack

| Category | Tools & Libraries | Purpose |
| :--- | :--- | :--- |
| **Language & Env** | **Python 3.10+**, Jupyter Notebook | Core programming and interactive prototyping. |
| **Data Analysis** | **Pandas**, **NumPy** | Data cleaning, pivot tables, and matrix operations. |
| **NLP** | **NLTK**, Regex | Text cleaning, tokenization, and lemmatization. |
| **Feature Engineering**| **TF-IDF (Scikit-Learn)** | Converting text into weighted numerical vectors. |
| **Machine Learning** | **Scikit-Learn**, **XGBoost** | Training 4 models (LR, RF, XGB, Naive Bayes) & evaluation. |
| **Imbalance Handling** | **SMOTE** (Imbalanced-learn) | Handling skewed sentiment class distributions. |
| **Web Framework** | **Flask**, Jinja2 | Building the web interface and API routing. |
| **Production Server** | **Gunicorn** | WSGI HTTP Server for stable cloud deployment. |
| **Deployment** | **Render** | Cloud hosting for the live web application. |
| **Frontend** | HTML5, CSS3, **Bootstrap** | Designing a responsive user interface. |

---

## 🛠️ Technical Workflow

### 1. Data Cleaning and Pre-Processing
* **Quality Checks:** Addressed missing values in `reviews_username` and `user_sentiment`.
* **Feature Engineering:** Merged `reviews_title` and `reviews_text` into a single `reviews_combined` feature to maximize context.
* **Datatypes:** Standardized ratings and labels for mathematical modeling.

### 2. Text Processing (NLP)
A modular pipeline was built using **NLTK** to clean raw user feedback:
* **Tokenization & Lowercasing**: Standardizing text format.
* **Stop-word Removal**: Custom logic to **retain negation words** (e.g., "not", "no") as they are critical for sentiment.
* **Lemmatization**: Reducing words to their root forms using `WordNetLemmatizer`.

### 3. Feature Extraction
* **Vectorization**: Used **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* **N-Grams**: Utilized (1,2) N-Grams to capture contextual phrases like "not happy".

### 4. Model Building (Sentiment Analysis)
We evaluated **four** distinct classifiers to determine the optimal production model:

| Model | Accuracy | F1-Score | Analysis |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~92% | **High** | **Winner**: Best balance of speed and accuracy for deployment. |
| **Random Forest** | ~89% | Medium | Robust but creates very large file sizes. |
| **XGBoost** | ~91% | High | Strong performer, but slightly higher latency. |
| **Naive Bayes** | ~87% | Medium | Good baseline, struggles with complex N-gram nuance. |

### 5. Recommendation Engine
Compared two Collaborative Filtering approaches using **RMSE**:
* **User-User Filtering (Selected)**: Based on user similarity; provided superior performance and lower error rates.
* **Item-Item Filtering**: Based on product similarity.

---

## 💻 Deployment & Optimization
To handle memory constraints (512MB RAM) on the hosting platform:
* **Matrix Serialization**: Converted the heavy similarity matrix into a light-weight **Python Dictionary** (~2MB).
* **Skinny DataFrames**: Removed non-essential columns from lookup data to ensure sub-second response times.
* **Embedded UI**: The HTML frontend is embedded directly within `app.py` for a consolidated deployment structure.

---

## 📁 Repository Structure
```text
├── models/
│   ├── sentiment_model.pkl       # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF Vectorizer
│   ├── user_final_rating.pkl     # Optimized Recommendation Dictionary
│   └── df_cleaned.pkl            # Skinny lookup DataFrame for deployment
├── .gitignore                    # To ignore __pycache__ and local envs
├── app.py                        # Flask file + Embedded HTML code
├── model.py                      # Core ML and Recommendation initialization logic
├── requirements.txt              # List of dependencies
├── Procfile                      # Deployment instructions for Gunicorn
├── runtime.txt                   # Specified Python runtime version
├── Sentiment_Based_Product_Recommendation_Hari_Vittal_Mahendrakar.ipynb
└── README.md                     # Project documentation
