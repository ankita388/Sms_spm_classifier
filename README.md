# Sms_spm_classifier
📩 Spam SMS Detection using Machine Learning
A simple yet powerful Spam SMS Classifier built with Machine Learning, deployed using Streamlit, and powered by libraries like scikit-learn, nltk, xgboost, pandas, and matplotlib. This project demonstrates how NLP and supervised learning techniques can be used to automatically detect spam messages from real-world SMS data.


🧠 Features 


✅ Text pre-processing using NLTK (stopwords removal, stemming)

✅ Vectorization using TF-IDF

✅ Models: Multinomial Naive Bayes, Logistic Regression, XGBoost, etc.

✅ Exploratory Data Analysis with word clouds, bar plots, and histograms

✅ Real-time prediction through a clean Streamlit UI

🧰 Tech Stack


Category	Tools/Libraries Used

💻 Frontend	Streamlit

🧪 ML Models	scikit-learn, xgboost

🧹 NLP	nltk (stopwords, stemming), wordcloud

📊 Visualization	matplotlib, seaborn

📁 Data Handling	pandas, numpy


📂 Project Structure


├── app.py   # Streamlit app

├── spam_classifier.pkl     # Trained model

├── vectorizer.pkl          # TF-IDF vectorizer

├── data/                   # SMS dataset

├── visuals/                # Plots and wordclouds

├── README.md               # Project documentation

└── requirements.txt        # Dependencies


📈 Model Performance

Accuracy: ~97% with XGBoost

High Precision and Recall for spam class

Real-time prediction speed

📌 Future Improvements

Add deep learning model with LSTM

Integrate email or web spam detection

Multilingual support

