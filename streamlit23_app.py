import streamlit as st
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the SVM model and the TF-IDF vectorizer
with open('svm_model2.pkl', 'rb') as f:
    svm_model2 = pickle.load(f)
with open('tfidf_vec.pkl', 'rb') as f:
    tfidf_vec = pickle.load(f)

# Define the Flask app
app = Flask(__name__)

# Define the routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user's question from the form
    question = request.form['question']

    # Use the vectorizer to transform the question into a vector
    question_vec = tfidf_vec.transform([question])

    # Use the SVM model to generate an answer
    answer = svm_model2.predict(question_vec)[0]

    # Return the answer to the user
    return render_template('predict.html', answer=answer)

# Run the Flask app in the Streamlit app
def run():
    with st.spinner('Loading model...'):
        # Run the app
        app.run(port=8000)

if __name__ == '__main__':
    run()
