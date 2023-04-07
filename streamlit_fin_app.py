import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import numpy as np

# Load the preprocessed data
df = pd.read_csv('BankFAQs.csv')
df1 = pd.read_csv('BankFAQs1.csv', encoding='ISO-8859-1')
data1 = pd.concat([df1, df])

# Define the TD-IDF vectorizer and fit it to the data
tdidf = TfidfVectorizer()
tdidf.fit(data1['Question'].str.lower())

# Define the support vector machine model and fit it to the data
svc_model = SVC(kernel='linear')
svc_model.fit(tdidf.transform(data1['Question'].str.lower()), data1['Class'])

# Define a function to get the answer to a given question
def get_answer(question):
    # Vectorize the question
    question_tdidf = tdidf.transform([question.lower()])
    
    # Calculate the cosine similarity between both vectors
    cosine_sims = cosine_similarity(question_tdidf, tdidf.transform(data1['Question'].str.lower()))

    # Get the index of the most similar text to the query
    most_similar_idx = np.argmax(cosine_sims)

    # Get the predicted class of the query
    predicted_class = svc_model.predict(question_tdidf)[0]
    
    # If the predicted class is not the same as the actual class, return an error message
    if predicted_class != data1.iloc[most_similar_idx]['Class']:
        return {'error': 'Could not find an appropriate answer.'}
    
    # Get the answer and construct the response
    answer = data1.iloc[most_similar_idx]['Answer']
    response = {
        'answer': answer,
        'predicted_class': predicted_class
    }
    
    return response

# Define the Streamlit app
def app():
    st.title('Banking FAQ Chatbot')
    question = st.text_input('Enter your question')
    if st.button('Get answer'):
        response = get_answer(question)
        st.write(response['answer'])
