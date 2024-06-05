import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Function to transform the text
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Restaurent Review Checker ")

input_review = st.text_input("Enter Your Review")

if st.button('Predict'):
    transform_review = transform(input_review)

    vector_input = tfidf.transform([transform_review])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Positive")
    else:
        st.header("Negative")


# streamlit run app.py  - To run app