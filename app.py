import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_texts(text):
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

st.title("Spam Classifier")

st.header("How to --")

st.text(" 1. Here you can input the msg you received into the text box can we can predict whether it's a spam or not. \n 2. Remember, Spam messages are lucrative in nature so think calmy before responding. \n 3. Spam are mostly lengthy and contain more words so observe . \n 4. Always be cautious of to whom and why you are sharing your personal mobile no and email . \n 5. If you happen to be a victim of such Fraud Activities report it. \n ")
st.text(" Visit the site for more info--> https://spamlaws.com/ ")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_texts(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")