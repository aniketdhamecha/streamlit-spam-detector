import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download tokenizers (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('spam_detector_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Initialize Stemmer
ps = PorterStemmer()

# Define text preprocessing function (same as in training)
def transform_text(text):
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

# Streamlit UI
st.set_page_config(page_title="Spam Email Detector", page_icon="📩", layout="centered")

st.title("📧 Email/SMS Spam Detection App")
st.write("Enter your message below and find out whether it's **Spam** or **Not Spam**!")

input_sms = st.text_area("✉️ Enter the message:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before clicking Predict.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 'spam':
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**.")

st.caption("Model trained with Multinomial Naive Bayes and TF-IDF features.")
