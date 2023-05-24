import streamlit as st
from PIL import Image
import pandas as pd
from pickle import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import nltk
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import itertools # confusion matrix
nltk.download('stopwords')
nltk.download('wordnet')


# Set the page configuration with the title and background image
st.set_page_config(
    page_title="Condition and Drug Name Prediction",
    page_icon=":pill:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and vectorizer
model = load(open('pass_tf.pkl', 'rb'))
vectorizer = load(open('tfidf3_vectorizer.pkl', 'rb'))

# Display the doctor image
st.image('doctor_image.jpg', width=200)

html_temp="""
<div style ="background-color:Blue;padding:5px">
<h2 style="color:white;text-align:center;"> What concerns you
about your health today? </h2>
"""
st.markdown(html_temp,unsafe_allow_html=True)

# Load the data
df2 = pd.read_table('drugsCom_raw.tsv')
condition1 = ['Depression','High Blood Pressure','Diabetes, Type 2']
df1 = df2[df2['condition'].isin(condition1)]
X = df1.drop(['Unnamed: 0','rating','date','usefulCount','drugName'],axis=1)

# Clean the reviews
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmatized_words)

X['review_clean'] = X['review'].apply(review_to_words)

# Split the data into train and test sets
X_feat = X['review_clean']
y = X['condition']
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, stratify=y, test_size=0.2, random_state=0)

# Vectorize the text data
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Train the model
pass_tf1 = PassiveAggressiveClassifier()
pass_tf1.fit(tfidf_train, y_train)

# Create text input for user to enter review
text = st.text_input('Please enter the latest symptoms on basis of severity : ')

# Create predict button to predict condition and recommended drugs
# Create predict button to predict condition and recommended drugs
if st.button('Lets diagnose'):
    test = vectorizer.transform([text])
    pred1 = pass_tf1.predict(test)[0]
    st.write("Condition:")
    st.subheader(pred1)
    st.write(" Recommended Drugs based on review database ")

    if pred1 == "Depression":
        st.subheader('Trintellix , Vortioxetine , Duloxetine')
    elif pred1 == 'High Blood Pressure':
        st.subheader('Amlodipine or olmesartan , Azor , Hydrochlorothiazide or olmesartan')
    elif pred1 == 'Diabetes, Type 2':
        st.subheader('Liraglutide , Victoza , Bydureon')


# Add a warning message for predicting patient's condition
warning_message = """
    <div style="background-color: #ec891d; padding: 10px; border-radius: 10px;">
        <p style="color: white; font-weight: bold;">Warning: This prediction is for informational purposes only and should not be considered a medical diagnosis. Please consult a healthcare professional for an accurate diagnosis.</p>
    </div>
"""

# Display the warning message
st.markdown(warning_message, unsafe_allow_html=True)










