import pandas as pd
df = pd.read_csv('dataset/spam.csv',encoding='latin-1')
df = df.dropna(how="any", axis=1)
df.columns = ['target', 'text']
print ("---First 5 Rows of the dataset---")
print (df.head())
print("\n---Datsset Distribution---")
print(df['target'].value_counts())     

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and clean the data
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
df = df.dropna(how="any", axis=1)
df.columns = ['target', 'text']

# 2. Split into Training (80%) and Testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# 3. Convert text messages to numbers (TF-IDF Vectorization)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train the AI Model
print("Training the Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Evaluate the Model's performance
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Training Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# 6. Test with your own custom emails!
print("\n--- Testing Custom Messages ---")
custom_emails = [
    "https://rzp.io/rzp/9vKTMCx: click here for money! "
]

# Transform the custom text using our learned vectorizer
custom_vec = vectorizer.transform(custom_emails)
predictions = model.predict(custom_vec)

for email, prediction in zip(custom_emails, predictions):
    print(f"Message: {email}")
    print(f"Prediction: {prediction.upper()}\n")
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Inject Visual Style Configuration
st.set_page_config(page_title="AI Spam Shield", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
    /* Gradient Title Text */
    .main-title {
        font-size: 40px !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FFAC4B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 25px;
    }
    /* Description Styling */
    .sub-text {
        font-size: 18px;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Render Headers
st.markdown('<h1 class="main-title">🛡️ AI Spam Filter Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Paste messages below to inspect elements via live telemetry machine modeling</p>', unsafe_allow_html=True)

# 2. Optimized Pipeline Caching
@st.cache_resource
def build_pipeline():
    df = pd.read_csv('dataset/spam.csv', encoding='latin-1').dropna(how="any", axis=1)
    df.columns = ['target', 'text']
    vec = TfidfVectorizer()
    X_vec = vec.fit_transform(df['text'])
    mdl = MultinomialNB()
    mdl.fit(X_vec, df['target'])
    return vec, mdl

vectorizer, model = build_pipeline()

# 3. Interactive Component Panels
user_input = st.text_area("📨 Enter the text message payload below:", height=130, placeholder="Paste text messages here...")

if st.button("🚀 Process & Analyze Content", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Input prompt payload cannot be blank. Please supply message strings.")
    else:
        # Transformation & Matrix Prediction
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        
        # 4. Colorized Results Matrix Mapping
        if prediction == "spam":
            st.markdown("""
                <div style="background-color: #FFDEE2; border-left: 6px solid #FF4B5C; padding: 20px; border-radius: 8px; margin-top: 20px;">
                    <h3 style="color: #8A1F2A; margin: 0; font-size: 22px;">🚨 High Risk Threat Detected</h3>
                    <p style="color: #A93240; margin: 8px 0 0 0; font-size: 16px; font-weight: 500;">
                        Analysis flags this text structure as intentional spam. Suspicious signature layout verified.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.toast("Security Alert: System blocked potential exploit link signatures.", icon="🚨")
        else:
            st.markdown("""
                <div style="background-color: #D4EDDA; border-left: 6px solid #28A745; padding: 20px; border-radius: 8px; margin-top: 20px;">
                    <h3 style="color: #155724; margin: 0; font-size: 22px;">✅ Safe Transmission (HAM)</h3>
                    <p style="color: #1E7E34; margin: 8px 0 0 0; font-size: 16px; font-weight: 500;">
                        No threat patterns detected. Clean context verification match confirmed. Safe to engage.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.toast("System Notification: Message verification pass confirmed.", icon="🛡️")