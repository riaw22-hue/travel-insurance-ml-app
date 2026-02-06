import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Travel Insurance Prediction ✈️",
    page_icon="🎒",
    layout="wide"
)

# Button style
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Function to load Lottie animation from URL
# --------------------------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Confetti animation (from lottiefiles.com)
confetti_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
try:
    df = pd.read_csv("TravelInsurancePrediction.csv")
except FileNotFoundError:
    st.error("❌ CSV file not found! Make sure 'TravelInsurancePrediction.csv' is in the project folder.")
    st.stop()

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# --------------------------------------------------
# Convert categorical → numeric
# --------------------------------------------------
df["Employment Type"] = df["Employment Type"].map({
    "Government Sector": 0,
    "Private Sector/Self Employed": 1
})

df["GraduateOrNot"] = df["GraduateOrNot"].map({
    "No": 0,
    "Yes": 1
})

df["ChronicDiseases"] = df["ChronicDiseases"].map({
    "No": 0,
    "Yes": 1
})

df["FrequentFlyer"] = df["FrequentFlyer"].map({
    "No": 0,
    "Yes": 1
})

df["EverTravelledAbroad"] = df["EverTravelledAbroad"].map({
    "No": 0,
    "Yes": 1
})

# --------------------------------------------------
# Split features and target
# --------------------------------------------------
X = df.drop("TravelInsurance", axis=1)
y = df["TravelInsurance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("Customer Details")

age = st.sidebar.number_input("Age", 18, 100, 30, step=1)
employment = st.sidebar.selectbox(
    "Employment Type",
    [0, 1],
    format_func=lambda x: "Government Sector" if x == 0 else "Private Sector / Self Employed"
)
graduate = st.sidebar.selectbox(
    "Graduate",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
income = st.sidebar.number_input(
    "Annual Income",
    min_value=0,
    value=500000,
    step=50000
)
family = st.sidebar.number_input("Family Members", 1, 10, 4, step=1)
chronic = st.sidebar.selectbox(
    "Chronic Disease",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
frequent = st.sidebar.selectbox(
    "Frequent Flyer",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
travelled = st.sidebar.selectbox(
    "Ever Travelled Abroad",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("✈️ Travel Insurance Prediction App")
st.write(f"### Model Accuracy: **{accuracy*100:.2f}%**")

# Prediction with cute celebration
if st.button("Predict"):
    input_data = np.array([[age, employment, graduate, income, family, chronic, frequent, travelled]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.success("🎉✅ Customer is LIKELY to buy Travel Insurance! 🎉")
        # Show confetti animation
        if confetti_animation:
            st_lottie(confetti_animation, height=300, key="confetti")
        st.balloons()
    else:
        st.warning("😅❌ Customer is NOT likely to buy Travel Insurance.")
        st.snow()

    st.info(f"📊 Probability of buying insurance: **{probability*100:.2f}%**")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Riya | Fun Travel Predictor 🌍")
