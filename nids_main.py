import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="centered"
)

# -------------------------------
# Session State
# -------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None

# -------------------------------
# Title
# -------------------------------
st.title("AI-Based Network Intrusion Detection System (NIDS)")
st.write("ML-based Intrusion Detection using CIC-IDS2017 Dataset")

# -------------------------------
# Load CSV Data
# -------------------------------
@st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert labels: BENIGN = 0, Attack = 1
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with column median
    df.fillna(df.median(), inplace=True)

    # OPTIONAL (HIGHLY RECOMMENDED): reduce size for Streamlit
    df = df.sample(5000, random_state=42)

    return df


# -------------------------------
# Train Model
# -------------------------------
def train_model(df):
    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, X.columns.tolist()

# -------------------------------
# Load Data
# -------------------------------
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.write(f"Total Samples: {df.shape[0]}")
st.write(f"Total Features: {df.shape[1] - 1}")

# -------------------------------
# Train Button
# -------------------------------
if st.button("Train Model Now"):
    model, acc, feature_names = train_model(df)
    st.session_state.model = model
    st.session_state.accuracy = acc
    st.session_state.features = feature_names

    st.success("✅ Model trained successfully")
    st.write(f"Model Accuracy: **{acc:.2f}**")

# -------------------------------
# Live Prediction
# -------------------------------
st.subheader("Live Traffic Prediction")

if st.session_state.model is None:
    st.warning("Train the model first to enable prediction.")
else:
    input_data = []

    for feature in st.session_state.features:
        val = st.number_input(feature, value=0.0)
        input_data.append(val)

    if st.button("Check Traffic"):
        sample = np.array([input_data])
        pred = st.session_state.model.predict(sample)

        if pred[0] == 1:
            st.error("🚨 Intrusion Detected (Attack Traffic)")
        else:
            st.success("✅ Normal Traffic (BENIGN)")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AI-Based NIDS using CIC-IDS2017 Dataset")
