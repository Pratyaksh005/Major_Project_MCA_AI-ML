import streamlit as st
import joblib
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess

# ---------------- LOAD MODELS ----------------
reg_model = joblib.load("reg_model.pkl")
prob_model = joblib.load("prob_model.pkl")

# ---------------- LOAD DATA ----------------
df = load_and_preprocess("dataset.csv")

# ---------------- TITLE ----------------
st.title("🚀 Workforce Prediction System")

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Dataset Preview")
st.write(df.head())

# ---------------- KPI SECTION ----------------
st.subheader("📈 System Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(df))
col2.metric("High Risk Employees", int(df['Risk'].sum()))
col3.metric("Avg Assigned Hours", int(df['Assigned_Hours'].mean()))

# ---------------- INPUT PARAMETERS ----------------
st.subheader("🎛 Input Parameters")

available = st.slider("Available Hours", 20, 60, 40)
assigned = st.slider("Assigned Hours", 10, 70, 35)
complexity = st.slider("Task Complexity", 1, 10, 5)
deadline = st.slider("Deadline Days", 1, 30, 10)

# ---------------- WHAT-IF ANALYSIS ----------------
st.subheader("🧪 What-if Analysis")

scenario = st.selectbox("Scenario", ["Normal", "Increase Workload"])

# Apply scenario logic
if scenario == "Increase Workload":
    assigned *= 1.2

# ---------------- MODEL PREDICTION ----------------
future = reg_model.predict([[available, assigned, complexity]])
risk = prob_model.predict_proba([[available, assigned, deadline]])[0][1]

# ---------------- RESULTS ----------------
st.subheader("🔮 Results")

st.write(f"Predicted Workload: {future[0]:.2f}")
st.write(f"Risk Score: {risk:.2f}")

# ---------------- BASIC VISUALIZATION ----------------
st.subheader("📊 Workload Distribution")

fig = plt.figure()
plt.hist(df['Assigned_Hours'])
plt.xlabel("Assigned Hours")
plt.ylabel("Number of Employees")

st.pyplot(fig)