import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# Page Config & Styling
# ---------------------------
st.set_page_config(page_title="Student Performance Prediction", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f3f3f3;
        color: #333333;
    }
    .main {
        background-color: #ffffff;
        padding: 32px;
        border-radius: 24px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
import streamlit as st


    </style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance Prediction")
st.markdown("##### A Mini ML Research Project")
st.write("Now you can predict your exam performance based on your study habits, sleep, past scores, and attendance.")

# ---------------------------
# 1. Generate Synthetic Dataset
# ---------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500
    hours_studied = np.random.uniform(1, 10, n)
    sleep_hours = np.random.uniform(4, 9, n)
    previous_score = np.random.uniform(40, 90, n)
    attendance = np.random.uniform(50, 100, n)
    final_score = (hours_studied * 7) + (sleep_hours * 2) + \
                  (previous_score * 0.5) + (attendance * 0.3) + \
                  np.random.normal(0, 2, n)
    df = pd.DataFrame({
        'Hours_Studied': hours_studied,
        'Sleep_Hours': sleep_hours,
        'Previous_Score': previous_score,
        'Attendance': attendance,
        'Final_Score': final_score
    })
    return df

data = generate_data()

# ---------------------------
# 2. EDA - Correlation Heatmap
# ---------------------------
st.subheader("📊 Exploratory Data Analysis")

with st.expander("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(data.corr(), annot=True, cmap="crest", ax=ax)
    plt.title("Feature Correlation with Final Score")
    st.pyplot(fig)

# ---------------------------
# 3. Train Models & Compare
# ---------------------------
st.subheader("🤖 Model Training & Comparison")

X = data[['Hours_Studied', 'Sleep_Hours', 'Previous_Score', 'Attendance']]
y = data['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)

# Metrics
lr_r2 = r2_score(y_test, y_lr)
rf_r2 = r2_score(y_test, y_rf)

col1, col2 = st.columns(2)
with col1:
    st.metric("Linear Regression R²", f"{lr_r2:.3f}")
    st.metric("Linear Regression RMSE", f"{np.sqrt(mean_squared_error(y_test, y_lr)):.2f}")
with col2:
    st.metric("Random Forest R²", f"{rf_r2:.3f}")
    st.metric("Random Forest RMSE", f"{np.sqrt(mean_squared_error(y_test, y_rf)):.2f}")

# Feature Importance
st.markdown("### 🔍 Feature Importance (Random Forest)")
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(5, 3))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette="viridis", ax=ax2)
st.pyplot(fig2)

# ---------------------------
# 4. Interactive Prediction
# ---------------------------
st.subheader("🎯 Predict Your Score")

col1, col2 = st.columns(2)
with col1:
    hours = st.slider("Hours Studied per Day", 1.0, 10.0, 5.0)
    sleep = st.slider("Average Sleep Hours", 4.0, 9.0, 7.0)
with col2:
    prev = st.number_input("Previous Exam Score", 0.0, 100.0, 70.0)
    attendance = st.slider("Attendance (%)", 40.0, 100.0, 85.0)

if st.button("Predict My Final Score"):
    input_data = pd.DataFrame([[hours, sleep, prev, attendance]],
                              columns=['Hours_Studied', 'Sleep_Hours', 'Previous_Score', 'Attendance'])
    pred = rf.predict(input_data)[0]
    st.success(f"✅ Predicted Final Score: *{pred:.2f} / 100*")

st.markdown("---")
st.caption("Developed as a mini-research ML project by Debabrata Das")