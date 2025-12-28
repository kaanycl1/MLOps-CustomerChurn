import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 Customer Churn Prediction System")
st.markdown("Welcome! Use the sidebar to navigate between Prediction and Monitoring pages.")
st.info("👈 Select a page from the sidebar to get started")

st.sidebar.markdown("### 🚀 Quick Start")
st.sidebar.markdown("""
1. **Start API:**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts churn-api
   ```

2. **Make Predictions:**
   - Go to 📊 Prediction page
   - Fill in customer details
   - Get churn predictions

3. **Monitor Drift:**
   - Go to 📈 Monitoring page
   - Generate drift reports
   - View data drift analysis
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Pages")
st.sidebar.markdown("""
- **📊 Prediction**: Make churn predictions
- **📈 Monitoring**: View drift reports
""")

