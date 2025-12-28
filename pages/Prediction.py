import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details to predict churn probability")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000", help="URL of the FastAPI server")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure", min_value=0, value=12, step=1)
        usage_frequency = st.number_input("Usage Frequency", min_value=0, value=10, step=1)
        support_calls = st.number_input("Support Calls", min_value=0, value=0, step=1)
    
    with col2:
        payment_delay = st.number_input("Payment Delay", min_value=0, value=0, step=1)
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Premium", "Standard"])
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend", min_value=0.0, value=1000.0, step=100.0)
        last_interaction = st.number_input("Last Interaction (days)", min_value=0, value=30, step=1)
    
    threshold = st.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

if submitted:
    payload = {
        "rows": [{
            "Age": int(age),
            "Gender": gender,
            "Tenure": int(tenure),
            "Usage Frequency": int(usage_frequency),
            "Support Calls": int(support_calls),
            "Payment Delay": int(payment_delay),
            "Subscription Type": subscription_type,
            "Contract Length": contract_length,
            "Total Spend": float(total_spend),
            "Last Interaction": int(last_interaction)
        }],
        "threshold": float(threshold)
    }
    
    try:
        with st.spinner("Calling API..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
        
        prob = data.get("probabilities", [0])[0]
        pred = data.get("predictions", [0])[0]
        explanations = data.get("explanations", [{}])[0]
        
        st.success("✅ Prediction completed!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{prob:.1%}")
        
        with col2:
            churn_status = "🔴 Churn" if pred == 1 else "🟢 No Churn"
            st.metric("Prediction", churn_status)
        
        with col3:
            confidence = prob if pred == 1 else (1 - prob)
            st.metric("Confidence", f"{confidence:.1%}")
        
        st.divider()
        
        if explanations:
            st.subheader("📈 Feature Importance (SHAP Values)")
            
            shap_df = pd.DataFrame([
                {"Feature": k, "SHAP Value": v}
                for k, v in explanations.items()
            ]).sort_values("SHAP Value", key=abs, ascending=False)
            
            fig = px.bar(
                shap_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                color="SHAP Value",
                color_continuous_scale=["red", "gray", "green"],
                title="Feature Impact on Prediction"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 Detailed SHAP Values"):
                st.dataframe(shap_df, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Probability Visualization")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Prediction Summary")
            st.json({
                "Churn Probability": f"{prob:.4f}",
                "Prediction": "Churn" if pred == 1 else "No Churn",
                "Threshold Used": threshold,
                "Customer Details": {
                    "Age": age,
                    "Gender": gender,
                    "Tenure": f"{tenure} months",
                    "Subscription": subscription_type,
                    "Contract": contract_length
                }
            })
    
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the API is running on port 8000.")
        st.info("💡 Start the API with: `docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts churn-api`")
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The API might be slow or unresponsive.")
    
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e}")
        try:
            error_detail = response.json()
            st.json(error_detail)
        except:
            st.text(response.text)
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.exception(e)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Instructions")
st.sidebar.markdown("""
1. Make sure the API is running:
   ```bash
   docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts churn-api
   ```

2. Fill in the customer details

3. Adjust the threshold if needed

4. Click "Predict Churn" to get predictions
""")

if st.sidebar.button("🔍 Test API Connection"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ API is connected!")
        else:
            st.sidebar.error("❌ API returned an error")
    except:
        st.sidebar.error("❌ Cannot connect to API")

