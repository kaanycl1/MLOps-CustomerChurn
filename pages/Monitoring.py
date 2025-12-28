import streamlit as st
import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset
import streamlit.components.v1 as components

st.set_page_config(page_title="Data Drift Monitoring", page_icon="📈", layout="wide")
report_path="artifacts/drift_report.html"
st.title("📈 Data Drift Monitoring")
st.markdown("Monitor data drift between reference data and current inference logs")

if 'report_generated' not in st.session_state:
    st.session_state['report_generated'] = False
if 'report_path' not in st.session_state:
    st.session_state['report_path'] = None

if os.path.exists(report_path) and not st.session_state.get('report_generated', False):
    st.session_state['report_generated'] = True
    st.session_state['report_path'] = report_path

ref_path = st.sidebar.text_input(
    "Reference Data Path", 
    value="data/raw/customer_churn_dataset-training-master.csv",
    help="Path to reference dataset"
)

curr_path = st.sidebar.text_input(
    "Current Data Path",
    value="artifacts/inference_logs.csv",
    help="Path to inference logs"
)

report_path = "artifacts/drift_report.html"

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Generate Drift Report", use_container_width=True):
        if not os.path.exists(ref_path):
            st.error(f"❌ Reference data not found: {ref_path}")
            st.stop()
        
        if not os.path.exists(curr_path):
            st.error(
                f"❌ Inference logs not found: {curr_path}\n\n"
                "💡 Make predictions first to generate inference logs."
            )
            st.stop()
        
        try:
            with st.spinner("Loading data and generating drift report..."):
                reference_data = pd.read_csv(ref_path)
                current_data = pd.read_csv(curr_path)
                
                if len(current_data) == 0:
                    st.error(f"❌ Inference logs file is empty: {curr_path}")
                    st.stop()
                
                if len(reference_data) > 5000:
                    st.info("📊 Reference data is large, sampling 5000 rows...")
                    reference_data = reference_data.sample(n=5000, random_state=42)
                
                common_cols = [c for c in reference_data.columns if c in current_data.columns]
                
                if not common_cols:
                    st.error("❌ No common columns found between reference and current data")
                    st.stop()
                
                report = Report([DataDriftPreset()])
                
                my_eval = report.run(
                    current_data=current_data[common_cols],
                    reference_data=reference_data[common_cols]
                )
                
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                my_eval.save_html(report_path)
                
            st.success(f"✅ Drift report generated successfully!")
            st.balloons()
            
            st.session_state['report_generated'] = True
            st.session_state['report_path'] = report_path
            
        except Exception as e:
            st.error(f"❌ Error generating report: {e}")
            st.exception(e)

with col2:
    if st.button("📊 View Latest Report", use_container_width=True):
        if os.path.exists(report_path):
            st.session_state['report_generated'] = True
            st.session_state['report_path'] = report_path
        else:
            st.warning("⚠️ No drift report found. Generate one first.")

report_path_to_show = st.session_state.get('report_path', report_path)
if st.session_state.get('report_generated', False) and os.path.exists(report_path_to_show):
    st.divider()
    st.subheader("📄 Drift Report")
    
    with open(report_path_to_show, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    components.html(html_content, height=800, scrolling=True)
    
    with st.expander("📥 Download Report"):
        with open(report_path_to_show, 'rb') as f:
            st.download_button(
                label="⬇️ Download HTML Report",
                data=f,
                file_name="drift_report.html",
                mime="text/html"
            )

else:
    st.info("👆 Click 'Generate Drift Report' to create a new drift analysis report")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Statistics")

if os.path.exists(curr_path):
    try:
        current_data = pd.read_csv(curr_path)
        st.sidebar.metric("Inference Logs", f"{len(current_data)} records")
        
        if 'timestamp' in current_data.columns:
            st.sidebar.markdown("**Latest Prediction:**")
            latest = pd.to_datetime(current_data['timestamp']).max()
            st.sidebar.write(latest.strftime("%Y-%m-%d %H:%M:%S"))
    except:
        st.sidebar.warning("Could not load inference logs")

if os.path.exists(ref_path):
    try:
        reference_data = pd.read_csv(ref_path)
        st.sidebar.metric("Reference Data", f"{len(reference_data)} records")
    except:
        pass

