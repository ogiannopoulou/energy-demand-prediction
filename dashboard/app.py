"""Streamlit dashboard for Italian electricity demand forecasting."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Italian Energy Demand Forecast",
    page_icon="⚡",
    layout="wide",
)

# Constants
API_URL = "http://localhost:8000"
MODELS_DIR = Path(__file__).parent.parent / "models"

st.title("⚡ Italian Electricity Demand Forecasting")
st.markdown("Real-time demand forecasting using ML models trained on ENTSO-E data")


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception:
        return False, None


def get_predictions(start_date, end_date, model_name):
    """Get predictions from the API for a date range."""
    predictions = []
    current = start_date
    
    while current <= end_date:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "datetime": current.isoformat(),
                    "features": {}
                },
                params={"model_name": model_name},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                predictions.append({
                    "datetime": current,
                    "predicted_demand_mw": data["predicted_demand_mw"]
                })
        except Exception as e:
            st.warning(f"Failed to get prediction for {current}: {e}")
        
        current += timedelta(hours=1)
    
    return pd.DataFrame(predictions)


def load_model_metadata():
    """Load metadata for all available models."""
    models = []
    
    if MODELS_DIR.exists():
        for meta_path in MODELS_DIR.glob("*.json"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                models.append(meta)
            except Exception:
                continue
    
    return models


# Sidebar
st.sidebar.header("Configuration")

# API status
api_healthy, health_data = check_api_health()
if api_healthy:
    st.sidebar.success("✓ API Connected")
    if health_data:
        st.sidebar.info(f"Model: {health_data.get('model_name', 'N/A')}")
else:
    st.sidebar.error("✗ API Disconnected")
    st.sidebar.info("Start API with: `make api`")


# Model selection
model_metadata = load_model_metadata()
if model_metadata:
    model_names = [m["name"] for m in model_metadata]
    selected_model = st.sidebar.selectbox("Select Model", model_names)
    
    # Show model details
    selected_meta = next((m for m in model_metadata if m["name"] == selected_model), None)
    if selected_meta:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Model Details:**")
        st.sidebar.markdown(f"- Version: {selected_meta.get('version', 'N/A')}")
        st.sidebar.markdown(f"- Trained: {selected_meta.get('trained_at', 'N/A')[:10]}")
        
        metrics = selected_meta.get("metrics", {})
        if metrics:
            st.sidebar.markdown("**Metrics:**")
            for k, v in metrics.items():
                st.sidebar.markdown(f"- {k.upper()}: {v:.4f}")
else:
    selected_model = "lightgbm_demand_forecast"
    st.sidebar.warning("No models found. Run `make train` first.")


# Main content
tab1, tab2, tab3 = st.tabs(["Forecast", "Model Performance", "Data Explorer"])

with tab1:
    st.header("Demand Forecast")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date(),
            min_value=datetime.now().date(),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=start_date + timedelta(days=7),
            min_value=start_date,
        )
    
    if st.button("Generate Forecast", type="primary"):
        if not api_healthy:
            st.error("API is not running. Start it with `make api`")
        else:
            with st.spinner("Generating forecasts..."):
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.min.time())
                
                predictions_df = get_predictions(start_dt, end_dt, selected_model)
                
                if not predictions_df.empty:
                    # Plot
                    fig = px.line(
                        predictions_df,
                        x="datetime",
                        y="predicted_demand_mw",
                        title="Predicted Electricity Demand",
                        labels={"predicted_demand_mw": "Demand (MW)", "datetime": "Time"}
                    )
                    fig.update_layout(xaxis_title="Time", yaxis_title="Demand (MW)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Demand", f"{predictions_df['predicted_demand_mw'].mean():,.0f} MW")
                    col2.metric("Peak Demand", f"{predictions_df['predicted_demand_mw'].max():,.0f} MW")
                    col3.metric("Min Demand", f"{predictions_df['predicted_demand_mw'].min():,.0f} MW")
                    
                    # Download
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "predictions.csv",
                        "text/csv",
                    )
                else:
                    st.warning("No predictions generated")


with tab2:
    st.header("Model Performance")
    
    if model_metadata:
        # Comparison table
        df_metrics = pd.DataFrame([
            {
                "Model": m["name"],
                "R²": m.get("metrics", {}).get("r2", "N/A"),
                "MAE (MW)": m.get("metrics", {}).get("mae", "N/A"),
                "MAPE (%)": m.get("metrics", {}).get("mape", "N/A"),
                "Trained": m.get("trained_at", "N/A")[:10],
            }
            for m in model_metadata
        ])
        
        st.dataframe(df_metrics, use_container_width=True)
        
        # Metrics comparison chart
        if len(model_metadata) > 1:
            fig = go.Figure()
            
            for m in model_metadata:
                metrics = m.get("metrics", {})
                fig.add_trace(go.Bar(
                    name=m["name"],
                    x=["R²", "MAE", "MAPE"],
                    y=[metrics.get("r2", 0), metrics.get("mae", 0) / 1000, metrics.get("mape", 0)],
                ))
            
            fig.update_layout(
                title="Model Comparison",
                barmode="group",
                yaxis_title="Value",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model metadata found. Train models with `make train`.")


with tab3:
    st.header("Data Explorer")
    
    # Check for cached data
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    if data_dir.exists():
        data_files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
        
        if data_files:
            selected_file = st.selectbox(
                "Select Data File",
                [f.name for f in data_files]
            )
            
            file_path = data_dir / selected_file
            
            try:
                if file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                st.dataframe(df.head(100), use_container_width=True)
                
                # Basic stats
                st.markdown("---")
                st.markdown("**Dataset Statistics:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{len(df):,}")
                col2.metric("Columns", len(df.columns))
                col3.metric("Date Range", f"{df.index.min()} to {df.index.max()}")
                
                # Plot
                if "demand_mw" in df.columns:
                    fig = px.line(df, y="demand_mw", title="Demand Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.info("No data files found. Download data with the API.")
    else:
        st.info("Data directory not found.")


# Footer
st.markdown("---")
st.markdown(
    """
    **Italian Electricity Demand Forecasting** | 
    [GitHub](https://github.com/ogiannopoulou/energy-demand-prediction) |
    Built with FastAPI + Streamlit + MLflow
    """
)
