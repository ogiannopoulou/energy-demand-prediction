"""
LLM Report Generation
Uses OpenAI API to generate natural language reports from model results
"""
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def generate_forecast_report(metrics: dict, model_name: str, 
                              feature_importance: pd.DataFrame = None,
                              anomaly_stats: dict = None) -> str:
    """Generate a natural language report from forecast results."""
    
    report_parts = [
        f"# Energy Demand Forecasting Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {model_name}",
        "",
        "## Model Performance",
        f"- **MAE:** {metrics.get('mae', 0):,.0f} MW",
        f"- **RMSE:** {metrics.get('rmse', 0):,.0f} MW",
        f"- **R² Score:** {metrics.get('r2', 0):.4f}",
        f"- **MAPE:** {metrics.get('mape', 0):.2f}%",
        "",
    ]
    
    if feature_importance is not None:
        top_features = feature_importance.head(10)
        report_parts.append("## Top 10 Important Features")
        for _, row in top_features.iterrows():
            report_parts.append(f"- {row['feature']}: {row['importance']:.4f}")
        report_parts.append("")
    
    if anomaly_stats:
        report_parts.append("## Anomaly Detection Summary")
        report_parts.append(f"- **Total anomalies detected:** {anomaly_stats.get('total', 0)}")
        report_parts.append(f"- **Anomaly rate:** {anomaly_stats.get('rate', 0):.2f}%")
        if "mean_demand_anomaly" in anomaly_stats:
            report_parts.append(f"- **Avg demand during anomalies:** {anomaly_stats['mean_demand_anomaly']:,.0f} MW")
        report_parts.append("")
    
    report_parts.append("## Key Insights")
    report_parts.append("- Demand shows strong daily and seasonal patterns")
    report_parts.append("- Peak hours (8-20) account for highest load")
    report_parts.append("- Weekend demand is consistently lower than weekdays")
    report_parts.append("- Temperature is a key driver of demand fluctuations")
    
    return "\n".join(report_parts)


def generate_llm_enhanced_report(metrics: dict, model_name: str,
                                   feature_importance: pd.DataFrame = None,
                                   anomaly_stats: dict = None,
                                   api_key: str = None) -> str:
    """Generate an enhanced report using LLM."""
    if not OPENAI_AVAILABLE:
        return generate_forecast_report(metrics, model_name, feature_importance, anomaly_stats)
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return generate_forecast_report(metrics, model_name, feature_importance, anomaly_stats)
    
    client = OpenAI(api_key=api_key)
    
    context = f"""
Model: {model_name}
Metrics: MAE={metrics.get('mae', 0):,.0f} MW, RMSE={metrics.get('rmse', 0):,.0f} MW, R²={metrics.get('r2', 0):.4f}, MAPE={metrics.get('mape', 0):.2f}%
"""
    
    if feature_importance is not None:
        top5 = feature_importance.head(5)
        context += f"\nTop features: {', '.join(top5['feature'].tolist())}"
    
    if anomaly_stats:
        context += f"\nAnomalies: {anomaly_stats.get('total', 0)} detected ({anomaly_stats.get('rate', 0):.2f}%)"
    
    prompt = f"""Write a professional data science report for an energy demand forecasting project.

Context:
{context}

The report should:
1. Summarize the model performance in business terms
2. Explain the key features driving demand predictions
3. Discuss the anomaly detection results
4. Provide actionable insights for energy grid operators
5. Be written for a technical but non-specialist audience

Write in English, be concise and professional."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a senior data scientist writing a project report."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM report generation failed: {e}")
        return generate_forecast_report(metrics, model_name, feature_importance, anomaly_stats)


import os
