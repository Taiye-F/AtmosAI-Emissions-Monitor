# AtmosAI: Industrial Emissions Compliance Monitor (RegTech)

**Transforming environmental enforcement from reactive to proactive.**

AtmosAI is an AI-driven early warning system designed for **NESREA** (Nigeria's National Environmental Standards and Regulations Enforcement Agency). It utilizes machine learning to predict the probability of air quality violations 24 hours in advance, allowing for surgical regulatory intervention.

[Live Web Application - Click Here to View](https://atmosai-emissions-monitor-yg2ygsn4uuepqyfbgahjp5.streamlit.app/)

## The Solution
Traditional environmental monitoring in Nigeria is often retrospective. AtmosAI shifts the paradigm by:
- **Live Telemetry Ingestion:** Consuming real-time PM10, PM2.5, and NO2 data via the Open-Meteo Satellite API across 11 key Nigerian zones.
- **Predictive Intelligence:** Utilizing a bias-corrected XGBoost classifier to forecast violations of World Health Organization (WHO) safety limits.
- **Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) and The Local "Waterfall" Plot (The "Bias-Breaker") to ensure transparency in regulatory decision-making.

## The Breakthrough: Bias Correction
A critical challenge in West African environmental modeling is the **"Harmattan Effect"**—where natural Saharan dust (PM10) in northern states (like Taraba) can be mistaken for industrial pollution.

**My Technical Pivot:**
- Identified a significant geographic bias in initial models.
- Retrained the model to prioritize **Nitrogen Dioxide (NO2)** and **PM2.5** (industrial markers) over simple particulate volume.
- **Result:** Achieved a **96.44% Balanced Accuracy** that accurately distinguishes between natural weather events and illegal industrial activity.

## Key Insights
- **The Weekend Gap:** SHAP analysis revealed a spike in violation probability from Friday to Sunday, suggesting industries may capitalize on reduced weekend oversight.
- **Feature Importance:** Nitrogen Dioxide was identified as the primary "smoking gun" for anthropogenic (man-made) pollution.

## Tech Stack
- **Language:** Python 3.11
- **Modeling:** XGBoost (Extreme Gradient Boosting)
- **Explainability:** SHAP
- **Data Pipeline:** REST APIs (Open-Meteo), Pandas, NumPy
- **Interface:** Streamlit Cloud

## Team AtmosAI


