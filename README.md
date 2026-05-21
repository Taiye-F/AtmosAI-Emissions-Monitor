# AtmosAI: Industrial Emissions Compliance Monitor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-purple)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-red?logo=streamlit&logoColor=white)

**Transforming environmental enforcement from reactive to proactive.**

**AtmosAI** is an AI-driven early warning Regulatory Technology system. It utilizes machine learning to predict the probability of severe air quality violations 24 hours in advance, allowing for surgical, data-driven regulatory intervention.

**[Live Web Application - Click Here to Test the Dashboard](https://atmosai-emissions-monitor-yg2ygsn4uuepqyfbgahjp5.streamlit.app/)**

---

## The Problem vs. The Solution
Traditional environmental monitoring in Nigeria is often retrospective, relying on physical inspections after ecological damage has already occurred. 

**AtmosAI shifts this paradigm by:**
- **Live Telemetry Ingestion:** Autonomously consuming real-time PM10, PM2.5, and NO2 data via the Open-Meteo Satellite API across 11 key Nigerian industrial and control zones.
- **Predictive Intelligence:** Utilizing a bias-corrected XGBoost classifier to forecast imminent violations of World Health Organization (WHO) air safety limits.
- **Explainable AI (XAI):** Integrating SHAP (SHapley Additive exPlanations) to ensure absolute transparency and legal justification in regulatory decision-making.

---

## The Breakthrough: Bias Correction & The "Harmattan Effect"
A critical challenge in West African environmental modeling is the **"Harmattan Effect"**—where natural Saharan dust (PM10) in northern agrarian states (like Taraba or Benue) can trigger false positives, tricking basic AI models into flagging them as industrial polluters.

**The Technical Pivot:**
- Identified a significant geographic and environmental bias in baseline modeling.
- Engineered a bias-corrected model that prioritizes **Nitrogen Dioxide (NO2)** and **PM2.5** (chemical markers of industrial burning and illegal refining) over simple particulate volume.
- **Result:** Achieved a **96.44% Balanced Accuracy** that flawlessly distinguishes between natural atmospheric weather events and illegal industrial activity.

---

## Key Data Insights
Using SHAP Summary and Waterfall plots, the model exposed critical behavioral and environmental trends:
1. **The Weekend Gap:** Analysis revealed a spike in violation probability from Friday evening to Sunday, suggesting that industries may capitalize on reduced weekend government oversight to vent emissions.
2. **Feature Importance:** Nitrogen Dioxide (NO2) was identified as the primary "smoking gun" for anthropogenic (man-made) pollution, superseding PM10.
3. **Infrastructure Independence:** The AI learned to prioritize live atmospheric chemistry over official "Power Plant Counts," successfully detecting unregistered/illegal refining hubs that exist off the official grid.

---

## Tech Stack & Architecture
- **Language:** Python 3.11
- **Machine Learning:** Scikit-Learn, XGBoost (Extreme Gradient Boosting)
- **Explainable AI:** SHAP
- **Data Engineering:** REST APIs (Open-Meteo ERA5), Pandas, NumPy
- **Frontend & Deployment:** Streamlit, Streamlit Community Cloud

---

Developed by Taiye Fagbolade
