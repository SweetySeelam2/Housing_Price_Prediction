import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Housing Price Prediction & Explainability",
                   layout="wide", page_icon="🏠")

USD_CONV = 83  # CURRENCY: 1 USD = 83 INR

def inr_usd_fmt(amount):
    dollars = amount / USD_CONV
    return f"₹{amount:,.0f} / ${dollars:,.0f}"

def currency_note():
    st.markdown(
        "<div style='color:gray; font-size:0.95em; margin-top:1em'>"
        "Note: The original dataset used Indian rupees (INR, ₹) as the currency. "
        "All prices and error metrics are shown as <b>₹ INR / $ USD</b> (USD conversion at 1 USD = 83 INR, June 2025). "
        "Interpret results accordingly for your region.</div>", unsafe_allow_html=True
    )

@st.cache_resource
def load_artifacts():
    models = {
        'Linear Regression': joblib.load('linear_regression_model.pkl'),
        'Ridge Regression': joblib.load('ridge_regression_model.pkl'),
        'Lasso Regression': joblib.load('lasso_regression_model.pkl'),
        'Random Forest': joblib.load('random_forest_model.pkl'),
        'Gradient Boosting': joblib.load('gradient_boosting_model.pkl')
    }
    scaler = joblib.load('feature_scaler.pkl')
    feature_names = np.load('feature_names.npy', allow_pickle=True)
    return models, scaler, feature_names

models, scaler, feature_names = load_artifacts()

MODEL_METRICS = {
    "Gradient Boosting": {"r2": 0.70, "mae": 706979, "mse": 9.97e11},
    "Ridge Regression":  {"r2": 0.70, "mae": 734026, "mse": 9.98e11},
    "Lasso Regression":  {"r2": 0.69, "mae": 735049, "mse": 1.00e12},
    "Random Forest":     {"r2": 0.64, "mae": 776471, "mse": 1.19e12},
    "Linear Regression": {"r2": 0.62, "mae": 787090, "mse": 1.22e12},
}

# --- Demo data ---
DEMO_PATH = 'demo_sample.csv'
if not os.path.exists(DEMO_PATH):
    DEMO_DF = pd.DataFrame(
        [[7420, 4, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0]],
        columns=feature_names
    )
    DEMO_DF.to_csv(DEMO_PATH, index=False)
else:
    DEMO_DF = pd.read_csv(DEMO_PATH)

# --- Sidebar ---
st.sidebar.title("🏠 Housing Price App")
page = st.sidebar.radio("Go to:", [
    "Project Overview",
    "Data Exploration",
    "Predict & Test",
    "Explainability (SHAP & LIME)",
    "Business Value & Recommendations",
    "About & Credits"
])

def footer():
    st.markdown(
        "<hr> <span style='font-size:0.85em'> Proprietary &copy; 2025 Sweety Seelam – All Rights Reserved. </span>",
        unsafe_allow_html=True
    )

# --- 1. Project Overview ---
if page == "Project Overview":
    st.title("🏠 Housing Price Prediction & Explainability App")
    st.markdown(f"""
    **Built by Sweety Seelam**

    - **Business Problem:** Accurately predicting housing prices is critical for real estate companies, property tech firms, and financial institutions to optimize investments, reduce risks, and maximize returns.
    - **Dataset:** [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
    - **Models Used:** Linear Regression, Ridge Regression, Lasso Regression, Random Forest, Gradient Boosting.
    - **Key Results:**  
        - Best model R²: **0.70** (Gradient Boosting)  
        - MAE: {inr_usd_fmt(700000)} (absolute error)
        - SHAP & LIME explainability for every prediction
    - **Business Impact:**  
      Models like these enable smarter property investments, targeted marketing, and risk reduction, saving millions for companies like **Zillow, Redfin, Realtor.com, Housing.com**, and major banks.
    - **Live Demo:** Test on real data, see prediction drivers, or upload your own!
    """)
    st.image("images/Actual vs Predicted plot.png", caption="Actual vs. Predicted Prices")
    st.markdown("""
    <b>Interpretation:</b><br>
    - Each dot shows a house: x = actual sale price, y = predicted price.<br>
    - The blue line shows ideal fit (perfect prediction).<br>
    - Most points closely follow the line, showing the model is accurate for the majority of cases.<br>
    - Some spread at high prices indicates a typical MAE of about {mae} (see next pages for full metrics).
    """.format(mae=inr_usd_fmt(700000)), unsafe_allow_html=True)
    currency_note()
    footer()

# --- 2. Data Exploration ---
elif page == "Data Exploration":
    st.title("🔍 Data Exploration")
    st.markdown("Preview the data and check key visualizations:")
    st.dataframe(pd.read_csv("Housing.csv").head(10))
    st.image("images/RegressionModel-MetricsPerformance.png", caption="Regression Model Metrics")
    st.markdown("""
    <b>Interpretation (Linear Regression):</b><br>
    - Linear Regression serves as the baseline model.<br>
    - Captures overall trends but not complex non-linear effects.<br>
    - Shows higher error (MAE, MSE) and lower R² than advanced models.<br>
    - Best used for quick, interpretable estimates or as a benchmark.<br>
    - For business use, ensemble models (like Gradient Boosting) are preferred for higher accuracy and lower risk.
    - Error is measured in ₹/$. Higher R² means better prediction.
    """, unsafe_allow_html=True)
    st.image("images/Histogram-ResidualDistribution.png", caption="Residuals Distribution")
    st.markdown("""
    <b>Interpretation:</b><br>
    - Shows difference between predicted and actual price.<br>
    - Centered around zero, suggesting unbiased predictions.<br>
    - Symmetric, bell-shaped curve indicates no major outliers or bias.
    """, unsafe_allow_html=True)
    st.image("images/shap_summary_gbr.png", caption="SHAP Summary: Gradient Boosting")
    st.markdown("""
    <b>Interpretation:</b><br>
    - Top features: area, bathrooms, air conditioning, stories, parking.<br>
    - Blue/red colors show whether feature increases/decreases price.<br>
    - Larger values on right have strongest effect on raising predicted price.
    """, unsafe_allow_html=True)
    currency_note()
    footer()

# --- 3. Predict & Test ---
elif page == "Predict & Test":
    st.title("🏠 Predict Housing Price")
    st.write("Choose a model and test with our sample or upload your own data!")

    MODEL_INTERPRETATION = {
        "Gradient Boosting": (
            "This is the most accurate model, capturing complex relationships and minimizing error—ideal for business-critical pricing."
        ),
        "Ridge Regression": (
            "Very close to Gradient Boosting; robust to overfitting but slightly higher error. Suits linear patterns with small non-linearities."
        ),
        "Lasso Regression": (
            "Provides a simple, interpretable model that highlights key features, though with marginally higher error. Good for feature selection."
        ),
        "Random Forest": (
            "Captures non-linear patterns but is less accurate than Gradient Boosting. Useful for quick, robust baseline predictions."
        ),
    }

    with st.expander("ℹ️ Model Results & Metrics (from the project's Jupyter analysis)"):
        for name in ["Gradient Boosting", "Ridge Regression", "Lasso Regression", "Random Forest"]:
            m = MODEL_METRICS[name]
            st.markdown(
                f"""
                - **{name}:**
                    - R²: **{m['r2']:.2f}**
                    - MSE: {m['mse']:.2e}
                    - MAE: {inr_usd_fmt(m['mae'])}
                    - *Interpretation:* {MODEL_INTERPRETATION[name]}
                """
            )

    model_name = st.selectbox("Select model:", list(models.keys()), index=4)
    input_method = st.radio("Input data:", ["Use Demo Sample", "Upload Your CSV"])

    if input_method == "Use Demo Sample":
        input_df = DEMO_DF.copy()
        st.success("Demo sample loaded. You can edit below:")
        input_df = st.data_editor(input_df, num_rows="dynamic")
    else:
        uploaded = st.file_uploader("Upload your CSV file (must match model features)", type="csv")
        if uploaded:
            input_df = pd.read_csv(uploaded)
            st.success("File uploaded!")
            st.dataframe(input_df.head())
        else:
            st.warning("Upload a file to continue.")
            input_df = None

    if input_df is not None and st.button("Predict!"):
        input_scaled = scaler.transform(input_df)
        preds = models[model_name].predict(input_scaled)
        input_df['Predicted Price'] = preds.astype(int)
        st.dataframe(input_df)

        mean_pred = np.mean(preds)
        mean_pred_usd = mean_pred / USD_CONV
        st.success(
            f"Prediction completed! Avg predicted price: ₹{mean_pred:,.0f} / ${mean_pred_usd:,.0f}"
        )

        # Dynamic interpretation per selected model
        metrics = MODEL_METRICS.get(model_name, {"r2": "N/A", "mae": "N/A"})
        st.markdown(f"""
        **Interpretation:**  
        - The predicted prices reflect features like area, bathrooms, bedrooms, and amenities.
        - Based on your selection, the typical error margin is {inr_usd_fmt(metrics['mae'])} (MAE).
        - This model explains up to {metrics['r2']:.0%} of the variance in housing prices in the test set.
        """)
    currency_note()
    footer()

# --- 4. Explainability (SHAP & LIME) ---
elif page == "Explainability (SHAP & LIME)":
    st.title("🔬 Model Explainability (SHAP & LIME)")
    st.markdown("See exactly *why* the model predicts these prices! View SHAP summary or test an individual case.")

    st.image("images/shap_summary_gbr.png", caption="SHAP Feature Importance – Gradient Boosting")
    st.markdown("""
    <b>Interpretation:</b><br>
    - Features like area, bathrooms, AC, stories, parking have most impact.<br>
    - Blue dots: lower feature value, Red dots: higher feature value.<br>
    - Farther from zero = greater impact on prediction.<br>
    """, unsafe_allow_html=True)

    with st.expander("🧪 Try SHAP or LIME on Demo Row:"):
        row = DEMO_DF.iloc[[0]]
        model = models["Gradient Boosting"]
        explainer = shap.Explainer(model, DEMO_DF)
        shap_values = explainer(row)
        st.write("Input:", row)
        plt.close('all')  # Prevents plot stacking
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)
        st.pyplot(plt.gcf())  # Display current figure
        st.info("Blue = decreases price; Red = increases price")
        st.markdown("""
        <b>Interpretation:</b><br>
        - Waterfall plot: Each bar shows how much a feature moves the prediction up or down.<br>
        - Most influential = area, bathrooms, stories, AC, prefarea.<br>
        - Explains the model's reasoning for this specific house.<br>
        """, unsafe_allow_html=True)
        st.markdown("Or see the full HTML LIME output below:")
        with open("lime_rf_example.html", "r", encoding="utf-8") as f:
            html_data = f.read()
            st.components.v1.html(html_data, height=400, scrolling=True)
        st.markdown("""
        <b>Interpretation (LIME):</b><br>
        - Bar chart shows positive (orange) and negative (blue) feature contributions.<br>
        - Useful for showing which features increase or decrease the prediction in plain English.<br>
        """, unsafe_allow_html=True)
    currency_note()
    footer()

# --- 5. Business Value & Recommendations ---
elif page == "Business Value & Recommendations":
    st.title("💡 Business Value, Impact & Recommendations")
    st.markdown(f"""
**Business Impact:**  
- **For Real Estate Companies:**  
    - Enables accurate price setting, improved profitability, and trust with customers.
- **For Portals/FinTechs (e.g., Housing.com, Realtor.com, Zillow):**  
    - Drives smarter recommendations, targeted ads, and risk reduction in mortgages.
- **Model Performance:**  
    - Up to **70% of housing price variation explained** (R² = 0.70)
    - Error margin of ~{inr_usd_fmt(700000)} (MAE)
    - Real-world iBuying losses like Zillow’s $881M underscore the need for advanced, explainable ML to avoid costly business failures and regulatory penalties [Zillow’s Shuttered Home-Flipping Business](https://www.wsj.com/business/earnings/zillows-shuttered-home-flipping-business-lost-881-million-in-2021-11644529656)
    - Can help save millions annually in mispricing costs for large portfolios (see [Zillow Group News](https://investors.zillowgroup.com/investors/news-and-events/news/default.aspx) for industry trends)
- **Adoption Effect:**  
    - Can reduce overpricing/underpricing errors by >30%, increasing transaction speed and volume.
    - Transparency via SHAP/LIME improves trust and regulatory compliance.

**Recommendations:**  
- Deploy this model for:
    - Automated pricing on listings
    - Portfolio valuation for banks/investors
    - Customer-facing price calculators
- Combine with external factors (location scores, market trends) for even higher accuracy.
- Integrate explainability for transparency with stakeholders.

**Relevant Companies:**  
- **Housing.com, Realtor.com, Zillow, Redfin, Opendoor, Bank of America, Quicken Loans**

**References:**  
- [Housing Prices Dataset – Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- [Zillow’s Shuttered Home-Flipping Business – WSJ, 2022](https://www.wsj.com/business/earnings/zillows-shuttered-home-flipping-business-lost-881-million-in-2021-11644529656)
- [Zillow Group News – Industry Trends (2025)](https://investors.zillowgroup.com/investors/news-and-events/news/default.aspx)
- [Interpretable Machine Learning Book (SHAP & LIME, Regulatory Trends, 2024)](https://christophm.github.io/interpretable-ml-book/)
- [Analytics Vidhya – Linear Regression Code](https://medium.com/analytics-vidhya/applying-multiple-linear-regression-in-house-price-prediction-47dacb42942b)
""")
    currency_note()
    footer()

# --- 6. About & Credits ---
elif page == "About & Credits":
    st.title("ℹ️ About & Credits")
    st.markdown("""
    **Built by Sweety Seelam** 
    🔒 License & Usage:                         
    Proprietary & All Rights Reserved                         
    © 2025 Sweety Seelam.                 
    This work is proprietary and protected by copyright.
    No part of this project, app, code, or analysis may be copied, reproduced, distributed, or used for any purpose—commercial or otherwise—without explicit written permission from the author. 

    - **Dataset:** [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
    - **Linear Regression & OLS code:** Adapted from [Aakash's Analytics Vidhya Medium Article](https://medium.com/analytics-vidhya/applying-multiple-linear-regression-in-house-price-prediction-47dacb42942b)
    - **Hyperparameter models, explainability, business analysis, recommendations:** Developed by Sweety Seelam
    - **Live Demo Data:** See demo_sample.csv or try your own file!
    - **References:**
        - SHAP, LIME documentation for explainability
        - Zillow, Redfin, Realtor.com (for business impact estimation)
                
    **Contact:**  
    - [LinkedIn](https://www.linkedin.com/in/sweetyrao670/)  
    - [GitHub](https://github.com/SweetySeelam2/Housing_Price_Prediction)
    - [Email](sweetyseelam2@gmail.com)
    """)
    currency_note()
    footer()