
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housingprice-prediction.streamlit.app/)

[![All Rights Reserved](https://img.shields.io/badge/Copyright-All%20Rights%20Reserved-red)](https://github.com/SweetySeelam2/Housing_Price_Prediction)

> **All content, models, and code are fully copyrighted.  
> No reuse, redistribution, or commercial use allowed without written permission.**

---

# 🏠 Housing Price Prediction & Model Explainability App

**A full-scale, production-ready, ML-powered web application for accurate housing price prediction and model explainability, built by [Sweety Seelam](https://www.linkedin.com/in/sweetyrao670/).**

---

## 🚀 Project Overview

Housing price prediction is a critical real-world problem for real estate, banking, fintech, and proptech sectors.  
This application enables you to **predict house prices** using advanced machine learning models, visualize model performance, and understand predictions via SHAP and LIME explainability.

---

## 💼 **Business Problem:**  
  Accurate price estimation enables smarter investments, transparent lending, risk reduction, and optimized sales for platforms like Housing.com, Realtor.com, Zillow, Redfin, and major banks.

---

## ✨ **Key Features:**  
  - Large-scale regression models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting (all trained & tuned).
  - Upload your own data or test on a live demo sample.
  - **Model Explainability:** See why each prediction is made using SHAP & LIME.
  - Multi-page interactive navigation, error analysis, and visual analytics.
  - Business value, recommendations, and ROI simulation for companies.
  - Original dataset: **Indian Rupees (₹)**; all predictions displayed as `₹INR / $USD` (with conversion at 1 USD = 83 INR, Jun 2025).
  - MIT License, fully credited and open for learning, research, or enterprise use.

---

## 📊 Dataset

- **Source:** [Housing Prices Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)  
- **Original author (code inspiration for linear/OLS):** [Aakash, Analytics Vidhya](https://medium.com/analytics-vidhya/applying-multiple-linear-regression-in-house-price-prediction-47dacb42942b)

**Features:**
| Feature                | Type        | Description                               |
|------------------------|-------------|-------------------------------------------|
| price                  | int         | Sale price (target variable)              |
| area                   | int         | Plot area (sq. ft.)                       |
| bedrooms               | int         | Number of bedrooms                        |
| bathrooms              | int         | Number of bathrooms                       |
| stories                | int         | Number of stories                         |
| mainroad               | yes/no      | Is house on main road?                    |
| guestroom              | yes/no      | Is guestroom present?                     |
| basement               | yes/no      | Is basement present?                      |
| hotwaterheating        | yes/no      | Hot water heating available?              |
| airconditioning        | yes/no      | Air conditioning available?               |
| parking                | int         | Number of parking spaces                  |
| prefarea               | yes/no      | Is house in preferred area?               |
| furnishingstatus       | categorical | Furnishing: furnished/semi/unfurnished    |

---

## 📈 Model Performance (Test Set)

All results are **from your final Jupyter analysis and used in the deployed app**.

| Model               | R² Score | MSE         | MAE              |
|---------------------|----------|-------------|------------------|
| Gradient Boosting   | 0.70     | 9.97e+11    | ₹706,979 / $8,522 |
| Ridge Regression    | 0.70     | 9.98e+11    | ₹734,026 / $8,847 |
| Lasso Regression    | 0.69     | 1.00e+12    | ₹735,049 / $8,859 |
| Random Forest       | 0.64     | 1.19e+12    | ₹776,471 / $9,356 |
| Linear Regression   | 0.69     | 1.00e+12    | ~₹735,000 / $8,859 |

- **Interpretation:**  
  - Best model (Gradient Boosting) explains 70% of price variance, with average prediction error (MAE) of about ₹707K / $8.5K per house.
  - Business-level predictions are robust, though errors may increase for rare/unusual properties.

---

## ⚙️ App Features

- **Multi-Model Selection:** Choose among 5 trained ML models.
- **Live Prediction:**  
  - Use demo sample or upload your own `.csv` with identical features.
  - Editable input grid for rapid scenario analysis.
- **Explainability (SHAP & LIME):**  
  - Visualize feature importance for both the dataset and individual cases.
  - Interactive SHAP waterfall & summary plots.
  - LIME HTML visualization for human-interpretability.
- **Business Value Module:**  
  - ROI simulation, business case studies, and actionable recommendations for real estate companies and banks.
  - see [Zillow Group News](https://investors.zillowgroup.com/investors/news-and-events/news/default.aspx) for industry trends.
- **Universal Currency Support:**  
  - All outputs shown as `₹INR / $USD`, so global users can interpret immediately.
  - Note on currency shown at bottom of each page for transparency.
- **Attractive, large-scale Streamlit UI:**  
  - Streamlit [live demo badge](https://housingprice-prediction.streamlit.app/), responsive layout, project branding, and MIT copyright.
- **Open, Reproducible & Cited:**  
  - All code, models, and explainability available in the repo and app.
  - Dataset, methodology, and external sources are cited throughout.

---

## 🚀 Try the Live App - How to Use?
                                                                  
1. **Open the App:**                                                   
   [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housingprice-prediction.streamlit.app/)
2. **Explore the Data:**  
   See the dataset sample and core visuals on the "Data Exploration" page.
3. **Predict Housing Prices:**  
   - Select a model (default: Gradient Boosting, best performing).
   - Edit demo features or upload your own data.
   - Click "Predict!" to see predictions and error bands.
4. **Explain Model Decisions:**  
   - Go to "Explainability (SHAP & LIME)".
   - Explore feature importance and the drivers for specific predictions.
5. **Business Value:**  
   - See potential impact and recommendations for real-world adoption.

---

## 🏗️ Project Structure

├── app.py                                                                       
├── README.md                                                                                         
├── requirements.txt                                                                                            
├── Housing_LinearReg_Gradient.ipynb                                                                                     
├── Housing.csv                                                                                                
├── demo_sample.csv                                                                                            
├── feature_scaler.pkl                                                                                                        
├── feature_names.npy                                                                                                  
├── linear_regression_model.pkl                                                        
├── ridge_regression_model.pkl                                                                            
├── lasso_regression_model.pkl                                                                         
├── random_forest_model.pkl                                                                             
├── gradient_boosting_model.pkl                                                                                     
├── lime_rf_example.html                                                                                    
├── images/                                                                                                
│ ├── Actual vs Predicted plot.png                                                                                  
│ ├── RegressionModel-MetricsPerformance.png                                                                            
│ ├── Histogram-ResidualDistribution.png                                                                             
│ ├── shap_summary_gbr.png                                                     

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/SweetySeelam2/Housing_Price_Prediction.git
cd Housing_Price_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📚 References & Credits
**Dataset:**                                
[Kaggle: Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

**Linear Regression/OLS code:**                                    
Adapted from [Aakash's Analytics Vidhya Medium Article](https://medium.com/analytics-vidhya/applying-multiple-linear-regression-in-house-price-prediction-47dacb42942b)

**Business Impact:**                                 
[Zillow Group News](https://investors.zillowgroup.com/investors/news-and-events/news/default.aspx)

**Explainability:**                                              
Interpretable Machine Learning Book (SHAP & LIME)

***Note:***
- All price values are displayed as ₹INR / $USD (at 1 USD = 83 INR, June 2025).
- This project is designed for research, business simulation, and professional demonstration.

---

## 👩‍💻 Author:                               
**Sweety Seelam** | Business Analyst | Aspiring Data Scientist                                         
[LinkedIn](https://www.linkedin.com/in/sweetyrao670/)                                     
[GitHub](https://github.com/SweetySeelam2/Housing_Price_Prediction)                                                            
[Portfolio](https://sweetyseelam2.github.io/SweetySeelam.github.io/)
[Email](sweetyseelam2@gmail.com)                    

---

## 🔒 Proprietary & All Rights Reserved
© 2025 Sweety Seelam. This work is proprietary and protected by copyright. All content, models, code, and visuals are © 2025 Sweety Seelam. No part of this project, app, code, or analysis may be copied, reproduced, distributed, or used for any purpose—commercial or otherwise—without explicit written permission from the author.

For licensing, commercial use, or collaboration inquiries, please contact: Email: sweetyseelam2@gmail.com
