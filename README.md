# 📊 Apartment Price Prediction in Polish Cities (Streamlit)!

This application allows users to interactively explore how different features influence apartment prices in major Polish cities.
* **Python libraries:** plotly, pandas, streamlit, joblib, xgboost
* **Data source:** .csv.

The data used to train the model was sourced from Kaggle, and the predictive model is based on an optimized XGBoost Regressor (XGBRegressor) with the following best hyperparameters:

colsample_bytree: 1.0
learning_rate: 0.1
max_depth: 10
n_estimators: 200
subsample: 0.8

The model predicts apartment prices based on user-defined input features such as area, number of rooms, building age, proximity to amenities, and more.
By adjusting the sliders, you can examine how changes in individual variables affect the predicted apartment price across different cities.

👉 **Check out the live website here:** :
https://apartmentpriceprediction.streamlit.app/
