import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Introduction
st.markdown("""
### Apartment Price Prediction in Polish Cities

This application allows users to interactively explore how different features influence apartment prices in major Polish cities.  
The data used to train the model was sourced from Kaggle, and the predictive model is based on an optimized **XGBoost Regressor (XGBRegressor)** with the following best hyperparameters:

- `colsample_bytree`: 1.0  
- `learning_rate`: 0.1  
- `max_depth`: 10  
- `n_estimators`: 200  
- `subsample`: 0.8

The model predicts apartment prices based on user-defined input features such as area, number of rooms, building age, proximity to amenities, and more.  
By adjusting the sliders below, you can examine how changes in individual variables affect the predicted apartment price across different cities.
""")

# Load the trained model
xgb = joblib.load("xgb_model.pkl")

# City mapping
city_mapping = {
    0: 'Białystok', 1: 'Bydgoszcz', 2: 'Częstochowa', 3: 'Gdańsk', 4: 'Gdynia',
    5: 'Katowice', 6: 'Kraków', 7: 'Łódź', 8: 'Lublin', 9: 'Poznań', 10: 'Radom',
    11: 'Rzeszów', 12: 'Szczecin', 13: 'Warszawa', 14: 'Wrocław'
}

# List of cities (encoded values)
cities = list(city_mapping.keys())

# Two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    # Sliders for input features
    squareMeters = st.slider("Area (m²)", 20, 150, 30)
    rooms = st.slider("Number of rooms", 1, 6, 1)
    collegeDistance = st.slider("Distance to the nearest university (km)", 0, 20, 2)
    centreDistance = st.slider("Distance to city center (km)", 0, 20, 10)
    floorCount = st.slider("Number of floors in the building", 1, 30, 2)
    poiCount = st.slider("Number of nearby points of interest", 0, 10, 3)
    hasElevator_yes = st.slider("Does the apartment have an elevator?", 0, 1, 1)
    age_of_building = st.slider("Building age (years)", 0, 100, 4)

# Predict apartment prices for all cities
predictions = []
for city in cities:
    input_data = {
        'squareMeters': squareMeters,
        'rooms': rooms,
        'collegeDistance': collegeDistance,
        'centreDistance': centreDistance,
        'floorCount': floorCount,
        'poiCount': poiCount,
        'hasElevator_yes': hasElevator_yes,
        'age_of_building': age_of_building,
        'city_encoded': city
    }

    X_new = pd.DataFrame([input_data])
    predicted_price = xgb.predict(X_new)
    predictions.append(predicted_price[0])

# Create DataFrame with results
df_predictions = pd.DataFrame({
    'City': [city_mapping[c] for c in cities],
    'Predicted Price (PLN)': predictions
})

# Sort values from highest to lowest
df_predictions = df_predictions.sort_values(by='Predicted Price (PLN)', ascending=False)

# Set bar colors (highlight Kraków in a pastel shade)
df_predictions['Color'] = df_predictions['City'].apply(lambda x: '#ffcccc' if x == 'Kraków' else '#cce7ff')

with col2:
    # Display the table
    st.subheader("Predicted Apartment Prices")
    st.dataframe(df_predictions, use_container_width=True)

    # Create a horizontal bar chart using Plotly
    fig = px.bar(df_predictions, x='Predicted Price (PLN)', y='City',
                 title="Predicted Apartment Prices Across Cities", orientation='h',
                 text=df_predictions['Predicted Price (PLN)'].apply(lambda x: f'{x:,.2f} PLN'),
                 color='Color', color_discrete_map={'#ffcccc': '#ffcccc', '#cce7ff': '#cce7ff'})

    # Reverse Y-axis order to show highest price at the top
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_showticklabels=False)

    # Format text labels to appear inside the bars
    fig.update_traces(texttemplate='%{text}', textposition='inside',
                      textfont=dict(family='Arial', size=16, color='black', weight='bold'))

    st.plotly_chart(fig)





