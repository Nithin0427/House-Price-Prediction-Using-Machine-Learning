import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("house_price_model.pkl")

# Sample input: you can change these values
input_data = {
    "SquareFeet": 1600,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Neighborhood": "Suburb",  # Options: Rural, Suburb, Urban (based on your data)
    "YearBuilt": 2015
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
predicted_price = model.predict(input_df)[0]

# Format in Indian Rupees
def rupee_format(n):
    s = f"{int(n):,}"
    s = s.replace(",", ",")
    return f"₹{s}{f'.{int(n*100)%100:02d}'}"

print("📢 Predicted House Price:", rupee_format(predicted_price))
