import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("housing_price_dataset.csv")

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Column types
categorical_features = ["Neighborhood"]
numerical_features = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features)
])

# Pipeline with Linear Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Format function for ₹
def rupee(n):
    return f"₹{n:,.2f}"

# Print results
print("\n✅ Model Performance on Test Data")
print(f"Mean Absolute Error (MAE): {rupee(mae)}")
print(f"Root Mean Squared Error (RMSE): {rupee(rmse)}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

# Save model
joblib.dump(model, "house_price_model.pkl")
print("\n✅ Model saved as 'house_price_model.pkl'")
