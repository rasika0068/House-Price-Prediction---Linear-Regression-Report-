# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------
# 1. Load Dataset
# -------------------------
# Example: Kaggle "House Prices - Advanced Regression Techniques"
# Download CSV from Kaggle and place it in the same folder
df = pd.read_csv("house_prices.csv")

print("Dataset shape:", df.shape)
print(df.head())

# -------------------------
# 2. Preprocessing
# -------------------------
# Drop irrelevant columns (like IDs if present)
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# Handle missing values (fill with median for numeric, mode for categorical)
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical variables
label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col])

# -------------------------
# 3. Feature Selection
# -------------------------
# Target variable
y = df["SalePrice"]   # Adjust if dataset target is different
X = df.drop("SalePrice", axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 4. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------
# 5. Train Linear Regression Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# 6. Predictions & Evaluation
# -------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# -------------------------
# 7. Visualization
# -------------------------
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# -------------------------
# 8. Example Prediction
# -------------------------
# Example: Predict for first test row
sample = X_test[0].reshape(1, -1)
predicted_price = model.predict(sample)[0]
print("\nExample Prediction:")
print("Actual Price:", y_test.iloc[0])
print("Predicted Price:", predicted_price)
