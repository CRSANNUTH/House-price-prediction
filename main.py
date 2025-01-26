import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, cv
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load your dataset
file_name = 'Bengaluru_House_Data_Cleaned.csv'  # Replace with your dataset file name
data = pd.read_csv(file_name)

# Handle missing values
data = data.dropna()

# Extract 'size' as numeric
data['size'] = data['size'].astype(str)
data['size'] = data['size'].str.extract(r'(\d+)').astype(float)

# Convert 'total_sqft' to numeric, handling ranges
def convert_sqft_to_num(value):
    try:
        return float(value)
    except:
        if '-' in value:
            values = value.split('-')
            return (float(values[0]) + float(values[1])) / 2
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)

# Drop rows with missing or invalid data
data = data.dropna()

# Encode categorical variables
label_encoder_area = LabelEncoder()
label_encoder_location = LabelEncoder()

data['area_type'] = label_encoder_area.fit_transform(data['area_type'])
data['location'] = label_encoder_location.fit_transform(data['location'])

# Prepare features and target
X = data[['area_type', 'location', 'size', 'total_sqft', 'bath', 'balcony']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBRegressor model
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

# Perform Cross-Validation with XGBoost's cv function
params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

dtrain = xgboost.DMatrix(X_train, label=y_train)

cv_results = cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    nfold=3,
    metrics="rmse",
    early_stopping_rounds=10,
    seed=42
)

# Best iteration from cross-validation
best_iteration = cv_results['test-rmse-mean'].idxmin()
print(f"Best iteration: {best_iteration}")

# Train model with best params
best_model = XGBRegressor(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=best_iteration,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

best_model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = best_model.predict(X_test)

# Calculate R² score (Accuracy)
r2 = r2_score(y_test, y_pred)
print(f"Accuracy (R² Score): {r2:.2f}")

# Function to predict the price for new input
def predict_price(area_type, location, size, total_sqft, bath, balcony):
    # Handle unseen labels for new inputs
    if area_type not in label_encoder_area.classes_:
        label_encoder_area.classes_ = list(label_encoder_area.classes_) + [area_type]
        label_encoder_area.fit(label_encoder_area.classes_)
    if location not in label_encoder_location.classes_:
        label_encoder_location.classes_ = list(label_encoder_location.classes_) + [location]
        label_encoder_location.fit(label_encoder_location.classes_)

    # Transform the input data
    area_type_encoded = label_encoder_area.transform([area_type])[0]
    location_encoded = label_encoder_location.transform([location])[0]

    # Create input features as a DataFrame
    input_features = pd.DataFrame([[area_type_encoded, location_encoded, size, total_sqft, bath, balcony]],
                                  columns=['area_type', 'location', 'size', 'total_sqft', 'bath', 'balcony'])

    # Predict the price
    predicted_price = best_model.predict(input_features)

    return predicted_price[0]

# Function to take input from the user
def predict_from_user():
    area_type = input(
        "Enter area type (e.g., Super built-up Area, Plot Area, Built-up Area, Villa, Studio Apartment): ")
    location = input("Enter location (e.g., Whitefield, Koramangala, Indiranagar): ")
    size = int(input("Enter size (e.g., number of bedrooms, 2, 3, etc.): "))
    total_sqft = float(input("Enter total square feet (e.g., 1200, 1500, etc.): "))
    bath = int(input("Enter number of bathrooms: "))
    balcony = int(input("Enter number of balconies: "))

    predicted_price = predict_price(area_type, location, size, total_sqft, bath, balcony)
    print(f"\nPredicted Price: {predicted_price:.2f} Lakhs")
    print("-" * 80)

# Run the input loop
while True:
    print("\n--- Predict House Price ---")
    predict_from_user()
    cont = input("Do you want to predict another house price? (yes/no): ").strip().lower()
    if cont != 'yes':
        break
