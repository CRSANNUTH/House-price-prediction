# Real Estate Price Prediction using Flask

## About the Project

This is a Flask-based web application that predicts real estate prices based on user inputs such as area type, location, size, total square feet, number of bathrooms, and balconies.


## 🌟 Algorithm Explanation

The real estate price prediction model follows the following key steps:

### 1️⃣ **Data Preprocessing**
   - **Missing Data Handling**: Rows with missing values are removed to ensure the dataset is clean and ready for training.
   - **Size Conversion**: The "size" feature, which represents the number of bedrooms in the property, is extracted as a numeric value.
   - **Total Square Feet Conversion**: The `total_sqft` feature, which sometimes contains ranges (e.g., "1200-1400"), is converted into a single numeric value by averaging the range.
   - **Categorical Data Encoding**: The categorical features, **area_type** and **location**, are converted into numerical values using **Label Encoding**.

### 2️⃣ **Model Training and Hyperparameter Tuning**
   - **Training Set Split**: The dataset is split into a training set (80%) and a testing set (20%).
   - **XGBoost Regressor**: The model uses the **XGBRegressor**, a variant of the XGBoost algorithm, to predict continuous values (house prices).
   - **Cross-Validation**: We perform **cross-validation** to find the optimal number of boosting rounds (iterations), preventing overfitting and ensuring the best model performance. The evaluation metric used is **RMSE** (Root Mean Squared Error).
   - **Hyperparameters**:
     - **n_estimators**: Number of boosting rounds (200).
     - **max_depth**: Maximum depth of trees (6).
     - **learning_rate**: Learning rate for boosting rounds (0.05).
     - **subsample** and **colsample_bytree**: Used for regularization to avoid overfitting (0.9).

### 3️⃣ **Model Evaluation**
   After training, the model's performance is evaluated on the test set using the **R² Score** (coefficient of determination), which measures how well the model’s predictions match the actual values. An R² score close to 1 indicates good model accuracy.

### 4️⃣ **Prediction**
   The trained model is used to predict the house price for new user inputs. The user provides the following details:
   - **Area Type**: The type of the area (e.g., Super built-up Area, Plot Area).
   - **Location**: The location of the property.
   - **Size**: The number of bedrooms.
   - **Total Square Feet**: The area in square feet.
   - **Number of Bathrooms**.
   - **Number of Balconies**.
   
   These inputs are preprocessed (e.g., encoding and feature transformation) before being fed into the model for price prediction.

### 5️⃣ **Interactive Prediction**
   The model interacts with the user through a command-line interface. The user can input their property details, and the model predicts the house price in real-time.

## Application Interface

Here is an example of the web form users can fill out to predict property prices:

![Form Screenshot](screenshots/image.png "Form Interface")

---



---

## Application Architecture

The following diagram illustrates the architecture of the app:

![Architecture Diagram](screenshots/image2.png "Application Architecture")

---

## Features

- User-friendly web interface for data input.
- Real-time price prediction based on user inputs.
- Displays model accuracy (currently hardcoded as a placeholder).
- Built using Flask for backend and Python for logic.

## Usage

1. Open your browser and navigate to `http://127.0.0.1:5000/`.
2. Enter the property details into the form.
3. Click "Submit" to get the predicted price.


## Prerequisites
Before running the application, ensure you have the following installed:
- Python 3.x
- Flask
- NumPy
- Pickle (if using a saved model)

Install the required dependencies:
```bash
pip install flask numpy
