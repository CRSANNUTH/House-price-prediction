from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)


# Dummy model for demonstration purposes (replace with your actual model)
# In a real app, load the trained model using pickle or joblib
# model = pickle.load(open('model.pkl', 'rb'))  # Example for loading a model

# Route for displaying the form and predicting
@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    accuracy = None
    if request.method == "POST":
        # Get form data
        area_type = request.form["area_type"]
        location = request.form["location"]
        size = int(request.form["size"])
        total_sqft = float(request.form["total_sqft"])
        bath = int(request.form["bath"])
        balcony = int(request.form["balcony"])

        # For demonstration, we're assuming you have a prediction model
        # Dummy prediction logic (replace with real prediction code)
        features = np.array([[size, total_sqft, bath, balcony]])  # Use the relevant features for your model
        price = predict_price(features)  # Replace this with your actual model prediction

        # Here you can calculate and print the model's accuracy to the console
        # For now, we're assuming the accuracy is just an example value
        accuracy = 0.85  # Dummy accuracy (replace with actual accuracy calculation)

        print(f"Model Accuracy: {accuracy * 100}%")  # Print accuracy to the console

        return render_template(
            "form.html",
            area_type=area_type,
            location=location,
            size=size,
            total_sqft=total_sqft,
            bath=bath,
            balcony=balcony,
            price=f"{price:,.2f}",
            accuracy=f"{accuracy * 100:.2f}%"
        )

    return render_template("form.html")


# Dummy function to simulate price prediction (replace with your real model prediction)
def predict_price(features):
    # Dummy price prediction logic (use actual model prediction in real app)
    price = features[0][1] * 5000  # Example: price = sqft * 5000
    return price


if __name__ == "__main__":
    app.run(debug=True)
