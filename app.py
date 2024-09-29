from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('models/best_model.pkl')

feature_names = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
                 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service',
                 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
                 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 
                 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
                 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 
                 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
                 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 
                 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'BothStreamingServices', 'AvgMonthlyCharge']

# Define the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (JSON format)
    try:
        # Get JSON data from POST request
        input_data = request.json

        # Convert input data to a DataFrame using the correct feature names
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)
        print(prediction)
        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        })


@app.route('/', methods=['GET'])
def welcome():
    # Get the data from the POST request (JSON format)
    return "Server running successfully"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
