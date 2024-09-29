from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('models/best_model.pkl')

# Define the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (JSON format)
    data = request.get_json()

    # Extract the features from the input JSON
    features = np.array([data['features']])  # Ensure input is a 2D array

    # Make a prediction
    prediction = model.predict(features)
   
    
    # Return the prediction result as JSON
    return jsonify({
        'prediction': "Yes" if int(prediction[0]) else "No"
    })


@app.route('/', methods=['GET'])
def welcome():
    # Get the data from the POST request (JSON format)
    return "Server running successfully"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
