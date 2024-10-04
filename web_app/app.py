import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the heart disease prediction model using the updated path
model_path = r'C:\Users\ABDUL RAHMAN\Desktop\Disease-Project\models\my_model.h5'
print(f"Loading model from: {model_path}")  # Debug print to verify path
model1 = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model1', methods=['GET', 'POST'])
def model1_page():
    if request.method == 'POST':
        # Get form data
        input_data = [float(request.form['age']),
                      float(request.form['sex']),
                      float(request.form['cp']),
                      float(request.form['trestbps']),
                      float(request.form['chol']),
                      float(request.form['fbs']),
                      float(request.form['restecg']),
                      float(request.form['thalach']),
                      float(request.form['exang']),
                      float(request.form['oldpeak']),
                      float(request.form['slope']),
                      float(request.form['ca']),
                      float(request.form['thal'])]
        input_data = np.array([input_data])
        
        # Reshape input data to match the model's expected input shape
        input_data = np.resize(input_data, (1, 100))  # Adjust this to match your model's input shape
        
        # Make prediction
        prediction = model1.predict(input_data)
        result = 'Heart Disease Detected' if prediction[0][0] > 0.5 else 'No Heart Disease'

        return render_template('model1.html', result=result)
    return render_template('model1.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
