import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load models and pre-trained PCA transformer
model1_path = r'C:\Users\ABDUL RAHMAN\Desktop\Disease-Project\models\my_model.h5'
pca_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'pca_model.pkl'))

# Load the saved model, scaler, and PCA
random_search_loaded = joblib.load('alzheimers_model_up.pkl')
scaler_al = joblib.load('scaler.pkl')
pca_al = joblib.load('pca.pkl')
model2 = random_search_loaded.best_estimator_

print(f"Loading model1 from: {model1_path}")
print(f"Loading PCA from: {pca_path}")

model1 = load_model(model1_path)
pca = joblib.load(pca_path)

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

@app.route('/model2', methods=['GET', 'POST'])
def model2_page():
    if request.method == 'POST':
        input_data = [float(request.form['Age']),
                      float(request.form['Gender']),
                      float(request.form['Ethnicity']),
                      float(request.form['EducationLevel']),
                      float(request.form['BMI']),
                      float(request.form['Smoking']),
                      float(request.form['AlcoholConsumption']),
                      float(request.form['PhysicalActivity']),
                      float(request.form['DietQuality']),
                      float(request.form['SleepQuality']),
                      float(request.form['FamilyHistoryAlzheimers']),
                      float(request.form['CardiovascularDisease']),
                      float(request.form['Diabetes']),
                      float(request.form['Depression']),
                      float(request.form['HeadInjury']),
                      float(request.form['Hypertension']),
                      float(request.form['SystolicBP']),
                      float(request.form['DiastolicBP']),
                      float(request.form['CholesterolTotal']),
                      float(request.form['CholesterolLDL']),
                      float(request.form['CholesterolHDL']),
                      float(request.form['CholesterolTriglycerides']),
                      float(request.form['MMSE']),
                      float(request.form['FunctionalAssessment']),
                      float(request.form['MemoryComplaints']),
                      float(request.form['BehavioralProblems']),
                      float(request.form['ADL']),
                      float(request.form['Confusion']),
                      float(request.form['Disorientation']),
                      float(request.form['PersonalityChanges']),
                      float(request.form['DifficultyCompletingTasks']),
                      float(request.form['Forgetfulness'])]
        input_data = np.array([input_data])
        input_data_pca = pca_al.transform(input_data)
        input_data_scaled = scaler_al.transform(input_data_pca)

        prediction = model2.predict(input_data_scaled)
        # Transform input data using the pre-trained PCA
        #input_data_pca = pca.transform(input_data)
        
        # Predict
        #prediction = model2.predict(input_data_pca)
        result = 'Diagnosis' if prediction[0] > 0.5 else 'No Diagnosis'

        return render_template('model2.html', result=result)
    return render_template('model2.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
