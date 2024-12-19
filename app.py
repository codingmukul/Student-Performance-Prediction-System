from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get the input data from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),  # Corrected the field
            writing_score=float(request.form.get('writing_score'))   # Corrected the field
        )
        
        # Get the data as a DataFrame for prediction
        pred_df = data.get_data_as_data_frame()
        print(f"Input Data for Prediction:\n{pred_df}")
        
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Making Prediction...")
        
        # Make the prediction
        results = predict_pipeline.predict(pred_df)
        print("Prediction Completed")
        
        # Render the result back to the 'home.html' page
        return render_template('home.html', results=results[0])
    
    # If method is GET, return the home page for the user to input data
    return render_template('home.html')

# Main driver to run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
