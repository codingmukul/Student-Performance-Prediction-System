# Student Performance Prediction System

## Project Overview

This project implements a machine learning system to predict student performance in mathematics based on various demographic and academic factors. The system uses multiple regression algorithms to find the best-performing model for predicting student math scores.

## Features

- Data ingestion pipeline with automatic train-test splitting
- Advanced data preprocessing for both numerical and categorical features
- Multiple regression models comparison (Random Forest, XGBoost, CatBoost, etc.)
- Automated model selection based on R2 score
- Web interface for real-time predictions
- Comprehensive logging and error handling

## Tech Stack

- Python 3.10+
- Scikit-learn
- Pandas
- NumPy
- XGBoost
- CatBoost
- Flask
- HTML/CSS

## Project Structure

```
Student-Performance-Prediction-System/
├── artifacts/
├── notebook/
│   └── data/
│       └── student_data.csv
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
├── templates/
│   ├── home.html
│   └── index.html
├── app.py
├── setup.py
└── requirements.txt
```

## Features Used for Prediction

1. Gender
2. Race/Ethnicity
3. Parental Level of Education
4. Lunch Type
5. Test Preparation Course
6. Reading Score
7. Writing Score

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Student-Performance-Prediction-System
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the training pipeline:
```bash
python src/pipeline/train_pipeline.py
```

5. Start the Flask application:
```bash
python app.py
```

## Usage

1. Training:
   - The system automatically handles data preprocessing
   - Trains multiple models and selects the best performer
   - Saves the model and preprocessor in the artifacts directory

2. Prediction:
   - Access the web interface at `http://localhost:5000`
   - Input student details
   - Get predicted math score

## Model Details

The system trains and compares the following models:
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regression
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

## Data Preprocessing

- Numerical Features:
  - Median imputation for missing values
  - Standard scaling

- Categorical Features:
  - Most frequent imputation for missing values
  - One-hot encoding
  - Standard scaling

## Performance Metrics

The system uses R2 score as the primary metric for model evaluation. Models with scores below 0.6 are rejected to ensure prediction quality.

## Error Handling

The system implements custom exception handling throughout the pipeline for better error tracking and debugging.

## Logging

Comprehensive logging is implemented for all major operations:
- Data ingestion
- Transformation
- Model training
- Predictions

## Acknowledgments

- Dataset provided is focused on student performance metrics
- Thanks to all contributors and maintainers