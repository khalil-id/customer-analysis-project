from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.data_extraction import extract_and_save_data
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model, evaluate_model, get_feature_importance, save_model, load_model, preprocess_and_predict
from src.config import DATA_PATH, MODEL_PATH
from main import main_train
from src.database_operations import insert_customer_data

app = Flask(__name__)

# Load model and preprocessor at startup
model, preprocessor = load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    main_train()
    return render_template('index.html', message="Model training completed successfully!")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        new_data = {
            'customer_id': request.form['customer_id'],
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'estimated_salary': float(request.form['estimated_salary']),
            'credit_score': float(request.form['credit_score']),
            'products_number': float(request.form['products_number']),
            'tenure': float(request.form['tenure']),
            'credit_card': int(request.form['credit_card']),
            'active_member': int(request.form['active_member']),
            'balance': float(request.form['balance']),
            'country': request.form['country']
        }
        
        # Convert to DataFrame
        new_data_df = pd.DataFrame([new_data])
        
        # Make prediction
        prediction, probability = preprocess_and_predict(model, preprocessor, new_data_df)
        
        # Prepare result
        result = {
            'customer_id': new_data['customer_id'],
            'prediction': 'Churn' if prediction[0] else 'Stay',
            'probability': f"{probability[0]:.2f}"
        }
        
        # Add prediction to new_data for database insertion
        new_data['prediction'] = 1 if prediction[0] else 0
        
        # Store the new data in the database
        #insert_customer_data(new_data)
        
        return render_template('result.html', result=result)
    
    # If GET request, show the prediction form
    return render_template('predict_form.html')

if __name__ == '__main__':
    app.run(debug=True)