import pandas as pd
from src.model_training import load_model, preprocess_and_predict
from src.visualization import visualize_prediction
from src.config import MODEL_PATH, DATA_PATH

def get_user_input():
    print("Enter customer information:")
    customer_id = input("Customer ID: ")
    gender = input("Gender (Male/Female): ")
    age = float(input("Age: "))
    estimated_salary = float(input("Estimated Salary: "))
    credit_score = float(input("Credit Score: "))
    products_number = float(input("Number of Products: "))
    tenure = float(input("Tenure (years): "))
    credit_card = float(input("Has Credit Card (0 for No, 1 for Yes): "))
    active_member = float(input("Is Active Member (0 for No, 1 for Yes): "))
    balance = float(input("Account Balance: "))
    
    # Load the original dataset to get valid countries
    df = pd.read_csv(DATA_PATH)
    VALID_COUNTRIES = df['country'].unique().tolist()
    while True:
        country = input(f"Country ({', '.join(VALID_COUNTRIES)}): ")
        if country in VALID_COUNTRIES:
            break
        print(f"Invalid country. Please enter one of: {', '.join(VALID_COUNTRIES)}")

    new_data = pd.DataFrame({
        'customer_id': [customer_id],
        'gender': [gender],
        'age': [age],
        'estimated_salary': [estimated_salary],
        'credit_score': [credit_score],
        'products_number': [products_number],
        'tenure': [tenure],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'balance': [balance],
        'country': [country]
    })

    return new_data

def main_predict():
    # Load model and preprocessor
    model, preprocessor = load_model(MODEL_PATH)

    # Get new customer data
    new_data = get_user_input()
    
    # Load existing data
    df = pd.read_csv(DATA_PATH)
    # Prepare data for visualization
    df = df[df['balance'] != 0.00]
    df = df.reset_index(drop=True)
    
    X = df.drop(['customer_id', 'attrition'], axis=1)
    y = df['attrition']
    feature_names = X.columns.tolist()
    
    # Make prediction
    prediction, probability = preprocess_and_predict(model, preprocessor, new_data)
    
    print(f"\nCustomer ID: {new_data['customer_id'].values[0]}")
    print(f"Prediction: {'Churn' if prediction[0] else 'Stay'}")
    print(f"Probability of churning: {probability[0]:.2f}")
    
    # Visualize prediction
    visualize_prediction(X, y, new_data.drop('customer_id', axis=1), probability[0], feature_names)
    return prediction, probability

if __name__ == "__main__":
    main_predict()