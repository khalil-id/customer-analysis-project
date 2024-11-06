import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.config import DATA_PATH

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def get_feature_importance(model, feature_names):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return feature_importance

def save_model(model, preprocessor, model_path):
    joblib.dump((model, preprocessor), model_path)
    print(f"Model and preprocessor saved to {model_path}")

def load_model(model_path):
    model, preprocessor = joblib.load(model_path)
    return model, preprocessor

def preprocess_and_predict(model, preprocessor, new_data):
    # Ensure new_data has all required columns
    required_columns = ['customer_id', 'credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary',
                        'gender', 'credit_card', 'active_member', 'country', 'attrition']
    for col in required_columns:
        if col not in new_data.columns:
            new_data[col] = 0  # or some appropriate default value

    # Ensure all columns are float (except 'country' and 'gender', which need to be processed)
    for col in new_data.columns:
        if col not in ['country', 'gender']:
            new_data[col] = new_data[col].astype(float)

    # Preprocess the new data
    X_new_processed = preprocessor.transform(new_data)

    # Get the feature names expected by the model
    feature_names_expected = model.feature_names_in_
    # Convert processed data back to a DataFrame with proper feature names
    feature_names_transformed = preprocessor.get_feature_names_out()
    feature_names_cleaned = [name.split('__')[-1] for name in feature_names_transformed]
    
    # Convert the processed data back to a DataFrame with the cleaned feature names
    X_new_processed_df = pd.DataFrame(X_new_processed, columns=feature_names_cleaned)
    
    # Reorder columns to match the model's training columns
    X_new_processed_df = X_new_processed_df[feature_names_expected]
    
    # Make predictions
    prediction = model.predict(X_new_processed_df)
    probability = model.predict_proba(X_new_processed_df)[:, 1]

    return prediction, probability