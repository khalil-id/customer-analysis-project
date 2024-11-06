import pandas as pd
from src.data_extraction import extract_and_save_data
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model, evaluate_model, get_feature_importance, save_model
from src.config import DATA_PATH, MODEL_PATH

def main_train():
    # Extract and save data
    DATA = extract_and_save_data()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessing_pipeline, dff = load_and_preprocess_data(DATA)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Get feature importance
    feature_names = X_train.columns
    feature_importance = get_feature_importance(model, feature_names)
    print("\nFeature Importance:")
    print(feature_importance)

    # Save model and preprocessing pipeline
    save_model(model, preprocessing_pipeline, MODEL_PATH)
    print(dff.head(20))

if __name__ == "__main__":
    main_train()
