import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(df):
    #df = pd.read_csv(file_path)
    
    df = df[df['balance'] != 0.00]
    df = df.reset_index(drop=True)
    
    # Separate features and target
    X = df.drop(['attrition', 'customer_id'], axis=1)
    y = df['attrition']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define column types
    numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
    categorical_features = ['gender', 'country', 'credit_card', 'active_member']
    
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit the preprocessor
    preprocessor.fit(X_train)
    
    # Transform the data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = (numerical_features + 
                     preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
    
    # Convert to DataFrame
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, df