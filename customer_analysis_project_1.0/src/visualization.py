import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def visualize_prediction(X, y, new_data, prediction, feature_names):
    # Ensure X and new_data are DataFrames
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data, columns=feature_names)

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)

    # Align new_data_encoded with X_encoded
    new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_encoded)
    new_data_pca = pca.transform(new_data_encoded)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='Churn Status')

    # Plot new data point
    plt.scatter(new_data_pca[0, 0], new_data_pca[0, 1], color='red', s=200, marker='*', edgecolors='black', linewidth=2)

    plt.title('Customer Churn Prediction Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    # Add legend
    plt.legend(['Existing Customers', 'New Customer'])

    # Annotate new data point
    plt.annotate(f'Prediction: {"Churn" if prediction>=0.5 else "Stay"}', 
                 (new_data_pca[0, 0], new_data_pca[0, 1]), 
                 xytext=(5, 5), textcoords='offset points')

    plt.show()

    # Print feature importance for PCA components
    print("\nFeature Importance in PCA:")
    for i, component in enumerate(pca.components_):
        sorted_importance = sorted(zip(X_encoded.columns, component), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop 5 features for PC{i+1}:")
        for feature, importance in sorted_importance[:5]:
            print(f"{feature}: {importance:.4f}")

    # Calculate and print the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"\nExplained Variance Ratio: PC1 = {explained_variance_ratio[0]:.4f}, PC2 = {explained_variance_ratio[1]:.4f}")
    print(f"Total Explained Variance: {sum(explained_variance_ratio):.4f}")





"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def visualize_prediction(X, y, new_data, prediction, feature_names):
    if isinstance(X, pd.DataFrame):
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)
        new_data_encoded = pd.get_dummies(new_data, drop_first=True)

        # Align new_data_encoded with X_encoded
        new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_encoded)
    new_data_pca = pca.transform(new_data_encoded)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter)

    # Plot new data point
    plt.scatter(new_data_pca[0, 0], new_data_pca[0, 1], color='red', s=200, marker='*', edgecolors='black', linewidth=2)

    plt.title('Customer Churn Prediction Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    # Add legend
    plt.legend(['Existing Customers', 'New Customer'])

    # Annotate new data point
    plt.annotate(f'Prediction: {"Churn" if prediction>=0.5 else "Stay"}', 
                 (new_data_pca[0, 0], new_data_pca[0, 1]), 
                 xytext=(5, 5), textcoords='offset points')

    plt.show()

    # Print feature importance for PCA components
    print("\nFeature Importance in PCA:")
    for i, component in enumerate(pca.components_):
        sorted_importance = sorted(zip(feature_names, component), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop 5 features for PC{i+1}:")
        for feature, importance in sorted_importance[:5]:
            print(f"{feature}: {importance:.4f}")
"""