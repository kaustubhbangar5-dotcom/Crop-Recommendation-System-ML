"""
Crop Recommendation System - Machine Learning Model Training (9 Features, 24 Crop Classes)
Trains a Random Forest Classifier on agricultural parameters to recommend suitable crops.
Features: N, P, K, temperature, humidity, ph, rainfall, soil_moisture, sunlight_hours
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate_model():
    dataset_path = 'Crop_recommendation.csv'
    model_save_path = 'crop_model.pkl'
    
    print("=" * 65)
    print("--- CROP RECOMMENDATION SYSTEM - 9 FEATURES & 24 CROPS ---")
    print("=" * 65)
    
    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nDataset Sample:")
    print(df.head())
    
    # Step 2: Data Preprocessing & Validation
    print("\n[Step 2] Data Preprocessing & Inspection...")
    print("Missing values per column:")
    print(df.isnull().sum())
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Duplicates removed. Remaining rows: {df.shape[0]}")
        
    print("\nDataset Summary Statistics:")
    print(df.describe())
    
    features = [
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 
        'rainfall', 'soil_moisture', 'sunlight_hours'
    ]
    target = 'label'
    
    X = df[features]
    y = df[target]
    
    print(f"\nTotal Target Crop Classes ({len(y.unique())}):")
    print(sorted(y.unique().tolist()))
    
    # Step 3: Exploratory Data Analysis & Visualizations
    print("\n[Step 3] Generating Exploratory Data Analysis (EDA) charts...")
    os.makedirs('static/img/eda', exist_ok=True)
    
    try:
        # 1. Feature Distributions (3x3 Grid for 9 Features)
        plt.figure(figsize=(16, 12))
        for i, col in enumerate(features, 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[col], kde=True, color='#2e7d32', bins=25)
            plt.title(f'Distribution of {col}', fontsize=11, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('static/img/eda/feature_distributions.png', dpi=150)
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        numeric_df = df[features]
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f', square=True, linewidths=0.5)
        plt.title('9-Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig('static/img/eda/correlation_matrix.png', dpi=150)
        plt.close()
        
        print("EDA charts saved to 'static/img/eda/'.")
    except Exception as e:
        print(f"Warning: Could not save EDA plots: {e}")

    # Step 4: Model Training (Train/Test Split & Random Forest)
    print("\n[Step 4] Splitting dataset and training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        random_state=42,
        n_jobs=1
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest Classifier trained successfully!", flush=True)
    
    # Step 5: Model Evaluation
    print("\n[Step 5] Evaluating model performance...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 6: Feature Importance Analysis (Top Feature Importances)
    print("\n[Step 6] Top Feature Importances (All 9 Features):")
    importances = rf_model.feature_importances_
    sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(sorted_features, 1):
        print(f" #{rank} {feat:16s}: {imp * 100:6.2f}%")
        
    # Step 7: Save Model
    print("\n[Step 7] Saving trained model...")
    model_payload = {
        'model': rf_model,
        'features': features,
        'crops': sorted(y.unique().tolist()),
        'accuracy': accuracy,
        'feature_importances': {feat: round(float(imp * 100), 2) for feat, imp in sorted_features}
    }
    joblib.dump(model_payload, model_save_path)
    print(f"Model successfully saved to '{model_save_path}'")
    
    # Step 8: Test Single Sample Predictions
    print("\n[Step 8] Testing sample predictions...")
    
    # Test sample 1: Rice
    sample_rice = pd.DataFrame([[90, 42, 43, 20.87, 82.0, 6.5, 202.9, 82.0, 6.5]], columns=features)
    pred_rice = rf_model.predict(sample_rice)[0]
    conf_rice = rf_model.predict_proba(sample_rice).max() * 100
    print(f"Sample Rice Prediction: {pred_rice} (Confidence: {conf_rice:.2f}%)")
    
    # Test sample 2: Wheat
    sample_wheat = pd.DataFrame([[105, 55, 40, 18.5, 58.0, 6.7, 70.0, 52.0, 8.0]], columns=features)
    pred_wheat = rf_model.predict(sample_wheat)[0]
    conf_wheat = rf_model.predict_proba(sample_wheat).max() * 100
    print(f"Sample Wheat Prediction: {pred_wheat} (Confidence: {conf_wheat:.2f}%)")
    
    # Test sample 3: Sugarcane
    sample_sugar = pd.DataFrame([[148, 68, 92, 30.5, 80.0, 6.9, 195.0, 78.0, 8.8]], columns=features)
    pred_sugar = rf_model.predict(sample_sugar)[0]
    conf_sugar = rf_model.predict_proba(sample_sugar).max() * 100
    print(f"Sample Sugarcane Prediction: {pred_sugar} (Confidence: {conf_sugar:.2f}%)")
    
    print("\n" + "=" * 65)
    print(" Model training pipeline completed successfully!")
    print("=" * 65)

if __name__ == '__main__':
    train_and_evaluate_model()
