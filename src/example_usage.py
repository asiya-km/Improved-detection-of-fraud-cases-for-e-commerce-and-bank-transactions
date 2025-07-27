"""
Example Usage Script for Fraud Detection System
==============================================

This script demonstrates how to use the fraud detection system
with sample data and various use cases.

Author: Asiya KM
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# Import our modules
from src.fraud_detection_pipeline import FraudDetectionPipeline
from src.data_preprocessing import preprocess_fraud_data
from src.model_training import FraudDetectionModelTrainer
from src.deployment import FraudDetectionAPI

def example_1_complete_pipeline():
    """
    Example 1: Run the complete fraud detection pipeline.
    """
    print("="*60)
    print("EXAMPLE 1: Complete Fraud Detection Pipeline")
    print("="*60)
    
    # Initialize the pipeline
    pipeline = FraudDetectionPipeline()
    
    # Run the complete pipeline
    pipeline.run_complete_pipeline()
    
    print("Complete pipeline executed successfully!")
    print("Check the 'models/' directory for saved models and results.")

def example_2_data_preprocessing():
    """
    Example 2: Data preprocessing only.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Data Preprocessing")
    print("="*60)
    
    # Preprocess the data
    processed_data, feature_summary = preprocess_fraud_data(
        data_path='../Data/',
        output_path='../Data/processed/'
    )
    
    print(f"Data preprocessing completed!")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Total features created: {len(processed_data.columns)}")
    print(f"Class distribution:")
    print(processed_data['class'].value_counts(normalize=True))
    
    # Display feature summary
    print("\nFeature Summary (first 10 features):")
    print(feature_summary.head(10))

def example_3_model_training():
    """
    Example 3: Model training with custom parameters.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Training")
    print("="*60)
    
    # Load preprocessed data
    try:
        processed_data = pd.read_csv('../Data/processed/processed_fraud_data.csv')
    except FileNotFoundError:
        print("Processed data not found. Running preprocessing first...")
        processed_data, _ = preprocess_fraud_data()
    
    # Initialize trainer with custom parameters
    trainer = FraudDetectionModelTrainer(random_state=42)
    
    # Prepare data
    trainer.prepare_data(processed_data, target_column='class', test_size=0.2)
    
    # Handle class imbalance with different methods
    print("Testing different class imbalance handling methods...")
    
    methods = ['smote', 'undersample', 'smoteenn']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} ---")
        trainer.handle_class_imbalance(method=method)
        trainer.initialize_models()
        trainer.train_models()
        
        # Store results
        results[method] = {}
        for name, result in trainer.results.items():
            results[method][name] = {
                'f1_score': result['f1_score'],
                'roc_auc': result['roc_auc'],
                'precision': result['precision'],
                'recall': result['recall']
            }
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON OF CLASS IMBALANCE METHODS")
    print("="*60)
    
    for method, method_results in results.items():
        print(f"\n{method.upper()}:")
        for model_name, metrics in method_results.items():
            print(f"  {model_name}:")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")
            print(f"    ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")

def example_4_api_usage():
    """
    Example 4: Using the API for predictions.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: API Usage")
    print("="*60)
    
    # Sample transaction data
    sample_transaction = {
        "user_id": "user_12345",
        "purchase_value": 150.50,
        "age": 28,
        "device_id": "device_67890",
        "ip_address": "192168001001",
        "source": "SEO",
        "browser": "Chrome",
        "sex": "M",
        "country": "US",
        "purchase_time": "2024-01-15T14:30:00",
        "signup_time": "2024-01-10T09:15:00",
        "device_transaction_count": 3,
        "ip_transaction_count": 2,
        "user_transaction_count": 5,
        "country_transaction_count": 1000,
        "time_since_prev_txn_user": 2.5,
        "time_since_prev_txn_device": 1.0,
        "time_since_prev_txn_ip": 0.5,
        "time_since_signup": 120.0,
        "device_risk_score": 450.0,
        "ip_risk_score": 300.0,
        "user_risk_score": 750.0,
        "country_risk_score": 150000.0,
        "composite_risk_score": 37500.0,
        "high_value_transaction": 0,
        "very_high_value_transaction": 0,
        "purchase_value_percentile": 0.6,
        "user_device_count": 1,
        "user_ip_count": 1,
        "user_country_count": 1,
        "multiple_devices": 0,
        "multiple_ips": 0,
        "multiple_countries": 0,
        "rapid_successive_txn": 0
    }
    
    # Multiple transactions for batch prediction
    batch_transactions = [
        sample_transaction,
        {
            **sample_transaction,
            "purchase_value": 2500.00,
            "high_value_transaction": 1,
            "composite_risk_score": 50000.0
        },
        {
            **sample_transaction,
            "purchase_value": 50.00,
            "time_since_prev_txn_user": 0.1,
            "rapid_successive_txn": 1
        }
    ]
    
    print("Sample transaction data:")
    print(json.dumps(sample_transaction, indent=2))
    
    # Note: This would require the API to be running
    print("\nTo test the API, start it first:")
    print("python src/deployment.py --debug")
    print("\nThen you can make requests like:")
    
    # Example API calls (commented out since API might not be running)
    """
    # Single prediction
    response = requests.post('http://localhost:5000/predict', 
                           json=sample_transaction)
    result = response.json()
    print(f"Prediction: {result}")
    
    # Batch prediction
    response = requests.post('http://localhost:5000/predict_batch',
                           json={'transactions': batch_transactions})
    results = response.json()
    print(f"Batch predictions: {results}")
    """

def example_5_feature_engineering_demo():
    """
    Example 5: Demonstrate feature engineering process.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Feature Engineering Demo")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample fraud data
    sample_data = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_samples)],
        'purchase_value': np.random.exponential(100, n_samples),
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'device_id': [f'device_{i%50}' for i in range(n_samples)],
        'ip_address': [f'192.168.1.{i%100}' for i in range(n_samples)],
        'source': np.random.choice(['SEO', 'Ads', 'Direct'], n_samples),
        'browser': np.random.choice(['Chrome', 'Safari', 'Firefox'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU'], n_samples),
        'purchase_time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'signup_time': pd.date_range('2023-12-01', periods=n_samples, freq='D'),
        'class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud
    })
    
    print("Original data shape:", sample_data.shape)
    print("Class distribution:")
    print(sample_data['class'].value_counts(normalize=True))
    
    # Demonstrate feature engineering steps
    from src.data_preprocessing import (
        create_transaction_frequency_features,
        create_time_based_features,
        create_value_based_features,
        create_risk_score_features,
        create_behavioral_features
    )
    
    # Step 1: Transaction frequency features
    print("\n1. Creating transaction frequency features...")
    sample_data = create_transaction_frequency_features(sample_data)
    print(f"   Added features: device_transaction_count, ip_transaction_count, etc.")
    
    # Step 2: Time-based features
    print("\n2. Creating time-based features...")
    sample_data = create_time_based_features(sample_data)
    print(f"   Added features: hour_of_day, day_of_week, time_since_prev_txn, etc.")
    
    # Step 3: Value-based features
    print("\n3. Creating value-based features...")
    sample_data = create_value_based_features(sample_data)
    print(f"   Added features: purchase_value_log, high_value_transaction, etc.")
    
    # Step 4: Risk score features
    print("\n4. Creating risk score features...")
    sample_data = create_risk_score_features(sample_data)
    print(f"   Added features: device_risk_score, composite_risk_score, etc.")
    
    # Step 5: Behavioral features
    print("\n5. Creating behavioral features...")
    sample_data = create_behavioral_features(sample_data)
    print(f"   Added features: multiple_devices, rapid_successive_txn, etc.")
    
    print(f"\nFinal data shape: {sample_data.shape}")
    print(f"Total features created: {len(sample_data.columns)}")
    
    # Show some engineered features
    print("\nSample engineered features:")
    feature_columns = [col for col in sample_data.columns if col not in [
        'user_id', 'device_id', 'ip_address', 'source', 'browser', 'sex', 
        'country', 'purchase_time', 'signup_time', 'class'
    ]]
    
    print(sample_data[feature_columns[:10]].head())

def example_6_model_comparison():
    """
    Example 6: Compare different models and their performance.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Model Comparison")
    print("="*60)
    
    # Load preprocessed data
    try:
        processed_data = pd.read_csv('../Data/processed/processed_fraud_data.csv')
    except FileNotFoundError:
        print("Processed data not found. Running preprocessing first...")
        processed_data, _ = preprocess_fraud_data()
    
    # Initialize trainer
    trainer = FraudDetectionModelTrainer(random_state=42)
    
    # Prepare data
    trainer.prepare_data(processed_data, target_column='class')
    trainer.handle_class_imbalance(method='smote')
    trainer.initialize_models()
    trainer.train_models()
    trainer.perform_cross_validation()
    
    # Create comparison table
    comparison_data = []
    
    for name, result in trainer.results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'ROC AUC': f"{result['roc_auc']:.4f}",
            'F1 Score': f"{result['f1_score']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'CV F1 Mean': f"{result['cv_f1']['mean']:.4f}",
            'CV F1 Std': f"{result['cv_f1']['std']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model for each metric
    print("\nBest models by metric:")
    metrics = ['accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']
    
    for metric in metrics:
        best_model = max(trainer.results.items(), key=lambda x: x[1][metric])
        print(f"  Best {metric}: {best_model[0]} ({best_model[1][metric]:.4f})")

def main():
    """
    Run all examples.
    """
    print("FRAUD DETECTION SYSTEM - EXAMPLE USAGE")
    print("="*60)
    
    # Run examples
    try:
        example_1_complete_pipeline()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_data_preprocessing()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_model_training()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_api_usage()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_5_feature_engineering_demo()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    try:
        example_6_model_comparison()
    except Exception as e:
        print(f"Example 6 failed: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated models in the 'models/' directory")
    print("2. Start the API server: python src/deployment.py --debug")
    print("3. Make predictions using the API endpoints")
    print("4. Deploy to production using Docker: docker-compose up")


if __name__ == "__main__":
    main() 