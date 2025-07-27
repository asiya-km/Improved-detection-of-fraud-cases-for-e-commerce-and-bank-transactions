"""
Fraud Detection Pipeline for E-commerce and Bank Transactions
============================================================

This module provides a complete implementation of the fraud detection pipeline including:
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Model deployment utilities

Author: Asiya KM
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc,
                           roc_curve, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
import pickle
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    A comprehensive pipeline for fraud detection in e-commerce and bank transactions.
    
    This class handles the complete workflow from data loading to model deployment,
    including data cleaning, feature engineering, model training, and evaluation.
    """
    
    def __init__(self, data_path='../Data/'):
        """
        Initialize the fraud detection pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the data directory containing the datasets
        """
        self.data_path = data_path
        self.fraud_data = None
        self.ip_to_country = None
        self.credit_card = None
        self.merged_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = None
        self.feature_names = None
        
    def load_data(self):
        """
        Load all datasets required for fraud detection.
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Loading datasets...")
        
        # Load the main datasets
        self.fraud_data = pd.read_csv(f'{self.data_path}Fraud_Data.csv')
        self.ip_to_country = pd.read_csv(f'{self.data_path}IpAddress_to_Country.csv')
        self.credit_card = pd.read_csv(f'{self.data_path}creditcard.csv')
        
        print(f"Fraud data shape: {self.fraud_data.shape}")
        print(f"IP to Country data shape: {self.ip_to_country.shape}")
        print(f"Credit card data shape: {self.credit_card.shape}")
        
        return self
    
    def clean_data(self):
        """
        Perform comprehensive data cleaning including:
        - Handling missing values
        - Removing duplicates
        - Converting data types
        - Basic data validation
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Cleaning datasets...")
        
        # Check for missing values
        print("\nMissing values check:")
        print("Fraud_Data.csv missing values:")
        print(self.fraud_data.isnull().sum())
        print("\nIpAddress_to_Country.csv missing values:")
        print(self.ip_to_country.isnull().sum())
        print("\ncreditcard.csv missing values:")
        print(self.credit_card.isnull().sum())
        
        # Check for duplicates
        print(f"\nDuplicates check:")
        print(f"Fraud_Data.csv duplicates: {self.fraud_data.duplicated().sum()}")
        print(f"IpAddress_to_Country.csv duplicates: {self.ip_to_country.duplicated().sum()}")
        print(f"creditcard.csv duplicates: {self.credit_card.duplicated().sum()}")
        
        # Remove duplicates
        self.fraud_data = self.fraud_data.drop_duplicates()
        self.ip_to_country = self.ip_to_country.drop_duplicates()
        self.credit_card = self.credit_card.drop_duplicates()
        
        # Convert data types
        self.fraud_data['signup_time'] = pd.to_datetime(self.fraud_data['signup_time'])
        self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'])
        self.ip_to_country['lower_bound_ip_address'] = pd.to_numeric(
            self.ip_to_country['lower_bound_ip_address'], errors='coerce')
        self.ip_to_country['upper_bound_ip_address'] = pd.to_numeric(
            self.ip_to_country['upper_bound_ip_address'], errors='coerce')
        
        print("Data cleaning completed successfully!")
        return self
    
    def merge_geolocation_data(self):
        """
        Merge fraud data with IP-to-country mapping using efficient merge_asof.
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Merging geolocation data...")
        
        # Ensure columns are numeric and sorted
        self.fraud_data['ip_address'] = pd.to_numeric(self.fraud_data['ip_address'], errors='coerce')
        self.ip_to_country['lower_bound_ip_address'] = pd.to_numeric(
            self.ip_to_country['lower_bound_ip_address'], errors='coerce')
        self.ip_to_country['upper_bound_ip_address'] = pd.to_numeric(
            self.ip_to_country['upper_bound_ip_address'], errors='coerce')
        
        # Sort both DataFrames for merge_asof
        fraud_data_sorted = self.fraud_data.sort_values('ip_address').reset_index(drop=True)
        ip_to_country_sorted = self.ip_to_country.sort_values('lower_bound_ip_address').reset_index(drop=True)
        
        # Use merge_asof to find the lower bound
        merged = pd.merge_asof(
            fraud_data_sorted,
            ip_to_country_sorted,
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='backward'
        )
        
        # Filter to only those where ip_address <= upper_bound_ip_address
        merged['country'] = np.where(
            merged['ip_address'] <= merged['upper_bound_ip_address'],
            merged['country'],
            'Unknown'
        )
        
        # Keep the original order
        self.merged_data = merged.sort_index()
        
        print(f"Merged data shape: {self.merged_data.shape}")
        return self
    
    def engineer_features(self):
        """
        Create comprehensive feature engineering including:
        - Transaction frequency features
        - Time-based features
        - Behavioral patterns
        - Risk indicators
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Engineering features...")
        
        # Transaction frequency features
        self.merged_data['device_transaction_count'] = self.merged_data.groupby('device_id')['device_id'].transform('count')
        self.merged_data['ip_transaction_count'] = self.merged_data.groupby('ip_address')['ip_address'].transform('count')
        self.merged_data['country_transaction_count'] = self.merged_data.groupby('country')['country'].transform('count')
        self.merged_data['user_transaction_count'] = self.merged_data.groupby('user_id')['user_id'].transform('count')
        
        # Time-based features
        # Sort by user and purchase time
        self.merged_data = self.merged_data.sort_values(['user_id', 'purchase_time'])
        self.merged_data['time_since_prev_txn_user'] = self.merged_data.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
        
        # Sort by device and purchase time
        self.merged_data = self.merged_data.sort_values(['device_id', 'purchase_time'])
        self.merged_data['time_since_prev_txn_device'] = self.merged_data.groupby('device_id')['purchase_time'].diff().dt.total_seconds() / 3600
        
        # Sort by IP address and purchase time
        self.merged_data = self.merged_data.sort_values(['ip_address', 'purchase_time'])
        self.merged_data['time_since_prev_txn_ip'] = self.merged_data.groupby('ip_address')['purchase_time'].diff().dt.total_seconds() / 3600
        
        # Time between signup and purchase
        self.merged_data['time_since_signup'] = (self.merged_data['purchase_time'] - self.merged_data['signup_time']).dt.total_seconds() / 3600
        
        # Hour of day and day of week
        self.merged_data['hour_of_day'] = self.merged_data['purchase_time'].dt.hour
        self.merged_data['day_of_week'] = self.merged_data['purchase_time'].dt.dayofweek
        
        # Purchase value features
        self.merged_data['purchase_value_log'] = np.log1p(self.merged_data['purchase_value'])
        
        # Risk indicators
        self.merged_data['high_value_transaction'] = (self.merged_data['purchase_value'] > 
                                                     self.merged_data['purchase_value'].quantile(0.95)).astype(int)
        
        # Device and IP risk scores
        self.merged_data['device_risk_score'] = self.merged_data['device_transaction_count'] * self.merged_data['purchase_value']
        self.merged_data['ip_risk_score'] = self.merged_data['ip_transaction_count'] * self.merged_data['purchase_value']
        
        print(f"Feature engineering completed. Total features: {len(self.merged_data.columns)}")
        return self
    
    def prepare_features(self):
        """
        Prepare features for modeling by:
        - Selecting relevant features
        - Handling categorical variables
        - Scaling numerical features
        - Splitting into train/test sets
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Preparing features for modeling...")
        
        # Select features for modeling
        feature_columns = [
            'purchase_value', 'purchase_value_log', 'age',
            'device_transaction_count', 'ip_transaction_count', 
            'country_transaction_count', 'user_transaction_count',
            'time_since_prev_txn_user', 'time_since_prev_txn_device',
            'time_since_prev_txn_ip', 'time_since_signup',
            'hour_of_day', 'day_of_week', 'high_value_transaction',
            'device_risk_score', 'ip_risk_score'
        ]
        
        # Handle categorical variables
        categorical_features = ['source', 'browser', 'sex', 'country']
        for feature in categorical_features:
            if feature in self.merged_data.columns:
                dummies = pd.get_dummies(self.merged_data[feature], prefix=feature, drop_first=True)
                self.merged_data = pd.concat([self.merged_data, dummies], axis=1)
                feature_columns.extend(dummies.columns.tolist())
        
        # Create feature matrix
        X = self.merged_data[feature_columns].copy()
        y = self.merged_data['class'].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = feature_columns
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Class distribution in training set:")
        print(self.y_train.value_counts(normalize=True))
        
        return self
    
    def handle_class_imbalance(self, method='smote'):
        """
        Handle class imbalance using various techniques.
        
        Parameters:
        -----------
        method : str
            Method to use for handling class imbalance ('smote', 'undersample', 'smoteenn')
            
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print(f"Handling class imbalance using {method}...")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            self.X_train_resampled, self.y_train_resampled = undersampler.fit_resample(
                self.X_train_scaled, self.y_train
            )
        elif method == 'smoteenn':
            smoteenn = SMOTEENN(random_state=42)
            self.X_train_resampled, self.y_train_resampled = smoteenn.fit_resample(
                self.X_train_scaled, self.y_train
            )
        
        print(f"Resampled training set shape: {self.X_train_resampled.shape}")
        print(f"Class distribution after resampling:")
        print(pd.Series(self.y_train_resampled).value_counts(normalize=True))
        
        return self
    
    def train_models(self):
        """
        Train multiple machine learning models and evaluate their performance.
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Training models...")
        
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train and evaluate each model
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            
            # Train the model
            model.fit(self.X_train_resampled, self.y_train_resampled)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': (y_pred == self.y_test).mean(),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'f1_score': f1_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {self.results[name]['accuracy']:.4f}")
            print(f"{name} - ROC AUC: {self.results[name]['roc_auc']:.4f}")
            print(f"{name} - F1 Score: {self.results[name]['f1_score']:.4f}")
            print(f"{name} - Precision: {self.results[name]['precision']:.4f}")
            print(f"{name} - Recall: {self.results[name]['recall']:.4f}")
        
        return self
    
    def evaluate_models(self):
        """
        Perform comprehensive model evaluation including:
        - Classification reports
        - Confusion matrices
        - ROC and Precision-Recall curves
        - Cross-validation scores
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Evaluating models...")
        
        # Detailed evaluation for each model
        for name, result in self.results.items():
            print(f"\n{'='*50}")
            print(f"Detailed Results for {name}")
            print(f"{'='*50}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, result['predictions']))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, result['predictions'])
            print(f"\nConfusion Matrix:")
            print(cm)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            pr_auc = auc(recall, precision)
            print(f"\nPrecision-Recall AUC: {pr_auc:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(result['model'], self.X_train_resampled, 
                                      self.y_train_resampled, cv=5, scoring='f1')
            print(f"\nCross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self
    
    def plot_results(self):
        """
        Create comprehensive visualization of model results.
        
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC curves
        ax1 = axes[0, 0]
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True)
        
        # Precision-Recall curves
        ax2 = axes[0, 1]
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True)
        
        # Confusion matrices
        for i, (name, result) in enumerate(self.results.items()):
            ax = axes[1, i]
            cm = confusion_matrix(self.y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name}\nConfusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def save_models(self, output_dir='models/'):
        """
        Save trained models and preprocessing objects.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the models
            
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Saving models...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        for name, result in self.results.items():
            model_filename = f"{output_dir}{name.lower().replace(' ', '_')}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(result['model'], f)
            print(f"Saved {name} to {model_filename}")
        
        # Save scaler
        scaler_filename = f"{output_dir}scaler.pkl"
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_filename}")
        
        # Save feature names
        feature_filename = f"{output_dir}feature_names.json"
        with open(feature_filename, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"Saved feature names to {feature_filename}")
        
        return self
    
    def run_complete_pipeline(self, output_dir='models/'):
        """
        Run the complete fraud detection pipeline from start to finish.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the models
            
        Returns:
        --------
        self : FraudDetectionPipeline
            Returns self for method chaining
        """
        print("Starting complete fraud detection pipeline...")
        print("="*60)
        
        (self.load_data()
         .clean_data()
         .merge_geolocation_data()
         .engineer_features()
         .prepare_features()
         .handle_class_imbalance()
         .train_models()
         .evaluate_models()
         .plot_results()
         .save_models(output_dir))
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        
        return self


def main():
    """
    Main function to run the fraud detection pipeline.
    """
    # Initialize the pipeline
    pipeline = FraudDetectionPipeline()
    
    # Run the complete pipeline
    pipeline.run_complete_pipeline()
    
    # Print final results summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for name, result in pipeline.results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        print(f"  F1 Score: {result['f1_score']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")


if __name__ == "__main__":
    main() 