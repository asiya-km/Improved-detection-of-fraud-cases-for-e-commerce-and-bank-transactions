"""
Model Training Module for Fraud Detection
========================================

This module contains comprehensive model training and evaluation functions including:
- Multiple ML algorithms
- Hyperparameter tuning
- Model evaluation metrics
- Cross-validation
- Model comparison and selection

Author: Asiya KM
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc,
                           roc_curve, f1_score, precision_score, recall_score,
                           accuracy_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
import pickle
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class FraudDetectionModelTrainer:
    """
    A comprehensive model trainer for fraud detection.
    
    This class handles model training, hyperparameter tuning, evaluation,
    and model selection for fraud detection tasks.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data(self, data, target_column='class', test_size=0.2):
        """
        Prepare data for training by splitting into train/test sets and scaling.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Preprocessed dataset
        target_column : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Preparing data for training...")
        
        # Separate features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
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
            Method to use ('smote', 'undersample', 'smoteenn')
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print(f"Handling class imbalance using {method}...")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            self.X_train_resampled, self.y_train_resampled = undersampler.fit_resample(
                self.X_train_scaled, self.y_train
            )
        elif method == 'smoteenn':
            smoteenn = SMOTEENN(random_state=self.random_state)
            self.X_train_resampled, self.y_train_resampled = smoteenn.fit_resample(
                self.X_train_scaled, self.y_train
            )
        
        print(f"Resampled training set shape: {self.X_train_resampled.shape}")
        print(f"Class distribution after resampling:")
        print(pd.Series(self.y_train_resampled).value_counts(normalize=True))
        
        return self
    
    def initialize_models(self):
        """
        Initialize multiple machine learning models for comparison.
        
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        return self
    
    def train_models(self):
        """
        Train all initialized models and evaluate their performance.
        
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Training models...")
        
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
                'accuracy': accuracy_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'f1_score': f1_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred)
            }
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = np.abs(model.coef_[0])
            
            print(f"{name} - Accuracy: {self.results[name]['accuracy']:.4f}")
            print(f"{name} - ROC AUC: {self.results[name]['roc_auc']:.4f}")
            print(f"{name} - F1 Score: {self.results[name]['f1_score']:.4f}")
            print(f"{name} - Precision: {self.results[name]['precision']:.4f}")
            print(f"{name} - Recall: {self.results[name]['recall']:.4f}")
        
        return self
    
    def perform_cross_validation(self, cv_folds=5):
        """
        Perform cross-validation for all models.
        
        Parameters:
        -----------
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"\nCross-validation for {name}...")
            
            # Cross-validation scores
            cv_accuracy = cross_val_score(model, self.X_train_resampled, 
                                        self.y_train_resampled, cv=cv, scoring='accuracy')
            cv_f1 = cross_val_score(model, self.X_train_resampled, 
                                  self.y_train_resampled, cv=cv, scoring='f1')
            cv_roc_auc = cross_val_score(model, self.X_train_resampled, 
                                       self.y_train_resampled, cv=cv, scoring='roc_auc')
            
            # Store CV results
            self.results[name]['cv_accuracy'] = {
                'mean': cv_accuracy.mean(),
                'std': cv_accuracy.std(),
                'scores': cv_accuracy
            }
            self.results[name]['cv_f1'] = {
                'mean': cv_f1.mean(),
                'std': cv_f1.std(),
                'scores': cv_f1
            }
            self.results[name]['cv_roc_auc'] = {
                'mean': cv_roc_auc.mean(),
                'std': cv_roc_auc.std(),
                'scores': cv_roc_auc
            }
            
            print(f"CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
            print(f"CV F1 Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
            print(f"CV ROC AUC: {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
        
        return self
    
    def hyperparameter_tuning(self, model_name, param_grid, cv_folds=5):
        """
        Perform hyperparameter tuning for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        param_grid : dict
            Parameter grid for tuning
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        # Initialize base model
        base_model = self.models[model_name]
        
        # Perform grid search
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='f1', 
            n_jobs=-1, verbose=1, random_state=self.random_state
        )
        
        grid_search.fit(self.X_train_resampled, self.y_train_resampled)
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        return self
    
    def select_best_model(self, metric='f1_score'):
        """
        Select the best model based on a specified metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for model selection
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print(f"Selecting best model based on {metric}...")
        
        # Find the best model
        best_score = -1
        best_model_name = None
        
        for name, result in self.results.items():
            if result[metric] > best_score:
                best_score = result[metric]
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        
        print(f"Best model: {best_model_name}")
        print(f"Best {metric}: {best_score:.4f}")
        
        return self
    
    def evaluate_models(self):
        """
        Perform comprehensive evaluation of all models.
        
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Performing comprehensive model evaluation...")
        
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
            
            # Cross-validation results if available
            if 'cv_f1' in result:
                print(f"\nCross-validation F1 scores: {result['cv_f1']['scores']}")
                print(f"Mean CV F1 score: {result['cv_f1']['mean']:.4f} (+/- {result['cv_f1']['std'] * 2:.4f})")
        
        return self
    
    def plot_results(self, save_path='model_evaluation_results.png'):
        """
        Create comprehensive visualization of model results.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        # Model comparison bar chart
        ax3 = axes[0, 2]
        metrics = ['accuracy', 'roc_auc', 'f1_score', 'precision', 'recall']
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (name, result) in enumerate(self.results.items()):
            values = [result[metric] for metric in metrics]
            ax3.bar(x + i * width, values, width, label=name)
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x + width * (len(self.results) - 1) / 2)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True)
        
        # Confusion matrices
        for i, (name, result) in enumerate(self.results.items()):
            ax = axes[1, i]
            cm = confusion_matrix(self.y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name}\nConfusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def plot_feature_importance(self, feature_names, top_n=20, save_path='feature_importance.png'):
        """
        Plot feature importance for models that support it.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        top_n : int
            Number of top features to display
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Creating feature importance plots...")
        
        models_with_importance = [name for name in self.feature_importance.keys()]
        
        if not models_with_importance:
            print("No models with feature importance available")
            return self
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, model_name in enumerate(models_with_importance[:4]):
            if i >= len(axes):
                break
                
            importance = self.feature_importance[model_name]
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            ax = axes[i]
            feature_importance_df.plot(x='feature', y='importance', kind='barh', ax=ax)
            ax.set_title(f'{model_name} - Top {top_n} Features')
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def save_models(self, output_dir='models/'):
        """
        Save trained models and related objects.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the models
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Saving models...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_filename = f"{output_dir}{name.lower().replace(' ', '_')}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_filename}")
        
        # Save scaler
        scaler_filename = f"{output_dir}scaler.pkl"
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_filename}")
        
        # Save results summary
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'accuracy': result['accuracy'],
                'roc_auc': result['roc_auc'],
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall']
            }
        
        results_filename = f"{output_dir}model_results.json"
        with open(results_filename, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Saved results summary to {results_filename}")
        
        return self
    
    def run_complete_training_pipeline(self, data, target_column='class'):
        """
        Run the complete model training pipeline.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Preprocessed dataset
        target_column : str
            Name of the target column
            
        Returns:
        --------
        self : FraudDetectionModelTrainer
            Returns self for method chaining
        """
        print("Starting complete model training pipeline...")
        print("="*60)
        
        (self.prepare_data(data, target_column)
         .handle_class_imbalance()
         .initialize_models()
         .train_models()
         .perform_cross_validation()
         .select_best_model()
         .evaluate_models()
         .plot_results()
         .save_models())
        
        print("\n" + "="*60)
        print("Training pipeline completed successfully!")
        
        return self


def main():
    """
    Main function to run the model training pipeline.
    """
    # Load preprocessed data
    data = pd.read_csv('../Data/processed/processed_fraud_data.csv')
    
    # Initialize trainer
    trainer = FraudDetectionModelTrainer()
    
    # Run complete pipeline
    trainer.run_complete_training_pipeline(data)
    
    # Print final results summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for name, result in trainer.results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        print(f"  F1 Score: {result['f1_score']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")


if __name__ == "__main__":
    main() 