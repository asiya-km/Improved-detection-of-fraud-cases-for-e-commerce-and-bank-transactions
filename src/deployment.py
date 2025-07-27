"""
Deployment Module for Fraud Detection System
===========================================

This module provides production-ready deployment capabilities including:
- Flask API for real-time predictions
- Model loading and serving
- Input validation and preprocessing
- Health checks and monitoring
- Docker configuration utilities

Author: Asiya KM
Date: 2024
"""

import os
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionAPI:
    """
    Flask API for fraud detection predictions.
    
    This class provides a RESTful API for real-time fraud detection
    with comprehensive input validation and error handling.
    """
    
    def __init__(self, model_path: str = 'models/', host: str = '0.0.0.0', port: int = 5000):
        """
        Initialize the fraud detection API.
        
        Parameters:
        -----------
        model_path : str
            Path to the directory containing trained models
        host : str
            Host address for the API
        port : int
            Port number for the API
        """
        self.model_path = model_path
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        
        # Load models and scaler
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load models
        self.load_models()
        
        # Register routes
        self.register_routes()
        
    def load_models(self):
        """Load trained models and preprocessing objects."""
        try:
            # Load the best model (assuming it's saved as 'best_model.pkl')
            model_file = os.path.join(self.model_path, 'best_model.pkl')
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Best model loaded successfully")
            else:
                # Try loading individual models
                model_files = {
                    'logistic_regression': 'logistic_regression.pkl',
                    'random_forest': 'random_forest.pkl',
                    'gradient_boosting': 'gradient_boosting.pkl'
                }
                
                for model_name, filename in model_files.items():
                    filepath = os.path.join(self.model_path, filename)
                    if os.path.exists(filepath):
                        with open(filepath, 'rb') as f:
                            self.model = pickle.load(f)
                        logger.info(f"{model_name} model loaded successfully")
                        break
            
            # Load scaler
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            
            # Load feature names
            feature_file = os.path.join(self.model_path, 'feature_names.json')
            if os.path.exists(feature_file):
                with open(feature_file, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info("Feature names loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def register_routes(self):
        """Register API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'model_loaded': self.model is not None,
                'scaler_loaded': self.scaler is not None
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Make fraud detection predictions."""
            try:
                # Get input data
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Validate and preprocess input
                processed_data = self.preprocess_input(data)
                
                if processed_data is None:
                    return jsonify({'error': 'Invalid input data'}), 400
                
                # Make prediction
                prediction, probability = self.make_prediction(processed_data)
                
                # Determine risk level
                risk_level = self.determine_risk_level(probability)
                
                return jsonify({
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'risk_level': risk_level,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/predict_batch', methods=['POST'])
        def predict_batch():
            """Make batch fraud detection predictions."""
            try:
                # Get input data
                data = request.get_json()
                
                if not data or 'transactions' not in data:
                    return jsonify({'error': 'No transactions data provided'}), 400
                
                transactions = data['transactions']
                results = []
                
                for i, transaction in enumerate(transactions):
                    try:
                        # Preprocess transaction
                        processed_data = self.preprocess_input(transaction)
                        
                        if processed_data is not None:
                            # Make prediction
                            prediction, probability = self.make_prediction(processed_data)
                            risk_level = self.determine_risk_level(probability)
                            
                            results.append({
                                'transaction_id': i,
                                'prediction': int(prediction),
                                'probability': float(probability),
                                'risk_level': risk_level
                            })
                        else:
                            results.append({
                                'transaction_id': i,
                                'error': 'Invalid transaction data'
                            })
                    except Exception as e:
                        results.append({
                            'transaction_id': i,
                            'error': str(e)
                        })
                
                return jsonify({
                    'results': results,
                    'total_transactions': len(transactions),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Batch prediction error: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            """Get information about the loaded model."""
            if self.model is None:
                return jsonify({'error': 'No model loaded'}), 500
            
            model_info = {
                'model_type': type(self.model).__name__,
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'scaler_available': self.scaler is not None,
                'feature_names': self.feature_names[:10] if self.feature_names else []  # Show first 10
            }
            
            return jsonify(model_info)
    
    def preprocess_input(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        data : dict
            Input transaction data
            
        Returns:
        --------
        numpy.ndarray or None
            Preprocessed feature vector
        """
        try:
            # Create feature vector
            features = {}
            
            # Basic transaction features
            features['purchase_value'] = float(data.get('purchase_value', 0))
            features['purchase_value_log'] = np.log1p(features['purchase_value'])
            features['age'] = int(data.get('age', 0))
            
            # Time-based features
            purchase_time = pd.to_datetime(data.get('purchase_time', datetime.now()))
            features['hour_of_day'] = purchase_time.hour
            features['day_of_week'] = purchase_time.weekday()
            features['is_weekend'] = int(purchase_time.weekday() >= 5)
            features['is_business_hours'] = int(9 <= purchase_time.hour <= 17)
            
            # Transaction frequency features (default to 1 for new entities)
            features['device_transaction_count'] = int(data.get('device_transaction_count', 1))
            features['ip_transaction_count'] = int(data.get('ip_transaction_count', 1))
            features['user_transaction_count'] = int(data.get('user_transaction_count', 1))
            features['country_transaction_count'] = int(data.get('country_transaction_count', 1))
            
            # Time since previous transaction features
            features['time_since_prev_txn_user'] = float(data.get('time_since_prev_txn_user', 24))
            features['time_since_prev_txn_device'] = float(data.get('time_since_prev_txn_device', 24))
            features['time_since_prev_txn_ip'] = float(data.get('time_since_prev_txn_ip', 24))
            features['time_since_signup'] = float(data.get('time_since_signup', 24))
            
            # Risk score features
            features['device_risk_score'] = float(data.get('device_risk_score', 0))
            features['ip_risk_score'] = float(data.get('ip_risk_score', 0))
            features['user_risk_score'] = float(data.get('user_risk_score', 0))
            features['country_risk_score'] = float(data.get('country_risk_score', 0))
            features['composite_risk_score'] = float(data.get('composite_risk_score', 0))
            
            # Value-based features
            features['high_value_transaction'] = int(data.get('high_value_transaction', 0))
            features['very_high_value_transaction'] = int(data.get('very_high_value_transaction', 0))
            features['purchase_value_percentile'] = float(data.get('purchase_value_percentile', 0.5))
            
            # Behavioral features
            features['user_device_count'] = int(data.get('user_device_count', 1))
            features['user_ip_count'] = int(data.get('user_ip_count', 1))
            features['user_country_count'] = int(data.get('user_country_count', 1))
            features['multiple_devices'] = int(data.get('multiple_devices', 0))
            features['multiple_ips'] = int(data.get('multiple_ips', 0))
            features['multiple_countries'] = int(data.get('multiple_countries', 0))
            features['rapid_successive_txn'] = int(data.get('rapid_successive_txn', 0))
            
            # Categorical features (one-hot encoded)
            source = data.get('source', 'unknown')
            browser = data.get('browser', 'unknown')
            sex = data.get('sex', 'unknown')
            country = data.get('country', 'unknown')
            
            # Add one-hot encoded features (simplified)
            for cat_feature in ['source', 'browser', 'sex', 'country']:
                for value in ['SEO', 'Ads', 'Direct', 'Chrome', 'Safari', 'Firefox', 'M', 'F', 'US', 'UK', 'CA']:
                    features[f'{cat_feature}_{value}'] = int(data.get(cat_feature, 'unknown') == value)
            
            # Convert to DataFrame and ensure correct order
            if self.feature_names:
                feature_df = pd.DataFrame([features])
                # Ensure all required features are present
                for feature in self.feature_names:
                    if feature not in feature_df.columns:
                        feature_df[feature] = 0
                # Reorder columns to match training data
                feature_df = feature_df[self.feature_names]
            else:
                feature_df = pd.DataFrame([features])
            
            # Convert to numpy array
            feature_array = feature_df.values
            
            # Scale features if scaler is available
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            return None
    
    def make_prediction(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction using the loaded model.
        
        Parameters:
        -----------
        features : numpy.ndarray
            Preprocessed feature vector
            
        Returns:
        --------
        tuple : (prediction, probability)
            Prediction (0 or 1) and probability score
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0, 1]
        
        return prediction, probability
    
    def determine_risk_level(self, probability: float) -> str:
        """
        Determine risk level based on prediction probability.
        
        Parameters:
        -----------
        probability : float
            Prediction probability (0-1)
            
        Returns:
        --------
        str
            Risk level ('low', 'medium', 'high')
        """
        if probability >= 0.7:
            return 'high'
        elif probability >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def run(self, debug: bool = False):
        """
        Run the Flask API server.
        
        Parameters:
        -----------
        debug : bool
            Enable debug mode
        """
        logger.info(f"Starting fraud detection API on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def create_dockerfile():
    """Create a Dockerfile for containerized deployment."""
    dockerfile_content = """
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src/deployment.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("Dockerfile created successfully")


def create_docker_compose():
    """Create a docker-compose.yml file for easy deployment."""
    compose_content = """
version: '3.8'

services:
  fraud-detection-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)
    
    print("docker-compose.yml created successfully")


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fraud Detection API')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--model-path', default='models/', help='Path to models directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--create-docker', action='store_true', help='Create Docker files')
    
    args = parser.parse_args()
    
    if args.create_docker:
        create_dockerfile()
        create_docker_compose()
        return
    
    # Initialize and run API
    api = FraudDetectionAPI(
        model_path=args.model_path,
        host=args.host,
        port=args.port
    )
    
    api.run(debug=args.debug)


if __name__ == "__main__":
    main() 