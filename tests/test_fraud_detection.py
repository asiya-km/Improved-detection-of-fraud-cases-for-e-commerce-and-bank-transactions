"""
Unit Tests for Fraud Detection System
====================================

This module contains comprehensive unit tests for all components
of the fraud detection system.

Author: Asiya KM
Date: 2024
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection_pipeline import FraudDetectionPipeline
from data_preprocessing import (
    load_and_validate_data,
    clean_fraud_data,
    clean_ip_data,
    map_ip_to_country,
    create_transaction_frequency_features,
    create_time_based_features,
    create_value_based_features,
    create_risk_score_features,
    create_behavioral_features,
    encode_categorical_features,
    handle_missing_values
)
from model_training import FraudDetectionModelTrainer
from utils import get_geolocation


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample fraud data
        self.sample_fraud_data = pd.DataFrame({
            'user_id': ['user_1', 'user_2', 'user_3', 'user_4', 'user_5'],
            'signup_time': ['2024-01-01 10:00:00', '2024-01-02 11:00:00', 
                           '2024-01-03 12:00:00', '2024-01-04 13:00:00', 
                           '2024-01-05 14:00:00'],
            'purchase_time': ['2024-01-01 15:00:00', '2024-01-02 16:00:00',
                             '2024-01-03 17:00:00', '2024-01-04 18:00:00',
                             '2024-01-05 19:00:00'],
            'purchase_value': [100.0, 200.0, 150.0, 300.0, 250.0],
            'device_id': ['device_1', 'device_2', 'device_1', 'device_3', 'device_2'],
            'source': ['SEO', 'Ads', 'Direct', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Safari', 'Firefox', 'Chrome', 'Safari'],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'age': [25, 30, 35, 28, 32],
            'ip_address': ['192168001001', '192168001002', '192168001003',
                          '192168001004', '192168001005'],
            'class': [0, 1, 0, 0, 1]
        })
        
        # Create sample IP data
        self.sample_ip_data = pd.DataFrame({
            'lower_bound_ip_address': ['192168001000', '192168001010', '192168001020'],
            'upper_bound_ip_address': ['192168001009', '192168001019', '192168001029'],
            'country': ['US', 'UK', 'CA']
        })
    
    def test_clean_fraud_data(self):
        """Test fraud data cleaning."""
        # Add some problematic data
        test_data = self.sample_fraud_data.copy()
        test_data.loc[5] = test_data.iloc[0]  # Add duplicate
        test_data.loc[6] = test_data.iloc[0]  # Add another duplicate
        test_data.loc[7] = [None, None, None, -50, None, None, None, None, -5, None, None]
        
        cleaned_data = clean_fraud_data(test_data)
        
        # Check that duplicates are removed
        self.assertEqual(len(cleaned_data), len(self.sample_fraud_data))
        
        # Check that invalid data is removed
        self.assertNotIn(-50, cleaned_data['purchase_value'].values)
        self.assertNotIn(-5, cleaned_data['age'].values)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['signup_time']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['purchase_time']))
    
    def test_clean_ip_data(self):
        """Test IP data cleaning."""
        # Add some problematic data
        test_data = self.sample_ip_data.copy()
        test_data.loc[3] = [None, None, None]  # Add missing values
        test_data.loc[4] = ['192168001030', '192168001020', 'Invalid']  # Invalid range
        
        cleaned_data = clean_ip_data(test_data)
        
        # Check that missing values are removed
        self.assertEqual(len(cleaned_data), len(self.sample_ip_data))
        
        # Check that invalid ranges are removed
        self.assertNotIn('Invalid', cleaned_data['country'].values)
    
    def test_map_ip_to_country(self):
        """Test IP to country mapping."""
        merged_data = map_ip_to_country(self.sample_fraud_data, self.sample_ip_data)
        
        # Check that country column is added
        self.assertIn('country', merged_data.columns)
        
        # Check that all transactions have a country
        self.assertEqual(len(merged_data), len(self.sample_fraud_data))
        self.assertTrue(merged_data['country'].notna().all())
    
    def test_create_transaction_frequency_features(self):
        """Test transaction frequency feature creation."""
        data = create_transaction_frequency_features(self.sample_fraud_data)
        
        # Check that frequency features are added
        expected_features = ['device_transaction_count', 'ip_transaction_count', 
                           'country_transaction_count', 'user_transaction_count']
        
        for feature in expected_features:
            self.assertIn(feature, data.columns)
        
        # Check that device_1 has 2 transactions
        device_1_data = data[data['device_id'] == 'device_1']
        self.assertEqual(device_1_data['device_transaction_count'].iloc[0], 2)
    
    def test_create_time_based_features(self):
        """Test time-based feature creation."""
        data = create_time_based_features(self.sample_fraud_data)
        
        # Check that time features are added
        expected_features = ['hour_of_day', 'day_of_week', 'is_weekend', 
                           'is_business_hours']
        
        for feature in expected_features:
            self.assertIn(feature, data.columns)
        
        # Check that hour_of_day is between 0 and 23
        self.assertTrue((data['hour_of_day'] >= 0).all())
        self.assertTrue((data['hour_of_day'] <= 23).all())
    
    def test_create_value_based_features(self):
        """Test value-based feature creation."""
        data = create_value_based_features(self.sample_fraud_data)
        
        # Check that value features are added
        expected_features = ['purchase_value_log', 'high_value_transaction', 
                           'very_high_value_transaction', 'purchase_value_percentile']
        
        for feature in expected_features:
            self.assertIn(feature, data.columns)
        
        # Check that log transformation is applied
        self.assertTrue((data['purchase_value_log'] >= 0).all())
    
    def test_create_risk_score_features(self):
        """Test risk score feature creation."""
        data = create_risk_score_features(self.sample_fraud_data)
        
        # Check that risk features are added
        expected_features = ['device_risk_score', 'ip_risk_score', 
                           'user_risk_score', 'country_risk_score', 
                           'composite_risk_score']
        
        for feature in expected_features:
            self.assertIn(feature, data.columns)
        
        # Check that risk scores are non-negative
        for feature in expected_features:
            self.assertTrue((data[feature] >= 0).all())
    
    def test_create_behavioral_features(self):
        """Test behavioral feature creation."""
        data = create_behavioral_features(self.sample_fraud_data)
        
        # Check that behavioral features are added
        expected_features = ['user_device_count', 'user_ip_count', 
                           'user_country_count', 'multiple_devices', 
                           'multiple_ips', 'multiple_countries']
        
        for feature in expected_features:
            self.assertIn(feature, data.columns)
        
        # Check that user_device_count is at least 1
        self.assertTrue((data['user_device_count'] >= 1).all())
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        categorical_columns = ['source', 'browser', 'sex']
        encoded_data = encode_categorical_features(self.sample_fraud_data, categorical_columns)
        
        # Check that dummy variables are created
        expected_dummy_features = ['source_Ads', 'source_Direct', 'browser_Safari', 
                                 'browser_Firefox', 'sex_F']
        
        for feature in expected_dummy_features:
            self.assertIn(feature, encoded_data.columns)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Add missing values
        test_data = self.sample_fraud_data.copy()
        test_data.loc[0, 'purchase_value'] = np.nan
        test_data.loc[1, 'age'] = np.nan
        
        # Test fill_zero strategy
        filled_data = handle_missing_values(test_data, strategy='fill_zero')
        self.assertEqual(filled_data.isnull().sum().sum(), 0)
        
        # Test drop strategy
        dropped_data = handle_missing_values(test_data, strategy='drop')
        self.assertLess(len(dropped_data), len(test_data))


class TestModelTraining(unittest.TestCase):
    """Test cases for model training functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
    
    def test_model_trainer_initialization(self):
        """Test model trainer initialization."""
        trainer = FraudDetectionModelTrainer(random_state=42)
        
        self.assertEqual(trainer.random_state, 42)
        self.assertEqual(trainer.models, {})
        self.assertIsNone(trainer.best_model)
    
    def test_prepare_data(self):
        """Test data preparation."""
        trainer = FraudDetectionModelTrainer()
        trainer.prepare_data(self.sample_data, target_column='class')
        
        # Check that data is split
        self.assertIsNotNone(trainer.X_train)
        self.assertIsNotNone(trainer.X_test)
        self.assertIsNotNone(trainer.y_train)
        self.assertIsNotNone(trainer.y_test)
        
        # Check shapes
        self.assertEqual(len(trainer.X_train) + len(trainer.X_test), len(self.sample_data))
        self.assertEqual(len(trainer.y_train) + len(trainer.y_test), len(self.sample_data))
    
    def test_handle_class_imbalance(self):
        """Test class imbalance handling."""
        trainer = FraudDetectionModelTrainer()
        trainer.prepare_data(self.sample_data, target_column='class')
        
        # Test SMOTE
        trainer.handle_class_imbalance(method='smote')
        self.assertIsNotNone(trainer.X_train_resampled)
        self.assertIsNotNone(trainer.y_train_resampled)
        
        # Check that resampled data has more samples
        self.assertGreaterEqual(len(trainer.X_train_resampled), len(trainer.X_train))
    
    def test_initialize_models(self):
        """Test model initialization."""
        trainer = FraudDetectionModelTrainer()
        trainer.initialize_models()
        
        # Check that models are initialized
        expected_models = ['Logistic Regression', 'Random Forest', 
                          'Gradient Boosting', 'SVM']
        
        for model_name in expected_models:
            self.assertIn(model_name, trainer.models)
    
    def test_train_models(self):
        """Test model training."""
        trainer = FraudDetectionModelTrainer()
        trainer.prepare_data(self.sample_data, target_column='class')
        trainer.handle_class_imbalance()
        trainer.initialize_models()
        trainer.train_models()
        
        # Check that results are stored
        self.assertIsNotNone(trainer.results)
        self.assertGreater(len(trainer.results), 0)
        
        # Check that each model has results
        for model_name in trainer.models.keys():
            self.assertIn(model_name, trainer.results)
            
            result = trainer.results[model_name]
            self.assertIn('accuracy', result)
            self.assertIn('roc_auc', result)
            self.assertIn('f1_score', result)
    
    def test_select_best_model(self):
        """Test best model selection."""
        trainer = FraudDetectionModelTrainer()
        trainer.prepare_data(self.sample_data, target_column='class')
        trainer.handle_class_imbalance()
        trainer.initialize_models()
        trainer.train_models()
        trainer.select_best_model(metric='f1_score')
        
        # Check that best model is selected
        self.assertIsNotNone(trainer.best_model)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.ip_to_country = pd.DataFrame({
            'lower_bound_ip_address': [192168001000, 192168001010, 192168001020],
            'upper_bound_ip_address': [192168001009, 192168001019, 192168001029],
            'country': ['US', 'UK', 'CA']
        })
    
    def test_get_geolocation_valid_ip(self):
        """Test geolocation lookup with valid IP."""
        ip_address = 192168001005
        country = get_geolocation(ip_address, self.ip_to_country)
        self.assertEqual(country, 'US')
    
    def test_get_geolocation_invalid_ip(self):
        """Test geolocation lookup with invalid IP."""
        ip_address = 999999999999
        country = get_geolocation(ip_address, self.ip_to_country)
        self.assertEqual(country, 'Unknown')
    
    def test_get_geolocation_string_ip(self):
        """Test geolocation lookup with string IP."""
        ip_address = "192168001005"
        country = get_geolocation(ip_address, self.ip_to_country)
        self.assertEqual(country, 'US')


class TestFraudDetectionPipeline(unittest.TestCase):
    """Test cases for the complete fraud detection pipeline."""
    
    def setUp(self):
        """Set up test data."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data files
        self.create_sample_data_files()
    
    def create_sample_data_files(self):
        """Create sample data files for testing."""
        # Sample fraud data
        fraud_data = pd.DataFrame({
            'user_id': ['user_1', 'user_2', 'user_3'],
            'signup_time': ['2024-01-01 10:00:00', '2024-01-02 11:00:00', '2024-01-03 12:00:00'],
            'purchase_time': ['2024-01-01 15:00:00', '2024-01-02 16:00:00', '2024-01-03 17:00:00'],
            'purchase_value': [100.0, 200.0, 150.0],
            'device_id': ['device_1', 'device_2', 'device_1'],
            'source': ['SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Safari', 'Firefox'],
            'sex': ['M', 'F', 'M'],
            'age': [25, 30, 35],
            'ip_address': ['192168001001', '192168001002', '192168001003'],
            'class': [0, 1, 0]
        })
        
        # Sample IP data
        ip_data = pd.DataFrame({
            'lower_bound_ip_address': ['192168001000', '192168001010'],
            'upper_bound_ip_address': ['192168001009', '192168001019'],
            'country': ['US', 'UK']
        })
        
        # Save files
        fraud_data.to_csv(os.path.join(self.temp_dir, 'Fraud_Data.csv'), index=False)
        ip_data.to_csv(os.path.join(self.temp_dir, 'IpAddress_to_Country.csv'), index=False)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = FraudDetectionPipeline(data_path=self.temp_dir + '/')
        
        self.assertEqual(pipeline.data_path, self.temp_dir + '/')
        self.assertIsNone(pipeline.fraud_data)
        self.assertIsNone(pipeline.ip_to_country)
    
    def test_load_data(self):
        """Test data loading."""
        pipeline = FraudDetectionPipeline(data_path=self.temp_dir + '/')
        pipeline.load_data()
        
        self.assertIsNotNone(pipeline.fraud_data)
        self.assertIsNotNone(pipeline.ip_to_country)
        self.assertEqual(len(pipeline.fraud_data), 3)
        self.assertEqual(len(pipeline.ip_to_country), 2)
    
    def test_clean_data(self):
        """Test data cleaning."""
        pipeline = FraudDetectionPipeline(data_path=self.temp_dir + '/')
        pipeline.load_data()
        pipeline.clean_data()
        
        # Check that data types are correct
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(pipeline.fraud_data['signup_time']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(pipeline.fraud_data['purchase_time']))
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataPreprocessing,
        TestModelTraining,
        TestUtils,
        TestFraudDetectionPipeline
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 