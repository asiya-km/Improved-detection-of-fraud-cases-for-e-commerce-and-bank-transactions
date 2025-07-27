"""
Data Preprocessing Module for Fraud Detection
============================================

This module contains comprehensive data preprocessing functions including:
- Data cleaning and validation
- Feature engineering
- Data transformation utilities
- Geolocation mapping functions

Author: Asiya KM
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def load_and_validate_data(data_path='../Data/'):
    """
    Load and perform initial validation of all datasets.
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory
        
    Returns:
    --------
    tuple : (fraud_data, ip_to_country, credit_card)
        Loaded and validated datasets
    """
    print("Loading and validating datasets...")
    
    # Load datasets
    fraud_data = pd.read_csv(f'{data_path}Fraud_Data.csv')
    ip_to_country = pd.read_csv(f'{data_path}IpAddress_to_Country.csv')
    credit_card = pd.read_csv(f'{data_path}creditcard.csv')
    
    # Basic validation
    print(f"Fraud data shape: {fraud_data.shape}")
    print(f"IP to Country data shape: {ip_to_country.shape}")
    print(f"Credit card data shape: {credit_card.shape}")
    
    # Check for required columns
    required_fraud_cols = ['user_id', 'signup_time', 'purchase_time', 'purchase_value', 
                          'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class']
    required_ip_cols = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    
    missing_fraud_cols = set(required_fraud_cols) - set(fraud_data.columns)
    missing_ip_cols = set(required_ip_cols) - set(ip_to_country.columns)
    
    if missing_fraud_cols:
        raise ValueError(f"Missing columns in fraud data: {missing_fraud_cols}")
    if missing_ip_cols:
        raise ValueError(f"Missing columns in IP data: {missing_ip_cols}")
    
    return fraud_data, ip_to_country, credit_card

def clean_fraud_data(fraud_data):
    """
    Clean the fraud dataset by handling missing values, duplicates, and data types.
    
    Parameters:
    -----------
    fraud_data : pandas.DataFrame
        Raw fraud dataset
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned fraud dataset
    """
    print("Cleaning fraud data...")
    
    # Check for missing values
    missing_values = fraud_data.isnull().sum()
    print("Missing values:")
    print(missing_values[missing_values > 0])
    
    # Check for duplicates
    duplicates = fraud_data.duplicated().sum()
    print(f"Duplicates found: {duplicates}")
    
    # Remove duplicates
    if duplicates > 0:
        fraud_data = fraud_data.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    # Convert data types
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data['ip_address'] = pd.to_numeric(fraud_data['ip_address'], errors='coerce')
    
    # Validate age range
    fraud_data = fraud_data[(fraud_data['age'] >= 0) & (fraud_data['age'] <= 120)]
    
    # Validate purchase value
    fraud_data = fraud_data[fraud_data['purchase_value'] > 0]
    
    print(f"Cleaned fraud data shape: {fraud_data.shape}")
    return fraud_data

def clean_ip_data(ip_to_country):
    """
    Clean the IP-to-country mapping dataset.
    
    Parameters:
    -----------
    ip_to_country : pandas.DataFrame
        Raw IP-to-country dataset
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned IP-to-country dataset
    """
    print("Cleaning IP-to-country data...")
    
    # Check for missing values
    missing_values = ip_to_country.isnull().sum()
    print("Missing values:")
    print(missing_values[missing_values > 0])
    
    # Remove rows with missing values
    ip_to_country = ip_to_country.dropna()
    
    # Convert IP addresses to numeric
    ip_to_country['lower_bound_ip_address'] = pd.to_numeric(
        ip_to_country['lower_bound_ip_address'], errors='coerce')
    ip_to_country['upper_bound_ip_address'] = pd.to_numeric(
        ip_to_country['upper_bound_ip_address'], errors='coerce')
    
    # Remove invalid IP ranges
    ip_to_country = ip_to_country[
        (ip_to_country['lower_bound_ip_address'] >= 0) &
        (ip_to_country['upper_bound_ip_address'] >= 0) &
        (ip_to_country['lower_bound_ip_address'] <= ip_to_country['upper_bound_ip_address'])
    ]
    
    # Remove duplicates
    duplicates = ip_to_country.duplicated().sum()
    if duplicates > 0:
        ip_to_country = ip_to_country.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    print(f"Cleaned IP data shape: {ip_to_country.shape}")
    return ip_to_country

def map_ip_to_country(fraud_data, ip_to_country):
    """
    Map IP addresses to countries using efficient merge_asof.
    
    Parameters:
    -----------
    fraud_data : pandas.DataFrame
        Fraud dataset with IP addresses
    ip_to_country : pandas.DataFrame
        IP-to-country mapping dataset
        
    Returns:
    --------
    pandas.DataFrame
        Fraud dataset with country information added
    """
    print("Mapping IP addresses to countries...")
    
    # Sort both DataFrames for merge_asof
    fraud_data_sorted = fraud_data.sort_values('ip_address').reset_index(drop=True)
    ip_to_country_sorted = ip_to_country.sort_values('lower_bound_ip_address').reset_index(drop=True)
    
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
    merged = merged.sort_index()
    
    # Drop unnecessary columns
    merged = merged.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, errors='ignore')
    
    print(f"Merged data shape: {merged.shape}")
    print(f"Countries found: {merged['country'].nunique()}")
    
    return merged

def create_transaction_frequency_features(data):
    """
    Create transaction frequency features for various entities.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with transaction information
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with frequency features added
    """
    print("Creating transaction frequency features...")
    
    # Device transaction count
    data['device_transaction_count'] = data.groupby('device_id')['device_id'].transform('count')
    
    # IP transaction count
    data['ip_transaction_count'] = data.groupby('ip_address')['ip_address'].transform('count')
    
    # Country transaction count
    data['country_transaction_count'] = data.groupby('country')['country'].transform('count')
    
    # User transaction count
    data['user_transaction_count'] = data.groupby('user_id')['user_id'].transform('count')
    
    print("Transaction frequency features created")
    return data

def create_time_based_features(data):
    """
    Create time-based features for fraud detection.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with timestamp information
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with time-based features added
    """
    print("Creating time-based features...")
    
    # Time since previous transaction for each user
    data = data.sort_values(['user_id', 'purchase_time'])
    data['time_since_prev_txn_user'] = data.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
    
    # Time since previous transaction for each device
    data = data.sort_values(['device_id', 'purchase_time'])
    data['time_since_prev_txn_device'] = data.groupby('device_id')['purchase_time'].diff().dt.total_seconds() / 3600
    
    # Time since previous transaction for each IP
    data = data.sort_values(['ip_address', 'purchase_time'])
    data['time_since_prev_txn_ip'] = data.groupby('ip_address')['purchase_time'].diff().dt.total_seconds() / 3600
    
    # Time between signup and purchase
    data['time_since_signup'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds() / 3600
    
    # Hour of day and day of week
    data['hour_of_day'] = data['purchase_time'].dt.hour
    data['day_of_week'] = data['purchase_time'].dt.dayofweek
    data['month'] = data['purchase_time'].dt.month
    
    # Weekend indicator
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Business hours indicator (9 AM to 5 PM)
    data['is_business_hours'] = ((data['hour_of_day'] >= 9) & (data['hour_of_day'] <= 17)).astype(int)
    
    print("Time-based features created")
    return data

def create_value_based_features(data):
    """
    Create value-based features for fraud detection.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with purchase value information
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with value-based features added
    """
    print("Creating value-based features...")
    
    # Log transformation of purchase value
    data['purchase_value_log'] = np.log1p(data['purchase_value'])
    
    # High value transaction indicator (95th percentile)
    high_value_threshold = data['purchase_value'].quantile(0.95)
    data['high_value_transaction'] = (data['purchase_value'] > high_value_threshold).astype(int)
    
    # Very high value transaction indicator (99th percentile)
    very_high_value_threshold = data['purchase_value'].quantile(0.99)
    data['very_high_value_transaction'] = (data['purchase_value'] > very_high_value_threshold).astype(int)
    
    # Purchase value percentiles
    data['purchase_value_percentile'] = data['purchase_value'].rank(pct=True)
    
    print("Value-based features created")
    return data

def create_risk_score_features(data):
    """
    Create risk score features based on various indicators.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with transaction information
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with risk score features added
    """
    print("Creating risk score features...")
    
    # Device risk score (frequency * average value)
    device_avg_value = data.groupby('device_id')['purchase_value'].transform('mean')
    data['device_risk_score'] = data['device_transaction_count'] * device_avg_value
    
    # IP risk score (frequency * average value)
    ip_avg_value = data.groupby('ip_address')['purchase_value'].transform('mean')
    data['ip_risk_score'] = data['ip_transaction_count'] * ip_avg_value
    
    # User risk score (frequency * average value)
    user_avg_value = data.groupby('user_id')['purchase_value'].transform('mean')
    data['user_risk_score'] = data['user_transaction_count'] * user_avg_value
    
    # Country risk score (frequency * average value)
    country_avg_value = data.groupby('country')['purchase_value'].transform('mean')
    data['country_risk_score'] = data['country_transaction_count'] * country_avg_value
    
    # Composite risk score
    data['composite_risk_score'] = (
        data['device_risk_score'] + 
        data['ip_risk_score'] + 
        data['user_risk_score'] + 
        data['country_risk_score']
    ) / 4
    
    print("Risk score features created")
    return data

def create_behavioral_features(data):
    """
    Create behavioral pattern features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with transaction information
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with behavioral features added
    """
    print("Creating behavioral features...")
    
    # Multiple devices per user
    user_device_count = data.groupby('user_id')['device_id'].nunique()
    data['user_device_count'] = data['user_id'].map(user_device_count)
    
    # Multiple IPs per user
    user_ip_count = data.groupby('user_id')['ip_address'].nunique()
    data['user_ip_count'] = data['user_id'].map(user_ip_count)
    
    # Multiple countries per user
    user_country_count = data.groupby('user_id')['country'].nunique()
    data['user_country_count'] = data['user_id'].map(user_country_count)
    
    # Suspicious behavior indicators
    data['multiple_devices'] = (data['user_device_count'] > 1).astype(int)
    data['multiple_ips'] = (data['user_ip_count'] > 1).astype(int)
    data['multiple_countries'] = (data['user_country_count'] > 1).astype(int)
    
    # Rapid successive transactions (within 1 hour)
    data['rapid_successive_txn'] = (data['time_since_prev_txn_user'] < 1).astype(int)
    
    print("Behavioral features created")
    return data

def encode_categorical_features(data, categorical_columns):
    """
    Encode categorical features using one-hot encoding.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with categorical features
    categorical_columns : list
        List of categorical column names
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with encoded categorical features
    """
    print("Encoding categorical features...")
    
    # Create dummy variables for categorical features
    for column in categorical_columns:
        if column in data.columns:
            dummies = pd.get_dummies(data[column], prefix=column, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            print(f"Encoded {column}: {len(dummies.columns)} new features")
    
    return data

def handle_missing_values(data, strategy='fill_zero'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with missing values
    strategy : str
        Strategy to handle missing values ('fill_zero', 'drop', 'interpolate')
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with missing values handled
    """
    print(f"Handling missing values using strategy: {strategy}")
    
    if strategy == 'fill_zero':
        data = data.fillna(0)
    elif strategy == 'drop':
        data = data.dropna()
    elif strategy == 'interpolate':
        data = data.interpolate(method='linear')
    
    print(f"Missing values after handling: {data.isnull().sum().sum()}")
    return data

def create_feature_summary(data):
    """
    Create a summary of all features in the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset to summarize
        
    Returns:
    --------
    pandas.DataFrame
        Feature summary
    """
    print("Creating feature summary...")
    
    feature_summary = pd.DataFrame({
        'feature_name': data.columns,
        'data_type': data.dtypes,
        'missing_values': data.isnull().sum(),
        'missing_percentage': (data.isnull().sum() / len(data)) * 100,
        'unique_values': data.nunique(),
        'min_value': data.min() if data.dtypes != 'object' else None,
        'max_value': data.max() if data.dtypes != 'object' else None,
        'mean_value': data.mean() if data.dtypes != 'object' else None
    })
    
    return feature_summary

def preprocess_fraud_data(data_path='../Data/', output_path='../Data/processed/'):
    """
    Complete preprocessing pipeline for fraud detection data.
    
    Parameters:
    -----------
    data_path : str
        Path to raw data directory
    output_path : str
        Path to save processed data
        
    Returns:
    --------
    pandas.DataFrame
        Fully preprocessed dataset ready for modeling
    """
    print("Starting complete data preprocessing pipeline...")
    
    # Load and validate data
    fraud_data, ip_to_country, credit_card = load_and_validate_data(data_path)
    
    # Clean data
    fraud_data = clean_fraud_data(fraud_data)
    ip_to_country = clean_ip_data(ip_to_country)
    
    # Merge with geolocation data
    merged_data = map_ip_to_country(fraud_data, ip_to_country)
    
    # Create features
    merged_data = create_transaction_frequency_features(merged_data)
    merged_data = create_time_based_features(merged_data)
    merged_data = create_value_based_features(merged_data)
    merged_data = create_risk_score_features(merged_data)
    merged_data = create_behavioral_features(merged_data)
    
    # Encode categorical features
    categorical_columns = ['source', 'browser', 'sex', 'country']
    merged_data = encode_categorical_features(merged_data, categorical_columns)
    
    # Handle missing values
    merged_data = handle_missing_values(merged_data, strategy='fill_zero')
    
    # Create feature summary
    feature_summary = create_feature_summary(merged_data)
    
    # Save processed data
    import os
    os.makedirs(output_path, exist_ok=True)
    merged_data.to_csv(f'{output_path}processed_fraud_data.csv', index=False)
    feature_summary.to_csv(f'{output_path}feature_summary.csv', index=False)
    
    print(f"Preprocessing completed. Processed data shape: {merged_data.shape}")
    print(f"Total features created: {len(merged_data.columns)}")
    
    return merged_data, feature_summary 