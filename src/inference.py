"""
Inference module for customer purchase prediction.

This module handles loading trained models and making predictions
on new customer data without requiring the training pipeline.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from features import FeatureEngineer
from preprocessing import DataPreprocessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PurchasePredictor:
    """
    Production-ready predictor for customer purchase likelihood.

    Loads trained models and preprocessing pipelines to make predictions
    on new customer data.
    """

    def __init__(self,
                 models_dir: Optional[str] = None,
                 data_dir: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing data files
        """
        # Set up paths
        self.project_root = Path(__file__).parent.parent

        if models_dir is None:
            self.models_dir = self.project_root / "models"
        else:
            self.models_dir = Path(models_dir)

        if data_dir is None:
            self.data_dir = self.project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        # Initialize components
        self.model = None
        self.feature_engineer = None
        self.preprocessor = None

        # Load components
        self._load_components()

    def _load_components(self) -> None:
        """
        Load all necessary components for inference.
        """
        try:
            # Load the best trained model
            model_path = self.models_dir / "best_model.joblib"
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")

            # Initialize feature engineer (will load scaler during prediction)
            self.feature_engineer = FeatureEngineer()

            # Initialize preprocessor
            self.preprocessor = DataPreprocessor(str(self.data_dir))

            logger.info("All inference components loaded successfully")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required model file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading inference components: {e}")

    def preprocess_new_data(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new customer data for prediction.

        This matches the preprocessing used during training:
        - Handle missing values
        - Scale numerical features
        - One-hot encode categorical features

        Args:
            customer_data: Raw customer data DataFrame

        Returns:
            Processed DataFrame ready for prediction
        """
        # For prediction data, we don't expect a 'purchase' column
        df_copy = customer_data.copy()

        # Handle missing values according to business rules (matching training preprocessing)
        df_copy['time_spent'] = df_copy['time_spent'].fillna(df_copy['time_spent'].median() if not df_copy['time_spent'].empty else 34.33)
        df_copy['pages_viewed'] = df_copy['pages_viewed'].fillna(df_copy['pages_viewed'].mean() if not df_copy['pages_viewed'].empty else 9.78)
        df_copy['basket_value'] = df_copy['basket_value'].fillna(0.0)
        df_copy['device_type'] = df_copy['device_type'].fillna('Unknown')
        df_copy['customer_type'] = df_copy['customer_type'].fillna('New')

        # Ensure correct data types
        df_copy['customer_id'] = df_copy['customer_id'].astype(int)
        df_copy['pages_viewed'] = df_copy['pages_viewed'].astype(int)

        # Scale numerical features (fit on prediction data for now)
        numerical_cols = ['time_spent', 'pages_viewed', 'basket_value']
        df_scaled = self.feature_engineer.scale_numerical_features(df_copy, fit=True)

        # One-hot encode categorical features
        df_encoded = self.feature_engineer.encode_categorical_features(df_scaled)

        # Remove customer_id from features (not used for prediction)
        if 'customer_id' in df_encoded.columns:
            df_encoded = df_encoded.drop('customer_id', axis=1)

        return df_encoded

    def predict(self,
               customer_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
               return_probabilities: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Make purchase predictions for new customer data.

        Args:
            customer_data: Customer data in various formats:
                          - pandas DataFrame
                          - Single customer dict
                          - List of customer dicts
            return_probabilities: Whether to return prediction probabilities

        Returns:
            DataFrame or dict with predictions and optional probabilities
        """
        # Convert input to DataFrame
        if isinstance(customer_data, dict):
            # Single customer
            df = pd.DataFrame([customer_data])
            single_customer = True
        elif isinstance(customer_data, list):
            # Multiple customers
            df = pd.DataFrame(customer_data)
            single_customer = False
        else:
            # Already a DataFrame
            df = customer_data.copy()
            single_customer = False

        # Validate required columns
        required_cols = ['customer_id', 'time_spent', 'pages_viewed',
                        'basket_value', 'device_type', 'customer_type']

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Store customer IDs for output
        customer_ids = df['customer_id'].tolist()

        # Preprocess the data
        processed_data = self.preprocess_new_data(df)

        # Make predictions (customer_id already removed in preprocessing)
        predictions = self.model.predict(processed_data)

        # Get prediction probabilities if requested
        probabilities = None
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_data)[:, 1]

        # Create results DataFrame
        results_df = pd.DataFrame({
            'customer_id': customer_ids,
            'purchase_prediction': predictions.astype(int)
        })

        if probabilities is not None:
            results_df['purchase_probability'] = probabilities

        # Add prediction confidence interpretation
        results_df['prediction_confidence'] = pd.cut(
            results_df['purchase_probability'] if probabilities is not None else predictions.astype(float),
            bins=[0, 0.3, 0.7, 1.0],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )

        # Return single customer as dict if input was single customer
        if single_customer:
            result_dict = results_df.iloc[0].to_dict()
            return result_dict

        return results_df

    def predict_from_csv(self,
                        csv_path: str,
                        output_path: Optional[str] = None,
                        return_probabilities: bool = True) -> pd.DataFrame:
        """
        Make predictions from a CSV file.

        Args:
            csv_path: Path to CSV file with customer data
            output_path: Optional path to save predictions
            return_probabilities: Whether to include probabilities

        Returns:
            DataFrame with predictions
        """
        # Load data from CSV
        input_path = Path(csv_path)
        customer_data = pd.read_csv(input_path)
        logger.info(f"Loaded data from {csv_path}: {len(customer_data)} customers")

        # Make predictions
        predictions = self.predict(customer_data, return_probabilities=return_probabilities)

        # Save predictions if output path provided
        if output_path:
            output_path = Path(output_path)
            predictions.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")

        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and pipeline.

        Returns:
            Dictionary with model and pipeline information
        """
        info = {
            'model_type': type(self.model).__name__,
            'feature_engineering': {
                'scaling_method': self.feature_engineer.scaling_method,
                'engineered_features': [
                    'engagement_score',
                    'basket_intensity',
                    'behavioral_segment'
                ]
            },
            'expected_features': self.feature_engineer.get_feature_names()
        }

        # Try to get model parameters
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()

        return info

    def validate_input_data(self, data: pd.DataFrame) -> List[str]:
        """
        Validate input data and return list of issues.

        Args:
            data: Input customer data DataFrame

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check required columns
        required_cols = ['customer_id', 'time_spent', 'pages_viewed',
                        'basket_value', 'device_type', 'customer_type']

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

        # Check for missing values in critical columns
        for col in ['time_spent', 'pages_viewed', 'basket_value']:
            if col in data.columns and data[col].isnull().any():
                issues.append(f"Column '{col}' has missing values that will be imputed")

        # Check data types
        if 'customer_id' in data.columns and not pd.api.types.is_numeric_dtype(data['customer_id']):
            issues.append("customer_id should be numeric")

        # Check categorical values
        if 'device_type' in data.columns:
            valid_devices = ['Mobile', 'Desktop', 'Tablet', 'Unknown']
            invalid_devices = set(data['device_type'].dropna()) - set(valid_devices)
            if invalid_devices:
                issues.append(f"Unknown device_type values: {invalid_devices}")

        if 'customer_type' in data.columns:
            valid_types = ['New', 'Returning']
            invalid_types = set(data['customer_type'].dropna()) - set(valid_types)
            if invalid_types:
                issues.append(f"Unknown customer_type values: {invalid_types}")

        return issues


def main():
    """Main function for testing inference pipeline."""
    predictor = PurchasePredictor()

    # Example: Predict for sample customers
    sample_customers = [
        {
            'customer_id': 99991,
            'time_spent': 25.5,
            'pages_viewed': 8,
            'basket_value': 75.0,
            'device_type': 'Mobile',
            'customer_type': 'Returning'
        },
        {
            'customer_id': 99992,
            'time_spent': 5.2,
            'pages_viewed': 2,
            'basket_value': 0.0,
            'device_type': 'Desktop',
            'customer_type': 'New'
        }
    ]

    print("Testing Inference Pipeline:")
    print("=" * 40)

    # Get model info
    model_info = predictor.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Scaling Method: {model_info['feature_engineering']['scaling_method']}")

    # Make predictions
    predictions = predictor.predict(sample_customers, return_probabilities=True)
    print("\nPredictions:")
    print(predictions)

    print("\nInference pipeline test completed successfully!")


if __name__ == "__main__":
    main()