"""
Feature engineering module for customer purchase prediction.

This module handles feature scaling, encoding, and creation of engineered features
to improve model performance and capture behavioral patterns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, List, Tuple
from pathlib import Path


class FeatureEngineer:
    """
    Handles feature engineering for customer purchase prediction.

    Engineered Features:
    1. engagement_score: Normalized combination of time_spent and pages_viewed
       - Captures overall user engagement level
       - Higher scores indicate more active browsing behavior

    2. basket_intensity: basket_value divided by pages_viewed (with smoothing)
       - Measures spending efficiency per page viewed
       - Indicates purchase intent strength

    3. behavioral_segment: Categorical segmentation based on activity patterns
       - 'high_engagement': High time + high pages
       - 'focused_shopper': Low time + high basket_value
       - 'casual_browser': Low engagement overall
       - 'window_shopper': High time + low basket_value
    """

    def __init__(self, scaling_method: str = "minmax"):
        """
        Initialize the feature engineer.

        Args:
            scaling_method: 'minmax' or 'standard' scaling for numerical features
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.numerical_cols = ['time_spent', 'pages_viewed', 'basket_value']
        self.categorical_cols = ['device_type', 'customer_type']

    def create_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Create engagement score from time_spent and pages_viewed.

        This feature captures overall user engagement by combining browsing time
        and page activity. Higher scores indicate more engaged users who are
        more likely to convert to purchases.

        Formula: (time_spent_norm + pages_viewed_norm) / 2
        """
        # Normalize components to 0-1 range for equal weighting
        time_norm = (df['time_spent'] - df['time_spent'].min()) / (df['time_spent'].max() - df['time_spent'].min())
        pages_norm = (df['pages_viewed'] - df['pages_viewed'].min()) / (df['pages_viewed'].max() - df['pages_viewed'].min())

        return (time_norm + pages_norm) / 2

    def create_basket_intensity(self, df: pd.DataFrame) -> pd.Series:
        """
        Create basket intensity feature.

        Measures how much value a customer adds to their basket per page viewed.
        Higher intensity suggests stronger purchase intent per unit of browsing effort.

        Formula: basket_value / (pages_viewed + 1) - adds smoothing to avoid division by zero
        """
        return df['basket_value'] / (df['pages_viewed'] + 1)

    def create_behavioral_segment(self, df: pd.DataFrame) -> pd.Series:
        """
        Create behavioral segmentation based on activity patterns.

        Segments users into behavioral categories that can help predict purchase likelihood:
        - high_engagement: Active browsers with high time and page views
        - focused_shopper: Efficient shoppers with high basket value relative to time
        - casual_browser: Low engagement across all metrics
        - window_shopper: High browsing time but low basket value
        """
        # Calculate percentiles for segmentation thresholds
        time_median = df['time_spent'].median()
        pages_median = df['pages_viewed'].median()
        basket_median = df['basket_value'].median()

        segments = []

        for _, row in df.iterrows():
            if row['time_spent'] > time_median and row['pages_viewed'] > pages_median:
                segment = 'high_engagement'
            elif row['basket_value'] > basket_median and row['pages_viewed'] <= pages_median:
                segment = 'focused_shopper'
            elif row['time_spent'] > time_median and row['basket_value'] <= basket_median:
                segment = 'window_shopper'
            else:
                segment = 'casual_browser'

            segments.append(segment)

        return pd.Series(segments, index=df.index)

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features to the DataFrame.

        Args:
            df: Input DataFrame with base features

        Returns:
            DataFrame with additional engineered features
        """
        df_engineered = df.copy()

        # Add engagement score
        df_engineered['engagement_score'] = self.create_engagement_score(df)

        # Add basket intensity
        df_engineered['basket_intensity'] = self.create_basket_intensity(df)

        # Add behavioral segmentation
        df_engineered['behavioral_segment'] = self.create_behavioral_segment(df)

        return df_engineered

    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using the specified scaling method.

        Args:
            df: DataFrame with numerical features to scale
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            DataFrame with scaled numerical features
        """
        df_scaled = df.copy()

        if fit:
            if self.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaling_method == "standard":
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")

            df_scaled[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_scaled[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])

        return df_scaled

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features with all possible categories.

        Args:
            df: DataFrame with categorical features

        Returns:
            DataFrame with one-hot encoded categorical features
        """
        # Define all possible categories (matching training data)
        device_categories = ['Desktop', 'Mobile', 'Tablet', 'Unknown']
        customer_categories = ['New', 'Returning']

        # One-hot encode with all possible categories
        device_encoded = pd.get_dummies(df['device_type'], prefix='device_type')
        customer_encoded = pd.get_dummies(df['customer_type'], prefix='customer_type')

        # Ensure all categories exist (add missing ones as False)
        for cat in device_categories:
            col_name = f'device_type_{cat}'
            if col_name not in device_encoded.columns:
                device_encoded[col_name] = False

        for cat in customer_categories:
            col_name = f'customer_type_{cat}'
            if col_name not in customer_encoded.columns:
                customer_encoded[col_name] = False

        # Reorder columns to match expected order
        device_encoded = device_encoded[['device_type_Desktop', 'device_type_Mobile',
                                       'device_type_Tablet', 'device_type_Unknown']]
        customer_encoded = customer_encoded[['customer_type_New', 'customer_type_Returning']]

        # Drop original categorical columns and add encoded columns
        df_encoded = df.drop(columns=self.categorical_cols)
        df_encoded = pd.concat([df_encoded, device_encoded, customer_encoded], axis=1)

        return df_encoded

    def process_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Complete feature processing pipeline.

        Args:
            df: Raw feature DataFrame
            fit_scaler: Whether to fit the scaler (training) or use existing (inference)

        Returns:
            Processed feature DataFrame ready for modeling
        """
        # Add engineered features
        df_with_features = self.add_engineered_features(df)

        # Scale numerical features
        df_scaled = self.scale_numerical_features(df_with_features, fit=fit_scaler)

        # One-hot encode categorical features
        df_final = self.encode_categorical_features(df_scaled)

        return df_final

    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features that will be created by the pipeline.

        Returns:
            List of feature names in the processed dataset
        """
        base_features = ['customer_id', 'time_spent', 'pages_viewed', 'basket_value', 'purchase']
        engineered_features = ['engagement_score', 'basket_intensity', 'behavioral_segment']

        # Add one-hot encoded categorical features (assuming all possible categories)
        encoded_features = [
            'device_type_Desktop', 'device_type_Mobile', 'device_type_Tablet', 'device_type_Unknown',
            'customer_type_New', 'customer_type_Returning'
        ]

        return base_features + engineered_features + encoded_features


def main():
    """Main function for testing feature engineering."""
    from preprocessing import DataPreprocessor

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.load_and_clean()

    # Create feature engineer and process features
    feature_engineer = FeatureEngineer()
    processed_features = feature_engineer.process_features(clean_data)

    print(f"Original features: {len(clean_data.columns)}")
    print(f"Processed features: {len(processed_features.columns)}")
    print(f"New engineered features: engagement_score, basket_intensity, behavioral_segment")
    print(f"One-hot encoded categorical features added")
    print(f"\nSample processed features:\n{processed_features.head()}")


if __name__ == "__main__":
    main()