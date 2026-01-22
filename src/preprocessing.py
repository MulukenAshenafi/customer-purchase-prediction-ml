"""
Data preprocessing module for customer purchase prediction.

This module handles data loading, missing value imputation, and basic data cleaning
according to the specified data requirements.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class DataPreprocessor:
    """
    Handles preprocessing of raw customer data for the purchase prediction model.

    Data requirements:
    - time_spent: Float, missing values -> median
    - pages_viewed: Integer, missing values -> mean
    - basket_value: Float, missing values -> 0
    - device_type: String, missing values -> "Unknown"
    - customer_type: String, missing values -> "New"
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the preprocessor.

        Args:
            data_dir: Path to data directory. If None, uses default project structure.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

    def load_raw_data(self, filename: str = "raw_customer_data.csv") -> pd.DataFrame:
        """
        Load raw customer data from CSV file.

        Args:
            filename: Name of the raw data file

        Returns:
            Raw customer DataFrame
        """
        file_path = self.data_dir / "raw" / filename
        return pd.read_csv(file_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the customer data according to requirements.

        Args:
            df: Raw customer DataFrame

        Returns:
            Cleaned DataFrame with proper data types and handled missing values
        """
        # Create a copy to avoid modifying the original
        clean_df = df.copy()

        # Handle missing values according to specifications
        clean_df['time_spent'] = clean_df['time_spent'].fillna(clean_df['time_spent'].median())
        clean_df['pages_viewed'] = clean_df['pages_viewed'].fillna(clean_df['pages_viewed'].mean())
        clean_df['basket_value'] = clean_df['basket_value'].fillna(0.0)
        clean_df['device_type'] = clean_df['device_type'].fillna('Unknown')
        clean_df['customer_type'] = clean_df['customer_type'].fillna('New')

        # Ensure correct data types
        clean_df['customer_id'] = clean_df['customer_id'].astype(int)
        clean_df['pages_viewed'] = clean_df['pages_viewed'].astype(int)
        clean_df['purchase'] = clean_df['purchase'].astype(int)

        return clean_df

    def load_and_clean(self, filename: str = "raw_customer_data.csv") -> pd.DataFrame:
        """
        Load raw data and apply cleaning pipeline.

        Args:
            filename: Name of the raw data file

        Returns:
            Cleaned DataFrame ready for feature engineering
        """
        raw_data = self.load_raw_data(filename)
        return self.clean_data(raw_data)

    def save_processed_data(self, df: pd.DataFrame, filename: str = "cleaned_data.csv") -> None:
        """
        Save processed data to the processed data directory.

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.data_dir / "processed" / filename
        df.to_csv(output_path, index=False)


def main():
    """Main function for testing the preprocessing pipeline."""
    preprocessor = DataPreprocessor()

    # Load and clean data
    clean_data = preprocessor.load_and_clean()

    print(f"Loaded and cleaned {len(clean_data)} customer records")
    print(f"Missing values remaining: {clean_data.isnull().sum().sum()}")
    print(f"Data types:\n{clean_data.dtypes}")

    # Save cleaned data
    preprocessor.save_processed_data(clean_data)
    print("Cleaned data saved to data/processed/cleaned_data.csv")


if __name__ == "__main__":
    main()