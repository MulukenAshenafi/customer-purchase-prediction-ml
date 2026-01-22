"""
Model training module for customer purchase prediction.

This module handles training multiple classifiers, model comparison,
and saving trained models for later use.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training and comparison of multiple classification models.

    Trains Logistic Regression, Random Forest, and XGBoost classifiers
    with proper cross-validation and reproducible results.
    """

    def __init__(self,
                 models_dir: Optional[str] = None,
                 random_seed: int = 42):
        """
        Initialize the model trainer.

        Args:
            models_dir: Directory to save trained models
            random_seed: Random seed for reproducibility
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed
        self.models = {}

        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)

    def load_training_data(self, filename: str = "input_model_features.csv") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load preprocessed training data.

        Args:
            filename: Name of the training data file

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        data_path = Path(__file__).parent.parent / "data" / "processed" / filename
        df = pd.read_csv(data_path)

        # Separate features and target
        X = df.drop(['customer_id', 'purchase'], axis=1)
        y = df['purchase']

        logger.info(f"Loaded training data: {len(X)} samples, {len(X.columns)} features")
        return X, y

    def create_train_val_test_split(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   train_size: float = 0.7,
                                   val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create train/validation/test splits.

        Args:
            X: Feature DataFrame
            y: Target Series
            train_size: Proportion for training set
            val_size: Proportion for validation set

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        test_size = 1 - train_size - val_size
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )

        # Second split: separate train and validation from temp
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_seed, stratify=y_temp
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with appropriate hyperparameters.

        Returns:
            Dictionary of model names to model instances
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_seed,
                n_jobs=-1
            )
        }

        self.models = models
        return models

    def train_model(self, model_name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a single model.

        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        logger.info(f"{model_name} training completed")
        return model

    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all models and return trained instances.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Dictionary of trained models
        """
        trained_models = {}

        for model_name, model in self.models.items():
            trained_model = self.train_model(model_name, model, X_train, y_train)
            trained_models[model_name] = trained_model

        return trained_models

    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate a model on given data.

        Args:
            model: Trained model
            X: Feature data
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }

        return metrics

    def save_model(self, model: Any, model_name: str) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Trained model instance
            model_name: Name for the saved model file
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

    def save_training_results(self,
                             trained_models: Dict[str, Any],
                             X_train: pd.DataFrame,
                             X_val: pd.DataFrame,
                             y_train: pd.Series,
                             y_val: pd.Series) -> str:
        """
        Save training results and model performance to JSON.

        Args:
            trained_models: Dictionary of trained models
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels

        Returns:
            Path to the saved results file
        """
        results = {
            'training_info': {
                'random_seed': self.random_seed,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'features': list(X_train.columns)
            },
            'model_performance': {}
        }

        # Evaluate each model on train and validation sets
        for model_name, model in trained_models.items():
            train_metrics = self.evaluate_model(model, X_train, y_train)
            val_metrics = self.evaluate_model(model, X_val, y_val)

            results['model_performance'][model_name] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }

        # Save results to JSON
        results_path = self.models_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training results saved to {results_path}")
        return str(results_path)

    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary containing trained models and training results
        """
        logger.info("Starting model training pipeline...")

        # Load training data
        X, y = self.load_training_data()

        # Create train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_val_test_split(X, y)

        # Initialize models
        self.initialize_models()

        # Train all models
        trained_models = self.train_all_models(X_train, y_train)

        # Save trained models
        for model_name, model in trained_models.items():
            self.save_model(model, model_name)

        # Save training results
        results_path = self.save_training_results(trained_models, X_train, X_val, y_train, y_val)

        # Also save the test set for final evaluation
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data_path = self.models_dir / "test_data.csv"
        test_data.to_csv(test_data_path, index=False)

        logger.info("Training pipeline completed successfully!")

        return {
            'trained_models': trained_models,
            'results_path': results_path,
            'test_data_path': str(test_data_path)
        }


def main():
    """Main function to run model training."""
    trainer = ModelTrainer()
    results = trainer.run_training_pipeline()

    # Print summary of results
    print("\nTraining Summary:")
    print("=" * 50)

    # Load and display results
    with open(results['results_path'], 'r') as f:
        training_results = json.load(f)

    for model_name, performance in training_results['model_performance'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Train F1: {performance['train_metrics']['f1_score']:.4f}")
        print(f"  Val F1: {performance['val_metrics']['f1_score']:.4f}")
        print(f"  Val Accuracy: {performance['val_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()