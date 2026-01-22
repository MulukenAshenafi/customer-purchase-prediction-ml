"""
Model evaluation module for customer purchase prediction.

This module provides comprehensive model evaluation including metrics calculation,
confusion matrix analysis, and business-aware model selection.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Any, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for classification tasks.

    Provides detailed metrics, confusion matrix analysis, and business-aware
    model selection for customer purchase prediction.
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model evaluator.

        Args:
            models_dir: Directory containing trained models
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)

    def load_trained_models(self) -> Dict[str, Any]:
        """
        Load all trained models from the models directory.

        Returns:
            Dictionary of model names to loaded model instances
        """
        model_files = list(self.models_dir.glob("*.joblib"))
        trained_models = {}

        for model_file in model_files:
            if model_file.name != "best_model.joblib":  # Skip the best model file if it exists
                model_name = model_file.stem
                model = joblib.load(model_file)
                trained_models[model_name] = model
                logger.info(f"Loaded model: {model_name}")

        return trained_models

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test data for evaluation.

        Returns:
            Tuple of (X_test, y_test)
        """
        test_data_path = self.models_dir / "test_data.csv"
        test_df = pd.read_csv(test_data_path)

        X_test = test_df.drop('purchase', axis=1)
        y_test = test_df['purchase']

        logger.info(f"Loaded test data: {len(X_test)} samples")
        return X_test, y_test

    def calculate_comprehensive_metrics(self,
                                       y_true: pd.Series,
                                       y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for AUC)

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }

        # Add AUC if probabilities are available
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                logger.warning("Could not calculate AUC - probabilities may not be available")

        return metrics

    def analyze_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Analyze confusion matrix and derive business insights.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with confusion matrix analysis
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        analysis = {
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            },
            'business_insights': {
                'correct_predictions': int(tp + tn),
                'incorrect_predictions': int(fp + fn),
                'missed_purchases': int(fn),  # False negatives - missed revenue opportunities
                'false_alarms': int(fp),      # False positives - wasted marketing spend
                'total_purchases': int(tp + fn),
                'predicted_purchases': int(tp + fp)
            }
        }

        # Calculate business-relevant ratios
        total_samples = len(y_true)
        analysis['business_insights'].update({
            'purchase_prediction_rate': (tp + fp) / total_samples,
            'actual_purchase_rate': (tp + fn) / total_samples,
            'missed_opportunity_rate': fn / (tp + fn) if (tp + fn) > 0 else 0
        })

        return analysis

    def evaluate_single_model(self, model: Any, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.

        Args:
            model: Trained model instance
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels

        Returns:
            Comprehensive evaluation results for the model
        """
        logger.info(f"Evaluating {model_name}...")

        # Get predictions
        y_pred = model.predict(X_test)

        # Get probabilities if available (for AUC)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except:
                pass

        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_prob)

        # Analyze confusion matrix
        cm_analysis = self.analyze_confusion_matrix(y_test, y_pred)

        # Get classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Combine all results
        evaluation_results = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix_analysis': cm_analysis,
            'classification_report': class_report,
            'test_samples': len(X_test)
        }

        return evaluation_results

    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models on test data.

        Returns:
            Dictionary of model names to their evaluation results
        """
        # Load models and test data
        trained_models = self.load_trained_models()
        X_test, y_test = self.load_test_data()

        # Evaluate each model
        evaluation_results = {}
        for model_name, model in trained_models.items():
            results = self.evaluate_single_model(model, model_name, X_test, y_test)
            evaluation_results[model_name] = results

        return evaluation_results

    def select_best_model(self, evaluation_results: Dict[str, Dict[str, Any]],
                         criteria: str = "f1_score") -> Tuple[str, Dict[str, Any]]:
        """
        Select the best model based on specified criteria.

        For customer purchase prediction, different metrics matter:
        - f1_score: Balanced measure of precision and recall
        - precision: Minimize false positives (marketing waste)
        - recall: Minimize false negatives (missed opportunities)

        Args:
            evaluation_results: Results from evaluate_all_models
            criteria: Metric to use for model selection ('f1_score', 'precision', 'recall', 'auc')

        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        best_model_name = None
        best_score = -1
        best_results = None

        for model_name, results in evaluation_results.items():
            score = results['metrics'].get(criteria, 0)

            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_results = results

        logger.info(f"Best model selected: {best_model_name} (criterion: {criteria}, score: {best_score:.4f})")

        return best_model_name, best_results

    def save_best_model(self, model_name: str) -> None:
        """
        Save the best model with a special filename.

        Args:
            model_name: Name of the best model
        """
        source_path = self.models_dir / f"{model_name}.joblib"
        target_path = self.models_dir / "best_model.joblib"

        # Copy the best model
        import shutil
        shutil.copy2(source_path, target_path)

        logger.info(f"Best model saved as best_model.joblib")

    def save_evaluation_results(self,
                               evaluation_results: Dict[str, Dict[str, Any]],
                               best_model_name: str,
                               best_model_results: Dict[str, Any]) -> str:
        """
        Save comprehensive evaluation results to JSON.

        Args:
            evaluation_results: Results for all models
            best_model_name: Name of the selected best model
            best_model_results: Detailed results for the best model

        Returns:
            Path to saved results file
        """
        # Create summary structure
        summary = {
            'evaluation_summary': {
                'best_model': best_model_name,
                'selection_criteria': 'f1_score',  # Default criteria
                'total_models_evaluated': len(evaluation_results),
                'model_comparison': {}
            },
            'detailed_results': evaluation_results,
            'business_insights': {}
        }

        # Add model comparison
        for model_name, results in evaluation_results.items():
            summary['evaluation_summary']['model_comparison'][model_name] = {
                'f1_score': results['metrics']['f1_score'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'accuracy': results['metrics']['accuracy'],
                'auc': results['metrics'].get('auc', 'N/A')
            }

        # Add business insights for the best model
        best_cm = best_model_results['confusion_matrix_analysis']['business_insights']
        summary['business_insights'] = {
            'selected_model': best_model_name,
            'missed_purchase_opportunities': best_cm['missed_purchases'],
            'wasted_marketing_spend': best_cm['false_alarms'],
            'correct_predictions': best_cm['correct_predictions'],
            'model_precision': best_model_results['metrics']['precision'],
            'model_recall': best_model_results['metrics']['recall'],
            'trade_off_analysis': self._analyze_tradeoffs(best_model_results)
        }

        # Save to JSON
        results_path = self.models_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Evaluation results saved to {results_path}")
        return str(results_path)

    def _analyze_tradeoffs(self, model_results: Dict[str, Any]) -> str:
        """
        Analyze precision vs recall tradeoffs for business context.

        Args:
            model_results: Results for a specific model

        Returns:
            Analysis string explaining the tradeoff
        """
        precision = model_results['metrics']['precision']
        recall = model_results['metrics']['recall']
        f1 = model_results['metrics']['f1_score']

        if precision > recall:
            analysis = ".2f"".2f"f"business context where minimizing marketing waste is more important than capturing every potential purchase."
        elif recall > precision:
            analysis = ".2f"".2f"f"business context where maximizing revenue opportunities is prioritized over precise targeting."
        else:
            analysis = ".2f"".2f"f"balanced approach suitable for most business scenarios."

        return analysis

    def run_evaluation_pipeline(self, selection_criteria: str = "f1_score") -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.

        Args:
            selection_criteria: Metric to use for model selection

        Returns:
            Dictionary with evaluation results and best model info
        """
        logger.info("Starting model evaluation pipeline...")

        # Evaluate all models
        evaluation_results = self.evaluate_all_models()

        # Select best model
        best_model_name, best_model_results = self.select_best_model(
            evaluation_results, criteria=selection_criteria
        )

        # Save best model
        self.save_best_model(best_model_name)

        # Save evaluation results
        results_path = self.save_evaluation_results(
            evaluation_results, best_model_name, best_model_results
        )

        logger.info("Evaluation pipeline completed successfully!")

        return {
            'evaluation_results': evaluation_results,
            'best_model_name': best_model_name,
            'best_model_results': best_model_results,
            'results_path': results_path
        }


def main():
    """Main function to run model evaluation."""
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation_pipeline()

    print("\nModel Evaluation Summary:")
    print("=" * 60)
    print(f"Best Model: {results['best_model_name']}")
    print("\nModel Performance Comparison:")

    # Load detailed results
    with open(results['results_path'], 'r') as f:
        eval_summary = json.load(f)

    for model_name, metrics in eval_summary['evaluation_summary']['model_comparison'].items():
        print(f"\n{model_name.upper()}:")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

    print(f"\nBusiness Insights for {results['best_model_name']}:")
    insights = eval_summary['business_insights']
    print(f"- Missed purchase opportunities: {insights['missed_purchase_opportunities']}")
    print(f"- Wasted marketing spend: {insights['wasted_marketing_spend']}")
    print(f"- Correct predictions: {insights['correct_predictions']}")
    print(f"\nTradeoff Analysis: {insights['trade_off_analysis']}")


if __name__ == "__main__":
    main()