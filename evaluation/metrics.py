import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ClassificationMetrics:
    """
    Computes and stores classification metrics for a single model run.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    training_time : float, optional
        Wall-clock seconds spent training the model.
    inference_time : float, optional
        Wall-clock seconds spent predicting on the evaluation set.
    n_features : int, optional
        Dimensionality of the feature matrix used for training.
    """

    def __init__(
        self,
        y_true,
        y_pred,
        training_time: float = 0.0,
        inference_time: float = 0.0,
        n_features: int = 0,
    ):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.training_time = training_time
        self.inference_time = inference_time
        self.n_features = n_features

        self._compute()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute(self):
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision = precision_score(
            self.y_true, self.y_pred, pos_label="positive", zero_division=0
        )
        self.recall = recall_score(
            self.y_true, self.y_pred, pos_label="positive", zero_division=0
        )
        self.f1 = f1_score(
            self.y_true, self.y_pred, pos_label="positive", zero_division=0
        )
        self.confusion_matrix = confusion_matrix(
            self.y_true, self.y_pred, labels=["negative", "positive"]
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return all metrics as a flat dictionary (suitable for MLflow logging)."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "training_time_s": self.training_time,
            "inference_time_s": self.inference_time,
            "n_features": self.n_features,
        }

    def report(self) -> str:
        """Return a human-readable classification report."""
        return classification_report(
            self.y_true,
            self.y_pred,
            target_names=["negative", "positive"],
            zero_division=0,
        )

    def __repr__(self) -> str:
        return (
            f"ClassificationMetrics("
            f"acc={self.accuracy:.4f}, "
            f"prec={self.precision:.4f}, "
            f"rec={self.recall:.4f}, "
            f"f1={self.f1:.4f})"
        )
