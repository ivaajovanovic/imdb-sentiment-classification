import time
import copy
import pandas as pd
import numpy as np
import mlflow

from evaluation.metrics import ClassificationMetrics
from models.base import BaseModel
from vectorization.base import BaseVectorizer


class Experiment:
    
    def __init__(
        self,
        model: BaseModel,
        vectorizer: BaseVectorizer,
        run_name: str = "",
    ) -> None:
        self.model = model
        self.vectorizer = vectorizer
        self.run_name = run_name

        self._is_trained: bool = False
        self._retrained: bool = False
        self._train_metrics: ClassificationMetrics | None = None
        self._val_metrics: ClassificationMetrics | None = None


    def run(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        tags: dict | None = None,
    ) -> "Experiment":
       
        with mlflow.start_run(run_name=self.run_name or None):
            if tags:
                mlflow.set_tags(tags)

            mlflow.log_params(self._collect_params())

            X_train, X_val = self._fit_and_transform(df_train, df_val)
            self._train(X_train, df_train["sentiment"])
            self._evaluate_train_val(X_train, X_val, df_train, df_val)

        self._is_trained = True
        return self

    def retrain_on(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> "Experiment":
       
        df_combined = pd.concat([df_train, df_val], ignore_index=True)

        new_model = copy.deepcopy(self.model)
        new_vectorizer = copy.deepcopy(self.vectorizer)
        new_experiment = Experiment(
            model=new_model,
            vectorizer=new_vectorizer,
            run_name=self.run_name,
        )

        X_combined = new_experiment.vectorizer.fit_transform(df_combined)
        new_experiment._train(X_combined, df_combined["sentiment"])
        new_experiment._is_trained = True
        new_experiment._retrained = True

        return new_experiment

    def evaluate_on_test(
        self,
        df_test: pd.DataFrame,
        extra_metrics: dict | None = None,
        tags: dict | None = None,
    ) -> "Experiment":
        
        if not self._retrained:
            raise RuntimeError(
                "evaluate_on_test() requires the experiment to have been produced "
                "by retrain_on(). Call retrain_on(df_train, df_val) first."
            )

        with mlflow.start_run(run_name=(self.run_name or "") + "_final"):
            if tags:
                mlflow.set_tags(tags)

            mlflow.log_params(self._collect_params())

            if extra_metrics:
                mlflow.log_metrics(extra_metrics)

            X_test = self.vectorizer.transform(df_test)

            t0 = time.perf_counter()
            y_pred = self.model.predict(X_test)
            inference_time = time.perf_counter() - t0

            test_metrics = ClassificationMetrics(
                y_true=df_test["sentiment"],
                y_pred=y_pred,
                training_time=0.0,
                inference_time=inference_time,
                n_features=X_test.shape[1],
            )

            prefixed = {f"test_{k}": v for k, v in test_metrics.to_dict().items()}
            mlflow.log_metrics(prefixed)

        return self


    def _fit_and_transform(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
   
        X_train = self.vectorizer.fit_transform(df_train)
        X_val = self.vectorizer.transform(df_val)
        return X_train, X_val

    def _train(self, X_train, y_train: pd.Series) -> None:
       
        t0 = time.perf_counter()
        self.model.train(X_train, y_train)
        self._last_training_time = time.perf_counter() - t0

    def _evaluate_train_val(
        self,
        X_train,
        X_val,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> None:
       
        n_features = X_train.shape[1]

        t0 = time.perf_counter()
        y_pred_train = self.model.predict(X_train)
        inference_train = time.perf_counter() - t0

        self._train_metrics = ClassificationMetrics(
            y_true=df_train["sentiment"],
            y_pred=y_pred_train,
            training_time=self._last_training_time,
            inference_time=inference_train,
            n_features=n_features,
        )

        t0 = time.perf_counter()
        y_pred_val = self.model.predict(X_val)
        inference_val = time.perf_counter() - t0

        self._val_metrics = ClassificationMetrics(
            y_true=df_val["sentiment"],
            y_pred=y_pred_val,
            training_time=self._last_training_time,
            inference_time=inference_val,
            n_features=n_features,
        )

        # --- MLflow logging ---
        train_log = {f"train_{k}": v for k, v in self._train_metrics.to_dict().items()}
        val_log = {f"val_{k}": v for k, v in self._val_metrics.to_dict().items()}

        mlflow.log_metrics(train_log)
        mlflow.log_metrics(val_log)

    def _collect_params(self) -> dict:
      
        params: dict = {"model": type(self.model).__name__}

        vec = self.vectorizer
        if hasattr(vec, "vectorizer"):
            sklearn_vec = vec.vectorizer
            for attr in ("ngram_range", "min_df"):
                if hasattr(sklearn_vec, attr):
                    params[attr] = getattr(sklearn_vec, attr)

        model_params = self.model.get_params()
        params.update(model_params)

        return params

  
    @property
    def train_metrics(self) -> ClassificationMetrics | None:
        """Training-set metrics from the last ``run()`` call."""
        return self._train_metrics

    @property
    def val_metrics(self) -> ClassificationMetrics | None:
        """Validation-set metrics from the last ``run()`` call."""
        return self._val_metrics