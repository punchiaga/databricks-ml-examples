"""
Model Training Module for PySpark ML
Supports Classification and Regression tasks
"""

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    DecisionTreeClassifier,
    NaiveBayes,
    LinearSVC
)
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
    DecisionTreeRegressor
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame
import mlflow
import mlflow.spark
from typing import Dict, List, Tuple, Optional, Any
import time


class ModelTrainer:
    """Base class for training ML models"""
    
    def __init__(
        self,
        features_col: str = "features",
        label_col: str = "label",
        prediction_col: str = "prediction"
    ):
        self.features_col = features_col
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.models = {}
        self.results = {}
    
    def train(
        self,
        model,
        train_data: DataFrame,
        preprocessing_pipeline: Optional[Pipeline] = None
    ) -> PipelineModel:
        """
        Train a model with optional preprocessing
        """
        if preprocessing_pipeline:
            # Combine preprocessing and model into single pipeline
            stages = preprocessing_pipeline.getStages() + [model]
            pipeline = Pipeline(stages=stages)
        else:
            pipeline = Pipeline(stages=[model])
        
        start_time = time.time()
        fitted_model = pipeline.fit(train_data)
        training_time = time.time() - start_time
        
        return fitted_model, training_time
    
    def evaluate(
        self,
        model: PipelineModel,
        test_data: DataFrame,
        evaluators: List
    ) -> Dict[str, float]:
        """
        Evaluate model on test data
        """
        predictions = model.transform(test_data)
        metrics = {}
        
        for evaluator in evaluators:
            metric_name = evaluator.getMetricName()
            metrics[metric_name] = evaluator.evaluate(predictions)
        
        return metrics, predictions


class ClassificationTrainer(ModelTrainer):
    """Trainer for classification tasks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type = "classification"
    
    def get_model(self, model_type: str, **params) -> Any:
        """
        Get a classification model by type
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'gbt', 'decision_tree', 'naive_bayes', 'svm')
            **params: Model-specific parameters
        """
        model_params = {
            "featuresCol": self.features_col,
            "labelCol": self.label_col,
            "predictionCol": self.prediction_col
        }
        model_params.update(params)
        
        if model_type == "logistic":
            return LogisticRegression(**model_params)
        elif model_type == "random_forest":
            return RandomForestClassifier(**model_params)
        elif model_type == "gbt":
            return GBTClassifier(**model_params)
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(**model_params)
        elif model_type == "naive_bayes":
            return NaiveBayes(**model_params)
        elif model_type == "svm":
            return LinearSVC(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_evaluators(self, is_binary: bool = True) -> List:
        """
        Get evaluators for classification
        """
        evaluators = []
        
        if is_binary:
            evaluators.append(
                BinaryClassificationEvaluator(
                    labelCol=self.label_col,
                    rawPredictionCol="rawPrediction",
                    metricName="areaUnderROC"
                )
            )
            evaluators.append(
                BinaryClassificationEvaluator(
                    labelCol=self.label_col,
                    rawPredictionCol="rawPrediction",
                    metricName="areaUnderPR"
                )
            )
        
        evaluators.extend([
            MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="accuracy"
            ),
            MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="f1"
            ),
            MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="weightedPrecision"
            ),
            MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="weightedRecall"
            )
        ])
        
        return evaluators
    
    def train_with_mlflow(
        self,
        model_type: str,
        train_data: DataFrame,
        test_data: DataFrame,
        preprocessing_pipeline: Optional[Pipeline] = None,
        model_params: Dict = None,
        experiment_name: str = None,
        run_name: str = None,
        is_binary: bool = True
    ) -> Tuple[PipelineModel, Dict, DataFrame]:
        """
        Train classification model with MLflow tracking
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        if model_params is None:
            model_params = {}
        
        with mlflow.start_run(run_name=run_name or f"{model_type}_classification"):
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("train_size", train_data.count())
            mlflow.log_param("test_size", test_data.count())
            mlflow.log_params(model_params)
            
            # Get model
            model = self.get_model(model_type, **model_params)
            
            # Train model
            fitted_model, training_time = self.train(
                model, train_data, preprocessing_pipeline
            )
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Evaluate on train and test
            evaluators = self.get_evaluators(is_binary=is_binary)
            
            train_metrics, train_predictions = self.evaluate(
                fitted_model, train_data, evaluators
            )
            test_metrics, test_predictions = self.evaluate(
                fitted_model, test_data, evaluators
            )
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log model
            mlflow.spark.log_model(fitted_model, "model")
            
            # Store results
            results = {
                "model_type": model_type,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "training_time": training_time
            }
            
            self.models[model_type] = fitted_model
            self.results[model_type] = results
            
            return fitted_model, results, test_predictions
    
    def train_multiple_models(
        self,
        model_types: List[str],
        train_data: DataFrame,
        test_data: DataFrame,
        preprocessing_pipeline: Optional[Pipeline] = None,
        model_configs: Dict[str, Dict] = None,
        experiment_name: str = None,
        is_binary: bool = True
    ) -> Dict:
        """
        Train multiple classification models and compare
        """
        if model_configs is None:
            model_configs = {model_type: {} for model_type in model_types}
        
        all_results = {}
        
        for model_type in model_types:
            print(f"\nTraining {model_type}...")
            
            model_params = model_configs.get(model_type, {})
            
            _, results, _ = self.train_with_mlflow(
                model_type=model_type,
                train_data=train_data,
                test_data=test_data,
                preprocessing_pipeline=preprocessing_pipeline,
                model_params=model_params,
                experiment_name=experiment_name,
                run_name=f"{model_type}_classification",
                is_binary=is_binary
            )
            
            all_results[model_type] = results
            
            print(f"{model_type} - Test Metrics:")
            for metric, value in results["test_metrics"].items():
                print(f"  {metric}: {value:.4f}")
        
        return all_results


class RegressionTrainer(ModelTrainer):
    """Trainer for regression tasks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type = "regression"
    
    def get_model(self, model_type: str, **params) -> Any:
        """
        Get a regression model by type
        
        Args:
            model_type: Type of model ('linear', 'random_forest', 'gbt', 'decision_tree')
            **params: Model-specific parameters
        """
        model_params = {
            "featuresCol": self.features_col,
            "labelCol": self.label_col,
            "predictionCol": self.prediction_col
        }
        model_params.update(params)
        
        if model_type == "linear":
            return LinearRegression(**model_params)
        elif model_type == "random_forest":
            return RandomForestRegressor(**model_params)
        elif model_type == "gbt":
            return GBTRegressor(**model_params)
        elif model_type == "decision_tree":
            return DecisionTreeRegressor(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_evaluators(self) -> List:
        """
        Get evaluators for regression
        """
        return [
            RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="rmse"
            ),
            RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="mse"
            ),
            RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="mae"
            ),
            RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="r2"
            )
        ]
    
    def train_with_mlflow(
        self,
        model_type: str,
        train_data: DataFrame,
        test_data: DataFrame,
        preprocessing_pipeline: Optional[Pipeline] = None,
        model_params: Dict = None,
        experiment_name: str = None,
        run_name: str = None
    ) -> Tuple[PipelineModel, Dict, DataFrame]:
        """
        Train regression model with MLflow tracking
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        if model_params is None:
            model_params = {}
        
        with mlflow.start_run(run_name=run_name or f"{model_type}_regression"):
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("train_size", train_data.count())
            mlflow.log_param("test_size", test_data.count())
            mlflow.log_params(model_params)
            
            # Get model
            model = self.get_model(model_type, **model_params)
            
            # Train model
            fitted_model, training_time = self.train(
                model, train_data, preprocessing_pipeline
            )
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Evaluate on train and test
            evaluators = self.get_evaluators()
            
            train_metrics, train_predictions = self.evaluate(
                fitted_model, train_data, evaluators
            )
            test_metrics, test_predictions = self.evaluate(
                fitted_model, test_data, evaluators
            )
            
            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log model
            mlflow.spark.log_model(fitted_model, "model")
            
            # Store results
            results = {
                "model_type": model_type,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "training_time": training_time
            }
            
            self.models[model_type] = fitted_model
            self.results[model_type] = results
            
            return fitted_model, results, test_predictions


class HyperparameterTuner:
    """Hyperparameter tuning with cross-validation"""
    
    def __init__(
        self,
        features_col: str = "features",
        label_col: str = "label"
    ):
        self.features_col = features_col
        self.label_col = label_col
    
    def create_param_grid(
        self,
        model,
        param_configs: Dict[str, List]
    ) -> List:
        """
        Create parameter grid for tuning
        
        Args:
            model: The model to tune
            param_configs: Dictionary mapping parameter names to lists of values
                          Example: {"maxDepth": [5, 10], "numTrees": [50, 100]}
        """
        builder = ParamGridBuilder()
        
        for param_name, values in param_configs.items():
            param = getattr(model, param_name)
            builder = builder.addGrid(param, values)
        
        return builder.build()
    
    def tune_with_cross_validation(
        self,
        model,
        param_grid: List,
        train_data: DataFrame,
        evaluator,
        num_folds: int = 3,
        preprocessing_pipeline: Optional[Pipeline] = None,
        experiment_name: str = None
    ) -> CrossValidator:
        """
        Perform hyperparameter tuning with cross-validation
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{type(model).__name__}_cv_tuning"):
            
            # Create pipeline
            if preprocessing_pipeline:
                stages = preprocessing_pipeline.getStages() + [model]
                pipeline = Pipeline(stages=stages)
            else:
                pipeline = Pipeline(stages=[model])
            
            # Create CrossValidator
            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=num_folds,
                seed=42,
                parallelism=2
            )
            
            # Log parameters
            mlflow.log_param("num_folds", num_folds)
            mlflow.log_param("num_param_combinations", len(param_grid))
            
            # Fit cross-validator
            print(f"Starting cross-validation with {len(param_grid)} parameter combinations...")
            start_time = time.time()
            cv_model = cv.fit(train_data)
            cv_time = time.time() - start_time
            
            # Log results
            mlflow.log_metric("cv_time_seconds", cv_time)
            mlflow.log_metric("best_cv_score", float(max(cv_model.avgMetrics)))
            
            # Log best parameters
            best_model = cv_model.bestModel
            if hasattr(best_model.stages[-1], "extractParamMap"):
                best_params = best_model.stages[-1].extractParamMap()
                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param.name}", value)
            
            # Log model
            mlflow.spark.log_model(cv_model.bestModel, "best_model")
            
            print(f"\nCross-validation completed in {cv_time:.2f} seconds")
            print(f"Best CV Score: {max(cv_model.avgMetrics):.4f}")
            
            return cv_model
    
    def tune_with_train_validation_split(
        self,
        model,
        param_grid: List,
        train_data: DataFrame,
        evaluator,
        train_ratio: float = 0.8,
        preprocessing_pipeline: Optional[Pipeline] = None,
        experiment_name: str = None
    ) -> TrainValidationSplit:
        """
        Perform hyperparameter tuning with train-validation split
        Faster than cross-validation but less robust
        """
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{type(model).__name__}_tvs_tuning"):
            
            # Create pipeline
            if preprocessing_pipeline:
                stages = preprocessing_pipeline.getStages() + [model]
                pipeline = Pipeline(stages=stages)
            else:
                pipeline = Pipeline(stages=[model])
            
            # Create TrainValidationSplit
            tvs = TrainValidationSplit(
                estimator=pipeline,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                trainRatio=train_ratio,
                seed=42,
                parallelism=2
            )
            
            # Log parameters
            mlflow.log_param("train_ratio", train_ratio)
            mlflow.log_param("num_param_combinations", len(param_grid))
            
            # Fit
            print(f"Starting train-validation split with {len(param_grid)} parameter combinations...")
            start_time = time.time()
            tvs_model = tvs.fit(train_data)
            tvs_time = time.time() - start_time
            
            # Log results
            mlflow.log_metric("tvs_time_seconds", tvs_time)
            mlflow.log_metric("best_validation_score", float(max(tvs_model.validationMetrics)))
            
            # Log model
            mlflow.spark.log_model(tvs_model.bestModel, "best_model")
            
            print(f"\nTrain-validation split completed in {tvs_time:.2f} seconds")
            print(f"Best Validation Score: {max(tvs_model.validationMetrics):.4f}")
            
            return tvs_model


class ModelComparer:
    """Compare multiple trained models"""
    
    @staticmethod
    def compare_results(results: Dict[str, Dict]) -> None:
        """
        Print comparison of model results
        """
        import pandas as pd
        
        # Extract test metrics
        comparison_data = []
        for model_name, model_results in results.items():
            row = {"Model": model_name}
            row.update(model_results["test_metrics"])
            row["Training Time (s)"] = model_results["training_time"]
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print("\nModel Comparison:")
        print(df.to_string(index=False))
    
    @staticmethod
    def get_best_model(results: Dict[str, Dict], metric: str = "areaUnderROC") -> str:
        """
        Get the best model based on a specific metric
        """
        best_model = None
        best_score = -float('inf')
        
        for model_name, model_results in results.items():
            score = model_results["test_metrics"].get(metric)
            if score and score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score
