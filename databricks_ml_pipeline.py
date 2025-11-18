# Databricks notebook source
"""
Machine Learning Pipeline with PySpark in Databricks
This notebook demonstrates:
- Data loading and exploration
- Feature engineering
- Model training with multiple algorithms
- Model evaluation and comparison
- Model deployment preparation
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, mean, stddev, isnull
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# ML imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, 
    StandardScaler, 
    StringIndexer, 
    OneHotEncoder,
    Imputer
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# MLflow for experiment tracking
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Spark Session

# COMMAND ----------

# Spark session is already available in Databricks as 'spark'
# Configure for optimal performance
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

print(f"Spark Version: {spark.version}")
print(f"Databricks Runtime: {sc.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Loading

# COMMAND ----------

# Example 1: Load data from Databricks File System (DBFS)
# df = spark.read.format("csv") \
#     .option("header", "true") \
#     .option("inferSchema", "true") \
#     .load("/dbfs/mnt/data/your_data.csv")

# Example 2: Load from Delta Lake
# df = spark.read.format("delta").load("/mnt/delta/table_name")

# Example 3: Create sample dataset for demonstration
from pyspark.sql.functions import rand, randn

# Create sample classification dataset
sample_size = 10000
df = spark.range(sample_size) \
    .withColumn("feature1", randn(seed=42) * 10 + 50) \
    .withColumn("feature2", randn(seed=43) * 5 + 30) \
    .withColumn("feature3", rand(seed=44) * 100) \
    .withColumn("feature4", randn(seed=45) * 15 + 75) \
    .withColumn("category", (col("id") % 3).cast("string")) \
    .withColumn("label", when(col("feature1") + col("feature2") > 80, 1).otherwise(0))

print(f"Dataset size: {df.count()} rows, {len(df.columns)} columns")
df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Exploratory Data Analysis

# COMMAND ----------

# Display schema
print("Dataset Schema:")
df.printSchema()

# Summary statistics
print("\nSummary Statistics:")
df.describe().show()

# Check for missing values
print("\nMissing Values:")
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

# Class distribution for classification
print("\nClass Distribution:")
df.groupBy("label").count().show()

# COMMAND ----------

# Visualize feature distributions
pdf = df.select("feature1", "feature2", "feature3", "feature4", "label").toPandas()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
features = ["feature1", "feature2", "feature3", "feature4"]

for idx, feature in enumerate(features):
    row, col = idx // 2, idx % 2
    axes[row, col].hist(pdf[feature], bins=50, alpha=0.7, edgecolor='black')
    axes[row, col].set_title(f'Distribution of {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Preprocessing

# COMMAND ----------

class DataPreprocessor:
    """Handle data preprocessing for ML pipeline"""
    
    def __init__(self, categorical_cols=None, numerical_cols=None, label_col="label"):
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.label_col = label_col
        self.stages = []
    
    def build_pipeline(self):
        """Build preprocessing pipeline"""
        
        # Handle categorical features
        if self.categorical_cols:
            for cat_col in self.categorical_cols:
                # String indexing
                indexer = StringIndexer(
                    inputCol=cat_col, 
                    outputCol=f"{cat_col}_indexed",
                    handleInvalid="keep"
                )
                self.stages.append(indexer)
                
                # One-hot encoding
                encoder = OneHotEncoder(
                    inputCols=[f"{cat_col}_indexed"],
                    outputCols=[f"{cat_col}_encoded"]
                )
                self.stages.append(encoder)
        
        # Handle missing values in numerical features
        if self.numerical_cols:
            imputer = Imputer(
                inputCols=self.numerical_cols,
                outputCols=[f"{col}_imputed" for col in self.numerical_cols],
                strategy="mean"
            )
            self.stages.append(imputer)
        
        # Assemble feature vector
        feature_cols = []
        if self.categorical_cols:
            feature_cols.extend([f"{col}_encoded" for col in self.categorical_cols])
        if self.numerical_cols:
            feature_cols.extend([f"{col}_imputed" for col in self.numerical_cols])
        
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="assembled_features"
        )
        self.stages.append(assembler)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="assembled_features",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        self.stages.append(scaler)
        
        return Pipeline(stages=self.stages)

# COMMAND ----------

# Define feature columns
numerical_features = ["feature1", "feature2", "feature3", "feature4"]
categorical_features = ["category"]

# Create preprocessor
preprocessor = DataPreprocessor(
    categorical_cols=categorical_features,
    numerical_cols=numerical_features,
    label_col="label"
)

# Build preprocessing pipeline
preprocessing_pipeline = preprocessor.build_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train-Test Split

# COMMAND ----------

# Split data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training set size: {train_df.count()}")
print(f"Testing set size: {test_df.count()}")

# Cache datasets for better performance
train_df.cache()
test_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Training with MLflow

# COMMAND ----------

# Set up MLflow experiment
mlflow.set_experiment("/Users/your_username/ml_pipeline_experiment")

# COMMAND ----------

def train_classification_model(model, model_name, train_data, test_data, preprocessing_pipeline):
    """
    Train and evaluate a classification model with MLflow tracking
    """
    with mlflow.start_run(run_name=model_name):
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", train_data.count())
        mlflow.log_param("test_size", test_data.count())
        
        # Create full pipeline (preprocessing + model)
        pipeline = Pipeline(stages=preprocessing_pipeline.getStages() + [model])
        
        # Train model
        print(f"\nTraining {model_name}...")
        model_fitted = pipeline.fit(train_data)
        
        # Make predictions
        train_predictions = model_fitted.transform(train_data)
        test_predictions = model_fitted.transform(test_data)
        
        # Evaluate model
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        evaluator_accuracy = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        
        # Calculate metrics
        train_auc = evaluator_auc.evaluate(train_predictions)
        test_auc = evaluator_auc.evaluate(test_predictions)
        train_accuracy = evaluator_accuracy.evaluate(train_predictions)
        test_accuracy = evaluator_accuracy.evaluate(test_predictions)
        train_f1 = evaluator_f1.evaluate(train_predictions)
        test_f1 = evaluator_f1.evaluate(test_predictions)
        
        # Log metrics
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)
        
        # Log model
        mlflow.spark.log_model(model_fitted, "model")
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
        
        return model_fitted, test_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Logistic Regression

# COMMAND ----------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8
)

lr_model, lr_predictions = train_classification_model(
    lr, "Logistic Regression", train_df, test_df, preprocessing_pipeline
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Random Forest Classifier

# COMMAND ----------

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42
)

rf_model, rf_predictions = train_classification_model(
    rf, "Random Forest", train_df, test_df, preprocessing_pipeline
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Gradient Boosted Trees

# COMMAND ----------

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    maxDepth=5,
    seed=42
)

gbt_model, gbt_predictions = train_classification_model(
    gbt, "Gradient Boosted Trees", train_df, test_df, preprocessing_pipeline
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Hyperparameter Tuning with Cross-Validation

# COMMAND ----------

def tune_model_with_cv(model, param_grid, train_data, preprocessing_pipeline):
    """
    Perform hyperparameter tuning using cross-validation
    """
    with mlflow.start_run(run_name=f"{type(model).__name__}_CV"):
        
        # Create pipeline
        pipeline = Pipeline(stages=preprocessing_pipeline.getStages() + [model])
        
        # Set up evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        # Create cross-validator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=42
        )
        
        # Train with cross-validation
        print(f"\nPerforming cross-validation for {type(model).__name__}...")
        cv_model = cv.fit(train_data)
        
        # Get best model and parameters
        best_model = cv_model.bestModel
        
        # Log best parameters
        mlflow.log_param("cv_folds", 3)
        mlflow.log_metric("best_cv_score", max(cv_model.avgMetrics))
        
        print(f"\nBest CV Score: {max(cv_model.avgMetrics):.4f}")
        
        return cv_model

# COMMAND ----------

# Example: Tune Random Forest
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
    .build()

# Uncomment to run hyperparameter tuning (can be time-consuming)
# rf_cv_model = tune_model_with_cv(rf, rf_param_grid, train_df, preprocessing_pipeline)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Evaluation and Comparison

# COMMAND ----------

# Compare model predictions
print("Sample Predictions Comparison:")
comparison_df = test_df.select("label") \
    .withColumn("lr_prediction", lr_predictions.select("prediction").rdd.map(lambda x: x[0]).collect()) \
    .withColumn("rf_prediction", rf_predictions.select("prediction").rdd.map(lambda x: x[0]).collect()) \
    .withColumn("gbt_prediction", gbt_predictions.select("prediction").rdd.map(lambda x: x[0]).collect())

comparison_df.show(20)

# COMMAND ----------

# Confusion Matrix for best model (Random Forest)
from sklearn.metrics import confusion_matrix, classification_report

rf_pred_df = rf_predictions.select("label", "prediction").toPandas()

print("\nRandom Forest - Confusion Matrix:")
cm = confusion_matrix(rf_pred_df["label"], rf_pred_df["prediction"])
print(cm)

print("\nRandom Forest - Classification Report:")
print(classification_report(rf_pred_df["label"], rf_pred_df["prediction"]))

# COMMAND ----------

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Feature Importance Analysis

# COMMAND ----------

# Extract feature importances from Random Forest model
rf_best_model = rf_model.stages[-1]  # Get the RF model from pipeline

if hasattr(rf_best_model, 'featureImportances'):
    feature_importance = rf_best_model.featureImportances.toArray()
    
    # Create feature names
    all_features = []
    for cat_col in categorical_features:
        all_features.append(f"{cat_col}_encoded")
    for num_col in numerical_features:
        all_features.append(num_col)
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importance_df)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances - Random Forest')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Model Persistence and Registry

# COMMAND ----------

# Save best model to DBFS
best_model_path = "/dbfs/mnt/models/best_classification_model"

# Save model
rf_model.write().overwrite().save(best_model_path)
print(f"Model saved to: {best_model_path}")

# Register model with MLflow Model Registry
model_name = "classification_model_prod"

with mlflow.start_run(run_name="model_registration"):
    mlflow.spark.log_model(
        rf_model,
        "model",
        registered_model_name=model_name
    )

print(f"\nModel registered as: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Batch Prediction

# COMMAND ----------

def batch_predict(model, data):
    """
    Perform batch predictions on new data
    """
    predictions = model.transform(data)
    return predictions.select("features", "prediction", "probability")

# Example batch prediction
new_predictions = batch_predict(rf_model, test_df.limit(100))
new_predictions.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Model Deployment Preparation

# COMMAND ----------

# Create a UDF for real-time predictions
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

def create_prediction_udf(model_path):
    """
    Create a UDF for model predictions
    """
    from pyspark.ml import PipelineModel
    
    # Load model
    loaded_model = PipelineModel.load(model_path)
    
    def predict(features):
        # Transform features and return prediction
        # This is a simplified version
        return float(loaded_model.transform(features).select("prediction").first()[0])
    
    return udf(predict, DoubleType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Model Monitoring and Metrics Dashboard

# COMMAND ----------

# Calculate and display model performance metrics over time
def display_model_metrics():
    """
    Display comprehensive model metrics
    """
    metrics = {
        "Model": ["Logistic Regression", "Random Forest", "GBT"],
        "Test AUC": [
            BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(lr_predictions),
            BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(rf_predictions),
            BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(gbt_predictions)
        ],
        "Test Accuracy": [
            MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(lr_predictions),
            MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(rf_predictions),
            MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(gbt_predictions)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics)
    print("\nModel Comparison:")
    print(metrics_df)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].bar(metrics_df["Model"], metrics_df["Test AUC"])
    axes[0].set_title("Model Comparison - AUC")
    axes[0].set_ylabel("AUC Score")
    axes[0].set_ylim([0.5, 1.0])
    
    axes[1].bar(metrics_df["Model"], metrics_df["Test Accuracy"])
    axes[1].set_title("Model Comparison - Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.show()

display_model_metrics()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Cleanup

# COMMAND ----------

# Unpersist cached DataFrames
train_df.unpersist()
test_df.unpersist()

print("Cleanup completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook demonstrated:
# MAGIC - Data loading and exploration in Databricks
# MAGIC - Feature engineering with PySpark ML
# MAGIC - Training multiple classification models
# MAGIC - Model evaluation and comparison
# MAGIC - Hyperparameter tuning with cross-validation
# MAGIC - MLflow integration for experiment tracking
# MAGIC - Model persistence and registry
# MAGIC - Batch prediction capabilities
# MAGIC - Feature importance analysis
# MAGIC 
# MAGIC Next steps:
# MAGIC - Deploy model to production using Databricks Model Serving
# MAGIC - Set up automated retraining pipelines
# MAGIC - Implement A/B testing for model versions
# MAGIC - Monitor model performance in production
