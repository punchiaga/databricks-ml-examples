"""
Example: Classification Task with Databricks and PySpark
This script demonstrates how to use the model training modules
"""

# Setup - Import required modules
from pyspark.sql import SparkSession
from data_preprocessing import (
    DataQualityChecker,
    FeatureEngineer,
    PipelineBuilder,
    create_preprocessing_pipeline
)
from model_training import (
    ClassificationTrainer,
    HyperparameterTuner,
    ModelComparer
)

# Initialize Spark Session (already available in Databricks as 'spark')
# spark = SparkSession.builder.appName("ML_Classification").getOrCreate()

# 1. LOAD DATA
# Load your data from various sources
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/path/to/your/data.csv")

# Or use Delta Lake
# df = spark.read.format("delta").load("/mnt/delta/table_name")

# Or from a table
# df = spark.table("database.table_name")

print(f"Dataset loaded: {df.count()} rows, {len(df.columns)} columns")

# 2. DATA QUALITY CHECK
checker = DataQualityChecker(df)
quality_report = checker.generate_quality_report(categorical_cols=['category_col'])

print("\nData Quality Report:")
print(f"Total Rows: {quality_report['total_rows']}")
print(f"Total Columns: {quality_report['total_columns']}")
print(f"Duplicates: {quality_report['duplicates']}")
print(f"Missing Values: {quality_report['missing_values']}")

# 3. FEATURE ENGINEERING (Optional)
# Create interaction features
df = FeatureEngineer.create_interaction_features(
    df, 'feature1', 'feature2', operation='multiply'
)

# Create polynomial features
df = FeatureEngineer.create_polynomial_features(
    df, 'feature1', degree=2
)

# Create binned features
df = FeatureEngineer.create_binned_features(
    df, 'feature1', num_bins=5, method='quantile'
)

# 4. DEFINE FEATURES
numerical_features = ['feature1', 'feature2', 'feature3']
categorical_features = ['category_col']
label_col = 'label'

# 5. CREATE PREPROCESSING PIPELINE
preprocessing_pipeline = create_preprocessing_pipeline(
    numerical_cols=numerical_features,
    categorical_cols=categorical_features,
    imputation_strategy='mean',
    scaling_method='standard'
)

# 6. SPLIT DATA
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

print(f"\nTrain set: {train_df.count()} rows")
print(f"Test set: {test_df.count()} rows")

# 7. TRAIN MULTIPLE MODELS
trainer = ClassificationTrainer(
    features_col="scaled_features",  # Output from preprocessing pipeline
    label_col=label_col
)

# Define model configurations
model_configs = {
    "logistic": {
        "maxIter": 100,
        "regParam": 0.01
    },
    "random_forest": {
        "numTrees": 100,
        "maxDepth": 10
    },
    "gbt": {
        "maxIter": 100,
        "maxDepth": 5
    }
}

# Train and compare models
results = trainer.train_multiple_models(
    model_types=["logistic", "random_forest", "gbt"],
    train_data=train_df,
    test_data=test_df,
    preprocessing_pipeline=preprocessing_pipeline,
    model_configs=model_configs,
    experiment_name="/Users/your_username/classification_experiment",
    is_binary=True
)

# 8. COMPARE RESULTS
ModelComparer.compare_results(results)
best_model_name, best_score = ModelComparer.get_best_model(results, metric="areaUnderROC")
print(f"\nBest Model: {best_model_name} with AUC = {best_score:.4f}")

# 9. HYPERPARAMETER TUNING (Optional)
# Tune the best model further
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

tuner = HyperparameterTuner()

rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol=label_col
)

# Define parameter grid
param_configs = {
    "numTrees": [100, 200, 300],
    "maxDepth": [10, 15, 20],
    "minInstancesPerNode": [1, 5]
}
param_grid = tuner.create_param_grid(rf, param_configs)

# Create evaluator
evaluator = BinaryClassificationEvaluator(
    labelCol=label_col,
    metricName="areaUnderROC"
)

# Perform tuning
cv_model = tuner.tune_with_cross_validation(
    model=rf,
    param_grid=param_grid,
    train_data=train_df,
    evaluator=evaluator,
    num_folds=3,
    preprocessing_pipeline=preprocessing_pipeline,
    experiment_name="/Users/your_username/classification_experiment"
)

# 10. EVALUATE BEST MODEL
best_model = cv_model.bestModel
test_predictions = best_model.transform(test_df)

final_auc = evaluator.evaluate(test_predictions)
print(f"\nFinal Model Test AUC: {final_auc:.4f}")

# 11. SAVE MODEL
model_path = "/dbfs/mnt/models/best_classification_model"
best_model.write().overwrite().save(model_path)
print(f"\nModel saved to: {model_path}")

# 12. BATCH PREDICTIONS
# Make predictions on new data
new_predictions = best_model.transform(test_df.limit(100))
new_predictions.select("label", "prediction", "probability").show(10)

# 13. CLEANUP
train_df.unpersist()
test_df.unpersist()

print("\n✓ Pipeline completed successfully!")
