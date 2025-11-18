"""
Example: Regression Task with Databricks and PySpark
This script demonstrates how to use the model training modules for regression
"""

# Setup - Import required modules
from pyspark.sql import SparkSession
from data_preprocessing import (
    DataQualityChecker,
    OutlierHandler,
    create_preprocessing_pipeline
)
from model_training import RegressionTrainer

# Initialize Spark Session (already available in Databricks as 'spark')
# spark = SparkSession.builder.appName("ML_Regression").getOrCreate()

# 1. LOAD DATA
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/path/to/your/regression_data.csv")

print(f"Dataset loaded: {df.count()} rows, {len(df.columns)} columns")

# 2. DATA QUALITY CHECK
checker = DataQualityChecker(df)
stats = checker.get_statistics()
stats.show()

missing_values = checker.check_missing_values()
missing_values.show()

# 3. HANDLE OUTLIERS
# Detect outliers
df = OutlierHandler.detect_outliers_iqr(df, 'target_variable', multiplier=1.5)

# Cap outliers
df = OutlierHandler.cap_outliers(df, 'feature1', multiplier=1.5)

# Or remove outliers (use with caution)
# df = OutlierHandler.remove_outliers(df, 'feature1', multiplier=1.5)

# 4. DEFINE FEATURES
numerical_features = ['feature1', 'feature2', 'feature3', 'feature4']
categorical_features = ['category1', 'category2']
label_col = 'target_variable'

# 5. CREATE PREPROCESSING PIPELINE
preprocessing_pipeline = create_preprocessing_pipeline(
    numerical_cols=numerical_features,
    categorical_cols=categorical_features,
    imputation_strategy='median',  # Use median for regression
    scaling_method='standard'
)

# 6. SPLIT DATA
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

print(f"\nTrain set: {train_df.count()} rows")
print(f"Test set: {test_df.count()} rows")

# 7. TRAIN REGRESSION MODELS
trainer = RegressionTrainer(
    features_col="scaled_features",
    label_col=label_col
)

# Train Linear Regression
lr_model, lr_results, lr_predictions = trainer.train_with_mlflow(
    model_type="linear",
    train_data=train_df,
    test_data=test_df,
    preprocessing_pipeline=preprocessing_pipeline,
    model_params={"maxIter": 100, "regParam": 0.1},
    experiment_name="/Users/your_username/regression_experiment",
    run_name="linear_regression"
)

# Train Random Forest Regressor
rf_model, rf_results, rf_predictions = trainer.train_with_mlflow(
    model_type="random_forest",
    train_data=train_df,
    test_data=test_df,
    preprocessing_pipeline=preprocessing_pipeline,
    model_params={"numTrees": 100, "maxDepth": 10},
    experiment_name="/Users/your_username/regression_experiment",
    run_name="random_forest_regression"
)

# Train Gradient Boosted Trees
gbt_model, gbt_results, gbt_predictions = trainer.train_with_mlflow(
    model_type="gbt",
    train_data=train_df,
    test_data=test_df,
    preprocessing_pipeline=preprocessing_pipeline,
    model_params={"maxIter": 100, "maxDepth": 5},
    experiment_name="/Users/your_username/regression_experiment",
    run_name="gbt_regression"
)

# 8. COMPARE RESULTS
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

print("\nLinear Regression:")
for metric, value in lr_results["test_metrics"].items():
    print(f"  {metric}: {value:.4f}")

print("\nRandom Forest Regression:")
for metric, value in rf_results["test_metrics"].items():
    print(f"  {metric}: {value:.4f}")

print("\nGradient Boosted Trees:")
for metric, value in gbt_results["test_metrics"].items():
    print(f"  {metric}: {value:.4f}")

# 9. SELECT BEST MODEL (by R² score)
models_results = {
    "Linear Regression": lr_results,
    "Random Forest": rf_results,
    "GBT": gbt_results
}

best_model_name = max(
    models_results.keys(),
    key=lambda x: models_results[x]["test_metrics"]["r2"]
)
best_r2 = models_results[best_model_name]["test_metrics"]["r2"]

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"Test R² Score: {best_r2:.4f}")
print(f"{'='*60}")

# 10. FEATURE IMPORTANCE (for tree-based models)
if best_model_name in ["Random Forest", "GBT"]:
    best_model = rf_model if best_model_name == "Random Forest" else gbt_model
    
    # Extract the model from pipeline
    model_stage = best_model.stages[-1]
    
    if hasattr(model_stage, 'featureImportances'):
        import pandas as pd
        
        feature_importance = model_stage.featureImportances.toArray()
        
        # Create feature names
        all_features = []
        for cat_col in categorical_features:
            all_features.append(f"{cat_col}_encoded")
        for num_col in numerical_features:
            all_features.append(num_col)
        
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(importance_df.head(10))

# 11. RESIDUAL ANALYSIS
# Calculate residuals
best_predictions = rf_predictions if best_model_name == "Random Forest" else gbt_predictions
residuals_df = best_predictions.select(
    label_col,
    "prediction"
).toPandas()

residuals_df['residual'] = residuals_df[label_col] - residuals_df['prediction']

import matplotlib.pyplot as plt
import seaborn as sns

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Predicted vs Actual
axes[0, 0].scatter(residuals_df['prediction'], residuals_df[label_col], alpha=0.5)
axes[0, 0].plot([residuals_df[label_col].min(), residuals_df[label_col].max()],
                [residuals_df[label_col].min(), residuals_df[label_col].max()],
                'r--', lw=2)
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_title('Predicted vs Actual')

# Residual plot
axes[0, 1].scatter(residuals_df['prediction'], residuals_df['residual'], alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')

# Residual distribution
axes[1, 0].hist(residuals_df['residual'], bins=50, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution')

# Q-Q plot
from scipy import stats
stats.probplot(residuals_df['residual'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# 12. SAVE BEST MODEL
best_model_to_save = rf_model if best_model_name == "Random Forest" else gbt_model
model_path = f"/dbfs/mnt/models/best_regression_model_{best_model_name.lower().replace(' ', '_')}"
best_model_to_save.write().overwrite().save(model_path)
print(f"\nModel saved to: {model_path}")

# 13. MAKE PREDICTIONS ON NEW DATA
# Example: Make predictions on a sample
sample_predictions = best_model_to_save.transform(test_df.limit(20))
sample_predictions.select(label_col, "prediction").show()

# 14. CLEANUP
train_df.unpersist()
test_df.unpersist()

print("\n✓ Regression pipeline completed successfully!")
