# Machine Learning in Databricks with PySpark

This project provides a comprehensive framework for building machine learning models in Databricks using Python and PySpark. It includes end-to-end ML pipeline implementation with data preprocessing, model training, evaluation, and deployment capabilities.

## 📁 Project Structure

```
LLMOPS/
├── databricks_ml_pipeline.py    # Main Databricks notebook with complete ML workflow
├── data_preprocessing.py         # Data preprocessing utilities and transformations
├── model_training.py            # Model training, evaluation, and hyperparameter tuning
└── README.md                    # This file
```

## 🚀 Features

### 1. **Data Preprocessing** (`data_preprocessing.py`)
- **Data Quality Checker**: Identify missing values, duplicates, and data quality issues
- **Feature Engineering**: Create interaction, polynomial, and binned features
- **Outlier Handling**: Detect and handle outliers using IQR method
- **Data Balancing**: Handle imbalanced datasets with oversampling/undersampling
- **Pipeline Builder**: Build complete preprocessing pipelines with:
  - Imputation strategies
  - String indexing and one-hot encoding
  - Feature scaling (StandardScaler, MinMaxScaler)
  - Vector assembly

### 2. **Model Training** (`model_training.py`)
- **Classification Models**:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosted Trees
  - Decision Tree Classifier
  - Naive Bayes
  - Linear SVM

- **Regression Models**:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosted Trees Regressor
  - Decision Tree Regressor

- **Hyperparameter Tuning**:
  - Cross-Validation
  - Train-Validation Split
  - Automated parameter grid search

- **MLflow Integration**:
  - Experiment tracking
  - Model versioning
  - Parameter and metric logging
  - Model registry

### 3. **Complete ML Pipeline** (`databricks_ml_pipeline.py`)
A full Databricks notebook demonstrating:
- Data loading from multiple sources (CSV, Delta Lake, DBFS)
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Model training with multiple algorithms
- Model evaluation and comparison
- Feature importance analysis
- Model persistence and deployment
- Batch prediction capabilities

## 📋 Prerequisites

### Databricks Environment
- Databricks Runtime 11.0+ (with ML runtime recommended)
- Cluster with appropriate compute resources

### Python Packages
The following packages are pre-installed in Databricks ML Runtime:
- `pyspark`
- `mlflow`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## 🔧 Installation & Setup

### 1. Import to Databricks

**Option A: Direct Import**
1. Open your Databricks workspace
2. Navigate to Workspace → Users → [Your Username]
3. Click "Import"
4. Upload `databricks_ml_pipeline.py`
5. The file will be imported as a Databricks notebook

**Option B: Git Integration**
1. Set up Repos in Databricks
2. Clone this repository
3. The notebook will be available in your Repos folder

### 2. Upload Support Modules
Upload `data_preprocessing.py` and `model_training.py` to:
- DBFS: `/FileStore/scripts/` or
- Workspace: Create a folder and upload

### 3. Configure Notebook
In the notebook, update the import paths if needed:
```python
# If using DBFS
import sys
sys.path.append('/dbfs/FileStore/scripts/')

from data_preprocessing import *
from model_training import *
```

## 💡 Usage Examples

### Quick Start with the Main Notebook

1. **Open the notebook** in Databricks
2. **Attach to a cluster** (ML Runtime recommended)
3. **Run all cells** to see the complete pipeline in action

The notebook includes sample data generation, so you can run it immediately without external data.

### Using Individual Modules

#### Data Preprocessing

```python
from data_preprocessing import DataQualityChecker, FeatureEngineer, PipelineBuilder

# Check data quality
checker = DataQualityChecker(df)
quality_report = checker.generate_quality_report(categorical_cols=['category'])
print(quality_report)

# Create interaction features
df_engineered = FeatureEngineer.create_interaction_features(
    df, 'feature1', 'feature2', operation='multiply'
)

# Build preprocessing pipeline
builder = PipelineBuilder()
pipeline = builder \
    .add_imputer(['col1', 'col2'], strategy='mean') \
    .add_string_indexer(['category']) \
    .add_one_hot_encoder(['category']) \
    .add_vector_assembler(['col1', 'col2', 'category_encoded']) \
    .add_standard_scaler() \
    .build()

# Fit and transform
model = pipeline.fit(train_df)
transformed_df = model.transform(train_df)
```

#### Classification Training

```python
from model_training import ClassificationTrainer

# Initialize trainer
trainer = ClassificationTrainer(
    features_col="features",
    label_col="label"
)

# Train single model
model, results, predictions = trainer.train_with_mlflow(
    model_type="random_forest",
    train_data=train_df,
    test_data=test_df,
    model_params={"numTrees": 100, "maxDepth": 10},
    experiment_name="/Users/your_username/ml_experiment",
    is_binary=True
)

# Train multiple models and compare
results = trainer.train_multiple_models(
    model_types=["logistic", "random_forest", "gbt"],
    train_data=train_df,
    test_data=test_df,
    experiment_name="/Users/your_username/ml_experiment"
)
```

#### Hyperparameter Tuning

```python
from model_training import HyperparameterTuner
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize tuner
tuner = HyperparameterTuner()

# Create model
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Define parameter grid
param_configs = {
    "numTrees": [50, 100, 200],
    "maxDepth": [5, 10, 15],
    "minInstancesPerNode": [1, 5, 10]
}
param_grid = tuner.create_param_grid(rf, param_configs)

# Create evaluator
evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

# Perform cross-validation
cv_model = tuner.tune_with_cross_validation(
    model=rf,
    param_grid=param_grid,
    train_data=train_df,
    evaluator=evaluator,
    num_folds=3
)

# Get best model
best_model = cv_model.bestModel
```

## 📊 Working with Your Own Data

### Loading Data from Various Sources

#### CSV from DBFS
```python
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/dbfs/mnt/data/your_data.csv")
```

#### Delta Lake
```python
df = spark.read.format("delta").load("/mnt/delta/table_name")
```

#### Azure Blob Storage
```python
df = spark.read.format("csv") \
    .option("header", "true") \
    .load("wasbs://container@account.blob.core.windows.net/path/file.csv")
```

#### AWS S3
```python
df = spark.read.format("csv") \
    .option("header", "true") \
    .load("s3a://bucket-name/path/file.csv")
```

#### Databricks Tables
```python
df = spark.table("database.table_name")
```

## 🔍 MLflow Experiment Tracking

### View Experiments
1. In Databricks, click on "Experiments" in the left sidebar
2. Find your experiment (e.g., `/Users/your_username/ml_experiment`)
3. View all runs, compare metrics, and analyze results

### Register Models
```python
import mlflow

# Log and register model
with mlflow.start_run():
    mlflow.spark.log_model(
        model,
        "model",
        registered_model_name="my_production_model"
    )
```

### Load Registered Models
```python
# Load specific version
model_uri = "models:/my_production_model/1"
loaded_model = mlflow.spark.load_model(model_uri)

# Load latest production version
model_uri = "models:/my_production_model/Production"
loaded_model = mlflow.spark.load_model(model_uri)
```

## 🎯 Model Evaluation Metrics

### Classification Metrics
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Weighted precision across classes
- **Recall**: Weighted recall across classes

### Regression Metrics
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

## 🚀 Model Deployment

### Save Model to DBFS
```python
model_path = "/dbfs/mnt/models/my_model"
fitted_model.write().overwrite().save(model_path)
```

### Batch Predictions
```python
# Load saved model
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load(model_path)

# Make predictions
predictions = loaded_model.transform(new_data)
predictions.select("features", "prediction", "probability").show()
```

### Real-time Serving
1. Register model in MLflow Model Registry
2. Use Databricks Model Serving to deploy:
   - Go to "Models" → Select your model
   - Click "Serve Model"
   - Configure endpoint settings
3. Access via REST API

## 📈 Best Practices

### 1. Data Preparation
- Always check data quality before training
- Handle missing values appropriately
- Scale features for better model performance
- Create meaningful feature interactions

### 2. Model Training
- Start with simple models (Logistic Regression, Decision Trees)
- Use cross-validation for robust evaluation
- Track all experiments with MLflow
- Compare multiple models before choosing

### 3. Performance Optimization
- Cache DataFrames when reusing them multiple times
- Use appropriate cluster size for your data
- Leverage Databricks Auto-scaling
- Partition data appropriately

### 4. Production Deployment
- Version your models properly
- Monitor model performance over time
- Implement A/B testing for new models
- Set up automated retraining pipelines

## 🔧 Troubleshooting

### Common Issues

**Issue**: Import errors for support modules
**Solution**: Ensure modules are uploaded to DBFS or Workspace and paths are correctly configured

**Issue**: Out of memory errors
**Solution**: 
- Increase cluster size
- Use sampling for initial development
- Optimize DataFrame operations with `.cache()` and `.persist()`

**Issue**: Slow training times
**Solution**:
- Reduce cross-validation folds
- Use TrainValidationSplit instead of CrossValidator
- Reduce parameter grid size
- Optimize Spark configurations

## 📚 Additional Resources

- [Databricks ML Documentation](https://docs.databricks.com/machine-learning/index.html)
- [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Databricks Academy](https://academy.databricks.com/)

## 🤝 Contributing

Feel free to extend this framework with:
- Additional feature engineering methods
- New model types
- Custom evaluation metrics
- Advanced deployment strategies

## 📝 License

This project is provided as-is for educational and commercial use.

## 👤 Author

Created for machine learning practitioners working with Databricks and PySpark.

---

**Happy Machine Learning! 🎉**

For questions or issues, please refer to the Databricks documentation or community forums.
