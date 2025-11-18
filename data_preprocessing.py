"""
Data Preprocessing Utilities for PySpark ML Pipeline
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, count, isnull, isnan, 
    mean, stddev, percentile_approx, lit
)
from pyspark.ml.feature import (
    VectorAssembler, 
    StandardScaler, 
    MinMaxScaler,
    StringIndexer, 
    OneHotEncoder,
    Imputer,
    Bucketizer,
    QuantileDiscretizer
)
from pyspark.ml import Pipeline
from typing import List, Dict, Optional


class DataQualityChecker:
    """Check data quality and generate reports"""
    
    def __init__(self, df: DataFrame):
        self.df = df
        
    def check_missing_values(self) -> DataFrame:
        """
        Check for missing values in all columns
        """
        missing_counts = self.df.select([
            count(when(isnull(c) | isnan(c), c)).alias(c) 
            for c in self.df.columns
        ])
        return missing_counts
    
    def get_statistics(self) -> DataFrame:
        """
        Get summary statistics for numerical columns
        """
        return self.df.describe()
    
    def check_duplicates(self) -> int:
        """
        Check for duplicate rows
        """
        total_rows = self.df.count()
        distinct_rows = self.df.distinct().count()
        duplicates = total_rows - distinct_rows
        return duplicates
    
    def check_cardinality(self, categorical_cols: List[str]) -> Dict[str, int]:
        """
        Check cardinality of categorical columns
        """
        cardinality = {}
        for col_name in categorical_cols:
            cardinality[col_name] = self.df.select(col_name).distinct().count()
        return cardinality
    
    def generate_quality_report(self, categorical_cols: List[str] = None) -> Dict:
        """
        Generate comprehensive data quality report
        """
        report = {
            "total_rows": self.df.count(),
            "total_columns": len(self.df.columns),
            "missing_values": self.check_missing_values().collect()[0].asDict(),
            "duplicates": self.check_duplicates(),
            "statistics": self.get_statistics().toPandas().to_dict()
        }
        
        if categorical_cols:
            report["cardinality"] = self.check_cardinality(categorical_cols)
        
        return report


class FeatureEngineer:
    """Advanced feature engineering operations"""
    
    @staticmethod
    def create_interaction_features(
        df: DataFrame, 
        col1: str, 
        col2: str, 
        operation: str = "multiply"
    ) -> DataFrame:
        """
        Create interaction features between two columns
        
        Args:
            df: Input DataFrame
            col1: First column name
            col2: Second column name
            operation: Type of operation ('multiply', 'add', 'subtract', 'divide')
        """
        new_col_name = f"{col1}_{operation}_{col2}"
        
        if operation == "multiply":
            return df.withColumn(new_col_name, col(col1) * col(col2))
        elif operation == "add":
            return df.withColumn(new_col_name, col(col1) + col(col2))
        elif operation == "subtract":
            return df.withColumn(new_col_name, col(col1) - col(col2))
        elif operation == "divide":
            return df.withColumn(
                new_col_name, 
                when(col(col2) != 0, col(col1) / col(col2)).otherwise(0)
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def create_polynomial_features(
        df: DataFrame, 
        col_name: str, 
        degree: int = 2
    ) -> DataFrame:
        """
        Create polynomial features for a given column
        """
        result_df = df
        for d in range(2, degree + 1):
            new_col_name = f"{col_name}_pow_{d}"
            result_df = result_df.withColumn(new_col_name, col(col_name) ** d)
        return result_df
    
    @staticmethod
    def create_binned_features(
        df: DataFrame,
        col_name: str,
        num_bins: int = 5,
        method: str = "quantile"
    ) -> DataFrame:
        """
        Create binned categorical features from continuous variables
        
        Args:
            df: Input DataFrame
            col_name: Column to bin
            num_bins: Number of bins
            method: 'quantile' or 'equal_width'
        """
        output_col = f"{col_name}_binned"
        
        if method == "quantile":
            discretizer = QuantileDiscretizer(
                numBuckets=num_bins,
                inputCol=col_name,
                outputCol=output_col,
                handleInvalid="keep"
            )
            return discretizer.fit(df).transform(df)
        else:
            # Calculate equal-width splits
            stats = df.select(
                min(col(col_name)).alias("min"),
                max(col(col_name)).alias("max")
            ).collect()[0]
            
            min_val = stats["min"]
            max_val = stats["max"]
            width = (max_val - min_val) / num_bins
            splits = [min_val + i * width for i in range(num_bins + 1)]
            splits[0] = float('-inf')
            splits[-1] = float('inf')
            
            bucketizer = Bucketizer(
                splits=splits,
                inputCol=col_name,
                outputCol=output_col,
                handleInvalid="keep"
            )
            return bucketizer.transform(df)
    
    @staticmethod
    def create_aggregation_features(
        df: DataFrame,
        group_col: str,
        agg_col: str,
        agg_functions: List[str] = ["mean", "sum", "count"]
    ) -> DataFrame:
        """
        Create aggregation features based on grouping
        """
        from pyspark.sql.functions import avg, sum as _sum, count, max as _max, min as _min
        
        agg_exprs = []
        for func in agg_functions:
            if func == "mean":
                agg_exprs.append(avg(agg_col).alias(f"{group_col}_{agg_col}_mean"))
            elif func == "sum":
                agg_exprs.append(_sum(agg_col).alias(f"{group_col}_{agg_col}_sum"))
            elif func == "count":
                agg_exprs.append(count(agg_col).alias(f"{group_col}_{agg_col}_count"))
            elif func == "max":
                agg_exprs.append(_max(agg_col).alias(f"{group_col}_{agg_col}_max"))
            elif func == "min":
                agg_exprs.append(_min(agg_col).alias(f"{group_col}_{agg_col}_min"))
        
        agg_df = df.groupBy(group_col).agg(*agg_exprs)
        return df.join(agg_df, on=group_col, how="left")


class PipelineBuilder:
    """Build ML preprocessing pipelines"""
    
    def __init__(self):
        self.stages = []
    
    def add_imputer(
        self,
        input_cols: List[str],
        strategy: str = "mean"
    ):
        """
        Add imputation stage for missing values
        
        Args:
            input_cols: List of columns to impute
            strategy: Imputation strategy ('mean', 'median', 'mode')
        """
        imputer = Imputer(
            inputCols=input_cols,
            outputCols=[f"{col}_imputed" for col in input_cols],
            strategy=strategy
        )
        self.stages.append(imputer)
        return self
    
    def add_string_indexer(
        self,
        input_cols: List[str],
        handle_invalid: str = "keep"
    ):
        """
        Add string indexing stage for categorical variables
        """
        for col in input_cols:
            indexer = StringIndexer(
                inputCol=col,
                outputCol=f"{col}_indexed",
                handleInvalid=handle_invalid
            )
            self.stages.append(indexer)
        return self
    
    def add_one_hot_encoder(
        self,
        input_cols: List[str]
    ):
        """
        Add one-hot encoding stage
        """
        encoder = OneHotEncoder(
            inputCols=[f"{col}_indexed" for col in input_cols],
            outputCols=[f"{col}_encoded" for col in input_cols],
            handleInvalid="keep"
        )
        self.stages.append(encoder)
        return self
    
    def add_vector_assembler(
        self,
        input_cols: List[str],
        output_col: str = "features"
    ):
        """
        Add vector assembler to combine features
        """
        assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol=output_col,
            handleInvalid="keep"
        )
        self.stages.append(assembler)
        return self
    
    def add_standard_scaler(
        self,
        input_col: str = "features",
        output_col: str = "scaled_features",
        with_mean: bool = True,
        with_std: bool = True
    ):
        """
        Add standard scaler for feature normalization
        """
        scaler = StandardScaler(
            inputCol=input_col,
            outputCol=output_col,
            withMean=with_mean,
            withStd=with_std
        )
        self.stages.append(scaler)
        return self
    
    def add_minmax_scaler(
        self,
        input_col: str = "features",
        output_col: str = "scaled_features",
        min_val: float = 0.0,
        max_val: float = 1.0
    ):
        """
        Add MinMax scaler for feature normalization
        """
        scaler = MinMaxScaler(
            inputCol=input_col,
            outputCol=output_col,
            min=min_val,
            max=max_val
        )
        self.stages.append(scaler)
        return self
    
    def build(self) -> Pipeline:
        """
        Build and return the pipeline
        """
        return Pipeline(stages=self.stages)


class OutlierHandler:
    """Handle outliers in numerical features"""
    
    @staticmethod
    def detect_outliers_iqr(
        df: DataFrame,
        col_name: str,
        multiplier: float = 1.5
    ) -> DataFrame:
        """
        Detect outliers using IQR method
        """
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return df.withColumn(
            f"{col_name}_is_outlier",
            when(
                (col(col_name) < lower_bound) | (col(col_name) > upper_bound),
                lit(1)
            ).otherwise(lit(0))
        )
    
    @staticmethod
    def cap_outliers(
        df: DataFrame,
        col_name: str,
        multiplier: float = 1.5
    ) -> DataFrame:
        """
        Cap outliers at IQR bounds
        """
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return df.withColumn(
            f"{col_name}_capped",
            when(col(col_name) < lower_bound, lower_bound)
            .when(col(col_name) > upper_bound, upper_bound)
            .otherwise(col(col_name))
        )
    
    @staticmethod
    def remove_outliers(
        df: DataFrame,
        col_name: str,
        multiplier: float = 1.5
    ) -> DataFrame:
        """
        Remove rows with outliers
        """
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return df.filter(
            (col(col_name) >= lower_bound) & (col(col_name) <= upper_bound)
        )


class DataBalancer:
    """Handle imbalanced datasets"""
    
    @staticmethod
    def oversample_minority(
        df: DataFrame,
        label_col: str = "label",
        ratio: float = 1.0
    ) -> DataFrame:
        """
        Oversample minority class
        
        Args:
            df: Input DataFrame
            label_col: Name of label column
            ratio: Target ratio of minority to majority class
        """
        # Calculate class counts
        class_counts = df.groupBy(label_col).count().collect()
        class_counts_dict = {row[label_col]: row["count"] for row in class_counts}
        
        # Identify majority and minority classes
        majority_class = max(class_counts_dict, key=class_counts_dict.get)
        minority_class = min(class_counts_dict, key=class_counts_dict.get)
        
        majority_count = class_counts_dict[majority_class]
        minority_count = class_counts_dict[minority_class]
        
        # Calculate required samples
        target_count = int(majority_count * ratio)
        oversample_ratio = target_count / minority_count
        
        # Oversample minority class
        minority_df = df.filter(col(label_col) == minority_class)
        majority_df = df.filter(col(label_col) == majority_class)
        
        # Sample with replacement
        oversampled_minority = minority_df.sample(
            withReplacement=True,
            fraction=oversample_ratio,
            seed=42
        )
        
        return majority_df.union(oversampled_minority)
    
    @staticmethod
    def undersample_majority(
        df: DataFrame,
        label_col: str = "label",
        ratio: float = 1.0
    ) -> DataFrame:
        """
        Undersample majority class
        """
        # Calculate class counts
        class_counts = df.groupBy(label_col).count().collect()
        class_counts_dict = {row[label_col]: row["count"] for row in class_counts}
        
        # Identify majority and minority classes
        majority_class = max(class_counts_dict, key=class_counts_dict.get)
        minority_class = min(class_counts_dict, key=class_counts_dict.get)
        
        minority_count = class_counts_dict[minority_class]
        target_count = int(minority_count / ratio)
        
        # Undersample majority class
        minority_df = df.filter(col(label_col) == minority_class)
        majority_df = df.filter(col(label_col) == majority_class)
        
        # Calculate fraction for sampling
        fraction = target_count / class_counts_dict[majority_class]
        
        undersampled_majority = majority_df.sample(
            withReplacement=False,
            fraction=fraction,
            seed=42
        )
        
        return minority_df.union(undersampled_majority)


def create_preprocessing_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    imputation_strategy: str = "mean",
    scaling_method: str = "standard"
) -> Pipeline:
    """
    Create a complete preprocessing pipeline
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        imputation_strategy: Strategy for imputing missing values
        scaling_method: Method for scaling features ('standard' or 'minmax')
    
    Returns:
        PySpark ML Pipeline
    """
    builder = PipelineBuilder()
    
    # Add imputation for numerical features
    if numerical_cols:
        builder.add_imputer(numerical_cols, strategy=imputation_strategy)
    
    # Add string indexing and one-hot encoding for categorical features
    if categorical_cols:
        builder.add_string_indexer(categorical_cols)
        builder.add_one_hot_encoder(categorical_cols)
    
    # Prepare feature columns for assembly
    feature_cols = []
    if numerical_cols:
        feature_cols.extend([f"{col}_imputed" for col in numerical_cols])
    if categorical_cols:
        feature_cols.extend([f"{col}_encoded" for col in categorical_cols])
    
    # Add vector assembler
    builder.add_vector_assembler(feature_cols, output_col="features")
    
    # Add scaling
    if scaling_method == "standard":
        builder.add_standard_scaler(
            input_col="features",
            output_col="scaled_features"
        )
    elif scaling_method == "minmax":
        builder.add_minmax_scaler(
            input_col="features",
            output_col="scaled_features"
        )
    
    return builder.build()
