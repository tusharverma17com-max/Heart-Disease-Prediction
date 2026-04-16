"""
Heart Disease Prediction using Apache Spark (PySpark)
Dataset: 1025 patient records, 13 features, binary target (0=No Disease, 1=Disease)
"""

# ─────────────────────────────────────────────
# 1. IMPORTS & SPARK SESSION
# ─────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder \
    .appName("HeartDiseasePrediction") \
    .config("spark.ui.showConsoleProgress", "false") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark Session Started:", spark.version)

# ─────────────────────────────────────────────
# 2. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
DATA_PATH = r"C:\Users\91841\Downloads\archive (4)\HeartDiseaseTrain-Test.csv"

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

print("\n" + "="*60)
print("📊 DATASET OVERVIEW")
print("="*60)
print(f"Rows: {df.count()}   Columns: {len(df.columns)}")
print("\nSchema:")
df.printSchema()

print("\nFirst 5 rows:")
df.show(5, truncate=False)

print("\nTarget Distribution:")
df.groupBy("target").count().orderBy("target").show()

print("\nBasic Statistics (Numeric):")
numeric_cols = ["age","resting_blood_pressure","cholestoral","Max_heart_rate","oldpeak"]
df.select(numeric_cols).describe().show()

# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("🔧 PREPROCESSING")
print("="*60)

# Check nulls
null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
print("Null value counts:")
null_counts.show()

# Categorical columns to encode
categorical_cols = [
    "sex", "chest_pain_type", "fasting_blood_sugar",
    "rest_ecg", "exercise_induced_angina", "slope",
    "vessels_colored_by_flourosopy", "thalassemia"
]

# Numeric columns
numeric_cols = [
    "age", "resting_blood_pressure", "cholestoral",
    "Max_heart_rate", "oldpeak"
]

# Build indexers for each categorical column
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]

# OneHotEncoders
encoders = [
    OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe")
    for c in categorical_cols
]

# Assemble all features into one vector
ohe_cols    = [c + "_ohe" for c in categorical_cols]
feature_cols = numeric_cols + ohe_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler    = StandardScaler(inputCol="raw_features", outputCol="features",
                           withMean=False, withStd=True)

print("Pipeline stages built:")
print(f"  → {len(indexers)} StringIndexers")
print(f"  → {len(encoders)} OneHotEncoders")
print(f"  → VectorAssembler ({len(feature_cols)} feature inputs)")
print(f"  → StandardScaler")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"\nTrain size: {train_df.count()}  |  Test size: {test_df.count()}")

# ─────────────────────────────────────────────
# 5. DEFINE MODELS
# ─────────────────────────────────────────────
lr  = LogisticRegression(featuresCol="features", labelCol="target",
                          maxIter=100, regParam=0.01)
dt  = DecisionTreeClassifier(featuresCol="features", labelCol="target",
                              maxDepth=5, seed=42)
rf  = RandomForestClassifier(featuresCol="features", labelCol="target",
                               numTrees=100, seed=42)
gbt = GBTClassifier(featuresCol="features", labelCol="target",
                     maxIter=50, seed=42)

models = {
    "Logistic Regression": lr,
    "Decision Tree":       dt,
    "Random Forest":       rf,
    "Gradient Boosting":   gbt,
}

# ─────────────────────────────────────────────
# 6. TRAIN, EVALUATE, COMPARE
# ─────────────────────────────────────────────
bin_eval  = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
acc_eval  = MulticlassClassificationEvaluator(labelCol="target",
                                               predictionCol="prediction",
                                               metricName="accuracy")
f1_eval   = MulticlassClassificationEvaluator(labelCol="target",
                                               predictionCol="prediction",
                                               metricName="f1")
prec_eval = MulticlassClassificationEvaluator(labelCol="target",
                                               predictionCol="prediction",
                                               metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(labelCol="target",
                                               predictionCol="prediction",
                                               metricName="weightedRecall")

results = {}

print("\n" + "="*60)
print("🤖 MODEL TRAINING & EVALUATION")
print("="*60)

for name, classifier in models.items():
    print(f"\n▶ Training: {name} ...")

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, classifier])
    model    = pipeline.fit(train_df)
    preds    = model.transform(test_df)

    auc  = bin_eval.evaluate(preds)
    acc  = acc_eval.evaluate(preds)
    f1   = f1_eval.evaluate(preds)
    prec = prec_eval.evaluate(preds)
    rec  = rec_eval.evaluate(preds)

    results[name] = {
        "AUC-ROC":   round(auc,  4),
        "Accuracy":  round(acc,  4),
        "F1 Score":  round(f1,   4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
    }

    print(f"   AUC-ROC : {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Score: {f1:.4f}")

# ─────────────────────────────────────────────
# 7. RESULTS SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("📈 RESULTS COMPARISON")
print("="*60)
header = f"{'Model':<25} {'AUC-ROC':>8} {'Accuracy':>9} {'F1':>8} {'Precision':>10} {'Recall':>8}"
print(header)
print("-" * 72)
for name, m in results.items():
    print(f"{name:<25} {m['AUC-ROC']:>8} {m['Accuracy']:>9} {m['F1 Score']:>8} {m['Precision']:>10} {m['Recall']:>8}")

best = max(results, key=lambda k: results[k]["AUC-ROC"])
print(f"\n🏆 Best Model: {best}  (AUC-ROC = {results[best]['AUC-ROC']})")

# ─────────────────────────────────────────────
# 8. CROSS-VALIDATION ON BEST MODEL (Random Forest)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("🔁 CROSS-VALIDATION — Random Forest (5-fold)")
print("="*60)

rf_cv = RandomForestClassifier(featuresCol="features", labelCol="target", seed=42)
pipeline_cv = Pipeline(stages=indexers + encoders + [assembler, scaler, rf_cv])

paramGrid = ParamGridBuilder() \
    .addGrid(rf_cv.numTrees, [50, 100]) \
    .addGrid(rf_cv.maxDepth, [5, 10]) \
    .build()

cv = CrossValidator(estimator=pipeline_cv,
                    estimatorParamMaps=paramGrid,
                    evaluator=bin_eval,
                    numFolds=5,
                    seed=42)

cv_model  = cv.fit(train_df)
cv_preds  = cv_model.transform(test_df)

cv_auc = bin_eval.evaluate(cv_preds)
cv_acc = acc_eval.evaluate(cv_preds)
cv_f1  = f1_eval.evaluate(cv_preds)

print(f"Best CV AUC-ROC : {cv_auc:.4f}")
print(f"Best CV Accuracy: {cv_acc:.4f}")
print(f"Best CV F1 Score: {cv_f1:.4f}")

best_rf = cv_model.bestModel.stages[-1]
print(f"Best numTrees   : {best_rf.getNumTrees}")
print(f"Best maxDepth   : {best_rf.getOrDefault('maxDepth')}")

# Feature Importances
print("\nTop 10 Feature Importances (Random Forest):")
feat_imp = best_rf.featureImportances.toArray()

# Recover feature names after OHE (approximate — use raw numeric names + ohe tags)
feat_names = numeric_cols.copy()
for c in categorical_cols:
    feat_names.append(c + "_ohe")   # grouped label

for idx in sorted(range(len(feat_imp)), key=lambda i: feat_imp[i], reverse=True)[:10]:
    fname = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
    print(f"  [{idx:>3}] {fname:<40} {feat_imp[idx]:.4f}")

# ─────────────────────────────────────────────
# 9. CONFUSION MATRIX (Best CV Model)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("🔢 CONFUSION MATRIX (Best CV Random Forest)")
print("="*60)

from pyspark.sql.functions import col as scol
tp = cv_preds.filter((scol("target")==1) & (scol("prediction")==1)).count()
tn = cv_preds.filter((scol("target")==0) & (scol("prediction")==0)).count()
fp = cv_preds.filter((scol("target")==0) & (scol("prediction")==1)).count()
fn = cv_preds.filter((scol("target")==1) & (scol("prediction")==0)).count()

print(f"\n                Predicted 0   Predicted 1")
print(f"  Actual 0         {tn:>5}         {fp:>5}")
print(f"  Actual 1         {fn:>5}         {tp:>5}")
print(f"\n  Sensitivity (Recall) : {tp/(tp+fn):.4f}")
print(f"  Specificity          : {tn/(tn+fp):.4f}")
print(f"  Precision            : {tp/(tp+fp):.4f}")

print("\n✅ Heart Disease Prediction Pipeline Complete!")
spark.stop()
