# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "seaborn",
# ]
# ///
# DBTITLE 1,Install seaborn
# MAGIC %pip install seaborn

# COMMAND ----------

# MAGIC %md
# MAGIC Task 1 — Data preprocessing and cleanup

# COMMAND ----------

# DBTITLE 1,Load the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# COMMAND ----------

# MAGIC %md
# MAGIC Data Ingestion

# COMMAND ----------

file_path = "workspace.default.patient_readmission_data_"

#The table is loaded from Unity Catalog into a DataFrame called df. The head() command shows the first five admissions
df = spark.table(file_path).toPandas()

print("Rows and columns:", df.shape)
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC the dataset has 3000 rows and 10 columns

# COMMAND ----------

# DBTITLE 1,Inspect the columns and data types

#This helps you see whether columns are numeric or text and summarises the data. For example, age should be numeric, while gender should be categorical.

df.info()

display(df.describe(include="all").T)

# COMMAND ----------

# MAGIC %md
# MAGIC the dataset has  3

# COMMAND ----------

#checking the number of null values on the dataset
df.isnull().sum()

# COMMAND ----------

# DBTITLE 1,Check missing values and duplicates

#We check for missing data and duplicates before making changes. If you find none, record that in your report rather than inventing cleaning work.
print("Missing values by column:")
display(df.isna().sum())

print("Fully duplicated rows:", df.duplicated().sum())
print("Duplicate Patient IDs:", df["Patient ID"].duplicated().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC Exploring Data

# COMMAND ----------

# DBTITLE 1,Check categories and valid ranges
#The brief gives expected ranges for the numeric fields. This code identifies records outside those ranges; it does not automatically delete them

print("Gender values:", df["Gender"].unique())


# COMMAND ----------

print("Admission Type values:", df["Admission Type"].unique())

sns.countplot(data=df, x="Admission Type")
plt.title("Distribution of Admission Types")
plt.xlabel("Admission Type")
plt.ylabel("Number of admissions")
plt.tight_layout()
plt.show()

counts = df["Admission Type"].value_counts()
percentages = (counts / len(df) * 100).round(2)
display(pd.DataFrame({"Count": counts, "Percentage": percentages}))


# COMMAND ----------

print("Readmission values:", df["Readmission"].unique())

sns.countplot(data=df, x="Readmission")
plt.title("Distribution of Readmission Values")
plt.xlabel("Readmitted within 30 days")
plt.ylabel("Number of admissions")
plt.tight_layout()
plt.show()

counts = df["Readmission"].value_counts()
percentages = (counts / len(df) * 100).round(2)
display(pd.DataFrame({"Count": counts, "Percentage": percentages}))



# COMMAND ----------

valid_ranges = {
    "Age": (18, 89),
    "Length of Stay": (1, 29),
    "Number of Diagnoses": (1, 9),
    "Blood Pressure": (60, 179),
    "Blood Sugar Levels": (51, 199),
    "Previous Admissions": (0, 4)
}

for col, (minimum, maximum) in valid_ranges.items():
    invalid = df[~df[col].between(minimum, maximum)]
    print(col, "invalid records:", len(invalid))

# COMMAND ----------

# DBTITLE 1,Clean category labels carefully
#This removes unnecessary spaces and standardises capitalisation. Afterward, check that the values match the expected categories
df["Gender"] = df["Gender"].astype("string").str.strip().str.title()
df["Admission Type"] = (
    df["Admission Type"].astype("string").str.strip().str.title()
)
df["Readmission"] = df["Readmission"].astype("string").str.strip().str.title()

print("Gender:", df["Gender"].unique())
print("Admission Type:", df["Admission Type"].unique())
print("Readmission:", df["Readmission"].unique())

# COMMAND ----------

# DBTITLE 1,Split the data before fitting preprocessing
#This creates a training set and a held-out test set while preserving the approximate proportion of readmissions in both. With 3,000 rows and an 80/20 split, you should have 2,400 training rows and 600 test rows.

from sklearn.model_selection import train_test_split

numeric_features = [
    "Age",
    "Length of Stay",
    "Number of Diagnoses",
    "Blood Pressure",
    "Blood Sugar Levels",
    "Previous Admissions"
]

categorical_features = ["Gender", "Admission Type"]
feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = df["Readmission"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected Readmission label. Check the labels in Step 4.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))



# COMMAND ----------

print("Training target proportions:")
display(y_train.value_counts(normalize=True))



# COMMAND ----------

print("Testing target proportions:")
display(y_test.value_counts(normalize=True))

# COMMAND ----------

# DBTITLE 1,Build the preprocessing pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("Preprocessing pipeline created.")

#The median fills missing numeric values if any appear in future data.
#The most frequent category fills missing categorical values.
#One-hot encoding converts categories into numerical features.
#Standard scaling puts numeric features on comparable scales.
#The pipeline ensures that the imputer, scaler and encoder are fitted using training data when the model is trained.

# COMMAND ----------

# MAGIC %md
# MAGIC Task 2 — Exploratory data analysis

# COMMAND ----------

# DBTITLE 1,Check the balance of the target
eda = X_train.copy()
eda["Readmission"] = y_train.map({0: "No", 1: "Yes"})

target_counts = eda["Readmission"].value_counts()
target_percentages = eda["Readmission"].value_counts(normalize=True) * 100

display(pd.DataFrame({
    "Number of admissions": target_counts,
    "Percentage": target_percentages.round(2)
}))

sns.countplot(data=eda, x="Readmission")
plt.title("Figure 1: Distribution of 30-Day Readmission")
plt.xlabel("Readmitted within 30 days")
plt.ylabel("Number of admissions")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Plot numeric distributions
numeric_columns = numeric_features

eda[numeric_columns].hist(
    figsize=(14, 10),
    bins=20,
    edgecolor="black"
)

plt.suptitle("Figure 2: Distributions of Numeric Patient Variables")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Compare readmission rates by category
gender_data = X_train[["Gender"]].copy()
gender_data["Readmission"] = y_train

gender_data["Readmission"] = gender_data["Readmission"].map(
    {0: "No", 1: "Yes"}
)

gender_rates = (
    gender_data.assign(Readmitted=gender_data["Readmission"].eq("Yes"))
    .groupby("Gender")["Readmitted"]
    .mean()
    .mul(100)
    .reset_index(name="Readmission rate (%)")
)

sns.barplot(data=gender_rates, x="Gender", y="Readmission rate (%)")
plt.title("Figure 3: Readmission Rate by Gender")
plt.ylabel("Readmission rate (%)")
plt.xlabel("Gender")
plt.tight_layout()
plt.show()

display(gender_rates.round(2))

# COMMAND ----------

admission_data = X_train[["Admission Type"]].copy()
admission_data["Readmitted"] = y_train.eq(1).to_numpy()

admission_rates = (
    admission_data.groupby("Admission Type")["Readmitted"]
    .mean()
    .mul(100)
    .reset_index(name="Readmission rate (%)")
)

sns.barplot(
    data=admission_rates,
    x="Admission Type",
    y="Readmission rate (%)"
)
plt.title("Figure 4: Readmission Rate by Admission Type")
plt.ylabel("Readmission rate (%)")
plt.xlabel("Admission Type")
plt.tight_layout()
plt.show()

display(admission_rates.round(2))

# COMMAND ----------

# DBTITLE 1,Compare numeric variables by outcome
plot_data = X_train.copy()
plot_data["Readmission"] = y_train.map({0: "No", 1: "Yes"})

for col in ["Age", "Length of Stay", "Previous Admissions"]:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=plot_data, x="Readmission", y=col)
    plt.title(f"Figure: {col} by Readmission Outcome")
    plt.xlabel("Readmitted within 30 days")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Understanding Numerical Data by creating a correlation matrix

# COMMAND ----------

# DBTITLE 1,Explore numeric correlations
correlation_data = X_train[numeric_features].copy()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Figure 5: Correlation Between Numeric Predictors")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Summarise the numeric data
display(X_train[numeric_features].describe().T)

# COMMAND ----------

# DBTITLE 1,Summarise the numeric data
eda[numeric_columns].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC **Previous Admissions**: Most patients have been here before. The average patient has been admitted roughly 2 times previously. 
# MAGIC **Length of Stay**: Patients stay in the hospital for an average of 15 days

# COMMAND ----------

import pandas as pd

# 1. (Optional) Recreate your summary data DataFrame for context
# Replace this with your actual DataFrame variable name, e.g., df = eda[numeric_columns]
columns = ['Age', 'Length of Stay', 'Number of Diagnoses', 'Blood Pressure', 'Blood Sugar Levels', 'Previous Admissions']

# 2. Automatically find and count outliers using the IQR rule
print("--- OUTLIER ANALYSIS ---")

outlier_counts = {}

for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_counts[col] = len(outliers)
    
    print(f"{col}: Found {len(outliers)} outliers (Limits: {lower_bound:.1f} to {upper_bound:.1f})")



# COMMAND ----------

# MAGIC %md
# MAGIC Task 3 — Feature selection

# COMMAND ----------

# DBTITLE 1,Test numeric features
from scipy.stats import mannwhitneyu

test_data = X_train.copy()
test_data["target"] = y_train

numeric_test_results = []

for col in numeric_features:
    no_group = test_data.loc[test_data["target"] == 0, col]
    yes_group = test_data.loc[test_data["target"] == 1, col]

    statistic, p_value = mannwhitneyu(
        no_group,
        yes_group,
        alternative="two-sided"
    )

    numeric_test_results.append({
        "Feature": col,
        "No median": no_group.median(),
        "Yes median": yes_group.median(),
        "U statistic": statistic,
        "p-value": p_value
    })

numeric_tests = pd.DataFrame(numeric_test_results)
display(numeric_tests.round(4))

# COMMAND ----------

# DBTITLE 1,Test categorical features
from scipy.stats import chi2_contingency

categorical_test_results = []

for col in categorical_features:
    table = pd.crosstab(X_train[col], y_train)
    chi2, p_value, dof, expected = chi2_contingency(table)

    categorical_test_results.append({
        "Feature": col,
        "Chi-square": chi2,
        "Degrees of freedom": dof,
        "p-value": p_value
    })

categorical_tests = pd.DataFrame(categorical_test_results)
display(categorical_tests.round(4))

# COMMAND ----------

# DBTITLE 1,Calculate mutual information
from sklearn.feature_selection import mutual_info_classif

mi_preprocessor = preprocessor.fit(X_train)

X_train_transformed = mi_preprocessor.transform(X_train)
transformed_names = mi_preprocessor.get_feature_names_out()

mi_scores = mutual_info_classif(
    X_train_transformed,
    y_train,
    random_state=42
)

mi_results = pd.DataFrame({
    "Feature": transformed_names,
    "Mutual information": mi_scores
}).sort_values("Mutual information", ascending=False)

display(mi_results)

#Mutual information estimates how much information each transformed feature provides about the target, including some non-linear relationships. A score near zero suggests little detected information by this measure; it does not prove that a variable is useless

# COMMAND ----------

# DBTITLE 1,Examine Random Forest feature importance
from sklearn.ensemble import RandomForestClassifier

importance_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

importance_model.fit(X_train, y_train)

rf = importance_model.named_steps["classifier"]
names = importance_model.named_steps["preprocessor"].get_feature_names_out()

importance_results = pd.DataFrame({
    "Feature": names,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

display(importance_results)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=importance_results.head(10),
    x="Importance",
    y="Feature"
)
plt.title("Figure 6: Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

#These scores show how the fitted forest used its features. They are not causal effects, and importance can be divided across related features.

# COMMAND ----------

# DBTITLE 1,Create a keep/drop decision table
decision_table = pd.DataFrame({
    "Feature": [
        "Patient ID",
        "Age",
        "Gender",
        "Admission Type",
        "Length of Stay",
        "Number of Diagnoses",
        "Blood Pressure",
        "Blood Sugar Levels",
        "Previous Admissions"
    ],
    "Initial decision": [
        "Drop",
        "Review evidence",
        "Review evidence",
        "Review evidence",
        "Review evidence",
        "Review evidence",
        "Review evidence",
        "Review evidence",
        "Review evidence"
    ],
    "Reason": [
        "Identifier, not a meaningful predictor",
        "Assess statistical and clinical relevance",
        "Assess relevance and fairness implications",
        "Assess relevance to readmission",
        "Assess relevance and timing of availability",
        "Assess relevance and redundancy",
        "Assess relevance and data quality",
        "Assess relevance and data quality",
        "Assess relevance to prior healthcare use"
    ]
})

display(decision_table)

# COMMAND ----------

# MAGIC %md
# MAGIC Task 4 — Model building

# COMMAND ----------

# DBTITLE 1,Create the baseline model
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

baseline_scores = cross_val_score(
    baseline,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)

print("Baseline ROC-AUC mean:", round(baseline_scores.mean(), 3))
print("Baseline ROC-AUC standard deviation:", round(baseline_scores.std(), 3))

#Cross-validation divides the training data into five parts. Each part is held out once while the model is trained on the others. The mean and standard deviation show average performance and how much it varies across folds.

# COMMAND ----------

# DBTITLE 1,Compare three models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )
}

cv_results = []

for name, classifier in models.items():
    candidate = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        candidate,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    cv_results.append({
        "Model": name,
        "Mean ROC-AUC": scores.mean(),
        "Standard deviation": scores.std()
    })

cv_table = pd.DataFrame(cv_results).sort_values(
    "Mean ROC-AUC",
    ascending=False
)

display(cv_table.round(3))

# COMMAND ----------

# DBTITLE 1,Tune Logistic Regression
from sklearn.model_selection import GridSearchCV

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

logistic_grid = {
    "classifier__C": [0.01, 0.1, 1, 10]
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    param_grid=logistic_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

logistic_search.fit(X_train, y_train)

print("Best logistic parameters:", logistic_search.best_params_)
print("Best logistic CV ROC-AUC:", round(logistic_search.best_score_, 3))

#C controls the strength of regularisation. The grid search tests the specified values using cross-validation on the training set

# COMMAND ----------

# DBTITLE 1,Tune Random Forest
forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

forest_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 8, None],
    "classifier__min_samples_leaf": [5, 10]
}

forest_search = GridSearchCV(
    forest_pipeline,
    param_grid=forest_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

forest_search.fit(X_train, y_train)

print("Best forest parameters:", forest_search.best_params_)
print("Best forest CV ROC-AUC:", round(forest_search.best_score_, 3))

#This searches 12 combinations of tree count, maximum depth and minimum leaf size. Limiting depth and requiring more observations in a leaf can help control overfitting.

# COMMAND ----------

# DBTITLE 1,Choose the final model
tuned_models = {
    "Logistic Regression": logistic_search.best_estimator_,
    "Random Forest": forest_search.best_estimator_
}

tuned_results = pd.DataFrame([
    {
        "Model": name,
        "Best CV ROC-AUC": search.best_score_,
        "Best parameters": search.best_params_
    }
    for name, search in [
        ("Logistic Regression", logistic_search),
        ("Random Forest", forest_search)
    ]
]).sort_values("Best CV ROC-AUC", ascending=False)

display(tuned_results)

best_name = max(
    tuned_models,
    key=lambda name: (
        logistic_search.best_score_
        if name == "Logistic Regression"
        else forest_search.best_score_
    )
)

final_model = tuned_models[best_name]

print("Selected model:", best_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Task 5 — Model evaluation and testing

# COMMAND ----------

# DBTITLE 1,Evaluate the final model on the test set
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

final_model.fit(X_train, y_train)

y_probability = final_model.predict_proba(X_test)[:, 1]
y_prediction = (y_probability >= 0.5).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test, y_prediction),
    "Precision": precision_score(y_test, y_prediction, zero_division=0),
    "Recall": recall_score(y_test, y_prediction, zero_division=0),
    "F1-score": f1_score(y_test, y_prediction, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_probability)
}

display(pd.DataFrame({
    "Metric": list(metrics.keys()),
    "Score": list(metrics.values())
}).round(3))

print(classification_report(
    y_test,
    y_prediction,
    target_names=["Not readmitted", "Readmitted"],
    zero_division=0
))

#Accuracy: How many predictions were correct overall?
#Precision: Of the patients flagged as likely to return, how many actually returned?
#Recall: Of all patients who returned, how many did the model identify?
#F1-score: A balance between precision and recall.
#ROC-AUC: How well the model ranks readmitted patients above non-readmitted patients across thresholds.

# COMMAND ----------

# DBTITLE 1,Display the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_prediction, labels=[0, 1])

display(pd.DataFrame(
    cm,
    index=["Actual: No", "Actual: Yes"],
    columns=["Predicted: No", "Predicted: Yes"]
))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not readmitted", "Readmitted"]
).plot()

plt.title("Figure 7: Confusion Matrix on the Test Set")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Plot the ROC curve and precision–recall curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

RocCurveDisplay.from_predictions(y_test, y_probability)
plt.title("Figure 8: ROC Curve on the Test Set")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_predictions(y_test, y_probability)
plt.title("Figure 9: Precision–Recall Curve on the Test Set")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Choose a decision threshold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

oof_probability = cross_val_predict(
    final_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_train,
    oof_probability
)

# Prefer the highest precision among thresholds with recall >= 80%
eligible = np.where(recall[:-1] >= 0.80)[0]

if len(eligible) > 0:
    best_index = eligible[np.argmax(precision[eligible])]
    chosen_threshold = thresholds[best_index]
else:
    # If 80% recall is not achievable, choose the best F1 threshold
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )
    chosen_threshold = thresholds[np.argmax(f1_values)]

print("Chosen threshold:", round(chosen_threshold, 3))

# COMMAND ----------

y_prediction_threshold = (
    y_probability >= chosen_threshold
).astype(int)

print("Recall:", round(
    recall_score(y_test, y_prediction_threshold, zero_division=0), 3
))

print("Precision:", round(
    precision_score(y_test, y_prediction_threshold, zero_division=0), 3
))

print("F1-score:", round(
    f1_score(y_test, y_prediction_threshold, zero_division=0), 3
))

# COMMAND ----------

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(
    chosen_threshold,
    linestyle="--",
    label="Chosen threshold"
)
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("Figure 10: Precision and Recall by Threshold")
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC An 80% recall target is an illustrative choice for this exercise, not a clinical standard. A hospital would need to agree on an appropriate target based on resources and patient-safety considerations.

# COMMAND ----------

# DBTITLE 1,Check for overfitting
# Overfitting happens when a model performs well on training data but substantially worse on unseen data.

from sklearn.metrics import roc_auc_score

train_probability = final_model.predict_proba(X_train)[:, 1]
test_probability = final_model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_probability)
test_auc = roc_auc_score(y_test, test_probability)

print("Training ROC-AUC:", round(train_auc, 3))
print("Testing ROC-AUC:", round(test_auc, 3))
print("Train-test gap:", round(train_auc - test_auc, 3))

# COMMAND ----------

# MAGIC %md
# MAGIC A large positive gap can indicate overfitting, while a small gap suggests more consistent ranking performance. Neither result alone proves that a model generalises well.

# COMMAND ----------

# DBTITLE 1,Save final model
import joblib

joblib.dump(final_model, "/tmp/patient_readmission_model.joblib")

print("Final model saved.")