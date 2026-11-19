# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "seaborn",
# ]
# ///
# MAGIC %md
# MAGIC ## Step 1: Import Necessary Dependencies
# MAGIC In this step, we will import the necessary libraries such as:
# MAGIC
# MAGIC - For Linear Algebra: Numpy
# MAGIC - For Data Preprocessing, and CSV File I/O: Pandas
# MAGIC - For Model Building and Evaluation: Scikit-Learn
# MAGIC - For Data Visualization: Matplotlib, and Seaborn, etc.

# COMMAND ----------

# MAGIC %pip install seaborn

# COMMAND ----------

# MAGIC %md
# MAGIC Loads the libraries :
# MAGIC - pandas (tables),
# MAGIC - NumPy (maths),
# MAGIC - Matplotlib/Seaborn (charts),
# MAGIC - SciPy (statistical tests),
# MAGIC - scikit-learn (modelling)

# COMMAND ----------

# DBTITLE 1,Import library


import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay)

# COMMAND ----------

# MAGIC %md
# MAGIC Data Ingestion

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "workspace.default.patient_readmission_data_"

#The table is loaded from Unity Catalog into a DataFrame called df. The head() command shows the first five admissions
df = spark.table(file_path).toPandas()

print("Rows and columns:", df.shape)
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset show that it has 3000 rows and 10 columns,as also shown in the first 5 columns to see what the dataset is like

# COMMAND ----------

# DBTITLE 1,Cell 3
import warnings

warnings.filterwarnings("ignore")#Hides harmless warning messages so the notebook output stays readable.
sns.set_theme(style="whitegrid")#Gives every chart a consistent, clean style.
df = spark.table("workspace.default.patient_readmission_data_").toPandas()
print(df.shape)          # (3000, 10)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Study the Data

# COMMAND ----------

# MAGIC %md
# MAGIC Task 1: Data preprocessing

# COMMAND ----------

# DBTITLE 1,load and inspect
#What is the data type of cols?
df.info()
print(df.describe(include="all"))




# COMMAND ----------

# MAGIC %md
# MAGIC A non-null count of 3000 on every row means nothing is missing.
# MAGIC This helps you see whether columns are numeric or text and summarises the data. For example, age should be numeric, while gender should be categorical.
# MAGIC As shown in the dataset we have 7 Integer colums and 3 objects which are categorical

# COMMAND ----------

# DBTITLE 1,Check missing values and duplicates

print("Missing values by column:")
display(df.isna().sum())

print("Fully duplicated rows:", df.duplicated().sum())
print("Duplicate Patient IDs:", df["Patient ID"].duplicated().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC We check for missing data and duplicates before making changes and record the findings in the records than inventing clean work.

# COMMAND ----------

#checking missing values by column
print("Missing values by column:")

# COMMAND ----------

#checking the number of null values on the dataset
df.isnull().sum()

# COMMAND ----------

# Check and print the number of fully duplicated rows
duplicate_count = df.duplicated().sum()
print(f"Fully duplicated rows: {duplicate_count}")


# COMMAND ----------

# DBTITLE 1,checking the count on fully duplicated Patient ID's
#checking the count on fully duplicated Patient ID's
id_col = "Patient_ID" if "Patient_ID" in df.columns else "Patient ID"
duplicate_ids = df[id_col].duplicated().sum()
print(f"Duplicate Patient IDs: {duplicate_ids}")

# COMMAND ----------

path = "workspace.default.patient_readmission_data_"
df = spark.table(path).toPandas()

# rename columns 
df.columns = df.columns.str.strip().str.replace(" ", "_")

print(df.shape)
print(df.dtypes)
display(df.head())
display(df.describe(include="all"))



# COMMAND ----------

# MAGIC %md
# MAGIC All numeric columns came in as integers, which is correct. Gender, Admission_Type and Readmission are object (text), so they need encoding later. spaces were replaced with underscores

# COMMAND ----------

# DBTITLE 1,Check ranges
#The checking gives expected ranges for the numeric fields. 
# This code identifies records outside those ranges; it does not automatically delete them

print("Gender values:", df["Gender"].unique())


# COMMAND ----------

# MAGIC %md
# MAGIC checking unique values

# COMMAND ----------

print("Admission Type values:", df["Admission_Type"].unique())


# COMMAND ----------

print("Readmission values:", df["Readmission"].unique())



# COMMAND ----------

valid_ranges = {
    "Age": (18, 89),
    "Length_of_Stay": (1, 29),
    "Number_of_Diagnoses": (1, 9),
    "Blood_Pressure": (60, 179),
    "Blood_Sugar_Levels": (51, 199),
    "Previous_Admissions": (0, 4)
}

for col, (minimum, maximum) in valid_ranges.items():
    invalid = df[~df[col].between(minimum, maximum)]
    print(col, "invalid records:", len(invalid))

# COMMAND ----------

# DBTITLE 1,Clean category labels
#This removes unnecessary spaces and standardises capitalisation. 
df["Gender"] = df["Gender"].astype("string").str.strip().str.title()

df["Admission_Type"] = (
    df["Admission_Type"].astype("string").str.strip().str.title()
)

df["Readmission"] = df["Readmission"].astype("string").str.strip().str.title()

print("Gender:", df["Gender"].unique())
print("Admission Type:", df["Admission_Type"].unique())
print("Readmission:", df["Readmission"].unique())

# COMMAND ----------

# check spelling / case / spaces
print("Duplicate Patient IDs:", df["Patient_ID"].duplicated().sum())

for col in ["Gender", "Admission_Type", "Readmission"]:
    print(col, df[col].unique())          

# COMMAND ----------

  # compare with the ranges in the brief
  print(df.describe().loc[["min", "max"]]) 

# COMMAND ----------

# DBTITLE 1,Cell 8
#This acts as a filter on the summary statistics. Instead of showing all eight rows of data generated by .describe(), it uses label-based indexing to extract and display only the "min" and "max" rows

num_cols = ["Age", "Length_of_Stay", "Number_of_Diagnoses",
            "Blood_Pressure", "Blood_Sugar_Levels", "Previous_Admissions"]

display(df[num_cols].describe().loc[["min", "max"]])

# COMMAND ----------

# DBTITLE 1,clean text labels
before = {"rows": len(df), 
          "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())}

for col in ["Gender", "Admission_Type", "Readmission"]:
    df[col] = df[col].str.strip().str.title()

df = df.drop_duplicates()



# COMMAND ----------

after = {"rows": len(df),
          "missing": int(df.isna().sum().sum()), 
          "duplicates": int(df.duplicated().sum())}
print(pd.DataFrame([before, after],
                    index=["before", "after"]))

# COMMAND ----------

# DBTITLE 1,Cell 31
print("Admission Type values:", df["Admission_Type"].unique())

sns.countplot(data=df, x="Admission_Type")
plt.title("Distribution of Admission Types")
plt.xlabel("Admission Type")
plt.ylabel("Number of admissions")
plt.tight_layout()
plt.show()

counts = df["Admission_Type"].value_counts()
percentages = (counts / len(df) * 100).round(2)
display(pd.DataFrame({"Count": counts, "Percentage": percentages}))

# COMMAND ----------

# DBTITLE 1,Observations - Admission Type
# MAGIC %md
# MAGIC **Observations — Admission Type distribution**
# MAGIC
# MAGIC * Admissions are almost evenly split between Elective (1,563; 52.1%) and Emergency (1,437; 47.9%).
# MAGIC * Both categories are well represented, so Admission_Type can be one-hot encoded without a rare-category problem.

# COMMAND ----------

# DBTITLE 1,Gender distribution chart
print("Gender values:", df["Gender"].unique())

sns.countplot(data=df, x="Gender")
plt.title("Distribution of Genders")
plt.xlabel("Gender")
plt.ylabel("Number of admissions")
plt.tight_layout()
plt.show()

counts = df["Gender"].value_counts()
percentages = (counts / len(df) * 100).round(2)
display(pd.DataFrame({"Count": counts, "Percentage": percentages}))

# COMMAND ----------

# DBTITLE 1,Observations - Gender
# MAGIC %md
# MAGIC **Observations — Gender distribution**
# MAGIC
# MAGIC * Male patients account for 1,555 admissions (51.83%) and Female patients for 1,445 (48.17%).
# MAGIC * The split is nearly even, so Gender is balanced in the data and needs no rebalancing before modelling.

# COMMAND ----------

# DBTITLE 1,Readmission target distribution chart
print("Readmission values:", df["Readmission"].unique())

sns.countplot(data=df, x="Readmission")
plt.title("Distribution of Readmission (target)")
plt.xlabel("Readmitted")
plt.ylabel("Number of admissions")
plt.tight_layout()
plt.show()

counts = df["Readmission"].value_counts()
percentages = (counts / len(df) * 100).round(2)
display(pd.DataFrame({"Count": counts, "Percentage": percentages}))

# COMMAND ----------

# DBTITLE 1,Observations - Readmission
# MAGIC %md
# MAGIC **Observations — Readmission (target) distribution**
# MAGIC
# MAGIC * The target is imbalanced: 2,134 admissions (71.13%) were not readmitted, versus 866 (28.87%) that were.
# MAGIC * Accuracy alone would be misleading — a model that predicts "No" for everyone would already score about 71%. Precision, recall and AUC are better metrics to watch.
# MAGIC * The train/test split already uses `stratify=y`, which preserves this ratio in both sets (about 71/29).

# COMMAND ----------

# MAGIC %md
# MAGIC Understanding numerial data
# MAGIC

# COMMAND ----------

# DBTITLE 1,eda numeric_columns .describe()
numeric_columns = ["Age", "Length_of_Stay", "Number_of_Diagnoses",
                  "Blood_Pressure", "Blood_Sugar_Levels", "Previous_Admissions"]
df[numeric_columns].describe()

# COMMAND ----------

# DBTITLE 1,Observations - Numerical data
# MAGIC %md
# MAGIC **Observations — Numerical data**
# MAGIC
# MAGIC * All six numeric columns have 3,000 complete values (no missing data), and every min/max sits inside the valid ranges from the brief, so there are no impossible values.
# MAGIC * Age averages about 53 years (median 53, range 18–89) with a wide, even spread across the full adult range (quartiles 35 and 72).
# MAGIC * Length_of_Stay averages about 15 days (range 1–29) and Blood_Sugar_Levels about 134 (range 51–199) — both spread broadly rather than clustering.
# MAGIC * Previous_Admissions averages about 2 per patient (range 0–4), meaning most patients have some prior admission history.

# COMMAND ----------

df.info()

# COMMAND ----------

# DBTITLE 1,outliers (IQR rule)
#The IQR rule (or 1.5 × IQR rule) identifies potential outliers in a data set by setting lower and upper boundaries outside the middle 50% of the data

num_cols = ["Age", "Length_of_Stay", "Number_of_Diagnoses",
            "Blood_Pressure", "Blood_Sugar_Levels", "Previous_Admissions"]
#Q₁ (First Quartile): The 25th percentile (median of the lower half of the data).Q₃ (Third Quartile): The 75th percentile (median of the upper half of the data)
for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum()
    print(c, "outliers:", n_out)

df[num_cols].plot(kind="box", subplots=True, layout=(2, 3), figsize=(12, 6))
plt.tight_layout(); 
plt.show()

# COMMAND ----------

# DBTITLE 1,Observations - Outliers (IQR rule)
# MAGIC %md
# MAGIC **Observations — Outliers (IQR rule)**
# MAGIC
# MAGIC * No outliers were detected in any of the six numeric columns: every column has 0 records outside the 1.5 × IQR fences.
# MAGIC * The box plots confirm this — whiskers span the full min–max ranges with no individual points beyond them.
# MAGIC * Together with the earlier valid-ranges check, the data appears pre-cleaned or synthetically generated, so no outlier treatment (removal or capping) is needed before modelling.

# COMMAND ----------

# DBTITLE 1,encode target and split (split before scaling)
df["Readmission_Flag"] = (df["Readmission"] == "Yes").astype(int)

X = df.drop(columns=["Patient_ID", "Readmission", "Readmission_Flag"])
y = df["Readmission_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape, X_test.shape)
print("Train balance:\n", y_train.value_counts(normalize=True))
print("Test balance:\n", y_test.value_counts(normalize=True))

# COMMAND ----------

# DBTITLE 1,Observations - Target encoding and split
# MAGIC %md
# MAGIC **Observations — Target encoding and split**
# MAGIC
# MAGIC * The target is encoded as `Readmission_Flag` (1 = Yes, 0 = No), and `Patient_ID`, `Readmission` and `Readmission_Flag` are all excluded from the features, leaving 8 predictor columns.
# MAGIC * The 80/20 split gives 2,400 training and 600 test admissions (2,400 + 600 = 3,000, matching the full dataset).
# MAGIC * `stratify=y` preserves the 71/29 class ratio almost exactly in both sets (71.1/28.9 train vs 71.2/28.8 test), so no information about class balance is lost to the split.
# MAGIC * `random_state=42` makes the split reproducible on every rerun.

# COMMAND ----------

#preprocessing pipeline (fits on training data only inside the models)
#one-hot (not 0/1/2) because Gender and Admission Type have no order. 
# Standard scaling helps logistic regression, whereas trees and forests don’t need it. 
# The imputer is there so the pipeline can be rerun on next month’s export.

cat_cols = ["Gender", "Admission_Type"]

def make_prep(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]), cat),
    ])

make_prep(num_cols, cat_cols)

# COMMAND ----------

# DBTITLE 1,Task 2 heading
# MAGIC %md
# MAGIC # Task 2: Exploratory data analysis

# COMMAND ----------

# DBTITLE 1,target balance and descriptive table
train = X_train.copy()
train["Readmission_Flag"] = y_train

print(y_train.value_counts())
print(y_train.value_counts(normalize=True))
display(train[num_cols].describe().round(2))

y_train.value_counts().plot(kind="bar")
plt.title("Figure 1: Readmission count (training set)")
plt.xlabel("Readmitted (0 = No, 1 = Yes)"); plt.ylabel("Number of patients")
plt.show()

# COMMAND ----------

# DBTITLE 1,Observations - Figure 1
# MAGIC %md
# MAGIC **Observations — Figure 1: Readmission count (training set)**
# MAGIC
# MAGIC * For every 100 patients in the training data, about 71 did not come back and about 29 did. So being readmitted is the less common outcome.
# MAGIC * This matters when we judge the models later: always guessing "no readmission" would already be right 7 times out of 10. So a useful model has to do better than 71% accuracy to prove it learned anything.
# MAGIC * The averages for the training patients look just like the full dataset (age around 54, stays around 15 days, about 2 previous admissions), so the training set is a fair sample — nothing odd happened in the split.

# COMMAND ----------

# DBTITLE 1,train = X_train.copy() train["Readmission_Flag"] = y_train  print(y_train.value_counts()) print(y_train.value_counts(normalize=True)) display(train[num_cols].describe().round(2))  y_train.value_counts().plot(kind="bar") plt.title("Figure 1: Readmission count (training set)") plt.xlabel("Readmitted (0 = No, 1 = Yes)"); plt.ylabel("Number of patients") plt.show()
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.ravel(), num_cols):
    sns.histplot(train[c], bins=20, ax=ax)
    ax.set_title(c); ax.set_ylabel("Count")
plt.suptitle("Figure 2: Distribution of numeric variables")
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Observations - Figure 2
# MAGIC %md
# MAGIC **Observations — Figure 2: Distribution of numeric variables**
# MAGIC
# MAGIC * None of the six numeric measures pile up in one place or have strange gaps — patient ages, stay lengths, blood pressure, blood sugar, number of diagnoses and previous admissions are all spread fairly evenly across their realistic ranges.
# MAGIC * In everyday terms: the training set contains a natural mix of patients — young and old, short stays and long stays, mild and complex cases — rather than one dominant type.
# MAGIC * Because the values are evenly spread, there are no extreme spikes to distort the models, and no single "typical patient" the data is biased towards.

# COMMAND ----------

# DBTITLE 1,readmission rate by category
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, c in zip(axes, ["Gender", "Admission_Type", "Number_of_Diagnoses", "Previous_Admissions"]):
    train.groupby(c)["Readmission_Flag"].mean().plot(kind="bar", ax=ax)
    ax.set_title(f"Readmission rate by {c}"); ax.set_ylabel("Rate")
plt.suptitle("Figure 3: Readmission rate by group")
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Observations - Figure 3
# MAGIC %md
# MAGIC **Observations — Figure 3: Readmission rate by group**
# MAGIC
# MAGIC * The differences between groups are small. Men come back slightly more often than women (about 30% vs 28%), and elective patients slightly more often than emergency ones (about 30% vs 27%).
# MAGIC * As the number of diagnoses grows, the readmission rate edges up (from about 25% to 33%), and previous admissions show a similar mild pattern (27% to 31%).
# MAGIC * In plain terms: no single factor on its own strongly separates patients who come back from patients who don't. The groups overlap heavily, so the models will need to combine several weak signals — which is a realistic but harder prediction problem.

# COMMAND ----------

# DBTITLE 1,box plots by outcome
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, c in zip(axes, ["Age", "Length_of_Stay", "Blood_Pressure", "Blood_Sugar_Levels"]):
    sns.boxplot(x="Readmission_Flag", y=c, data=train, ax=ax)
    ax.set_xlabel("Readmitted (0 = No, 1 = Yes)")
plt.suptitle("Figure 4: Numeric variables by readmission outcome")
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Observations - Figure 4
# MAGIC %md
# MAGIC **Observations — Figure 4: Numeric variables by readmission outcome**
# MAGIC
# MAGIC * For every measurement, the boxes for patients who came back (1) and patients who did not (0) sit at almost the same height. Their medians, spreads and ranges overlap almost completely.
# MAGIC * In plain language: on paper, a patient who gets readmitted looks statistically like one who doesn't — no single measurement visibly separates the two groups.
# MAGIC * This tells us any useful prediction will have to come from subtle combinations of factors rather than one obvious red flag.

# COMMAND ----------

# DBTITLE 1,correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(train[num_cols + ["Readmission_Flag"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Figure 5: Correlation matrix")
plt.show()

#class balance, which variables look related to the outcome and how strongly, and any redundant pairs. Don’t call a correlation of about 0.03 a “relationship”.

# COMMAND ----------

# DBTITLE 1,Observations - Figure 5
# MAGIC %md
# MAGIC **Observations — Figure 5: Correlation matrix**
# MAGIC
# MAGIC * Every correlation between a measurement and readmission is essentially zero — the closest is Age at about 0.03, and as the brief warns, that is not a relationship worth calling one.
# MAGIC * No two measurements tell the same story either: the strongest pair (Length_of_Stay and Number_of_Diagnoses) is only about 0.03, so there are no redundant columns to drop.
# MAGIC * In plain terms: none of these numbers, alone or in pairs, meaningfully drives readmission. Combined with the small group differences in Figure 3, we should expect the models to find only modest signal — and we should be careful not to over-claim any single factor as a cause of readmission.
# MAGIC * Gender and Admission Type are deliberately left out of this matrix because correlation only works on numbers — they are text categories. Their link to readmission is measured the proper way instead: group rates in Figure 3 and a chi-square test in Task 3 (Gender p = 0.31, well within chance).

# COMMAND ----------

# MAGIC %md
# MAGIC Task 3: Feature selection

# COMMAND ----------

df["Readmission"] = df["Readmission"].map({
    "Yes": 1,
    "No": 0
})

#Convert Readmission into numerical values

numerical_df = df.select_dtypes(include=["int64", "float64"])

# COMMAND ----------

correlation_matrix = numerical_df.corr(method="pearson")
display(correlation_matrix)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Pearson Correlation Matrix")
plt.show()

# COMMAND ----------

target_correlation = correlation_matrix["Readmission"]

display(target_correlation.sort_values(ascending=False))
#we want to know which features are associated with the target variable.

# COMMAND ----------

# DBTITLE 1,Identify weak and strong correlations
weak_features = target_correlation[
    target_correlation.abs() < 0.10
]

strong_features = target_correlation[
    target_correlation.abs() >= 0.30
]

print("Weakly correlated features:")
display(weak_features)

print("Strongly correlated features:")
display(strong_features)

# COMMAND ----------

# MAGIC %md
# MAGIC The Pearson correlation analysis was used to examine the linear relationship between the numerical features and the Readmission target variable. Features with correlation values close to zero showed weak linear associations with readmission. Previous Admissions and Number of Diagnoses were examined as potentially important predictors because of their observed correlations. Age and Gender were also evaluated to determine their statistical contribution.
# MAGIC
# MAGIC Features were not removed based on correlation alone. Further feature selection and model performance testing were used to determine which variables should be retained.

# COMMAND ----------

# DBTITLE 1,display candidate features
numerical_column = df[num_cols + ["Readmission"]]
features = numerical_column.drop("Readmission", axis=1)
y = numerical_column["Readmission"]
display(num_cols + cat_cols)

print("numerical_column — first 5 rows:")
display(numerical_column.head())

print("features — first 5 rows:")
display(features.head())

print("y — target values:")
display(y.value_counts())

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,X and y for feature importance
# Readmission_Flag is a copy of the target and Patient_ID is only an identifier — both must be
# excluded, otherwise the models "cheat" by seeing the answer (target leakage).
X = df.drop(columns=["Readmission", "Readmission_Flag", "Patient_ID"])

y = df["Readmission"]

# COMMAND ----------

X = pd.get_dummies(X, drop_first=True)

# COMMAND ----------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X, y)

# COMMAND ----------

# DBTITLE 1,Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

# COMMAND ----------

# DBTITLE 1,sort feature importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

display(feature_importance)

# COMMAND ----------

# DBTITLE 1,Visualise feature Importance
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()

# COMMAND ----------

# DBTITLE 1,Group Features
def classify_importance(score):

    if score >= 0.15:
        return "High Significance"

    elif score >= 0.05:
        return "Moderate Significance"

    else:
        return "Very Low Significance"

# COMMAND ----------

# DBTITLE 1,Applying Classification
feature_importance["Significance"] = feature_importance[
    "Importance"
].apply(classify_importance)

display(feature_importance)

# COMMAND ----------

# DBTITLE 1,Feature Selection Decision
feature_importance["Decision"] = feature_importance[
    "Significance"
].apply(
    lambda x: "Keep" if x != "Very Low Significance" else "Review"
)

display(feature_importance)

# COMMAND ----------

# DBTITLE 1,Observations - Corrected feature importance
# MAGIC %md
# MAGIC **Observations — Random Forest feature importance (corrected)**
# MAGIC
# MAGIC * This is the honest version of the importance ranking: the first run accidentally included `Readmission_Flag` — a copy of the answer — which "explained" 95.4% of the model and made every real feature look useless. With the target copy and `Patient_ID` removed, the forest now spreads importance across the eight genuine features.
# MAGIC * The four strongest are Blood Sugar Levels (0.206), Blood Pressure (0.205), Age (0.189) and Length of Stay (0.164); the two categorical flags carry the least (Admission Type 0.030, Gender 0.027) — the same broad order as the chi-square tests and the permutation importance in Figure 6.
# MAGIC * A caution for the write-up: impurity importances always sum to 1 on the training set, so "high" here only means *relatively strongest within a weak-signal dataset* — every model built on these features still scores near coin-flip AUC, as the model sections below confirm.

# COMMAND ----------

# DBTITLE 1,chi-square (categoricals) and t-test + effect size (numerics)
rows = []
for c in cat_cols:
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(X_train[c], y_train))
    rows.append({"feature": c, "test": "chi-square", "statistic": chi2, "p_value": p})



stat_results = pd.DataFrame(rows)
display(stat_results.round(4))

# COMMAND ----------

# DBTITLE 1,Observations - Chi-square tests
# MAGIC %md
# MAGIC **Observations — Chi-square tests (categorical features)**
# MAGIC
# MAGIC * Neither Gender (p = 0.31) nor Admission Type (p = 0.11) shows a statistically meaningful link with readmission — both p-values sit well above the usual 0.05 threshold.
# MAGIC * In plain language: whether a patient is male or female, or came in electively rather than as an emergency, does not meaningfully change the chance of coming back. The small rate gaps seen in Figure 3 are within what chance alone would produce.

# COMMAND ----------

# DBTITLE 1,t-test + effect size (numerics)
for c in num_cols:
    a = X_train.loc[y_train == 1, c]; b = X_train.loc[y_train == 0, c]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)   # Cohen's d = effect size
    rows.append({"feature": c, "test": "t-test", "statistic": t, "p_value": p, "effect_size_d": d})

stat_results = pd.DataFrame(rows)
display(stat_results.round(4))

# COMMAND ----------

# DBTITLE 1,Observations - T-tests
# MAGIC %md
# MAGIC **Observations — T-tests with effect size (numeric features)**
# MAGIC
# MAGIC * On every numeric measurement, readmitted and non-readmitted patients are statistically indistinguishable: all p-values range from 0.22 to 0.91, far above the 0.05 threshold.
# MAGIC * The effect sizes are tiny — Cohen's d is at most 0.06 (Age), with the rest around 0.01–0.05. For context, 0.2 is considered a *small* effect, so these are negligible.
# MAGIC * In plain language: no single feature passes a statistical test for a link with readmission. Combined with the chi-square results above, this confirms the EDA story — the models will have to look for subtle combinations of factors, and we should expect only modest predictive power (consistent with Figures 4 and 5).

# COMMAND ----------

# DBTITLE 1,mutual information and permutation importance
X_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)

mi = pd.Series(mutual_info_classif(X_enc, y_train, random_state=42), index=X_enc.columns)

rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42,
                            class_weight="balanced").fit(X_enc, y_train)
perm = permutation_importance(rf, X_enc, y_train, n_repeats=10, random_state=42, scoring="roc_auc")
perm_imp = pd.Series(perm.importances_mean, index=X_enc.columns)

importance = pd.DataFrame({"mutual_info": mi, "perm_importance": perm_imp}).sort_values("perm_importance")
display(importance.round(4))

importance["perm_importance"].plot(kind="barh", figsize=(7, 4))
plt.title("Figure 6: Permutation importance (training set)")
plt.xlabel("Drop in ROC-AUC when shuffled")
plt.show()

# COMMAND ----------

# DBTITLE 1,Observations - Figure 6
# MAGIC %md
# MAGIC **Observations — Figure 6: Mutual information and permutation importance**
# MAGIC
# MAGIC * Permutation importance tells us how much the model's score drops when a feature's values are shuffled. The biggest drop is Blood Sugar Levels (about 0.14 in ROC-AUC), followed by Length of Stay (0.11) — but even the top feature is worth only ~0.14 AUC on a model sitting near the 0.5 coin-flip baseline, so no single feature carries the prediction.
# MAGIC * The two categorical dummies rank last (Gender_Male about 0.03, Admission_Type_Emergency about 0.03), which matches the chi-square verdict that Gender and Admission Type carry almost no signal.
# MAGIC * Mutual information is essentially zero for most features (highest is Number_of_Diagnoses at only 0.013) — a second, independent method confirming there is barely any one-to-one relationship with readmission.
# MAGIC * Reading both methods together with the t-tests and chi-square above: every feature contributes a little, none contributes a lot, so the sensible call is to keep all eight features and let the models look for combinations. One caution: these importances are measured on the *training* set, where the forest can lean on noise, so treat the ranking as indicative rather than exact.

# COMMAND ----------

# DBTITLE 1,decide and set final features
#Patient ID is dropped because it is an identifier, not a patient characteristic

final_num = ["Age", "Length_of_Stay", "Number_of_Diagnoses",
             "Blood_Pressure", "Blood_Sugar_Levels", "Previous_Admissions"]   # edit after analysis
final_cat = ["Gender", "Admission_Type"]                                  # edit after analysis
final_features = final_num + final_cat

# COMMAND ----------

# DBTITLE 1,Observations - Final feature set
# MAGIC %md
# MAGIC **Observations — Final feature set**
# MAGIC
# MAGIC * The final feature set keeps all eight candidate features: six numeric measurements (Age, Length_of_Stay, Number_of_Diagnoses, Blood_Pressure, Blood_Sugar_Levels, Previous_Admissions) plus the two categorical ones (Gender, Admission_Type).
# MAGIC * Patient ID is the only column dropped — it is just an identifier with no clinical meaning, and keeping it would risk the model memorising patients rather than learning patterns.
# MAGIC * Why keep everything else? The statistical tests and Figure 6 showed that no single feature is useless either: every one carries at least a little signal, and none is so weak that removing it would simplify the model without losing information.
# MAGIC * In plain language: with a weak-signal dataset like this, the safest choice is to let the models weigh all eight factors and find combinations themselves rather than pre-deciding which single factor matters.

# COMMAND ----------

# MAGIC %md
# MAGIC Task 4: Model building

# COMMAND ----------

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# COMMAND ----------

# DBTITLE 1,Separate features
#Seperate features
# Readmission_Flag is a copy of the target and Patient_ID is only an identifier — both must be
# excluded, otherwise the models "cheat" by seeing the answer (target leakage).
X = df.drop(columns=["Readmission", "Readmission_Flag", "Patient_ID"])

y = df["Readmission"]

# COMMAND ----------

# DBTITLE 1,Encode  Categories
X = pd.get_dummies(X, drop_first=True)

# COMMAND ----------

# DBTITLE 1,Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# COMMAND ----------

# DBTITLE 1,Baseline Classifier
baseline_model = DummyClassifier(
    strategy="stratified",
    random_state=42
)

# COMMAND ----------

# DBTITLE 1,Train Baseline Model
baseline_model.fit(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Generate Predictions
baseline_predictions = baseline_model.predict(X_test)

# COMMAND ----------

# DBTITLE 1,Baseline Accuracy
baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("Baseline Accuracy:", round(baseline_accuracy * 100, 2), "%")

# COMMAND ----------

print(classification_report(
    y_test,
    baseline_predictions
))

# COMMAND ----------

# DBTITLE 1,Confusion Matrics
cm = confusion_matrix(
    y_test,
    baseline_predictions
)

display(pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
))

# COMMAND ----------

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Baseline Model Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.show()

# COMMAND ----------

# DBTITLE 1,baseline narrative
# MAGIC %md
# MAGIC A Stratified Dummy Classifier was implemented to establish a baseline for evaluating the performance of the machine learning models. The classifier generated predictions based on the distribution of the target classes without using patient-related features.
# MAGIC The dataset was divided into training and testing sets using an 80:20 split, with stratification applied to preserve the proportion of readmitted and non-readmitted patients.
# MAGIC The baseline achieved an accuracy of approximately 58.67%. This benchmark provides a reference point against which the performance of the more advanced classification models can be compared.

# COMMAND ----------

# DBTITLE 1,baseline and candidates on the same folds

# Imbalance is handled with class_weight="balanced" because it is simple and leak-free. .
prep = make_prep(final_num, final_cat)
Xtr = df.loc[y_train.index, final_features]   # from df directly: stays valid after the later re-split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

models = {
    "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = []
for name, clf in models.items():
    pipe = Pipeline([("prep", make_prep(final_num, final_cat)), ("clf", clf)])
    s = cross_val_score(pipe, Xtr, y_train, cv=cv, scoring="roc_auc")
    results.append({"model": name, "cv_auc_mean": s.mean(), "cv_auc_std": s.std()})

display(pd.DataFrame(results).round(3))

# COMMAND ----------

# DBTITLE 1,Observations - Baseline and candidates
# MAGIC %md
# MAGIC **Observations — Baseline and candidate models (cross-validated AUC)**
# MAGIC
# MAGIC * All four real models score within ±0.02 of the 0.500 baseline: Logistic Regression 0.494, Decision Tree 0.506, Random Forest 0.522, K-Nearest Neighbors 0.523 — none shows clear predictive skill yet.
# MAGIC * K-Nearest Neighbors (k=5, newly added) posts the best CV mean (0.523), but its 0.001 edge over the Random Forest is far inside the fold-to-fold noise — it is another coin-flip candidate, not a leader, and must still prove itself on the test set.
# MAGIC * The fold-to-fold variation (standard deviations of 0.013–0.026) is as large as the gaps between the models, so the differences among them are not meaningful — the Random Forest's small lead is within noise.
# MAGIC * The baseline's exact 0.500 with zero spread is expected: it always predicts the majority class, so it is the yardstick everything else must clearly beat.
# MAGIC * This matches the story so far: with every feature carrying only a whisper of signal, no model family can conjure strong performance — and the class imbalance is handled inside each model with `class_weight="balanced"`, which is simple and leak-free.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# COMMAND ----------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

# COMMAND ----------

# DBTITLE 1,Training the 3 models
for name, model in models.items():

    model.fit(X_train, y_train)

    print(name, "training completed")

# COMMAND ----------

# DBTITLE 1,Predictions
predictions = {}

for name, model in models.items():

    predictions[name] = model.predict(X_test)

# COMMAND ----------

# DBTITLE 1,Evaluate Model Performance
results = []

for name, prediction in predictions.items():

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test, prediction, zero_division=0
    )

    recall = recall_score(
        y_test, prediction, zero_division=0
    )

    f1 = f1_score(
        y_test, prediction, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

# COMMAND ----------

# DBTITLE 1,Comparing Models
comparison_df = pd.DataFrame(results)

display(
    comparison_df.sort_values(
        by="F1-Score",
        ascending=False
    )
)

# COMMAND ----------

# DBTITLE 1,Visualising
plt.figure(figsize=(10, 6))

comparison_melted = comparison_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

sns.barplot(
    data=comparison_melted,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Performance Score")
plt.xlabel("Classification Models")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.show()

# COMMAND ----------

# DBTITLE 1,Observations - Simple-model comparison (leakage corrected)
# MAGIC %md
# MAGIC **Observations — Simple-model comparison (leakage corrected)**
# MAGIC
# MAGIC * Before the fix this table showed impossible perfect 1.0 scores for all three models — that was target leakage (`Readmission_Flag`, a copy of the target, was sitting inside the features). With the leak removed, every model is honestly ordinary again.
# MAGIC * The plain Decision Tree is the best of the three (accuracy 0.595, precision 0.299, recall 0.301, F1 0.300); plain KNN catches only 11.6% of readmissions (recall 0.116); plain Logistic Regression degenerates into always predicting "No" (accuracy 0.712 with zero recall) because the 71/29 class imbalance is no longer counter-weighted.
# MAGIC * These untuned models are a sanity check, not the final word — the class-weighted, cross-validated and tuned pipeline in the surrounding cells is what the final model choice is based on.

# COMMAND ----------

# DBTITLE 1,Observations - Test-set evaluation of the simple models
# MAGIC %md
# MAGIC **Observations — Test-set evaluation of the simple models**
# MAGIC
# MAGIC * On unseen data the plain Decision Tree again leads (ROC-AUC 0.507), ahead of the stratified baseline (0.500), Logistic Regression (0.492) and K-Nearest Neighbors (0.490) — effectively a four-way tie around the coin-flip line.
# MAGIC * KNN's recall of 0.116 means it catches only about 1 in 9 actual readmissions — the worst catch-rate of any model in the notebook — which rules it out for a screening task where missed patients are the costly error.
# MAGIC * These numbers also confirm the leak fix from the other side: no model is anywhere near the impossible 1.0 it scored before the correction.

# COMMAND ----------

best_model = comparison_df.loc[
    comparison_df["F1-Score"].idxmax()
]

print("Best Performing Model:", best_model["Model"])

print("Accuracy:", round(best_model["Accuracy"], 3))

print("Precision:", round(best_model["Precision"], 3))

print("Recall:", round(best_model["Recall"], 3))

print("F1-Score:", round(best_model["F1-Score"], 3))

# COMMAND ----------

# DBTITLE 1,Stratifield Fold Comparing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# COMMAND ----------

cv_results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"]
    )

    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-Score": scores["test_f1"].mean()
    })

# COMMAND ----------

cv_comparison = pd.DataFrame(cv_results)

display(cv_comparison.sort_values(
    by="F1-Score",
    ascending=False
))

# COMMAND ----------

# DBTITLE 1,tune at least two models (training data only)
def tune(clf, grid):
    pipe = Pipeline([("prep", make_prep(final_num, final_cat)), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc")
    gs.fit(Xtr, y_train)
    print(gs.best_params_, round(gs.best_score_, 3))
    return gs

gs_lr = tune(LogisticRegression(max_iter=1000, class_weight="balanced"),
             {"clf__C": [0.01, 0.1, 1, 10]})

gs_tree = tune(DecisionTreeClassifier(class_weight="balanced", random_state=42),
               {"clf__max_depth": [3, 5, 7], "clf__min_samples_leaf": [10, 30, 50]})

gs_rf = tune(RandomForestClassifier(class_weight="balanced", random_state=42),
             {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, 8],
              "clf__min_samples_leaf": [10, 30]})

# COMMAND ----------

# DBTITLE 1,Observations - Hyperparameter tuning
# MAGIC %md
# MAGIC **Observations — Hyperparameter tuning (training data only)**
# MAGIC
# MAGIC * Grid search picked: Logistic Regression C = 10 (CV AUC 0.494), Decision Tree depth 7 / min samples per leaf 30 (0.522), Random Forest 100 trees / depth 8 / leaf 30 (0.519).
# MAGIC * Tuning barely moves the needle: every tuned model still sits in the 0.49–0.52 band, essentially coin-flip territory — no hidden parameter combination rescues the weak signal.
# MAGIC * The tuned Decision Tree and Random Forest converge to nearly the same score (~0.52), suggesting both are extracting whatever little pattern exists; the simpler tree edges ahead by a hair.
# MAGIC * Since the tuned Decision Tree holds the best cross-validated AUC (0.522), it is carried forward as the final model in the evaluation section — a sensible choice, as the simpler tree is also easier to explain to clinicians than a forest.

# COMMAND ----------

# MAGIC %md
# MAGIC Task 5: Evaluation

# COMMAND ----------

# DBTITLE 1,Evaluation Libraries
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Evaluation Models
evaluation_models = {
    "Baseline Classifier": baseline_model,
    "Logistic Regression": models["Logistic Regression"],
    "K-Nearest Neighbors": models["K-Nearest Neighbors"],
    "Decision Tree Classifier": models["Decision Tree"]
}

# COMMAND ----------

# DBTITLE 1,Evaluation Metrics
evaluation_results = []

for name, model in evaluation_models.items():

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    evaluation_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

# COMMAND ----------

evaluation_df = pd.DataFrame(evaluation_results)

display(
    evaluation_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )
)

# COMMAND ----------

for name, model in evaluation_models.items():

    predictions = model.predict(X_test)

    print("\n", name)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

# COMMAND ----------

# DBTITLE 1,Confusion Matrics
for name, model in evaluation_models.items():

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(name)

    display(pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    ))

# COMMAND ----------

# DBTITLE 1,test metrics for all tuned models plus the baseline

#The train_auc versus roc_auc columns give you the overfitting comparison. A large gap means overfitting.

dummy = Pipeline([("prep", make_prep(final_num, final_cat)),
                  ("clf", DummyClassifier(strategy="most_frequent"))]).fit(Xtr, y_train)

candidates = {"Baseline": dummy, "Logistic Regression": gs_lr.best_estimator_,
              "Decision Tree": gs_tree.best_estimator_, "Random Forest": gs_rf.best_estimator_}
Xte = df.loc[y_test.index, final_features]   # same rows as X_test but keeps the raw columns the prep pipeline expects

rows = []
for name, m in candidates.items():
    pred = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
    rows.append({"model": name,
                 "accuracy": accuracy_score(y_test, pred),
                 "precision": precision_score(y_test, pred, zero_division=0),
                 "recall": recall_score(y_test, pred),
                 "f1": f1_score(y_test, pred),
                 "roc_auc": roc_auc_score(y_test, proba),
                 "train_auc": roc_auc_score(y_train, m.predict_proba(Xtr)[:, 1])})
display(pd.DataFrame(rows).round(3))

# COMMAND ----------

# DBTITLE 1,Observations - Test-set metrics
# MAGIC %md
# MAGIC **Observations — Test-set metrics for all models (with overfitting check)**
# MAGIC
# MAGIC * On unseen data every model lands back at coin-flip AUC: Logistic Regression 0.492, Decision Tree 0.513, Random Forest 0.475 — the small cross-validation leads did not survive contact with the test set.
# MAGIC * The `train_auc` vs `roc_auc` columns expose overfitting: the Random Forest scores 0.782 on its own training data but only 0.475 on test data — a 0.31 gap, meaning it largely memorised noise. The Decision Tree shows a 0.16 gap (0.67 vs 0.513). Logistic Regression has the smallest gap (0.534 vs 0.492), making it the most honest model — it just has nothing to find.
# MAGIC * The baseline's 0.712 "accuracy" is an illusion of the class imbalance: always predicting "No" is right 71% of the time but finds zero readmitted patients (precision, recall and F1 all 0).
# MAGIC * The Decision Tree is the best of a weak bunch: the highest test AUC (0.513) and by far the best recall (0.549) — it catches about half of true readmissions, which matters if a missed at-risk patient is the costly error.

# COMMAND ----------

# DBTITLE 1,threshold analysis

#A missed at-risk patient (false negative) is usually costlier than an unnecessary follow-up call, so favouring recall is a common justification. 
#all models score barely above the 0.50 baseline AUC on this dataset, so threshold tuning is operating on a weak signal 

final_model = candidates["Decision Tree"]   # best tuned CV AUC (0.522); defined here since no earlier cell assigns it

proba = final_model.predict_proba(Xte)[:, 1]
rows = []
for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
    pred = (proba >= t).astype(int)
    rows.append({"threshold": t,
                 "precision": precision_score(y_test, pred, zero_division=0),
                 "recall": recall_score(y_test, pred)})
display(pd.DataFrame(rows).round(3))

# COMMAND ----------

# DBTITLE 1,Observations - Threshold analysis
# MAGIC %md
# MAGIC **Observations — Threshold analysis**
# MAGIC
# MAGIC * Lowering the decision threshold trades precision for recall exactly as expected: at 0.2 the model catches 93% of true readmissions, while at 0.6 recall collapses to 0.23.
# MAGIC * Precision stays remarkably flat (~0.29) across every threshold — a symptom of the weak signal: loosening the bar admits more true positives and false positives in the same proportion, so there is no "sweet spot" to find.
# MAGIC * Since a missed at-risk patient (false negative) is usually costlier than an unnecessary follow-up call, a threshold below 0.5 — say 0.3, with recall 0.87 — is the defensible operating point. But with AUC ≈ 0.51, no threshold can turn this model into a reliable predictor; tuning only redistributes its errors.

# COMMAND ----------

# DBTITLE 1,learning curve (overfitting check)
from sklearn.model_selection import learning_curve

sizes, tr, va = learning_curve(final_model, Xtr, y_train, cv=cv, scoring="roc_auc",
                               train_sizes=np.linspace(0.2, 1.0, 5))
plt.plot(sizes, tr.mean(axis=1), label="Training")
plt.plot(sizes, va.mean(axis=1), label="Validation")
plt.xlabel("Training set size (patients)"); plt.ylabel("ROC-AUC")
plt.title("Figure 10: Learning curve"); plt.legend(); plt.show()

#what the model can’t tell the hospital, which extra data would help (diagnosis codes, medications, discharge details), and the ethical issues, such as fairness across groups and not letting a score replace clinical judgement.

# COMMAND ----------

# DBTITLE 1,Observations - Figure 10
# MAGIC %md
# MAGIC **Observations — Figure 10: Learning curve**
# MAGIC
# MAGIC * Training AUC starts high (~0.71 with 400 patients) and drifts down to ~0.68 as data grows, while validation AUC stays pinned near 0.50–0.52 the whole way.
# MAGIC * The persistent ~0.15–0.2 gap between the two curves is the signature of overfitting: the tree keeps finding patterns in the training data that simply do not hold for new patients.
# MAGIC * More data alone will not fix this — validation performance barely improves with size and even dips mid-curve (touching ~0.50 around 1,600 patients). The bottleneck is signal, not sample size: richer features (diagnosis codes, medications, discharge details) would help far more than more rows.

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

final_model.fit(Xtr, y_train)   # Xtr keeps the raw feature columns the prep pipeline expects

y_probability = final_model.predict_proba(Xte)[:, 1]
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

# COMMAND ----------

# DBTITLE 1,Observations - Final model evaluation
# MAGIC %md
# MAGIC **Observations — Final model on the test set**
# MAGIC
# MAGIC * The final model (the tuned Decision Tree) scores: Accuracy 0.492, Precision 0.295, Recall 0.549, F1 0.384, ROC-AUC 0.513.
# MAGIC * Accuracy below 0.5 looks alarming next to the baseline's 0.712 — but the baseline earns its number by never flagging anyone. This model deliberately flags more patients in order to catch readmissions, trading raw accuracy for recall.
# MAGIC * The honest summary is ROC-AUC 0.513: barely better than a coin flip. The model can serve only as a weak screening aid, never as a decision-maker.
# MAGIC * Per the classification report, the "Yes" (readmitted) class is where performance is weakest: precision ≈ 0.30 means roughly 7 of every 10 patients flagged would not actually have been readmitted.

# COMMAND ----------

# DBTITLE 1,from sklearn.metrics import (     accuracy_score,     precision_score,     recall_score,     f1_score,     roc_auc_score,     classification_report )  final_model.fit(X_train, y_train)  y_probability = final_model.predict_proba(X_test)[:, 1] y_prediction = (y_probability >= 0.5).astype(int)  metrics = {     "Accuracy": accuracy_score(y_test, y_prediction),     "Precision": precision_score(y_test,
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

# DBTITLE 1,Observations - Figure 7
# MAGIC %md
# MAGIC **Observations — Figure 7: Confusion matrix on the test set**
# MAGIC
# MAGIC * Of the 600 test patients: 200 were correctly cleared (true negatives), 227 were false alarms, 78 readmissions were missed, and 95 were correctly caught (true positives).
# MAGIC * The model catches 95 of 173 true readmissions (recall ≈ 0.55) — but fires 227 false alarms to do it, so precision is only ≈ 0.30: roughly 7 of every 10 alerts are unnecessary follow-ups.
# MAGIC * It also misses 78 patients who genuinely came back — with such a weak signal, catching more inevitably means flagging more.
# MAGIC * In plain language: as a screening tool it leans toward over-alerting, which is arguably the right direction when a missed patient is the costlier error — but 227 false alarms out of 322 alerts would limit its practical usefulness in a real hospital workflow.

# COMMAND ----------

best_model = evaluation_df.loc[
    evaluation_df["ROC-AUC"].idxmax()
]

print("Best Model:", best_model["Model"])

print("Accuracy:", round(best_model["Accuracy"], 3))

print("Recall:", round(best_model["Recall"], 3))

print("ROC-AUC:", round(best_model["ROC-AUC"], 3))

# COMMAND ----------

# DBTITLE 1,justified decision threshold (F2 sweep)
# Justifying the decision threshold with a recall-weighted score (F2).
# F2 weights recall twice as much as precision, matching the clinical view that
# a missed at-risk patient (false negative) costs more than an unnecessary follow-up call.
from sklearn.metrics import fbeta_score

proba_test = final_model.predict_proba(Xte)[:, 1]
rows = []
for t in np.arange(0.05, 0.96, 0.05):
    pred = (proba_test >= t).astype(int)
    rows.append({
        "threshold": round(float(t), 2),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "F2 (recall-weighted)": fbeta_score(y_test, pred, beta=2),
        "patients flagged": int(pred.sum()),
    })
thr = pd.DataFrame(rows)
display(thr.round(3))
best = thr.loc[thr["F2 (recall-weighted)"].idxmax(), "threshold"]
print(f"Threshold justified by best F2: {best}")

# COMMAND ----------

# DBTITLE 1,Observations - Justified threshold
# MAGIC %md
# MAGIC **Observations — Justified decision threshold (F2 sweep)**
# MAGIC
# MAGIC * The sweep exposes an uncomfortable truth: precision is stuck near 0.29 at every threshold, so a pure recall-weighted score (F2) keeps rising as the bar drops — its mathematical "optimum" is threshold 0.05, which simply flags all 600 patients, catches everything, and helps no one prioritise.
# MAGIC * A defensible operating point must balance catch-rate against alert volume: threshold 0.3 keeps recall at 0.87 but flags 518 of 600 patients (86% of the ward); 0.5 flags 322 (54%) and still catches 55% of readmissions.
# MAGIC * In plain language: on this dataset no threshold is statistically *good* — the choice is clinical, not mathematical (how many follow-up calls can staff actually make?). Note also that this sweep was judged on the test set for demonstration; in production the threshold must be fixed on a separate validation set first.

# COMMAND ----------

# DBTITLE 1,train vs test gap per model
# Train vs test gap: train_auc minus test_auc. A large gap means the model
# memorised the training data instead of learning patterns that generalise (overfitting).
rows = []
for name, m in candidates.items():
    tr_auc = roc_auc_score(y_train, m.predict_proba(Xtr)[:, 1])
    te_auc = roc_auc_score(y_test, m.predict_proba(Xte)[:, 1])
    rows.append({"model": name, "train_auc": tr_auc, "test_auc": te_auc,
                 "gap (overfitting)": tr_auc - te_auc})
display(pd.DataFrame(rows).round(3))

# COMMAND ----------

# DBTITLE 1,Observations - Train vs test gap
# MAGIC %md
# MAGIC **Observations — Train vs test gap (overfitting check)**
# MAGIC
# MAGIC * The gaps quantify overfitting directly: Random Forest 0.31 (0.782 on training data → 0.475 on test — worse than chance on new patients), Decision Tree 0.16, Logistic Regression 0.04, and the baseline 0 by construction.
# MAGIC * The pattern is textbook: the more flexible the model, the bigger the gap — flexibility lets tree ensembles memorise training idiosyncrasies that do not transfer to new patients.
# MAGIC * Logistic Regression's tiny gap shows a regularised linear model is already near its ceiling; the trees' large gaps are mostly memorised noise, which is exactly what the mitigation comparison below tries to claw back.

# COMMAND ----------

# DBTITLE 1,overfitting mitigations compared
# Overfitting mitigations: constrain the models so they cannot memorise the training
# data, then compare cross-validated AUC against the train-vs-test gap.
mitigations = {
    "current tuned tree (depth 7, leaf 30)":
        DecisionTreeClassifier(max_depth=7, min_samples_leaf=30, class_weight="balanced", random_state=42),
    "shallower tree (depth 3, leaf 50)":
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=42),
    "pruned tree (ccp_alpha=0.005)":
        DecisionTreeClassifier(max_depth=7, min_samples_leaf=30, ccp_alpha=0.005, class_weight="balanced", random_state=42),
    "regularised logistic regression (C=0.1)":
        LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"),
}
rows = []
for name, clf in mitigations.items():
    pipe = Pipeline([("prep", make_prep(final_num, final_cat)), ("clf", clf)])
    cv_auc = cross_val_score(pipe, Xtr, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(Xtr, y_train)
    tr_auc = roc_auc_score(y_train, pipe.predict_proba(Xtr)[:, 1])
    te_auc = roc_auc_score(y_test, pipe.predict_proba(Xte)[:, 1])
    rows.append({"variant": name, "cv_auc": cv_auc, "train_auc": tr_auc,
                 "test_auc": te_auc, "gap (overfitting)": tr_auc - te_auc})
display(pd.DataFrame(rows).round(3))

# COMMAND ----------

# DBTITLE 1,Observations - Overfitting mitigations
# MAGIC %md
# MAGIC **Observations — Overfitting mitigations compared**
# MAGIC
# MAGIC * Constraining the tree shrinks the gap dramatically: the shallower tree (depth 3, leaf 50) cuts overfitting from 0.157 to 0.065, and aggressive pruning (ccp_alpha = 0.005) eliminates it entirely — but prunes the tree all the way down to a stump with CV AUC 0.500, i.e. no signal left at all.
# MAGIC * The trade-off is explicit: regularisation removes memorised noise *and* the little real signal together. The current tuned tree keeps the best cross-validated AUC (0.522), but pays for it with a 0.16 train-vs-test gap.
# MAGIC * With signal this weak, the mitigation choice is a judgement call: if trust and generalisation matter most, the shallower tree (CV 0.508, gap 0.065) is the more defensible final model — the marginal 0.014 of CV AUC the deeper tree buys is not worth its instability.

# COMMAND ----------

# DBTITLE 1,Data adequacy check — header
# MAGIC %md
# MAGIC ## Data Adequacy Check: Does Each Patient Appear Once, and Would More Rows Help?
# MAGIC
# MAGIC Two questions for the dataset itself:
# MAGIC
# MAGIC * Is every patient represented by exactly one row (no repeat visits that could leak information between the training and test sets)?
# MAGIC * Do the recorded features carry any real signal about readmission — i.e. would collecting more rows with the same columns improve the models?

# COMMAND ----------

# DBTITLE 1,Check patient uniqueness and feature signal
# Data adequacy check.
# (1) Patient uniqueness: if the same Patient ID appeared in both the training and
#     the test set, the models could "recognise" patients instead of learning patterns.
# (2) Feature signal: point-biserial correlation (numeric features vs the Yes/No target)
#     and chi-square tests (categorical features). A p-value below ~0.05 would mean a
#     real relationship; nothing here reaches that bar.

import pandas as pd
from scipy import stats

check_df = spark.table("workspace.default.patient_readmission_data_").toPandas()

print("Rows:", len(check_df))
print("Unique Patient IDs:", check_df["Patient ID"].nunique())
print("Every patient appears exactly once:", check_df["Patient ID"].is_unique)
print("Duplicate full rows:", check_df.duplicated().sum())

y_check = (check_df["Readmission"] == "Yes").astype(int)
print("\nReadmission rate (share of 'Yes'):", round(y_check.mean(), 3))

rows = []
for col in ["Age", "Length of Stay", "Number of Diagnoses",
            "Blood Pressure", "Blood Sugar Levels", "Previous Admissions"]:
    r, p = stats.pointbiserialr(y_check, check_df[col])
    rows.append({"feature": col, "relationship to Readmission": round(r, 3),
                 "p-value": round(p, 3)})
for col in ["Gender", "Admission Type"]:
    ct = pd.crosstab(check_df[col], y_check)
    _, p, _, _ = stats.chi2_contingency(ct)
    rows.append({"feature": col, "relationship to Readmission": "chi-square",
                 "p-value": round(p, 3)})
display(pd.DataFrame(rows))

# COMMAND ----------

# DBTITLE 1,Observations - Data adequacy check
# MAGIC %md
# MAGIC **Observations — Data adequacy check**
# MAGIC
# MAGIC * Every patient appears exactly once: 3,000 rows carry 3,000 unique Patient IDs, with 0 duplicated rows. There is no repeat-visit leakage between the training and test sets, so the weak results cannot be blamed on that.
# MAGIC * No feature has a real relationship with readmission: every numeric correlation sits between -0.007 and +0.029, and no p-value comes close to the 0.05 significance bar (the closest is Age at 0.106; Gender 0.438 and Admission Type 0.555 are just as empty). Readmission behaves as if randomly assigned relative to these eight features.
# MAGIC * The consequence for "would more rows help?": no. More rows drawn from the same eight columns would only push the models more firmly toward the ~0.50 coin-flip AUC they already sit at — more data cannot create signal that the columns do not contain. What would genuinely help is new columns: diagnosis codes, medications, lab results, and discharge information — the real-world drivers this dataset lacks.

# COMMAND ----------

# DBTITLE 1,Accuracy ceiling — header
# MAGIC %md
# MAGIC ## Accuracy Ceiling: Can Any Model Reach 80%?
# MAGIC
# MAGIC A natural follow-up to the data adequacy check: the models score in the 0.49–0.60 accuracy range — could better tuning push accuracy past 80%?
# MAGIC
# MAGIC This analysis gives every model its best possible shot: train the strongest candidates, sweep **every** decision threshold from 0.02 to 0.98, and keep the highest accuracy each one can achieve. If the best of all that equals the "always say No" baseline, the columns contain nothing a model can convert into extra accuracy.

# COMMAND ----------

# DBTITLE 1,Sweep every threshold for the best achievable accuracy
# Accuracy ceiling analysis: can any model built from these columns reach 80%?
# Self-contained (own imports, own data load) so it runs regardless of notebook state.

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ceil_df = spark.table("workspace.default.patient_readmission_data_").toPandas()
ceil_df.columns = [c.replace(" ", "_") for c in ceil_df.columns]
y_ceil = (ceil_df["Readmission"] == "Yes").astype(int)
X_ceil = ceil_df.drop(columns=["Patient_ID", "Readmission"])

Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_ceil, y_ceil, test_size=0.2, random_state=42, stratify=y_ceil
)

num_cols = ["Age", "Length_of_Stay", "Number_of_Diagnoses",
            "Blood_Pressure", "Blood_Sugar_Levels", "Previous_Admissions"]
cat_cols = ["Gender", "Admission_Type"]
prep_c = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

baseline_acc = (yte_c == 0).mean()
thresholds = np.arange(0.02, 0.99, 0.02)

models = {
    "Tuned Decision Tree (depth 7, leaf 30)":
        DecisionTreeClassifier(max_depth=7, min_samples_leaf=30, random_state=42),
    "Random Forest (200 trees)":
        RandomForestClassifier(n_estimators=200, random_state=42),
}

rows = [{"model": "Always predict 'No' (baseline)",
         "best accuracy at ANY threshold": round(baseline_acc, 3)}]
for name, clf in models.items():
    pipe = Pipeline([("prep", prep_c), ("clf", clf)])
    pipe.fit(Xtr_c, ytr_c)
    proba = pipe.predict_proba(Xte_c)[:, 1]
    best_t, best_acc = max(
        ((t, accuracy_score(yte_c, (proba >= t).astype(int))) for t in thresholds),
        key=lambda x: x[1]
    )
    rows.append({"model": name, "best accuracy at ANY threshold": round(best_acc, 3)})
display(pd.DataFrame(rows))

needed = int(0.8 * len(yte_c))
already_right = int((yte_c == 0).sum())
print(f"To reach 80% accuracy a model must get {needed} of {len(yte_c)} test patients right.")
print(f"Always saying 'No' already gets {already_right} right (every non-readmission).")
print(f"The extra {needed - already_right} must come from correctly flagging true readmissions,")
print("which requires features that actually relate to readmission - these eight do not.")

# COMMAND ----------

# DBTITLE 1,Observations - Accuracy ceiling
# MAGIC %md
# MAGIC **Observations — Accuracy ceiling analysis**
# MAGIC
# MAGIC * The sweep gives every model its best possible shot: 49 decision thresholds from 0.02 to 0.98, keeping the highest accuracy. The result: the tuned Decision Tree tops out at 0.712 and the Random Forest at 0.712 — identical to the always-say-"No" baseline of 0.712 (427 of 600 test patients).
# MAGIC * No model built from these eight columns can exceed ~71.2% accuracy, so 80% is out of reach. The arithmetic is unforgiving: 80% means 480 of 600 test patients correct, and 427 of those come free from the non-readmissions. The remaining 53 can only come from correctly flagging true readmissions — and the data adequacy check showed no feature carries that information.
# MAGIC * Any report of 80%+ accuracy on this dataset would therefore signal a methodological error, not a good model: target leakage (the earlier draft's fake 100% came from including Readmission_Flag in X) or evaluating on training data (Random Forest: 0.782 on training vs 0.475 on test — memorisation, not prediction). The honest ceiling is simply the class balance: 71% No / 29% Yes.

# COMMAND ----------

# DBTITLE 1,limitations and ethics
# MAGIC %md
# MAGIC **Limitations and ethics**
# MAGIC
# MAGIC *Data limitations*
# MAGIC * The dataset holds only 3,000 patients and eight basic characteristics. The strongest real-world drivers of readmission — diagnosis codes, medications, lab results, discharge planning, social factors — are entirely absent, so the model is blind to what matters most.
# MAGIC * Everything here describes *this* dataset. With test AUC ≈ 0.51, none of these numbers should be read as real-world readmission risk; a hospital deploying this would need its own data and a properly validated rebuild.
# MAGIC
# MAGIC *Method limitations*
# MAGIC * One single train/test split: results can shift with a different random split; repeated cross-validation or several splits would make the estimates sturdier.
# MAGIC * The threshold sweep above is judged on the test set for demonstration — in production the threshold must be chosen on a separate validation set, then confirmed once on untouched test data.
# MAGIC * Permutation importances were measured on training data, so the ranking is indicative rather than definitive.
# MAGIC
# MAGIC *Ethics and fairness*
# MAGIC * **Fairness across groups:** even though Gender and Admission Type showed no overall signal, the model must be checked for *differential* errors — e.g. does it miss readmissions more often for one group? Equal overall accuracy can hide unequal harm.
# MAGIC * **Burden of false alarms:** at the recall-friendly threshold, ~7 of 10 alerts are false. Excessive alerts can worry vulnerable patients, erode trust in the care team, and waste staff time.
# MAGIC * **Cost of misses:** the 78 missed readmissions are the ethical core — a screening tool that misses half of true cases must never be presented as assurance of safety.
# MAGIC * **Human oversight:** the score should support, never replace, clinical judgement. Follow-up decisions should remain with clinicians who can weigh context the model cannot see.
# MAGIC * **Privacy and transparency:** patients' records are used to build the score, so consent, data protection, and explainability (why was this patient flagged?) are prerequisites for deployment.

# COMMAND ----------

# DBTITLE 1,Overall evaluation and best model choice
# MAGIC %md
# MAGIC Overall evaluation and best model choice

# COMMAND ----------

# DBTITLE 1,overall evaluation table and best model
# Overall evaluation: one table combining cross-validation, the overfitting gap,
# and test-set performance for every model, so the final choice rests on all the evidence.
rows = []
for name, m in candidates.items():
    cv_auc = cross_val_score(m, Xtr, y_train, cv=cv, scoring="roc_auc").mean()
    tr_auc = roc_auc_score(y_train, m.predict_proba(Xtr)[:, 1])
    te_auc = roc_auc_score(y_test, m.predict_proba(Xte)[:, 1])
    pred = m.predict(Xte)
    rows.append({
        "model": name,
        "cv_auc": cv_auc,
        "test_auc": te_auc,
        "overfit_gap": tr_auc - te_auc,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    })
overall = pd.DataFrame(rows)
display(overall.round(3))

best = overall.loc[overall["test_auc"].idxmax(), "model"]
print(f"Best model by test AUC: {best}")

# COMMAND ----------

# DBTITLE 1,Observations - Overall evaluation
# MAGIC %md
# MAGIC **Observations — Overall evaluation table**
# MAGIC
# MAGIC * Read across all the evidence at once, the ranking is clear: the Decision Tree wins on test AUC (0.513), recall (0.549) and F1 (0.384) simultaneously — it is the only model that is best-or-near-best on every measure that matters for screening.
# MAGIC * The Random Forest's story collapses on unseen data: the best-looking training score (0.782 → gap 0.307) turns into the *worst* test AUC (0.475). Flexibility without signal is pure memorisation.
# MAGIC * Logistic Regression is the most stable (gap 0.042) but weakest (test 0.492, recall 0.474) — honest, but with nothing to find.
# MAGIC * K-Nearest Neighbors, evaluated in the simpler model section, does not change the verdict: the best cross-validated AUC (0.523) collapsed to test AUC 0.490 with recall only 0.116 — without class weighting, and with neighbours memorised rather than learned, KNN is unsuitable for this imbalanced screening problem.
# MAGIC * The Decision Tree's own caveat stands: its 0.157 overfit gap means some of its apparent skill is memorised noise — which is why the shallower tree (gap 0.065, CV 0.508) remains the trust-conscious alternative flagged in the mitigations analysis.

# COMMAND ----------

# DBTITLE 1,Overall evaluation result and best model choice
# MAGIC %md
# MAGIC **Overall evaluation result and best prediction model choice**
# MAGIC
# MAGIC *Verdict on the overall evaluation*
# MAGIC * No model in this study genuinely predicts readmission: every test AUC sits in the 0.475–0.513 band against a 0.500 coin-flip baseline, and the learning curve showed the bottleneck is missing signal in the data, not model choice. The honest overall result is: *this dataset cannot support a reliable readmission predictor*.
# MAGIC
# MAGIC *Best prediction model: the tuned Decision Tree (depth 7, min samples per leaf 30)*
# MAGIC * **Why it wins:** best test AUC (0.513), best recall (0.549 — catching 95 of 173 true readmissions), best F1 (0.384), and the best cross-validated AUC (0.522). It is also the most explainable choice for clinicians: a single tree whose flagged patients can be traced rule-by-rule.
# MAGIC * **Why not the others:** the Random Forest memorises the training set (gap 0.31) and scores below chance on test (0.475); Logistic Regression is stable but weakest on every measure; K-Nearest Neighbors posts the best cross-validated AUC (0.523) but collapses on the test set (AUC 0.490, recall 0.116) — it memorises neighbours instead of learning patterns and has no class weighting for the 71/29 imbalance.
# MAGIC * **The caveat, stated once more:** the tree's 0.157 overfit gap means part of its edge is noise. If the priority were robustness over raw numbers, the shallower tree (depth 3, leaf 50; CV 0.508, gap 0.065) is the defensible substitute.
# MAGIC
# MAGIC *How the best model should be used*
# MAGIC * Deploy, if at all, only as a **screening aid at a lowered threshold (~0.3)** to prioritise follow-up calls — never as an automated decision, and always with clinicians reviewing who gets contacted.
# MAGIC * The single most valuable next step is **not more modelling but more data**: diagnosis codes, medications, labs and discharge information would give the models real signal to work with.