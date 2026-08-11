import pandas as pd

# Load raw data
df = pd.read_csv("url_features_extracted1.csv")

print("Before cleaning:", df.shape)

# 1. Drop rows with missing label (can't train on unlabeled data)
df = df.dropna(subset=["ClassLabel"])

# 2. Drop exact duplicate rows (same URL + same features = same entry twice)
df = df.drop_duplicates()

# 3. Ensure label is proper int (0 = phishing, 1 = legitimate)
df["ClassLabel"] = df["ClassLabel"].astype(int)

# 4. Sanity check: confirm no other missing values snuck in
assert df.isnull().sum().sum() == 0, "Unexpected missing values remain"

# 5. Reset index after dropping rows
df = df.reset_index(drop=True)

print("After cleaning:", df.shape)
print(df["ClassLabel"].value_counts())

# Save cleaned version
df.to_csv("dataphishing_cleaned.csv", index=False)

from sklearn.model_selection import train_test_split

# Load your cleaned data
df = pd.read_csv("dataphishing_cleaned.csv")

# Separate URL (not a model feature — keep only for reference/debugging)
url_reference = df["URL"]

# Features and label
feature_cols = [
    "url_length", "has_ip_address", "dot_count", "https_flag",
    "url_entropy", "token_count", "subdomain_count", "query_param_count",
    "tld_length", "path_length", "has_hyphen_in_domain", "number_of_digits",
    "tld_popularity", "suspicious_file_extension", "domain_name_length",
    "percentage_numeric_chars"
]
X = df[feature_cols]
y = df["ClassLabel"]

# Stratified split — preserves the 63/37 class ratio in both sets
X_train, X_test, y_train, y_test, url_train, url_test = train_test_split(
    X, y, url_reference,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape, "  Test shape:", X_test.shape)
print("Train class balance:\n", y_train.value_counts(normalize=True))
print("Test class balance:\n", y_test.value_counts(normalize=True))

# Save splits so training/eval notebooks don't need to re-split (guarantees consistency)
X_train.to_csv("data_X_train.csv", index=False)
X_test.to_csv("data_X_test.csv", index=False)
y_train.to_csv("data_y_train.csv", index=False)
y_test.to_csv("data_y_test.csv", index=False)

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the splits saved earlier
X_train = pd.read_csv("data_X_train.csv")
X_test = pd.read_csv("data_X_test.csv")
y_train = pd.read_csv("data_y_train.csv").values.ravel()
y_test = pd.read_csv("data_y_test.csv").values.ravel()

# ---- Baseline: Logistic Regression (needs scaled features) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled = scaler.transform(X_test)          # transform test with train's scaler

log_reg = LogisticRegression(
    class_weight="balanced",   # handles the 63/37 imbalance
    max_iter=1000,
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)

# ---- Main model: Random Forest (no scaling needed) ----
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("Both models trained.")

# Save everything needed for evaluation + deployment
joblib.dump(rf, "models_random_forest.joblib")
joblib.dump(log_reg, "models_logistic_regression.joblib")
joblib.dump(scaler, "models_scaler.joblib")   # needed at inference time for LR only

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

# Load test data and saved models
X_test = pd.read_csv("data_X_test.csv")
y_test = pd.read_csv("data_y_test.csv").values.ravel()

rf = joblib.load("models_random_forest.joblib")
log_reg = joblib.load("models_logistic_regression.joblib")
scaler = joblib.load("models_scaler.joblib")

X_test_scaled = scaler.transform(X_test)

# ---- Predictions ----
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

lr_preds = log_reg.predict(X_test_scaled)
lr_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

# ---- Metrics table ----
def get_metrics(y_true, preds, probs):
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1": f1_score(y_true, preds),
        "ROC-AUC": roc_auc_score(y_true, probs),
    }

results = pd.DataFrame({
    "Logistic Regression": get_metrics(y_test, lr_preds, lr_probs),
    "Random Forest": get_metrics(y_test, rf_preds, rf_probs),
}).T

print(results.round(4))
results.round(4).to_csv("results_model_comparison.csv")

# ---- Confusion matrix (final model = Random Forest) ----
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Legitimate"])
disp.plot(cmap="Blues")
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig("results_confusion_matrix.png", dpi=150)
plt.close()

# ---- ROC curve — both models on one plot ----
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, rf_probs):.3f})")
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, lr_probs):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("results_roc_curve.png", dpi=150)
plt.close()

# ---- Feature importance (Random Forest) ----
importances = pd.Series(rf.feature_importances_, index=X_test.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, color="steelblue")
plt.title("Random Forest — Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results_feature_importance.png", dpi=150)
plt.close()

print("\nTop 5 features:")
print(importances.head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X_train = pd.read_csv("data_X_train.csv")
y_train = pd.read_csv("data_y_train.csv").values.ravel()

top_features = ["path_length", "token_count", "https_flag", "percentage_numeric_chars", "number_of_digits"]

# ---- 1. Visual check: distribution by class ----
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()
for i, feat in enumerate(top_features):
    sns.boxplot(x=y_train, y=X_train[feat], ax=axes[i])
    axes[i].set_xticklabels(["Phishing (0)", "Legitimate (1)"])
    axes[i].set_title(feat)
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig("results_feature_separability.png", dpi=150)
plt.show()

# ---- 2. Numeric check: how well does ONE feature alone classify? ----
print("Single-feature accuracy (5-fold CV, depth-1 decision stump):\n")
for feat in top_features:
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    scores = cross_val_score(stump, X_train[[feat]], y_train, cv=5, scoring="accuracy")
    print(f"{feat:30s} → {scores.mean():.4f}")

# ---- 3. Combined check: all 5 top features together, no RF, just linear-ish stump depth 3 ----
stump_combined = DecisionTreeClassifier(max_depth=3, random_state=42)
scores_combined = cross_val_score(stump_combined, X_train[top_features], y_train, cv=5, scoring="accuracy")
print(f"\n{'All 5 top features (depth-3 tree)':30s} → {scores_combined.mean():.4f}")