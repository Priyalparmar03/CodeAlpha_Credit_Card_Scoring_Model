import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    "data/german.data-numeric",
    sep=r"\s+",
    header=None
)

print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# Features & Target
# =========================

X = df.iloc[:, :-1]   # First 24 columns

y = df.iloc[:, -1]    # Last column

# Convert target:
# 1 = Good Credit
# 2 = Bad Credit

y = y.map({
    1: 1,
    2: 0
})

print("\nTarget Distribution:")
print(y.value_counts())

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Models
# =========================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

best_model = None
best_score = 0

# =========================
# Training & Evaluation
# =========================

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    roc = roc_auc_score(
        y_test,
        predictions
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    if accuracy > best_score:
        best_score = accuracy
        best_model = model

# =========================
# Save Best Model
# =========================

joblib.dump(
    best_model,
    "best_credit_model.pkl"
)

print("\nBest Model Saved Successfully!")
print(f"Best Accuracy = {best_score:.4f}")

# =========================
# Feature Importance
# =========================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance":
            best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title("Feature Importance")
    plt.tight_layout()

    plt.show()