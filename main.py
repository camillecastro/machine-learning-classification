from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# Educational machine learning classification comparison
# Dataset: Breast Cancer Wisconsin Diagnostic
# All models use the same reproducible train/test split.

OUT = Path("results")
OUT.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
}

results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, model) in zip(axes, models.items()):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_f1 = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1"
    ).mean()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "CV F1 (train)": round(cv_f1, 4),
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(precision_score(y_test, predictions), 4),
        "Recall": round(recall_score(y_test, predictions), 4),
        "F1": round(f1_score(y_test, predictions), 4)
    })

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Malignant", "Benign"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )

    ax.set_title(name)

fig.suptitle("Classification Model Comparison")
fig.tight_layout()

fig.savefig(
    OUT / "confusion_matrices.png",
    dpi=180
)

plt.close(fig)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "F1",
    ascending=False
)

results_df.to_csv(
    OUT / "model_metrics.csv",
    index=False
)

print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 50)
print(results_df.to_string(index=False))

print("\nFiles generated:")
print("results/model_metrics.csv")
print("results/confusion_matrices.png")
