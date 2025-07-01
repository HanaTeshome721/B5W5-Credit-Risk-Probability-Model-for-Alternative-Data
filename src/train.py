import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.model_utils import evaluate_model

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")

# Features and Target
X = df.drop(columns=["CustomerId", "is_high_risk"])
y = df["is_high_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Define models and hyperparameters
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier()
}

params = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "GradientBoosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
}

# Train and track
for model_name, model in models.items():
    print(f"Training {model_name}...")
    clf = GridSearchCV(model, param_grid=params[model_name], cv=5, scoring="roc_auc")

    with mlflow.start_run(run_name=model_name):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)

        # Log metrics and model
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        mlflow.sklearn.log_model(clf.best_estimator_, model_name)

        print(f"{model_name} AUC: {metrics['roc_auc']:.3f}")

print("✅ Training and logging complete.")
