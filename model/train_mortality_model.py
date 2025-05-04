"""
train_sepsis_rf.py

Load sepsis features, train a Random Forest classifier with optional feature selection,
then save the trained model for future inference.

This script hardcodes the input path but can be easily adapted.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def main():
    # Load data
    input_path = "/Users/apple/GitHubRepos/OpenManus/AI_agent_train_sepsis.csv"
    df = pd.read_csv(input_path)

    # Specify target column
    target_col = 'mortality_90d'  # replace with your actual label column name
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build pipeline: imputing, scaling, feature selection, random forest
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {auc:.4f}")

    # Save the pipeline
    model_path = "sepsis_rf_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Trained model saved to {model_path}")


if __name__ == '__main__':
    main()
