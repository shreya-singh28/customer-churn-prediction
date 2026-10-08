# training.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ---------------- CONFIG ----------------
DATA_PATH = r"Telcom-Customer-Churn.csv"
RANDOM_STATE = 42


# ---------------- LOAD DATA ----------------
def load_data(path):

    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    return df


# ---------------- FEATURE ENGINEERING ----------------
def feature_engineering(df):

    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["HighCharges"] = (df["MonthlyCharges"] > 80).astype(int)

    df["LongTermCustomer"] = (df["tenure"] > 24).astype(int)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1,12,24,48,72],
        labels=[0,1,2,3]
    )

    df["TenureGroup"] = df["TenureGroup"].astype(float).fillna(0).astype(int)

    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    return df


# ---------------- ENCODING ----------------
def encode(df):

    encoders = {}

    for col in df.select_dtypes(include="object").columns:

        if col == "customerID":
            continue

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

    return df, encoders


# ---------------- MAIN ----------------
def main():

    print("Loading dataset...")
    df = load_data(DATA_PATH)

    df = feature_engineering(df)

    df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

    df_model = df.drop(columns=["customerID"])

    df_model, encoders = encode(df_model)

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    # ---------------- TRAIN TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # ---------------- SMOTE ----------------
    print("Applying SMOTE...")

    smote = SMOTE(random_state=RANDOM_STATE)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ---------------- SCALING ----------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- GRID SEARCH XGBOOST ----------------
    print("\nTuning XGBoost with GridSearch...")

    param_grid = {
        "n_estimators":[300,500],
        "max_depth":[5,7,9],
        "learning_rate":[0.01,0.03],
        "subsample":[0.8,0.9],
        "colsample_bytree":[0.8,0.9]
    }

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_xgb = grid.best_estimator_

    print("Best XGBoost Parameters:", grid.best_params_)

    # ---------------- MODELS ----------------
    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE
            ),

        "XGBoost":
            best_xgb
    }

    results = {}

    print("\nModel Comparison")
    print("----------------")

    best_model = None
    best_accuracy = 0

    for name, model in models.items():

        if name == "Logistic Regression":

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        else:

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        results[name] = acc

        print(f"{name} Accuracy: {acc}")

        if acc > best_accuracy:

            best_accuracy = acc
            best_model = model

    print("\nBest Model:", best_model)

    # ---------------- CROSS VALIDATION ----------------
    print("\nCross Validation Score:")

    scores = cross_val_score(best_model, X, y, cv=5)

    print("Average CV Accuracy:", scores.mean())

    # ---------------- FINAL EVALUATION ----------------
    y_pred = best_model.predict(X_test)

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn","Churn"],
        yticklabels=["No Churn","Churn"]
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("confusion_matrix.png")

    plt.close()
    from sklearn.metrics import roc_curve, auc
    y_proba = best_model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()
    # ---------------- FEATURE IMPORTANCE ----------------
    if hasattr(best_model, "feature_importances_"):

        importance = pd.Series(
            best_model.feature_importances_,
            index=X.columns
        ).sort_values()

        plt.figure(figsize=(8,6))

        importance.plot(kind="barh")

        plt.title("Feature Importance")

        plt.tight_layout()

        plt.savefig("feature_importance.png")

        plt.close()

    # ---------------- MODEL COMPARISON PLOT ----------------
    plt.figure()

    sns.barplot(
        x=list(results.keys()),
        y=list(results.values())
    )

    plt.ylabel("Accuracy")

    plt.title("Model Comparison")

    plt.savefig("model_comparison.png")

    plt.close()

    # ---------------- SAVE MODEL ----------------
    joblib.dump(best_model, "model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(best_accuracy, "accuracy.pkl")

    print("\nModel saved successfully")


if __name__ == "__main__":
    main()