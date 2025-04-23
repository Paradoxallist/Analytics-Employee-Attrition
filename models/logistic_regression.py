from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class LogisticRegressionModel:
    def __init__(self, **model_params):
        self.model = LogisticRegression(**model_params)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
            print("⚠️ Пропущенные значения — заменены нулями в train")
            X_train = X_train.fillna(0)
            y_train = y_train.fillna(0)

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        if X_test.isnull().sum().sum() > 0 or y_test.isnull().sum() > 0:
            print("⚠️ Пропущенные значения — заменены нулями в test")
            X_test = X_test.fillna(0)
            y_test = y_test.fillna(0)

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        print("\n[Logistic Regression] Classification Report:")
        print(classification_report(y_test, y_pred))

        roc_score = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_score:.4f}")

        return y_test, y_pred, roc_score

    def plot_feature_importance(self, feature_names: pd.Index):
        if hasattr(self.model, "coef_"):
            importances = pd.Series(np.abs(self.model.coef_[0]), index=feature_names)
            sorted_importances = importances.sort_values(ascending=False)

            plt.figure(figsize=(10, 6))
            sorted_importances.head(30).plot(kind="barh")
            plt.gca().invert_yaxis()
            plt.title("Top Feature Weights (Logistic Regression)")
            plt.tight_layout()
            plt.show()

    def get_model(self):
        return self.model
