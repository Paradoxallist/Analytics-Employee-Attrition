from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd


class GradientBoostingModel:
    def __init__(self, **model_params):
        self.model = GradientBoostingClassifier(**model_params)

    def train(self, X_train, y_train):
        if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
            print("⚠️ Пропущенные значения — заменены нулями в train")
            X_train = X_train.fillna(0)
            y_train = y_train.fillna(0)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if X_test.isnull().sum().sum() > 0 or y_test.isnull().sum() > 0:
            print("⚠️ Пропущенные значения — заменены нулями в test")
            X_test = X_test.fillna(0)
            y_test = y_test.fillna(0)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n[Gradient Boosting] Classification Report:")
        print(classification_report(y_test, y_pred))

        roc_score = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_score:.4f}")

        return y_test, y_pred, roc_score

    def plot_feature_importance(self, feature_names: pd.Index):
        importances = pd.Series(self.model.feature_importances_, index=feature_names)
        sorted_importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sorted_importances.head(30).plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title("Top Feature Importances (Gradient Boosting)")
        plt.tight_layout()
        plt.show()

    def get_model(self):
        return self.model
