from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class RandomForestModel:
    def __init__(self, **model_params):
        self.model = RandomForestClassifier(**model_params)

    def train(self, X_train, y_train):
        if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
            print("⚠️ Внимание: Обнаружены пропущенные значения в обучающем наборе. Будет применено заполнение 0.")
            X_train = X_train.fillna(0)
            y_train = y_train.fillna(0)

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if X_test.isnull().sum().sum() > 0 or y_test.isnull().sum() > 0:
            print("⚠️ Внимание: Обнаружены пропущенные значения в тестовом наборе. Будет применено заполнение 0.")
            X_test = X_test.fillna(0)
            y_test = y_test.fillna(0)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        roc_score = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_score:.4f}")

        return y_test, y_pred, roc_score

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_names: pd.Index):
        importances = pd.Series(self.model.feature_importances_, index=feature_names)
        sorted_importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(12, max(6, len(importances) // 3)))
        sns.barplot(x=sorted_importances.values, y=sorted_importances.index)
        plt.title("Feature Importances (All Features)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

        print("\n10 Least Important Features:")
        print(sorted_importances.tail(10))

    def get_model(self):
        return self.model
