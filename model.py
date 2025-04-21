from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ChurnModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42, class_weight='balanced')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        roc_score = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_score:.4f}")

        return y_test, y_pred

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_names: pd.Index):
        importances = pd.Series(self.model.feature_importances_, index=feature_names)
        top_importances = importances.sort_values(ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances.values, y=top_importances.index)
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    def get_model(self):
        return self.model
