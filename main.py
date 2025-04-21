from model import ChurnModel
from preprocessor import Preprocessor
import pandas as pd

data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
processor = Preprocessor(data)
X_train, X_test, y_train, y_test = processor.train_test_split()

model = ChurnModel()
model.train(X_train, y_train)
y_test, y_pred = model.evaluate(X_test, y_test)
model.plot_confusion_matrix(y_test, y_pred)
model.plot_feature_importance(X_train.columns)

