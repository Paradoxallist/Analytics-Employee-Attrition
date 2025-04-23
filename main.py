import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import Preprocessor
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.gradient_boosting import GradientBoostingModel
from config import MODEL_CONFIGS

# Загрузка данных
raw_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Обработка версий
processor = Preprocessor(raw_data)
split_versions = processor.train_test_split_versions()

# Модели
model_classes = {
    "RandomForest": RandomForestModel,
    "LogisticRegression": LogisticRegressionModel,
    "GradientBoosting": GradientBoostingModel
}

# Сбор результатов
results = []

for version_name, (X_train, X_test, y_train, y_test) in split_versions.items():
    for model_name, model_class in model_classes.items():
        base_params = MODEL_CONFIGS[model_name]["base"]

        if model_name == "RandomForest":
            model = model_class(**base_params)
            print(f"\n🔎 Версия: {version_name} | Модель: {model_name} (base)")
            model.train(X_train, y_train)
            _, _, roc_auc = model.evaluate(X_test, y_test)
            results.append({"Version": version_name, "Model": f"{model_name}_base", "ROC AUC": roc_auc})

            grid_params = MODEL_CONFIGS[model_name]["grid"]
            from itertools import product
            sample_grid = list(product(
                grid_params["n_estimators"][::50],
                grid_params["max_depth"][::5],
                grid_params["min_samples_split"][::25],
                grid_params["min_samples_leaf"][::10]
            ))[:3]

            for idx, (n_est, depth, min_split, min_leaf) in enumerate(sample_grid):
                custom_params = {
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "min_samples_split": min_split,
                    "min_samples_leaf": min_leaf,
                    "random_state": 42,
                    "class_weight": "balanced"
                }
                model = model_class(**custom_params)
                print(f"\n🔎 Версия: {version_name} | Модель: {model_name}_tuned_{idx}")
                model.train(X_train, y_train)
                _, _, roc_auc = model.evaluate(X_test, y_test)
                label = f"{model_name}_tuned_{idx}"
                results.append({"Version": version_name, "Model": label, "ROC AUC": roc_auc})
        else:
            model = model_class(**base_params)
            print(f"\n🔎 Версия: {version_name} | Модель: {model_name}")
            model.train(X_train, y_train)
            _, _, roc_auc = model.evaluate(X_test, y_test)
            results.append({"Version": version_name, "Model": model_name, "ROC AUC": roc_auc})

# Преобразуем в DataFrame
results_df = pd.DataFrame(results)

# Вывод в консоль отсортированных результатов
print("\n===== Модели по убыванию ROC AUC =====")
sorted_display = results_df.sort_values(by="ROC AUC", ascending=False)
for idx, row in sorted_display.iterrows():
    print(f"{row['Version']} | {row['Model']}: {row['ROC AUC']:.4f}")

# Визуализация
plt.figure(figsize=(14, 9))
sorted_results = results_df.sort_values(by="ROC AUC", ascending=True)
labels = sorted_results["Version"] + " | " + sorted_results["Model"]
plt.barh(labels, sorted_results["ROC AUC"])
plt.xlabel("ROC AUC Score")
plt.title("Сравнение моделей и параметров по версиям данных")
plt.tight_layout()
plt.show()

