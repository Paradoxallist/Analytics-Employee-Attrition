# Конфигурация параметров моделей

MODEL_CONFIGS = {
    "RandomForest": {
        "base": {
            "random_state": 42,
            "class_weight": "balanced"
        },
        "grid": {
            "n_estimators": list(range(10, 300, 2000)),
            "max_depth": list(range(1, 20)),
            "min_samples_split": list(range(2, 150)),
            "min_samples_leaf": list(range(2, 60))
        }
    },
    "LogisticRegression": {
        "base": {
            "solver": "lbfgs",
            "max_iter": 100000
        },
        "grid": {}
    },
    "GradientBoosting": {
        "base": {
            "random_state": 42
        },
        "grid": {}
    }
}
