import pandas as pd
import numpy as np
import itertools


class FeatureGenerator:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

    def generate_arithmetic_features(self):
        new_features = {}
        for col1, col2 in itertools.combinations(self.numeric_columns, 2):
            if col1 == col2:
                continue
            new_features[f"{col1}_minus_{col2}"] = self.data[col1] - self.data[col2]
            if not (self.data[col2] == 0).any():
                new_features[f"{col1}_div_{col2}"] = self.data[col1] / self.data[col2]
                new_features[f"{col1}_mod_{col2}"] = self.data[col1] % self.data[col2]

        new_df = pd.concat([self.data, pd.DataFrame(new_features)], axis=1)
        self.data = new_df
        return self.data

    def generate_binned_features(self, step_percent: float = 0.1):
        for col in self.numeric_columns:
            min_val, max_val = self.data[col].min(), self.data[col].max()
            step = (max_val - min_val) * step_percent
            if step == 0:
                continue
            bins = np.arange(min_val, max_val + step, step)
            labels = [f"{col}_bin_{i}" for i in range(len(bins)-1)]
            self.data[f"{col}_binned"] = pd.cut(self.data[col], bins=bins, labels=labels, include_lowest=True)
        return pd.get_dummies(self.data)

    def generate_logical_features(self):
        if "JobLevel" in self.data.columns and "OverTime_Yes" in self.data.columns:
            self.data["SeniorOverloaded"] = ((self.data["JobLevel"] > 3) & (self.data["OverTime_Yes"] == 1)).astype(int)
        if "YearsAtCompany" in self.data.columns and "YearsSinceLastPromotion" in self.data.columns:
            self.data["LongNoPromotion"] = ((self.data["YearsAtCompany"] > 5) & (self.data["YearsSinceLastPromotion"] >= 3)).astype(int)
        return self.data
