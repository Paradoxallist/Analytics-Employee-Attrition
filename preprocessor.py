import pandas as pd
from sklearn.model_selection import train_test_split
from feature_generator import FeatureGenerator
import numpy as np


class Preprocessor:
    def __init__(self, data: pd.DataFrame, target_column: str = 'Attrition'):
        self.data = data.copy()
        self.target_column = target_column
        self.target_encoded_column = 'AttritionFlag'
        self.X_versions = {}
        self.y = None

    def clean_data(self, df):
        columns_to_drop = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
        return df.drop(columns=columns_to_drop)

    def extract_target(self, df):
        df[self.target_encoded_column] = df[self.target_column].map({'Yes': 1, 'No': 0})
        self.y = df[self.target_encoded_column]
        return df.drop(columns=[self.target_column, self.target_encoded_column])

    def encode_categorical(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'bool']).columns.tolist()
        self.data = pd.get_dummies(self.data, columns=categorical_columns)
        return self

    def combine_selected_correlated_features(self, df):
        combined_df = df.copy()
        if "JobLevel" in df.columns and "MonthlyIncome" in df.columns:
            combined_df["JobLevel_MonthlyIncome_sum"] = combined_df["JobLevel"] + combined_df["MonthlyIncome"]
            combined_df.drop(columns=["JobLevel", "MonthlyIncome"], inplace=True)
        if "PerformanceRating" in df.columns and "PercentSalaryHike" in df.columns:
            combined_df["Performance_Peercent_sum"] = combined_df["PerformanceRating"] + combined_df["PercentSalaryHike"]
            combined_df.drop(columns=["PerformanceRating", "PercentSalaryHike"], inplace=True)
        required_cols = ["YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
        if all(col in df.columns for col in required_cols):
            combined_df["Years_Combined_sum"] = sum([combined_df[col] for col in required_cols])
            combined_df.drop(columns=required_cols, inplace=True)
        return combined_df

    def generate_versions(self):
        # raw
        df_raw = self.data.copy()
        self.X_versions["raw"] = pd.get_dummies(df_raw.drop(columns=[self.target_column]))

        # cleaned
        df_cleaned = self.clean_data(df_raw)
        self.X_versions["cleaned"] = pd.get_dummies(df_cleaned.drop(columns=[self.target_column]))

        # combined
        df_combined = self.combine_selected_correlated_features(df_cleaned)
        self.X_versions["combined"] = pd.get_dummies(df_combined.drop(columns=[self.target_column]))

        # filtered = same as combined for now, no drop of least important
        self.X_versions["filtered"] = self.X_versions["combined"].copy()

        # full feature generation
        df_processed = self.extract_target(df_combined)
        self.data = pd.get_dummies(df_processed)
        self.encode_categorical()

        generator = FeatureGenerator(self.data)
        df_gen = generator.generate_arithmetic_features()
        df_gen = generator.generate_logical_features()
        df_gen = generator.generate_binned_features()
        df_gen = df_gen.select_dtypes(include=[int, float])

        self.X_versions["gen_full"] = df_gen

        # version with correlation filtering (gen_pruned)
        corr_matrix = df_gen.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        df_pruned = df_gen.drop(columns=to_drop)
        self.X_versions["gen_pruned"] = df_pruned

        return self

    def get_versions_and_target(self):
        return self.X_versions, self.y

    def train_test_split_versions(self, test_size: float = 0.2, random_state: int = 42):
        self.generate_versions()
        split_data = {}
        for name, df in self.X_versions.items():
            X_train, X_test, y_train, y_test = train_test_split(df, self.y, test_size=test_size, random_state=random_state)
            split_data[name] = (X_train, X_test, y_train, y_test)
        return split_data
