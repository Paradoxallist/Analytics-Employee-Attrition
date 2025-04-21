import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, data: pd.DataFrame, target_column: str = 'Attrition'):
        self.data = data.copy()
        self.target_column = target_column
        self.target_encoded_column = 'AttritionFlag'
        self.X = None
        self.y = None

    def clean_data(self):
        columns_to_drop = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
        self.data.drop(columns=columns_to_drop, inplace=True)
        return self

    def extract_target(self):
        self.data[self.target_encoded_column] = self.data[self.target_column].map({'Yes': 1, 'No': 0})
        self.y = self.data[self.target_encoded_column]
        self.data.drop(columns=[self.target_column, self.target_encoded_column], inplace=True)
        return self

    def encode_categorical(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'bool']).columns.tolist()
        self.data = pd.get_dummies(self.data, columns=categorical_columns)
        return self

    def preprocess_all(self):
        return self.clean_data().extract_target().encode_categorical()

    def get_features_and_target(self):
        self.X = self.data.copy()
        return self.X, self.y

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        self.preprocess_all()
        self.get_features_and_target()
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def get_processed_data(self):
        return self.X, self.y
