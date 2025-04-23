import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:

    @staticmethod
    def plot_pearson_correlation_heatmap(data: pd.DataFrame, attrition_column: str = 'AttritionFlag'):
        """
        Строит тепловую карту корреляций между числовыми признаками.
        """
        numerical_columns = data.select_dtypes(include=['int64', 'float64']).drop(columns=[attrition_column])
        correlation_matrix = numerical_columns.corr(method='pearson')

        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
        plt.title("Pearson Correlation Heatmap Between Numerical Features")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_top_attrition_correlations(data: pd.DataFrame, attrition_column: str = 'AttritionFlag', top_n: int = 10):
        """
        Визуализирует топ-N признаков по модулю корреляции с целевой переменной.
        """
        correlation_series = (
            data.corr(numeric_only=True)[attrition_column]
            .drop(attrition_column)
            .apply(abs)
            .sort_values(ascending=False)
        )

        top_features = correlation_series.head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_features.values, y=top_features.index, orient='h')
        plt.title(f"Top {top_n} Features Correlated with AttritionFlag (by Pearson)")
        plt.xlabel("Absolute Correlation")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
