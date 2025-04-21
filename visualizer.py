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

    @staticmethod
    def plot_attrition_distribution_comparison(data: pd.DataFrame, feature: str, attrition_column: str = 'AttritionFlag', bins: int = 20):
        """
        Строит 3 графика в одном окне:
        1. Распределение признака по классам AttritionFlag
        2. Процент уволившихся по каждому уникальному значению признака
        3. Процент оставшихся по каждому уникальному значению признака
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 14))

        # Гистограмма распределения
        sns.histplot(data=data, x=feature, hue=attrition_column, multiple="stack", bins=bins, ax=axes[0])
        axes[0].set_title(f"Distribution of {feature} by {attrition_column}")

        # Процент уволившихся
        grouped = data.groupby(feature)[attrition_column]
        attrition_percent = grouped.mean() * 100
        axes[1].plot(attrition_percent.index, attrition_percent.values, marker='o', color='red')
        axes[1].set_title(f"% Attrition by {feature}")
        axes[1].set_ylabel("% Attrition")
        axes[1].set_xlabel(feature)
        axes[1].grid(True)

        # Процент оставшихся
        stay_percent = (1 - grouped.mean()) * 100
        axes[2].plot(stay_percent.index, stay_percent.values, marker='o', color='green')
        axes[2].set_title(f"% Stayed by {feature}")
        axes[2].set_ylabel("% Stayed")
        axes[2].set_xlabel(feature)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_conditional_density_distribution(data: pd.DataFrame, feature: str, attrition_column: str = 'AttritionFlag'):
        """
        Строит условное вероятностное распределение признака при Attrition=1 и Attrition=0
        + гистограмму распределения по классам Attrition — на одном полотне.
        """
        data_yes = data[data[attrition_column] == 1][feature].dropna()
        data_no = data[data[attrition_column] == 0][feature].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Условные плотности
        sns.kdeplot(data_yes, fill=True, label='Attrition = Yes', color='red', common_norm=False, ax=axes[0])
        sns.kdeplot(data_no, fill=True, label='Attrition = No', color='green', common_norm=False, ax=axes[0])
        axes[0].set_title(f"Conditional Density of {feature} by AttritionFlag")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Density (normalized)")
        axes[0].legend()
        axes[0].grid(True)

        # Гистограмма
        sns.histplot(data=data, x=feature, hue=attrition_column, multiple="stack", bins=20, ax=axes[1])
        axes[1].set_title(f"Stacked Histogram of {feature} by AttritionFlag")
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel("Count")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
