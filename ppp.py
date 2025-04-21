import pandas as pd
from visualizer import Visualizer

# Загрузка данных
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
pd.options.display.max_columns = 100

# Удаление константных и служебных столбцов
columns_to_drop = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
data.drop(columns=columns_to_drop, inplace=True)

# Преобразование целевого признака в числовой формат
data['AttritionFlag'] = data['Attrition'].map({'Yes': 1, 'No': 0})

# Выделение числовых признаков
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('AttritionFlag')

# Выделение категориальных признаков
categorical_features = data.select_dtypes(include=['object', 'bool']).columns.tolist()

# Визуализация корреляций
Visualizer.plot_pearson_correlation_heatmap(data)
Visualizer.plot_top_attrition_correlations(data)

#numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
#numerical_columns.remove('AttritionFlag')

#for feature in numerical_columns:
#    Visualizer.plot_conditional_density_distribution(data, feature)