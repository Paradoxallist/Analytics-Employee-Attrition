import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import seaborn
import numpy as np

data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
pd.options.display.max_columns = 100
print(data.info())
print(data.isnull().sum())
print(data.describe())