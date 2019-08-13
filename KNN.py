import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_log_error
import statsmodels.api as sm
import csv

# Import the dataset
train_dataset = pd.read_csv('train_final.csv')
test_dataset = pd.read_csv('test_final.csv')

df_train = DataFrame(train_dataset, columns=['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day'])

xTrain = df_train[['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day']] #Hozi, just add the column names OJ sends over here.
yTrain = df_train['revenue']

X_train, X_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size=0.4, random_state=42)


df_test = DataFrame(test_dataset, columns=['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day'])
xTest = df_test[['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day']] #Hozi, just add the column names OJ sends over here.

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 10)
knn_model = knn.fit(X_train, y_train)
yPrediction_KNN = knn.predict(X_test)

error = np.sqrt(mean_squared_log_error(y_test, yPrediction_KNN))
print(error)

knn = KNeighborsRegressor(n_neighbors = 10)
knn_model = knn.fit(xTrain, yTrain)
yPrediction_KNN = knn.predict(xTest)
final_df = pd.DataFrame(columns = ['id','revenue'])
final_df['id'] = df_test['id']
final_df['revenue'] = yPrediction_KNN
final_df.set_index('id', inplace=True)
final_df.to_csv('KNN_Predictions.csv')
