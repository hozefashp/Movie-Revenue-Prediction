import pandas as pd                
from ast import literal_eval
from collections import Counter
import numpy as np
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['source'] = 'train'                              
test['source'] = 'test'

df = pd.concat([train, test],ignore_index=True, sort=True)
df = df[df.budget != 0]
df['cast_character'] = df['cast'].fillna('[]').apply(literal_eval).apply(lambda x: [i['character'] for i in x] if isinstance(x, list) else [])
text_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
for column in text_columns:
    df[column] = df[column].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

df['belongs_to_collection'] = df['belongs_to_collection'].notnull()
df['homepage'] = df['homepage'].notnull()
df['is_english_original_language'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
df['is_US'] = df['production_countries'].apply(lambda x: 1 if 'United States of America' in x else 0)
df['belongs_to_collection'] *=1
df['has_homepage'] = df['homepage']*1

count_columns = ['production_companies', 'Keywords', 'crew']
for column in count_columns:
    df['total_' + column] = df[column].apply(lambda x: len(x) if x!= [] else 0)

list_of_cast_names = list(df['cast'].apply(lambda x: [i for i in x] if x != {} else []).values)
top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(20)]
for g in top_cast_names: 
    df['cast_name_' + g] = df['cast'].apply(lambda x: 1 if g in str(x) else 0)
cast_name_column = [col for col in df if col.startswith('cast_name')]
weight = pd.DataFrame(pd.Series([21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2], index=cast_name_column, name=0))
df['cast_power'] = (df[cast_name_column] * weight[0]).sum(1)
df.drop(cast_name_column, axis=1, inplace=True)

list_of_cast_characters = list(df['cast_character'].apply(lambda x: [i for i in x] if x != {} else []).values)
top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(20)]
for g in top_cast_characters:
    df['cast_character_' + g] = df['cast_character'].apply(lambda x: 1 if g in str(x) else 0)
cast_character_column = [col for col in df if col.startswith('cast_character')]
weight = pd.DataFrame(pd.Series([21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], index=cast_character_column, name=0))
df['cast_power'] += (df[cast_character_column] * weight[0]).sum(1)
df.drop(cast_character_column, axis=1, inplace=True)

list_of_genres = list(df['genres'].apply(lambda x: [i for i in x] if x != {} else []).values)
top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(10)]
for g in top_genres: 
    df['genre_' + g] = df['genres'].apply(lambda x: 1 if g in str(x) else 0)
genre_column = [col for col in df if col.startswith('genre_')]
weight = pd.DataFrame(pd.Series([10,9,8,7,6,5,4,3,2,1], index=genre_column, name=0))
df['genre_power'] = (df[genre_column] * weight[0]).sum(1)
df.drop(genre_column, axis=1, inplace=True)

df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

df.drop(['genres', 'homepage', 'original_language', 'spoken_languages', 'production_countries', 'production_companies','Keywords', 'cast', 'crew', 'genre_power', 'overview','imdb_id', 'title','tagline', 'status','poster_path','original_title','release_date','release_month','release_year'], axis=1, inplace=True)

df=df.fillna(0)

train = df.loc[df['source'] == 'train']
test = df.loc[df['source'] == 'test']

train.drop('source', axis=1, inplace=True)
test.drop('source', axis=1, inplace=True)

df_train = DataFrame(train, columns=['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day'])

xTrain = df_train[['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day']] #Hozi, just add the column names OJ sends over here.
yTrain = df_train['revenue']

X_train, X_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size=0.4, random_state=42)


df_test = DataFrame(test, columns=['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day'])
xTest = df_test[['belongs_to_collection', 'budget', 'id', 'popularity', 'revenue', 'runtime', 'is_english_original_language', 'is_US', 'has_homepage', 'total_production_companies', 'total_Keywords', 'total_crew', 'cast_power', 'release_day']] #Hozi, just add the column names OJ sends over here.

'''linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)
yPrediction_LR = linearRegressor.predict(xTest)

with open('prediction_LR.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(yPrediction_LR)
csvFile.close()'''

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
