import pandas as pd
import numpy as np
from numpy.random.mtrand import permutation
import math
from sklearn.neighbors import KNeighborsRegressor

# INFORMAÇÃO DA BASSE DE DADOS

ds = pd.read_csv(f'/content/drive/MyDrive/Infinity School/IA/aula02/nba_2013.csv', sep = ",")
print(ds.columns.values)
print(ds.head())
print(ds.info())

# ZERANDO VALORES NULOS 
ds['x3p.'].fillna(0, inplace=True)
ds['ft.'].fillna(0, inplace=True)
ds['fg.'].fillna(0, inplace=True)
ds['x2p.'].fillna(0, inplace=True)
ds['x2p'].fillna(0, inplace=True)
ds['efg.'].fillna(0, inplace=True)
print(ds.isnull().sum())

# NORMALIZANDO BASE
ds_normalized = ds.select_dtypes('number')
ds_normalized = (ds_normalized - ds_normalized())/ds.std()
display(ds_normalized.head())

# Separar Base X Base Teste
# Embaralhar dados da base
random_index = permutation(ds_normalized.index)

# Calculara a quantidade de dados de teste
q_teste = math.floor(len(ds_normalized) / 4)

# separar a base de teste
test = ds_normalized.loc[random_index[1:q_teste]]
# Separar base de treino
train = ds_normalized.loc[random_index[q_teste:]]
# Separar Colunas que irão ajudar a definir o target


print(ds_normalized.head(30))
# DEFININDO O X
ds_temp = ds_normalized.drop(columns=['pts'])
x_columns = ds_temp.columns
print("x_columns: ", x_columns)

# DEFINIDO O Y

y_column = ['pts']
print("y_column: ", y_column)

knn = KNeighborsRegressor(n_neighbors=3)

print(ds_normalized.isnull().sum())
print(ds_normalized.isna().sum())

knn.fit(train[x_columns], train[y_column])

predictions = knn.predict(test[x_columns])


# Calcular erro médio das predições


result_test = test[y_column]

mse = (((predictions - result_test) ** 2).sum()) / len(predictions)

print("MSE: ", mse)

