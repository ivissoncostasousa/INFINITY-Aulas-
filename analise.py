import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os



os.system('cls')
def categorize(price):
    if price >= 500000:
        return 'expansive'
    elif price >= 220000:
        return 'fear'
    else:
        return 'cheap'
    


dataset = pd.read_csv('kc_house_data.csv', sep=',')

bedrooms = dataset['bedrooms']
bedrooms.dropna(inplace=True)
values = np.array(bedrooms)
print(values.tolist())
sum = 0


#verificar quantos campos nulos temos no dataset
print(dataset.isnull().sum())
for value in values:
    if value != 33.0:
        sum += value
media =sum/len(values)-1   
print(media)

#Substituir os campos nulos de quartos para a media de quartos do dataset
dataset['bedrooms'].fillna(media, inplace=True)
dataset.dropna(inplace=True)
print(dataset.isnull().sum())



# Aplicação do Zscore para eliminar outliers
# Verificar a existencia de outlier

# sns.boxplot(x=dataset['bedrooms'])
# plt.show()

# Utiliza o Zscore
bedrooms = pd.DataFrame(values)
bedrooms.columns = ['bedrooms']
z = np.abs(stats.zscore(bedrooms))
z_bedrooms = bedrooms[(z<3).all(axis=1)]
sns.boxplot(x=z_bedrooms['bedrooms'])
plt.show()

