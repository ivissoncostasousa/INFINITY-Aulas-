import pandas as pd
import seaborn as sns
import matblotlib.pyplot as plt
from sklearn import cluster
from sklearn.decomposition import _pca
from sklearn.cluster import KMeans

#LENDO DS DE MUSICAS DO SPOTIFY
ds = pd.read_csv("genres_v2.csv", sep=",")
#print(ds.info())
#Normalizando o ds para aolicar PCA
ds_normalized = ds.select_dtypes('number')
# ds_normalized.drop(columns="Unnamed: 0", axis=1, inplace=True)
print(len(ds_normalized.columns))
ds_normalized = (ds_normalized - ds_normalized.mean())/ds_normalized.std()


#Aplicar o PCA
pca = PCA(n_components=5)
pca.fit(ds_normalized)
print(pca.explained_variance_ration_) 
percent = 0
for component in pca.explained_variance_ratio_:
    percent += component
    
print("Total de cobertura dos dados na nossa base: {:.2f}%".format(percent*100))   

# Aplicando  o PCA depois 
pca = PCA(n_components=8).fit_transform(ds_normalized)


# Rodar o algoritmo k-means em cima da base de dados com o PCA
kmeans = KMeans(n_clusters=4)
kmeans.fit(pca)
print(kmeans.labels_)




# Pegar os valores dos labels e colocar como uma nova coluna dentro do original
ds['cluster'] = kmeans.labels_

# Visualizando clusters distintos

print(len(ds[ds['cluster']==0]))

print(pd.value_counts(ds['cluster']))
