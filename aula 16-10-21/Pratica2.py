import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
# transformando o dataset em dataframe do pandas
ds = pd.DataFrame(iris.data, columns=iris.feature_names)
# conhecendo
print(ds.head())
print(ds.info())
# verificando existencia de valores nulos
print(ds.isnull().sum())
columns = ds.columns
# plotar valores para saber se tem outliers
for c in columns:
    # sns.boxplot(x=ds[c])
    # plt.show()
#exibição dos dados em pares
# sns.pairplot(ds)
# plt.show()

wcss = []
for k in range(1,8):
    KMeans = KMeans(n_clusters=R)  
    
    
    
    
    
    
    
    
kmeans =KMeans(m_clusters=3, init='k-means++', n_init=10, max_iter=300)
clusters=kmeans.fit_predict(ds)
print(clusters)
ds['target'] = clusters
# sns.pairplot(ds, hue='target')
# plt.show()
    
# print("clusters originais: ", iris.target)
# ds['origin_target'] = iris.target
# sn.pairplot(ds, hue='origin_target')
# plt.show()


ds_cluster0 = ds[ds['target'] == 0]
ds_cluster1 = ds[ds['target'] == 2]
ds_cluster2 = ds[ds['target'] == 2]

ds_cluster1.to_csv('cluster1.csv', sep=',')
ds_cluster2.to_csv('cluster2.csv', sep=',')
ds_cluster3.to_csv('cluster3.csv', sep=',')
