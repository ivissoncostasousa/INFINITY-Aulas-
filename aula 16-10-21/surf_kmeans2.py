import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def  classify_board_adequate(board_adequate):
    if board_adequate == 'Very inadequate':
        return 0
    if board_adequate == 'Inadequate':
        return 1
    if board_adequate == 'More or less':
        return 2
    if board_adequate == 'Suitable':
        return 3
    if board_adequate == 'Very suitable':
        return 4


ds = pd.read_csv('df_surf2.csv', sep=',')
print(ds.head(20))
print(ds.info())
print(ds['board_adequate'].unique())
ds['board_adequate'] = ds['board_adequate'].apply(classify_board_adequate)


ds = ds.select_dtypes('number')
ds.drop(columns=['surfer_weight_distribution','board_tail_rocker','board_nose_rocker'], axis=1, inplace=True)
print(ds.info())
#Outliers de board_how_many
ds[ds['board_how_many']==60] = 6.0
ds[ds['board_how_many']==10] = 1.0

# isso será uma função para ser aplicada em todas as colunas
def remove_na(column_name):
    column = ds[column_name]
    column.dropna(inplace=True)
    column = column.tolist()
    sum = 0
    for d in column:
        sum = sum + d
    media = sum / len(column)
    print("media da coluna {} é {}".format(column_name, media))
    ds[column_name].fillna(media, inplace=True)
   # print(ds.info())
#aqui termina

# esta forma é muito trabalhosa!!
# remove_na('board_how_many')
# remove_na('board_length')


#Facilitando a vida:
for column in ds.columns:
    remove_na(column)
print(ds.info())

# wcss =[]
# for i in range(1,15):
#     kmeans = KMeans(n_clusters=i, init='k-means++')
#     kmeans.fit(ds)
#     wcss.append(kmeans.inertia_)

# print(wcss)
# plt.plot(range(1,15), wcss)
# plt.xlabel("Clusters")
# plt.ylabel("WCSS")
# plt.show()
# dropar os índices para n influenciar no algoritmo
ds.drop(columns='Unnamed: 0', inplace=True)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300)
clusters = kmeans.fit_predict(ds)
print(clusters)
ds['cluster'] = clusters
print(ds.head(25))

sns.pairplot(ds, x_vars=['cluster'], y_vars=['board_length'])
plt.show()