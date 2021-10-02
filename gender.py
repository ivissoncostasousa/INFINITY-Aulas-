import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

ds = pd.read_csv("gender.csv", sep=',')
# print(ds.info())
# print(ds.head(20))

# sns.boxplot(x=ds['forehead+height_cm'])
# plt.show()

x =de.drop(columns='Male', axis=1)
y = ds['Male']

x_train, x_test, y_train, y_test = train_test_split(ds, test_size=0.25, random_state=50)

print("tamanho da base de teste: ",len(x_test))
print("tamanho da base de treino: ",len(x_train))

knm = KNeighborsClassifier()

k_values = dict(n_neighbors=[1,2,3,4,5,6,7,8,9,10])

gritSearch = GridSearchCV(knn, k_values, scoring='accuracy')

gs = gridSearch.fit(x_train, y_train)

print("Acurácia para o valor de {} é : {}". format(gs.best_params_, gs.best_score_))


