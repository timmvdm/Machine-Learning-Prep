import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris_df = sns.load_dataset('iris')
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = iris_df[numeric_cols]
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3 , random_state=42)
model.fit(X)
print(model.cluster_centers_, model.inertia_)

preds = model.predict(X)

options = range(2,11)
inertias = []

for n_clusters in options:
    model = KMeans(n_clusters, random_state=42).fit(X)
    inertias.append(model.inertia_)

# plt.title("No. of clusters vs. Inertia")
# plt.plot (options, inertias, '-o')
# plt.xlabel ('No. of clusters (K)')
# plt.ylabel ('Inertia')
# plt.show()


from sklearn.decomposition import PCA


pca = PCA(n_components=2)
pca.fit(X)
transformed = pca.transform(X)
sns.scatterplot(x = transformed[:,0], y = transformed[:,1], hue = iris_df['species'])



from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
transformed = tsne.fit_transform(X)
sns.scatterplot(x= transformed[:,0], y= transformed[:,1], hue = iris_df['species'])
plt.show()