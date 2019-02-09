# Hierarchical clustering 

# generate random sample data
import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns = variables, index = labels)
df

# calculate the distance matrix 
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric = 'euclidean')),
                        columns = labels, index = labels)
row_dist

# apply complete linkage agglomeration to the clusters
from scipy.cluster.hierarchy import linkage
help(linkage) # look at the function documentation

# INCORRECT APPROACH (using the squareform distance matrix)
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(row_dist,
                       method = 'complete',
                       metric = 'euclidean')

# CORRECT APPROACH (using the condensed distance matrix)
row_clusters = linkage(pdist(df, 
                       metric = 'euclidean'),
                       method = 'complete')

# CORRECT APPROACH (using the input sample matrix)
row_clusters = linkage(df.values, 
                       method = 'complete',
                       metric = 'euclidean')

# clustering results
pd.DataFrame(row_clusters,
             columns = ['row label 1',
                        'row label 2',
                        'distance',
                        'no. of items in cluster'],
             index = ['cluster %d' % (i + 1) for i in range(
                      row_clusters.shape[0])])

# visualize the results in a dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters,
                       labels = labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold = np.inf
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


# attach the dendrogram to a heat map
fig = plt.figure(figsize = (8, 8))
axd = fig.add_axes([0.09, 0.10, 0.20, 0.60])

row_dendr = dendrogram(row_clusters, orientation = 'left')

df_rowclust = df.ix[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.10, 0.60, 0.60])
cax = axm.matshow(df_rowclust,
                  interpolation = 'nearest',
                  cmap = 'hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()


# Agglomerative Clustering with scikit-learn
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 2,
                             affinity = 'euclidean',
                             linkage = 'complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)
