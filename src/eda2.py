import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from seaborn import violinplot
from sklearn.metrics import silhouette_samples, silhouette_score

from pprint import pprint
from time import time
import logging

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def under_sample_binary_0_G0(x, y, test_reserve=.2, oversample=1.5):
    '''
    undersamples based on y <= 0 and y > 0 as binary options
    will reserve test_reserve portion from minority class
    will oversample remaining minority class to oversample times its original size with replacement
    will undersample majority class to match size of oversampled minority
    returns in the same format as train_test_split
    '''
    y = y.astype(int)
    y_0_idxs = np.where(y <= 0)[0]
    y_G0_idxs = np.where(y > 0)[0]
    if len(y_0_idxs) > len(y_G0_idxs):
        minority_test_idxs = np.random.choice(y_G0_idxs, round(len(y_G0_idxs)*test_reserve), replace=False)
        minority_idxs = np.setdiff1d(y_G0_idxs, minority_test_idxs)
        minority_idxs = np.random.choice(minority_idxs, round(len(y_G0_idxs)*oversample), replace=True)
        majority_idxs = np.random.choice(y_0_idxs, len(minority_idxs), replace=False)
        majority_test_idxs = np.setdiff1d(y_0_idxs, majority_idxs)
    elif len(y_0_idxs) == len(y_G0_idxs):
        minority_test_idxs = np.random.choice(y_G0_idxs, round(len(y_G0_idxs)*test_reserve), replace=False)
        minority_idxs = np.setdiff1d(y_G0_idxs, minority_test_idxs)
        minority_idxs = np.random.choice(minority_idxs, round(len(y_G0_idxs)*oversample), replace=True)
        majority_test_idxs = np.random.choice(y_0_idxs, len(minority_test_idxs), replace=False)
        majority_idxs = np.setdiff1d(y_0_idxs, majority_test_idxs)
        majority_idxs = np.random.choice(majority_idxs, len(minority_idxs), replace=True)
    else:
        minority_test_idxs = np.random.choice(y_0_idxs, round(len(y_0_idxs)*test_reserve), replace=False)
        minority_idxs = np.setdiff1d(y_0_idxs, minority_test_idxs)
        minority_idxs = np.random.choice(minority_idxs, round(len(y_0_idxs)*oversample), replace=True)
        majority_idxs = np.random.choice(y_G0_idxs, len(minority_idxs), replace=False)
        majority_test_idxs = np.setdiff1d(y_G0_idxs, majority_idxs)
    train_idx = np.concatenate((minority_idxs, majority_idxs), axis=0)
    test_idx = np.concatenate((minority_test_idxs, majority_test_idxs), axis=0)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    return x_train, x_test, y_train, y_test

def cum_scree_plot(tsvd, num_components, title=None):
    ind = np.arange(num_components)
    vals = []
    for num in range(num_components):
        vals.append(np.sum(tsvd.explained_variance_ratio_[0:num]))
    plt.figure(figsize=(10, 5), dpi=250)
    ax = plt.subplot(111)

    cm = plt.cm.get_cmap('cool')

    ax.bar(ind, vals, 0.35, color=cm(vals))

    for i in range(num_components):
        if i % 5 == 0:
            ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=4)

    #ax.set_xticklabels(ind, fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, num_components+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)
    plt.show()

def tsne_3d_scatter(x, y, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[0], x[1], x[2], zdir='z', s=20, depthshade=True, c=y[0], cmap = 'cool', )
    ax.set_xlabel('TNSE Parameter 1')
    ax.set_ylabel('TNSE Parameter 2')
    ax.set_zlabel('TNSE Parameter 3')
    if title is not None:
        plt.title(title, fontsize=16)
    plt.show()

x = np.load('data/x_array.npy')
y = np.load('data/y_array.npy')
vocab = pd.read_csv('data/cat_to_num.csv', usecols=['ID']).values.astype(str)
vocab = vocab[:,0].tolist()
#standard train test split to cut the data in half
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)

#undersample split to compare with standard
# x_train, x_test, y_train, y_test = under_sample_binary_0_G0(x, y)


n_components=150

tfidf_pipe = Pipeline([
    ('vect', CountVectorizer(vocabulary=vocab)),
    ('tfidf', TfidfTransformer()),
    ('tsvd', TruncatedSVD(n_components=n_components)),
])
# tfidf_pipe.fit(x_train)
# x_train = tfidf_pipe.transform(x_train)
# cum_scree_plot(tfidf_pipe.named_steps['tsvd'], n_components, title="Website Categories TSVD\nStandard Split")
#
# tfidf_pipe.fit(x_under)
# x_under = tfidf_pipe.transform(x_under)
#cum_scree_plot(tfidf_pipe.named_steps['tsvd'], n_components, title="Website Categories TSVD\nUnder Sampling")

'''The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.'''

silhouette_scores_a_average = []
silhouette_scores_k_average = []
silhouette_scores_s_average = []
silhouette_scores_d_average = []
predication_average_var_k = []
predication_average_var_a = []
predication_average_var_s = []
predication_average_var_d = []
for n_clusters in range(2, 10):
    silhouette_scores_a = []
    silhouette_scores_k = []
    silhouette_scores_s = []
    silhouette_scores_d = []
    for run in range(5):
        x_train, x_test, y_train, y_test = under_sample_binary_0_G0(x, y)
        x_train = tfidf_pipe.fit_transform(x_train)

        cluster_ward = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_a = cluster_ward.fit_predict(x_train)

        cluster_kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=500, n_jobs=-1)
        cluster_k = cluster_kmeans.fit_predict(x_train)

        cluster_spectral = SpectralClustering(n_clusters=n_clusters) #using look for n_clusters to also loop through diminstions in spectral
        cluster_s = cluster_spectral.fit_predict(x_train)

        cluster_dbscan = DBSCAN(eps=(np.log(n_clusters)/50))#using look for n_clusters to also loop through eps in dbscan
        cluster_d = cluster_dbscan.fit_predict(x_train)

        silhouette_scores_a.append(silhouette_score(x_train, cluster_a))
        silhouette_scores_k.append(silhouette_score(x_train, cluster_k))
        silhouette_scores_s.append(silhouette_score(x_train, cluster_s))
        silhouette_scores_d.append(silhouette_score(x_train, cluster_d))

        print(f'run: {run}')
    predication_average_var_k.append(np.var(silhouette_scores_k))
    predication_average_var_a.append(np.var(silhouette_scores_a))
    predication_average_var_s.append(np.var(silhouette_scores_s))
    predication_average_var_d.append(np.var(silhouette_scores_d))

    silhouette_scores_a_average.append(np.mean(silhouette_scores_a))
    silhouette_scores_k_average.append(np.mean(silhouette_scores_k))
    silhouette_scores_s_average.append(np.mean(silhouette_scores_s))
    silhouette_scores_d_average.append(np.mean(silhouette_scores_d))
    print(f'cluster: {n_clusters}')

print(silhouette_scores_a_average)
print('\n')
print(silhouette_scores_k_average)
print('\n')
print(predication_average_var_k)
print('\n')
print(predication_average_var_a)
print('\n')
print(silhouette_scores_s_average)
print('\n')
print(predication_average_var_s)
print('\n')
print(silhouette_scores_d_average)
print('\n')
print(predication_average_var_d)

plt.plot(range(2,10), silhouette_scores_a_average, c='darkorange', label='Agglomerative Average Silhouette Score')
plt.plot(range(2,10), silhouette_scores_k_average, c='palegreen', label='KMeans Average Silhouette Score')
plt.plot(range(2,10), silhouette_scores_s_average, c='goldenrod', label='Spectral Average Silhouette Score')
plt.plot(range(2,10), silhouette_scores_d_average, c='skyblue', label='DBSCAN Average Silhouette Score')
plt.title('Clustering Scores - Basic\nAverage Silhouette Score\nOver 100 Bootstrapped UnderOver Samples')
plt.legend()
plt.show()


plt.plot(range(2,10), predication_average_var_k, c='deepskyblue', label='KMeans Average Silhouette Variance')
plt.plot(range(2,10), predication_average_var_a, c='lightcoral', label='Agglomerative Average Silhouette Variance')
plt.plot(range(2,10), predication_average_var_s, c='gold', label='Spectral Average Silhouette Variance')
plt.plot(range(2,10), predication_average_var_d, c='powderblue', label='DBSCAN Average Silhouette Variance')
plt.title('Clustering Scores - Basic\nAverage Silhouette Variance\nOver 100 Bootstrapped UnderOver Samples')
plt.legend()
plt.show()



#
# plt.scatter(cluster_under, y_under, c=y_under, cmap='cool')
# plt.show()
# violinplot(x=cluster_under, y=y_under)
# plt.show()
