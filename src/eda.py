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
from sklearn.cluster import AgglomerativeClustering
from seaborn import violinplot
from sklearn.metrics import silhouette_samples, silhouette_score

from pprint import pprint
from time import time
import logging

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def under_sample_binary_0_G0(x, y):
    '''
    undersamples based on y <= 0 and y > 0 as binary options
    will use all of smaller class and random sample of larger class to match
    '''
    y = y.astype(int)
    y_0_idxs = np.where(y <= 0)[0]
    y_G0_idxs = np.where(y > 0)[0]
    if len(y_0_idxs) > len(y_G0_idxs):
        minority_idxs = y_G0_idxs
        majority_idxs = np.random.choice(y_0_idxs, len(y_G0_idxs), replace=False)
    else:
        minority_idxs = y_0_idxs
        majority_idxs = np.random.choice(y_G0_idxs, len(y_0_idxs), replace=False)
    idx = np.concatenate((minority_idxs, majority_idxs), axis=0)
    x_under = x[idx]
    y_under = y[idx]
    return x_under, y_under, idx

def basic_classifier(x,y):
    '''code example for pipeline and grid search'''
    # #############################################################################
    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }


    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(x, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

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

#standard train test split to cut the data in half
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)

#undersample split to compare with standard
x_under, y_under, under_idx = under_sample_binary_0_G0(x, y)


n_components=150

tfidf_pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('tsvd', TruncatedSVD(n_components=n_components)),
])
# tfidf_pipe.fit(x_train)
# x_train = tfidf_pipe.transform(x_train)
# cum_scree_plot(tfidf_pipe.named_steps['tsvd'], n_components, title="Website Categories TSVD\nStandard Split")
#
tfidf_pipe.fit(x_under)
x_under = tfidf_pipe.transform(x_under)
#cum_scree_plot(tfidf_pipe.named_steps['tsvd'], n_components, title="Website Categories TSVD\nUnder Sampling")

'''The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.'''

silhouette_scores = []
for n_clusters in range(2, 100):
    cluster_ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='manhattan', linkage='complete')
    cluster_under = cluster_ward.fit_predict(x_under)
    silhouette_scores.append(silhouette_score(x_under, cluster_under))
plt.plot(range(2,100), silhouette_scores)
plt.title('Agglormerative Clustering - L2 Average\nSilhouette Score by Number of Clusters\nUnder Sampling')
plt.show()



# plt.scatter(cluster_under, y_under, c=y_under, cmap='cool')
# plt.show()
# violinplot(x=cluster_under, y=y_under)
# plt.show()



'''ran for 1.5 hours, not done'''
# x_train_tsne = TSNE(n_components=3).fit_transform(tfidf_pipe.transform(x_train))
# tsne_3d_scatter(x_train_tsne, y_train, 'TSNE Top 3 Diminsions\nColored By Actual Conversions\nStandard Split')
#
# mds = MDS(n_components=3,  n_jobs=-1)
# x_train_mds = mds.fit_transform(tfidf_pipe.transform(x_train))
# print('saving model')
# pickle.dump(mds, 'mds_standard.pkl')
# print('attempting to plot')
# tsne_3d_scatter(x_train_mds, y_train, 'MDS Top 3 Diminsions\nColored By Actual Conversions\nStandard Split')
