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
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering, MeanShift
from seaborn import violinplot
from sklearn.metrics import silhouette_samples, silhouette_score, recall_score, mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


def under_sample_binary_0_G0(x, y, test_reserve=.2, oversample=1.5, majority_percent=.5, x_part2 = None):
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
        majority_idxs = np.random.choice(y_0_idxs, round((len(minority_idxs)/(1-majority_percent))*majority_percent), replace=False)
        majority_test_idxs = np.setdiff1d(y_0_idxs, majority_idxs)
    elif len(y_0_idxs) == len(y_G0_idxs):
        print('the classes started balanced')
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
        majority_idxs = np.random.choice(y_G0_idxs, round((len(minority_idxs)/(1-majority_percent))*majority_percent), replace=False)
        majority_test_idxs = np.setdiff1d(y_G0_idxs, majority_idxs)
    train_idx = np.concatenate((minority_idxs, majority_idxs), axis=0)
    test_idx = np.concatenate((minority_test_idxs, majority_test_idxs), axis=0)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    if x_part2 is None:
        return x_train, x_test, y_train, y_test
    else:
        return x_train, x_test, x_part2[train_idx], x_part2[test_idx], y_train, y_test


def print_tree(dtree):
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())

def decision_tree(x_train, x_test, y_train, y_test, x, y, description=''):
    dt = DecisionTreeClassifier()
    scores = cross_validate(estimator=dt, X=x_train, y=y_train, scoring='recall', n_jobs=-1)
    f1scores = cross_validate(estimator=dt, X=x_train, y=y_train, scoring='f1', n_jobs=-1)
    print(f'DecisionTreeClassifier F1 Train, {description}: {f1scores["test_score"]}')
    print(f'DecisionTreeClassifier Recall Train, {description}: {scores["test_score"]}')
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print(f'DecisionTreeClassifier F1 Test, {description}: {f1_score(y_test, y_pred)}')
    print(f'DecisionTreeClassifier Recall Test, {description}: {recall_score(y_test, y_pred)}')
    y_pred = dt.predict(x)
    print(f'DecisionTreeClassifier F1 Full, {description}: {f1_score(y, y_pred)}')
    print(f'DecisionTreeClassifier Recall Full, {description}: {recall_score(y, y_pred)}')
    #print_tree(dt)

def random_forest(x_train, x_test, y_train, y_test, x, y, description=''):
    rfc = RandomForestClassifier(n_estimators=350, n_jobs=-1)
    scores = cross_validate(estimator=rfc, X=x_train, y=y_train, scoring='recall', n_jobs=-1)
    f1scores = cross_validate(estimator=rfc, X=x_train, y=y_train, scoring='f1', n_jobs=-1)
    print(f'RandomForestClassifier F1 Train, just profile: {f1scores["test_score"]}')
    print(f'RandomForestClassifier Recall Train, just profile: {scores["test_score"]}')
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    print(f'RandomForestClassifier F1 Test, just profile: {f1_score(y_test, y_pred)}')
    print(f'RandomForestClassifier Recall Test, just profile: {recall_score(y_test, y_pred)}')
    y_pred = rfc.predict(x)
    print(f'RandomForestClassifier F1 Full, just profile: {f1_score(y, y_pred)}')
    print(f'RandomForestClassifier Recall Full, just profile: {recall_score(y, y_pred)}')


def stripped_down_w_wo_clusters(user_profile, bag_of_words, y, vocab):

    x = user_profile

    for run in range(5):
        #undersample split to compare with standard
        x_train, x_test, words_train, words_test, y_train, y_test = under_sample_binary_0_G0(x, y, test_reserve=.2, majority_percent=.7, oversample=1.5, x_part2=bag_of_words)
        print(f'round: {run}')

        decision_tree(x_train, x_test, y_train, y_test, x, y, 'just profile')
        random_forest(x_train, x_test, y_train, y_test, x, y, 'just profile')

        tfidf_pipe = Pipeline([
        ('vect', CountVectorizer(vocabulary=vocab))
        ])
        word_vecs_train = tfidf_pipe.fit_transform(words_train)
        word_vecs_test = tfidf_pipe.transform(words_test)
        word_vecs_full = tfidf_pipe.transform(bag_of_words)
        x_train_tran = hstack([x_train, word_vecs_train])
        x_test_tran = hstack([x_test, word_vecs_test])
        x_full_tran = hstack([x, word_vecs_full])

        decision_tree(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, 'with word vecs')
        random_forest(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, 'with word vecs')

        tfidf_pipe = Pipeline([
        ('vect', CountVectorizer(vocabulary=vocab)),
        ('tfidf', TfidfTransformer())
        ])
        word_vecs_train = tfidf_pipe.fit_transform(words_train)
        word_vecs_test = tfidf_pipe.transform(words_test)
        word_vecs_full = tfidf_pipe.transform(bag_of_words)
        x_train_tran = hstack([x_train, word_vecs_train])
        x_test_tran = hstack([x_test, word_vecs_test])
        x_full_tran = hstack([x, word_vecs_full])

        decision_tree(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, 'with tfidf vecs')
        random_forest(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, 'with tfidf vecs')

        tfidf_pipe = Pipeline([
        ('vect', CountVectorizer(vocabulary=vocab)),
        ('tfidf', TfidfTransformer()),
        ('tsvd', TruncatedSVD(n_components=150))
        ])
        word_vecs_train = tfidf_pipe.fit_transform(words_train)
        word_vecs_test = tfidf_pipe.transform(words_test)
        word_vecs_full = tfidf_pipe.transform(bag_of_words)
        x_train_tran = hstack([x_train, word_vecs_train])
        x_test_tran = hstack([x_test, word_vecs_test])
        x_full_tran = hstack([x, word_vecs_full])

        decision_tree(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, 'with tfidf tsvd')
        random_forest(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, 'with tfidf tsvd')

        for num in range(2,100, 2):
            tfidf_pipe = Pipeline([
            ('vect', CountVectorizer(vocabulary=vocab)),
            ('cluster', KMeans(n_clusters=num, n_init=50, max_iter=250, n_jobs=-1))
            ])
            word_vecs_train = tfidf_pipe.fit_predict(words_train)
            word_vecs_test = tfidf_pipe.predict(words_test)
            word_vecs_full = tfidf_pipe.predict(bag_of_words)
            x_train_tran = hstack([x_train, word_vecs_train])
            x_test_tran = hstack([x_test, word_vecs_test])
            x_full_tran = hstack([x, word_vecs_full])

            decision_tree(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, f'with {num} cluster on word vec')
            random_forest(x_train_tran, x_test_tran, y_train, y_test, x_full_tran, y, f'with {num} cluster on word vec')




def random_forest_w_wo_clusters(x,y):

    # rfr_score = []
    # rfr_model = []
    # rfr_score_full = []
    # rfr_score_clustered_full = []
    rfc_score = []
    rfc_f1score = []
    rfc_score_test = []
    rfc_f1score_test = []
    rfc_model = []
    rfc_score_full = []
    rfc_f1score_full = []
    rfc_score_clustered = []
    rfc_f1score_clustered = []
    rfc_score_clustered_test = []
    rfc_f1score_clustered_test = []
    rfc_score_clustered_full = []
    rfc_f1score_clustered_full = []
    # best_tfidf_pipe = []
    # rfr_score_clustered = []
    # rfr_model_clustered = []
    rfc_model_clustered = []
    pipelines = []

    for run in range(5):
        #undersample split to compare with standard
        x_train, x_test,y_train, y_test = under_sample_binary_0_G0(x, y, .2, 1.5, .7)
        x_train_raw, x_test_raw,y_train_raw, y_test_raw  = under_sample_binary_0_G0(x, y_raw)
        print(f'round: {run}')
        # x_train_under, x_test_under, y_train_under, y_test_under = train_test_split(x_under,y_under)

        x_pipe = Pipeline([
    	('vect', CountVectorizer(vocabulary=vocab))
        ])
        x_train_tran = x_pipe.fit_transform(x_train)
        x_test_tran = x_pipe.transform(x_test)
        x_full = x_pipe.transform(x)
        pipelines.append(x_pipe)
        # x_train_raw_tran = x_pipe.fit_transform(x_train_raw)
        # x_test_raw_tran = x_pipe.transform(x_test_raw)
        # pipelines.append(x_pipe)
        #
        # rfr = RandomForestRegressor(n_estimators=350, n_jobs=-1)
        # scores = cross_validate(estimator=rfr, X=x_train_raw_tran, y=y_train_raw, scoring='neg_mean_squared_error', n_jobs=-1)
        # rfr_score.append(np.mean(scores['test_score']))
        # print(rfr_score[-1])
        # rfr.fit(x_train_raw_tran, y_train_raw)
        # rfr_model.append(rfr)
        # rfr_score_full.append(mean_squared_error(y_test_raw, rfr.predict(x_test_raw_tran)))
        # print(rfr_score_full[-1])


        rfc = RandomForestClassifier(n_estimators=350, n_jobs=-1)
        scores = cross_validate(estimator=rfc, X=x_train_tran, y=y_train, scoring='recall', n_jobs=-1)
        f1scores = cross_validate(estimator=rfc, X=x_train_tran, y=y_train, scoring='f1', n_jobs=-1)
        rfc_score.append(np.mean(scores['test_score']))
        rfc_f1score.append(np.mean(f1scores['test_score']))
        print(rfc_score[-1])
        print(rfc_f1score[-1])
        rfc.fit(x_train_tran, y_train)
        rfc_model.append(rfc)
        y_pred = rfc.predict(x_test_tran)
        rfc_score_test.append(recall_score(y_test, y_pred))
        rfc_f1score_test.append(f1_score(y_test, y_pred))
        print(rfc_score_test[-1])
        print(rfc_f1score_test[-1])
        y_pred = rfc.predict(x_full)
        rfc_score_full.append(recall_score(y, y_pred))
        rfc_f1score_full.append(f1_score(y, y_pred))
        print(rfc_score_full[-1])
        print(rfc_f1score_full[-1])

        # rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1)
        # scores = cross_validate(rfr, x_pipe.fit_transform(x), y_raw, scoring='neg_mean_squared_error', cv=None, n_jobs=-1)
        # rfr_score_full.append(np.mean(scores['test_score']))
        # print(rfr_score_full[-1])
        # rfr.fit(x_pipe.fit_transform(x), y_raw)
        # rfr_model_full.append(rfr)
        #
        # rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        # scores = cross_validate(rfc, x_pipe.fit_transform(x), y, scoring='recall', cv=None, n_jobs=-1)
        # rfc_score_full.append(np.mean(scores['test_score']))
        # print(rfc_score_full[-1])
        # rfc.fit(x_pipe.fit_transform(x), y)
        # rfc_model_full.append(rfc)
        #
        n_components=150
        tfidf_pipe = Pipeline([
    	('vect', CountVectorizer(vocabulary=vocab)),
    	('tfidf', TfidfTransformer()),
    	('tsvd', TruncatedSVD(n_components=n_components)),
    	('cluster', KMeans(n_clusters=100, n_init=50, max_iter=250, n_jobs=-1))
        ])
        # x_train_raw_cluster = tfidf_pipe.fit_predict(x_train_raw)
        # x_train_raw_tran = hstack([x_train_raw_tran, x_train_raw_cluster.reshape(-1,1)])
        # x_test_raw_cluster = tfidf_pipe.predict(x_test_raw)
        # x_test_raw_tran = hstack([x_test_raw_tran, x_test_raw_cluster.reshape(-1,1)])
        # pipelines.append(tfidf_pipe)

        x_train_cluster = tfidf_pipe.fit_predict(x_train)
        x_train_tran = hstack([x_train_tran, x_train_cluster.reshape(-1,1)])
        x_test_cluster = tfidf_pipe.predict(x_test)
        x_test_tran = hstack([x_test_tran, x_test_cluster.reshape(-1,1)])
        x_full_cluster = tfidf_pipe.predict(x)
        x_full = hstack([x_full, x_full_cluster.reshape(-1,1)])
        pipelines.append(tfidf_pipe)
        #
        # rfr = RandomForestRegressor(n_estimators=350, n_jobs=-1)
        # scores = cross_validate(rfr, x_train_raw_tran, y_train_raw, scoring='neg_mean_squared_error', n_jobs=-1)
        # rfr_score_clustered.append(np.mean(scores['test_score']))
        # print(rfr_score_clustered[-1])
        # rfr.fit(x_train_raw_tran, y_train_raw)
        # rfr_model_clustered.append(rfr)
        # rfr_score_clustered_full.append(mean_squared_error(y_test_raw, rfr.predict(x_test_raw_tran)))
        # print(rfr_score_clustered_full[-1])

        rfc = RandomForestClassifier(n_estimators=350, n_jobs=-1)
        scores = cross_validate(rfc, x_train_tran, y_train, scoring='recall',  n_jobs=-1)
        f1scores = cross_validate(rfc, x_train_tran, y_train, scoring='f1',  n_jobs=-1)
        rfc_score_clustered.append(np.mean(scores['test_score']))
        rfc_f1score_clustered.append(np.mean(f1scores['test_score']))
        print(rfc_score_clustered[-1])
        print(rfc_f1score_clustered[-1])
        rfc.fit(x_train_tran, y_train)
        rfc_model_clustered.append(rfc)
        y_pred = rfc.predict(x_test_tran)
        rfc_score_clustered_test.append(recall_score(y_test, y_pred))
        rfc_f1score_clustered_test.append(f1_score(y_test, y_pred))
        print(rfc_score_clustered_test[-1])
        print(rfc_f1score_clustered_test[-1])
        y_pred = rfc.predict(x_full)
        rfc_score_clustered_full.append(recall_score(y, y_pred))
        rfc_f1score_clustered_full.append(f1_score(y, y_pred))
        print(rfc_score_clustered_full[-1])
        print(rfc_f1score_clustered_full[-1])

    print(rfc_score, rfc_f1score, rfc_score_test, rfc_f1score_test, rfc_score_full,
        rfc_f1score_full, rfc_score_clustered, rfc_f1score_clustered, rfc_score_clustered_test,
        rfc_f1score_clustered_test, rfc_score_clustered_full, rfc_f1score_clustered_full)

    # with open('rfr_model.pkl', 'wb') as f:
    #     pickle.dump(rfr_model, f)
    with open('rfc_model.pkl', 'wb') as f:
        pickle.dump(rfc_model, f)
    # with open('rfr_model_scores.pkl', 'wb') as f:
    #     pickle.dump(rfr_score, f)
    with open('rfc_model_scores.pkl', 'wb') as f:
        pickle.dump(rfc_score, f)
    with open('rfc_model_f1scores.pkl', 'wb') as f:
        pickle.dump(rfc_f1score, f)
    with open('pipelines.pkl', 'wb') as f:
        pickle.dump(pipelines, f)

    # with open('rfr_model_clustered.pkl', 'wb') as f:
    #     pickle.dump(rfr_model_clustered, f)
    # with open('rfc_model_clustered.pkl', 'wb') as f:
    #     pickle.dump(rfc_model_clustered, f)
    # with open('rfr_score_clustered.pkl', 'wb') as f:
    #     pickle.dump(rfr_score_clustered, f)
    with open('rfc_score_clustered.pkl', 'wb') as f:
        pickle.dump(rfc_score_clustered, f)
    with open('rfc_f1score_clustered.pkl', 'wb') as f:
        pickle.dump(rfc_f1score_clustered, f)

    # with open('rfr_score_full.pkl', 'wb') as f:
    #     pickle.dump(rfr_score_full, f)
    with open('rfc_score_full.pkl', 'wb') as f:
        pickle.dump(rfc_score_full, f)
    with open('rfc_f1score_full.pkl', 'wb') as f:
        pickle.dump(rfc_f1score_full, f)
    # with open('rfr_score_clustered_full.pkl', 'wb') as f:
    #     pickle.dump(rfr_score_clustered_full, f)
    with open('rfc_score_clustered_full.pkl', 'wb') as f:
        pickle.dump(rfc_score_clustered_full, f)
    with open('rfc_f1score_clustered_full.pkl', 'wb') as f:
        pickle.dump(rfc_f1score_clustered_full, f)

    with open('rfc_score_test.pkl', 'wb') as f:
        pickle.dump(rfc_score_test, f)
    with open('rfc_f1score_test.pkl', 'wb') as f:
        pickle.dump(rfc_f1score_test, f)
    with open('rfc_score_clustered_test.pkl', 'wb') as f:
        pickle.dump(rfc_score_clustered_test, f)
    with open('rfc_f1score_clustered_test.pkl', 'wb') as f:
        pickle.dump(rfc_f1score_clustered_test, f)
    print_Random_Forest_Results()


def print_Random_Forest_Results():
    # with open('rfr_score_clustered.pkl', 'rb') as f:
    #     rfr_score_clustered = pickle.load(f)
    with open('rfc_score_clustered.pkl', 'rb') as f:
        rfc_score_clustered = list(pickle.load(f))
    with open('rfc_f1score_clustered.pkl', 'rb') as f:
        rfc_f1score_clustered = list(pickle.load(f))
    # with open('rfr_score_full.pkl', 'rb') as f:
    #     rfr_score_full = pickle.load(f)
    with open('rfc_score_full.pkl', 'rb') as f:
        rfc_score_full = list(pickle.load(f))
    with open('rfc_f1score_full.pkl', 'rb') as f:
        rfc_f1score_full = list(pickle.load(f))
    # with open('rfr_score_clustered_full.pkl', 'rb') as f:
    #     rfr_score_clustered_full = pickle.load(f)
    with open('rfc_score_clustered_full.pkl', 'rb') as f:
        rfc_score_clustered_full = list(pickle.load(f))
    with open('rfc_f1score_clustered_full.pkl', 'rb') as f:
        rfc_f1score_clustered_full = list(pickle.load(f))
    with open('rfc_score_test.pkl', 'rb') as f:
        rfc_score_test = list(pickle.load(f))
    with open('rfc_f1score_test.pkl', 'rb') as f:
        rfc_f1score_test = list(pickle.load(f))
    with open('rfc_score_clustered_test.pkl', 'rb') as f:
        rfc_score_clustered_test = list(pickle.load(f))
    with open('rfc_f1score_clustered_test.pkl', 'rb') as f:
        rfc_f1score_clustered_test = list(pickle.load(f))


    # plt.scatter(range(len(rfr_score)), rfr_score, c='m', label='RandomForestRegressor NMSE')
    # plt.scatter(range(len(rfr_score_clustered)), rfr_score_clustered, c='navy', label='RandomForestRegressor NMSE with Clustering')
    # plt.scatter(range(len(rfr_score_full)), rfr_score_full, c='darkgreen', label='RandomForestRegressor MSE Test')
    # plt.scatter(range(len(rfr_score_clustered_full)), rfr_score_clustered_full, c='firebrick', label='RandomForestRegressor MSE Test with Clustering')
    # plt.title('RandomForestRegressor Scores\nTrain and Test\nWith and Without KMeans Clustering')
    # plt.legend(loc=0)
    # plt.show()

    'train data is 50/50 so recall and f1 would be .5 with random, total true 3570'
    'test data is 595 true to 131406, which is .45%, so recall would still be .5 and the f1 score would be .0081'
    plt.scatter(range(len(rfc_score)), rfc_score, c='mediumblue', label='RandomForestClassifier Recall')
    plt.scatter(range(len(rfc_score_clustered)), rfc_score_clustered, c='maroon', label='RandomForestClassifier Recall with Clustering')
    plt.scatter(range(len(rfc_score_full)), rfc_score_test, c='c', label='RandomForestClassifier Recall Test')
    plt.scatter(range(len(rfc_score_clustered_full)), rfc_score_clustered_test, c='darkviolet', label='RandomForestClassifier Recall Test with Clustering')
    plt.scatter(range(len(rfc_score_full)), rfc_score_full, c='navy', label='RandomForestClassifier Recall Full Dataset')
    plt.scatter(range(len(rfc_score_clustered_full)), rfc_score_clustered_full, c='darkgreen', label='RandomForestClassifier Recall Full Dataset with Clustering')
    plt.plot(range(5), [.5]*5, c='dimgray', label='Recall - Random Guess')
    plt.title('RandomForestClassifier Scores\nTrain and Test\nWith and Without KMeans Clustering')
    plt.legend(loc=0)
    plt.show()

    plt.scatter(range(len(rfc_f1score)), rfc_f1score, c='m', label='RandomForestClassifier F1')
    plt.scatter(range(len(rfc_f1score_clustered)), rfc_f1score_clustered, c='navy', label='RandomForestClassifier F1 with Clustering')
    plt.scatter(range(len(rfc_f1score_full)), rfc_f1score_test, c='navy', label='RandomForestClassifier F1 Test')
    plt.scatter(range(len(rfc_f1score_clustered_full)), rfc_f1score_clustered_test, c='c', label='RandomForestClassifierF1 Test with Clustering')
    plt.scatter(range(len(rfc_f1score_full)), rfc_f1score_full, c='darkgreen', label='RandomForestClassifier F1 Full Dataset')
    plt.scatter(range(len(rfc_f1score_clustered_full)), rfc_f1score_clustered_full, c='firebrick', label='RandomForestClassifierF1 Full Dataset with Clustering')
    plt.plot(range(5), [.3750]*5, c='dimgray', label='F1 - Random Guess - Train')
    plt.plot(range(5), [.0091]*5, c='r', label='F1 - Random Guess - Test')
    plt.plot(range(5), [.0385]*5, c='violet', label='F1 - Random Guess - Full Dataset')
    plt.title('RandomForestClassifier Scores\nTrain and Test\nWith and Without KMeans Clustering')
    plt.legend(loc=0)
    plt.show()

def best_clustering(x,y, vocab):
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

def single_cluster_model_sil(x, y, vocab):
    n_components=150
    silhouette_scores=[]
    kmeans_scores=[]
    for num in range(25):
        print(num)
        x_train, x_test, y_train, y_test = under_sample_binary_0_G0(x, y, test_reserve=.2, oversample=1, majority_percent=.5)
        tfidf_pipe = Pipeline([
        ('vect', CountVectorizer(vocabulary=vocab)),
        ('tfidf', TfidfTransformer()),
        ('tsvd', TruncatedSVD(n_components=n_components))
        ])
        x_tran = tfidf_pipe.fit_transform(x_train)
        kmn = KMeans(n_clusters=num, n_init=50, max_iter=250, n_jobs=-1)
        clusters = kmn.fit_predict(x_tran)
        silhouette_scores.append(silhouette_score(x_tran, clusters))
        print(silhouette_scores[-1])
        #kmeans_scores.append(kmn.score(x_tran))
    plt.plot(range(25), silhouette_scores, c='palegreen', label='Silhouette Score')
    #plt.plot(range(2,500), kmeans_scores, c='violet', label='KMeans Score')
    plt.title('MeanShift Clustering\nScores Across Cluster Counts')
    plt.legend(loc=0)
    plt.show()

def evaluate_clusters(x, y, vocab, n_clusters):
    x_train, x_test, y_train, y_test = under_sample_binary_0_G0(x, y, test_reserve=.2, oversample=1.5, majority_percent=.7)
    tfidf_pipe = Pipeline([
    ('vect', CountVectorizer(vocabulary=vocab)),
    ('tfidf', TfidfTransformer()),
    ('tsvd', TruncatedSVD(n_components=150)),
    ('cluster', KMeans(n_clusters=n_clusters, n_init=50, max_iter=250, n_jobs=-1))
    ])
    clusters = tfidf_pipe.fit_predict(x_train)

    plt.scatter(clusters, y_train, c=y_train, cmap='cool')
    plt.title('Customer Conversion Across Clusters')
    plt.show()
    violinplot(x=clusters, y=y_train)
    plt.title('Customer Conversion Across Clusters')
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



if __name__ == '__main__':
    # user_profile = np.load('data/user_profile_modes_array_dummies')
    x = np.load('data/x_array.npy')
    y = np.load('data/y_array.npy')
    vocab = pd.read_csv('data/cat_to_num.csv', usecols=['ID']).values.astype(str)
    vocab = vocab[:,0].tolist()

    y = np.where(y > 0, 1, 0)

    best_clustering(x,y, vocab)
    single_cluster_model_sil(x, y, vocab)
    evaluate_clusters(x, y, vocab, 50)
    random_forest_w_wo_clusters(x,y)


    #stripped_down_w_wo_clusters(user_profile, x, y, vocab)

    # best_clustering(x,y, vocab)

    #standard train test split to cut the data in half
    # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.5)

    #undersample split to compare with standard
    # x_train, x_test, y_train, y_test = under_sample_binary_0_G0(x, y)



            # with open('rfr_model_scores.pkl', 'rb') as f:
            #     rfr_score = pickle.load(f)
            # with open('rfc_model_scores.pkl', 'rb') as f:
            #     rfc_score = pickle.load(f)
            #
            #
            # x_pipe = Pipeline([('vect', CountVectorizer())])
            #
            # with open('rfr_model.pkl', 'rb') as f:
            #     rfr_models = pickle.load(f)
            # with open('rfc_model.pkl', 'rb') as f:
            #     rfc_models = pickle.load(f)
            #
            # x_tran = x_pipe.fit_transform(x)
            # rfc_f1 = [f1_score(y, model.predict(x_tran)) for model in rfc_models]
            # rfc_recall = [recall_score(y, model.predict(x_tran)) for model in rfc_models]
            # rfr_mse = [mean_squared_error(y_raw, model.predict(x_tran)) for model in rfr_models]
            #
            #
            # plt.scatter(range(5), -1*rfr_score, c='pink', label='RandomForestRegressorMSE with Under Sampling')
            # plt.scatter(range(5), rfc_score, c='springgreen', label='RandomForestClassifierRecall with Under Sampling')
            # plt.scatter(range(5), rfr_mse, c='hotpink', label='RandomForestRegressorMSE with All Data')
            # plt.scatter(range(5), rfc_recall, c='green', label='RandomForestClassifierRecall with All Data')
            # plt.scatter(range(5), rfc_f1, c='darkgreen', label='RandomForestClassifierF1 with All Data')
            # plt.title("Regression and Classification Scores\nCategory Counts For Browsing Behavior\nWithout Clustering")
            # plt.legend()
            # plt.show()
