import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate

y = pickle.load( open( "y_full.pkl", "rb" ) )
y = y.values
x = np.load('user_profile_modes_array_dummies')

y_class = y = np.where(y > 0, 1, 0)

rfr_score = []
rfr_model = []
rfc_score = []
rfc_model = []
# best_tfidf_pipe = []
rfr_score_clustered = []
rfc_score_clustered = []
rfr_model_clustered = []
rfc_model_clustered = []

for run in range(5):
    #undersample split to compare with standard
#     x_under, y_under, under_idx = under_sample_binary_0_G0(x, y)
#     print(f'y_under: {np.unique(y_under)}')
#     x_under_raw, y_under_raw, under_idx_raw = under_sample_binary_0_G0(x, y_raw)
#     print(f'y_under_raw: {np.unique(y_under_raw)}')
    # x_train_under, x_test_under, y_train_under, y_test_under = train_test_split(x_under,y_under)

#     x_pipe = Pipeline([
#         ('vect', CountVectorizer())
#     ])
    rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    scores = cross_validate(rfr, x, y, scoring='neg_mean_squared_error', cv=None, n_jobs=-1)
    rfr_score.append(np.mean(scores['test_score']))
    print(rfr_score[-1])
    rfr.fit(x, y)
    rfr_model.append(rfr)

    rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    scores = cross_validate(rfc, x, y_class, scoring='recall', cv=None, n_jobs=-1)
    rfc_score.append(np.mean(scores['test_score']))
    print(rfc_score[-1])
    rfc.fit(x, y_class)
    rfc_model.append(rfc)
    #
    # n_components=150
    # tfidf_pipe = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('tsvd', TruncatedSVD(n_components=n_components)),
    #     ('cluster'), KMeans(n_clusters=100, n_init=50, max_iter=500, n_jobs=-1)
    # ])
    # x_under_raw_cluster = tfidf_pipe.fit_transform(x_under_raw)
    # x_under_raw = np.hstack(x_pipe.fit_transform(x_under_raw), x_under_raw_cluster.reshape(-1,1))
    # x_under_cluster = tfidf_pipe.fit_transform(x_under)
    # x_under = np.hstack(x_pipe.fit_transform(x_under), x_under_cluster.reshape(-1,1))
    #
    # rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    # scores = cross_validate(rfr, x_under_raw, y_under_raw, scoring='neg_mean_squared_error', cv=None, n_jobs=-1)
    # rfr_score_clustered.append(np.mean(scores['test_score']))
    # print(rfr_score_clustered[-1])
    # rfr.fit(x_pipe.fit_transform(x_under_raw), y_under_raw)
    # rfr_model_clustered.append(rfr)
    #
    # rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    # scores = cross_validate(rfc, x_under, y_under, scoring='recall', cv=None, n_jobs=-1)
    # rfc_score_clustered.append(np.mean(scores['test_score']))
    # print(rfc_score_clustered[-1])
    # rfc.fit(x_pipe.fit_transform(x_under), y_under)
    # rfc_model_clustered.append(rfc)

pickle.dump(rfr_model, 'rfr_model_profile.pkl')
pickle.dump(rfc_model, 'rfc_model_profile.pkl')
pickle.dump(rfr_score, 'rfr_model_scores_profile.pkl')
pickle.dump(rfc_score, 'rfc_model_scores_profile.pkl')
# pickle.dump(rfr_model_clustered, 'rfr_model_clustered.pkl')
# pickle.dump(rfc_model_clustered, 'rfc_model_clustered.pkl')

plt.scatter(range(5), rfr_score[:,0], c='pink', label='RandomForestRegressorNMSE')
plt.scatter(range(5), rfc_score[:,0], c='springgreen', label='RandomForestClassifierRecall')
# plt.scatter(range(100), rfr_score_clustered[:,0], c='hotpink', label='RandomForestRegressorNMSE with Clusters')
# plt.scatter(range(100), rfc_score_clustered[:,0], c='green', label='RandomForestClassifierRecall with Clusters')
plt.legend()
plt.show()
