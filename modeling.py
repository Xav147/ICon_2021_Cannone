import numpy as np
import pandas as pd
import os
import joblib
import logging

from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from data_preparation import tracks, clustering_df
from classification_dataset import classification_df

X = classification_df.drop('genre', axis=1)
y = classification_df['genre']


def kmeans_clustering(df):
    if os.path.exists('./clustering/clusters'):
        clusters = joblib.load('./clustering/clusters')
    else:
        clusters = KMeans(n_clusters=1000, random_state=0).fit(df)
        joblib.dump(clusters, './clustering/clusters')
    return clusters


def split_classification_df(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
    return x_train, x_test, y_train, y_test


def train_model(model, x_train, y_train, savepath):
    model.fit(x_train.values, y_train)
    joblib.dump(model, savepath)
    return model


def get_stats(model):
    y_pred = model.predict(x_test.values)
    cvs = cross_val_score(model, X=X, y=y, cv=10)
    stats = {
        "Accuracy": metrics.accuracy_score(y_test, y_pred),
        "Misclassification rate": 1 - metrics.accuracy_score(y_test, y_pred),
        "Precision score": metrics.precision_score(y_test, y_pred, average='weighted'),
        "Recall score": metrics.recall_score(y_test, y_pred, average='weighted'),
        "Cross validation scores": cvs
    }
    return stats


x_train, x_test, y_train, y_test = split_classification_df(X, y)
kmeans = kmeans_clustering(clustering_df)
score = []

savepath = "./classifiers/"
presets = [
    # {'model': RandomForestClassifier(), 'savepath': savepath + "RandomForest"},
    # {'model': LogisticRegression(max_iter=5000), 'savepath': savepath + "LogisticRegression"},
    # {'model': MLPClassifier(max_iter=5000), 'savepath': savepath + "MLPC-NN"},
    # {'model': DecisionTreeClassifier(), 'savepath': savepath + "DecisionTree"},
    # {'model': GradientBoostingClassifier(), 'savepath': savepath + "GradientBoosting"},
    {'model': GaussianNB(), 'savepath': savepath + "NaiveBayes"},
]

classifiers = []
stats = {}

for preset in presets:
    if os.path.exists(preset['savepath']):
        model = joblib.load(preset['savepath'])
    else:
        model = train_model(preset['model'], x_train, y_train, preset['savepath'])
    classifiers.append(model)
    model_stats = get_stats(model)
    stats[model.__class__.__name__] = model_stats
    print(model_stats)

print(classifiers)
print(stats)


def predict_genre(song_features, model):
    try:
        song_features = np.array(list(song_features.values())).reshape(1, -1)
        probabilities = model.predict_proba(song_features)
        song_cluster = kmeans.predict(song_features)
        suggestions = []

        for i in range(len(clustering_df)):
            suggestions.append(tracks.iloc[i])
        filter_suggestions = []
        for sugg in suggestions:
            filter_suggestions.append({'name': sugg['name'], 'artist': sugg.artists_upd[0]})
    except Exception as e:
        logging.error(f"Predict genre Exception: {e}")
    return probabilities, filter_suggestions
