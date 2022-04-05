import numpy as np
import pandas as pd
import os
import joblib

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from data_preparation import tracks, clustering_df, classification_df, top_20_genres


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    print("    " + fst_empty_cell, end=" ")

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def kmeans_clustering(df):
    if (os.path.exists('./clustering/clusters')):
        clusters = joblib.load('./clustering/clusters')
    else:
        clusters = KMeans(n_clusters=1000, random_state=0).fit(df)
        joblib.dump(clusters, './clustering/clusters')
    return clusters


def split_classification_df(df):
    y = df['consolidates_genre_lists']
    x = df.drop('consolidates_genre_lists', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
    return x_train, x_test, y_train, y_test


def train_model(model, X_train, X_test, y_train, y_test, filename):
    model.fit(X_train.values, y_train)
    joblib.dump(model, filename)
    score = model.score(X_test, y_test)
    return model, score


def get_stats(name, model):
    y_pred = model.predict(x_test)
    stats = {
        "Name": name,
        "Accuracy": metrics.accuracy_score(y_test, y_pred),
        "Misclassification rate": 1 - metrics.accuracy_score(y_test, y_pred),
        "Precision score": metrics.precision_score(y_test, y_pred, average='weighted'),
        "Recall score": metrics.recall_score(y_test, y_pred, average='weighted')
    }
    return stats


x_train, x_test, y_train, y_test = split_classification_df(classification_df)
kmeans = kmeans_clustering(clustering_df)
score = []

models = {
    'RandomForest': [RandomForestClassifier(), "./classifiers_no_feature_selection/RandomForest"],
    'LogisticRegression': [LogisticRegression(max_iter=5000), "./classifiers_no_feature_selection/LogisticRegression"],
    'NeuralNetwork': [MLPClassifier(), "./classifiers_no_feature_selection/NeuralNetwork"]
}

model = RandomForestClassifier()
filename = "./classifiers_no_feature_selection/RandomForest"
if os.path.exists(filename):
    model = joblib.load(filename)
else:
    model, acc = train_model(model, x_train, x_test, y_train, y_test, filename)


def predict_genre(song_features, model):
    try:
        song_features = np.array(list(song_features.values())).reshape(1, -1)
        probabilities = model.predict_proba(song_features)
        song_cluster = kmeans.predict(song_features)
        suggestions = []

        for i in range(len(clustering_df)):
            if kmeans.labels_[i] == song_cluster[0]:
                suggestions.append(tracks.iloc[i])
        filter_suggestions = []
        for sugg in suggestions:
            filter_suggestions.append({'name': sugg['name'], 'artist': sugg.artists_upd[0]})
    except Exception as e:
        print("Predict genre Exception: ")
        print(e)
    return probabilities, filter_suggestions
