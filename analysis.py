import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from spoty_api import sp
from data_preparation import tracks, reals

import numpy as np
from data_preparation import top_20_genres, clustering_df
from modeling import predict_genre, model, kmeans

# complete_feature_set = create_feature_set(tracks, reals)
# complete_feature_set.to_csv('feature_set_x10.csv', index=False)
complete_feature_set = pd.read_csv('feature_set_x10.csv')


# complete_feature_set = pd.read_csv('feature_set_x4.csv')


def ohe_prep(df, column, new_name):
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df


def build_tfidf_set(df, float_cols):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    genre_df.reset_index(drop=True, inplace=True)

    year_ohe = ohe_prep(df, 'year', 'year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_bins', 'pop') * 0.15

    floats = df[float_cols].reset_index(drop=True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2

    # concantenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)

    # add song id
    final['id'] = df['id'].values

    return final


def create_necessary_outputs(playlist_name, id_dic, df):
    playlist = pd.DataFrame()
    playlist_name = playlist_name

    for ix, i in enumerate(sp.playlist(id_dic)['tracks']['items']):
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id']  # ['uri'].split(':')[2]
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])

    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added', ascending=False)

    return playlist


def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    complete_feature_set_playlist = complete_feature_set[
        complete_feature_set['id'].isin(playlist_df['id'].values)]  # .drop('id', axis = 1).mean(axis =0)
    print(complete_feature_set_playlist.shape)
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id', 'date_added']], on='id',
                                                                        how='inner')
    complete_feature_set_nonplaylist = complete_feature_set[
        ~complete_feature_set['id'].isin(playlist_df['id'].values)]  # .drop('id', axis = 1)

    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added', ascending=False)

    most_recent_date = playlist_feature_set.iloc[0, -1]

    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix, 'months_from_recent'] = int(
            (most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)

    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))

    playlist_feature_set_weighted = playlist_feature_set.copy()
    playlist_feature_set_weighted.update(
        playlist_feature_set_weighted.iloc[:, :-4].mul(playlist_feature_set_weighted.weight, 0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]

    return playlist_feature_set_weighted_final.sum(axis=0), complete_feature_set_nonplaylist


def generate_playlist_recos(df, features, nonplaylist_features):
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis=1).values,
                                               features.values.reshape(1, -1))[:, 0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending=False).head(40)
    non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].apply(
        lambda x: sp.track(x)['album']['images'][1]['url'])

    return non_playlist_df_top_40


def analyze_url(url):
    type = -1
    payload = []
    try:
        id = url.split("/")[4].split("?")[0]
    except:
        return type, payload
    try:
        sp.playlist(id)
        type = 1
        payload = analyze_playlist(id)
        return type, payload
    except Exception as e:
        print("URL Exception 1: ")
        print(e)
        try:
            sp.track(id)
            type = 2
            payload = [analyze_track(id)]
            return type, payload
        except:
            try:
                sp.album(id)
                type = 3
                return type, payload
            except:
                return type, payload


def analyze_playlist(id):
    playlist = sp.playlist(id)
    playlist_name = playlist['name']
    original_tracks = playlist['tracks']
    songs = []
    artist = []
    image_url = []
    for track in original_tracks['items']:
        songs.append(track['track']['name'])
        artist.append(track['track']['artists'][0]['name'])
        image_url.append(track['track']['album']['images'][0]['url'])

    analyzed = np.stack((songs, artist, image_url), axis=1)
    outputs = create_necessary_outputs(playlist_name, id, tracks)
    playlist_feature_set, general_feature_set = generate_playlist_feature(complete_feature_set, outputs, 1.09)
    recommended = generate_playlist_recos(tracks, playlist_feature_set, general_feature_set)

    return analyzed, recommended[['name', 'artists', 'url']].values, playlist_name


def analyze_track(id):
    track_data = sp.track(id)
    track_features = sp.audio_features(id)[0]
    track_features_array = np.array(list(track_features.items()))
    features = dict()
    # Aggiungo a feature_array solo i valori delle feature
    for feature in track_features_array:
        features[feature[0]] = feature[1]

    features['popularity_bins'] = int(track_data['popularity'] / 5)
    # Aggiungo ulteriori dati alle feature
    features.pop('type')
    features.pop('id')
    features.pop('uri')
    features.pop('track_href')
    features.pop('analysis_url')
    song_info = dict()
    song_info['album_url'] = track_data['album']['images'][0]['url']
    song_info['name'] = track_data['name']
    song_info['artist'] = track_data['artists'][0]['name']
    prediction_probabilities, suggestions = predict_genre(features, model)
    probabilities = dict()
    try:
        # Format predictions
        for genre in top_20_genres:
            probabilities[genre] = int(prediction_probabilities[0][list(top_20_genres).index(genre)] * 100)
    except Exception as e:
        print("Analyze Track Exception:")
        print(e)

    probabilities = sorted(probabilities.items(), reverse=True, key=lambda x: x[1])

    return features, song_info, probabilities, suggestions
