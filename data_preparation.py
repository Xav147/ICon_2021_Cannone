import pandas as pd
import numpy as np
import re
import itertools

tracks = pd.read_csv("./data/spotify160k/tracks.csv")
artists = pd.read_csv("./data/spotify160k/artists.csv")

data_split = 1
tracks_split = int(len(tracks) / data_split)
tracks = tracks.iloc[:tracks_split]
artists_split = int(len(artists) / data_split)
artists = artists.iloc[:artists_split]

# Cambio colonna ID artista
artists.rename(columns={'id': 'artist_id'}, inplace=True)

# Creo le feature year e decade
tracks['year'] = tracks['release_date'].apply(lambda x: x.split('-')[0])
tracks['year'] = tracks['year'].apply(lambda x: int(x))
tracks['decade'] = tracks['year'].apply(lambda x: int(x / 10) * 10 - 1900 if x < 2000 else int(x / 10) * 10 - 2000)

# Salvo le feature float del dataset per riutilizzarle dopo
reals = tracks.dtypes[tracks.dtypes == 'float64'].index.values

# Rendo popularity un valore in 5 bin
tracks['popularity_bins'] = tracks['popularity'].apply(lambda x: int(x / 5))

# Semplifico generi
artists['genres_upd'] = artists['genres'].apply(lambda x: [re.sub(' ', '_', i) for i in re.findall(r"'([^']*)'", x)])

# Semplifico artisti
tracks['artists_upd_v1'] = tracks['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
tracks[tracks['artists_upd_v1'].apply(lambda x: not x)].head(5)
tracks['artists_upd_v2'] = tracks['artists'].apply(lambda x: re.findall('\"(.*?)\"', x))
tracks['artists_upd'] = np.where(tracks['artists_upd_v1'].apply(lambda x: not x), tracks['artists_upd_v2'],
                                 tracks['artists_upd_v1'])
# Creo la feature artists_song = Artista + Titolo canzone
tracks['artists_song'] = tracks.apply(lambda row: str(row['artists_upd'][0]) + str(row['name']), axis=1)

# Ordino per titolo e release date
tracks.sort_values(['artists_song', 'release_date'], ascending=False, inplace=True)

# Elimino duplicati
tracks.drop_duplicates('artists_song', inplace=True)

# Creo generi per artista
artists_exploded = tracks[['artists_upd', 'id']].explode('artists_upd')

artists_exploded_enriched = artists_exploded.merge(artists, how='left', left_on='artists_upd', right_on='name')
artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres.isnull()]

artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()

artists_genres_consolidated['genre'] = artists_genres_consolidated['genres_upd'].apply(
    lambda x: list(set(list(itertools.chain.from_iterable(x)))))

# Unisco generi artista con generi traccia
tracks = tracks.merge(artists_genres_consolidated[['id', 'genre']], on='id', how='left')
tracks['genre'] = tracks['genre'].apply(
    lambda d: d if isinstance(d, list) else [])

# Elimino colonne inutili
tracks.drop(columns=['id_artists', 'artists_upd_v1', 'artists_upd_v2', 'release_date'], inplace=True)

# Preparo copie del dataframe per classificazione e clustering
clustering_df = tracks.drop(
    columns=['id', 'genre', 'artists', 'artists_song', 'name', 'artists_upd', 'explicit', 'decade',
             'popularity'])
