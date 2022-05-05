from data_preparation import tracks

classification_df = tracks.drop(
    columns=['id', 'artists', 'name', 'artists_song', 'artists_upd', 'popularity', 'explicit', 'decade'])

# Elimino le righe che non hanno alcun genere specificato
check = classification_df['genre'].apply(lambda x: x == [])
classification_df = classification_df.loc[~check]

# Prendo il primo genere nella lista
classification_df['genre'] = classification_df['genre'].apply(lambda x: x[0])

genres = ['rock', 'pop', 'classical', 'jazz', 'metal', 'punk', 'folk', 'techno', 'country', 'dance', 'disco', 'rap',
          'hip_hop', 'classic', 'opera', 'romantic']

# Semplifico i generi
for genre in genres:
    classification_df['genre'] = classification_df['genre'].apply(
        lambda x: genre if genre in x else x)

n_genres = 10
# Genero una lista dei generi più presenti nel dataset

top_genres = classification_df['genre'].value_counts().head(n_genres).index

# Filtro il dataframe con la top n_genres dei generi presenti
check = classification_df['genre'].apply(lambda x: x in top_genres)
classification_df = classification_df.loc[check]

# Passo da un valore stringa a un valore numerico partendo da 1
classification_df['genre'] = classification_df['genre'].apply(
    lambda x: (top_genres.get_loc(x) + 1))
