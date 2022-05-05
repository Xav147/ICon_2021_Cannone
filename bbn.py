import pandas as pd

from classification_dataset import classification_df

from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController


def create_bins(df, column_filter):
    for col in df.columns:
        if col not in column_filter:
            df[col] = pd.cut(df[col], bins=[df[col].min(), df[col].mean(), df[col].max()],
                             labels=['Low', 'High'],
                             include_lowest=True)


def probs(data, child, parent1=None, parent2=None):
    if parent1 is None:
        # Calculate probabilities
        prob = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(
            -1).tolist()
    elif parent1 is not None:
        # Check if child node has 1 parent or 2 parents
        if parent2 is None:
            # Caclucate probabilities
            prob = pd.crosstab(data[parent1], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        else:
            # Caclucate probabilities
            prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        print("Error in Probability Frequency Calculations")
    return prob


# Define a function for printing marginal probabilities
def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')


def evidence(ev, nod, cat, val):
    ev = EvidenceBuilder() \
        .with_node(join_tree.get_bbn_node_by_name(nod)) \
        .with_evidence(cat, val) \
        .build()
    join_tree.set_observation(ev)


base_df = classification_df.copy()

base_df.drop(columns=['duration_ms', 'popularity_bins', 'time_signature', 'year'], axis=1, inplace=True)

exclude = ['key', 'genre', 'mode']
create_bins(base_df, exclude)

loudness = BbnNode(Variable('loudness', 'loudness', ['Low', 'High']), probs(base_df, 'loudness'))
energy = BbnNode(Variable('energy', 'energy', ['Low', 'High']), probs(base_df, 'energy', 'loudness'))
tempo = BbnNode(Variable('tempo', 'tempo', ['Low', 'High']), probs(base_df, 'tempo'))
key = BbnNode(Variable('key', 'key', [str(x) for x in range(0, 12)]), probs(base_df, 'key'))
mode = BbnNode(Variable('mode', 'mode', ['0', '1']), probs(base_df, 'mode'))
danceability = BbnNode(Variable('danceability', 'danceability', ['Low', 'High']),
                       probs(base_df, 'danceability', 'energy', 'tempo'))
valence = BbnNode(Variable('valence', 'valence', ['Low', 'High']), probs(base_df, 'valence', 'mode', 'key'))
genre = BbnNode(Variable('genre', 'genre', [str(x) for x in range(1, 11)]),
                probs(base_df, 'genre', 'valence', 'danceability'))

bbn = Bbn() \
    .add_node(loudness) \
    .add_node(energy) \
    .add_node(danceability) \
    .add_node(valence) \
    .add_node(tempo) \
    .add_node(key) \
    .add_node(mode) \
    .add_node(genre) \
    .add_edge(Edge(loudness, energy, EdgeType.DIRECTED)) \
    .add_edge(Edge(energy, danceability, EdgeType.DIRECTED)) \
    .add_edge(Edge(tempo, danceability, EdgeType.DIRECTED)) \
    .add_edge(Edge(mode, valence, EdgeType.DIRECTED)) \
    .add_edge(Edge(key, valence, EdgeType.DIRECTED)) \
    .add_edge(Edge(valence, genre, EdgeType.DIRECTED)) \
    .add_edge(Edge(danceability, genre, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)


def convert_features(df, features):
    for feature in features:
        if feature not in ['key', 'mode']:
            if features[feature] < df[feature].mean():
                features[feature] = 'Low'
            else:
                features[feature] = 'High'
        else:
            features[feature] = str(features[feature])


def bayes_predict(song_features):
    convert_features(classification_df, song_features)
    bbn_features = ['loudness', 'energy', 'tempo', 'key', 'mode', 'danceability', 'valence']
    for feature in bbn_features:
        evidence(feature, feature, song_features[feature], 1.0)
    print_probs()
