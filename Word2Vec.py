import numpy as np
from gensim import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def createmodel(df):
    print('create model:')
    docs = df['processed_body']
    asentences = createsentences(docs.iloc[0])
    model = models.Word2Vec(sentences=asentences, min_count=0, vector_size=200, workers=5, window=7, sg=1)
    for i in tqdm(range(1, len(df))):
        Asentences = createsentences(df['processed_body'].iloc[i])
        token_count = len(Asentences)
        model.train(corpus_iterable=Asentences, word_count=0, total_examples=token_count, epochs=10)
    return model


def createsentences(doc):
    sentences = []
    where = [0]
    docl = doc.split()

    # Create a list of indexes of words with periods
    for i in range(0, len(docl)):
        if '.' in docl[i]:
            where.append(i)
            where.append(i + 1)

    # Create a list of sentences according to the indexes of words with periods
    for i in range(0, int(len(where) / 2) + 1, 2):
        sentence = []
        for j in range(where[i], where[i + 1] + 1):
            sentence.append(docl[j])
        sentences.append(sentence)

    # Remove periods from final sentences list
    for i in range(0, len(sentences)):
        sent = sentences[i]
        for j in range(0, len(sent)):
            sent[j] = sent[j].replace('.', '', )

    return sentences


def plot_with_matplotlib(x_vals, y_vals, labels):
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    labels = list(range(len(labels)))
    for i in labels:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    return plt

def reduce_dimensions(model):
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)

    pca = PCA(n_components=2)
    vectors = pca.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels