import re

import gensim.models.keyedvectors
import pandas as pd
import Parse
from tqdm import tqdm
import Json_reader
import Word2Vec
import TfIdf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == '__main__':
    # setting the settings for pandas
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 2000)

    # loading the data and creating a dataframe out of the csv file in the folder:
    path = 'C:/Users/User/Desktop/finalProj/metadata/'
    json_path = path + 'document_parses/pdf_json/'
    csv_path = path + 'metadata.csv'
    metaDf = pd.read_csv(csv_path)
    metaDf.dropna(axis=0, how='all', subset=['pdf_json_files'], inplace=True)
    a = f'{json_path}d1aafb70c066a2068b02786f8929fd9c900897fb.json'
    #creating a list of all filepaths:
    Jp = []
    for i in range(0, len(metaDf)):
        json_id = metaDf['pdf_json_files'].iloc[i]
        Jp.append(f'{path}{json_id}')

    def containswords(body):
        if re.search('[a-zA-Z]', body):
            return True
        else:
            return False
    # using the read class we created for a new dataframe
    j = 0
    t = 100
    a = []
    for i in tqdm(Jp, total=t):
        if (t == j):
            break
        try:
            data = Json_reader.Read(i, metaDf)
        except Exception:
            continue
        if len(data.body) == 0:
            continue
        if not containswords(data.body):
            continue
        id = data.id
        abstract = data.abstract
        body = data.body
        doi = data.doi
        url = data.url
        author = data.author
        title = data.title
        row = [id, abstract, body, doi, url, author, title]
        a.append(row)
        j = j + 1
    clean = pd.DataFrame(columns=['id', 'abstract', 'body', 'doi', 'url', 'author', 'title'], data=a)
    clean.to_csv(path_or_buf='D:\FinalProj/clean100000.csv')
    # clean=pd.read_csv(filepath_or_buffer='D:\FinalProj/clean.csv')
    parsedclean = Parse.parsedata(clean)
    parsedclean.to_csv(path_or_buf='D:\FinalProj/parsedclean100000.csv')
    parsedclean=pd.read_csv(filepath_or_buffer='D:\FinalProj/parsedclean100000.csv')
    print(len(parsedclean))
    print(parsedclean['processed_body'].iloc[0])

    unique = TfIdf.uniquewords(parsedclean)
    w2vmodel = Word2Vec.createmodel(parsedclean)
    w2vmodel = gensim.models.Word2Vec.load('D:\FinalProj/word2vec100000.model')
    labels = np.asarray(w2vmodel.wv.index_to_key)
    def presenthigher(num):
        for i in labels:
            sims = w2vmodel.wv.most_similar(str(i), topn=1)
            if sims[0][1]>=num:
                print(f'{i}: {sims}')

    presenthigher(0.5)


    x_vals, y_vals, labels = Word2Vec.reduce_dimensions(w2vmodel)
    def plot_with_matplotlib(x_vals, y_vals, labels):
        plt.figure(figsize=(12, 12))
        plt.scatter(x_vals, y_vals)

        indices = list(range(len(labels)))
        for i in indices:
            plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plot_with_matplotlib(x_vals, y_vals, labels)
    plt.show()
    w2vmodel.save("D:\FinalProj/word2vec100000.model")

    #The tfidf part
    tfidf = TfIdf.TfIdftransform(parsedclean)
    tfidf = tfidf.T
    tfidf.to_csv(path_or_buf='D:\FinalProj/tfidf100000.csv')
    tfidf = pd.read_csv(filepath_or_buffer='D:\FinalProj/tfidf100000.csv')
    print(tfidf.head(10))
    print(tfidf.info())
    #
    X_std = StandardScaler().fit_transform(tfidf)
    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(X_std)
    features = range(pca.n_components_)

    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist

    # run kmeans with many different k
    distortions = []
    K = range(1, 50)
    for k in K:
        k_means = KMeans(n_clusters=k).fit(principalComponents)
        k_means.fit(principalComponents)
        distortions.append(sum(np.min(cdist(principalComponents, k_means.cluster_centers_, 'euclidean'), axis=1)) / tfidf.shape[0])

    X_line = [K[0], K[-1]]
    Y_line = [distortions[0], distortions[-1]]

    # Plot the elbow
    plt.plot(K, distortions, 'b-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The best k for the k-means operation')
    plt.show()
    #
    k = 30
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(principalComponents)
    parsedclean['y'] = y_pred

    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X_std)

    import seaborn as sns
    # sns settings
    sns.set(rc={'figure.figsize': (15, 15)})

    # colors
    palette = sns.color_palette("bright", 1)

    # plot
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], palette=palette)
    plt.title('PCA with no Labels')
    plt.savefig("covid19.png")
    plt.show()

    from matplotlib import pyplot as plt
    import seaborn as sns

    # sns settings
    sns.set(rc={'figure.figsize': (13, 9)})

    # colors
    palette = sns.hls_palette(30, l=.4, s=.9)

    # plot
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y_pred, legend='full', palette=palette)
    plt.title('PCA with Kmeans Labels')
    plt.savefig("improved_cluster_tsne.png")
    plt.show()