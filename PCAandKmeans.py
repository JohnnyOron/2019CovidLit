import pandas as pd
from sklearn.decomposition import PCA

myPCA = PCA(n_components=2)
def usePCA(df):
    ndf = myPCA.fit_transform(df)
    return ndf

