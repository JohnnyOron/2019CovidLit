import pandas as pd
from tqdm import tqdm
import math


# creating the Tf-Idf table using only a dataframe
def TfIdftransform(df):
    unique = uniquewords(df)
    tfidf = pd.DataFrame(index=unique)
    tfdf = tf(df, unique)
    idfdf = idfl(df, unique)
    print('tfidf:')
    for i in tqdm(range(0, len(df))):
        ii = str(i)
        col = []
        for word in range(0, len(unique)):
            tfi = tfdf[ii].iloc[word]
            idfi = idfdf['idf'].iloc[word]
            tfidfi = tfi * idfi
            col.append(tfidfi)
        tfidf[ii] = col
    return tfidf


# defining a function to get the list of all the unique words in the corpus
def uniquewords(df):
    print('unique:')
    for i in tqdm(range(0, len(df))):
        body = df['processed_body'].iloc[i]
        bodyl = body.split()
        unique = []
        for word in bodyl:
            if not (word in unique):
                unique.append(word)
    print(len(unique))
    return unique


# defining the tf(t,d) function
def tf(df, uni):
    tfdf = pd.DataFrame(index=uni)
    print('tf:')
    for i in tqdm(range(0, len(df))):
        body = df['processed_body'].iloc[i]
        bodyl = body.split()
        column = []
        for word in uni:
            count = bodyl.count(word)
            column.append(count / len(bodyl))
        index = str(i)
        tfdf[index] = column
    return tfdf


# defining the idf(t)
def idfl(df, uni):
    idfdf = pd.DataFrame(index=uni)
    col = []
    print('idf:')
    for word in tqdm(uni):
        num = 0
        for i in range(0, len(df)):
            body = df['processed_body'].iloc[i]
            if (word in body):
                num = num + 1
        col.append(num)
    idfdf['docf'] = col
    idf = []
    for i in idfdf['docf']:
        n = len(df)
        iidf = math.log(n / (i + 1))
        iidf = abs(iidf)
        idf.append(iidf)
    idfdf['idf'] = idf
    return idfdf
