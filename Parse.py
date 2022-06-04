import re
from langdetect import detect
from tqdm import tqdm


def parsedata(df):
    tempdf = df
    checkLanguage(tempdf)
    a = []
    for i in tqdm(tempdf['body']):
        nsw = stopwordreduce(i)
        a.append(nsw)
    tempdf['processed_body'] = a
    tempdf = tempdf[tempdf['language'] == 'en']
    return tempdf


def checkLanguage(df):
    langs = []
    for i in tqdm(df['body']):
        a = i.split(' ')
        if len(a) >= 50:
            try:
                lang = detect(" ".join(a[:50]))
            except Exception:
                lang = 'null'
        elif len(a) > 0:
            try:
                lang = detect(" ".join(a))
            except Exception:
                lang = 'null'
        langs.append(lang)
    df['language'] = langs

def stopwordreduce(body):
    # the spacy stop-word list:
    # list of stop-words that will be taken out of the body
    # the list can be found on https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/
    stopwords = ['really', 'sometimes', 'go', 'since', 'whither', 'they', 'its', 'them', 'well',
                 'meanwhile', 'seems', 'and', 'latterly', 'regarding', 'somehow', 'sixty', 'whole', 'anyway', 'else',
                 'few', 'beside', 'to', 'namely', 'someone', 'see', 'moreover', 'wherein', 'for', 'former', 'bottom',
                 'it',
                 'next', 'six', 'along', 'once', 'might', 'whenever', 'below', 'another', 'yourself', 'each', 'just',
                 'ourselves',
                 'everyone', 'any', 'across', 'get', 'that', 'eight', 'we', 'which', 'therefore', 'may', 'keep',
                 'among', 'give',
                 'such', 'are', 'indeed', 'everywhere', 'same', 'herself', 'yourselves', 'alone', 'were', 'was', 'take',
                 'seem',
                 'say', 'why', 'show', 'between', 'during', 'elsewhere', 'or', 'though', 'forty', 'made', 'others',
                 'whereafter', 'formerly', 'several', 'via', 'does', 'please', 'three', 'also', 'fifty', 'afterwards',
                 'noone',
                 'no one', 'do', 'perhaps', 'further', 'I', 'beforehand', 'myself', 'empty', 'yet', 'thereby', 'been',
                 'both', 'never',
                 'put', 'without', 'him', 'a', 'nothing', 'thereafter', 'make', 'then', 'whom', 'must', 'sometime',
                 'against', 'through', 'being', 'four', 'back', 'become', 'our', 'himself', 'because', 'anything',
                 'nor',
                 'therein', 'due', 'until', 'own', 'most', 'now', 'while', 'of', 'only', 'am', 'itself', 'too',
                 'nobody', 'if',
                 'one', 'whereas', 'twelve', 'together', 'can', 'who', 'even', 'be', 'she', 'besides', 'herein', 'off',
                 'last', 'no', 'whereupon', 'the', 'thru', 'out', 'hereupon', 'by', 'us', 'already', 'became', 'here',
                 'hers',
                 'onto', 'beyond', 'down', 'enough', 'did', 'some', 'over', 'serious', 'quite', 'move', 'around',
                 'nowhere',
                 'amongst', 'but', 'so', 'wherever', 'twenty', 'often', 'part', 'again', 'where', 're', 'within', 'at',
                 'yours',
                 'front', 'unless', 'could', 'anyone', 'third', 'whatever', 'doing', 'nevertheless', 'before', 'rather',
                 'fifteen', 'her', 'me', 'thereupon', 'mostly', 'throughout', 'hence', 'mine', 'ten', 'hundred', 'nine',
                 'call',
                 'when', 'about', 'will', 'whereby', 'this', 'upon', 'you', 'should', 'always', 'themselves', 'not',
                 'has',
                 'behind', 'on', 'anywhere', 'side', 'their', 'hereby', 'latter', 'after', 'none', 'these', 'name',
                 'every',
                 'although', 'however', 'he', 'becoming', 'how', 'whose', 'still', 'hereafter', 'whether', 'towards',
                 'more', 'everything', 'whoever', 'seemed', 'cannot', 'up', 'otherwise', 'in', 'would', 'under', 'done',
                 'thence', 'whence', 'seeming', 'either', 'other', 'with', 'into', 'amount', 'five', 'much', 'except',
                 'his', 'thus',
                 'what', 'almost', 'becomes', 'least', 'ever', 'above', 'is', 'first', 'there', 'somewhere', 'top',
                 'than', 'have',
                 'toward', 'per', 'all', 'ours', 'full', 'anyhow', 'as', 'many', 'various', 'your', 'had', 'eleven',
                 'from', 'something',
                 'less', 'those', 'an', 'two', 'my', 'very', 'neither']
    # list of strings that will be taken out of words
    parts_stopwords = ["'d", "'m", "n't", "'ve", "'re", "'s", "'ll", ",", "(", ")", "[", "]", "-", "/", ":", "%", '•',
                       ';', '®', '>', '∝', '<']
    # my custom word list:
    custom = ['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'author', 'figure',
              'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
              'al.', 'Elsevier', 'PMC', 'CZI', 'ago', 'de', 'com']
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    parts = parts_stopwords + nums
    stopwords = stopwords + custom
    no_stop_words = body.lower()

    # Reduction of parts of other words
    for i in parts:
        no_stop_words = no_stop_words.replace(str(i), '')

    # Removing periods from the middle of words
    nswlist = no_stop_words.split()
    for i in range(0, len(nswlist)):
        if '.' in nswlist[i]:
            if (nswlist[i].count('.') > 1) and (nswlist[i][-1] == '.'):
                nswlist[i].replace('.', '', (nswlist[i].count('.') - 1))
            elif (nswlist[i].count('.') == 1) and (nswlist[i][-1] == '.'):
                continue
            else:
                nswlist[i].replace('.', '')

    # Adding a period to the last word of the list if needed
    if nswlist[-1][-1] != '.':
        nswlist[-1] = nswlist[-1] + '.'
    no_stop_words = ' '.join(nswlist)

    # Full words reduction:
    for i in stopwords:
        no_stop_words = re.sub(r'\s*\b' + i + r'\b\s*', ' ', no_stop_words)
    return no_stop_words
