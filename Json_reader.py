import json


class Read:
    def __init__(self, path, df):
        file = open(path)
        data = json.load(file)
        self.id = data['paper_id']
        self.abstract = []
        self.body = []
        self.doi = []
        self.url = []
        self.author = []
        self.title = []
        abst = []
        bod = []
        for i in data['abstract']:
            abst.append(i['text'])
        for i in data['body_text']:
            bod.append(i['text'])
        self.abstract = ' '.join(abst)
        self.body = ' '.join(bod)
        self.doi = df['doi'].loc[df['sha'] == self.id]
        self.url = df['url'].loc[df['sha'] == self.id]
        self.author = df['authors'].loc[df['sha'] == self.id]
        self.title = df['title'].loc[df['sha'] == self.id]

    def __repr__(self):
        return f'id-{self.id}: doi-{self.doi.to_string()} url-{self.url.to_string()} author-{self.author.to_string()} title-{self.title.to_string()} abstract-{self.abstract[:100]}... body-{self.body[:100]}'
