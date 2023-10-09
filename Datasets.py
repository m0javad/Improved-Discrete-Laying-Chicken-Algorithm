#Asosoft dataset
class Aso():

    def __init__(self, path, n_top=2000):
        self.path = path
        self.n_top = n_top

    def read_corpus(self):
        corpus = []
        with open(self.path ,encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            corpus.append(line)
        return corpus

    def get_labels(self):
        corpus = self.read_corpus() 
        y = []
        sentences = []
        for line in corpus:
            temp = line.split(" ")
            y.append(temp.pop()[:-1])
            sentences.append(temp)
        sentences = [' '.join(i) for i in sentences]
        label = []
        for i in range (0,len(y)):
            if y[i] == 'Literature':
                a = 0
            elif y[i] == 'Social':
                a = 4
            elif y[i] == 'Sports':
                a = 5
            elif y[i] == 'Religious':
                a = 2
            elif y[i] == 'Political':
                a = 1
            elif y[i] == 'Scientific':
                a = 3
            label.append(a)
        return sentences,label

    def TF (self):
        sentences, label = self.get_labels()
        vectorizer = CountVectorizer(max_features = self.n_top)
        X = vectorizer.fit_transform(sentences)
        return X.toarray()

    def TF_IDF(self):
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(self.TF())
        return tfidf.toarray()
    pass
# 20newsgroup dataset
class newsgroups():
    
    def __init__(self, sentences, n_top=2000):
        self.n_top = n_top
        self.sentences = sentences
        
    def TF (self):
        sentences = self.sentences
        vectorizer = CountVectorizer(max_features = self.n_top)
        X = vectorizer.fit_transform(self.sentences)
        return X.toarray()

    def TF_IDF(self):
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(self.TF())
        return tfidf.toarray()
    pass

#AGNEWS
class AGNEWS():
    import requests
    r = requests.get('https://storage.googleapis.com/kaggle-data-sets/612351/1095715/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230506%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230506T114456Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=342dd58e34061ad5d565ee453b955b132ccc0933ddf40718ba9fc1ba13cca0aae688d4a547d6fed6463eca40177569affe059b2efa33d69ff7305f861fab5f95259b6f6b9609d811cdb24049277d6515b9e18ac0e9cc21da430f65fd20e6a9251bb39d39810ed5f8d25ab91d6c61647c05282465e2c051edc6929587452b1024882dc51f3bfc30e9933cd1421c4990ceccb244a3249088d07e78cf19c9a1c58296ddf96289b844788149a98c53833be7e434074a8ac2bd8e2f542372ee036fadedf0d36eb5e72537d6fd283f19daf2bdc4859747876adea8550284fa334da6a05078a2f809fecf4871dae2c1f703009d5b52f73cba63bf96c5c0e7c4b1fc25c5', allow_redirects=True)
    with open('AGNEWS.zip', 'wb') as f:
        f.write(r.content)
    # !unzip "/content/AGNEWS.zip" -d "/content/AGNEWS"
    raw_news = pd.read_csv('train.csv')
    y= []
    sentences = []
    for label in raw_news['Class Index']:
        y.append(label)
    label = np.array([int(y[x]) for x in range(0,len(y))])
    news = raw_news['Description'] # + raw_news['Title'] 
    sentences = []
    for sentence in news:
        sentences.append(sentence)
    sentences = sentences[-10000:]
    label = label[-10000:]
    sentences = preprocesses(stop_words,sentences)
    sentences = np.array([str(sentences[x]) for x in range(0,len(sentences))])
    news_dataset = newsgroups(sentences=sentences, n_top=2000)
    TF_IDF_vector = news_dataset.TF_IDF()

#BBC NEWS
class BBCNEWS():
    import requests
    r = requests.get('http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip', allow_redirects=True)
    with open('BBCNEWS.zip', 'wb') as f:
        f.write(r.content)
    # !unzip "/content/BBCNEWS.zip" -d "/content/BBCNEWS"
    folder_path = 'bbc'
    sentences = []
    label = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(root,filename)
                    with open (file_path , 'r') as f :
                        file_contents = f.read()
                        labels = root.split('/')[-1]
                        if labels == 'bbc\\entertainment':
                            labels = 0
                        if labels == 'bbc\\business':
                            labels = 1
                        if labels == 'bbc\\politics':
                            labels = 2
                        if labels == 'bbc\\sport':
                            labels = 3
                        if labels == 'bbc\\tech':
                            labels = 4
                        sentences.append(file_contents)
                        label.append(labels)
    label = np.array([int(label[x]) for x in range(0,len(label))])
    sentences = preprocesses(stop_words,sentences)
    sentences = np.array([str(sentences[x]) for x in range(0,len(sentences))])
    news_dataset = newsgroups(sentences=sentences, n_top=2000)
    TF_IDF_vector = news_dataset.TF_IDF()

#Reuters
class Reuters():
    def __init__(self):
        pass
    def discrete():
        from keras import utils
        from keras.datasets import reuters
        from keras.preprocessing.text import Tokenizer
        (x_reuters_dataset, y_reuters_dataset),(X_test, y_test) = reuters.load_data(test_split = 0, oov_char=0, skip_top=100, start_char=None)
        t = Tokenizer(num_words=2100)
        t.fit_on_sequences(x_reuters_dataset)
        TF_IDF_vector = t.sequences_to_matrix(x_reuters_dataset, mode='tfidf')
        label = y_reuters_dataset
        # label = utils.to_categorical(y_reuters_dataset, max(y_reuters_dataset) + 1)
    def text():
        import numpy as np
        import pandas as pd
        def make_reuters_data(data_dir):

            # download reuters data
            data_path = data_dir
            import os
            print('Downloading data...')
            os.system('wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz -P %s' % data_path)
            os.system('wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz -P %s' % data_path)
            os.system('wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz -P %s' % data_path)
            os.system('wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz -P %s' % data_path)
            os.system('wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz -P %s' % data_path)

            os.system('wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz -P %s' % data_path)

            print('Unzipping data...')
            os.system('gunzip %s/lyrl2004_tokens_test_pt0.dat.gz' % data_path)
            os.system('gunzip %s/lyrl2004_tokens_test_pt1.dat.gz' % data_path)
            os.system('gunzip %s/lyrl2004_tokens_test_pt2.dat.gz' % data_path)
            os.system('gunzip %s/lyrl2004_tokens_test_pt3.dat.gz' % data_path)
            os.system('gunzip %s/lyrl2004_tokens_train.dat.gz' % data_path)
            os.system('gunzip %s/rcv1-v2.topics.qrels.gz' % data_path)

            np.random.seed(1234)
            from sklearn.feature_extraction.text import CountVectorizer
            from os.path import join
            did_to_cat = {}
            cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
            with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
                for line in fin.readlines():
                    line = line.strip().split(' ')
                    cat = line[0]
                    did = int(line[1])
                    if cat in cat_list:
                        did_to_cat[did] = did_to_cat.get(did, []) + [cat]
                # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
                for did in list(did_to_cat.keys()):
                    if len(did_to_cat[did]) > 1:
                        del did_to_cat[did]

            dat_list = ['lyrl2004_tokens_test_pt0.dat',
                        'lyrl2004_tokens_test_pt1.dat',
                        'lyrl2004_tokens_test_pt2.dat',
                        'lyrl2004_tokens_test_pt3.dat',
                        'lyrl2004_tokens_train.dat']
            data = []
            target = []
            cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
            del did
            for dat in dat_list:
                with open(join(data_dir, dat)) as fin:
                    for line in fin.readlines():
                        if line.startswith('.I'):
                            if 'did' in locals():
                                assert doc != ''
                                if did in did_to_cat:
                                    data.append(doc)
                                    target.append(cat_to_cid[did_to_cat[did][0]])
                            did = int(line.strip().split(' ')[1])
                            doc = ''
                        elif line.startswith('.W'):
                            assert doc == ''
                        else:
                            doc += line
                print(len(data),len(target))

            print((len(data), 'and', len(did_to_cat)))
        #     assert len(data) == len(did_to_cat)
            x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
            y = np.asarray(target)

            from sklearn.feature_extraction.text import TfidfTransformer
            x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
            x = x[:10000].astype(np.float32)
            print(x.dtype, x.size)
            y = y[:10000]
            x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
            print('todense succeed')

            assert x.shape[0] == y.shape[0]
            x = x.reshape((x.shape[0], -1))
            np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})
            return data, target

        def load_reuters(data_path='/content/reuters/reuters'):
            import os

            if not os.path.exists(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))

            if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
                print('making reuters idf features')
                make_reuters_data(data_path)
                print(('reutersidf saved to ' + data_path))
            data = np.load(os.path.join(data_path, 'reutersidf10k.npy'),allow_pickle=True).item()
            # has been shuffled
            x = data['data']
            y = data['label']
            x = x.reshape((x.shape[0], -1)).astype('float64')
            # scale to [0,1]
            from sklearn.preprocessing import MinMaxScaler
            x = MinMaxScaler().fit_transform(x)
            y = y.reshape((y.size,))
            print(('REUTERSIDF10K samples', x.shape))
            return x, y

TF_IDF_vector, label = load_reuters()

#SPAM 
class spam():
    import string
    # !git clone 'https://github.com/milindsoorya/Spam-Classifier-in-python'
    data = pd.read_csv('/content/Spam-Classifier-in-python/dataset/spam.csv', encoding='latin-1')
    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    data = data.rename(columns={"v2" : "text", "v1":"label"})
    data = data.replace(['ham','spam'],[0, 1])
    def text_process(text):

        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

        return " ".join(text)
    data['text'] = data['text'].apply(text_process)
    text = pd.DataFrame(data['text'])
    sentences = text.to_numpy()
    sentences = np.array([str(sentences[x]) for x in range(0,len(sentences))])
    sentences = preprocesses(stop_words,sentences)
    y = pd.DataFrame(data['label'])
    label = y.to_numpy()
    label = np.array([int(label[x]) for x in range(0,len(label))])
    news_dataset = newsgroups(sentences=sentences, n_top=1000)
    TF_IDF_vector = news_dataset.TF_IDF()
