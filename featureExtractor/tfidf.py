from sklearn.feature_extraction.text import TfidfVectorizer
from spam_detector import load_data
from preprocessing.Tokenizer import word_tokenizer
from preprocessing.Stopwords import remove_stopwords
from preprocessing.Stemmer import stem
from sklearn.model_selection import train_test_split



class TfIdf(object):
    def __init__(self, data_path):
        self.data = load_data(data_path)
        self.tokenized_docs = []
        self.docs = self.preprocess()
        self.labels = [d[1] for d in self.data]
        tfidf_transformer = TfidfVectorizer()
        self.all_features = tfidf_transformer.fit_transform(self.docs)
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(self.all_features, self.labels, test_size=0.3,
                                                                                    random_state=1234)

    def preprocess(self):
        docs = []
        for d in self.data:
            words = word_tokenizer(d[0])
            words = remove_stopwords(words)
            words = stem(words)
            doc = " ".join(words)
            docs.append(doc)
        return docs


