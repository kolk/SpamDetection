from nltk.stem import PorterStemmer

def stem(words):
    porter = PorterStemmer()
    stemmed_words = [porter.stem(w) for w in words]
    return stemmed_words