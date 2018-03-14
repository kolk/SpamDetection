from nltk.corpus import stopwords
from Tokenizer import word_tokenizer

def remove_stopwords(words):
    """
    Removes stopwords and returns a list non-stopwords
    :param text:
    :return:
    """
    final_words = []

    # Remove punctuation
    words = [w.lower() for w in words if w.isalpha()]

    # Remove stopwords
    stopwords_english = set(stopwords.words('english'))

    for word in words:
        if word not in stopwords_english:
            final_words.append(word)
    return final_words



if __name__ == "__main__":
    words = word_tokenizer("This is Andrew's text, isn't it?")
    final_words = remove_stopwords(words)
    print(final_words)