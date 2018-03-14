from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import string

def word_tokenizer(text):
    """
    Tokenizes a string of text into a list of words
    :param text: string
    :return: list of words
    """
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(text)
    return words


def sent_tokenizer(text):
    return sent_tokenizer(text)

