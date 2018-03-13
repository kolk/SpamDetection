import os
import random
from defaults import *
from preprocessing.Tokenizer import word_tokenizer

def read_file(file_path):
    with open(file_path, "r") as f:
        lines = f.read()
    return lines

def load_data(input_path):
    spam_path =  "enron1/spam"
    ham_path = "enron1/ham"

    # list the files
    spam_files = os.listdir(os.path.join(input_path, spam_path))
    spam_paths = [os.path.join(input_path, spam_path, f) for f in spam_files]

    ham_files = os.listdir(os.path.join(input_path, ham_path))
    ham_paths = [os.path.join(input_path, ham_path, f) for f in ham_files]

    # read the files from each category
    spam_list = []
    for file_path in spam_paths:
        spam_list.append(read_file(file_path))

    ham_list = []
    for file_path in ham_paths:
        ham_list.append(read_file(file_path))

    spam_list = [(txt, "spam") for txt in spam_list]
    ham_list = [(txt, "ham") for txt in ham_list]
    all_emails = spam_list + ham_list
    random.shuffle(all_emails)
    return all_emails



if __name__ == "__main__":
    data = load_data("/home/abzooba/Downloads")

    """
    if CLASSIFIER["bi-lstm"]:
        pass
    else:
        if FEATURE_EXTRACTOR["tf-idf"]:
            pass
    """
    print(data[0])
    #word_tokenizer()