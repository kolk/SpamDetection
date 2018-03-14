from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV

from featureExtractor.tfidf import TfIdf

class NaiveBayes(object):
    def __init__(self, num_trials=10):
        self.mnb = MultinomialNB()
        self.num_trials = num_trials
        self.parameter_grid = {
            "alpha": [0.01, 0.2, 1.0, 2.0]
        }

    def fit(self, train_features, train_labels):
        print('Starting cross-validation')
        # Loop for each trial
        for i in range(self.num_trials):
            # cross-validation split
            parameter_opt_cv = KFold(n_splits=4, shuffle=True)

            # Grid search for parameter tuning
            clf = GridSearchCV(estimator=self.mnb, param_grid=self.parameter_grid, cv=parameter_opt_cv)
            clf.fit(train_features, train_labels)
            print('trial {0} grid search score {1}'.format(i, clf.best_score_))

        self.clf = clf

    def predict(self, features_test):
        return self.clf.predict(features_test)


"""
if __name__ == "__main__":
    DATA_PATH = "/home/abzooba/Downloads"
    nb = NaiveBayes()
    tfidf = TfIdf(DATA_PATH)
    nb = NaiveBayes()
    nb.fit(tfidf.features_train, tfidf.labels_train)
    pred_y = nb.predict(tfidf.features_test)
    acc = accuracy_score(tfidf.labels_test, pred_y)
    print(acc)
"""