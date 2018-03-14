from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV

class SVC_(object):
    def __init__(self, num_trails=10):
        self.svm = SVC("rbf")
        self.num_trails = num_trails
        self.parameter_grid = {
            "C": [1, 10, 100],
            "gamma": [0.001, 0.01, 0.1]
        }

    def fit(self, train_features, train_labels):
        print('Starting cross-validation')
        # Loop for each trial
        for i in range(self.num_trails):

            # cross-validation split
            parameter_opt_cv = KFold(n_splits=4, shuffle=False)

            # Grid search for parameter tuning
            clf = GridSearchCV(estimator=self.svm, param_grid=self.parameter_grid, cv=parameter_opt_cv)
            clf.fit(train_features, train_labels)
            print('trial {0} grid search score {1}'.format(i, clf.best_score_))

        self.clf = clf

    def predict(self, test_features, test_labels=[]):
        y_pred = self.clf.predict(test_features)
        return y_pred




"""
if __name__ == "__main__":
    DATA_PATH = "/home/abzooba/Downloads"
    from featureExtractor.tfidf import TfIdf
    from sklearn.metrics import accuracy_score

    tfidf = TfIdf(DATA_PATH)
    from models.svm import SVC_
    svm = SVC_(2)
    svm.fit(tfidf.features_train, tfidf.labels_train)
    pred_y = svm.predict(tfidf.features_test, tfidf.labels_test)
    acc = accuracy_score(tfidf.labels_test, pred_y)
    print(acc)
"""