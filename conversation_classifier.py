import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.inspection import permutation_importance
import json


def create_dataset(data):
    X = []
    y = []
    for datapoint in data:
        X.append([datapoint["depth"], datapoint["size"], datapoint["width"], datapoint["structural_virality"],
                  datapoint["density"], datapoint["diameter"], datapoint["reply_count"], datapoint["unique_users"]])
        y.append(datapoint["party"])
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    data = json.load(open("conversation_metrics.json"))
    X, y = create_dataset(data)
    # First test that we have roughly equal size of republicans and democrats to ensure that we dont introduce bias
    print(sum((y == "Democrat")))
    # if very skewed, need to balance dataset


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = svm.SVC(C=1)
    scaler = preprocessing.StandardScaler()
    clf = make_pipeline(scaler, model)
    print(cross_val_score(clf, X_train, y_train, cv=10))

    # DONT FORGET TO SCALE TEST SET PRIOR TO EVALUATING

"""
r = permutation_importance(model, X_val, y_val,
                           n_repeats=30,
                           random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{diabetes.feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
"""
