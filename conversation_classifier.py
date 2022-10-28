import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import json


def create_dataset(data):
    X = []
    y = []
    for datapoint in data:
        X.append([datapoint["depth"], datapoint["size"], datapoint["width"], datapoint["structural_virality"],
                  datapoint["density"], datapoint["diameter"], datapoint["reply_count"], datapoint["unique_users"]])
        y.append(datapoint["author_id"])
    X = np.array(X)
    y = np.array(y)
    return X, y


def stratified_kFold(X, y, clf):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    lst_accu_stratified = []

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf.fit(X_train_fold, y_train_fold)
        lst_accu_stratified.append(clf.score(X_test_fold, y_test_fold))

    print('List of possible accuracy:', lst_accu_stratified)
    print('\nMaximum Accuracy That can be obtained from this model is:',
          max(lst_accu_stratified) * 100, '%')
    print('\nMinimum Accuracy:',
          min(lst_accu_stratified) * 100, '%')
    print('\nOverall Accuracy:',
          np.mean(lst_accu_stratified) * 100, '%')
    print('\nStandard Deviation is:', np.std(lst_accu_stratified))


if __name__ == "__main__":
    data = json.load(open("conversation_metrics.json"))
    X, y = create_dataset(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    #model = svm.SVC(C=1)
    model = LogisticRegression(random_state=0)
    scaler = preprocessing.StandardScaler()
    clf = make_pipeline(scaler, model)
    stratified_kFold(X_train, y_train, clf)
    print(f"\nThe share in Democrats is {sum((y == 'Democrat'))/len(y)}")






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




