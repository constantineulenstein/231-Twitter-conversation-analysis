import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import json
import matplotlib.pyplot as plt
import scipy


def create_dataset(data, cutoff_size):
    X = []
    y = []
    feature_names = [feature for feature in data[0].keys()][2:]
    for datapoint in data:
        if datapoint["size"] >= cutoff_size:
            X.append([datapoint[feature] for feature in feature_names])
            y.append(datapoint["author_id"])
        # X.append([datapoint["depth"], datapoint["size"], datapoint["width"], datapoint["structural_virality"],
        #          datapoint["density"], datapoint["diameter"], datapoint["reply_count"], datapoint["unique_users"]])
        # y.append(datapoint["author_id"])
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


def generate_ccdf_plots(dems, reps, feature):
    def _ccdf(a):
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, 1 - cusum / cusum[-1]

    x_dems, y_dems = _ccdf(dems)
    #x_dems = np.insert(x_dems, 0, 0.) #Add so plot always starts at 0
    #y_dems = np.insert(y_dems, 0, 1.)

    x_reps, y_reps = _ccdf(reps)
    #x_reps = np.insert(x_reps, 0, 0.)
    #y_reps = np.insert(y_reps, 0, 1.)

    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x_dems, y_dems, drawstyle='steps-post', label="Dems")
    plt.plot(x_reps, y_reps, drawstyle='steps-post', label="Reps")
    plt.legend()
    plt.xlabel(f"{feature}")
    plt.ylabel("CCDF (%)")
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in current_values])
    #plt.tight_layout()
    plt.show()


def plot_distribution(X, y, feature_name_dict, feature):
    values = X[:, feature_name_dict[feature]]
    dem_idx = np.argwhere(y == "Democrat")
    rep_idx = np.argwhere(y == "Republican")
    dem_values = values[dem_idx]
    rep_values = values[rep_idx]

    dem_values = np.sort(dem_values.flatten())
    rep_values = np.sort(rep_values.flatten())

    generate_ccdf_plots(dem_values, rep_values, feature)


if __name__ == "__main__":
    plot_distributions = True
    cutoff_size = 5
    data = json.load(open("conversation_metrics_temp.json"))
    feature_name_dict = {
        name: idx for idx, name in enumerate(list(data[0].keys())[2:])
    }
    X, y = create_dataset(data, cutoff_size)
    print(f"The dataset contains {len(y[y=='Democrat'])} conversations of Democrats and {len(y[y=='Republican'])} "
          f"conversations of Republicans.")


    if plot_distributions:
        for feature in feature_name_dict.keys():
            print(feature)
            plot_distribution(X, y, feature_name_dict, feature)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = svm.SVC(C=2)
    # model = LogisticRegression(random_state=0)
    scaler = preprocessing.StandardScaler()
    clf = make_pipeline(scaler, model)
    stratified_kFold(X_train, y_train, clf)
    print(f"\nThe share in Democrats is {sum((y == 'Democrat')) / len(y)}")

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
