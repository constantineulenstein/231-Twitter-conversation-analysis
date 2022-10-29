import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
import json
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


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


def stratified_kFold(X, y, clf, verbose):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    lst_accu_stratified = []

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf.fit(X_train_fold, y_train_fold)
        lst_accu_stratified.append(clf.score(X_test_fold, y_test_fold))

    if verbose:
        print('List of possible accuracy:', lst_accu_stratified)
        print('\nMaximum Accuracy That can be obtained from this model is:',
              max(lst_accu_stratified) * 100, '%')
        print('\nMinimum Accuracy:',
              min(lst_accu_stratified) * 100, '%')
        print('\nOverall Accuracy:',
              np.mean(lst_accu_stratified) * 100, '%')
        print('\nStandard Deviation is:', np.std(lst_accu_stratified))

    return np.mean(lst_accu_stratified)


def generate_ccdf_plots(dems, reps, feature):
    def _ccdf(a):
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, 1 - cusum / cusum[-1]

    x_dems, y_dems = _ccdf(dems)
    # x_dems = np.insert(x_dems, 0, 0.) #Add so plot always starts at 0
    # y_dems = np.insert(y_dems, 0, 1.)

    x_reps, y_reps = _ccdf(reps)
    # x_reps = np.insert(x_reps, 0, 0.)
    # y_reps = np.insert(y_reps, 0, 1.)

    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x_dems, y_dems, drawstyle='steps-post', label="Dems")
    plt.plot(x_reps, y_reps, drawstyle='steps-post', label="Reps")
    plt.legend()
    plt.xlabel(f"{feature}")
    plt.ylabel("CCDF (%)")
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in current_values])
    # plt.tight_layout()
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
    plot_distributions = False
    cutoff_size = 5
    data = json.load(open("conversation_metrics_v2.json"))
    feature_name_dict = {
        name: idx for idx, name in enumerate(list(data[0].keys())[2:])
    }
    X, y = create_dataset(data, cutoff_size)
    print(f"The dataset contains {len(y[y == 'Democrat'])} conversations of Democrats and {len(y[y == 'Republican'])} "
          f"conversations of Republicans.")

    if plot_distributions:
        for feature in feature_name_dict.keys():
            print(feature)
            plot_distribution(X, y, feature_name_dict, feature)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
    ]

    model_accs = {}
    for i, model in enumerate(classifiers):
        scaler = preprocessing.StandardScaler()
        clf = make_pipeline(scaler, model)
        acc = stratified_kFold(X_train, y_train, clf, verbose=False)
        model_accs[names[i]] = acc

    print(f"\nThe share in Democrats is {sum((y == 'Democrat')) / len(y)}")
    print(f"\nModel accuracies are: \n", model_accs)

    best_model = max(model_accs, key=model_accs.get)
    print(f"Best model is {best_model} with following metrics: \n")

    scaler = preprocessing.StandardScaler()
    clf = make_pipeline(scaler, classifiers[names.index(best_model)])
    acc = stratified_kFold(X_train, y_train, clf, verbose=True)
    clf.fit(X_train, y_train)
    print(f"\n The accuracy on final test set is {clf.score(X_test, y_test) * 100} %")

    r = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        print(f"{[feature for feature in feature_name_dict if feature_name_dict[feature] == i][0]}: "
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
