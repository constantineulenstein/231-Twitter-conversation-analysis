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
from sklearn.linear_model import LogisticRegression


def create_dataset(data, cutoff_size, cutoff_max, feature_names):
    X = []
    y = []
    for datapoint in data:
        if datapoint["size"] >= cutoff_size and datapoint["size"] <= cutoff_max:
            X.append([datapoint[feature] for feature in feature_names])
            y.append(datapoint["party"])

    X = np.array(X)
    y = np.array(y)
    return X, y


def stratified_kFold(X, y, clf, verbose):
    skf = StratifiedKFold(n_splits=10, shuffle=True)#, random_state=42)
    lst_accu_stratified = []
    lst_accu_dems_stratified = []
    lst_accu_reps_stratified = []

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf.fit(X_train_fold, y_train_fold)
        lst_accu_stratified.append(clf.score(X_test_fold, y_test_fold))
        lst_accu_dems_stratified.append(clf.score(X_test_fold[np.argwhere(y_test_fold=="Democrat").flatten()], y_test_fold[y_test_fold=="Democrat"]))
        lst_accu_reps_stratified.append(clf.score(X_test_fold[np.argwhere(y_test_fold=="Republican").flatten()], y_test_fold[y_test_fold=="Republican"]))

    if verbose:
        #print('List of possible accuracy:', lst_accu_stratified)
        print('\nMaximum Accuracy That can be obtained from this model is:',
              max(lst_accu_stratified) * 100, '%')
        print('\nMinimum Accuracy:',
              min(lst_accu_stratified) * 100, '%')
        print('\nOverall Accuracy:',
              np.mean(lst_accu_stratified) * 100, '%')
        print('\nOverall Accuracy on Democrats:',
              np.mean(lst_accu_dems_stratified) * 100, '%')
        print('\nOverall Accuracy on Republicans:',
              np.mean(lst_accu_reps_stratified) * 100, '%')
        print('\nStandard Deviation is:', np.std(lst_accu_stratified))

    return np.mean(lst_accu_stratified)


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
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(x_dems, y_dems, drawstyle='steps-post', label="Dems")
    plt.plot(x_reps, y_reps, drawstyle='steps-post', label="Reps")
    plt.legend()
    plt.xlabel(f"{feature}")
    plt.ylabel("CCDF (%)")
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in current_values])
    plt.tight_layout()
    plt.savefig(f"plots/ccdf_graph_{feature}.png")


def plot_distribution(X, y, feature_name_dict, feature):
    values = X[:, feature_name_dict[feature]]
    dem_idx = np.argwhere(y == "Democrat")
    rep_idx = np.argwhere(y == "Republican")
    dem_values = values[dem_idx]
    rep_values = values[rep_idx]

    dem_values = np.sort(dem_values.flatten())
    rep_values = np.sort(rep_values.flatten())

    generate_ccdf_plots(dem_values, rep_values, feature)


def run_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    scaler = preprocessing.StandardScaler()
    model = LogisticRegression()
    clf = make_pipeline(scaler, model)
    clf.fit(X_train, y_train)
    #acc = stratified_kFold(X_train, y_train, clf, verbose=False)
    print(f"\nThe accuracy on final test set is {clf.score(X_test, y_test) * 100} %")
    importance = model.coef_[0]
    for i, v in enumerate(importance):
        print(f'Feature: {feature_names[i]}, Score: {v}')

    plt.figure()
    feature_names = [feature.replace('_to_', '-to-') for feature in feature_names]
    feature_names = [feature.replace('_with_', '-with-') for feature in feature_names]
    feature_names = [feature.replace('_', '\n') for feature in feature_names]
    plt.bar(feature_names, importance)
    plt.xticks(fontsize=9, rotation=90)
    plt.subplots_adjust(bottom=0.25)
    plt.title("Coefficients of Logistic Regression Model")
    plt.savefig(f"plots/logistic_regression_feature_importance.png")

def run_model_evaluation(X_train, X_test, y_train, y_test, feature_name_dict):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "Logistic Regression"
    ]
    classifiers = [
        KNeighborsClassifier(5),
        SVC(kernel="linear"),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        LogisticRegression()
    ]

    classifiers = [
        KNeighborsClassifier(5),
        SVC(kernel="linear"),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(hidden_layer_sizes=(10,10,), max_iter=100000),
        AdaBoostClassifier(),
        GaussianNB(),
        LogisticRegression()
    ]

    model_accs = {}
    for i, model in enumerate(classifiers):
        scaler = preprocessing.StandardScaler()
        clf = make_pipeline(scaler, model)
        acc = stratified_kFold(X_train, y_train, clf, verbose=False)
        model_accs[names[i]] = acc

    print(f"Model accuracies are: \n", model_accs)

    best_model = max(model_accs, key=model_accs.get)
    print(f"\nBest model is {best_model} with following metrics: \n")

    scaler = preprocessing.StandardScaler()
    model = classifiers[names.index(best_model)]
    clf = make_pipeline(scaler, model)
    _ = stratified_kFold(X_train, y_train, clf, verbose=True)
    clf.fit(X_train, y_train)
    print(f"\nThe share in Democrats is {sum((y == 'Democrat')) / len(y)}")
    print(
        f"\n The accuracy on final test set is {clf.score(X_test, y_test) * 100} %, the accuracy on final test set for "
        f"Democrats is {clf.score(X_test[np.argwhere(y_test == 'Democrat').flatten()], y_test[y_test == 'Democrat']) * 100} "
        f"%, the accuracy on final test set "
        f"for Republicans is {clf.score(X_test[np.argwhere(y_test == 'Republican').flatten()], y_test[y_test == 'Republican']) * 100} %")

    r = permutation_importance(clf, X_test, y_test, n_repeats=30)
    means = []
    stds = []
    for i in r.importances_mean.argsort()[::-1]:
        means.append(r.importances_mean[i])
        stds.append(r.importances_std[i])
        print(f"{[feature for feature in feature_name_dict if feature_name_dict[feature] == i][0]}: "
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

    plt.figure()
    feature_names = list(feature_name_dict.keys())
    feature_names = [feature.replace('_to_', '-to-') for feature in feature_names]
    feature_names = [feature.replace('_with_', '-with-') for feature in feature_names]
    feature_names = [feature.replace('_', '\n') for feature in feature_names]
    plt.bar(feature_names, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.title('Permutation Importance of Features')
    plt.xticks(fontsize=9, rotation=90)
    plt.subplots_adjust(bottom=0.25)
    plt.grid()
    plt.savefig(f"plots/permutation_importance.png")


def calculate_statistics(X, y, feature_name_dict):
    dem_idx = np.argwhere(y == "Democrat")
    rep_idx = np.argwhere(y == "Republican")
    dem_values = X[dem_idx]
    rep_values = X[rep_idx]

    dem_means = np.mean(dem_values, axis=0)[0]
    rep_means = np.mean(rep_values, axis=0)[0]

    dem_std = np.std(dem_values, axis=0)[0]
    rep_std = np.std(rep_values, axis=0)[0]

    return {feature: (dem_means[i], dem_std[i]) for i, feature in enumerate(feature_name_dict)}, \
           {feature: (rep_means[i], rep_std[i]) for i, feature in enumerate(feature_name_dict)},


if __name__ == "__main__":
    plot_distributions = False
    check_logistic_regression = False
    evaluate_models = True
    calculate_stats = False
    cutoff_size = 20
    cutoff_max = 500000
    data = json.load(open("conversation_metrics_v6.json"))
    feature_names = list(data[0].keys())[2:-1]
    #feature_names = ['size', 'width', 'depth', 'density', 'reply_to_reply_proportion', 'echo_chamber_proportion', "assortativity"]
    feature_names = ['width', 'reply_with_reply_proportion', 'reciprocity', 'density', 'unique_users']

    feature_name_dict = {
        name: idx for idx, name in enumerate(feature_names)
    }

    # Uncomment for getting the graphs with max feature values
    max_conv_dict = {}
    min_conv_dict = {}
    for feature in feature_name_dict:
       data_filtered = filter(lambda row: row["size"] > 100 and row["size"] < 1000, data)
       max_conv = max(data_filtered, key=lambda x: x[feature])
       data_filtered = filter(lambda row: row["size"] > 100 and row["size"] < 1000, data)
       min_conv = min(data_filtered, key=lambda x: x[feature])
       max_conv_dict[feature] = max_conv
       min_conv_dict[feature] = min_conv
    json.dump(max_conv_dict, open("maxConvs.json", "w"))
    json.dump(min_conv_dict, open("minConvs.json", "w"))

    X, y = create_dataset(data, cutoff_size, cutoff_max, feature_names)
    print(f"The dataset contains {len(y[y == 'Democrat'])} conversations of Democrats and {len(y[y == 'Republican'])} "
          f"conversations of Republicans.")

    if calculate_stats:
        dem_stats, rep_stats = calculate_statistics(X, y, feature_name_dict)
        print("Dem Stats: ", dem_stats)
        print("Rep Stats: ", rep_stats)

    if plot_distributions:
        for feature in feature_name_dict.keys():
            print(feature)
            plot_distribution(X, y, feature_name_dict, feature)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)#, random_state=42)

    if check_logistic_regression:
        run_logistic_regression(X_train, X_test, y_train, y_test, feature_names)

    if evaluate_models:
        run_model_evaluation(X_train, X_test, y_train, y_test, feature_name_dict)




