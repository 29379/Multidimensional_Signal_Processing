from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from collections import defaultdict


hidden_layer_sizes = [
    (1, 10),
    (2, 10),
    (3, 10),
    (4, 10),
    (5, 10)
]

def return_classifiers():
    classifiers = []
    for hidden_layer_size in hidden_layer_sizes:
        classifiers.append(MLPClassifier(hidden_layer_sizes=hidden_layer_size))
    return classifiers


def evaluate_classifier(classifier, X, y, rskf):
    scores = []
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clone(classifier)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, y_pred))
    return scores


def ex1():
    dataset = load_digits()
    X = dataset['data']
    y = dataset['target']
    print(dataset.keys())
    print(X.shape)
    print(y.shape)

    fig, ax = plt.subplots(1, 10, figsize=(15, 4))
    for i in range(10):
        ax[i].imshow(X[i].reshape(8, 8), cmap='binary')
        ax[i].set_title(y[i])
    plt.tight_layout()
    plt.show()

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    classifiers = return_classifiers()

    mean_scores = []
    logs = []
    for idx, classifier in enumerate(classifiers):
        scores = evaluate_classifier(classifier, X, y, rskf)
        mean_score = np.mean(scores)
        mean_scores.append(mean_score.item())
        logs.append(f"Mean results for hidden layers size {hidden_layer_sizes[idx]}: {mean_score}")

    best_size_index = np.argmax(mean_scores)
    best_size = hidden_layer_sizes[best_size_index]
    for l in logs:
        print(l)
    print(mean_scores)
    print(f"Best hidden layer size: {best_size}")
    return best_size



def ex2(hidden_layer_size):
    dataset = load_digits()
    X = dataset['data']
    y = dataset['target']
    
    mask = y < 5
    X_closed = X[mask]
    y_closed = y[mask]

    y_binary = np.where(y < 5, 1, 0)

    for i in range(10):
        print(y[i], y_binary[i])
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_size)
    
    logs = []
    # closed dataset
    scores_closed = evaluate_classifier(classifier, X, y, rskf)
    mean_score_closed = np.mean(scores_closed)
    logs.append(f"Mean results for closed dataset: {mean_score_closed}")

    # open dataset
    threshold = 0.8
    inner_scores = []
    outer_scores = []
    for train_index, test_index in rskf.split(X_closed, y_closed):
        X_train, X_test = X_closed[train_index], X_closed[test_index]
        y_train, y_test = y_closed[train_index], y_closed[test_index]
        clf = clone(classifier)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        inner_scores.append(balanced_accuracy_score(y_test, y_pred))

        y_proba = clf.predict_proba(X)
        y_pred_bin = (np.max(y_proba, axis=1) >= threshold).astype(int)
        outer_scores.append(balanced_accuracy_score(y_binary, y_pred_bin))

    mean_inner_score = np.mean(inner_scores)
    mean_outer_score = np.mean(outer_scores)
    logs.append(f"Inner score: {mean_inner_score}")
    logs.append(f"Outer score: {mean_outer_score}")
    print()
    for l in logs:
        print(l)


def ex3(hidden_layer_size):
    thresholds = np.linspace(0.5, 1, 100)
    print(thresholds)

    dataset = load_digits()
    X = dataset['data']
    y = dataset['target']
    
    mask = y < 5
    X_closed = X[mask]
    y_closed = y[mask]

    y_binary = np.where(y < 5, 1, 0)
    
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_size)

    inner_scores = []
    outer_scores = defaultdict(list)
    for train_index, test_index in rskf.split(X_closed, y_closed):
        X_train, X_test = X_closed[train_index], X_closed[test_index]
        y_train, y_test = y_closed[train_index], y_closed[test_index]
        clf = clone(classifier)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        inner_scores.append(balanced_accuracy_score(y_test, y_pred))

        y_proba = clf.predict_proba(X)
        for t in thresholds:
            y_pred = (np.max(y_proba, axis=1) >= t).astype(int)
            outer_scores[t].append(balanced_accuracy_score(y_binary, y_pred))

    mean_inner_score = np.mean(inner_scores)
    mean_outer_scores = []
    for k, v in outer_scores.items():
        mean_outer_scores.append(np.mean(v))
    
    best_threshold_index = np.argmax(mean_outer_scores)
    best_threshold = thresholds[best_threshold_index]

    print("\n\n- - - - - -- - - -- - - - - -- - - - -- - - -\n")
    print(f"Inner score: {mean_inner_score}")
    print("\nOuter scores:")
    for i, outer_score in enumerate(mean_outer_scores):
        print(f"Outer score for {thresholds[i]}: {outer_score}")
    print(f"Best result:\n* Index: {best_threshold_index}\n* Threshold: {best_threshold}\n* Score: {mean_outer_scores[best_threshold_index]}")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(thresholds, mean_outer_scores)
    ax.plot(best_threshold, mean_outer_scores[best_threshold_index], 'ro')
    ax.grid()
    plt.xlabel("threshold")
    plt.ylabel("outer balanced accuracy")
    plt.tight_layout()
    plt.show()


def main():
    hidden_layer_size = ex1()
    print(hidden_layer_size)
    ex2(hidden_layer_size)
    ex3(hidden_layer_size)


if __name__ == "__main__":
    main()
