### Pattern Recognition - Project 4 ###
# Nathan Bartley

import numpy as np
from pathlib import Path

# N_100_path = Path("C:\\ImageProcessing\\Pattern_Rec_ECE5363\\Project_4\\Proj4Train100.txt")
# N_1000_path = Path("C:\\ImageProcessing\\Pattern_Rec_ECE5363\\Project_4\\Proj4Train1000.txt")
# Test_path = Path("C:\\ImageProcessing\\Pattern_Rec_ECE5363\\Project_4\\Proj4Test.txt")

N_100_path = Path(input("Enter training data file path for N=100: ")).resolve()
N_1000_path = Path(input("Enter training data file path for N=1000: ")).resolve()
Test_path = Path(input("Enter test data file path: ")).resolve()

# Small regularization to avoid singular covariance or zero variance.
EPS = 1e-6


def load_data(file_path):
    if file_path.suffix.lower() == ".csv":
        return np.loadtxt(file_path, delimiter=",", dtype=float)
    if file_path.suffix.lower() == ".txt":
        return np.loadtxt(file_path, dtype=float)
    raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")


def split_features_labels(data):
    return data[:, :5], data[:, 5].astype(int)


def log_gaussian_diag(x, mean, var):
    return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)


def log_gaussian_full(x, mean, cov):
    d = mean.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + EPS * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)

    diff = x - mean
    quad = diff.T @ np.linalg.solve(cov, diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


# i) Naive Bayes classifier (feature independence)
def naive_bayes_classifier(X_train, y_train, X_test):
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]
    print(f"Classes: {classes}, n_classes: {n_classes}, n_features: {n_features}")

    priors = np.array([np.mean(y_train == c) for c in classes])
    means = np.zeros((n_classes, n_features))
    variances = np.zeros((n_classes, n_features))

    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        means[i] = np.mean(X_c, axis=0)
        variances[i] = np.var(X_c, axis=0) + EPS

    y_pred = np.zeros(X_test.shape[0], dtype=int)
    for i, x in enumerate(X_test):
        log_posteriors = np.zeros(n_classes)
        for j in range(n_classes):
            log_likelihood = log_gaussian_diag(x, means[j], variances[j])
            log_posteriors[j] = np.log(priors[j] + EPS) + log_likelihood
        y_pred[i] = classes[np.argmax(log_posteriors)]

    return y_pred


# ii) Bayes classifier with parameters estimated by MLE (full covariance)
def bayes_classifier_mle(X_train, y_train, X_test):
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]

    priors = np.array([np.mean(y_train == c) for c in classes])
    means = np.zeros((n_classes, n_features))
    covariances = np.zeros((n_classes, n_features, n_features))

    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        means[i] = np.mean(X_c, axis=0)
        covariances[i] = np.cov(X_c, rowvar=False, bias=True) + EPS * np.eye(n_features)

    y_pred = np.zeros(X_test.shape[0], dtype=int)
    for i, x in enumerate(X_test):
        log_posteriors = np.zeros(n_classes)
        for j in range(n_classes):
            log_likelihood = log_gaussian_full(x, means[j], covariances[j])
            log_posteriors[j] = np.log(priors[j] + EPS) + log_likelihood
        y_pred[i] = classes[np.argmax(log_posteriors)]

    return y_pred


# iii) Bayes classifier using true parameter values.
# True generating model parameters:
#   Class 1: N([0,0,0,0,0], S1), Class 2: N([1,1,1,1,1], S2), equal priors.
def bayes_classifier_true_params(X_test):
    means = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    covariances = np.array([
        [
            [0.8, 0.2, 0.1, 0.05, 0.01],
            [0.2, 0.7, 0.1, 0.03, 0.02],
            [0.1, 0.1, 0.8, 0.02, 0.01],
            [0.05, 0.03, 0.02, 0.9, 0.01],
            [0.01, 0.02, 0.01, 0.01, 0.8],
        ],
        [
            [0.9, 0.1, 0.05, 0.02, 0.01],
            [0.1, 0.8, 0.1, 0.02, 0.02],
            [0.05, 0.1, 0.7, 0.02, 0.01],
            [0.02, 0.02, 0.02, 0.6, 0.02],
            [0.01, 0.02, 0.01, 0.02, 0.7],
        ],
    ])
    priors = np.array([0.5, 0.5])

    classes = np.array([1, 2])
    y_pred = np.zeros(X_test.shape[0], dtype=int)

    for i, x in enumerate(X_test):
        log_posteriors = np.zeros(2)
        for j in range(2):
            log_likelihood = log_gaussian_full(x, means[j], covariances[j])
            log_posteriors[j] = np.log(priors[j]) + log_likelihood
        y_pred[i] = classes[np.argmax(log_posteriors)]

    return y_pred


def test_error(y_true, y_pred):
    return np.sum(y_true != y_pred), np.mean(y_true != y_pred)

def evaluate_for_training_size(X_train, y_train, X_test, y_test, n_train):
    y_pred_nb = naive_bayes_classifier(X_train, y_train, X_test)
    y_pred_mle = bayes_classifier_mle(X_train, y_train, X_test)
    y_pred_true = bayes_classifier_true_params(X_test)

    misclass_nb = test_error(y_test, y_pred_nb)
    misclass_mle = test_error(y_test, y_pred_mle)
    misclass_true = test_error(y_test, y_pred_true)

    print(f"\n=== Test Results (N_train = {n_train}) ===")
    print(f"Naive Bayes:                    Misclassified = {int(misclass_nb[0]):4d} / {len(y_test)},  Error Rate = {misclass_nb[1]:.4f} ({misclass_nb[1]*100:.2f}%)")
    print(f"Bayes (MLE, full cov):          Misclassified = {int(misclass_mle[0]):4d} / {len(y_test)},  Error Rate = {misclass_mle[1]:.4f} ({misclass_mle[1]*100:.2f}%)")
    print(f"Bayes (true parameters):        Misclassified = {int(misclass_true[0]):4d} / {len(y_test)},  Error Rate = {misclass_true[1]:.4f} ({misclass_true[1]*100:.2f}%)")


if __name__ == "__main__":
    train_100 = load_data(N_100_path)
    train_1000 = load_data(N_1000_path)
    test_data = load_data(Test_path)

    X_train_100, y_train_100 = split_features_labels(train_100)
    X_train_1000, y_train_1000 = split_features_labels(train_1000)
    X_test, y_test = split_features_labels(test_data)

    evaluate_for_training_size(X_train_100, y_train_100, X_test, y_test, n_train=100)
    evaluate_for_training_size(X_train_1000, y_train_1000, X_test, y_test, n_train=1000)
