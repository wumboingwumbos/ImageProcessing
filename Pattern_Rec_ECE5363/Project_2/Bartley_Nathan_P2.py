### Project 2: CVXOPT for linearly nonseparable (soft margin) SVMs
# in its Dual Form for C=.1 and C=100 with plots
# part 2: compare comptational efficiency with packaged python SVMs (e.g. sklearn)

# Nathan Bartley
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn import svm
import time

# load data from excel file (x1, x2, y)
data_path = Path(__file__).resolve().parent / "Proj2&3DataSet.txt"
A = np.loadtxt(data_path, dtype=float)

# split data into features and labels
X = A[:, :2]  # features (x1, x2)
y = A[:, 2]   # labels (y)

def train_svm_cvxopt(X, y, C):
    n = X.shape[0]
    K = X @ X.T  # linear kernel Gram matrix
    l = X.shape[1]

    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n)) # b in notes

    # Inequality constraints: 0 <= alpha_i <= C
    G = matrix(np.vstack((-np.eye(n), np.eye(n))))
    h = matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
    # Equality constraint: sum(alpha_i * y_i) = 0
    A = matrix(y, (1, n), "d")
    b = matrix(0.0)

    solvers.options["show_progress"] = True
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution["x"])

    w = (alphas * y) @ X
    # support vectors
    support_mask = alphas > 1e-6
    margin_mask = (alphas > 1e-6) & (alphas < C - 1e-6)
    if np.any(margin_mask):
        b_val = np.mean(y[margin_mask] - X[margin_mask] @ w)
    elif np.any(support_mask):
        b_val = np.mean(y[support_mask] - X[support_mask] @ w)
    else:
        b_val = 0.0
    return w, b_val, alphas


# function to plot decision boundary with margains and support vectors
def plot_decision_boundary(w, b, alphas, X, y, title):
    misclassified_mask = (y * (X @ w + b) <= 0)
    misclass_count = np.sum(misclassified_mask)
    print(f"{title}: Misclassified samples = {misclass_count}")

    plt.figure(figsize=(8, 6))
    support_mask = alphas > 1e-6
    #plot support vectors
    if np.any(support_mask):
        plt.scatter(X[support_mask, 0], X[support_mask, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = w[0] * xx + w[1] * yy + b
    plt.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0], colors=['blue', 'black', 'red'], linestyles=['--', '-', '--'])
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(['Support Vectors'])
    plt.text(
        0.02,
        0.98,
        f"Misclassified: {misclass_count}\nSupport Vectors: {np.sum(support_mask)}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    plt.grid()
    plt.show()
# Train SVM with C=0.1 and C = 100, and plot decision boundaries
C1 = 0.1
C2 = 100
w1, b1, a1 = train_svm_cvxopt(X, y, C1)
plot_decision_boundary(w1, b1, a1, X, y, f'SVM with C={C1}')
w2, b2, a2 = train_svm_cvxopt(X, y, C2)
plot_decision_boundary(w2, b2, a2, X, y, f'SVM with C={C2}')

#------------------ Part 2: Compare with sklearn SVM ------------------#
# Compare the computational efficiency of your implementation of SVM with one in Matlab
# or Python that employs the SMO approach. Present this comparison as a plot of the number
# of training samples versus execution time.


def generate_gaussian_data(seed=100):
    rng = np.random.default_rng(seed)
    class1 = rng.multivariate_normal([1, 3], [[1, 0], [0, 1]], 60)
    class2 = rng.multivariate_normal([4, 1], [[2, 0], [0, 2]], 40)
    Xg = np.vstack((class1, class2))
    yg = np.hstack((np.ones(class1.shape[0]), -np.ones(class2.shape[0])))
    idx = rng.permutation(Xg.shape[0])
    return Xg[idx], yg[idx]


def benchmark_cvxopt_vs_smo(C=1.0):
    Xg, yg = generate_gaussian_data()
    sample_sizes = [20, 40, 50, 65, 85, 100]
    cvxopt_times = []
    smo_times = []
    solvers.options["show_progress"] = False
    for n in sample_sizes:
        Xn = Xg[:n]
        yn = yg[:n]

        t0 = time.perf_counter()
        train_svm_cvxopt(Xn, yn, C)
        cvxopt_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        clf = svm.SVC(kernel="linear", C=C)
        clf.fit(Xn, yn)
        smo_times.append(time.perf_counter() - t0)

    plt.figure(figsize=(7, 5))
    plt.plot(sample_sizes, cvxopt_times, marker='o', label='CVXOPT (QP)')
    plt.plot(sample_sizes, smo_times, marker='s', label='sklearn SVC (SMO/libsvm)')
    plt.xlabel('Number of training samples')
    plt.ylabel('Execution time (s)')
    plt.title('SVM Training Time vs Sample Size')
    plt.grid(True)
    plt.legend()
    plt.show()


benchmark_cvxopt_vs_smo(C=1.0)

