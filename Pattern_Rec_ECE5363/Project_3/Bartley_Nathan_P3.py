from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

data_path = Path(input("Enter data csv or txt file path: ")).resolve()

# handle txt or csv
if data_path.suffix.lower() == '.csv':
    A = np.loadtxt(data_path, delimiter=',', dtype=float)
elif data_path.suffix.lower() == '.txt':
    A = np.loadtxt(data_path, dtype=float)

# Features and labels
X = A[:, :2]
y = A[:, 2]

def gaussian_kernel(X1, X2, sigma=1.75):
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    dists = (
        np.sum(X1**2, axis=1)[:, np.newaxis]
        + np.sum(X2**2, axis=1)
        - 2 * X1 @ X2.T
    )
    return np.exp(-dists / (2 * sigma**2))

def decision_function(X_eval, X_train, y_train, alphas, b, sigma=1.75):
    K = gaussian_kernel(X_eval, X_train, sigma)   # shape: (m, n)
    return K @ (alphas * y_train) + b

def train_svm_cvxopt(X, y, C, sigma=1.75):
    n = X.shape[0]
    K = gaussian_kernel(X, X, sigma)

    P = matrix(np.outer(y, y) * K, tc='d')
    q = matrix(-np.ones(n), tc='d')

    G = matrix(np.vstack([-np.eye(n), np.eye(n)]), tc='d')
    h = matrix(np.hstack([np.zeros(n), np.ones(n) * C]), tc='d')

    Aeq = matrix(y.reshape(1, -1), tc='d')
    beq = matrix(0.0, tc='d')

    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, Aeq, beq)

    alphas = np.ravel(solution['x'])

    tol = 1e-5
    support_mask = alphas > tol
    margin_mask = (alphas > tol) & (alphas < C - tol)

    support_vectors = X[support_mask]

    # Compute bias b using margin support vectors
    if np.any(margin_mask):
        b_vals = []
        for i in np.where(margin_mask)[0]:
            b_i = y[i] - np.sum(alphas * y * K[:, i])
            b_vals.append(b_i)
        b = np.mean(b_vals)
    else:
        # Fallback if no margin SVs are found numerically
        sv_idx = np.where(support_mask)[0]
        b_vals = []
        for i in sv_idx:
            b_i = y[i] - np.sum(alphas * y * K[:, i])
            b_vals.append(b_i)
        b = np.mean(b_vals)

    # Predictions on training set
    f_train = decision_function(X, X, y, alphas, b, sigma)
    y_pred = np.sign(f_train)
    y_pred[y_pred == 0] = 1

    misclassified = np.sum(y_pred != y)

    print(f"\nC = {C}")
    print(f"Number of support vectors: {np.sum(support_mask)}")
    print(f"Number of margin SVs (0 < alpha < C): {np.sum(margin_mask)}")
    print(f"Bias b = {b:.6f}")
    print(f"Misclassified samples = {misclassified}")

    return alphas, b, support_vectors, support_mask, margin_mask, misclassified

def plot_decision_boundary(X, y, alphas, b, support_mask, margin_mask, C, sigma=1.75):
    plt.figure(figsize=(8, 6))

    # Plot samples
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='o', alpha=0.7, label='Class -1')
    plt.scatter(X[y ==  1, 0], X[y ==  1, 1], c='blue', marker='s', alpha=0.7, label='Class +1')

    # Highlight support vectors
    plt.scatter(
        X[support_mask, 0], X[support_mask, 1],
        s=120, facecolors='none', edgecolors='k', linewidths=1.5,
        label='Support Vectors'
    )

    # margin 
    if np.any(margin_mask):
        plt.scatter(
            X[margin_mask, 0], X[margin_mask, 1],
            s=180, facecolors='none', edgecolors='green', linewidths=1.5,
            label='Margin SVs'
        )

    # Build grid
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = decision_function(grid, X, y, alphas, b, sigma)
    Z = Z.reshape(xx.shape)

    # Plot filled regions (optional but helpful)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.2, cmap='bwr')

    # Plot margin curves f(x)=±1
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='gray', linestyles='--', linewidths=1.5)

    # Plot decision boundary f(x)=0
    plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='-', linewidths=2)

    plt.title(f"Gaussian Kernel SVM (C={C}, sigma={sigma})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.text(x_min + 0.5, y_max - 0.5, f"Misclassified: {np.sum(y != np.sign(decision_function(X, X, y, alphas, b, sigma)))}", fontsize=10)
    plt.text(x_min + 0.5, y_max - 1.0, f"Support Vectors: {np.sum(support_mask)}", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- Train and plot for both C values ----
sigma = 1.75

for C in [10, 100]:
    alphas, b, support_vectors, support_mask, margin_mask, misclassified = train_svm_cvxopt(X, y, C, sigma)
    plot_decision_boundary(X, y, alphas, b, support_mask, margin_mask, C, sigma)