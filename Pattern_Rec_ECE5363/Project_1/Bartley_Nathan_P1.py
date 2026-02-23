import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = input("Enter the path to the CSV file: ")
data = pd.read_csv(csv_file)

Features = ["meas_1", "meas_2", "meas_3", "meas_4"]
Classes = data["species"].unique()


def sw_formula(df, class_col, value_col):
    N = len(df)
    out = 0.0
    for cls, g in df.groupby(class_col):
        Pj = len(g)/N
        sigma2 = g[value_col].var(ddof=0)   # population variance
        out += Pj * sigma2
    return out

def sb_formula(df, class_col, value_col):
    N = len(df)
    mu = df[value_col].mean()
    out = 0.0
    for cls, g in df.groupby(class_col):
        Pj = len(g)/N
        muj = g[value_col].mean()
        out += Pj * (muj - mu)**2
    return out

# for feature in Features:
#     wcv = sw_formula(data, "species", feature)
#     bcv = sb_formula(data, "species", feature)
#     print(f"Feature: {feature}, Within-Class Variance: {wcv:.4f}, Between-Class Variance: {bcv:.4f}")

def batch_perceptron(X, y01, rho=0.01, max_iters=200):
    y = np.where(y01 == 1, 1.0, -1.0)

    N, d = X.shape
    X_aug = np.hstack([np.ones((N, 1)), X])
    w = np.zeros(d + 1)

    for k in range(max_iters):
        scores = X_aug @ w
        mis = (y * scores) <= 0
        n_mis = int(mis.sum())
        if n_mis == 0:
            break

        w = w + rho * (y[mis, None] * X_aug[mis]).sum(axis=0)
    print(f"Batch Perceptron converged in {k+1} iterations with {n_mis} misclassified samples.")
    return w

def least_squares(X, y01):
    y = np.where(y01 == 1, 1.0, -1.0)
    n_samples = X.shape[0]
    X_aug = np.hstack([np.ones((n_samples, 1)), X])

    w = np.linalg.pinv(X_aug) @ y

    missclassified = ((X_aug @ w) * y) <= 0
    n_mis = int(missclassified.sum())
    print(f"Least Squares solution has {n_mis} misclassified samples.")

    return w
def one_versus_one(X, y, class_pair):
    class_1, class_2 = class_pair
    mask = (y == class_1) | (y == class_2)
    X_pair = X[mask]
    y_pair = y[mask]
    y_binary = np.where(y_pair == class_1, -1, 1)
    w = least_squares(X_pair, y_binary)
    return w
# def plot_decision_boundary_2d(ax, w, b, X, label="Decision boundary", **kwargs):
#     """
#     Plots boundary for w[0]*x + w[1]*y + b = 0
#     Uses X to choose plotting range and clips to axes.
#     """
#     w0, w1 = float(w[0]), float(w[1])
#     eps = 1e-12

#     # Set limits from data with a small padding
#     x_min, x_max = X[:, 0].min(), X[:, 0].max()
#     y_min, y_max = X[:, 1].min(), X[:, 1].max()
#     x_pad = 0.05 * (x_max - x_min)
#     y_pad = 0.05 * (y_max - y_min)

#     x_min -= x_pad; x_max += x_pad
#     y_min -= y_pad; y_max += y_pad

#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)

#     # Vertical boundary: w0*x + b = 0
#     if abs(w1) < eps:
#         if abs(w0) < eps:
#             return  # degenerate
#         x0 = -b / w0
#         ax.plot([x0, x0], [y_min, y_max], label=label, **kwargs)
#         return

#     # Regular line
#     xs = np.linspace(x_min, x_max, 300)
#     ys = -(w0 * xs + b) / w1

#     # Clip to visible y-range to avoid huge lines
#     mask = (ys >= y_min) & (ys <= y_max)
#     ax.plot(xs[mask], ys[mask], label=label, **kwargs)
def plot_boundary_contour(ax, w_aug, X, label="Decision boundary"):
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid_aug = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()]
    zz = (grid_aug @ w_aug).reshape(xx.shape)
    cs = ax.contour(xx, yy, zz, levels=[0], linewidths=2)
    cs.collections[0].set_label(label)

######################### Case 1: Setosa Vs. Versi+Vergi with all features ##########################

print("======Case 1: Setosa Vs. Versi+Vergi with all features=========")
class_1 = data[data["species"] == "setosa"]
class_2 = data[data["species"].isin(["versicolor", "virginica"])]
X = pd.concat([class_1[Features], class_2[Features]]).values
y = np.hstack((np.zeros(len(class_1)), np.ones(len(class_2))))
print()
print("________Batch Perceptron__________:")
weights_perceptron = batch_perceptron(X, y)
print("BP_Weights:", weights_perceptron)
print()

print("________Least Squares__________:")
weights_ls = least_squares(X, y)
print("LS_Weights:", weights_ls)
print()
print()

######################### Case 2: Setosa Vs. Versi+Vergi with Features 3 and 4##########################
print("======Case 2: Setosa Vs. Versi+Vergi with Features 3 and 4=========")
class_1 = data[data["species"] == "setosa"]
class_2 = data[data["species"].isin(["versicolor", "virginica"])]
feat_3_4 = ["meas_3", "meas_4"]
X = pd.concat([class_1[feat_3_4], class_2[feat_3_4]]).values
y = np.hstack((np.zeros(len(class_1)), np.ones(len(class_2))))
print()
print("________Batch Perceptron__________:")
weights_perceptron = batch_perceptron(X, y)
print("BP_Weights:", weights_perceptron)

fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(X[y==1, 0], X[y==1, 1], label="Class 1", alpha=0.8)
ax.scatter(X[y==0, 0], X[y==0, 1], label="Class 0", alpha=0.8)

plot_boundary_contour(ax, weights_perceptron, X)

ax.set_xlabel("meas_3")
ax.set_ylabel("meas_4")
ax.legend()
ax.set_title("Case 2.1")

fig.savefig("Case2_1.png")
plt.show()


print()
print("________Least Squares__________:")
weights_ls = least_squares(X, y)
print("LS_Weights:", weights_ls)

fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(X[y==1, 0], X[y==1, 1], label="Class 1", alpha=0.8)
ax.scatter(X[y==0, 0], X[y==0, 1], label="Class 0", alpha=0.8)

plot_boundary_contour(ax, weights_ls, X)

ax.set_xlabel("meas_3")
ax.set_ylabel("meas_4")
ax.legend()
ax.set_title("Case 2.2")
fig.savefig("Case2_2.png")
plt.show()

######################### Case 3: Vergi Vs. Versi+Setosa with all features ##########################

print("======Case 3: Vergi Vs. Versi+Setosa with all features=========")
class_1 = data[data["species"] == "virginica"]
class_2 = data[data["species"].isin(["versicolor", "setosa"])]
X = pd.concat([class_1[Features], class_2[Features]]).values
y = np.hstack((np.zeros(len(class_1)), np.ones(len(class_2))))
print()
print("________Batch Perceptron__________:")
weights_perceptron = batch_perceptron(X, y)
print("BP_Weights:", weights_perceptron)
print()

print("________Least Squares__________:")
weights_ls = least_squares(X, y)
print("LS_Weights:", weights_ls)
print()
print()

######################### Case 4: Vergi Vs. Versi+Setosa with Features 3 and 4 ##########################
print("======Case 4: Vergi Vs. Versi+Setosa with Features 3 and 4=========")
class_1 = data[data["species"] == "virginica"]
class_2 = data[data["species"].isin(["versicolor", "setosa"])]
feat_3_4 = ["meas_3", "meas_4"]
X = pd.concat([class_1[feat_3_4], class_2[feat_3_4]]).values
y = np.hstack((np.zeros(len(class_1)), np.ones(len(class_2))))
print()
print("________Batch Perceptron__________:")
weights_perceptron = batch_perceptron(X, y)
print("BP_Weights:", weights_perceptron)
fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(X[y==1, 0], X[y==1, 1], label="Class 1", alpha=0.8)
ax.scatter(X[y==0, 0], X[y==0, 1], label="Class 0", alpha=0.8)

plot_boundary_contour(ax, weights_perceptron, X)

ax.set_xlabel("meas_3")
ax.set_ylabel("meas_4")
ax.legend()
ax.set_title("Case 4.1")
fig.savefig("Case4_1.png")
plt.show()


print()
print("________Least Squares__________:")
weights_ls = least_squares(X, y)
print("LS_Weights:", weights_ls)

fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(X[y==1, 0], X[y==1, 1], label="Class 1", alpha=0.8)
ax.scatter(X[y==0, 0], X[y==0, 1], label="Class 0", alpha=0.8)

plot_boundary_contour(ax, weights_ls, X)

ax.set_xlabel("meas_3")
ax.set_ylabel("meas_4")
ax.legend()
ax.set_title("Case 4.2")
fig.savefig("Case4_2.png")
plt.show()
######################### Case 5: Multiclass: Setosa Vs. Versi Vs. Vergi Features 3 and 4 ##########################
print("======Case 5: Setosa Vs. Versi Vs. Vergi with Features 3 and 4=========")
class_1 = data[data["species"] == "setosa"]
class_2 = data[data["species"] == "versicolor"]
class_3 = data[data["species"] == "virginica"]
feat_3_4 = ["meas_3", "meas_4"]

X = pd.concat([class_1[feat_3_4], class_2[feat_3_4], class_3[feat_3_4]]).values
y = np.hstack((np.zeros(len(class_1)), np.ones(len(class_2)), np.full(len(class_3), 2)))
print()
print("________Multiclass Least Squares__________:")
# one versus one
w_12 = one_versus_one(X, y, (0, 1))
w_13 = one_versus_one(X, y, (0, 2))
w_23 = one_versus_one(X, y, (1, 2))
print("OVO_Weights (setosa vs versicolor):", w_12)
print("OVO_Weights (setosa vs virginica):", w_13)
print("OVO_Weights (versicolor vs virginica):", w_23)

for w, (c1, c2) in zip([w_12, w_13, w_23], [(0, 1), (0, 2), (1, 2)]):
    fig, ax = plt.subplots(figsize=(6,5))

    mask = (y == c1) | (y == c2)
    X_pair = X[mask]
    y_pair = y[mask]

    ax.scatter(X_pair[y_pair==c1, 0], X_pair[y_pair==c1, 1], label=f"Class {c1}", alpha=0.8)
    ax.scatter(X_pair[y_pair==c2, 0], X_pair[y_pair==c2, 1], label=f"Class {c2}", alpha=0.8)

    plot_boundary_contour(ax, w, X_pair)

    ax.set_xlabel("meas_3")
    ax.set_ylabel("meas_4")
    ax.legend()
    ax.set_title(f"OVO: Class {c1} vs Class {c2}")
    fig.savefig(f"OVO_{c1}_vs_{c2}.png")
    plt.show()

missclassified = 0
for i in range(len(X)):
    votes = []
    for w, (c1, c2) in zip([w_12, w_13, w_23], [(0, 1), (0, 2), (1, 2)]):
        score = np.dot(np.hstack([1.0, X[i]]), w)
        if score >= 0:
            votes.append(c2)
        else:
            votes.append(c1)
    pred = max(set(votes), key=votes.count)
    if pred != y[i]:
        missclassified += 1

print("Number of misclassified samples:", missclassified)