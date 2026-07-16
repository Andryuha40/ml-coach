# Feature selection and dimensionality reduction (PCA)

These notes were assembled from the transcripts "Feature selection and
dimensionality reduction" (lecture section
`content/lectures/Лекции__17. Запись лекций_30 мая 2026.txt`)
and "Feature selection" (seminar section
`content/seminars/Семинары__17. Запись семинара_2 поток_3 июня_2026.txt`),
as well as Evgeny Patochenko's repository: lecture `lesson_12` "Dimensionality
reduction" (PCA, SVD, MDS, t-SNE — a more rigorous mathematical derivation)
and two notebooks — seminar `lesson_03` "Regression and feature selection"
(a practical feature-selection pipeline) and seminar `lesson_12`
"Dimensionality reduction. Visualization" (PCA in practice, kernel PCA,
t-SNE).

Both source files (lecture 17 and seminar 17) are shared with other topics
of the course: a significant part of them is devoted to ensembles/boosting
(topic `ensembles-bagging-boosting`), class imbalance, and feature encoding
(topics `class-imbalance`, `eda-and-pipeline`) — that material is **not**
duplicated here; only the part relating to feature selection and
dimensionality reduction has been taken.

## Why feature selection and dimensionality reduction are needed

Feature selection is a procedure that removes some features from the data.
Reasons:

- Too many features is bad for classical machine learning models: unlike
  deep models, they cannot learn efficiently from an excessively large
  number of features, and it increases training time.
- Extra features that are unrelated to the target variable create noise,
  make it harder for the model to find the real dependency, and contribute
  to overfitting.
- Features that are strongly dependent (correlated) on each other
  (multicollinearity) should also be partly removed so the model does not
  overfit.

### The curse of dimensionality

The higher the dimensionality of the data (the more features), the:
- sparser the data becomes and the harder the computations get;
- more the risk of overfitting grows;
- less informative the distance between points becomes (which is especially
  critical for metric methods like kNN — see topic `knn`);
- exponentially more data is required to preserve the same model quality.

### Feature selection vs feature extraction

Two fundamentally different ways of reducing dimensionality:

| | Feature Selection | Feature Extraction |
|---|---|---|
| Approach | Selects a subset of relevant features from the original set | Transforms the original features into a new, more informative set |
| Mechanism | Reduces dimensionality while keeping the original features | Reduces dimensionality by transforming the data into a new space |
| Requirements | Requires domain knowledge and feature engineering | Can be applied to raw data without prior engineering |
| Downsides | May lead to loss of useful information when important features are removed | May introduce redundancy and noise if the extracted features are poorly defined |
| Example methods | Filter, Wrapper, Embedded | PCA, LDA, Kernel PCA, autoencoders |

## Feature selection: filter methods

Features are selected based on a pre-computed statistic — the fastest
methods, but they do not account for the joint (combined) influence of
features on each other.

### Quasi-constant features (Variance Threshold)

If a feature takes the same (or almost the same) value on nearly all
objects, it has small variance and can be considered almost a constant that
carries no meaningful information, and removed. There is no exact universal
threshold (a rule of thumb — roughly, variance below one). A reliable way to
check: remove the suspicious feature and see whether the model quality
changed — if not, the feature can indeed be dropped.

### Selection by correlation with the target variable

For each feature, its correlation with the target is computed, and features
with a small absolute correlation are removed. The upside is speed (the
correlation is computed once per feature). The main coefficients: Pearson
(the classic one, maximum 1, no relationship — 0), Spearman, Kendall.

Downsides:
- the absence of (linear, Pearson) correlation does not mean the absence of
  a relationship — there may be a complex nonlinear dependency; there are
  other correlation coefficients for capturing it;
- correlation does not mean a causal relationship (a classic illustrative
  example is the spurious correlation between US spending on science and
  space and the number of suicides by strangulation, with a coefficient of
  about 0.87 — there is most likely no real causal link here);
- **the main problem of all univariate methods** — they do not account for
  the joint influence of features. The fruit example: shape by itself is
  weakly related to the answer "is this an apple", color also correlates
  weakly on its own — but the combination "red and round" already points
  fairly unambiguously to an apple.

### Limitations of univariate methods — in more detail

An example from the seminar: suppose in a two-dimensional feature space
(x1, x2) the classes are linearly separable. If you project the sample onto
a single axis (that is, discard the second feature), objects of the classes
partly mix together and the separability disappears. At the same time, a
univariate importance estimate may show that each feature individually is
weakly informative and suggest removing both — even though together they
give good separability. The problem is even more pronounced if features
affect the target only in combination (interaction). The rule: "don't throw
the baby out with the bathwater" — feature-selection methods can improve the
model, have no effect, or sometimes even worsen quality.

### The chi-square criterion (χ²)

Checks how well the observed frequencies match the theoretically expected
ones under the assumption that the variables are independent. Formula: the
sum over all cells of (observed frequency − expected frequency)² / expected
frequency. The larger the χ² value, the stronger the relationship between
the variables. It works for categorical features — an analog of the
correlation coefficient, but for categorical data. After computing χ² for
all features, a cutoff threshold is set, or the features are sorted in
descending order of χ² and the least significant ones are dropped.

### Mutual information

Estimates the strength of the relationship between a feature and the target
based on the joint probability density of the two variables divided by the
product of their individual (marginal) densities — it shows how much
information about the target variable is contained in a particular feature.
Features with a small mutual-information value are considered of little use.
It is implemented out of the box in scikit-learn.

In a similar way, a feature can be evaluated via **ROC-AUC**: treating the
feature itself as a prediction of the target and computing AUC — the higher
the AUC, the more important the feature.

### The Phik matrix (φk)

A commonly used approach that relies under the hood on the chi-square
criterion and other statistics — it lets you uniformly estimate the
relationship between numeric and categorical features at once for the whole
table ("everything against everything").

**An important caveat about all filter methods**: from the standpoint of
rigorous statistics, their use for feature selection in machine learning is
not entirely rigorous — a "loose" interpretation. Nevertheless, in practice
this approach works well, even if theory and practice do not fully agree
here. You should also be prepared for the fact that a random "noise" feature
sometimes unexpectedly ends up at the top of the importance ranking under
any of these methods — this is a normal feature of such experiments, not a
reason to consider the method broken.

## Wrapper methods

They use greedy (sequential) feature selection — the most accurate from a
practical standpoint, but noticeably slower than filter methods.

**Backward elimination.** Suppose there are 1000 features and you need to
keep 100. At each step the model is trained in turn without each of the
remaining features (with 1000 features — about 1000 training runs), and the
one whose removal improved the model quality the most (or degraded it the
least) is removed. The procedure is repeated on the remaining features until
the desired number is left.

**Forward selection.** The reverse variant: at each step the feature that
gives the largest gain in quality is added to the final set.

There is also a combined strategy: first greedy addition, while it improves
quality, then, once addition stops helping — greedy removal of the features
already accumulated.

## Embedded methods

Feature selection happens as a side effect of training the model itself — in
terms of speed, embedded methods are usually faster than wrapper methods but
slower than purely filter methods.

- **L1 regularization (Lasso)** — zeroes out part of the weights, thereby
  automatically performing feature selection (for more on L1/L2/ElasticNet —
  see topic `linear-regression-gd`). In scikit-learn — `SelectFromModel`
  wrapped around a model with L1 regularization automatically removes
  features with zero/small weight.
- **L0 regularization** — encountered less often, mainly in deep learning;
  it penalizes directly for the number of non-zero weights.
- **Feature importance of trees and tree ensembles** (impurity-based feature
  importance) — the more a feature reduces uncertainty (impurity) at splits,
  the more important it is. The estimate via a random forest is considered
  on average more honest than the weights of linear regression, though not
  perfect either — a random noise feature still sometimes ends up in high
  positions.
- **Built-in selection methods in boosting** — CatBoost and XGBoost have
  their own methods (for example, `select_features`) that let you explicitly
  set the desired number of final features.

Feature selection based on a single model (for example, a fast logistic
regression or a random forest) is a separate stage of feature engineering,
and the resulting set of features can then be used with a completely
different model (not necessarily the one that produced the selection).

## Permutation Importance

The idea: take a single feature column and randomly shuffle the values in it
(thereby "breaking" the feature's link to the objects), then look at how the
model quality changed without retraining:

- if the quality did not change (or almost did not change) — the feature is
  practically useless and can be removed;
- if the quality dropped noticeably — the feature is important;
- if the quality suddenly **improved** after shuffling "random garbage" in
  the column — this is an alarming sign: most likely something is wrong with
  the model or the data (data leakage is possible), and it is worth
  investigating rather than being glad about such an improvement.

The method is simple, but at the same time quite effective.

## The Boruta algorithm

A modification of the permutation-importance idea, arranged as a kind of
"contest":

1. For each original (real) feature, a "shadow" copy is created — the same
   column, but with randomly shuffled values.
2. The real and shadow features are combined into one dataset, a model is
   trained on it, and the importance (feature importance) of all features is
   evaluated.
3. Only those real features whose importance turned out to be higher than
   the importance of their shadow "twin" are kept in the final set.
4. The old shadow features are removed, new ones are generated, and the
   process is repeated several times.

Implemented in the `boruta` library for Python. In the demonstration
example, out of the full set of features the algorithm kept only 4, while
the model quality was 0.87.

## Methodology for running feature-selection experiments

The key principle: you cannot apply all selection methods "in one go" in the
hope that the model will figure it out itself. The right approach is to take
one method at a time, apply it, record the change in model quality, analyze
the reasons — and only then move on to the next method. Experiments should
be as independent of each other as possible.

Since some methods (primarily wrapper) are quite time-consuming, you do not
have to run experiments on the full dataset: you can take a representative
subsample (for example, 50-100 thousand records instead of a million),
having first verified its representativeness with a statistical test — this
speeds up the process several times over without loss of quality of the
conclusions.

**Feature selection for speeding up inference.** Besides improving quality,
feature selection can be applied solely for the sake of speeding up
inference: if the quality of a heavy model is, say, 0.67 by the metric while
the business is satisfied with quality around 0.6, you can remove some
features and get a model that works several times faster, sacrificing part
of the quality. Such selection is done at the final stage, when the model
for production is practically decided.

**A recommended general pipeline** (from the lecture): first — selection by
correlation (remove the clearly weakly correlated features), then — train a
simple model with built-in selection mechanisms (L1/L2 regularization or
feature importance for trees/ensembles) and remove the least significant
features. And only if that is not enough, and time and resources allow —
move on to wrapper methods as the longest and most expensive stage; in
practice you often simply do not get to it.

## Code: a practical feature-selection pipeline

An example from seminar `lesson_03` — predicting the level of violent crime
from socio-demographic features (the Communities and Crime dataset): scaling
→ selection by variance (`VarianceThreshold`) → selection by correlation
with the target (`SelectKBest`) → embedded selection via L1 regularization
(`SelectFromModel` + `Lasso`), all inside a single `Pipeline` with a
hyperparameter search via `GridSearchCV`:

```python
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import pandas as pd
```

```python
# feature selection based on variance
vs_transformer = VarianceThreshold(0.01)

X_train_var = pd.DataFrame(
    data=vs_transformer.fit_transform(X_train_scaled),
    columns=X_train_scaled.columns[vs_transformer.get_support()],
)
X_test_var = pd.DataFrame(
    data=vs_transformer.transform(X_test_scaled),
    columns=X_test_scaled.columns[vs_transformer.get_support()],
)
```

```python
# feature selection based on correlation with the target variable
# Select the 15 best features
sb = SelectKBest(f_regression, k=15)

X_train_kbest = pd.DataFrame(
    data=sb.fit_transform(X_train_var, y_train),
    columns=X_train_var.columns[sb.get_support()],
)
X_test_kbest = pd.DataFrame(
    data=sb.transform(X_test_var), columns=X_test_var.columns[sb.get_support()]
)
```

```python
# selection based on regression with L1 regularization
lasso = Lasso(5.0)
l1_select = SelectFromModel(lasso)

X_train_l1 = pd.DataFrame(
    data=l1_select.fit_transform(X_train_var, y_train),
    columns=X_train_var.columns[l1_select.get_support()],
)
X_test_l1 = pd.DataFrame(
    data=l1_select.transform(X_test_var),
    columns=X_test_var.columns[l1_select.get_support()],
)
```

```python
# the whole pipeline together, with a hyperparameter search
pipe = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        ("variance", VarianceThreshold(0.01)),
        ("selection", SelectFromModel(Lasso(5.0))),
        ("regressor", LinearRegression()),
    ]
)

pipe.fit(X_train, y_train)

param_grid = {
    "variance__threshold": [0.005, 0.0075, 0.009, 0.01, 0.011, 0.012],
    "selection__estimator__alpha": [0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
}
grid_search = GridSearchCV(pipe, param_grid, cv=5, verbose=True)
grid_search.fit(X_train, y_train)

pipe_best = grid_search.best_estimator_
```

## Principal component analysis (PCA)

Unlike feature selection, PCA does not throw away the original features but
builds **new** features from the old ones (automated feature engineering) —
there are far fewer new features than original ones, but they still describe
the same objects. The new features are a linear combination of the old ones.

**Two simplifications of the problem that classical PCA makes**: (1) the
target variable is ignored — the method does not look at the answer at all;
(2) it builds specifically a linear combination of features. Ignoring the
target seems strange at first glance, but in practice it is not so bad: data
often has an internal structure of lower dimensionality that is in no way
related to the target variable, so optimal new features can be built without
looking at the answer.

### Problem statement

There are original numeric features of dimension m, and we want to obtain
new numeric features of dimension d, where d is significantly smaller than
m. Requirements: (1) the new features are linearly expressed through the
original ones (a transition from the old basis to a new one); (2) the
transition should lose the smallest amount of the original information.

**Geometric interpretation.** PCA is a projection of an m-dimensional
feature space onto a d-dimensional subspace. If points in a two-dimensional
space are projected onto a line (a transition from 2D to 1D), with a
well-chosen line the projection will preserve a significant part of the
original spread of the data, while with a poorly chosen one most points will
"collapse" together and information about the spread will be lost.

**Formalizing the requirement.** We want the variance of the projection of
the sample onto the new feature to be maximal — the larger the variance in
the new features, the more of the original information is preserved. The idea
of PCA is to choose a projection so that the variance is as large as possible
among all possible projections. Another, equivalent interpretation: the
search for principal components is a search for a subspace of lower
dimensionality where the squared deviations of the points' projections to
this subspace are minimal.

### Mechanism (step by step)

1. **Standardization of the data (centering).** Before applying PCA, the
   data is centered — its mean value is subtracted from each feature (for
   the method from the repository — it is also divided by the standard
   deviation, bringing the feature to mean 0 and standard deviation 1).
   Centering is important because PCA is sensitive to the scale of features.
2. **Computing the covariance matrix** — PCA computes the covariance between
   features to understand how they are related to each other (do they grow
   together, are they inversely related, and so on).
3. **Finding the principal components.** The method finds the directions
   along which the data is "scattered" the most (maximal variance) — the
   eigenvectors and eigenvalues of the covariance matrix are computed. The
   first component (PC1) is the direction with the largest variance; the
   second (PC2) is perpendicular to the first, with the largest remaining
   variance, and so on. The components must be orthogonal to each other and
   have unit length (normalized).
4. **Selecting the top components and projecting the data.** After ranking
   the components by significance, the first few are selected (for example,
   so that they cover ~95% of the variance), and the original data is
   projected onto the space formed by these components.

### Mathematical derivation (the Lagrangian)

Suppose there are N centered points (their mean is zero) and a vector v (of
unit length) onto which they are projected. The length of the projection of
the i-th point is the dot product of the point with v. The variance along
the direction v is decomposed through the covariance matrix C and is written
as the product (v squared, multiplied by C) — that is, as the quadratic form
v-transposed times C times v.

The task is to maximize this quadratic form subject to the constraint on the
unit length of the vector v (v-transposed times v equals 1). A Lagrange
function is constructed: the quadratic form minus λ times (the norm of v
squared minus 1), where λ is the Lagrange multiplier. Setting the
derivatives with respect to v to zero, we get that the sought vector v must
satisfy the equation Cv = λv — that is, **v is an eigenvector of the
covariance matrix C**, and the corresponding λ is its eigenvalue. The first
principal component is the eigenvector with the largest eigenvalue, the
second — with the next-largest, and so on.

### How many components to keep

There is no single formula — a rule of thumb (for example, if there are more
features than objects in the sample, you can limit yourself to a number of
components on the order of half the number of objects). A practical approach:
the eigenvalues of the covariance matrix are ordered in descending order. You
can compute the proportion of explained variance for the first k components
(the sum of the first k eigenvalues divided by the sum of all) and the
proportion of unexplained variance (one minus this proportion). If you plot
the dependence of the proportion of unexplained variance on the number of
components, it usually first drops sharply and then levels off into a plateau
— the "kink" point (the scree criterion) is usually taken as a suitable
number of components.

### Interpretability and limitations

After applying PCA, the components are usually very hard to interpret — it is
unclear what exactly a given new component means. A well-known exception is
PCA applied to images of faces (**eigenfaces**): the first components (in
descending order of significance) correspond to the most significant facial
features (eyes, nose, mouth, oval of the face), and each subsequent one adds
ever finer details. The transformation can be inverted to reconstruct the
original image from a limited number of components (partially
compress/decompress the data), although in practice other algorithms are
used for image compression.

Limitations of PCA:
- the new components are linear combinations of the original features, so
  they can be hard to interpret;
- PCA is sensitive to the scale of features — you must standardize the data
  in advance;
- with too strong a reduction of dimensionality, important details may be
  lost (loss of information is possible);
- it is assumed that the important information is in the linear combinations
  of features; if the relationships between features are nonlinear, PCA may
  be ineffective (then you need Kernel PCA, t-SNE, etc.).
- it makes sense to use it when there is a clearly expressed correlation
  between features in the data — if the features are already weakly
  correlated with each other, PCA will not give significant compression
  without loss of variance.

## Singular value decomposition (SVD) and its connection to PCA

PCA arrives at exactly the same result as **singular value decomposition**
(SVD) — the decomposition of the data matrix X into a product of three
matrices: U · Σ · V-transposed, where U and V are orthogonal matrices
(responsible for rotating the space), and Σ is a diagonal matrix with
singular values on the diagonal (the singular values are the roots of the
eigenvalues of the matrix X-transposed times X).

The connection to PCA: the columns of the matrix V are the eigenvectors of
the matrix X-transposed times X, that is, they are the principal components.
The columns of the matrix U·Σ are the new features (projections of the
original data onto the principal components). To reduce the dimensionality to
k dimensions, take the first k columns of U and the top k×k block of Σ — the
resulting matrix U_k · Σ_k contains k new features corresponding to the first
k principal components.

**PCA or SVD?** Both methods lead to the same result; the question is
computational convenience. PCA (directly finding the eigenvalues of the
matrix XᵀX) has computational difficulties; for SVD there exist efficient
iterative algorithms for directly computing the decomposition (without
explicitly finding the eigenvalues). In practice, for large datasets SVD
usually works faster, but there is no unambiguous rule. SVD is also actively
used in recommender systems (a separate topic of the course).

## Multidimensional scaling (MDS) and t-SNE — for visualization

PCA (and SVD) can be used not only for compressing the feature space before
training a model, but also for **visualization**: multidimensional data is
reduced to two or three dimensions, which makes graphical representation
possible, helps simplify analysis, visually reveal clusters, and detect
outliers. But PCA as a visualization tool has a limit — the method is linear,
and if the structure of the data is substantially nonlinear, a two-
dimensional PCA projection may poorly reflect the real structure. For this
there are specialized nonlinear methods that are usually used only for
visualization, not as features for further training of a model.

### Multidimensional Scaling (MDS)

The idea is to minimize the squared deviations between the original and the
new pairwise distances between objects. The MDS objective function (raw
stress) is the sum over all pairs of objects of the squared difference
between the original pairwise distance and the distance between the same
objects already in the embedded (reduced) space.

### t-SNE (t-distributed stochastic neighbor embedding)

A nonlinear method oriented toward **preserving the local structure** (unlike
MDS, which tries to preserve exactly the distances): what matters is not the
preservation of absolute distances between objects, but the preservation of
the proportions/neighborhoods between them, thanks to which t-SNE has less of
a tendency to squeeze points into the center than other algorithms. It is
especially convenient when you need to "see" the clusters in the data.

The method minimizes the Kullback-Leibler divergence (KL-divergence) between
the joint probabilities of object similarity in the original and in the
embedded (reduced) space — using gradient descent.

Downsides of t-SNE:
- it is computationally expensive — on datasets with millions of objects the
  computation can take hours, whereas PCA will finish in seconds or minutes;
- the algorithm is stochastic: with different initial conditions it may
  arrive at different local minima of the KL-divergence — it is useful to run
  the algorithm with different initial values and choose the variant with the
  smallest divergence;
- the global structure of the data is not explicitly preserved (distances
  between distant clusters on a t-SNE visualization are uninformative) — this
  downside is partly mitigated by initializing the points with the PCA method
  (`init='pca'`).

## Code: PCA in practice, kernel PCA, and t-SNE

A demonstration of PCA on synthetic two-dimensional data — the method
correctly finds the direction of the largest variance and does not depend on
the rotation of the data (since it works with the covariance matrix, not with
the original axes):

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def PCA_show(dataset):
    plt.scatter(*zip(*dataset), alpha=0.5)

    dec = PCA()
    dec.fit(dataset) # train PCA
    ax = plt.gca()
    for comp_ind in range(dec.components_.shape[0]): # iterate over the extracted components (axes)
        component = dec.components_[comp_ind, :]     # the component
        var = dec.explained_variance_[comp_ind]      # explained variance
        start, end = dec.mean_, component * var
        ax.arrow(start[0], start[1], end[0], end[1],
                 head_width=0.2, head_length=0.4, fc='r', ec='r')

    ax.set_aspect('equal', adjustable='box')
```

On nonlinear data (concentric circles, half-moons, blob clusters) PCA finds
directions that are not good features — the method works well only when the
"main direction of variability" really does describe the data linearly; for
strongly nonlinear manifolds you need kernel PCA, t-SNE, UMAP, and similar
methods.

**PCA on faces (eigenfaces) — compressing and reconstructing images:**

```python
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces(shuffle=True, random_state=432540)
faces_images = faces.data
faces_ids = faces.target
image_shape = (64, 64)

mean_face = faces_images.mean(axis=0)

red = PCA()
faces_images -= mean_face
red.fit(faces_images)
```

```python
# compression by a factor of N and reconstruction
base_size = min(image_shape[0] * image_shape[1], faces_images.shape[0])

def compress_and_show(compress_ratio):
    red = PCA(n_components=int(base_size * compress_ratio))
    red.fit(faces_images)

    faces_compressed = red.transform(faces_images) # project the points onto the extracted axes
    faces_restored = red.inverse_transform(faces_compressed) + mean_face # inverse transformation

    plt.figure(figsize=(16, 8))
    rows, cols = 2, 4
    n_samples = rows * cols
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(faces_restored[i, :].reshape(image_shape), interpolation='none',
                   cmap='gray')
        plt.xticks(())
        plt.yticks(())

compress_and_show(0.5)
```

Using PCA features instead of the original pixels can give higher
classification quality (this was checked via `GridSearchCV` on a
`RandomForestClassifier`, training once on the raw pixels and a second time —
on the first 100 principal components).

**Kernel PCA** — since PCA effectively works not with the original features
but with the matrix of their covariances (essentially — with dot products),
you can replace the ordinary dot product with an arbitrary kernel K(xᵢ, xⱼ) —
this corresponds to a transition into another space where the assumption of
linearity may already make sense (analogous to the kernel trick in SVM, see
topic `svm`). The problem is that it is not clear in advance how to choose the
kernel:

```python
from sklearn.decomposition import KernelPCA

def KPCA_show(X, y):
    kernels_params = [
        dict(kernel='rbf', gamma=5),
        dict(kernel='poly', gamma=10),
        dict(kernel='cosine', gamma=10),
    ]

    for i, p in enumerate(kernels_params):
        dec = KernelPCA(**p)
        X_transformed = dec.fit_transform(X)
        # if the result comes out linearly separable — the kernel was chosen well
```

**t-SNE on the digits dataset:**

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X = digits["data"]
y = digits["target"]

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()
```
