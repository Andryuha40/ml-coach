# Decision trees

These notes were assembled from a lecture in Evgeny Patochenko's repository (`lesson_08` — the
foundation, the theory of splitting criteria), a course lecture ("Lecture recording, April 21" —
which gives a more detailed, conversational version of the same material plus an introduction to
ensembles and a bias-variance experiment that isn't in the repository), and a seminar from the
repository, "Decision trees" (practical code, including a hand-written implementation of the error
criterion).

## Why we need decision trees

Up to this topic we covered linear algorithms: they can only draw a linear separating hyperplane. If
the data are linearly inseparable (for classification) or the relationship between features and the
target variable is nonlinear (for regression), linear models inevitably make mistakes. Decision
trees are a family of models that can find arbitrary nonlinear patterns while remaining easily
interpretable (unlike, for example, neural networks): the decision-making process is structurally
similar to how a human reasons, asking themselves a sequence of simple questions.

The second, no less important reason to study trees is that they serve as a "building block" for
ensembles (random forest, boosting), which in practice are considered some of the most powerful
algorithms of classical machine learning.

## Anatomy of a tree: nodes, root, leaves, predicates

A decision tree is a kind of graph (a directed tree) whose vertices (nodes) hold **predicates** —
simple checkable conditions ("questions") on one of the features. Depending on the answer to the
predicate, we move down the tree into one of the child vertices.

- **Root** — the vertex where the process begins: it receives the entire training set (during
  training) or the object for which a prediction is being made.
- Exactly one edge enters each vertex, and two or more may leave it.
- **Leaf** — a vertex with no outgoing edges; it is precisely in the leaves that the final
  prediction is formed.
- **Binary tree** — exactly two edges leave each vertex: the predicate takes the value "true/false",
  and depending on the answer we go left or right. This is the classic (though not the only
  possible) way of building a tree.
- **Tree depth** — the number of levels of vertices with predicates (leaves are not counted).

Formally: each vertex v is assigned a predicate function that checks a condition on an object and
returns 0 or 1. Each leaf vertex is assigned a prediction: for classification — a class or a class
probability; for regression — a real number.

## Building a predicate: comparing a feature against a threshold

The usual form of a predicate in a node is a comparison of the value of a single feature against a
threshold, for example "x₁ > t". If the condition holds, the object goes one way; if not, the other.
Even if the original sample is linearly inseparable by a single such condition, several predicates
can be applied in sequence, "cutting out" rectangular regions of feature space (the analogy is
splitting the area under a curve into small rectangles when taking an integral: the finer and more
frequent the partitions, the more accurate the final approximation of the complex shape).

How the best threshold for a specific feature is chosen: all the values the feature takes in the
training set are taken and tried one by one as candidate thresholds. For example, if x₁ takes the
values 18, 24, 32, 45, the conditions x₁ > 18, x₁ > 24, x₁ > 32, x₁ > 45 are tried. With n distinct
values of the feature you get n − 1 meaningful thresholds (thresholds outside the range of the
feature's values are pointless — then all objects go the same way). Thresholds for the remaining
features are tried in the same manner, and out of all options (feature + threshold) the one that
gives the best split by the chosen criterion is selected.

**More complex predicates** (for example, a slanted hyperplane — a linear combination of several
features instead of comparing a single feature against a threshold) are technically possible but
almost never used in practice: for each node you would have to train a separate model, which sharply
increases computational cost and easily leads to overfitting when there are few objects in the
sub-sample. The gain in quality usually doesn't justify this complexity — the same result can be
achieved by trying simple threshold predicates more often.

## The greedy tree-building algorithm

The training set arrives at the root. Then successive splitting happens: at each vertex the predicate
that gives the best split right "here and now" is chosen — without looking ahead to what will come
out several steps later. This is exactly what is called a **greedy algorithm**. The step is then
recursively repeated for each of the resulting sub-samples until a stopping criterion is met.

The greedy algorithm is not the most optimal: an exhaustive search over all possible ways to build a
tree with different combinations of predicates would be an NP-complete problem (that is, not solvable
in reasonable time). So in practice greedy (locally optimal at each step) construction is always
used.

At prediction time, an object travels down the tree from the root, checking the corresponding
predicate at each node, and eventually "rolls down" into one of the leaves — where the answer is
formed.

## The split quality functional and the splitting criterion

Simply counting the fraction of errors is not enough: the goal is to achieve maximal **homogeneity**
of objects in the resulting sub-samples (for example, if one sub-sample has 10 objects of class A and
2 of class B, and the other the reverse, that is already a good split, even if it isn't entirely free
of "impurities").

A **splitting criterion** H is introduced — an impurity function that measures how the target
variable is distributed among the objects of a set. The more homogeneous the sub-sample (all objects
of one class, or, for regression, all objects with close target values), the smaller the value of H;
a perfectly homogeneous sub-sample gives H = 0.

The split quality functional (for feature j and threshold t) is the weighted sum of the splitting
criteria of the left and right subtrees after the split:

Q(split) = (fraction of objects that fell into the left subtree) × H(left subtree) + (fraction of
objects that fell into the right subtree) × H(right subtree)

The task is to minimize Q, that is, to choose the predicate (feature and threshold) at which the
weighted impurity after the split is the smallest. This is equivalent to maximizing the **information
gain** — the difference between the impurity in the original vertex before the split and the total
weighted impurity of the child vertices after the split.

It's important that the weight of each term is precisely the fraction of objects, not simply the
number of objects or an unweighted sum of H. Otherwise the tree could "cheat": isolate literally a
couple of objects into a separate small vertex, get zero impurity there, and artificially lower the
loss function while barely improving the real quality of the model.

## Splitting criteria for classification: Gini and entropy

Let there be K classes in a classification task, and let pₖ be the fraction of objects of class k that
fell into the vertex. There are two main criteria:

- **Gini criterion (Gini impurity)**: H = 1 − sum of pₖ² over all classes k. An equivalent form
  (after rearranging): sum of pₖ·(1 − pₖ) over all classes. It doesn't require computing a logarithm,
  so it's computationally cheaper — which is why it's more often chosen by default.
- **Entropy**: H = minus the sum of pₖ·log(pₖ) over all classes present in the vertex (if pₖ = 0, the
  corresponding term is simply ignored, since the logarithm of zero is undefined). The base of the
  logarithm is not fixed explicitly — usually the binary logarithm (log₂) is used, but sometimes the
  natural one; in calculation problems it's worth checking the problem statement.

Both criteria have the same needed property: they equal zero when all objects of the sub-sample belong
to one class (maximal homogeneity), and are maximal when the classes are distributed as evenly as
possible in the sub-sample. Trees built with different criteria differ slightly in structure but
behave fundamentally similarly. There are also separate historical names for the algorithms: a tree
with the entropy criterion is sometimes called ID3, a tree with the Gini criterion — CART
(Classification And Regression Tree).

The simplest alternative (a cruder one) is the **classification error**: the fraction of objects not
belonging to the most popular class in the node. Its drawback is that it accounts only for the
fraction of a single (the most frequent) class and doesn't reward prediction confidence in the rest of
the distribution, so in practice Gini or entropy is preferred.

### Example entropy calculation

If a leaf received 9 objects of one class and 11 of another (fractions 9/20 and 11/20), the binary
entropy of such a sub-sample is roughly 1 (close to the maximum, since the classes are distributed
almost evenly).

## The splitting criterion for regression: MSE / variance

For a regression task the tree predicts a single constant c in each leaf. If you take the mean squared
error (MSE) between this constant and the true values of the objects that fell into the leaf as the
splitting criterion, then the minimum of that error is reached exactly when c equals the mean value of
the target variable over the objects of the leaf. From this follows an elegant result: the splitting
criterion H(R) at a vertex for regression is simply the **variance** of the target variable of the
objects that fell into the vertex: the smaller the variance, the smaller the spread of target values
in that leaf.

If instead of MSE you use MAE (mean absolute error), the optimal constant answer in the leaf becomes
not the mean but the **median** of the target variable of the leaf's objects.

Trying thresholds for a specific feature in a regression task works the same as in classification: the
gaps between neighboring (sorted) feature values are tried, the splitting criterion (for example, MSE)
is computed for each, and the threshold with the best value is chosen.

## The prediction in a leaf

- **Classification**: the answer is the probability of belonging to each class — the fraction of
  objects of the corresponding class among those that fell into the leaf. For example, if a leaf
  received 3 objects of class A and 1 of class B out of 4, the probability of class A is 3/4, and of
  class B — 1/4.
- **Regression**: the answer is an aggregated statistic over the target variable of the leaf's
  objects: the mean (if the criterion is MSE) or the median (if the criterion is MAE).

## Overfitting of decision trees

If you let a tree grow without limit, it can perfectly separate any training set — to the point where
each leaf ends up with a single object. Such a tree shows zero error on training, but overfits: it
starts adapting to noisy (outlier) objects, carving out separate tiny regions for them even when there
is no real relationship between the specific object and the target.

Visually, in classification this looks like a heavily jagged separating boundary with individual
"islands" of one or two objects; in regression — like a "staircase" of disproportionate steps
perfectly fitted to every point of the training set, but predicting new objects poorly.

## Stopping criteria (tree regularization)

To avoid driving the tree into overfitting, its growth is stopped early. The main hyperparameters (a
specific library implementation may have up to a couple dozen, but in practice these are the ones
varied first):

- **max_depth** — the maximum depth of the tree. The most common and often the most important of the
  tuned hyperparameters; trees are usually kept shallow (often no deeper than 5 levels — though the
  optimal value always depends on the specific data).
- **min_samples_leaf** — the minimum number of objects required for a vertex to become a leaf; if a
  split creates a group smaller than this value, that split is forbidden.
- **min_samples_split** — the minimum number of objects in a node required for the node to be split
  further at all.
- **max_leaf_nodes** — a limit on the maximum number of leaves in the tree.
- **min_impurity_decrease** — the minimum threshold for the reduction in impurity at a split: if a
  given predicate reduces impurity by less than this threshold, the split is not performed.
- stopping if all objects in a leaf belong to one class.

All these parameters are hyperparameters: they aren't derived analytically but tuned empirically, for
example by grid search (`GridSearchCV`).

## Pruning

An alternative approach to regularization: first build a tree of maximum depth (overfitted, with zero
error on training), then "prune" it — remove those leaf vertices (or subtrees) whose removal does not
increase (or reduces) the error on a separate validation set; a new leaf is formed in place of the
removed subtree.

Mathematically this can be expressed by adding to the tree's error functional a term proportional to
the number of vertices in the tree (a penalty for tree size) — such an updated functional captures
exactly the idea of pruning. One well-known variant is Cost-Complexity Pruning, where the strength of
the penalty is controlled by a parameter α, tuned via cross-validation.

Pruning was popular when a tree was used as a standalone model. Nowadays individual trees are almost
never used: in ensembles the base algorithms are usually already under-fitted, shallow trees
("decision stumps", of depth 1–3), so the need for pruning almost never arises in practice, and in
most libraries it isn't even implemented.

## Handling categorical features and missing values

In theory a decision tree does not require normalization/scaling of features (unlike linear models) —
the model compares a feature's value against a threshold, and the absolute scale of the value does not
affect this comparison. In practice the specific behavior depends on the library.

**Categorical features.** The straightforward approach is to set up a separate edge for each category
value, but with a large number of unique categories the tree grows wide and overfits more easily. A
better option is a predicate of the form "the feature belongs to some subset of categories" (for
example, "brand ∈ {A, B}"). In practice, even libraries that support categorical features "out of the
box" (for example, CatBoost) internally perform encoding into numbers — for example, they order
categories by the mean target value (for regression) or by the fraction of the positive class (for
classification) and replace a category with its ordinal position in that list (something similar to
Mean Target Encoding).

**Missing values.** A tree can handle missing values "out of the box" — one of its practical
advantages. During training: an object with a missing value in a feature involved in a predicate is
simply ignored when computing the loss function for that split. At prediction time: if an object that
fell into a vertex with the predicate "x < t" has a missing value of x, where to send it is unknown,
so the object is sent into both branches at once, reaches both corresponding leaves, and the final
prediction is the average of the two resulting answers, weighted in proportion to the number of
objects that fell into each branch during training. Not all libraries support this (for example, plain
scikit-learn doesn't), but modern gradient boosting libraries (CatBoost, XGBoost, etc.) can handle
missing values under the hood.

## Pros and cons of decision trees

**Pros:**
- Can find nonlinear relationships of arbitrary complexity (unlike linear models).
- High interpretability: understandable predicates ("age > 25"), you can visualize the whole tree and
  explain why the model gave exactly that answer — no "black box" effect.
- Require no special data preprocessing: no normalization needed, categorical features and missing
  values can be handled "out of the box" (though in practice much depends on the specific
  implementation).
- Train fast and produce predictions fast; few parameters; a lightweight model, cheap inference.

**Cons:**
- A strong tendency to overfit without control of depth/stopping criteria.
- **Instability**: a small change in the training set can strongly change the structure of the tree
  (unlike, for example, linear regression, which barely changes under a small "wobbling" of points
  around the line). This instability later turns out to be key for bagging — what was a drawback of a
  single tree becomes a source of "diversity" among the base models in an ensemble.
- The separating boundary is piecewise-linear (composed of hyperplanes perpendicular to the feature
  axes), with limitations in shape.
- Poor extrapolation: outside the range of feature values the tree was trained on, predictions can
  behave unpredictably.
- Finding the optimal tree is an NP-complete problem, so in reality only a greedy (not guaranteed
  optimal) tree is actually built.

## Experiment: bias and variance using a tree

Consider an experiment illustrating the bias-variance tradeoff. Suppose there is a true relationship
y = f(x) + ε with normal noise ε (mean around zero, some known variance). The scheme: many times we
generate a sample from the same distribution, each time retrain the model on the new sample, and make
a prediction at one and the same point x.

- For **linear regression** the predictions at this point are fairly stable (they don't "wander" much
  from sample to sample) — the model is little sensitive to the specific training set.
- For a **decision tree** the predictions are noticeably less stable — the spread of predictions is
  higher.

On this basis two terms are introduced:

- **Bias** — the difference between the model's prediction averaged over many retrainings and the true
  value of the relationship f(x). It characterizes the model's ability to recover the sought
  relationship without getting distracted by noise.
- **Variance** — the variance of the model's answers when retrained on different samples from the same
  distribution. It characterizes the sensitivity of the model's answers to changes in the training
  set.

The familiar model error (for example, MSE) can be decomposed into three terms: the irreducible noise
of the data, the variance, and the squared bias (bias²). The noise can't be influenced, but the
variance and bias can be regulated by changing the complexity of the model (hyperparameters). As a
rule, as the complexity of the model increases, the bias falls while the variance grows, and vice
versa — that is, there exists some optimal ratio (the **bias-variance trade-off**) at which the total
error is minimal. This is an empirical regularity, often observed in practice, but not a strict
theoretical law — there are exceptions (for example, for certain model compositions or for deep neural
networks).

For a decision tree a typical combination is: relatively low bias (the tree approximates complex,
including nonlinear, relationships well) and high variance (instability of structure when the sample
changes). It is precisely at eliminating this high variance while preserving low bias that ensemble
methods on top of trees (bagging, random forest — the topic `ensembles-bagging-boosting`) are aimed.

## Practical code: a tree in sklearn

Comparison of a linear model and a tree on linearly inseparable data (seminar code, the reference for
checking code exercises):

```python
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

np.random.seed(13)
n = 500
X = np.zeros(shape=(n, 2))
X[:, 0] = np.linspace(-5, 5, 500)
X[:, 1] = X[:, 0] + 0.5 * np.random.normal(size=n)
y = (X[:, 1] > X[:, 0]).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)

lr = LogisticRegression(random_state=13)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Linear model accuracy: {accuracy_score(y_pred_lr, y_test):.2f}")

dt = DecisionTreeClassifier(random_state=13)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(f"Decision tree accuracy: {accuracy_score(y_pred_dt, y_test):.2f}")
```

The effect of the hyperparameters `max_depth` and `min_samples_leaf` on tree structure and
overfitting (a grid search over values and visualization of decision boundaries):

```python
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

for i, max_depth in enumerate([3, 5, None]):
    for j, min_samples_leaf in enumerate([15, 5, 1]):
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=13)
        dt.fit(X, y)
        ax[i][j].set_title("max_depth = {} | min_samples_leaf = {}".format(max_depth, min_samples_leaf))
        ax[i][j].axis("off")
        plot_decision_regions(X, y, dt, ax=ax[i][j])  # function from the mlxtend library

plt.show()
```

A regression tree, a grid search over `max_depth`/`min_samples_leaf`/`min_samples_split` by MSE,
feature importances (`feature_importances_`) — on the Boston housing prices dataset (note: `load_boston`
is deprecated and removed from modern versions of sklearn, but the original notebook uses precisely
it):

```python
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree

dt = DecisionTreeRegressor(max_depth=3, random_state=13)
dt.fit(X_train, y_train)

plt.figure(figsize=(12, 10))
plot_tree(dt, feature_names=X.columns, filled=True, rounded=True)
plt.show()

mean_squared_error(y_test, dt.predict(X_test))

max_depth_array = range(2, 20)
mse_array = []

for max_depth in max_depth_array:
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=13)
    dt.fit(X_train, y_train)
    mse_array.append(mean_squared_error(y_test, dt.predict(X_test)))

plt.plot(max_depth_array, mse_array)
plt.title("Dependence of MSE on max depth")
plt.xlabel("max depth")
plt.ylabel("MSE")
plt.show()

dt = DecisionTreeRegressor(max_depth=6, random_state=13)
dt.fit(X_train, y_train)
dt.feature_importances_

pd.DataFrame({
    "feature": X.columns,
    "importance": dt.feature_importances_
}).sort_values(by="importance", ascending=False).reset_index(drop=True)
```

A check that a decision tree does not need feature standardization (unlike linear models) — the quality
barely changes when a `StandardScaler` is added:

```python
print("No scaling is applied\n")
for max_depth in [3, 6]:
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=13)
    dt.fit(X_train, y_train)
    print(f"MSE on test set for depth {max_depth}: {mean_squared_error(y_test, dt.predict(X_test)):.2f}")

print("Standard scaling is applied\n")
for max_depth in [3, 6]:
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=13)
    dt.fit(X_train_scaled, y_train)
    print(f"MSE on test set for depth {max_depth}: {mean_squared_error(y_test, dt.predict(X_test_scaled)):.2f}")
```

## A hand-written implementation of the error criterion (for code exercises)

The seminar shows how to "by hand" try thresholds for a single feature and compute the split quality
criterion Q (the weighted variance of the target variable in the left and right parts) — reference,
non-optimized code, to understand the mechanics of a tree before calling a ready-made `sklearn`:

```python
# sort the dataframe by the feature so it's convenient to iterate over the threshold
data_train.sort_values("CRIM", inplace=True)

# iterate over all possible splits by the feature
# for each split compute Q:
# first for the left and right subtree compute the splitting criterion - the variance of the answers,
# then add up the values of the splitting criterion with weights
quals = []
for i in range(data_train.shape[0]):
    quality = data_train["target"][:i].std()**2 * i / data_train.shape[0] + \
        data_train["target"][i:].std()**2 * (1 - i / data_train.shape[0])
    quals.append(quality)

plt.plot(data_train["CRIM"], quals)
plt.xlabel("feature CRIM")
plt.ylabel("Q")
```

A bonus seminar task (ready-made signature stubs, the implementation deliberately left as `pass` — a
good scaffold for a "fill in the function" code exercise):

```python
from typing import Iterable, List, Tuple

def H(R: np.array) -> float:
    """
    Compute impurity criterion for a fixed set of objects R.
    Last column is assumed to contain target value
    """
    pass


def split_node(R_m: np.array, feature: str, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a fixed set of objects R_m given feature number and threshold t
    """
    pass


def q_error(R_m: np.array, feature: str, t: float) -> float:
    """
    Compute error criterion for given split parameters
    """
    pass


def get_optimal_split(R_m: np.array, feature: str) -> Tuple[float, List[float]]:
    Q_array = []
    feature_values = np.unique(R_m[feature])

    for t in feature_values:
        Q_array.append(q_error(R_m, feature, t))

    opt_threshold = # your code here

    return opt_threshold, Q_array
```

## Introduction to ensembles (a preview of the next topic)

The idea underlying the next topic (`ensembles-bagging-boosting`): "two heads are better than one" —
combining several base algorithms a₁, a₂, ..., a_K into a common ensemble (a composition) whose answers
are aggregated often improves the final accuracy. Two main approaches:

- **Bagging** — the answers of the base algorithms are averaged, but each base algorithm is trained on
  its own, slightly different sub-sample (via Bootstrap sampling — sampling with replacement of the
  same size as the original).
- **Boosting** — each next algorithm in the composition corrects (improves) the answer of the previous
  one, that is, incrementally improves the overall answer of the whole composition.

There's no strict rule for how many base models to include in an ensemble — it's an empirically tuned
hyperparameter: usually you plot quality against the number of base algorithms and stop when the
growth in quality plateaus (for a random forest this often happens already after ~100 trees).

The intuition for why an ensemble reduces error (for regression, if the ensemble's answer is the
arithmetic mean of the answers of the base models h₁, ..., h_K): if the base models are unbiased and
their answers don't correlate with each other, the ensemble's prediction is also unbiased, and the
ensemble's variance decreases in proportion to the number of base algorithms K (Var(ensemble) =
Var(base) / K). But if the answers correlate (correlation ρ > 0), the ensemble's variance as K grows
tends not to zero but to ρ·Var(base) — that is, the smaller the correlation between the base
algorithms, the greater the gain from ensembling. It is precisely the instability of decision trees
(their high sensitivity to the training set) that makes them an especially good "building material" for
bagging — Bootstrap sub-samples yield noticeably differing trees, and their predictions correlate
weakly with each other.
