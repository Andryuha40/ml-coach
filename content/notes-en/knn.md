# The k-nearest neighbors method (kNN) and approximate nearest-neighbor search

These notes are assembled from the transcripts of the seminars "The k-nearest
neighbors method (kNN)" and "Locality-Sensitive Hashing and the bias-variance
decomposition"
(`content/seminars/Семинары__12. Запись семинара_2 поток_22 апреля_2026.txt`,
`content/seminars/Семинары__13. Запись семинара_2 поток_29 апреля_2026.txt`)
and material from Evgeny Patochenko's repository `lesson_07`: the lecture
"Nonlinear classification methods" (naive Bayes, kNN, Kernel SVM) and the
seminar notebook "The kNN method" (worked problems + practical code). The
Kernel SVM material from the same `lesson_07` lecture is not duplicated here —
it is covered in detail in the `svm` topic (which also has the more rigorous
kernel formulas).

## The general idea and the compactness hypothesis

The k-nearest neighbors method (k Nearest Neighbors, kNN) is a nonlinear
algorithm applicable both to classification and to regression. It is precisely
where classic machine-learning courses often begin, although strictly speaking
it is more of a heuristic metric algorithm than a model with trainable weights.

The method is based on the **compactness hypothesis**: similar objects are
located close to each other in feature space, and dissimilar ones far apart. If
a new object is surrounded predominantly by objects of one class, it is logical
to assign the object itself to that class too ("if it looks like a duck and
quacks like a duck, then it's probably a duck").

## The algorithm

Classifying a new object:
1. k is a hyperparameter of the algorithm (the number of neighbors), set in
   advance.
2. For the new object, the distance to every object in the training set is
   computed.
3. The objects are sorted by distance, and the k nearest are selected.
4. The prediction becomes the class that occurs most often among these k
   objects.

Formally: the prediction is the class y from the set of all classes Y for which
the sum of the indicators "the i-th of the k nearest neighbors belongs to class
y" (over all k neighbors) is maximal.

**What happens at the `fit` stage.** Unlike models with weights (linear
regression, SVM), kNN has no process of fitting weights from the training set —
k is a hyperparameter set from the outside, not learned. In the "naive" version,
`fit()` simply stores (copies) the entire training set inside the model — there
is no formal training phase. All computations (measuring distances, sorting,
voting) happen already at the `predict` stage, when a new query object arrives.
This is called **lazy learning**.

Consequence: a kNN model weighs roughly as much as the dataset itself (unlike,
for example, a linear model, where only a single row of parameters is stored).

## Distance metrics

The second key hyperparameter of kNN is the metric by which the distance between
objects is measured.

- **Euclidean metric**: the square root of the sum of squared differences across
  coordinates — the ordinary distance (the Pythagorean theorem). The most common
  choice for continuous features.
- **Manhattan metric**: the sum of the absolute values of the differences across
  coordinates. Works better on data with outliers.
- **Minkowski metric** — a generalization of the Euclidean and Manhattan metrics
  through the degree parameter p (at p=1 it is Manhattan, at p=2 it is
  Euclidean).
- **Hamming distance** — a special case of the Minkowski metric for
  categorical/binary features: the number of positions in which the values of
  the vectors do not match. Example: the vectors (1,0,1,1,1) and (1,1,0,1,1)
  differ in 2 positions → Hamming distance = 2.
- **Cosine distance** — for text data: one minus the cosine of the angle between
  the vectors.
- **Jaccard measure**: the ratio of the cardinality of the intersection of sets
  to the cardinality of their union — often used to measure text similarity (see
  the section on LSH below for details).

**A calculation example.** Given two vectors (1, 1, 2) and (2, 0, 3). The
Manhattan distance: |1−2| + |1−0| + |2−3| = 1 + 1 + 1 = 3.

## Feature scaling

Scaling (normalizing) the data before applying kNN is mandatory. The reason: kNN
computes the distance over all features at once, and if the features are on
different scales, a feature with a large spread of values "drowns out" the rest.

An example from the seminar: two men aged 31 and 38 with very different incomes
(on the order of a million for one and on the order of 120 conventional units
for the other) and a woman substantially older, with an income of the same order
as the first man's. By age and gender both men are much closer to each other
than to the woman, but when computing the Euclidean distance over unscaled
features, the enormous difference in income "crushes" the contribution of age,
and it turns out that the man with the high income is closer by distance to the
woman with the high income than to the other man with the low income — even
though in essence they ought to be similar more by age and gender.

An important clarification: if k is already fixed (the model is "trained," i.e.
the sample is stored), simply scaling the data after the fact does not change the
order of closeness of objects to each other — it only shrinks or expands the
distances proportionally. So scaling needs to be built in beforehand, before
using the model, rather than as a way to fix a bad prediction "after the fact."

## Weighted (generalized) kNN

In the "naive" algorithm, all k nearest neighbors are counted with equal weight.
Because of this a counterintuitive situation is possible: among a new object's k
nearest neighbors there turn out to be more representatives of a more distant
class than of the nearer one, and the algorithm returns an unexpected answer (for
example, the object is clearly next to a red circle, but the algorithm predicts a
green triangle, because there happened to be more green triangles among the k
nearest).

The idea of the fix: the closer a neighbor is to the query object, the greater
its weight in the vote should be.

**Weighting by order (rank).** Neighbors are ordered by increasing distance and
assigned a weight inversely proportional to the ordinal number: the nearest gets
weight 1, the next 1/2, the next 1/3, and so on.

**Parzen window method.** A more advanced variant is to account not only for the
order but also for the distance value itself, via a kernel function K, which maps
a distance to some coefficient (weight). The prediction is the class y that
maximizes the sum over the k neighbors of [the indicator of belonging to class y]
× K(distance to the neighbor / h), where h is the window width. Examples of
kernels:

- **Rectangular**: 1/2 for |x| ≤ 1, otherwise 0 — objects farther than the window
  width h get zero weight and do not affect the prediction.
- **Triangular** — another variant of the kernel function.
- **Gaussian**: 1/sqrt(2π) × the exponential of minus x²/2.

Memorizing the specific kernel formulas is not necessary — what matters is the
general principle: the farther an object is from the query, the smaller its
contribution to the vote, and very distant objects can be zeroed out entirely.
The flip side: if an anomalous (outlier) neighbor happens to be near the query,
then because of the high weight given to close objects it can influence the
prediction more strongly — accounting for distance slightly raises the risk of
overfitting (excessive sensitivity to outliers).

## kNN for regression

The naive approach: the prediction becomes the mean value of the target variable
among the k nearest neighbors (instead of voting over classes, as in
classification).

The weighted variant is the **Nadaraya-Watson formula** (a weighted average that
accounts for the distance to the neighbors, by analogy with the Parzen window
method): the prediction equals the sum over all neighbors of (the neighbor's
answer × the kernel weight of the distance to the neighbor), divided by the sum
of the kernel weights. A similar principle is used, for example, in estimating
housing prices: one often goes by the prices of similar objects that are closest
by characteristics.

## The hyperparameter k: overfitting and underfitting

Formalization from the notebook: let k be the number of neighbors; for each
object u, the k objects nearest to it from the training set are taken, and the
class of object u is determined as the argmax over y of the sum of the indicators
"the neighbor belongs to class y" (for the weighted variant — with a weight
inversely proportional to the distance to the neighbor).

As k grows, the model becomes more robust to random outliers and noise in the
data, but at too large a k the model becomes overly simplified and stops
capturing the real patterns — underfitting sets in (in the limit, at k equal to
the number of objects in the sample, the same answer will be predicted for all
queries). At small k, on the contrary, there is a high probability of reacting to
a noisy outlier object — the model overfits (at k=1, an "island" of a
prediction of the wrong class forms around each noisy object, which is logically
incorrect).

Thus increasing k is a way to combat overfitting, but with excessive increases in
k the algorithm slides into underfitting. The optimal value of k needs to be
selected on a validation set or by cross-validation (between these extremes) —
comparing different k by quality on the train set is useless: each object is its
own nearest neighbor, and the optimal value will always turn out to be k=1.

## Advantages and disadvantages of kNN

Advantages: it can solve both classification and regression; it makes no
assumptions about linear separability or the distribution of the data; it is
fairly interpretable (you can explain the answer through the nearest neighbors);
it requires no formal training (lazy learning).

Disadvantages:
1. The model stores the entire training set — it requires a lot of memory.
2. For each prediction you need to compute the distance from the query to all
   objects in the sample — expensive computationally.
3. The data must be scaled.
4. You have to select the number of neighbors k and a distance metric that
   really reflects the similarity of objects well in the specific task — not
   always trivial.

## The computational complexity of kNN

To find the k nearest neighbors for one query object, you need to compute the
distance to each of the l objects in the training set; computing the distance
between a pair of objects takes O(d) operations, where d is the number of
features. The resulting complexity of one prediction is **O(l · d)**.

The nearest-neighbor search task is relevant not only in kNN but also, for
example, in searching for similar texts or in vector databases. For large volumes
of data, exact brute force becomes too expensive — **approximate**
nearest-neighbor search algorithms are used, which find neighbors faster while
preserving acceptable accuracy.

## LSH (Locality-Sensitive Hashing) — approximate nearest-neighbor search

The general idea: objects are passed through a hash function that distributes
them across "bins" (buckets). With a well-chosen hash function, similar objects
more often fall into the same (or neighboring) buckets across different random
hashings, while dissimilar ones almost always fall into different ones. Then, to
search for a new object's neighbors, it is enough to look only at the objects
from the same bucket rather than the whole sample.

### The random projections method

Suitable for tabular data, where an object is represented by a point in a
multidimensional feature space.

1. Several random hyperplanes are drawn through the origin.
2. For each hyperplane, the dot product of the object with the vector defining
   the hyperplane is computed: if it is positive — a 1 is assigned, if negative —
   a 0.
3. After several such hyperplanes (for example, 10), each object is encoded by a
   bit sequence.
4. The similarity of two objects is assessed through the Hamming distance between
   their bit codes. In the example worked in class with 6 hyperplanes, two
   objects matched in 5 of 6 positions (similarity — 5/6).

### MinHash

Used, in particular, to assess the similarity of texts.

**Jaccard measure.** For sets A and B: the ratio of the cardinality of the
intersection to the cardinality of the union. A text (document) is split into
n-grams (shingles) — for example, bigrams. The word "Саша" → bigrams {са, аш,
ша}, the word "Маша" → {ма, аш, ша} (the example uses character bigrams of the
original Russian words). The intersection is {аш, ша} (2 elements), the union is
{ма, аш, ша, са} (4 elements). The Jaccard measure = 2/4 = 0.5.

**The "shingles — documents" matrix.** Rows are all the unique shingles of the
corpus, columns are documents; at the intersection is a 1 if the shingle occurs
in the document, otherwise 0. In practice such a matrix is enormous (tens of
thousands of shingles across millions of documents) and sparse (mostly zeros) —
storing and processing it directly is expensive.

**The MinHash signature-building algorithm** — compresses this large sparse
matrix into a compact signature matrix that approximately preserves the Jaccard
measure:

1. All shingles (the rows of the matrix) are numbered.
2. The rows of the matrix are randomly shuffled (a random permutation of the row
   indices).
3. For each document (column), scanning the rows top to bottom in the shuffled
   order, the minimum index of a row where a 1 stands is found — this value is
   recorded as the document's hash for the given permutation.
4. The procedure (a new random permutation → a new minimum index) is repeated
   many times (the number of repetitions is a hyperparameter of the algorithm:
   the more permutations, the more accurate the approximation to the true Jaccard
   measure).
5. The result is a compact signature matrix: the number of rows equals the number
   of permutations, the number of columns equals the number of documents.

The Jaccard measure computed from the signature matrix approaches the Jaccard
measure computed from the original (large, sparse) shingle matrix.

### Searching for similar documents: splitting into bands (banding)

The final step is searching for similar documents with some specified
probability:

1. The signature matrix is split into "bands" of equal width — groups of rows
   with the same number of rows in each (for example, a matrix of 100
   permutations is divided into 25 bands of 4 rows).
2. Within each band, the columns (chunks of the documents' signatures) are hashed
   into separate buckets.
3. Two documents are considered similar if their columns fell into the same
   bucket in at least one band (not necessarily in all).

If s is the similarity estimate (the Jaccard measure) from one signature row, and
r is the number of rows in one band, then the probability of a match in a
specific band is s to the power of r, and the probability of not matching in any
of the b bands is (1 − s to the power of r) to the power of b. With a sufficient
number of bands this probability becomes very small — that is, similar documents
will with high probability be caught in at least one band.

LSH is actively used in practice in nearest-neighbor search tasks over large
volumes of data, in particular in vector databases.

## Additional: the bias-variance decomposition

This material was covered in the same seminar (13), right after LSH — as a
general tool for assessing the quality of any model, not just kNN.

**A probabilistic model of the data.** It is assumed that the target variable y
is described by a true (unknown) dependency f(x) plus noise: y = f(x) + ε, where
the noise ε is normally distributed with mean 0.

**The error decomposition.** The error of a model a(x) on a fixed object is the
mathematical expectation of the squared difference between the true answer and
the model's prediction. Expanding this expression, we get the sum of three terms:
the squared bias (Bias²) + the variance (Variance) + the noise (σ²).

- **Bias** — how much, on average, the model's prediction differs from the true
  value; it reflects how well the model is in principle able to learn the target
  dependency.
- **Variance** — how much the model's predictions can differ if it is trained on
  different training subsamples; it reflects the model's overfitting.
- **Noise (σ², the irreducible error)** — associated with the randomness in the
  data itself; the model cannot influence this component.

**The "dartboard" illustration**: low bias + low variance — predictions are
accurate and tightly clustered (close to the ideal); low bias + high variance —
accurate on average, but individual predictions are widely scattered (risk of
overfitting); high bias + low variance — predictions are steadily clustered but
systematically far from the truth; high bias + high variance — the worst case.

**The influence of model complexity.** As a model is made more complex (the
number of parameters grows), the bias usually decreases (the model fits the
training set more precisely), but at the same time the variance grows (the model
fits the specific peculiarities of the training set more strongly —
overfitting). The task is to find the optimal model complexity at which the total
error (Bias² + Variance) is minimal. This regularity is not a strict law but an
empirical rule of thumb.

## Additional: the naive Bayes classifier (brief)

Naive Bayes is a supervised learning algorithm based on Bayes' theorem with a
"naive" assumption of conditional independence between each pair of features given
the value of the class. The prediction is the class with the maximum posterior
probability (the denominator of Bayes' theorem — the total probability of the
features — need not be computed, since it is the same for all classes). Thanks to
the independence assumption, the joint probability of the features given the class
decomposes into a product of the separate conditional probabilities of each
feature.

**Gaussian naive Bayes** is often used — it approximates numerical features with
a normal distribution whose parameters (the mean and variance for each class) are
estimated by the maximum-likelihood method. Other variants: multinomial (for
text, for example, spam filters), Bernoulli (binary features), categorical.

Pluses: simplicity of implementation, low computational costs, optimal if the
features really are independent. Minus: in most real tasks the features are not
fully independent, so quality often suffers.

## Code: worked problems and practice on the digits dataset

### Problem (compute by hand)

Classification into 3 classes by 2 features, k=3, Manhattan metric, training set:

| Feature 1 | Feature 2 | Class |
|-----------|-----------|-------|
| 1         | -1        | 1     |
| 2         | 2         | 1     |
| 3         | 2         | 2     |
| 1         | 0         | 3     |
| 2         | -2        | 3     |

Prediction for the object x=(2, -1): distances to objects 1-5 (Manhattan) — 1, 3,
4, 2, 1. The three nearest are objects 1, 4, 5 (classes 1, 3, 3). The majority is
class 3, prediction: 3.

The same feature sample but with continuous answers (regression), k=3: the
neighbors are the same (1, 4, 5) with answers 3.5, -0.4, 0.1 — prediction: the
mean (3.5 − 0.4 + 0.1) / 3 = 1.1.

### Practical code (the digits dataset, selecting k on a validation set)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

clf = KNeighborsClassifier(n_neighbors=3)

data = load_digits()
X = data.images
y = data.target

# flatten the square image into a vector to get an object-feature matrix
X = X.reshape(X.shape[0], -1)

# shuffle the data
X, y = shuffle(X, y)

X_train, y_train = X[:700, :], y[:700]
X_val, y_val = X[700:1300, :], y[700:1300]
X_test, y_test = X[1300:, :], y[1300:]

# Train the classifier and make predictions
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)

# Compute the simplest quality metric of the algorithm - the fraction of correct answers
print("Accuracy is: ", np.mean(y_test==y_predicted))
```

```python
# Selecting k on the validation set:
k_best = -1
best_accuracy = 0

for k in range(1, 20):
    y_predicted = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_val)

    val_accuracy = np.mean(y_predicted==y_val)
    print(f"k = {k}; accuracy = {val_accuracy:.3f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        k_best = k

k_best
```

```python
clf = KNeighborsClassifier(n_neighbors=k_best)
clf.fit(X_train, y_train)

for X_data, y_data in zip([X_train, X_val, X_test], [y_train, y_val, y_test]):
    y_predicted = clf.predict(X_data)
    print(f"Accuracy: {np.mean(y_predicted==y_data):.3f}")
```

The takeaway from the notebook: quality on the training set is the best, but it is
deceptive (the algorithm already knows these objects); quality on the validation
set is also not entirely honest (it was used to select k); the honest estimate is
only the quality on the test set, which was used nowhere — neither in training nor
in selecting the hyperparameters.
