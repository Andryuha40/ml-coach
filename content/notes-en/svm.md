# Support vector machines (SVM)

These notes were assembled from the transcript of the seminar "Support vector
machines (SVM)"
(`content/seminars/Семинары__11. Запись семинара_2 поток_15 апреля_2026.txt`,
the basis for the intuition and the derivation of the loss function) and
materials from Evgeny Patochenko's repository: lecture `lesson_06` (a more
rigorous/formal derivation of the margin and multiclass classification) and
two notebooks — seminar `lesson_06` (multiclass SVM classification in
practice) and seminar `lesson_07` (a visual comparison of classifiers).

## The general problem statement

In a feature space there are objects split into classes. The classification
task comes down to drawing in this space a line (in a plane), a plane (in a
three-dimensional space), or a hyperplane (in a space of arbitrary
dimensionality) that separates the objects into classes.

SVM does the same thing as logistic regression — it builds a separating
hyperplane. The difference is in exactly how it does this: SVM maximizes the
width of the band (gap) between the classes, rather than just fitting a
probabilistic model.

A sample is called **linearly separable** if there exists a parameter vector
such that the corresponding classifier makes no errors on this sample.

## Pros and cons of SVM

Pros:
- robust to overfitting (logistic regression is more vulnerable in this
  respect);
- robust to outliers (logistic regression may react inadequately to outliers
  and overfit because of them);
- gives an acceptable result on small amounts of data;
- can build nonlinearly separating hyperplanes thanks to the kernel trick —
  that is, it can also solve problems for linearly non-separable samples.

Cons, because of which the method is used fairly rarely in practice:
- trains slowly (especially noticeable during a hyperparameter search);
- works poorly with probabilities — unlike logistic regression, SVM in
  general is not designed to produce probability estimates;
- when the kernel trick is used, the computational complexity rises sharply,
  and the model becomes harder to interpret.

## Margin and support vectors

The separating hyperplane is defined by the equation: the dot product of the
weight vector w with an object's features x plus the intercept b equals zero
(⟨w, x⟩ + b = 0).

**The margin** of an object: M(xᵢ) = yᵢ · (⟨w, xᵢ⟩ + b), where yᵢ is the true
class label of the object (+1 or −1).

- if the sign of the prediction matches the true label, then M(xᵢ) > 0 — the
  object is classified correctly;
- if the signs do not match, then M(xᵢ) < 0 — the object is classified
  incorrectly.

The absolute value of the margin |M(xᵢ)| is geometrically interpreted as the
distance from the object to the separating hyperplane, that is, as the degree
of the classifier's confidence: the larger the absolute value of the margin,
the farther the object is from the hyperplane and the more confident the
classifier. Hence the idea: the best classifier is the one whose sample
objects are at the maximum distance from the separating hyperplane (maximal
margins).

**Support vectors** are the objects (points) lying closest to the separating
hyperplane: they lie exactly on the boundary of the separating band (gap) or,
in the case of a soft margin, enter inside it. It is precisely on these
objects that the construction of the separating band "rests" — hence the name
of the method.

Besides the main hyperplane, two auxiliary (support) hyperplanes parallel to
it are introduced, passing through the support vectors: ⟨w, x⟩ + b = 1 and
⟨w, x⟩ + b = −1. The width of the band (gap) between them equals 2 / ‖w‖,
where ‖w‖ is the norm of the weight vector.

**A rigorous derivation (repository).** The distance from a point x₀ to the
hyperplane H, defined by the normal vector w and the offset b, equals the
absolute value of (⟨w, x₀⟩ + b) divided by the norm of w. Using the freedom
to choose the scale of the weights, you can normalize them so that for the
nearest sample object x* it holds that min|⟨w, x⟩ + b| = 1. Then the distance
to the nearest object equals 1/‖w‖, and the width of the gap between the
positive and negative classes is twice that: M = 2/‖w‖. It is exactly this
that SVM maximizes.

## Hard Margin

The sample is linearly separable, and there must not be a single error inside
the separating band — all objects of one class lie strictly on one side, and
those of the other — strictly on the other. Formally: yᵢ · (⟨w, xᵢ⟩ + b) ≥ 1
for all i.

## Hinge Loss

To formalize the correctness of the prediction, the piecewise-linear loss
function Hinge Loss is used:

For a single object: L(xᵢ) = 0 if yᵢ · (⟨w, xᵢ⟩ + b) ≥ 1 (the object is
classified correctly and lies outside the band); otherwise
L(xᵢ) = 1 − yᵢ · (⟨w, xᵢ⟩ + b). Compactly:
L(xᵢ) = max(0, 1 − yᵢ · (⟨w, xᵢ⟩ + b)).

The loss function averaged over the sample is the mean value of L(xᵢ) over
all objects. The more objects fall beyond the boundary of the support
hyperplanes (that is, the worse the classification), the larger the value of
the hinge loss.

## The final SVM loss function

The SVM problem consists of two parts: (1) the correctness of the prediction
(minimizing the hinge loss) and (2) maximizing the gap between the classes
(the wider the gap, the more confident the classifier). Since the width of
the gap equals 2/‖w‖, maximizing the gap is equivalent to minimizing the
squared norm of the weights (‖w‖²).

The final loss function to minimize: the sum of the squared norm of the
weights and the hinge loss averaged over the sample — that is,
‖w‖² + (1/n) · the sum over all objects of max(0, 1 − yᵢ · (⟨w, xᵢ⟩ + b)). The
first term is responsible for maximizing the gap, the second — for minimizing
the classification error.

### Training via subgradient descent

The hinge loss is piecewise-linear and not differentiable at the kink point,
so ordinary gradient descent is not directly applicable to it — the method of
**subgradient descent** is used: the gradient is computed separately for each
point depending on how it is classified.

- If an object is classified correctly and lies outside the band
  (yᵢ · (⟨w, xᵢ⟩ + b) ≥ 1), only the contribution from maximizing the gap is
  taken into account: the gradient with respect to w equals 2w, the gradient
  with respect to b equals 0.
- If the condition does not hold (the object is classified incorrectly or has
  fallen inside the band), the gradient also contains the contribution from
  the classification error: the gradient with respect to w equals 2w − yᵢxᵢ,
  the gradient with respect to b equals −yᵢ.

The training algorithm: initialize w and b (with zeros or at random); in a
loop over a given number of iterations, for each sample object check the
margin condition, compute the corresponding gradient, and update the weights
in the direction of the anti-gradient with step η (learning rate); stop by
the number of iterations or by a convergence criterion.

## Soft Margin and the C parameter

In practice, classes are rarely arranged perfectly "cleanly" — almost always
objects of different classes touch or partly fall into the gap region. A
coefficient C is added to the loss function, allowing the classifier to make
errors on some objects:

L(w, b) = ‖w‖² + C · (1/n) · the sum over objects of max(0, 1 − yᵢ · (⟨w, xᵢ⟩ + b))

The coefficient C regulates the ratio between the contribution of maximizing
the gap and the contribution of minimizing the prediction error:
- C = 1 corresponds to the case of a hard margin (Hard Margin);
- C from 0 to 1 — the role of the classification error decreases (the model
  is more tolerant of errors, the gap is wider);
- C greater than 1 — the role of the classification error increases (the
  model is less tolerant of errors, the gap is narrower, the behavior is
  closer to a hard margin).

**A formalization from the repository (an equivalent notation).** A penalty
ξᵢ ≥ 0 is introduced for each object, allowing a violation of the margin: the
condition Mᵢ(w) ≥ 1 − ξᵢ. Then the task is to minimize half the squared norm
of w plus C times the sum of the penalties ξᵢ, subject to this constraint.
Such a problem has a unique solution and simultaneously requires the
penalties to be as small as possible and the gap as wide as possible.

Three situations are possible regarding the arrangement of the classes:
1. The classes are linearly separable without errors — Hard Margin SVM.
2. The classes are linearly separable with errors (objects partly enter the
   gap) — Soft Margin SVM.
3. The classes are linearly non-separable in principle (for example, one
   class is a circle completely surrounded by another class forming a ring) —
   no straight line will separate the classes, the kernel trick is needed.

## The kernel trick for linearly non-separable data

The idea: the data is transferred from the original feature space into a
space of higher dimensionality (a new dimension is added). In the new space
it becomes possible to separate the data with a hyperplane.

Example: for two concentric circles in the plane (x1, x2), you can introduce
a new feature z = the square root of (x1² + x2²). Then the two-dimensional
data turns into three-dimensional data (x1, x2, z), forming a figure like a
paraboloid of revolution, in which the classes can now be separated by a
plane ("cut off" the apex of the paraboloid).

Directly adding such features (for example, polynomial ones) is a workable
but computationally expensive path: the number of features after such a
transformation can increase hundreds of times. The **kernel trick** solves
this problem differently: instead of explicitly building new features, a
**kernel** K(a, b) is used — a function that directly computes the dot product
of the already transformed vectors, without explicitly computing the
transformation itself. For example, for a polynomial transformation of degree
n, the dot product of the transformed vectors equals the dot product of the
original vectors raised to the power n. This makes it possible to build
complex separating boundaries without an explosive growth of dimensionality
and computational complexity.

**The main types of kernels** (the more rigorous formulas are from the
repository):

- **Linear**: K(a, b) = the dot product of a and b — an ordinary linear
  classifier with no transformation of the space.
- **Polynomial**: K(a, b) = (γ · ⟨a, b⟩ + r) to the power d. The degree d
  controls the complexity of the boundary; γ is a parameter that determines
  the radius of influence of the training objects (the smaller γ, the
  smoother the boundary); r is a shift coefficient that affects the
  flexibility of the kernel.
- **RBF (Radial Basis Function, Gaussian)**: K(a, b) = exp(−γ · the squared
  distance between a and b). Less prone to overfitting and well suited to
  data of a complex nonlinear shape; it takes into account not only the
  values of the features but also their distribution.
- **Sigmoid**: K(a, b) = tanh(γ · ⟨a, b⟩ + r). Imitates the behavior of a
  two-layer neural network with a sigmoid activation function; works well
  with complex nonlinear cases, but may overfit with noise and outliers.

The type of kernel is another hyperparameter of the model that has to be
tuned; it is not always clear in advance which kernel will suit particular
data.

**A practical demonstration (seminar, `SVC` from scikit-learn):**
- on synthetic concentric circles, an ordinary linear classifier cannot cope,
  while an RBF kernel (`kernel='rbf'`) separates the circles correctly;
- on data in the shape of two half-moons (`make_moons`), the RBF kernel did
  not work completely correctly, and a polynomial kernel coped better
  (although also with some errors) — that is, there is no guarantee that the
  chosen kernel will fit.

## Multiclass classification

SVM is originally designed for binary classification. Classification into K
classes (K > 2), where each object belongs to exactly one class (multiclass,
as opposed to multilabel, where an object can belong to several classes at
once), is reduced to binary problems by one of two approaches.

### One-vs-Rest (OvR, also One-vs-All)

K binary classifiers are trained — each solves the problem "does the object
belong to class k or not". The final prediction is the class of the
classifier that gave the most confident answer (the largest value of the
decision function). The problem with this approach: the classifiers may have
different scales of the decision function, which makes comparing them directly
("who is more confident") not entirely correct.

### One-vs-One (OvO, also All-vs-All)

For each pair of classes i and j, a separate binary classifier is trained,
solving the problem "class i or class j". With K classes there are K(K−1)/2
classifiers, and each is trained only on the objects of its two classes. The
final prediction is the class predicted by the largest number of classifiers
(voting). The problem: with large K the number of classifiers grows
quadratically, which can be computationally expensive.

`sklearn.svm.SVC`, for a multiclass problem, uses by default a built-in
strategy based on One-vs-One. The `sklearn.multiclass` module also provides
separate meta-estimators `OneVsOneClassifier` and `OneVsRestClassifier`, with
which you can wrap any binary classifier.

### Multiclass vs multilabel

**Multiclass** — each object can belong to only one class. **Multilabel** —
each object can belong to several classes at once (a problem with overlapping
classes).

## Quality metrics for multiclass classification: averaging

To compute the algorithm's quality across all classes at once, various ways
of averaging the quality computed on each class separately are applied:

- **Macro-average** — the metric (for example, precision) is computed
  separately for each class (as for a binary classifier "class against the
  rest"), then these K values are averaged with equal weight. It does not
  account for the imbalance in class sizes.
- **Micro-average** — the quantities TP, TN, FP, FN are summed over all
  classes at once (over the whole confusion matrix), and the metric is
  computed from these aggregate values.
- **Weighted-average** — like macro-averaging, but with weights proportional
  to the number of objects of each class in the sample (it accounts for the
  imbalance).

An additional metric for imbalanced data is **balanced accuracy**: the mean
value of the recall computed for each class separately (it was used in the
example from the notebook, see the code below).

## Practical code: OvR/OvO for SVM and handling class imbalance

An example from seminar `lesson_06` (the Yeast dataset — predicting the
localization of a protein inside a cell, 10 classes, 8 features) — a
comparison of the built-in `SVC` strategy, `OneVsOneClassifier`, and
`OneVsRestClassifier`, including handling of class imbalance via
`class_weight="balanced"`:

```python
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate
)

from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
)

from sklearn.svm import SVC

scoring = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]

# Cross-validation
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0)

# Base SVM model
svm = SVC(kernel="linear", C=1.0, random_state=0)

# Different multiclass classification strategies
ovo_svm = OneVsOneClassifier(svm)
ovr_svm = OneVsRestClassifier(svm)

# Evaluation via cross-validation
cv_results_svm = cross_validate(svm, X, y, cv=cv, n_jobs=3, scoring=scoring)
cv_results_ovo = cross_validate(ovo_svm, X, y, cv=cv, n_jobs=3, scoring=scoring)
cv_results_ovr = cross_validate(ovr_svm, X, y, cv=cv, n_jobs=3, scoring=scoring)
```

```python
# The same thing, but with handling of class imbalance
svm_balanced = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=0)

ovo_svm_balanced = OneVsOneClassifier(svm_balanced)
ovr_svm_balanced = OneVsRestClassifier(svm_balanced)

cv_results_svm_balanced = cross_validate(svm_balanced, X, y, cv=cv, n_jobs=3, scoring=scoring)
cv_results_ovo_balanced = cross_validate(ovo_svm_balanced, X, y, cv=cv, n_jobs=3, scoring=scoring)
cv_results_ovr_balanced = cross_validate(ovr_svm_balanced, X, y, cv=cv, n_jobs=3, scoring=scoring)
```

The conclusion from the notebook: for the SVC (built-in OvO) and OvO
strategies, when `class_weight="balanced"` is added, accuracy decreases, but
the metrics sensitive to class imbalance grow (macro-precision/recall/F1,
balanced_accuracy) — the model becomes "fairer" to the rare classes at the
expense of the frequent ones. But for OvR the balancing behaved differently:
accuracy rose slightly, while balanced_accuracy and f1_macro even declined
slightly. The reason lies in the OvR scheme itself: each binary classifier
solves the problem "one class against the nine others merged into one", and
with `class_weight="balanced"` the small positive class gets a
disproportionately large weight against the merged negative one, while the
final label is chosen by argmax over the decision functions of independently
trained classifiers that are not comparable in scale — so the effect of
balancing on the macro-metrics is not necessarily positive. The conclusion:
the effect of `class_weight="balanced"` depends on the chosen strategy for
reducing to binary problems.

## Code: a visual comparison of the separating boundaries of different classifiers

The seminar notebook `lesson_07` compares kNN, linear and RBF SVM, a decision
tree, a random forest, and naive Bayes on three synthetic datasets
(half-moons, concentric circles, linearly separable data), drawing the
separating boundary of each classifier:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=SEED),
    SVC(gamma=2, C=1, random_state=SEED),
    DecisionTreeClassifier(max_depth=5, random_state=SEED),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=SEED),
    GaussianNB(),
]
```

The conclusion from the notebook: in high-dimensional spaces the data may turn
out to be linearly separable, and in such cases simpler methods (for example,
linear SVM or naive Bayes) may provide better generalization than more complex
models — it is important to keep this in mind and not choose a model that is
"complex by default".

## A practical note from the seminar

SVM is applied fairly rarely in practice — except in certain tasks (for
example, anomaly detection); experienced practitioners often do not use it at
all over years of work. Nevertheless, the concept of the method (margin,
support vectors, the kernel trick) is important for understanding and is often
asked about in interviews.
