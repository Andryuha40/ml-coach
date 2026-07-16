# Ensembles: bagging, random forest, stacking, boosting

These notes were assembled from lectures in Evgeny Patochenko's repository (`lesson_09` —
ensemble methods, `lesson_10` — gradient boosting; the basis for the
formulas and tables), course lectures ("Lecture recording, May 16" — a detailed
explanation of bagging/random forest/OOB/boosting in plain words with a walkthrough of
student questions, and the second half of "Lecture recording, May 30" — a detailed
breakdown of XGBoost/CatBoost/LightGBM with preserved details missing from the
repository), a course seminar ("Seminar recording, May 20" — stacking,
a practical comparison of techniques on sales data from Ecuador, an important takeaway
about feature engineering) and repository seminars (`09_Семинар`,
`10. Семинар. Бустинги с подбором гиперпараметров` — code).

The first half of "Lecture 17" (about the RAG homework, feature selection and PCA) does not
belong to this topic — it is covered in the topics `rag-systems` and
`feature-selection-dim-reduction`.

## Why we need ensembles: recap of models' strengths and weaknesses

By this point in the course we have already covered linear models, kNN and decision
trees — and each family has its own pros and cons:

- **kNN** — simple interpretation (the model just memorizes the sample), but
  mediocre generalization ability.
- **Linear models** — robust to noise in the data (a small "wobble" of the
  points barely shifts the separating line), but poor at capturing
  nonlinear dependencies.
- **Decision trees** — good at finding nonlinear dependencies, but prone
  to overfitting and unstable: a small change in the training sample can
  greatly change the tree's structure.

The idea lying on the surface: if different models make errors differently on
the same task, combining their predictions can yield a higher-quality
prediction than any of the models on its own. **Ensembles (compositions) of
models** are built on this idea — one of the key
approaches in machine learning, especially when applied to tabular data.
Three main approaches are distinguished: bagging, stacking and boosting.

## Decomposing the error into bias, variance and noise (recap)

A model's error (for example, MSE) can be decomposed into three components:

- **Bias** — how well, on average, the model is able to
  approximate the true dependence f(x). To estimate it you take many subsamples
  of the training sample, train a model on each, average the predictions and
  look at how close the averaged model is to the true dependence.
  Small bias — a good prediction on average; large bias — the model is
  too simple (underfitting).
- **Variance** — how strongly the model's predictions at a
  specific point change depending on which training
  sample it was trained on. Small variance — a model robust to changes in the data;
  large variance — a heavily overfitted, sensitive model.
- **Noise** — the irreducible part of the error, tied to the very nature of the data
  (randomness, measurement imprecision, insufficiency of features to
  exhaustively describe the dependence). You cannot affect it — either
  clean the data, or accept that the dependence is weak.

A visual demonstration is the `mlxtend` library, the function
`bias_variance_decomp`: you pass it a model, train/test data, a loss
function and the number of random subsamples for the estimate, and it returns the total
error, bias and variance. On the California Housing dataset it is shown that:

- **Linear regression**: almost all the error is bias (the model is robust
  to noise, but describes the complex dependence poorly).
- **Decision tree** without constraints: the bias is noticeably lower, but the variance is
  much higher (the tree easily overfits and changes greatly when the
  sample changes).

Conclusion: if we could "knock down" the tree's variance to the level of linear regression,
the total error would drop sharply — this is exactly the idea of
ensembling on top of trees: reduce variance while preserving low
bias.

As model complexity grows, bias usually falls and variance rises (and
vice versa) — this is the bias-variance trade-off, covered in more detail in the
`decision-trees` topic.

## The general scheme of an ensemble

An ensemble of models is an approach that builds a prediction using not
one model, but a set of base models and an aggregating (meta-)model
that combines their predictions into the final answer.

Ways to aggregate the answers of the base algorithms:

**For regression:**
- the arithmetic mean of the base models' answers;
- the median of the answers;
- a weighted average (with weights reflecting the "trust" in each base
  model).

**For classification:**
- hard voting — the final class is the one for which the majority of
  base models voted;
- soft voting — the class probabilities predicted by each base model
  are averaged (more accurate than hard voting, but requires the base
  models to have a `predict_proba` method);
- a unanimity committee — a stricter variant where an answer is counted
  only if all base models agree.

Ideally the base models should be as independent (uncorrelated) as possible
from each other — then ensembling gives the greatest benefit. Full
independence is hard to achieve, but in practice one strives for it:
using models of different classes, different hyperparameters, different
initialization, different subsets of the training sample or different loss
functions.

**Ways to obtain different subsamples for the base models:**
- cross-validation — K overlapping subsamples;
- pasting — sampling objects without replacement;
- the random subspace method — all objects, but a random subset of
  features (without replacement);
- the random patches method — a combination of random sampling of both
  objects and features;
- **bootstrap** — sampling objects with replacement; it underlies
  bagging and random forest.

## Bootstrap

From the original sample of size N a new subsample of the same (classically)
size is formed as follows: objects are drawn from the original sample
randomly, one at a time, **with replacement** ("we draw an object from the bag,
write it into the subsample and put it back"), and so on N times. As a result,
in the subsample some objects of the original sample are repeated several times, while
some do not make it in at all.

In the classic bootstrap the subsample size strictly matches the size of the
original sample (this matters, for example, for certain statistical
estimates), but in the context of random forest an exact match is not essential
— the goal here is simply to obtain maximally diverse subsamples for
the different base trees.

How many such subsamples are needed (that is, how many base models to build) —
there is no strict theoretical answer, it is a hyperparameter. An empirical
guideline is a number of trees on the order of a few hundred (100–1000); a practical
way to tune it is to build a plot of quality (or out-of-bag error) against the number
of base models and stop where the plot reaches a plateau.

## Bagging (Bootstrap Aggregating)

**Idea**: preserve the good properties of the base algorithm (low bias)
by building a composition of several instances trained on different
bootstrap subsamples, so as to reduce the variance (instability)
of the final prediction.

**Procedure:**
1. From the original sample, N bootstrap subsamples are formed.
2. On each, its own base model is trained (not necessarily a tree, but
   trees fit best — see below why).
3. The base models' predictions are aggregated: averaging for regression,
   voting or averaging probabilities for classification.

### Why bagging works

Two key properties:

1. **Bagging does not worsen bias**: the bias of the composition equals the bias
   of a single base model. This follows from the linearity of expectation
   — averaging is linear, so the average bias does not change when
   new models are added.
2. **If the base algorithms are uncorrelated with each other**, the variance
   of a composition of N models is N times smaller than the variance of a single
   base model: Var(ensemble) = Var(base) / N. If the correlation between
   the base models ρ is not zero, the formula gets more complicated: the variance
   of the ensemble tends not to zero but to ρ · Var(base) as N grows — that is,
   the benefit of adding new models decreases as the correlation between
   them grows.

Hence two practical conclusions:
- Since bagging's bias equals the bias of the base model, for a good
  result **the base algorithm itself must predict the target reasonably well**
  — this is exactly why bagging is most often built over decision
  trees (and over overfitted, deep trees at that: averaging does not
  change their bias, but reduces their high variance well).
- To decorrelate the base models, you need to "shake up" their training data
  as much as possible. It is precisely the tree's instability (its
  sensitivity to the training sample), which was a drawback of a
  single tree, that becomes an advantage here: bootstrap subsamples
  give trees that differ noticeably in structure, and hence weakly
  correlated predictions. If you take more stable algorithms as base models
  (for example, linear ones), bagging will work noticeably
  worse — their predictions are already strongly correlated with each other.

Adding one more tree, trained on a new independent bootstrap subsample,
to a composition that already has a large number of trees does not change the
composition's bias, but reduces its variance — that is, with bagging (unlike
boosting, see below) you can grow the number of base models with impunity: in the
worst case it just spends extra computational resources without
harming quality.

## Random Forest

A single bootstrap may not be enough to decorrelate the trees —
the subsamples still partially overlap. Random forest adds a second
source of diversity: when building each node of a tree, the optimal
predicate is searched for not among all features, but among a **random subset
of features** of size m (while the hyperparameters of all trees in the forest are
the same — diversity is ensured only by randomness in the subsamples
of objects and features).

**Pseudocode for building a random forest:**
1. Generate a bootstrap subsample.
2. Build a tree on this subsample, but at each step when searching for a
   predicate select a random subset of m features out of all p and
   search for the best predicate only among them.
3. Constrain the tree construction with the condition: no fewer than n_min objects in
   each leaf.
4. Repeat steps 1–3 N times (by the number of trees) and combine the trees into a
   composition.

**Empirical default hyperparameter values** (guidelines established in the
community, not strict formulas):
- the number of features m considered at each step: for classification —
  the square root of the total number of features p (m = √p), for regression —
  about a third of the features (m ≈ p / 3);
- the minimum number of objects in a leaf n_min: for classification — 1, for
  regression — often 5 (in various sources 3 is also encountered).

**Depth of the base trees**: shallow trees have few parameters,
poorly capture patterns in the data and have large bias;
deep trees memorize the training sample too strongly and have
large variance. Since averaging in bagging does not change the bias
of the base algorithm and only reduces variance, for a random forest it makes
sense to take precisely **deep** (individually overfitted) trees —
their high variance is then "damped out" by averaging.

**Number of trees**: increasing the number of trees reduces variance, but does not
affect bias; at the same time the reduction of variance is not unbounded (it is limited by
the number of available features and the size of the subsamples), while the training time grows.
In practice you build a plot of error against the number of trees and choose the moment
when the improvement becomes negligible (for a random forest — a typical
guideline is on the order of 100 trees, after which quality usually reaches a plateau).

**Number of features m**: the more features considered at each
step, the more similar the trees turn out (higher correlation, weaker
ensembling effect); the fewer features — the more diverse the trees,
but the weaker each one individually.

### Out-of-Bag (OOB) error

Since each tree is trained on its own bootstrap subsample, on average
about 67% of the unique objects of the original sample make it into it (when the size of the
subsample equals the size of the original sample), and the remaining roughly 33%
the tree did not see during training. These "unseen" objects are not wasted —
for each tree you can compute the error precisely on the part
of the data on which it was not trained, and average such errors over all
trees. Formally:

OOB = the sum over all objects n from 1 to N of the loss function between the true
answer yₙ and the averaged prediction of all trees that did NOT see
object xₙ during training (denote the set of such trees I(n))

There is a theoretical statement: as the number of trees tends to
infinity, the OOB error tends to the leave-one-out estimate of the error — that is,
this is a very accurate, "free" estimate of the forest's generalization ability.

**Practical consequences:**
- Thanks to the OOB estimate, for a random forest you do not have to set aside
  a separate test sample — you can train the forest right away on all available
  data, assessing quality via the OOB score.
- The OOB error is convenient for tuning the number of trees: as n_estimators
  grows, at some point it reaches a plateau — beyond that, increasing the number of
  trees no longer makes sense.

## Stacking

Random forest is good, but training and inference of a large number of trees (hundreds
or thousands) take time. An alternative is stacking: it uses a
relatively small number of base models, possibly from different families
(for example, a tree, linear regression and kNN at the same time).

**Procedure:**
1. The data is split into training and test samples.
2. The training part is split into n folds (as in cross-validation).
3. Each base model is trained on n − 1 folds and makes predictions
   on the remaining one — this is how "meta-features" are obtained (the base
   models' predictions become new features, while the original target variable
   stays the same).
4. On these meta-features a meta-model is trained (for example, a simple shallow
   tree), which makes the final prediction.

Stacking does not directly aim to reduce specifically bias or variance,
but in practice it often reduces the overall error.

**The main problem of stacking** is a high tendency to overfit: if
among the base models there is a heavily overfitted model, the meta-model
will most likely "lean" toward its predictions and overfit too.
Ways to fight this: regularize the base models; strictly separate the data
for training the base models and the meta-model (either through a separate held-out
part, or through a full-fledged K-fold scheme above). The downside of all these
complications — they give back some of the training time that stacking
was originally supposed to save compared to a forest of thousands of trees.

## Boosting: the general idea

Unlike bagging and stacking (where the base models are trained independently
of each other), in **boosting** the training is sequential: each next
model is built so as to reduce the error of the composition accumulated so far —
that is, each next algorithm corrects the errors of the previous one.

### Building the composition using regression as an example

The final composition is the sum of the base algorithms: a(x) = b₁(x) + b₂(x) + ...
+ b_N(x), where the base algorithms are usually (though not necessarily) decision
trees.

**Step 1.** We build the first tree b₁, minimizing, for example, the mean squared
error between b₁(xᵢ) and the true yᵢ over all objects of the training sample.

**Step 2.** We compute the vector of residuals (the error) of the first tree on each
object: sᵢ = yᵢ − b₁(xᵢ).

**Step 3.** We train the second tree b₂ so that it predicts not
the original target y, but the vector of residuals s — that is, b₂ is "aimed" at
predicting what b₁ could not predict. The composition after two
steps: a = b₁ + b₂.

**Step 4 and onward.** We compute the new vector of residuals, now on the predictions of the
current composition, and build the next tree, minimizing precisely this
new vector. And so on: each next tree is trained to predict
what remains "unexplained" by the composition from the previous step.

### Why it is called gradient boosting

Using ordinary residuals (y − a(x)) as the target variable for the
next tree is justified precisely when the overall task is solved by
minimizing MSE (the residuals participate directly in its computation). But what to do
with an arbitrary, possibly complex loss function L? We need to choose such
numbers sᵢ at object xᵢ that most strongly reduce this loss function
with the already-built composition a^(N)(x) fixed. From gradient
descent it is known that the vector of numbers that most strongly reduces the function
is its **antigradient**. Therefore in the general case, as the target
variable for the next tree we take not simply the residual, but minus the
derivative of the loss function L(y, p) with respect to the prediction p, evaluated at
the point of the composition's already-obtained answers:

sᵢ = − ∂L(yᵢ, p) / ∂p, evaluated at p = a^(N)(xᵢ)

When the loss function is MSE, the antigradient exactly coincides with the ordinary
residual (y − a(x)), so "naive" boosting on residuals is a special
case of gradient boosting. But for an arbitrary loss function (not
necessarily simply differentiable) the antigradient scheme always works
— this is exactly the "magic" of gradient boosting: despite the fact that
the real loss function may be complex, each next tree is still
trained on a simple and well-studied task — approximating a
specific vector of numbers (the antigradient) by minimizing MSE.

Hence the name — **gradient boosting**: at each step the composition literally
"descends" along the loss function, as in gradient descent, only the
step is made not in the parameter space of one model, but by adding a
new base algorithm to the composition. Replacing the error vector with the
antigradient is not the only way to build boosting (there are non-gradient
variants too), but in practice gradient boosting works better and
became dominant.

### Loss functions for regression and classification

**Regression:**
- MSE: the antigradient is the ordinary residual sᵢ = yᵢ − a(xᵢ).
- MAE: the antigradient is the sign of the difference, sᵢ = −sign(a(xᵢ) − yᵢ).

**Classification** (logistic loss function L(y, p) = log(1 + e^(−yp))):
the antigradient is computed by the formula sᵢ = yᵢ / (1 + e^(yᵢ·a(xᵢ))).

### Weighted composition and learning rate

In the general form the composition is built not by simple summation, but by a weighted
sum: a(x) = the sum of wₙ·bₙ(x), where the coefficient wₙ can be tuned by a separate
minimization (for example, one-dimensional gradient descent) — this helps
fix problems such as MAE growing under simple summation of answers.

A simpler and more common way of regularizing is shrinking the step
(learning rate) α: a^(N)(x) = a^(N−1)(x) + α · w_N · b^(N)(x),
where α belongs to the interval (0, 1]. The smaller the learning rate, the less
"trust" is placed in each individual base algorithm, and the better,
as a rule, the final quality of the composition turns out (at the cost of a larger
number of iterations).

**Stochastic gradient boosting** — by analogy with stochastic gradient
descent, each tree is trained not on the whole sample, but on a
random subsample (usually half the sample). This reduces the noise level
and speeds up computation.

### Bias, variance and overfitting in boosting

Boosting works oppositely to bagging: each next tree
purposefully reduces the composition's bias (compared to a single
tree), but at the same time fits the composition to the training
data ever more strongly, increasing variance. Therefore:
- as base models for boosting you usually take models with **high
  bias and low variance** — that is, deliberately underfitted,
  shallow trees (a typical depth guideline is 3–6, the starting point is
  3–4);
- unlike bagging, where increasing the number of trees does not harm quality,
  in boosting the training error decreases monotonically as the number of trees grows,
  while the test error has a characteristic **U-shape**: first it falls,
  then it starts to rise — the optimal number of iterations corresponds to the minimum
  of this U-shaped curve on validation/test, right before the
  train/test curves begin to diverge.

Too simple base algorithms (for example, stumps — trees of depth 1) do not
let boosting "grab onto" nonlinearities in the data, and adding new
equally simple trees does not help — the whole composition ends up underfitting.
Too complex (deep) trees quickly overfit and, what
is more important, the next trees have almost no error left that could
be corrected — the boosting mechanism stops working as intended.

The boosting hyperparameters (tree depth, learning rate, number of trees)
are usually shared across all base models of the composition (although each tree has
its own individual target — the antigradient vector at the given step).

### Boosting versus bagging: summary

| | Bagging / random forest | Boosting |
|---|---|---|
| Bias | equals the bias of a single base model | noticeably lower than that of a single base model |
| Variance | roughly N times lower than that of the base model (under independence) | can be regulated by hyperparameters, but by construction grows with the number of iterations |
| Number of base models | can be grown with almost no risk | requires careful tuning — threat of overfitting |
| Training of base models | independent, parallelizable | sequential |

It is considered that with proper tuning gradient boosting usually gives higher
quality than random forest (although this is an empirical rule, not a
strict law — with certain refinement a random forest is capable of
showing comparable results). In practice there is usually less fiddling with hyperparameters
when using modern boosting implementations than when using a random forest —
which is why gradient boosting is informally called the "king" of classical machine
learning algorithms.

## Gradient boosting implementations: XGBoost, CatBoost, LightGBM

The gradient boosting algorithm itself was described as far back as a 1999 paper, but
a truly efficient implementation appeared only in 2014 —
**XGBoost**, after which it literally "blew up" Kaggle: in practically all
competitions with tabular data, boosting-based solutions started to win. In 2017 two more
implementations came out almost simultaneously:
**CatBoost** (Yandex) and **LightGBM** (Microsoft). All three significantly
surpass "naive" classical boosting, and it is not always clear in advance
which implementation will turn out best on a specific task — it is
recommended to try all three with default parameters and compare.

### XGBoost (eXtreme Gradient Boosting)

The key idea is to replace the simple approximation of the antigradient (usually via
MSE) with an approximation of an arbitrary loss function via a **Taylor
series expansion** of the second order in the neighborhood of the composition's current
prediction. Taylor's formula (a reminder): f(x₀ + Δx) is approximated by the sum f(x₀) +
f′(x₀)·Δx + f″(x₀)/2·Δx² + ... — that is, a polynomial built from
the function's derivatives at a point. Using not only the first but also the **second
derivative** of the loss function, the base algorithm gets information not
only about the direction of error reduction (this is given by the first derivative), but
also about how quickly the error changes in this direction (the second
derivative) — that is, it learns the optimal "step size", not just the
direction. This makes it possible to choose tree splits more accurately, update weights
more efficiently, train faster, overfit less and converge more stably on complex loss
functions.

Additionally, two regularization terms are added to the loss function:
- a penalty proportional to the number of leaves of the tree (hyperparameter γ) — reduces
  the tendency to build too many leaves;
- a penalty proportional to the sum of the squares of the predictions in the leaves
  (hyperparameter λ) — reduces the "confidence" of individual predictions.

The logic of the regularization: the leaves accumulate a prediction of the previous
step's error, and if a tree predicts too large a (and
potentially wrong) value, this error will have to be corrected by the next
trees, which spins up overfitting down the chain. By penalizing the number of leaves
and the magnitude of predictions, XGBoost reduces the level of trust in each
individual prediction.

**Advantages**: parallel computation; use of the second
derivative; support for linear base models; regularization; a shrinkage
factor; selection of a subset of columns; handling of missing values.
**Disadvantages**: you need to pass through the whole dataset when splitting nodes; high
complexity of the preliminary sorting of the data.
**Typical hyperparameters**: `learning_rate`, `max_depth`,
`min_child_weight`, `colsample_bytree`, `subsample`, `n_estimators`.

### CatBoost (Categorical Boosting)

A Yandex development, a superstructure over the ideas of XGBoost with several
key features:

1. **Symmetric decision trees.** At each level of the tree the same
   condition is checked (for example, "feature > t" is checked in all
   nodes of the given level), resulting in a fully
   balanced binary tree. Pros: the tree's structure is known in advance
   by depth; such trees are noticeably more robust to small
   changes in the training sample (unlike ordinary trees, whose
   structure "wanders" a lot); they are compactly stored in memory.
2. **Automatic encoding of categorical features.** CatBoost can
   encode categorical features on its own under the hood, combining
   one-hot encoding, target-mean encoding, counter-based
   encoding (sometimes with noise/permutations to avoid target
   leakage). Several models with different encoding
   strategies are trained at once, and the best is chosen. The downside — such
   encoding is less transparent for interpretation.
3. **Ordered boosting.** In classical boosting each
   tree is trained on residuals computed on the same data on which the
   previous trees were already trained — this can lead to a peculiar
   leak of information. CatBoost fights this by randomly shuffling/
   permuting the training data when adding each new tree.

**Practical pros**: trains faster than classical boosting;
shows good results even without fine-tuning hyperparameters (that
is, it is often recommended as a fast first baseline — you can feed in
raw data without manual encoding of categories); built-in cross-validation;
a built-in overfitting detector (can automatically stop training
when the error on held-out data starts to rise); built-in computation
of metrics and visualization of learning curves; works with missing values out of the box.
**Disadvantages**: handling categorical features requires a lot of memory and
time; the random number generator settings can influence the final
result.
**Typical hyperparameters**: `learning_rate`, `depth`, `l2_reg_leaf`,
`cat_features`, `one_hot_max_size`, `rsm`, `iterations`.

### LightGBM (Light Gradient-Boosting Machine)

An implementation from Microsoft whose main advantage (reflected in the
name) is speed. The key idea is **leaf-wise
tree construction** instead of the more common level-wise: in
level-wise construction at each step all nodes of the current depth are split at once,
whereas LightGBM at each step adds one specific
leaf — the one that most strongly reduces the current error.
This gives higher quality for the same tree-construction "budget",
but can lead to deeper, asymmetric trees and increased
risk of overfitting.

Additional speedup — **feature binning**: a continuous
feature is split into a limited number of ordered "bins", which
sharply reduces the number of split options that need to be enumerated when
building a tree (it speeds up training and slightly reduces overfitting).
LightGBM can also efficiently (in O(k·log k) operations) handle
categorical features without explicit encoding, directing an object into a branch
by the categorical value directly.

**Advantages**: a histogram-based algorithm; gradient-based
one-side sampling (GOSS); exclusive feature bundling
(EFB); leaf-wise construction; parallel speedup of data processing;
cache optimization.
**Disadvantages**: can create deeper trees under leaf-wise
construction, which provokes overfitting; sensitivity to noise.
**Typical hyperparameters**: `learning_rate`, `max_depth`,
`min_data_in_leaf`, `categorical_feature`, `feature_fraction`,
`bagging_fraction`, `num_iterations`.

### Comparison table (in general terms)

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree construction | level-wise (balanced) | leaf-wise (faster, risk of overfitting) | symmetric splits via ordered boosting |
| Categorical features | need to encode manually | need to manually specify the categorical columns | handled automatically (target encoding under the hood) |
| Speed/memory | fast, uses more memory | very fast and memory-efficient | normal speed, works great with a large number of categories |
| Best scenario | small/medium numeric datasets | speed is critical, large tabular data | mixed data types, many categories, minimal tuning |

Besides standard regression/classification, gradient boosting is also
widely and effectively applied to time-series forecasting.

## Practice: comparing techniques on real data (seminar takeaways)

In practice (data on sales in a chain of stores, Kaggle) a
sequential comparison of techniques was performed on one and the same pipeline (`Pipeline`
+ `ColumnTransformer` for encoding categories and scaling):

| Model | Error (MSLE) |
|---|---|
| A single tree (after GridSearchCV over hyperparameters) | 0.46 |
| Stacking (tree + linear regression + kNN, default parameters) | 0.51 |
| Bagging (10 trees of depth 10) | 0.44 |
| Random forest | 0.44 |

**Important practical takeaways:**
- Stacking with default parameters of the base models showed a result worse than
  a carefully tuned single tree — because for the tree an extensive
  search over hyperparameters was carried out, while the stacking's base models
  were almost not tuned. Applying a more complex ensembling
  technique by itself does not guarantee a gain in quality if the base models are not
  tuned.
- The number of base models in bagging can be pre-tuned by the
  plot of error against `n_estimators` at default hyperparameters
  of the base model, and the hyperparameters of the base model can be searched over separately
  — this saves time compared to simultaneously searching over both.
- **The main practical takeaway**: choosing the ensembling technique and its
  hyperparameters gives a far smaller gain in quality than work with
  features (feature engineering). In the example covered, limiting the
  training window in time (not using data that is too old, which
  could have been collected under different macroeconomic conditions) and
  adding a meaningful feature (the presence of a promo) reduced the
  error from 0.44 to 0.33 — far more than any manipulations with
  ensembles. Ensembles are a useful but not the main source of quality growth
  in classical ML on tabular data.
- Tree-based models (and their ensembles) do not require feature
  scaling — a tree does not care what scale a feature is in, since it only
  compares a value with a threshold. It is worth scaling only if trees are
  used in combination with other models (for example, in stacking), so as
  not to later hunt for elusive errors.

## Practical code

### Bagging and random forest on real data (apartment prices)

Seminar code (repository, `lesson_09`) — a comparison of linear regression,
a tree and a random forest on one real-estate dataset for Moscow and the region:

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# separate the numeric features and the target variable
X = df_1.drop('Price', axis=1)
y = df_1['Price']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)

# a base tree without constraints (overfits: the train error is much lower than the test error)
dt = DecisionTreeRegressor()
dt.fit(train_x, train_y)

# random forest
rf = RandomForestRegressor(random_state=42, max_depth=15)
rf.fit(train_x, train_y)

pred_train_rf = rf.predict(train_x)
pred_test_rf = rf.predict(test_x)
print(f'RMSE train: {round(root_mean_squared_error(train_y, pred_train_rf), 5)}')
print(f'RMSE test: {round(root_mean_squared_error(test_y, pred_test_rf), 5)}')

# tuning the random forest's hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(train_x, train_y)
best_rf = grid_search.best_estimator_
```

### Gradient boosting "by hand" (reference for code exercises)

Seminar code (repository, `lesson_10`) — step-by-step construction of the composition
through residuals, and then a generalization to an arbitrary loss function
through the antigradient:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generating the sample
np.random.seed(123)
N = 100
X = np.linspace(0, 1, N).reshape(-1, 1)
y = np.sin(X)[:, 0] + np.random.normal(0, 0.1, size=N)

# Step 0: empty composition
a = 0

# Step 1: first tree (decision stump — depth 1)
dt1 = DecisionTreeRegressor(max_depth=1)
dt1.fit(X, y)
dt1_pred = dt1.predict(X)
a = dt1_pred

# Step 2: compute the residuals (for MSE the residuals coincide with the antigradient)
s1 = y - a

# Step 3: the second tree is trained to predict the residuals, not the original y
dt2 = DecisionTreeRegressor(max_depth=1)
dt2.fit(X, s1)
dt2_pred = dt2.predict(X)
a += dt2_pred  # include the second tree's predictions in the composition

# Steps 4+: repeat — each new tree is trained on the residuals of the current composition
s2 = y - a
dt3 = DecisionTreeRegressor(max_depth=1)
dt3.fit(X, s2)
a += dt3.predict(X)
```

Generalization to MAE (the antigradient is the sign of the difference, not the difference itself):

```python
# Antigradient of MAE: s_i = -sign(a(x_i) - y_i)
s1 = -np.sign(a - y)

dt2 = DecisionTreeRegressor(max_depth=1)
dt2.fit(X, s1)
a += dt2.predict(X)
```

Generalization to the logistic loss function (binary classification):

```python
def log_loss(y, y_pred):
    return np.log(1 + np.exp(-y * y_pred)).mean()

# Antigradient of the logistic loss function
s1 = y / (1 + np.exp(y * a))

dt2 = DecisionTreeRegressor(max_depth=1)
dt2.fit(X, s1)
a += dt2.predict(X)
```

A demonstration of boosting overfitting (unlike random forest, where the
train/test error reaches a plateau as the number of trees grows, in boosting the
train error keeps falling, while the test error — after a certain point —
rises):

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

trees = [1, 2, 5, 20, 100, 500, 1000]
loss_rf_test, loss_gb_test = [], []

for ts in trees:
    rf = RandomForestRegressor(n_estimators=ts, max_depth=2)
    gb = GradientBoostingRegressor(n_estimators=ts, max_depth=1, learning_rate=0.1)

    rf.fit(X_train, y_train)
    loss_rf_test.append(mean_squared_error(y_test, rf.predict(X_test)))

    gb.fit(X_train, y_train)
    loss_gb_test.append(mean_squared_error(y_test, gb.predict(X_test)))
```

### Comparing boosting implementations and tuning hyperparameters

```python
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# XGBoost
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary:logistic', eval_metric='auc',
    tree_method='hist', random_state=123, n_jobs=-1
)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict_proba(X_test)
print(f'AUC ROC XGBoost: {roc_auc_score(y_test, pred_xgb[:, 1])}')

# CatBoost
cat = CatBoostClassifier(
    iterations=500, learning_rate=0.1, depth=6,
    loss_function='Logloss', eval_metric='AUC',
    random_seed=123, verbose=False
)
cat.fit(X_train, y_train)
pred_cat = cat.predict_proba(X_test)
print(f'AUC ROC CatBoost: {roc_auc_score(y_test, pred_cat[:, 1])}')

# LightGBM
lgbm = LGBMClassifier(
    n_estimators=500, learning_rate=0.1, max_depth=-1, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary', random_state=123, n_jobs=-1
)
lgbm.fit(X_train, y_train)
pred_lgbm = lgbm.predict_proba(X_test)
print(f'AUC ROC LightGBM: {roc_auc_score(y_test, pred_lgbm[:, 1])}')
```

Tuning hyperparameters three different ways:

```python
# 1) RandomizedSearchCV for XGBoost
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

param_dist_xgb = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
search_xgb = RandomizedSearchCV(
    xgb, param_distributions=param_dist_xgb, n_iter=20,
    cv=cv, scoring='roc_auc', verbose=1, random_state=123
)
search_xgb.fit(X_train, y_train)

# 2) Optuna for LightGBM
import optuna
from sklearn.model_selection import cross_val_score

def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "random_state": 123, "verbose": -1
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective_lgb, n_trials=30)

# 3) CatBoost's built-in grid search
param_grid_cat = {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "iterations": [300, 500, 800]}
grid_result = cat.grid_search(
    param_grid_cat, X=X_train, y=y_train, cv=3,
    shuffle=True, partition_random_seed=123, verbose=False
)
```

Feature importance and SHAP (Shapley values — an interpretation method
that lets you understand the contribution of each feature to a specific prediction, in
contrast to ordinary feature importance, which gives only the overall picture):

```python
importance_xgb = xgb.feature_importances_
top_idx = np.argsort(importance_xgb)[-30:]

import shap
explainer_cat = shap.Explainer(cat)
shap_values_cat = explainer_cat.shap_values(X_train)
shap.summary_plot(shap_values_cat, X_train, max_display=10)
```
