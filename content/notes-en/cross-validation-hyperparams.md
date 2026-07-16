# Cross-validation and hyperparameter tuning

These notes were assembled from a local lecture, "Basic methods of training models"
(`content/lectures/05. Лекция. Базовые методы обучения моделей.pdf`), and a transcript of the seminar
"Overview of models... Generalization ability... Cross-validation... Hyperparameter tuning"
(`content/seminars/Семинары__10. Запись семинара_2 поток_8 апреля_2026.txt`). This topic has no separate
`repo_lesson` source — according to the course map (`_coach/curriculum.yaml`) it uses only these two
files.

The lecture "Basic methods of training models" is roughly half devoted to overfitting/underfitting and
regularization (Ridge/Lasso/ElasticNet) — this is already covered in detail in the topic
`linear-regression-gd` and isn't duplicated here. Below is what isn't there: the details of
train/validation/test, the mechanics of cross-validation, the dependence of quality on the amount of
data, and a full overview of hyperparameter tuning methods.

## A model's generalization ability

The key concept when evaluating a model is its **generalization ability**: what matters is not how
well the model fitted the training data, but how well it works on new, previously unseen data, because
it's precisely new data that the model will deal with in real use (production).

An illustrative example on Fisher's Irises: if you split the sample with the `train_test_split`
function with `shuffle=False`, the data isn't shuffled and goes in order (first the objects of class 0,
then 1, then 2). Then train mostly gets classes 0 and 1, and test gets only class 2, which the model
never saw during training. Logistic regression in this case gives an accuracy of just 0.76. If you turn
on `shuffle=True` (the default value), the quality rises to 0.97.

But there's a nuance here too: random shuffling doesn't guarantee preserving the class proportions
between train and test — for example, with an original class ratio of 70/30, after a random split train
might end up with 90/10, and test with 60/40. To preserve the original proportions, the `stratify=y`
parameter is used. But even after that the model's quality can jump noticeably when `random_state`
changes (for example, between quality values of 1.0 and 0.95) — that is, a single train/test split by
itself gives an unstable, random quality estimate. It is precisely to remove this dependence of the
estimate on the random split that cross-validation is needed.

## Train / Validation / Test: why a third part is needed

Hyperparameter tuning is usually done on a validation set. But if you use only train and test for it,
it turns out that the validation role (essentially the "homework" for the hyperparameters) goes either
to train (then the hyperparameters overfit to the training set) or to test (then test stops being an
honest, independent final check — it too gets "peeked at" when choosing the hyperparameters).

To avoid this, the original data is split into three parts:

- **Train** — the main data space on which the model learns (the model's parameters are tuned).
- **Validation** — data for calibrating the model: the quality of algorithms with different
  hyperparameter values is measured on it, and the best hyperparameters are chosen based on that
  measurement.
- **Test** — the final measurement of the model's quality on data that never participated in either
  training or hyperparameter tuning.

## Cross-validation

**Definition.** Cross-validation is a procedure for an honest estimate of a model's generalization
ability. Canonically it's needed precisely for **estimating** a model's quality, not for choosing a
specific final model and not for training — although in practice it's often used to choose a model too
(see below).

### k-Fold CV

The most frequently used mechanism. The training set is divided into k equal parts (folds). At each
step one part is used as the test (validation) part, and the remaining k−1 for training. The process is
repeated k times, successively changing the validation part. The final quality of the model is the mean
value of the metric across all folds (or the quality on a separately held-out test set).

By default in scikit-learn k=5, but you can set any value — usually 3, 5, or 7. As a result, every
object of the original data has been in both the training and the test part at least once — this is what
removes the dependence of the estimate on the randomness of a single train/test split.

**What to do with the k trained models you get?** Canonically, cross-validation is an evaluation tool:
after it, the mean metric across folds is computed, and, if desired, the model is retrained on the
entire training set. Some practitioners use cross-validation differently — as a way to choose the final
model from the k obtained: they either take the best by quality, or average the predictions of all k
models (for regression — averaging, for classification — voting, which is why an odd number of folds is
often taken). Both approaches occur in practice.

**Choosing the number of folds k.**

- With small k (for example, k=2) the estimate can be pessimistically understated because of the small
  size of the training part at each step.
- With large k (for example, k=10) the estimate can have large variance because of the small size of
  the validation part, and you have to spend a long time training many models.

**The spread of the metric across folds as a diagnostic.** If the quality on cross-validation jumps a
lot from fold to fold (for example, from 0.5 to 1.0), that's a bad sign: the model is unstable with
respect to the input data and will most likely behave just as unpredictably on new data in production.
This is a signal to consider abandoning the current model — to try other hyperparameters or another
family of models. That is, cross-validation gives not only an averaged metric but also an indirect
indicator of the model's stability — the spread of the metric across folds.

### Stratified k-Fold

Stratification is used for splitting into folds — each fold contains approximately the same class ratio
as the entire original set. It's especially important with class imbalance: with an ordinary random
split, some folds may contain too few objects of certain classes or none at all.

An important practical nuance: by default, for classification tasks the `cross_val_score` function
under the hood uses not the ordinary `KFold` but precisely `StratifiedKFold`. This can be demonstrated
on the unshuffled Irises dataset (the classes go in order in blocks): when splitting "as is" via
`cross_val_score`, the quality stays high (around 0.98), because stratification is at work under the
hood; but if you explicitly specify the classic `KFold` without stratification on the same unshuffled
data, you get a quality close to zero on each fold — because a class the model didn't see during
training ends up entirely in the test fold. This is a frequent tricky question in interviews.

### Leave-One-Out (LOO)

A special case of k-Fold where k equals the number of objects in the sample N: the size of the test
fold is one — exactly one object goes into the test. If there are 100 objects in the sample, 100 models
will be trained.

LOO loses a lot on time: in the class demonstration, 5-fold cross-validation took about 4 seconds,
while LOO on the same dataset took about 139 seconds, that is, roughly 30 times longer. LOO is usually
applied on small datasets when you want to be maximally cautious and make sure the model shows
consistently adequate quality on all objects without any randomness in the split (for example, in tasks
that need very reliable validation — say, predicting a diagnosis). In practice LOO is applied fairly
rarely.

### cross_val_score and cross_validate — technical details

`cross_val_score` takes a model and data, uses 5 folds by default; the number of folds is set by the
`cv` parameter (for example, `cv=10`). Useful parameters:

- **verbose** — the level of detail in the output during training; useful when training takes a long
  time and you want to see progress across folds.
- **n_jobs** — parallelization across CPU cores; `n_jobs=-1` means use all available cores. Supported
  for most models (trees, random forest, boosting, SVM, kNN, etc.), and if parallelization isn't
  supported for a specific model, specifying `n_jobs=-1` won't break the code — the model will just run
  as usual, which is why it's recommended to specify it almost always.
- **scoring** — the metric by which quality is evaluated (a string name from the scikit-learn
  documentation). Metrics that by their nature need to be minimized (for example, MSE) are represented
  in scikit-learn in an "inverted", multiplied-by-−1 form with the `neg_` prefix (for example,
  `neg_mean_squared_error`) — because all auxiliary tools (including cross-validation) by convention
  maximize the passed metric.

`cross_validate` is a more elaborate function: it gives additional information (the training time
`fit_time` and the prediction time `score_time` on each fold) and lets you pass a list of several
metrics at once via `scoring` (for example, `['accuracy', 'f1_macro']`), getting an estimate for each of
them in a single call. It's convenient to wrap the result in a `pandas.DataFrame`.

The `sklearn.datasets` module also has functions for generating synthetic data for a specific task —
for example, `make_regression`, `make_classification`.

## The dependence of model quality on the amount of data

A practical observation: if a model's quality is poor, it often seems that it's enough to "pour in more
data" — and this is partly true, but obtaining additional data isn't always easy in practice
(especially if the data needs to be labeled by assessors — and labeling costs money), so you want to
understand how much data is actually needed, so as not to spend extra and not to "under-pour".

For most classical machine learning models, the dependence of quality on the amount of training data
looks like a curve: quality grows as the amount of data increases and then plateaus (with small
fluctuations) — after a certain point, further increasing the amount of data stops yielding a gain in
quality. The practical conclusion: if you see that the model has plateaued, there's no point in
labeling additional data.

You can assess this dependence with the `learning_curve` function: it takes a model and training data
(X, y) as input, and outputs three arrays: `train_sizes` (the sizes of the sub-samples on which
training was done), `train_scores`, and `test_scores`. Under the hood the data is split into
sub-samples of increasing size, and for each such sub-sample cross-validation is run (usually with 5
folds): the model is trained on the training part and validated on the held-out part. This forms the
arrays of quality metrics on train and on test for each sub-sample size — which lets you plot the
dependence of quality on the amount of data.

## Hyperparameters and model parameters (a reminder)

- **Model parameters** — quantities tuned automatically from the training set (for example, the weights
  w in linear regression).
- **Model hyperparameters** — quantities controlling the training process, which cannot be tuned from
  the training set directly (for example, the regularization coefficient λ or the learning rate η in
  gradient descent).

Each model has its own set of hyperparameters (the list for a specific model — the `get_params()`
method). Example: a random forest has the hyperparameter `n_estimators` (the number of trees). When
trying its values from 1 to ~180, the model's quality grows only up to a certain point (for example, up
to ~61 trees), after which further increasing the number of trees barely changes the quality — this
check helps understand where increasing the hyperparameter value further makes no sense.

But trying each hyperparameter manually one by one is inconvenient, especially when there are many of
them and you want to simultaneously assess the influence of several hyperparameters and their
combinations at once. For this there are several approaches.

## Hyperparameter tuning

In essence, when building a model you control three things: (1) the data (you try to make it as close as
possible to what will be in production), (2) the choice of the model itself (it's not known in advance
which model will turn out best), and (3) the hyperparameters of the chosen model. Trying hyperparameters
over a full grid can take a very long time and sooner or later runs up against computational resources.

### Grid Search — trying over a grid

Several values are fixed for each hyperparameter, and all possible combinations are tried in sequence:
on each combination a model is trained and tested, and the model with the best quality is chosen. For
example, if for an SVM 3 hyperparameters are tried with 3, 2, and 4 value options respectively, you get
3 × 2 × 4 = 24 models.

`GridSearchCV` in scikit-learn takes a model, a dictionary of hyperparameters and their values, the
number of cross-validation folds (`cv`), and a metric (`scoring`). After `fit`, the following are
available:

- **best_params_** — the best found set of hyperparameters;
- **best_estimator_** — the best trained model itself (it can be used right away for `predict`, without
  reassembling anything by hand);
- **cv_results_** — the results for each combination of hyperparameters (convenient to wrap in a
  `pandas.DataFrame`); the table has a `rank_test_score` column — the model's rank by quality (1 — the
  best; when several models have the same result, ranks may coincide, and the next model gets a rank
  with numbers skipped).

The method's downside: if there are too many value combinations, the search will drag on for an
unreasonably long time.

### Random Search — random search

For each hyperparameter a distribution is specified, from which values are chosen by sampling; only the
selected random combinations are tried (rather than all possible ones, as in Grid Search). The
advantage: with a sufficiently large number of tried combinations, Random Search can, over the same
number of iterations as Grid Search, consider more diverse values — that is, it optimizes the search
without the risk of guaranteed missing a good combination somewhere at the seam of the grid.

An example of the scale of the problem: for a random forest the total number of possible hyperparameter
combinations can reach ~65,000 — a full grid search in that case is unrealistic time-wise, and
`RandomizedSearchCV` (for example, with 30 random combinations instead of all) becomes a practical
alternative.

A practical two-stage scheme: first run `RandomizedSearchCV` over a wide range of hyperparameters to
roughly understand the region of the hyperparameter space with the best quality, and then investigate
the neighborhood of the found values in a targeted way, via a full search (`GridSearchCV`) — a rough
random estimate, then an exact search around the promising region.

### Successive Halving (SH)

All hyperparameter combinations are evaluated with a small amount of resources at the first iteration
(for example, on a small sub-sample of the data or with a small number of training iterations). Only
the combinations that showed the best results proceed to the second iteration, with increased
resources. The process is repeated until the best combination is found. In scikit-learn it's implemented
both over a full grid (`HalvingGridSearchCV`) and with sampling (`HalvingRandomSearchCV`).

### Bayesian optimization and other methods

**Bayesian optimization** — instead of trying and training all hyperparameter combinations in a row, it
estimates the probability that a given combination of hyperparameters will turn out to be a good one,
and thus predicts promising combinations even before training. Bayesian optimization isn't implemented
in scikit-learn — for it separate libraries are used, the most widespread in practice being **Optuna**;
there are others, for example **Hyperopt**, and large companies often use their own internal
implementations.

Other methods worth knowing by name (without needing to memorize the details at this level of the
course): **Tree-structured Parzen Estimator (TPE)**, **Population Based Training (PBT)**, **Hyperband**,
as well as genetic algorithms (which imitate the evolutionary process to search for good hyperparameter
combinations) — a narrowly specialized direction.

In simple cases, Grid Search and Random Search are most often used in practice; Bayesian optimization
(Optuna) is applied when there are many hyperparameters and training models takes a long time.

**The main libraries for hyperparameter tuning:** scikit-learn, Hyperopt, Optuna.

## Conclusions

Cross-validation is needed for an honest estimate of a model's generalization ability (although in
practice it's sometimes used to choose a model too). For hyperparameter tuning there are several
families of methods: full grid search (Grid Search), random search (Random Search), Successive Halving,
and Bayesian optimization (for example, via Optuna). Separately, it's important to be able to assess the
dependence of a model's quality on the amount of data (`learning_curve`) — this helps understand at what
volume further growth in quality stops, which has practical significance when making decisions about the
amount of data labeling.

For more on regularization (L1/L2/ElasticNet) and overfitting/underfitting — see the topic
`linear-regression-gd`.
