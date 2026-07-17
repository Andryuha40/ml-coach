# Linear methods of regression, gradient descent, regularization

These notes were assembled from lectures in Evgeny Patochenko's repository
(`lesson_02`, `lesson_03`, `lesson_04` — the basis) and from course seminars
("Training linear regression", "Gradient descent and its variations" — examples,
code, unique details such as an explicit definition of underfitting).

## Linear model

Linear regression is the basic model for predicting a continuous target variable
from one or several features. General form:

`y = β0 + β1·x1 + β2·x2 + ... + βn·xn + ε`

where y is the target variable, x1...xn are the features, β0 is the intercept,
β1...βn are the coefficients (weights) of the model, and ε is the model error
(noise). In shorthand this is written as the dot product of the weight vector and
the feature vector (with an added one for the intercept): `y(x) = (β, x)`.

Training linear regression means minimizing the MSE (mean squared error) over the
weights β: we take the average over all objects in the sample of the squared
difference between the model's prediction and the true answer, and we choose the β
for which this average is minimal.

### The closed-form solution (OLS) and its drawbacks

For linear regression with MSE there is an exact formula for the optimal weights
(the method of least squares) — it is expressed through matrix operations on the
feature matrix and the vector of answers. Drawbacks of such a closed-form
solution:
- matrix inversion is an expensive operation, cubic in the number of features;
- the matrix `XᵀX` can be singular or ill-conditioned (this happens, in
  particular, under strong correlation between features — multicollinearity, see
  below on regularization);
- if the error functional is not MSE but something else — a closed-form solution
  most often cannot be found at all.

Therefore, in practice the weights are usually chosen iteratively — with gradient
descent (see below) — rather than directly by formula.

### Why MSE specifically? (a probabilistic justification)

Even if the target variable in theory depends linearly on the features, no ideal
model exists — the real answers will differ slightly from the predictions due to
noise in the data: `y ≈ (β, x) + ε`. If we assume the noise ε is normally
distributed (which is a reasonable assumption for many real tasks), then
maximizing the likelihood of the data under this assumption is mathematically
equivalent to minimizing precisely the mean squared error — hence the popularity
of MSE as the default loss function for regression.

## Loss functions and quality metrics for regression

An important distinction: the **loss function** is what is minimized during
training (fitting the weights); the **quality metric** is what is used to measure
the quality of an already-trained model on new data. The same function can often
be used as both.

The implementations below (seminar code, the reference for checking code
exercises):

```python
def MSE(y: np.array, y_pred: np.array) -> np.float64:
    return ((y - y_pred) ** 2).mean()

def RMSE(y: np.array, y_pred: np.array) -> np.float64:
    return np.sqrt(MSE(y, y_pred))

def R_squared(y: np.array, y_pred: np.array) -> np.float64:
    std = ((y - np.mean(y)) ** 2).mean()
    return 1 - MSE(y, y_pred) / std

def MAE(y: np.array, y_pred: np.array) -> np.float64:
    return np.abs(y - y_pred).mean()
```

- **MSE** (mean squared error) — the average of the squared deviations of the
  prediction from the true value. Because of the squaring, it heavily penalizes
  large deviations (meaning it is sensitive to outliers), and its units of
  measurement are the square of the target variable's units (for example,
  "kilograms squared"), which is poorly interpretable by a human.
- **RMSE** — the square root of MSE, which solves the units-of-measurement
  problem (kilograms again), but remains an unbounded quantity, so from a single
  value it is hard to tell whether the model is "good" or not.
- **R² (coefficient of determination)** — a normalized version of MSE: the
  fraction of the target variable's variance explained by the model. R² close to
  1 — the model explains the data well; R² close to 0 — the model is no better
  than a constant prediction (for example, the mean); R² negative — the model is
  worse than simply predicting the sample mean. It can be difficult to interpret
  for a non-technical stakeholder.
- **MAE** (mean absolute error) — the average of the absolute deviations. Unlike
  MSE, it does not square the error, so it is less affected by outliers, but the
  absolute value is not a smooth function, which complicates direct optimization
  with gradient methods (sklearn has no separate "linear regression by MAE"
  class — for this, quantile regression with the parameter q=0.5 from the
  `statsmodels` library is used).

More specialized loss functions worth knowing by name (without needing to
memorize the formulas by heart at this level of the course): **Huber Loss** — a
hybrid of MSE and MAE (robust to outliers like MAE, but smooth and lightly
penalizing small deviations like MSE); **MSLE** — penalizes under-predictions
less than over-predictions (suitable only for a non-negative target variable);
**Quantile Loss** — the penalty depends not only on the magnitude of the error
but also on its sign (useful when over- and under-estimation have different
business costs, for example in demand forecasting).

## Overfitting and underfitting

- **Overfitting** — the model's quality on new data (which it did not see during
  training) is substantially worse than on the training set. It arises with a
  too-complex model, too-long training, or an unfortunate (non-representative)
  training set.
- **Underfitting** — the model's error remains large even on the training set. It
  arises with a too-simple model or stopping training too early.

Overfitting is always present when the parameters are fitted on a finite
(necessarily incomplete) sample — the question is its degree. With a small number
of objects there is a higher risk that the model will "memorize" specific
examples instead of learning the general regularity; with excessive model
complexity (too many weights), the extra degrees of freedom get "spent" fitting
the noise of the training set.

## Regularization

**Why it's needed.** Under strong feature correlation (multicollinearity), the
weight-optimization problem can have infinitely many solutions, the weights lose
their physical meaning (to the point of paradox — a feature that logically should
raise the target variable gets a negative weight), and the model becomes
inaccurate and uninterpretable.

**The idea.** We modify the loss function by adding a penalty for weights that are
too large: the final functional = the original loss function + λ × the
regularizer, where λ (lambda) is a hyperparameter reflecting the strength of the
penalty (chosen on a validation set, usually by a sweep on a logarithmic scale,
for example from 0.01 to 100).

- **L2 regularization (Ridge)** — the penalty equals the sum of the squares of the
  weights (the L2 norm of the weight vector). It penalizes large weights, pulling
  them toward zero, but does not zero them out entirely. It reduces the influence
  of noise in the data. In sklearn — the `Ridge` class.
- **L1 regularization (Lasso)** — the penalty equals the sum of the absolute
  values of the weights (the L1 norm). It penalizes all weights "equally hard,"
  which is why the least important weights are zeroed out entirely — that is,
  Lasso performs feature selection along the way. In sklearn — the `Lasso` class.
- **ElasticNet** — a combination of L1 and L2 (the sum of both penalties with two
  separate λ's). It combines the advantages of both: it both selects features and
  smooths the weights. In sklearn — the `ElasticNet` class.

Advantages of regularization: stability of the weights under multicollinearity,
better generalization ability, reduced risk of overfitting. Drawback: choosing the
optimal regularization strength λ requires cross-validation, which is
computationally costly.

### Multicollinearity in practice (VIF)

A sign of multicollinearity in a trained model is weights that are inadequate in
sign or magnitude. One tool for a quantitative check is **VIF (variance inflation
factor)**: the higher a feature's VIF, the more strongly it is linearly explained
by the other features (a rule of thumb: VIF > 5-10 is an alarming signal). An
example from the seminar (`statsmodels`):

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
```

Under multicollinearity, the standard error of the coefficients in
`OLS.summary()` (from `statsmodels`) also grows noticeably — the model may have a
high R² (explains the data well overall), yet for individual features the p-value
becomes large, meaning one cannot confidently say which features exactly are
significant.

## Gradient descent

### The optimization problem in general form

Training is the fitting of the model's weights to minimize the loss function, so
the machine learning task reduces to an optimization task: among all possible
weight values, find the one at which the objective function (for example, MSE) is
minimal. For linear regression with MSE this function is convex — that is, its
local and global minima coincide (this is not always true for more complex
models, such as neural networks).

### The gradient and the anti-gradient

The **gradient** of a function at a point is a vector pointing in the direction in
which the function grows fastest. For a function of several variables this is a
vector of all the partial derivatives with respect to each variable; for a
function of one variable it is simply the derivative. The **anti-gradient** is a
vector in the opposite direction: toward where the function decreases fastest. It
is precisely along the anti-gradient that gradient descent moves (although in the
general case this direction does not point exactly at the minimum itself).

### The algorithm and a manual implementation

The general idea (the reference seminar code for checking code exercises):

```python
def f(x):
    return x**2

def grad_f(x):
    return 2 * x

def gradient_descent(x0, alpha, num_iters):
    x = x0
    for i in range(1, num_iters + 1):
        grad = grad_f(x)
        x_new = x - alpha * grad
        x = x_new
    return x

gradient_descent(x0=5.0, alpha=0.1, num_iters=5)
```

That is, at each step: we compute the gradient of the function at the current
point, take a step in the direction of the anti-gradient of size `alpha` (the
learning rate), obtain a new point, and so on until a stopping criterion is met
(for example, a number of steps, or the change in the function/weights between
steps has become smaller than a given threshold). When training a model, the
"point" is not a single number but a vector of weights, and the update formula is
written the same way, just for the weight vector: new weights = old weights −
learning rate × the gradient of the loss function with respect to the weights.

**Initializing the weights**: with zeros, with small random values, by training on
a small random subsample, or by multi-start (several runs from different random
points, choosing the best result).

**Learning rate (α)** — a hyperparameter that determines the step size. A small α
— accurate convergence, but slow; a large α — faster, but you can "overshoot" the
minimum or fail to converge at all.

**Drawbacks of ordinary (full/batch) gradient descent**: at each step you need to
compute the gradient over all objects in the sample at once — this is expensive in
time and memory; the method can also get stuck in a local minimum (for non-convex
functions).

### Modifications of gradient descent

- **SGD (stochastic gradient descent)** — at each step the gradient is computed on
  a single random object rather than the whole sample. It is cheaper on resources
  but may require more steps to converge (a "noisier" trajectory).
- **Mini-batch SGD** — a compromise: the gradient is computed on a random
  subsample of fixed size (batch size, for example, 16, 32, 64). A practical
  example in PyTorch (the structure of the standard training loop — useful for the
  neural-network topics too):

```python
model_sgd = nn.Linear(1, 1)
criterion = nn.MSELoss()
sgd_optimizer = torch.optim.SGD(model_sgd.parameters(), lr=0.1)

for epoch in range(10):
    epoch_loss = 0
    for X_batch, y_batch in dataloader:
        sgd_optimizer.zero_grad()      # zero out gradients from the previous step
        y_pred = model_sgd(X_batch)    # forward pass — prediction
        loss = criterion(y_pred, y_batch)
        loss.backward()                # compute gradients (backprop)
        sgd_optimizer.step()           # update the weights
        epoch_loss += loss.item()
```

  Five steps per batch: zero out the gradients → forward pass → compute the loss →
  `backward()` (compute the gradients) → `step()` (update the weights).

- **Momentum** — adds "inertia": the next step takes into account not only the
  current gradient but also the accumulated "velocity" from previous steps
  (analogy — a ball rolling down a slope does not stop instantly on a flat
  stretch). It reduces the "jitter" of the trajectory and helps jump over small
  local minima.
- **Nesterov Momentum (NAG)** — a modification of Momentum: the gradient is
  computed not at the current point but at the point where inertia would carry us
  anyway — "looking ahead." It is usually a bit faster and more stable than
  ordinary Momentum.
- **Adagrad** — adapts the learning rate individually for each weight based on the
  sum of the squares of all its past gradients: the more a weight has already been
  updated, the smaller its personal learning rate becomes. It works well with
  sparse data (NLP, recommender systems), but the drawback is that the learning
  rate decreases monotonically and can "freeze."
- **RMSProp** — solves Adagrad's decay problem: instead of the sum of all past
  squared gradients, it uses an exponential moving average (old gradients are
  gradually "forgotten").
- **Adam** — the most popular default optimizer in deep learning: it combines
  Momentum (smoothing the gradient itself) and RMSProp (an adaptive learning rate
  per coordinate), plus a bias correction on the first steps. It works well out of
  the box in almost any task; when using it, it is important to choose α — people
  often start with 0.0003 (3e-4).

### Learning rate schedules (schedulers)

It is often more advantageous not to keep the learning rate constant, but to lower
it over the course of training (at the start — a large step for quick progress,
closer to the end — a small one, so as not to "overshoot" the minimum):
- **StepLR** — multiplies the learning rate by a coefficient every N epochs (in
  steps).
- **ReduceLROnPlateau** — a reactive approach: it watches a metric (for example,
  the validation loss) and reduces the learning rate if there have been no
  improvements over a given number of epochs (`patience`).
- **Warm-up** — over the first few steps the learning rate instead grows from zero
  to the target value, and only then decreases. It is needed because at the very
  start of training the weights are random and the gradients are large and noisy —
  a step that is too large right away could "blow up" the weights.

## Stopping criteria for gradient descent

- stopping after a predefined number of steps;
- stopping when the change in the loss function between steps is below a threshold;
- stopping when the change in the weights between steps is below a threshold.

## Data preprocessing for linear regression

Briefly (detailed EDA — in the topic `eda-and-pipeline`): for linear models
preprocessing is especially important, since it affects not only quality but also
the interpretability of the weights.

- **Missing values** — fill with a statistic (mean/median), a constant, or predict
  from the other features; for linear models, filling with a "special value"
  (unlike trees/boosting) is usually not appropriate.
- **Scaling** — bringing features to a single scale is important for training
  speed, numerical stability, and interpreting the weights as a measure of a
  feature's significance. Standardization (`StandardScaler`: subtract the mean,
  divide by the standard deviation) — when the data resembles a normal
  distribution. Normalization (`MinMaxScaler`: compress into the range [0, 1]) —
  when there are strong outliers or different ranges across groups of values.
- **Nonlinear feature transformations** (polynomial features, logarithm, square
  root, trigonometric functions) — allow linear regression to model nonlinear
  dependencies, if exploratory analysis suggests a nonlinear form of the relation
  between a feature and the target variable (for example, `plt.scatter(x, y)`
  shows a parabola — it's worth trying `x**2` as an additional feature).

## When linear regression works well (and when it doesn't)

It works well if: the relationship between the features and the target variable is
genuinely linear, the data is clean (without strong outliers), and the
statistical assumptions of the method are met. If the assumptions are violated —
the model may give incorrect predictions; then it is worth considering polynomial
regression, regularization, or moving on to decision trees and ensembles (the next
topics of the course).

Advantages: simplicity of implementation, fast training, interpretability
(coefficients that make sense). Disadvantages: sensitivity to outliers, poor
handling of nonlinear dependencies out of the box, a requirement that the
statistical assumptions be met.
