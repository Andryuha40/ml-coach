# Linear methods of classification and quality metrics

These notes were assembled from a lecture in Evgeny Patochenko's repository
(`lesson_05` — the basis; it combines exactly the same material as the course's
local lectures "09. Linear methods of classification" and "10. Classification
quality metrics" — it is a practically word-for-word duplicate of those same two
PDFs, slide for slide, with no unique additions, so it is not retold separately),
the `lesson_05` repository seminar "Binary classification, logistic regression"
(classifying irises and wine — code and building the ROC curve by hand), and the
`lesson_05` repository seminar "Classification (predicting a customer's response
to a marketing campaign)" (an end-to-end example: EDA → preprocessing → logistic
regression → metrics on a realistic dataset).

## Classification as a machine learning task

**Classification** is a task in which the target variable is categorical
(discrete): you need to predict not a number but the class/label of an object.
It can be:
- **binary** — two classes, Y = {0, 1} (or {−1, +1});
- **multiclass with non-overlapping classes** — an object belongs to exactly one
  of M classes, Y = {1, ..., M};
- **multiclass with overlapping classes (multi-label)** — an object can be
  assigned several labels at once simultaneously.

What follows is primarily about binary classification — it's simpler to analyze,
and multiclass is usually reduced to a combination of binary tasks (for example,
via the one-vs-rest strategy).

## Linear model of binary classification

The simplest linear model for binary classification (with classes encoded as −1
and +1) predicts the class through the sign of a linear combination of features:

f(x, β) = sign(sum of βᵢ·xᵢ over all features i)

- If the sum of βᵢ·xᵢ is greater than zero — the sign is positive, the object is
  assigned to the positive class (+1).
- If the sum is less than zero — the sign is negative, the object is assigned to
  the negative class (−1).
- The equation "sum of βᵢ·xᵢ = 0" defines the **decision boundary**: in the
  two-dimensional case this is a line, in the multidimensional case a
  hyperplane.

### Training: minimizing the error rate

When training a linear classifier, we would naturally like to directly minimize
the fraction of incorrect answers — a function that, for each object, equals 1 if
the prediction is wrong and 0 otherwise, averaged over the entire sample.

### Margin

A convenient concept is introduced — the **margin** of an object: Mᵢ = yᵢ · (β,
xᵢ), that is, the product of the object's true class and the value of the linear
combination before applying the sign. The margin shows the classifier's degree
of confidence in its answer:
- a large positive margin — the object is far from the decision boundary **and**
  classified correctly (the algorithm is "confident" and doesn't err) — such
  objects are called **reliable**;
- a margin close to zero — the object is near the boundary, the classifier is
  not very confident — **borderline** objects;
- a negative margin — the object is classified incorrectly — **noise** objects
  (or genuine outliers/anomalies).

The task of minimizing the error rate is equivalent to minimizing the fraction
of objects with a negative margin (Mᵢ < 0).

### The problem of the threshold loss function and smooth approximations

Such a loss function (the fraction of objects with a negative margin) is called a
**threshold** function — it is discontinuous (jumps at zero), which makes it
extremely inconvenient to optimize with gradient methods (the derivative is
almost everywhere zero, and at the discontinuity it is undefined).

The solution is to replace the threshold function with a smooth **upper bound**:
a continuous, differentiable function that is always greater than or equal to the
threshold function in value — then minimizing this smooth function automatically
"pulls down" the original threshold one as well. The specific choice of such a
function depends on the task and determines the specific algorithm:
- V(M) = 1 − M — piecewise linear (used, for example, in SVM with hinge loss);
- H(M) = −M — piecewise linear;
- L(M) = log(1 + e^(−M)) — **logistic** (used in logistic regression);
- Q(M) = (1 − M)² — quadratic;
- S(M) = 2 / (1 + e^M) — sigmoidal;
- E(M) = e^(−M) — exponential (used, in particular, in AdaBoost).

## Logistic regression

### Definition and the sigmoid

Logistic regression is a linear classifier that predicts not just a class but the
**probability** of belonging to a class: f(x, β) is an estimate of the
probability P(y = +1 | x; β). This probability is modeled through the **sigmoid**
applied to the linear combination of features:

f(x, β) = σ(βᵀx), where σ(z) = 1 / (1 + e^(−z))

The sigmoid is a function that "squeezes" any real number z into the interval
from 0 to 1, which is why it is convenient to interpret it as a probability.

### Why the sigmoid specifically

- The range of σ(z) is exactly the segment [0, 1], which naturally corresponds to
  the definition of a probability.
- Interpretable symmetry: the probability of the opposite event is σ(−z) = 1 −
  σ(z).
- The sigmoid has an inverse function (the ability to recover z from a
  probability s): z(s) = ln(s / (1 − s)) — this is the so-called logit, widely
  used in statistics.
- The sigmoid differentiates conveniently: its derivative is expressed through
  the function itself — σ′(z) = σ(z)·(1 − σ(z)) — which greatly simplifies
  computing the gradient during training.

### Why not the quadratic loss function

If you take the ordinary quadratic loss function for logistic regression (compare
the predicted probability with the 0/1 answer and square it), two problems arise:
- the resulting function is **not convex** in the parameters β — during
  optimization you can get stuck in a local minimum without reaching the global
  one;
- for completely wrong predictions the penalty turns out too mild: if the model
  confidently (with a probability around 0%) predicted a positive object as
  negative, the penalty is (1 − 0)² = 1 — that is, finite and not "punishing" the
  model in proportion to the crudeness of the error.

### The logistic loss function (log-loss)

Instead of the quadratic one, the logistic loss function is used:

logloss = −(1/n) × sum over all objects [yᵢ·log(f(xᵢ, β)) + (1 − yᵢ)·log(1 − f(xᵢ, β))]

The key difference from the quadratic function: if the algorithm confidently and
correctly predicted the class (f → 1 when y = 1), the penalty tends to 0; but if
the algorithm confidently and **incorrectly** predicted it (f → 0 when y = 1),
the penalty tends to **plus infinity** — that is, for a confident error the model
is punished disproportionately hard, which is exactly what motivates it to be
careful in its probabilistic predictions.

### The constant solution of log-loss (intuition)

If the model could output only a single constant a for all objects (without
regard to features), then with n₁ objects of the positive class and n₂ objects of
the negative class, the minimum of log-loss is reached exactly at a = n₁ / n —
that is, **the optimal constant is exactly equal to the fraction of positive
objects in the sample**. This is an intuitive reference point: even the most
primitive model without features should output a probability equal to the class
frequency in the data, and any meaningful model should show a log-loss better
than this constant baseline.

## Classification quality metrics

### Accuracy

**Accuracy** is the fraction of correct answers of the classifier:

accuracy = (number of correctly classified objects) / (total number of objects)

**The main problem with accuracy is class imbalance.** Example: diagnosing a rare
disease that occurs in 3 people out of 1000 (0.3%). A constant classifier that
always predicts "healthy" will achieve accuracy = 0.997 — formally an impressive
result, but such a model is completely useless (it never finds a single sick
person). Under strong class imbalance, accuracy stops reflecting the real quality
of the model — other metrics are needed.

### Confusion matrix

For binary classification, all objects are divided into 4 groups based on the
prediction result:

| | predicted positive | predicted negative |
|---|---|---|
| **actually positive** | TP (True Positive) | FN (False Negative) |
| **actually negative** | FP (False Positive) | TN (True Negative) |

- **Type I error (False Positive)** — the true class is negative, but positive
  was predicted (false alarm).
- **Type II error (False Negative)** — the true class is positive, but negative
  was predicted (missed target).

Which error is more critical depends on the task. For example, for disease
diagnosis FN (failing to notice a sick patient) is usually far more dangerous
than FP (sending a healthy person for additional testing); whereas for a spam
filter, conversely, FP (an important email in the spam folder) may be more
unacceptable than FN (a little spam in the inbox).

### Precision

precision = TP / (TP + FP)

Shows **how much you can trust** the classifier when it has produced a positive
answer: what fraction of the objects labeled positive by the model are actually
positive.

### Recall

recall = TP / (TP + FN)

Shows **how many objects of the positive class the classifier finds** out of all
those that actually exist: what fraction of the truly positive objects was
correctly found.

**An example comparing two models**: model 1 has precision = 0.8, recall = 0.8;
model 2 has precision = 0.96, recall = 0.48. Model 2 is much more precise in its
positive predictions, but finds a much smaller fraction of the actual positive
objects — which model is "better" depends on what matters more in the specific
task: not missing positive objects (recall) or not creating false alarms
(precision).

### F1 measure

A metric combining precision and recall into a single number — the harmonic mean:

F = 2 × precision × recall / (precision + recall)

The harmonic mean (rather than the ordinary arithmetic one) is chosen
deliberately: it more strongly "penalizes" the situation where one of the two
quantities is low, even if the other is high — that is, a high F1 requires both
metrics, precision and recall, to be sufficiently good simultaneously.

### The classification threshold and adjusting it

In its "soft" form the classifier outputs not a class right away but a certain
confidence p(x) — the probability that object x belongs to the positive class. By
default, if p(x) > 0.5, the object is assigned to the positive class, otherwise to
the negative one. But this threshold t need not be 0.5, and can be varied in the
range from 0 to 1 depending on the task:
- **raising the threshold** t — the model becomes more "cautious" in positive
  predictions: precision usually rises, while recall falls (the model finds fewer
  positive objects, but with almost no false alarms);
- **lowering the threshold** t — conversely, recall rises, but precision usually
  falls.

Choosing the specific threshold is a business/domain decision, depending on which
of the two errors (FP or FN) is more costly in the specific task.

## ROC-AUC

**ROC-AUC** is a metric that measures the quality of the **entire family** of
classifiers (that is, the model at all possible values of the threshold t) with a
single number, without being tied to a specific chosen threshold.

Additional notation (equivalent to some of the quantities above, but
traditionally used specifically in the context of the ROC curve):
- **TPR (True Positive Rate)** = TP / (TP + FN) — the same thing as recall (the
  fraction of correctly found positive objects);
- **FPR (False Positive Rate)** = FP / (FP + TN) — the fraction of negative
  objects mistakenly classified as positive.

The **ROC curve** is a curve made up of points with coordinates (FPR, TPR)
obtained by sweeping over all possible values of the threshold t. **AUC (Area
Under Curve)** is the area under this curve, a number in the segment [0, 1]:
- AUC = 1 — perfect classification (there exists a threshold at which TPR = 1 and
  FPR = 0 simultaneously — all positives found, not a single false alarm);
- AUC = 0.5 — the quality of a random classifier (guessing at random); on the
  graph this corresponds to the diagonal line from (0,0) to (1,1).

### Geometric construction of the ROC curve by hand

An intuitive way to build the ROC curve by hand: order all objects by decreasing
predicted probability/confidence of the classifier. Then divide the unit square
into m × n parts, where m is the number of positive-class objects and n is the
number of negative-class objects. Start from the point (0, 0) and move along the
list of ordered objects: if the next object is truly positive — take a step **up**
(by 1/m); if truly negative — a step **to the right** (by 1/n). If several
consecutive objects have the same predicted probability but different true
classes — the step is taken "diagonally" (simultaneously up and to the right, in
proportion to the number of objects of each class among the "tied" ones). The
route inevitably ends at the point (1, 1) — and the resulting broken line is the
ROC curve.

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, label="ROC")
print(f'The area under the ROC curve is {auc(fpr, tpr)}')
```

### Probabilistic interpretation of AUC

The area under the ROC curve equals the **fraction of pairs of observations** (one
object of the truly positive class, the other of the truly negative) that the
classifier ordered correctly by its predicted probability (that is, assigned the
positive object in the pair a higher probability than the negative one). This is a
convenient and intuitive alternative reading of AUC — as the probability that a
randomly chosen positive object receives a higher score from the model than a
randomly chosen negative one.

## AUC-PR (area under the precision-recall curve)

Under strong class imbalance (a small fraction of positive-class objects), ROC-AUC
can give a deceptively good result — because FPR is computed relative to a large
number of negative objects, and even a noticeable number of false triggers barely
changes FPR. Therefore, for tasks with imbalanced classes it is preferable to use
the **precision-recall curve** (a curve in recall–precision coordinates as the
threshold is swept) and the area under it — **AUC-PR**:

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_score)
pr_auc = average_precision_score(y_test, y_score)
```

## Practical code

### Classifying irises: a basic logistic regression example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = load_iris()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = data['target']

# reduce the multiclass task to a binary one: versicolor (class 1) vs. the rest
y[y != 1] = -1

X = X[['sepal length (cm)', 'sepal width (cm)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# IMPORTANT: fit_transform on train, transform (without re-fitting) on test — otherwise data leakage
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.coef_)

print('F1 score:', f1_score(y_test, lr.predict(X_test)))
```

### Effect of the regularization coefficient on the decision boundary

In sklearn, `LogisticRegression`'s regularization is set by the parameter `C`,
which is the **inverse** of the regularization coefficient (C = 1/α): the smaller
C, the stronger the regularization.

```python
C = [0.01, 0.05, 10]
lr1 = LogisticRegression(C=C[0])
lr2 = LogisticRegression(C=C[1])
lr3 = LogisticRegression(C=C[2])
# with small C the decision boundary becomes simpler (stronger regularization,
# the model is less eager to adapt to specific points of the training set);
# with large C the boundary fits the data more closely, but the risk of overfitting is higher
```

### Tuning the regularization hyperparameter via cross-validation

```python
from sklearn.model_selection import cross_validate

scores_lr = []
for c in np.arange(0.1, 10, 1):
    lr = LogisticRegression(C=c)
    cv_lr = cross_validate(lr, X_train, y_train, cv=5, scoring='f1')['test_score']
    scores_lr.append(cv_lr.mean())

best_c = np.arange(0.1, 10, 1)[np.argmax(scores_lr)]
lr = LogisticRegression(C=best_c)
lr.fit(X_train, y_train)
```

### End-to-end example: EDA → preprocessing → metrics (marketing campaign)

Fragments from the seminar "predicting a customer's response to a marketing
campaign" — an illustration of the full cycle on realistic "dirty" data:

```python
# discovering "junk" categories in the data during EDA
df[(df['Marital_Status'] == 'Absurd') | (df['Marital_Status'] == 'YOLO')]
df['Marital_Status'] = np.where(
    df['Marital_Status'].isin(['Absurd', 'YOLO', 'Alone']), 'Single', df['Marital_Status']
)

# One-Hot Encoding of categorical features
data = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

# cleaning missing values and outliers in a specific feature
data.dropna(subset=['Income'], inplace=True)
data = data[(data['Income'] <= data['Income'].quantile(0.995))]

# engineering features from the date and year of birth
data['days'] = (datetime.datetime(2015, 1, 1) - pd.to_datetime(data['Dt_Customer'])).dt.days
data['Age'] = 2015 - data['Year_Birth']

# training and the full set of metrics, including ROC-AUC
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve

lr = LogisticRegression(solver='liblinear')
lr.fit(Xtrain, ytrain)
pred_test = lr.predict(Xtest)
pred_test_prob = lr.predict_proba(Xtest)

print('Accuracy:', round(accuracy_score(ytest, pred_test), 3))
print('Recall:', round(recall_score(ytest, pred_test), 3))
print('Precision:', round(precision_score(ytest, pred_test), 3))
print('ROC AUC:', round(roc_auc_score(ytest, pred_test_prob[:, 1]), 3))
```

**An important technique for evaluating a model — comparison with naive
baselines**:

```python
# naive model "no one will respond"
pred_test_0 = np.zeros(Xtest.shape[0])
print('Accuracy:', round(accuracy_score(ytest, pred_test_0), 3))  # high accuracy under imbalance, but recall = 0

# naive model "the response to the new campaign will be like the response to campaign 2"
pred_test_2 = Xtest['AcceptedCmp2']
```

Such a comparison nicely illustrates the accuracy trap under class imbalance in
practice: a naive constant classifier can show deceptively high accuracy but zero
recall — that is, be completely useless for the task (similar to the rare-disease
diagnosis example above).

## In brief: how to choose a metric

- **Accuracy** — suitable only when the classes are roughly balanced and when
  errors of both types "cost" the same.
- **Precision** — matters when the cost of a false trigger (FP) is critical — for
  example, if a positive prediction launches an expensive action.
- **Recall** — matters when the cost of missing a positive object (FN) is critical
  — for example, in medical diagnosis or fraud detection.
- **F1** — a compromise when both precision and recall matter simultaneously and a
  single number for comparing models is enough.
- **ROC-AUC** — good when you need to compare models independently of the choice of
  a specific threshold, with relatively balanced classes.
- **AUC-PR** — preferable to ROC-AUC under strong class imbalance (more on methods
  for dealing with imbalance — in the topic `class-imbalance`).
