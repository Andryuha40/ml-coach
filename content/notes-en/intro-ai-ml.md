# Introduction to AI and machine learning

These notes were assembled from: a lecture in Evgeny Patochenko's repository (`lesson_01`,
the basis), a local lecture "Artificial intelligence" (a unique historical
part) and a seminar "Introduction to machine learning" (the formal statement of the
task, an example on Fisher's Irises, sklearn code).

## A brief history of AI

- **1947** — Alan Turing puts forward the idea of "intelligent machines"
  that change their internal state based on experience; in 1950 he proposes the
  Turing test for assessing a computer's intelligence.
- **1956** — at the Dartmouth conference John McCarthy introduces the term
  "artificial intelligence".
- **1960** — Rosenblatt's perceptron — the first working algorithm
  that models the operation of a brain neuron.
- **1966** — the conversational system Eliza imitates a conversation with a psychotherapist,
  substituting the patient's significant words into template phrases (one of the first
  natural language processing systems).
- **1970s-1980s** — the flourishing of rule-based expert systems (for example,
  MYCIN for diagnosing bacterial infections); in 1984 — the first algorithm
  for automatically building a decision tree.
- **1997** — Deep Blue (IBM) beats Garry Kasparov at chess.
- **2002** — Roomba, the first household robot vacuum cleaner.
- **2016** — neural machine translation significantly surpasses in quality
  the previous rule-based approaches.
- **2017** — the paper *Attention Is All You Need* introduces the attention mechanism,
  which became the basis of almost all modern SOTA solutions.
- **2022-2023** — the explosive growth of generative models and large language
  models (LLMs), for example ChatGPT.

Three waves worth distinguishing by their time of appearance: machine learning
(from the early 1990s) → deep learning / neural networks (from the mid-2000s) →
generative AI and LLMs (from 2022).

## What machine learning is

Three equivalent definitions that appear in the course:
- the science of extracting patterns from a limited number of examples
  (E. Sokolov);
- the science of algorithms that automatically improve through experience
  (Yandex);
- the process that gives computers the ability to learn to do something without
  explicitly writing code for a specific task (A. Samuel).

The common idea of all three: instead of manually spelling out rules
(`if... then...`), we show the algorithm examples and it finds the
pattern on its own.

## Object, features, training sample

To apply an ML algorithm to anything (a car, a text, a patient), that
something first needs to be **formalized** — represented by a set of numbers,
which is called a **feature description**.

- **Object** — the entity for which we want to make a prediction (a customer,
  an apartment, a photograph...).
- **Feature** — one coordinate of the feature description, a specific
  measurable characteristic of the object.
- **Training sample** — a set of objects for which the correct answer is
  already known.
- **Target variable (answer)** — what we want to predict.
- **Model (algorithm)** — a function that, from the feature description of an
  object, produces a prediction.

An example — the scoring task: from a customer's characteristics (gender, age, income,
credit history...) predict whether they will repay the loan. Here: objects are
the customers from the dataset, features are their characteristics, the target variable is
the number 1 (will repay) or 0 (will not repay).

An example of a feature description done by hand: take a car. It can be described
by color, engine power, presence of an audio system, top speed,
mass, make. Non-numeric characteristics (color, make) are converted into numbers
by numbering the possible values: for example, the colors "green/red/blue"
are encoded as 0/1/2, the presence of an audio system — as 0/1. Then a specific
car turns into a vector of numbers of the form [1, 100, 1, 150, 800, 0] —
it is exactly such a vector that is fed into the algorithm.

### Types of features

- **Binary** — exactly two values (usually 0/1 or -1/1). Example: presence of
  an audio system, gender, "bought / did not buy".
- **Categorical** — a finite set of values; for the model they need to be
  encoded with numbers. They are divided into:
  - **nominal** — the values cannot be compared ("the color blue is greater than
    red" is meaningless) — for example, the make of a car, color;
  - **ordinal** — the values are ordered ("a major is senior to a sergeant" makes
    sense) — for example, military rank, an exam grade.
- **Continuous** — a number from a continuous range (mass, price, height).
  They can be integer or real.
- **Geographic** — latitude/longitude pairs, requiring specific
  processing (logistics tasks).
- **Date and time** — a special format (`datetime` in pandas), important for
  time series and time-based indexing.

## The model as a family of functions

Our goal is to find an algorithm that is very similar to the pattern hidden from us
(the true dependence between features and the answer). We ourselves
decide from which class of functions to choose this algorithm — this is what is called
**model selection**.

A model is not one specific function, but a whole **family of functions**
depending on parameters. Training is the tuning of specific values
of the parameters within this family.

Example: suppose we suspect that the true dependence resembles a parabola.
Then we choose a model — the family of all second-degree polynomials of the form
`A(x) = a·x² + b·x + c`, where a, b, c are parameters. Training is the tuning of
specific a, b, c at which this parabola best describes the data.
When the parameters are tuned, we get a specific function, which is our
final algorithm.

## The loss function and the ML task as an optimization task

The course's formal notation (useful in other topics too):
- N — the number of objects (rows) in the sample; M — the number of features (columns).
- x — one object (more precisely, its feature description, a vector of numbers);
  X — the whole set of objects (a matrix).
- y — the correct answer for an object (or a vector of answers for a set of objects).
- a(x) — the result of applying algorithm a to object x, that is, the prediction.

The **loss function** is a function that, from an object x and the correct answer y,
shows how badly the algorithm erred on this one object. It is often
denoted L(x, y) (a fuller notation is L with the index of algorithm a).
For example, for a regression task (predicting the price of a car from its
features) the loss function can simply be the absolute value of the difference between
the prediction and the true value: "by how much in absolute terms the predicted
price differs from the real one".

The reasoning further is this: we want not just a small error on a
specific object, but a small error **on average over the whole training
sample** — that is, for the model to work well not on one example, but
in general. Therefore the task is formulated as follows: compute the average value of the
loss function over all N objects of the training sample and tune the parameters
of the model so that this average is as small as possible. This is called
the **empirical risk**, and its minimization over the parameters is the same as
training the model.

The result is the key idea of the course: **the machine learning task reduces to
an optimization task** — minimizing the average loss function over the parameters
of the model. Solving the optimization task itself (how exactly to find the minimum) is
a separate field (see the topic on gradient descent).

## Types of machine learning tasks

- **Regression** — the target variable is continuous (a number). Examples:
  the price of an apartment, a restaurant's profit, a graduate's salary.
  The general form of a linear model: prediction = intercept + the sum of
  (coefficient × feature) over all features, plus the error. The coefficients are
  exactly the parameters of the model that are tuned during training.
- **Classification** — the target variable is categorical (a class/label).
  It can be binary (two classes, for example "sick/healthy"), multiclass
  (the class is one of M variants, for example an animal's breed) or
  multiclass with overlapping classes (an object may belong at once
  to several labels). Examples: medical diagnosis, credit
  scoring, customer churn, a click on a banner.
- **Clustering** — unsupervised learning: you need to split objects into
  groups (clusters) so that objects within a group are similar to one
  another, and those from different groups are distinct. Unlike classification, here
  there are **no** correct answers known in advance and no training sample in
  the usual sense — the classes are not set in advance, the algorithm itself finds
  structure in the data.
- Other types of tasks (briefly, without details at this level of the course):
  ranking, recommendations, dimensionality reduction, anomaly detection,
  time-series forecasting, object detection.

## The process of solving an ML task and exploratory data analysis (EDA) — briefly

A typical pipeline: obtaining the data → exploratory analysis (EDA) → feature
selection/engineering → model selection and training on the training
sample → quality assessment on the test sample → deployment to production.

EDA (Exploratory Data Analysis) is the first substantive step after
obtaining the data: to understand the volume and completeness of the data, the characteristics of the features,
their distributions, the dependencies between features and with the target variable,
to find missing values and outliers. A detailed breakdown of EDA techniques and `sklearn.Pipeline`
is in the topic "Exploratory data analysis, sklearn Pipeline, feature
encoding" (`eda-and-pipeline`); here only the place of EDA in the overall
process matters.

## Training and validation: train/test split

Splitting the data into a **training (train)** and a **test (test)** sample is
one of the central ideas of the course. The point: if you assess a model's quality on
the same data on which it was trained, the assessment will be deceptively
optimistic — the model could have simply adapted to these specific
examples rather than learning the general pattern.

The course's analogy: it is like homework and a test at school. Train is
the homework: it trains, and both a weak and a strong student can more or less solve it
(including by peeking at the answers). Test is the
exam: problems the student has not seen before. Only someone who has really
understood the topic, rather than memorized specific
examples, will solve the exam well. Therefore it is correct to assess a model's quality only on the test.

Sometimes the sample is split not into two but into three parts — a
**validation** sample is added. It is needed for tuning the model's hyperparameters
(and for tracking quality during training in models that take a
long time to learn, for example neural networks) — that is, the validation sample plays the role
of "homework" specifically for the hyperparameters, while the final check is still done
on a separate test. In the course such a three-part split is almost never
encountered outside of neural networks.

### Parameters vs hyperparameters

- **Parameters** of a model are tuned automatically during training (on
  train). Example: the weights in linear regression, the structure of a decision tree.
- **Hyperparameters** are set and fixed in advance, before training, manually
  or through a separate search. Example: the learning rate of gradient descent,
  the strength of regularization, the depth of a tree.

## The sklearn interface (using Fisher's Irises as an example)

Fisher's Irises is a classic teaching dataset: for each of 150 flowers
of three species, 4 continuous features are measured (length/width of the sepal and
petal), and the task is classification by species.

The `sklearn` library sets a single interface for almost all classical
ML algorithms — the ability to use it transfers to linear models,
trees, SVM, ensembles, kNN, etc. (modules `sklearn.linear_model`,
`sklearn.tree`, `sklearn.svm`, `sklearn.ensemble`, `sklearn.cluster`,
`sklearn.neighbors`).

The basic workflow — the code below is reproduced verbatim from the seminar
(the reference for checking code exercises on this topic):

```python
import pandas as pd
import seaborn as sns

import sklearn

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Loading the dataset
X, y = load_iris(return_X_y=True)

# Splitting the sample into training and test
X_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# Training the model
model = LogisticRegression().fit(X_train, y_train)

# Prediction on the test sample
preds = model.predict(x_test)

# Quality assessment — the fraction of correct answers
acc = accuracy_score(y_test, preds)
```

Three key methods common to almost all sklearn models:
- `model.fit(X, y)` — training (tuning parameters) on features `X` and
  answers `y`; for unsupervised learning — `model.fit(X)` without `y`.
- `model.predict(X)` — obtain the model's predictions for new objects
  (the model did not see them during training).
- `accuracy_score(y_true, y_pred)` from `sklearn.metrics` — the fraction of correct
  answers: one of the simplest classification quality metrics (more on
  metrics — in the topic `classification-metrics`).

## Mini-glossary of notation (for reference)

- N — the number of objects, M — the number of features.
- x, X — an object (feature vector) and a matrix of objects respectively.
- y — the correct answer (or a vector of answers).
- a(x) — the prediction of algorithm a on object x.
- L(x, y) — the loss function: the magnitude of the algorithm's error on object x given
  the correct answer y.
- "Empirical risk" — the average value of the loss function over the whole training
  sample; training = minimizing the empirical risk over the model's parameters.
