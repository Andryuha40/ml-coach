# Homework: classification (homework 3, basic)

Source: `content/ml_basics_course/homeworks/homework 3 (Classification)/homework_3_classification_(basic).ipynb`.
The assignment practices comparing several classification algorithms, grid search over hyperparameters,
and assessing the effect of scaling and encoding categorical features — things that aren't in the notes
(the notes are focused on logistic regression and metrics as such).

## Data

A binary classification task: from a set of census features, determine whether a person's average
annual income exceeds the $50,000 threshold (the Adult / Census Income dataset).

Features: `age`, `workclass` (type of employment), `fnlwgt` (the observation's representativeness
weight in the population), `education` (level of education), `education-num` (the numeric equivalent of
the education level), `marital-status`, `occupation`, `relationship` (type of family ties), `race`,
`sex`, `capital-gain` (profit from selling assets), `capital-loss` (loss from selling assets),
`hours-per-week` (working hours per week). The target variable is `>50K,<=50K`.

The quality metric in the assignment is **AUC-ROC**.

```python
import pandas as pd
import numpy as np

PATH = 'https://github.com/evgpat/ml_basics_course/raw/main/datasets/adult_census_income.csv'
data = pd.read_csv(PATH, sep=',')
```

Missing values in the data are marked with a question mark `?` (not `NaN`).

## On hyperparameter tuning (context for the assignment)

The assignment reminds you: a model's parameters are tuned during training (the weights of a linear
model, the structure of a tree), while hyperparameters are set in advance (regularization, tree depth)
and tuned separately — usually via grid search. Three schemes for fairly comparing models when tuning
hyperparameters:
1. train/test — with a large number of tried combinations, risks "overfitting to the test";
2. train/validation/test — validation for comparing models, test for the final estimate;
3. cross-validation (Leave-One-Out, K-Fold, repeated random splitting) — computationally more
   expensive, but more reliable.

Trade-offs under limited time: a sparser grid of values; fewer cross-validation folds (but a noisier
estimate); sequential (greedy) tuning of parameters one after another instead of a full search; random
search over part of the combinations instead of all.

## Task 0. Loading the data

Load the dataset and print the first few rows to make sure it loaded correctly.

## Task 1 (0.5 points)

1.1. Find all features that have missing values. Remove from the sample all objects with missing
values.

1.2. Extract the target variable into a separate variable, remove it from the dataset, and convert it
to binary format. Not all features are real-valued — separate out the real-valued features (in the next
section we work only with them).

## Task 2 (1.5 points)

We consider 5 algorithms: kNN, SGD Linear Classifier, Naive Bayes Classifier, Logistic Regression, SVC
(Support Vector Classifier). For the first two, choose one hyperparameter to optimize:
- kNN — the number of neighbors (`n_neighbors`);
- SGD Linear Classifier — the optimized function (`loss`).

The remaining parameters — by default. Use `GridSearchCV` with 5-fold cross-validation.

2.1. Tune the optimal values of the specified hyperparameters for the first two algorithms. Plot the
mean cross-validation quality as a function of the hyperparameter value, with a confidence interval
`[mean - std, mean + std]`.

2.2. Comment on the resulting plots.

## Task 3 (1.5 points)

Tune the regularization parameter `C` in the `LogisticRegression` and `SVC` algorithms.

## Task 4 (1 point)

Study the documentation for the Naive Bayes Classifier algorithm and tune the possible hyperparameters
for this algorithm.

## Task 5 (0.5 points)

Some of the algorithms used are sensitive to the scale of the features.

5.1. Plot histograms for the features `age`, `fnlwgt`, `capital-gain`.

5.2. Based on the plots, write:
- what the peculiarity of the data is;
- which algorithms this might affect;
- can scaling affect the performance of these algorithms?

## Task 6 (1 point)

Feature scaling can be done, for example, by:
- standardization: x_new = (x − mean) / standard deviation (`StandardScaler`,
  `sklearn.preprocessing.scale`);
- normalization to the range [0, 1]: x_new = (x − min) / (max − min) (`MinMaxScaler`).

6.1. Scale all real-valued features by one of the specified methods and tune the optimal hyperparameter
values as in tasks 2/3.

6.2. Did the quality change for some algorithms?

## Task 7 (2 points)

7.1. Do a grid search over several hyperparameters and find the optimal combinations (the best mean
cross-validation quality) for each algorithm, for example:
- KNN — the number of neighbors (`n_neighbors`) and the metric (`metric`);
- SGDClassifier — the optimized function (`loss`) and `penalty`;
- for the remaining three algorithms, decide for yourself which sets of hyperparameters to try.

Keep in mind that this operation can be resource-intensive.

7.2. Which of the algorithms has the best quality?

## Task 9 (0.5 points)

Up to now, non-numeric (categorical) features have not been used.

9.1. Convert all categorical features using one-hot encoding (`pandas.get_dummies` or
`sklearn.feature_extraction.DictVectorizer`).

## Task 10 (0.5 points)

10.1. Add the encoded categorical features to the scaled real-valued ones and train the algorithms with
the best hyperparameters from the previous point (a new hyperparameter search is not required in this
point). Did adding the new features give a gain in quality? Measure quality, as before, via 5-Fold CV —
it's convenient to use `cross_val_score`.

10.2. Is the best classifier now different from the best one in the previous point?

## Task 12 (1 point)

12.1. For each type of classifier (kNN, SGD classifier, etc.) choose the one that gave the best
cross-validation quality, and plot a box plot (boxplot) — all classifiers should be shown on a single
plot (`matplotlib.pyplot.boxplot` or `seaborn.boxplot`).

12.2. Draw general final conclusions about the classifiers in terms of how they work with features and
the complexity of the model itself (what hyperparameters the model has, whether a change in the
hyperparameter value strongly affects the model's quality).
