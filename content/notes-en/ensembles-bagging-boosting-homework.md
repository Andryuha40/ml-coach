# Homework: ensembles (the main part of homework 4)

Source: `content/ml_basics_course/homeworks/homework 4 (Trees and Ensembles)/homework_4_trees_and_ensembles.ipynb`.
This homework is shared between two topics — `decision-trees` and
`ensembles-bagging-boosting`. Here the main practical part is collected
(10 points) — applying ensembles to a real task. The bonus
theoretical part about entropy and decision trees is in the file
`content/notes/decision-trees-homework.md`.

## About the assignment

We are working with the task of predicting diabetes in a patient (the Pima Indians
Diabetes Database dataset, Kaggle). The dataset is loaded from a direct link:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DIABETES = 'https://github.com/evgpat/ml_basics_course/raw/main/datasets/diabetes.csv'
data = pd.read_csv(DIABETES)
```

The target variable is the `Outcome` column (1 — diabetes diagnosed, 0 —
not).

## Task 1 (1 point)

Split the sample into training and test parts in a 70:30 ratio. Do not
forget to separate the target variable from the features (so as not to accidentally
include it in training as a feature).

## Task 2 (1 point)

Train a `BaggingClassifier` on trees (parameter
`base_estimator=DecisionTreeClassifier()`). Assess the classification quality
on the test sample by the metrics accuracy, precision and recall.

## Task 3 (1 point)

Train a Random Forest with the number of trees equal to 50. Assess the
classification quality by the same metrics. Which of the two built models
performed better?

## Task 4 (1 point)

For the random forest, analyze the AUC-ROC value on the same data
depending on changes in the parameters (you can do a simple search with
training/testing in a loop):
- `n_estimators` (you can go through about 10 values from the interval from 10 to
  1500);
- `min_samples_leaf` (you can choose the grid of values at your own discretion).

Build the corresponding plots of AUC-ROC against these parameters.
What conclusions can you draw?

## Task 5 (1 point)

For the best random forest model, compute the feature importances and
build a bar plot using the `plt.bar` function. Which feature turned out to be
the most important for determining diabetes?

## Task 6 (1 point)

By analogy with the random forest, go through various values of the number of
trees for the `GradientBoostingClassifier` and build a plot of AUC-ROC
against the number of trees. What do you observe? Does this plot differ from
the analogous plot for the random forest?

## Task 7 (2 points)

Choose two other gradient boosting implementations (for example, XGBoost,
CatBoost or LightGBM), do a parameter search for each of them.
Compute the same metrics: accuracy, precision and recall.

## Task 8 (2 points)

Compare the results of all the models built in the assignment and choose the best one.
Draw conclusions on the work done.
