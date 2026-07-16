# Homework: decision trees (bonus part of homework 4)

Source: `content/ml_basics_course/homeworks/homework 4 (Trees and Ensembles)/homework_4_trees_and_ensembles.ipynb`.
This homework is shared between two topics — `decision-trees` and
`ensembles-bagging-boosting`. Only the bonus part (theoretical problems on
decision trees) is collected here; the main practical part (bagging, random
forest, gradient boosting on the diabetes data) is in the file
`content/notes/ensembles-bagging-boosting-homework.md`.

## Problem 1 (0.25 points)

A leaf of a tree ends up with 10 objects, 8 of which are from one class and 2
— from the second. Compute the (binary — with a logarithm base 2) entropy of
the resulting sample in the leaf. Round the answer to two decimal places.

## Problem 2 (0.25 points)

For the table below, compute how many predicates of the form "feature = some
value" (that is, predicates of the form [xⱼ = a]) you need to go through in
order to build the first node of a decision tree.

| x1 | x2 | x3 | y |
|----|----|----|---|
| A1 | A2 | A3 | A |
| B1 | A2 | A3 | A |
| C1 | B2 | A3 | B |
| A1 | C2 | B3 | A |
| B1 | D2 | A3 | B |
| B1 | C2 | B3 | B |
| C1 | D2 | B3 | A |

## Problem 3 (0.25 points)

Using the table below, determine which feature should be used to form the
first node of a decision tree, if we want to predict y. Use entropy as the
informativeness criterion, and the indicators [xⱼ = a] as the splitting
criteria.

| x1 | x2 | x3 | x4 | y |
|----|----|----|----|---|
| A1 | A2 | A3 | A4 | A |
| B1 | A2 | B3 | A4 | A |
| C1 | C2 | A3 | A4 | A |
| A1 | A2 | D3 | B4 | A |
| C1 | B2 | C3 | A4 | B |
| B1 | C2 | D3 | B4 | A |
| A1 | B2 | B3 | A4 | A |
| C1 | C2 | C3 | B4 | B |
| B1 | B2 | C3 | B4 | B |
| A1 | C2 | C3 | A4 | B |

Hint: look carefully at the data.

## Problem 4 (0.25 points)

From the listed sets of objects of different classes, choose the set with the
**lowest** binary entropy.

1. 30 objects of class 0, 10 objects of class 1
2. 20 objects of class 0, 10 objects of class 1, 10 objects of class 2
3. 35 objects of class 0, 5 objects of class 1, 5 objects of class 2
4. 20 objects of class 0, 20 objects of class 1
