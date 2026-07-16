# Homework: EDA on student data (homework 1, basic)

Source: `content/ml_basics_course/homeworks/homework 1 (Pandas and EDA)/homework_01_(basic).ipynb`.
All the tasks have equal weight. The assignment drills EDA practice on a new
dataset (unlike the material in the notes, where EDA was worked through on
telecom customer churn data and housing-price data) plus merging several
tables via `pandas` — something the notes do not cover.

## Data description

The data contains information about students, split into 10 groups. The files
are divided into two categories:
- `Students_info_i` — information about the students from group `i`;
- `Students_marks_i` — the exam marks of the students from group `i`.

The data is downloaded as an archive:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

!wget https://github.com/evgpat/datasets/raw/main/students.zip
!unzip students.zip
```

It is assumed that afterwards you will need to predict the math mark for each
student based on their other characteristics.

A hint on the methods for merging tables in pandas (the analog of SQL
`merge`/`join`/`concat`):
https://www.kaggle.com/residentmario/renaming-and-combining#Combining

## Task 1

1.1. Assemble all the information about the students into a single table
`data`: the resulting table should contain the information and marks of all
the students from all 10 groups. Print several rows of the table to
demonstrate the result.

1.2. Write down why EDA is needed. Suggest what information, and in what form,
you need to look at — based on the available dataset and taking into account
the subsequent task of predicting the math mark — and for what purpose.

## Task 2

Drop the `index` column from the resulting table. Print the first 10 rows of
the table.

## Task 3

3.1. Print the dimensions of the resulting table.

3.2. Print the statistical characteristics of the table's numeric columns
(minimum, maximum, mean value, standard deviation).

3.3. Print the information about the remaining features.

3.4. Check whether the table has any missing values.

## Task 4

Build distribution plots of the numeric features.

## Task 5

5.1. Print the correlation matrix for the numeric features.

5.2. Comment on the result in text.

## Task 6

Print the students' average scores for each subject (`math`, `reading`,
`writing`).

Build bar charts of the average writing score depending on:
- gender (`gender`);
- racial/ethnic group (`race/ethnicity`);
- parental level of education (`parental level of education`);
- lunch type (`lunch`);
- completion of the preparation course (`test preparation course`).

## Task 7

How do the marks depend on whether the student took the exam preparation
course (`test preparation course`)? Print, for each subject separately, the
average score of the students who took the exam preparation course and those
who did not.

## Task 8

Rename the column `parental level of education` to `education`, and
`test preparation course` to `test preparation`, using the pandas `rename`
method.

## Task 9

Let us fix the minimum score for passing an exam: `passmark = 50`.

Answer the questions:
- What proportion of students passed the math exam (score > `passmark`)?
- What proportion of the students who took the exam preparation course passed
  the math exam?
- What proportion of the women who did not take the exam preparation course
  failed the math exam?

## Task 10

Using `groupby`, do the following:
- for each ethnic group, print the average reading exam score;
- for each level of education, print the minimum writing exam score.

## Task 11

How many students successfully passed the math exam? Create a new column in
the `data` table called `Math_PassStatus` and write `F` into it if the student
failed the math exam (exam score < `passmark`), and `P` otherwise. Count the
number of students who passed and failed the math exam.

## Task 12

Let us convert scores into grades by the system:
- [90; 100] = A
- [80; 90) = B
- [70; 80) = C
- [60; 70) = D
- [50; 60) = E
- [0; 50) = F (Fail)

Create a helper function `GetGrade(average_mark)` that assigns a grade to a
student based on the average score across the three exams according to the
specified criteria. Create a `Grade` column and write each student's grade
into it. Print the number of students who received each of the grades.

```python
def GetGrade(average_mark):
    # your code here
    ...
```
