# Домашнее задание: ансамбли (основная часть homework 4)

Источник: `content/ml_basics_course/homeworks/homework 4 (Trees and Ensembles)/homework_4_trees_and_ensembles.ipynb`.
Это домашнее задание общее для двух тем — `decision-trees` и
`ensembles-bagging-boosting`. Здесь собрана основная практическая часть
(10 баллов) — применение ансамблей на реальной задаче. Бонусная
теоретическая часть про энтропию и решающие деревья — в файле
`content/notes/decision-trees-homework.md`.

## О задании

Работаем с задачей предсказания диабета у пациента (датасет Pima Indians
Diabetes Database, Kaggle). Датасет грузится по прямой ссылке:

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

Целевая переменная — столбец `Outcome` (1 — диабет диагностирован, 0 —
нет).

## Задача 1 (1 балл)

Разбейте выборку на обучающую и тестовую части в отношении 70:30. Не
забудьте отделить целевую переменную от признаков (чтобы случайно не
включить её в обучение как признак).

## Задача 2 (1 балл)

Обучите `BaggingClassifier` на деревьях (параметр
`base_estimator=DecisionTreeClassifier()`). Оцените качество классификации
на тестовой выборке по метрикам accuracy, precision и recall.

## Задача 3 (1 балл)

Обучите Random Forest с числом деревьев, равным 50. Оцените качество
классификации по тем же метрикам. Какая из двух построенных моделей
показала себя лучше?

## Задача 4 (1 балл)

Для случайного леса проанализируйте значение AUC-ROC на этих же данных в
зависимости от изменения параметров (можно сделать обычный перебор с
обучением/тестированием в цикле):
- `n_estimators` (можно перебрать около 10 значений из отрезка от 10 до
  1500);
- `min_samples_leaf` (сетку значений можете выбрать на своё усмотрение).

Постройте соответствующие графики зависимости AUC-ROC от этих параметров.
Какие выводы вы можете сделать?

## Задача 5 (1 балл)

Для лучшей модели случайного леса посчитайте важность признаков и
постройте bar plot с помощью функции `plt.bar`. Какой признак оказался
самым важным для определения диабета?

## Задача 6 (1 балл)

По аналогии со случайным лесом переберите различные значения числа
деревьев для `GradientBoostingClassifier` и постройте график зависимости
AUC-ROC от числа деревьев. Что вы наблюдаете? Отличается ли этот график от
аналогичного графика для случайного леса?

## Задача 7 (2 балла)

Выберите две другие реализации градиентного бустинга (например, XGBoost,
CatBoost или LightGBM), сделайте перебор параметров по каждой из них.
Посчитайте те же метрики: accuracy, precision и recall.

## Задача 8 (2 балла)

Сравните результаты всех построенных в задании моделей и выберите лучшую.
Сделайте выводы по проделанной работе.
